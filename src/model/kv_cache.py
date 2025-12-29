from torch import Tensor
import torch
from torch import nn
from tensordict import TensorDict

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    BlockMask
)


def make_block_mask(T: int, L: int, written: torch.Tensor) -> BlockMask:
    """
    T: Q length for this frame
    L: KV capacity == written.numel()
    written: [L] bool, True where there is valid KV data
    """
    BS = _DEFAULT_SPARSE_BLOCK_SIZE
    KV_blocks = (L + BS - 1) // BS
    Q_blocks = (T + BS - 1) // BS

    # [KV_blocks, BS]
    written_blocks = torch.nn.functional.pad(written, (0, KV_blocks * BS - L)).view(KV_blocks, BS)

    # Block-level occupancy
    block_any = written_blocks.any(-1)    # block has at least one written token
    block_all = written_blocks.all(-1)    # block is fully written

    # Every Q-block sees the same KV-block pattern
    nonzero_bm = block_any[None, :].expand(Q_blocks, KV_blocks)       # [Q_blocks, KV_blocks]
    full_bm = block_all[None, :].expand_as(nonzero_bm)                # [Q_blocks, KV_blocks]
    partial_bm = nonzero_bm & ~full_bm                                # [Q_blocks, KV_blocks]

    def dense_to_ordered(dense_mask: torch.Tensor):
        # dense_mask: [Q_blocks, KV_blocks] bool
        # returns: [1,1,Q_blocks], [1,1,Q_blocks,KV_blocks]
        num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)        # [Q_blocks]
        indices = dense_mask.argsort(dim=-1, descending=True, stable=True).to(torch.int32)
        return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

    # Partial blocks (need mask_mod)
    kv_num_blocks, kv_indices = dense_to_ordered(partial_bm)

    # Full blocks (mask_mod can be skipped entirely)
    full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)

    def mask_mod(b, h, q, kv):
        return written[kv]

    bm = BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=BS,
        mask_mod=mask_mod,
        seq_lengths=(T, L),
        compute_q_blocks=False,  # no backward, avoids the transpose/_ordered_to_dense path
    )

    return bm


class LayerKVCache(nn.Module):
    """
    Ring-buffer KV cache with fixed capacity L (tokens) for history plus
    one extra frame (tokens_per_frame) at the tail holding the current frame.
    """

    def __init__(self, B, H, L, Dh, dtype, tokens_per_frame: int, pinned_dilation: int = 1):
        super().__init__()
        self.tpf = tokens_per_frame
        self.L = L
        # total KV capacity: ring (L) + tail frame (tpf)
        self.capacity = L + self.tpf
        self.pinned_dilation = pinned_dilation
        self.num_buckets = (L // self.tpf) // self.pinned_dilation
        assert (L // self.tpf) % pinned_dilation == 0 and L % self.tpf == 0

        # KV buffer: [2, B, H, capacity, Dh]
        self.kv = nn.Buffer(
            torch.zeros(2, B, H, self.capacity, Dh, dtype=dtype),
            persistent=False,
        )

        # which slots have ever been written
        # tail slice [L, L+tpf) always holds the current frame and is considered written
        written = torch.zeros(self.capacity, dtype=torch.bool)
        written[L:] = True
        self.written = nn.Buffer(written, persistent=False)

        # Precompute indices:
        #   frame_offsets: [0, 1, ..., tpf-1] (for ring indexing)
        #   current_idx:   [L, L+1, ..., L+tpf-1] (tail slice)
        self.frame_offsets = nn.Buffer(torch.arange(self.tpf, dtype=torch.long), persistent=False)
        self.current_idx = nn.Buffer(self.frame_offsets + L, persistent=False)

    def reset(self):
        self.kv.zero_()
        self.written.zero_()
        self.written[self.L:].fill_(True)

    def upsert(self, kv: Tensor, pos_ids: TensorDict, is_frozen: bool):
        """
        kv: [2, B, H, T, Dh] for a single frame (T = tokens_per_frame)
        t_pos: [B, T], all equal per frame (ignoring -1)
        """
        T = self.tpf
        t_pos = pos_ids["t_pos"]

        if not torch.compiler.is_compiling():
            torch._check(kv.size(3) == self.tpf, "KV cache expects exactly one frame per upsert")
            torch._check(t_pos.shape == (kv.size(1), T), "t_pos must be [B, T]")
            torch._check(self.tpf <= self.L, "frame longer than KV ring capacity")
            torch._check(self.L % self.tpf == 0,
                         f"L ({self.L}) must be a multiple of tokens_per_frame ({self.tpf})")
            torch._check(self.kv.size(3) == self.capacity,
                         "KV buffer has unexpected length (expected L + tokens_per_frame)")
            torch._check(
                (t_pos >= 0).all().item(),
                "t_pos must be non-negative during inference",
            )
            torch._check(((t_pos == t_pos[:, :1]).all()).item(), "t_pos must be constant within frame")

        frame_t = t_pos[0, 0]

        # map frame_t to a bucket, each bucket owns T contiguous slots
        bucket = (frame_t + (self.pinned_dilation - 1)) // self.pinned_dilation
        slot = bucket % self.num_buckets
        base = slot * T

        # indices in the ring for this frame: [T] in [0, L)
        ring_idx = self.frame_offsets + base

        # Always write current frame into the tail slice [L, L+T):
        # this is the "self-attention component" for the current frame.
        self.kv.index_copy_(3, self.current_idx, kv)

        write_step = (frame_t.remainder(self.pinned_dilation) == 0)
        mask_written = self.written.clone()
        mask_written[ring_idx] = mask_written[ring_idx] & ~write_step
        bm = make_block_mask(T, self.capacity, mask_written)

        # Persist current frame into the ring for future queries when unfrozen.
        if not is_frozen:
            # Persist current frame into the ring for future queries.
            dst = torch.where(write_step, ring_idx, self.current_idx)
            self.kv.index_copy_(3, dst, kv)
            self.written[dst] = True

        k, v = self.kv.unbind(0)
        return k, v, bm


class StaticKVCache(nn.Module):
    def __init__(self, config, batch_size, dtype):
        super().__init__()

        self.tpf = config.tokens_per_frame

        local_L = config.local_window * self.tpf
        global_L = config.global_window * self.tpf

        period = config.global_attn_period
        off = getattr(config, "global_attn_offset", 0) % period
        self.layers = nn.ModuleList([
            LayerKVCache(
                batch_size,
                getattr(config, "n_kv_heads", config.n_heads),
                global_L if ((layer_idx - off) % period == 0) else local_L,
                config.d_model // config.n_heads,
                dtype,
                self.tpf,
                config.global_pinned_dilation if ((layer_idx - off) % period == 0) else 1,
            )
            for layer_idx in range(config.n_layers)
        ])

        self._is_frozen = True

    def reset(self):
        for layer in self.layers:
            layer.reset()
        self._is_frozen = True

    def set_frozen(self, is_frozen: bool):
        self._is_frozen = is_frozen

    def upsert(self, k: Tensor, v: Tensor, pos_ids: TensorDict, layer: int):
        kv = torch.stack([k, v], dim=0)
        return self.layers[layer].upsert(kv, pos_ids, self._is_frozen)
