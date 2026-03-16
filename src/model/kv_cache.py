from torch import Tensor
import torch
from torch import nn
from tensordict import TensorDict
import os

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    BlockMask
)


def make_block_mask(T: int, L: int, written: torch.Tensor):
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

    return bm, block_any.contiguous()


def _block_any_for_size(written: torch.Tensor, block_size: int) -> torch.Tensor:
    kv_blocks = (written.numel() + block_size - 1) // block_size
    padded = torch.nn.functional.pad(written, (0, kv_blocks * block_size - written.numel()))
    return padded.view(kv_blocks, block_size).any(-1).contiguous()


def _metal_block_size() -> int:
    env = os.environ.get("WORLD_METAL_BLOCK_SIZE")
    if env is None:
        return 4
    try:
        parsed = int(env)
        return parsed if parsed > 0 else 4
    except ValueError:
        return 4


def _kv_runtime_checks_enabled() -> bool:
    return os.environ.get("WORLD_KV_RUNTIME_CHECKS", "0") == "1"


def _compute_active_blocks_enabled() -> bool:
    return os.environ.get("WORLD_KV_COMPUTE_ACTIVE_BLOCKS", "0") == "1"


def _using_metal_backend() -> bool:
    return os.environ.get("WORLD_ATTENTION_BACKEND", "flex").lower() == "metal"


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
        self._num_buckets_mask = (self.num_buckets - 1) if (self.num_buckets & (self.num_buckets - 1)) == 0 else -1

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
        self._mask_written = nn.Buffer(torch.empty_like(written), persistent=False)
        self._block_written = nn.Buffer(torch.empty(0, dtype=torch.uint8), persistent=False)
        self._all_blocks_i32 = nn.Buffer(torch.empty(0, dtype=torch.int32), persistent=False)
        self._tmp_block_written = nn.Buffer(torch.empty(0, dtype=torch.uint8), persistent=False)
        self._metal_bs_cache = 0
        self._blocks_per_frame = 0
        self._seen_slots: set[int] = set()
        self._seen_slots_ordered: list[int] = []
        self._slot_block_ranges: list[torch.Tensor] = []
        self._tail_block_range: torch.Tensor | None = None
        self._can_build_active_without_nonzero = False

        # Precompute indices:
        #   frame_offsets: [0, 1, ..., tpf-1] (for ring indexing)
        #   current_idx:   [L, L+1, ..., L+tpf-1] (tail slice)
        self.frame_offsets = nn.Buffer(torch.arange(self.tpf, dtype=torch.long), persistent=False)
        self.current_idx = nn.Buffer(self.frame_offsets + L, persistent=False)
        self._metal_backend = _using_metal_backend()
        self._need_active_metadata = self._metal_backend or _compute_active_blocks_enabled()
        self._configured_metal_bs = _metal_block_size()

    def reset(self):
        self.kv.zero_()
        self.written.zero_()
        self.written[self.L:].fill_(True)
        self._metal_bs_cache = 0
        self._blocks_per_frame = 0
        self._seen_slots.clear()
        self._seen_slots_ordered = []
        self._slot_block_ranges = []
        self._tail_block_range = None
        self._can_build_active_without_nonzero = False
        if self._block_written.numel() > 0:
            self._block_written.zero_()
        if self._all_blocks_i32.numel() > 0:
            self._all_blocks_i32 = self._all_blocks_i32.new_empty((0,), dtype=torch.int32)
        if self._tmp_block_written.numel() > 0:
            self._tmp_block_written.zero_()

    def _ensure_block_written(self, metal_bs: int):
        if self._metal_bs_cache == metal_bs and self._block_written.numel() > 0:
            return
        block_any = _block_any_for_size(self.written, metal_bs).to(torch.uint8).contiguous()
        self._block_written = block_any
        if self._tmp_block_written.numel() != block_any.numel():
            self._tmp_block_written = torch.empty_like(block_any)
        self._all_blocks_i32 = torch.arange(block_any.numel(), device=block_any.device, dtype=torch.int32)
        self._metal_bs_cache = metal_bs
        self._blocks_per_frame = (self.tpf + metal_bs - 1) // metal_bs
        self._slot_block_ranges = []
        self._tail_block_range = None
        self._can_build_active_without_nonzero = False
        if (self.tpf % metal_bs) == 0 and (self.L % metal_bs) == 0:
            frame_blocks = self.tpf // metal_bs
            base = torch.arange(frame_blocks, device=block_any.device, dtype=torch.int32)
            self._slot_block_ranges = [
                (base + (slot * frame_blocks)).contiguous()
                for slot in range(self.num_buckets)
            ]
            self._tail_block_range = (base + (self.L // metal_bs)).contiguous()
            self._can_build_active_without_nonzero = True

    def rebuild_seen_slots(self):
        ring_tokens = self.num_buckets * self.tpf
        ring_written = self.written.narrow(0, 0, ring_tokens).view(self.num_buckets, self.tpf)
        occupied = ring_written.any(dim=1).to("cpu")
        self._seen_slots = {i for i, w in enumerate(occupied.tolist()) if bool(w)}
        self._seen_slots_ordered = sorted(self._seen_slots)

    def _active_blocks_for_block_written(
        self,
        block_written: torch.Tensor,
        write_step: bool,
        ring_block_start: int,
        slot: int,
    ) -> torch.Tensor:
        if self._can_build_active_without_nonzero and self._tail_block_range is not None:
            parts: list[torch.Tensor] = []
            for seen_slot in self._seen_slots_ordered:
                if write_step and seen_slot == slot:
                    continue
                parts.append(self._slot_block_ranges[seen_slot])
            parts.append(self._tail_block_range)
            if len(parts) == 1:
                return parts[0]
            return torch.cat(parts, dim=0).contiguous()

        # In steady state, ring blocks are fully written and mask updates are
        # contiguous; build active indices arithmetically to avoid nonzero sync.
        if len(self._seen_slots) == self.num_buckets:
            all_blocks = self._all_blocks_i32
            if not write_step:
                return all_blocks
            start = ring_block_start
            end = min(start + self._blocks_per_frame, all_blocks.numel())
            if start <= 0:
                return all_blocks.narrow(0, end, all_blocks.numel() - end).contiguous()
            if end >= all_blocks.numel():
                return all_blocks.narrow(0, 0, start).contiguous()
            left = all_blocks.narrow(0, 0, start)
            right = all_blocks.narrow(0, end, all_blocks.numel() - end)
            return torch.cat((left, right), dim=0).contiguous()
        return torch.nonzero(block_written, as_tuple=False).flatten().to(torch.int32).contiguous()

    def upsert(self, kv: Tensor, pos_ids: TensorDict, is_frozen: bool, frame_idx_int: int | None = None):
        """
        kv: [2, B, H, T, Dh] for a single frame (T = tokens_per_frame)
        t_pos: [B, T], all equal per frame (ignoring -1)
        """
        T = self.tpf
        f_pos = pos_ids["f_pos"]

        if _kv_runtime_checks_enabled() and not torch.compiler.is_compiling():
            torch._check(kv.size(3) == self.tpf, "KV cache expects exactly one frame per upsert")
            torch._check(f_pos.shape == (kv.size(1), T), "t_pos must be [B, T]")
            torch._check(self.tpf <= self.L, "frame longer than KV ring capacity")
            torch._check(self.L % self.tpf == 0, f"L ({self.L}) must be a multiple of tokens_per_frame ({self.tpf})")
            torch._check(self.kv.size(3) == self.capacity, "KV buffer too long (expected L + tokens_per_frame)")
            torch._check((f_pos >= 0).all().item(), "t_pos must be non-negative during inference")
            torch._check(((f_pos == f_pos[:, :1]).all()).item(), "t_pos must be constant within frame")

        frame_idx = int(f_pos[0, 0].item()) if frame_idx_int is None else int(frame_idx_int)

        # map frame_t to a bucket, each bucket owns T contiguous slots
        bucket = (frame_idx + (self.pinned_dilation - 1)) // self.pinned_dilation
        slot = (bucket & self._num_buckets_mask) if self._num_buckets_mask >= 0 else (bucket % self.num_buckets)
        base = slot * T

        ring_start = int(base)
        ring_end = ring_start + T

        # Always write current frame into the tail slice [L, L+T):
        # this is the "self-attention component" for the current frame.
        self.kv.narrow(3, self.L, T).copy_(kv)

        bm = None
        metal_bs = self._configured_metal_bs
        need_active_metadata = self._need_active_metadata
        write_step = (frame_idx % self.pinned_dilation) == 0
        if self._metal_backend:
            self._ensure_block_written(metal_bs)
            ring_block_start = ring_start // metal_bs
            if write_step:
                block_written = self._tmp_block_written
                block_written.copy_(self._block_written)
                block_written.narrow(0, ring_block_start, self._blocks_per_frame).zero_()
            else:
                block_written = self._block_written
        else:
            mask_written = self._mask_written
            mask_written.copy_(self.written)
            if write_step:
                mask_written.narrow(0, ring_start, T).fill_(False)
            bm, _ = make_block_mask(T, self.capacity, mask_written)
            block_written = _block_any_for_size(mask_written, metal_bs).to(torch.uint8).contiguous()
        active_blocks = None
        if need_active_metadata:
            ring_block_start = ring_start // metal_bs
            active_blocks = self._active_blocks_for_block_written(block_written, write_step, ring_block_start, slot)

        # Persist current frame into the ring for future queries when unfrozen.
        if not is_frozen:
            # Persist current frame into the ring for future queries.
            # If write_step is false, current frame remains only in tail.
            if write_step:
                self.kv.narrow(3, ring_start, T).copy_(kv)
                first_write_for_slot = (slot not in self._seen_slots)
                if first_write_for_slot:
                    self.written.narrow(0, ring_start, T).fill_(True)
                    if need_active_metadata:
                        self._seen_slots.add(slot)
                        if slot not in self._seen_slots_ordered:
                            self._seen_slots_ordered.append(slot)
                            self._seen_slots_ordered.sort()
                    if self._metal_backend:
                        ring_block_start = ring_start // metal_bs
                        self._block_written.narrow(0, ring_block_start, self._blocks_per_frame).fill_(1)

        k, v = self.kv.unbind(0)
        return k, v, bm, block_written, active_blocks, metal_bs


class StaticKVCache(nn.Module):
    def __init__(self, config, batch_size, dtype):
        super().__init__()

        self.tpf = config.height * config.width

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
        self._cached_fpos_ptr = -1
        self._cached_fpos_version = -1
        self._cached_fpos_value = 0
        self._frame_idx_hint: int | None = None

    def reset(self):
        for layer in self.layers:
            layer.reset()
        self._is_frozen = True
        self._cached_fpos_ptr = -1
        self._cached_fpos_version = -1
        self._cached_fpos_value = 0
        self._frame_idx_hint = None

    @torch.inference_mode()
    def get_state(self):
        layers = [(layer.kv.detach().clone(), layer.written.detach().clone()) for layer in self.layers]
        return {"_is_frozen": self._is_frozen, "layers": layers}

    @torch.inference_mode()
    def load_state(self, state):
        self._is_frozen = bool(state.get("_is_frozen", True))
        for layer, (kv, written) in zip(self.layers, state["layers"]):
            layer.kv.copy_(kv)
            layer.written.copy_(written)
            layer.rebuild_seen_slots()
            layer._metal_bs_cache = 0
        self._cached_fpos_ptr = -1
        self._cached_fpos_version = -1
        self._cached_fpos_value = 0
        self._frame_idx_hint = None

    def set_frozen(self, is_frozen: bool):
        self._is_frozen = is_frozen

    def set_frame_idx_int(self, frame_idx_int: int):
        self._frame_idx_hint = int(frame_idx_int)

    def get_frame_idx(self, pos_ids: TensorDict) -> int:
        if self._frame_idx_hint is not None:
            return self._frame_idx_hint
        fpos = pos_ids["f_pos"]
        ptr = int(fpos.data_ptr())
        version = int(fpos._version)
        if ptr != self._cached_fpos_ptr or version != self._cached_fpos_version:
            self._cached_fpos_ptr = ptr
            self._cached_fpos_version = version
            self._cached_fpos_value = int(fpos[0, 0].item())
        return self._cached_fpos_value

    def upsert(self, k: Tensor, v: Tensor, pos_ids: TensorDict, layer: int, frame_idx_int: int | None = None):
        kv = torch.stack([k, v], dim=0)
        return self.layers[layer].upsert(kv, pos_ids, self._is_frozen, frame_idx_int=frame_idx_int)
