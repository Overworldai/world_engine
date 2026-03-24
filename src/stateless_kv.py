"""
Stateless KV cache operations for OpenVINO export.
Replaces the stateful LayerKVCache / StaticKVCache with pure functions
that take KV buffers as inputs and return updated buffers as outputs.
No nn.Module state mutation, no nn.Buffer, no in-place ops.

Updated for wp-1.5: uses f_pos (frame index) for KV cache indexing,
tpf computed as height*width.
"""
import torch
from torch import Tensor


def make_dense_mask(T: int, capacity: int, written: Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """
    Build a dense additive attention mask from the written boolean vector.

    Returns: [1, 1, T, capacity] float mask (0.0 = attend, -inf = ignore)
    Uses arithmetic instead of torch.where to avoid Intel GPU OpenCL Select bug.
    """
    # ~written (invalid slots) -> large negative, written (valid) -> 0.0
    inv = (~written).to(dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, T, capacity)
    return inv * (-1e9)


def upsert_stateless(
    kv_buf: Tensor,
    written: Tensor,
    new_k: Tensor,
    new_v: Tensor,
    pos_ids: dict,
    is_frozen: Tensor,
    tpf: int,
    L: int,
    num_buckets: int,
    pinned_dilation: int,
) -> tuple:
    """
    Stateless KV cache upsert. Mirrors LayerKVCache.upsert() logic but without
    in-place mutation.

    Args:
        kv_buf: [2, B, H, capacity, Dh] — current KV state
        written: [capacity] bool — which slots have been written
        new_k: [B, H, T, Dh] — current frame keys
        new_v: [B, H, T, Dh] — current frame values
        pos_ids: dict with "f_pos" [B, T] (frame index for KV cache)
        is_frozen: scalar bool tensor
        tpf: tokens per frame
        L: ring buffer length (capacity - tpf)
        num_buckets: L // tpf // pinned_dilation
        pinned_dilation: dilation factor for pinned slots

    Returns:
        k_full: [B, H, capacity, Dh] — full K for attention
        v_full: [B, H, capacity, Dh] — full V for attention
        attn_mask: [1, 1, T, capacity] — dense additive mask
        kv_buf_out: [2, B, H, capacity, Dh] — updated KV state
        written_out: [capacity] bool — updated written mask
    """
    T = tpf
    capacity = L + T
    f_pos = pos_ids["f_pos"]
    frame_idx = f_pos[0, 0]

    # Compute ring buffer slot
    bucket = (frame_idx + (pinned_dilation - 1)) // pinned_dilation
    slot = bucket % num_buckets
    base = slot * T

    # Ring indices for this frame
    frame_offsets = torch.arange(T, dtype=torch.long, device=kv_buf.device)
    ring_idx = frame_offsets + base
    current_idx = frame_offsets + L  # tail slice indices

    # Stack new K,V — cast to match KV buffer dtype
    new_kv = torch.stack([new_k, new_v], dim=0).to(kv_buf.dtype)  # [2, B, H, T, Dh]

    # Always write current frame into the tail slice [L, L+T)
    kv_out = kv_buf.clone()
    kv_out[:, :, :, current_idx, :] = new_kv

    # Build attention mask: mask out the ring slot being overwritten
    # Avoid scatter on bool/int — use broadcast comparison for OV GPU compatibility
    all_idx = torch.arange(capacity, dtype=torch.long, device=kv_buf.device)
    is_ring = (all_idx.unsqueeze(-1) == ring_idx.unsqueeze(0)).any(-1)  # [capacity]

    write_step = (frame_idx.remainder(pinned_dilation) == 0)
    # When write_step: clear ring slots from mask. Otherwise: keep as-is.
    mask_written = written & ~(is_ring & write_step)

    attn_mask = make_dense_mask(T, capacity, mask_written, dtype=new_k.dtype)

    # If not frozen, persist current frame into the ring buffer
    # Compute destination: write_step -> ring_idx, else -> current_idx
    # Use arithmetic instead of torch.where to avoid Intel GPU OpenCL Select bug
    ws = write_step.long()
    dst = ws * ring_idx + (1 - ws) * current_idx
    is_dst = (all_idx.unsqueeze(-1) == dst.unsqueeze(0)).any(-1)  # [capacity]

    # Compute unfrozen KV state
    kv_written = kv_out.clone()
    kv_written[:, :, :, dst, :] = new_kv
    written_written = written | is_dst

    # Select based on is_frozen using arithmetic (no torch.where)
    frozen_f = is_frozen.reshape(()).to(kv_out.dtype)  # 1.0 if frozen, 0.0 if not
    kv_out = frozen_f * kv_out + (1.0 - frozen_f) * kv_written
    # For written mask: frozen -> keep original, unfrozen -> use written_written
    frozen_b = is_frozen.reshape(())
    written_out = (frozen_b & written) | (~frozen_b & written_written)

    k_full, v_full = kv_out.unbind(0)
    return k_full, v_full, attn_mask, kv_out, written_out


class StatelessKVManager:
    """
    Manages KV cache state externally as plain tensors.
    Replaces StaticKVCache for the portable model.
    """

    def __init__(self, config, batch_size: int, dtype: torch.dtype, device: torch.device):
        self.tpf = config.height * config.width
        self.n_layers = config.n_layers
        self.dtype = dtype
        self.device = device

        local_L = config.local_window * self.tpf
        global_L = config.global_window * self.tpf
        period = config.global_attn_period
        off = getattr(config, "global_attn_offset", 0) % period
        n_kv_heads = getattr(config, "n_kv_heads", config.n_heads)
        d_head = config.d_model // config.n_heads

        self.layer_configs = []
        self.kv_bufs = []
        self.written_bufs = []

        for layer_idx in range(config.n_layers):
            is_global = ((layer_idx - off) % period == 0)
            L = global_L if is_global else local_L
            pd = config.global_pinned_dilation if is_global else 1
            capacity = L + self.tpf
            num_buckets = (L // self.tpf) // pd

            self.layer_configs.append({
                "L": L,
                "capacity": capacity,
                "num_buckets": num_buckets,
                "pinned_dilation": pd,
            })

            kv = torch.zeros(2, batch_size, n_kv_heads, capacity, d_head, dtype=dtype, device=device)
            written = torch.zeros(capacity, dtype=torch.bool, device=device)
            written[L:] = True  # tail slice always considered written

            self.kv_bufs.append(kv)
            self.written_bufs.append(written)

    def reset(self):
        for i, cfg in enumerate(self.layer_configs):
            self.kv_bufs[i].zero_()
            self.written_bufs[i].zero_()
            self.written_bufs[i][cfg["L"]:] = True

    def get_state(self):
        """Returns (kv_bufs, written_bufs) as lists of tensors."""
        return self.kv_bufs, self.written_bufs

    def set_state(self, kv_bufs, written_bufs):
        """Update state from model outputs."""
        self.kv_bufs = kv_bufs
        self.written_bufs = written_bufs
