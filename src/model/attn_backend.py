from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import os

import torch
from torch import Tensor

from torch.nn.attention.flex_attention import flex_attention


class AttnBackend(str, Enum):
    PYTORCH_FLEX = "pytorch-flex"
    METAL = "metal-op"
    AUTO = "auto"

    @staticmethod
    def default() -> "AttnBackend":
        """
        Resolve the default backend from the WORLD_ATTENTION_BACKEND env var.
        """
        import os

        value = os.getenv("WORLD_ATTENTION_BACKEND", "pytorch-flex").lower()
        if value == "metal":
            return AttnBackend.METAL
        if value == "auto":
            return AttnBackend.AUTO
        return AttnBackend.PYTORCH_FLEX


def _metal_impl_mode() -> str:
    # WORLD_METAL_IMPL=ref|fast
    mode = os.getenv("WORLD_METAL_IMPL", "ref").lower()
    return "fast" if mode == "fast" else "ref"


def _metal_use_causal(cfg: "AttnConfig") -> bool:
    """
    Keep Metal behavior aligned with flex_attention backend semantics.

    The flex path in this repo does not pass `causal` explicitly into
    `flex_attention`; masking semantics are encoded by BlockMask metadata.
    To preserve CPU/CUDA parity, Metal defaults to non-causal unless
    explicitly overridden.
    """
    if os.getenv("WORLD_METAL_FORCE_CAUSAL", "0") == "1":
        return bool(cfg.causal)
    return False


@dataclass
class AttnConfig:
    """
    Backend-agnostic attention configuration.

    This object is intentionally small and forward-only: it encodes only what
    the kernel needs at runtime. Training- and autograd-specific concerns are
    out of scope for the hybrid Metal inference path.
    """

    causal: bool = True
    enable_gqa: bool = False


@dataclass
class AttnMeta:
    """
    Backend-agnostic metadata describing the KV layout for a single attention
    call. This is the hook where we will eventually encode block/window
    sparsity and cache positions for the Metal kernel.

    For the initial implementation, we allow passing the existing BlockMask
    object through as `flex_block_mask` so the PyTorch flex backend can keep
    working while we design a compact Metal-friendly format. In parallel we
    expose basic sequence length information that the Metal backend will use
    to size its tiles.
    """

    # Optional flex BlockMask used by the PyTorch flex backend today.
    flex_block_mask: Optional[object] = None

    # Logical query and KV lengths for this attention call.
    q_len: Optional[int] = None
    kv_len: Optional[int] = None
    block_written: Optional[Tensor] = None
    active_blocks: Optional[Tensor] = None
    block_size: Optional[int] = None

    # Future fields for the Metal backend (block size, bucket indices, validity
    # masks, etc.) will live here as we iterate on the sparsity encoding.


def world_flex_attn_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    meta: Optional[AttnMeta],
    cfg: AttnConfig,
    backend: Optional[AttnBackend] = None,
) -> Tensor:
    """
    Backend-neutral attention entrypoint used by high-level modules.

    Args:
        q, k, v: [B, H, T, Dh] tensors on the same device (MPS, CUDA, etc.).
        meta:   Backend-agnostic metadata describing KV/cache layout.
        cfg:    Small configuration object with behavioral flags.
        backend:
            - PYTORCH_FLEX: call torch.nn.attention.flex_attention directly.
            - METAL:        call the custom Metal op (to be added).
            - AUTO:         choose PYTORCH_FLEX or METAL based on device /
                            availability.
    """
    if backend is None:
        backend = AttnBackend.default()

    if backend is AttnBackend.AUTO:
        backend = AttnBackend.METAL if q.device.type == "mps" else AttnBackend.PYTORCH_FLEX

    if backend is AttnBackend.PYTORCH_FLEX:
        block_mask = meta.flex_block_mask if meta is not None else None
        return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=cfg.enable_gqa)

    if backend is AttnBackend.METAL:
        mask = None
        mode = _metal_impl_mode()
        use_causal = _metal_use_causal(cfg)
        if mode == "fast":
            if meta is not None and meta.active_blocks is not None and meta.block_size is not None:
                return torch.ops.world.flex_attn_metal_fast_active(
                    q, k, v, meta.active_blocks, int(meta.block_size), use_causal
                )
            if meta is not None and meta.block_written is not None and meta.block_size is not None:
                return torch.ops.world.flex_attn_metal_fast_blocks(
                    q, k, v, meta.block_written, int(meta.block_size), use_causal
                )
            return torch.ops.world.flex_attn_metal_fast(q, k, v, mask, use_causal)
        return torch.ops.world.flex_attn_metal_ref(q, k, v, mask, use_causal)

    raise ValueError(f"Unknown attention backend: {backend}")

