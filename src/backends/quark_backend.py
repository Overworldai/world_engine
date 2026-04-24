"""Torch-free DiT inference via the ``quark`` stack.

v1 scope — the torch VAE (taehv / InferenceAE) still runs for encode/
decode; we bridge to ``QuarkTensor`` at the latent boundary via
``QuarkTensor.borrow`` (zero-copy). Only the DiT hot path (denoise +
commit) runs on quark kernels.

Not supported in v1 (tracked TODOs):
- Prompt cross-attention: ``quark.nn`` has no ``CrossAttention`` yet.
  ``set_prompt`` warns + no-ops; prompted ckpts produce degraded
  output until the cross-attn port lands.
- ``get_state`` / ``load_state``: quark KV ring-buffer layout differs
  from ``StaticKVCache``; round-trip punted to follow-up.
- Quantization modes other than ``fp8w8a8`` (FP4 needs flashinfer,
  intw8a8 needs gemlite — both torch-only for now).
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Optional

import torch


def _resolve_safetensors_path(model_uri: str) -> str:
    """Return a local path to ``model.safetensors`` for ``model_uri``.

    Accepts a local directory or an HF repo id.
    """
    if os.path.isdir(model_uri):
        return os.path.join(model_uri, "model.safetensors")
    import huggingface_hub

    local_dir = huggingface_hub.snapshot_download(
        model_uri, allow_patterns=["config.yaml", "model.safetensors"]
    )
    return os.path.join(local_dir, "model.safetensors")


def _qt_borrow(t: torch.Tensor, dtype: str = "bf16"):
    """Zero-copy wrap a contiguous CUDA torch tensor as a ``QuarkTensor``.
    Holds a reference to the torch tensor via ``owner`` so its storage
    outlives the wrapper.
    """
    from quark.runtime.tensor import QuarkTensor

    assert t.is_cuda, "QuarkBackend requires CUDA tensors"
    assert t.is_contiguous(), "borrowed tensor must be contiguous"
    return QuarkTensor.borrow(
        t.data_ptr(),
        t.numel() * t.element_size(),
        tuple(t.shape),
        dtype,
        owner=t,
    )


class QuarkBackend:
    """Delegate owned by ``WorldEngine`` when ``backend="quark"``.

    Public surface matches the methods ``WorldEngine`` forwards:
    ``reset``, ``set_prompt``, ``append_frame``, ``gen_frame``,
    ``get_state``, ``load_state``.
    """

    def __init__(
        self,
        model_uri: str,
        *,
        quant: Optional[str] = None,
        model_config_overrides: Optional[Dict] = None,
        device=None,
        dtype=torch.bfloat16,
        load_weights: bool = True,
    ):
        from quark.models.waypoint_15 import (
            GenerateFrame,
            Waypoint15,
            Waypoint15Config,
            remap_world_engine_state_dict,
        )
        from quark.nn.io import load_safetensors

        from ..ae import get_ae
        from ..model import WorldModel

        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype

        self.we_cfg = WorldModel.load_config(model_uri)
        if model_config_overrides:
            self.we_cfg.merge_with(model_config_overrides)

        if self.we_cfg.prompt_conditioning is not None:
            warnings.warn(
                "QuarkBackend: prompt_conditioning is enabled on this config "
                "but prompt cross-attention has not been ported to quark yet. "
                "The prompt will be ignored — outputs will be degraded vs. "
                "the torch backend. Use backend='torch' for prompted inference.",
                RuntimeWarning,
                stacklevel=2,
            )

        # ── Model ────────────────────────────────────────────────
        # use_f16=False pins the residual stream to bf16, which matches
        # the ``torch.bfloat16`` latent + ctrl + VAE pipe on this side
        # and avoids an f16↔bf16 cast at the borrowed-buffer boundary.
        cfg = Waypoint15Config.from_world_engine(self.we_cfg, use_f16=False)
        self.cfg = cfg
        self.model = Waypoint15(cfg)

        if load_weights:
            raw = load_safetensors(_resolve_safetensors_path(model_uri), dtype="bf16")
            self.model.load_state_dict(remap_world_engine_state_dict(raw, cfg), strict=False)

        if quant is None:
            fp8 = False
        elif quant == "fp8w8a8":
            fp8 = True
        else:
            raise NotImplementedError(
                f"QuarkBackend: quant={quant!r} not supported (only fp8w8a8 / None). "
                f"Use backend='torch' for intw8a8/nvfp4."
            )
        self.model.prepare(fp8=fp8)
        self.gen = GenerateFrame(self.model)
        self._graph_ready = False

        # Persistent ctrl input: stable bf16 device buffer + host packer.
        # ``fill(ctrl)`` memcpy_htod's into the same pointer every frame so
        # the graph-captured input address stays stable.
        self._ctrl_dev, self._ctrl_fill = self.model.make_ctrl_buffer()

        # ── VAE (torch side) ─────────────────────────────────────
        pH, pW = cfg.patch
        self.vae = get_ae(
            self.we_cfg.ae_uri,
            is_taehv_ae=self.we_cfg.taehv_ae,
            auto_aspect_ratio=self.we_cfg.auto_aspect_ratio,
            dtype=dtype,
            device=self.device,
            **(
                {"height": cfg.height * pH, "width": cfg.width * pW}
                if self.we_cfg.taehv_ae
                else {}
            ),
        )
        # quark.Waypoint15 consumes a flat (1, C*H*W) latent — Patchify
        # does the spatial reshape internally. The VAE wants (1, C, H, W),
        # so we track both shapes and reshape at the boundary.
        pixel_h, pixel_w = cfg.height * pH, cfg.width * pW
        self._pixel_shape = (1, cfg.channels, pixel_h, pixel_w)
        self._flat_shape = (1, cfg.channels * pixel_h * pixel_w)
        # Kept for backwards-compat with anything reading ``engine.frm_shape``.
        self.frm_shape = (1, 1, cfg.channels, pixel_h, pixel_w)

        # Frame-rate mult retained for parity with the torch path.
        latent_fps = self.we_cfg.inference_fps / self.we_cfg.temporal_compression
        assert self.we_cfg.base_fps % latent_fps == 0
        self.ts_mult = int(self.we_cfg.base_fps // latent_fps)
        self._frame_counter = 0

        # Stable noise + output buffers for the graph hot path. torch
        # owns the memory; quark borrows the pointer. Noise buffer is
        # flat (quark input shape); output buffer is 4D (VAE input).
        # Refilling noise via ``torch.randn(out=...)`` keeps the borrowed
        # QuarkTensor's pointer graph-stable.
        self._noise_torch = torch.empty(self._flat_shape, device=self.device, dtype=self.dtype)
        self._noise_qt = _qt_borrow(self._noise_torch)
        self._latent_out_torch = torch.empty(self._pixel_shape, device=self.device, dtype=self.dtype)

    # ── Public API (matches WorldEngine's forwarded methods) ─────

    def reset(self) -> None:
        """Reset KV caches, frame counter, and VAE state."""
        self.gen.reset()
        self._frame_counter = 0
        if hasattr(self.vae, "reset"):
            self.vae.reset()

    def set_prompt(self, prompt: str) -> None:
        # TODO: port ``CrossAttention`` to ``quark.nn`` + wire prompt_emb
        # through ``Waypoint15``. For now, ignored.
        warnings.warn(
            "QuarkBackend.set_prompt: ignored — prompt cross-attention not "
            "yet ported to quark. Switch to backend='torch' for prompted "
            "inference.",
            RuntimeWarning,
            stacklevel=2,
        )

    @torch.inference_mode()
    def append_frame(self, img: torch.Tensor, ctrl=None) -> torch.Tensor:
        """VAE-encode ``img``, run a commit-only pass (no denoise) to
        write this frame's KV, then VAE-decode and return the image."""
        x0_pixel = self.vae.encode(img)  # (1, C, H, W) for VAE round-trip
        # quark.Waypoint15 wants the flat (1, C*H*W) layout — Patchify
        # does the spatial reshape internally.
        x0_flat = x0_pixel.reshape(self._flat_shape).contiguous()
        x0_qt = _qt_borrow(x0_flat)
        ctrl_qt = self._encode_ctrl(ctrl)
        ctrl_emb = self.model.encode_ctrl(ctrl_qt)
        self.model(
            x0_qt,
            sigma_idx=len(self.cfg.scheduler_sigmas) - 1,
            frame_t=self.gen.frame_t,
            ctrl_emb=ctrl_emb,
            frozen=False,
        )
        self.gen.frame_t.increment()
        self._frame_counter += 1
        return self.vae.decode(x0_pixel)

    @torch.inference_mode()
    def gen_frame(self, ctrl=None, return_img: bool = True):
        """Refill the stable noise buffer, run ``GenerateFrame`` (denoise
        + commit + ft++) via a captured CUDA graph, D2D-copy the latent
        into a torch buffer for VAE decode. First call captures the graph
        via ``gen.prepare_graph(...)``.
        """
        # Refill in place. torch.randn is a native CUDA kernel (µs); the
        # borrowed QuarkTensor keeps the same pointer, so the capture
        # buffer stays graph-stable across frames.
        torch.randn(self._flat_shape, out=self._noise_torch)
        ctrl_qt = self._encode_ctrl(ctrl)

        if not self._graph_ready:
            self.gen.prepare_graph(self._noise_qt, ctrl_qt, start_frame_t=self._frame_counter)
            self._graph_ready = True

        latent_qt = self.gen(self._noise_qt, ctrl_qt)
        self._frame_counter += 1

        # QuarkTensor → torch: D2D memcpy into the pre-allocated torch
        # output buffer. No host round-trip.
        latent_qt.copy_into_ptr(self._latent_out_torch.data_ptr())

        # self._latent_out_torch is (1, C, H, W) — the VAE's expected shape.
        return self.vae.decode(self._latent_out_torch) if return_img else self._latent_out_torch

    def get_state(self):
        raise NotImplementedError(
            "QuarkBackend.get_state: KV ring-buffer round-trip not implemented. "
            "Tracked as a v1 follow-up."
        )

    def load_state(self, state):
        raise NotImplementedError(
            "QuarkBackend.load_state: KV ring-buffer round-trip not implemented. "
            "Tracked as a v1 follow-up."
        )

    # ── Internals ────────────────────────────────────────────────

    def _encode_ctrl(self, ctrl):
        """Pack a ``CtrlInput`` into the stable ``[1, padded_in]`` bf16
        device buffer ``Waypoint15.encode_ctrl`` expects.

        Returns ``None`` when ``ctrl_conditioning`` is off. ``ctrl_fill``
        (from ``model.make_ctrl_buffer()``) writes in place on the
        pre-allocated device buffer so the graph-captured input pointer
        stays stable across frames.
        """
        if self._ctrl_fill is None:
            return None
        if ctrl is None:
            from ..world_engine import CtrlInput

            ctrl = CtrlInput()
        return self._ctrl_fill(ctrl)
