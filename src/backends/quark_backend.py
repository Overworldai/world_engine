"""Torch-free DiT inference via the ``quark`` stack.

v1 scope — the torch VAE (taehv / InferenceAE) still runs for encode/
decode; we bridge to ``QuarkTensor`` at the latent boundary via a
zero-copy ``_BorrowedStorage`` wrapper. Only the DiT hot path (denoise
+ commit) runs on quark kernels.

Not supported in v1 (tracked TODOs):
- Prompt cross-attention: ``quark.nn`` has no ``CrossAttention`` yet.
  ``set_prompt`` warns + no-ops; prompted ckpts produce degraded output
  until the cross-attn port lands.
- ``get_state`` / ``load_state``: quark KV ring-buffer layout differs
  from ``StaticKVCache``; round-trip punted to follow-up.
- Quantization modes other than ``fp8w8a8`` (FP4 needs flashinfer,
  intw8a8 needs gemlite — both torch-only for now).
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Optional

import numpy as np
import torch


_TORCH_TO_QUARK_DTYPE = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
}


def _qt_borrow(t: torch.Tensor):
    """Wrap a contiguous CUDA torch tensor as a ``QuarkTensor`` (zero-copy).

    The borrowed storage holds a reference to the torch tensor so it
    stays alive for the QuarkTensor's lifetime.
    """
    from quark.runtime.tensor import (
        QuarkTensor,
        _BorrowedStorage,
        _contiguous_strides,
    )

    assert t.is_cuda, "QuarkBackend requires CUDA tensors"
    assert t.is_contiguous(), "borrowed tensor must be contiguous"
    nbytes = t.numel() * t.element_size()
    storage = _BorrowedStorage(t.data_ptr(), nbytes, owner=t)
    return QuarkTensor(
        storage,
        tuple(t.shape),
        _contiguous_strides(tuple(t.shape)),
        0,
        _TORCH_TO_QUARK_DTYPE[t.dtype],
    )


def _map_config(we_cfg):
    """Map ``world_engine`` OmegaConf config → ``Waypoint15Config``."""
    from quark.models.waypoint_15 import Waypoint15Config

    return Waypoint15Config(
        d_model=we_cfg.d_model,
        n_layers=we_cfg.n_layers,
        n_heads=we_cfg.n_heads,
        n_kv_heads=we_cfg.n_kv_heads,
        mlp_ratio=we_cfg.mlp_ratio,
        channels=we_cfg.channels,
        patch=tuple(we_cfg.patch),
        height=we_cfg.height,
        width=we_cfg.width,
        local_window=we_cfg.local_window,
        global_window=we_cfg.global_window,
        global_pinned_dilation=we_cfg.global_pinned_dilation,
        global_attn_period=we_cfg.global_attn_period,
        global_attn_offset=we_cfg.global_attn_offset,
        value_residual=we_cfg.value_residual,
        fourier_dim=we_cfg.fourier_dim,
        scheduler_sigmas=tuple(we_cfg.scheduler_sigmas),
        n_buttons=we_cfg.n_buttons,
        ctrl_conditioning=we_cfg.ctrl_conditioning is not None,
        ctrl_conditioning_period=we_cfg.ctrl_conditioning_period,
        # TODO: confirm whether the Waypoint-1.5 ckpt is f16 or bf16.
        use_f16=False,
    )


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
        from quark.models.waypoint_15 import GenerateFrame, Waypoint15
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
        cfg = _map_config(self.we_cfg)
        self.cfg = cfg
        self.model = Waypoint15(cfg)

        if load_weights:
            state = load_safetensors(_resolve_safetensors_path(model_uri), dtype="bf16")
            self.model.load_state_dict(state)

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
        self.frm_shape = (1, 1, cfg.channels, cfg.height * pH, cfg.width * pW)

        # Frame-rate mult retained for parity with the torch path. The
        # quark model owns ``frame_t`` on device; we keep a host mirror
        # for ``get_state``/``load_state`` follow-up work.
        latent_fps = self.we_cfg.inference_fps / self.we_cfg.temporal_compression
        assert self.we_cfg.base_fps % latent_fps == 0
        self.ts_mult = int(self.we_cfg.base_fps // latent_fps)
        self._frame_counter = 0

        # Ctrl staging buffer (host f32 → cast to bf16 on upload).
        if cfg.ctrl_conditioning:
            self._padded_in = self.model.ctrl_emb._padded_in
            self._n_buttons = cfg.n_buttons
            self._ctrl_host = np.zeros((1, self._padded_in), dtype=np.float32)
        else:
            self._ctrl_host = None

    # ── Public API (matches WorldEngine's forwarded methods) ─────

    def reset(self) -> None:
        """Reset KV caches, frame counter, and VAE state."""
        # TODO: expose a ``Waypoint15.reset()`` helper in quark so we
        # don't reach into ``blocks[i].kv_cache`` from the outside.
        for block in self.model.blocks:
            kvc = getattr(block, "kv_cache", None)
            if kvc is not None and hasattr(kvc, "reset"):
                kvc.reset()
        self.gen.set_frame_t(0)
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
        x0 = self.vae.encode(img).unsqueeze(1).contiguous()
        x0_qt = _qt_borrow(x0)
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
        return self.vae.decode(x0.squeeze(1))

    @torch.inference_mode()
    def gen_frame(self, ctrl=None, return_img: bool = True):
        """Sample noise on torch (faster RNG), borrow into QuarkTensor,
        run ``GenerateFrame`` (denoise + commit + ft++), VAE-decode."""
        x = torch.randn(self.frm_shape, device=self.device, dtype=self.dtype)
        x_qt = _qt_borrow(x)
        ctrl_qt = self._encode_ctrl(ctrl)

        # GenerateFrame runs denoise + commit + ft increment. The final
        # latent is written in place into ``x`` (borrowed), so the torch
        # side already owns the result.
        self.gen(x_qt, ctrl_qt)
        self._frame_counter += 1

        x0 = x.squeeze(1)
        return self.vae.decode(x0) if return_img else x0

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
        """Pack a ``CtrlInput`` into the ``[1, padded_in]`` bf16 tensor
        ``Waypoint15.encode_ctrl`` expects.

        Layout matches ``ControllerInputEmbedding.forward`` in
        ``world_engine.model.world_model``:
            ``cat((mouse[2], button[n_buttons], scroll[1]), dim=-1)``
        padded up to ``padded_in``.
        """
        if self._ctrl_host is None:
            return None
        if ctrl is None:
            # Import lazily to avoid circular import at module load.
            from ..world_engine import CtrlInput

            ctrl = CtrlInput()

        buf = self._ctrl_host
        buf.fill(0.0)
        buf[0, 0] = float(ctrl.mouse[0])
        buf[0, 1] = float(ctrl.mouse[1])
        for bid in ctrl.button:
            if 0 <= bid < self._n_buttons:
                buf[0, 2 + bid] = 1.0
        buf[0, 2 + self._n_buttons] = float(ctrl.scroll_wheel)

        # TODO: allocate ``QuarkTensor`` once and memcpy_htod in place —
        # ``from_numpy`` allocates a fresh device buffer every frame.
        from quark.runtime.tensor import QuarkTensor

        return QuarkTensor.from_numpy(buf, dtype="bf16")
