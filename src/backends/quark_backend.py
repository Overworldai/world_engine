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
    """Map ``world_engine`` OmegaConf config → ``Waypoint15Config``.

    Only fields actually present on the yaml are forwarded — the rest
    fall back to ``Waypoint15Config`` dataclass defaults, which match
    the implicit defaults baked into the torch world_engine code.
    """
    from quark.models.waypoint_15 import Waypoint15Config

    _SCALAR_FIELDS = (
        "d_model", "n_layers", "n_heads", "n_kv_heads", "mlp_ratio",
        "channels", "height", "width",
        "local_window", "global_window", "global_pinned_dilation",
        "global_attn_period", "global_attn_offset",
        "value_residual", "fourier_dim",
        "n_buttons", "ctrl_conditioning_period",
    )
    kwargs = {name: we_cfg[name] for name in _SCALAR_FIELDS if name in we_cfg}
    if "patch" in we_cfg:
        kwargs["patch"] = tuple(we_cfg.patch)
    if "scheduler_sigmas" in we_cfg:
        kwargs["scheduler_sigmas"] = tuple(we_cfg.scheduler_sigmas)
    if "ctrl_conditioning" in we_cfg:
        kwargs["ctrl_conditioning"] = we_cfg.ctrl_conditioning is not None
    # TODO: confirm whether the Waypoint-1.5 ckpt is f16 or bf16.
    kwargs["use_f16"] = False
    return Waypoint15Config(**kwargs)


def _remap_state_dict(raw_sd: dict, cfg) -> dict:
    """Transform a world_engine-saved state dict into the layout quark's
    ``Waypoint15`` expects (merged ``qkv_proj``, split ``ctrl_fusion``,
    per-block ``attn_cond``/``mlp_cond`` heads, etc.).

    CUDA-only copy of ``scripts/generate.py::remap_state_dict`` in the
    quark repo. TODO: factor the canonical version into a shared
    ``quark.models.convert`` module and import from there.
    """
    from quark.runtime.tensor import QuarkTensor

    sd: dict = {}
    d = cfg.d_model
    ph, pw = cfg.patch
    C = cfg.channels

    w = raw_sd["patchify.weight"]
    sd["patchify.weight"] = w.reshape(d, C * ph * pw) if w.ndim == 4 else w

    w = raw_sd["unpatchify.weight"]
    if w.ndim == 4:
        w = w.permute(1, 2, 3, 0).reshape(C * ph * pw, d)
    sd["unpatchify.weight"] = w

    b = raw_sd["unpatchify.bias"]
    if int(b.shape[0]) == C:
        # Tile [C] → [C*ph*pw] by repeating each element ph*pw times.
        import struct as _struct

        b_f32 = b.astype("f32")
        b_vals = list(_struct.unpack(f"<{C}f", b_f32.to_bytes()))
        tiled = []
        for v in b_vals:
            tiled.extend([v] * (ph * pw))
        b = QuarkTensor.from_list(tiled, dtype="f32").astype(raw_sd["unpatchify.bias"].dtype)
    sd["unpatchify.bias"] = b

    sd["noise_fc1.weight"] = raw_sd["denoise_step_emb.mlp.fc1.weight"]
    sd["noise_fc2.weight"] = raw_sd["denoise_step_emb.mlp.fc2.weight"]
    sd["out_norm_proj.weight"] = raw_sd["out_norm.fc.weight"]

    for i in range(cfg.n_layers):
        p = f"transformer.blocks.{i}."

        if p + "cond_head.bias_in" in raw_sd:
            sd[f"blocks.{i}.attn_cond_bias_in"] = raw_sd[p + "cond_head.bias_in"]
            sd[f"blocks.{i}.mlp_cond_bias_in"] = raw_sd[p + "cond_head.bias_in"]
            for j in range(6):
                sd[f"blocks.{i}.cond_projs.{j}.weight"] = raw_sd[
                    p + f"cond_head.cond_proj.{j}.weight"
                ]
        else:
            sd[f"blocks.{i}.attn_cond_bias_in"] = raw_sd[p + "attn_cond_head.bias_in"]
            sd[f"blocks.{i}.mlp_cond_bias_in"] = raw_sd[p + "mlp_cond_head.bias_in"]
            for j in range(3):
                sd[f"blocks.{i}.cond_projs.{j}.weight"] = raw_sd[
                    p + f"attn_cond_head.cond_proj.{j}.weight"
                ]
                sd[f"blocks.{i}.cond_projs.{j+3}.weight"] = raw_sd[
                    p + f"mlp_cond_head.cond_proj.{j}.weight"
                ]

        q_w = raw_sd[p + "attn.q_proj.weight"]
        k_w = raw_sd[p + "attn.k_proj.weight"]
        v_w = raw_sd[p + "attn.v_proj.weight"]
        sd[f"blocks.{i}.qkv_proj.weight"] = QuarkTensor.cat([q_w, k_w, v_w], dim=0)
        sd[f"blocks.{i}.out_proj.weight"] = raw_sd[p + "attn.out_proj.weight"]

        fc1 = p + ("mlp.fc1.weight" if p + "mlp.fc1.weight" in raw_sd else "dit_mlp.fc1.weight")
        fc2 = p + ("mlp.fc2.weight" if p + "mlp.fc2.weight" in raw_sd else "dit_mlp.fc2.weight")
        sd[f"blocks.{i}.mlp.fc1.weight"] = raw_sd[fc1]
        sd[f"blocks.{i}.mlp.fc2.weight"] = raw_sd[fc2]

        if cfg.value_residual and p + "attn.v_lamb" in raw_sd:
            lamb = raw_sd[p + "attn.v_lamb"]
            if lamb.ndim == 0:
                lamb = lamb.reshape(1)
            sd[f"blocks.{i}.v_residual.lamb"] = lamb.astype("f32")

        if cfg.ctrl_conditioning:
            fc1_x_key = p + "ctrl_mlpfusion.fc1_x.weight"
            fc1_c_key = p + "ctrl_mlpfusion.fc1_c.weight"
            fc2_key = p + "ctrl_mlpfusion.fc2.weight"
            if fc1_x_key in raw_sd:
                sd[f"blocks.{i}.ctrl_fusion.fc1_x.weight"] = raw_sd[fc1_x_key]
                sd[f"blocks.{i}.ctrl_fusion.fc1_c.weight"] = raw_sd[fc1_c_key]
                sd[f"blocks.{i}.ctrl_fusion.fc2.weight"] = raw_sd[fc2_key]
            elif p + "ctrl_mlpfusion.mlp.fc1.weight" in raw_sd:
                fc1_cat = raw_sd[p + "ctrl_mlpfusion.mlp.fc1.weight"]
                d_model = int(fc1_cat.shape[1]) // 2
                sd[f"blocks.{i}.ctrl_fusion.fc1_x.weight"] = fc1_cat[:, :d_model]
                sd[f"blocks.{i}.ctrl_fusion.fc1_c.weight"] = fc1_cat[:, d_model:]
                sd[f"blocks.{i}.ctrl_fusion.fc2.weight"] = raw_sd[
                    p + "ctrl_mlpfusion.mlp.fc2.weight"
                ]

    ctrl_fc1_key = "ctrl_emb.mlp.fc1.weight"
    if cfg.ctrl_conditioning and ctrl_fc1_key in raw_sd:
        fc1_w = raw_sd[ctrl_fc1_key]  # [d_mid, n_buttons+3]
        # Pad columns to a multiple of 16 for GEMM tile alignment.
        raw_k = int(fc1_w.shape[1])
        padded_k = ((raw_k + 15) // 16) * 16
        if padded_k > raw_k:
            w_np = (fc1_w.astype("f32") if fc1_w.dtype != "f32" else fc1_w).to_numpy()
            padded = np.zeros((w_np.shape[0], padded_k), dtype=np.float32)
            padded[:, :raw_k] = w_np
            half_dt = "f16" if cfg.use_f16 else "bf16"
            fc1_w = QuarkTensor.from_numpy(padded, dtype=half_dt)
        sd["ctrl_emb.fc1.weight"] = fc1_w
        sd["ctrl_emb.fc2.weight"] = raw_sd["ctrl_emb.mlp.fc2.weight"]

    return sd


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
            raw = load_safetensors(_resolve_safetensors_path(model_uri), dtype="bf16")
            self.model.load_state_dict(_remap_state_dict(raw, cfg), strict=False)

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

        # Stable noise + output buffers for the graph hot path. torch
        # owns the memory; quark borrows the pointer. Refill noise every
        # frame via ``torch.randn(out=self._noise_torch)`` so the capture
        # buffer stays at a fixed address (graph-stable).
        self._noise_torch = torch.empty(self.frm_shape, device=self.device, dtype=self.dtype)
        self._noise_qt = _qt_borrow(self._noise_torch)
        self._latent_out_torch = torch.empty(self.frm_shape, device=self.device, dtype=self.dtype)

        # Stable ctrl device buffer (bf16). Filled in place per frame via
        # memcpy_htod so the graph captures a fixed pointer.
        if cfg.ctrl_conditioning:
            from quark.runtime.tensor import QuarkTensor

            self._padded_in = self.model.ctrl_emb._padded_in
            self._n_buttons = cfg.n_buttons
            self._ctrl_host = np.zeros((1, self._padded_in), dtype=np.float32)
            self._ctrl_staged = np.zeros((1, self._padded_in), dtype=np.uint16)
            self._ctrl_dev = QuarkTensor.zeros(1, self._padded_in, dtype="bf16")
        else:
            self._ctrl_host = None
            self._ctrl_dev = None

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
        """Sample noise on torch (faster RNG) into a stable borrowed
        buffer, run ``GenerateFrame`` (denoise + commit + ft++) via a
        captured CUDA graph, D2D-copy the latent into a torch buffer
        for VAE decode.

        First call does the eager warmup + ``gen.graph(...)`` capture
        dance (see ``scripts/generate.py``): run one frame eagerly so
        every cached output buffer is allocated, rewind ``frame_t``,
        capture, rewind again — otherwise cuMemAllocAsync nodes bake
        into the graph at addresses that desync on replay.
        """
        from quark.runtime.cuda import CudaRuntime

        # Refill the stable noise buffer in place. torch.randn on CUDA
        # is a native kernel (~µs), much faster than QuarkTensor.randn's
        # scalar-Python-loop RNG — and refilling in place keeps the
        # borrowed QuarkTensor's pointer graph-stable.
        torch.randn(self.frm_shape, out=self._noise_torch)
        ctrl_qt = self._encode_ctrl(ctrl)

        if not self._graph_ready:
            # Eager warmup — ensures every cached output buffer in the
            # model tree is already allocated before capture.
            _ = self.gen(self._noise_qt, ctrl_qt)
            CudaRuntime.instance().stream_synchronize(0)
            # Warmup bumped frame_t; rewind to the real value.
            self.gen.set_frame_t(self._frame_counter)
            self.gen.graph(self._noise_qt, ctrl_qt)
            # Capture ran one forward pass → bumped again.
            self.gen.set_frame_t(self._frame_counter)
            CudaRuntime.instance().stream_synchronize(0)
            self._graph_ready = True

        # Replay: D2D-copies inputs into the stable capture buffers,
        # replays, returns a cloned output QuarkTensor.
        latent_qt = self.gen(self._noise_qt, ctrl_qt)
        self._frame_counter += 1

        # QuarkTensor → torch: D2D memcpy into our pre-allocated torch
        # output buffer. No host round-trip.
        nbytes = (
            self._latent_out_torch.numel() * self._latent_out_torch.element_size()
        )
        CudaRuntime.instance().memcpy_dtod(
            self._latent_out_torch.data_ptr(), latent_qt.data_ptr(), nbytes
        )

        x0 = self._latent_out_torch.squeeze(1)
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
        """Pack a ``CtrlInput`` into the stable ``[1, padded_in]`` bf16
        device buffer ``Waypoint15.encode_ctrl`` expects.

        Layout matches ``ControllerInputEmbedding.forward`` in
        ``world_engine.model.world_model``:
            ``cat((mouse[2], button[n_buttons], scroll[1]), dim=-1)``
        padded up to ``padded_in``. Writes happen in place on the
        pre-allocated ``self._ctrl_dev`` so the graph-captured input
        pointer stays stable across frames.
        """
        if self._ctrl_dev is None:
            return None
        if ctrl is None:
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

        # f32 → bf16: truncate each 32-bit word to its high 16 bits.
        self._ctrl_staged[:] = (buf.view(np.uint32) >> 16).astype(np.uint16)

        from quark.runtime.cuda import CudaRuntime

        CudaRuntime.instance().memcpy_htod(
            self._ctrl_dev.data_ptr(),
            self._ctrl_staged.ctypes.data,
            self._ctrl_staged.nbytes,
        )
        return self._ctrl_dev
