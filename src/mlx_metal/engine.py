"""
MLXWorldEngine — WorldEngine subclass for Apple Silicon.

World model runs on MLX (Metal GPU with W8A8 NAX acceleration).
TAEHV encoder/decoder runs on ANE via CoreML (0% GPU).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

import mlx.core as mx
import numpy as np
import torch

from ..world_engine import WorldEngine, CtrlInput
from ..ae import get_ae
from .mlx_world_model import load_from_pytorch, compute_rope_angles


class MLXWorldEngine(WorldEngine):
    """WorldEngine on Apple Silicon: MLX world model + ANE TAEHV.

    Usage:
        engine = MLXWorldEngine(model_uri, ane_vae=True)
        engine.append_frame(seed_img)
        for ctrl in controls:
            img = engine.gen_frame_pipelined(ctrl)
            if img is not None: display(img)
        img = engine.flush_pipeline()
    """

    def __init__(
        self,
        model_uri: str,
        int8_profile: Optional[str] = "speed",
        kv_cache_mode: str = "fp16",
        attention_mode: str = "fp16",
        ane_vae: bool = True,
        model_config_overrides=None,
    ):
        # Base class sets up model_cfg, frm_shape, frame_ts, scheduler_sigmas, _ctx, vae, etc.
        # load_weights=False avoids loading PyTorch world model weights we'd immediately discard.
        super().__init__(model_uri, model_config_overrides=model_config_overrides, load_weights=False)

        # Replace VAE with ANE version (exports CoreML models on first run)
        if ane_vae:
            pH, pW = self.model_cfg.patch
            self.vae = get_ae(
                self.model_cfg.ae_uri,
                is_taehv_ae=self.model_cfg.taehv_ae,
                auto_aspect_ratio=self.model_cfg.auto_aspect_ratio,
                ane=True,
                dtype=torch.float32,
                height=self.model_cfg.height * pH,
                width=self.model_cfg.width * pW,
            )

        # Load MLX world model (replaces the PyTorch model from super().__init__)
        self.mlx_model, _ = load_from_pytorch(
            model_uri, int8_profile=int8_profile,
            kv_cache_mode=kv_cache_mode, attention_mode=attention_mode,
        )

        # Pipelined decode state
        self._decode_executor = ThreadPoolExecutor(max_workers=1) if ane_vae else None
        self._pending_decode: Optional[Future] = None

    def reset(self):
        """Reset all state for new generation."""
        self.flush_pipeline()
        for kv in self.mlx_model.kv_caches:
            kv.keys = mx.zeros_like(kv.keys)
            kv.values = mx.zeros_like(kv.values)
            kv.written_slots.clear()
        self.frame_ts.zero_()
        self.vae.reset()

    def _rope(self, frame_idx: int):
        return compute_rope_angles(
            frame_idx, self.mlx_model.ts_mult,
            self.mlx_model.rope_xy, self.mlx_model.rope_inv_t,
        )

    def _ctrl_to_mlx(self, ctrl: CtrlInput):
        mouse = mx.array([[list(ctrl.mouse)]], dtype=mx.float16)
        button = mx.zeros((1, 1, self.model_cfg.n_buttons), dtype=mx.float16)
        if ctrl.button:
            btn_list = list(ctrl.button)
            button_np = np.zeros((1, 1, self.model_cfg.n_buttons), dtype=np.float16)
            for b in btn_list:
                button_np[0, 0, b] = 1.0
            button = mx.array(button_np)
        scroll = mx.array([[[ctrl.scroll_wheel]]], dtype=mx.float16)
        return mouse, button, scroll

    def append_frame(self, img, ctrl: CtrlInput = None):
        ctrl = ctrl or CtrlInput()
        assert img.dtype == torch.uint8

        with torch.inference_mode():
            latent_pt = self.vae.encode(img)

        frame_idx = self.frame_ts.item()
        rope_cos, rope_sin = self._rope(frame_idx)
        mouse, button, scroll = self._ctrl_to_mlx(ctrl)

        latent_mx = mx.array(latent_pt.float().numpy()).astype(mx.float16)
        latent_mx = mx.reshape(latent_mx, self._mlx_latent_shape())

        self.mlx_model.cache_write(latent_mx, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        self.frame_ts.add_(1)
        return img

    def gen_frame(self, ctrl: CtrlInput = None, return_img: bool = True):
        ctrl = ctrl or CtrlInput()
        frame_idx = self.frame_ts.item()
        rope_cos, rope_sin = self._rope(frame_idx)
        mouse, button, scroll = self._ctrl_to_mlx(ctrl)

        # Denoise (4 Euler steps on MLX/Metal)
        shape = self._mlx_latent_shape()
        x = mx.array(np.random.randn(*shape).astype(np.float16))
        x0 = self.mlx_model.denoise(x, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        mx.eval(x0)

        # Cache write
        self.mlx_model.cache_write(x0, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        self.frame_ts.add_(1)

        if not return_img:
            return self._mlx_to_torch_latent(x0)

        # Decode
        latent_pt = self._mlx_to_torch_latent(x0)
        with torch.inference_mode():
            return self.vae.decode(latent_pt)

    def gen_frame_pipelined(self, ctrl: CtrlInput = None):
        """Pipelined: MLX denoise on GPU, ANE decode in background thread."""
        if self._decode_executor is None:
            return self.gen_frame(ctrl, return_img=True)

        ctrl = ctrl or CtrlInput()
        frame_idx = self.frame_ts.item()
        rope_cos, rope_sin = self._rope(frame_idx)
        mouse, button, scroll = self._ctrl_to_mlx(ctrl)

        # MLX denoise (GPU)
        shape = self._mlx_latent_shape()
        x = mx.array(np.random.randn(*shape).astype(np.float16))
        x0 = self.mlx_model.denoise(x, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        mx.eval(x0)

        # Collect previous ANE decode (should already be done — 17ms ANE < 147ms GPU)
        prev_img = None
        if self._pending_decode is not None:
            prev_img = self._pending_decode.result()
            self._pending_decode = None

        # Submit ANE decode BEFORE cache_write so they overlap on GPU + ANE
        latent_pt = self._mlx_to_torch_latent(x0)
        self._pending_decode = self._decode_executor.submit(self.vae.decode, latent_pt)

        # Cache write (GPU) — runs in parallel with ANE decode above
        self.mlx_model.cache_write(x0, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        self.frame_ts.add_(1)

        return prev_img

    def flush_pipeline(self):
        """Drain pending pipelined decode."""
        if self._pending_decode is not None:
            result = self._pending_decode.result()
            self._pending_decode = None
            return result
        return None

    def _mlx_latent_shape(self):
        _, _, c, h, w = self.frm_shape
        return (1, 1, c, h, w)

    def _mlx_to_torch_latent(self, x0: mx.array) -> torch.Tensor:
        return torch.from_numpy(np.array(x0.squeeze(0))).to(dtype=torch.float32)
