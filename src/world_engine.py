from typing import Dict, Optional, Set, Tuple
import os
import torch
from torch import Tensor
from dataclasses import dataclass, field

from .model import WorldModel, StaticKVCache, PromptEncoder
from .ae import get_ae
from .patch_model import apply_inference_patches
from .quantize import quantize_model


# Global torch optimizations
torch._dynamo.config.recompile_limit = 64
torch.set_float32_matmul_precision("medium")  # low: bf16, medium: tf32, high: fp32

# fix graph break:
torch._dynamo.config.capture_scalar_outputs = True
# Avoid per-layer recompiles from static integer attrs like layer_idx on MPS mixed-compile paths.
torch._dynamo.config.allow_unspec_int_on_nn_module = True

COMPILE_OPTIONS = {
    "max_autotune": True,
    "coordinate_descent_tuning": True,
    "triton.cudagraphs": True,
    # Negligible improvement in throughput:
    # "epilogue_fusion": True,
    # "shape_padding": True,
}


@dataclass
class CtrlInput:
    button: Set[int] = field(default_factory=set)  # pressed button IDs
    mouse: Tuple[float, float] = (0.0, 0.0)  # (x, y) velocity
    scroll_wheel: int = 0  # bwd, stationary, or fwd -> (-1, 0, 1)


class WorldEngine:
    def __init__(
        self,
        model_uri: str,
        quant: Optional[str] = None,
        model_config_overrides: Optional[Dict] = None,
        device=None,
        dtype=torch.bfloat16,
        load_weights: bool = True,
        scheduler_steps: Optional[int] = None,
        cache_interval: int = 1,
    ):
        """
        model_uri: HF URI or local folder containing model.safetensors and config.yaml
        quant: None | w8a8 | nvfp4
        model_config_overrides: Dict to override model config values
        """
        self.device = torch.get_default_device() if device is None else device
        self.dtype = torch.get_default_dtype() if dtype is None else dtype
        if cache_interval <= 0:
            raise ValueError("cache_interval must be >= 1")
        self.cache_interval = int(cache_interval)
        self._gen_count = 0
        force_compile_metal = os.getenv("WORLD_FORCE_COMPILE_METAL", "0") == "1"
        hybrid_compile_metal = os.getenv("WORLD_HYBRID_COMPILE_METAL", "0") == "1"
        self._disable_compile = (
            str(self.device).startswith("mps")
            and os.getenv("WORLD_ATTENTION_BACKEND", "flex").lower() == "metal"
            and not force_compile_metal
        )

        self.model_cfg = WorldModel.load_config(model_uri)

        if model_config_overrides:
            self.model_cfg.merge_with(model_config_overrides)

        with torch.device(self.device):
            # Load Model / Modules
            self.vae = get_ae(self.model_cfg.ae_uri, getattr(self.model_cfg, "taehv_ae", False), dtype=dtype)

            self.prompt_encoder = None
            if self.model_cfg.prompt_conditioning is not None:
                pe_uri = getattr(self.model_cfg, "prompt_encoder_uri", "google/umt5-xl")
                self.prompt_encoder = PromptEncoder(pe_uri, dtype=dtype).eval()

            self.model = WorldModel.from_pretrained(
                model_uri, cfg=self.model_cfg, device=self.device, dtype=dtype, load_weights=load_weights
            ).eval()
            apply_inference_patches(self.model)
            if quant is not None:
                quantize_model(self.model, quant)

            self.kv_cache = StaticKVCache(self.model_cfg, batch_size=1, dtype=dtype)

            # Inference Scheduler
            self.scheduler_sigmas = torch.tensor(self.model_cfg.scheduler_sigmas, dtype=dtype)
            if scheduler_steps is not None:
                if scheduler_steps <= 0:
                    raise ValueError("scheduler_steps must be > 0 when provided")
                if scheduler_steps > int(self.scheduler_sigmas.numel()):
                    raise ValueError(
                        f"scheduler_steps={scheduler_steps} exceeds available "
                        f"{int(self.scheduler_sigmas.numel())}"
                    )
                self.scheduler_sigmas = self.scheduler_sigmas[: int(scheduler_steps)].contiguous()
            self.scheduler_dsigmas = self.scheduler_sigmas.diff().contiguous()
            self.scheduler_step_sigmas = self.scheduler_sigmas[:-1].contiguous()
            self._sigma_zero = torch.zeros((1, 1), dtype=dtype)

            pH, pW = getattr(self.model_cfg, "patch", [1, 1])
            self.frm_shape = 1, 1, self.model_cfg.channels, self.model_cfg.height * pH, self.model_cfg.width * pW

            # State
            inference_fps = getattr(self.model_cfg, "inference_fps", self.model_cfg.base_fps)
            latent_fps = inference_fps / getattr(self.model_cfg, "temporal_compression", 1)
            self.ts_mult = int(int(self.model_cfg.base_fps) // latent_fps)
            self.frame_ts = torch.tensor([[0]], dtype=torch.long)
            self._frame_idx_int = 0

            # Static input context tensors
            self._ctx = {
                "button": torch.zeros((1, 1, self.model_cfg.n_buttons), dtype=dtype),
                "mouse": torch.zeros((1, 1, 2), dtype=dtype),
                "scroll": torch.zeros((1, 1, 1), dtype=dtype),
                "frame_timestamp": torch.empty((1, 1), dtype=torch.long),
                "frame_idx": torch.empty((1, 1), dtype=torch.long),
            }

            self._prompt_ctx = {"prompt_emb": None, "prompt_pad_mask": None}
            metal_runtime = str(self.device).startswith("mps") and os.getenv("WORLD_ATTENTION_BACKEND", "flex").lower() == "metal"
            if (force_compile_metal or hybrid_compile_metal) and metal_runtime:
                # Allow graph breaks around custom Metal ops while still compiling dense surrounding math.
                self._cache_pass_fn = self._cache_pass_mixed
                self._denoise_pass_fn = self._denoise_pass_mixed
            else:
                self._cache_pass_fn = self._cache_pass_eager if self._disable_compile else self._cache_pass
                self._denoise_pass_fn = self._denoise_pass_eager if self._disable_compile else self._denoise_pass

    @torch.inference_mode()
    def reset(self):
        """Reset state for new generation"""
        self.kv_cache.reset()
        self._gen_count = 0
        self._frame_idx_int = 0
        self.frame_ts.zero_()
        for v in self._ctx.values():
            v.zero_()
        self.vae.reset()

    @torch.inference_mode()
    def get_state(self):
        """Captures a world state to continue via load_state. Doesn't save model"""
        return {"kv_cache": self.kv_cache.get_state(), "frame_ts": self.frame_ts.detach().clone()}

    @torch.inference_mode()
    def load_state(self, state):
        """Loads a world state object saved via save_state. Doesn't load or change model"""
        self.kv_cache.load_state(state["kv_cache"])
        self.frame_ts.copy_(state["frame_ts"])
        self._frame_idx_int = int(self.frame_ts[0, 0].item())

    def set_prompt(self, prompt: str):
        """Apply text conditioning for T2V"""
        if self.prompt_encoder is None:
            raise RuntimeError("prompt_conditioning enabled but prompt_encoder is not initialized")
        self._prompt_ctx["prompt_emb"], self._prompt_ctx["prompt_pad_mask"] = self.prompt_encoder([prompt])

    @torch.inference_mode()
    def append_frame(self, img: Tensor, ctrl: CtrlInput = None):
        assert img.dtype == torch.uint8, img.dtype
        x0 = self.vae.encode(img).unsqueeze(1)
        inputs = self.prep_inputs(x=x0, ctrl=ctrl)
        self._cache_pass_fn(x0, inputs, self.kv_cache)
        return img

    @torch.inference_mode()
    def gen_frame(self, ctrl: CtrlInput = None, return_img: bool = True):
        x = torch.randn(self.frm_shape, device=self.device, dtype=self.dtype)
        inputs = self.prep_inputs(x=x, ctrl=ctrl)
        x0 = self._denoise_pass_fn(x, inputs, self.kv_cache)
        if (self._gen_count % self.cache_interval) == 0:
            self._cache_pass_fn(x0, inputs, self.kv_cache)
        self._gen_count += 1
        return (self.vae.decode(x0.squeeze(1)) if return_img else x0.squeeze(1))

    @torch.compile
    def _prep_inputs(self, mouse_x: float, mouse_y: float, scroll_wheel: float):
        self._ctx["mouse"][0, 0, 0] = mouse_x
        self._ctx["mouse"][0, 0, 1] = mouse_y
        self._ctx["scroll"][0, 0, 0] = scroll_wheel

        self._ctx["frame_idx"].copy_(self.frame_ts)
        self._ctx["frame_timestamp"].copy_(self.frame_ts).mul_(self.ts_mult)
        self.frame_ts.add_(1)

        return self._ctx

    def prep_inputs(self, x, ctrl=None):
        ctrl = ctrl if ctrl is not None else CtrlInput()
        self._ctx["button"].zero_()
        if ctrl.button:
            self._ctx["button"][..., list(ctrl.button)] = 1.0
        mx, my = ctrl.mouse
        mouse_x = float(mx)
        mouse_y = float(my)
        if ctrl.scroll_wheel > 0:
            scroll_wheel = 1.0
        elif ctrl.scroll_wheel < 0:
            scroll_wheel = -1.0
        else:
            scroll_wheel = 0.0
        ctx = self._prep_inputs(mouse_x, mouse_y, scroll_wheel)
        # Thread a cheap Python-side frame index hint to avoid per-layer scalar syncs.
        self.kv_cache.set_frame_idx_int(self._frame_idx_int)
        self._frame_idx_int += 1

        # prepare prompt conditioning
        if self.model_cfg.prompt_conditioning is None:
            return ctx
        if self._prompt_ctx["prompt_emb"] is None:
            self.set_prompt("An explorable world")
        return {**ctx, **self._prompt_ctx}

    @torch.compile(fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
    def _denoise_pass(self, x, ctx: Dict[str, Tensor], kv_cache):
        kv_cache.set_frozen(True)
        bt = (x.size(0), x.size(1))
        for i in range(self.scheduler_dsigmas.numel()):
            sigma_bt = self.scheduler_step_sigmas[i].expand(bt)
            v = self.model(x, sigma_bt, **ctx, kv_cache=kv_cache, ctrl_cond=True, prompt_cond=True)
            x = x + self.scheduler_dsigmas[i] * v
        return x

    def _denoise_pass_eager(self, x, ctx: Dict[str, Tensor], kv_cache):
        kv_cache.set_frozen(True)
        bt = (x.size(0), x.size(1))
        for i in range(self.scheduler_dsigmas.numel()):
            sigma_bt = self.scheduler_step_sigmas[i].expand(bt)
            v = self.model(x, sigma_bt, **ctx, kv_cache=kv_cache, ctrl_cond=True, prompt_cond=True)
            x = x + self.scheduler_dsigmas[i] * v
        return x

    @torch.compile(dynamic=False, options=COMPILE_OPTIONS)
    def _denoise_pass_mixed(self, x, ctx: Dict[str, Tensor], kv_cache):
        kv_cache.set_frozen(True)
        bt = (x.size(0), x.size(1))
        for i in range(self.scheduler_dsigmas.numel()):
            sigma_bt = self.scheduler_step_sigmas[i].expand(bt)
            v = self.model(x, sigma_bt, **ctx, kv_cache=kv_cache, ctrl_cond=True, prompt_cond=True)
            x = x + self.scheduler_dsigmas[i] * v
        return x

    @torch.compile(fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
    def _cache_pass(self, x, ctx: Dict[str, Tensor], kv_cache):
        """Side effect: updates kv cache"""
        kv_cache.set_frozen(False)
        self.model(
            x, self._sigma_zero.expand((x.size(0), x.size(1))), **ctx, kv_cache=kv_cache, ctrl_cond=True, prompt_cond=True
        )

    def _cache_pass_eager(self, x, ctx: Dict[str, Tensor], kv_cache):
        """Side effect: updates kv cache"""
        kv_cache.set_frozen(False)
        self.model(
            x, self._sigma_zero.expand((x.size(0), x.size(1))), **ctx, kv_cache=kv_cache, ctrl_cond=True, prompt_cond=True
        )

    @torch.compile(dynamic=False, options=COMPILE_OPTIONS)
    def _cache_pass_mixed(self, x, ctx: Dict[str, Tensor], kv_cache):
        """Side effect: updates kv cache"""
        kv_cache.set_frozen(False)
        self.model(
            x, self._sigma_zero.expand((x.size(0), x.size(1))), **ctx, kv_cache=kv_cache, ctrl_cond=True, prompt_cond=True
        )
