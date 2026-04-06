from typing import Dict, Optional, Set, Tuple
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
    mouse: Tuple[float, float] = (0.0, 0.0)  # (dx, dy) velocity
    scroll_wheel: int = 0  # down, stationary, or up -> (-1, 0, 1)


class WorldEngine:
    def __init__(
        self,
        model_uri: str,
        quant: Optional[str] = None,
        model_config_overrides: Optional[Dict] = None,
        device=None,
        dtype=torch.bfloat16,
        load_weights: bool = True
    ):
        """
        model_uri: HF URI or local folder containing model.safetensors and config.yaml
        quant: None | w8a8 | nvfp4

        model_config_overrides: Dict to override model config values
        - auto_aspect_ratio: set to False to work in ae raw space, otherwise in/out are 720p or 360p
        """
        self.device = torch.get_default_device() if device is None else device
        self.dtype = torch.get_default_dtype() if dtype is None else dtype

        self.model_cfg = WorldModel.load_config(model_uri)

        if model_config_overrides:
            self.model_cfg.merge_with(model_config_overrides)

        with torch.device(self.device):
            # Load Model / Modules
            self.vae = get_ae(
                self.model_cfg.ae_uri,
                is_taehv_ae=self.model_cfg.taehv_ae,
                auto_aspect_ratio=self.model_cfg.auto_aspect_ratio,
                dtype=dtype,
                device=device,
            )

            self.prompt_encoder = None
            if self.model_cfg.prompt_conditioning is not None:
                self.prompt_encoder = PromptEncoder(self.model_cfg.prompt_encoder_uri, dtype=dtype).eval()

            self.model = WorldModel.from_pretrained(
                model_uri, cfg=self.model_cfg, device=self.device, dtype=dtype, load_weights=load_weights
            ).eval()
            apply_inference_patches(self.model)
            if quant is not None:
                quantize_model(self.model, quant)

            self.kv_cache = StaticKVCache(self.model_cfg, batch_size=1, dtype=dtype).to(device=device)

            # Inference Scheduler
            self.scheduler_sigmas = torch.tensor(self.model_cfg.scheduler_sigmas, dtype=dtype, device=device)

            pH, pW = self.model_cfg.patch
            self.frm_shape = 1, 1, self.model_cfg.channels, self.model_cfg.height * pH, self.model_cfg.width * pW

            # State
            latent_fps = self.model_cfg.inference_fps / self.model_cfg.temporal_compression
            self.ts_mult = int(self.model_cfg.base_fps) // latent_fps
            self.frame_ts = torch.tensor([[0]], dtype=torch.long)

            # Static input context tensors
            self._ctx = {
                "button": torch.zeros((1, 1, self.model_cfg.n_buttons), dtype=dtype),
                "mouse": torch.zeros((1, 1, 2), dtype=dtype),
                "scroll": torch.zeros((1, 1, 1), dtype=dtype),
                "frame_timestamp": torch.empty((1, 1), dtype=torch.long),
                "frame_idx": torch.empty((1, 1), dtype=torch.long),
            }

            self._prompt_ctx = {"prompt_emb": None, "prompt_pad_mask": None}

    @torch.inference_mode()
    def reset(self):
        """Reset state for new generation"""
        self.kv_cache.reset()
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
        self._cache_pass(x0, inputs, self.kv_cache)
        return img

    @torch.inference_mode()
    def gen_frame(self, ctrl: CtrlInput = None, return_img: bool = True):
        x = torch.randn(self.frm_shape, device=self.device, dtype=self.dtype)
        inputs = self.prep_inputs(x=x, ctrl=ctrl)
        x0 = self._denoise_pass(x, inputs, self.kv_cache).clone()
        self._cache_pass(x0, inputs, self.kv_cache)
        return (self.vae.decode(x0.squeeze(1)) if return_img else x0.squeeze(1))

    @torch.compile
    def _prep_inputs(self, x, ctrl=None):
        self._ctx["button"].zero_()
        self._ctx["button"][..., ctrl.button] = 1.0

        self._ctx["mouse"][0, 0, 0] = ctrl.mouse[0]
        self._ctx["mouse"][0, 0, 1] = ctrl.mouse[1]

        self._ctx["scroll"][0, 0, 0] = ctrl.scroll_wheel

        self._ctx["frame_idx"].copy_(self.frame_ts)
        self._ctx["frame_timestamp"].copy_(self.frame_ts).mul_(self.ts_mult)
        self.frame_ts.add_(1)

        return self._ctx

    def prep_inputs(self, x, ctrl=None):
        ctrl = ctrl if ctrl is not None else CtrlInput()
        ctrl.button = torch.as_tensor(list(ctrl.button), dtype=torch.int64).to(x.device, non_blocking=True)
        ctrl.mouse = torch.as_tensor(ctrl.mouse).to(x.device, non_blocking=True)
        ctrl.scroll_wheel = torch.as_tensor(ctrl.scroll_wheel).to(x.device, non_blocking=True)
        ctx = self._prep_inputs(x, ctrl)

        # prepare prompt conditioning
        if self.model_cfg.prompt_conditioning is None:
            return ctx
        if self._prompt_ctx["prompt_emb"] is None:
            self.set_prompt("An explorable world")
        return {**ctx, **self._prompt_ctx}

    @torch.compile(fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
    def _denoise_pass(self, x, ctx: Dict[str, Tensor], kv_cache):
        """Run Deterministic Euler ODE Solver"""
        kv_cache.set_frozen(True)
        sigma = x.new_empty((x.size(0), x.size(1)))
        for step_sig, step_dsig in zip(self.scheduler_sigmas, self.scheduler_sigmas.diff()):
            v = self.model(x, sigma.fill_(step_sig), **ctx, kv_cache=kv_cache)
            x = (x.float() + step_dsig.float() * v.float()).type_as(x)
        return x

    @torch.compile(fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
    def _cache_pass(self, x, ctx: Dict[str, Tensor], kv_cache):
        """Side effect: updates kv cache"""
        kv_cache.set_frozen(False)
        self.model(x, x.new_zeros((x.size(0), x.size(1))), **ctx, kv_cache=kv_cache)
