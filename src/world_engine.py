from typing import Dict, Optional, Set, Tuple
import torch
from torch import Tensor
from dataclasses import dataclass, field

from owl_wms.models.world import WorldModel
from owl_wms.nn.kv_cache import StaticKVCache

from world_engine.ae import InferenceAE
from world_engine.patch_model import apply_inference_patches
from world_engine.quantize import quantize_model


# Global torch optimizations
torch._dynamo.config.recompile_limit = 64
torch.set_float32_matmul_precision("medium")  # low: bf16, medium: tf32, high: fp32

# fix graph break:
torch._dynamo.config.capture_scalar_outputs = True


@dataclass
class CtrlInput:
    button: Set[int] = field(default_factory=set)  # pressed button IDs
    mouse: Tuple[float, float] = (0.0, 0.0)  # (x, y) velocity


class WorldEngine:
    def __init__(
        self,
        model_uri: str,
        quant: Optional[str] = None,
        apply_patches: bool = True,
        model_config_overrides: Optional[Dict] = None,
        device=None,
        dtype=torch.bfloat16,
    ):
        """
        model_uri: HF URI or local folder containing model.safetensors and config.yaml
        quant: None | w8a8 | nvfp4
        """
        # Meta
        self.device, self.dtype = device, dtype
        self.model_cfg = WorldModel.load_config(model_uri)

        # TODO: remove these hardcoding hacks:
        self.model_cfg.mlp_gradient_checkpointing = getattr(self.model_cfg, "mlp_gradient_checkpointing", False)

        if model_config_overrides:
            self.model_cfg.merge_with(model_config_overrides)

        # Model
        self.vae = InferenceAE.from_pretrained(model_uri, device=device, dtype=dtype)
        # self.prompt_encoder = PromptEncoder("google/umt5-xl").to(device).eval()  # TODO: dont hardcode
        self.model = WorldModel.from_pretrained(model_uri, cfg=self.model_cfg).to(device=device, dtype=dtype).eval()
        if apply_patches:
            apply_inference_patches(self.model)
        if quant is not None:
            quantize_model(self.model, quant)

        # Inference Scheduler
        self.scheduler_sigmas = torch.tensor(self.model_cfg.scheduler_sigmas, device=device, dtype=dtype)

        pH, pW = getattr(self.model_cfg, "patch", [1, 1])
        self.frm_shape = 1, 1, self.model_cfg.channels, self.model_cfg.height * pH, self.model_cfg.width * pW

        # State
        self.kv_cache = StaticKVCache(self.model_cfg, max_seq_len=None, batch_size=1, dtype=dtype).to(device)
        self.frame_ts = torch.tensor([[0]], dtype=torch.long, device=device)

    @torch.inference_mode()
    def reset(self):
        """Reset state for new generation"""
        self.frame_ts.zero_()
        self.kv_cache.reset()  # in-place reset

    def set_prompt(self, prompt: str, timestamp: float = 0.0):
        """Apply text conditioning for T2V"""
        import warnings
        warnings.warn("Not Implemented")

    @torch.inference_mode()
    def append_frame(self, img: Tensor, ctrl: CtrlInput = None):
        assert img.dtype == torch.uint8, img.dtype
        x0 = self.vae.encode(img).unsqueeze(1)
        inputs = self._prep_inputs(x=x0, ctrl=ctrl)
        self.kv_cache = self._cache_pass(x0, inputs, self.kv_cache)
        return img

    @torch.inference_mode()
    def gen_frame(self, ctrl: CtrlInput = None, return_img: bool = True):
        x = torch.randn(self.frm_shape, device=self.device, dtype=self.dtype)
        inputs = self._prep_inputs(x=x, ctrl=ctrl)
        x0 = self._denoise_pass(x, inputs, self.kv_cache).clone()
        self.kv_cache = self._cache_pass(x0, inputs, self.kv_cache)
        with torch.amp.autocast('cuda', torch.bfloat16):
            x0 = x0.squeeze(1)
            return (self.vae.decode(x0) if return_img else x0)

    def _prep_inputs(self, x, ctrl=None):
        ctrl = ctrl if ctrl is not None else CtrlInput()
        button = x.new_zeros(1, 1, self.model_cfg.n_buttons)
        button[..., x.new_tensor(tuple(ctrl.button or ()), dtype=torch.long)] = 1.0
        mouse = x.new_tensor(ctrl.mouse, dtype=self.dtype)[None, None]
        out = {"button": button, "mouse": mouse, "frame_timestamp": self.frame_ts.clone()}
        self.frame_ts += 1
        return out

    @torch.compile(fullgraph=True, mode="max-autotune", dynamic=False)
    def _denoise_pass(self, x, ctx: Dict[str, Tensor], kv_cache):
        kv_cache.set_frozen(True)
        sigma = x.new_empty((x.size(0), x.size(1)))
        for step_sig, step_dsig in zip(self.scheduler_sigmas, self.scheduler_sigmas.diff()):
            v = self.model(x, sigma.fill_(step_sig), **ctx, kv_cache=kv_cache)
            x = x + step_dsig * v
        return x

    @torch.compile(fullgraph=True, mode="max-autotune", dynamic=False)
    def _cache_pass(self, x, ctx: Dict[str, Tensor], kv_cache):
        kv_cache.set_frozen(False)
        self.model(x, x.new_zeros((x.size(0), x.size(1))), **ctx, kv_cache=kv_cache)
        return kv_cache


# TODO
# - RoPE for inference
