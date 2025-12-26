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
        self.device, self.dtype = device, dtype

        self.model_cfg = WorldModel.load_config(model_uri)

        model_config_overrides["global_attn_offset"] = False  # TODO, remove hack

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
        # Static input context tensors
        self._ctx = {
            "button": torch.zeros((1, 1, self.model_cfg.n_buttons), device=device, dtype=dtype),
            "mouse": torch.zeros((1, 1, 2), device=device, dtype=dtype),
            "frame_timestamp": torch.empty((1, 1), device=device, dtype=torch.long),
        }

    @torch.inference_mode()
    def reset(self):
        """Reset state for new generation"""
        self.kv_cache.reset()
        self.frame_ts.zero_()
        for v in self._ctx.values():
            v.zero_()

    def set_prompt(self, prompt: str):
        """Apply text conditioning for T2V"""
        import warnings
        warnings.warn("Not Implemented")

    @torch.inference_mode()
    def append_frame(self, img: Tensor, ctrl: CtrlInput = None):
        assert img.dtype == torch.uint8, img.dtype
        x0 = self.vae.encode(img).unsqueeze(1)
        inputs = self._prep_inputs(x=x0, ctrl=ctrl)
        self._cache_pass(x0, inputs, self.kv_cache)
        return img

    @torch.inference_mode()
    def gen_frame(self, ctrl: CtrlInput = None, return_img: bool = True):
        x = torch.randn(self.frm_shape, device=self.device, dtype=self.dtype)
        inputs = self._prep_inputs(x=x, ctrl=ctrl)
        x0 = self._denoise_pass(x, inputs, self.kv_cache).clone()
        self._cache_pass(x0, inputs, self.kv_cache)
        return (self.vae.decode(x0.squeeze(1)) if return_img else x0.squeeze(1))

    @torch.compile
    def _prep_inputs(self, x, ctrl=None):
        ctrl = ctrl if ctrl is not None else CtrlInput()
        self._ctx["button"].zero_()
        if ctrl.button:
            idx = torch.as_tensor(list(ctrl.button), device=self._ctx["button"].device, dtype=torch.long)
            self._ctx["button"][..., idx] = 1.0

        self._ctx["mouse"][0, 0, 0] = ctrl.mouse[0]
        self._ctx["mouse"][0, 0, 1] = ctrl.mouse[1]

        self._ctx["frame_timestamp"].copy_(self.frame_ts)
        self.frame_ts.add_(1)
        return self._ctx

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
        """Side effect: updates kv cache"""
        kv_cache.set_frozen(False)
        self.model(x, x.new_zeros((x.size(0), x.size(1))), **ctx, kv_cache=kv_cache)


# TODO
# - RoPE for inference
