from typing import Dict, List, Optional, Set, Tuple
from torch import Tensor
from dataclasses import dataclass, field

import gc
import torch

from owl_wms.models.world import WorldModel
from owl_wms.nn.kv_cache import StaticKVCache

from world_engine.ae import InferenceAE


# Global torch optimizations
torch._dynamo.config.recompile_limit = 64
torch.set_float32_matmul_precision("medium")  # low: bf16, medium: tf32, high: fp32


@dataclass
class CtrlInput:
    button: Set[int] = field(default_factory=set)  # pressed button IDs
    mouse: Tuple[float, float] = (0.0, 0.0)  # (x, y) position


@dataclass
class InferenceConfig:
    scheduler_sigmas: List[float] = field(default_factory=lambda: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    noise_prev: float = 0.0


@dataclass
class OptimizationConfig:
    pass  # TODO


class WorldEngine:
    def __init__(
        self,
        model_uri: str,
        inference_config: Optional[InferenceConfig] = None,
        optimization_config: Optional[OptimizationConfig] = None,
        model_config_overrides: Optional[Dict] = None,
        device=None,
        dtype=torch.bfloat16,
    ):
        # Meta
        self.device, self.dtype = device, dtype
        self.inference_config = inference_config or InferenceConfig()
        self.model_cfg = WorldModel.load_config(model_uri)

        # TODO: remove these hardcoding hacks:
        self.model_cfg.n_buttons = self.model_cfg.n_controller_inputs - 2

        if model_config_overrides:
            self.model_cfg.merge_with(model_config_overrides)

        # Model
        self.model = WorldModel.from_pretrained(model_uri, cfg=self.model_cfg).to(device=device, dtype=dtype).eval()
        self.vae = InferenceAE.from_pretrained(model_uri, device=device, dtype=dtype)
        # self.prompt_encoder = PromptEncoder("google/umt5-xl").to(device).eval()  # TODO: dont hardcode

        # Inference Scheduler
        assert self.inference_config.noise_prev == 0.0, "Only zero-noise supported currently"
        self.scheduler_sigmas = torch.tensor(self.inference_config.scheduler_sigmas, device=device, dtype=dtype)

        # State
        self.uncached_buffer = self.kv_cache = self.frame_ts = None
        self.reset()

    @torch.inference_mode()
    def reset(self, collect_garbage: bool = True):
        """Reset state for new generation"""
        if collect_garbage:
            del self.kv_cache, self.uncached_buffer
            gc.collect()
            torch.cuda.empty_cache()

        cfg, dev, dt = self.model_cfg, self.device, self.dtype
        self.frame_ts = torch.tensor([[0]], dtype=torch.long, device=dev)
        self.uncached_buffer = {
            "x": torch.empty(1, 0, cfg.channels, cfg.height, cfg.width, device=dev, dtype=dt),
            "frame_timestamp": torch.empty(1, 0, device=dev, dtype=torch.long),
            "mouse": torch.empty(1, 0, 2, device=dev, dtype=dt),
            "button": torch.empty(1, 0, cfg.n_buttons, device=dev, dtype=dt),
        }
        self.kv_cache = StaticKVCache(cfg, max_seq_len=512, batch_size=1, dtype=dt).to(dev)

    @torch.inference_mode()
    def _push_frame_state(self, x, ctrl=None):
        ctrl = ctrl if ctrl is not None else CtrlInput()
        new_frame_state = {
            "button": torch.bincount(
                torch.tensor(list(ctrl.button), dtype=torch.long), minlength=self.model_cfg.n_buttons,
            ).to(x, non_blocking=True)[None, None],
            "mouse": x.new_tensor(ctrl.mouse)[None, None],
            "frame_timestamp": self.frame_ts,
            "x": x,
        }
        for k, v in new_frame_state.items():
            self.uncached_buffer[k] = torch.cat([self.uncached_buffer[k], v], dim=1)
        self.frame_ts = self.frame_ts + 1

    @torch.inference_mode()
    def append_frame(self, img: Tensor, ctrl: CtrlInput = None):
        assert img.dtype == torch.uint8
        # push frame inputs w/ clean latent
        self._push_frame_state(
            x=self.vae.encode(img.to(self.device)).to(self.dtype).unsqueeze(1),
            ctrl=ctrl
        )
        return img

    @torch.inference_mode()
    def gen_frame(self, ctrl: CtrlInput = None):
        shape = self.uncached_buffer["x"].shape
        # prepare frame inputs + random N latent
        self._push_frame_state(
            x=torch.randn(shape[0], 1, *shape[2:], device=self.device, dtype=self.dtype),
            ctrl=ctrl,
        )
        with torch.amp.autocast('cuda', torch.bfloat16):
            self.uncached_buffer, self.kv_cache = self.denoise_frame(self.uncached_buffer, self.kv_cache)
            img = self.vae.decode(self.uncached_buffer["x"][:, -1])  # decode last frame -> image
            return img

    def set_prompt(self, prompt: str, timestamp: float = 0.0):
        """Apply text conditioning for T2V"""
        import warnings
        warnings.warn("Not Implemented")

    @torch.compile
    def denoise_frame(self, uncached_buffer: Dict[str, Tensor], kv_cache):
        """Advance state. Side effects: Updates uncached_buffer, kv_cache"""
        x = uncached_buffer["x"]
        state = {k: v for k, v in uncached_buffer.items() if k != "x"}
        sigma = x.new_full((x.size(0), x.size(1)), self.inference_config.noise_prev)
        # Note: doesn't renoise latents, only noise_prev == 0.0 supported

        for step_sig, step_dsig in zip(self.scheduler_sigmas, self.scheduler_sigmas.diff()):
            sigma[:, -1] = step_sig  # update rollout sigma
            v = self.model(x, sigma, **state, kv_cache=kv_cache)
            x[:, -1:] = (x[:, -1:] + step_dsig * v[:, -1:]).type_as(x)

            # remove cached portions from sequence
            state = {k: s[:, -kv_cache.n_uncached_frames:] for k, s in state.items()}
            x = x[:, -kv_cache.n_uncached_frames:]
            sigma = sigma[:, -kv_cache.n_uncached_frames:]

        state["x"] = x
        return state, kv_cache


# TODO
# - Push inference config overrides to hub (if reasonable, set an inference config in training)
# - Apply noise_prev
# - RoPE for inference
