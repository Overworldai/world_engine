from typing import Set, Tuple
from torch import Tensor
from dataclasses import dataclass, field

import torch

from owl_wms.models.world import WorldModel
from owl_wms.nn.kv_cache import StaticKVCache

from world_engine.ae import InferenceAE


@dataclass
class CtrlInput:
    button: Set[int] = field(default_factory=set)  # pressed button IDs
    mouse: Tuple[float, float] = (0.0, 0.0)  # (x, y) position


@dataclass
class InferenceConfig:
    scheduler_sigmas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    fps: int = 60
    noise_prev: float = 0.0


# Global torch optimizations
torch._dynamo.config.recompile_limit = 64

torch.set_float32_matmul_precision("medium")  # or "medium" for more TF32, "highest" to disable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class WorldEngine:
    uncached_k = 1  # Number of uncached frames, a property of our denoising strategy

    def __init__(self, model_uri, inference_cfg=None, device=None, dtype=torch.bfloat16):
        # Meta
        self.device, self.dtype = device, dtype
        self.inference_cfg = inference_cfg or InferenceConfig()
        self.model_cfg = WorldModel.load_config(model_uri)

        # TODO: remove these hardcoding hacks:
        self.model_cfg.n_buttons = self.model_cfg.n_controller_inputs - 2
        self.model_cfg.causal = True

        # Model
        self.model = WorldModel.from_pretrained(model_uri, cfg=self.model_cfg).to(device=device, dtype=dtype).eval()
        self.vae = InferenceAE.from_pretrained(model_uri, device=device, dtype=dtype)
        # self.prompt_encoder = PromptEncoder("google/umt5-xl").to(device).eval()  # TODO: dont hardcode

        # Inference Scheduler
        self.scheduler_sigmas = torch.tensor(self.inference_cfg.scheduler_sigmas, device=device, dtype=dtype)

        # State
        self.uncached_buffer = self.kv_cache = self.frame_idx = None
        self.reset()

    @torch.inference_mode()
    def reset(self):
        """Reset state for new generation"""
        cfg, dev, dt = self.model_cfg, self.device, self.dtype
        self.uncached_buffer = {
            "x": torch.empty(1, 0, cfg.channels, cfg.height, cfg.width, device=dev, dtype=dt),
            "frame_timestamp": torch.empty(1, 0, device=dev, dtype=torch.long),
            "mouse": torch.empty(1, 0, 2, device=dev, dtype=dt),
            "button": torch.empty(1, 0, cfg.n_buttons, device=dev, dtype=dt),
        }
        self.kv_cache = StaticKVCache(
            cfg, max_seq_len=512, batch_size=1, dtype=dt, n_uncached_frames=self.uncached_k
        ).to(dev)
        self.frame_idx = 0

    @torch.inference_mode()
    def _push_frame_state(self, x, ctrl=None):
        new_frame_state = {
            "button": torch.bincount(
                torch.tensor(list(ctrl.button), dtype=torch.long), minlength=self.model_cfg.n_buttons,
            ).to(x, non_blocking=True)[None, None],
            "mouse": x.new_tensor(ctrl.mouse)[None, None],
            "frame_timestamp": x.new_tensor([[self.frame_idx]], dtype=torch.long),
            "x": x,
        }
        for k, v in new_frame_state.items():
            self.uncached_buffer[k] = torch.cat([self.uncached_buffer[k], v], dim=1)
        self.frame_idx += 1

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
        self.denoise_frame()  # Denoise last frame. Side effect: update buffer / kv cache
        img = self.vae.decode(self.uncached_buffer["x"][:, -1])  # decode last frame -> image
        return img

    def set_prompt(self, prompt: str, timestamp: float = 0.0):
        """Apply text conditioning for T2V"""
        import warnings
        warnings.warn("Not Implemented")

    @torch.compile
    def denoise_frame(self):
        """Advance state. Side effects: Updates uncached_buffer, kv_cache"""
        x = self.uncached_buffer["x"]
        state = {k: v for k, v in self.uncached_buffer.items() if k != "x"}
        sigma = x.new_full((x.size(0), x.size(1)), self.inference_cfg.noise_prev)

        for step_sig, step_dsig in zip(self.scheduler_sigmas, self.scheduler_sigmas.diff()):
            sigma[:, -1] = step_sig  # update rollout sigma
            v = self.model(x, sigma, **state, kv_cache=self.kv_cache)
            x[:, -1:] = (x[:, -1:] + step_dsig * v[:, -1:]).type_as(x)

            # remove cached portions from sequence
            state = {k: s[:, -self.uncached_k:] for k, s in state.items()}
            x = x[:, -self.uncached_k:]
            sigma = sigma[:, -self.uncached_k:]

        state["x"] = x
        self.uncached_buffer = state


# AFTER MVP
# - Apply noise_prev
# - RoPE for inference


def test_world_engine(device="cuda"):
    import time
    model_uri = "OpenWorldLabs/CoD-BiV2-Base"
    uint8_img = torch.randint(
        0, 256, (512, 512, 3),
        dtype=torch.uint8, device=device
    )  # random sample img

    engine = WorldEngine(model_uri, device=device)

    # frame idx = 0
    # -------------

    # persistent prompt (until new prompt is set)
    engine.set_prompt("A fun game")

    # record_frame: Add a clean frame to sequence. No denoising. Advance state
    first_ctrl = CtrlInput()
    img = engine.append_frame(uint8_img, ctrl=first_ctrl)

    # frame idxs = 1+
    # ---------------
    for controller_input in [
            CtrlInput(button={48, 42}, mouse=[0.4, 0.3]),
            CtrlInput(mouse=[0.1, 0.2]),
            CtrlInput(button={95, 32, 105}),
    ]:
        start = time.time()
        img = engine.gen_frame(ctrl=controller_input)  # Generate image, advance to next frame
        print("Step time:", time.time() - start)

    engine.reset()  # Clean up, prepare for new generation
