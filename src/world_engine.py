from typing import Dict, List, Optional, Set, Tuple
from torch import Tensor
from dataclasses import dataclass, field

import torch

from owl_wms.models.world import WorldModel
# from owl_wms.nn.kv_cache import StaticKVCache

from world_engine.ae import InferenceAE


# Global torch optimizations
torch._dynamo.config.recompile_limit = 64
torch.set_float32_matmul_precision("medium")  # low: bf16, medium: tf32, high: fp32

# fix graph break:
torch._dynamo.config.capture_scalar_outputs = True


@dataclass
class CtrlInput:
    button: Set[int] = field(default_factory=set)  # pressed button IDs
    mouse: Tuple[float, float] = (0.0, 0.0)  # (x, y) position


@dataclass
class InferenceConfig:
    quantize: bool = False  # False: bf16, true: w8a8
    # TODO: use model config scheduler sigmas
    # scheduler_sigmas: Optional[List[float]] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25, 0.0])
    # noise_prev: float = 0.0  # always 0 due to self forcing


@dataclass
class OptimizationConfig:
    pass  # TODO
"""
TODO: quantization options
- bf16
- a8w8: Float8DynamicActivationFloat8WeightConfig
"""


####
# REMOVE
import torch.nn as nn


class LayerKVCache(nn.Module):
    def __init__(self, B, H, L, Dh, dtype, tokens_per_frame: int, n_uncached_tok: int):
        super().__init__()
        self.tpf = tokens_per_frame
        self.n_uncached_tok = n_uncached_tok

        # Per-layer KV buffers
        self.k = nn.Buffer(torch.empty(B, H, L, Dh, dtype=dtype), persistent=False)
        self.v = nn.Buffer(torch.empty(B, H, L, Dh, dtype=dtype), persistent=False)
        self.kv_offset = nn.Buffer(torch.zeros((), dtype=torch.long), persistent=False)

        # Per-layer t_pos buffers
        self.t_pos = nn.Buffer(torch.zeros(B, L, dtype=torch.long), persistent=False)
        self.t_pos_offset = nn.Buffer(torch.zeros((), dtype=torch.long), persistent=False)

    def upsert(self, k: Tensor, v: Tensor, t_pos: Tensor):
        # --- KV upsert ---
        T = k.size(2)

        # Optional: keep this check, but only in eager mode so it doesn't bother Dynamo
        if not torch._dynamo.is_compiling():
            assert T % self.tpf == 0, "KV insert must be frame-aligned"

        kv_start = self.kv_offset          # 0-D long tensor
        kv_end = kv_start + T              # 0-D long tensor

        if not torch._dynamo.is_compiling():
            kv_end_int = int(kv_end)
            assert kv_end_int <= self.k.size(2), "KV cache overflow"

        # Build an index tensor [kv_start, kv_start+1, ..., kv_end-1]
        idx = torch.arange(T, device=self.k.device, dtype=torch.long) + kv_start

        # Write into KV cache along sequence dimension (dim=2)
        self.k.index_copy_(2, idx, k)
        self.v.index_copy_(2, idx, v)

        new_kv_off = torch.maximum(
            kv_start,
            kv_end - self.n_uncached_tok
        )
        self.kv_offset.copy_(new_kv_off)

        # --- t_pos upsert (per-layer) ---
        if not torch.compiler.is_compiling():
            torch._assert(t_pos.ndim == 2, "t_pos must be 2D")
            torch._assert(
                t_pos.size(0) == self.t_pos.size(0),
                "Batch mismatch in t_pos",
            )

        S = t_pos.size(1)
        # per-layer analogue of: start = max(t_pos_offset, kv_offset.min())
        t_start = torch.maximum(self.t_pos_offset, self.kv_offset)
        t_end = t_start + S
        if not torch.compiler.is_compiling():
            torch._assert(
                t_end <= self.t_pos.size(1),
                f"KV cache overflow (t_pos), {t_end}, {self.t_pos.shape}",
            )

        t_idx = torch.arange(S, device=self.t_pos.device, dtype=torch.long) + t_start
        self.t_pos.index_copy_(1, t_idx, t_pos)

        new_t_off = torch.maximum(t_start, t_end - self.n_uncached_tok)
        self.t_pos_offset.copy_(new_t_off)

        # t_pos_view = self.t_pos[:, :t_end]  # TODO: use for applying attn rules

        idx = torch.arange(T, device=self.k.device, dtype=torch.long) + (kv_end - T)
        return (
            self.k.index_select(2, idx),
            self.v.index_select(2, idx),
            self.t_pos.index_select(1, idx)
        )


class StaticKVCacheNew(nn.Module):
    def __init__(self, config, max_seq_len, batch_size, dtype, n_uncached_frames: int = 1):
        super().__init__()

        self.tpf = config.tokens_per_frame
        self.n_uncached_frames = n_uncached_frames
        self.n_uncached_tok = self.tpf * n_uncached_frames

        B = batch_size
        H = getattr(config, "n_kv_heads", config.n_heads)
        L = max_seq_len * self.tpf
        Dh = config.d_model // config.n_heads
        NL = config.n_layers

        self.layers = nn.ModuleList([
            LayerKVCache(B, H, L, Dh, dtype, self.tpf, self.n_uncached_tok)
            for _ in range(NL)
        ])

    def upsert(self, k: Tensor, v: Tensor, t_pos: Tensor, layer: int):
        return self.layers[layer].upsert(k, v, t_pos)

class StaticKVCache(nn.Module):
    def __init__(self, config, max_seq_len, batch_size, dtype, n_uncached_frames=1):
        super().__init__()

        # Exclude last N tokens from caching
        self.tpf = config.tokens_per_frame
        self.n_uncached_frames = n_uncached_frames
        self.n_uncached_tok = self.tpf * self.n_uncached_frames

        B = batch_size
        H = getattr(config, "n_kv_heads", config.n_heads)
        L = max_seq_len * config.tokens_per_frame
        Dh = config.d_model // config.n_heads
        NL = config.n_layers

        self.k = nn.Buffer(torch.empty(NL, B, H, L, Dh, dtype=dtype), persistent=False)
        self.v = nn.Buffer(torch.empty(NL, B, H, L, Dh, dtype=dtype), persistent=False)
        self.kv_offset = nn.Buffer(torch.zeros(NL, dtype=torch.long), persistent=False)

        # shared between layers
        self.t_pos = nn.Buffer(torch.zeros(B, L, dtype=torch.long), persistent=False)
        self.t_pos_offset = nn.Buffer(torch.zeros((), dtype=torch.long), persistent=False)

    def upsert(self, k: Tensor, v: Tensor, layer: int):
        T = k.size(2)
        torch._assert((T % self.tpf) == 0, "KV insert must be frame-aligned")

        start = self.kv_offset[layer]
        end = start + T

        torch._assert(end <= self.k.size(3), "KV cache overflow")

        self.k[layer, :, :, start:end, :].copy_(k)
        self.v[layer, :, :, start:end, :].copy_(v)
        new_off = torch.clamp(end - self.n_uncached_tok, min=0)
        new_off = torch.maximum(new_off, start)
        self.kv_offset[layer].copy_(new_off)

        return self.k[layer, :, :, :end, :], self.v[layer, :, :, :end, :]  # TODO: make static kv

    def upsert_t_pos(self, t_pos):
        """Insert per-batch frame ids and return the KV-length view."""
        assert t_pos.ndim == 2
        torch._assert(t_pos.size(0) == self.t_pos.size(0), "Batch mismatch in t_pos")

        start = torch.maximum(self.t_pos_offset, self.kv_offset.min())
        S = t_pos.size(1)
        end = start + S
        torch._assert(end <= self.t_pos.size(1), f"KV cache overflow (t_pos), {end}, {self.t_pos.shape}")

        self.t_pos[:, start:end].copy_(t_pos)
        new_off = torch.maximum(start, torch.clamp(end - self.n_uncached_tok, min=0))
        self.t_pos_offset.copy_(new_off)
        return self.t_pos[:, :end]


####


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
        inference_config = inference_config or InferenceConfig()
        self.model_cfg = WorldModel.load_config(model_uri)

        # TODO: remove these hardcoding hacks:
        if not hasattr(self.model_cfg, "n_buttons"):
            self.model_cfg.n_buttons = self.model_cfg.n_controller_inputs - 2
        self.model_cfg.mlp_gradient_checkpointing = getattr(self.model_cfg, "mlp_gradient_checkpointing", False)

        if model_config_overrides:
            self.model_cfg.merge_with(model_config_overrides)

        # Model
        self.vae = InferenceAE.from_pretrained(model_uri, device=device, dtype=dtype)
        # self.prompt_encoder = PromptEncoder("google/umt5-xl").to(device).eval()  # TODO: dont hardcode
        self.model = WorldModel.from_pretrained(model_uri, cfg=self.model_cfg).to(device=device, dtype=dtype).eval()

        # TODO: clean up quantization logic
        if inference_config.quantize:
            self.quantize(self.model)

        # Inference Scheduler
        self.scheduler_sigmas = torch.tensor(self.model_cfg.scheduler_sigmas, device=device, dtype=dtype)

        # State
        self.uncached_buffer = self.kv_cache = self.frame_ts = None
        self.reset()

    def quantize(self, model):
        from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig, PerRow

        def is_bf16_linear(mod: nn.Module, _: str) -> bool:
            weight = getattr(mod, "weight", None)
            return isinstance(mod, nn.Linear) and getattr(weight, "dtype", None) is torch.bfloat16

        quant_cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        quantize_(
            self.model,
            quant_cfg,
            filter_fn=is_bf16_linear,
            device=self.device,
        )

    @torch.inference_mode()
    def reset(self):
        """Reset state for new generation"""
        cfg, dev, dt = self.model_cfg, self.device, self.dtype
        pH, pW = getattr(cfg, "patch", [1, 1])
        self.frame_ts = torch.tensor([[0]], dtype=torch.long, device=dev)
        self.uncached_buffer = {
            "x": torch.empty(1, 0, cfg.channels, cfg.height * pH, cfg.width * pW, device=dev, dtype=dt),
            "frame_timestamp": torch.empty(1, 0, device=dev, dtype=torch.long),
            "mouse": torch.empty(1, 0, 2, device=dev, dtype=dt),
            "button": torch.empty(1, 0, cfg.n_buttons, device=dev, dtype=dt),
        }
        self.kv_cache = StaticKVCache(cfg, max_seq_len=512, batch_size=1, dtype=dt).to(dev)

    @torch.inference_mode()
    @torch.compile
    def _push_frame_state(self, x, ctrl=None):
        ctrl = ctrl if ctrl is not None else CtrlInput()
        button = x.new_zeros(1, 1, self.model_cfg.n_buttons)
        if ctrl.button:
            idx = x.new_tensor(list(ctrl.button), dtype=torch.long)
            button.index_fill_(-1, idx, 1.0)
        mouse = x.new_tensor(ctrl.mouse, dtype=self.dtype)[None, None]

        new_frame_state = {
            "button": button,
            "mouse": mouse,
            "frame_timestamp": self.frame_ts,
            "x": x,
        }
        for k, v in new_frame_state.items():
            self.uncached_buffer[k] = torch.cat([self.uncached_buffer[k], v], dim=1)
        self.frame_ts = self.frame_ts + 1

    @torch.inference_mode()
    def append_frame(self, img: Tensor, ctrl: CtrlInput = None):
        assert img.dtype == torch.uint8, img.dtype
        # push frame inputs w/ clean latent
        self._push_frame_state(
            x=self.vae.encode(img).unsqueeze(1),
            ctrl=ctrl
        )
        return img

    @torch.inference_mode()
    def gen_frame(self, ctrl: CtrlInput = None, return_img: bool = True):
        shape = self.uncached_buffer["x"].shape
        # prepare frame inputs + random N latent
        self._push_frame_state(
            x=torch.randn(shape[0], 1, *shape[2:], device=self.device, dtype=self.dtype),
            ctrl=ctrl,
        )
        self.uncached_buffer, self.kv_cache = self.denoise_frame(self.uncached_buffer, self.kv_cache)
        x = self.uncached_buffer["x"][:, -1]
        with torch.amp.autocast('cuda', torch.bfloat16):
            return (self.vae.decode(x) if return_img else x)

    def set_prompt(self, prompt: str, timestamp: float = 0.0):
        """Apply text conditioning for T2V"""
        import warnings
        warnings.warn("Not Implemented")

    #@torch.compile(fullgraph=True, mode="max-autotune", dynamic=False)
    @torch.compile
    def denoise_frame(self, uncached_buffer: Dict[str, Tensor], kv_cache):
        """Advance state, generate new frame, provide updated state and kv cache"""
        x = uncached_buffer["x"].clone()  # needed for max-autotune / reduce-overhead
        state = {k: v for k, v in uncached_buffer.items() if k != "x"}
        sigma = x.new_zeros((x.size(0), x.size(1)))

        for step_sig, step_dsig in zip(self.scheduler_sigmas, self.scheduler_sigmas.diff()):
            sigma[:, -1] = step_sig  # update rollout sigma
            v = self.model(x, sigma, **state, kv_cache=kv_cache)
            x[:, -1:] = x[:, -1:] + step_dsig * v[:, -1:]

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
