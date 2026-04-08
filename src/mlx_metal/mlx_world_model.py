"""
MLX implementation of the WorldModel for Apple Silicon inference.

Restores the known-good fp16 baseline plus the selective fused W8A8 NAX
linear acceleration path that previously delivered the best MLX speed profile.
"""
from __future__ import annotations

import fnmatch
from typing import Optional, Tuple, List, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from we_kernels import w8a8_gemm_nax
from we_kernels import fused_silu_quant, fused_rmsnorm_adaln_quant, w8a8_gemm_prequantized, w8a8_silu_gemm_nax
from we_kernels import scatter_sdpa

DTYPE = mx.float16

N_LAYERS = 24
D_MODEL = 2048
N_HEADS = 32
N_KV_HEADS = 32
D_HEAD = 64
D_ROPE = D_HEAD // 2
T = 512
C = 32
HP, WP = 16, 32
PH, PW = 2, 2
N_BUTTONS = 256
SIGMAS = [1.0, 0.9, 0.75, 0.3, 0.0]
DSIGMAS = [-0.1, -0.15, -0.45, -0.3]
INT8_MAX = 127.0
INT8_EPS = 1e-6

DEFAULT_INT8_NAX_QKV_PATTERNS = ("transformer.blocks.*.attn",)
DEFAULT_INT8_NAX_COND_PATTERNS = ("transformer.blocks.*.cond_head",)
DEFAULT_INT8_NAX_STANDALONE_PATTERNS: Sequence[str] = ()
INT8_NAX_STANDALONE_COMPONENT_PATTERNS = {
    "attn_out": ("transformer.blocks.*.attn.out_proj",),
    "mlp_fc1": ("transformer.blocks.*.mlp.fc1",),
    "mlp_fc2": ("transformer.blocks.*.mlp.fc2",),
    "ctrl_fc1_x": ("transformer.blocks.*.ctrl_mlpfusion.fc1_x",),
    "ctrl_fc1_c": ("transformer.blocks.*.ctrl_mlpfusion.fc1_c",),
    "ctrl_fc2": ("transformer.blocks.*.ctrl_mlpfusion.fc2",),
    "unpatchify": ("unpatchify",),
    "denoise_fc1": ("denoise_step_emb.mlp.fc1",),
    "denoise_fc2": ("denoise_step_emb.mlp.fc2",),
    "out_norm": ("out_norm.fc",),
}
INT8_NAX_GROUP_COMPONENTS = ("qkv", "cond")
INT8_NAX_SPEED_COMPONENTS = (
    *INT8_NAX_GROUP_COMPONENTS,
    "attn_out",
    "mlp_fc1",
    "mlp_fc2",
    "unpatchify",
)
INT8_NAX_MAX_QAT_COMPONENTS = (
    *INT8_NAX_SPEED_COMPONENTS,
    "ctrl_fc1_x",
    "ctrl_fc1_c",
    "ctrl_fc2",
    "denoise_fc1",
    "denoise_fc2",
    "out_norm",
)


def _symmetric_int8_quantize_last_axis(x: mx.array) -> Tuple[mx.array, mx.array]:
    x_f32 = x.astype(mx.float32)
    eps = mx.array(INT8_EPS, dtype=mx.float32)
    scale = mx.maximum(mx.max(mx.abs(x_f32), axis=-1, keepdims=True) / INT8_MAX, eps)
    x_q = mx.clip(mx.round(x_f32 / scale), -INT8_MAX, INT8_MAX).astype(mx.int8)
    return x_q, scale


def _symmetric_int8_quantize_rows(w: mx.array) -> Tuple[mx.array, mx.array]:
    w_q, w_scale = _symmetric_int8_quantize_last_axis(w)
    return w_q, mx.reshape(w_scale, (-1,))


def _path_set(root, path: str, value):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if isinstance(parent, list) and part == "blocks":
            continue
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


def _path_matches(path: str, patterns: Optional[Sequence[str]]) -> bool:
    if not patterns:
        return False
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def _linear_is_eligible(lin) -> bool:
    return isinstance(lin, nn.Linear) and lin.weight.shape[-1] % 64 == 0 and lin.weight.dtype == mx.float16


def get_int8_nax_profile_components(profile: str = "speed") -> tuple[str, ...]:
    profile = profile.lower()
    if profile in {"grouped", "group", "base"}:
        return INT8_NAX_GROUP_COMPONENTS
    if profile in {"speed", "default"}:
        return INT8_NAX_SPEED_COMPONENTS
    if profile in {"max", "max_qat", "all", "full"}:
        return INT8_NAX_MAX_QAT_COMPONENTS
    raise ValueError(f"Unknown int8 NAX profile: {profile}")


class Int8NaxLinear(nn.Module):
    def __init__(self, weight_q: mx.array, weight_scale: mx.array, bias: Optional[mx.array] = None, output_splits: Optional[Sequence[int]] = None, smooth_scale: Optional[mx.array] = None):
        super().__init__()
        self.weight_q = weight_q
        self.weight_scale = weight_scale
        self.bias = bias.astype(mx.float32) if bias is not None else None
        self.output_splits = tuple(output_splits) if output_splits is not None else None
        self.smooth_scale = smooth_scale.astype(mx.float16) if smooth_scale is not None else None
        self.freeze()

    @classmethod
    def from_linear(cls, lin: nn.Linear, smooth_scale: Optional[mx.array] = None):
        weight_q, weight_scale = _symmetric_int8_quantize_rows(lin.weight)
        return cls(weight_q, weight_scale, getattr(lin, "bias", None), smooth_scale=smooth_scale)

    @classmethod
    def from_linears(cls, linears: Sequence[nn.Linear], smooth_scale: Optional[mx.array] = None):
        weights = []
        scales = []
        biases = []
        splits = []
        for lin in linears:
            w_q, w_scale = _symmetric_int8_quantize_rows(lin.weight)
            weights.append(w_q)
            scales.append(w_scale)
            splits.append(lin.weight.shape[0])
            lin_bias = getattr(lin, "bias", None)
            if lin_bias is not None:
                biases.append(lin_bias.astype(mx.float32))
        bias = mx.concatenate(biases, axis=0) if biases else None
        return cls(mx.concatenate(weights, axis=0), mx.concatenate(scales, axis=0), bias=bias, output_splits=splits, smooth_scale=smooth_scale)

    def _apply_smooth(self, x: mx.array) -> mx.array:
        if self.smooth_scale is not None:
            return x * self.smooth_scale
        return x

    def __call__(self, x: mx.array):
        x = x.astype(mx.float16) if x.dtype != mx.float16 else x
        x = self._apply_smooth(x)
        y = w8a8_gemm_nax(x, self.weight_q, w_scales=self.weight_scale, bias=self.bias)
        if self.output_splits is not None:
            return mx.split(y, np.cumsum(self.output_splits[:-1]).tolist(), axis=-1)
        return y

    def forward_prequantized(self, x_q: mx.array, x_scales: mx.array):
        """GEMM with pre-quantized int8 activations — skips activation quantization."""
        y = w8a8_gemm_prequantized(x_q, x_scales, self.weight_q, w_scales=self.weight_scale, bias=self.bias)
        if self.output_splits is not None:
            return mx.split(y, np.cumsum(self.output_splits[:-1]).tolist(), axis=-1)
        return y


class Int8NaxSiLULinear(nn.Module):
    """W8A8 linear that applies fused SiLU+Quant before the GEMM.

    Used for MLP fc2 where the input is fc1's fp16 output and SiLU
    is applied before fc2's GEMM. Fuses SiLU + int8 quantization
    into a single Metal kernel dispatch.
    """
    def __init__(self, weight_q: mx.array, weight_scale: mx.array, bias: Optional[mx.array] = None):
        super().__init__()
        self.weight_q = weight_q
        self.weight_scale = weight_scale
        self.bias = bias.astype(mx.float32) if bias is not None else None
        self.freeze()

    @classmethod
    def from_linear(cls, lin: nn.Linear):
        weight_q, weight_scale = _symmetric_int8_quantize_rows(lin.weight)
        return cls(weight_q, weight_scale, getattr(lin, "bias", None))

    def __call__(self, x: mx.array):
        return w8a8_silu_gemm_nax(x, self.weight_q, w_scales=self.weight_scale, bias=self.bias)


def enable_int8_nax_components(model: nn.Module, components: Sequence[str], smooth_scales: Optional[dict] = None) -> dict:
    components = tuple(components)
    qkv_patterns = DEFAULT_INT8_NAX_QKV_PATTERNS if "qkv" in components else ()
    cond_patterns = DEFAULT_INT8_NAX_COND_PATTERNS if "cond" in components else ()
    standalone_patterns = []
    for name in components:
        if name in {"qkv", "cond"}:
            continue
        standalone_patterns.extend(INT8_NAX_STANDALONE_COMPONENT_PATTERNS[name])
    return enable_int8_nax_linear(model, qkv_patterns=qkv_patterns, cond_patterns=cond_patterns, standalone_patterns=tuple(standalone_patterns), smooth_scales=smooth_scales)


def _merge_qkv_smooth_scales(
    smooth_scales: dict, idx: int, q_proj, k_proj, v_proj
) -> Optional[mx.array]:
    """Merge separate q/k/v smooth scales into a unified QKV smooth scale.

    Mirrors PyTorch's merge_qkv_smoothscales: takes element-wise max of
    the individual smoothing factors, then adjusts q/k/v weights so that
    a single unified scale can be applied to the shared input.

    Smooth scales are stored as 1/s (reciprocal of smoothing factor).
    Unified scale = 1 / max(s_q, s_k, s_v) per channel.
    Weight adjustment: W_new[section] *= s_uni / s_individual.
    """
    s_q = smooth_scales.get(f"transformer.blocks.{idx}.attn.q_proj")
    s_k = smooth_scales.get(f"transformer.blocks.{idx}.attn.k_proj")
    s_v = smooth_scales.get(f"transformer.blocks.{idx}.attn.v_proj")
    if s_q is None and s_k is None and s_v is None:
        return None

    D = q_proj.weight.shape[-1]
    ones = mx.ones((D,), dtype=mx.float32)
    # Smooth scales are stored as 1/s; convert to s for merging
    sq = (1.0 / s_q.astype(mx.float32)) if s_q is not None else ones
    sk = (1.0 / s_k.astype(mx.float32)) if s_k is not None else ones
    sv = (1.0 / s_v.astype(mx.float32)) if s_v is not None else ones
    s_uni = mx.maximum(mx.maximum(sq, sk), sv)

    # Adjust weights: W_new = W * (s_uni / s_individual)
    q_adj = (s_uni / sq).astype(mx.float16)
    k_adj = (s_uni / sk).astype(mx.float16)
    v_adj = (s_uni / sv).astype(mx.float16)
    q_proj.weight = q_proj.weight * q_adj
    k_proj.weight = k_proj.weight * k_adj
    v_proj.weight = v_proj.weight * v_adj

    # Return unified scale as 1/s_uni (reciprocal form)
    return (1.0 / s_uni).astype(mx.float16)


def enable_int8_nax_linear(
    model: nn.Module,
    *,
    qkv_patterns: Optional[Sequence[str]] = DEFAULT_INT8_NAX_QKV_PATTERNS,
    cond_patterns: Optional[Sequence[str]] = DEFAULT_INT8_NAX_COND_PATTERNS,
    standalone_patterns: Optional[Sequence[str]] = DEFAULT_INT8_NAX_STANDALONE_PATTERNS,
    smooth_scales: Optional[dict] = None,
) -> dict:
    stats = {"fused_qkv_groups": 0, "fused_cond_groups": 0, "fused_mlp_modules": 0, "standalone_linears": 0, "replaced_linears": 0, "smooth_applied": 0}
    if smooth_scales is None:
        smooth_scales = {}

    for idx, blk in enumerate(model.transformer):
        attn_path = f"transformer.blocks.{idx}.attn"
        if _path_matches(attn_path, qkv_patterns):
            linears = [blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj]
            if all(_linear_is_eligible(lin) for lin in linears):
                qkv_smooth = _merge_qkv_smooth_scales(
                    smooth_scales, idx, blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj
                )
                blk.attn.qkv_proj = Int8NaxLinear.from_linears(linears, smooth_scale=qkv_smooth)
                stats["fused_qkv_groups"] += 1
                stats["replaced_linears"] += 3
                if qkv_smooth is not None:
                    stats["smooth_applied"] += 1
        cond_path = f"transformer.blocks.{idx}.cond_head"
        if _path_matches(cond_path, cond_patterns):
            if all(_linear_is_eligible(lin) for lin in blk.cond_head.cond_proj):
                blk.cond_head.cond_proj_group = Int8NaxLinear.from_linears(blk.cond_head.cond_proj)
                stats["fused_cond_groups"] += 1
                stats["replaced_linears"] += len(blk.cond_head.cond_proj)

    if standalone_patterns is None:
        standalone_patterns = ("*",)
    if standalone_patterns:
        candidates = []
        for idx, blk in enumerate(model.transformer):
            candidates.extend([
                (f"transformer.blocks.{idx}.attn.out_proj", blk.attn.out_proj),
                (f"transformer.blocks.{idx}.mlp.fc1", blk.mlp.fc1),
                (f"transformer.blocks.{idx}.mlp.fc2", blk.mlp.fc2),
            ])
            if blk.has_ctrl:
                candidates.extend([
                    (f"transformer.blocks.{idx}.ctrl_mlpfusion.fc1_x", blk.ctrl_mlpfusion.fc1_x),
                    (f"transformer.blocks.{idx}.ctrl_mlpfusion.fc1_c", blk.ctrl_mlpfusion.fc1_c),
                    (f"transformer.blocks.{idx}.ctrl_mlpfusion.fc2", blk.ctrl_mlpfusion.fc2),
                ])
        candidates.extend([
            ("unpatchify", model.unpatchify),
            ("denoise_step_emb.mlp.fc1", model.denoise_step_emb.mlp.fc1),
            ("denoise_step_emb.mlp.fc2", model.denoise_step_emb.mlp.fc2),
            ("out_norm.fc", model.out_norm.fc),
        ])
        # Patterns where fc2 receives SiLU output — use fused SiLU+Quant variant
        _silu_fc2_patterns = {f"transformer.blocks.{idx}.mlp.fc2" for idx in range(len(model.transformer))}
        for path, mod in candidates:
            if not isinstance(mod, nn.Linear):
                continue
            if not _linear_is_eligible(mod) or not _path_matches(path, standalone_patterns):
                continue
            ss = smooth_scales.get(path)
            if path in _silu_fc2_patterns:
                _path_set(model, path, Int8NaxSiLULinear.from_linear(mod))
            else:
                _path_set(model, path, Int8NaxLinear.from_linear(mod, smooth_scale=ss))
                if ss is not None:
                    stats["smooth_applied"] += 1
            stats["standalone_linears"] += 1
            stats["replaced_linears"] += 1
    return stats


def enable_int8_nax_profile(model: nn.Module, profile: str = "speed", smooth_scales: Optional[dict] = None) -> dict:
    return enable_int8_nax_components(model, get_int8_nax_profile_components(profile), smooth_scales=smooth_scales)


def ortho_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    y0 = x0 * cos - x1 * sin
    y1 = x1 * cos + x0 * sin
    return mx.concatenate([y0, y1], axis=-1)


def compute_rope_angles(frame_idx: int, ts_mult: float, rope_xy: mx.array, rope_inv_t: mx.array) -> Tuple[mx.array, mx.array]:
    idx = mx.arange(T)
    x_norm = (2.0 * (idx % WP).astype(mx.float32) + 1.0) / WP - 1.0
    y_norm = (2.0 * (idx // WP).astype(mx.float32) + 1.0) / HP - 1.0
    t_val = mx.full((T,), float(frame_idx * ts_mult), dtype=mx.float32)
    freqs = mx.concatenate([mx.expand_dims(x_norm, -1) * rope_xy, mx.expand_dims(y_norm, -1) * rope_xy, mx.expand_dims(t_val, -1) * rope_inv_t], axis=-1)
    cos = mx.reshape(mx.cos(freqs), (1, 1, T, D_ROPE)).astype(DTYPE)
    sin = mx.reshape(mx.sin(freqs), (1, 1, T, D_ROPE)).astype(DTYPE)
    return cos, sin


class RingKVCache:
    BK = 32  # scatter_sdpa block granularity

    def __init__(self, capacity: int, L: int, dilation: int, num_buckets: int):
        self.capacity = capacity
        self.L = L
        self.dilation = dilation
        self.num_buckets = num_buckets
        self.keys = mx.zeros((1, N_KV_HEADS, capacity, D_HEAD), dtype=DTYPE)
        self.values = mx.zeros((1, N_KV_HEADS, capacity, D_HEAD), dtype=DTYPE)
        # Track which ring slots are populated (pure Python — no GPU sync needed)
        self.written_slots: set[int] = set()

    def set_frozen(self, frozen: bool):
        self.frozen = frozen

    def compute_block_offsets(self, frame_idx: int) -> mx.array:
        """Precompute BK-aligned block offsets for scatter-read SDPA.

        Slightly differs from pytorch implementation, usually would use .argwhere or 
        .nonzero, but mlx doesn't have those equivalent primitives. Using np could 
        force GPU->CPU sync, so we instead track the indices with a python set.
        Uses self.written_slots to know which ring buckets are populated.
        Called once per forward_single.
        """
        # Determine stale bucket to exclude (being overwritten this frame)
        stale_slot = -1
        write_step = (frame_idx % self.dilation) == 0
        if self.num_buckets > 0 and write_step:
            bucket = (frame_idx + (self.dilation - 1)) // self.dilation
            stale_slot = bucket % self.num_buckets

        # Collect BK-aligned block offsets for written, non-stale ring buckets
        offsets = []
        for slot in sorted(self.written_slots):
            if slot == stale_slot:
                continue
            start = slot * T
            for blk in range(T // self.BK):
                offsets.append(start + blk * self.BK)

        # Tail is always valid
        for blk in range(T // self.BK):
            offsets.append(self.L + blk * self.BK)

        return mx.array(np.array(offsets, dtype=np.int32))

    def upsert(self, k_new: mx.array, v_new: mx.array, frame_idx: int,
               block_offsets: mx.array | None = None) -> None:
        """Write current frame to tail, optionally persist to ring."""
        # 1. Always write current frame to tail
        self.keys[:, :, self.L:self.L + T, :] = k_new
        self.values[:, :, self.L:self.L + T, :] = v_new

        # 2. Persist to ring (only when unfrozen — cache_write pass)
        write_step = (frame_idx % self.dilation) == 0
        if write_step and self.num_buckets > 0 and not getattr(self, "frozen", False):
            bucket = (frame_idx + (self.dilation - 1)) // self.dilation
            slot = bucket % self.num_buckets
            rs = slot * T
            self.keys[:, :, rs:rs + T, :] = k_new
            self.values[:, :, rs:rs + T, :] = v_new
            self.written_slots.add(slot)


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=False)

    def __call__(self, x, *, fc1_q: mx.array | None = None, fc1_scales: mx.array | None = None):
        if fc1_q is not None and isinstance(self.fc1, Int8NaxLinear):
            h = self.fc1.forward_prequantized(fc1_q, fc1_scales)
        else:
            h = self.fc1(x)
        if isinstance(self.fc2, Int8NaxSiLULinear):
            return self.fc2(h)  # SiLU fused inside fc2
        return self.fc2(nn.silu(h))


class CtrlFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_x = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.fc1_c = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.fc2 = nn.Linear(D_MODEL, D_MODEL, bias=False)

    def __call__(self, x4: mx.array, cond: mx.array):
        h = nn.silu(self.fc1_x(x4) + mx.expand_dims(self.fc1_c(cond), 2))
        return self.fc2(h)


class CondHead(nn.Module):
    def __init__(self, noise_conditioning: str, shared_cond_proj=None):
        super().__init__()
        self.bias_in = mx.zeros((D_MODEL,)) if noise_conditioning == "wan" else None
        # cond_proj weights are shared across all blocks (PyTorch ties them).
        # bias_in is per-block.
        self.cond_proj = shared_cond_proj if shared_cond_proj is not None else [nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(6)]
        self.cond_proj_group = None

    def __call__(self, cond: mx.array):
        c = cond + self.bias_in if self.bias_in is not None else cond
        h = nn.silu(c)
        if self.cond_proj_group is not None:
            return self.cond_proj_group(h)
        return tuple(p(h) for p in self.cond_proj)


class Attention(nn.Module):
    def __init__(self, value_residual: bool, gated_attn: bool):
        super().__init__()
        self.value_residual = value_residual
        if value_residual:
            self.v_lamb = mx.array(0.5, dtype=mx.float16)
        self.q_proj = nn.Linear(D_MODEL, N_HEADS * D_HEAD, bias=False)
        self.k_proj = nn.Linear(D_MODEL, N_KV_HEADS * D_HEAD, bias=False)
        self.v_proj = nn.Linear(D_MODEL, N_KV_HEADS * D_HEAD, bias=False)
        self.qkv_proj = None
        self.out_proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.gated_attn = gated_attn
        if gated_attn:
            self.gate_proj = nn.Linear(N_HEADS, N_HEADS, bias=False)


class TransformerBlock(nn.Module):
    def __init__(self, layer_idx: int, is_global: bool, has_ctrl: bool, cfg, shared_cond_proj=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_global = is_global
        self.has_ctrl = has_ctrl
        self.attn = Attention(getattr(cfg, "value_residual", False), getattr(cfg, "gated_attn", False))
        self.cond_head = CondHead(getattr(cfg, "noise_conditioning", "wan"), shared_cond_proj=shared_cond_proj)
        self.mlp = MLP(D_MODEL, int(D_MODEL * cfg.mlp_ratio), D_MODEL)
        if has_ctrl:
            self.ctrl_mlpfusion = CtrlFusion()

    def __call__(self, x: mx.array, cond: mx.array, ctrl_emb: mx.array, rope_cos: mx.array, rope_sin: mx.array, v1_in: Optional[mx.array], kv_cache: RingKVCache, frame_idx: int, block_offsets: mx.array | None = None) -> Tuple[mx.array, mx.array]:
        s0, b0, g0, s1, b1, g1 = self.cond_head(cond)
        residual = x

        # Attention: RMSNorm + AdaLN + QKV projection
        if self.attn.qkv_proj is not None and isinstance(self.attn.qkv_proj, Int8NaxLinear):
            # Fused RMSNorm+AdaLN+(SmoothQuant)+Quant → pre-quantized QKV
            x_2d = mx.reshape(x, (-1, D_MODEL))
            x_q, x_scales = fused_rmsnorm_adaln_quant(
                x_2d, mx.reshape(s0, (-1,)), mx.reshape(b0, (-1,)),
                smooth_scale=self.attn.qkv_proj.smooth_scale,
            )
            q_raw, k_raw, v_raw = self.attn.qkv_proj.forward_prequantized(x_q, x_scales)
        else:
            x4 = mx.reshape(x, (1, 1, T, D_MODEL))
            x_n = mx.fast.rms_norm(x4, None, 1e-5) * (1 + mx.expand_dims(s0, 2)) + mx.expand_dims(b0, 2)
            x_n = mx.reshape(x_n, (1, T, D_MODEL))
            if self.attn.qkv_proj is not None:
                q_raw, k_raw, v_raw = self.attn.qkv_proj(x_n)
            else:
                q_raw = self.attn.q_proj(x_n)
                k_raw = self.attn.k_proj(x_n)
                v_raw = self.attn.v_proj(x_n)

        q = mx.reshape(q_raw, (1, T, N_HEADS, D_HEAD)).transpose(0, 2, 1, 3)
        k_new = mx.reshape(k_raw, (1, T, N_KV_HEADS, D_HEAD)).transpose(0, 2, 1, 3)
        v_new = mx.reshape(v_raw, (1, T, N_KV_HEADS, D_HEAD)).transpose(0, 2, 1, 3)
        if v1_in is not None and self.attn.value_residual:
            v1_r = mx.reshape(v1_in, (1, N_KV_HEADS, T, D_HEAD))
            v_new = v_new + self.attn.v_lamb * (v1_r - v_new)
            v1_out = v1_in
        else:
            v1_out = v_new
        q = ortho_rope(mx.fast.rms_norm(q, None, 1e-5), rope_cos, rope_sin)
        k_new = ortho_rope(mx.fast.rms_norm(k_new, None, 1e-5), rope_cos, rope_sin)

        # KV cache upsert + scatter-read SDPA
        kv_cache.upsert(k_new, v_new, frame_idx, block_offsets=block_offsets)
        n_kv = kv_cache.keys.shape[1]
        q_3d = mx.reshape(q, (N_HEADS, T, D_HEAD))
        k_3d = mx.reshape(kv_cache.keys, (n_kv, kv_cache.capacity, D_HEAD))
        v_3d = mx.reshape(kv_cache.values, (n_kv, kv_cache.capacity, D_HEAD))
        y_3d = scatter_sdpa(q_3d, k_3d, v_3d, block_offsets, float(D_HEAD ** -0.5), "bq32_bk32_wm2")
        # y_3d: [N_HEADS, T, D_HEAD] → transpose to [T, N_HEADS, D_HEAD] → [1, T, N_HEADS*D_HEAD]
        y = mx.reshape(y_3d.transpose(1, 0, 2), (1, T, N_HEADS * D_HEAD))
        y = self.attn.out_proj(y)
        x4_y = mx.reshape(y, (1, 1, T, D_MODEL))
        x = mx.reshape(x4_y * mx.expand_dims(g0, 2), (1, T, D_MODEL)) + residual
        if self.has_ctrl:
            x_n2 = mx.fast.rms_norm(x, None, 1e-5)
            c_n = mx.fast.rms_norm(ctrl_emb, None, 1e-5)
            x4_2 = mx.reshape(x_n2, (1, 1, T, D_MODEL))
            fused = self.ctrl_mlpfusion(x4_2, c_n)
            x = mx.reshape(fused, (1, T, D_MODEL)) + x

        # MLP: RMSNorm + AdaLN + fc1 + SiLU + fc2
        if isinstance(self.mlp.fc1, Int8NaxLinear):
            # Fused RMSNorm + AdaLN + (SmoothQuant) + Quant → pre-quantized fc1
            x_2d = mx.reshape(x, (-1, D_MODEL))
            fc1_q, fc1_scales = fused_rmsnorm_adaln_quant(
                x_2d, mx.reshape(s1, (-1,)), mx.reshape(b1, (-1,)),
                smooth_scale=self.mlp.fc1.smooth_scale,
            )
            mo = self.mlp(None, fc1_q=fc1_q, fc1_scales=fc1_scales)
        else:
            x4_m = mx.reshape(x, (1, 1, T, D_MODEL))
            mn = mx.fast.rms_norm(x4_m, None, 1e-5) * (1 + mx.expand_dims(s1, 2)) + mx.expand_dims(b1, 2)
            mo = self.mlp(mn)

        x4_g = mx.reshape(mo, (1, 1, T, D_MODEL))
        x = mx.reshape(x4_g * mx.expand_dims(g1, 2), (1, T, D_MODEL)) + x
        return x, v1_out


class OutNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(D_MODEL, 2 * D_MODEL, bias=False)


class CtrlEmbed(nn.Module):
    def __init__(self, mlp_ratio: int):
        super().__init__()
        self.mlp = MLP(N_BUTTONS + 3, D_MODEL * mlp_ratio, D_MODEL)


class NoiseConditioner(nn.Module):
    def __init__(self, mlp_ratio: int):
        super().__init__()
        self.freq = mx.zeros((256,))
        self.mlp = MLP(512, D_MODEL * mlp_ratio, D_MODEL)


class MLXWorldModel(nn.Module):
    def __init__(self, cfg, kv_cache_mode: str = "fp16", attention_mode: str = "fp16"):
        super().__init__()
        self.cfg = cfg
        self.kv_cache_mode = kv_cache_mode
        self.attention_mode = attention_mode
        inference_fps = getattr(cfg, "inference_fps", cfg.base_fps)
        latent_fps = inference_fps / getattr(cfg, "temporal_compression", 1)
        self.ts_mult = int(cfg.base_fps) // latent_fps
        self.patchify = nn.Conv2d(C, D_MODEL, kernel_size=(PH, PW), stride=(PH, PW), bias=False)
        self.unpatchify = nn.Linear(D_MODEL, C * PH * PW, bias=True)
        self.out_norm = OutNorm()
        self.ctrl_emb = CtrlEmbed(cfg.mlp_ratio)
        self.denoise_step_emb = NoiseConditioner(cfg.mlp_ratio)
        self.rope_xy = mx.zeros((D_HEAD // 8,))
        self.rope_inv_t = mx.zeros((D_HEAD // 4,))
        period = cfg.global_attn_period
        off = getattr(cfg, "global_attn_offset", 0) % period
        ctrl_period = cfg.ctrl_conditioning_period
        noise_cond_type = getattr(cfg, "noise_conditioning", "wan")
        shared_cond_proj = [nn.Linear(D_MODEL, D_MODEL, bias=False) for _ in range(6)]
        self.transformer = [
            TransformerBlock(i, ((i - off) % period == 0), (ctrl_period is not None and i % ctrl_period == 0), cfg, shared_cond_proj)
            for i in range(N_LAYERS)
        ]
        local_L = cfg.local_window * T
        global_L = cfg.global_window * T
        self.kv_caches: List[RingKVCache] = []
        for i in range(N_LAYERS):
            is_global = ((i - off) % period == 0)
            L = global_L if is_global else local_L
            cap = L + T
            dilation = cfg.global_pinned_dilation if is_global else 1
            num_buckets = (L // T) // dilation
            self.kv_caches.append(RingKVCache(cap, L, dilation, num_buckets))

    def noise_cond(self, sigma: float) -> mx.array:
        s = mx.array([sigma], dtype=mx.float32) * 1000.0
        phase = mx.expand_dims(s, -1) * mx.expand_dims(self.denoise_step_emb.freq, 0)
        emb = mx.concatenate([mx.sin(phase), mx.cos(phase)], axis=-1) * (2 ** 0.5)
        mlp = self.denoise_step_emb.mlp
        emb = mlp.fc2(nn.silu(mlp.fc1(emb)))
        return mx.reshape(emb, (1, 1, D_MODEL)).astype(DTYPE)

    def ctrl_embed(self, mouse, button, scroll) -> mx.array:
        inp = mx.concatenate([mouse, button, scroll], axis=-1)
        mlp = self.ctrl_emb.mlp
        return mlp.fc2(nn.silu(mlp.fc1(inp))).astype(DTYPE) + 1e-7

    def forward_single(self, x: mx.array, cond: mx.array, rope_cos: mx.array, rope_sin: mx.array, mouse: mx.array, button: mx.array, scroll: mx.array, frame_idx: int, _block_offsets: list | None = None) -> mx.array:
        ctrl_emb = self.ctrl_embed(mouse, button, scroll)
        x_2d = mx.reshape(x, (1, C, HP * PH, WP * PW)).transpose(0, 2, 3, 1)
        x_pat = self.patchify(x_2d)
        x_seq = mx.reshape(x_pat, (1, T, D_MODEL))
        # Use precomputed block offsets if provided, else compute
        if _block_offsets is not None:
            block_offsets_list = _block_offsets
        else:
            _off_cache = {}
            block_offsets_list = []
            for kv in self.kv_caches:
                key = (kv.capacity, kv.dilation, kv.num_buckets)
                if key not in _off_cache:
                    _off_cache[key] = kv.compute_block_offsets(frame_idx)
                block_offsets_list.append(_off_cache[key])

        v1 = None
        for blk, kv, bo in zip(self.transformer, self.kv_caches, block_offsets_list):
            x_seq, v1 = blk(x_seq, cond, ctrl_emb, rope_cos, rope_sin, v1, kv, frame_idx, bo)
        y = nn.silu(cond)
        ab = self.out_norm.fc(y)
        ab = mx.reshape(mx.broadcast_to(mx.expand_dims(ab, 2), (1, 1, T, 2 * D_MODEL)), (1, T, 2 * D_MODEL))
        a, b_ = mx.split(ab, 2, axis=-1)
        x_seq = nn.silu(mx.fast.rms_norm(x_seq, None, 1e-5) * (1 + a) + b_)
        x_out = self.unpatchify(x_seq)
        x_out = mx.reshape(x_out, (1, HP, WP, C, PH, PW))
        x_out = x_out.transpose(0, 3, 1, 4, 2, 5)
        return mx.reshape(x_out, (1, 1, C, HP * PH, WP * PW))

    def denoise(self, x, rope_cos, rope_sin, mouse, button, scroll, frame_idx: int) -> mx.array:
        for kv in self.kv_caches:
            kv.set_frozen(True)

        # Precompute block offsets once (same for all 4 denoise steps)
        _off_cache = {}
        block_offsets_list = []
        for kv in self.kv_caches:
            key = (kv.capacity, kv.dilation, kv.num_buckets)
            if key not in _off_cache:
                _off_cache[key] = kv.compute_block_offsets(frame_idx)
            block_offsets_list.append(_off_cache[key])

        for sigma, dsigma in zip(SIGMAS[:4], DSIGMAS):
            cond = self.noise_cond(sigma)
            v = self.forward_single(
                x, cond, rope_cos, rope_sin, mouse, button, scroll, frame_idx,
                _block_offsets=block_offsets_list,
            )
            x = x + dsigma * v
            mx.eval(x)
        for kv in self.kv_caches:
            kv.set_frozen(False)
        return x

    def cache_write(self, x, rope_cos, rope_sin, mouse, button, scroll, frame_idx: int):
        cond = self.noise_cond(0.0)
        self.forward_single(x, cond, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        # Eval to materialize KV cache writes before next frame
        mx.eval(*[arr for kv in self.kv_caches for arr in [kv.keys, kv.values]])


def _extract_smooth_scales(pt_model) -> dict:
    """Extract SmoothQuant scales from PyTorch model buffers.

    Returns a dict mapping MLX-style paths (e.g. 'transformer.blocks.0.mlp.fc1')
    to mx.array smooth scales of shape [D].
    """
    smooth_scales = {}
    for name, buf in pt_model.named_buffers():
        if not name.endswith("._smooth_scale"):
            continue
        # e.g. 'transformer.blocks.0.attn.q_proj._smooth_scale' -> 'transformer.blocks.0.attn.q_proj'
        layer_path = name[: -len("._smooth_scale")]
        arr = mx.array(buf.detach().float().numpy()).astype(DTYPE)
        smooth_scales[layer_path] = arr
    return smooth_scales


def load_from_pytorch(model_uri: str, int8_profile: Optional[str] = "speed", kv_cache_mode: str = "fp16", attention_mode: str = "fp16") -> Tuple[MLXWorldModel, object]:
    import torch
    from src.model import WorldModel

    cfg = WorldModel.load_config(model_uri)
    global N_LAYERS, D_MODEL, N_HEADS, N_KV_HEADS, D_HEAD, D_ROPE, T, C, HP, WP, PH, PW, N_BUTTONS, SIGMAS, DSIGMAS
    N_LAYERS = cfg.n_layers
    D_MODEL = cfg.d_model
    N_HEADS = cfg.n_heads
    N_KV_HEADS = getattr(cfg, "n_kv_heads", cfg.n_heads)
    D_HEAD = cfg.d_model // cfg.n_heads
    D_ROPE = D_HEAD // 2
    T = cfg.height * cfg.width
    C = cfg.channels
    HP, WP = cfg.height, cfg.width
    PH, PW = tuple(cfg.patch)
    N_BUTTONS = cfg.n_buttons
    SIGMAS = list(cfg.scheduler_sigmas)
    DSIGMAS = [SIGMAS[i + 1] - SIGMAS[i] for i in range(len(SIGMAS) - 1)]

    pt_model = WorldModel.from_pretrained(model_uri, cfg=cfg, device="cpu", dtype=torch.bfloat16).eval()
    mlx_model = MLXWorldModel(cfg, kv_cache_mode=kv_cache_mode, attention_mode=attention_mode)

    # Extract SmoothQuant scales before weight conversion
    smooth_scales = _extract_smooth_scales(pt_model)
    if smooth_scales:
        print(f"  SmoothQuant: loaded {len(smooth_scales)} smooth scales")

    weights = []
    for name, param in pt_model.named_parameters():
        arr = mx.array(param.detach().float().numpy()).astype(DTYPE)
        # MLX model uses transformer[i] (plain list), PyTorch uses transformer.blocks[i]
        name = name.replace("transformer.blocks.", "transformer.")
        if name == "patchify.weight":
            arr = mx.transpose(arr, (0, 2, 3, 1))
        if "ctrl_mlpfusion.mlp.fc1.weight" in name:
            wx = arr[:, :D_MODEL]
            wc = arr[:, D_MODEL:]
            base = name.replace("mlp.fc1.weight", "")
            weights.append((base + "fc1_x.weight", wx))
            weights.append((base + "fc1_c.weight", wc))
            continue
        if "ctrl_mlpfusion.mlp.fc2.weight" in name:
            weights.append((name.replace("mlp.fc2.weight", "fc2.weight"), arr))
            continue
        weights.append((name, arr))
    mlx_model.load_weights(weights, strict=False)
    mlx_model.rope_xy = mx.array(pt_model.transformer.rope_angles.xy.detach().float().numpy())
    mlx_model.rope_inv_t = mx.array(pt_model.transformer.rope_angles.inv_t.detach().float().numpy())
    mlx_model.denoise_step_emb.freq = mx.array(pt_model.denoise_step_emb.freq.detach().float().numpy())
    del pt_model  # Free PyTorch memory immediately

    from mlx.utils import tree_map
    def cast_dtype(x):
        return x.astype(DTYPE) if isinstance(x, mx.array) and x.dtype == mx.float32 and x.ndim >= 2 else x
    mlx_model.update(tree_map(cast_dtype, mlx_model.parameters()))
    if int8_profile is not None:
        stats = enable_int8_nax_profile(mlx_model, int8_profile, smooth_scales=smooth_scales if smooth_scales else None)
        mlx_model.int8_profile = int8_profile
        mlx_model.int8_stats = stats
    else:
        mlx_model.int8_profile = None
        mlx_model.int8_stats = None
    mx.eval(mlx_model.parameters())
    return mlx_model, cfg
