"""World Engine Metal kernels — custom MLX C++ extensions for Apple Silicon."""
from __future__ import annotations

import mlx.core as mx

from we_kernels import _ext


def w8a8_gemm_nax(
    x: mx.array,
    weight_q: mx.array,
    *,
    w_scales: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """W8A8 symmetric GEMM using NAX hardware tensor cores.

    Parameters
    ----------
    x : mx.array, fp16, shape [..., K]  — quantised to int8 internally
    weight_q : mx.array, int8, shape [N, K]
    w_scales : mx.array, fp32, shape [N]
    bias : mx.array | None, fp32, shape [N]

    Returns
    -------
    mx.array, fp16, shape [..., N]
    """
    orig_shape = x.shape
    K = orig_shape[-1]
    N = weight_q.shape[0]

    x_2d = mx.reshape(x, (-1, K)).astype(mx.float16)

    # Dynamic int8 quantisation of activations
    x_f32 = x_2d.astype(mx.float32)
    x_absmax = mx.max(mx.abs(x_f32), axis=-1)
    x_scales = mx.maximum(x_absmax / 127.0, 1e-6)
    x_q = mx.clip(
        mx.round(x_f32 / mx.expand_dims(x_scales, -1)),
        -127, 127,
    ).astype(mx.int8)

    w_scales = w_scales.astype(mx.float32)
    bias_data = bias if bias is not None else mx.zeros((N,), dtype=mx.float32)

    y = _ext.w8a8_gemm(x_q, weight_q, x_scales, w_scales, bias_data)

    out_shape = orig_shape[:-1] + (N,)
    return mx.reshape(y, out_shape)


def fused_silu_quant(x: mx.array) -> tuple[mx.array, mx.array]:
    """Fused SiLU + per-row int8 quantization.

    Returns (x_q [M, K] int8, x_scales [M] fp32).
    """
    x_2d = mx.reshape(x, (-1, x.shape[-1])).astype(mx.float16)
    result = _ext.fused_silu_quant(x_2d)
    return result[0], result[1]


def fused_rmsnorm_quant(x: mx.array, eps: float = 1e-5) -> tuple[mx.array, mx.array]:
    """Fused RMSNorm + per-row int8 quantization.

    Returns (x_q [M, K] int8, x_scales [M] fp32).
    """
    x_2d = mx.reshape(x, (-1, x.shape[-1])).astype(mx.float16)
    result = _ext.fused_rmsnorm_quant(x_2d, eps)
    return result[0], result[1]


def fused_rmsnorm_adaln_quant(
    x: mx.array, s: mx.array, b: mx.array, eps: float = 1e-5,
    smooth_scale: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Fused RMSNorm + AdaLN(*(1+s)+b) + optional SmoothQuant + per-row int8 quantization.

    Returns (x_q [M, K] int8, x_scales [M] fp32).
    """
    x_2d = mx.reshape(x, (-1, x.shape[-1])).astype(mx.float16)
    s_1d = mx.reshape(s, (-1,)).astype(mx.float16)
    b_1d = mx.reshape(b, (-1,)).astype(mx.float16)
    if smooth_scale is not None:
        ss = mx.reshape(smooth_scale, (-1,)).astype(mx.float16)
        result = _ext.fused_rmsnorm_adaln_smooth_quant(x_2d, s_1d, b_1d, ss, eps)
    else:
        result = _ext.fused_rmsnorm_adaln_quant(x_2d, s_1d, b_1d, eps)
    return result[0], result[1]


def fused_rmsnorm_smooth_quant(
    x: mx.array, smooth_scale: mx.array, eps: float = 1e-5,
) -> tuple[mx.array, mx.array]:
    """Fused RMSNorm + SmoothQuant + per-row int8 quantization.

    Returns (x_q [M, K] int8, x_scales [M] fp32).
    """
    x_2d = mx.reshape(x, (-1, x.shape[-1])).astype(mx.float16)
    ss = mx.reshape(smooth_scale, (-1,)).astype(mx.float16)
    result = _ext.fused_rmsnorm_smooth_quant(x_2d, ss, eps)
    return result[0], result[1]


def w8a8_gemm_prequantized(
    x_q: mx.array,
    x_scales: mx.array,
    weight_q: mx.array,
    *,
    w_scales: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """W8A8 GEMM with pre-quantized int8 activations (no activation quant step)."""
    N = weight_q.shape[0]
    w_scales = w_scales.astype(mx.float32)
    bias_data = bias if bias is not None else mx.zeros((N,), dtype=mx.float32)
    return _ext.w8a8_gemm(x_q, weight_q, x_scales, w_scales, bias_data)


def w8a8_silu_gemm_nax(
    x: mx.array,
    weight_q: mx.array,
    *,
    w_scales: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    """SiLU(x) -> int8 quantize -> W8A8 GEMM, fused pipeline."""
    orig_shape = x.shape
    K = orig_shape[-1]
    N = weight_q.shape[0]

    x_2d = mx.reshape(x, (-1, K)).astype(mx.float16)
    x_q, x_scales = _ext.fused_silu_quant(x_2d)

    w_scales = w_scales.astype(mx.float32)
    bias_data = bias if bias is not None else mx.zeros((N,), dtype=mx.float32)

    y = _ext.w8a8_gemm(x_q, weight_q, x_scales, w_scales, bias_data)

    out_shape = orig_shape[:-1] + (N,)
    return mx.reshape(y, out_shape)
