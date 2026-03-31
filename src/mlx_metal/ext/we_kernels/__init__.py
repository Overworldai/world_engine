"""World Engine Metal kernels — custom MLX C++ extensions for Apple Silicon."""
from __future__ import annotations

import mlx.core as mx

from we_kernels._ext import w8a8_gemm as _w8a8_gemm_raw


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
    M = x_2d.shape[0]

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

    y = _w8a8_gemm_raw(x_q, weight_q, x_scales, w_scales, bias_data)

    out_shape = orig_shape[:-1] + (N,)
    return mx.reshape(y, out_shape)
