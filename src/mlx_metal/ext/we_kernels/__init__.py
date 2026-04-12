"""World Engine Metal kernels — custom MLX C++ extensions for Apple Silicon."""
from __future__ import annotations

import sys as _sys

if _sys.platform != "darwin":
    raise ImportError("we_kernels requires macOS (Apple Silicon)")

import mlx.core as mx

from we_kernels import _ext


def fused_quant(x: mx.array) -> tuple[mx.array, mx.array]:
    """Plain per-row symmetric int8 quantization (no RMSNorm).

    Single Metal dispatch replacing Python-side abs+max+div+round+clip.

    Returns (x_q [M, K] int8, x_scales [M] fp32).
    """
    x_2d = mx.reshape(x, (-1, x.shape[-1])).astype(mx.float16)
    result = _ext.fused_quant(x_2d)
    return result[0], result[1]


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

    # Minimize graph nodes: skip redundant reshape/astype when possible
    x_2d = x if (x.ndim == 2 and x.dtype == mx.float16) else mx.reshape(x, (-1, K)).astype(mx.float16)

    # Fused Metal kernel: single dispatch for activation quantization
    x_q, x_scales = _ext.fused_quant(x_2d)

    bias_data = bias if bias is not None else mx.zeros((N,), dtype=mx.float32)
    y = _ext.w8a8_gemm(x_q, weight_q, x_scales, w_scales, bias_data)

    return y if len(orig_shape) == 2 else mx.reshape(y, orig_shape[:-1] + (N,))


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
    bias_data = bias if bias is not None else mx.zeros((N,), dtype=mx.float32)
    return _ext.w8a8_gemm(x_q, weight_q, x_scales, w_scales, bias_data)


def scatter_sdpa(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    block_offsets: mx.array,
    scale: float,
) -> mx.array:
    """Scatter-read flash attention.

    Fused SDPA that reads K/V from cache at valid block offsets.
    Uses NAX MMA and online softmax. Tile config selected in C++.

    Parameters
    ----------
    Q : mx.array, fp16, shape [N_Q, T, D_HEAD]
    K : mx.array, fp16, shape [N_KV, capacity, D_HEAD]
    V : mx.array, fp16, shape [N_KV, capacity, D_HEAD]
    block_offsets : mx.array, int32, shape [N_BLOCKS] — BK-aligned token offsets
    scale : float — typically 1/sqrt(D_HEAD)

    Returns
    -------
    mx.array, fp16, shape [N_Q, T, D_HEAD]
    """
    return _ext.scatter_sdpa(
        Q.astype(mx.float16),
        K.astype(mx.float16),
        V.astype(mx.float16),
        block_offsets.astype(mx.int32),
        float(scale),
    )


def seq_sdpa(
    Q: mx.array,
    K: mx.array,
    V: mx.array,
    num_kv_tokens: int,
    scale: float,
) -> mx.array:
    """Sequential-scan attention with contiguous K/V (no block_offsets).

    K/V are read sequentially from offset 0 to num_kv_tokens. Q reads
    from device memory (L2 cached). Tile config selected in C++.

    Parameters
    ----------
    Q : mx.array, fp16, shape [N_Q, T, D_HEAD]
    K : mx.array, fp16, shape [N_KV, capacity, D_HEAD] — reads [0, num_kv_tokens)
    V : mx.array, fp16, shape [N_KV, capacity, D_HEAD]
    num_kv_tokens : int — number of valid KV tokens (contiguous from 0)
    scale : float — typically 1/sqrt(D_HEAD)

    Returns
    -------
    mx.array, fp16, shape [N_Q, T, D_HEAD]
    """
    return _ext.seq_sdpa(
        Q.astype(mx.float16),
        K.astype(mx.float16),
        V.astype(mx.float16),
        int(num_kv_tokens),
        float(scale),
    )


def kv_cache_upsert(
    cache_k: mx.array,
    cache_v: mx.array,
    k_new: mx.array,
    v_new: mx.array,
    rs: int,
) -> tuple[mx.array, mx.array]:
    """In-place KV cache ring buffer upsert.

    Writes k_new/v_new [N_KV, T, D] into cache [N_KV, L, D] at token
    offset rs. Single Metal dispatch instead of MLX slice assignment
    which copies the entire cache tensor.

    Returns (cache_k_updated, cache_v_updated).
    """
    result = _ext.kv_cache_upsert(cache_k, cache_v, k_new, v_new, int(rs))
    return result[0], result[1]


def fused_qkv_norm_rope(
    qkv: mx.array,
    rope_cos: mx.array,
    rope_sin: mx.array,
    n_q: int,
    n_k: int,
    n_v: int,
    eps: float = 1e-5,
) -> tuple[mx.array, mx.array, mx.array]:
    """Fused QKV split + per-head RMSNorm + OrthoRoPE.

    Takes flat QKV GEMM output and produces head-split Q, K (with norm+rope)
    and V (transposed only). Eliminates separate split/reshape/norm/rope ops.

    Parameters
    ----------
    qkv : mx.array, fp16, shape [T, (N_Q+N_K+N_V)*D_HEAD]
    rope_cos : mx.array, fp16, shape [T, D_ROPE]
    rope_sin : mx.array, fp16, shape [T, D_ROPE]
    n_q, n_k, n_v : number of Q, K, V heads
    eps : RMSNorm epsilon

    Returns
    -------
    tuple of (q [N_Q, T, D_HEAD], k [N_K, T, D_HEAD], v [N_V, T, D_HEAD])
    """
    qkv_2d = mx.reshape(qkv, (-1, qkv.shape[-1])).astype(mx.float16)
    # rope_cos/sin come as [1, 1, T, D_ROPE] — flatten to [T, D_ROPE]
    rc = mx.reshape(rope_cos, (-1, rope_cos.shape[-1])).astype(mx.float16)
    rs = mx.reshape(rope_sin, (-1, rope_sin.shape[-1])).astype(mx.float16)
    result = _ext.fused_qkv_norm_rope(qkv_2d, rc, rs, n_q, n_k, n_v, eps)
    return result[0], result[1], result[2]


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

    x_2d = x if (x.ndim == 2 and x.dtype == mx.float16) else mx.reshape(x, (-1, K)).astype(mx.float16)
    x_q, x_scales = _ext.fused_silu_quant(x_2d)

    bias_data = bias if bias is not None else mx.zeros((N,), dtype=mx.float32)
    y = _ext.w8a8_gemm(x_q, weight_q, x_scales, w_scales, bias_data)

    return y if len(orig_shape) == 2 else mx.reshape(y, orig_shape[:-1] + (N,))
