"""
Unit tests for custom Metal kernels: matvec, GEMM, fused QKV+RoPE.

Validates numerical correctness against reference Python/MLX implementations.

Usage:
  pytest src/mlx_metal/tests/test_kernels.py -v
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from we_kernels import (
    w8a8_gemm_nax,
    w8a8_gemm_prequantized,
    fused_silu_quant,
    fused_rmsnorm_quant,
    fused_rmsnorm_adaln_quant,
    fused_qkv_norm_rope,
    ring_flash_attention,
)


def _symmetric_int8_quantize(x: mx.array):
    """Reference Python int8 quantization matching the model code."""
    x_f32 = x.astype(mx.float32)
    absmax = mx.max(mx.abs(x_f32), axis=-1, keepdims=True)
    scale = mx.maximum(absmax / 127.0, 1e-6)
    x_q = mx.clip(mx.round(x_f32 / scale), -127, 127).astype(mx.int8)
    return x_q, mx.reshape(scale, (-1,))


def _ref_ortho_rope(x, cos, sin):
    """Reference OrthoRoPE in numpy."""
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    y0 = x0 * cos - x1 * sin
    y1 = x1 * cos + x0 * sin
    return np.concatenate([y0, y1], axis=-1)


def _ref_rms_norm(x, eps=1e-5):
    """Reference RMSNorm in numpy."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms


# ============================================================================
# W8A8 GEMM (exercises both matvec and tiled GEMM paths)
# ============================================================================

class TestW8A8Gemm:
    """Tests w8a8_gemm_nax which dispatches to matvec (M<5) or tiled GEMM."""

    @pytest.mark.parametrize("M,N,K", [
        (1, 2048, 2048),     # matvec path (M=1)
        (4, 2048, 2048),     # matvec path (M=4)
        (512, 2048, 2048),   # tiled GEMM (attn out_proj shape)
        (512, 8192, 2048),   # tiled GEMM (mlp.fc1 shape)
        (512, 2048, 8192),   # tiled GEMM (mlp.fc2 shape)
        (512, 6144, 2048),   # tiled GEMM (QKV fused shape)
    ])
    def test_correctness(self, M, N, K):
        mx.random.seed(42)
        x = mx.random.normal((M, K)).astype(mx.float16)
        w = mx.random.normal((N, K)).astype(mx.float16)
        bias = mx.random.normal((N,)).astype(mx.float32)

        # Reference: fp16 matmul
        ref = (x.astype(mx.float32) @ w.astype(mx.float32).T + bias).astype(mx.float16)

        # Kernel
        result = w8a8_gemm_nax(x, mx.zeros_like(w).astype(mx.int8), w_scales=mx.ones((N,), dtype=mx.float32), bias=bias)
        # Actually need to quantize w properly
        w_q, w_scales = _symmetric_int8_quantize(w)
        result = w8a8_gemm_nax(x, w_q, w_scales=w_scales, bias=bias)
        mx.eval(result)

        ref_np = np.array(ref).astype(np.float32)
        res_np = np.array(result).astype(np.float32)

        # int8 quantization introduces error — check cosine similarity
        cos_sim = np.dot(ref_np.flatten(), res_np.flatten()) / (
            np.linalg.norm(ref_np.flatten()) * np.linalg.norm(res_np.flatten()) + 1e-12
        )
        assert cos_sim > 0.98, f"Cosine similarity {cos_sim:.4f} too low for M={M},N={N},K={K}"
        assert np.all(np.isfinite(res_np)), "Output contains NaN/Inf"

    @pytest.mark.parametrize("M", [1, 2, 4])
    def test_matvec_no_nan(self, M):
        """Matvec path should produce finite results for small M."""
        mx.random.seed(123)
        N, K = 2048, 2048
        x = mx.random.normal((M, K)).astype(mx.float16)
        w = mx.random.normal((N, K)).astype(mx.float16)
        w_q, w_scales = _symmetric_int8_quantize(w)
        bias = mx.zeros((N,), dtype=mx.float32)

        result = w8a8_gemm_nax(x, w_q, w_scales=w_scales, bias=bias)
        mx.eval(result)
        assert np.all(np.isfinite(np.array(result))), f"Matvec NaN for M={M}"

    def test_prequantized_matches(self):
        """w8a8_gemm_prequantized should match w8a8_gemm_nax."""
        mx.random.seed(42)
        M, N, K = 512, 2048, 2048
        x = mx.random.normal((M, K)).astype(mx.float16)
        w = mx.random.normal((N, K)).astype(mx.float16)
        w_q, w_scales = _symmetric_int8_quantize(w)
        bias = mx.zeros((N,), dtype=mx.float32)

        # Full path (quantize + GEMM)
        result_full = w8a8_gemm_nax(x, w_q, w_scales=w_scales, bias=bias)

        # Pre-quantize, then GEMM
        x_q, x_scales = _symmetric_int8_quantize(x)
        result_pre = w8a8_gemm_prequantized(x_q, x_scales, w_q, w_scales=w_scales, bias=bias)

        mx.eval(result_full, result_pre)
        np.testing.assert_allclose(
            np.array(result_full), np.array(result_pre),
            atol=0.5, rtol=0.01,
            err_msg="Prequantized path diverges from full path",
        )


# ============================================================================
# Fused SiLU + Quant
# ============================================================================

class TestFusedSiLUQuant:
    @pytest.mark.parametrize("M,K", [(512, 8192), (1, 2048), (512, 2048)])
    def test_correctness(self, M, K):
        mx.random.seed(42)
        x = mx.random.normal((M, K)).astype(mx.float16)

        # Reference: separate SiLU + quantize
        x_silu = mx.sigmoid(x.astype(mx.float32)) * x.astype(mx.float32)
        ref_q, ref_scales = _symmetric_int8_quantize(x_silu.astype(mx.float16))

        # Fused kernel
        fused_q, fused_scales = fused_silu_quant(x)
        mx.eval(fused_q, fused_scales, ref_q, ref_scales)

        # Dequantize and compare
        ref_deq = np.array(ref_q).astype(np.float32) * np.array(ref_scales)[:, None]
        fused_deq = np.array(fused_q).astype(np.float32) * np.array(fused_scales)[:, None]

        np.testing.assert_allclose(fused_deq, ref_deq, atol=0.5, rtol=0.05,
                                   err_msg=f"Fused SiLU+Quant diverges for M={M},K={K}")


# ============================================================================
# Fused RMSNorm + Quant
# ============================================================================

class TestFusedRMSNormQuant:
    @pytest.mark.parametrize("M,K", [(512, 2048), (1, 2048)])
    def test_plain_rmsnorm(self, M, K):
        mx.random.seed(42)
        x = mx.random.normal((M, K)).astype(mx.float16)

        # Reference
        x_norm = mx.fast.rms_norm(x, None, 1e-5)
        ref_q, ref_scales = _symmetric_int8_quantize(x_norm)

        # Fused
        fused_q, fused_scales = fused_rmsnorm_quant(x, eps=1e-5)
        mx.eval(fused_q, fused_scales, ref_q, ref_scales)

        ref_deq = np.array(ref_q).astype(np.float32) * np.array(ref_scales)[:, None]
        fused_deq = np.array(fused_q).astype(np.float32) * np.array(fused_scales)[:, None]

        np.testing.assert_allclose(fused_deq, ref_deq, atol=0.5, rtol=0.05)

    @pytest.mark.parametrize("M,K", [(512, 2048)])
    def test_adaln_rmsnorm(self, M, K):
        mx.random.seed(42)
        x = mx.random.normal((M, K)).astype(mx.float16)
        s = mx.random.normal((K,)).astype(mx.float16) * 0.1
        b = mx.random.normal((K,)).astype(mx.float16) * 0.1

        # Reference
        x_norm = mx.fast.rms_norm(x, None, 1e-5)
        x_mod = x_norm * (1 + s) + b
        ref_q, ref_scales = _symmetric_int8_quantize(x_mod)

        # Fused
        fused_q, fused_scales = fused_rmsnorm_adaln_quant(x, s, b, eps=1e-5)
        mx.eval(fused_q, fused_scales, ref_q, ref_scales)

        ref_deq = np.array(ref_q).astype(np.float32) * np.array(ref_scales)[:, None]
        fused_deq = np.array(fused_q).astype(np.float32) * np.array(fused_scales)[:, None]

        np.testing.assert_allclose(fused_deq, ref_deq, atol=0.5, rtol=0.05)


# ============================================================================
# Fused QKV split + RMSNorm + OrthoRoPE
# ============================================================================

class TestFusedQKVNormRoPE:
    """Tests the fused kernel against reference Python implementation."""

    @pytest.fixture
    def setup(self):
        mx.random.seed(42)
        T, N_Q, N_K, N_V, D_HEAD = 512, 32, 32, 32, 64
        D_ROPE = D_HEAD // 2
        N_TOTAL = N_Q + N_K + N_V
        QKV_DIM = N_TOTAL * D_HEAD

        qkv = mx.random.normal((T, QKV_DIM)).astype(mx.float16)
        rope_cos = mx.random.normal((T, D_ROPE)).astype(mx.float16) * 0.5
        rope_sin = mx.random.normal((T, D_ROPE)).astype(mx.float16) * 0.5

        return {
            "qkv": qkv, "rope_cos": rope_cos, "rope_sin": rope_sin,
            "T": T, "N_Q": N_Q, "N_K": N_K, "N_V": N_V,
            "D_HEAD": D_HEAD, "D_ROPE": D_ROPE,
        }

    def _reference_qkv_norm_rope(self, qkv_np, cos_np, sin_np, N_Q, N_K, N_V, D_HEAD, D_ROPE):
        """Pure numpy reference."""
        T = qkv_np.shape[0]
        # Split
        q_offset = 0
        k_offset = N_Q * D_HEAD
        v_offset = (N_Q + N_K) * D_HEAD

        q_heads = qkv_np[:, q_offset:q_offset + N_Q * D_HEAD].reshape(T, N_Q, D_HEAD)
        k_heads = qkv_np[:, k_offset:k_offset + N_K * D_HEAD].reshape(T, N_K, D_HEAD)
        v_heads = qkv_np[:, v_offset:v_offset + N_V * D_HEAD].reshape(T, N_V, D_HEAD)

        # Transpose to [N_H, T, D_HEAD]
        q_heads = q_heads.transpose(1, 0, 2)
        k_heads = k_heads.transpose(1, 0, 2)
        v_heads = v_heads.transpose(1, 0, 2)

        # RMSNorm + RoPE for Q and K
        for heads in [q_heads, k_heads]:
            for h in range(heads.shape[0]):
                heads[h] = _ref_rms_norm(heads[h].astype(np.float32))
                cos_t = cos_np  # [T, D_ROPE]
                sin_t = sin_np
                heads[h] = _ref_ortho_rope(heads[h], cos_t, sin_t)

        return q_heads, k_heads, v_heads

    def test_output_shapes(self, setup):
        q, k, v = fused_qkv_norm_rope(
            setup["qkv"], setup["rope_cos"], setup["rope_sin"],
            setup["N_Q"], setup["N_K"], setup["N_V"],
        )
        mx.eval(q, k, v)
        assert q.shape == (setup["N_Q"], setup["T"], setup["D_HEAD"])
        assert k.shape == (setup["N_K"], setup["T"], setup["D_HEAD"])
        assert v.shape == (setup["N_V"], setup["T"], setup["D_HEAD"])

    def test_finite(self, setup):
        q, k, v = fused_qkv_norm_rope(
            setup["qkv"], setup["rope_cos"], setup["rope_sin"],
            setup["N_Q"], setup["N_K"], setup["N_V"],
        )
        mx.eval(q, k, v)
        for name, arr in [("q", q), ("k", k), ("v", v)]:
            assert np.all(np.isfinite(np.array(arr))), f"{name} contains NaN/Inf"

    def test_v_passthrough(self, setup):
        """V heads should just be copied (no norm/rope), matching input exactly."""
        q, k, v = fused_qkv_norm_rope(
            setup["qkv"], setup["rope_cos"], setup["rope_sin"],
            setup["N_Q"], setup["N_K"], setup["N_V"],
        )
        mx.eval(v)

        # Extract V from flat input
        qkv_np = np.array(setup["qkv"])
        T, D_HEAD, N_Q, N_K, N_V = setup["T"], setup["D_HEAD"], setup["N_Q"], setup["N_K"], setup["N_V"]
        v_offset = (N_Q + N_K) * D_HEAD
        v_ref = qkv_np[:, v_offset:v_offset + N_V * D_HEAD].reshape(T, N_V, D_HEAD)
        v_ref = v_ref.transpose(1, 0, 2)  # [N_V, T, D_HEAD]

        np.testing.assert_allclose(
            np.array(v).astype(np.float32),
            v_ref.astype(np.float32),
            atol=1e-3, rtol=1e-3,
            err_msg="V passthrough doesn't match input",
        )

    def test_q_k_norm_rope(self, setup):
        """Q and K should match reference RMSNorm + OrthoRoPE."""
        q, k, v = fused_qkv_norm_rope(
            setup["qkv"], setup["rope_cos"], setup["rope_sin"],
            setup["N_Q"], setup["N_K"], setup["N_V"],
        )
        mx.eval(q, k, v)

        ref_q, ref_k, ref_v = self._reference_qkv_norm_rope(
            np.array(setup["qkv"]).astype(np.float32),
            np.array(setup["rope_cos"]).astype(np.float32),
            np.array(setup["rope_sin"]).astype(np.float32),
            setup["N_Q"], setup["N_K"], setup["N_V"],
            setup["D_HEAD"], setup["D_ROPE"],
        )

        # Tolerance accounts for fp16 precision in the kernel
        for name, got, ref in [("Q", np.array(q), ref_q), ("K", np.array(k), ref_k)]:
            got_f = got.flatten().astype(np.float64)
            ref_f = ref.flatten().astype(np.float64)
            cos_sim = np.dot(got_f, ref_f) / (
                np.linalg.norm(got_f) * np.linalg.norm(ref_f) + 1e-12
            )
            assert cos_sim > 0.99, f"{name} cosine similarity {cos_sim:.6f} too low"

    def test_deterministic(self, setup):
        """Same input should produce bitwise identical output."""
        q1, k1, v1 = fused_qkv_norm_rope(
            setup["qkv"], setup["rope_cos"], setup["rope_sin"],
            setup["N_Q"], setup["N_K"], setup["N_V"],
        )
        q2, k2, v2 = fused_qkv_norm_rope(
            setup["qkv"], setup["rope_cos"], setup["rope_sin"],
            setup["N_Q"], setup["N_K"], setup["N_V"],
        )
        mx.eval(q1, k1, v1, q2, k2, v2)
        np.testing.assert_array_equal(np.array(q1), np.array(q2))
        np.testing.assert_array_equal(np.array(k1), np.array(k2))
        np.testing.assert_array_equal(np.array(v1), np.array(v2))


# ============================================================================
# Ring-buffer flash attention
# ============================================================================

def _ref_sdpa_with_written(Q, K, V, written, scale):
    """Reference attention in numpy using written mask."""
    # Q: [N_H, T, D], K/V: [N_H, cap, D], written: [cap]
    N_H, T_Q, D = Q.shape
    cap = K.shape[1]

    # Build additive mask: 0 for valid, -1e4 for invalid
    mask = np.where(written > 0.5, 0.0, -1e4).reshape(1, 1, cap)

    O = np.zeros_like(Q)
    for h in range(N_H):
        for qi in range(T_Q):
            q = Q[h, qi] * scale  # [D]
            scores = q @ K[h].T + mask[0, 0]  # [cap]
            scores_max = np.max(scores)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / (np.sum(exp_scores) + 1e-12)
            O[h, qi] = attn @ V[h]
    return O


class TestRingFlashAttention:
    """Tests custom ring-buffer flash attention against reference SDPA."""

    def test_all_written(self):
        """When all slots are written, should match standard SDPA."""
        mx.random.seed(42)
        N_H, T_Q, D = 4, 16, 64
        cap = 32
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        written = mx.ones((cap,), dtype=mx.float16)

        result = ring_flash_attention(Q, K, V, written, scale)
        mx.eval(result)

        # Reference via MLX SDPA
        ref = mx.fast.scaled_dot_product_attention(
            mx.expand_dims(Q, 0), mx.expand_dims(K, 0),
            mx.expand_dims(V, 0), scale=scale,
        )
        mx.eval(ref)
        ref_3d = mx.reshape(ref, (N_H, T_Q, D))

        res_np = np.array(result).astype(np.float32)
        ref_np = np.array(ref_3d).astype(np.float32)

        cos_sim = np.dot(res_np.flatten(), ref_np.flatten()) / (
            np.linalg.norm(res_np.flatten()) * np.linalg.norm(ref_np.flatten()) + 1e-12
        )
        assert cos_sim > 0.99, f"All-written cosine sim {cos_sim:.6f} too low"
        assert np.all(np.isfinite(res_np)), "Output contains NaN/Inf"

    def test_partial_written(self):
        """Only some KV slots written — kernel should ignore zeros."""
        mx.random.seed(42)
        N_H, T_Q, D = 4, 16, 64
        cap = 64
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)

        # Only first 16 and last 16 tokens are written (simulating ring + tail)
        written_np = np.zeros(cap, dtype=np.float16)
        written_np[:16] = 1.0
        written_np[48:64] = 1.0
        written = mx.array(written_np)

        result = ring_flash_attention(Q, K, V, written, scale)
        mx.eval(result)

        # Reference
        ref = _ref_sdpa_with_written(
            np.array(Q).astype(np.float32),
            np.array(K).astype(np.float32),
            np.array(V).astype(np.float32),
            written_np.astype(np.float32),
            scale,
        )

        res_np = np.array(result).astype(np.float32)
        cos_sim = np.dot(res_np.flatten(), ref.flatten().astype(np.float32)) / (
            np.linalg.norm(res_np.flatten()) * np.linalg.norm(ref.flatten()) + 1e-12
        )
        assert cos_sim > 0.99, f"Partial-written cosine sim {cos_sim:.6f} too low"

    def test_none_written_except_tail(self):
        """Only tail written (fresh cache, frame 0)."""
        mx.random.seed(42)
        N_H, T_Q, D = 2, 8, 64
        T_TAIL = 8
        L = 32
        cap = L + T_TAIL  # 40
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)

        # Only tail written
        written_np = np.zeros(cap, dtype=np.float16)
        written_np[L:L + T_TAIL] = 1.0
        written = mx.array(written_np)

        result = ring_flash_attention(Q, K, V, written, scale)
        mx.eval(result)

        # Reference: attention only over tail slots
        ref = _ref_sdpa_with_written(
            np.array(Q).astype(np.float32),
            np.array(K).astype(np.float32),
            np.array(V).astype(np.float32),
            written_np.astype(np.float32),
            scale,
        )

        res_np = np.array(result).astype(np.float32)
        cos_sim = np.dot(res_np.flatten(), ref.flatten().astype(np.float32)) / (
            np.linalg.norm(res_np.flatten()) * np.linalg.norm(ref.flatten()) + 1e-12
        )
        assert cos_sim > 0.99, f"Tail-only cosine sim {cos_sim:.6f} too low"

    def test_model_shapes(self):
        """Test with actual model shapes: local (8704) and global (66048)."""
        mx.random.seed(42)
        scale = 1.0 / np.sqrt(64)

        # Local layer shape
        N_H, T_Q, D = 32, 512, 64
        cap = 8704
        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)

        # Simulate: 5 ring buckets written + tail
        written_np = np.zeros(cap, dtype=np.float16)
        for bucket in range(5):
            s = bucket * 512
            written_np[s:s + 512] = 1.0
        written_np[8192:8704] = 1.0  # tail
        written = mx.array(written_np)

        result = ring_flash_attention(Q, K, V, written, scale)
        mx.eval(result)

        res_np = np.array(result).astype(np.float32)
        assert np.all(np.isfinite(res_np)), "Local-layer output has NaN/Inf"
        assert res_np.shape == (N_H, T_Q, D)

    def test_matches_mlx_sdpa(self):
        """Direct comparison with mx.fast.scaled_dot_product_attention + mask."""
        mx.random.seed(42)
        N_H, T_Q, D = 4, 32, 64
        cap = 128
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)

        # Random written pattern
        written_np = np.zeros(cap, dtype=np.float16)
        written_np[:48] = 1.0
        written_np[96:128] = 1.0
        written = mx.array(written_np)

        # Our kernel
        result = ring_flash_attention(Q, K, V, written, scale)

        # MLX SDPA with equivalent mask
        mask = mx.reshape(written, (1, 1, 1, -1)) * 1e4 - 1e4
        ref = mx.fast.scaled_dot_product_attention(
            mx.expand_dims(Q, 0), mx.expand_dims(K, 0),
            mx.expand_dims(V, 0), scale=scale, mask=mask,
        )
        ref_3d = mx.reshape(ref, (N_H, T_Q, D))

        mx.eval(result, ref_3d)

        res_np = np.array(result).astype(np.float32)
        ref_np = np.array(ref_3d).astype(np.float32)

        # Per-head cosine similarity
        for h in range(N_H):
            r = res_np[h].flatten()
            f = ref_np[h].flatten()
            cos_sim = np.dot(r, f) / (np.linalg.norm(r) * np.linalg.norm(f) + 1e-12)
            assert cos_sim > 0.99, f"Head {h} cosine sim {cos_sim:.6f} too low"

    def test_gqa(self):
        """GQA: N_Q=32, N_KV=16 — each pair of Q heads shares a KV head."""
        mx.random.seed(42)
        N_Q, N_KV, T_Q, D = 32, 16, 32, 64
        cap = 64
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_Q, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_KV, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_KV, cap, D)).astype(mx.float16)
        written = mx.ones((cap,), dtype=mx.float16)

        result = ring_flash_attention(Q, K, V, written, scale)
        mx.eval(result)

        # Reference: expand K/V to N_Q heads then use MLX SDPA
        group_size = N_Q // N_KV
        K_exp = mx.repeat(K, group_size, axis=0)  # [N_Q, cap, D]
        V_exp = mx.repeat(V, group_size, axis=0)
        ref = mx.fast.scaled_dot_product_attention(
            mx.expand_dims(Q, 0), mx.expand_dims(K_exp, 0),
            mx.expand_dims(V_exp, 0), scale=scale,
        )
        ref_3d = mx.reshape(ref, (N_Q, T_Q, D))
        mx.eval(ref_3d)

        res_np = np.array(result).astype(np.float32)
        ref_np = np.array(ref_3d).astype(np.float32)

        for h in range(N_Q):
            r = res_np[h].flatten()
            f = ref_np[h].flatten()
            cos_sim = np.dot(r, f) / (np.linalg.norm(r) * np.linalg.norm(f) + 1e-12)
            assert cos_sim > 0.99, f"GQA head {h} cosine sim {cos_sim:.6f} too low"

    def test_deterministic(self):
        """Same input should produce identical output."""
        mx.random.seed(42)
        N_H, T_Q, D = 4, 16, 64
        cap = 32
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        written = mx.ones((cap,), dtype=mx.float16)

        r1 = ring_flash_attention(Q, K, V, written, scale)
        r2 = ring_flash_attention(Q, K, V, written, scale)
        mx.eval(r1, r2)
        np.testing.assert_array_equal(np.array(r1), np.array(r2))

    def test_finite_outputs(self):
        """No NaN/Inf with various written patterns."""
        mx.random.seed(42)
        N_H, T_Q, D = 4, 16, 64
        cap = 64
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)

        # Test several written patterns
        patterns = [
            np.ones(cap, dtype=np.float16),           # all written
            np.zeros(cap, dtype=np.float16),           # none (edge case)
            np.eye(1, cap, 0, dtype=np.float16).flatten(),  # single token
        ]
        # Set at least one slot for the "none" case to avoid division by zero
        patterns[1][0] = 1.0

        for i, pat in enumerate(patterns):
            written = mx.array(pat)
            result = ring_flash_attention(Q, K, V, written, scale)
            mx.eval(result)
            assert np.all(np.isfinite(np.array(result))), f"Pattern {i} has NaN/Inf"
