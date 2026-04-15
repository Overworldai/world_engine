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
    scatter_sdpa,
    seq_sdpa,
    seq_sdpa_int8block,
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

def _blocks_from_valid(valid_offsets, BK=32):
    """Convert a list of valid token offsets to BK-aligned block offsets."""
    blocks = sorted(set(off - (off % BK) for off in valid_offsets))
    return np.array(blocks, dtype=np.int32)


class TestScatterSDPA:
    """Tests scatter-read flash attention against MLX SDPA."""

    def test_all_valid(self):
        """All KV slots valid — should match standard SDPA."""
        mx.random.seed(42)
        N_H, T_Q, D = 4, 16, 64
        cap = 64
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        block_offsets = mx.array(np.arange(0, cap, 32, dtype=np.int32))  # [0, 32]

        result = scatter_sdpa(Q, K, V, block_offsets, scale)
        mx.eval(result)

        ref = mx.fast.scaled_dot_product_attention(
            mx.expand_dims(Q, 0), mx.expand_dims(K, 0),
            mx.expand_dims(V, 0), scale=scale,
        )
        ref_3d = mx.reshape(ref, (N_H, T_Q, D))
        mx.eval(ref_3d)

        res_np = np.array(result).astype(np.float32)
        ref_np = np.array(ref_3d).astype(np.float32)
        cos_sim = np.dot(res_np.flatten(), ref_np.flatten()) / (
            np.linalg.norm(res_np.flatten()) * np.linalg.norm(ref_np.flatten()) + 1e-12
        )
        assert cos_sim > 0.99, f"All-valid cosine sim {cos_sim:.6f} too low"
        assert np.all(np.isfinite(res_np))

    def test_partial_blocks(self):
        """Only some blocks valid — kernel should attend only to those."""
        mx.random.seed(42)
        N_H, T_Q, D = 4, 32, 64
        cap = 128
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)

        # Only blocks at offsets 0, 32, 96 are valid (skip 64)
        block_offsets = mx.array([0, 32, 96], dtype=mx.int32)
        result = scatter_sdpa(Q, K, V, block_offsets, scale)
        mx.eval(result)

        # Reference: gather + SDPA
        valid_idx = np.concatenate([np.arange(0, 64), np.arange(96, 128)]).astype(np.int32)
        K_c = K[:, mx.array(valid_idx), :]
        V_c = V[:, mx.array(valid_idx), :]
        ref = mx.fast.scaled_dot_product_attention(
            mx.expand_dims(Q, 0), mx.expand_dims(K_c, 0),
            mx.expand_dims(V_c, 0), scale=scale,
        )
        ref_3d = mx.reshape(ref, (N_H, T_Q, D))
        mx.eval(ref_3d)

        res_np = np.array(result).astype(np.float32)
        ref_np = np.array(ref_3d).astype(np.float32)
        cos_sim = np.dot(res_np.flatten(), ref_np.flatten()) / (
            np.linalg.norm(res_np.flatten()) * np.linalg.norm(ref_np.flatten()) + 1e-12
        )
        assert cos_sim > 0.99, f"Partial-blocks cosine sim {cos_sim:.6f} too low"

    def test_gqa(self):
        """GQA: N_Q=32, N_KV=16."""
        mx.random.seed(42)
        N_Q, N_KV, T_Q, D = 32, 16, 32, 64
        cap = 64
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_Q, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_KV, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_KV, cap, D)).astype(mx.float16)
        block_offsets = mx.array([0, 32], dtype=mx.int32)

        result = scatter_sdpa(Q, K, V, block_offsets, scale)
        mx.eval(result)

        # Reference: expand K/V for GQA then SDPA
        group_size = N_Q // N_KV
        K_exp = mx.repeat(K, group_size, axis=0)
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

    def test_model_shapes(self):
        """Test with actual model shapes: 32 Q heads, 16 KV heads, cap=8704."""
        mx.random.seed(42)
        N_Q, N_KV, T_Q, D = 32, 16, 512, 64
        cap = 8704
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_Q, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_KV, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_KV, cap, D)).astype(mx.float16)

        # 5 ring buckets (each 512 tokens = 16 blocks of 32) + tail (16 blocks)
        offsets = []
        for bucket in range(5):
            for blk in range(16):
                offsets.append(bucket * 512 + blk * 32)
        for blk in range(16):
            offsets.append(8192 + blk * 32)
        block_offsets = mx.array(offsets, dtype=mx.int32)

        result = scatter_sdpa(Q, K, V, block_offsets, scale)
        mx.eval(result)

        res_np = np.array(result).astype(np.float32)
        assert np.all(np.isfinite(res_np)), "Model-shape output has NaN/Inf"
        assert res_np.shape == (N_Q, T_Q, D)

    def test_deterministic(self):
        """Same input produces identical output."""
        mx.random.seed(42)
        N_H, T_Q, D = 4, 16, 64
        cap = 64
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        block_offsets = mx.array([0, 32], dtype=mx.int32)

        r1 = scatter_sdpa(Q, K, V, block_offsets, scale)
        r2 = scatter_sdpa(Q, K, V, block_offsets, scale)
        mx.eval(r1, r2)
        np.testing.assert_array_equal(np.array(r1), np.array(r2))

    # --- Int8 block SDPA tests ---

    def test_int8block_vs_fp16(self):
        """Int8 block SDPA should match fp16 SDPA within quantization tolerance."""
        mx.random.seed(42)
        N_Q, N_KV, T_Q, D = 32, 32, 8, 64
        cap = 256
        BK = 32
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_Q, T_Q, D)).astype(mx.float16)
        K_fp = mx.random.normal((N_KV, cap, D)).astype(mx.float16) * 0.5
        V_fp = mx.random.normal((N_KV, cap, D)).astype(mx.float16) * 0.5
        mx.eval(Q, K_fp, V_fp)

        # Quantize K, V per-block
        from we_kernels import fused_quant
        K_flat = mx.reshape(K_fp, (N_KV * (cap // BK), BK * D))
        K_q, K_s = fused_quant(K_flat)
        K_q = mx.reshape(K_q, (N_KV, cap, D))
        K_s = mx.reshape(K_s.astype(mx.float16), (N_KV, cap // BK))
        V_flat = mx.reshape(V_fp, (N_KV * (cap // BK), BK * D))
        V_q, V_s = fused_quant(V_flat)
        V_q = mx.reshape(V_q, (N_KV, cap, D))
        V_s = mx.reshape(V_s.astype(mx.float16), (N_KV, cap // BK))

        num_kv = 128
        O_fp = seq_sdpa(Q, K_fp, V_fp, num_kv, scale)
        O_i8 = seq_sdpa_int8block(Q, K_q, K_s, V_q, V_s, num_kv, scale)
        mx.eval(O_fp, O_i8)

        res = np.array(O_i8).astype(np.float32)
        ref = np.array(O_fp).astype(np.float32)
        mae = np.abs(res - ref).mean()
        ref_max = np.abs(ref).max()
        assert mae / ref_max < 0.005, f"Int8 block MAE {mae:.6f} / ref_max {ref_max:.4f} = {mae/ref_max*100:.2f}% (>0.5%)"
        assert np.all(np.isfinite(res))

    def test_int8block_large_kv(self):
        """Int8 block SDPA at production shape (8192 KV tokens)."""
        mx.random.seed(42)
        N_Q, N_KV, T_Q, D = 32, 32, 512, 64
        cap = 8192
        BK = 32
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_Q, T_Q, D)).astype(mx.float16)
        K_fp = mx.random.normal((N_KV, cap, D)).astype(mx.float16) * 0.5
        V_fp = mx.random.normal((N_KV, cap, D)).astype(mx.float16) * 0.5
        mx.eval(Q, K_fp, V_fp)

        from we_kernels import fused_quant
        K_flat = mx.reshape(K_fp, (N_KV * (cap // BK), BK * D))
        K_q, K_s = fused_quant(K_flat)
        K_q = mx.reshape(K_q, (N_KV, cap, D))
        K_s = mx.reshape(K_s.astype(mx.float16), (N_KV, cap // BK))
        V_flat = mx.reshape(V_fp, (N_KV * (cap // BK), BK * D))
        V_q, V_s = fused_quant(V_flat)
        V_q = mx.reshape(V_q, (N_KV, cap, D))
        V_s = mx.reshape(V_s.astype(mx.float16), (N_KV, cap // BK))

        O = seq_sdpa_int8block(Q, K_q, K_s, V_q, V_s, cap, scale)
        mx.eval(O)
        res = np.array(O).astype(np.float32)
        assert res.shape == (N_Q, T_Q, D)
        assert np.all(np.isfinite(res)), "Large-KV int8 block output has NaN/Inf"

    def test_int8block_deterministic(self):
        """Same input produces identical int8 block output."""
        mx.random.seed(42)
        N_Q, N_KV, T_Q, D = 4, 4, 8, 64
        cap = 128
        BK = 32
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_Q, T_Q, D)).astype(mx.float16)
        K_q = mx.random.randint(-127, 127, (N_KV, cap, D)).astype(mx.int8)
        K_s = mx.random.uniform(shape=(N_KV, cap // BK)).astype(mx.float16) * 0.02
        V_q = mx.random.randint(-127, 127, (N_KV, cap, D)).astype(mx.int8)
        V_s = mx.random.uniform(shape=(N_KV, cap // BK)).astype(mx.float16) * 0.02
        mx.eval(Q, K_q, K_s, V_q, V_s)

        r1 = seq_sdpa_int8block(Q, K_q, K_s, V_q, V_s, 64, scale)
        r2 = seq_sdpa_int8block(Q, K_q, K_s, V_q, V_s, 64, scale)
        mx.eval(r1, r2)
        np.testing.assert_array_equal(np.array(r1), np.array(r2))

    def test_kv_cache_upsert_int8_block(self):
        """Int8 block KV cache upsert roundtrip."""
        N_KV, L, T_Q, D = 4, 256, 16, 64
        BK = 32
        rs = 32

        cache_k_q = mx.zeros((1, N_KV, L, D), dtype=mx.int8)
        cache_k_s = mx.zeros((1, N_KV, L // BK), dtype=mx.float16)
        cache_v_q = mx.zeros((1, N_KV, L, D), dtype=mx.int8)
        cache_v_s = mx.zeros((1, N_KV, L // BK), dtype=mx.float16)

        k_new_q = mx.random.randint(-127, 127, (1, N_KV, T_Q, D)).astype(mx.int8)
        k_new_s = mx.random.uniform(shape=(1, N_KV, T_Q // BK)).astype(mx.float16) * 0.05
        v_new_q = mx.random.randint(-127, 127, (1, N_KV, T_Q, D)).astype(mx.int8)
        v_new_s = mx.random.uniform(shape=(1, N_KV, T_Q // BK)).astype(mx.float16) * 0.05
        mx.eval(cache_k_q, cache_k_s, cache_v_q, cache_v_s, k_new_q, k_new_s, v_new_q, v_new_s)

        from we_kernels import kv_cache_upsert_int8_block
        rs_blk = rs // BK
        ck, cs, cv, vs = kv_cache_upsert_int8_block(
            cache_k_q, cache_k_s, cache_v_q, cache_v_s,
            k_new_q, k_new_s, v_new_q, v_new_s, rs, rs_blk)
        mx.eval(ck, cs, cv, vs)

        # Verify data was written at the correct offset
        assert bool(mx.all(ck[:, :, rs:rs+T_Q] == k_new_q))
        assert bool(mx.all(cs[:, :, rs_blk:rs_blk+T_Q//BK] == k_new_s))
        assert bool(mx.all(cv[:, :, rs:rs+T_Q] == v_new_q))
        assert bool(mx.all(vs[:, :, rs_blk:rs_blk+T_Q//BK] == v_new_s))

    def test_finite_single_block(self):
        """Single block should produce finite results."""
        mx.random.seed(42)
        N_H, T_Q, D = 4, 32, 64
        cap = 128
        scale = 1.0 / np.sqrt(D)

        Q = mx.random.normal((N_H, T_Q, D)).astype(mx.float16)
        K = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        V = mx.random.normal((N_H, cap, D)).astype(mx.float16)
        block_offsets = mx.array([64], dtype=mx.int32)

        result = scatter_sdpa(Q, K, V, block_offsets, scale)
        mx.eval(result)
        assert np.all(np.isfinite(np.array(result)))
