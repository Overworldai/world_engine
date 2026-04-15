#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "mlx/mlx.h"
#include "kernels/we_ops.h"

namespace nb = nanobind;
namespace mx = mlx::core;

NB_MODULE(_ext, m) {
  m.doc() = "World Engine Metal kernels — custom MLX C++ extensions for Apple Silicon";

  m.def(
      "w8a8_gemm",
      [](const mx::array& x_q,
         const mx::array& w_q,
         const mx::array& x_scales,
         const mx::array& w_scales,
         const mx::array& bias) {
        return we_kernels::w8a8_gemm(x_q, w_q, x_scales, w_scales, bias);
      },
      nb::arg("x_q"),
      nb::arg("w_q"),
      nb::arg("x_scales"),
      nb::arg("w_scales"),
      nb::arg("bias"),
      R"(W8A8 GEMM using NAX hardware tensor cores.

Args:
    x_q: int8 quantised activations [M, K]
    w_q: int8 quantised weights [N, K]
    x_scales: fp32 per-row activation scales [M]
    w_scales: fp32 per-row weight scales [N]
    bias: fp32 bias [N] (pass zeros for no bias)

Returns:
    fp16 output [M, N])");

  m.def(
      "fused_silu_quant",
      [](const mx::array& x) {
        return we_kernels::fused_silu_quant(x);
      },
      nb::arg("x"),
      R"(Fused SiLU activation + per-row int8 quantization.

Args:
    x: fp16 activations [M, K]

Returns:
    list[int8 x_q [M, K], fp32 x_scales [M]])");

  m.def(
      "fused_quant",
      [](const mx::array& x) {
        return we_kernels::fused_quant(x);
      },
      nb::arg("x"),
      R"(Plain per-row symmetric int8 quantization (no RMSNorm).

Args:
    x: fp16 activations [M, K]

Returns:
    list[int8 x_q [M, K], fp32 x_scales [M]])");

  m.def(
      "fused_rmsnorm_quant",
      [](const mx::array& x, float eps) {
        return we_kernels::fused_rmsnorm_quant(x, eps);
      },
      nb::arg("x"),
      nb::arg("eps") = 1e-5f,
      R"(Fused RMSNorm + per-row int8 quantization.

Args:
    x: fp16 activations [M, K]
    eps: RMSNorm epsilon

Returns:
    list[int8 x_q [M, K], fp32 x_scales [M]])");

  m.def(
      "fused_rmsnorm_adaln_quant",
      [](const mx::array& x,
         const mx::array& adaln_s,
         const mx::array& adaln_b,
         float eps) {
        return we_kernels::fused_rmsnorm_adaln_quant(x, adaln_s, adaln_b, eps);
      },
      nb::arg("x"),
      nb::arg("adaln_s"),
      nb::arg("adaln_b"),
      nb::arg("eps") = 1e-5f,
      R"(Fused RMSNorm + AdaLN modulation + per-row int8 quantization.

Args:
    x: fp16 activations [M, K]
    adaln_s: fp16 scale modulation [K]
    adaln_b: fp16 bias modulation [K]
    eps: RMSNorm epsilon

Returns:
    list[int8 x_q [M, K], fp32 x_scales [M]])");

  m.def(
      "fused_rmsnorm_smooth_quant",
      [](const mx::array& x,
         const mx::array& smooth_scale,
         float eps) {
        return we_kernels::fused_rmsnorm_smooth_quant(x, smooth_scale, eps);
      },
      nb::arg("x"),
      nb::arg("smooth_scale"),
      nb::arg("eps") = 1e-5f,
      R"(Fused RMSNorm + SmoothQuant + per-row int8 quantization.

Args:
    x: fp16 activations [M, K]
    smooth_scale: fp16 per-channel smooth scale [K]
    eps: RMSNorm epsilon

Returns:
    list[int8 x_q [M, K], fp32 x_scales [M]])");

  m.def(
      "fused_rmsnorm_adaln_smooth_quant",
      [](const mx::array& x,
         const mx::array& adaln_s,
         const mx::array& adaln_b,
         const mx::array& smooth_scale,
         float eps) {
        return we_kernels::fused_rmsnorm_adaln_smooth_quant(x, adaln_s, adaln_b, smooth_scale, eps);
      },
      nb::arg("x"),
      nb::arg("adaln_s"),
      nb::arg("adaln_b"),
      nb::arg("smooth_scale"),
      nb::arg("eps") = 1e-5f,
      R"(Fused RMSNorm + AdaLN + SmoothQuant + per-row int8 quantization.

Args:
    x: fp16 activations [M, K]
    adaln_s: fp16 scale modulation [K]
    adaln_b: fp16 bias modulation [K]
    smooth_scale: fp16 per-channel smooth scale [K]
    eps: RMSNorm epsilon

Returns:
    list[int8 x_q [M, K], fp32 x_scales [M]])");

  m.def(
      "fused_qkv_norm_rope",
      [](const mx::array& qkv,
         const mx::array& rope_cos,
         const mx::array& rope_sin,
         uint32_t n_q, uint32_t n_k, uint32_t n_v,
         float eps) {
        return we_kernels::fused_qkv_norm_rope(qkv, rope_cos, rope_sin, n_q, n_k, n_v, eps);
      },
      nb::arg("qkv"),
      nb::arg("rope_cos"),
      nb::arg("rope_sin"),
      nb::arg("n_q"),
      nb::arg("n_k"),
      nb::arg("n_v"),
      nb::arg("eps") = 1e-5f,
      R"(Fused QKV split + per-head RMSNorm + OrthoRoPE.

Args:
    qkv: fp16 [T, (N_Q+N_K+N_V)*D_HEAD] — flat QKV from GEMM
    rope_cos: fp16 [T, D_ROPE] — precomputed cos angles
    rope_sin: fp16 [T, D_ROPE] — precomputed sin angles
    n_q: number of Q heads
    n_k: number of K heads
    n_v: number of V heads
    eps: RMSNorm epsilon

Returns:
    list[fp16 q [N_Q, T, D_HEAD], fp16 k [N_K, T, D_HEAD], fp16 v [N_V, T, D_HEAD]])");

  m.def(
      "scatter_sdpa",
      [](const mx::array& Q,
         const mx::array& K,
         const mx::array& V,
         const mx::array& block_offsets,
         float scale) {
        return we_kernels::scatter_sdpa(Q, K, V, block_offsets, scale);
      },
      nb::arg("Q"), nb::arg("K"), nb::arg("V"),
      nb::arg("block_offsets"), nb::arg("scale"));

  m.def(
      "seq_sdpa",
      [](const mx::array& Q,
         const mx::array& K,
         const mx::array& V,
         int num_kv_tokens,
         float scale) {
        return we_kernels::seq_sdpa(Q, K, V,
            static_cast<uint32_t>(num_kv_tokens), scale);
      },
      nb::arg("Q"), nb::arg("K"), nb::arg("V"),
      nb::arg("num_kv_tokens"), nb::arg("scale"));

  m.def(
      "seq_sdpa_int8block",
      [](const mx::array& Q, const mx::array& K_q, const mx::array& K_scales,
         const mx::array& V_q, const mx::array& V_scales,
         int num_kv_tokens, float scale, int bk) {
        uint32_t N_Q = static_cast<uint32_t>(Q.shape(0));
        uint32_t N_KV = static_cast<uint32_t>(K_q.shape(0));
        uint32_t T = static_cast<uint32_t>(Q.shape(1));
        uint32_t D = static_cast<uint32_t>(Q.shape(2));
        auto stream = mx::to_stream({});
        return mx::array(
            {static_cast<int>(N_Q), static_cast<int>(T), static_cast<int>(D)},
            mx::float16,
            std::make_shared<we_kernels::SeqSDPAInt8Block>(
                stream, N_Q, N_KV, T, D,
                static_cast<uint32_t>(num_kv_tokens), scale,
                static_cast<uint32_t>(bk)),
            {mx::contiguous(Q, false, stream),
             mx::contiguous(K_q, false, stream),
             mx::contiguous(K_scales, false, stream),
             mx::contiguous(V_q, false, stream),
             mx::contiguous(V_scales, false, stream)});
      },
      nb::arg("Q"), nb::arg("K_q"), nb::arg("K_scales"),
      nb::arg("V_q"), nb::arg("V_scales"),
      nb::arg("num_kv_tokens"), nb::arg("scale"),
      nb::arg("bk") = 32,
      "SageAttention-style: per-block K/V int8 quant, int8 Q@K^T MMA.");

  m.def(
      "kv_cache_upsert",
      [](const mx::array& cache_k,
         const mx::array& cache_v,
         const mx::array& k_new,
         const mx::array& v_new,
         int rs) {
        return we_kernels::kv_cache_upsert(cache_k, cache_v, k_new, v_new,
            static_cast<uint32_t>(rs));
      },
      nb::arg("cache_k"), nb::arg("cache_v"),
      nb::arg("k_new"), nb::arg("v_new"), nb::arg("rs"),
      "In-place KV cache upsert (coalesced half4).");

  m.def(
      "fused_quant_upsert",
      [](const mx::array& k_new, const mx::array& v_new,
         const mx::array& cache_k_q, const mx::array& cache_k_s,
         const mx::array& cache_v_q, const mx::array& cache_v_s,
         int rs, int rs_BLK, int BK) {
        return we_kernels::fused_quant_upsert(
            k_new, v_new, cache_k_q, cache_k_s, cache_v_q, cache_v_s,
            static_cast<uint32_t>(rs), static_cast<uint32_t>(rs_BLK),
            static_cast<uint32_t>(BK));
      },
      nb::arg("k_new"), nb::arg("v_new"),
      nb::arg("cache_k_q"), nb::arg("cache_k_s"),
      nb::arg("cache_v_q"), nb::arg("cache_v_s"),
      nb::arg("rs"), nb::arg("rs_BLK"), nb::arg("BK"),
      "Fused fp16 K/V → per-block int8 quant → cache write (one dispatch).");

  m.def(
      "kv_cache_upsert_int8_block",
      [](const mx::array& cache_k_q, const mx::array& cache_k_scale,
         const mx::array& cache_v_q, const mx::array& cache_v_scale,
         const mx::array& k_new_q, const mx::array& k_new_scale,
         const mx::array& v_new_q, const mx::array& v_new_scale,
         int rs, int rs_BLK) {
        return we_kernels::kv_cache_upsert_int8_block(
            cache_k_q, cache_k_scale, cache_v_q, cache_v_scale,
            k_new_q, k_new_scale, v_new_q, v_new_scale,
            static_cast<uint32_t>(rs), static_cast<uint32_t>(rs_BLK));
      },
      nb::arg("cache_k_q"), nb::arg("cache_k_scale"),
      nb::arg("cache_v_q"), nb::arg("cache_v_scale"),
      nb::arg("k_new_q"), nb::arg("k_new_scale"),
      nb::arg("v_new_q"), nb::arg("v_new_scale"),
      nb::arg("rs"), nb::arg("rs_BLK"),
      "In-place int8 KV cache upsert with per-block scales (SageAttention).");

  m.def(
      "repro_half4_tg",
      [](const mx::array& x) { return we_kernels::repro_half4_tg(x); },
      nb::arg("x"),
      "Diagnostic A: pure x→TG→y copy via vector half4 TG writes.");

  m.def(
      "repro_half4_tg_reduce",
      [](const mx::array& x) { return we_kernels::repro_half4_tg_reduce(x); },
      nb::arg("x"),
      "Diagnostic B: + interleaved sum_sq reduction (RMSNorm Phase 1).");

  m.def(
      "repro_half4_tg_rmw",
      [](const mx::array& x) { return we_kernels::repro_half4_tg_rmw(x); },
      nb::arg("x"),
      "Diagnostic C: + Phase 2 RMW + Phase 3 copy. Closest to RMSNorm minus AdaLN.");

  m.def(
      "repro_half4_tg_adaln",
      [](const mx::array& x, const mx::array& adaln_s,
         const mx::array& adaln_b) {
        return we_kernels::repro_half4_tg_adaln(x, adaln_s, adaln_b);
      },
      nb::arg("x"), nb::arg("adaln_s"), nb::arg("adaln_b"),
      "Diagnostic D: full RMSNorm Phase 1+2+3 structure (with AdaLN device reads).");

#define BIND_REPRO_WORKAROUND(py_name, cpp_fn, doc)                       \
  m.def(                                                                   \
      py_name,                                                             \
      [](const mx::array& x, const mx::array& adaln_s,                     \
         const mx::array& adaln_b) {                                       \
        return we_kernels::cpp_fn(x, adaln_s, adaln_b);                    \
      },                                                                   \
      nb::arg("x"), nb::arg("adaln_s"), nb::arg("adaln_b"),                \
      doc);
  BIND_REPRO_WORKAROUND("repro_half4_tg_dualflag",    repro_half4_tg_dualflag,
      "Workaround E: same as D but uses mem_threadgroup|mem_device barrier.");
  BIND_REPRO_WORKAROUND("repro_half4_tg_regprefetch", repro_half4_tg_regprefetch,
      "Workaround F: same as D but adaln pre-loaded to per-thread registers.");
  BIND_REPRO_WORKAROUND("repro_half4_tg_tgprefetch",  repro_half4_tg_tgprefetch,
      "Workaround G: same as D but adaln pre-loaded to a TG scratch buffer.");
  BIND_REPRO_WORKAROUND("repro_half4_tg_volatile",    repro_half4_tg_volatile,
      "Workaround H: same as D but x_cache declared volatile threadgroup.");
#undef BIND_REPRO_WORKAROUND

}
