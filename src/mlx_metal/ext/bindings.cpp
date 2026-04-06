#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "mlx/mlx.h"
#include "kernels/w8a8_gemm.h"

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
      "ring_flash_attention",
      [](const mx::array& Q,
         const mx::array& K,
         const mx::array& V,
         const mx::array& written,
         float scale) {
        return we_kernels::ring_flash_attention(Q, K, V, written, scale);
      },
      nb::arg("Q"),
      nb::arg("K"),
      nb::arg("V"),
      nb::arg("written"),
      nb::arg("scale"),
      R"(Ring-buffer flash attention.

Args:
    Q: fp16 [N_H, T, D_HEAD] — query
    K: fp16 [N_H, capacity, D_HEAD] — key cache
    V: fp16 [N_H, capacity, D_HEAD] — value cache
    written: fp16 [capacity] — 1.0 for valid, 0.0 for empty
    scale: float — 1/sqrt(D_HEAD)

Returns:
    fp16 [N_H, T, D_HEAD])");
}
