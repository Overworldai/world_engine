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
}
