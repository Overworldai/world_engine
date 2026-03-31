#include <nanobind/nanobind.h>

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

Computes: out = (x_q @ w_q.T) * diag(x_scales) * diag(w_scales) + bias

Args:
    x_q: int8 quantised activations [M, K]
    w_q: int8 quantised weights [N, K]
    x_scales: fp32 per-row activation scales [M]
    w_scales: fp32 per-row weight scales [N]
    bias: fp32 bias [N] (pass zeros for no bias)

Returns:
    fp16 output [M, N])");
}
