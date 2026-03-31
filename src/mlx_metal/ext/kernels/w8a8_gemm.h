#pragma once

#include "mlx/mlx.h"
#include "mlx/primitives.h"

namespace we_kernels {

namespace mx = mlx::core;

class W8A8Gemm : public mx::Primitive {
 public:
  W8A8Gemm(mx::Stream stream, uint32_t M, uint32_t N, uint32_t K, bool has_bias)
      : mx::Primitive(stream), M_(M), N_(N), K_(K), has_bias_(has_bias) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "W8A8Gemm"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {mx::Shape{static_cast<int>(M_), static_cast<int>(N_)}};
  }

 private:
  uint32_t M_, N_, K_;
  bool has_bias_;
};

mx::array w8a8_gemm(
    const mx::array& x_q,
    const mx::array& w_q,
    const mx::array& x_scales,
    const mx::array& w_scales,
    const mx::array& bias,
    mx::StreamOrDevice s = {});

}  // namespace we_kernels
