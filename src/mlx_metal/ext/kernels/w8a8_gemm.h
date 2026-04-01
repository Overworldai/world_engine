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

// Fused SiLU + per-row int8 quantization (ZeroQuant pre-GEMM fusion)
class FusedSiLUQuant : public mx::Primitive {
 public:
  FusedSiLUQuant(mx::Stream stream, uint32_t M, uint32_t K)
      : mx::Primitive(stream), M_(M), K_(K) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "FusedSiLUQuant"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {mx::Shape{static_cast<int>(M_), static_cast<int>(K_)},
            mx::Shape{static_cast<int>(M_)}};
  }

 private:
  uint32_t M_, K_;
};

std::vector<mx::array> fused_silu_quant(
    const mx::array& x,
    mx::StreamOrDevice s = {});

// Fused RMSNorm (+ optional AdaLN) + per-row int8 quantization
class FusedRMSNormQuant : public mx::Primitive {
 public:
  FusedRMSNormQuant(mx::Stream stream, uint32_t M, uint32_t K, float eps, bool has_adaln)
      : mx::Primitive(stream), M_(M), K_(K), eps_(eps), has_adaln_(has_adaln) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "FusedRMSNormQuant"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {mx::Shape{static_cast<int>(M_), static_cast<int>(K_)},
            mx::Shape{static_cast<int>(M_)}};
  }

 private:
  uint32_t M_, K_;
  float eps_;
  bool has_adaln_;
};

std::vector<mx::array> fused_rmsnorm_quant(
    const mx::array& x,
    float eps = 1e-5f,
    mx::StreamOrDevice s = {});

std::vector<mx::array> fused_rmsnorm_adaln_quant(
    const mx::array& x,
    const mx::array& adaln_s,
    const mx::array& adaln_b,
    float eps = 1e-5f,
    mx::StreamOrDevice s = {});

}  // namespace we_kernels
