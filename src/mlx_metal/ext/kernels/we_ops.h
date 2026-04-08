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

// Fused RMSNorm (+ optional AdaLN) (+ optional SmoothQuant) + per-row int8 quantization
class FusedRMSNormQuant : public mx::Primitive {
 public:
  FusedRMSNormQuant(mx::Stream stream, uint32_t M, uint32_t K, float eps, bool has_adaln, bool has_smooth = false)
      : mx::Primitive(stream), M_(M), K_(K), eps_(eps), has_adaln_(has_adaln), has_smooth_(has_smooth) {}

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
  bool has_smooth_;
};

std::vector<mx::array> fused_rmsnorm_quant(
    const mx::array& x,
    float eps = 1e-5f,
    mx::StreamOrDevice s = {});

std::vector<mx::array> fused_rmsnorm_smooth_quant(
    const mx::array& x,
    const mx::array& smooth_scale,
    float eps = 1e-5f,
    mx::StreamOrDevice s = {});

std::vector<mx::array> fused_rmsnorm_adaln_quant(
    const mx::array& x,
    const mx::array& adaln_s,
    const mx::array& adaln_b,
    float eps = 1e-5f,
    mx::StreamOrDevice s = {});

std::vector<mx::array> fused_rmsnorm_adaln_smooth_quant(
    const mx::array& x,
    const mx::array& adaln_s,
    const mx::array& adaln_b,
    const mx::array& smooth_scale,
    float eps = 1e-5f,
    mx::StreamOrDevice s = {});

// Fused QKV split + per-head RMSNorm + OrthoRoPE
class FusedQKVNormRoPE : public mx::Primitive {
 public:
  FusedQKVNormRoPE(mx::Stream stream, uint32_t T, uint32_t N_Q, uint32_t N_K,
                    uint32_t N_V, uint32_t D_HEAD, uint32_t D_ROPE, float eps)
      : mx::Primitive(stream), T_(T), N_Q_(N_Q), N_K_(N_K), N_V_(N_V),
        D_HEAD_(D_HEAD), D_ROPE_(D_ROPE), eps_(eps) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "FusedQKVNormRoPE"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    int T = static_cast<int>(T_);
    int D = static_cast<int>(D_HEAD_);
    return {mx::Shape{static_cast<int>(N_Q_), T, D},
            mx::Shape{static_cast<int>(N_K_), T, D},
            mx::Shape{static_cast<int>(N_V_), T, D}};
  }

 private:
  uint32_t T_, N_Q_, N_K_, N_V_, D_HEAD_, D_ROPE_;
  float eps_;
};

std::vector<mx::array> fused_qkv_norm_rope(
    const mx::array& qkv,
    const mx::array& rope_cos,
    const mx::array& rope_sin,
    uint32_t N_Q, uint32_t N_K, uint32_t N_V,
    float eps = 1e-5f,
    mx::StreamOrDevice s = {});

// Scatter-read flash attention with GQA support
class ScatterSDPA : public mx::Primitive {
 public:
  ScatterSDPA(mx::Stream stream, uint32_t N_Q, uint32_t N_KV,
              uint32_t T, uint32_t capacity, uint32_t D_HEAD,
              uint32_t n_blocks, float scale,
              std::string kernel_name = "scatter_sdpa",
              uint32_t bq = 32, uint32_t tg_size = 32)
      : mx::Primitive(stream), N_Q_(N_Q), N_KV_(N_KV), T_(T),
        capacity_(capacity), D_HEAD_(D_HEAD), n_blocks_(n_blocks),
        scale_(scale), kernel_name_(std::move(kernel_name)),
        bq_(bq), tg_size_(tg_size) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "ScatterSDPA"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {mx::Shape{static_cast<int>(N_Q_), static_cast<int>(T_),
                      static_cast<int>(D_HEAD_)}};
  }

 private:
  uint32_t N_Q_, N_KV_, T_, capacity_, D_HEAD_, n_blocks_;
  float scale_;
  std::string kernel_name_;
  uint32_t bq_, tg_size_;
};

mx::array scatter_sdpa(
    const mx::array& Q,
    const mx::array& K,
    const mx::array& V,
    const mx::array& block_offsets,
    float scale,
    const std::string& variant = "",
    mx::StreamOrDevice s = {});

}  // namespace we_kernels
