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

// Plain per-row symmetric int8 quantization (no RMSNorm)
class FusedQuant : public mx::Primitive {
 public:
  FusedQuant(mx::Stream stream, uint32_t M, uint32_t K)
      : mx::Primitive(stream), M_(M), K_(K) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "FusedQuant"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {mx::Shape{static_cast<int>(M_), static_cast<int>(K_)},
            mx::Shape{static_cast<int>(M_)}};
  }

 private:
  uint32_t M_, K_;
};

std::vector<mx::array> fused_quant(
    const mx::array& x,
    mx::StreamOrDevice s = {});

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
              uint32_t n_blocks, float scale)
      : mx::Primitive(stream), N_Q_(N_Q), N_KV_(N_KV), T_(T),
        capacity_(capacity), D_HEAD_(D_HEAD), n_blocks_(n_blocks),
        scale_(scale) {}

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
};

mx::array scatter_sdpa(
    const mx::array& Q,
    const mx::array& K,
    const mx::array& V,
    const mx::array& block_offsets,
    float scale,
    mx::StreamOrDevice s = {});

// Sequential-scan attention: K/V contiguous from offset 0, no block_offsets.
class SeqSDPA : public mx::Primitive {
 public:
  SeqSDPA(mx::Stream stream, uint32_t N_Q, uint32_t N_KV,
          uint32_t T, uint32_t D_HEAD, uint32_t num_kv_tokens,
          float scale)
      : mx::Primitive(stream), N_Q_(N_Q), N_KV_(N_KV), T_(T),
        D_HEAD_(D_HEAD), num_kv_tokens_(num_kv_tokens),
        scale_(scale) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "SeqSDPA"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {mx::Shape{static_cast<int>(N_Q_), static_cast<int>(T_),
                      static_cast<int>(D_HEAD_)}};
  }

 private:
  uint32_t N_Q_, N_KV_, T_, D_HEAD_, num_kv_tokens_;
  float scale_;
};

mx::array seq_sdpa(
    const mx::array& Q,
    const mx::array& K,
    const mx::array& V,
    uint32_t num_kv_tokens,
    float scale,
    mx::StreamOrDevice s = {});

// SageAttention-style: per-block K and V quantization, int8 Q@K MMA.
// K_scales and V_scales shape: [N_KV, N_BLK_CAP = CAP/BK]
class SeqSDPAInt8Block : public mx::Primitive {
 public:
  SeqSDPAInt8Block(mx::Stream stream, uint32_t N_Q, uint32_t N_KV,
                   uint32_t T, uint32_t D_HEAD, uint32_t num_kv_tokens,
                   float scale, uint32_t bk = 32)
      : mx::Primitive(stream), N_Q_(N_Q), N_KV_(N_KV), T_(T),
        D_HEAD_(D_HEAD), num_kv_tokens_(num_kv_tokens), scale_(scale), bk_(bk) {}
  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  const char* name() const override { return "SeqSDPAInt8Block"; }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {mx::Shape{static_cast<int>(N_Q_), static_cast<int>(T_),
                      static_cast<int>(D_HEAD_)}};
  }
 private:
  uint32_t N_Q_, N_KV_, T_, D_HEAD_, num_kv_tokens_, bk_;
  float scale_;
};

mx::array seq_sdpa_int8block(
    const mx::array& Q, const mx::array& K_q, const mx::array& K_scales,
    const mx::array& V_q, const mx::array& V_scales,
    uint32_t num_kv_tokens, float scale, mx::StreamOrDevice s = {});

// In-place KV cache ring buffer upsert
class KVCacheUpsert : public mx::Primitive {
 public:
  KVCacheUpsert(mx::Stream stream, uint32_t N_KV, uint32_t L, uint32_t T,
                uint32_t D, uint32_t rs)
      : mx::Primitive(stream), N_KV_(N_KV), L_(L), T_(T), D_(D), rs_(rs) {}

  void eval_cpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  void eval_gpu(
      const std::vector<mx::array>& inputs,
      std::vector<mx::array>& outputs) override;

  const char* name() const override { return "KVCacheUpsert"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape(), inputs[1].shape()};
  }

 private:
  uint32_t N_KV_, L_, T_, D_, rs_;
};

std::vector<mx::array> kv_cache_upsert(
    const mx::array& cache_k, const mx::array& cache_v,
    const mx::array& k_new, const mx::array& v_new,
    uint32_t rs, mx::StreamOrDevice s = {});

// Fused quantize + upsert: fp16 K/V → int8 per-block quant → cache write (one dispatch)
class FusedQuantUpsert : public mx::Primitive {
 public:
  FusedQuantUpsert(mx::Stream stream, uint32_t N_KV, uint32_t L, uint32_t T,
                   uint32_t D, uint32_t rs, uint32_t L_BLK, uint32_t rs_BLK, uint32_t BK)
      : mx::Primitive(stream), N_KV_(N_KV), L_(L), T_(T), D_(D), rs_(rs),
        L_BLK_(L_BLK), rs_BLK_(rs_BLK), BK_(BK) {}
  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  const char* name() const override { return "FusedQuantUpsert"; }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {inputs[2].shape(), inputs[3].shape(),
            inputs[4].shape(), inputs[5].shape()};
  }
 private:
  uint32_t N_KV_, L_, T_, D_, rs_, L_BLK_, rs_BLK_, BK_;
};

std::vector<mx::array> fused_quant_upsert(
    const mx::array& k_new_fp16, const mx::array& v_new_fp16,
    const mx::array& cache_k_q, const mx::array& cache_k_s,
    const mx::array& cache_v_q, const mx::array& cache_v_s,
    uint32_t rs, uint32_t rs_BLK, uint32_t BK,
    mx::StreamOrDevice s = {});

// Block-scale upsert (SageAttention-style) — scales have a different layout
// from data (L_BLK = L / BK).
class KVCacheUpsertInt8Block : public mx::Primitive {
 public:
  KVCacheUpsertInt8Block(mx::Stream stream, uint32_t N_KV, uint32_t L,
                         uint32_t T, uint32_t D, uint32_t rs,
                         uint32_t L_BLK, uint32_t T_BLK, uint32_t rs_BLK)
      : mx::Primitive(stream), N_KV_(N_KV), L_(L), T_(T), D_(D), rs_(rs),
        L_BLK_(L_BLK), T_BLK_(T_BLK), rs_BLK_(rs_BLK) {}
  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  const char* name() const override { return "KVCacheUpsertInt8Block"; }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape(), inputs[1].shape(),
            inputs[2].shape(), inputs[3].shape()};
  }
 private:
  uint32_t N_KV_, L_, T_, D_, rs_, L_BLK_, T_BLK_, rs_BLK_;
};

std::vector<mx::array> kv_cache_upsert_int8_block(
    const mx::array& cache_k_q, const mx::array& cache_k_scale,
    const mx::array& cache_v_q, const mx::array& cache_v_scale,
    const mx::array& k_new_q, const mx::array& k_new_scale,
    const mx::array& v_new_q, const mx::array& v_new_scale,
    uint32_t rs, uint32_t rs_BLK, mx::StreamOrDevice s = {});

// -----------------------------------------------------------------------
// Diagnostic: minimal repro for the half4-TG-store race we hit in
// the RMSNorm Phase 1 pattern. Pure copy x → TG cache → y, using the
// vectorized half4→TG write path through our .metallib + Primitive
// dispatch. If this reproduces the race, the bug is narrowed to the
// metallib/primitive path (same pattern via mx.fast.metal_kernel is
// clean). See APPLE_SILICON_VS_CUDA.md and tests/repro_half4_tg_race.py.
// -----------------------------------------------------------------------
class ReproHalf4TG : public mx::Primitive {
 public:
  ReproHalf4TG(mx::Stream stream, uint32_t M, uint32_t K)
      : mx::Primitive(stream), M_(M), K_(K) {}
  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  const char* name() const override { return "ReproHalf4TG"; }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape()};
  }
 private:
  uint32_t M_, K_;
};

mx::array repro_half4_tg(const mx::array& x, mx::StreamOrDevice s = {});

// Variant B: adds sum_sq reduction (simd_sum + sg_reduce writes)
// interleaved with the half4 TG writes, exactly mirroring RMSNorm.
class ReproHalf4TGReduce : public mx::Primitive {
 public:
  ReproHalf4TGReduce(mx::Stream stream, uint32_t M, uint32_t K)
      : mx::Primitive(stream), M_(M), K_(K) {}
  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  const char* name() const override { return "ReproHalf4TGReduce"; }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape(), mx::Shape{static_cast<int>(M_)}};
  }
 private:
  uint32_t M_, K_;
};

std::vector<mx::array> repro_half4_tg_reduce(
    const mx::array& x, mx::StreamOrDevice s = {});

// Variant C: + Phase 2 read-modify-write + Phase 3 copy-out.
// Closest structural match to RMSNorm minus the AdaLN device reads.
class ReproHalf4TGRMW : public mx::Primitive {
 public:
  ReproHalf4TGRMW(mx::Stream stream, uint32_t M, uint32_t K)
      : mx::Primitive(stream), M_(M), K_(K) {}
  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  const char* name() const override { return "ReproHalf4TGRMW"; }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape(), mx::Shape{static_cast<int>(M_)}};
  }
 private:
  uint32_t M_, K_;
};

std::vector<mx::array> repro_half4_tg_rmw(
    const mx::array& x, mx::StreamOrDevice s = {});

// Variant D: + AdaLN device reads. Full RMSNorm match.
class ReproHalf4TGAdaLN : public mx::Primitive {
 public:
  ReproHalf4TGAdaLN(mx::Stream stream, uint32_t M, uint32_t K)
      : mx::Primitive(stream), M_(M), K_(K) {}
  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  const char* name() const override { return "ReproHalf4TGAdaLN"; }
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape(), mx::Shape{static_cast<int>(M_)}};
  }
 private:
  uint32_t M_, K_;
};

std::vector<mx::array> repro_half4_tg_adaln(
    const mx::array& x, const mx::array& adaln_s, const mx::array& adaln_b,
    mx::StreamOrDevice s = {});

// Workaround candidates — same signature as variant D, only differ in
// which kernel name they dispatch. Shared base class cuts boilerplate.
class ReproHalf4TGAdaLNBase : public mx::Primitive {
 public:
  ReproHalf4TGAdaLNBase(mx::Stream stream, uint32_t M, uint32_t K,
                        const char* kernel_name)
      : mx::Primitive(stream), M_(M), K_(K), kernel_name_(kernel_name) {}
  void eval_cpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  void eval_gpu(const std::vector<mx::array>& inputs,
                std::vector<mx::array>& outputs) override;
  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape(), mx::Shape{static_cast<int>(M_)}};
  }
 protected:
  uint32_t M_, K_;
  const char* kernel_name_;
};

// Concrete subclasses so MLX graph dedup doesn't collapse variants.
#define DECL_REPRO_VARIANT(ClassName, NameTag)                            \
  class ClassName : public ReproHalf4TGAdaLNBase {                        \
   public:                                                                 \
    ClassName(mx::Stream stream, uint32_t M, uint32_t K)                   \
        : ReproHalf4TGAdaLNBase(stream, M, K, NameTag) {}                  \
    const char* name() const override { return NameTag; }                  \
  };
DECL_REPRO_VARIANT(ReproHalf4TGDualFlag,    "repro_half4_tg_dualflag")
DECL_REPRO_VARIANT(ReproHalf4TGRegPrefetch, "repro_half4_tg_regprefetch")
DECL_REPRO_VARIANT(ReproHalf4TGTGPrefetch,  "repro_half4_tg_tgprefetch")
DECL_REPRO_VARIANT(ReproHalf4TGVolatile,    "repro_half4_tg_volatile")
#undef DECL_REPRO_VARIANT

std::vector<mx::array> repro_half4_tg_dualflag(
    const mx::array& x, const mx::array& adaln_s, const mx::array& adaln_b,
    mx::StreamOrDevice s = {});
std::vector<mx::array> repro_half4_tg_regprefetch(
    const mx::array& x, const mx::array& adaln_s, const mx::array& adaln_b,
    mx::StreamOrDevice s = {});
std::vector<mx::array> repro_half4_tg_tgprefetch(
    const mx::array& x, const mx::array& adaln_s, const mx::array& adaln_b,
    mx::StreamOrDevice s = {});
std::vector<mx::array> repro_half4_tg_volatile(
    const mx::array& x, const mx::array& adaln_s, const mx::array& adaln_b,
    mx::StreamOrDevice s = {});

}  // namespace we_kernels
