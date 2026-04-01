#include "kernels/w8a8_gemm.h"

#include <dlfcn.h>
#include <filesystem>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace we_kernels {

struct TileConfig {
  int bm, bn, bk, wm, wn;
  const char* kernel_name;
};

// v1: both A and B staged through threadgroup
// Wins for wide N (N >= 4096) where B staging amortizes well.
static constexpr TileConfig V1_TINY_M   = { 32,  64, 128, 1, 2, "w8a8_gemm_nax_bm32_bn64_bk128_wm1_wn2"};
static constexpr TileConfig V1_SMALL    = { 64,  64,  64, 2, 2, "w8a8_gemm_nax_bm64_bn64_bk64_wm2_wn2"};
static constexpr TileConfig V1_DEEP_K   = { 64,  64, 128, 2, 2, "w8a8_gemm_nax_bm64_bn64_bk128_wm2_wn2"};
static constexpr TileConfig V1_DEEP_K2  = { 64,  64, 192, 2, 2, "w8a8_gemm_nax_bm64_bn64_bk192_wm2_wn2"};
static constexpr TileConfig V1_WIDE_N   = { 64, 128, 128, 2, 4, "w8a8_gemm_nax_bm64_bn128_bk128_wm2_wn4"};
static constexpr TileConfig V1_LARGE    = {128, 128,  64, 4, 4, "w8a8_gemm_nax_bm128_bn128_bk64_wm4_wn4"};

// v2: A direct from device, B staged through threadgroup
// Wins for square/narrow N shapes where A staging overhead exceeds benefit.
// Halves threadgroup usage, allows larger BK (e.g. BK=256).
static constexpr TileConfig V2_SMALL    = { 64,  64,  64, 2, 2, "w8a8_gemm_v2_bm64_bn64_bk64_wm2_wn2"};
static constexpr TileConfig V2_DEEP_K   = { 64,  64, 256, 2, 2, "w8a8_gemm_v2_bm64_bn64_bk256_wm2_wn2"};
static constexpr TileConfig V2_WIDE_N   = { 64, 128, 128, 2, 4, "w8a8_gemm_v2_bm64_bn128_bk128_wm2_wn4"};
static constexpr TileConfig V2_LARGE    = {128, 128,  64, 4, 4, "w8a8_gemm_v2_bm128_bn128_bk64_wm4_wn4"};

static const TileConfig& select_tile(uint32_t M, uint32_t N, uint32_t K) {
  // Thresholds from bench_gemm.py on M5 Max (see w8a8_gemm.metal header).
  // N >= 4096: v1 both-staged wins (mlp.fc1 N=8192, qkv N=6144)
  // N < 4096:  v2 A-direct wins (attn.out N=2048, mlp.fc2 K=8192)
  if (M >= 128 && N >= 128) {
    return (N >= 4096) ? V1_LARGE : V2_LARGE;
  }
  if (M >= 64 && N >= 128) {
    return (N >= 4096) ? V1_WIDE_N : V2_WIDE_N;
  }
  // K >= 192: deep-K v1 tile fits BK=128 in threadgroup budget, better K-reuse
  if (K >= 192) return V1_DEEP_K;
  return V1_SMALL;
}

struct W8A8Params {
  uint32_t M;
  uint32_t N;
  uint32_t K;
};

static const std::string& lib_path() {
  static std::string path = []() {
    Dl_info info;
    dladdr(reinterpret_cast<void*>(&lib_path), &info);
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return path;
}

void W8A8Gemm::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("W8A8Gemm: CPU not supported");
}

void W8A8Gemm::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& x_q = inputs[0];
  auto& w_q = inputs[1];
  auto& x_scales = inputs[2];
  auto& w_scales = inputs[3];
  auto& bias = inputs[4];
  auto& out = outputs[0];

  out.set_data(mx::allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  auto mtl_lib = d.get_library("we_kernels", lib_path());

  auto& enc = d.get_command_encoder(s.index);

  enc.set_input_array(x_q, 0);
  enc.set_input_array(w_q, 1);
  enc.set_input_array(x_scales, 2);
  enc.set_input_array(w_scales, 3);
  enc.set_input_array(bias, 4);
  enc.set_output_array(out, 5);

  W8A8Params params{M_, N_, K_};
  enc.set_bytes(params, 6);

  // Matvec for small M (1-4 tokens, decode path). K must be divisible by
  // BLOCK_SIZE=1024 for aligned 128-bit loads. Benchmarked crossover at M=5.
  constexpr uint32_t MATVEC_LIMIT = 5;
  if (M_ < MATVEC_LIMIT && K_ % 1024 == 0) {
    auto kernel = d.get_kernel("w8a8_matvec", mtl_lib);
    enc.set_compute_pipeline_state(kernel);
    constexpr int BN_MV = 8; // from matvec kernel
    MTL::Size grid_dims(M_, (N_ + BN_MV - 1) / BN_MV, 1);
    MTL::Size group_dims(64, 1, 1); // 2 simdgroups × 32
    enc.dispatch_threadgroups(grid_dims, group_dims);
  } else {
    const auto& tile = select_tile(M_, N_, K_);
    auto kernel = d.get_kernel(tile.kernel_name, mtl_lib);
    enc.set_compute_pipeline_state(kernel);
    uint32_t tiles_n = (N_ + tile.bn - 1) / tile.bn;
    uint32_t tiles_m = (M_ + tile.bm - 1) / tile.bm;
    MTL::Size grid_dims(tiles_n, tiles_m, 1);
    MTL::Size group_dims(32, tile.wn, tile.wm);
    enc.dispatch_threadgroups(grid_dims, group_dims);
  }
}

mx::array w8a8_gemm(
    const mx::array& x_q,
    const mx::array& w_q,
    const mx::array& x_scales,
    const mx::array& w_scales,
    const mx::array& bias,
    mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x_q.shape(0));
  uint32_t K = static_cast<uint32_t>(x_q.shape(1));
  uint32_t N = static_cast<uint32_t>(w_q.shape(0));
  bool has_bias = bias.size() == static_cast<size_t>(N);

  auto stream = mx::to_stream(s);
  return mx::array(
      {static_cast<int>(M), static_cast<int>(N)},
      mx::float16,
      std::make_shared<W8A8Gemm>(stream, M, N, K, has_bias),
      {mx::contiguous(x_q, false, stream),
       mx::contiguous(w_q, false, stream),
       mx::contiguous(x_scales, false, stream),
       mx::contiguous(w_scales, false, stream),
       mx::contiguous(bias, false, stream)});
}

// ===========================================================================
// Fused SiLU + int8 quantization (ZeroQuant pre-GEMM fusion)
// ===========================================================================

void FusedSiLUQuant::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("FusedSiLUQuant: CPU not supported");
}

void FusedSiLUQuant::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& x = inputs[0];
  auto& x_q = outputs[0];
  auto& x_scales = outputs[1];

  x_q.set_data(mx::allocator::malloc(x_q.nbytes()));
  x_scales.set_data(mx::allocator::malloc(x_scales.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel("fused_silu_quant", mtl_lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(x, 0);
  enc.set_output_array(x_q, 1);
  enc.set_output_array(x_scales, 2);

  struct Params { uint32_t M; uint32_t K; };
  Params params{M_, K_};
  enc.set_bytes(params, 3);

  MTL::Size grid(M_, 1, 1);
  MTL::Size group(256, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

std::vector<mx::array> fused_silu_quant(
    const mx::array& x,
    mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {mx::Shape{static_cast<int>(M), static_cast<int>(K)},
       mx::Shape{static_cast<int>(M)}},
      {mx::int8, mx::float32},
      std::make_shared<FusedSiLUQuant>(stream, M, K),
      {mx::contiguous(x, false, stream)});
}

// ===========================================================================
// Fused RMSNorm (+ optional AdaLN) + int8 quantization
// ===========================================================================

void FusedRMSNormQuant::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("FusedRMSNormQuant: CPU not supported");
}

void FusedRMSNormQuant::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& x = inputs[0];
  auto& x_q = outputs[0];
  auto& x_scales = outputs[1];

  x_q.set_data(mx::allocator::malloc(x_q.nbytes()));
  x_scales.set_data(mx::allocator::malloc(x_scales.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());

  auto& enc = d.get_command_encoder(s.index);

  struct Params { uint32_t M; uint32_t K; float eps; };
  Params params{M_, K_, eps_};

  if (has_adaln_) {
    auto& adaln_s = inputs[1];
    auto& adaln_b = inputs[2];

    auto kernel = d.get_kernel("fused_rmsnorm_adaln_quant", mtl_lib);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array(x, 0);
    enc.set_output_array(x_q, 1);
    enc.set_output_array(x_scales, 2);
    enc.set_input_array(adaln_s, 3);
    enc.set_input_array(adaln_b, 4);
    enc.set_bytes(params, 5);
  } else {
    auto kernel = d.get_kernel("fused_rmsnorm_quant", mtl_lib);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array(x, 0);
    enc.set_output_array(x_q, 1);
    enc.set_output_array(x_scales, 2);
    enc.set_bytes(params, 3);
  }

  MTL::Size grid(M_, 1, 1);
  MTL::Size group(256, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

std::vector<mx::array> fused_rmsnorm_quant(
    const mx::array& x,
    float eps,
    mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {mx::Shape{static_cast<int>(M), static_cast<int>(K)},
       mx::Shape{static_cast<int>(M)}},
      {mx::int8, mx::float32},
      std::make_shared<FusedRMSNormQuant>(stream, M, K, eps, false),
      {mx::contiguous(x, false, stream)});
}

std::vector<mx::array> fused_rmsnorm_adaln_quant(
    const mx::array& x,
    const mx::array& adaln_s,
    const mx::array& adaln_b,
    float eps,
    mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {mx::Shape{static_cast<int>(M), static_cast<int>(K)},
       mx::Shape{static_cast<int>(M)}},
      {mx::int8, mx::float32},
      std::make_shared<FusedRMSNormQuant>(stream, M, K, eps, true),
      {mx::contiguous(x, false, stream),
       mx::contiguous(adaln_s, false, stream),
       mx::contiguous(adaln_b, false, stream)});
}

}  // namespace we_kernels
