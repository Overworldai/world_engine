#include "kernels/we_ops.h"

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
  // bench_steady validated (M5 Max, saturated KV cache, all optimizations):
  //
  // V1_SMALL (64×64×64) wins for ALL shapes. Maximum TG count gives best
  // occupancy and load balancing across 40 GPU cores in lazy eval.
  //
  // Tested alternatives (all slower in end-to-end):
  //   V2_DEEP_K for fc2 (K=8192): +15ms despite fewer K-loop iters
  //   V1_DEEP_K for QKV/fc1 (N>=4096): +10ms despite halved K-loop
  //   V1_LARGE, V1_WIDE_N: +10-20ms
  //
  // CAUTION: isolated GEMM benchmarks with mx.eval() are misleading.
  // V2_DEEP_K appears ~20% faster in isolation but ~10% slower end-to-end
  // because lazy eval batches kernels differently than per-call sync.
  // Always validate with bench_steady, never isolated mx.eval() timings.
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

  auto& enc = mx::metal::get_command_encoder(s);

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
    constexpr int BN_MV = 32; // from matvec kernel: 4 SGs × 8 results each
    MTL::Size grid_dims(M_, (N_ + BN_MV - 1) / BN_MV, 1);
    MTL::Size group_dims(128, 1, 1); // 4 simdgroups × 32
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
      {x_q, w_q, x_scales, w_scales, bias});
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

  auto& enc = mx::metal::get_command_encoder(s);
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
// Plain per-row int8 quantization (no RMSNorm)
// ===========================================================================

void FusedQuant::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("FusedQuant: CPU not supported");
}

void FusedQuant::eval_gpu(
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
  auto kernel = d.get_kernel("fused_quant", mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(x, 0);
  enc.set_output_array(x_q, 1);
  enc.set_output_array(x_scales, 2);

  struct Params { uint32_t M; uint32_t K; float eps; };
  Params params{M_, K_, 0.0f};  // eps unused but matches struct layout
  enc.set_bytes(params, 3);

  MTL::Size grid(M_, 1, 1);
  MTL::Size group(256, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

std::vector<mx::array> fused_quant(
    const mx::array& x,
    mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {mx::Shape{static_cast<int>(M), static_cast<int>(K)},
       mx::Shape{static_cast<int>(M)}},
      {mx::int8, mx::float32},
      std::make_shared<FusedQuant>(stream, M, K),
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

  if (K_ > 2048) {
    throw std::runtime_error(
        "FusedRMSNormQuant: K=" + std::to_string(K_) +
        " exceeds MAX_K=2048. Increase MAX_K in w8a8_fused_rmsnorm_quant.metal.");
  }

  x_q.set_data(mx::allocator::malloc(x_q.nbytes()));
  x_scales.set_data(mx::allocator::malloc(x_scales.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());

  auto& enc = mx::metal::get_command_encoder(s);

  struct Params { uint32_t M; uint32_t K; float eps; };
  Params params{M_, K_, eps_};

  // 4 variants: {adaln, no_adaln} x {smooth, no_smooth}
  // Input layout:
  //   no_adaln, no_smooth:  [x]
  //   no_adaln, smooth:     [x, smooth_scale]
  //   adaln, no_smooth:     [x, adaln_s, adaln_b]
  //   adaln, smooth:        [x, adaln_s, adaln_b, smooth_scale]

  if (has_adaln_ && has_smooth_) {
    auto& adaln_s = inputs[1];
    auto& adaln_b = inputs[2];
    auto& smooth = inputs[3];

    auto kernel = d.get_kernel("fused_rmsnorm_adaln_smooth_quant", mtl_lib);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array(x, 0);
    enc.set_output_array(x_q, 1);
    enc.set_output_array(x_scales, 2);
    enc.set_input_array(adaln_s, 3);
    enc.set_input_array(adaln_b, 4);
    enc.set_input_array(smooth, 5);
    enc.set_bytes(params, 6);
  } else if (has_adaln_) {
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
  } else if (has_smooth_) {
    auto& smooth = inputs[1];

    auto kernel = d.get_kernel("fused_rmsnorm_smooth_quant", mtl_lib);
    enc.set_compute_pipeline_state(kernel);

    enc.set_input_array(x, 0);
    enc.set_output_array(x_q, 1);
    enc.set_output_array(x_scales, 2);
    enc.set_input_array(smooth, 3);
    enc.set_bytes(params, 4);
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
      std::make_shared<FusedRMSNormQuant>(stream, M, K, eps, true, false),
      {mx::contiguous(x, false, stream),
       mx::contiguous(adaln_s, false, stream),
       mx::contiguous(adaln_b, false, stream)});
}

std::vector<mx::array> fused_rmsnorm_smooth_quant(
    const mx::array& x,
    const mx::array& smooth_scale,
    float eps,
    mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {mx::Shape{static_cast<int>(M), static_cast<int>(K)},
       mx::Shape{static_cast<int>(M)}},
      {mx::int8, mx::float32},
      std::make_shared<FusedRMSNormQuant>(stream, M, K, eps, false, true),
      {mx::contiguous(x, false, stream),
       mx::contiguous(smooth_scale, false, stream)});
}

std::vector<mx::array> fused_rmsnorm_adaln_smooth_quant(
    const mx::array& x,
    const mx::array& adaln_s,
    const mx::array& adaln_b,
    const mx::array& smooth_scale,
    float eps,
    mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {mx::Shape{static_cast<int>(M), static_cast<int>(K)},
       mx::Shape{static_cast<int>(M)}},
      {mx::int8, mx::float32},
      std::make_shared<FusedRMSNormQuant>(stream, M, K, eps, true, true),
      {mx::contiguous(x, false, stream),
       mx::contiguous(adaln_s, false, stream),
       mx::contiguous(adaln_b, false, stream),
       mx::contiguous(smooth_scale, false, stream)});
}

// ===========================================================================
// Fused QKV split + per-head RMSNorm + OrthoRoPE
// ===========================================================================

void FusedQKVNormRoPE::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("FusedQKVNormRoPE: CPU not supported");
}

void FusedQKVNormRoPE::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& qkv = inputs[0];
  auto& rope_cos = inputs[1];
  auto& rope_sin = inputs[2];
  auto& q_out = outputs[0];
  auto& k_out = outputs[1];
  auto& v_out = outputs[2];

  q_out.set_data(mx::allocator::malloc(q_out.nbytes()));
  k_out.set_data(mx::allocator::malloc(k_out.nbytes()));
  v_out.set_data(mx::allocator::malloc(v_out.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel("fused_qkv_norm_rope", mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(qkv, 0);
  enc.set_output_array(q_out, 1);
  enc.set_output_array(k_out, 2);
  enc.set_output_array(v_out, 3);
  enc.set_input_array(rope_cos, 4);
  enc.set_input_array(rope_sin, 5);

  struct Params {
    uint32_t T, N_Q, N_K, N_V, D_HEAD, D_ROPE;
    float eps;
  };
  Params params{T_, N_Q_, N_K_, N_V_, D_HEAD_, D_ROPE_, eps_};
  enc.set_bytes(params, 6);

  uint32_t N_TOTAL = N_Q_ + N_K_ + N_V_;
  constexpr uint32_t HEADS_PER_TG = 8;
  uint32_t head_groups = (N_TOTAL + HEADS_PER_TG - 1) / HEADS_PER_TG;
  MTL::Size grid(T_, head_groups, 1);
  MTL::Size group(HEADS_PER_TG * 32, 1, 1);  // 8 simdgroups, each handles one head
  enc.dispatch_threadgroups(grid, group);
}

std::vector<mx::array> fused_qkv_norm_rope(
    const mx::array& qkv,
    const mx::array& rope_cos,
    const mx::array& rope_sin,
    uint32_t N_Q, uint32_t N_K, uint32_t N_V,
    float eps,
    mx::StreamOrDevice s) {
  uint32_t T = static_cast<uint32_t>(qkv.shape(0));
  uint32_t QKV_DIM = static_cast<uint32_t>(qkv.shape(1));
  uint32_t N_TOTAL = N_Q + N_K + N_V;
  uint32_t D_HEAD = QKV_DIM / N_TOTAL;
  uint32_t D_ROPE = D_HEAD / 2;

  auto stream = mx::to_stream(s);
  int iT = static_cast<int>(T);
  int iD = static_cast<int>(D_HEAD);
  return mx::array::make_arrays(
      {mx::Shape{static_cast<int>(N_Q), iT, iD},
       mx::Shape{static_cast<int>(N_K), iT, iD},
       mx::Shape{static_cast<int>(N_V), iT, iD}},
      {mx::float16, mx::float16, mx::float16},
      std::make_shared<FusedQKVNormRoPE>(stream, T, N_Q, N_K, N_V, D_HEAD, D_ROPE, eps),
      {mx::contiguous(qkv, false, stream),
       mx::contiguous(rope_cos, false, stream),
       mx::contiguous(rope_sin, false, stream)});
}

// ===========================================================================
// Scatter-read flash attention
// ===========================================================================

void ScatterSDPA::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("ScatterSDPA: CPU not supported");
}

void ScatterSDPA::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& Q = inputs[0];
  auto& K = inputs[1];
  auto& V = inputs[2];
  auto& block_offsets = inputs[3];
  auto& O = outputs[0];

  O.set_data(mx::allocator::malloc(O.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());

  // BQ=32, BK=32, WM=2: best for T=512 N_HEADS=32 (bench_tune validated)
  auto kernel = d.get_kernel("scatter_sdpa_bq32_bk32_wm2", mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(Q, 0);
  enc.set_input_array(K, 1);
  enc.set_input_array(V, 2);
  enc.set_input_array(block_offsets, 3);
  enc.set_output_array(O, 4);

  struct Params {
    uint32_t N_Q_HEADS, N_KV_HEADS, T, CAPACITY, D_HEAD, N_BLOCKS;
    float scale;
  };
  Params params{N_Q_, N_KV_, T_, capacity_, D_HEAD_, n_blocks_, scale_};
  enc.set_bytes(params, 5);

  constexpr uint32_t BQ = 32;
  uint32_t q_blocks = (T_ + BQ - 1) / BQ;
  MTL::Size grid(q_blocks, N_Q_, 1);
  MTL::Size group(64, 1, 1);  // 2 simdgroups
  enc.dispatch_threadgroups(grid, group);
}

mx::array scatter_sdpa(
    const mx::array& Q,
    const mx::array& K,
    const mx::array& V,
    const mx::array& block_offsets,
    float scale,
    mx::StreamOrDevice s) {
  uint32_t N_Q = static_cast<uint32_t>(Q.shape(0));
  uint32_t N_KV = static_cast<uint32_t>(K.shape(0));
  uint32_t T = static_cast<uint32_t>(Q.shape(1));
  uint32_t D = static_cast<uint32_t>(Q.shape(2));
  uint32_t capacity = static_cast<uint32_t>(K.shape(1));
  uint32_t n_blocks = static_cast<uint32_t>(block_offsets.shape(0));

  // BQ=32 BK=32 WM=2: best for T=512 N_HEADS=32 (512 TGs, 72% NAX eff)
  auto stream = mx::to_stream(s);
  return mx::array(
      {static_cast<int>(N_Q), static_cast<int>(T), static_cast<int>(D)},
      mx::float16,
      std::make_shared<ScatterSDPA>(
          stream, N_Q, N_KV, T, capacity, D, n_blocks, scale),
      {mx::contiguous(Q, false, stream),
       mx::contiguous(K, false, stream),
       mx::contiguous(V, false, stream),
       mx::contiguous(block_offsets, false, stream)});
}

// ===========================================================================
// Sequential-scan attention (contiguous K/V, no block_offsets)
// ===========================================================================

void SeqSDPA::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("SeqSDPA: CPU not supported");
}

void SeqSDPA::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& Q = inputs[0];
  auto& K = inputs[1];
  auto& V = inputs[2];
  auto& O = outputs[0];

  O.set_data(mx::allocator::malloc(O.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel("scatter_sdpa_seq_direct_bq32_bk32_wm2", mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(Q, 0);
  enc.set_input_array(K, 1);
  enc.set_input_array(V, 2);
  enc.set_output_array(O, 3);

  uint32_t capacity = static_cast<uint32_t>(K.shape(1));
  struct Params {
    uint32_t N_Q_HEADS, N_KV_HEADS, T_Q, D_HEAD, CAPACITY, NUM_KV_TOKENS;
    float scale;
  };
  Params params{N_Q_, N_KV_, T_, D_HEAD_, capacity, num_kv_tokens_, scale_};
  enc.set_bytes(params, 4);

  // BQ=32, WM=2 → 64 threads, 4KB Q staging
  constexpr uint32_t BQ = 32;
  uint32_t q_blocks = (T_ + BQ - 1) / BQ;
  MTL::Size grid(q_blocks, N_Q_, 1);
  MTL::Size group(64, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

mx::array seq_sdpa(
    const mx::array& Q,
    const mx::array& K,
    const mx::array& V,
    uint32_t num_kv_tokens,
    float scale,
    mx::StreamOrDevice s) {
  uint32_t N_Q = static_cast<uint32_t>(Q.shape(0));
  uint32_t N_KV = static_cast<uint32_t>(K.shape(0));
  uint32_t T = static_cast<uint32_t>(Q.shape(1));
  uint32_t D = static_cast<uint32_t>(Q.shape(2));

  auto stream = mx::to_stream(s);
  return mx::array(
      {static_cast<int>(N_Q), static_cast<int>(T), static_cast<int>(D)},
      mx::float16,
      std::make_shared<SeqSDPA>(
          stream, N_Q, N_KV, T, D, num_kv_tokens, scale),
      {mx::contiguous(Q, false, stream),
       mx::contiguous(K, false, stream),
       mx::contiguous(V, false, stream)});
}

// ===========================================================================
// In-place KV cache ring buffer upsert
// ===========================================================================

void KVCacheUpsert::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("KVCacheUpsert: CPU not supported");
}

void KVCacheUpsert::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: [cache_k, cache_v, k_new, v_new]
  // outputs: [cache_k_updated, cache_v_updated] — share buffers with inputs
  auto& cache_k = inputs[0];
  auto& cache_v = inputs[1];
  auto& k_new = inputs[2];
  auto& v_new = inputs[3];

  // Donate input buffers to outputs (in-place mutation)
  outputs[0].copy_shared_buffer(cache_k);
  outputs[1].copy_shared_buffer(cache_v);

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  const char* kname = "kv_cache_upsert";
  auto kernel = d.get_kernel(kname, mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_output_array(outputs[0], 0);
  enc.set_output_array(outputs[1], 1);
  enc.set_input_array(k_new, 2);
  enc.set_input_array(v_new, 3);

  struct Params {
    uint32_t N_KV, L, T, D, rs;
  };
  Params params{N_KV_, L_, T_, D_, rs_};
  enc.set_bytes(params, 4);

  constexpr uint32_t tg_size = 256;
  uint32_t n_groups = (T_ + tg_size - 1) / tg_size;
  MTL::Size grid(n_groups, N_KV_, 2);
  MTL::Size group(tg_size, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

std::vector<mx::array> kv_cache_upsert(
    const mx::array& cache_k,
    const mx::array& cache_v,
    const mx::array& k_new,
    const mx::array& v_new,
    uint32_t rs,
    mx::StreamOrDevice s) {
  // Support both [N_KV, L, D] and [1, N_KV, L, D] — use last 3 dims
  int nd = k_new.ndim();
  uint32_t N_KV = static_cast<uint32_t>(k_new.shape(nd - 3));
  uint32_t T = static_cast<uint32_t>(k_new.shape(nd - 2));
  uint32_t D = static_cast<uint32_t>(k_new.shape(nd - 1));
  uint32_t L = static_cast<uint32_t>(cache_k.shape(cache_k.ndim() - 2));

  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {cache_k.shape(), cache_v.shape()},
      {mx::float16, mx::float16},
      std::make_shared<KVCacheUpsert>(stream, N_KV, L, T, D, rs),
      {cache_k, cache_v,
       mx::contiguous(k_new, false, stream),
       mx::contiguous(v_new, false, stream)});
}

}  // namespace we_kernels
