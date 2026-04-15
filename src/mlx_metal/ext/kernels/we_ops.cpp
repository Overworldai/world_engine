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

// M=32 MMA variant: 50% fewer MMA calls, same compute
static constexpr TileConfig M32_SMALL = { 64, 64, 64, 2, 2, "w8a8_gemm_m32_bm64_bn64_bk64_wm2_wn2"};

static const TileConfig& select_tile(uint32_t M, uint32_t N, uint32_t K) {
  // bench_steady validated (M5 Max, saturated KV cache):
  //
  // V1_SMALL (64×64×64, M=16 NAXFrag) is the baseline winner across shapes:
  // maximum TG count gives best occupancy across 40 GPU cores in lazy eval.
  //
  // M32_SMALL uses matmul2d_descriptor(32,32,16) — 2× work per MMA call.
  // Auto-dispatched for wide N (QKV N=6144, gate_up N=8192) where the
  // halved MMA instruction count translates to a small speedup (+3-8% isolated).
  // Falls back to V1_SMALL for narrow N (N<6000) where M=32 is -6% due to
  // overhead dominating over compute. End-to-end effect is small but positive.
  //
  // Tested alternatives (all slower end-to-end):
  //   V2_DEEP_K for fc2 (K=8192): +15ms despite fewer K-loop iters
  //   V1_DEEP_K for QKV/fc1 (N>=4096): +10ms despite halved K-loop
  //   V1_LARGE, V1_WIDE_N: +10-20ms
  //
  // CAUTION: isolated GEMM benchmarks with mx.eval() are misleading.
  // V2_DEEP_K appears ~20× faster in isolation but ~10% slower end-to-end
  // because lazy eval batches kernels differently than per-call sync.
  // Always validate with bench_steady, never isolated mx.eval() timings.
  //
  // WE_GEMM_M32: 0=always M=16, 1=always M=32, 3=auto (default).
  static int gemm_mode = std::getenv("WE_GEMM_M32") ? std::atoi(std::getenv("WE_GEMM_M32")) : 3;
  if (gemm_mode == 1) return M32_SMALL;
  if (gemm_mode == 3 && N >= 6000) return M32_SMALL;
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
// SageAttention-style per-block int8 SDPA
// ===========================================================================

void SeqSDPAInt8Block::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("SeqSDPAInt8Block: CPU not supported");
}

void SeqSDPAInt8Block::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& Q = inputs[0];
  auto& K_q = inputs[1];
  auto& K_scales = inputs[2];
  auto& V_q = inputs[3];
  auto& V_scales = inputs[4];
  auto& O = outputs[0];

  O.set_data(mx::allocator::malloc(O.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);

  auto mtl_lib = d.get_library("we_kernels", lib_path());
  const char* kname;
  if (bk_ == 64) {
    kname = "seq_sdpa_int8block_bk64_isolated";
  } else {
    kname = "seq_sdpa_int8block_bq32_bk32_wm2";
  }
  auto kernel = d.get_kernel(kname, mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(Q, 0);
  enc.set_input_array(K_q, 1);
  enc.set_input_array(K_scales, 2);
  enc.set_input_array(V_q, 3);
  enc.set_input_array(V_scales, 4);
  enc.set_output_array(O, 5);

  uint32_t capacity = static_cast<uint32_t>(K_q.shape(1));
  struct Params {
    uint32_t N_Q_HEADS, N_KV_HEADS, T_Q, D_HEAD, CAPACITY, NUM_KV_TOKENS;
    float scale;
  };
  Params params{N_Q_, N_KV_, T_, D_HEAD_, capacity, num_kv_tokens_, scale_};
  enc.set_bytes(params, 6);

  constexpr uint32_t BQ = 32;
  uint32_t q_blocks = (T_ + BQ - 1) / BQ;
  MTL::Size grid(q_blocks, N_Q_, 1);
  MTL::Size group(64, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

mx::array seq_sdpa_int8block(
    const mx::array& Q, const mx::array& K_q, const mx::array& K_scales,
    const mx::array& V_q, const mx::array& V_scales,
    uint32_t num_kv_tokens, float scale, mx::StreamOrDevice s) {
  uint32_t N_Q = static_cast<uint32_t>(Q.shape(0));
  uint32_t N_KV = static_cast<uint32_t>(K_q.shape(0));
  uint32_t T = static_cast<uint32_t>(Q.shape(1));
  uint32_t D = static_cast<uint32_t>(Q.shape(2));
  auto stream = mx::to_stream(s);
  return mx::array(
      {static_cast<int>(N_Q), static_cast<int>(T), static_cast<int>(D)},
      mx::float16,
      std::make_shared<SeqSDPAInt8Block>(stream, N_Q, N_KV, T, D, num_kv_tokens, scale, 32),
      {mx::contiguous(Q, false, stream),
       mx::contiguous(K_q, false, stream),
       mx::contiguous(K_scales, false, stream),
       mx::contiguous(V_q, false, stream),
       mx::contiguous(V_scales, false, stream)});
}

// ===========================================================================
// In-place KV cache ring buffer upsert
// ===========================================================================

void KVCacheUpsert::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("KVCacheUpsert: CPU not supported");
}

// Helper shared by both KVCacheUpsert variants — only the kernel name differs.
namespace {
struct KVUpsertParams {
  uint32_t N_KV, L, T, D, rs;
};

static void dispatch_kv_upsert_kernel(
    const char* kernel_name,
    const mx::array& cache_k,
    const mx::array& cache_v,
    const mx::array& k_new,
    const mx::array& v_new,
    uint32_t N_KV, uint32_t L, uint32_t T, uint32_t D, uint32_t rs,
    std::vector<mx::array>& outputs,
    mx::Stream s) {
  outputs[0].copy_shared_buffer(cache_k);
  outputs[1].copy_shared_buffer(cache_v);

  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel(kernel_name, mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);

  enc.set_output_array(outputs[0], 0);
  enc.set_output_array(outputs[1], 1);
  enc.set_input_array(k_new, 2);
  enc.set_input_array(v_new, 3);

  KVUpsertParams params{N_KV, L, T, D, rs};
  enc.set_bytes(params, 4);

  constexpr uint32_t tg_size = 256;
  uint32_t n_groups = (T + tg_size - 1) / tg_size;
  MTL::Size grid(n_groups, N_KV, 2);
  MTL::Size group(tg_size, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}
}  // anonymous namespace

void KVCacheUpsert::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  dispatch_kv_upsert_kernel(
      "kv_cache_upsert",
      inputs[0], inputs[1], inputs[2], inputs[3],
      N_KV_, L_, T_, D_, rs_, outputs, stream());
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

void FusedQuantUpsert::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("FusedQuantUpsert: CPU not supported");
}

void FusedQuantUpsert::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  // inputs: k_new_fp16, v_new_fp16, cache_k_q, cache_k_s, cache_v_q, cache_v_s
  auto& k_new = inputs[0];
  auto& v_new = inputs[1];

  // Donate cache buffers to outputs (in-place mutation)
  outputs[0].copy_shared_buffer(inputs[2]);  // cache_k_q
  outputs[1].copy_shared_buffer(inputs[3]);  // cache_k_s
  outputs[2].copy_shared_buffer(inputs[4]);  // cache_v_q
  outputs[3].copy_shared_buffer(inputs[5]);  // cache_v_s

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel("fused_quant_upsert", mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(k_new, 0);
  enc.set_input_array(v_new, 1);
  enc.set_output_array(outputs[0], 2);
  enc.set_output_array(outputs[1], 3);
  enc.set_output_array(outputs[2], 4);
  enc.set_output_array(outputs[3], 5);

  struct Params {
    uint32_t N_KV, L, T, D, rs, L_BLK, rs_BLK, BK;
  };
  Params params{N_KV_, L_, T_, D_, rs_, L_BLK_, rs_BLK_, BK_};
  enc.set_bytes(params, 6);

  uint32_t n_blocks = T_ / BK_;
  MTL::Size grid(n_blocks, N_KV_, 2);
  MTL::Size group(64, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

std::vector<mx::array> fused_quant_upsert(
    const mx::array& k_new, const mx::array& v_new,
    const mx::array& cache_k_q, const mx::array& cache_k_s,
    const mx::array& cache_v_q, const mx::array& cache_v_s,
    uint32_t rs, uint32_t rs_BLK, uint32_t BK, mx::StreamOrDevice s) {
  int nd = k_new.ndim();
  uint32_t N_KV = static_cast<uint32_t>(k_new.shape(nd - 3));
  uint32_t T = static_cast<uint32_t>(k_new.shape(nd - 2));
  uint32_t D = static_cast<uint32_t>(k_new.shape(nd - 1));
  uint32_t L = static_cast<uint32_t>(cache_k_q.shape(cache_k_q.ndim() - 2));
  uint32_t L_BLK = static_cast<uint32_t>(cache_k_s.shape(cache_k_s.ndim() - 1));

  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {cache_k_q.shape(), cache_k_s.shape(),
       cache_v_q.shape(), cache_v_s.shape()},
      {mx::int8, mx::float16, mx::int8, mx::float16},
      std::make_shared<FusedQuantUpsert>(stream, N_KV, L, T, D, rs, L_BLK, rs_BLK, BK),
      {mx::contiguous(k_new, false, stream),
       mx::contiguous(v_new, false, stream),
       cache_k_q, cache_k_s, cache_v_q, cache_v_s});
}

void KVCacheUpsertInt8Block::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("KVCacheUpsertInt8Block: CPU not supported");
}

void KVCacheUpsertInt8Block::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& cache_k_q     = inputs[0];
  auto& cache_k_scale = inputs[1];
  auto& cache_v_q     = inputs[2];
  auto& cache_v_scale = inputs[3];
  auto& k_new_q       = inputs[4];
  auto& k_new_scale   = inputs[5];
  auto& v_new_q       = inputs[6];
  auto& v_new_scale   = inputs[7];

  outputs[0].copy_shared_buffer(cache_k_q);
  outputs[1].copy_shared_buffer(cache_k_scale);
  outputs[2].copy_shared_buffer(cache_v_q);
  outputs[3].copy_shared_buffer(cache_v_scale);

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel("kv_cache_upsert_int8_block", mtl_lib);
  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_output_array(outputs[0], 0);
  enc.set_output_array(outputs[1], 1);
  enc.set_output_array(outputs[2], 2);
  enc.set_output_array(outputs[3], 3);
  enc.set_input_array(k_new_q, 4);
  enc.set_input_array(k_new_scale, 5);
  enc.set_input_array(v_new_q, 6);
  enc.set_input_array(v_new_scale, 7);

  struct Params {
    uint32_t N_KV, L, T, D, rs, L_BLK, T_BLK, rs_BLK;
  };
  Params params{N_KV_, L_, T_, D_, rs_, L_BLK_, T_BLK_, rs_BLK_};
  enc.set_bytes(params, 8);

  constexpr uint32_t tg_size = 256;
  uint32_t n_groups = (T_ + tg_size - 1) / tg_size;
  MTL::Size grid(n_groups, N_KV_, 2);
  MTL::Size group(tg_size, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

std::vector<mx::array> kv_cache_upsert_int8_block(
    const mx::array& cache_k_q, const mx::array& cache_k_scale,
    const mx::array& cache_v_q, const mx::array& cache_v_scale,
    const mx::array& k_new_q, const mx::array& k_new_scale,
    const mx::array& v_new_q, const mx::array& v_new_scale,
    uint32_t rs, uint32_t rs_BLK, mx::StreamOrDevice s) {
  int nd = k_new_q.ndim();
  uint32_t N_KV = static_cast<uint32_t>(k_new_q.shape(nd - 3));
  uint32_t T = static_cast<uint32_t>(k_new_q.shape(nd - 2));
  uint32_t D = static_cast<uint32_t>(k_new_q.shape(nd - 1));
  uint32_t L = static_cast<uint32_t>(cache_k_q.shape(cache_k_q.ndim() - 2));
  // Scales shape last dim is blocks
  uint32_t T_BLK = static_cast<uint32_t>(k_new_scale.shape(k_new_scale.ndim() - 1));
  uint32_t L_BLK = static_cast<uint32_t>(cache_k_scale.shape(cache_k_scale.ndim() - 1));

  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {cache_k_q.shape(), cache_k_scale.shape(),
       cache_v_q.shape(), cache_v_scale.shape()},
      {mx::int8, mx::float16, mx::int8, mx::float16},
      std::make_shared<KVCacheUpsertInt8Block>(
          stream, N_KV, L, T, D, rs, L_BLK, T_BLK, rs_BLK),
      {cache_k_q, cache_k_scale, cache_v_q, cache_v_scale,
       mx::contiguous(k_new_q, false, stream),
       mx::contiguous(k_new_scale, false, stream),
       mx::contiguous(v_new_q, false, stream),
       mx::contiguous(v_new_scale, false, stream)});
}

// ===========================================================================
// Diagnostic: minimal half4-TG-store race repro. Dispatches the
// repro_half4_tg kernel from the precompiled metallib — same path as
// our RMSNorm kernel, to isolate whether the metallib+primitive path
// alone triggers the race.
// ===========================================================================

void ReproHalf4TG::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("ReproHalf4TG: CPU not supported");
}

void ReproHalf4TG::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& x = inputs[0];
  auto& y = outputs[0];
  y.set_data(mx::allocator::malloc(y.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel("repro_half4_tg", mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0);
  enc.set_output_array(y, 1);

  struct Params { uint32_t M; uint32_t K; };
  Params params{M_, K_};
  enc.set_bytes(params, 2);

  MTL::Size grid(M_, 1, 1);
  MTL::Size group(256, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

mx::array repro_half4_tg(const mx::array& x, mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  auto stream = mx::to_stream(s);
  return mx::array(
      x.shape(),
      mx::float16,
      std::make_shared<ReproHalf4TG>(stream, M, K),
      {mx::contiguous(x, false, stream)});
}

// ----- Variant B: + sum_sq reduction, mirroring RMSNorm Phase 1 -----

void ReproHalf4TGReduce::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("ReproHalf4TGReduce: CPU not supported");
}

void ReproHalf4TGReduce::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& x = inputs[0];
  auto& y = outputs[0];
  auto& sums = outputs[1];
  y.set_data(mx::allocator::malloc(y.nbytes()));
  sums.set_data(mx::allocator::malloc(sums.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel("repro_half4_tg_reduce", mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0);
  enc.set_output_array(y, 1);
  enc.set_output_array(sums, 2);

  struct Params { uint32_t M; uint32_t K; };
  Params params{M_, K_};
  enc.set_bytes(params, 3);

  MTL::Size grid(M_, 1, 1);
  MTL::Size group(256, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

std::vector<mx::array> repro_half4_tg_reduce(
    const mx::array& x, mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {x.shape(), mx::Shape{static_cast<int>(M)}},
      {mx::float16, mx::float32},
      std::make_shared<ReproHalf4TGReduce>(stream, M, K),
      {mx::contiguous(x, false, stream)});
}

// ----- Variant C: + Phase 2 RMW + Phase 3 copy-out -----

void ReproHalf4TGRMW::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("ReproHalf4TGRMW: CPU not supported");
}

void ReproHalf4TGRMW::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& x = inputs[0];
  auto& y = outputs[0];
  auto& rms = outputs[1];
  y.set_data(mx::allocator::malloc(y.nbytes()));
  rms.set_data(mx::allocator::malloc(rms.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel("repro_half4_tg_rmw", mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0);
  enc.set_output_array(y, 1);
  enc.set_output_array(rms, 2);

  struct Params { uint32_t M; uint32_t K; };
  Params params{M_, K_};
  enc.set_bytes(params, 3);

  MTL::Size grid(M_, 1, 1);
  MTL::Size group(256, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

std::vector<mx::array> repro_half4_tg_rmw(
    const mx::array& x, mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {x.shape(), mx::Shape{static_cast<int>(M)}},
      {mx::float16, mx::float32},
      std::make_shared<ReproHalf4TGRMW>(stream, M, K),
      {mx::contiguous(x, false, stream)});
}

// ----- Variant D: + AdaLN device reads -----

void ReproHalf4TGAdaLN::eval_cpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  throw std::runtime_error("ReproHalf4TGAdaLN: CPU not supported");
}

void ReproHalf4TGAdaLN::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& x = inputs[0];
  auto& adaln_s = inputs[1];
  auto& adaln_b = inputs[2];
  auto& y = outputs[0];
  auto& rms = outputs[1];
  y.set_data(mx::allocator::malloc(y.nbytes()));
  rms.set_data(mx::allocator::malloc(rms.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel("repro_half4_tg_adaln", mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0);
  enc.set_output_array(y, 1);
  enc.set_output_array(rms, 2);
  enc.set_input_array(adaln_s, 3);
  enc.set_input_array(adaln_b, 4);

  struct Params { uint32_t M; uint32_t K; };
  Params params{M_, K_};
  enc.set_bytes(params, 5);

  MTL::Size grid(M_, 1, 1);
  MTL::Size group(256, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

std::vector<mx::array> repro_half4_tg_adaln(
    const mx::array& x, const mx::array& adaln_s, const mx::array& adaln_b,
    mx::StreamOrDevice s) {
  uint32_t M = static_cast<uint32_t>(x.shape(0));
  uint32_t K = static_cast<uint32_t>(x.shape(1));
  auto stream = mx::to_stream(s);
  return mx::array::make_arrays(
      {x.shape(), mx::Shape{static_cast<int>(M)}},
      {mx::float16, mx::float32},
      std::make_shared<ReproHalf4TGAdaLN>(stream, M, K),
      {mx::contiguous(x, false, stream),
       mx::contiguous(adaln_s, false, stream),
       mx::contiguous(adaln_b, false, stream)});
}

// ----- Workaround variants share eval_gpu via base class -----

void ReproHalf4TGAdaLNBase::eval_cpu(
    const std::vector<mx::array>&, std::vector<mx::array>&) {
  throw std::runtime_error("ReproHalf4TGAdaLNBase: CPU not supported");
}

void ReproHalf4TGAdaLNBase::eval_gpu(
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>& outputs) {
  auto& x = inputs[0];
  auto& adaln_s = inputs[1];
  auto& adaln_b = inputs[2];
  auto& y = outputs[0];
  auto& rms = outputs[1];
  y.set_data(mx::allocator::malloc(y.nbytes()));
  rms.set_data(mx::allocator::malloc(rms.nbytes()));

  auto& s = stream();
  auto& d = mx::metal::device(s.device);
  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel(kernel_name_, mtl_lib);

  auto& enc = mx::metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  enc.set_input_array(x, 0);
  enc.set_output_array(y, 1);
  enc.set_output_array(rms, 2);
  enc.set_input_array(adaln_s, 3);
  enc.set_input_array(adaln_b, 4);

  struct Params { uint32_t M; uint32_t K; };
  Params params{M_, K_};
  enc.set_bytes(params, 5);

  MTL::Size grid(M_, 1, 1);
  MTL::Size group(256, 1, 1);
  enc.dispatch_threadgroups(grid, group);
}

#define DEFINE_WORKAROUND_DISPATCH(fn_name, ClassName)                     \
  std::vector<mx::array> fn_name(                                          \
      const mx::array& x, const mx::array& adaln_s,                        \
      const mx::array& adaln_b, mx::StreamOrDevice s) {                    \
    uint32_t M = static_cast<uint32_t>(x.shape(0));                        \
    uint32_t K = static_cast<uint32_t>(x.shape(1));                        \
    auto stream = mx::to_stream(s);                                        \
    return mx::array::make_arrays(                                         \
        {x.shape(), mx::Shape{static_cast<int>(M)}},                       \
        {mx::float16, mx::float32},                                        \
        std::make_shared<ClassName>(stream, M, K),                         \
        {mx::contiguous(x, false, stream),                                 \
         mx::contiguous(adaln_s, false, stream),                           \
         mx::contiguous(adaln_b, false, stream)});                         \
  }

DEFINE_WORKAROUND_DISPATCH(repro_half4_tg_dualflag,    ReproHalf4TGDualFlag)
DEFINE_WORKAROUND_DISPATCH(repro_half4_tg_regprefetch, ReproHalf4TGRegPrefetch)
DEFINE_WORKAROUND_DISPATCH(repro_half4_tg_tgprefetch,  ReproHalf4TGTGPrefetch)
DEFINE_WORKAROUND_DISPATCH(repro_half4_tg_volatile,    ReproHalf4TGVolatile)
#undef DEFINE_WORKAROUND_DISPATCH

}  // namespace we_kernels
