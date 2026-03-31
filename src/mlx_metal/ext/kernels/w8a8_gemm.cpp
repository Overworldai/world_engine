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

static constexpr TileConfig TILE_SMALL    = { 64,  64,  64, 2, 2, "w8a8_gemm_nax_bm64_bn64_bk64_wm2_wn2"};
static constexpr TileConfig TILE_DEEP_K   = { 64,  64, 192, 2, 2, "w8a8_gemm_nax_bm64_bn64_bk192_wm2_wn2"};
static constexpr TileConfig TILE_WIDE_N   = { 64, 128, 128, 2, 4, "w8a8_gemm_nax_bm64_bn128_bk128_wm2_wn4"};
static constexpr TileConfig TILE_LARGE    = {128, 128,  64, 4, 4, "w8a8_gemm_nax_bm128_bn128_bk64_wm4_wn4"};
// Threadgroup budget: BM*(BK+16) + BN*(BK+16) <= 32768

static const TileConfig& select_tile(uint32_t M, uint32_t N, uint32_t K) {
  if (M >= 128 && N >= 128) {
    return TILE_LARGE;
  }
  if (N >= 128) {
    return TILE_WIDE_N;
  }
  if (K >= 192) {
    return TILE_DEEP_K;
  }
  return TILE_SMALL;
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

  const auto& tile = select_tile(M_, N_, K_);

  auto mtl_lib = d.get_library("we_kernels", lib_path());
  auto kernel = d.get_kernel(tile.kernel_name, mtl_lib);

  auto& enc = d.get_command_encoder(s.index);
  enc.set_compute_pipeline_state(kernel);

  enc.set_input_array(x_q, 0);
  enc.set_input_array(w_q, 1);
  enc.set_input_array(x_scales, 2);
  enc.set_input_array(w_scales, 3);
  enc.set_input_array(bias, 4);
  enc.set_output_array(out, 5);

  W8A8Params params{M_, N_, K_};
  enc.set_bytes(params, 6);

  uint32_t tiles_n = (N_ + tile.bn - 1) / tile.bn;
  uint32_t tiles_m = (M_ + tile.bm - 1) / tile.bm;
  MTL::Size grid_dims(tiles_n, tiles_m, 1);
  MTL::Size group_dims(32, tile.wn, tile.wm);
  enc.dispatch_threadgroups(grid_dims, group_dims);
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

}  // namespace we_kernels
