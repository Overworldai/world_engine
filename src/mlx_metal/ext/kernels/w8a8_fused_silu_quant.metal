// Fused SiLU activation + per-row symmetric int8 quantization.
// One threadgroup per row, 256 threads.
// Input:  fp16 [M, K]  (GEMM output)
// Output: int8 [M, K] + fp32 scales [M]

#include <metal_stdlib>
using namespace metal;

struct FusedSiLUQuantParams {
    uint M;
    uint K;
};

constant constexpr int TG_SIZE = 256;
constant constexpr int MAX_K = 8192;

[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void fused_silu_quant(
    device const half* x        [[buffer(0)]],
    device int8_t*     x_q      [[buffer(1)]],
    device float*      x_scales [[buffer(2)]],
    constant FusedSiLUQuantParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half* x_row = x + row * K;
    device int8_t* q_row = x_q + row * K;

    threadgroup half silu_cache[MAX_K];
    threadgroup float sg_max[TG_SIZE / 32];

    // Phase 1: SiLU + cache + absmax
    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_row[k];
        float s = v / (1.0f + exp(-v));  // SiLU
        silu_cache[k] = (half)s;
        local_max = max(local_max, abs(s));
    }

    // SIMD reduce
    local_max = simd_max(local_max);

    // Cross-simdgroup reduce
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_max[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_max[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) {
            sg_max[0] = max(v / 127.0f, 1e-6f);
            x_scales[row] = sg_max[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_scale = 1.0f / sg_max[0];

    // Phase 2: quantize from cache
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)silu_cache[k] * inv_scale;
        q_row[k] = (int8_t)clamp(rint(v), -127.0f, 127.0f);
    }
}