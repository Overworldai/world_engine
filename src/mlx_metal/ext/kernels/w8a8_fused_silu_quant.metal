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

    threadgroup float sg_max[TG_SIZE / 32];

    // Cross-lane with half4 vector loads: thread tid at iter i hits
    // half4 index (tid + i*TG_SIZE). Adjacent threads hit adjacent half4s.
    const uint per_thread_4 = K / (TG_SIZE * 4);
    const device half4* x_row4 = reinterpret_cast<const device half4*>(x_row);
    device char4* q_row4 = reinterpret_cast<device char4*>(q_row);

    // Pass 1: SiLU + absmax
    float local_max = 0.0f;
    for (uint i = 0; i < per_thread_4; i++) {
        half4 h = x_row4[tid + i * TG_SIZE];
        float4 v = float4(h);
        float4 s = v / (1.0f + fast::exp(-v));
        float4 a = abs(s);
        local_max = max(local_max, max(max(a.x, a.y), max(a.z, a.w)));
    }

    local_max = simd_max(local_max);
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

    // Pass 2: re-read + SiLU + quantize (half4 in, char4 out).
    for (uint i = 0; i < per_thread_4; i++) {
        uint idx = tid + i * TG_SIZE;
        half4 h = x_row4[idx];
        float4 v = float4(h);
        float4 s = v / (1.0f + fast::exp(-v));
        float4 scaled = s * inv_scale;
        char4 q;
        q.x = (int8_t)clamp(rint(scaled.x), -127.0f, 127.0f);
        q.y = (int8_t)clamp(rint(scaled.y), -127.0f, 127.0f);
        q.z = (int8_t)clamp(rint(scaled.z), -127.0f, 127.0f);
        q.w = (int8_t)clamp(rint(scaled.w), -127.0f, 127.0f);
        q_row4[idx] = q;
    }
}