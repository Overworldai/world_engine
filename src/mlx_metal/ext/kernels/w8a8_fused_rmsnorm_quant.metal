// Fused RMSNorm (+ optional AdaLN modulation) + per-row symmetric int8 quantization.
// One threadgroup per row, 256 threads.
// Input:  fp16 [M, K], optional adaln_s [K], adaln_b [K]
// Output: int8 [M, K] + fp32 scales [M]

#include <metal_stdlib>
using namespace metal;

struct FusedRMSNormQuantParams {
    uint M;
    uint K;
    float eps;
};

constant constexpr int TG_SIZE = 256;
constant constexpr int MAX_K = 2048;  // D_MODEL; saves 12KB TG mem vs 8192

// RMSNorm + AdaLN(*(1+s)+b) + int8 quantization
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void fused_rmsnorm_adaln_quant(
    device const half* x        [[buffer(0)]],
    device int8_t*     x_q      [[buffer(1)]],
    device float*      x_scales [[buffer(2)]],
    device const half* adaln_s  [[buffer(3)]],
    device const half* adaln_b  [[buffer(4)]],
    constant FusedRMSNormQuantParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const float eps = params.eps;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half* x_row = x + row * K;
    device int8_t* q_row = x_q + row * K;

    threadgroup half x_cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Phase 1: read + cache + sum of squares
    float sum_sq = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_row[k];
        x_cache[k] = (half)v;
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) sg_reduce[0] = rsqrt(v / (float)K + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = sg_reduce[0];

    // Phase 2: normalize + AdaLN + absmax
    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_cache[k] * rms_inv;
        v = v * (1.0f + (float)adaln_s[k]) + (float)adaln_b[k];
        x_cache[k] = (half)v;
        local_max = max(local_max, abs(v));
    }

    local_max = simd_max(local_max);
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) {
            sg_reduce[0] = max(v / 127.0f, 1e-6f);
            x_scales[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_scale = 1.0f / sg_reduce[0];

    // Phase 3: quantize from cache
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_cache[k] * inv_scale;
        q_row[k] = (int8_t)clamp(rint(v), -127.0f, 127.0f);
    }
}

// RMSNorm + AdaLN + SmoothQuant(per-channel scale) + int8 quantization
// smooth_scale [K] is applied after AdaLN modulation, before quantization:
//   v = (rms_norm(x) * (1+s) + b) * smooth_scale[k]
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void fused_rmsnorm_adaln_smooth_quant(
    device const half* x            [[buffer(0)]],
    device int8_t*     x_q          [[buffer(1)]],
    device float*      x_scales     [[buffer(2)]],
    device const half* adaln_s      [[buffer(3)]],
    device const half* adaln_b      [[buffer(4)]],
    device const half* smooth_scale [[buffer(5)]],
    constant FusedRMSNormQuantParams& params [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const float eps = params.eps;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half* x_row = x + row * K;
    device int8_t* q_row = x_q + row * K;

    threadgroup half x_cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Phase 1: read + cache + sum of squares
    float sum_sq = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_row[k];
        x_cache[k] = (half)v;
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) sg_reduce[0] = rsqrt(v / (float)K + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = sg_reduce[0];

    // Phase 2: normalize + AdaLN + SmoothQuant + absmax
    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_cache[k] * rms_inv;
        v = v * (1.0f + (float)adaln_s[k]) + (float)adaln_b[k];
        v *= (float)smooth_scale[k];
        x_cache[k] = (half)v;
        local_max = max(local_max, abs(v));
    }

    local_max = simd_max(local_max);
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) {
            sg_reduce[0] = max(v / 127.0f, 1e-6f);
            x_scales[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_scale = 1.0f / sg_reduce[0];

    // Phase 3: quantize from cache
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_cache[k] * inv_scale;
        q_row[k] = (int8_t)clamp(rint(v), -127.0f, 127.0f);
    }
}

// RMSNorm (no AdaLN) + int8 quantization
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void fused_rmsnorm_quant(
    device const half* x        [[buffer(0)]],
    device int8_t*     x_q      [[buffer(1)]],
    device float*      x_scales [[buffer(2)]],
    constant FusedRMSNormQuantParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const float eps = params.eps;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half* x_row = x + row * K;
    device int8_t* q_row = x_q + row * K;

    threadgroup half x_cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Phase 1: read + cache + sum of squares
    float sum_sq = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_row[k];
        x_cache[k] = (half)v;
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) sg_reduce[0] = rsqrt(v / (float)K + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = sg_reduce[0];

    // Phase 2: normalize + absmax
    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_cache[k] * rms_inv;
        x_cache[k] = (half)v;
        local_max = max(local_max, abs(v));
    }

    local_max = simd_max(local_max);
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) {
            sg_reduce[0] = max(v / 127.0f, 1e-6f);
            x_scales[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_scale = 1.0f / sg_reduce[0];

    // Phase 3: quantize from cache
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_cache[k] * inv_scale;
        q_row[k] = (int8_t)clamp(rint(v), -127.0f, 127.0f);
    }
}

// Plain per-row symmetric int8 quantization (no RMSNorm, no AdaLN).
// Replaces Python-side quant (abs+max+div+round+clip) with a single dispatch.
// No threadgroup cache — re-reads from device on second pass (matches MLX's
// pattern for simple two-pass kernels). Saves 16KB TG memory → higher occupancy.
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void fused_quant(
    device const half* x        [[buffer(0)]],
    device int8_t*     x_q      [[buffer(1)]],
    device float*      x_scales [[buffer(2)]],
    constant FusedRMSNormQuantParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half* x_row = x + row * K;
    device int8_t* q_row = x_q + row * K;

    threadgroup float sg_reduce[TG_SIZE / 32];

    // Pass 1: absmax
    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        local_max = max(local_max, abs((float)x_row[k]));
    }

    local_max = simd_max(local_max);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) {
            sg_reduce[0] = max(v / 127.0f, 1e-6f);
            x_scales[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_scale = 1.0f / sg_reduce[0];

    // Pass 2: re-read from device + quantize
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_row[k] * inv_scale;
        q_row[k] = (int8_t)clamp(rint(v), -127.0f, 127.0f);
    }
}

// RMSNorm + SmoothQuant + int8 quantization (no AdaLN)
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void fused_rmsnorm_smooth_quant(
    device const half* x            [[buffer(0)]],
    device int8_t*     x_q          [[buffer(1)]],
    device float*      x_scales     [[buffer(2)]],
    device const half* smooth_scale [[buffer(3)]],
    constant FusedRMSNormQuantParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const float eps = params.eps;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half* x_row = x + row * K;
    device int8_t* q_row = x_q + row * K;

    threadgroup half x_cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Phase 1: read + cache + sum of squares
    float sum_sq = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_row[k];
        x_cache[k] = (half)v;
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) sg_reduce[0] = rsqrt(v / (float)K + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = sg_reduce[0];

    // Phase 2: normalize + SmoothQuant + absmax
    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_cache[k] * rms_inv * (float)smooth_scale[k];
        x_cache[k] = (half)v;
        local_max = max(local_max, abs(v));
    }

    local_max = simd_max(local_max);
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) {
            sg_reduce[0] = max(v / 127.0f, 1e-6f);
            x_scales[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_scale = 1.0f / sg_reduce[0];

    // Phase 3: quantize from cache
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_cache[k] * inv_scale;
        q_row[k] = (int8_t)clamp(rint(v), -127.0f, 127.0f);
    }
}
