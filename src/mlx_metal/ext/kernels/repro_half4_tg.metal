// Minimal isolating repro for the half4-TG-store race.
//
// Structure: pure copy x → TG cache (via vectorized half4 writes)
// → TG cache → y (via scalar reads at different indices).
// No sum_sq, no reduction, no multi-phase — just the essential pattern.
//
// Paired with:
//   - C++ primitive `ReproHalf4TG` in we_ops.cpp
//   - Python entry `repro_half4_tg` in we_kernels/__init__.py
//   - Driver in tests/repro_half4_tg_race.py (--mode metallib)
//
// Goal: if THIS reproduces the race, the bug is narrowed to
// .metallib + custom-Primitive dispatch; the RMSNorm complexity is
// incidental.

#include <metal_stdlib>
using namespace metal;

struct ReproHalf4TGParams {
    uint M;
    uint K;
};

constant constexpr int TG_SIZE = 256;
constant constexpr int MAX_K = 2048;

// Variant A: pure copy via half4 TG writes (minimal). Does NOT reproduce.
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void repro_half4_tg(
    device const half* x        [[buffer(0)]],
    device       half* y        [[buffer(1)]],
    constant ReproHalf4TGParams& params [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const uint K4 = K / 4;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half4* x_row4 =
        reinterpret_cast<const device half4*>(x + row * K);
    device half* y_row = y + row * K;

    threadgroup half cache[MAX_K];

    for (uint k4 = tid; k4 < K4; k4 += TG_SIZE) {
        half4 h = x_row4[k4];
        uint k = k4 * 4;
        cache[k+0] = h.x;
        cache[k+1] = h.y;
        cache[k+2] = h.z;
        cache[k+3] = h.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = cache[k];
    }
}


// Variant B: adds the RMSNorm-style sum_sq reduction before the barrier.
// Writes x_cache via half4 AND writes sg_reduce via lane-0 scalar stores,
// then a SINGLE threadgroup_barrier. This is the pattern that races in
// our real kernel: after the barrier, some rows read stale x_cache[]
// values that should have been overwritten by Phase 1.
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void repro_half4_tg_reduce(
    device const half* x        [[buffer(0)]],
    device       half* y        [[buffer(1)]],
    device       float* sum_out [[buffer(2)]],
    constant ReproHalf4TGParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const uint K4 = K / 4;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half4* x_row4 =
        reinterpret_cast<const device half4*>(x + row * K);
    device half* y_row = y + row * K;

    threadgroup half cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Phase 1: half4 device read → 4 scalar TG writes + sum_sq register
    // accumulation (matches RMSNorm Phase 1 exactly).
    float sum_sq = 0.0f;
    for (uint k4 = tid; k4 < K4; k4 += TG_SIZE) {
        half4 h = x_row4[k4];
        uint k = k4 * 4;
        cache[k+0] = h.x;
        cache[k+1] = h.y;
        cache[k+2] = h.z;
        cache[k+3] = h.w;
        float4 v = float4(h);
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    // Simd-level reduce, then cross-simdgroup reduce via TG memory.
    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            sg_reduce[0] = v;
            sum_out[row] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: scalar TG → device copy. Correct output = x (a pure copy
    // through TG). If the race is present, some elements don't match.
    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = cache[k];
    }
}


// Variant C: adds Phase 2 read-modify-write to x_cache, mirroring
// RMSNorm's normalize+absmax pass exactly. This is the last shape
// difference vs the real kernel.
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void repro_half4_tg_rmw(
    device const half* x        [[buffer(0)]],
    device       half* y        [[buffer(1)]],
    device       float* sum_out [[buffer(2)]],
    constant ReproHalf4TGParams& params [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const uint K4 = K / 4;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half4* x_row4 =
        reinterpret_cast<const device half4*>(x + row * K);
    device half* y_row = y + row * K;

    threadgroup half cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Phase 1: vectorized half4 → 4 scalar TG writes + sum_sq.
    float sum_sq = 0.0f;
    for (uint k4 = tid; k4 < K4; k4 += TG_SIZE) {
        half4 h = x_row4[k4];
        uint k = k4 * 4;
        cache[k+0] = h.x;
        cache[k+1] = h.y;
        cache[k+2] = h.z;
        cache[k+3] = h.w;
        float4 v = float4(h);
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            sg_reduce[0] = rsqrt(v / (float)K + 1e-5f);
            sum_out[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = sg_reduce[0];

    // Phase 2: read x_cache → scale by rms_inv → WRITE BACK to x_cache
    // + accumulate local_max. Matches RMSNorm exactly (minus AdaLN).
    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)cache[k] * rms_inv;
        cache[k] = (half)v;
        local_max = max(local_max, abs(v));
    }
    local_max = simd_max(local_max);
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) sg_reduce[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: copy cache → y. Correct output: y[r,k] = x[r,k]*rms_inv.
    // Race symptom: some rows show wildly different values because
    // Phase 1's writes didn't all land before Phase 2 scaled them.
    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = cache[k];
    }
}


// Variant D: + AdaLN device reads in Phase 2. Full structural match
// to fused_rmsnorm_adaln_quant Phase 1/2.
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void repro_half4_tg_adaln(
    device const half* x          [[buffer(0)]],
    device       half* y          [[buffer(1)]],
    device       float* sum_out   [[buffer(2)]],
    device const half* adaln_s    [[buffer(3)]],
    device const half* adaln_b    [[buffer(4)]],
    constant ReproHalf4TGParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const uint K4 = K / 4;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half4* x_row4 =
        reinterpret_cast<const device half4*>(x + row * K);
    device half* y_row = y + row * K;

    threadgroup half cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    float sum_sq = 0.0f;
    for (uint k4 = tid; k4 < K4; k4 += TG_SIZE) {
        half4 h = x_row4[k4];
        uint k = k4 * 4;
        cache[k+0] = h.x;
        cache[k+1] = h.y;
        cache[k+2] = h.z;
        cache[k+3] = h.w;
        float4 v = float4(h);
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            sg_reduce[0] = rsqrt(v / (float)K + 1e-5f);
            sum_out[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = sg_reduce[0];

    // Phase 2 with AdaLN-style device reads (adaln_s[k], adaln_b[k]).
    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)cache[k] * rms_inv;
        v = v * (1.0f + (float)adaln_s[k]) + (float)adaln_b[k];
        cache[k] = (half)v;
        local_max = max(local_max, abs(v));
    }
    local_max = simd_max(local_max);
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) sg_reduce[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = cache[k];
    }
}


// =====================================================================
// Workaround candidates for variant D's race. Each preserves BOTH
// Phase 1 half4 TG writes AND Phase 2 device reads, but changes the
// barrier semantics or where the device reads happen.
// =====================================================================

// Variant E: barrier with BOTH mem_threadgroup AND mem_device flags.
// Hypothesis: under Family 9 unified cache, forcing both memory views
// to synchronize should flush cache lines for subsequent device loads.
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void repro_half4_tg_dualflag(
    device const half* x          [[buffer(0)]],
    device       half* y          [[buffer(1)]],
    device       float* sum_out   [[buffer(2)]],
    device const half* adaln_s    [[buffer(3)]],
    device const half* adaln_b    [[buffer(4)]],
    constant ReproHalf4TGParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const uint K4 = K / 4;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half4* x_row4 =
        reinterpret_cast<const device half4*>(x + row * K);
    device half* y_row = y + row * K;

    threadgroup half cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    float sum_sq = 0.0f;
    for (uint k4 = tid; k4 < K4; k4 += TG_SIZE) {
        half4 h = x_row4[k4];
        uint k = k4 * 4;
        cache[k+0] = h.x; cache[k+1] = h.y;
        cache[k+2] = h.z; cache[k+3] = h.w;
        float4 v = float4(h);
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            sg_reduce[0] = rsqrt(v / (float)K + 1e-5f);
            sum_out[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    float rms_inv = sg_reduce[0];

    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)cache[k] * rms_inv;
        v = v * (1.0f + (float)adaln_s[k]) + (float)adaln_b[k];
        cache[k] = (half)v;
        local_max = max(local_max, abs(v));
    }
    local_max = simd_max(local_max);
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) sg_reduce[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = cache[k];
    }
}


// Variant F: pre-load adaln_s / adaln_b into per-thread REGISTERS
// before Phase 2 begins.
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void repro_half4_tg_regprefetch(
    device const half* x          [[buffer(0)]],
    device       half* y          [[buffer(1)]],
    device       float* sum_out   [[buffer(2)]],
    device const half* adaln_s    [[buffer(3)]],
    device const half* adaln_b    [[buffer(4)]],
    constant ReproHalf4TGParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const uint K4 = K / 4;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half4* x_row4 =
        reinterpret_cast<const device half4*>(x + row * K);
    device half* y_row = y + row * K;

    threadgroup half cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Pre-load adaln_s / adaln_b into per-thread registers.
    // At K=2048 / TG_SIZE=256 each thread holds 8 halves of each.
    constexpr uint kPerThread = 8;
    float s_reg[kPerThread], b_reg[kPerThread];
    for (uint i = 0; i < kPerThread; i++) {
        uint k = tid + i * TG_SIZE;
        s_reg[i] = (float)adaln_s[k];
        b_reg[i] = (float)adaln_b[k];
    }

    float sum_sq = 0.0f;
    for (uint k4 = tid; k4 < K4; k4 += TG_SIZE) {
        half4 h = x_row4[k4];
        uint k = k4 * 4;
        cache[k+0] = h.x; cache[k+1] = h.y;
        cache[k+2] = h.z; cache[k+3] = h.w;
        float4 v = float4(h);
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            sg_reduce[0] = rsqrt(v / (float)K + 1e-5f);
            sum_out[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = sg_reduce[0];

    // Phase 2 uses s_reg/b_reg (register reads), not device memory.
    float local_max = 0.0f;
    for (uint i = 0; i < kPerThread; i++) {
        uint k = tid + i * TG_SIZE;
        float v = (float)cache[k] * rms_inv;
        v = v * (1.0f + s_reg[i]) + b_reg[i];
        cache[k] = (half)v;
        local_max = max(local_max, abs(v));
    }
    local_max = simd_max(local_max);
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) sg_reduce[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = cache[k];
    }
}


// Variant G: pre-load adaln_s / adaln_b into a TG scratch buffer once
// at the start of the kernel. Phase 2 reads from TG (not device).
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void repro_half4_tg_tgprefetch(
    device const half* x          [[buffer(0)]],
    device       half* y          [[buffer(1)]],
    device       float* sum_out   [[buffer(2)]],
    device const half* adaln_s    [[buffer(3)]],
    device const half* adaln_b    [[buffer(4)]],
    constant ReproHalf4TGParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const uint K4 = K / 4;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half4* x_row4 =
        reinterpret_cast<const device half4*>(x + row * K);
    device half* y_row = y + row * K;

    threadgroup half cache[MAX_K];
    threadgroup half s_tg[MAX_K];
    threadgroup half b_tg[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Phase 0: load adaln into TG scratch (scalar stores).
    for (uint k = tid; k < K; k += TG_SIZE) {
        s_tg[k] = adaln_s[k];
        b_tg[k] = adaln_b[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_sq = 0.0f;
    for (uint k4 = tid; k4 < K4; k4 += TG_SIZE) {
        half4 h = x_row4[k4];
        uint k = k4 * 4;
        cache[k+0] = h.x; cache[k+1] = h.y;
        cache[k+2] = h.z; cache[k+3] = h.w;
        float4 v = float4(h);
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            sg_reduce[0] = rsqrt(v / (float)K + 1e-5f);
            sum_out[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = sg_reduce[0];

    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)cache[k] * rms_inv;
        v = v * (1.0f + (float)s_tg[k]) + (float)b_tg[k];
        cache[k] = (half)v;
        local_max = max(local_max, abs(v));
    }
    local_max = simd_max(local_max);
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) sg_reduce[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = cache[k];
    }
}


// Variant H: volatile threadgroup qualifier on x_cache.
[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void repro_half4_tg_volatile(
    device const half* x          [[buffer(0)]],
    device       half* y          [[buffer(1)]],
    device       float* sum_out   [[buffer(2)]],
    device const half* adaln_s    [[buffer(3)]],
    device const half* adaln_b    [[buffer(4)]],
    constant ReproHalf4TGParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint M = params.M;
    const uint K = params.K;
    const uint K4 = K / 4;
    const uint row = tgid.x;
    if (row >= M) return;

    const device half4* x_row4 =
        reinterpret_cast<const device half4*>(x + row * K);
    device half* y_row = y + row * K;

    volatile threadgroup half cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    float sum_sq = 0.0f;
    for (uint k4 = tid; k4 < K4; k4 += TG_SIZE) {
        half4 h = x_row4[k4];
        uint k = k4 * 4;
        cache[k+0] = h.x; cache[k+1] = h.y;
        cache[k+2] = h.z; cache[k+3] = h.w;
        float4 v = float4(h);
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }
    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            sg_reduce[0] = rsqrt(v / (float)K + 1e-5f);
            sum_out[row] = sg_reduce[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = sg_reduce[0];

    float local_max = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)cache[k] * rms_inv;
        v = v * (1.0f + (float)adaln_s[k]) + (float)adaln_b[k];
        cache[k] = (half)v;
        local_max = max(local_max, abs(v));
    }
    local_max = simd_max(local_max);
    if (lane == 0) sg_reduce[sgid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_max(v);
        if (lane == 0) sg_reduce[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = cache[k];
    }
}
