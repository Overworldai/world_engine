// Ring-buffer flash attention kernel with GQA support.
//
// Tiled flash-attention that skips unwritten KV cache blocks.
// Dispatch: one threadgroup per (q_block, q_head).
// Each threadgroup processes BQ=16 queries and streams through KV blocks
// of size BK=32 using 4 simdgroups (128 threads).
//
// Each simdgroup handles BQ/4=4 queries across all D dims.
// For each KV block, all simdgroups cooperatively load K/V into threadgroup
// memory, then each computes dot products for its queries.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct RingAttnParams {
    uint N_Q_HEADS;
    uint N_KV_HEADS;
    uint T;
    uint CAPACITY;
    uint D_HEAD;
    float scale;
};

constant constexpr int BQ = 16;     // queries per threadgroup
constant constexpr int BK = 32;     // KV tokens per block
constant constexpr int NUM_SG = 4;  // simdgroups per threadgroup
constant constexpr int TG_SIZE = NUM_SG * 32;  // 128 threads
constant constexpr int QS_PER_SG = BQ / NUM_SG; // 4 queries per simdgroup
constant constexpr int MAX_D = 64;

[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void ring_flash_attention(
    device const half*  Q          [[buffer(0)]],
    device const half*  K          [[buffer(1)]],
    device const half*  V          [[buffer(2)]],
    device const half*  written    [[buffer(3)]],
    device half*        O          [[buffer(4)]],
    constant RingAttnParams& params [[buffer(5)]],
    uint3 tgid      [[threadgroup_position_in_grid]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  sgid      [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    const uint N_Q = params.N_Q_HEADS;
    const uint N_KV = params.N_KV_HEADS;
    const uint T_Q = params.T;
    const uint CAP = params.CAPACITY;
    const uint D = params.D_HEAD;
    const float scale = params.scale;

    const uint q_block = tgid.x;
    const uint q_head  = tgid.y;

    if (q_head >= N_Q) return;

    const uint q_start = q_block * BQ;
    if (q_start >= T_Q) return;
    const uint q_end = min(q_start + (uint)BQ, T_Q);

    // GQA mapping
    const uint kv_head = q_head / (N_Q / N_KV);

    const device half* Q_h = Q + q_head * T_Q * D;
    const device half* K_h = K + kv_head * CAP * D;
    const device half* V_h = V + kv_head * CAP * D;
    device half* O_h = O + q_head * T_Q * D;

    // Each simdgroup handles QS_PER_SG=4 queries
    // Each thread in simdgroup handles 2 dims (for D=64, 32 threads × 2 = 64)
    const uint sq_start = q_start + sgid * QS_PER_SG;
    const uint d0 = simd_lid * 2;
    const uint d1 = d0 + 1;

    // Load queries for this simdgroup (pre-scaled)
    float qv[QS_PER_SG][2];
    for (int qi = 0; qi < QS_PER_SG; qi++) {
        uint gq = sq_start + qi;
        if (gq < q_end && d0 < D) {
            qv[qi][0] = (float)Q_h[gq * D + d0] * scale;
            qv[qi][1] = (float)Q_h[gq * D + d1] * scale;
        } else {
            qv[qi][0] = 0.0f;
            qv[qi][1] = 0.0f;
        }
    }

    // Online softmax state per query
    float row_max[QS_PER_SG];
    float row_sum[QS_PER_SG];
    float acc[QS_PER_SG][2];
    for (int qi = 0; qi < QS_PER_SG; qi++) {
        row_max[qi] = -1e10f;
        row_sum[qi] = 0.0f;
        acc[qi][0] = 0.0f;
        acc[qi][1] = 0.0f;
    }

    // Threadgroup memory for KV tiles
    threadgroup half K_tile[BK * MAX_D];  // [BK, D]
    threadgroup half V_tile[BK * MAX_D];  // [BK, D]
    threadgroup bool block_valid;

    // Stream through KV blocks
    for (uint blk_start = 0; blk_start < CAP; blk_start += BK) {
        // Cooperative block-level skip check
        // Thread 0 checks written for this block
        if (tid == 0) {
            bool any = false;
            uint blk_end_check = min(blk_start + (uint)BK, CAP);
            for (uint i = blk_start; i < blk_end_check; i++) {
                if ((float)written[i] > 0.5f) { any = true; break; }
            }
            block_valid = any;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!block_valid) continue;

        // Cooperative load K and V tiles into threadgroup memory
        // 128 threads, BK*D = 32*64 = 2048 halves → 16 halves per thread
        uint blk_end = min(blk_start + (uint)BK, CAP);
        uint n_kv = blk_end - blk_start;

        for (uint t = tid; t < (uint)BK * D; t += TG_SIZE) {
            uint row = t / D;
            uint col = t % D;
            uint gkv = blk_start + row;
            if (row < n_kv && gkv < CAP) {
                K_tile[row * D + col] = K_h[gkv * D + col];
                V_tile[row * D + col] = V_h[gkv * D + col];
            } else {
                K_tile[row * D + col] = 0;
                V_tile[row * D + col] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention for this KV block
        for (uint ki = 0; ki < n_kv; ki++) {
            uint gkv = blk_start + ki;
            if ((float)written[gkv] < 0.5f) continue;

            // Load K values for this thread's dims from threadgroup
            float k0 = (float)K_tile[ki * D + d0];
            float k1 = (float)K_tile[ki * D + d1];

            // Load V values
            float v0 = (float)V_tile[ki * D + d0];
            float v1 = (float)V_tile[ki * D + d1];

            #pragma unroll
            for (int qi = 0; qi < QS_PER_SG; qi++) {
                uint gq = sq_start + qi;
                if (gq >= q_end) continue;

                // dot(q, k)
                float dot = qv[qi][0] * k0 + qv[qi][1] * k1;
                float s = simd_sum(dot);

                // Online softmax
                float old_max = row_max[qi];
                row_max[qi] = max(row_max[qi], s);
                float exp_diff = exp(old_max - row_max[qi]);
                float exp_s = exp(s - row_max[qi]);

                acc[qi][0] = acc[qi][0] * exp_diff + exp_s * v0;
                acc[qi][1] = acc[qi][1] * exp_diff + exp_s * v1;
                row_sum[qi] = row_sum[qi] * exp_diff + exp_s;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Normalize and write output
    for (int qi = 0; qi < QS_PER_SG; qi++) {
        uint gq = sq_start + qi;
        if (gq >= q_end) continue;
        float inv = (row_sum[qi] > 0.0f) ? 1.0f / row_sum[qi] : 0.0f;
        if (d0 < D) O_h[gq * D + d0] = (half)(acc[qi][0] * inv);
        if (d1 < D) O_h[gq * D + d1] = (half)(acc[qi][1] * inv);
    }
}
