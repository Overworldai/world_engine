// Fused QKV split + per-head RMSNorm + OrthoRoPE kernel.
//
// Takes flat QKV output [T, (N_Q + N_K + N_V) * D_HEAD] from GEMM and produces:
//   q [N_Q, T, D_HEAD] — RMSNorm + OrthoRoPE applied
//   k [N_K, T, D_HEAD] — RMSNorm + OrthoRoPE applied
//   v [N_V, T, D_HEAD] — copied (no norm/rope)
//
// Each threadgroup handles one token × HEADS_PER_TG heads.
// Each simdgroup within the TG processes one head (D_HEAD elements).
// Reduces TG count 8× vs one-TG-per-head approach.
//
// Grid: (T, ceil(N_TOTAL / HEADS_PER_TG), 1)
// Group: (HEADS_PER_TG * 32, 1, 1)

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct FusedQKVRoPEParams {
    uint T;
    uint N_Q;
    uint N_K;
    uint N_V;
    uint D_HEAD;
    uint D_ROPE;
    float eps;
};

constant constexpr int HEADS_PER_TG = 8;
constant constexpr int SIMD_SIZE = 32;
constant constexpr int TG_SIZE = HEADS_PER_TG * SIMD_SIZE;  // 256

[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void fused_qkv_norm_rope(
    device const half* qkv_in   [[buffer(0)]],
    device half*       q_out    [[buffer(1)]],
    device half*       k_out    [[buffer(2)]],
    device half*       v_out    [[buffer(3)]],
    device const half* rope_cos [[buffer(4)]],
    device const half* rope_sin [[buffer(5)]],
    constant FusedQKVRoPEParams& params [[buffer(6)]],
    uint2 tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_index_in_threadgroup]])
{
    const uint T = params.T;
    const uint N_Q = params.N_Q;
    const uint N_K = params.N_K;
    const uint N_V = params.N_V;
    const uint D_HEAD = params.D_HEAD;
    const uint D_ROPE = params.D_ROPE;
    const float eps = params.eps;
    const uint N_TOTAL = N_Q + N_K + N_V;

    const uint token = tgid.x;
    if (token >= T) return;

    // Each simdgroup handles one head
    const uint sg_id = tid / SIMD_SIZE;
    const uint lane = tid % SIMD_SIZE;
    const uint head = tgid.y * HEADS_PER_TG + sg_id;
    if (head >= N_TOTAL) return;

    // Input pointer for this (token, head)
    const uint qkv_stride = N_TOTAL * D_HEAD;
    const device half* src = qkv_in + token * qkv_stride + head * D_HEAD;

    const bool is_q = (head < N_Q);
    const bool is_k = (!is_q && head < N_Q + N_K);

    const uint elems_per_thread = (D_HEAD + SIMD_SIZE - 1) / SIMD_SIZE;

    if (is_q || is_k) {
        // Q or K: RMSNorm + OrthoRoPE

        float vals[4];
        float sum_sq = 0.0f;

        for (uint e = 0; e < elems_per_thread; e++) {
            uint idx = lane + e * SIMD_SIZE;
            if (idx < D_HEAD) {
                float v = (float)src[idx];
                vals[e] = v;
                sum_sq += v * v;
            }
        }

        sum_sq = simd_sum(sum_sq);
        float rms_inv = rsqrt(sum_sq / (float)D_HEAD + eps);

        device half* dst;
        if (is_q) {
            dst = q_out + head * T * D_HEAD + token * D_HEAD;
        } else {
            dst = k_out + (head - N_Q) * T * D_HEAD + token * D_HEAD;
        }

        const device half* rc = rope_cos + token * D_ROPE;
        const device half* rs = rope_sin + token * D_ROPE;

        for (uint e = 0; e < elems_per_thread; e++) {
            uint idx = lane + e * SIMD_SIZE;
            if (idx >= D_HEAD) continue;

            float normed = vals[e] * rms_inv;

            bool is_even = (idx % 2 == 0);
            uint pair_lane = is_even ? (lane + 1) : (lane - 1);
            float pair_normed = simd_shuffle(normed, pair_lane);

            uint rope_idx = idx / 2;
            float c = (float)rc[rope_idx];
            float s = (float)rs[rope_idx];

            if (is_even) {
                dst[rope_idx] = (half)(normed * c - pair_normed * s);
            } else {
                dst[D_ROPE + rope_idx] = (half)(normed * c + pair_normed * s);
            }
        }
    } else {
        // V: just copy/transpose
        uint v_head = head - N_Q - N_K;
        device half* dst = v_out + v_head * T * D_HEAD + token * D_HEAD;

        for (uint e = 0; e < elems_per_thread; e++) {
            uint idx = lane + e * SIMD_SIZE;
            if (idx < D_HEAD) {
                dst[idx] = src[idx];
            }
        }
    }
}
