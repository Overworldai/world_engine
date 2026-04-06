// Fused QKV split + per-head RMSNorm + OrthoRoPE kernel.
//
// Takes flat QKV output [T, (N_Q + N_K + N_V) * D_HEAD] from GEMM and produces:
//   q [N_Q, T, D_HEAD] — RMSNorm + OrthoRoPE applied
//   k [N_K, T, D_HEAD] — RMSNorm + OrthoRoPE applied
//   v [N_V, T, D_HEAD] — copied (no norm/rope)
//
// One threadgroup per (token, head) pair. D_HEAD elements processed by
// one simdgroup of 32 threads (2 elements per thread for D_HEAD=64).
//
// Fuses: mx.split + reshape + transpose + 2x rms_norm + 2x ortho_rope
// Saves ~6 kernel dispatches and intermediate memory traffic per block.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct FusedQKVRoPEParams {
    uint T;          // number of tokens (512)
    uint N_Q;        // number of Q heads (32)
    uint N_K;        // number of K heads (32)
    uint N_V;        // number of V heads (32)
    uint D_HEAD;     // head dimension (64)
    uint D_ROPE;     // rope dimension (D_HEAD/2 = 32)
    float eps;       // RMSNorm epsilon
};

constant constexpr int TG_SIZE = 32;  // one simdgroup

[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void fused_qkv_norm_rope(
    device const half* qkv_in   [[buffer(0)]],   // [T, (N_Q+N_K+N_V)*D_HEAD]
    device half*       q_out    [[buffer(1)]],    // [N_Q, T, D_HEAD]
    device half*       k_out    [[buffer(2)]],    // [N_K, T, D_HEAD]
    device half*       v_out    [[buffer(3)]],    // [N_V, T, D_HEAD]
    device const half* rope_cos [[buffer(4)]],    // [1, 1, T, D_ROPE]
    device const half* rope_sin [[buffer(5)]],    // [1, 1, T, D_ROPE]
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

    const uint token = tgid.x;   // which token [0, T)
    const uint head  = tgid.y;   // which head [0, N_Q+N_K+N_V)

    if (token >= T || head >= N_TOTAL) return;

    // Input: row-major [T, N_TOTAL * D_HEAD]
    const uint qkv_stride = N_TOTAL * D_HEAD;
    const device half* src = qkv_in + token * qkv_stride + head * D_HEAD;

    // Determine if this is a Q, K, or V head
    const bool is_q = (head < N_Q);
    const bool is_k = (!is_q && head < N_Q + N_K);
    // else is_v

    // Each thread handles ceil(D_HEAD / TG_SIZE) elements
    // For D_HEAD=64, TG_SIZE=32: 2 elements per thread
    const uint elems_per_thread = (D_HEAD + TG_SIZE - 1) / TG_SIZE;

    if (is_q || is_k) {
        // Q or K: apply RMSNorm + OrthoRoPE

        // Phase 1: load + sum of squares
        float vals[4];  // max 4 elements per thread (supports D_HEAD up to 128)
        float sum_sq = 0.0f;

        for (uint e = 0; e < elems_per_thread; e++) {
            uint idx = tid + e * TG_SIZE;
            if (idx < D_HEAD) {
                float v = (float)src[idx];
                vals[e] = v;
                sum_sq += v * v;
            }
        }

        // SIMD reduce sum of squares
        sum_sq = simd_sum(sum_sq);
        float rms_inv = rsqrt(sum_sq / (float)D_HEAD + eps);

        // Phase 2: normalize + OrthoRoPE + store
        // OrthoRoPE output layout (split, matching Python concat([y0, y1])):
        //   x0 = x[0::2], x1 = x[1::2]   (even/odd elements)
        //   y0[i] = x0[i]*cos[i] - x1[i]*sin[i]   → stored at dst[i]
        //   y1[i] = x1[i]*cos[i] + x0[i]*sin[i]   → stored at dst[D_ROPE + i]
        //
        // Thread layout for D_HEAD=64, TG_SIZE=32:
        //   Thread k processes input indices k (e=0) and k+32 (e=1).
        //   Even threads (k even): handle pairs (k, k+1) via shuffle with lane k+1
        //   Odd threads (k odd): handle pairs (k-1, k) via shuffle with lane k-1

        device half* dst;
        if (is_q) {
            dst = q_out + head * T * D_HEAD + token * D_HEAD;
        } else {
            dst = k_out + (head - N_Q) * T * D_HEAD + token * D_HEAD;
        }

        const device half* rc = rope_cos + token * D_ROPE;
        const device half* rs = rope_sin + token * D_ROPE;

        for (uint e = 0; e < elems_per_thread; e++) {
            uint idx = tid + e * TG_SIZE;
            if (idx >= D_HEAD) continue;

            float normed = vals[e] * rms_inv;

            // Get paired element via simd_shuffle
            bool is_even = (idx % 2 == 0);
            uint pair_lane = is_even ? (tid + 1) : (tid - 1);

            float pair_normed = simd_shuffle(normed, pair_lane);

            uint rope_idx = idx / 2;  // which cos/sin pair
            float c = (float)rc[rope_idx];
            float s = (float)rs[rope_idx];

            if (is_even) {
                // This is x0[rope_idx], pair is x1[rope_idx]
                // y0[rope_idx] = x0*cos - x1*sin → store at dst[rope_idx]
                float y0 = normed * c - pair_normed * s;
                dst[rope_idx] = (half)y0;
            } else {
                // This is x1[rope_idx], pair is x0[rope_idx]
                // y1[rope_idx] = x1*cos + x0*sin → store at dst[D_ROPE + rope_idx]
                float y1 = normed * c + pair_normed * s;
                dst[D_ROPE + rope_idx] = (half)y1;
            }
        }
    } else {
        // V head: just copy/transpose, no norm or rope
        uint v_head = head - N_Q - N_K;
        device half* dst = v_out + v_head * T * D_HEAD + token * D_HEAD;

        for (uint e = 0; e < elems_per_thread; e++) {
            uint idx = tid + e * TG_SIZE;
            if (idx < D_HEAD) {
                dst[idx] = src[idx];
            }
        }
    }
}
