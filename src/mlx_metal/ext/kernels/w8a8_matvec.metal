// W8A8 matrix-vector kernel for M=1..4 (single/few-token decode)
//
// v2: Wider tiles (BN=32) with 4 simdgroups, each producing 8 results.
// Each thread loads 32 int8 values via 2x int4 (16-byte) reads, computes
// partial dot products, then simd_sum reduces across 32 lanes.
//
// Improvements over v1:
//   - 4 simdgroups × 8 results each = BN=32 (was 2×4=8): 4× fewer TGs
//   - Vectorized 4-wide char4 dot using multiply-add chains
//   - Better GPU occupancy from larger threadgroups (128 threads)

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct W8A8Params {
    uint M;
    uint N;
    uint K;
};

constant constexpr int SIMD_SIZE = 32;
constant constexpr int VALUES_PER_THREAD = 32;
constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE; // 1024
constant constexpr int RESULTS_PER_SG = 8;
constant constexpr int NUM_SG = 4;
constant constexpr int BN = NUM_SG * RESULTS_PER_SG; // 32
constant constexpr int TG_SIZE = NUM_SG * SIMD_SIZE;  // 128

inline int dot16(int4 a, int4 b) {
    // Unpack 2x int4 (each int4 = 4x int32 holding 4x int8 as char4)
    // into 16 int8 pairs, compute sum of products.
    char4 a0 = as_type<char4>(a[0]); char4 b0 = as_type<char4>(b[0]);
    char4 a1 = as_type<char4>(a[1]); char4 b1 = as_type<char4>(b[1]);
    char4 a2 = as_type<char4>(a[2]); char4 b2 = as_type<char4>(b[2]);
    char4 a3 = as_type<char4>(a[3]); char4 b3 = as_type<char4>(b[3]);
    int sum = 0;
    sum += (int)a0.x*b0.x + (int)a0.y*b0.y + (int)a0.z*b0.z + (int)a0.w*b0.w;
    sum += (int)a1.x*b1.x + (int)a1.y*b1.y + (int)a1.z*b1.z + (int)a1.w*b1.w;
    sum += (int)a2.x*b2.x + (int)a2.y*b2.y + (int)a2.z*b2.z + (int)a2.w*b2.w;
    sum += (int)a3.x*b3.x + (int)a3.y*b3.y + (int)a3.z*b3.z + (int)a3.w*b3.w;
    return sum;
}

[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void w8a8_matvec(
    device const int8_t* x_q      [[buffer(0)]],
    device const int8_t* w_q      [[buffer(1)]],
    device const float*  x_scales [[buffer(2)]],
    device const float*  w_scales [[buffer(3)]],
    device const float*  bias     [[buffer(4)]],
    device half*         out      [[buffer(5)]],
    constant W8A8Params& params   [[buffer(6)]],
    uint3 tid       [[threadgroup_position_in_grid]],
    uint  sgid      [[simdgroup_index_in_threadgroup]],
    uint  simd_lid  [[thread_index_in_simdgroup]])
{
    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    const uint m_row = tid.x;
    const uint n_base = tid.y * BN;

    if (m_row >= M) return;

    const uint out_row_base = n_base + sgid * RESULTS_PER_SG;

    const device int4* x_vec = reinterpret_cast<device const int4*>(
        x_q + m_row * K) + simd_lid * 2;

    float x_sc = x_scales[m_row];

    int result[RESULTS_PER_SG] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (uint k = 0; k < K; k += BLOCK_SIZE) {
        int4 xv0 = x_vec[0];
        int4 xv1 = x_vec[1];

        #pragma unroll
        for (int row = 0; row < RESULTS_PER_SG; row++) {
            uint out_row = out_row_base + row;
            if (out_row >= N) continue;

            const device int4* w_vec = reinterpret_cast<device const int4*>(
                w_q + out_row * K + k) + simd_lid * 2;

            result[row] += dot16(xv0, w_vec[0]) + dot16(xv1, w_vec[1]);
        }

        x_vec += SIMD_SIZE * 2;
    }

    #pragma unroll
    for (int row = 0; row < RESULTS_PER_SG; row++) {
        uint out_row = out_row_base + row;
        if (out_row >= N) continue;

        int total = simd_sum(result[row]);
        if (simd_lid == 0) {
            float val = (float)total * x_sc * w_scales[out_row] + bias[out_row];
            out[m_row * N + out_row] = (half)val;
        }
    }
}
