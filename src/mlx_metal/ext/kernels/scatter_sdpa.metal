// Scatter-read flash attention — v5 (NAX variant).
//
// Direct device memory reads via NAXTile — no threadgroup staging.
// Adapted from MLX Steel's attention_nax kernel with scatter-read
// K/V addressing via block_offsets.
//
// BQ=32, BK=32, BD=64, WM=1, WN=1 → 1 simdgroup, 32 threads.

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;
using namespace mlx::steel;

struct ScatterAttnParams {
    uint N_Q_HEADS;
    uint N_KV_HEADS;
    uint T_Q;
    uint CAPACITY;
    uint D_HEAD;
    uint N_BLOCKS;
    float scale;
};

struct MaxOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct SumOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct ExpSubOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};
struct MulOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};

template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename AccumType = float>
void scatter_sdpa_nax_impl(
    const device T* Q,
    const device T* K_head,
    const device T* V_head,
    const device int* block_offsets,
    device T* O,
    constant ScatterAttnParams& params,
    uint simd_group_id,
    uint3 tid)
{
    const uint N_Q = params.N_Q_HEADS;
    const uint N_KV = params.N_KV_HEADS;
    const uint T_Q_len = params.T_Q;
    const uint D = params.D_HEAD;
    const uint N_BLK = params.N_BLOCKS;

    const uint q_head = tid.y;
    if (q_head >= N_Q) return;
    const uint q_off = tid.x * BQ;
    if (q_off >= T_Q_len) return;

    const uint kv_head = q_head / (N_Q / N_KV);

    // Set up pointers
    Q += q_head * T_Q_len * D + q_off * D;
    O += q_head * T_Q_len * D + q_off * D;
    const device T* K_base = K_head + kv_head * params.CAPACITY * D;
    const device T* V_base = V_head + kv_head * params.CAPACITY * D;

    const float scale2 = params.scale * M_LOG2E_F;

    // NAX tile dimensions
    constexpr short kU = 16;
    constexpr int kNWarps = WM * WN;
    constexpr int TQ = BQ / (kNWarps * kU);  // 1
    constexpr int TD = BD / kU;              // 4
    constexpr short TK = BK / kU;            // 2

    using otile_t = NAXTile<AccumType, TQ, TD>;
    otile_t Otile;
    Otile.clear();

    const short tm = kU * TQ * simd_group_id;
    Q += tm * D;

    const uint n_q = min(q_off + (uint)BQ, T_Q_len) - q_off;
    const short lim_rows_q = n_q - tm;
    const bool is_last_q = (q_off + BQ > T_Q_len);

    // Online softmax state
    constexpr short kRowsPT = otile_t::kRowsPerThread;
    metal::vec<AccumType, kRowsPT> max_score;
    metal::vec<AccumType, kRowsPT> sum_score{0};
    for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = Limits<AccumType>::finite_min;
    }

    // Main loop: iterate over valid KV blocks
    for (uint blk = 0; blk < N_BLK; blk++) {
        int kv_offset = block_offsets[blk];
        const device T* K_blk = K_base + kv_offset * D;
        const device T* V_blk = V_base + kv_offset * D;

        // S = Q @ K^T
        using stile_t = NAXTile<AccumType, TQ, TK>;
        stile_t Stile;
        Stile.clear();

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik += 2) {
                STEEL_PRAGMA_UNROLL
                for (short id = 0; id < TD; id++) {
                    NAXTile<T, 1, 1> Qtile;
                    NAXTile<T, 2, 1> Ktile;

                    const int Q_off = iq * kU * D + id * kU;
                    const int K_off = ik * kU * D + id * kU;

                    if (is_last_q) {
                        Qtile.load_rows(Q + Q_off, D, lim_rows_q - iq * kU);
                    } else {
                        Qtile.load(Q + Q_off, D);
                    }

                    Ktile.load(K_blk + K_off, D);

                    stile_t::NAXFrag_t::mma(
                        Stile.frag_at(iq, ik),
                        Stile.frag_at(iq, ik + 1),
                        Qtile.frag_at(0, 0),
                        metal::false_type{},
                        Ktile.frag_at(0, 0),
                        Ktile.frag_at(1, 0),
                        metal::true_type{});
                }
            }
        }

        // Scale
        STEEL_PRAGMA_UNROLL
        for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
            Stile.elems()[ii] *= scale2;
        }

        // Online softmax
        metal::vec<AccumType, kRowsPT> new_max;
        metal::vec<AccumType, kRowsPT> factor;
        for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];

        Stile.template row_reduce<MaxOp>(new_max);
        Stile.template row_bin_op<ExpSubOp>(new_max);

        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
            sum_score[i] = sum_score[i] * factor[i];
        }
        Stile.template row_reduce<SumOp>(sum_score);
        Otile.template row_bin_op<MulOp>(factor);

        simdgroup_barrier(mem_flags::mem_none);

        // O += S_exp @ V
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id += 2) {
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    NAXTile<T, 1, 2> Vtile;
                    const int V_off = ik * kU * D + id * kU;
                    Vtile.load(V_blk + V_off, D);

                    otile_t::NAXFrag_t::mma(
                        Otile.frag_at(iq, id),
                        Otile.frag_at(iq, id + 1),
                        Stile.frag_at(iq, ik),
                        metal::false_type{},
                        Vtile.frag_at(0, 0),
                        Vtile.frag_at(0, 1),
                        metal::false_type{});
                }
            }
        }
    }

    // Normalize
    threadgroup_barrier(mem_flags::mem_none);
    metal::vec<AccumType, kRowsPT> rcp;
    for (short i = 0; i < kRowsPT; ++i) rcp[i] = 1.f / sum_score[i];
    Otile.template row_bin_op<MulOp>(rcp);

    // Store
    O += tm * D;
    if (is_last_q) {
        if (lim_rows_q <= 0) return;
        Otile.store_rows(O, D, lim_rows_q);
    } else {
        Otile.store(O, D);
    }
}

// Kernel instantiations — different tile configs for autotuning
// Naming: scatter_sdpa_bqXX_bkYY_wmZ

#define SCATTER_SDPA_KERNEL(suffix, _BQ, _BK, _BD, _WM, _WN)                  \
[[kernel, max_total_threads_per_threadgroup(_WM * _WN * 32)]]                  \
void scatter_sdpa_##suffix(                                                     \
    const device half* Q [[buffer(0)]],                                         \
    const device half* K [[buffer(1)]],                                         \
    const device half* V [[buffer(2)]],                                         \
    const device int* block_offsets [[buffer(3)]],                              \
    device half* O [[buffer(4)]],                                               \
    constant ScatterAttnParams& params [[buffer(5)]],                           \
    uint simd_lane_id [[thread_index_in_simdgroup]],                            \
    uint simd_group_id [[simdgroup_index_in_threadgroup]],                      \
    uint3 tid [[threadgroup_position_in_grid]])                                 \
{                                                                               \
    (void)simd_lane_id;                                                         \
    scatter_sdpa_nax_impl<half, _BQ, _BK, _BD, _WM, _WN>(                     \
        Q, K, V, block_offsets, O, params, simd_group_id, tid);                \
}

// BQ=32, 2 SG — production config (dispatched by ScatterSDPA::eval_gpu)
SCATTER_SDPA_KERNEL(bq32_bk32_wm2,  32, 32, 64, 2, 1)

// ---------------------------------------------------------------------------
// Sequential-scan attention: K/V contiguous from offset 0, no block_offsets.
// Q reads from device memory (L2 cached). No threadgroup memory needed.
// Used when KV cache layout guarantees contiguous active slots.
// ---------------------------------------------------------------------------

struct SeqAttnParams {
    uint N_Q_HEADS;
    uint N_KV_HEADS;
    uint T_Q;
    uint D_HEAD;
    uint CAPACITY;        // stride between KV heads in memory
    uint NUM_KV_TOKENS;   // how many tokens to attend to (contiguous from 0)
    float scale;
};

// Sequential-scan attention with Q staged in threadgroup memory.
// Q loaded once from device → TG memory, reused across all KV blocks.
// K/V read directly from device (sequential scan, no block_offsets).
//
// BQ=32, BD=64: Q_smem = 32×64 halfs = 4KB threadgroup memory.
// TG layout: Q_smem[BQ][BD] with compile-time stride BD.

template <
    typename T,
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename AccumType = float>
void scatter_sdpa_seq_staged_impl(
    const device T* Q,
    const device T* K_head,
    const device T* V_head,
    device T* O,
    threadgroup T* Q_smem,
    constant SeqAttnParams& params,
    uint simd_group_id,
    uint simd_lane_id,
    uint3 tgid)
{
    const uint N_Q = params.N_Q_HEADS;
    const uint N_KV = params.N_KV_HEADS;
    const uint T_Q_len = params.T_Q;
    const uint D = params.D_HEAD;
    const uint KV_LEN = params.NUM_KV_TOKENS;
    const uint CAP = params.CAPACITY;

    const uint q_head = tgid.y;
    if (q_head >= N_Q) return;
    const uint q_off = tgid.x * BQ;
    if (q_off >= T_Q_len) return;

    const uint kv_head = q_head / (N_Q / N_KV);

    const device T* Q_dev = Q + q_head * T_Q_len * D + q_off * D;
    O += q_head * T_Q_len * D + q_off * D;
    const device T* K_base = K_head + kv_head * CAP * D;
    const device T* V_base = V_head + kv_head * CAP * D;

    const float scale2 = params.scale * M_LOG2E_F;

    constexpr short kU = 16;
    constexpr int kNWarps = WM * WN;
    constexpr int TQ = BQ / (kNWarps * kU);
    constexpr int TD = BD / kU;
    constexpr short TK = BK / kU;

    const short tm = kU * TQ * simd_group_id;

    const uint n_q = min(q_off + (uint)BQ, T_Q_len) - q_off;
    const short lim_rows_q = n_q - tm;
    const bool is_last_q = (q_off + BQ > T_Q_len);

    // Stage Q into threadgroup memory with pre-scaling.
    // Pre-multiply Q by scale2 so MMA output is already in log2 domain —
    // eliminates the per-block Stile *= scale2 element-wise multiply (256 saves).
    {
        const uint tid = simd_group_id * 32 + simd_lane_id;
        const uint total = BQ * BD;
        const uint stride = kNWarps * 32;
        for (uint i = tid; i < total; i += stride) {
            uint row = i / BD;
            uint col = i % BD;
            float v = (q_off + row < T_Q_len) ? (float)Q_dev[row * D + col] : 0.0f;
            Q_smem[row * BD + col] = (T)(v * scale2);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pointer to this simdgroup's Q rows in threadgroup memory
    const threadgroup T* Q_tg = Q_smem + tm * BD;

    using otile_t = NAXTile<AccumType, TQ, TD>;
    otile_t Otile;
    Otile.clear();

    constexpr short kRowsPT = otile_t::kRowsPerThread;
    metal::vec<AccumType, kRowsPT> max_score;
    metal::vec<AccumType, kRowsPT> sum_score{0};
    for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = Limits<AccumType>::finite_min;
    }

    const uint N_BLK = (KV_LEN + BK - 1) / BK;
    const uint KV_STRIDE = BK * D;
    const device T* K_blk = K_base;
    const device T* V_blk = V_base;
    for (uint blk = 0; blk < N_BLK; blk++, K_blk += KV_STRIDE, V_blk += KV_STRIDE) {

        using stile_t = NAXTile<AccumType, TQ, TK>;
        stile_t Stile;
        Stile.clear();

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
            STEEL_PRAGMA_UNROLL
            for (short ik = 0; ik < TK; ik += 2) {
                STEEL_PRAGMA_UNROLL
                for (short id = 0; id < TD; id++) {
                    NAXTile<T, 1, 1> Qtile;
                    NAXTile<T, 2, 1> Ktile;

                    // Q from threadgroup (compile-time stride BD)
                    Qtile.template load<T, BD, 1>(Q_tg + iq * kU * BD + id * kU);
                    // K from device
                    Ktile.load(K_blk + ik * kU * D + id * kU, D);

                    stile_t::NAXFrag_t::mma(
                        Stile.frag_at(iq, ik),
                        Stile.frag_at(iq, ik + 1),
                        Qtile.frag_at(0, 0),
                        metal::false_type{},
                        Ktile.frag_at(0, 0),
                        Ktile.frag_at(1, 0),
                        metal::true_type{});
                }
            }
        }

        // scale2 baked into Q during TG staging





        metal::vec<AccumType, kRowsPT> new_max;
        metal::vec<AccumType, kRowsPT> factor;
        for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];

        Stile.template row_reduce<MaxOp>(new_max);
        Stile.template row_bin_op<ExpSubOp>(new_max);

        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
            sum_score[i] = sum_score[i] * factor[i];
        }
        Stile.template row_reduce<SumOp>(sum_score);
        Otile.template row_bin_op<MulOp>(factor);

        simdgroup_barrier(mem_flags::mem_none);

        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id += 2) {
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    NAXTile<T, 1, 2> Vtile;
                    Vtile.load(V_blk + ik * kU * D + id * kU, D);

                    otile_t::NAXFrag_t::mma(
                        Otile.frag_at(iq, id),
                        Otile.frag_at(iq, id + 1),
                        Stile.frag_at(iq, ik),
                        metal::false_type{},
                        Vtile.frag_at(0, 0),
                        Vtile.frag_at(0, 1),
                        metal::false_type{});
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_none);
    metal::vec<AccumType, kRowsPT> rcp;
    for (short i = 0; i < kRowsPT; ++i) rcp[i] = 1.f / sum_score[i];
    Otile.template row_bin_op<MulOp>(rcp);

    O += tm * D;
    if (is_last_q) {
        if (lim_rows_q <= 0) return;
        Otile.store_rows(O, D, lim_rows_q);
    } else {
        Otile.store(O, D);
    }
}

// BQ=32, WM=2: 512 TGs, 4KB Q staging
[[kernel, max_total_threads_per_threadgroup(64)]]
void scatter_sdpa_seq_direct_bq32_bk32_wm2(
    const device half* Q [[buffer(0)]],
    const device half* K [[buffer(1)]],
    const device half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant SeqAttnParams& params [[buffer(4)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]])
{
    threadgroup half Q_smem[32 * 64];  // 4KB
    scatter_sdpa_seq_staged_impl<half, 32, 32, 64, 2, 1>(
        Q, K, V, O, Q_smem, params,
        simd_group_id, simd_lane_id, tgid);
}

// ---------------------------------------------------------------------------
// SageAttention-B style: per-block K quant (one scale per BK tokens) and
// per-block V quant. Scales: K_scales [N_KV, N_BLK], V_scales [N_KV, N_BLK].
// Q quantized per-SDPA-block (inside kernel). All scales applied as scalars.
// ---------------------------------------------------------------------------

template <
    int BQ,
    int BK,
    int BD,
    int WM,
    int WN,
    typename AccumType = float>
void scatter_sdpa_seq_int8block_impl(
    const device half* Q,
    const device char* K_head,
    const device half* K_scales_head,   // [N_KV, N_BLK]
    const device char* V_head,
    const device half* V_scales_head,   // [N_KV, N_BLK] (N_KV, N_SCALE_PER_HEAD)
    device half* O,
    threadgroup int8_t* Q_smem,
    threadgroup float* sg_max_smem,
    constant SeqAttnParams& params,
    uint simd_group_id,
    uint simd_lane_id,
    uint3 tgid)
{
    const uint N_Q = params.N_Q_HEADS;
    const uint N_KV = params.N_KV_HEADS;
    const uint T_Q_len = params.T_Q;
    const uint D = params.D_HEAD;
    const uint KV_LEN = params.NUM_KV_TOKENS;
    const uint CAP = params.CAPACITY;
    // Scale arrays are always per-32-token blocks (not per-BK), stride accordingly.
    constexpr uint SCALE_BK = 32;
    const uint N_SCALE_PER_HEAD = CAP / SCALE_BK;

    const uint q_head = tgid.y;
    if (q_head >= N_Q) return;
    const uint q_off = tgid.x * BQ;
    if (q_off >= T_Q_len) return;

    const uint kv_head = q_head / (N_Q / N_KV);

    const device half* Q_dev = Q + q_head * T_Q_len * D + q_off * D;
    O += q_head * T_Q_len * D + q_off * D;
    const device char* K_base = K_head + kv_head * CAP * D;
    const device char* V_base = V_head + kv_head * CAP * D;
    const device half* K_scales_base = K_scales_head + kv_head * N_SCALE_PER_HEAD;
    const device half* V_scales_base = V_scales_head + kv_head * N_SCALE_PER_HEAD;


    const float scale2 = params.scale * M_LOG2E_F;

    constexpr short kU = 16;
    constexpr int kNWarps = WM * WN;
    constexpr int TQ = BQ / (kNWarps * kU);
    constexpr int TD = BD / kU;
    constexpr short TK = BK / kU;

    const short tm = kU * TQ * simd_group_id;
    const uint n_q = min(q_off + (uint)BQ, T_Q_len) - q_off;
    const short lim_rows_q = n_q - tm;
    const bool is_last_q = (q_off + BQ > T_Q_len);

    // Stage Q + compute per-SDPA-block max_abs + quantize to int8
    const uint tid = simd_group_id * 32 + simd_lane_id;
    constexpr uint TOTAL = BQ * BD;
    constexpr uint STRIDE = kNWarps * 32;
    constexpr uint kPerThread = TOTAL / STRIDE;
    float q_vals[kPerThread];
    float thread_max = 0.0f;
    {
        STEEL_PRAGMA_UNROLL
        for (uint i = 0; i < kPerThread; i++) {
            uint idx = tid + i * STRIDE;
            uint row = idx / BD;
            uint col = idx % BD;
            float v = (q_off + row < T_Q_len) ? (float)Q_dev[row * D + col] : 0.0f;
            q_vals[i] = v;
            thread_max = max(thread_max, fabs(v));
        }
    }
    float sg_max = simd_max(thread_max);
    if (simd_lane_id == 0) sg_max_smem[simd_group_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float block_max = max(sg_max_smem[0], sg_max_smem[1]);
    float q_block_scale = (block_max > 0.0f) ? (block_max / 127.0f) : 1.0f;
    float q_inv_scale = (block_max > 0.0f) ? (127.0f / block_max) : 0.0f;
    {
        STEEL_PRAGMA_UNROLL
        for (uint i = 0; i < kPerThread; i++) {
            uint idx = tid + i * STRIDE;
            int qi = (int)rint(q_vals[i] * q_inv_scale);
            qi = clamp(qi, -127, 127);
            Q_smem[idx] = (int8_t)qi;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const threadgroup int8_t* Q_tg = Q_smem + tm * BD;

    using otile_t = NAXTile<AccumType, TQ, TD>;
    otile_t Otile;
    Otile.clear();

    constexpr short kRowsPT = otile_t::kRowsPerThread;
    metal::vec<AccumType, kRowsPT> max_score;
    metal::vec<AccumType, kRowsPT> sum_score{0};
    for (short i = 0; i < kRowsPT; ++i) {
        max_score[i] = Limits<AccumType>::finite_min;
    }

    const uint N_BLK = (KV_LEN + BK - 1) / BK;
    const uint KV_STRIDE = BK * D;
    const device char* K_blk = K_base;
    const device char* V_blk = V_base;
    for (uint blk = 0; blk < N_BLK; blk++,
                                    K_blk += KV_STRIDE,
                                    V_blk += KV_STRIDE) {

        // Per-sub-block scales: BK tokens may span multiple 32-token scale entries.
        // Each TK/2 pair of frags (= 32 K tokens) has its own scale.
        constexpr short SCALE_STRIDE = BK / 32;  // 1 for BK=32, 2 for BK=64

        // MMA + convert in tight scope so compiler frees int32 regs before softmax.
        using stile_t = NAXTile<AccumType, TQ, TK>;
        using FragF = typename stile_t::NAXFrag_t;
        stile_t Stile;
        {
            using stile_int_t = NAXTile<int, TQ, TK>;
            stile_int_t Stile_int;
            Stile_int.clear();

            STEEL_PRAGMA_UNROLL
            for (short iq = 0; iq < TQ; iq++) {
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik += 2) {
                    STEEL_PRAGMA_UNROLL
                    for (short id = 0; id < TD; id++) {
                        NAXTile<int8_t, 1, 1> Qtile;
                        NAXTile<int8_t, 2, 1> Ktile;

                        Qtile.template load<int8_t, BD, 1>(
                            reinterpret_cast<const threadgroup int8_t*>(Q_tg + iq * kU * BD + id * kU));
                        Ktile.load(reinterpret_cast<const device int8_t*>(
                            K_blk + ik * kU * D + id * kU), D);

                        stile_int_t::NAXFrag_t::mma(
                            Stile_int.frag_at(iq, ik),
                            Stile_int.frag_at(iq, ik + 1),
                            Qtile.frag_at(0, 0),
                            metal::false_type{},
                            Ktile.frag_at(0, 0),
                            Ktile.frag_at(1, 0),
                            metal::true_type{});
                    }
                }
            }

            // Convert int32 → float with per-sub-block K scale.
            // Precompute sub-block scales to reduce register pressure in inner loop.
            float k_sub_s[SCALE_STRIDE];
            STEEL_PRAGMA_UNROLL
            for (short ss = 0; ss < SCALE_STRIDE; ss++)
                k_sub_s[ss] = scale2 * q_block_scale *
                    (float)K_scales_base[blk * SCALE_STRIDE + ss];

            STEEL_PRAGMA_UNROLL
            for (short tc = 0; tc < TK; tc++) {
                float s = k_sub_s[tc / 2];
                STEEL_PRAGMA_UNROLL
                for (short iq = 0; iq < TQ; iq++) {
                    thread auto& fd = Stile.frag_at(iq, tc);
                    thread auto& fs = Stile_int.frag_at(iq, tc);
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < FragF::kElemRows * FragF::kElemCols; i++) {
                        fd[i] = (float)fs[i] * s;
                    }
                }
            }
        }  // Stile_int out of scope — registers reclaimable

        // Online softmax (NAXTile ops — compiler-optimized, faster than manual fusion)
        metal::vec<AccumType, kRowsPT> new_max;
        metal::vec<AccumType, kRowsPT> factor;
        for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
        Stile.template row_reduce<MaxOp>(new_max);
        Stile.template row_bin_op<ExpSubOp>(new_max);
        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
            sum_score[i] = sum_score[i] * factor[i];
        }
        Stile.template row_reduce<SumOp>(sum_score);
        Otile.template row_bin_op<MulOp>(factor);

        // Scale P by per-sub-block V scale (if V is int8)
        {
            STEEL_PRAGMA_UNROLL
            for (short tc = 0; tc < TK; tc++) {
                float vs = (float)V_scales_base[blk * SCALE_STRIDE + tc / 2];
                STEEL_PRAGMA_UNROLL
                for (short iq = 0; iq < TQ; iq++) {
                    thread auto& fd = Stile.frag_at(iq, tc);
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < FragF::kElemRows * FragF::kElemCols; i++) {
                        fd[i] *= vs;
                    }
                }
            }
        }

        simdgroup_barrier(mem_flags::mem_none);

        // P @ V fp16 MMA
        STEEL_PRAGMA_UNROLL
        for (short iq = 0; iq < TQ; iq++) {
            STEEL_PRAGMA_UNROLL
            for (short id = 0; id < TD; id += 2) {
                STEEL_PRAGMA_UNROLL
                for (short ik = 0; ik < TK; ik++) {
                    NAXTile<half, 1, 2> Vtile;
                    Vtile.load(V_blk + ik * kU * D + id * kU, D);

                    otile_t::NAXFrag_t::mma(
                        Otile.frag_at(iq, id),
                        Otile.frag_at(iq, id + 1),
                        Stile.frag_at(iq, ik),
                        metal::false_type{},
                        Vtile.frag_at(0, 0),
                        Vtile.frag_at(0, 1),
                        metal::false_type{});
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_none);
    metal::vec<AccumType, kRowsPT> rcp;
    for (short i = 0; i < kRowsPT; ++i) rcp[i] = 1.f / sum_score[i];
    Otile.template row_bin_op<MulOp>(rcp);

    O += tm * D;
    if (is_last_q) {
        if (lim_rows_q <= 0) return;
        Otile.store_rows(O, D, lim_rows_q);
    } else {
        Otile.store(O, D);
    }
}

[[kernel, max_total_threads_per_threadgroup(64)]]
void seq_sdpa_int8block_bq32_bk32_wm2(
    const device half* Q          [[buffer(0)]],
    const device char* K_q        [[buffer(1)]],
    const device half* K_scales   [[buffer(2)]],
    const device char* V_q        [[buffer(3)]],
    const device half* V_scales   [[buffer(4)]],
    device half*       O          [[buffer(5)]],
    constant SeqAttnParams& params [[buffer(6)]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    threadgroup int8_t Q_smem[32 * 64];
    threadgroup float sg_max[2];
    scatter_sdpa_seq_int8block_impl<32, 32, 64, 2, 1>(
        Q, K_q, K_scales, V_q, V_scales, O, Q_smem, sg_max, params,
        simd_group_id, simd_lane_id, tgid);
}

// BK=64 variant: in separate file scatter_sdpa_bk64.metal for
// isolated compilation (register allocation is compiler-context-sensitive).


