// BK=64 int8 SDPA kernel — isolated compilation unit for stable register allocation.
// Separated from scatter_sdpa.metal because Metal's compiler register allocator
// is sensitive to other kernels in the same compilation unit.

#include "mlx/backend/metal/kernels/steel/attn/nax.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/kernels/steel/utils.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;
using namespace mlx::steel;

struct SeqAttnParamsBK64 {
    uint N_Q_HEADS;
    uint N_KV_HEADS;
    uint T_Q;
    uint D_HEAD;
    uint CAPACITY;
    uint NUM_KV_TOKENS;
    float scale;
};

struct MaxOp64 {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct SumOp64 {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct ExpSubOp64 {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};
struct MulOp64 {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};

[[kernel, max_total_threads_per_threadgroup(64)]]
void seq_sdpa_int8block_bk64_isolated(
    const device half* Q          [[buffer(0)]],
    const device char* K_q        [[buffer(1)]],
    const device half* K_scales   [[buffer(2)]],
    const device char* V_q        [[buffer(3)]],
    const device half* V_scales   [[buffer(4)]],
    device half*       O          [[buffer(5)]],
    constant SeqAttnParamsBK64& params [[buffer(6)]],
    uint simd_lane_id  [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tgid         [[threadgroup_position_in_grid]])
{
    const uint N_Q = params.N_Q_HEADS;
    const uint N_KV = params.N_KV_HEADS;
    const uint T_Q_len = params.T_Q;
    const uint D = params.D_HEAD;
    const uint KV_LEN = params.NUM_KV_TOKENS;
    const uint CAP = params.CAPACITY;
    constexpr uint SCALE_BK = 32;
    const uint N_SCALE_PER_HEAD = CAP / SCALE_BK;

    const uint q_head = tgid.y;
    if (q_head >= N_Q) return;
    const uint q_off = tgid.x * 32;
    if (q_off >= T_Q_len) return;
    const uint kv_head = q_head / (N_Q / N_KV);

    const device half* Q_dev = Q + q_head * T_Q_len * D + q_off * D;
    O += q_head * T_Q_len * D + q_off * D;
    const device char* K_base = K_q + kv_head * CAP * D;
    const device char* V_base = V_q + kv_head * CAP * D;
    const device half* K_scales_base = K_scales + kv_head * N_SCALE_PER_HEAD;
    const device half* V_scales_base = V_scales + kv_head * N_SCALE_PER_HEAD;

    const float scale2 = params.scale * M_LOG2E_F;

    constexpr short kU = 16;
    constexpr int BQ = 32, BK = 64, BD = 64, WM = 2, WN = 1;
    constexpr int kNWarps = WM * WN;
    constexpr int TQ = BQ / (kNWarps * kU);  // 1
    constexpr int TD = BD / kU;               // 4
    constexpr short TK = BK / kU;            // 4
    constexpr short SCALE_STRIDE = BK / 32;  // 2

    const short tm = kU * TQ * simd_group_id;
    const uint n_q = min(q_off + (uint)BQ, T_Q_len) - q_off;
    const short lim_rows_q = n_q - tm;
    const bool is_last_q = (q_off + BQ > T_Q_len);

    // Stage Q + quantize to int8 (per-SDPA-block)
    threadgroup int8_t Q_smem[32 * 64];
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
    float sg_max_val = simd_max(thread_max);
    threadgroup float sg_max_smem[2];
    if (simd_lane_id == 0) sg_max_smem[simd_group_id] = sg_max_val;
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

    using otile_t = NAXTile<float, TQ, TD>;
    otile_t Otile;
    Otile.clear();

    constexpr short kRowsPT = otile_t::kRowsPerThread;
    metal::vec<float, kRowsPT> max_score;
    metal::vec<float, kRowsPT> sum_score{0};
    for (short i = 0; i < kRowsPT; ++i)
        max_score[i] = Limits<float>::finite_min;

    const uint N_BLK = (KV_LEN + BK - 1) / BK;
    const uint KV_STRIDE = BK * D;
    const device char* K_blk = K_base;
    const device char* V_blk = V_base;
    for (uint blk = 0; blk < N_BLK; blk++, K_blk += KV_STRIDE, V_blk += KV_STRIDE) {

        using stile_t = NAXTile<float, TQ, TK>;
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
                    for (short i = 0; i < FragF::kElemRows * FragF::kElemCols; i++)
                        fd[i] = (float)fs[i] * s;
                }
            }
        }

        metal::vec<float, kRowsPT> new_max;
        metal::vec<float, kRowsPT> factor;
        for (short i = 0; i < kRowsPT; ++i) new_max[i] = max_score[i];
        Stile.template row_reduce<MaxOp64>(new_max);
        Stile.template row_bin_op<ExpSubOp64>(new_max);
        for (short i = 0; i < kRowsPT; ++i) {
            factor[i] = fast::exp2(max_score[i] - new_max[i]);
            max_score[i] = new_max[i];
            sum_score[i] = sum_score[i] * factor[i];
        }
        Stile.template row_reduce<SumOp64>(sum_score);
        Otile.template row_bin_op<MulOp64>(factor);

        {
            STEEL_PRAGMA_UNROLL
            for (short tc = 0; tc < TK; tc++) {
                float vs = (float)V_scales_base[blk * SCALE_STRIDE + tc / 2];
                STEEL_PRAGMA_UNROLL
                for (short iq = 0; iq < TQ; iq++) {
                    thread auto& fd = Stile.frag_at(iq, tc);
                    STEEL_PRAGMA_UNROLL
                    for (short i = 0; i < FragF::kElemRows * FragF::kElemCols; i++)
                        fd[i] *= vs;
                }
            }
        }

        simdgroup_barrier(mem_flags::mem_none);

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
    metal::vec<float, kRowsPT> rcp;
    for (short i = 0; i < kRowsPT; ++i) rcp[i] = 1.f / sum_score[i];
    Otile.template row_bin_op<MulOp64>(rcp);

    O += tm * D;
    if (is_last_q) {
        if (lim_rows_q <= 0) return;
        Otile.store_rows(O, D, lim_rows_q);
    } else {
        Otile.store(O, D);
    }
}
