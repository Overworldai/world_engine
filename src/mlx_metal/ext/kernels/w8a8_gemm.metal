// W8A8 NAX GEMM — int8×int8→int32 using MLX Steel NAXTile abstractions
//
// Two A-loading strategies selected at dispatch time:
//   v1 (both-staged):  A and B both staged through threadgroup. Wins for wide N
//                       (N >= 4096) where double-staging amortizes well.
//   v2 (A-direct):     A loads direct from device, B staged. Halves threadgroup
//                       usage, allows larger BK. Wins for square/narrow N shapes.
//
// Key optimizations:
//   - MLX NAXTile with Int<1> compile-time stride for vectorized 4-byte loads
//   - const_for_loop with compile-time indices for static register allocation
//   - Cooperative 128-bit coalesced device->threadgroup loads
//   - int8 threadgroup is 2x denser than fp16, enabling larger BK tiles
//   - Multiple tile specializations selected at runtime based on M, N, K

#include <metal_stdlib>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
using namespace metal;
using namespace mlx::steel;

struct W8A8Params {
    uint M;
    uint N;
    uint K;
};

// ---- Epilogue: scale int32 accumulators by per-row/col scales, store as fp16 ----

template <typename DTile>
inline void store_scaled(
    thread DTile& Dtile,
    device half* dst,
    uint m_base, uint n_base,
    uint M, uint N,
    const device float* x_scales,
    const device float* w_scales,
    const device float* bias)
{
    constexpr short _TM = DTile::kTileRows;
    constexpr short _TN = DTile::kTileCols;
    const short2 sc = BaseNAXFrag::get_coord();

    STEEL_PRAGMA_UNROLL
    for (short fi = 0; fi < _TM; fi++) {
        STEEL_PRAGMA_UNROLL
        for (short fj = 0; fj < _TN; fj++) {
            thread auto& frag = Dtile.frag_at(fi, fj);
            STEEL_PRAGMA_UNROLL
            for (short i = 0; i < 2; i++) {
                uint gm = m_base + sc.y + fi * 16 + i * 8;
                if (gm >= M) continue;
                float x_sc = x_scales[gm];
                STEEL_PRAGMA_UNROLL
                for (short j = 0; j < 4; j++) {
                    uint gn = n_base + sc.x + fj * 16 + j;
                    if (gn >= N) continue;
                    float val = (float)frag[i * 4 + j] * x_sc * w_scales[gn] + bias[gn];
                    dst[gm * N + gn] = (half)val;
                }
            }
        }
    }
}

// ===========================================================================
// V1: Both A and B staged through threadgroup
// ===========================================================================

template <
    short _BM, short _BN, short _BK,
    short _WM, short _WN>
void w8a8_gemm_impl(
    device const int8_t* x_q,
    device const int8_t* w_q,
    device const float*  x_scales,
    device const float*  w_scales,
    device const float*  bias,
    device half*         out,
    constant W8A8Params& params,
    threadgroup int8_t*  As,
    threadgroup int8_t*  Bs,
    uint3 tgid, uint sgid, uint lane)
{
    constexpr short _SM = _BM / _WM;
    constexpr short _SN = _BN / _WN;
    constexpr short _SK = 32;
    constexpr short _TM = _SM / 16;
    constexpr short _TN = _SN / 16;
    constexpr short _TK = _SK / 16;
    constexpr short _A_PAD = 16;
    constexpr short _B_PAD = 16;
    constexpr short _LDA_TG = _BK + _A_PAD;
    constexpr short _LDB_TG = _BK + _B_PAD;
    constexpr short _TG_SIZE = _WM * _WN * 32;

    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    const uint m_base = tgid.y * _BM;
    const uint n_base = tgid.x * _BN;

    if (m_base >= M || n_base >= N) return;

    const short sg_m = sgid / _WN;
    const short sg_n = sgid % _WN;
    const short tm = sg_m * _SM;
    const short tn = sg_n * _SN;

    const bool sg_valid = (m_base + tm < M) && (n_base + tn < N);
    const uint flat = sgid * 32 + lane;

    NAXTile<int, _TM, _TN> Dtile;
    Dtile.clear();

    for (uint k_base = 0; k_base < K; k_base += _BK) {

        // Cooperative coalesced load A [BM, BK] -> threadgroup
        for (uint t = flat; t < uint(_BM) * uint(_BK) / 16u; t += _TG_SIZE) {
            short r = t / (_BK / 16);
            short c = (t % (_BK / 16)) * 16;
            uint gm = m_base + r;
            uint gk = k_base + c;
            threadgroup int8_t* dst = As + r * _LDA_TG + c;
            if (gm < M && gk + 16 <= K) {
                *reinterpret_cast<threadgroup int4*>(dst) =
                    *reinterpret_cast<device const int4*>(x_q + gm * K + gk);
            } else {
                for (short i = 0; i < 16; i++)
                    dst[i] = (gm < M && gk + i < K) ? x_q[gm * K + gk + i] : int8_t(0);
            }
        }

        // Cooperative coalesced load B [BN, BK] -> threadgroup
        for (uint t = flat; t < uint(_BN) * uint(_BK) / 16u; t += _TG_SIZE) {
            short r = t / (_BK / 16);
            short c = (t % (_BK / 16)) * 16;
            uint gn = n_base + r;
            uint gk = k_base + c;
            threadgroup int8_t* dst = Bs + r * _LDB_TG + c;
            if (gn < N && gk + 16 <= K) {
                *reinterpret_cast<threadgroup int4*>(dst) =
                    *reinterpret_cast<device const int4*>(w_q + gn * K + gk);
            } else {
                for (short i = 0; i < 16; i++)
                    dst[i] = (gn < N && gk + i < K) ? w_q[gn * K + gk + i] : int8_t(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_valid) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < _BK; kk += _SK) {
                NAXTile<int8_t, _TM, _TK> Atile;
                Atile.template load<int8_t, _LDA_TG, 1>(
                    As + tm * _LDA_TG + kk);

                NAXTile<int8_t, _TN, _TK> Btile;
                Btile.template load<int8_t, _LDB_TG, 1>(
                    Bs + tn * _LDB_TG + kk);

                tile_matmad_nax(
                    Dtile,
                    Atile, metal::bool_constant<false>{},
                    Btile, metal::bool_constant<true>{});
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sg_valid) {
        store_scaled(Dtile, out,
            m_base + tm, n_base + tn, M, N,
            x_scales, w_scales, bias);
    }
}

// ===========================================================================
// V2: A direct from device, B staged through threadgroup
// ===========================================================================

template <
    short _BM, short _BN, short _BK,
    short _WM, short _WN>
void w8a8_gemm_v2_impl(
    device const int8_t* x_q,
    device const int8_t* w_q,
    device const float*  x_scales,
    device const float*  w_scales,
    device const float*  bias,
    device half*         out,
    constant W8A8Params& params,
    threadgroup int8_t*  Bs,
    uint3 tgid, uint sgid, uint lane)
{
    constexpr short _SM = _BM / _WM;
    constexpr short _SN = _BN / _WN;
    constexpr short _SK = 32;
    constexpr short _TM = _SM / 16;
    constexpr short _TN = _SN / 16;
    constexpr short _TK = _SK / 16;
    constexpr short _B_PAD = 16;
    constexpr short _LDB_TG = _BK + _B_PAD;
    constexpr short _TG_SIZE = _WM * _WN * 32;

    const uint M = params.M;
    const uint N = params.N;
    const uint K = params.K;

    const uint m_base = tgid.y * _BM;
    const uint n_base = tgid.x * _BN;

    if (m_base >= M || n_base >= N) return;

    const short sg_m = sgid / _WN;
    const short sg_n = sgid % _WN;
    const short tm = sg_m * _SM;
    const short tn = sg_n * _SN;

    const bool sg_valid = (m_base + tm < M) && (n_base + tn < N);
    const uint flat = sgid * 32 + lane;

    const short sgp_sm = sg_valid ? short(metal::min(int(M) - int(m_base + tm), int(_SM))) : 0;

    const device int8_t* A_sg = x_q + (m_base + tm) * K;

    NAXTile<int, _TM, _TN> Dtile;
    Dtile.clear();

    for (uint k_base = 0; k_base < K; k_base += _BK) {

        // Cooperative coalesced load B only -> threadgroup
        for (uint t = flat; t < uint(_BN) * uint(_BK) / 16u; t += _TG_SIZE) {
            short r = t / (_BK / 16);
            short c = (t % (_BK / 16)) * 16;
            uint gn = n_base + r;
            uint gk = k_base + c;
            threadgroup int8_t* dst = Bs + r * _LDB_TG + c;
            if (gn < N && gk + 16 <= K) {
                *reinterpret_cast<threadgroup int4*>(dst) =
                    *reinterpret_cast<device const int4*>(w_q + gn * K + gk);
            } else {
                for (short i = 0; i < 16; i++)
                    dst[i] = (gn < N && gk + i < K) ? w_q[gn * K + gk + i] : int8_t(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_valid) {
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < _BK; kk += _SK) {
                // A: load directly from device
                NAXTile<int8_t, _TM, _TK> Atile;
                if (sgp_sm >= _SM) {
                    Atile.load(A_sg + (k_base + kk), (int)K);
                } else {
                    Atile.load_safe(A_sg + (k_base + kk), (int)K,
                        short2(_SK, sgp_sm));
                }

                NAXTile<int8_t, _TN, _TK> Btile;
                Btile.template load<int8_t, _LDB_TG, 1>(
                    Bs + tn * _LDB_TG + kk);

                tile_matmad_nax(
                    Dtile,
                    Atile, metal::bool_constant<false>{},
                    Btile, metal::bool_constant<true>{});
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sg_valid) {
        store_scaled(Dtile, out,
            m_base + tm, n_base + tn, M, N,
            x_scales, w_scales, bias);
    }
}

// ===========================================================================
// Kernel instantiations
// ===========================================================================

#define W8A8_KERNEL(suffix, _BM, _BN, _BK, _WM, _WN)                        \
[[kernel, max_total_threads_per_threadgroup(_WM * _WN * 32)]]                 \
void w8a8_gemm_nax_##suffix(                                                  \
    device const int8_t* x_q      [[buffer(0)]],                              \
    device const int8_t* w_q      [[buffer(1)]],                              \
    device const float*  x_scales [[buffer(2)]],                              \
    device const float*  w_scales [[buffer(3)]],                              \
    device const float*  bias     [[buffer(4)]],                              \
    device half*         out      [[buffer(5)]],                              \
    constant W8A8Params& params   [[buffer(6)]],                              \
    uint3 tgid [[threadgroup_position_in_grid]],                              \
    uint  sgid [[simdgroup_index_in_threadgroup]],                            \
    uint  lane [[thread_index_in_simdgroup]])                                 \
{                                                                             \
    threadgroup int8_t As[_BM * (_BK + 16)];                                  \
    threadgroup int8_t Bs[_BN * (_BK + 16)];                                  \
    w8a8_gemm_impl<_BM, _BN, _BK, _WM, _WN>(                                \
        x_q, w_q, x_scales, w_scales, bias, out, params,                     \
        As, Bs, tgid, sgid, lane);                                            \
}

#define W8A8_V2_KERNEL(suffix, _BM, _BN, _BK, _WM, _WN)                     \
[[kernel, max_total_threads_per_threadgroup(_WM * _WN * 32)]]                 \
void w8a8_gemm_v2_##suffix(                                                   \
    device const int8_t* x_q      [[buffer(0)]],                              \
    device const int8_t* w_q      [[buffer(1)]],                              \
    device const float*  x_scales [[buffer(2)]],                              \
    device const float*  w_scales [[buffer(3)]],                              \
    device const float*  bias     [[buffer(4)]],                              \
    device half*         out      [[buffer(5)]],                              \
    constant W8A8Params& params   [[buffer(6)]],                              \
    uint3 tgid [[threadgroup_position_in_grid]],                              \
    uint  sgid [[simdgroup_index_in_threadgroup]],                            \
    uint  lane [[thread_index_in_simdgroup]])                                 \
{                                                                             \
    threadgroup int8_t Bs[_BN * (_BK + 16)];                                  \
    w8a8_gemm_v2_impl<_BM, _BN, _BK, _WM, _WN>(                             \
        x_q, w_q, x_scales, w_scales, bias, out, params,                     \
        Bs, tgid, sgid, lane);                                                \
}

// V1: Both A+B staged — budget: BM*(BK+16) + BN*(BK+16) <= 32768
W8A8_KERNEL(bm32_bn64_bk128_wm1_wn2,    32,  64, 128, 1, 2)  // small M (16-48)
W8A8_KERNEL(bm64_bn64_bk64_wm2_wn2,     64,  64,  64, 2, 2)
W8A8_KERNEL(bm64_bn64_bk128_wm2_wn2,    64,  64, 128, 2, 2)
W8A8_KERNEL(bm64_bn64_bk192_wm2_wn2,    64,  64, 192, 2, 2)
W8A8_KERNEL(bm64_bn128_bk128_wm2_wn4,   64, 128, 128, 2, 4)
W8A8_KERNEL(bm128_bn128_bk64_wm4_wn4,  128, 128,  64, 4, 4)

// V2: B-only staged — budget: BN*(BK+16) <= 32768
W8A8_V2_KERNEL(bm64_bn64_bk64_wm2_wn2,     64,  64,  64, 2, 2)
W8A8_V2_KERNEL(bm64_bn64_bk256_wm2_wn2,    64,  64, 256, 2, 2)
W8A8_V2_KERNEL(bm64_bn128_bk128_wm2_wn4,   64, 128, 128, 2, 4)
W8A8_V2_KERNEL(bm128_bn128_bk64_wm4_wn4,  128, 128,  64, 4, 4)
