// Custom M=32 NAXFrag for matmul2d_descriptor(32, 32, 16).
//
// Layout (confirmed via probe):
//   A (left, 32×16): ct_a[0..7] = BaseNAXFrag M=16 rows 0-15,
//                     ct_a[8..15] = BaseNAXFrag M=16 rows 16-31.
//   C (output, 32×32): Cn0[0..7] = M-rows 0-15 / N-cols 0-15,
//                       Cn0[8..15] = M-rows 0-15 / N-cols 16-31,
//                       Cn1[0..7] = M-rows 16-31 / N-cols 0-15,
//                       Cn1[8..15] = M-rows 16-31 / N-cols 16-31.
//   Each 8-element group uses BaseNAXFrag layout:
//     kElemRows=2, kElemCols=4, kElemRowsJump=8, get_coord()=(fn, fm).

#pragma once

#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/defines.h"

using namespace metal;

struct M32NAXFrag {
    STEEL_CONST short kFragRows = 32;
    STEEL_CONST short kFragCols = 16;
    STEEL_CONST short kElemsPerFrag = 16;

    template <typename U>
    using dtype_frag_t = metal::array<U, kElemsPerFrag>;

    METAL_FUNC static short2 get_coord() {
        const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
        const short qid = simd_lane_id >> 2;
        const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
        const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
        return short2{fn, fm};
    }

    // Load 32×16 from threadgroup. Two stacked BaseNAXFrag(M=16) halves.
    template <typename T, int StrX>
    METAL_FUNC static void load(
        thread dtype_frag_t<T>& dst,
        const threadgroup T* src,
        short off_x = 0, short off_y = 0)
    {
        const short2 sc = get_coord();
        // First half: M-rows 0-15 (base fm, fm+8)
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                dst[i * 4 + j] = src[(off_x + sc.y + i * 8) * StrX + off_y + sc.x + j];
            }
        }
        // Second half: M-rows 16-31 (base fm+16, fm+24)
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                dst[8 + i * 4 + j] = src[(off_x + 16 + sc.y + i * 8) * StrX + off_y + sc.x + j];
            }
        }
    }

    // Store Cn0 or Cn1 to device memory.
    // Cn0: M-rows {fm, fm+8} (first M=16 half), covering N-cols in two groups.
    //   Elements [0..7]: N-cols off_y + {fn..fn+3}, rows {fm, fm+8}
    //   Elements [8..15]: N-cols off_y + 16 + {fn..fn+3}, rows {fm, fm+8}
    // off_x = M-row offset (0 for Cn0, 16 for Cn1)
    template <typename T>
    METAL_FUNC static void store_half(
        const thread dtype_frag_t<T>& src,
        device T* dst, int ld,
        short off_x, short off_y,
        short src_offset = 0)
    {
        const short2 sc = get_coord();
        // First 8 elements: N-block 0 (off_y + fn)
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                dst[(off_x + sc.y + i * 8) * ld + off_y + sc.x + j] =
                    src[src_offset + i * 4 + j];
            }
        }
        // Second 8 elements: N-block 1 (off_y + 16 + fn)
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 2; i++) {
            STEEL_PRAGMA_UNROLL
            for (short j = 0; j < 4; j++) {
                dst[(off_x + sc.y + i * 8) * ld + off_y + 16 + sc.x + j] =
                    src[src_offset + 8 + i * 4 + j];
            }
        }
    }

    // M=32 MMA: A[32×16] × B[32×16]^T → C[32×32]
    template <typename CType, typename AType, typename BType, bool transpose_b = true>
    METAL_FUNC static void mma(
        thread dtype_frag_t<CType>& Cn0,
        thread dtype_frag_t<CType>& Cn1,
        const thread dtype_frag_t<AType>& A,
        const thread dtype_frag_t<BType>& Bn0,
        const thread dtype_frag_t<BType>& Bn1,
        metal::bool_constant<transpose_b> = {})
    {
        constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
            32, 32, 16, false, transpose_b, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
        mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

        auto ct_a = gemm_op.template get_left_input_cooperative_tensor<AType, BType, CType>();
        auto ct_b = gemm_op.template get_right_input_cooperative_tensor<AType, BType, CType>();
        auto ct_c = gemm_op.template get_destination_cooperative_tensor<
            decltype(ct_a), decltype(ct_b), CType>();

        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) ct_a[i] = A[i];
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
            ct_b[i] = Bn0[i];
            ct_b[kElemsPerFrag + i] = Bn1[i];
        }
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
            ct_c[i] = Cn0[i];
            ct_c[kElemsPerFrag + i] = Cn1[i];
        }

        gemm_op.run(ct_a, ct_b, ct_c);

        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) {
            Cn0[i] = ct_c[i];
            Cn1[i] = ct_c[kElemsPerFrag + i];
        }
    }

    template <typename T>
    METAL_FUNC static void clear(thread dtype_frag_t<T>& frag) {
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < kElemsPerFrag; i++) frag[i] = T(0);
    }
};
