#include <metal_stdlib>
using namespace metal;

/*
 Hybrid Metal attention kernel design (inference-only)
 ----------------------------------------------------

 Goals:
 - Forward-only attention for Q, K, V with block/window sparsity driven by
   metadata from Python.
 - Run entirely on Apple GPU (no CPU fallbacks), targeting M-series chips.
 - Serve as a drop-in backend for the world_flex_attn_forward API.

 Tensor layouts (matching AttnMeta):
 - Q: [B, H, T, Dh]  -> flattened as [B*H, T, Dh]
 - K: [B, H, L, Dh]  -> flattened as [B*H, L, Dh]
 - V: [B, H, L, Dh]  -> flattened as [B*H, L, Dh]
 - Output: [B, H, T, Dh] -> [B*H, T, Dh]

 Precision:
 - Inputs in fp16 or bf16; internally promote to fp32 for accumulation.
 - Outputs written in the same dtype as inputs.

 Tiling:
 - Each threadgroup processes a tile of (t_block, kv_block) for a single
   (batch, head) pair:
     - t_block: contiguous query positions in [0, T)
     - kv_block: contiguous key/value positions in [0, L)
 - Within a tile:
     - Load a small Dh chunk into threadgroup memory for Q and K.
     - Compute partial QK^T / sqrt(d) scores.
     - Apply block/window sparsity mask provided via metadata buffer.
     - Accumulate softmax-normalized attention * V to produce output.

 Sparsity metadata:
 - For the first implementation, the Metal kernel will consume a dense
   boolean mask per (t_block, kv_block) encoded as a uint8_t buffer, with:
     mask[b*h*T + t, L] == 1 for valid KV positions, 0 otherwise.
 - Later this can be compressed to a block-list representation that mirrors
   the BlockMask.from_kv_blocks semantics.

 NOTE:
 - The actual math implementation is intentionally left minimal and will be
   iterated on together with the C++/PyTorch custom op bridge.
 */

kernel void metal_flex_attn_forward(
    device const half*         q,
    device const half*         k,
    device const half*         v,
    device const int*          active_blocks, // [active_count] block indices
    device half*               out,
    constant uint&             B,
    constant uint&             Hq,
    constant uint&             T,
    constant uint&             L,
    constant uint&             Dh,
    constant uint&             block_size,
    constant uint&             active_count,
    constant uint&             causal,
    constant uint&             Hkv,
    constant uint&             fp16_accum,
    uint                       tid        [[thread_position_in_grid]],
    uint                       lane_id    [[thread_index_in_simdgroup]],
    uint                       simd_size  [[threads_per_simdgroup]]
) {
    (void)fp16_accum;
    const uint total_queries = B * Hq * T;
    if (simd_size == 0) {
        return;
    }
    const uint qid = tid / simd_size;
    if (qid >= total_queries) {
        return;
    }

    const uint bh = qid / T;
    const uint t = qid % T;
    const uint b = bh / Hq;
    const uint hq = bh % Hq;
    const uint group_size = max((uint)1, Hq / max((uint)1, Hkv));
    const uint hkv = min(hq / group_size, max((uint)0, Hkv - 1));

    const uint q_offset = (((b * Hq + hq) * T + t) * Dh);
    const uint kv_base = (((b * Hkv + hkv) * L) * Dh);
    const uint out_offset = q_offset;
    const float inv_sqrt_dh = (Dh == 64u) ? 0.125f : rsqrt((float)Dh);
    const uint safe_block_size = max((uint)1, block_size);
    const uint kMaxDh = 128;

    if (Dh > kMaxDh) {
        for (uint d = 0; d < Dh; ++d) {
            out[out_offset + d] = half(0.0h);
        }
        return;
    }

    const uint q_start = (L > T) ? (L - T) : 0u;
    const uint kv_limit = (causal != 0) ? min((uint)L, q_start + t + 1u) : (uint)L;
    if (kv_limit == 0) {
        for (uint d = 0; d < Dh; ++d) {
            out[out_offset + d] = half(0.0h);
        }
        return;
    }

    // SIMD-cooperative online softmax:
    // each lane owns a strided subset of Dh and collaborates on dot-product
    // reductions for every KV token.
    float m = -INFINITY;
    float l_acc = 0.0f;
    uint owned_dims[4];
    float q_regs[4];
    float acc[4];
    uint owned_count = 0;
    for (uint d = lane_id; d < Dh; d += simd_size) {
        if (owned_count < 4) {
            owned_dims[owned_count] = d;
            q_regs[owned_count] = (float)q[q_offset + d];
            acc[owned_count] = 0.0f;
            owned_count++;
        }
    }

    // Iterate by block to avoid per-token block-index division and reduce
    // branch pressure when many blocks are masked out.
    for (uint ai = 0; ai < active_count; ++ai) {
        const uint bidx = (uint)active_blocks[ai];
        const uint block_start = bidx * safe_block_size;
        if (block_start >= kv_limit) {
            break;
        }
        const uint block_end = min(kv_limit, block_start + safe_block_size);
        for (uint kv_idx = block_start; kv_idx < block_end; ++kv_idx) {
            float dot_local = 0.0f;
            const uint k_offset = kv_base + kv_idx * Dh;
            for (uint i = 0; i < owned_count; ++i) {
                const uint d = owned_dims[i];
                dot_local += q_regs[i] * (float)k[k_offset + d];
            }
            const float dot = simd_sum(dot_local);
            const float s = dot * inv_sqrt_dh;
            const float m_new = max(m, s);
            const float alpha = fast::exp(m - m_new);
            const float beta = fast::exp(s - m_new);
            const uint v_offset = kv_base + kv_idx * Dh;

            for (uint i = 0; i < owned_count; ++i) {
                const uint d2 = owned_dims[i];
                acc[i] = acc[i] * alpha + beta * (float)v[v_offset + d2];
            }
            l_acc = l_acc * alpha + beta;
            m = m_new;
        }
    }

    if (!(l_acc > 0.0f)) {
        for (uint i = 0; i < owned_count; ++i) {
            out[out_offset + owned_dims[i]] = half(0.0h);
        }
        return;
    }

    const float inv_l = 1.0f / l_acc;
    for (uint i = 0; i < owned_count; ++i) {
        out[out_offset + owned_dims[i]] = half(acc[i] * inv_l);
    }
}

kernel void metal_flex_attn_forward_from_block_written(
    device const half*         q,
    device const half*         k,
    device const half*         v,
    device const uchar*        block_written, // [kv_blocks] 0/1
    device half*               out,
    constant uint&             B,
    constant uint&             Hq,
    constant uint&             T,
    constant uint&             L,
    constant uint&             Dh,
    constant uint&             block_size,
    constant uint&             active_count, // interpreted as kv_blocks in this kernel
    constant uint&             causal,
    constant uint&             Hkv,
    constant uint&             fp16_accum,
    uint                       tid        [[thread_position_in_grid]],
    uint                       lane_id    [[thread_index_in_simdgroup]],
    uint                       simd_size  [[threads_per_simdgroup]]
) {
    (void)fp16_accum;
    const uint total_queries = B * Hq * T;
    if (simd_size == 0) {
        return;
    }
    const uint qid = tid / simd_size;
    if (qid >= total_queries) {
        return;
    }

    const uint bh = qid / T;
    const uint t = qid % T;
    const uint b = bh / Hq;
    const uint hq = bh % Hq;
    const uint group_size = max((uint)1, Hq / max((uint)1, Hkv));
    const uint hkv = min(hq / group_size, max((uint)0, Hkv - 1));

    const uint q_offset = (((b * Hq + hq) * T + t) * Dh);
    const uint kv_base = (((b * Hkv + hkv) * L) * Dh);
    const uint out_offset = q_offset;
    const float inv_sqrt_dh = (Dh == 64u) ? 0.125f : rsqrt((float)Dh);
    const uint safe_block_size = max((uint)1, block_size);
    const uint kv_blocks = active_count;
    const uint kMaxDh = 128;

    if (Dh > kMaxDh) {
        for (uint d = 0; d < Dh; ++d) {
            out[out_offset + d] = half(0.0h);
        }
        return;
    }

    const uint q_start = (L > T) ? (L - T) : 0u;
    const uint kv_limit = (causal != 0) ? min((uint)L, q_start + t + 1u) : (uint)L;
    if (kv_limit == 0) {
        for (uint d = 0; d < Dh; ++d) {
            out[out_offset + d] = half(0.0h);
        }
        return;
    }

    float m = -INFINITY;
    float l_acc = 0.0f;
    uint owned_dims[4];
    float q_regs[4];
    float acc[4];
    uint owned_count = 0;
    for (uint d = lane_id; d < Dh; d += simd_size) {
        if (owned_count < 4) {
            owned_dims[owned_count] = d;
            q_regs[owned_count] = (float)q[q_offset + d];
            acc[owned_count] = 0.0f;
            owned_count++;
        }
    }

    for (uint bidx = 0; bidx < kv_blocks; ++bidx) {
        if (block_written[bidx] == 0) {
            continue;
        }
        const uint block_start = bidx * safe_block_size;
        if (block_start >= kv_limit) {
            break;
        }
        const uint block_end = min(kv_limit, block_start + safe_block_size);
        for (uint kv_idx = block_start; kv_idx < block_end; ++kv_idx) {
            float dot_local = 0.0f;
            const uint k_offset = kv_base + kv_idx * Dh;
            for (uint i = 0; i < owned_count; ++i) {
                const uint d = owned_dims[i];
                dot_local += q_regs[i] * (float)k[k_offset + d];
            }
            const float dot = simd_sum(dot_local);
            const float s = dot * inv_sqrt_dh;
            const float m_new = max(m, s);
            const float alpha = fast::exp(m - m_new);
            const float beta = fast::exp(s - m_new);
            const uint v_offset = kv_base + kv_idx * Dh;

            for (uint i = 0; i < owned_count; ++i) {
                const uint d2 = owned_dims[i];
                acc[i] = acc[i] * alpha + beta * (float)v[v_offset + d2];
            }
            l_acc = l_acc * alpha + beta;
            m = m_new;
        }
    }

    if (!(l_acc > 0.0f)) {
        for (uint i = 0; i < owned_count; ++i) {
            out[out_offset + owned_dims[i]] = half(0.0h);
        }
        return;
    }

    const float inv_l = 1.0f / l_acc;
    for (uint i = 0; i < owned_count; ++i) {
        out[out_offset + owned_dims[i]] = half(acc[i] * inv_l);
    }
}

#if __METAL_VERSION__ >= 310
kernel void metal_flex_attn_forward_bf16(
    device const bfloat*       q,
    device const bfloat*       k,
    device const bfloat*       v,
    device const int*          active_blocks, // [active_count] block indices
    device bfloat*             out,
    constant uint&             B,
    constant uint&             Hq,
    constant uint&             T,
    constant uint&             L,
    constant uint&             Dh,
    constant uint&             block_size,
    constant uint&             active_count,
    constant uint&             causal,
    constant uint&             Hkv,
    constant uint&             fp16_accum,
    uint                       tid        [[thread_position_in_grid]],
    uint                       lane_id    [[thread_index_in_simdgroup]],
    uint                       simd_size  [[threads_per_simdgroup]]
) {
    (void)fp16_accum;
    const uint total_queries = B * Hq * T;
    if (simd_size == 0) {
        return;
    }
    const uint qid = tid / simd_size;
    if (qid >= total_queries) {
        return;
    }

    const uint bh = qid / T;
    const uint t = qid % T;
    const uint b = bh / Hq;
    const uint hq = bh % Hq;
    const uint group_size = max((uint)1, Hq / max((uint)1, Hkv));
    const uint hkv = min(hq / group_size, max((uint)0, Hkv - 1));

    const uint q_offset = (((b * Hq + hq) * T + t) * Dh);
    const uint kv_base = (((b * Hkv + hkv) * L) * Dh);
    const uint out_offset = q_offset;
    const float inv_sqrt_dh = (Dh == 64u) ? 0.125f : rsqrt((float)Dh);
    const uint safe_block_size = max((uint)1, block_size);
    const uint kMaxDh = 128;

    if (Dh > kMaxDh) {
        for (uint d = 0; d < Dh; ++d) {
            out[out_offset + d] = bfloat(0.0f);
        }
        return;
    }

    const uint q_start = (L > T) ? (L - T) : 0u;
    const uint kv_limit = (causal != 0) ? min((uint)L, q_start + t + 1u) : (uint)L;
    if (kv_limit == 0) {
        for (uint d = 0; d < Dh; ++d) {
            out[out_offset + d] = bfloat(0.0f);
        }
        return;
    }

    float m = -INFINITY;
    float l_acc = 0.0f;
    uint owned_dims[4];
    float q_regs[4];
    float acc[4];
    uint owned_count = 0;
    for (uint d = lane_id; d < Dh; d += simd_size) {
        if (owned_count < 4) {
            owned_dims[owned_count] = d;
            q_regs[owned_count] = (float)q[q_offset + d];
            acc[owned_count] = 0.0f;
            owned_count++;
        }
    }

    for (uint ai = 0; ai < active_count; ++ai) {
        const uint bidx = (uint)active_blocks[ai];
        const uint block_start = bidx * safe_block_size;
        if (block_start >= kv_limit) {
            break;
        }
        const uint block_end = min(kv_limit, block_start + safe_block_size);
        for (uint kv_idx = block_start; kv_idx < block_end; ++kv_idx) {
            float dot_local = 0.0f;
            const uint k_offset = kv_base + kv_idx * Dh;
            for (uint i = 0; i < owned_count; ++i) {
                const uint d = owned_dims[i];
                dot_local += q_regs[i] * (float)k[k_offset + d];
            }
            const float dot = simd_sum(dot_local);
            const float s = dot * inv_sqrt_dh;
            const float m_new = max(m, s);
            const float alpha = fast::exp(m - m_new);
            const float beta = fast::exp(s - m_new);
            const uint v_offset = kv_base + kv_idx * Dh;

            for (uint i = 0; i < owned_count; ++i) {
                const uint d2 = owned_dims[i];
                acc[i] = acc[i] * alpha + beta * (float)v[v_offset + d2];
            }
            l_acc = l_acc * alpha + beta;
            m = m_new;
        }
    }

    if (!(l_acc > 0.0f)) {
        for (uint i = 0; i < owned_count; ++i) {
            out[out_offset + owned_dims[i]] = bfloat(0.0f);
        }
        return;
    }

    const float inv_l = 1.0f / l_acc;
    for (uint i = 0; i < owned_count; ++i) {
        out[out_offset + owned_dims[i]] = bfloat(acc[i] * inv_l);
    }
}
#endif

kernel void metal_flex_attn_forward_dh64_bs4_single(
    device const half*         q,
    device const half*         k,
    device const half*         v,
    device const int*          active_blocks,
    device half*               out,
    constant uint&             B,
    constant uint&             Hq,
    constant uint&             T,
    constant uint&             L,
    constant uint&             Dh,
    constant uint&             block_size,
    constant uint&             active_count,
    constant uint&             causal,
    constant uint&             Hkv,
    constant uint&             fp16_accum,
    uint                       tid        [[thread_position_in_grid]],
    uint                       lane_id    [[thread_index_in_simdgroup]]
) {
    if (Dh != 64u || block_size != 4u) {
        return;
    }

    const uint total_queries = B * Hq * T;
    const uint qid = tid >> 5; // /32
    if (qid >= total_queries) {
        return;
    }

    const uint bh = qid / T;
    const uint t = qid % T;
    const uint b = bh / Hq;
    const uint hq = bh % Hq;
    const uint group_size = max((uint)1, Hq / max((uint)1, Hkv));
    const uint hkv = min(hq / group_size, max((uint)0, Hkv - 1));

    const uint q_offset = (((b * Hq + hq) * T + t) * 64u);
    const uint kv_base = (((b * Hkv + hkv) * L) * 64u);
    const uint out_offset = q_offset;
    const float inv_sqrt_dh = 0.125f;
    const uint d_pair = lane_id << 1; // contiguous pair in [0, 62]
    const uint q_start = (L > T) ? (L - T) : 0u;
    const uint kv_limit = (causal != 0) ? min((uint)L, q_start + t + 1u) : (uint)L;
    if (kv_limit == 0u) {
        out[out_offset + d_pair + 0u] = half(0.0h);
        out[out_offset + d_pair + 1u] = half(0.0h);
        return;
    }
    const float2 q2 = float2(
        (float)q[q_offset + d_pair + 0u],
        (float)q[q_offset + d_pair + 1u]
    );

    float m = -INFINITY;
    float l_acc = 0.0f;
    const bool use_fp16_accum = (fp16_accum != 0u);
    float2 acc2 = float2(0.0f);
    half2 acc2_h = half2((half)0.0h);

    for (uint ai = 0; ai < active_count; ++ai) {
        const uint block_start = ((uint)active_blocks[ai]) << 2;
        if (block_start >= kv_limit) {
            break;
        }

        const uint kv0 = block_start + 0u;
        if (kv0 < kv_limit) {
            const uint k0 = kv_base + kv0 * 64u;
            const float2 k20 = float2(
                (float)k[k0 + d_pair + 0u],
                (float)k[k0 + d_pair + 1u]
            );
            const float dot0 = simd_sum(q2.x * k20.x + q2.y * k20.y);
            const float s0 = dot0 * inv_sqrt_dh;
            const float m0 = max(m, s0);
            const float a0 = fast::exp(m - m0);
            const float b0 = fast::exp(s0 - m0);
            const uint v0 = kv_base + kv0 * 64u;
            if (use_fp16_accum) {
                const half2 v20_h = half2(v[v0 + d_pair + 0u], v[v0 + d_pair + 1u]);
                acc2_h = acc2_h * half(a0) + v20_h * half(b0);
            } else {
                const float2 v20 = float2(
                    (float)v[v0 + d_pair + 0u],
                    (float)v[v0 + d_pair + 1u]
                );
                acc2 = acc2 * a0 + v20 * b0;
            }
            l_acc = l_acc * a0 + b0;
            m = m0;
        }

        const uint kv1 = block_start + 1u;
        if (kv1 < kv_limit) {
            const uint k1 = kv_base + kv1 * 64u;
            const float2 k21 = float2(
                (float)k[k1 + d_pair + 0u],
                (float)k[k1 + d_pair + 1u]
            );
            const float dot1 = simd_sum(q2.x * k21.x + q2.y * k21.y);
            const float s1 = dot1 * inv_sqrt_dh;
            const float m1 = max(m, s1);
            const float a1 = fast::exp(m - m1);
            const float b1 = fast::exp(s1 - m1);
            const uint v1 = kv_base + kv1 * 64u;
            if (use_fp16_accum) {
                const half2 v21_h = half2(v[v1 + d_pair + 0u], v[v1 + d_pair + 1u]);
                acc2_h = acc2_h * half(a1) + v21_h * half(b1);
            } else {
                const float2 v21 = float2(
                    (float)v[v1 + d_pair + 0u],
                    (float)v[v1 + d_pair + 1u]
                );
                acc2 = acc2 * a1 + v21 * b1;
            }
            l_acc = l_acc * a1 + b1;
            m = m1;
        }

        const uint kv2 = block_start + 2u;
        if (kv2 < kv_limit) {
            const uint k2 = kv_base + kv2 * 64u;
            const float2 k22 = float2(
                (float)k[k2 + d_pair + 0u],
                (float)k[k2 + d_pair + 1u]
            );
            const float dot2 = simd_sum(q2.x * k22.x + q2.y * k22.y);
            const float s2 = dot2 * inv_sqrt_dh;
            const float m2 = max(m, s2);
            const float a2 = fast::exp(m - m2);
            const float b2 = fast::exp(s2 - m2);
            const uint v2 = kv_base + kv2 * 64u;
            if (use_fp16_accum) {
                const half2 v22_h = half2(v[v2 + d_pair + 0u], v[v2 + d_pair + 1u]);
                acc2_h = acc2_h * half(a2) + v22_h * half(b2);
            } else {
                const float2 v22 = float2(
                    (float)v[v2 + d_pair + 0u],
                    (float)v[v2 + d_pair + 1u]
                );
                acc2 = acc2 * a2 + v22 * b2;
            }
            l_acc = l_acc * a2 + b2;
            m = m2;
        }

        const uint kv3 = block_start + 3u;
        if (kv3 < kv_limit) {
            const uint k3 = kv_base + kv3 * 64u;
            const float2 k23 = float2(
                (float)k[k3 + d_pair + 0u],
                (float)k[k3 + d_pair + 1u]
            );
            const float dot3 = simd_sum(q2.x * k23.x + q2.y * k23.y);
            const float s3 = dot3 * inv_sqrt_dh;
            const float m3 = max(m, s3);
            const float a3 = fast::exp(m - m3);
            const float b3 = fast::exp(s3 - m3);
            const uint v3 = kv_base + kv3 * 64u;
            if (use_fp16_accum) {
                const half2 v23_h = half2(v[v3 + d_pair + 0u], v[v3 + d_pair + 1u]);
                acc2_h = acc2_h * half(a3) + v23_h * half(b3);
            } else {
                const float2 v23 = float2(
                    (float)v[v3 + d_pair + 0u],
                    (float)v[v3 + d_pair + 1u]
                );
                acc2 = acc2 * a3 + v23 * b3;
            }
            l_acc = l_acc * a3 + b3;
            m = m3;
        }
    }

    if (!(l_acc > 0.0f)) {
        out[out_offset + d_pair + 0u] = half(0.0h);
        out[out_offset + d_pair + 1u] = half(0.0h);
        return;
    }
    const float inv_l = 1.0f / l_acc;
    const float2 acc_out = use_fp16_accum ? float2(acc2_h) : acc2;
    out[out_offset + d_pair + 0u] = half(acc_out.x * inv_l);
    out[out_offset + d_pair + 1u] = half(acc_out.y * inv_l);
}

kernel void metal_flex_attn_forward_dh64_bs4_gqa2_single(
    device const half*         q,
    device const half*         k,
    device const half*         v,
    device const int*          active_blocks,
    device half*               out,
    constant uint&             B,
    constant uint&             Hq,
    constant uint&             T,
    constant uint&             L,
    constant uint&             Dh,
    constant uint&             block_size,
    constant uint&             active_count,
    constant uint&             causal,
    constant uint&             Hkv,
    constant uint&             fp16_accum,
    uint                       tid        [[thread_position_in_grid]],
    uint                       lane_id    [[thread_index_in_simdgroup]]
) {
    // Specialization for the common GQA=2 case (Hq = 2 * Hkv).
    if (Dh != 64u || block_size != 4u || Hq != (Hkv << 1)) {
        return;
    }

    const uint total_queries = B * Hq * T;
    const uint qid = tid >> 5; // /32
    if (qid >= total_queries) {
        return;
    }

    const uint bh = qid / T;
    const uint t = qid % T;
    const uint b = bh / Hq;
    const uint hq = bh % Hq;
    const uint hkv = hq >> 1; // exact for GQA=2

    const uint q_offset = (((b * Hq + hq) * T + t) * 64u);
    const uint kv_base = (((b * Hkv + hkv) * L) * 64u);
    const uint out_offset = q_offset;
    const float inv_sqrt_dh = 0.125f;
    const uint d_pair = lane_id << 1;
    const uint q_start = (L > T) ? (L - T) : 0u;
    const uint kv_limit = (causal != 0) ? min((uint)L, q_start + t + 1u) : (uint)L;
    if (kv_limit == 0u) {
        out[out_offset + d_pair + 0u] = half(0.0h);
        out[out_offset + d_pair + 1u] = half(0.0h);
        return;
    }

    const float2 q2 = float2(
        (float)q[q_offset + d_pair + 0u],
        (float)q[q_offset + d_pair + 1u]
    );

    float m = -INFINITY;
    float l_acc = 0.0f;
    const bool use_fp16_accum = (fp16_accum != 0u);
    float2 acc2 = float2(0.0f);
    half2 acc2_h = half2((half)0.0h);

    for (uint ai = 0; ai < active_count; ++ai) {
        const uint block_start = ((uint)active_blocks[ai]) << 2;
        if (block_start >= kv_limit) {
            break;
        }

        const uint kv0 = block_start + 0u;
        if (kv0 < kv_limit) {
            const uint k0 = kv_base + kv0 * 64u;
            const float2 k20 = float2((float)k[k0 + d_pair + 0u], (float)k[k0 + d_pair + 1u]);
            const float dot0 = simd_sum(q2.x * k20.x + q2.y * k20.y);
            const float s0 = dot0 * inv_sqrt_dh;
            const float m0 = max(m, s0);
            const float a0 = fast::exp(m - m0);
            const float b0 = fast::exp(s0 - m0);
            const uint v0 = kv_base + kv0 * 64u;
            if (use_fp16_accum) {
                const half2 v20_h = half2(v[v0 + d_pair + 0u], v[v0 + d_pair + 1u]);
                acc2_h = acc2_h * half(a0) + v20_h * half(b0);
            } else {
                const float2 v20 = float2((float)v[v0 + d_pair + 0u], (float)v[v0 + d_pair + 1u]);
                acc2 = acc2 * a0 + v20 * b0;
            }
            l_acc = l_acc * a0 + b0;
            m = m0;
        }

        const uint kv1 = block_start + 1u;
        if (kv1 < kv_limit) {
            const uint k1 = kv_base + kv1 * 64u;
            const float2 k21 = float2((float)k[k1 + d_pair + 0u], (float)k[k1 + d_pair + 1u]);
            const float dot1 = simd_sum(q2.x * k21.x + q2.y * k21.y);
            const float s1 = dot1 * inv_sqrt_dh;
            const float m1 = max(m, s1);
            const float a1 = fast::exp(m - m1);
            const float b1 = fast::exp(s1 - m1);
            const uint v1 = kv_base + kv1 * 64u;
            if (use_fp16_accum) {
                const half2 v21_h = half2(v[v1 + d_pair + 0u], v[v1 + d_pair + 1u]);
                acc2_h = acc2_h * half(a1) + v21_h * half(b1);
            } else {
                const float2 v21 = float2((float)v[v1 + d_pair + 0u], (float)v[v1 + d_pair + 1u]);
                acc2 = acc2 * a1 + v21 * b1;
            }
            l_acc = l_acc * a1 + b1;
            m = m1;
        }

        const uint kv2 = block_start + 2u;
        if (kv2 < kv_limit) {
            const uint k2 = kv_base + kv2 * 64u;
            const float2 k22 = float2((float)k[k2 + d_pair + 0u], (float)k[k2 + d_pair + 1u]);
            const float dot2 = simd_sum(q2.x * k22.x + q2.y * k22.y);
            const float s2 = dot2 * inv_sqrt_dh;
            const float m2 = max(m, s2);
            const float a2 = fast::exp(m - m2);
            const float b2 = fast::exp(s2 - m2);
            const uint v2 = kv_base + kv2 * 64u;
            if (use_fp16_accum) {
                const half2 v22_h = half2(v[v2 + d_pair + 0u], v[v2 + d_pair + 1u]);
                acc2_h = acc2_h * half(a2) + v22_h * half(b2);
            } else {
                const float2 v22 = float2((float)v[v2 + d_pair + 0u], (float)v[v2 + d_pair + 1u]);
                acc2 = acc2 * a2 + v22 * b2;
            }
            l_acc = l_acc * a2 + b2;
            m = m2;
        }

        const uint kv3 = block_start + 3u;
        if (kv3 < kv_limit) {
            const uint k3 = kv_base + kv3 * 64u;
            const float2 k23 = float2((float)k[k3 + d_pair + 0u], (float)k[k3 + d_pair + 1u]);
            const float dot3 = simd_sum(q2.x * k23.x + q2.y * k23.y);
            const float s3 = dot3 * inv_sqrt_dh;
            const float m3 = max(m, s3);
            const float a3 = fast::exp(m - m3);
            const float b3 = fast::exp(s3 - m3);
            const uint v3 = kv_base + kv3 * 64u;
            if (use_fp16_accum) {
                const half2 v23_h = half2(v[v3 + d_pair + 0u], v[v3 + d_pair + 1u]);
                acc2_h = acc2_h * half(a3) + v23_h * half(b3);
            } else {
                const float2 v23 = float2((float)v[v3 + d_pair + 0u], (float)v[v3 + d_pair + 1u]);
                acc2 = acc2 * a3 + v23 * b3;
            }
            l_acc = l_acc * a3 + b3;
            m = m3;
        }
    }

    if (!(l_acc > 0.0f)) {
        out[out_offset + d_pair + 0u] = half(0.0h);
        out[out_offset + d_pair + 1u] = half(0.0h);
        return;
    }
    const float inv_l = 1.0f / l_acc;
    const float2 acc_out = use_fp16_accum ? float2(acc2_h) : acc2;
    out[out_offset + d_pair + 0u] = half(acc_out.x * inv_l);
    out[out_offset + d_pair + 1u] = half(acc_out.y * inv_l);
}

kernel void metal_flex_attn_forward_dh64_bs4_gqa2_dualhead(
    device const half*         q,
    device const half*         k,
    device const half*         v,
    device const int*          active_blocks,
    device half*               out,
    constant uint&             B,
    constant uint&             Hq,
    constant uint&             T,
    constant uint&             L,
    constant uint&             Dh,
    constant uint&             block_size,
    constant uint&             active_count,
    constant uint&             causal,
    constant uint&             Hkv,
    constant uint&             fp16_accum,
    uint                       tid        [[thread_position_in_grid]],
    uint                       lane_id    [[thread_index_in_simdgroup]]
) {
    // One simdgroup handles a (b, hkv, t) triplet and computes both query heads
    // (2*hkv and 2*hkv+1), reusing each K/V load once.
    if (Dh != 64u || block_size != 4u || Hq != (Hkv << 1)) {
        return;
    }

    const uint total_pairs = B * Hkv * T;
    const uint pid = tid >> 5; // /32
    if (pid >= total_pairs) {
        return;
    }

    const uint bh = pid / T;
    const uint t = pid % T;
    const uint b = bh / Hkv;
    const uint hkv = bh % Hkv;
    const uint hq0 = hkv << 1;
    const uint hq1 = hq0 + 1u;

    const uint q_offset0 = (((b * Hq + hq0) * T + t) * 64u);
    const uint q_offset1 = (((b * Hq + hq1) * T + t) * 64u);
    const uint out_offset0 = q_offset0;
    const uint out_offset1 = q_offset1;
    const uint kv_base = (((b * Hkv + hkv) * L) * 64u);
    const float inv_sqrt_dh = 0.125f;
    const uint d_pair = lane_id << 1;
    const uint q_start = (L > T) ? (L - T) : 0u;
    const uint kv_limit = (causal != 0) ? min((uint)L, q_start + t + 1u) : (uint)L;
    if (kv_limit == 0u || active_count == 0u) {
        out[out_offset0 + d_pair + 0u] = half(0.0h);
        out[out_offset0 + d_pair + 1u] = half(0.0h);
        out[out_offset1 + d_pair + 0u] = half(0.0h);
        out[out_offset1 + d_pair + 1u] = half(0.0h);
        return;
    }

    const float2 q20 = float2(
        (float)q[q_offset0 + d_pair + 0u],
        (float)q[q_offset0 + d_pair + 1u]
    );
    const float2 q21 = float2(
        (float)q[q_offset1 + d_pair + 0u],
        (float)q[q_offset1 + d_pair + 1u]
    );

    const bool use_fp16_accum = (fp16_accum != 0u);
    float m0 = -INFINITY, m1 = -INFINITY;
    float l0 = 0.0f, l1 = 0.0f;
    float2 acc0 = float2(0.0f), acc1 = float2(0.0f);
    half2 acc0_h = half2((half)0.0h), acc1_h = half2((half)0.0h);

    for (uint ai = 0; ai < active_count; ++ai) {
        const uint block_start = ((uint)active_blocks[ai]) << 2;
        if (block_start >= kv_limit) {
            break;
        }
        const uint block_end = min(kv_limit, block_start + 4u);
        for (uint kv_idx = block_start; kv_idx < block_end; ++kv_idx) {
            const uint k_off = kv_base + kv_idx * 64u;
            const float2 k2 = float2((float)k[k_off + d_pair + 0u], (float)k[k_off + d_pair + 1u]);
            const float dot0 = simd_sum(q20.x * k2.x + q20.y * k2.y);
            const float dot1 = simd_sum(q21.x * k2.x + q21.y * k2.y);
            const float s0 = dot0 * inv_sqrt_dh;
            const float s1 = dot1 * inv_sqrt_dh;

            const float m0_new = max(m0, s0);
            const float a0 = fast::exp(m0 - m0_new);
            const float b0 = fast::exp(s0 - m0_new);
            const float m1_new = max(m1, s1);
            const float a1 = fast::exp(m1 - m1_new);
            const float b1 = fast::exp(s1 - m1_new);

            const uint v_off = kv_base + kv_idx * 64u;
            if (use_fp16_accum) {
                const half2 v2_h = half2(v[v_off + d_pair + 0u], v[v_off + d_pair + 1u]);
                acc0_h = acc0_h * half(a0) + v2_h * half(b0);
                acc1_h = acc1_h * half(a1) + v2_h * half(b1);
            } else {
                const float2 v2 = float2((float)v[v_off + d_pair + 0u], (float)v[v_off + d_pair + 1u]);
                acc0 = acc0 * a0 + v2 * b0;
                acc1 = acc1 * a1 + v2 * b1;
            }
            l0 = l0 * a0 + b0;
            l1 = l1 * a1 + b1;
            m0 = m0_new;
            m1 = m1_new;
        }
    }

    if (!(l0 > 0.0f)) {
        out[out_offset0 + d_pair + 0u] = half(0.0h);
        out[out_offset0 + d_pair + 1u] = half(0.0h);
    } else {
        const float inv_l0 = 1.0f / l0;
        const float2 out0 = use_fp16_accum ? float2(acc0_h) : acc0;
        out[out_offset0 + d_pair + 0u] = half(out0.x * inv_l0);
        out[out_offset0 + d_pair + 1u] = half(out0.y * inv_l0);
    }

    if (!(l1 > 0.0f)) {
        out[out_offset1 + d_pair + 0u] = half(0.0h);
        out[out_offset1 + d_pair + 1u] = half(0.0h);
    } else {
        const float inv_l1 = 1.0f / l1;
        const float2 out1 = use_fp16_accum ? float2(acc1_h) : acc1;
        out[out_offset1 + d_pair + 0u] = half(out1.x * inv_l1);
        out[out_offset1 + d_pair + 1u] = half(out1.y * inv_l1);
    }
}

kernel void metal_flex_attn_forward_dh64_bs4_gqa2_dense(
    device const half*         q,
    device const half*         k,
    device const half*         v,
    device const int*          active_blocks,
    device half*               out,
    constant uint&             B,
    constant uint&             Hq,
    constant uint&             T,
    constant uint&             L,
    constant uint&             Dh,
    constant uint&             block_size,
    constant uint&             active_count,
    constant uint&             causal,
    constant uint&             Hkv,
    constant uint&             fp16_accum,
    uint                       tid        [[thread_position_in_grid]],
    uint                       lane_id    [[thread_index_in_simdgroup]]
) {
    (void)active_blocks;
    if (Dh != 64u || block_size != 4u || Hq != (Hkv << 1)) {
        return;
    }

    const uint total_queries = B * Hq * T;
    const uint qid = tid >> 5;
    if (qid >= total_queries) {
        return;
    }

    const uint bh = qid / T;
    const uint t = qid % T;
    const uint b = bh / Hq;
    const uint hq = bh % Hq;
    const uint hkv = hq >> 1;

    const uint q_offset = (((b * Hq + hq) * T + t) * 64u);
    const uint kv_base = (((b * Hkv + hkv) * L) * 64u);
    const uint out_offset = q_offset;
    const float inv_sqrt_dh = 0.125f;
    const uint d_pair = lane_id << 1;
    const uint q_start = (L > T) ? (L - T) : 0u;
    const uint kv_limit = (causal != 0) ? min((uint)L, q_start + t + 1u) : (uint)L;
    if (kv_limit == 0u || active_count == 0u) {
        out[out_offset + d_pair + 0u] = half(0.0h);
        out[out_offset + d_pair + 1u] = half(0.0h);
        return;
    }

    const float2 q2 = float2(
        (float)q[q_offset + d_pair + 0u],
        (float)q[q_offset + d_pair + 1u]
    );

    float m = -INFINITY;
    float l_acc = 0.0f;
    float2 acc2 = float2(0.0f);

    for (uint kv_idx = 0u; kv_idx < kv_limit; ++kv_idx) {
        const uint k_off = kv_base + kv_idx * 64u;
        const float2 k2 = float2((float)k[k_off + d_pair + 0u], (float)k[k_off + d_pair + 1u]);
        const float dot = simd_sum(q2.x * k2.x + q2.y * k2.y);
        const float s = dot * inv_sqrt_dh;
        const float m_new = max(m, s);
        const float a = fast::exp(m - m_new);
        const float bcoef = fast::exp(s - m_new);
        const uint v_off = kv_base + kv_idx * 64u;
        const float2 v2 = float2((float)v[v_off + d_pair + 0u], (float)v[v_off + d_pair + 1u]);
        acc2 = acc2 * a + v2 * bcoef;
        l_acc = l_acc * a + bcoef;
        m = m_new;
    }

    if (!(l_acc > 0.0f)) {
        out[out_offset + d_pair + 0u] = half(0.0h);
        out[out_offset + d_pair + 1u] = half(0.0h);
        return;
    }
    const float inv_l = 1.0f / l_acc;
    out[out_offset + d_pair + 0u] = half(acc2.x * inv_l);
    out[out_offset + d_pair + 1u] = half(acc2.y * inv_l);
}


