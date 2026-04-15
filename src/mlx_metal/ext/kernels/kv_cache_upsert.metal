// In-place KV cache ring buffer upsert.
//
// Copies k_new [N_KV, T, D] into cache [N_KV, L, D] at ring offset rs.
// Single dispatch replaces MLX slice assignment (full-tensor copy).
//
// One thread per token row. Sequential half4 loop over D.
// A/B benchmarked against cross-lane (adjacent threads → adjacent half4
// addresses) on this kernel: per-thread-seq was 7-31% faster. The
// cross-lane variant was removed after the A/B concluded — see
// APPLE_SILICON_VS_CUDA.md.
//
// Grid: (ceil(T / TG_SIZE), N_KV, 2)   — z=0 for K, z=1 for V
// Group: (TG_SIZE, 1, 1)

#include <metal_stdlib>
using namespace metal;

struct KVCacheUpsertParams {
    uint N_KV;
    uint L;
    uint T;
    uint D;
    uint rs;
};

constant constexpr uint TG_SIZE = 256;

[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void kv_cache_upsert(
    device half*       cache_k  [[buffer(0)]],
    device half*       cache_v  [[buffer(1)]],
    const device half* k_new    [[buffer(2)]],
    const device half* v_new    [[buffer(3)]],
    constant KVCacheUpsertParams& params [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint L = params.L;
    const uint T = params.T;
    const uint D4 = params.D / 4;
    const uint rs = params.rs;

    const uint head = tgid.y;
    if (head >= params.N_KV) return;

    const uint t = tgid.x * TG_SIZE + tid;
    if (t >= T) return;

    const uint is_v = tgid.z;
    const device half4* src = reinterpret_cast<const device half4*>(
        (is_v ? v_new : k_new) + head * T * params.D);
    device half4* dst = reinterpret_cast<device half4*>(
        (is_v ? cache_v : cache_k) + head * L * params.D);

    const uint src_base = t * D4;
    const uint dst_base = (rs + t) * D4;

    for (uint i = 0; i < D4; i++) {
        dst[dst_base + i] = src[src_base + i];
    }
}



// ---------------------------------------------------------------------------
// Fused quantize + upsert: fp16 K/V → per-block int8 quant → write to cache.
// Eliminates separate fused_quant + upsert kernel launches.
// Grid: (T/BK, N_KV, 2)  z=0 for K, z=1 for V
// Group: (64, 1, 1)  — 2 simdgroups per TG
// ---------------------------------------------------------------------------

struct FusedQuantUpsertParams {
    uint N_KV;
    uint L;       // cache token capacity per head
    uint T;       // new tokens per frame
    uint D;       // head dim
    uint rs;      // token offset in cache
    uint L_BLK;   // cache scale capacity per head
    uint rs_BLK;  // scale offset in cache
    uint BK;      // block size (32)
};

[[kernel, max_total_threads_per_threadgroup(64)]]
void fused_quant_upsert(
    const device half* k_new      [[buffer(0)]],   // [1, N_KV, T, D] fp16
    const device half* v_new      [[buffer(1)]],   // [1, N_KV, T, D] fp16
    device char*       cache_k_q  [[buffer(2)]],   // [1, N_KV, L, D] int8
    device half*       cache_k_s  [[buffer(3)]],   // [1, N_KV, L_BLK] fp16
    device char*       cache_v_q  [[buffer(4)]],   // [1, N_KV, L, D] int8
    device half*       cache_v_s  [[buffer(5)]],   // [1, N_KV, L_BLK] fp16
    constant FusedQuantUpsertParams& params [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]],
    uint  sg_id [[simdgroup_index_in_threadgroup]])
{
    const uint L = params.L;
    const uint T = params.T;
    const uint D = params.D;
    const uint BK = params.BK;
    const uint rs = params.rs;

    const uint block_idx = tgid.x;
    const uint head = tgid.y;
    const uint is_v = tgid.z;
    if (head >= params.N_KV) return;

    const uint base_token = block_idx * BK;
    if (base_token >= T) return;

    const uint TG_SZ = 64;
    const uint elems = BK * D;  // 32 * 64 = 2048
    const uint per_thread = elems / TG_SZ;  // 32
    const uint per_thread_4 = per_thread / 4;  // 8 half4s

    // Source: fp16 K or V for this head and block (half4 aligned — D=64 so D%4=0)
    const device half* src = (is_v ? v_new : k_new) + head * T * D + base_token * D;
    const device half4* src4 = reinterpret_cast<const device half4*>(src);

    // half4 vector loads, per-thread-seq: thread tid owns per_thread_4
    // contiguous half4s starting at base_4.
    const uint base_4 = tid * per_thread_4;

    // Pass 1: load fp16 values, track max_abs
    float4 vals4[8];
    float thread_max = 0.0f;
    for (uint i = 0; i < per_thread_4; i++) {
        half4 h = src4[base_4 + i];
        float4 v = float4(h);
        vals4[i] = v;
        float4 a = abs(v);
        thread_max = max(thread_max, max(max(a.x, a.y), max(a.z, a.w)));
    }

    // Reduce max_abs across simdgroup + cross-SG
    float sg_max = simd_max(thread_max);
    threadgroup float sg_maxes[2];
    if (lane == 0) sg_maxes[sg_id] = sg_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float block_max = max(sg_maxes[0], sg_maxes[1]);
    float scale = (block_max > 0.0f) ? (block_max / 127.0f) : 1.0f;
    float inv_scale = (block_max > 0.0f) ? (127.0f / block_max) : 0.0f;

    // Pass 2: quantize and write directly to cache (char4 vector stores)
    device char* dst_q = (is_v ? cache_v_q : cache_k_q) + head * L * D + (rs + base_token) * D;
    device char4* dst_q4 = reinterpret_cast<device char4*>(dst_q);
    for (uint i = 0; i < per_thread_4; i++) {
        float4 scaled = vals4[i] * inv_scale;
        char4 q;
        q.x = (int8_t)clamp(rint(scaled.x), -127.0f, 127.0f);
        q.y = (int8_t)clamp(rint(scaled.y), -127.0f, 127.0f);
        q.z = (int8_t)clamp(rint(scaled.z), -127.0f, 127.0f);
        q.w = (int8_t)clamp(rint(scaled.w), -127.0f, 127.0f);
        dst_q4[base_4 + i] = q;
    }

    // Write per-block scale to cache
    if (tid == 0) {
        device half* dst_s = (is_v ? cache_v_s : cache_k_s) + head * params.L_BLK;
        dst_s[params.rs_BLK + block_idx] = (half)scale;
    }
}

// ---------------------------------------------------------------------------
// Int8 KV upsert with PER-BLOCK scales (SageAttention-style).
// Data: int8 [N_KV, L, D] with T tokens written at rs.
// Scales: fp16 [N_KV, L_BLK] with T_BLK blocks written at rs_BLK.
// Typically L_BLK = L / BK, T_BLK = T / BK, rs_BLK = rs / BK.
// ---------------------------------------------------------------------------

struct KVCacheUpsertInt8BlockParams {
    uint N_KV;
    uint L;       // token capacity per head
    uint T;       // token count per frame
    uint D;       // head dim
    uint rs;      // token offset in cache
    uint L_BLK;   // block-scale capacity per head
    uint T_BLK;   // block-scale count per frame
    uint rs_BLK;  // block-scale offset in cache
};

[[kernel, max_total_threads_per_threadgroup(TG_SIZE)]]
void kv_cache_upsert_int8_block(
    device char*       cache_k_q    [[buffer(0)]],
    device half*       cache_k_scale [[buffer(1)]],
    device char*       cache_v_q    [[buffer(2)]],
    device half*       cache_v_scale [[buffer(3)]],
    const device char* k_new_q      [[buffer(4)]],
    const device half* k_new_scale  [[buffer(5)]],
    const device char* v_new_q      [[buffer(6)]],
    const device half* v_new_scale  [[buffer(7)]],
    constant KVCacheUpsertInt8BlockParams& params [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]])
{
    const uint L = params.L;
    const uint T = params.T;
    const uint D = params.D;
    const uint D4 = D / 4;
    const uint rs = params.rs;

    const uint head = tgid.y;
    if (head >= params.N_KV) return;

    const uint t = tgid.x * TG_SIZE + tid;
    const uint is_v = tgid.z;

    // Copy int8 data (T tokens)
    if (t < T) {
        const device char4* src_q = reinterpret_cast<const device char4*>(
            (is_v ? v_new_q : k_new_q) + head * T * D);
        device char4* dst_q = reinterpret_cast<device char4*>(
            (is_v ? cache_v_q : cache_k_q) + head * L * D);
        uint sb = t * D4;
        uint db = (rs + t) * D4;
        for (uint i = 0; i < D4; i++) {
            dst_q[db + i] = src_q[sb + i];
        }
    }

    // Copy per-block scale values (T_BLK values, one scale per BK tokens)
    if (t < params.T_BLK) {
        const device half* src_s = (is_v ? v_new_scale : k_new_scale) + head * params.T_BLK;
        device half*       dst_s = (is_v ? cache_v_scale : cache_k_scale) + head * params.L_BLK;
        dst_s[params.rs_BLK + t] = src_s[t];
    }
}
