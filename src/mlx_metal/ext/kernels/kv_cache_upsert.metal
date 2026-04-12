// In-place KV cache ring buffer upsert.
//
// Copies k_new [N_KV, T, D] into cache [N_KV, L, D] at ring offset rs.
// Single dispatch replaces MLX slice assignment (full-tensor copy).
//
// One thread per token row. Sequential half4 loop over D — matches MLX's
// work-per-thread pattern. Apple Silicon prefetches per-thread linear
// access efficiently; CUDA-style cross-thread coalescing does not apply.
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
