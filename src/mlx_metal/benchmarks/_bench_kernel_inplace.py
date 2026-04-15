"""Internal helper: bench a kernel in-process after the metallib has been
rebuilt. Saves median/std/p95 to a JSON path for before/after comparison
by scripts that do `stash → bench → edit → rebuild → bench → compare`.

Not part of the public benchmark surface; invoked by
`_bench_loader_swap.sh` only.

Usage:
    python -m src.mlx_metal.benchmarks._bench_kernel_inplace \\
        <kernel_name> <output_json>
where <kernel_name> is one of: fused_quant_upsert, fused_silu_quant,
fused_rmsnorm_adaln_quant, fused_qkv_norm_rope.
"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np


def bench_fn(fn, warmup=10, iters=100):
    for _ in range(warmup):
        out = fn()
        mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(*out) if isinstance(out, (list, tuple)) else mx.eval(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.array(times)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
    }


def bench_fused_quant_upsert():
    from we_kernels import fused_quant_upsert
    mx.random.seed(42)
    N_KV, L, T, D, BK = 6, 4096, 512, 64, 32
    # Queue 96 calls per eval — matches per-frame count in the full model.
    BATCH = 96

    cache_k_q = mx.zeros((1, N_KV, L, D), dtype=mx.int8)
    cache_k_s = mx.zeros((1, N_KV, L // BK), dtype=mx.float16)
    cache_v_q = mx.zeros((1, N_KV, L, D), dtype=mx.int8)
    cache_v_s = mx.zeros((1, N_KV, L // BK), dtype=mx.float16)
    k_news = [mx.random.normal((1, N_KV, T, D)).astype(mx.float16)
              for _ in range(BATCH)]
    v_news = [mx.random.normal((1, N_KV, T, D)).astype(mx.float16)
              for _ in range(BATCH)]
    mx.eval(cache_k_q, cache_k_s, cache_v_q, cache_v_s, *k_news, *v_news)

    def run():
        ckq, cks, cvq, cvs = cache_k_q, cache_k_s, cache_v_q, cache_v_s
        for i in range(BATCH):
            rs = (i * T) % (L - T)
            rs_blk = rs // BK
            ckq, cks, cvq, cvs = fused_quant_upsert(
                k_news[i], v_news[i], ckq, cks, cvq, cvs,
                rs, rs_blk, BK)
        return ckq, cks, cvq, cvs
    return bench_fn(run)


def bench_fused_silu_quant():
    from we_kernels import fused_silu_quant
    mx.random.seed(42)
    M, K = 512, 8192
    BATCH = 48
    xs = [mx.random.normal((M, K)).astype(mx.float16) for _ in range(BATCH)]
    mx.eval(*xs)

    def run():
        outs = [fused_silu_quant(x) for x in xs]
        return [a for pair in outs for a in pair]
    return bench_fn(run)


def bench_fused_rmsnorm_adaln_quant():
    from we_kernels import fused_rmsnorm_adaln_quant
    mx.random.seed(42)
    M, K = 512, 2048
    BATCH = 96
    xs = [mx.random.normal((M, K)).astype(mx.float16) for _ in range(BATCH)]
    ss = [mx.random.normal((K,)).astype(mx.float16) * 0.1 for _ in range(BATCH)]
    bs = [mx.random.normal((K,)).astype(mx.float16) * 0.1 for _ in range(BATCH)]
    mx.eval(*xs, *ss, *bs)

    def run():
        outs = [fused_rmsnorm_adaln_quant(x, s, b)
                for x, s, b in zip(xs, ss, bs)]
        return [a for pair in outs for a in pair]
    return bench_fn(run)


def bench_fused_qkv_norm_rope():
    from we_kernels import fused_qkv_norm_rope
    mx.random.seed(42)
    T, N_Q, N_K, N_V, D_HEAD = 512, 32, 16, 16, 64
    D_ROPE = D_HEAD // 2
    QKV_DIM = (N_Q + N_K + N_V) * D_HEAD
    BATCH = 96
    qkvs = [mx.random.normal((T, QKV_DIM)).astype(mx.float16)
            for _ in range(BATCH)]
    rope_cos = mx.random.normal((T, D_ROPE)).astype(mx.float16)
    rope_sin = mx.random.normal((T, D_ROPE)).astype(mx.float16)
    mx.eval(*qkvs, rope_cos, rope_sin)

    def run():
        outs = [fused_qkv_norm_rope(q, rope_cos, rope_sin, N_Q, N_K, N_V)
                for q in qkvs]
        return [a for triple in outs for a in triple]
    return bench_fn(run)


BENCHES = {
    "fused_quant_upsert": bench_fused_quant_upsert,
    "fused_silu_quant": bench_fused_silu_quant,
    "fused_rmsnorm_adaln_quant": bench_fused_rmsnorm_adaln_quant,
    "fused_qkv_norm_rope": bench_fused_qkv_norm_rope,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kernel", choices=sorted(BENCHES.keys()))
    p.add_argument("output_json")
    args = p.parse_args()

    result = BENCHES[args.kernel]()
    result["kernel"] = args.kernel
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
