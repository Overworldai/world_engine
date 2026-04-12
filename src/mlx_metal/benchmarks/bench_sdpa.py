"""
Benchmark scatter_sdpa and seq_sdpa kernels with roofline analysis.

Usage:
  python -m src.mlx_metal.benchmarks.bench_sdpa
  python -m src.mlx_metal.benchmarks.bench_sdpa --n-blocks 80
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from we_kernels import scatter_sdpa, seq_sdpa

# Model constants (Waypoint-1.5)
N_HEADS = 32
N_KV_HEADS = 32
T = 512
D_HEAD = 64
BK = 32


def bench(fn, n_iter=100, warmup=10):
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        mx.eval(fn())
        times.append((time.perf_counter() - t0) * 1000)
    return times


def roofline(n_blocks: int, bw_gbps: float = 546.0):
    q_bytes = N_HEADS * T * D_HEAD * 2
    kv_bytes = N_KV_HEADS * n_blocks * BK * D_HEAD * 2 * 2
    o_bytes = N_HEADS * T * D_HEAD * 2
    total_bytes = q_bytes + kv_bytes + o_bytes
    flops_per_block = 2 * 2 * T * BK * D_HEAD
    total_flops = N_HEADS * n_blocks * flops_per_block
    return {
        "total_bytes_mb": total_bytes / 1e6,
        "total_gflops": total_flops / 1e9,
        "arithmetic_intensity": total_flops / total_bytes,
        "mem_bound_floor_ms": total_bytes / (bw_gbps * 1e9) * 1000,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark SDPA kernels")
    parser.add_argument("--n-blocks", type=int, default=None)
    parser.add_argument("--capacity", type=int, default=8192)
    parser.add_argument("--n-iter", type=int, default=100)
    args = parser.parse_args()

    scale = float(D_HEAD ** -0.5)
    capacity = args.capacity

    print(f"SDPA benchmark")
    print(f"  Q: [{N_HEADS}, {T}, {D_HEAD}]  K/V: [{N_KV_HEADS}, {capacity}, {D_HEAD}]")

    Q = mx.random.normal((N_HEADS, T, D_HEAD)).astype(mx.float16)
    K = mx.random.normal((N_KV_HEADS, capacity, D_HEAD)).astype(mx.float16)
    V = mx.random.normal((N_KV_HEADS, capacity, D_HEAD)).astype(mx.float16)
    mx.eval(Q, K, V)

    # --- Roofline at fixed n_blocks ---
    n_blocks_test = args.n_blocks or 64
    roof = roofline(n_blocks_test)
    kv_tokens = n_blocks_test * BK
    print(f"\n--- Roofline (n_blocks={n_blocks_test}, {kv_tokens} KV tokens) ---")
    print(f"  Data: {roof['total_bytes_mb']:.1f}MB  Compute: {roof['total_gflops']:.1f} GFLOP")
    print(f"  AI: {roof['arithmetic_intensity']:.0f} FLOPs/byte  Floor: {roof['mem_bound_floor_ms']:.2f}ms")

    # --- scatter_sdpa vs seq_sdpa ---
    offsets = mx.array(list(range(0, kv_tokens, BK)), dtype=mx.int32)
    mx.eval(offsets)

    print(f"\n--- Kernel comparison (n_blocks={n_blocks_test}) ---")
    for label, fn in [
        ("scatter_sdpa", lambda: scatter_sdpa(Q, K, V, offsets, scale)),
        ("seq_sdpa",     lambda: seq_sdpa(Q, K, V, kv_tokens, scale)),
    ]:
        times = bench(fn, n_iter=args.n_iter)
        med = np.median(times[10:])
        tops = roof['total_gflops'] / med
        eff = tops / 56 * 100
        print(f"  {label:16s}: {med:.3f}ms  {tops:.1f} TOPS  {eff:.0f}% NAX eff")

    # --- n_blocks sweep ---
    if args.n_blocks is None:
        print(f"\n--- n_blocks sweep (seq_sdpa) ---")
        print(f"  {'n_blocks':>8s} {'KV':>6s} {'Median':>8s} {'TOPS':>8s} {'Eff':>6s}")
        for nb in [4, 8, 16, 32, 48, 64, 80, 96, 128, 160]:
            kv = nb * BK
            if kv > capacity:
                break
            times = bench(lambda kv=kv: seq_sdpa(Q, K, V, kv, scale), n_iter=args.n_iter)
            med = np.median(times[10:])
            r = roofline(nb)
            tops = r['total_gflops'] / med
            eff = tops / 56 * 100
            print(f"  {nb:>8d} {kv:>6d} {med:7.3f}ms {tops:7.1f} {eff:5.0f}%")


if __name__ == "__main__":
    main()
