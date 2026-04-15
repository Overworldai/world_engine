"""Minimal reproducer for the half4 threadgroup-store race on M5 Max.

FINAL FINDING (narrowed via four probe kernels in
ext/kernels/repro_half4_tg.metal):

  The race reproduces ONLY when Phase 2 of the kernel does BOTH:
    (a) read-modify-write to the threadgroup cache (x_cache), AND
    (b) interleaved device-memory reads from auxiliary buffers (the
        adaln_s / adaln_b per-column parameters in our RMSNorm).

  Removing (a) or (b) makes it go away. Same half4 TG writes in Phase 1,
  same barrier, same TG geometry — variant A/B/C are all clean.

Probe kernels (see `ext/kernels/repro_half4_tg.metal`):
  A `repro_half4_tg`        — pure x → TG → y copy via half4 TG writes.
                              0/30 corrupt.
  B `repro_half4_tg_reduce` — A + sum_sq reduction via simd_sum +
                              sg_reduce TG writes. 0/30 corrupt.
  C `repro_half4_tg_rmw`    — B + Phase 2 read-modify-write to x_cache
                              (x_cache *= rms_inv). 0/30 corrupt.
  D `repro_half4_tg_adaln`  — C + per-column AdaLN device reads
                              (adaln_s[k], adaln_b[k]) in Phase 2.
                              18/30 corrupt, rows clustered in one
                              40-wide dispatch wave (= 40 GPU cores).

The standalone `mx.fast.metal_kernel` JIT path (see `SOURCES` below) also
does NOT reproduce even when ported to the same D-style structure,
which further suggests a metallib-toolchain / precompiled-PSO issue.

Usage:
    uv run python src/mlx_metal/tests/repro_half4_tg_race.py --mode primitive
    uv run python src/mlx_metal/tests/repro_half4_tg_race.py --mode probe
    uv run python src/mlx_metal/tests/repro_half4_tg_race.py --mode standalone
"""
from __future__ import annotations

import argparse
import sys

import mlx.core as mx
import numpy as np


HEADER = """
#include <metal_stdlib>
using namespace metal;

constant constexpr int TG_SIZE = 256;
constant constexpr int MAX_K = 2048;
"""


# Variant A: half4 device load → 4 scalar TG stores + sum_sq reduction
# (matches the RMSNorm Phase 1 structure that fails in production).
SOURCE_SCALAR_WRITES = """
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint K = K_param[0];
    const uint K4 = K / 4;

    const device half4* x_row4 =
        reinterpret_cast<const device half4*>(x_in + row * K);
    device half* y_row = y_out + row * K;
    device float* scales = scales_out + row;

    threadgroup half x_cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Phase 1: half4 device read, 4 scalar TG writes, sum_sq reduction
    // (exact structure of our failing RMSNorm kernel).
    float sum_sq = 0.0f;
    for (uint k4 = tid; k4 < K4; k4 += TG_SIZE) {
        half4 h = x_row4[k4];
        uint k = k4 * 4;
        x_cache[k+0] = h.x;
        x_cache[k+1] = h.y;
        x_cache[k+2] = h.z;
        x_cache[k+3] = h.w;
        float4 v = float4(h);
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    // Simdgroup reduce → threadgroup reduce (matches RMSNorm).
    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            sg_reduce[0] = v;
            *scales = v;  // write scale for verification
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: scalar TG read → device write. Per the Metal spec, the
    // barrier above makes Phase 1's TG writes visible to all threads.
    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = x_cache[k];
    }
"""


# Variant B: explicit half4* cast for the TG write (bypass compiler fusion).
SOURCE_EXPLICIT_HALF4 = """
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint K = K_param[0];
    const uint K4 = K / 4;

    const device half4* x_row4 =
        reinterpret_cast<const device half4*>(x_in + row * K);
    device half* y_row = y_out + row * K;
    device float* scales = scales_out + row;

    threadgroup half x_cache[MAX_K];
    threadgroup half4* x_cache4 =
        reinterpret_cast<threadgroup half4*>(x_cache);
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Phase 1: half4 device read → explicit half4 TG store + sum_sq reduce.
    float sum_sq = 0.0f;
    for (uint k4 = tid; k4 < K4; k4 += TG_SIZE) {
        half4 h = x_row4[k4];
        x_cache4[k4] = h;
        float4 v = float4(h);
        sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
    }

    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            sg_reduce[0] = v;
            *scales = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: scalar TG read → device write.
    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = x_cache[k];
    }
"""


# Known-good scalar baseline: no half4 stores to TG.
SOURCE_SCALAR_BASELINE = """
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint K = K_param[0];

    const device half* x_row = x_in + row * K;
    device half* y_row = y_out + row * K;
    device float* scales = scales_out + row;

    threadgroup half x_cache[MAX_K];
    threadgroup float sg_reduce[TG_SIZE / 32];

    // Phase 1: scalar device read → scalar TG write (no fusion).
    float sum_sq = 0.0f;
    for (uint k = tid; k < K; k += TG_SIZE) {
        float v = (float)x_row[k];
        x_cache[k] = (half)v;
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    uint sgid = tid / 32;
    uint lane = tid % 32;
    if (lane == 0) sg_reduce[sgid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0) {
        float v = (lane < (TG_SIZE / 32)) ? sg_reduce[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            sg_reduce[0] = v;
            *scales = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: scalar TG read → device write.
    for (uint k = tid; k < K; k += TG_SIZE) {
        y_row[k] = x_cache[k];
    }
"""


SOURCES = {
    "scalar_writes": SOURCE_SCALAR_WRITES,
    "explicit_half4": SOURCE_EXPLICIT_HALF4,
    "scalar_baseline": SOURCE_SCALAR_BASELINE,
}


def run_once(variant: str, M: int, K: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the kernel once; return (x_np, y_np). Each call uses a DIFFERENT
    seed so stale TG memory from prior runs won't alias to the current
    input."""
    kernel = mx.fast.metal_kernel(
        name=f"tg_race_{variant}",
        input_names=["x_in", "K_param"],
        output_names=["y_out", "scales_out"],
        source=SOURCES[variant],
        header=HEADER,
    )
    mx.random.seed(seed)
    x = mx.random.normal((M, K)).astype(mx.float16)
    mx.eval(x)
    K_param = mx.array([K], dtype=mx.uint32)
    # MLX's fast.metal_kernel grid is TOTAL THREADS (not threadgroups).
    outputs = kernel(
        inputs=[x, K_param],
        output_shapes=[(M, K), (M,)],
        output_dtypes=[mx.float16, mx.float32],
        grid=(M * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )
    y = outputs[0]
    mx.eval(y)
    return np.array(x), np.array(y)


def run_standalone(variant: str, runs: int, M: int, K: int, primer: bool):
    """JIT path — does NOT currently reproduce the race. Kept as negative
    control."""
    print(f"[standalone] variant={variant}  M={M}  K={K}  runs={runs}")

    if primer:
        for _ in range(50):
            _ = mx.random.normal((64, 64)).astype(mx.float16) * 2
            mx.eval(_)

    bad_runs = 0
    total_bad = 0
    for r in range(runs):
        x, y = run_once(variant, M, K, seed=1000 + r)
        bad = int((x != y).sum())
        total_bad += bad
        if bad > 0:
            bad_runs += 1
            bad_rows = np.where((x != y).any(axis=1))[0]
            print(f"run {r:2d}: {bad} bad elements, "
                  f"#bad rows={len(bad_rows)}")
        else:
            print(f"run {r:2d}: clean")

    print(f"\n[standalone] {bad_runs}/{runs} runs had corruption, "
          f"total={total_bad}/{M*K*runs} elements")


def run_probe(runs: int, M: int, K: int, primer: bool):
    """Sweep the four probe kernels (A/B/C/D) in ext/kernels/repro_half4_tg.metal
    through our metallib + Primitive path. Shows which structural
    ingredient is needed to trigger the race."""
    from we_kernels import (
        repro_half4_tg,          # A: pure copy
        repro_half4_tg_reduce,   # B: + sum_sq reduction
        repro_half4_tg_rmw,      # C: + Phase 2 RMW
        repro_half4_tg_adaln,    # D: + AdaLN device reads in Phase 2
    )

    if primer:
        mx.random.seed(0)
        for _ in range(100):
            y = (mx.random.normal((256, 256)).astype(mx.float16)
                 @ mx.random.normal((256, 256)).astype(mx.float16))
            mx.eval(y)

    def check_A(r):
        mx.random.seed(1000 + r)
        x = mx.random.normal((M, K)).astype(mx.float16); mx.eval(x)
        y = repro_half4_tg(x); mx.eval(y)
        return int((np.array(x) != np.array(y)).sum())

    def check_B(r):
        mx.random.seed(1000 + r)
        x = mx.random.normal((M, K)).astype(mx.float16); mx.eval(x)
        y, _ = repro_half4_tg_reduce(x); mx.eval(y)
        return int((np.array(x) != np.array(y)).sum())

    def check_C(r):
        mx.random.seed(1000 + r)
        x = mx.random.normal((M, K)).astype(mx.float16); mx.eval(x)
        y, rms = repro_half4_tg_rmw(x); mx.eval(y, rms)
        xn = np.array(x).astype(np.float32); rn = np.array(rms)
        expected = xn * rn[:, None]
        return int((np.abs(np.array(y).astype(np.float32) - expected) > 0.01).sum())

    def check_D(r):
        mx.random.seed(1000 + r)
        x = mx.random.normal((M, K)).astype(mx.float16)
        s = (mx.random.normal((K,)) * 0.1).astype(mx.float16)
        b = (mx.random.normal((K,)) * 0.1).astype(mx.float16)
        mx.eval(x, s, b)
        y, rms = repro_half4_tg_adaln(x, s, b); mx.eval(y, rms)
        xn = np.array(x).astype(np.float32)
        sn = np.array(s).astype(np.float32)
        bn = np.array(b).astype(np.float32)
        rn = np.array(rms)
        expected = (xn * rn[:, None]) * (1 + sn[None, :]) + bn[None, :]
        return int((np.abs(np.array(y).astype(np.float32) - expected) > 0.5).sum())

    for label, check in [("A (pure copy)", check_A),
                         ("B (+sum_sq)", check_B),
                         ("C (+Phase2 RMW)", check_C),
                         ("D (+AdaLN reads)", check_D)]:
        bad_runs = 0
        total_bad = 0
        for r in range(runs):
            bad = check(r)
            total_bad += bad
            if bad > 0:
                bad_runs += 1
        print(f"  {label:20s}  {bad_runs}/{runs} runs corrupt "
              f"(total bad elements: {total_bad})")


def run_primitive(runs: int, M: int, K: int, primer: bool):
    """Reproduce the race via our C++ primitive `fused_rmsnorm_adaln_quant`.
    Requires the kernel source to have vectorized Phase 1. Currently our
    shipping kernel is scalar-Phase-1 (workaround); this path will show
    no corruption until you temporarily re-vectorize Phase 1."""
    from we_kernels import fused_rmsnorm_adaln_quant

    print(f"[primitive] M={M}  K={K}  runs={runs}")
    print("(Requires vectorized Phase 1 in fused_rmsnorm_*_quant.metal)")

    if primer:
        for _ in range(100):
            y = (mx.random.normal((256, 256)).astype(mx.float16)
                 @ mx.random.normal((256, 256)).astype(mx.float16))
            mx.eval(y)

    mx.random.seed(42)
    x = mx.random.normal((M, K)).astype(mx.float16)
    s = (mx.random.normal((K,)) * 0.1).astype(mx.float16)
    b = (mx.random.normal((K,)) * 0.1).astype(mx.float16)
    mx.eval(x, s, b)

    bad_runs = 0
    for r in range(runs):
        fq, fs = fused_rmsnorm_adaln_quant(x, s, b, eps=1e-5)
        mx.eval(fq, fs)
        scales = np.array(fs)
        # Scales on this input should be ~0.02-0.04. >0.05 indicates
        # corrupted absmax from stale TG reads in Phase 2.
        bad = int((scales > 0.05).sum())
        if bad > 0:
            bad_runs += 1
            print(f"run {r:2d}: {bad} corrupt rows, max scale={scales.max():.4f}")
        else:
            print(f"run {r:2d}: clean (max scale={scales.max():.4f})")

    print(f"\n[primitive] {bad_runs}/{runs} runs had corrupted scales")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["standalone", "primitive", "probe"],
                   default="probe",
                   help="probe=sweep A/B/C/D minimal repro kernels (recommended); "
                        "standalone=mx.fast.metal_kernel JIT path (doesn't repro); "
                        "primitive=our fused_rmsnorm_adaln_quant via primitive "
                        "(repros only when kernel source has vectorized Phase 1)")
    p.add_argument("--variant", choices=list(SOURCES.keys()),
                   default="scalar_writes",
                   help="standalone mode only: kernel body variant")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("-M", type=int, default=512, help="# threadgroups (rows)")
    p.add_argument("-K", type=int, default=2048, help="row width in halves")
    p.add_argument("--no-primer", action="store_true",
                   help="Skip GPU pre-warm (primer empirically exposes bug)")
    args = p.parse_args()

    primer = not args.no_primer

    if args.mode == "standalone":
        run_standalone(args.variant, args.runs, args.M, args.K, primer)
    elif args.mode == "probe":
        run_probe(args.runs, args.M, args.K, primer)
    else:
        run_primitive(args.runs, args.M, args.K, primer)


if __name__ == "__main__":
    main()
