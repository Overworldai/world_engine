"""
Benchmark: ZeroQuant fused activation quantization kernels.

Compares fused (single Metal kernel) vs separate (MLX ops + Python-side quant)
for both standalone quant kernels and end-to-end quant + GEMM pipelines.

Usage:
  python -m src.mlx_metal.benchmarks.bench_fused_quant
  python -m src.mlx_metal.benchmarks.bench_fused_quant --accuracy
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

from we_kernels import (
    fused_silu_quant,
    fused_rmsnorm_quant,
    fused_rmsnorm_adaln_quant,
    fused_rmsnorm_smooth_quant,
    w8a8_gemm_prequantized,
    w8a8_silu_gemm_nax,
    w8a8_gemm_nax,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quantize_symmetric_int8(t_fp16):
    t_f32 = t_fp16.astype(mx.float32)
    scale = mx.maximum(mx.max(mx.abs(t_f32), axis=-1) / 127.0, 1e-6)
    t_q = mx.clip(mx.round(t_f32 / mx.expand_dims(scale, -1)), -127, 127).astype(mx.int8)
    return t_q, scale


def quantize_weights(N, K):
    w = mx.random.normal((N, K)).astype(mx.float16)
    w_q, w_sc = quantize_symmetric_int8(w)
    mx.eval(w_q, w_sc)
    return w_q, w_sc


def time_fn(fn, warmup=100, runs=500):
    for _ in range(warmup):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    mx.synchronize()
    return (t1 := time.perf_counter()) and (t1 - t0) / runs * 1e6


# ---------------------------------------------------------------------------
# Standalone kernel benchmarks
# ---------------------------------------------------------------------------

STANDALONE_SHAPES = [
    (512, 8192, "SiLU+Quant",  "mlp fc1 out → fc2 input"),
    (512, 2048, "SiLU+Quant",  "ctrl fusion fc2 input"),
    (512, 2048, "RMSNorm+AdaLN+Quant", "pre-QKV / pre-MLP"),
    (1,   2048, "RMSNorm+AdaLN+Quant", "cond (M=1)"),
]


def run_standalone(warmup, runs):
    print("=== Standalone Kernel: Fused vs Separate ===\n")
    hdr = (
        f"{'Op':>25s}  {'Shape':>12s}  {'Label':>22s}  "
        f"{'Separate':>10s}  {'Fused':>10s}  {'Speedup':>8s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for M, K, op, label in STANDALONE_SHAPES:
        tag = f"{M}x{K}"
        x = mx.random.normal((M, K)).astype(mx.float16)
        mx.eval(x)

        if op.startswith("SiLU"):
            def separate():
                xs = nn.silu(x).astype(mx.float32)
                absmax = mx.max(mx.abs(xs), axis=-1)
                sc = mx.maximum(absmax / 127.0, 1e-6)
                xq = mx.clip(mx.round(xs / mx.expand_dims(sc, -1)), -127, 127).astype(mx.int8)
                mx.eval(xq, sc)

            def fused():
                xq, sc = fused_silu_quant(x)
                mx.eval(xq, sc)
        else:
            s = mx.random.normal((K,)).astype(mx.float16)
            b = mx.random.normal((K,)).astype(mx.float16)
            mx.eval(s, b)

            def separate():
                x4 = mx.reshape(x, (1, 1, M, K))
                xn = mx.fast.rms_norm(x4, None, 1e-5) * (1 + s) + b
                xf = xn.astype(mx.float32)
                absmax = mx.max(mx.abs(xf), axis=-1)
                sc = mx.maximum(absmax / 127.0, 1e-6)
                xq = mx.clip(mx.round(xf / mx.expand_dims(sc, -1)), -127, 127).astype(mx.int8)
                mx.eval(xq, sc)

            def fused():
                xq, sc = fused_rmsnorm_adaln_quant(x, s, b, eps=1e-5)
                mx.eval(xq, sc)

        t_sep = time_fn(separate, warmup=warmup, runs=runs)
        t_fused = time_fn(fused, warmup=warmup, runs=runs)
        speedup = t_sep / t_fused

        print(
            f"{op:>25s}  {tag:>12s}  {label:>22s}  "
            f"{t_sep:>8.1f}us  {t_fused:>8.1f}us  {speedup:>6.2f}x"
        )


# ---------------------------------------------------------------------------
# End-to-end: fused quant + GEMM
# ---------------------------------------------------------------------------

E2E_SHAPES = [
    (512, 2048, 6144, "RMSNorm+AdaLN", "QKV proj",   24),
    (512, 2048, 8192, "RMSNorm+AdaLN", "mlp.fc1",    24),
    (512, 8192, 2048, "SiLU",          "mlp.fc2",     24),
    (512, 2048, 2048, "RMSNorm+AdaLN", "out_proj",    24),
]


def run_e2e(warmup, runs):
    print("\n=== End-to-End: Fused Quant + GEMM vs Current Path ===\n")
    hdr = (
        f"{'Label':>12s}  {'Shape':>18s}  {'Fusion':>16s}  {'#/fwd':>5s}  "
        f"{'Current':>10s}  {'Fused':>10s}  {'Speedup':>8s}  {'Saved/fwd':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))

    total_current = 0.0
    total_fused = 0.0

    for M, K, N, fusion_type, label, count in E2E_SHAPES:
        tag = f"{M}x{K}->{N}"
        x = mx.random.normal((M, K)).astype(mx.float16)
        w_q, w_sc = quantize_weights(N, K)
        bias = mx.zeros((N,), dtype=mx.float32)
        mx.eval(x)

        if fusion_type == "SiLU":
            def current():
                xs = nn.silu(x)
                y = w8a8_gemm_nax(xs, w_q, w_scales=w_sc, bias=bias)
                mx.eval(y)

            def fused():
                y = w8a8_silu_gemm_nax(x, w_q, w_scales=w_sc, bias=bias)
                mx.eval(y)
        else:
            s = mx.random.normal((K,)).astype(mx.float16)
            b = mx.random.normal((K,)).astype(mx.float16)
            mx.eval(s, b)

            def current():
                x4 = mx.reshape(x, (1, 1, M, K))
                xn = mx.fast.rms_norm(x4, None, 1e-5) * (1 + s) + b
                xn = mx.reshape(xn, (M, K))
                y = w8a8_gemm_nax(xn, w_q, w_scales=w_sc, bias=bias)
                mx.eval(y)

            def fused():
                xq, xsc = fused_rmsnorm_adaln_quant(x, s, b, eps=1e-5)
                y = w8a8_gemm_prequantized(xq, xsc, w_q, w_scales=w_sc, bias=bias)
                mx.eval(y)

        t_cur = time_fn(current, warmup=warmup, runs=runs)
        t_fused = time_fn(fused, warmup=warmup, runs=runs)
        speedup = t_cur / t_fused
        saved_per_fwd = (t_cur - t_fused) * count

        total_current += t_cur * count
        total_fused += t_fused * count

        print(
            f"{label:>12s}  {tag:>18s}  {fusion_type:>16s}  {count:>5d}  "
            f"{t_cur:>8.1f}us  {t_fused:>8.1f}us  {speedup:>6.2f}x  "
            f"{saved_per_fwd/1e3:>8.2f}ms"
        )

    print()
    saved_total = (total_current - total_fused) / 1e3
    print(
        f"{'Total (GEMM-bound ops)':>37s}  "
        f"{total_current/1e3:>8.2f}ms  {total_fused/1e3:>8.2f}ms  "
        f"{total_current/total_fused:>6.2f}x  {saved_total:>8.2f}ms"
    )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def run_accuracy():
    print("\n=== Accuracy: Fused vs Reference (cosine similarity) ===\n")
    hdr = f"{'Op':>25s}  {'Shape':>12s}  {'Scale diff':>12s}  {'Quant match':>12s}  {'Cos sim':>10s}"
    print(hdr)
    print("-" * len(hdr))

    tests = [
        ("SiLU+Quant", 512, 8192),
        ("SiLU+Quant", 512, 2048),
        ("RMSNorm+Quant", 512, 2048),
        ("RMSNorm+AdaLN+Quant", 512, 2048),
        ("RMSNorm+Smooth+Quant", 512, 2048),
        ("RMSNorm+AdaLN+Smooth+Quant", 512, 2048),
    ]

    for op, M, K in tests:
        x = mx.random.normal((M, K)).astype(mx.float16)
        mx.eval(x)

        if op == "SiLU+Quant":
            fused_q, fused_sc = fused_silu_quant(x)
            # Reference
            ref_v = nn.silu(x).astype(mx.float32)
            ref_absmax = mx.max(mx.abs(ref_v), axis=-1)
            ref_sc = mx.maximum(ref_absmax / 127.0, 1e-6)
            ref_q = mx.clip(mx.round(ref_v / mx.expand_dims(ref_sc, -1)), -127, 127).astype(mx.int8)

        elif op == "RMSNorm+Quant":
            fused_q, fused_sc = fused_rmsnorm_quant(x, eps=1e-5)
            x_f32 = x.astype(mx.float32)
            rms = mx.sqrt(mx.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-5)
            ref_v = (x_f32 / rms)
            ref_absmax = mx.max(mx.abs(ref_v), axis=-1)
            ref_sc = mx.maximum(ref_absmax / 127.0, 1e-6)
            ref_q = mx.clip(mx.round(ref_v / mx.expand_dims(ref_sc, -1)), -127, 127).astype(mx.int8)

        elif op == "RMSNorm+AdaLN+Quant":
            s = mx.random.normal((K,)).astype(mx.float16)
            b = mx.random.normal((K,)).astype(mx.float16)
            mx.eval(s, b)
            fused_q, fused_sc = fused_rmsnorm_adaln_quant(x, s, b, eps=1e-5)
            x_f32 = x.astype(mx.float32)
            rms = mx.sqrt(mx.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-5)
            ref_v = (x_f32 / rms) * (1.0 + s.astype(mx.float32)) + b.astype(mx.float32)
            ref_absmax = mx.max(mx.abs(ref_v), axis=-1)
            ref_sc = mx.maximum(ref_absmax / 127.0, 1e-6)
            ref_q = mx.clip(mx.round(ref_v / mx.expand_dims(ref_sc, -1)), -127, 127).astype(mx.int8)

        elif op == "RMSNorm+Smooth+Quant":
            sm = mx.random.uniform(shape=(K,), low=0.02, high=0.5).astype(mx.float16)
            mx.eval(sm)
            fused_q, fused_sc = fused_rmsnorm_smooth_quant(x, sm, eps=1e-5)
            x_f32 = x.astype(mx.float32)
            rms = mx.sqrt(mx.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-5)
            ref_v = (x_f32 / rms) * sm.astype(mx.float32)
            ref_absmax = mx.max(mx.abs(ref_v), axis=-1)
            ref_sc = mx.maximum(ref_absmax / 127.0, 1e-6)
            ref_q = mx.clip(mx.round(ref_v / mx.expand_dims(ref_sc, -1)), -127, 127).astype(mx.int8)

        else:  # RMSNorm+AdaLN+Smooth+Quant
            s = mx.random.normal((K,)).astype(mx.float16)
            b = mx.random.normal((K,)).astype(mx.float16)
            sm = mx.random.uniform(shape=(K,), low=0.02, high=0.5).astype(mx.float16)
            mx.eval(s, b, sm)
            fused_q, fused_sc = fused_rmsnorm_adaln_quant(x, s, b, eps=1e-5, smooth_scale=sm)
            x_f32 = x.astype(mx.float32)
            rms = mx.sqrt(mx.mean(x_f32 * x_f32, axis=-1, keepdims=True) + 1e-5)
            ref_v = ((x_f32 / rms) * (1.0 + s.astype(mx.float32)) + b.astype(mx.float32)) * sm.astype(mx.float32)
            ref_absmax = mx.max(mx.abs(ref_v), axis=-1)
            ref_sc = mx.maximum(ref_absmax / 127.0, 1e-6)
            ref_q = mx.clip(mx.round(ref_v / mx.expand_dims(ref_sc, -1)), -127, 127).astype(mx.int8)

        mx.eval(fused_q, fused_sc, ref_q, ref_sc)

        scale_diff = mx.max(mx.abs(fused_sc - ref_sc)).item()
        q_match = mx.mean((fused_q == ref_q).astype(mx.float32)).item()

        # Cosine similarity of dequantized outputs
        fused_deq = fused_q.astype(mx.float32) * mx.expand_dims(fused_sc, -1)
        ref_deq = ref_q.astype(mx.float32) * mx.expand_dims(ref_sc, -1)
        cos = (mx.sum(fused_deq * ref_deq) / (
            mx.sqrt(mx.sum(fused_deq * fused_deq)) *
            mx.sqrt(mx.sum(ref_deq * ref_deq)) + 1e-8
        )).item()

        print(
            f"{op:>25s}  {f'{M}x{K}':>12s}  "
            f"{scale_diff:>12.2e}  {q_match*100:>10.2f}%  {cos:>10.6f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ZeroQuant fused activation quantization kernels")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--accuracy", action="store_true")
    args = parser.parse_args()

    print(f"MLX {mx.__version__} — {mx.default_device()}")
    print(f"warmup={args.warmup}  runs={args.runs}\n")

    run_standalone(warmup=args.warmup, runs=args.runs)
    run_e2e(warmup=args.warmup, runs=args.runs)

    if args.accuracy:
        run_accuracy()


if __name__ == "__main__":
    main()
