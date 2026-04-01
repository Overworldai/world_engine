"""
Benchmark: GEMM kernel timing for W8A8 vs W8A16 vs fp16.

Four columns:
  fp16     — mx.matmul(fp16, fp16)
  W8A16    — mx.quantized_matmul(fp16 act, int8 weights) — no act quant
  W8A8 raw — w8a8_gemm(int8, int8) — pre-quantized, no quant cost
  W8A8 e2e — quantize_symmetric_int8(fp16) + w8a8_gemm — includes quant

Usage:
  python -m src.mlx_metal.benchmarks.bench_gemm
  python -m src.mlx_metal.benchmarks.bench_gemm --shapes sweep --accuracy
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
from we_kernels._ext import w8a8_gemm as w8a8_gemm_raw

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quantize_symmetric_int8(t_fp16):
    """Per-row symmetric int8 quantisation."""
    t_f32 = t_fp16.astype(mx.float32)
    scale = mx.maximum(mx.max(mx.abs(t_f32), axis=-1) / 127.0, 1e-6)
    t_q = mx.clip(mx.round(t_f32 / mx.expand_dims(scale, -1)), -127, 127).astype(mx.int8)
    return t_q, scale


def time_fn(fn, x, warmup=100, runs=500):
    """Chain fn(y) -> y to measure real kernel throughput."""
    y = x
    for _ in range(warmup):
        y = fn(y)
        mx.eval(y)

    y = x
    t0 = time.perf_counter()
    for _ in range(runs):
        y = fn(y)
        mx.eval(y)
    t1 = time.perf_counter()
    return (t1 - t0) / runs * 1e6  # microseconds


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

WORLD_MODEL_SHAPES = [
    (512,  2048, 2048, "qkv/out/ctrl",       168),
    (1,    2048, 2048, "cond/ctrl_c",        168),
    (512,  2048, 8192, "mlp.fc1",             24),
    (512,  8192, 2048, "mlp.fc2",             24),
    (512,  2048, 4096, "out_norm.fc",          1),
    (512,  2048,  128, "unpatchify",           1),
]

SWEEP_SHAPES = [
    (1,    256,  256,  "tiny",                 0),
    (1,    1024, 1024, "1K sq",                0),
    (1,    2048, 2048, "2K sq",                0),
    (1,    4096, 4096, "4K sq",                0),
    (16,   2048, 2048, "batch=16",             0),
    (64,   2048, 2048, "batch=64",             0),
    (256,  2048, 2048, "batch=256",            0),
    (512,  2048, 2048, "batch=512",            0),
    (1,    2048, 8192, "M=1 wide N",           0),
    (1,    8192, 2048, "M=1 wide K",           0),
    (512,  2048, 8192, "M=512 wide N",         0),
    (512,  8192, 2048, "M=512 wide K",         0),
]

MODEL_AND_SWEEP_SHAPES = [
    *SWEEP_SHAPES,
    *WORLD_MODEL_SHAPES,
]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(shapes, warmup, runs):
    hdr = (
        f"{'Layer':>16s}  {'Shape':>18s}  {'#/fwd':>5s}  "
        f"{'fp16':>6s}  {'W8A16':>6s}  {'W8A8':>6s}  {'W8A8 e2e':>8s}  "
        f"{'W8A8/fp16':>9s}  {'e2e/fp16':>8s}  {'e2e/W8A16':>9s}"
    )
    print(hdr)
    print("-" * len(hdr))

    totals = {k: 0.0 for k in ['fp16', 'w8a16', 'w8a8', 'e2e']}

    for M, K, N, label, count in shapes:
        tag = f"{M}x{K}->{N}"

        x_fp16 = mx.random.normal((M, K)).astype(mx.float16)
        w_fp16 = mx.random.normal((N, K)).astype(mx.float16)

        x_q, x_sc = quantize_symmetric_int8(x_fp16)
        w_q, w_sc = quantize_symmetric_int8(w_fp16)
        bias = mx.zeros((N,), dtype=mx.float32)

        w_q_mlx, sc_mlx, bi_mlx = mx.quantize(w_fp16, group_size=64, bits=8)
        mx.eval(x_q, x_sc, w_q, w_sc, bias, w_q_mlx, sc_mlx, bi_mlx)

        y0 = mx.zeros((M, N), dtype=mx.float16)
        mx.eval(y0)

        # fp16 matmul
        t_fp16 = time_fn(
            lambda y: x_fp16 @ w_fp16.T,
            y0, warmup=warmup, runs=runs)

        # W8A16: fp16 activations, int8 weights (no quant needed)
        t_w8a16 = time_fn(
            lambda y: mx.quantized_matmul(
                x_fp16, w_q_mlx, sc_mlx, bi_mlx,
                transpose=True, group_size=64, bits=8),
            y0, warmup=warmup, runs=runs)

        # W8A8 raw: pre-quantized int8 (quant cost excluded)
        t_w8a8 = time_fn(
            lambda y: w8a8_gemm_raw(x_q, w_q, x_sc, w_sc, bias),
            y0, warmup=warmup, runs=runs)

        # W8A8 end-to-end: quantize fp16→int8 + GEMM (quant cost included)
        def w8a8_e2e(y):
            xq, xs = quantize_symmetric_int8(x_fp16)
            return w8a8_gemm_raw(xq, w_q, xs, w_sc, bias)

        t_e2e = time_fn(w8a8_e2e, y0, warmup=warmup, runs=runs)

        print(
            f"{label:>16s}  {tag:>18s}  {count:>5d}  "
            f"{t_fp16:>4.0f}us  {t_w8a16:>4.0f}us  {t_w8a8:>4.0f}us  {t_e2e:>6.0f}us  "
            f"{t_w8a8/t_fp16:>7.2f}x  {t_e2e/t_fp16:>6.2f}x  {t_e2e/t_w8a16:>7.2f}x"
        )

        if count > 0:
            totals['fp16'] += t_fp16 * count
            totals['w8a16'] += t_w8a16 * count
            totals['w8a8'] += t_w8a8 * count
            totals['e2e'] += t_e2e * count

    if totals['fp16'] > 0:
        t = totals
        print()
        print(
            f"{'Forward pass (GEMM only)':>37s}  "
            f"{t['fp16']/1e3:>4.1f}ms  {t['w8a16']/1e3:>4.1f}ms  "
            f"{t['w8a8']/1e3:>4.1f}ms  {t['e2e']/1e3:>6.1f}ms  "
            f"{t['w8a8']/t['fp16']:>7.2f}x  "
            f"{t['e2e']/t['fp16']:>6.2f}x  "
            f"{t['e2e']/t['w8a16']:>7.2f}x"
        )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def run_accuracy(shapes):
    print("\n=== Accuracy (cosine similarity vs fp16 ground truth) ===\n")
    hdr = f"{'Layer':>16s}  {'Shape':>18s}  {'W8A8 cos':>9s}  {'W8A16 cos':>9s}  {'W8A8 maxerr':>11s}"
    print(hdr)
    print("-" * len(hdr))

    for M, K, N, label, _count in shapes:
        tag = f"{M}x{K}->{N}"

        x_fp16 = mx.random.normal((M, K)).astype(mx.float16)
        w_fp16 = mx.random.normal((N, K)).astype(mx.float16)

        x_q, x_sc = quantize_symmetric_int8(x_fp16)
        w_q, w_sc = quantize_symmetric_int8(w_fp16)
        bias = mx.zeros((N,), dtype=mx.float32)

        w_q_mlx, sc_mlx, bi_mlx = mx.quantize(w_fp16, group_size=64, bits=8)
        mx.eval(x_q, x_sc, w_q, w_sc, bias, w_q_mlx, sc_mlx, bi_mlx)

        y_gt = (x_fp16 @ w_fp16.T).astype(mx.float32)
        y_w8a8 = w8a8_gemm_raw(x_q, w_q, x_sc, w_sc, bias).astype(mx.float32)
        y_w8a16 = mx.quantized_matmul(
            x_fp16, w_q_mlx, sc_mlx, bi_mlx,
            transpose=True, group_size=64, bits=8).astype(mx.float32)
        mx.eval(y_gt, y_w8a8, y_w8a16)

        def cos(a, b):
            return (mx.sum(a * b) / (mx.sqrt(mx.sum(a*a)) * mx.sqrt(mx.sum(b*b)) + 1e-8)).item()

        max_err = mx.max(mx.abs(y_w8a8 - y_gt)).item()

        print(
            f"{label:>16s}  {tag:>18s}  "
            f"{cos(y_w8a8, y_gt):>9.6f}  "
            f"{cos(y_w8a16, y_gt):>9.6f}  "
            f"{max_err:>11.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GEMM kernels: W8A8 vs W8A16 vs fp16")
    parser.add_argument("--shapes", choices=["model", "sweep"], default="model")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--accuracy", action="store_true")
    args = parser.parse_args()

    shapes = WORLD_MODEL_SHAPES if args.shapes == "model" else MODEL_AND_SWEEP_SHAPES

    print(f"MLX {mx.__version__} — {mx.default_device()}")
    print(f"warmup={args.warmup}  runs={args.runs}")
    print(f"W8A8     = pre-quantized int8 input (GEMM only)")
    print(f"W8A8 e2e = fp16 input, includes activation quantization\n")

    print("=== Timing ===\n")
    run_benchmark(shapes, warmup=args.warmup, runs=args.runs)

    if args.accuracy:
        run_accuracy(shapes)


if __name__ == "__main__":
    main()
