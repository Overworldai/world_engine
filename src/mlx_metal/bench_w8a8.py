"""
Benchmark: raw GEMM kernel timing for W8A8 vs W8A16 vs fp16.

All three paths are timed on pre-quantized inputs — no activation
quantization overhead is included.  This isolates the GEMM kernel itself.

  fp16   — mx.matmul on fp16 inputs (Steel NAX)
  W8A16  — mx.quantized_matmul, 8-bit weights, fp16 activations (Steel QMM NAX)
  W8A8   — we_kernels.w8a8_gemm, int8 weights + int8 activations (MPP int8 MMA)

Usage:
  python -m src.mlx_metal.bench_w8a8
  python -m src.mlx_metal.bench_w8a8 --shapes sweep --accuracy
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
    """Per-row symmetric int8 quantisation (for both weights and activations)."""
    t_f32 = t_fp16.astype(mx.float32)
    scale = mx.maximum(mx.max(mx.abs(t_f32), axis=-1) / 127.0, 1e-6)
    t_q = mx.clip(mx.round(t_f32 / mx.expand_dims(scale, -1)), -127, 127).astype(mx.int8)
    return t_q, scale


def time_fn(fn, x, warmup=100, runs=500):
    """Chain fn(y) → y to measure real kernel throughput, not dispatch overhead."""
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

MODEL_SHAPES = [
    (1,    2048, 2048,  "single attn_out"),
    (1,    2048, 6144,  "single qkv_fused"),
    (1,    2048, 8192,  "single mlp_fc1"),
    (1,    8192, 2048,  "single mlp_fc2"),
    (512,  2048, 2048,  "frame attn_out"),
    (512,  2048, 6144,  "frame qkv_fused"),
    (512,  2048, 8192,  "frame mlp_fc1"),
    (512,  8192, 2048,  "frame mlp_fc2"),
]

SWEEP_SHAPES = [
    (1,    256,  256,   "tiny"),
    (1,    1024, 1024,  "1K sq"),
    (1,    2048, 2048,  "2K sq"),
    (1,    4096, 4096,  "4K sq"),
    (16,   2048, 2048,  "batch=16"),
    (64,   2048, 2048,  "batch=64"),
    (256,  2048, 2048,  "batch=256"),
    (512,  2048, 2048,  "batch=512"),
    *MODEL_SHAPES,
]


# ---------------------------------------------------------------------------
# Timing benchmark
# ---------------------------------------------------------------------------

def run_benchmark(shapes, warmup, runs):
    hdr = (
        f"{'Shape':>36s}  {'fp16':>8s}  {'W8A16':>8s}  "
        f"{'W8A8':>8s}  {'W8A8/fp16':>9s}  {'W8A8/W8A16':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for M, K, N, label in shapes:
        tag = f"{label} ({M}x{K}->{N})"

        x_fp16 = mx.random.normal((M, K)).astype(mx.float16)
        w_fp16 = mx.random.normal((N, K)).astype(mx.float16)

        x_q, x_sc = quantize_symmetric_int8(x_fp16)
        w_q, w_sc = quantize_symmetric_int8(w_fp16)
        bias = mx.zeros((N,), dtype=mx.float32)

        w_q_mlx, sc_mlx, bi_mlx = mx.quantize(w_fp16, group_size=64, bits=8)

        mx.eval(x_q, x_sc, w_q, w_sc, bias, w_q_mlx, sc_mlx, bi_mlx)

        # Seed value for chaining (same shape as output)
        y0 = mx.zeros((M, N), dtype=mx.float16)
        mx.eval(y0)

        t_fp16 = time_fn(
            lambda y: x_fp16 @ w_fp16.T,
            y0, warmup=warmup, runs=runs)

        t_w8a16 = time_fn(
            lambda y: mx.quantized_matmul(
                x_fp16, w_q_mlx, sc_mlx, bi_mlx,
                transpose=True, group_size=64, bits=8),
            y0, warmup=warmup, runs=runs)

        t_w8a8 = time_fn(
            lambda y: w8a8_gemm_raw(x_q, w_q, x_sc, w_sc, bias),
            y0, warmup=warmup, runs=runs)

        print(
            f"{tag:>36s}  {t_fp16:>6.0f}us  {t_w8a16:>6.0f}us  "
            f"{t_w8a8:>6.0f}us  "
            f"{t_w8a8/t_fp16:>7.2f}x  {t_w8a8/t_w8a16:>8.2f}x"
        )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def run_accuracy(shapes):
    print("\n=== Accuracy (cosine similarity vs fp16 ground truth) ===\n")
    hdr = f"{'Shape':>36s}  {'W8A8 cos':>9s}  {'W8A16 cos':>9s}  {'W8A8 maxerr':>11s}"
    print(hdr)
    print("-" * len(hdr))

    for M, K, N, label in shapes:
        tag = f"{label} ({M}x{K}->{N})"

        x_fp16 = mx.random.normal((M, K)).astype(mx.float16)
        w_fp16 = mx.random.normal((N, K)).astype(mx.float16)

        x_q, x_sc = quantize_symmetric_int8(x_fp16)
        w_q, w_sc = quantize_symmetric_int8(w_fp16)
        bias = mx.zeros((N,), dtype=mx.float32)

        w_q_mlx, sc_mlx, bi_mlx = mx.quantize(w_fp16, group_size=64, bits=8)
        mx.eval(x_q, x_sc, w_q, w_sc, bias, w_q_mlx, sc_mlx, bi_mlx)

        # Ground truth: fp16 matmul
        y_gt = (x_fp16 @ w_fp16.T).astype(mx.float32)

        # W8A8: raw kernel output, then apply scales in Python for comparison
        y_w8a8_raw = w8a8_gemm_raw(x_q, w_q, x_sc, w_sc, bias).astype(mx.float32)

        # W8A16
        y_w8a16 = mx.quantized_matmul(
            x_fp16, w_q_mlx, sc_mlx, bi_mlx,
            transpose=True, group_size=64, bits=8).astype(mx.float32)

        mx.eval(y_gt, y_w8a8_raw, y_w8a16)

        def cos(a, b):
            return (mx.sum(a * b) / (mx.sqrt(mx.sum(a*a)) * mx.sqrt(mx.sum(b*b)) + 1e-8)).item()

        max_err = mx.max(mx.abs(y_w8a8_raw - y_gt)).item()

        print(
            f"{tag:>36s}  "
            f"{cos(y_w8a8_raw, y_gt):>9.6f}  "
            f"{cos(y_w8a16, y_gt):>9.6f}  "
            f"{max_err:>11.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark raw GEMM kernels: W8A8 vs W8A16 vs fp16")
    parser.add_argument("--shapes", choices=["model", "sweep"], default="model")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--accuracy", action="store_true")
    args = parser.parse_args()

    shapes = MODEL_SHAPES if args.shapes == "model" else SWEEP_SHAPES

    print(f"MLX {mx.__version__} — {mx.default_device()}")
    print(f"warmup={args.warmup}  runs={args.runs}")
    print(f"Timing: raw kernel only (no activation quantisation overhead)\n")

    print("=== Timing ===\n")
    run_benchmark(shapes, warmup=args.warmup, runs=args.runs)

    if args.accuracy:
        run_accuracy(shapes)


if __name__ == "__main__":
    main()
