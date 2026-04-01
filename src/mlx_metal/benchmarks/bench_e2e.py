"""
Benchmark: end-to-end operator chains (activation + GEMM) across quantization strategies.

Measures the real cost of each operator chain as it appears in the model forward pass,
including the preceding activation function (RMSNorm+AdaLN or SiLU).

Five columns:
  fp16         — native fp16 ops + fp16 matmul
  W8A16        — native fp16 ops + mx.quantized_matmul (int8 weights, fp16 activations)
  W8A8 python  — native fp16 ops + Python-side int8 quant + NAX GEMM (current path)
  W8A8 fused   — fused Metal kernel (activation+quant) + NAX GEMM (ZeroQuant path)
  W8A8 raw     — pre-quantized int8 input + NAX GEMM (quant cost excluded, lower bound)

Usage:
  python -m src.mlx_metal.benchmarks.bench_e2e
  python -m src.mlx_metal.benchmarks.bench_e2e --accuracy
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

from we_kernels import (
    fused_rmsnorm_adaln_quant,
    fused_silu_quant,
    w8a8_gemm_nax,
    w8a8_gemm_prequantized,
    w8a8_silu_gemm_nax,
)
from we_kernels._ext import w8a8_gemm as w8a8_gemm_raw

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quantize_symmetric_int8(t_fp16):
    t_f32 = t_fp16.astype(mx.float32)
    scale = mx.maximum(mx.max(mx.abs(t_f32), axis=-1) / 127.0, 1e-6)
    t_q = mx.clip(mx.round(t_f32 / mx.expand_dims(scale, -1)), -127, 127).astype(mx.int8)
    return t_q, scale


def time_fn(fn, warmup=100, runs=500):
    for _ in range(warmup):
        fn()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    mx.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / runs * 1e6


# ---------------------------------------------------------------------------
# Operator chains as they appear in the model
# ---------------------------------------------------------------------------

# (M, K, N, chain_type, label, count_per_fwd)
#
# chain_type determines the preceding activation:
#   "rmsnorm_adaln" — RMSNorm + AdaLN(*(1+s)+b) → GEMM
#   "silu"          — SiLU → GEMM

E2E_SHAPES = [
    # Attention block
    (512, 2048, 6144, "rmsnorm_adaln", "QKV proj",     24),
    (512, 2048, 2048, "rmsnorm_adaln", "attn.out_proj", 24),
    # MLP block
    (512, 2048, 8192, "rmsnorm_adaln", "mlp.fc1",      24),
    (512, 8192, 2048, "silu",          "mlp.fc2",       24),
    # Single-occurrence layers
    (512, 2048, 4096, "rmsnorm_adaln", "out_norm.fc",    1),
]


def run_benchmark(shapes, warmup, runs):
    hdr = (
        f"{'Label':>14s}  {'Shape':>18s}  {'Chain':>16s}  {'#/fwd':>5s}  "
        f"{'fp16':>7s}  {'W8A16':>7s}  {'W8A8 py':>8s}  {'W8A8 fused':>10s}  {'W8A8 raw':>9s}  "
        f"{'fused/fp16':>10s}  {'fused/W8A16':>11s}"
    )
    print(hdr)
    print("-" * len(hdr))

    totals = {k: 0.0 for k in ["fp16", "w8a16", "w8a8_py", "w8a8_fused", "w8a8_raw"]}

    for M, K, N, chain_type, label, count in shapes:
        tag = f"{M}x{K}->{N}"

        # --- Shared setup ---
        x = mx.random.normal((M, K)).astype(mx.float16)
        w_fp16 = mx.random.normal((N, K)).astype(mx.float16)
        w_q, w_sc = quantize_symmetric_int8(w_fp16)
        x_q, x_sc = quantize_symmetric_int8(x)  # for raw baseline
        bias = mx.zeros((N,), dtype=mx.float32)
        w_q_mlx, sc_mlx, bi_mlx = mx.quantize(w_fp16, group_size=64, bits=8)
        mx.eval(x, w_fp16, w_q, w_sc, x_q, x_sc, bias, w_q_mlx, sc_mlx, bi_mlx)

        if chain_type == "rmsnorm_adaln":
            s = mx.random.normal((K,)).astype(mx.float16)
            b = mx.random.normal((K,)).astype(mx.float16)
            mx.eval(s, b)

            # fp16: rms_norm + adaln + matmul
            def fn_fp16():
                x4 = mx.reshape(x, (1, 1, M, K))
                xn = mx.fast.rms_norm(x4, None, 1e-5) * (1 + s) + b
                xn = mx.reshape(xn, (M, K))
                y = xn @ w_fp16.T
                mx.eval(y)

            # W8A16: rms_norm + adaln + quantized_matmul
            def fn_w8a16():
                x4 = mx.reshape(x, (1, 1, M, K))
                xn = mx.fast.rms_norm(x4, None, 1e-5) * (1 + s) + b
                xn = mx.reshape(xn, (M, K))
                y = mx.quantized_matmul(xn, w_q_mlx, sc_mlx, bi_mlx, transpose=True, group_size=64, bits=8)
                mx.eval(y)

            # W8A8 python: rms_norm + adaln + python quant + NAX GEMM
            def fn_w8a8_py():
                x4 = mx.reshape(x, (1, 1, M, K))
                xn = mx.fast.rms_norm(x4, None, 1e-5) * (1 + s) + b
                xn = mx.reshape(xn, (M, K))
                y = w8a8_gemm_nax(xn, w_q, w_scales=w_sc, bias=bias)
                mx.eval(y)

            # W8A8 fused: fused rmsnorm+adaln+quant kernel + NAX GEMM
            def fn_w8a8_fused():
                xq, xsc = fused_rmsnorm_adaln_quant(x, s, b, eps=1e-5)
                y = w8a8_gemm_prequantized(xq, xsc, w_q, w_scales=w_sc, bias=bias)
                mx.eval(y)

        else:  # silu
            # fp16: silu + matmul
            def fn_fp16():
                y = nn.silu(x) @ w_fp16.T
                mx.eval(y)

            # W8A16: silu + quantized_matmul
            def fn_w8a16():
                xs = nn.silu(x)
                y = mx.quantized_matmul(xs, w_q_mlx, sc_mlx, bi_mlx, transpose=True, group_size=64, bits=8)
                mx.eval(y)

            # W8A8 python: silu + python quant + NAX GEMM
            def fn_w8a8_py():
                xs = nn.silu(x)
                y = w8a8_gemm_nax(xs, w_q, w_scales=w_sc, bias=bias)
                mx.eval(y)

            # W8A8 fused: fused silu+quant kernel + NAX GEMM
            def fn_w8a8_fused():
                y = w8a8_silu_gemm_nax(x, w_q, w_scales=w_sc, bias=bias)
                mx.eval(y)

        # W8A8 raw: pre-quantized (lower bound, no activation cost)
        def fn_w8a8_raw():
            y = w8a8_gemm_raw(x_q, w_q, x_sc, w_sc, bias)
            mx.eval(y)

        t_fp16 = time_fn(fn_fp16, warmup=warmup, runs=runs)
        t_w8a16 = time_fn(fn_w8a16, warmup=warmup, runs=runs)
        t_py = time_fn(fn_w8a8_py, warmup=warmup, runs=runs)
        t_fused = time_fn(fn_w8a8_fused, warmup=warmup, runs=runs)
        t_raw = time_fn(fn_w8a8_raw, warmup=warmup, runs=runs)

        print(
            f"{label:>14s}  {tag:>18s}  {chain_type:>16s}  {count:>5d}  "
            f"{t_fp16:>5.0f}us  {t_w8a16:>5.0f}us  {t_py:>6.0f}us  "
            f"{t_fused:>8.0f}us  {t_raw:>7.0f}us  "
            f"{t_fused/t_fp16:>8.2f}x  {t_fused/t_w8a16:>9.2f}x"
        )

        if count > 0:
            totals["fp16"] += t_fp16 * count
            totals["w8a16"] += t_w8a16 * count
            totals["w8a8_py"] += t_py * count
            totals["w8a8_fused"] += t_fused * count
            totals["w8a8_raw"] += t_raw * count

    if totals["fp16"] > 0:
        t = totals
        print()
        print(
            f"{'Forward pass total':>41s}  "
            f"{t['fp16']/1e3:>5.1f}ms  {t['w8a16']/1e3:>5.1f}ms  "
            f"{t['w8a8_py']/1e3:>6.1f}ms  {t['w8a8_fused']/1e3:>8.1f}ms  "
            f"{t['w8a8_raw']/1e3:>7.1f}ms  "
            f"{t['w8a8_fused']/t['fp16']:>8.2f}x  "
            f"{t['w8a8_fused']/t['w8a16']:>9.2f}x"
        )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def run_accuracy(shapes):
    print("\n=== Accuracy (cosine similarity vs fp16 ground truth) ===\n")
    hdr = (
        f"{'Label':>14s}  {'Shape':>18s}  "
        f"{'W8A16 cos':>10s}  {'W8A8 py cos':>12s}  {'W8A8 fused cos':>15s}  "
        f"{'fused maxerr':>13s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for M, K, N, chain_type, label, _count in shapes:
        tag = f"{M}x{K}->{N}"

        x = mx.random.normal((M, K)).astype(mx.float16)
        w_fp16 = mx.random.normal((N, K)).astype(mx.float16)
        w_q, w_sc = quantize_symmetric_int8(w_fp16)
        bias = mx.zeros((N,), dtype=mx.float32)
        w_q_mlx, sc_mlx, bi_mlx = mx.quantize(w_fp16, group_size=64, bits=8)
        mx.eval(x, w_fp16, w_q, w_sc, bias, w_q_mlx, sc_mlx, bi_mlx)

        if chain_type == "rmsnorm_adaln":
            s = mx.random.normal((K,)).astype(mx.float16)
            b = mx.random.normal((K,)).astype(mx.float16)
            mx.eval(s, b)

            x4 = mx.reshape(x, (1, 1, M, K))
            xn = mx.fast.rms_norm(x4, None, 1e-5) * (1 + s) + b
            xn = mx.reshape(xn, (M, K))

            y_fp16 = (xn @ w_fp16.T).astype(mx.float32)
            y_w8a16 = mx.quantized_matmul(
                xn, w_q_mlx, sc_mlx, bi_mlx, transpose=True, group_size=64, bits=8
            ).astype(mx.float32)
            y_py = w8a8_gemm_nax(xn, w_q, w_scales=w_sc, bias=bias).astype(mx.float32)

            xq, xsc = fused_rmsnorm_adaln_quant(x, s, b, eps=1e-5)
            y_fused = w8a8_gemm_prequantized(xq, xsc, w_q, w_scales=w_sc, bias=bias).astype(mx.float32)
        else:
            xs = nn.silu(x)

            y_fp16 = (xs @ w_fp16.T).astype(mx.float32)
            y_w8a16 = mx.quantized_matmul(
                xs, w_q_mlx, sc_mlx, bi_mlx, transpose=True, group_size=64, bits=8
            ).astype(mx.float32)
            y_py = w8a8_gemm_nax(xs, w_q, w_scales=w_sc, bias=bias).astype(mx.float32)
            y_fused = w8a8_silu_gemm_nax(x, w_q, w_scales=w_sc, bias=bias).astype(mx.float32)

        mx.eval(y_fp16, y_w8a16, y_py, y_fused)

        def cos(a, b_):
            return (mx.sum(a * b_) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b_ * b_)) + 1e-8)).item()

        max_err = mx.max(mx.abs(y_fused - y_fp16)).item()

        print(
            f"{label:>14s}  {tag:>18s}  "
            f"{cos(y_w8a16, y_fp16):>10.6f}  "
            f"{cos(y_py, y_fp16):>12.6f}  "
            f"{cos(y_fused, y_fp16):>15.6f}  "
            f"{max_err:>13.4f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end operator chains: activation + GEMM")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--accuracy", action="store_true")
    args = parser.parse_args()

    print(f"MLX {mx.__version__} — {mx.default_device()}")
    print(f"warmup={args.warmup}  runs={args.runs}")
    print(f"fp16       = native fp16 activation + fp16 matmul")
    print(f"W8A16      = native fp16 activation + int8-weight matmul")
    print(f"W8A8 py    = native fp16 activation + Python int8 quant + NAX GEMM")
    print(f"W8A8 fused = fused Metal activation+quant kernel + NAX GEMM")
    print(f"W8A8 raw   = pre-quantized int8 + NAX GEMM (lower bound)\n")

    print("=== Timing ===\n")
    run_benchmark(shapes=E2E_SHAPES, warmup=args.warmup, runs=args.runs)

    if args.accuracy:
        run_accuracy(shapes=E2E_SHAPES)


if __name__ == "__main__":
    main()
