"""Benchmark: steady-state model performance with saturated KV cache.

Fills every ring buffer slot before timing, so measurements reflect the
worst-case (maximum attention context) the model will ever see. No VAE,
no encode — pure model throughput at saturation.

Usage:
  python -m src.mlx_metal.benchmarks.bench_steady
  python -m src.mlx_metal.benchmarks.bench_steady --frames 30 --profile fp16
  python -m src.mlx_metal.benchmarks.bench_steady --profile max_qat --warmup 3
  uv run -m src.mlx_metal.benchmarks.bench_steady --frames 30 --profile speed
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from ..mlx_world_model import load_from_pytorch, compute_rope_angles


MODEL_URI = "Overworld-Models/MR160k-smoothquant"


def saturation_frame(cfg) -> int:
    """Frame index at which every KV cache ring slot has been written at least once."""
    local_buckets = cfg.local_window
    dilation = cfg.global_pinned_dilation
    global_buckets = cfg.global_window // dilation
    # Local layers fill one bucket per frame (dilation=1).
    # Global layers fill one bucket every `dilation` frames.
    # Saturation = when the slower (global) ring wraps: global_buckets * dilation.
    return max(local_buckets, global_buckets * dilation)


def main():
    parser = argparse.ArgumentParser(description="Steady-state model benchmark (saturated KV cache)")
    parser.add_argument("--model-uri", default=MODEL_URI)
    parser.add_argument("--profile", choices=["fp16", "speed", "max_qat"], default="speed")
    parser.add_argument("--frames", type=int, default=20, help="Timed frames after saturation")
    parser.add_argument("--warmup", type=int, default=3, help="Extra warmup frames at saturation (not timed)")
    args = parser.parse_args()

    int8_profile = None if args.profile == "fp16" else args.profile

    print(f"Loading model: {args.model_uri} (profile={args.profile})")
    model, cfg = load_from_pytorch(args.model_uri, int8_profile=int8_profile)

    sat = saturation_frame(cfg)
    total_fill = sat + args.warmup
    total_bench = total_fill + args.frames

    pH, pW = cfg.patch
    latent_shape = (1, 1, 32, cfg.height * pH, cfg.width * pW)
    mouse = mx.zeros((1, 1, 2), dtype=mx.float16)
    button = mx.zeros((1, 1, 256), dtype=mx.float16)
    scroll = mx.zeros((1, 1, 1), dtype=mx.float16)

    # Use a fixed latent so denoise input is deterministic across frames.
    x_fixed = mx.random.normal(latent_shape).astype(mx.float16)
    mx.eval(x_fixed)

    # --- Phase 1: fill cache to saturation + warmup ---
    print(f"\nFilling KV cache: {sat} frames to saturate, +{args.warmup} warmup = {total_fill} frames")

    # Seed frame (frame 0)
    seed = mx.random.normal(latent_shape).astype(mx.float16)
    mx.eval(seed)
    rope_cos_0, rope_sin_0 = compute_rope_angles(0, model.ts_mult, model.rope_xy, model.rope_inv_t)
    model.cache_write(seed, rope_cos_0, rope_sin_0, mouse, button, scroll, 0)

    for fi in range(1, total_fill + 1):
        rope_cos, rope_sin = compute_rope_angles(fi, model.ts_mult, model.rope_xy, model.rope_inv_t)
        out = model.denoise(x_fixed, rope_cos, rope_sin, mouse, button, scroll, fi)
        mx.eval(out)
        model.cache_write(out, rope_cos, rope_sin, mouse, button, scroll, fi)
        if fi % 20 == 0 or fi == total_fill:
            n_written = len(model.kv_caches[0].written_slots)
            n_buckets = model.kv_caches[0].num_buckets
            print(f"  frame {fi:4d}  (local ring: {n_written}/{n_buckets} slots)")

    # Report cache state
    local_kv = model.kv_caches[0]
    global_indices = [i for i, kv in enumerate(model.kv_caches) if kv.dilation > 1]
    global_kv = model.kv_caches[global_indices[0]] if global_indices else None
    print(f"\nCache saturated:")
    print(f"  Local  — {len(local_kv.written_slots)}/{local_kv.num_buckets} slots, "
          f"capacity={local_kv.capacity} tokens, dilation={local_kv.dilation}")
    if global_kv:
        print(f"  Global — {len(global_kv.written_slots)}/{global_kv.num_buckets} slots, "
              f"capacity={global_kv.capacity} tokens, dilation={global_kv.dilation}")

    # Show attention span per layer
    n_local_tokens = len(local_kv.written_slots) * 512
    print(f"  Local  attention tokens: {n_local_tokens}")
    if global_kv:
        n_global_tokens = len(global_kv.written_slots) * 512
        print(f"  Global attention tokens: {n_global_tokens}")

    # --- Phase 2: timed frames ---
    print(f"\nBenchmarking {args.frames} frames at steady state...")

    denoise_times = []
    write_times = []
    frame_times = []

    for i in range(args.frames):
        fi = total_fill + 1 + i
        rope_cos, rope_sin = compute_rope_angles(fi, model.ts_mult, model.rope_xy, model.rope_inv_t)

        t0 = time.perf_counter()
        out = model.denoise(x_fixed, rope_cos, rope_sin, mouse, button, scroll, fi)
        mx.eval(out)
        t1 = time.perf_counter()

        model.cache_write(out, rope_cos, rope_sin, mouse, button, scroll, fi)
        mx.eval(*[arr for kv in model.kv_caches for arr in [kv.keys, kv.values]])
        t2 = time.perf_counter()

        d_ms = (t1 - t0) * 1000
        w_ms = (t2 - t1) * 1000
        f_ms = (t2 - t0) * 1000
        denoise_times.append(d_ms)
        write_times.append(w_ms)
        frame_times.append(f_ms)
        print(f"  frame {fi:4d}: denoise={d_ms:6.1f}ms  write={w_ms:5.1f}ms  total={f_ms:6.1f}ms")

    # --- Results ---
    d = np.array(denoise_times)
    w = np.array(write_times)
    f = np.array(frame_times)

    print(f"\n{'=' * 65}")
    print(f"Steady-state results ({args.frames} frames, profile={args.profile}):")
    print(f"  {'':12s} {'median':>8s} {'mean':>8s} {'std':>7s} {'min':>8s} {'max':>8s} {'p95':>8s}")
    for label, arr in [("denoise", d), ("cache_write", w), ("total", f)]:
        print(f"  {label:12s} {np.median(arr):7.1f}ms {np.mean(arr):7.1f}ms {np.std(arr):6.1f}ms "
              f"{np.min(arr):7.1f}ms {np.max(arr):7.1f}ms {np.percentile(arr, 95):7.1f}ms")
    fps = 1000.0 / np.median(f)
    print(f"\n  Median FPS: {fps:.2f} ({1000.0 / np.median(d):.2f} model-only)")


if __name__ == "__main__":
    main()
