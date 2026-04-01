"""Benchmark MLX fp16 and fused-int8 profiles."""
import argparse
import time

import mlx.core as mx
import numpy as np

from ..mlx_world_model import load_from_pytorch, compute_rope_angles


MODEL_URI = "Overworld-Models/MR160k"


def bench_model(model, label: str):
    cos, sin = compute_rope_angles(0, model.ts_mult, model.rope_xy, model.rope_inv_t)
    x = mx.random.normal((1, 1, 32, 32, 64)).astype(mx.float16)
    mouse = mx.zeros((1, 1, 2), dtype=mx.float16)
    button = mx.zeros((1, 1, 256), dtype=mx.float16)
    scroll = mx.zeros((1, 1, 1), dtype=mx.float16)
    cond = model.noise_cond(1.0)

    for _ in range(2):
        mx.eval(model.forward_single(x, cond, cos, sin, mouse, button, scroll, 0))

    times = []
    for _ in range(6):
        t0 = time.perf_counter()
        out = model.forward_single(x, cond, cos, sin, mouse, button, scroll, 0)
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    print(f"{label} single fwd: {1000 * np.mean(times):.1f}ms")

    for fi in range(3):
        cf, sf = compute_rope_angles(fi, model.ts_mult, model.rope_xy, model.rope_inv_t)
        model.cache_write(x, cf, sf, mouse, button, scroll, fi)

    times_f = []
    for fi in range(3, 6):
        cf, sf = compute_rope_angles(fi, model.ts_mult, model.rope_xy, model.rope_inv_t)
        t0 = time.perf_counter()
        out = model.denoise(x, cf, sf, mouse, button, scroll, fi)
        mx.eval(out)
        model.cache_write(out, cf, sf, mouse, button, scroll, fi)
        mx.eval(model.kv_caches[0].keys)
        times_f.append(time.perf_counter() - t0)
    print(f"{label} full frame: {1000 * np.mean(times_f):.0f}ms ({1000 / np.mean(times_f) / 1000:.1f} FPS)")
    return np.mean(times), np.mean(times_f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", default=MODEL_URI)
    args = parser.parse_args()

    summaries = []
    for profile in [None, "speed", "max_qat"]:
        model, cfg = load_from_pytorch(
            args.model_uri,
            int8_profile=profile,
            kv_cache_mode="fp16",
            attention_mode="fp16",
        )
        label = "fp16" if profile is None else f"fused w8a8 nax {profile}"
        print(
            {
                "model_uri": args.model_uri,
                "profile": profile,
                "n_heads": getattr(cfg, "n_heads", None),
                "n_kv_heads": getattr(cfg, "n_kv_heads", None),
                "ts_mult": model.ts_mult,
                "int8_stats": getattr(model, "int8_stats", None),
            }
        )
        single, frame = bench_model(model, label)
        summaries.append((label, single, frame))

    print("\nSummary:")
    for label, single, frame in summaries:
        print(f"  {label}: {1000*single:.1f}ms/fwd, {1000*frame:.0f}ms/frame ({1000/frame/1000:.1f} FPS)")


if __name__ == "__main__":
    main()
