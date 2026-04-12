"""
Capture a Metal GPU trace of the MLX world model for profiling in Xcode.

Produces a .gputrace file that can be opened in Xcode to inspect per-kernel
GPU timelines, ALU utilization, memory bandwidth, and scheduling gaps.

Usage:
  # Single transformer block (smallest trace, start here)
  MTL_CAPTURE_ENABLED=1 python -m src.mlx_metal.benchmarks.profile_capture --component single-layer

  # Single forward pass (all 24 layers, 1 denoise step)
  MTL_CAPTURE_ENABLED=1 python -m src.mlx_metal.benchmarks.profile_capture --component single-step

  # Full denoise (4 steps + cache write, default)
  MTL_CAPTURE_ENABLED=1 python -m src.mlx_metal.benchmarks.profile_capture

  # Full pipeline including ANE decode
  MTL_CAPTURE_ENABLED=1 python -m src.mlx_metal.benchmarks.profile_capture --component full

  # Unfused kernels (individual ops visible)
  MTL_CAPTURE_ENABLED=1 python -m src.mlx_metal.benchmarks.profile_capture --no-compile --component single-layer

Then: open profile_output/trace.gputrace  (opens in Xcode)

Tips:
  - Build MLX with CMAKE_ARGS="-DMLX_METAL_DEBUG=ON" for kernel names in trace
  - Use --no-compile to disable mx.compile fusion and see individual kernel costs
  - Start with single-layer, then single-step — full model traces can OOM Xcode
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil

import mlx.core as mx
import numpy as np
import torch

from ..mlx_world_model import load_from_pytorch, compute_rope_angles


MODEL_URI = "Overworld-Models/MR160k"
SMOOTHQUANT_URI = "Overworld-Models/MR160k-smoothquant"
SEED_IMAGE = pathlib.Path(__file__).parent / "frozen_valley_sniper.jpg"


def main():
    parser = argparse.ArgumentParser(description="Capture Metal GPU trace for Xcode profiling")
    parser.add_argument("--model-uri", default=MODEL_URI)
    parser.add_argument("--profile", choices=["fp16", "speed", "max_qat"], default="speed")
    parser.add_argument("--frames", type=int, default=1, help="Frames to capture (keep small — 1 frame = ~100 kernel dispatches)")
    parser.add_argument("--output", default="profile_output/trace.gputrace")
    parser.add_argument("--smoothquant", action="store_true")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable mx.compile to see individual kernel costs")
    parser.add_argument("--component", choices=["full", "model", "decode", "single-step", "single-layer"], default="model",
                        help="What to capture: model (denoise+cache), decode (ANE TAEHV), full, "
                             "single-step (1 forward pass), single-layer (1 transformer block)")
    args = parser.parse_args()

    if args.smoothquant and args.model_uri == MODEL_URI:
        args.model_uri = SMOOTHQUANT_URI

    if args.no_compile:
        mx.disable_compile()
        print("mx.compile DISABLED — trace will show individual kernels")

    if "MTL_CAPTURE_ENABLED" not in os.environ:
        print("ERROR: Set MTL_CAPTURE_ENABLED=1 before running.")
        print("  MTL_CAPTURE_ENABLED=1 python -m src.mlx_metal.benchmarks.profile_capture")
        return

    # Clean up existing trace
    trace_path = args.output
    if os.path.exists(trace_path):
        shutil.rmtree(trace_path)
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)

    int8_profile = None if args.profile == "fp16" else args.profile

    # --- Load model ---
    print(f"Loading model: {args.model_uri} (profile={args.profile})")
    model, cfg = load_from_pytorch(args.model_uri, int8_profile=int8_profile)

    pH, pW = cfg.patch
    latent_shape = (1, 1, cfg.channels, cfg.height * pH, cfg.width * pW)

    # --- Load VAE for decode capture ---
    vae = None
    if args.component in ("full", "decode"):
        from ...ae import get_ae
        vae = get_ae(
            cfg.ae_uri,
            is_taehv_ae=getattr(cfg, "taehv_ae", False),
            ane=True,
            dtype=torch.float32,
        )

    # --- Setup controls ---
    mouse = mx.zeros((1, 1, 2), dtype=mx.float16)
    button = mx.zeros((1, 1, cfg.n_buttons), dtype=mx.float16)
    scroll = mx.zeros((1, 1, 1), dtype=mx.float16)

    # --- Seed KV cache ---
    print("Seeding KV cache...")
    rope_cos_0, rope_sin_0 = compute_rope_angles(0, model.ts_mult, model.rope_xy, model.rope_inv_t)
    seed = mx.zeros(latent_shape, dtype=mx.float16)
    model.cache_write(seed, rope_cos_0, rope_sin_0, mouse, button, scroll, 0)

    # --- Warmup (critical — includes JIT compilation) ---
    print(f"Warming up ({args.frames} iterations)...")
    for fi in range(args.frames):
        frame_idx = fi + 1
        rope_cos, rope_sin = compute_rope_angles(frame_idx, model.ts_mult, model.rope_xy, model.rope_inv_t)
        x = mx.array(np.random.randn(*latent_shape).astype(np.float16))
        out = model.denoise(x, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        mx.eval(out)
        model.cache_write(out, rope_cos, rope_sin, mouse, button, scroll, frame_idx)

        if vae is not None:
            lat_pt = torch.from_numpy(np.array(out.squeeze(0))).to(dtype=torch.float32)
            with torch.inference_mode():
                vae.decode(lat_pt)

    # --- Capture ---
    print(f"Capturing {args.frames} frames to {trace_path} ...")
    print(f"  component={args.component}  compile={'OFF' if args.no_compile else 'ON'}")

    frame_idx_start = args.frames + 1  # continue from warmup

    mx.metal.start_capture(trace_path)

    if args.component == "single-layer":
        # Capture a single transformer block forward pass
        from ..mlx_world_model import SIGMAS, DTYPE, N_LAYERS, D_MODEL, T, N_KV_HEADS, D_HEAD
        frame_idx = frame_idx_start
        rope_cos, rope_sin = compute_rope_angles(frame_idx, model.ts_mult, model.rope_xy, model.rope_inv_t)
        cond = model.noise_cond(SIGMAS[0])
        ctrl_emb = model.ctrl_embed(mouse, button, scroll)
        x_seq = mx.random.normal((1, T, D_MODEL)).astype(DTYPE)
        v1 = mx.zeros((1, N_KV_HEADS, T, D_HEAD), dtype=DTYPE)
        blk = model.transformer[0]
        kv = model.kv_caches[0]
        kv.set_frozen(True)
        bo = kv.compute_block_offsets(frame_idx)
        x_seq, v1 = blk(x_seq, cond, ctrl_emb, rope_cos, rope_sin, v1, kv, frame_idx, bo)
        mx.eval(x_seq)

    elif args.component == "single-step":
        # Capture a single forward pass (1 denoise step, all layers)
        from ..mlx_world_model import SIGMAS
        frame_idx = frame_idx_start
        rope_cos, rope_sin = compute_rope_angles(frame_idx, model.ts_mult, model.rope_xy, model.rope_inv_t)
        cond = model.noise_cond(SIGMAS[0])
        x = mx.array(np.random.randn(*latent_shape).astype(np.float16))
        for kv in model.kv_caches:
            kv.set_frozen(True)
        out = model.forward_single(x, cond, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        mx.eval(out)

    else:
        for fi in range(args.frames):
            frame_idx = fi + frame_idx_start
            rope_cos, rope_sin = compute_rope_angles(frame_idx, model.ts_mult, model.rope_xy, model.rope_inv_t)

            if args.component in ("full", "model"):
                x = mx.array(np.random.randn(*latent_shape).astype(np.float16))
                out = model.denoise(x, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
                mx.eval(out)
                model.cache_write(out, rope_cos, rope_sin, mouse, button, scroll, frame_idx)

            if args.component in ("full", "decode") and vae is not None:
                if args.component == "decode":
                    out = mx.array(np.random.randn(*latent_shape).astype(np.float16))
                lat_pt = torch.from_numpy(np.array(out.squeeze(0))).to(dtype=torch.float32)
                with torch.inference_mode():
                    vae.decode(lat_pt)

    mx.metal.stop_capture()

    print(f"\nCapture saved to: {trace_path}")
    print(f"Open in Xcode:    open {trace_path}")
    print(f"\nMemory stats:")
    print(f"  Peak:   {mx.metal.get_peak_memory() / 1e9:.2f} GB")
    print(f"  Active: {mx.metal.get_active_memory() / 1e9:.2f} GB")
    print(f"  Cache:  {mx.metal.get_cache_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
