"""
Benchmark: full render pipeline — MLX world model + PyTorch TAEHV decode.

Encodes a seed image via TAEHV, runs the model forward pass per frame,
and decodes latents to RGB. Measures end-to-end wall time.

Usage:
  python -m src.mlx_metal.benchmarks.bench_render
  python -m src.mlx_metal.benchmarks.bench_render --frames 30 --save-frames
  python -m src.mlx_metal.benchmarks.bench_render --profile fp16 --no-decode
"""
from __future__ import annotations

import argparse
import pathlib
import time

import mlx.core as mx
import numpy as np
import torch

from ..mlx_world_model import load_from_pytorch, compute_rope_angles


MODEL_URI = "Overworld-Models/MR160k"
SMOOTHQUANT_URI = "Overworld-Models/MR160k-smoothquant"
SEED_IMAGE = pathlib.Path(__file__).parent / "frozen_valley_sniper.jpg"


def load_vae(model_uri: str, device: str = "cpu"):
    from src.model import WorldModel
    from src.ae import get_ae
    cfg = WorldModel.load_config(model_uri)
    ae_uri = getattr(cfg, "ae_uri", model_uri)
    is_taehv = getattr(cfg, "taehv_ae", False)
    return get_ae(ae_uri, is_taehv_ae=is_taehv, device=device, dtype=torch.float32)


def load_seed_image(path: pathlib.Path, height: int, width: int) -> torch.Tensor:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    return torch.from_numpy(np.array(img))


def mlx_to_torch(x: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(x))


def main():
    parser = argparse.ArgumentParser(description="Full render benchmark: MLX model + VAE decode")
    parser.add_argument("--model-uri", default=MODEL_URI)
    parser.add_argument("--seed-image", default=str(SEED_IMAGE))
    parser.add_argument("--profile", choices=["fp16", "speed", "max_qat"], default="speed")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to generate")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Warmup frames (not timed)")
    parser.add_argument("--save-frames", action="store_true", help="Save rendered frames as PNGs")
    parser.add_argument("--out-dir", default="bench_render_output", help="Output directory for saved frames")
    parser.add_argument("--no-decode", action="store_true", help="Skip VAE decode (measure model only)")
    parser.add_argument("--smoothquant", action="store_true", help="Use SmoothQuant model")
    args = parser.parse_args()

    if args.smoothquant and args.model_uri == MODEL_URI:
        args.model_uri = SMOOTHQUANT_URI

    int8_profile = None if args.profile == "fp16" else args.profile

    print(f"Loading model: {args.model_uri} (profile={args.profile})")
    model, cfg = load_from_pytorch(args.model_uri, int8_profile=int8_profile)
    print(f"  int8_stats: {getattr(model, 'int8_stats', None)}")

    pH, pW = cfg.patch
    pixel_h = cfg.height * pH * 16
    pixel_w = cfg.width * pW * 16
    latent_shape = (1, 1, cfg.channels, cfg.height * pH, cfg.width * pW)

    # --- Load VAE and encode seed image ---
    # The VAE expects the original video resolution (e.g. 720x1280) and
    # internally resizes to the latent pixel size (e.g. 512x1024).
    # Look up which input resolution maps to our latent pixel size.
    from src.ae import ChunkedStreamingTAEHV
    encode_h, encode_w = pixel_h, pixel_w
    for (src_h, src_w), (dst_h, dst_w) in ChunkedStreamingTAEHV._ENCODE_SIZES.items():
        if dst_h == pixel_h and dst_w == pixel_w:
            encode_h, encode_w = src_h, src_w
            break

    print(f"Loading VAE from model config")
    vae = load_vae(args.model_uri)

    print(f"Loading seed image: {args.seed_image} (resizing to {encode_w}x{encode_h})")
    seed_img = load_seed_image(pathlib.Path(args.seed_image), encode_h, encode_w)

    t_compress = getattr(cfg, "temporal_compression", 1)
    seed_batch = seed_img.unsqueeze(0).expand(t_compress, -1, -1, -1)

    vae.reset()
    print(f"Encoding seed frame (T={t_compress})...")
    with torch.inference_mode():
        seed_latent_pt = vae.encode(seed_batch)
    seed_latent = mx.array(seed_latent_pt.numpy()).astype(mx.float16)
    seed_latent = mx.reshape(seed_latent, latent_shape)
    mx.eval(seed_latent)
    print(f"  seed latent: {seed_latent.shape}")

    # --- Setup ---
    mouse = mx.zeros((1, 1, 2), dtype=mx.float16)
    button = mx.zeros((1, 1, cfg.n_buttons), dtype=mx.float16)
    scroll = mx.zeros((1, 1, 1), dtype=mx.float16)
    decode_enabled = not args.no_decode

    # --- Seed the KV cache ---
    # Write seed frame to cache and ring-copy so attention has context.
    print("Seeding KV cache...")
    rope_cos_0, rope_sin_0 = compute_rope_angles(0, model.ts_mult, model.rope_xy, model.rope_inv_t)
    model.cache_write(seed_latent, rope_cos_0, rope_sin_0, mouse, button, scroll, 0)

    # Prime the streaming decoder with the seed so it has temporal context
    if decode_enabled:
        vae.reset()
        with torch.inference_mode():
            vae.decode(seed_latent_pt)

    if args.save_frames:
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = args.warmup_frames + args.frames
    print(f"\nRendering {total_frames} frames ({args.warmup_frames} warmup + {args.frames} timed)...")
    print(f"  decode={'ON' if decode_enabled else 'OFF'}")

    # --- Frame loop ---
    # Each frame: denoise (4 forward passes) + cache_write (1 forward pass) + decode.
    # This matches bench_mlx.py's full-frame timing pattern.
    frame_times = []
    model_times = []
    decode_times = []

    for fi in range(total_frames):
        frame_idx = fi + 1  # frame 0 is the seed
        rope_cos, rope_sin = compute_rope_angles(frame_idx, model.ts_mult, model.rope_xy, model.rope_inv_t)

        # --- Model: denoise (4-step) + cache write ---
        t_model_start = time.perf_counter()
        x = mx.array(np.random.randn(*latent_shape).astype(np.float16))
        out = model.denoise(x, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        mx.eval(out)

        # Check for NaN (int8 quantization can diverge with large KV context)
        out_np = np.array(out)
        if np.isnan(out_np).sum() > 0:
            print(f"  frame {frame_idx:3d}: NaN — stopping generation")
            break

        model.cache_write(out, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        t_model_end = time.perf_counter()

        # --- VAE decode ---
        t_decode_start = time.perf_counter()
        img = None
        if decode_enabled:
            latent_pt = mlx_to_torch(out.squeeze(0)).to(dtype=torch.float32)
            with torch.inference_mode():
                img = vae.decode(latent_pt)
        t_decode_end = time.perf_counter()

        is_timed = fi >= args.warmup_frames
        model_ms = (t_model_end - t_model_start) * 1000
        decode_ms = (t_decode_end - t_decode_start) * 1000 if decode_enabled else 0
        frame_ms = (t_decode_end - t_model_start) * 1000

        if is_timed:
            model_times.append(model_ms)
            decode_times.append(decode_ms)
            frame_times.append(frame_ms)

        tag = "" if is_timed else " (warmup)"
        if decode_enabled:
            print(f"  frame {frame_idx:3d}: model={model_ms:6.1f}ms  decode={decode_ms:6.1f}ms  total={frame_ms:6.1f}ms{tag}")
        else:
            print(f"  frame {frame_idx:3d}: model={model_ms:6.1f}ms{tag}")

        if args.save_frames and img is not None and is_timed:
            from PIL import Image
            img_np = img.cpu().numpy()
            while img_np.ndim > 3:
                img_np = img_np[0]
            Image.fromarray(img_np).save(out_dir / f"frame_{frame_idx:04d}.png")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"Results ({args.frames} frames, profile={args.profile}):")
    print(f"  Model:  {np.mean(model_times):6.1f}ms avg  ({np.std(model_times):5.1f}ms std)")
    if decode_enabled:
        print(f"  Decode: {np.mean(decode_times):6.1f}ms avg  ({np.std(decode_times):5.1f}ms std)")
    print(f"  Total:  {np.mean(frame_times):6.1f}ms avg  ({np.std(frame_times):5.1f}ms std)")
    fps = 1000.0 / np.mean(frame_times)
    model_fps = 1000.0 / np.mean(model_times)
    print(f"  FPS:    {fps:.2f} (model-only: {model_fps:.2f})")
    if args.save_frames:
        print(f"  Frames saved to: {out_dir}/")


if __name__ == "__main__":
    main()
