"""
Benchmark MLXWorldEngine end-to-end: seed → gen_frame loop.

Measures model (denoise+cache), decode, and total frame times using
the MLXWorldEngine API. Supports both synchronous and pipelined (ANE) modes.

Usage:
  python -m src.mlx_metal.benchmarks.bench_engine
  python -m src.mlx_metal.benchmarks.bench_engine --ane --frames 30
  python -m src.mlx_metal.benchmarks.bench_engine --profile fp16 --no-decode
  python -m src.mlx_metal.benchmarks.bench_engine --ane --save-frames
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np
import torch

from ..engine import MLXWorldEngine
from ...world_engine import CtrlInput


MODEL_URI = "Overworld-Models/MR160k"
SMOOTHQUANT_URI = "Overworld-Models/MR160k-smoothquant"
SEED_IMAGE = pathlib.Path(__file__).parent / "frozen_valley_sniper.jpg"


def load_seed_image(path: pathlib.Path, height: int, width: int) -> torch.Tensor:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    return torch.from_numpy(np.array(img))


def main():
    parser = argparse.ArgumentParser(description="MLXWorldEngine end-to-end benchmark")
    parser.add_argument("--model-uri", default=MODEL_URI)
    parser.add_argument("--seed-image", default=str(SEED_IMAGE))
    parser.add_argument("--profile", choices=["fp16", "speed", "max_qat"], default="speed")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--no-ane", action="store_true", help="Disable ANE, run TAEHV on CPU instead")
    parser.add_argument("--no-decode", action="store_true", help="Skip VAE decode (model-only timing)")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--out-dir", default="bench_engine_output")
    parser.add_argument("--smoothquant", action="store_true")
    args = parser.parse_args()

    if args.smoothquant and args.model_uri == MODEL_URI:
        args.model_uri = SMOOTHQUANT_URI

    int8_profile = None if args.profile == "fp16" else args.profile
    ane = not args.no_ane

    # --- Build engine ---
    print(f"Loading MLXWorldEngine: {args.model_uri} (profile={args.profile}, ane={ane})")
    engine = MLXWorldEngine(
        args.model_uri,
        int8_profile=int8_profile,
        ane_vae=ane,
    )
    cfg = engine.model_cfg
    pH, pW = cfg.patch

    # --- Encode sizes lookup ---
    _ENCODE_SIZES = {(720, 1280): (512, 1024), (360, 640): (256, 512)}
    pixel_h = cfg.height * pH * 16
    pixel_w = cfg.width * pW * 16
    encode_h, encode_w = pixel_h, pixel_w
    for (src_h, src_w), (dst_h, dst_w) in _ENCODE_SIZES.items():
        if dst_h == pixel_h and dst_w == pixel_w:
            encode_h, encode_w = src_h, src_w
            break

    # --- Seed ---
    print(f"Loading seed image: {args.seed_image} ({encode_w}x{encode_h})")
    seed_img = load_seed_image(pathlib.Path(args.seed_image), encode_h, encode_w)

    t_compress = getattr(cfg, "temporal_compression", 1)
    seed_batch = seed_img.unsqueeze(0).expand(t_compress, -1, -1, -1).clone()

    engine.reset()
    engine.append_frame(seed_batch)

    decode_enabled = not args.no_decode
    use_pipeline = ane and decode_enabled

    if args.save_frames:
        out_dir = pathlib.Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = args.warmup_frames + args.frames
    mode = "pipelined" if use_pipeline else ("sync" if decode_enabled else "model-only")
    print(f"\nRendering {total_frames} frames ({args.warmup_frames} warmup + {args.frames} timed)")
    print(f"  mode={mode}  decode={'ON' if decode_enabled else 'OFF'}")

    # --- Frame loop ---
    frame_times = []

    for fi in range(total_frames):
        t0 = time.perf_counter()

        if use_pipeline:
            img = engine.gen_frame_pipelined()
        else:
            img = engine.gen_frame(return_img=decode_enabled)

        frame_ms = (time.perf_counter() - t0) * 1000
        is_timed = fi >= args.warmup_frames

        if is_timed:
            frame_times.append(frame_ms)

        tag = "" if is_timed else " (warmup)"
        print(f"  frame {fi + 1:3d}: {frame_ms:6.1f}ms{tag}")

        if args.save_frames and is_timed and img is not None:
            from PIL import Image
            frames_np = img.cpu().numpy()  # [T, H, W, 3]
            for t, frame in enumerate(frames_np):
                Image.fromarray(frame).save(out_dir / f"frame_{fi + 1:04d}_{t}.png")

    # Flush pipeline
    if use_pipeline:
        t0 = time.perf_counter()
        last = engine.flush_pipeline()
        flush_ms = (time.perf_counter() - t0) * 1000
        print(f"  flush:     {flush_ms:6.1f}ms")
        if args.save_frames and last is not None:
            from PIL import Image
            for t, frame in enumerate(last.cpu().numpy()):
                Image.fromarray(frame).save(out_dir / f"frame_{total_frames:04d}_{t}.png")

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"MLXWorldEngine Benchmark ({args.frames} timed frames)")
    print(f"  Profile: {args.profile}  ANE: {ane}  Mode: {mode}")
    t_upscale = getattr(cfg, "temporal_compression", 1)
    if t_upscale <= 1:
        t_upscale = 4  # taehv1_5 default: 1 latent → 4 video frames

    print(f"  Latent step: {np.mean(frame_times):6.1f}ms avg  ({np.std(frame_times):5.1f}ms std)")
    latent_fps = 1000.0 / np.mean(frame_times)
    video_fps = latent_fps * t_upscale
    print(f"  Latent FPS:  {latent_fps:.2f}")
    print(f"  Video FPS:   {video_fps:.2f}  (×{t_upscale} temporal upscale)")
    if args.save_frames:
        from PIL import Image
        saved = sorted(out_dir.glob("frame_*.png"))
        if saved:
            imgs = [Image.open(p) for p in saved]
            w, h = imgs[0].size
            cols = min(4, len(imgs))
            rows = (len(imgs) + cols - 1) // cols
            collage = Image.new("RGB", (w * cols, h * rows))
            for i, im in enumerate(imgs):
                collage.paste(im, ((i % cols) * w, (i // cols) * h))
            collage.save(out_dir / "collage.png")
            print(f"  Frames: {out_dir}/  Collage: {out_dir}/collage.png")


if __name__ == "__main__":
    main()
