"""
Benchmark: full render pipeline — MLX world model + PyTorch TAEHV decode.

Encodes a seed image via TAEHV, runs the model forward pass per frame,
and decodes latents to RGB. Measures end-to-end wall time.

Usage:
  python -m src.mlx_metal.benchmarks.bench_render                         # ANE decode (default)
  python -m src.mlx_metal.benchmarks.bench_render --no-ane                # CPU decode
  python -m src.mlx_metal.benchmarks.bench_render --frames 30 --save-frames
  python -m src.mlx_metal.benchmarks.bench_render --profile fp16 --no-decode
  python -m src.mlx_metal.benchmarks.bench_render --stability --frames 60
  python -m src.mlx_metal.benchmarks.bench_render --stability --smoothquant --save-frames --frames 45
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


def load_vae(model_uri: str, device: str = "cpu", ane: bool = False):
    from src.model import WorldModel
    from src.ae import get_ae
    cfg = WorldModel.load_config(model_uri)
    ae_uri = getattr(cfg, "ae_uri", model_uri)
    is_taehv = getattr(cfg, "taehv_ae", False)
    return get_ae(ae_uri, is_taehv_ae=is_taehv, device=device, dtype=torch.float32, ane=ane)


def load_seed_image(path: pathlib.Path, height: int, width: int) -> torch.Tensor:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    return torch.from_numpy(np.array(img))


def mlx_to_torch(x: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(x))


# ---------------------------------------------------------------------------
# Stability analysis: latent + pixel drift over a zero-control rollout
# ---------------------------------------------------------------------------

def run_stability(args):
    """Generate frames with zero control inputs and measure drift from frame 0."""
    int8_profile = None if args.profile == "fp16" else args.profile

    print(f"=== Stability Analysis ===")
    print(f"Loading model: {args.model_uri} (profile={args.profile})")
    model, cfg = load_from_pytorch(args.model_uri, int8_profile=int8_profile)

    pH, pW = cfg.patch
    pixel_h = cfg.height * pH * 16
    pixel_w = cfg.width * pW * 16
    latent_shape = (1, 1, cfg.channels, cfg.height * pH, cfg.width * pW)

    # --- VAE setup ---
    _ENCODE_SIZES = {(720, 1280): (512, 1024), (360, 640): (256, 512)}
    encode_h, encode_w = pixel_h, pixel_w
    for (src_h, src_w), (dst_h, dst_w) in _ENCODE_SIZES.items():
        if dst_h == pixel_h and dst_w == pixel_w:
            encode_h, encode_w = src_h, src_w
            break

    vae = load_vae(args.model_uri, ane=args.ane)

    print(f"Loading seed image: {args.seed_image} (resizing to {encode_w}x{encode_h})")
    seed_img = load_seed_image(pathlib.Path(args.seed_image), encode_h, encode_w)

    t_compress = getattr(cfg, "temporal_compression", 1)
    seed_batch = seed_img.unsqueeze(0).expand(t_compress, -1, -1, -1)

    vae.reset()
    with torch.inference_mode():
        seed_latent_pt = vae.encode(seed_batch)
    seed_latent = mx.array(seed_latent_pt.numpy()).astype(mx.float16)
    seed_latent = mx.reshape(seed_latent, latent_shape)
    mx.eval(seed_latent)

    # --- Zero control inputs ---
    mouse = mx.zeros((1, 1, 2), dtype=mx.float16)
    button = mx.zeros((1, 1, cfg.n_buttons), dtype=mx.float16)
    scroll = mx.zeros((1, 1, 1), dtype=mx.float16)

    # --- Seed KV cache ---
    rope_cos_0, rope_sin_0 = compute_rope_angles(0, model.ts_mult, model.rope_xy, model.rope_inv_t)
    model.cache_write(seed_latent, rope_cos_0, rope_sin_0, mouse, button, scroll, 0)

    # Prime streaming VAE decoder
    vae.reset()
    with torch.inference_mode():
        seed_rgb = vae.decode(seed_latent_pt)

    # Baselines: frame 0 latent and RGB
    latent_0 = np.array(seed_latent).astype(np.float32)
    rgb_0 = seed_rgb.cpu().numpy().astype(np.float32)

    # --- Stats accumulators ---
    n_frames = args.frames
    frame_indices = []

    # Consecutive-frame delta (frame N vs N-1)
    consec_latent_mae = []
    consec_latent_max = []
    consec_rgb_mae = []
    consec_rgb_max = []

    # Drift from frame 0 (frame N vs 0)
    drift_latent_mae = []
    drift_latent_max = []
    drift_latent_rmse = []
    drift_rgb_mae = []
    drift_rgb_max = []
    drift_rgb_rmse = []
    drift_rgb_psnr = []

    # Per-frame summaries
    latent_means = []
    latent_stds = []
    latent_abs_maxs = []
    nan_counts = []

    prev_latent = latent_0
    prev_rgb = rgb_0

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {n_frames} frames with zero control inputs...\n")
    print(f"{'frame':>5} | {'lat_mae_c':>10} {'lat_max_c':>10} | {'lat_mae_0':>10} {'lat_max_0':>10} {'lat_rmse_0':>10} | {'rgb_mae_0':>8} {'rgb_psnr':>8} | {'lat_mean':>9} {'lat_std':>9} {'NaN':>5}")
    print("-" * 130)

    for fi in range(n_frames):
        frame_idx = fi + 1
        frame_indices.append(frame_idx)
        rope_cos, rope_sin = compute_rope_angles(frame_idx, model.ts_mult, model.rope_xy, model.rope_inv_t)

        x = mx.array(np.random.randn(*latent_shape).astype(np.float16))
        out = model.denoise(x, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        mx.eval(out)

        out_np = np.array(out).astype(np.float32)
        n_nan = int(np.isnan(out_np).sum())
        nan_counts.append(n_nan)

        if n_nan > 0:
            print(f"{frame_idx:5d} | {'NaN — stopping':>60}")
            # Fill remaining stats with NaN so plots show the break
            for lst in [consec_latent_mae, consec_latent_max, consec_rgb_mae, consec_rgb_max,
                        drift_latent_mae, drift_latent_max, drift_latent_rmse,
                        drift_rgb_mae, drift_rgb_max, drift_rgb_rmse, drift_rgb_psnr,
                        latent_means, latent_stds, latent_abs_maxs]:
                lst.append(float("nan"))
            break

        # Latent stats
        latent_means.append(float(out_np.mean()))
        latent_stds.append(float(out_np.std()))
        latent_abs_maxs.append(float(np.abs(out_np).max()))

        # Consecutive delta (latent)
        diff_c = np.abs(out_np - prev_latent)
        consec_latent_mae.append(float(diff_c.mean()))
        consec_latent_max.append(float(diff_c.max()))

        # Drift from frame 0 (latent)
        diff_0 = np.abs(out_np - latent_0)
        drift_latent_mae.append(float(diff_0.mean()))
        drift_latent_max.append(float(diff_0.max()))
        drift_latent_rmse.append(float(np.sqrt((diff_0 ** 2).mean())))

        # Decode to RGB for pixel-space metrics
        model.cache_write(out, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
        latent_pt = mlx_to_torch(out.squeeze(0)).to(dtype=torch.float32)
        with torch.inference_mode():
            rgb = vae.decode(latent_pt)
        rgb_np = rgb.cpu().numpy().astype(np.float32)

        # Consecutive delta (RGB)
        rgb_diff_c = np.abs(rgb_np - prev_rgb)
        consec_rgb_mae.append(float(rgb_diff_c.mean()))
        consec_rgb_max.append(float(rgb_diff_c.max()))

        # Drift from frame 0 (RGB)
        rgb_diff_0 = np.abs(rgb_np - rgb_0)
        drift_rgb_mae.append(float(rgb_diff_0.mean()))
        drift_rgb_max.append(float(rgb_diff_0.max()))
        mse_rgb = float((rgb_diff_0 ** 2).mean())
        drift_rgb_rmse.append(float(np.sqrt(mse_rgb)))
        # PSNR (uint8 range 0-255)
        psnr = 10.0 * np.log10(255.0 ** 2 / max(mse_rgb, 1e-10))
        drift_rgb_psnr.append(psnr)

        print(f"{frame_idx:5d} | {consec_latent_mae[-1]:10.6f} {consec_latent_max[-1]:10.4f} | "
              f"{drift_latent_mae[-1]:10.6f} {drift_latent_max[-1]:10.4f} {drift_latent_rmse[-1]:10.6f} | "
              f"{drift_rgb_mae[-1]:8.3f} {drift_rgb_psnr[-1]:8.2f} | "
              f"{latent_means[-1]:9.4f} {latent_stds[-1]:9.4f} {n_nan:5d}")

        # Save frame PNG
        if args.save_frames:
            from PIL import Image
            img_np = rgb.cpu().numpy()
            while img_np.ndim > 3:
                img_np = img_np[0]
            Image.fromarray(img_np).save(out_dir / f"frame_{frame_idx:04d}.png")

        prev_latent = out_np
        prev_rgb = rgb_np

    # --- Save CSV ---
    csv_path = out_dir / "stability_stats.csv"
    with open(csv_path, "w") as f:
        cols = [
            "frame", "consec_lat_mae", "consec_lat_max", "consec_rgb_mae", "consec_rgb_max",
            "drift_lat_mae", "drift_lat_max", "drift_lat_rmse",
            "drift_rgb_mae", "drift_rgb_max", "drift_rgb_rmse", "drift_rgb_psnr",
            "lat_mean", "lat_std", "lat_abs_max", "nan_count",
        ]
        f.write(",".join(cols) + "\n")
        for i in range(len(frame_indices)):
            row = [
                frame_indices[i],
                consec_latent_mae[i] if i < len(consec_latent_mae) else "",
                consec_latent_max[i] if i < len(consec_latent_max) else "",
                consec_rgb_mae[i] if i < len(consec_rgb_mae) else "",
                consec_rgb_max[i] if i < len(consec_rgb_max) else "",
                drift_latent_mae[i] if i < len(drift_latent_mae) else "",
                drift_latent_max[i] if i < len(drift_latent_max) else "",
                drift_latent_rmse[i] if i < len(drift_latent_rmse) else "",
                drift_rgb_mae[i] if i < len(drift_rgb_mae) else "",
                drift_rgb_max[i] if i < len(drift_rgb_max) else "",
                drift_rgb_rmse[i] if i < len(drift_rgb_rmse) else "",
                drift_rgb_psnr[i] if i < len(drift_rgb_psnr) else "",
                latent_means[i] if i < len(latent_means) else "",
                latent_stds[i] if i < len(latent_stds) else "",
                latent_abs_maxs[i] if i < len(latent_abs_maxs) else "",
                nan_counts[i] if i < len(nan_counts) else "",
            ]
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"\nStats saved to {csv_path}")

    # --- Plot ---
    _plot_stability(
        out_dir, frame_indices, args.profile,
        consec_latent_mae, consec_latent_max,
        drift_latent_mae, drift_latent_max, drift_latent_rmse,
        consec_rgb_mae, drift_rgb_mae, drift_rgb_psnr,
        latent_means, latent_stds, latent_abs_maxs,
    )


def _plot_stability(
    out_dir, frames, profile,
    consec_lat_mae, consec_lat_max,
    drift_lat_mae, drift_lat_max, drift_lat_rmse,
    consec_rgb_mae, drift_rgb_mae, drift_rgb_psnr,
    lat_means, lat_stds, lat_abs_maxs,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle(f"Stability Analysis — profile={profile}, {len(frames)} frames, zero control", fontsize=14)

    # 1. Consecutive latent delta
    ax = axes[0, 0]
    ax.plot(frames, consec_lat_mae, label="MAE", color="tab:blue")
    ax.plot(frames, consec_lat_max, label="Max", color="tab:red", alpha=0.7)
    ax.set_title("Consecutive Frame Delta (latent)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Absolute difference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Drift from frame 0 (latent)
    ax = axes[0, 1]
    ax.plot(frames, drift_lat_mae, label="MAE", color="tab:blue")
    ax.plot(frames, drift_lat_rmse, label="RMSE", color="tab:orange")
    ax.plot(frames, drift_lat_max, label="Max", color="tab:red", alpha=0.7)
    ax.set_title("Drift from Frame 0 (latent)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Absolute difference")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Consecutive RGB delta
    ax = axes[1, 0]
    ax.plot(frames, consec_rgb_mae, label="MAE", color="tab:green")
    ax.set_title("Consecutive Frame Delta (RGB)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pixel MAE (0-255)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Drift from frame 0 (RGB) + PSNR
    ax = axes[1, 1]
    ax.plot(frames, drift_rgb_mae, label="RGB MAE vs frame 0", color="tab:green")
    ax.set_ylabel("Pixel MAE (0-255)")
    ax.set_xlabel("Frame")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(frames, drift_rgb_psnr, label="PSNR vs frame 0", color="tab:purple", linestyle="--")
    ax2.set_ylabel("PSNR (dB)")
    ax2.legend(loc="upper right")
    ax.set_title("Drift from Frame 0 (RGB) + PSNR")

    # 5. Latent distribution over time
    ax = axes[2, 0]
    ax.plot(frames, lat_means, label="Mean", color="tab:blue")
    ax.fill_between(frames,
                    [m - s for m, s in zip(lat_means, lat_stds)],
                    [m + s for m, s in zip(lat_means, lat_stds)],
                    alpha=0.2, color="tab:blue", label="Mean +/- Std")
    ax.set_title("Latent Distribution Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Latent abs max (numerical health)
    ax = axes[2, 1]
    ax.plot(frames, lat_abs_maxs, label="Max |latent|", color="tab:red")
    ax.axhline(y=65504, color="gray", linestyle=":", alpha=0.5, label="fp16 max")
    ax.set_title("Latent Absolute Max (numerical health)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Max |value|")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = out_dir / "stability_plots.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plots saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Full render benchmark: MLX model + VAE decode")
    parser.add_argument("--model-uri", default=MODEL_URI)
    parser.add_argument("--seed-image", default=str(SEED_IMAGE))
    parser.add_argument("--profile", choices=["fp16", "speed", "max_qat"], default="speed")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to generate")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Warmup frames (not timed)")
    parser.add_argument("--save-frames", action="store_true", help="Save rendered frames as PNGs")
    parser.add_argument("--out-dir", default="bench_render_output", help="Output directory for saved frames")
    parser.add_argument("--no-decode", action="store_true", help="Skip VAE decode (measure model only)")
    parser.add_argument("--smoothquant", action="store_true", help="Use SmoothQuant model")
    parser.add_argument("--stability", action="store_true",
                        help="Run stability analysis: measure latent/pixel drift over a zero-control rollout")
    parser.add_argument("--no-ane", action="store_true",
                        help="Disable ANE, run TAEHV on CPU instead (ANE is default — it frees GPU for the world model)")
    args = parser.parse_args()
    args.ane = not args.no_ane

    if args.smoothquant and args.model_uri == MODEL_URI:
        args.model_uri = SMOOTHQUANT_URI

    if args.stability:
        run_stability(args)
        return

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
    _ENCODE_SIZES = {(720, 1280): (512, 1024), (360, 640): (256, 512)}
    encode_h, encode_w = pixel_h, pixel_w
    for (src_h, src_w), (dst_h, dst_w) in _ENCODE_SIZES.items():
        if dst_h == pixel_h and dst_w == pixel_w:
            encode_h, encode_w = src_h, src_w
            break

    vae_label = "CoreML (stateful)" if args.ane else "CPU"
    print(f"Loading VAE from model config ({vae_label})")
    vae = load_vae(args.model_uri, ane=args.ane)

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

        # Build a collage of all saved frames
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
            collage_path = out_dir / "collage.png"
            collage.save(collage_path)
            print(f"  Collage: {collage_path}")


if __name__ == "__main__":
    main()
