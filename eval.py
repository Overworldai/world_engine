"""
WorldEngine quantization eval: forward-pass MSE.

Measures quantization degradation by running the same noisy latent through
both a bf16 reference model and a quantized model, then comparing their outputs.

Primary metric:  latent MSE  — MSE between denoised latents (what the model outputs)
Secondary metric: pixel MSE  — MSE between decoded images (more interpretable)

Usage:
    uv run --dev eval.py <model_uri> --quant <scheme> [options]
    uv run --dev eval.py Overworld/Waypoint-1.5-1B --quant intw8a8
    uv run --dev eval.py Overworld/Waypoint-1.5-1B --quant intw8a8 --n-trials 5
"""

import argparse
import urllib.request
import cv2
import numpy as np
import torch

from world_engine import WorldEngine, CtrlInput


SEED_FRAME_URLS = [
    "https://gist.github.com/user-attachments/assets/d81c6d26-a838-4afe-9d13-fd67677043c3",
    "https://gist.github.com/user-attachments/assets/b6d18c38-098e-43b0-8e61-66a16e5d8946",
    "https://gist.github.com/user-attachments/assets/0734a8c1-3eb4-4ffe-8c37-5665c45ab559",
    "https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed",
    "https://gist.github.com/user-attachments/assets/68c943a4-008a-4c25-948c-c81ab4c47d21",
]


def download_seed_frame(url_index: int = 0) -> np.ndarray:
    """Download a seed frame and repeat 4x to get [4, H, W, 3] uint8."""
    url = SEED_FRAME_URLS[url_index]
    print(f"Downloading seed frame from {url}")
    frame = cv2.imdecode(
        np.frombuffer(urllib.request.urlopen(url).read(), np.uint8),
        cv2.IMREAD_COLOR,
    )
    return np.repeat(frame[None], 4, axis=0)  # [4, H, W, 3]


@torch.inference_mode()
def forward_pass_mse(
    model_uri: str,
    quant: str,
    smooth: bool,
    seed_frame: np.ndarray,
    n_trials: int = 1,
    torch_seed: int = 42,
    device: str = "cuda",
) -> dict:
    """
    Measure quantization error via single forward-pass MSE.

    Loads bf16 and quantized engines, seeds their KV caches identically,
    runs the same noisy latent through both denoise passes, and compares outputs.

    Args:
        seed_frame: [T, H, W, 3] uint8 — used to seed KV cache context
        n_trials:   number of independent noise samples to average over

    Returns:
        dict with mean latent_mse and pixel_mse across trials
    """
    print(f"Loading bf16 engine: {model_uri}")
    engine_bf16 = WorldEngine("Overworld-Models/MR160k", quant=None, device=device)

    print(f"Loading quantized engine: {model_uri}  quant={quant}  smooth={smooth}")
    engine_quant = WorldEngine(model_uri, quant=quant, smooth=smooth, device=device)

    seed_tensor = torch.from_numpy(np.ascontiguousarray(seed_frame)).to(device)

    results = []
    for trial in range(n_trials):
        # Seed KV caches identically on both engines
        engine_bf16.reset()
        engine_quant.reset()
        engine_bf16.append_frame(seed_tensor)
        engine_quant.append_frame(seed_tensor)

        # Sample the same noise for both
        torch.manual_seed(torch_seed + trial)
        x_bf16 = torch.randn(engine_bf16.frm_shape, device=device, dtype=engine_bf16.dtype)
        x_quant = x_bf16.clone().to(engine_quant.dtype)

        ctrl = CtrlInput()
        inputs_bf16  = engine_bf16.prep_inputs(x_bf16, ctrl)
        inputs_quant = engine_quant.prep_inputs(x_quant, ctrl)

        lat_bf16  = engine_bf16._denoise_pass(x_bf16,  inputs_bf16,  engine_bf16.kv_cache).clone()
        lat_quant = engine_quant._denoise_pass(x_quant, inputs_quant, engine_quant.kv_cache).clone()

        # Primary: latent-space MSE
        latent_mse = (lat_bf16.float() - lat_quant.float()).pow(2).mean().item()

        # Secondary: pixel-space MSE after VAE decode
        img_bf16  = engine_bf16.vae.decode(lat_bf16.squeeze(1))
        img_quant = engine_quant.vae.decode(lat_quant.squeeze(1))
        pixel_mse = (img_bf16.float() - img_quant.float()).pow(2).mean().item()

        results.append({"latent_mse": latent_mse, "pixel_mse": pixel_mse})
        print(f"  trial {trial:3d}: latent_mse={latent_mse:.6f}  pixel_mse={pixel_mse:.2f}")

    means = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    print(f"\nResults  |  latent_mse={means['latent_mse']:.6f}  pixel_mse={means['pixel_mse']:.2f}  ({n_trials} trial(s))")
    return means


def main():
    parser = argparse.ArgumentParser(description="WorldEngine quantization eval (forward-pass MSE)")
    parser.add_argument("model_uri", help="HF model URI or local path")
    parser.add_argument("--quant", required=True, help="Quantization scheme to evaluate (e.g. intw8a8, fp8w8a8, nvfp4)")
    parser.add_argument("--smooth", action="store_true", help="Apply SmoothQuant")
    parser.add_argument("--n-trials", type=int, default=1, help="Number of noise samples to average over (default: 1)")
    parser.add_argument("--seed-frame", type=int, default=0, choices=range(len(SEED_FRAME_URLS)),
                        help="Index into seed frame URL list (default: 0)")
    parser.add_argument("--torch-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_frame = download_seed_frame(args.seed_frame)
    print(f"Seed frame shape: {seed_frame.shape}\n")

    forward_pass_mse(
        model_uri=args.model_uri,
        quant=args.quant,
        smooth=args.smooth,
        seed_frame=seed_frame,
        n_trials=args.n_trials,
        torch_seed=args.torch_seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
