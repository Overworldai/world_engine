"""
Generate video using the Core ML WorldModel with stateful KV cache.

Ring buffer management and stale-slot masking are handled inside the model.
The host computes noise conditioning, RoPE, and ring positions, then calls
predict() with no state round-trips needed.

Per-frame pipeline:
  1. Host: compute RoPE angles (fp32), ring positions (int math)
  2. CoreML predict x4: denoise steps (is_cache_write=0)
  3. CoreML predict x1: cache write  (is_cache_write=1)
  4. Host: VAE decode

Usage:
    PYTHONPATH=. .venv312/bin/python -m src.coreml_gen_video \
        --model diagnostics/out/world_model.mlpackage \
        --out diagnostics/out/coreml_output.mp4 \
        --frames 60
"""
import argparse
import io
import random
import time
import urllib.request

import cv2
import imageio.v3 as iio
import numpy as np
import torch
import coremltools as ct


SEED_URLS = [
    "https://gist.github.com/user-attachments/assets/d81c6d26-a838-4afe-9d13-fd67677043c3",
    "https://gist.github.com/user-attachments/assets/b6d18c38-098e-43b0-8e61-66a16e5d8946",
    "https://gist.github.com/user-attachments/assets/0734a8c1-3eb4-4ffe-8c37-5665c45ab559",
    "https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed",
    "https://gist.github.com/user-attachments/assets/68c943a4-008a-4c25-948c-c81ab4c47d21",
]

SIGMAS = [1.0, 0.9, 0.75, 0.3, 0.0]
DSIGMAS = [SIGMAS[i+1] - SIGMAS[i] for i in range(4)]

T = 512
N_LAYERS = 24


def compute_cond_host(sigma_val, noise_cond_module):
    with torch.no_grad():
        return noise_cond_module(torch.tensor([[sigma_val]], dtype=torch.float32)).half().numpy()


def compute_rope_host(frame_idx, ts_mult, rope_xy, rope_inv_t, cfg):
    tokens = cfg.height * cfg.width
    idx = torch.arange(tokens, dtype=torch.long)
    x_norm = (2.0 * idx.remainder(cfg.width).float() + 1.0) / cfg.width - 1.0
    y_norm = (2.0 * idx.div(cfg.width, rounding_mode="floor").float() + 1.0) / cfg.height - 1.0
    t_val = torch.full((tokens,), float(frame_idx * ts_mult), dtype=torch.float32)
    freqs = torch.cat((
        x_norm.unsqueeze(-1) * rope_xy,
        y_norm.unsqueeze(-1) * rope_xy,
        t_val.unsqueeze(-1) * rope_inv_t,
    ), dim=-1)
    return (freqs.cos().unsqueeze(0).unsqueeze(0).half().numpy(),
            freqs.sin().unsqueeze(0).unsqueeze(0).half().numpy())


def compute_ring_positions(frame_idx, cfg):
    """Compute ring buffer positions on the host (trivial integer math)."""
    # Local layers: dilation=1, num_buckets = local_window
    local_num_buckets = cfg.local_window  # L/T/dil = (16*512)/512/1 = 16
    ring_start_local = (frame_idx % local_num_buckets) * T

    # Global layers: dilation=8, num_buckets = global_window/dilation = 128/8 = 16
    global_dilation = cfg.global_pinned_dilation
    global_num_buckets = cfg.global_window // global_dilation  # 128/8 = 16
    bucket_global = (frame_idx + (global_dilation - 1)) // global_dilation
    ring_start_global = (bucket_global % global_num_buckets) * T

    write_step_global = 1.0 if (frame_idx % global_dilation) == 0 else 0.0

    return (np.array([float(ring_start_local)], dtype=np.float16),
            np.array([float(ring_start_global)], dtype=np.float16),
            np.array([write_step_global], dtype=np.float16))


def model_predict(mlmodel, state, x, cond_np, rope_cos_np, rope_sin_np,
                  is_cache_write, ring_start_local, ring_start_global, write_step_global,
                  mouse=(0, 0), buttons=None, scroll=0):
    inputs = {
        "x": np.array(x, dtype=np.float16),
        "cond": cond_np,
        "rope_cos": rope_cos_np,
        "rope_sin": rope_sin_np,
        "mouse": np.array([[[float(mouse[0]), float(mouse[1])]]], dtype=np.float16),
        "button": np.zeros((1, 1, 256), dtype=np.float16),
        "scroll": np.array([[[float(scroll)]]], dtype=np.float16),
        "is_cache_write": np.array([float(is_cache_write)], dtype=np.float16),
        "ring_start_local": ring_start_local,
        "ring_start_global": ring_start_global,
        "write_step_global": write_step_global,
    }
    if buttons:
        for b in buttons:
            if 0 <= b < 256:
                inputs["button"][0, 0, b] = 1.0
    return mlmodel.predict(inputs, state=state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="diagnostics/out/coreml_output.mp4")
    parser.add_argument("--seed-url", default="")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--config-uri", default="Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill")
    args = parser.parse_args()

    from src.model import WorldModel
    cfg = WorldModel.load_config(args.config_uri)

    inference_fps = getattr(cfg, "inference_fps", cfg.base_fps)
    latent_fps = inference_fps / getattr(cfg, "temporal_compression", 1)
    ts_mult = int(cfg.base_fps) // latent_fps
    print(f"[gen] ts_mult = {ts_mult}")

    print("[gen] Loading PyTorch model for host-side fp32 computation...")
    pt_model = WorldModel.from_pretrained(args.config_uri, cfg=cfg, device="cpu", dtype=torch.float16).eval()
    noise_cond = pt_model.denoise_step_emb
    rope_xy = pt_model.transformer.rope_angles.xy.float()
    rope_inv_t = pt_model.transformer.rope_angles.inv_t.float()

    print("[gen] Precomputing noise conditioning (fp32)...")
    cond_cache = {s: compute_cond_host(s, noise_cond) for s in SIGMAS}

    print("[gen] Loading Core ML model...")
    mlmodel = ct.models.MLModel(args.model, compute_units=ct.ComputeUnit.ALL)
    state = mlmodel.make_state()

    # Initialize written state: tail must be 1.0
    print("[gen] Initializing written state...")
    local_L = cfg.local_window * T
    global_L = cfg.global_window * T
    period = cfg.global_attn_period
    off = getattr(cfg, "global_attn_offset", 0) % period
    for i in range(N_LAYERS):
        is_global = ((i - off) % period == 0)
        cap = (global_L if is_global else local_L) + T
        w_key = f"blocks_{i}_written"
        w_np = state.read_state(w_key)
        w_np[cap - T:cap] = 1.0
        state.write_state(w_key, w_np)

    print("[gen] Loading VAE...")
    from src.ae import get_ae
    with torch.device("mps"):
        vae = get_ae(cfg.ae_uri, getattr(cfg, "taehv_ae", False), dtype=torch.float16)

    url = args.seed_url if args.seed_url else random.choice(SEED_URLS)
    print("[gen] Loading seed frame...")
    raw = urllib.request.urlopen(url).read()
    arr = iio.imread(io.BytesIO(raw))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    seed_img = cv2.resize(arr, (1024, 512))

    print("[gen] VAE encoding seed frame...")
    seed_t = torch.from_numpy(np.repeat(seed_img[None], 4, axis=0)).to(device="mps", dtype=torch.uint8)
    with torch.inference_mode():
        seed_latent = vae.encode(seed_t).unsqueeze(0)
    seed_np = seed_latent.cpu().numpy().astype(np.float16)

    # append_frame: cache write with seed latent
    frame_idx = 0
    cond_0 = cond_cache[0.0]
    rope_cos, rope_sin = compute_rope_host(frame_idx, ts_mult, rope_xy, rope_inv_t, cfg)
    rsl, rsg, wsg = compute_ring_positions(frame_idx, cfg)
    print("[gen] append_frame: cache write with seed latent...")
    model_predict(mlmodel, state, seed_np, cond_0, rope_cos, rope_sin,
                  is_cache_write=1.0, ring_start_local=rsl, ring_start_global=rsg, write_step_global=wsg)
    frame_idx += 1

    ctrl_seq = [
        {"mouse": (0.2, 0.2)}, {"buttons": {32}}, {}, {}, {},
        {"buttons": {1}}, {}, {}, {"buttons": {1, 32}},
        {}, {}, {}, {}, {}, {},
    ] * 4
    ctrl_seq += [{}] * 8
    ctrl_seq += [{"buttons": {32}}] * 10
    ctrl_seq += [{"buttons": {65}}] * 10
    ctrl_seq += [{"buttons": {68}}] * 10
    ctrl_seq += [{"buttons": {83}}] * 10
    ctrl_seq += [{}] * 10
    ctrl_seq = ctrl_seq[:args.frames]

    print(f"[gen] Generating {len(ctrl_seq)} frames...")
    all_latents = []
    frame_times = []

    for i, ctrl in enumerate(ctrl_seq):
        mouse = ctrl.get("mouse", (0, 0))
        buttons = ctrl.get("buttons", None)
        scroll = ctrl.get("scroll", 0)

        t0 = time.perf_counter()

        rope_cos, rope_sin = compute_rope_host(frame_idx, ts_mult, rope_xy, rope_inv_t, cfg)
        rsl, rsg, wsg = compute_ring_positions(frame_idx, cfg)

        # Denoise pass (4 steps)
        x = np.random.randn(1, 1, 32, 32, 64).astype(np.float16)
        for step in range(4):
            pred = model_predict(mlmodel, state, x, cond_cache[SIGMAS[step]], rope_cos, rope_sin,
                                 is_cache_write=0.0, ring_start_local=rsl, ring_start_global=rsg,
                                 write_step_global=wsg, mouse=mouse, buttons=buttons, scroll=scroll)
            v_key = list(pred.keys())[0]
            x = x + DSIGMAS[step] * pred[v_key]

        all_latents.append(x)

        # Cache write pass
        model_predict(mlmodel, state, x, cond_0, rope_cos, rope_sin,
                      is_cache_write=1.0, ring_start_local=rsl, ring_start_global=rsg,
                      write_step_global=wsg, mouse=mouse, buttons=buttons, scroll=scroll)

        dt = (time.perf_counter() - t0) * 1000
        frame_times.append(dt)
        frame_idx += 1

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  frame {i+1}: {dt:.0f} ms ({1000/dt:.1f} FPS)")

    mean_ms = sum(frame_times) / len(frame_times)
    print(f"\n[gen] Mean: {mean_ms:.0f} ms ({1000/mean_ms:.1f} FPS)")

    print(f"[gen] Decoding {len(all_latents)} frames with VAE...")
    with iio.imopen(args.out, "w", plugin="pyav") as out:
        for i, lat in enumerate(all_latents):
            lat_t = torch.from_numpy(lat).to(device="mps", dtype=torch.float16).squeeze(0)
            with torch.inference_mode():
                img = vae.decode(lat_t)
            img_np = img.cpu().numpy()
            if i == 0:
                out.write(img_np, fps=15, codec="libx264")
            else:
                out.write(img_np)
            if (i + 1) % 20 == 0:
                print(f"  decoded {i+1}/{len(all_latents)}")

    print(f"[gen] Video saved to {args.out}")


if __name__ == "__main__":
    main()
