"""
Generate video using the Core ML WorldModel with stateful KV cache.

Per-frame pipeline:
1. Host: compute cond (fp32), RoPE angles (fp32), mask stale ring slots
2. CoreML predict x4: denoise steps (frozen attention)
3. Host: restore written state
4. CoreML predict x1: cache write (sigma=0, unfrozen)
5. Host: copy tail -> ring slot, update written
6. Host: VAE decode

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
N_KV_HEADS = 32
D_HEAD = 64
D_ROPE = D_HEAD // 2  # 32


def get_layer_config(cfg):
    local_L = cfg.local_window * T
    global_L = cfg.global_window * T
    period = cfg.global_attn_period
    off = getattr(cfg, "global_attn_offset", 0) % period
    layers = []
    for i in range(N_LAYERS):
        is_global = ((i - off) % period == 0)
        L = global_L if is_global else local_L
        cap = L + T
        tpf = T
        num_buckets = (L // tpf) // (cfg.global_pinned_dilation if is_global else 1)
        dilation = cfg.global_pinned_dilation if is_global else 1
        layers.append({
            "cap": cap, "L": L, "tpf": tpf,
            "num_buckets": num_buckets, "dilation": dilation,
            "is_global": is_global,
        })
    return layers


def compute_cond_host(sigma_val, noise_cond_module):
    """Compute noise conditioning in fp32 on CPU, matching CUDA NoiseConditioner."""
    with torch.no_grad():
        s = torch.tensor([[sigma_val]], dtype=torch.float32)
        cond = noise_cond_module(s)  # fp32 computation, returns fp32
    return cond.half().numpy()  # [1, 1, D_MODEL] as fp16


def compute_rope_host(frame_idx, ts_mult, rope_xy, rope_inv_t, cfg):
    """Compute RoPE angles in fp32 on CPU, matching CUDA OrthoRoPEAngles."""
    tokens = cfg.height * cfg.width
    idx = torch.arange(tokens, dtype=torch.long)
    x_pos = idx.remainder(cfg.width)
    y_pos = idx.div(cfg.width, rounding_mode="floor")

    x_norm = (2.0 * x_pos.float() + 1.0) / cfg.width - 1.0
    y_norm = (2.0 * y_pos.float() + 1.0) / cfg.height - 1.0
    t_val = torch.full((tokens,), float(frame_idx * ts_mult), dtype=torch.float32)

    freqs = torch.cat(
        (x_norm.unsqueeze(-1) * rope_xy,
         y_norm.unsqueeze(-1) * rope_xy,
         t_val.unsqueeze(-1) * rope_inv_t),
        dim=-1,
    )  # [T, D_ROPE]

    cos_np = freqs.cos().unsqueeze(0).unsqueeze(0).half().numpy()  # [1, 1, T, D_ROPE]
    sin_np = freqs.sin().unsqueeze(0).unsqueeze(0).half().numpy()
    return cos_np, sin_np


def model_predict(mlmodel, state, x, cond_np, rope_cos_np, rope_sin_np,
                  mouse=(0, 0), buttons=None, scroll=0):
    inputs = {
        "x": np.array(x, dtype=np.float16),
        "cond": cond_np,
        "rope_cos": rope_cos_np,
        "rope_sin": rope_sin_np,
        "mouse": np.array([[[float(mouse[0]), float(mouse[1])]]], dtype=np.float16),
        "button": np.zeros((1, 1, 256), dtype=np.float16),
        "scroll": np.array([[[float(scroll)]]], dtype=np.float16),
    }
    if buttons:
        for b in buttons:
            if 0 <= b < 256:
                inputs["button"][0, 0, b] = 1.0
    return mlmodel.predict(inputs, state=state)


def mask_stale_slots(state, layer_configs, frame_idx):
    """Before denoise: mask out stale ring slots (matching CUDA frozen upsert behavior)."""
    saved = {}
    for i, lc in enumerate(layer_configs):
        dilation = lc["dilation"]
        write_step = (frame_idx % dilation) == 0
        if not write_step:
            continue

        tpf = lc["tpf"]
        num_buckets = lc["num_buckets"]

        bucket = (frame_idx + (dilation - 1)) // dilation
        slot = bucket % num_buckets
        ring_start = slot * tpf
        ring_end = ring_start + tpf

        w_key = f"blocks_{i}_written"
        w_np = np.array(state.read_state(w_key), copy=True)
        saved[w_key] = np.array(w_np, copy=True)
        w_np[ring_start:ring_end] = 0.0
        state.write_state(w_key, w_np)

    return saved


def restore_written(state, saved):
    """After denoise: restore the original written state."""
    for k, v in saved.items():
        state.write_state(k, v)


def update_ring_buffer(state, layer_configs, frame_idx):
    """After a cache-write predict, copy tail K/V -> ring slot and update written."""
    for i, lc in enumerate(layer_configs):
        cap = lc["cap"]
        L = lc["L"]
        tpf = lc["tpf"]
        dilation = lc["dilation"]
        num_buckets = lc["num_buckets"]

        write_step = (frame_idx % dilation) == 0
        if not write_step:
            continue

        bucket = (frame_idx + (dilation - 1)) // dilation
        slot = bucket % num_buckets
        ring_start = slot * tpf
        ring_end = ring_start + tpf
        tail_start = L
        tail_end = L + tpf

        k_key = f"blocks_{i}_k_cache"
        v_key = f"blocks_{i}_v_cache"
        w_key = f"blocks_{i}_written"

        k_np = state.read_state(k_key)
        v_np = state.read_state(v_key)
        w_np = state.read_state(w_key)

        k_np[:, :, ring_start:ring_end, :] = k_np[:, :, tail_start:tail_end, :]
        v_np[:, :, ring_start:ring_end, :] = v_np[:, :, tail_start:tail_end, :]
        w_np[ring_start:ring_end] = 1.0

        state.write_state(k_key, k_np)
        state.write_state(v_key, v_np)
        state.write_state(w_key, w_np)


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
    layer_configs = get_layer_config(cfg)

    # Compute ts_mult same as CUDA WorldEngine
    inference_fps = getattr(cfg, "inference_fps", cfg.base_fps)
    latent_fps = inference_fps / getattr(cfg, "temporal_compression", 1)
    ts_mult = int(cfg.base_fps) // latent_fps
    print(f"[gen] ts_mult = {ts_mult}")

    # Load PyTorch model on CPU to extract fp32 components
    print("[gen] Loading PyTorch model for host-side fp32 computation...")
    pt_model = WorldModel.from_pretrained(args.config_uri, cfg=cfg, device="cpu", dtype=torch.float16).eval()
    noise_cond = pt_model.denoise_step_emb  # NoiseConditioner (fp32 weights via NoCastModule)
    rope_xy = pt_model.transformer.rope_angles.xy.float()   # [d_xy] fp32
    rope_inv_t = pt_model.transformer.rope_angles.inv_t.float()  # [d_t] fp32

    # Precompute noise conditioning for all sigma values
    print("[gen] Precomputing noise conditioning (fp32)...")
    cond_cache = {}
    for s in SIGMAS:
        cond_cache[s] = compute_cond_host(s, noise_cond)
        print(f"  sigma={s:.2f}: cond stats mean={cond_cache[s].mean():.4f} std={cond_cache[s].std():.4f}")

    print("[gen] Loading Core ML model...")
    mlmodel = ct.models.MLModel(args.model, compute_units=ct.ComputeUnit.ALL)
    state = mlmodel.make_state()

    # CoreML make_state() initializes ALL buffers to zero, but the model expects
    # the tail of each written buffer to be 1.0 (so it can attend to its own
    # current-frame KV). Without this, the attention mask is all-False and the
    # model produces garbage output.
    print("[gen] Initializing written state (marking tail as attended)...")
    for i, lc in enumerate(layer_configs):
        cap = lc["cap"]
        w_key = f"blocks_{i}_written"
        w_np = state.read_state(w_key)
        w_np[cap - T:cap] = 1.0
        state.write_state(w_key, w_np)

    print("[gen] Loading VAE...")
    from src.ae import get_ae
    with torch.device("mps"):
        vae = get_ae(cfg.ae_uri, getattr(cfg, "taehv_ae", False), dtype=torch.float16)

    url = args.seed_url if args.seed_url else random.choice(SEED_URLS)
    print(f"[gen] Loading seed frame...")
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
    print(f"[gen] Seed latent: {seed_np.shape}")

    # append_frame: cache write with seed latent (sigma=0)
    print("[gen] append_frame: cache write with seed latent...")
    frame_idx = 0
    cond_0 = cond_cache[0.0]
    rope_cos, rope_sin = compute_rope_host(frame_idx, ts_mult, rope_xy, rope_inv_t, cfg)
    model_predict(mlmodel, state, seed_np, cond_0, rope_cos, rope_sin)
    update_ring_buffer(state, layer_configs, frame_idx)
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

        # Compute RoPE for this frame (fp32 on host)
        rope_cos, rope_sin = compute_rope_host(frame_idx, ts_mult, rope_xy, rope_inv_t, cfg)

        # Mask stale ring slots before denoise
        saved_written = mask_stale_slots(state, layer_configs, frame_idx)

        # Denoise pass (4 steps, frozen)
        x = np.random.randn(1, 1, 32, 32, 64).astype(np.float16)
        for step in range(4):
            cond_np = cond_cache[SIGMAS[step]]
            pred = model_predict(mlmodel, state, x, cond_np, rope_cos, rope_sin,
                                 mouse, buttons, scroll)
            v_key = list(pred.keys())[0]
            x = x + DSIGMAS[step] * pred[v_key]

        all_latents.append(x)

        # Restore written state after denoise
        restore_written(state, saved_written)

        # Cache write pass (sigma=0, unfrozen)
        model_predict(mlmodel, state, x, cond_0, rope_cos, rope_sin, mouse, buttons, scroll)
        update_ring_buffer(state, layer_configs, frame_idx)

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
