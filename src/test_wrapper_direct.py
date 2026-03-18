"""
Generate video using the PyTorch wrapper DIRECTLY on MPS.
No JIT, no CoreML - pure PyTorch inference.
This validates the wrapper logic is correct before introducing conversion.
"""
import argparse, io, random, time, urllib.request
import cv2, imageio.v3 as iio, numpy as np, torch
import torch.nn.functional as F

T = 512
N_LAYERS = 24
D_MODEL = 2048
D_HEAD = 64
D_ROPE = D_HEAD // 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="diagnostics/out/wrapper_direct.mp4")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--config-uri", default="Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from src.model import WorldModel
    from src.coreml_export import build_model, _patch_rms_norm_globally

    cfg = WorldModel.load_config(args.config_uri)

    # Load original model on CPU for fp32 host computations
    print("[gen] Loading PyTorch model on CPU for fp32 noise cond / RoPE...")
    pt_model = WorldModel.from_pretrained(args.config_uri, cfg=cfg, device="cpu", dtype=torch.float16).eval()
    noise_cond = pt_model.denoise_step_emb
    rope_xy = pt_model.transformer.rope_angles.xy.float()
    rope_inv_t = pt_model.transformer.rope_angles.inv_t.float()

    inference_fps = getattr(cfg, "inference_fps", cfg.base_fps)
    latent_fps = inference_fps / getattr(cfg, "temporal_compression", 1)
    ts_mult = int(cfg.base_fps) // latent_fps
    print(f"[gen] ts_mult = {ts_mult}")

    # Build the export wrapper on MPS
    print(f"[gen] Building StatefulWorldModelV3 on {args.device}...")
    _patch_rms_norm_globally()
    model = WorldModel.from_pretrained(args.config_uri, cfg=cfg, device=args.device, dtype=torch.float16).eval()
    from src.coreml_export import StatefulWorldModelV3
    stateful = StatefulWorldModelV3(model, cfg).to(args.device).eval()
    for p in stateful.parameters():
        p.data = p.data.to(dtype=torch.float16)
    for name, buf in stateful.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(dtype=torch.float16)
    del model
    print(f"[gen] Model ready on {args.device}")

    # Precompute sigmas
    SIGMAS = [1.0, 0.9, 0.75, 0.3, 0.0]
    DSIGMAS = [SIGMAS[i+1] - SIGMAS[i] for i in range(4)]

    def compute_cond(sigma_val):
        with torch.no_grad():
            s = torch.tensor([[sigma_val]], dtype=torch.float32)
            return noise_cond(s).half().to(args.device)

    cond_cache = {s: compute_cond(s) for s in SIGMAS}

    def compute_rope(frame_idx):
        idx = torch.arange(T, dtype=torch.long)
        x_norm = (2.0 * idx.remainder(cfg.width).float() + 1.0) / cfg.width - 1.0
        y_norm = (2.0 * idx.div(cfg.width, rounding_mode="floor").float() + 1.0) / cfg.height - 1.0
        t_val = torch.full((T,), float(frame_idx * ts_mult), dtype=torch.float32)
        freqs = torch.cat((
            x_norm.unsqueeze(-1) * rope_xy,
            y_norm.unsqueeze(-1) * rope_xy,
            t_val.unsqueeze(-1) * rope_inv_t,
        ), dim=-1)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0).half().to(args.device)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0).half().to(args.device)
        return cos, sin

    # KV cache ring buffer config
    local_L = cfg.local_window * T
    global_L = cfg.global_window * T
    period = cfg.global_attn_period
    off = getattr(cfg, "global_attn_offset", 0) % period
    layer_configs = []
    for i in range(N_LAYERS):
        is_global = ((i - off) % period == 0)
        L = global_L if is_global else local_L
        cap = L + T
        tpf = T
        dilation = cfg.global_pinned_dilation if is_global else 1
        num_buckets = (L // tpf) // dilation
        layer_configs.append({"cap": cap, "L": L, "tpf": tpf, "dilation": dilation, "num_buckets": num_buckets})

    def mask_stale_slots(frame_idx):
        saved = {}
        for i, lc in enumerate(layer_configs):
            dilation = lc["dilation"]
            if (frame_idx % dilation) != 0:
                continue
            tpf, num_buckets = lc["tpf"], lc["num_buckets"]
            bucket = (frame_idx + (dilation - 1)) // dilation
            slot = bucket % num_buckets
            ring_start = slot * tpf
            ring_end = ring_start + tpf
            blk = stateful.blocks[i]
            saved[i] = blk.written[ring_start:ring_end].clone()
            blk.written[ring_start:ring_end] = 0.0
        return saved

    def restore_written(saved):
        for i, data in saved.items():
            blk = stateful.blocks[i]
            lc = layer_configs[i]
            dilation = lc["dilation"]
            tpf, num_buckets = lc["tpf"], lc["num_buckets"]
            bucket = (0 + (dilation - 1)) // dilation  # placeholder, we saved the data
            blk.written[data.shape[0]:].clone()  # noop
            # Just restore directly from saved tensor
            # Need to recompute ring position... actually just use the data
        # Simpler: saved maps i -> original written[ring_start:ring_end]
        # But we need to know ring_start... let me refactor.

    # Actually, let me save/restore the full written buffer for simplicity
    def mask_stale_slots_v2(frame_idx):
        saved = {}
        for i, lc in enumerate(layer_configs):
            dilation = lc["dilation"]
            if (frame_idx % dilation) != 0:
                continue
            tpf, num_buckets = lc["tpf"], lc["num_buckets"]
            bucket = (frame_idx + (dilation - 1)) // dilation
            slot = bucket % num_buckets
            ring_start = slot * tpf
            ring_end = ring_start + tpf
            blk = stateful.blocks[i]
            saved[i] = (ring_start, ring_end, blk.written[ring_start:ring_end].clone())
            blk.written[ring_start:ring_end] = 0.0
        return saved

    def restore_written_v2(saved):
        for i, (rs, re, data) in saved.items():
            stateful.blocks[i].written[rs:re] = data

    def update_ring_buffer(frame_idx):
        for i, lc in enumerate(layer_configs):
            dilation = lc["dilation"]
            if (frame_idx % dilation) != 0:
                continue
            cap, L, tpf = lc["cap"], lc["L"], lc["tpf"]
            num_buckets = lc["num_buckets"]
            bucket = (frame_idx + (dilation - 1)) // dilation
            slot = bucket % num_buckets
            ring_start = slot * tpf
            ring_end = ring_start + tpf
            tail_start = L
            tail_end = L + tpf
            blk = stateful.blocks[i]
            blk.k_cache[:, :, ring_start:ring_end, :] = blk.k_cache[:, :, tail_start:tail_end, :].clone()
            blk.v_cache[:, :, ring_start:ring_end, :] = blk.v_cache[:, :, tail_start:tail_end, :].clone()
            blk.written[ring_start:ring_end] = 1.0

    # Load VAE
    print("[gen] Loading VAE...")
    from src.ae import get_ae
    with torch.device(args.device):
        vae = get_ae(cfg.ae_uri, getattr(cfg, "taehv_ae", False), dtype=torch.float16)

    # Load seed image
    SEED_URLS = [
        "https://gist.github.com/user-attachments/assets/d81c6d26-a838-4afe-9d13-fd67677043c3",
        "https://gist.github.com/user-attachments/assets/b6d18c38-098e-43b0-8e61-66a16e5d8946",
    ]
    url = random.choice(SEED_URLS)
    print(f"[gen] Loading seed frame...")
    raw = urllib.request.urlopen(url).read()
    arr = iio.imread(io.BytesIO(raw))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    seed_img = cv2.resize(arr, (1024, 512))

    print("[gen] VAE encoding seed frame...")
    seed_t = torch.from_numpy(np.repeat(seed_img[None], 4, axis=0)).to(device=args.device, dtype=torch.uint8)
    with torch.inference_mode():
        seed_latent = vae.encode(seed_t).unsqueeze(0)
    print(f"[gen] Seed latent: {seed_latent.shape}, device={seed_latent.device}")

    # append_frame: cache write with sigma=0
    frame_idx = 0
    rope_cos, rope_sin = compute_rope(frame_idx)
    cond_0 = cond_cache[0.0]
    mouse = torch.zeros(1, 1, 2, dtype=torch.float16, device=args.device)
    button = torch.zeros(1, 1, 256, dtype=torch.float16, device=args.device)
    scroll = torch.zeros(1, 1, 1, dtype=torch.float16, device=args.device)

    print("[gen] append_frame: cache write with seed latent...")
    with torch.inference_mode():
        _ = stateful(seed_latent, cond_0, rope_cos, rope_sin, mouse, button, scroll)
    update_ring_buffer(frame_idx)
    frame_idx += 1

    # Check first output stats
    print(f"[gen] First output stats: mean={_.float().mean():.4f}, std={_.float().std():.4f}")

    ctrl_seq = [
        {"mouse": (0.2, 0.2)}, {"buttons": {32}}, {}, {}, {},
        {"buttons": {1}}, {}, {}, {"buttons": {1, 32}},
        {}, {}, {}, {}, {}, {},
    ] * 2
    ctrl_seq += [{}] * 8
    ctrl_seq = ctrl_seq[:args.frames]

    print(f"[gen] Generating {len(ctrl_seq)} frames (DIRECT PyTorch on {args.device})...")
    all_latents = []
    frame_times = []

    for i, ctrl in enumerate(ctrl_seq):
        m = ctrl.get("mouse", (0, 0))
        buttons = ctrl.get("buttons", None)
        scr = ctrl.get("scroll", 0)

        mouse[0, 0, 0] = m[0]
        mouse[0, 0, 1] = m[1]
        button.zero_()
        if buttons:
            for b in buttons:
                if 0 <= b < 256:
                    button[0, 0, b] = 1.0
        scroll[0, 0, 0] = scr

        t0 = time.perf_counter()
        rope_cos, rope_sin = compute_rope(frame_idx)

        saved = mask_stale_slots_v2(frame_idx)

        x = torch.randn(1, 1, 32, 32, 64, dtype=torch.float16, device=args.device)
        with torch.inference_mode():
            for step in range(4):
                v = stateful(x, cond_cache[SIGMAS[step]], rope_cos, rope_sin, mouse, button, scroll)
                x = x + DSIGMAS[step] * v

        all_latents.append(x.clone())

        restore_written_v2(saved)

        with torch.inference_mode():
            _ = stateful(x, cond_0, rope_cos, rope_sin, mouse, button, scroll)
        update_ring_buffer(frame_idx)

        if args.device == "mps":
            torch.mps.synchronize()

        dt = (time.perf_counter() - t0) * 1000
        frame_times.append(dt)
        frame_idx += 1

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  frame {i+1}: {dt:.0f} ms ({1000/dt:.1f} FPS) | x: mean={x.float().mean():.4f} std={x.float().std():.4f}")

    mean_ms = sum(frame_times) / len(frame_times)
    print(f"\n[gen] Mean: {mean_ms:.0f} ms ({1000/mean_ms:.1f} FPS)")

    print(f"[gen] Decoding {len(all_latents)} frames with VAE...")
    with iio.imopen(args.out, "w", plugin="pyav") as out:
        for i, lat in enumerate(all_latents):
            lat_t = lat.squeeze(0)
            with torch.inference_mode():
                img = vae.decode(lat_t)
            img_np = img.cpu().numpy()
            if i == 0:
                out.write(img_np, fps=15, codec="libx264")
            else:
                out.write(img_np)
            if (i + 1) % 10 == 0:
                print(f"  decoded {i+1}/{len(all_latents)}")

    print(f"[gen] Video saved to {args.out}")


if __name__ == "__main__":
    main()
