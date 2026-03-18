"""Generate multiple CoreML video variants for comparison."""
import argparse, io, random, time, urllib.request
import cv2, imageio.v3 as iio, numpy as np, torch, coremltools as ct

from src.coreml_gen_video import (
    get_layer_config, compute_cond_host, compute_rope_host,
    model_predict, mask_stale_slots, restore_written, update_ring_buffer,
    SIGMAS, DSIGMAS, SEED_URLS, T, N_LAYERS,
)


def generate_video(mlmodel_path, out_path, config_uri, frames, compute_units, init_written, label):
    from src.model import WorldModel
    cfg = WorldModel.load_config(config_uri)
    layer_configs = get_layer_config(cfg)
    inference_fps = getattr(cfg, "inference_fps", cfg.base_fps)
    latent_fps = inference_fps / getattr(cfg, "temporal_compression", 1)
    ts_mult = int(cfg.base_fps) // latent_fps

    pt = WorldModel.from_pretrained(config_uri, cfg=cfg, device="cpu", dtype=torch.float16).eval()
    noise_cond = pt.denoise_step_emb
    rope_xy = pt.transformer.rope_angles.xy.float()
    rope_inv_t = pt.transformer.rope_angles.inv_t.float()
    cond_cache = {s: compute_cond_host(s, noise_cond) for s in SIGMAS}

    print(f"\n[{label}] Loading CoreML model (compute_units={compute_units})...")
    cu = {"ALL": ct.ComputeUnit.ALL, "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
          "CPU_ONLY": ct.ComputeUnit.CPU_ONLY, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE}
    mlmodel = ct.models.MLModel(mlmodel_path, compute_units=cu[compute_units])
    state = mlmodel.make_state()

    if init_written:
        local_L = cfg.local_window * T
        global_L = cfg.global_window * T
        period = cfg.global_attn_period
        off = getattr(cfg, "global_attn_offset", 0) % period
        for i in range(N_LAYERS):
            is_global = ((i - off) % period == 0)
            cap = (global_L if is_global else local_L) + T
            w = state.read_state(f"blocks_{i}_written")
            w[cap - T:] = 1.0
            state.write_state(f"blocks_{i}_written", w)
        print(f"[{label}] Written state initialized (tail=1.0)")
    else:
        print(f"[{label}] Written state NOT initialized (all zeros)")

    from src.ae import get_ae
    with torch.device("mps"):
        vae = get_ae(cfg.ae_uri, getattr(cfg, "taehv_ae", False), dtype=torch.float16)

    url = random.choice(SEED_URLS)
    raw = urllib.request.urlopen(url).read()
    arr = iio.imread(io.BytesIO(raw))
    if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
    if arr.shape[-1] > 3: arr = arr[..., :3]
    seed_img = cv2.resize(arr, (1024, 512))

    seed_t = torch.from_numpy(np.repeat(seed_img[None], 4, axis=0)).to(device="mps", dtype=torch.uint8)
    with torch.inference_mode():
        seed_latent = vae.encode(seed_t).unsqueeze(0)
    seed_np = seed_latent.cpu().numpy().astype(np.float16)

    frame_idx = 0
    cond_0 = cond_cache[0.0]
    rc, rs = compute_rope_host(frame_idx, ts_mult, rope_xy, rope_inv_t, cfg)
    model_predict(mlmodel, state, seed_np, cond_0, rc, rs)
    update_ring_buffer(state, layer_configs, frame_idx)
    frame_idx += 1

    ctrl_seq = ([{"mouse": (0.2, 0.2)}, {"buttons": {32}}, {}, {}, {},
                 {"buttons": {1}}, {}, {}, {"buttons": {1, 32}}, {}, {}, {}, {}, {}, {}] * 4
                + [{}] * 8 + [{"buttons": {32}}] * 10 + [{"buttons": {65}}] * 10
                + [{"buttons": {68}}] * 10 + [{"buttons": {83}}] * 10 + [{}] * 10)[:frames]

    all_latents, frame_times = [], []
    for i, ctrl in enumerate(ctrl_seq):
        mouse = ctrl.get("mouse", (0, 0))
        buttons = ctrl.get("buttons", None)
        scroll = ctrl.get("scroll", 0)
        t0 = time.perf_counter()
        rc, rs = compute_rope_host(frame_idx, ts_mult, rope_xy, rope_inv_t, cfg)
        saved = mask_stale_slots(state, layer_configs, frame_idx)
        x = np.random.randn(1, 1, 32, 32, 64).astype(np.float16)
        for step in range(4):
            pred = model_predict(mlmodel, state, x, cond_cache[SIGMAS[step]], rc, rs, mouse, buttons, scroll)
            vk = list(pred.keys())[0]
            x = x + DSIGMAS[step] * pred[vk]
        all_latents.append(x)
        restore_written(state, saved)
        model_predict(mlmodel, state, x, cond_0, rc, rs, mouse, buttons, scroll)
        update_ring_buffer(state, layer_configs, frame_idx)
        dt = (time.perf_counter() - t0) * 1000
        frame_times.append(dt)
        frame_idx += 1
        nan_count = np.isnan(x).sum()
        if (i+1) % 10 == 0 or i == 0:
            print(f"  [{label}] frame {i+1}: {dt:.0f}ms ({1000/dt:.1f}FPS) nan={nan_count}")

    mean_ms = sum(frame_times) / len(frame_times)
    print(f"  [{label}] Mean: {mean_ms:.0f}ms ({1000/mean_ms:.1f}FPS)")

    with iio.imopen(out_path, "w", plugin="pyav") as out:
        for i, lat in enumerate(all_latents):
            lat_t = torch.from_numpy(lat).to(device="mps", dtype=torch.float16).squeeze(0)
            with torch.inference_mode():
                img = vae.decode(lat_t)
            if i == 0:
                out.write(img.cpu().numpy(), fps=15, codec="libx264")
            else:
                out.write(img.cpu().numpy())
    print(f"  [{label}] Saved: {out_path}")


if __name__ == "__main__":
    URI = "Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill"
    MODEL = "diagnostics/out/world_model_v3.mlpackage"

    # v3e_cpugpu: CPU+GPU only (no ANE)
    generate_video(MODEL, "diagnostics/out/coreml_v3e_cpugpu.mp4", URI, 30, "CPU_AND_GPU", True, "v3e_cpugpu")

    # v3g_noinit: NO written state init (control test)
    generate_video(MODEL, "diagnostics/out/coreml_v3g_noinit.mp4", URI, 30, "ALL", False, "v3g_noinit")
