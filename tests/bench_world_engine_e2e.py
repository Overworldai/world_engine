import argparse
import io
import random
import time
import urllib.request
from pathlib import Path
import sys

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SEED_FRAME_URLS = [
    "https://gist.github.com/user-attachments/assets/d81c6d26-a838-4afe-9d13-fd67677043c3",
    "https://gist.github.com/user-attachments/assets/b6d18c38-098e-43b0-8e61-66a16e5d8946",
    "https://gist.github.com/user-attachments/assets/0734a8c1-3eb4-4ffe-8c37-5665c45ab559",
    "https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed",
    "https://gist.github.com/user-attachments/assets/68c943a4-008a-4c25-948c-c81ab4c47d21",
]


def _controller_sequence(CtrlInput):
    seq = [
        CtrlInput(mouse=[0.2, 0.2]), CtrlInput(button={32}), CtrlInput(), CtrlInput(), CtrlInput(),
        CtrlInput(button={1}), CtrlInput(), CtrlInput(), CtrlInput(button={1, 32}),
        CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(),
    ] * 4
    seq += [CtrlInput()] * 8
    seq += (
        [CtrlInput(button={32})] * 10 +
        [CtrlInput(button={65})] * 10 +
        [CtrlInput(button={68})] * 10 +
        [CtrlInput(button={83})] * 10
    )
    seq += [CtrlInput()] * 10
    return seq


def _sync_if_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _load_seed_frame(url: str) -> np.ndarray:
    raw = urllib.request.urlopen(url).read()
    arr = iio.imread(io.BytesIO(raw))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    t = F.interpolate(t, size=(512, 1024), mode="bilinear", align_corners=False)
    t = t.round().clamp(0, 255).to(torch.uint8)
    return t.squeeze(0).permute(1, 2, 0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="WorldEngine E2E generation script + latency stats.")
    parser.add_argument("--model-uri", default="Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill")
    parser.add_argument("--out", default="out.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=0, help="Override number of generated control frames (0=full sequence)")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--attention-backend", default="metal", choices=["metal", "flex", "auto"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--write-video", action="store_true", help="Write generated frames to --out via pyav")
    parser.add_argument("--return-img", action="store_true", help="Decode RGB images (otherwise benchmark latent-only)")
    parser.add_argument("--scheduler-steps", type=int, default=0, help="Override denoise scheduler steps (0=use model default)")
    parser.add_argument("--cache-interval", type=int, default=1, help="Run cache update every N generated frames")
    parser.add_argument("--quant", default="none", choices=["none", "w8a8", "nvfp4"], help="Optional model quantization mode")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend not available.")

    import os
    # torch.compile can be a net loss / long cold-start on MPS for this workload.
    if args.device == "mps":
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        os.environ.setdefault("WORLD_KV_RUNTIME_CHECKS", "0")
        os.environ.setdefault("WORLD_KV_COMPUTE_ACTIVE_BLOCKS", "0")
    os.environ["WORLD_ATTENTION_BACKEND"] = args.attention_backend
    from src.world_engine import WorldEngine, CtrlInput
    from src.metal.runtime import ensure_metal_attention_op_loaded
    if args.attention_backend == "metal" and args.device == "mps":
        os.environ.setdefault("WORLD_METAL_IMPL", "fast")
        os.environ.setdefault("WORLD_METAL_FAST_NO_FALLBACK", "1")
        os.environ.setdefault("WORLD_METAL_PREFER_ACTIVE_DISPATCH", "1")
        ensure_metal_attention_op_loaded()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(
        f"[e2e] cfg backend={args.attention_backend} device={args.device} dtype={args.dtype} "
        f"compile_disabled={os.environ.get('TORCHDYNAMO_DISABLE', '0')} "
        f"metal_impl={os.environ.get('WORLD_METAL_IMPL', 'ref')} "
        f"metal_no_fallback={os.environ.get('WORLD_METAL_FAST_NO_FALLBACK', '0')} "
        f"kv_checks={os.environ.get('WORLD_KV_RUNTIME_CHECKS', '0')} "
        f"kv_active_blocks={os.environ.get('WORLD_KV_COMPUTE_ACTIVE_BLOCKS', '0')}",
        flush=True,
    )
    print("[e2e] initializing engine...", flush=True)
    engine = WorldEngine(
        args.model_uri,
        quant=(None if args.quant == "none" else args.quant),
        device=args.device,
        dtype=dtype,
        scheduler_steps=(args.scheduler_steps if args.scheduler_steps > 0 else None),
        cache_interval=args.cache_interval,
    )
    print("[e2e] engine initialized", flush=True)
    print(
        f"[e2e] model n_layers={engine.model_cfg.n_layers} "
        f"n_heads={engine.model_cfg.n_heads} n_kv_heads={getattr(engine.model_cfg, 'n_kv_heads', engine.model_cfg.n_heads)} "
        f"scheduler_steps={int(engine.scheduler_sigmas.numel())} cache_interval={engine.cache_interval}",
        flush=True,
    )

    random.seed(args.seed)
    url = random.choice(SEED_FRAME_URLS)
    print("[e2e] loading seed frame...", flush=True)
    frame = _load_seed_frame(url)
    seed = torch.from_numpy(np.repeat(frame[None], 4, axis=0)).to(engine.device)
    print("[e2e] appending seed frame...", flush=True)
    engine.append_frame(seed)
    print("[e2e] seed frame appended", flush=True)

    ctrl_seq = _controller_sequence(CtrlInput)
    if args.frames > 0:
        ctrl_seq = ctrl_seq[: args.frames]

    totals_ms = []
    prep_ms = []
    denoise_ms = []
    cache_ms = []
    decode_ms = []

    def _step(ctrl):
        with torch.inference_mode():
            x = torch.randn(engine.frm_shape, device=engine.device, dtype=engine.dtype)

            t0 = time.perf_counter()
            inputs = engine.prep_inputs(x=x, ctrl=ctrl if ctrl is not None else CtrlInput())
            _sync_if_mps()
            t_prep = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            x0 = engine._denoise_pass_fn(x, inputs, engine.kv_cache)
            _sync_if_mps()
            t_denoise = (time.perf_counter() - t1) * 1000.0

            do_cache_update = (engine._gen_count % engine.cache_interval) == 0
            t2 = time.perf_counter()
            if do_cache_update:
                engine._cache_pass_fn(x0, inputs, engine.kv_cache)
                _sync_if_mps()
            t_cache = (time.perf_counter() - t2) * 1000.0
            engine._gen_count += 1

            t_decode = 0.0
            img = None
            if args.return_img or args.write_video:
                t3 = time.perf_counter()
                img = engine.vae.decode(x0.squeeze(1))
                _sync_if_mps()
                t_decode = (time.perf_counter() - t3) * 1000.0
            return img, t_prep, t_denoise, t_cache, t_decode, do_cache_update

    out = None
    if args.write_video:
        print("[e2e] opening video writer...", flush=True)
        out = iio.imopen(args.out, "w", plugin="pyav")

    try:
        steps = [None] + ctrl_seq
        for i, ctrl in enumerate(steps):
            label = "first" if i == 0 else f"ctrl_{i}"
            print(f"[e2e] generating {label} frame...", flush=True)
            t_total_start = time.perf_counter()
            img, t_prep, t_denoise, t_cache, t_decode, do_cache_update = _step(ctrl)
            t_total = (time.perf_counter() - t_total_start) * 1000.0

            totals_ms.append(t_total)
            prep_ms.append(t_prep)
            denoise_ms.append(t_denoise)
            cache_ms.append(t_cache)
            decode_ms.append(t_decode)

            if out is not None and img is not None:
                if i == 0:
                    out.write(img.cpu().numpy(), fps=60, codec="libx264")
                else:
                    out.write(img.cpu().numpy())
            print(
                f"[e2e] {label} done total={t_total:.3f}ms prep={t_prep:.3f} "
                f"denoise={t_denoise:.3f} cache={t_cache:.3f} decode={t_decode:.3f} "
                f"cache_update={int(do_cache_update)}",
                flush=True,
            )
    finally:
        if out is not None:
            out.close()

    def _summary(values):
        vals = sorted(values)
        n = len(vals)
        p50 = vals[n // 2]
        p95 = vals[max(0, int(0.95 * n) - 1)]
        mean = sum(vals) / max(1, n)
        return p50, p95, mean

    n = len(totals_ms)
    total_p50, total_p95, total_mean = _summary(totals_ms)
    prep_p50, prep_p95, prep_mean = _summary(prep_ms)
    den_p50, den_p95, den_mean = _summary(denoise_ms)
    cache_p50, cache_p95, cache_mean = _summary(cache_ms)
    dec_p50, dec_p95, dec_mean = _summary(decode_ms)

    print(
        f"model={args.model_uri} backend={args.attention_backend} device={args.device} "
        f"dtype={args.dtype} frames={n} return_img={args.return_img} write_video={args.write_video} "
        f"scheduler_steps={int(engine.scheduler_sigmas.numel())} cache_interval={args.cache_interval} quant={args.quant}"
    )
    print(f"total_ms   p50={total_p50:.3f} p95={total_p95:.3f} mean={total_mean:.3f}")
    print(f"prep_ms    p50={prep_p50:.3f} p95={prep_p95:.3f} mean={prep_mean:.3f}")
    print(f"denoise_ms p50={den_p50:.3f} p95={den_p95:.3f} mean={den_mean:.3f}")
    print(f"cache_ms   p50={cache_p50:.3f} p95={cache_p95:.3f} mean={cache_mean:.3f}")
    print(f"decode_ms  p50={dec_p50:.3f} p95={dec_p95:.3f} mean={dec_mean:.3f}")
    print(
        f"fps        p50={1000.0/max(total_p50,1e-9):.2f} "
        f"p95={1000.0/max(total_p95,1e-9):.2f} mean={1000.0/max(total_mean,1e-9):.2f}"
    )
    if args.write_video:
        print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
