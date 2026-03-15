import argparse
import io
import random
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


def _controller_sequence(CtrlInput):
    seq = [
        CtrlInput(mouse=[0.2, 0.2]), CtrlInput(button={32}), CtrlInput(), CtrlInput(), CtrlInput(),
        CtrlInput(button={1}), CtrlInput(), CtrlInput(), CtrlInput(button={1, 32}),
        CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(),
    ] * 2
    seq += [CtrlInput()] * 8
    return seq


def _sync_if_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Simple deterministic WorldEngine generator.")
    parser.add_argument("--model-uri", default="Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill")
    parser.add_argument("--out", default="out_simple.mp4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--attention-backend", default="metal", choices=["metal", "flex", "auto"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--scheduler-steps", type=int, default=4)
    parser.add_argument("--cache-interval", type=int, default=1)
    parser.add_argument("--seed-url", default="")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend not available.")

    import os
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

    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    engine = WorldEngine(
        args.model_uri,
        quant=None,
        device=args.device,
        dtype=dtype,
        scheduler_steps=args.scheduler_steps,
        cache_interval=args.cache_interval,
    )

    url = args.seed_url if args.seed_url else random.choice(SEED_FRAME_URLS)
    frame = _load_seed_frame(url)
    seed = torch.from_numpy(np.repeat(frame[None], 4, axis=0)).to(engine.device)
    engine.append_frame(seed)

    ctrl_seq = _controller_sequence(CtrlInput)
    if args.frames > 0:
        ctrl_seq = ctrl_seq[:args.frames]

    with iio.imopen(args.out, "w", plugin="pyav") as out:
        first = engine.gen_frame()
        _sync_if_mps()
        out.write(first.cpu().numpy(), fps=60, codec="libx264")
        for ctrl in ctrl_seq:
            img = engine.gen_frame(ctrl=ctrl)
            _sync_if_mps()
            out.write(img.cpu().numpy())

    print(f"wrote={args.out}")
    print(f"seed_url={url}")


if __name__ == "__main__":
    main()
