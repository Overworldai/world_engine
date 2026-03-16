import argparse
import io
import os
import random
import urllib.request

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from src.world_engine import WorldEngine, CtrlInput
from src.metal.runtime import ensure_metal_attention_op_loaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", default="Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--quant", default="none", choices=["none", "w8a8", "nvfp4"])
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-url", default="")
    parser.add_argument("--out", default="diagnostics/out/fp16_variant_metal_mps_saved.mp4")
    parser.add_argument("--hybrid-compile", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    os.environ["WORLD_HYBRID_COMPILE_METAL"] = "1" if args.hybrid_compile else "0"
    os.environ["WORLD_FORCE_COMPILE_METAL"] = "1" if args.force_compile else "0"
    ensure_metal_attention_op_loaded()

    quant = None if args.quant == "none" else args.quant
    engine = WorldEngine(
        args.model_uri,
        device=args.device,
        dtype=(torch.float16 if args.dtype == "float16" else torch.bfloat16),
        quant=quant,
    )
    # Compatibility for current world_engine timestamp math.
    engine.ts_mult = int(engine.ts_mult)

    urls = [
        "https://gist.github.com/user-attachments/assets/d81c6d26-a838-4afe-9d13-fd67677043c3",
        "https://gist.github.com/user-attachments/assets/b6d18c38-098e-43b0-8e61-66a16e5d8946",
        "https://gist.github.com/user-attachments/assets/0734a8c1-3eb4-4ffe-8c37-5665c45ab559",
        "https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed",
        "https://gist.github.com/user-attachments/assets/68c943a4-008a-4c25-948c-c81ab4c47d21",
    ]
    url = args.seed_url if args.seed_url else random.choice(urls)
    arr = iio.imread(io.BytesIO(urllib.request.urlopen(url).read()))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    t = F.interpolate(t, size=(512, 1024), mode="bilinear", align_corners=False)
    frame = t.round().clamp(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
    engine.append_frame(torch.from_numpy(np.repeat(frame[None], 4, axis=0)).to(engine.device))

    controller_sequence = [
        CtrlInput(mouse=[0.2, 0.2]),
        CtrlInput(button={32}),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(button={1}),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(button={1, 32}),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
    ] * 2
    controller_sequence += [CtrlInput()] * 8
    controller_sequence = controller_sequence[: max(0, args.frames)]

    out_path = args.out
    with iio.imopen(out_path, "w", plugin="pyav") as out:
        out.write(engine.gen_frame().cpu().numpy(), fps=60, codec="libx264")
        torch.mps.synchronize()
        for ctrl in controller_sequence:
            out.write(engine.gen_frame(ctrl=ctrl).cpu().numpy())
            torch.mps.synchronize()

    print(f"wrote={out_path}")
    print(f"seed_url={url}")


if __name__ == "__main__":
    main()

