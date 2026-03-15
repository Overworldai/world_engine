import argparse
import io
import json
import os
import random
import urllib.request
import inspect
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F


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


def _ctrl_sequence(CtrlInput, steps: int):
    seq = [
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
    ]
    if steps <= len(seq):
        return seq[:steps]
    seq = seq + [CtrlInput() for _ in range(steps - len(seq))]
    return seq


def _metrics(a: torch.Tensor, b: torch.Tensor):
    av = a.flatten()
    bv = b.flatten()
    d = (av - bv).abs()
    return {
        "cos": float(torch.nn.functional.cosine_similarity(av, bv, dim=0)),
        "mae": float(d.mean()),
        "rmse": float(torch.sqrt(((av - bv) ** 2).mean())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", default="Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--attention-backend", default="flex", choices=["flex", "metal", "auto"])
    parser.add_argument("--scheduler-steps", type=int, default=4)
    parser.add_argument("--cache-interval", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--seed-url",
        default="https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--disable-patch-cached-noise", action="store_true")
    parser.add_argument("--disable-patch-merge-qkv", action="store_true")
    parser.add_argument("--disable-patch-split-mlp", action="store_true")
    parser.add_argument("--force-direct-flex-wrapper", action="store_true")
    parser.add_argument("--metal-force-causal", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("WORLD_KV_RUNTIME_CHECKS", "0")
    os.environ.setdefault("WORLD_KV_COMPUTE_ACTIVE_BLOCKS", "0")
    os.environ["WORLD_ATTENTION_BACKEND"] = args.attention_backend
    if args.metal_force_causal:
        os.environ["WORLD_METAL_FORCE_CAUSAL"] = "1"
    else:
        os.environ.pop("WORLD_METAL_FORCE_CAUSAL", None)
    if args.attention_backend == "metal" and args.device == "mps":
        os.environ.setdefault("WORLD_METAL_IMPL", "fast")
        os.environ.setdefault("WORLD_METAL_FAST_NO_FALLBACK", "1")
        os.environ.setdefault("WORLD_METAL_PREFER_ACTIVE_DISPATCH", "1")

    import src.patch_model as patch_model
    import src.world_engine as world_engine_mod
    from src.world_engine import CtrlInput, WorldEngine

    # Patch toggles for ablation without editing model files.
    original_apply = world_engine_mod.apply_inference_patches

    def apply_with_toggles(model):
        if not args.disable_patch_cached_noise and next(model.parameters()).dtype == torch.bfloat16:
            patch_model.patch_cached_noise_conditioning(model)
        if not args.disable_patch_merge_qkv:
            patch_model.patch_Attn_merge_qkv(model)
        if not args.disable_patch_split_mlp:
            patch_model.patch_MLPFusion_split(model)

    world_engine_mod.apply_inference_patches = apply_with_toggles

    original_world_attn = getattr(patch_model, "world_flex_attn_forward", None)
    if args.force_direct_flex_wrapper and original_world_attn is not None:
        from torch.nn.attention.flex_attention import flex_attention

        def direct_flex(q, k, v, meta, cfg, backend=None):
            block_mask = meta.flex_block_mask if meta is not None else None
            return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=cfg.enable_gqa)

        patch_model.world_flex_attn_forward = direct_flex

    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame = _load_seed_frame(args.seed_url)
    seed = torch.from_numpy(np.repeat(frame[None], 4, axis=0))
    ctrl_seq = _ctrl_sequence(CtrlInput, args.steps)

    sig = inspect.signature(WorldEngine.__init__)
    kwargs = {
        "quant": None,
        "device": args.device,
        "dtype": dtype,
    }
    if "scheduler_steps" in sig.parameters:
        kwargs["scheduler_steps"] = args.scheduler_steps
    if "cache_interval" in sig.parameters:
        kwargs["cache_interval"] = args.cache_interval
    engine = WorldEngine(args.model_uri, **kwargs)
    if hasattr(engine, "_cache_pass_eager"):
        engine._cache_pass_fn = engine._cache_pass_eager
    if hasattr(engine, "_denoise_pass_eager"):
        engine._denoise_pass_fn = engine._denoise_pass_eager

    latents = []
    with torch.inference_mode():
        engine.append_frame(seed.to(engine.device))
        for i, ctrl in enumerate(ctrl_seq):
            x = torch.randn(
                (1, 1, 32, 32, 64),
                generator=torch.Generator(device="cpu").manual_seed(args.seed + i),
                dtype=torch.float32,
            ).to(engine.device, dtype=dtype)
            inp = engine.prep_inputs(x=x, ctrl=ctrl)
            y = engine._denoise_pass_fn(x, inp, engine.kv_cache)
            if hasattr(engine, "_cache_pass_fn"):
                engine._cache_pass_fn(y, inp, engine.kv_cache)
            else:
                engine._cache_pass(y, inp, engine.kv_cache)
            if hasattr(engine, "_gen_count"):
                engine._gen_count += 1
            if engine.device == "mps":
                torch.mps.synchronize()
            latents.append(y.detach().float().cpu())

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    lat_path = f"{out_prefix}.latents.pt"
    meta_path = f"{out_prefix}.meta.json"
    stats_path = f"{out_prefix}.stats.json"

    torch.save(latents, lat_path)
    meta = {
        "model_uri": args.model_uri,
        "device": args.device,
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "scheduler_steps": args.scheduler_steps,
        "cache_interval": args.cache_interval,
        "steps": args.steps,
        "seed_url": args.seed_url,
        "seed": args.seed,
        "disable_patch_cached_noise": args.disable_patch_cached_noise,
        "disable_patch_merge_qkv": args.disable_patch_merge_qkv,
        "disable_patch_split_mlp": args.disable_patch_split_mlp,
        "force_direct_flex_wrapper": args.force_direct_flex_wrapper,
        "metal_force_causal": args.metal_force_causal,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    adj = {}
    for i in range(1, len(latents)):
        adj[f"{i-1}->{i}"] = _metrics(latents[i - 1], latents[i])
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"adjacent_latent_metrics": adj}, f, indent=2)

    # restore monkeypatches
    if original_world_attn is not None:
        patch_model.world_flex_attn_forward = original_world_attn
    world_engine_mod.apply_inference_patches = original_apply

    print(json.dumps({"latents": lat_path, "meta": meta_path, "stats": stats_path}, indent=2))


if __name__ == "__main__":
    main()

