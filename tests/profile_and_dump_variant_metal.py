import argparse
import io
import json
import os
import random
import re
import time
import urllib.request
import subprocess
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from src.metal.runtime import ensure_metal_attention_op_loaded
from src.world_engine import CtrlInput, WorldEngine


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


def _controller_sequence(steps: int) -> list[CtrlInput]:
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
        CtrlInput(),
        CtrlInput(),
        CtrlInput(),
    ] * 4
    seq += [CtrlInput()] * 8
    seq += (
        [CtrlInput(button={32})] * 10
        + [CtrlInput(button={65})] * 10
        + [CtrlInput(button={68})] * 10
        + [CtrlInput(button={83})] * 10
    )
    seq += [CtrlInput()] * 10
    return seq[:steps]


def _sync_if_mps(device: str):
    if str(device).startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _sanitize(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return safe[:220]


def _to_dumpable(obj: Any):
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        if t.is_floating_point():
            t = t.to(torch.bfloat16)
        return t
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_dumpable(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _to_dumpable(v) for k, v in obj.items()}
    return obj


def _summary(obj: Any):
    if isinstance(obj, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "numel": int(obj.numel()),
        }
    if isinstance(obj, (list, tuple)):
        return {"kind": type(obj).__name__, "items": [_summary(x) for x in obj]}
    if isinstance(obj, dict):
        return {"kind": "dict", "items": {str(k): _summary(v) for k, v in obj.items()}}
    return {"kind": type(obj).__name__}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.array(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", default="Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-url", default="")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--profile-steps", type=int, default=64)
    parser.add_argument("--dump-phases", default="append,gen1")
    parser.add_argument("--output-dir", default="diagnostics/out/metal_profile_baseline")
    parser.add_argument("--write-video", action="store_true")
    parser.add_argument("--module-timing", action="store_true")
    parser.add_argument("--manifest-note", default="")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("WORLD_ATTENTION_BACKEND", "metal")
    os.environ.setdefault("WORLD_METAL_IMPL", "fast")
    os.environ.setdefault("WORLD_METAL_FAST_NO_FALLBACK", "1")
    os.environ.setdefault("WORLD_METAL_PREFER_ACTIVE_DISPATCH", "1")
    os.environ.setdefault("WORLD_KV_RUNTIME_CHECKS", "0")
    os.environ.setdefault("WORLD_KV_COMPUTE_ACTIVE_BLOCKS", "0")

    ensure_metal_attention_op_loaded()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    tensor_dir = output_dir / "tensors"
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    seed_url = args.seed_url if args.seed_url else random.choice(SEED_FRAME_URLS)
    frame = _load_seed_frame(seed_url)
    seed = torch.from_numpy(np.repeat(frame[None], 4, axis=0))

    engine = WorldEngine(args.model_uri, device=args.device, dtype=dtype)
    # Compatibility for restored world_engine path.
    if hasattr(engine, "ts_mult"):
        engine.ts_mult = int(engine.ts_mult)

    dump_phases = {x.strip() for x in args.dump_phases.split(",") if x.strip()}
    current_phase = {"name": None}
    dump_index: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    module_timing_enabled = bool(args.module_timing)
    module_timing: dict[str, dict[str, Any]] = {}
    start_times: dict[str, list[float]] = {}

    hooks = []
    for mod_name, mod in engine.model.named_modules():
        def _mk_hook(name: str, module: torch.nn.Module):
            def _hook(_m, _inp, out):
                phase = current_phase["name"]
                if phase is None or phase not in dump_phases:
                    return
                key = (phase, name)
                if key in seen:
                    return
                seen.add(key)
                data = _to_dumpable(out)
                summary = _summary(data)
                fname = f"{phase}__{_sanitize(name) if name else 'root'}__{len(dump_index):04d}.pt"
                path = tensor_dir / fname
                torch.save(data, path)
                dump_index.append(
                    {
                        "phase": phase,
                        "module_name": name,
                        "module_type": module.__class__.__name__,
                        "file": str(path),
                        "summary": summary,
                    }
                )
            return _hook
        hooks.append(mod.register_forward_hook(_mk_hook(mod_name, mod)))
        if module_timing_enabled:
            def _mk_pre_hook(name: str):
                def _pre(_m, _inp):
                    start_times.setdefault(name, []).append(time.perf_counter())
                return _pre

            def _mk_post_timing_hook(name: str, module: torch.nn.Module):
                def _post(_m, _inp, _out):
                    t0_list = start_times.get(name)
                    if not t0_list:
                        return
                    t0 = t0_list.pop()
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    rec = module_timing.setdefault(
                        name,
                        {
                            "module_name": name,
                            "module_type": module.__class__.__name__,
                            "count": 0,
                            "durations_ms": [],
                        },
                    )
                    rec["count"] += 1
                    rec["durations_ms"].append(float(dt_ms))
                return _post

            hooks.append(mod.register_forward_pre_hook(_mk_pre_hook(mod_name)))
            hooks.append(mod.register_forward_hook(_mk_post_timing_hook(mod_name, mod)))

    timings = {}
    with torch.inference_mode():
        t0 = time.perf_counter()
        current_phase["name"] = "append"
        engine.append_frame(seed.to(engine.device))
        _sync_if_mps(engine.device)
        t1 = time.perf_counter()
        timings["append_s"] = t1 - t0

        ctrl_seq = _controller_sequence(args.profile_steps)
        gen_times = []
        current_phase["name"] = "gen1"
        g0 = time.perf_counter()
        first = engine.gen_frame(ctrl=ctrl_seq[0])
        _sync_if_mps(engine.device)
        g1 = time.perf_counter()
        gen_times.append(g1 - g0)
        current_phase["name"] = None

        video_path = output_dir / "profile_run.mp4"
        writer = None
        if args.write_video:
            writer = iio.imopen(str(video_path), "w", plugin="pyav")
            writer.write(first.cpu().numpy(), fps=60, codec="libx264")

        for ctrl in ctrl_seq[1:]:
            g0 = time.perf_counter()
            frm = engine.gen_frame(ctrl=ctrl)
            _sync_if_mps(engine.device)
            g1 = time.perf_counter()
            gen_times.append(g1 - g0)
            if writer is not None:
                writer.write(frm.cpu().numpy())

        if writer is not None:
            writer.close()
            timings["video_path"] = str(video_path)

    for h in hooks:
        h.remove()

    gen_arr = np.array(gen_times, dtype=np.float64)
    timings.update(
        {
            "gen_frames": int(len(gen_times)),
            "gen_first_s": float(gen_arr[0]),
            "gen_mean_s": float(gen_arr.mean()),
            "gen_p50_s": float(np.percentile(gen_arr, 50)),
            "gen_p90_s": float(np.percentile(gen_arr, 90)),
            "gen_min_s": float(gen_arr.min()),
            "gen_max_s": float(gen_arr.max()),
            "gen_fps_mean": float(1.0 / gen_arr.mean()) if gen_arr.mean() > 0 else 0.0,
        }
    )

    module_timing_report = {
        "enabled": module_timing_enabled,
        "modules": [],
    }
    if module_timing_enabled:
        rows = []
        for name, rec in module_timing.items():
            durs = [float(x) for x in rec["durations_ms"]]
            rows.append(
                {
                    "module_name": name,
                    "module_type": rec["module_type"],
                    "count": int(rec["count"]),
                    "total_ms": float(sum(durs)),
                    "mean_ms": float(sum(durs) / len(durs)) if durs else 0.0,
                    "p50_ms": _percentile(durs, 50),
                    "p95_ms": _percentile(durs, 95),
                }
            )
        rows.sort(key=lambda x: x["total_ms"], reverse=True)
        module_timing_report["modules"] = rows

    # Torch profiler snapshot around one gen frame.
    with torch.inference_mode():
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            _ = engine.gen_frame(ctrl=CtrlInput())
            _sync_if_mps(engine.device)
    prof_table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=120)
    prof_path = output_dir / "torch_profiler_top_ops.txt"
    prof_path.write_text(prof_table, encoding="utf-8")
    module_timing_path = output_dir / "module_timing_report.json"
    module_timing_path.write_text(json.dumps(module_timing_report, indent=2), encoding="utf-8")

    git_sha = ""
    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=str(Path(__file__).resolve().parents[1]))
            .strip()
        )
    except Exception:
        git_sha = ""
    run_manifest = {
        "git_sha": git_sha,
        "model_uri": args.model_uri,
        "seed": args.seed,
        "seed_url": seed_url,
        "device": args.device,
        "dtype": args.dtype,
        "profile_steps": args.profile_steps,
        "dump_phases": args.dump_phases,
        "module_timing": module_timing_enabled,
        "write_video": bool(args.write_video),
        "manifest_note": args.manifest_note,
        "timestamp_unix_s": time.time(),
        "env": {
            "WORLD_ATTENTION_BACKEND": os.environ.get("WORLD_ATTENTION_BACKEND"),
            "WORLD_METAL_IMPL": os.environ.get("WORLD_METAL_IMPL"),
            "WORLD_METAL_FAST_NO_FALLBACK": os.environ.get("WORLD_METAL_FAST_NO_FALLBACK"),
            "WORLD_METAL_PREFER_ACTIVE_DISPATCH": os.environ.get("WORLD_METAL_PREFER_ACTIVE_DISPATCH"),
            "TORCHDYNAMO_DISABLE": os.environ.get("TORCHDYNAMO_DISABLE"),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    report = {
        "model_uri": args.model_uri,
        "seed": args.seed,
        "seed_url": seed_url,
        "device": args.device,
        "dtype": args.dtype,
        "env": {
            "WORLD_ATTENTION_BACKEND": os.environ.get("WORLD_ATTENTION_BACKEND"),
            "WORLD_METAL_IMPL": os.environ.get("WORLD_METAL_IMPL"),
            "WORLD_METAL_FAST_NO_FALLBACK": os.environ.get("WORLD_METAL_FAST_NO_FALLBACK"),
            "WORLD_METAL_PREFER_ACTIVE_DISPATCH": os.environ.get("WORLD_METAL_PREFER_ACTIVE_DISPATCH"),
            "TORCHDYNAMO_DISABLE": os.environ.get("TORCHDYNAMO_DISABLE"),
        },
        "timings": timings,
        "tensor_dump_count": len(dump_index),
        "tensor_dump_index_path": str(output_dir / "tensor_dump_index.json"),
        "module_timing_path": str(module_timing_path),
        "profiler_path": str(prof_path),
        "run_manifest_path": str(manifest_path),
    }

    (output_dir / "tensor_dump_index.json").write_text(json.dumps(dump_index, indent=2), encoding="utf-8")
    (output_dir / "profile_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

