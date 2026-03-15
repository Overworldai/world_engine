import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from bench_world_engine_e2e import _controller_sequence, _load_seed_frame
from src.metal.runtime import ensure_metal_attention_op_loaded
from src.world_engine import CtrlInput, WorldEngine

SEED_URLS = [
    "https://gist.github.com/user-attachments/assets/d81c6d26-a838-4afe-9d13-fd67677043c3",
    "https://gist.github.com/user-attachments/assets/b6d18c38-098e-43b0-8e61-66a16e5d8946",
    "https://gist.github.com/user-attachments/assets/0734a8c1-3eb4-4ffe-8c37-5665c45ab559",
    "https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed",
    "https://gist.github.com/user-attachments/assets/68c943a4-008a-4c25-948c-c81ab4c47d21",
]


def _sync_mps():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def _summary(values: list[float]) -> dict:
    vals = sorted(values)
    n = len(vals)
    return {
        "p50": vals[n // 2],
        "p95": vals[max(0, int(0.95 * n) - 1)],
        "mean": sum(vals) / max(1, n),
    }


def _run_trial(return_img: bool, warmup: int, steps: int) -> dict:
    engine = WorldEngine(
        "Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill",
        quant="w8a8",
        device="mps",
        dtype=torch.float16,
        scheduler_steps=4,
        cache_interval=1,
    )
    frame = _load_seed_frame(random.choice(SEED_URLS))
    seed = torch.from_numpy(np.repeat(frame[None], 4, axis=0)).to(engine.device)
    engine.append_frame(seed)
    ctrls = _controller_sequence(CtrlInput)

    with torch.inference_mode():
        for i in range(warmup):
            _ = engine.gen_frame(ctrl=ctrls[i % len(ctrls)], return_img=return_img)
    _sync_mps()

    total_ms, denoise_ms, cache_ms, decode_ms, prep_ms = [], [], [], [], []
    with torch.inference_mode():
        for i in range(steps):
            ctrl = ctrls[(warmup + i) % len(ctrls)]
            x = torch.randn(engine.frm_shape, device=engine.device, dtype=engine.dtype)

            ttot = time.perf_counter()

            t0 = time.perf_counter()
            inputs = engine.prep_inputs(x=x, ctrl=ctrl)
            _sync_mps()
            prep_ms.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            x0 = engine._denoise_pass_fn(x, inputs, engine.kv_cache)
            _sync_mps()
            denoise_ms.append((time.perf_counter() - t1) * 1000.0)

            t2 = time.perf_counter()
            engine._cache_pass_fn(x0, inputs, engine.kv_cache)
            engine._gen_count += 1
            _sync_mps()
            cache_ms.append((time.perf_counter() - t2) * 1000.0)

            dec = 0.0
            if return_img:
                t3 = time.perf_counter()
                _ = engine.vae.decode(x0.squeeze(1))
                _sync_mps()
                dec = (time.perf_counter() - t3) * 1000.0
            decode_ms.append(dec)

            total_ms.append((time.perf_counter() - ttot) * 1000.0)

    total = _summary(total_ms)
    result = {
        "stage": {
            "prep_ms": _summary(prep_ms),
            "denoise_ms": _summary(denoise_ms),
            "cache_ms": _summary(cache_ms),
            "decode_ms": _summary(decode_ms),
            "total_ms": total,
            "fps": {
                "p50": 1000.0 / max(total["p50"], 1e-9),
                "p95": 1000.0 / max(total["p95"], 1e-9),
                "mean": 1000.0 / max(total["mean"], 1e-9),
            },
        }
    }
    del engine
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        _sync_mps()
    return result


def main():
    parser = argparse.ArgumentParser(description="Capture/compare MPS performance baseline.")
    parser.add_argument("--output", type=Path, default=Path("docs/perf_baseline_mps_w8a8.json"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--max-regression", type=float, default=0.10, help="Allowed p50 total_ms regression fraction.")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend not available.")

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ["WORLD_ATTENTION_BACKEND"] = "metal"
    os.environ.setdefault("WORLD_METAL_IMPL", "fast")
    os.environ.setdefault("WORLD_METAL_FAST_NO_FALLBACK", "1")
    os.environ.setdefault("WORLD_METAL_PREFER_ACTIVE_DISPATCH", "1")
    os.environ.setdefault("WORLD_KV_RUNTIME_CHECKS", "0")
    os.environ.setdefault("WORLD_KV_COMPUTE_ACTIVE_BLOCKS", "0")

    ensure_metal_attention_op_loaded()
    runs_decode = []
    runs_latent = []

    for _ in range(args.repeats):
        runs_decode.append(_run_trial(return_img=True, warmup=args.warmup, steps=args.steps))
        runs_latent.append(_run_trial(return_img=False, warmup=args.warmup, steps=args.steps))

    result = {
        "meta": {
            "repeats": args.repeats,
            "warmup": args.warmup,
            "steps": args.steps,
            "quant": "w8a8",
            "scheduler_steps": 4,
            "cache_interval": 1,
        },
        "decode_runs": runs_decode,
        "latent_runs": runs_latent,
    }

    if not args.compare_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[perf-gate] wrote baseline: {args.output}")
        return

    if not args.output.exists():
        raise RuntimeError(f"Baseline file not found: {args.output}")
    baseline = json.loads(args.output.read_text(encoding="utf-8"))

    def _median_p50_total(payload, key):
        vals = [r["stage"]["total_ms"]["p50"] for r in payload[key]]
        vals = sorted(vals)
        return vals[len(vals) // 2]

    base_decode = _median_p50_total(baseline, "decode_runs")
    base_latent = _median_p50_total(baseline, "latent_runs")
    now_decode = _median_p50_total(result, "decode_runs")
    now_latent = _median_p50_total(result, "latent_runs")

    decode_reg = (now_decode - base_decode) / max(base_decode, 1e-9)
    latent_reg = (now_latent - base_latent) / max(base_latent, 1e-9)

    print(
        json.dumps(
            {
                "baseline_decode_p50_total_ms": base_decode,
                "current_decode_p50_total_ms": now_decode,
                "decode_regression_frac": decode_reg,
                "baseline_latent_p50_total_ms": base_latent,
                "current_latent_p50_total_ms": now_latent,
                "latent_regression_frac": latent_reg,
            },
            indent=2,
        )
    )

    if decode_reg > args.max_regression or latent_reg > args.max_regression:
        raise RuntimeError("Performance regression exceeded threshold.")


if __name__ == "__main__":
    main()

