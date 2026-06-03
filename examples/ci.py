#!/usr/bin/env python3
"""world_engine CI perf + consistency harness (issue #48).

Provider-agnostic: this script knows nothing about GCP/GitHub Actions. Given a
WorldEngine config it runs a performance rollout (LFPS) and/or a deterministic
consistency forward pass, writing JSON results (+ a latent .npy for
consistency) to --out. A separate `compare` subcommand turns a directory of
per-ref results into a markdown table (perf delta + consistency MSE).

The CI workflow runs the SAME copy of this script against two world_engine
installs (main and the PR HEAD) so that only the engine code differs.

Usage:
  # produce results for one (config, ref); run "main" first so it creates the
  # shared KV-cache state that "pr" then loads.
  python examples/ci.py run \
      --config-id wp15-1b-bf16 --ref main \
      --model-uri Overworld/Waypoint-1.5-1B --quant none \
      --shared-state results/state_wp15-1b-bf16.pt --out results

  # aggregate
  python examples/ci.py compare --results-dir results --out results/summary.md
"""
import argparse
import json
import platform
import time
from pathlib import Path


# How many gen_frame() steps to pre-roll before snapshotting the KV cache for
# the consistency check ("fully populated cache").
CONSISTENCY_PREROLL = 8


def env_info():
    import torch
    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "gpu": None,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        info["gpu"] = {
            "name": p.name,
            "capability": f"{p.major}.{p.minor}",
            "memory_gb": round(p.total_memory / 1e9, 1),
        }
    return info


def build_engine(model_uri, quant, overrides):
    from world_engine import WorldEngine
    return WorldEngine(
        model_uri,
        quant=quant,
        model_config_overrides=overrides,
        device="cuda",
    )


def run_perf(engine, n_frames):
    """Time an n_frames DiT-only rollout. LFPS = latent frames per second."""
    import torch

    for _ in range(3):  # warmup
        engine.gen_frame(return_img=False)
    engine.reset()
    engine.gen_frame(return_img=False)  # prime
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_frames):
        engine.gen_frame(return_img=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"status": "ok", "n_frames": n_frames, "elapsed_s": elapsed, "lfps": n_frames / elapsed}


def run_consistency(engine, shared_state: Path, latent_path: Path):
    """Deterministic single forward from a shared, fully-populated KV cache.

    The first ref to run creates the shared state (snapshotting the cache after
    a fixed pre-roll) and saves it; every other ref loads that exact state, so
    the only variable is the engine code. Latent is saved as .npy so the
    aggregator needs numpy only (no torch/GPU).
    """
    import numpy as np
    import torch

    # NB: full determinism also needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the
    # environment (set by the workflow) before CUDA initialises.
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    if shared_state.exists():
        engine.load_state(torch.load(shared_state))
        produced = False
    else:
        engine.reset()
        for _ in range(CONSISTENCY_PREROLL):
            engine.gen_frame(return_img=False)
        shared_state.parent.mkdir(parents=True, exist_ok=True)
        torch.save(engine.get_state(), shared_state)
        produced = True

    latent = engine.gen_frame(return_img=False).detach().float().cpu().numpy()
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(latent_path, latent)
    return {
        "status": "ok",
        "produced_shared_state": produced,
        "latent_shape": list(latent.shape),
        "latent_file": latent_path.name,
    }


def cmd_run(args):
    quant = None if (args.quant or "none").lower() in ("none", "null", "") else args.quant
    overrides = (
        json.loads(args.overrides)
        if args.overrides and args.overrides.lower() not in ("none", "null")
        else None
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "config_id": args.config_id,
        "ref": args.ref,
        "model_uri": args.model_uri,
        "quant": quant,
        "overrides": overrides,
        "env": env_info(),
        "perf": None,
        "consistency": None,
    }

    try:
        engine = build_engine(args.model_uri, quant, overrides)
    except Exception as e:  # engine init failure -> whole config marked failed
        result["error"] = f"engine init failed: {e!r}"
        _write_result(out, args, result)
        return

    if args.mode in ("perf", "both"):
        try:
            result["perf"] = run_perf(engine, args.n_frames)
        except Exception as e:
            result["perf"] = {"status": "failed", "error": repr(e)}

    if args.mode in ("consistency", "both"):
        try:
            result["consistency"] = run_consistency(
                engine,
                Path(args.shared_state),
                out / f"latent_{args.config_id}_{args.ref}.npy",
            )
        except Exception as e:
            result["consistency"] = {"status": "failed", "error": repr(e)}

    _write_result(out, args, result)


def _write_result(out: Path, args, result):
    path = out / f"result_{args.config_id}_{args.ref}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")


def _lfps(result):
    perf = (result or {}).get("perf") or {}
    return perf.get("lfps") if perf.get("status") == "ok" else None


def cmd_compare(args):
    import numpy as np

    rd = Path(args.results_dir)
    results = {}
    for f in sorted(rd.glob("result_*.json")):
        r = json.loads(f.read_text())
        results[(r["config_id"], r["ref"])] = r
    config_ids = sorted({c for c, _ in results})

    lines = ["## GPU benchmark (issue #48)", ""]
    gpu = next((r["env"].get("gpu") for r in results.values() if r.get("env")), None)
    if gpu:
        lines += [f"Machine: **{gpu['name']}** ({gpu['memory_gb']} GB, cc {gpu['capability']})", ""]

    lines += ["### Performance — LFPS (256-frame rollout)", "",
              "| Config | main | PR | Δ% |", "|---|---|---|---|"]
    for cid in config_ids:
        ml, pl = _lfps(results.get((cid, "main"))), _lfps(results.get((cid, "pr")))
        if isinstance(ml, float) and isinstance(pl, float) and ml:
            delta = f"{(pl - ml) / ml * 100:+.1f}%"
        else:
            delta = "—"
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else "**FAILED**"
        lines.append(f"| `{cid}` | {fmt(ml)} | {fmt(pl)} | {delta} |")

    lines += ["", "### Consistency — MSE(main, PR) latent", "",
              "| Config | MSE | status |", "|---|---|---|"]
    for cid in config_ids:
        try:
            a = np.load(rd / f"latent_{cid}_main.npy")
            b = np.load(rd / f"latent_{cid}_pr.npy")
            lines.append(f"| `{cid}` | {float(np.mean((a - b) ** 2)):.3e} | ok |")
        except Exception as e:
            lines.append(f"| `{cid}` | — | **FAILED**: {e} |")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run perf+consistency for one (config, ref)")
    r.add_argument("--config-id", required=True)
    r.add_argument("--ref", required=True, help="label for this engine build, e.g. main or pr")
    r.add_argument("--model-uri", required=True)
    r.add_argument("--quant", default="none")
    r.add_argument("--overrides", default=None, help="JSON dict of model_config_overrides")
    r.add_argument("--mode", choices=["perf", "consistency", "both"], default="both")
    r.add_argument("--n-frames", type=int, default=256)
    r.add_argument("--shared-state", required=True, help="path to the shared KV-cache state .pt")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="aggregate results into a markdown table")
    c.add_argument("--results-dir", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
