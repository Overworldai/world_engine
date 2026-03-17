import argparse
import json
import os
import subprocess
import sys
import statistics
from pathlib import Path
import torch


def _run(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path | None = None,
    check: bool = True,
) -> None:
    if log_path is None:
        subprocess.run(cmd, check=check, env=env)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, check=False, env=env, capture_output=True, text=True)
    blob = f"$ {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
    log_path.write_text(blob, encoding="utf-8")
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _preset_defaults(preset: str) -> dict[str, str]:
    if preset == "fp16":
        return {
            "dtype": "float16",
            "config": "tests/optimization_gate_config_fp16.json",
            "baseline_dump_dir": "diagnostics/out/fp16_baseline",
            "baseline_perf_report": "diagnostics/out/fp16_baseline/profile_report.json",
        }
    return {}


def _safety_check(dump_dir: Path) -> dict:
    index_path = dump_dir / "tensor_dump_index.json"
    if not index_path.exists():
        return {"checked": False, "reason": "tensor_dump_index_missing", "pass": False}
    idx = json.loads(index_path.read_text(encoding="utf-8"))
    nonfinite = 0
    tensors_checked = 0

    def _check_obj(obj):
        nonlocal nonfinite, tensors_checked
        if isinstance(obj, torch.Tensor):
            tensors_checked += 1
            if obj.is_floating_point():
                finite = torch.isfinite(obj).all().item()
                if not bool(finite):
                    nonfinite += 1
            return
        if isinstance(obj, (list, tuple)):
            for x in obj:
                _check_obj(x)
            return
        if isinstance(obj, dict):
            for x in obj.values():
                _check_obj(x)

    for entry in idx:
        p = Path(entry["file"])
        if not p.exists():
            continue
        obj = torch.load(p, map_location="cpu")
        _check_obj(obj)
    return {
        "checked": True,
        "tensors_checked": tensors_checked,
        "nonfinite_tensors": nonfinite,
        "pass": nonfinite == 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="none", choices=["none", "fp16"])
    parser.add_argument("--model-uri", default="Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--quant", default="none", choices=["none", "w8a8", "nvfp4"])
    parser.add_argument("--profile-steps", type=int, default=32)
    parser.add_argument("--perf-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-url", default="https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed")
    parser.add_argument("--baseline-dump-dir", default="diagnostics/out/metal_profile_baseline")
    parser.add_argument("--baseline-perf-report", default="diagnostics/out/metal_profile_perf_only/profile_report.json")
    parser.add_argument("--output-dir", default="diagnostics/out/optimization_gate_run")
    parser.add_argument("--config", default="tests/optimization_gate_config.json")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--visual-review-on-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-review-frames", type=int, default=32)
    parser.add_argument("--hybrid-compile", action="store_true")
    parser.add_argument("--force-compile", action="store_true")
    parser.add_argument("--capture-recompiles", action="store_true")
    parser.add_argument("--preclean-python", action="store_true")
    parser.add_argument("--isolate-ext-build", action="store_true")
    args = parser.parse_args()

    defaults = _preset_defaults(args.preset)
    if defaults:
        args.dtype = defaults["dtype"]
        args.config = defaults["config"]
        if parser.get_default("baseline_dump_dir") == args.baseline_dump_dir:
            args.baseline_dump_dir = defaults["baseline_dump_dir"]
        if parser.get_default("baseline_perf_report") == args.baseline_perf_report:
            args.baseline_perf_report = defaults["baseline_perf_report"]

    output_dir = Path(args.output_dir)
    perf_dir = output_dir / "perf"
    dump_dir = output_dir / "dump"
    compare_quick_dir = output_dir / "compare_quick"
    compare_full_dir = output_dir / "compare_full"
    for d in [output_dir, perf_dir, dump_dir, compare_quick_dir, compare_full_dir]:
        d.mkdir(parents=True, exist_ok=True)

    cfg = _load_json(Path(args.config))
    quick_cfg = cfg["quick_gate"]
    full_cfg = cfg["full_gate"]
    perf_cfg = cfg["performance_gate"]

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TORCHDYNAMO_DISABLE", "0")
    env.setdefault("WORLD_ATTENTION_BACKEND", "metal")
    env.setdefault("WORLD_METAL_IMPL", "fast")
    env.setdefault("WORLD_METAL_FAST_NO_FALLBACK", "1")
    env.setdefault("WORLD_METAL_PREFER_ACTIVE_DISPATCH", "1")
    env.setdefault("WORLD_KV_RUNTIME_CHECKS", "0")
    env.setdefault("WORLD_KV_COMPUTE_ACTIVE_BLOCKS", "0")
    env.setdefault("PYTHONPATH", ".")
    env["WORLD_HYBRID_COMPILE_METAL"] = "1" if args.hybrid_compile else "0"
    env["WORLD_FORCE_COMPILE_METAL"] = "1" if args.force_compile else "0"
    if args.preclean_python:
        subprocess.run(["pkill", "-9", "-f", "/opt/homebrew/Cellar/python@3.14"], check=False)
        subprocess.run(["pkill", "-9", "-f", "/Users/louiscastricato/overworld/world_engine/.venv/bin/python"], check=False)
    if args.isolate_ext_build:
        ext_dir = output_dir / "torch_extensions"
        ext_dir.mkdir(parents=True, exist_ok=True)
        env["TORCH_EXTENSIONS_DIR"] = str(ext_dir)
        env.setdefault("NINJA", "/Users/louiscastricato/overworld/world_engine/.venv/bin/ninja")
    recompile_dir = output_dir / "recompile_logs"
    if args.capture_recompiles:
        env["TORCH_LOGS"] = "recompiles"
        env.setdefault("TORCHDYNAMO_VERBOSE", "1")

    py = sys.executable
    profile_script = "tests/profile_and_dump_variant_metal.py"
    compare_script = "tests/compare_tensor_dumps.py"
    hotspot_script = "tests/summarize_hotspots.py"
    video_script = "tests/gen_world_variant_metal_save.py"
    bench_script = "tests/bench_world_engine_e2e.py"

    # 1) Perf-only run (repeat and aggregate median)
    perf_runs_dir = output_dir / "perf_runs"
    perf_runs_dir.mkdir(parents=True, exist_ok=True)
    perf_reports = []
    repeats = max(1, int(args.perf_repeats))
    for i in range(repeats):
        run_dir = perf_runs_dir / f"run_{i:02d}"
        _run(
            [
                py,
                profile_script,
                "--model-uri",
                args.model_uri,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--profile-steps",
                str(args.profile_steps),
                "--quant",
                args.quant,
                "--seed",
                str(args.seed + i),
                "--seed-url",
                args.seed_url,
                "--dump-phases",
                "none",
                "--output-dir",
                str(run_dir),
                "--manifest-note",
                f"optimization_gate_perf_run_{i:02d}",
            ],
            env,
            (recompile_dir / f"perf_run_{i:02d}.log") if args.capture_recompiles else None,
        )
        perf_reports.append(_load_json(run_dir / "profile_report.json"))

    gen_mean_values = [float(r["timings"]["gen_mean_s"]) for r in perf_reports]
    gen_p90_values = [float(r["timings"]["gen_p90_s"]) for r in perf_reports]
    perf_aggregate = {
        "perf_repeats": repeats,
        "gen_mean_s_values": gen_mean_values,
        "gen_p90_s_values": gen_p90_values,
        "gen_mean_s_median": float(statistics.median(gen_mean_values)),
        "gen_p90_s_median": float(statistics.median(gen_p90_values)),
    }
    (perf_dir / "perf_aggregate.json").write_text(json.dumps(perf_aggregate, indent=2), encoding="utf-8")
    # Keep a convenience report path for downstream consumers.
    (perf_dir / "profile_report.json").write_text(json.dumps(perf_reports[-1], indent=2), encoding="utf-8")

    # 1.5) Phase timing and latent/decoded FPS via e2e bench script.
    latent_json = perf_dir / "bench_latent.json"
    decoded_json = perf_dir / "bench_decoded.json"
    _run(
        [
            py,
            bench_script,
            "--model-uri",
            args.model_uri,
            "--device",
            args.device,
            "--attention-backend",
            "metal",
            "--dtype",
            args.dtype,
            "--quant",
            args.quant,
            "--frames",
            str(args.profile_steps),
            "--json-out",
            str(latent_json),
        ],
        env,
        (recompile_dir / "bench_latent.log") if args.capture_recompiles else None,
    )
    _run(
        [
            py,
            bench_script,
            "--model-uri",
            args.model_uri,
            "--device",
            args.device,
            "--attention-backend",
            "metal",
            "--dtype",
            args.dtype,
            "--quant",
            args.quant,
            "--frames",
            str(args.profile_steps),
            "--return-img",
            "--json-out",
            str(decoded_json),
        ],
        env,
        (recompile_dir / "bench_decoded.log") if args.capture_recompiles else None,
    )

    # 2) Dump run + module timing
    _run(
        [
            py,
            profile_script,
            "--model-uri",
            args.model_uri,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--profile-steps",
            str(args.profile_steps),
            "--quant",
            args.quant,
            "--seed",
            str(args.seed),
            "--seed-url",
            args.seed_url,
            "--dump-phases",
            "append,gen1",
            "--module-timing",
            "--output-dir",
            str(dump_dir),
            "--manifest-note",
            "optimization_gate_dump",
        ],
        env,
        (recompile_dir / "dump.log") if args.capture_recompiles else None,
    )

    # 2.5) Hotspot ranking summary from profiler + module timing.
    _run(
        [
            py,
            hotspot_script,
            "--profiler",
            str(dump_dir / "torch_profiler_top_ops.txt"),
            "--module-timing",
            str(dump_dir / "module_timing_report.json"),
            "--out",
            str(output_dir / "hotspot_summary.json"),
            "--top-k",
            "10",
        ],
        env,
        (recompile_dir / "hotspots.log") if args.capture_recompiles else None,
    )

    # 3) Quick compare gate (sentinel modules)
    _run(
        [
            py,
            compare_script,
            "--baseline-dir",
            args.baseline_dump_dir,
            "--candidate-dir",
            str(dump_dir),
            "--phase",
            "all",
            "--modules-regex",
            quick_cfg["sentinel_modules_regex"],
            "--cosine-min",
            str(quick_cfg["cosine_min"]),
            "--mae-max",
            str(quick_cfg["mae_max"]),
            "--rmse-max",
            str(quick_cfg["rmse_max"]),
            "--max-abs-max",
            str(quick_cfg["max_abs_max"]),
            "--out-dir",
            str(compare_quick_dir),
        ],
        env,
        (recompile_dir / "compare_quick.log") if args.capture_recompiles else None,
        check=False,
    )

    # 4) Full compare gate
    _run(
        [
            py,
            compare_script,
            "--baseline-dir",
            args.baseline_dump_dir,
            "--candidate-dir",
            str(dump_dir),
            "--phase",
            "all",
            "--cosine-min",
            str(full_cfg["cosine_min"]),
            "--mae-max",
            str(full_cfg["mae_max"]),
            "--rmse-max",
            str(full_cfg["rmse_max"]),
            "--max-abs-max",
            str(full_cfg["max_abs_max"]),
            "--out-dir",
            str(compare_full_dir),
        ],
        env,
        (recompile_dir / "compare_full.log") if args.capture_recompiles else None,
        check=False,
    )

    base_perf = _load_json(Path(args.baseline_perf_report))
    cur_perf = _load_json(perf_dir / "perf_aggregate.json")
    quick_cmp = _load_json(compare_quick_dir / "comparison_summary.json")
    full_cmp = _load_json(compare_full_dir / "comparison_summary.json")
    safety = _safety_check(dump_dir)
    bench_latent = _load_json(latent_json)
    bench_decoded = _load_json(decoded_json)

    base_mean = float(base_perf["timings"]["gen_mean_s"])
    cur_mean = float(cur_perf["gen_mean_s_median"])
    base_p90 = float(base_perf["timings"]["gen_p90_s"])
    cur_p90 = float(cur_perf["gen_p90_s_median"])
    improvement_pct = ((base_mean - cur_mean) / base_mean) * 100.0 if base_mean > 0 else 0.0
    p90_regression_pct = ((cur_p90 - base_p90) / base_p90) * 100.0 if base_p90 > 0 else 0.0

    perf_pass = (
        improvement_pct >= float(perf_cfg["min_median_improvement_pct"])
        and p90_regression_pct <= float(perf_cfg["max_p90_regression_pct"])
    )
    correctness_pass = bool(quick_cmp["pass"]) and bool(full_cmp["pass"])
    safety_pass = bool(safety.get("pass", False))
    overall_pass = correctness_pass and perf_pass and safety_pass
    visual_review_video = ""
    visual_review_required = False
    if args.visual_review_on_fail and (not correctness_pass or not safety_pass):
        visual_review_required = True
        visual_review_video = str(output_dir / "visual_review_fail.mp4")
        _run(
            [
                py,
                video_script,
                "--model-uri",
                args.model_uri,
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--quant",
                args.quant,
                "--frames",
                str(args.visual_review_frames),
                "--seed",
                str(args.seed),
                "--seed-url",
                args.seed_url,
                "--out",
                visual_review_video,
            ],
            env,
            (recompile_dir / "visual_review.log") if args.capture_recompiles else None,
        )

    report = {
        "baseline_perf_report": args.baseline_perf_report,
        "current_perf_report": str(perf_dir / "profile_report.json"),
        "current_perf_aggregate": str(perf_dir / "perf_aggregate.json"),
        "quick_compare_summary": str(compare_quick_dir / "comparison_summary.json"),
        "full_compare_summary": str(compare_full_dir / "comparison_summary.json"),
        "hotspot_summary": str(output_dir / "hotspot_summary.json"),
        "metrics": {
            "baseline_gen_mean_s": base_mean,
            "current_gen_mean_s": cur_mean,
            "improvement_pct": improvement_pct,
            "baseline_gen_p90_s": base_p90,
            "current_gen_p90_s": cur_p90,
            "p90_regression_pct": p90_regression_pct,
        },
        "runtime": {
            "quant": args.quant,
            "hybrid_compile": bool(args.hybrid_compile),
            "force_compile": bool(args.force_compile),
            "bench_latent_json": str(latent_json),
            "bench_decoded_json": str(decoded_json),
            "bench_latent_fps_mean": float(bench_latent["fps"]["mean"]),
            "bench_decoded_fps_mean": float(bench_decoded["fps"]["mean"]),
            "bench_phase_mean_ms": {
                "prep": float(bench_decoded["prep_ms"]["mean"]),
                "denoise": float(bench_decoded["denoise_ms"]["mean"]),
                "cache": float(bench_decoded["cache_ms"]["mean"]),
                "decode": float(bench_decoded["decode_ms"]["mean"]),
            },
        },
        "gates": {
            "quick_correctness_pass": bool(quick_cmp["pass"]),
            "full_correctness_pass": bool(full_cmp["pass"]),
            "correctness_pass": correctness_pass,
            "safety_pass": safety_pass,
            "performance_pass": perf_pass,
            "overall_pass": overall_pass,
            "visual_review_required": visual_review_required,
        },
        "safety": safety,
        "visual_review_video": visual_review_video,
        "recompile_log_dir": str(recompile_dir) if args.capture_recompiles else "",
        "thresholds": cfg,
    }

    gate_path = output_dir / "gate_report.json"
    gate_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.strict and not overall_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

