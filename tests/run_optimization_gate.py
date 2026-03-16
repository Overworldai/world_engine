import argparse
import json
import os
import subprocess
import sys
import statistics
from pathlib import Path


def _run(cmd: list[str], env: dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="none", choices=["none", "fp16"])
    parser.add_argument("--model-uri", default="Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--profile-steps", type=int, default=32)
    parser.add_argument("--perf-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-url", default="https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed")
    parser.add_argument("--baseline-dump-dir", default="diagnostics/out/metal_profile_baseline")
    parser.add_argument("--baseline-perf-report", default="diagnostics/out/metal_profile_perf_only/profile_report.json")
    parser.add_argument("--output-dir", default="diagnostics/out/optimization_gate_run")
    parser.add_argument("--config", default="tests/optimization_gate_config.json")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--visual-review-on-fail", action="store_true")
    parser.add_argument("--visual-review-frames", type=int, default=32)
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
    env.setdefault("TORCHDYNAMO_DISABLE", "1")
    env.setdefault("WORLD_ATTENTION_BACKEND", "metal")
    env.setdefault("WORLD_METAL_IMPL", "fast")
    env.setdefault("WORLD_METAL_FAST_NO_FALLBACK", "1")
    env.setdefault("WORLD_METAL_PREFER_ACTIVE_DISPATCH", "1")
    env.setdefault("WORLD_KV_RUNTIME_CHECKS", "0")
    env.setdefault("WORLD_KV_COMPUTE_ACTIVE_BLOCKS", "0")
    env.setdefault("PYTHONPATH", ".")

    py = sys.executable
    profile_script = "tests/profile_and_dump_variant_metal.py"
    compare_script = "tests/compare_tensor_dumps.py"
    hotspot_script = "tests/summarize_hotspots.py"
    video_script = "tests/gen_world_variant_metal_save.py"

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
    )

    base_perf = _load_json(Path(args.baseline_perf_report))
    cur_perf = _load_json(perf_dir / "perf_aggregate.json")
    quick_cmp = _load_json(compare_quick_dir / "comparison_summary.json")
    full_cmp = _load_json(compare_full_dir / "comparison_summary.json")

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
    overall_pass = correctness_pass and perf_pass
    visual_review_video = ""
    visual_review_required = False
    if args.visual_review_on_fail and not correctness_pass:
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
        "gates": {
            "quick_correctness_pass": bool(quick_cmp["pass"]),
            "full_correctness_pass": bool(full_cmp["pass"]),
            "correctness_pass": correctness_pass,
            "performance_pass": perf_pass,
            "overall_pass": overall_pass,
            "visual_review_required": visual_review_required,
        },
        "visual_review_video": visual_review_video,
        "thresholds": cfg,
    }

    gate_path = output_dir / "gate_report.json"
    gate_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.strict and not overall_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

