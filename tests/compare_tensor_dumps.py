import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


def _flatten_tensors(obj: Any) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        out.append(obj.detach().float().flatten())
        return out
    if isinstance(obj, (list, tuple)):
        for x in obj:
            out.extend(_flatten_tensors(x))
        return out
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            out.extend(_flatten_tensors(obj[k]))
        return out
    return out


def _metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    diff = (lhs - rhs).abs()
    denom = lhs.norm() * rhs.norm()
    cos = float(torch.dot(lhs, rhs) / denom) if float(denom) > 0.0 else 1.0
    return {
        "cosine": cos,
        "mae": float(diff.mean()),
        "rmse": float(torch.sqrt(((lhs - rhs) ** 2).mean())),
        "max_abs": float(diff.max()),
    }


def _load_index(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--phase", default="all", choices=["all", "append", "gen1"])
    parser.add_argument("--modules-regex", default="")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--cosine-min", type=float, default=0.999)
    parser.add_argument("--mae-max", type=float, default=1e-2)
    parser.add_argument("--rmse-max", type=float, default=1e-2)
    parser.add_argument("--max-abs-max", type=float, default=1e-1)
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    out_dir = Path(args.out_dir) if args.out_dir else candidate_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    b_index = _load_index(baseline_dir / "tensor_dump_index.json")
    c_index = _load_index(candidate_dir / "tensor_dump_index.json")

    rx = re.compile(args.modules_regex) if args.modules_regex else None

    b_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for e in b_index:
        key = (e["phase"], e["module_name"], e["module_type"])
        b_map[key] = e
    c_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for e in c_index:
        key = (e["phase"], e["module_name"], e["module_type"])
        c_map[key] = e

    common = sorted(set(b_map.keys()) & set(c_map.keys()))
    if args.phase != "all":
        common = [k for k in common if k[0] == args.phase]
    if rx is not None:
        common = [k for k in common if rx.search(k[1] or "")]

    per_module: list[dict[str, Any]] = []
    fail_count = 0
    for key in common:
        b_path = Path(b_map[key]["file"])
        c_path = Path(c_map[key]["file"])
        b_obj = torch.load(b_path, map_location="cpu")
        c_obj = torch.load(c_path, map_location="cpu")
        b_tensors = _flatten_tensors(b_obj)
        c_tensors = _flatten_tensors(c_obj)
        if len(b_tensors) != len(c_tensors):
            metrics = {"cosine": 0.0, "mae": float("inf"), "rmse": float("inf"), "max_abs": float("inf")}
            status = "fail"
            fail_count += 1
        else:
            # Concatenate all tensor leaves in a deterministic order.
            if len(b_tensors) == 0:
                metrics = {"cosine": 1.0, "mae": 0.0, "rmse": 0.0, "max_abs": 0.0}
            else:
                lhs = torch.cat(b_tensors)
                rhs = torch.cat(c_tensors)
                if lhs.numel() != rhs.numel():
                    metrics = {"cosine": 0.0, "mae": float("inf"), "rmse": float("inf"), "max_abs": float("inf")}
                else:
                    metrics = _metrics(lhs, rhs)
            status = "pass"
            if (
                metrics["cosine"] < args.cosine_min
                or metrics["mae"] > args.mae_max
                or metrics["rmse"] > args.rmse_max
                or metrics["max_abs"] > args.max_abs_max
            ):
                status = "fail"
                fail_count += 1

        per_module.append(
            {
                "phase": key[0],
                "module_name": key[1],
                "module_type": key[2],
                "baseline_file": str(b_path),
                "candidate_file": str(c_path),
                "status": status,
                "metrics": metrics,
            }
        )

    # Worst by cosine asc then mae desc.
    sorted_worst = sorted(
        per_module,
        key=lambda x: (x["metrics"]["cosine"], -x["metrics"]["mae"]),
    )
    worst = sorted_worst[: max(0, args.top_k)]

    summary = {
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "phase": args.phase,
        "modules_regex": args.modules_regex,
        "thresholds": {
            "cosine_min": args.cosine_min,
            "mae_max": args.mae_max,
            "rmse_max": args.rmse_max,
            "max_abs_max": args.max_abs_max,
        },
        "counts": {
            "baseline_index": len(b_index),
            "candidate_index": len(c_index),
            "compared_modules": len(per_module),
            "failed_modules": fail_count,
        },
        "pass": fail_count == 0,
    }

    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "comparison_worst_modules.json").write_text(json.dumps(worst, indent=2), encoding="utf-8")
    (out_dir / "comparison_full.json").write_text(json.dumps(per_module, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(json.dumps({"top_k_worst": worst}, indent=2))

    if args.strict and fail_count > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

