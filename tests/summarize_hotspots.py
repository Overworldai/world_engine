import argparse
import json
from pathlib import Path


def _parse_profiler_top_ops(path: Path, top_k: int) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    ops = []
    for line in lines:
        if "aten::" not in line and "world::" not in line:
            continue
        cols = [c.strip() for c in line.split("  ") if c.strip()]
        if len(cols) < 5:
            continue
        try:
            name = cols[0]
            self_cpu = cols[2]
            cpu_total = cols[4]
            ops.append({"name": name, "self_cpu": self_cpu, "cpu_total": cpu_total})
        except Exception:
            continue
    return ops[: max(0, top_k)]


def _parse_module_timing(path: Path, top_k: int) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    modules = data.get("modules", [])
    modules = sorted(modules, key=lambda x: float(x.get("total_ms", 0.0)), reverse=True)
    return modules[: max(0, top_k)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler", required=True)
    parser.add_argument("--module-timing", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    profiler_path = Path(args.profiler)
    module_timing_path = Path(args.module_timing)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "top_ops": _parse_profiler_top_ops(profiler_path, args.top_k),
        "top_modules": _parse_module_timing(module_timing_path, args.top_k),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

