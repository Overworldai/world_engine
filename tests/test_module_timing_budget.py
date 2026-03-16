from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.environ.get("RUN_SLOW_METAL_TESTS", "0") != "1", reason="enable with RUN_SLOW_METAL_TESTS=1")
@pytest.mark.skipif(os.environ.get("WORLD_ATTENTION_BACKEND", "metal") != "metal", reason="requires metal backend")
def test_module_timing_report_budget(tmp_path: Path):
    out_dir = tmp_path / "module_timing"
    cmd = [
        sys.executable,
        "tests/profile_and_dump_variant_metal.py",
        "--model-uri",
        "Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill",
        "--device",
        "mps",
        "--dtype",
        "bfloat16",
        "--profile-steps",
        "4",
        "--dump-phases",
        "none",
        "--module-timing",
        "--output-dir",
        str(out_dir),
    ]
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
    subprocess.run(cmd, check=True, cwd=str(_repo_root()), env=env)

    timing = json.loads((out_dir / "module_timing_report.json").read_text(encoding="utf-8"))
    assert timing["enabled"] is True
    assert isinstance(timing["modules"], list)
    assert len(timing["modules"]) > 0
    # Soft sanity: total measured module time should be positive.
    total = sum(float(m["total_ms"]) for m in timing["modules"])
    assert total > 0.0

