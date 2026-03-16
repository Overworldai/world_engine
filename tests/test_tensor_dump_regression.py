from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _baseline_dir() -> Path:
    return _repo_root() / "diagnostics" / "out" / "metal_profile_baseline"


@pytest.mark.skipif(not _baseline_dir().exists(), reason="baseline dump directory not found")
def test_compare_tensor_dumps_self_consistency(tmp_path: Path):
    compare_script = _repo_root() / "tests" / "compare_tensor_dumps.py"
    out_dir = tmp_path / "cmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(compare_script),
        "--baseline-dir",
        str(_baseline_dir()),
        "--candidate-dir",
        str(_baseline_dir()),
        "--phase",
        "all",
        "--strict",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(_repo_root()))

    summary = json.loads((out_dir / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["pass"] is True
    assert summary["counts"]["failed_modules"] == 0

