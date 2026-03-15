from __future__ import annotations

import time
from typing import Iterable

import pytest
import torch


def require_world_ops(op_names: Iterable[str]) -> None:
    if not hasattr(torch.ops, "world"):
        pytest.skip("Metal world namespace not registered")
    missing = [name for name in op_names if not hasattr(torch.ops.world, name)]
    if missing:
        pytest.skip(f"Required Metal ops not registered: {', '.join(missing)}")


def require_metal_attn_ops() -> None:
    require_world_ops(
        [
            "flex_attn_metal_ref",
            "flex_attn_metal_fast",
            "flex_attn_metal_fast_blocks",
            "flex_attn_metal_fast_active",
        ]
    )


def timed_ms_sync(fn, warmup: int = 5, iters: int = 20):
    for _ in range(warmup):
        fn()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    t = torch.tensor(samples, dtype=torch.float64)
    return {
        "mean_ms": float(t.mean().item()),
        "p50_ms": float(t.quantile(0.50).item()),
        "p95_ms": float(t.quantile(0.95).item()),
        "p99_ms": float(t.quantile(0.99).item()),
        "max_ms": float(t.max().item()),
    }
