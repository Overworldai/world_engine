import time

import pytest
import torch

from attn_backend import (
    AttnBackend,
    AttnConfig,
    AttnMeta,
    world_flex_attn_forward,
)
from metal_test_utils import require_metal_attn_ops, timed_ms_sync


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS backend not available on this system",
)


def _rand_attn_tensors(B: int, H: int, T: int, L: int, Dh: int, dtype: torch.dtype):
    q = torch.randn(B, H, T, Dh, device="mps", dtype=dtype)
    k = torch.randn(B, H, L, Dh, device="mps", dtype=dtype)
    v = torch.randn(B, H, L, Dh, device="mps", dtype=dtype)
    return q, k, v


def _rand_gqa_tensors(B: int, Hq: int, Hkv: int, T: int, L: int, Dh: int, dtype: torch.dtype):
    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=dtype)
    k = torch.randn(B, Hkv, L, Dh, device="mps", dtype=dtype)
    v = torch.randn(B, Hkv, L, Dh, device="mps", dtype=dtype)
    return q, k, v


_require_metal_ops = require_metal_attn_ops
_timed_ms_sync = timed_ms_sync


@pytest.mark.parametrize("dtype", [torch.float16])
def test_metal_backend_runs_and_is_stable(dtype):
    _require_metal_ops()

    B, H, T, L, Dh = 1, 8, 256, 256, 64
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, dtype)

    meta = AttnMeta(flex_block_mask=None, q_len=T, kv_len=L)
    cfg = AttnConfig(causal=True, enable_gqa=False)

    # Warmup
    for _ in range(3):
        _ = world_flex_attn_forward(q, k, v, meta, cfg, backend=AttnBackend.METAL)

    iters = 5
    start = time.perf_counter()
    for _ in range(iters):
        _ = world_flex_attn_forward(q, k, v, meta, cfg, backend=AttnBackend.METAL)
    elapsed = time.perf_counter() - start

    # This test intentionally only asserts that the kernel runs in a reasonable
    # amount of time; tighter perf targets can be added once the kernel body
    # is implemented and tuned.
    avg_ms = (elapsed / iters) * 1000.0
    assert avg_ms < 1000.0


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 64, 64, 64),
        (1, 4, 128, 128, 64),
        (1, 8, 256, 256, 64),
    ],
)
@pytest.mark.parametrize("mode", ["ref", "fast"])
def test_metal_impl_modes_perf_sanity(shape, mode, monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, H, T, L, Dh = shape
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, torch.float16)
    meta = AttnMeta(flex_block_mask=None, q_len=T, kv_len=L)
    cfg = AttnConfig(causal=True, enable_gqa=False)

    fn = torch.ops.world.flex_attn_metal_ref if mode == "ref" else torch.ops.world.flex_attn_metal_fast
    mask = torch.ones((B, H, T, L), device="mps", dtype=torch.uint8).contiguous()

    for _ in range(2):
        _ = fn(q, k, v, mask, cfg.causal)

    iters = 3
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(q, k, v, mask, cfg.causal)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert elapsed_ms > 0.0
    assert elapsed_ms < 2000.0


def test_metal_fast_strict_path_executes(monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")

    B, H, T, L, Dh = 1, 2, 32, 32, 64
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, torch.float16)
    mask = torch.ones((B, H, T, L), device="mps", dtype=torch.uint8).contiguous()
    out = torch.ops.world.flex_attn_metal_fast(q, k, v, mask, True)
    assert out.shape == q.shape


@pytest.mark.parametrize("mode", ["ref", "fast"])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 8, 2, 128, 128, 64),
        (1, 16, 4, 192, 192, 64),
    ],
)
def test_metal_gqa_modes_perf_sanity(shape, mode, monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, Hq, Hkv, T, L, Dh = shape
    q, k, v = _rand_gqa_tensors(B, Hq, Hkv, T, L, Dh, torch.float16)
    mask = torch.ones((B, Hq, T, L), device="mps", dtype=torch.uint8).contiguous()
    fn = torch.ops.world.flex_attn_metal_ref if mode == "ref" else torch.ops.world.flex_attn_metal_fast

    for _ in range(2):
        _ = fn(q, k, v, mask, True)

    iters = 3
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(q, k, v, mask, True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters

    assert out.shape == q.shape
    assert elapsed_ms > 0.0
    assert elapsed_ms < 3000.0


@pytest.mark.parametrize("causal", [False, True])
def test_metal_fast_long_context_stress(causal, monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, Hq, Hkv, T, L, Dh = 1, 16, 4, 256, 768, 64
    q, k, v = _rand_gqa_tensors(B, Hq, Hkv, T, L, Dh, torch.float16)
    mask = torch.ones((B, Hq, T, L), device="mps", dtype=torch.uint8).contiguous()

    # Warmup
    for _ in range(2):
        _ = torch.ops.world.flex_attn_metal_fast(q, k, v, mask, causal)

    iters = 4
    start = time.perf_counter()
    out = None
    for _ in range(iters):
        out = torch.ops.world.flex_attn_metal_fast(q, k, v, mask, causal)
    avg_ms = (time.perf_counter() - start) * 1000.0 / iters

    assert out is not None
    assert out.shape == q.shape
    assert torch.isfinite(out).all().item()
    # Generous ceiling for CI variance while still guarding hangs/regressions.
    assert avg_ms < 5000.0


def test_metal_fast_vs_ref_perf_ratio_gqa(monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, Hq, Hkv, T, L, Dh = 1, 16, 4, 192, 384, 64
    q, k, v = _rand_gqa_tensors(B, Hq, Hkv, T, L, Dh, torch.float16)
    mask = torch.ones((B, Hq, T, L), device="mps", dtype=torch.uint8).contiguous()

    for _ in range(2):
        _ = torch.ops.world.flex_attn_metal_ref(q, k, v, mask, True)
        _ = torch.ops.world.flex_attn_metal_fast(q, k, v, mask, True)

    iters = 3
    start = time.perf_counter()
    for _ in range(iters):
        _ = torch.ops.world.flex_attn_metal_ref(q, k, v, mask, True)
    ref_ms = (time.perf_counter() - start) * 1000.0 / iters

    start = time.perf_counter()
    for _ in range(iters):
        _ = torch.ops.world.flex_attn_metal_fast(q, k, v, mask, True)
    fast_ms = (time.perf_counter() - start) * 1000.0 / iters

    assert ref_ms > 0.0 and fast_ms > 0.0
    # Guard against extreme regressions while allowing room during early
    # kernel bring-up (current fast path is correctness-oriented, not tuned).
    assert fast_ms / ref_ms < 200.0
    assert fast_ms < 500.0


def test_metal_fast_blocks_perf_sanity(monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, Hq, Hkv, T, L, Dh = 1, 16, 4, 160, 320, 64
    q, k, v = _rand_gqa_tensors(B, Hq, Hkv, T, L, Dh, torch.float16)
    block_size = 4
    kv_blocks = (L + block_size - 1) // block_size
    block_written = torch.ones((kv_blocks,), device="mps", dtype=torch.uint8).contiguous()

    for _ in range(2):
        _ = torch.ops.world.flex_attn_metal_fast_blocks(q, k, v, block_written, block_size, True)

    iters = 3
    start = time.perf_counter()
    for _ in range(iters):
        out = torch.ops.world.flex_attn_metal_fast_blocks(q, k, v, block_written, block_size, True)
    avg_ms = (time.perf_counter() - start) * 1000.0 / iters
    assert out.shape == q.shape
    assert avg_ms > 0.0
    assert avg_ms < 5000.0


@pytest.mark.parametrize(
    "shape",
    [
        (1, 16, 4, 192, 384, 64),
        (1, 16, 4, 256, 768, 64),
        (1, 8, 8, 256, 512, 64),
        (2, 8, 2, 160, 320, 64),
    ],
)
@pytest.mark.parametrize("sparsity", [1.0, 0.5, 0.25])
def test_metal_fast_active_benchmark_matrix(shape, sparsity, monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, Hq, Hkv, T, L, Dh = shape
    q, k, v = _rand_gqa_tensors(B, Hq, Hkv, T, L, Dh, torch.float16)
    block_size = 4
    kv_blocks = (L + block_size - 1) // block_size
    block_written = (torch.rand((kv_blocks,), device="mps") < sparsity).to(torch.uint8).contiguous()
    active_blocks = torch.nonzero(block_written, as_tuple=False).flatten().to(torch.int32).contiguous()

    stats = _timed_ms_sync(
        lambda: torch.ops.world.flex_attn_metal_fast_active(
            q, k, v, active_blocks, block_size, True
        ),
        warmup=10,
        iters=40,
    )

    assert stats["mean_ms"] > 0.0
    assert stats["p50_ms"] > 0.0
    assert stats["p95_ms"] >= stats["p50_ms"]
    assert stats["max_ms"] < 50.0


def test_metal_fast_active_vs_blocks_latency(monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, Hq, Hkv, T, L, Dh = 1, 16, 4, 256, 768, 64
    q, k, v = _rand_gqa_tensors(B, Hq, Hkv, T, L, Dh, torch.float16)
    block_size = 4
    kv_blocks = (L + block_size - 1) // block_size
    block_written = torch.tensor([(i % 2) == 0 for i in range(kv_blocks)], device="mps", dtype=torch.uint8).contiguous()
    active_blocks = torch.nonzero(block_written, as_tuple=False).flatten().to(torch.int32).contiguous()

    blocks_stats = _timed_ms_sync(
        lambda: torch.ops.world.flex_attn_metal_fast_blocks(
            q, k, v, block_written, block_size, True
        ),
        warmup=10,
        iters=60,
    )
    active_stats = _timed_ms_sync(
        lambda: torch.ops.world.flex_attn_metal_fast_active(
            q, k, v, active_blocks, block_size, True
        ),
        warmup=10,
        iters=60,
    )

    # Active path should not regress significantly versus block-written path.
    assert active_stats["p50_ms"] <= blocks_stats["p50_ms"] * 1.25
    assert active_stats["p95_ms"] <= blocks_stats["p95_ms"] * 1.25


def test_world_backend_fast_active_stability(monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_ATTENTION_BACKEND", "metal")
    monkeypatch.setenv("WORLD_METAL_IMPL", "fast")
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, Hq, Hkv, T, L, Dh = 1, 16, 4, 192, 384, 64
    q, k, v = _rand_gqa_tensors(B, Hq, Hkv, T, L, Dh, torch.float16)
    block_size = 4
    kv_blocks = (L + block_size - 1) // block_size
    block_written = torch.tensor([(i % 2) == 0 for i in range(kv_blocks)], device="mps", dtype=torch.uint8).contiguous()
    active_blocks = torch.nonzero(block_written, as_tuple=False).flatten().to(torch.int32).contiguous()
    meta = AttnMeta(
        flex_block_mask=None,
        q_len=T,
        kv_len=L,
        block_written=block_written,
        active_blocks=active_blocks,
        block_size=block_size,
    )
    cfg = AttnConfig(causal=True, enable_gqa=True)

    stats = _timed_ms_sync(
        lambda: world_flex_attn_forward(q, k, v, meta, cfg, backend=AttnBackend.METAL),
        warmup=12,
        iters=80,
    )
    assert stats["mean_ms"] > 0.0
    assert stats["p95_ms"] < 20.0
    assert (stats["p95_ms"] / max(stats["p50_ms"], 1e-6)) < 3.0

