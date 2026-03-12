from pathlib import Path
import sys
import math
import random

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "model"))

from attn_backend import (
    AttnBackend,
    AttnConfig,
    AttnMeta,
    world_flex_attn_forward,
)


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS backend not available on this system",
)


def _rand_attn_tensors(B: int, H: int, T: int, L: int, Dh: int, dtype: torch.dtype):
    q = torch.randn(B, H, T, Dh, device="mps", dtype=dtype)
    k = torch.randn(B, H, L, Dh, device="mps", dtype=dtype)
    v = torch.randn(B, H, L, Dh, device="mps", dtype=dtype)
    return q, k, v


def _require_metal_op():
    if not hasattr(torch.ops, "world"):
        pytest.skip("Metal world namespace not registered")
    if not (
        hasattr(torch.ops.world, "flex_attn_metal_ref")
        and hasattr(torch.ops.world, "flex_attn_metal_fast")
        and hasattr(torch.ops.world, "flex_attn_metal_fast_blocks")
        and hasattr(torch.ops.world, "flex_attn_metal_fast_active")
    ):
        pytest.skip("Metal ref/fast/fast_blocks/fast_active ops not registered")


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Explicit SDPA reference that does not depend on flex_attention.
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)

    if qf.size(1) != kf.size(1):
        if qf.size(1) < kf.size(1) or (qf.size(1) % kf.size(1)) != 0:
            raise RuntimeError("GQA requires q_heads divisible by kv_heads")
        group_size = qf.size(1) // kf.size(1)
        head_idx = torch.arange(qf.size(1), device=q.device, dtype=torch.long) // group_size
        kf = kf.index_select(1, head_idx)
        vf = vf.index_select(1, head_idx)

    scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    if causal:
        t = q.size(-2)
        l = k.size(-2)
        causal_mask = torch.triu(
            torch.ones((t, l), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask[None, None], float("-inf"))

    # If a row is fully masked, define output as zero (to match kernel behavior).
    finite_row = torch.isfinite(scores).any(dim=-1, keepdim=True)
    safe_scores = torch.where(finite_row, scores, torch.zeros_like(scores))
    probs = torch.softmax(safe_scores, dim=-1)
    probs = torch.where(finite_row, probs, torch.zeros_like(probs))
    out = torch.matmul(probs, vf)
    return out.to(q.dtype)


def _dense_mask_from_block_written(
    block_written: torch.Tensor,
    t: int,
    l: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a 1D block-written mask [KV_blocks] into dense [1,1,T,L] uint8 mask.
    This mirrors Andrew's guidance: kernel consumes frame length, total length,
    and block-wise written state.
    """
    dense = torch.zeros((l,), device=device, dtype=torch.uint8)
    for bidx, is_written in enumerate(block_written.tolist()):
        if is_written:
            s = bidx * block_size
            e = min(l, s + block_size)
            dense[s:e] = 1
    return dense.view(1, 1, 1, l).expand(1, 1, t, l).contiguous()


@pytest.mark.parametrize("dtype", [torch.float16])
def test_metal_vs_reference_small_random(dtype):
    _require_metal_op()

    B, H, T, L, Dh = 1, 2, 8, 8, 64
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, dtype)

    ref_out = _reference_attention(q, k, v, causal=False, mask=None)
    metal_out = world_flex_attn_forward(
        q,
        k,
        v,
        AttnMeta(flex_block_mask=None, q_len=T, kv_len=L),
        AttnConfig(causal=False, enable_gqa=False),
        backend=AttnBackend.METAL,
    )

    ref_cpu = ref_out.to("cpu", dtype=torch.float32)
    metal_cpu = metal_out.to("cpu", dtype=torch.float32)

    max_abs_diff = (ref_cpu - metal_cpu).abs().max().item()
    mean_abs_diff = (ref_cpu - metal_cpu).abs().mean().item()

    assert max_abs_diff < 2e-1
    assert mean_abs_diff < 2e-2


@pytest.mark.parametrize("dtype", [torch.float16])
def test_metal_vs_reference_small_random_causal(dtype):
    _require_metal_op()

    B, H, T, L, Dh = 1, 2, 8, 8, 64
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, dtype)

    ref_out = _reference_attention(q, k, v, causal=True, mask=None)
    metal_out = world_flex_attn_forward(
        q,
        k,
        v,
        AttnMeta(flex_block_mask=None, q_len=T, kv_len=L),
        AttnConfig(causal=True, enable_gqa=False),
        backend=AttnBackend.METAL,
    )

    ref_cpu = ref_out.to("cpu", dtype=torch.float32)
    metal_cpu = metal_out.to("cpu", dtype=torch.float32)

    max_abs_diff = (ref_cpu - metal_cpu).abs().max().item()
    mean_abs_diff = (ref_cpu - metal_cpu).abs().mean().item()

    assert max_abs_diff < 2e-1
    assert mean_abs_diff < 2e-2


@pytest.mark.parametrize("dtype", [torch.float16])
def test_metal_mask_all_ones_and_all_zeros(dtype):
    _require_metal_op()

    B, H, T, L, Dh = 1, 2, 8, 8, 64
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, dtype)

    ones = torch.ones((B, H, T, L), device="mps", dtype=torch.uint8).contiguous()
    zeros = torch.zeros((B, H, T, L), device="mps", dtype=torch.uint8).contiguous()

    out_no_mask = torch.ops.world.flex_attn_metal(q, k, v, None, False)
    out_ones = torch.ops.world.flex_attn_metal(q, k, v, ones, False)
    out_zeros = torch.ops.world.flex_attn_metal(q, k, v, zeros, False)

    no_mask_cpu = out_no_mask.to("cpu", dtype=torch.float32)
    ones_cpu = out_ones.to("cpu", dtype=torch.float32)
    zeros_cpu = out_zeros.to("cpu", dtype=torch.float32)
    ref_zero = _reference_attention(q, k, v, causal=False, mask=zeros).to("cpu", dtype=torch.float32)

    assert torch.allclose(no_mask_cpu, ones_cpu, rtol=1e-2, atol=1e-2)
    assert torch.allclose(zeros_cpu, torch.zeros_like(zeros_cpu), rtol=0.0, atol=1e-6)
    assert torch.allclose(zeros_cpu, ref_zero, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("mode", ["ref", "fast"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 8, 2, 16, 16, 64),
        (1, 12, 3, 12, 16, 64),
        (1, 16, 4, 32, 32, 64),
    ],
)
def test_gqa_metal_impl_matches_reference(shape, causal, mode, monkeypatch):
    _require_metal_op()
    B, Hq, Hkv, T, L, Dh = shape
    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=torch.float16)
    k = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    v = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    dense_mask = torch.ones((B, Hq, T, L), device="mps", dtype=torch.uint8).contiguous()

    ref = _reference_attention(q, k, v, causal=causal, mask=dense_mask)
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    if mode == "fast":
        monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")
    out = torch.ops.world.flex_attn_metal_ref(q, k, v, dense_mask, causal) if mode == "ref" else torch.ops.world.flex_attn_metal_fast(q, k, v, dense_mask, causal)

    assert torch.allclose(
        out.to("cpu", dtype=torch.float32),
        ref.to("cpu", dtype=torch.float32),
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.parametrize("mode", ["ref", "fast"])
def test_world_flex_attn_forward_gqa_executes(mode, monkeypatch):
    _require_metal_op()
    B, Hq, Hkv, T, L, Dh = 1, 8, 2, 8, 8, 64
    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=torch.float16)
    k = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    v = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    meta = AttnMeta(flex_block_mask=None, q_len=T, kv_len=L)
    cfg = AttnConfig(causal=True, enable_gqa=True)

    monkeypatch.setenv("WORLD_METAL_IMPL", mode)
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    if mode == "fast":
        monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")
    out = world_flex_attn_forward(q, k, v, meta, cfg, backend=AttnBackend.METAL)
    assert out.shape == q.shape


@pytest.mark.parametrize("dtype", [torch.float16])
def test_ref_and_fast_op_shapes_and_parity(dtype):
    _require_metal_op()

    B, H, T, L, Dh = 1, 2, 8, 8, 64
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, dtype)

    out_ref = torch.ops.world.flex_attn_metal_ref(q, k, v, None, True)
    out_fast = torch.ops.world.flex_attn_metal_fast(q, k, v, None, True)

    assert out_ref.shape == q.shape
    assert out_fast.shape == q.shape
    assert out_ref.dtype == q.dtype
    assert out_fast.dtype == q.dtype
    assert torch.allclose(
        out_ref.to("cpu", dtype=torch.float32),
        out_fast.to("cpu", dtype=torch.float32),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("mode", ["ref", "fast"])
def test_world_flex_attn_forward_selects_metal_impl(mode, monkeypatch):
    _require_metal_op()

    B, H, T, L, Dh = 1, 2, 8, 8, 64
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, torch.float16)
    meta = AttnMeta(flex_block_mask=None, q_len=T, kv_len=L)
    cfg = AttnConfig(causal=True, enable_gqa=False)

    monkeypatch.setenv("WORLD_METAL_IMPL", mode)
    out = world_flex_attn_forward(q, k, v, meta, cfg, backend=AttnBackend.METAL)
    assert out.shape == q.shape


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 8, 8, 32),
        (1, 4, 12, 16, 64),
        (1, 8, 16, 16, 64),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mode", ["ref", "fast"])
def test_metal_impl_matches_reference_with_block_mask(shape, causal, mode, monkeypatch):
    _require_metal_op()
    B, H, T, L, Dh = shape
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, torch.float16)

    block_size = 4
    kv_blocks = (L + block_size - 1) // block_size
    # Fixed rolling-cache style block occupancy pattern.
    block_written = torch.tensor(
        [(i % 3) != 0 for i in range(kv_blocks)],
        device=q.device,
        dtype=torch.bool,
    )
    dense_mask = _dense_mask_from_block_written(block_written, T, L, block_size, q.device)
    dense_mask = dense_mask.expand(B, H, T, L).contiguous()

    ref = _reference_attention(q, k, v, causal=causal, mask=dense_mask)

    monkeypatch.setenv("WORLD_METAL_IMPL", mode)
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", str(block_size))
    if mode == "fast":
        monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")
    out = torch.ops.world.flex_attn_metal_ref(q, k, v, dense_mask, causal) if mode == "ref" else torch.ops.world.flex_attn_metal_fast(q, k, v, dense_mask, causal)

    assert torch.allclose(
        out.to("cpu", dtype=torch.float32),
        ref.to("cpu", dtype=torch.float32),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("seed", list(range(20)))
def test_metal_fast_strict_fuzz_block_mask_gqa(seed, monkeypatch):
    """
    Adversarial fuzz test:
    - odd T/L lengths
    - variable block sizes (including non-divisors of L)
    - mixed GQA factors
    - random block-written sparsity patterns
    - random causal mode
    """
    _require_metal_op()
    random.seed(seed)
    torch.manual_seed(seed)

    B = 1
    T = random.choice([1, 3, 7, 11, 15, 23, 31])
    L = random.choice([1, 5, 9, 13, 17, 29, 33, 47])
    Dh = random.choice([32, 64])
    Hkv = random.choice([1, 2, 4])
    gqa_group = random.choice([1, 2, 4])
    Hq = Hkv * gqa_group
    causal = bool(random.getrandbits(1))
    block_size = random.choice([1, 2, 3, 4, 5, 7, 8])

    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=torch.float16)
    k = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    v = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)

    kv_blocks = (L + block_size - 1) // block_size
    # Include very sparse and very dense block occupancy cases.
    p = random.choice([0.15, 0.35, 0.5, 0.8, 1.0])
    block_written = (torch.rand(kv_blocks, device=q.device) < p)
    # Keep at least one available block to avoid all-zero trivial outputs every time.
    if not bool(block_written.any()):
        block_written[random.randrange(kv_blocks)] = True

    dense_mask = _dense_mask_from_block_written(block_written, T, L, block_size, q.device)
    dense_mask = dense_mask.expand(B, Hq, T, L).contiguous()

    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", str(block_size))
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    out_fast = torch.ops.world.flex_attn_metal_fast(q, k, v, dense_mask, causal)
    ref = _reference_attention(q, k, v, causal=causal, mask=dense_mask)

    diff = (out_fast.to("cpu", dtype=torch.float32) - ref.to("cpu", dtype=torch.float32)).abs()
    assert diff.max().item() < 4e-2
    assert diff.mean().item() < 5e-3
    assert torch.isfinite(out_fast).all().item()


def test_metal_fast_strict_full_mask_rows_gqa(monkeypatch):
    """
    Hard edge case where all KV blocks are masked out. Output should be zeros
    (after safe softmax handling), even for GQA.
    """
    _require_metal_op()
    B, Hq, Hkv, T, L, Dh = 1, 8, 2, 19, 37, 64
    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=torch.float16)
    k = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    v = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)

    block_size = 4
    block_written = torch.zeros((L + block_size - 1) // block_size, device=q.device, dtype=torch.bool)
    dense_mask = _dense_mask_from_block_written(block_written, T, L, block_size, q.device)
    dense_mask = dense_mask.expand(B, Hq, T, L).contiguous()

    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", str(block_size))
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")
    out = torch.ops.world.flex_attn_metal_fast(q, k, v, dense_mask, True)
    assert torch.allclose(out.to("cpu", dtype=torch.float32), torch.zeros_like(out.to("cpu", dtype=torch.float32)), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 8, 2, 13, 29, 64),
        (1, 16, 4, 17, 33, 64),
    ],
)
def test_fast_blocks_op_matches_reference(shape, causal, monkeypatch):
    _require_metal_op()
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, Hq, Hkv, T, L, Dh = shape
    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=torch.float16)
    k = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    v = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)

    block_size = 4
    kv_blocks = (L + block_size - 1) // block_size
    block_written = torch.tensor([(i % 2) == 0 for i in range(kv_blocks)], device=q.device, dtype=torch.uint8).contiguous()
    dense_mask = _dense_mask_from_block_written(block_written.bool(), T, L, block_size, q.device).expand(B, Hq, T, L).contiguous()

    out = torch.ops.world.flex_attn_metal_fast_blocks(q, k, v, block_written, block_size, causal)
    ref = _reference_attention(q, k, v, causal=causal, mask=dense_mask)
    assert torch.allclose(
        out.to("cpu", dtype=torch.float32),
        ref.to("cpu", dtype=torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )


def test_world_flex_attn_forward_uses_block_metadata_path(monkeypatch):
    _require_metal_op()
    monkeypatch.setenv("WORLD_METAL_IMPL", "fast")
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, Hq, Hkv, T, L, Dh = 1, 8, 2, 11, 23, 64
    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=torch.float16)
    k = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    v = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    block_size = 4
    kv_blocks = (L + block_size - 1) // block_size
    block_written = torch.tensor([(i % 3) != 0 for i in range(kv_blocks)], device=q.device, dtype=torch.uint8).contiguous()

    meta = AttnMeta(
        flex_block_mask=None,
        q_len=T,
        kv_len=L,
        block_written=block_written,
        block_size=block_size,
    )
    cfg = AttnConfig(causal=True, enable_gqa=True)
    out = world_flex_attn_forward(q, k, v, meta, cfg, backend=AttnBackend.METAL)
    dense_mask = _dense_mask_from_block_written(block_written.bool(), T, L, block_size, q.device).expand(B, Hq, T, L).contiguous()
    ref = _reference_attention(q, k, v, causal=True, mask=dense_mask)
    assert torch.allclose(
        out.to("cpu", dtype=torch.float32),
        ref.to("cpu", dtype=torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )


def test_metal_fast_rejects_non_shared_mask(monkeypatch):
    _require_metal_op()
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")

    B, H, T, L, Dh = 1, 4, 8, 16, 64
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, torch.float16)
    mask = torch.ones((B, H, T, L), device="mps", dtype=torch.uint8).contiguous()
    # Violate shared-mask contract by changing one query row.
    mask[0, 1, 3, 5] = 0

    with pytest.raises(RuntimeError, match="shared mask"):
        _ = torch.ops.world.flex_attn_metal_fast(q, k, v, mask, True)


def test_metal_fast_rejects_non_blockwise_mask(monkeypatch):
    _require_metal_op()
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")

    B, H, T, L, Dh = 1, 4, 8, 16, 64
    q, k, v = _rand_attn_tensors(B, H, T, L, Dh, torch.float16)
    mask = torch.ones((B, H, T, L), device="mps", dtype=torch.uint8).contiguous()
    # Within a block [4,5,6,7], make token-level values differ.
    mask[..., 5] = 0

    with pytest.raises(RuntimeError, match="block-wise mask values"):
        _ = torch.ops.world.flex_attn_metal_fast(q, k, v, mask, False)


def test_metal_fast_batch2_shared_mask_matches_reference(monkeypatch):
    _require_metal_op()
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")

    B, Hq, Hkv, T, L, Dh = 2, 8, 2, 11, 23, 64
    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=torch.float16)
    k = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    v = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)

    kv_blocks = (L + 4 - 1) // 4
    block_written = torch.tensor([(i % 2) == 0 for i in range(kv_blocks)], device=q.device, dtype=torch.bool)
    base_mask = _dense_mask_from_block_written(block_written, T, L, 4, q.device)
    dense_mask = base_mask.expand(B, Hq, T, L).contiguous()

    out_fast = torch.ops.world.flex_attn_metal_fast(q, k, v, dense_mask, True)
    ref = _reference_attention(q, k, v, causal=True, mask=dense_mask)
    assert torch.allclose(
        out_fast.to("cpu", dtype=torch.float32),
        ref.to("cpu", dtype=torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.parametrize("seed", list(range(40)))
def test_metal_fast_active_strict_fuzz_matches_reference(seed, monkeypatch):
    _require_metal_op()
    random.seed(seed)
    torch.manual_seed(seed)
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B = random.choice([1, 2])
    Hkv = random.choice([1, 2, 4])
    gqa_group = random.choice([1, 2, 4])
    Hq = Hkv * gqa_group
    T = random.choice([1, 7, 15, 31, 63, 95])
    L = random.choice([5, 17, 37, 65, 129, 257])
    Dh = random.choice([32, 64])
    causal = bool(random.getrandbits(1))
    block_size = random.choice([1, 2, 4, 8])

    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=torch.float16)
    k = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    v = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)

    kv_blocks = (L + block_size - 1) // block_size
    p = random.choice([0.0, 0.1, 0.25, 0.5, 0.8, 1.0])
    block_written = (torch.rand(kv_blocks, device=q.device) < p).to(torch.uint8).contiguous()
    active_blocks = torch.nonzero(block_written, as_tuple=False).flatten().to(torch.int32).contiguous()

    out = torch.ops.world.flex_attn_metal_fast_active(q, k, v, active_blocks, block_size, causal)
    dense_mask = _dense_mask_from_block_written(block_written.bool(), T, L, block_size, q.device)
    dense_mask = dense_mask.expand(B, Hq, T, L).contiguous()
    ref = _reference_attention(q, k, v, causal=causal, mask=dense_mask)

    diff = (out.to("cpu", dtype=torch.float32) - ref.to("cpu", dtype=torch.float32)).abs()
    assert diff.max().item() < 5e-2
    assert diff.mean().item() < 8e-3
    assert torch.isfinite(out).all().item()

