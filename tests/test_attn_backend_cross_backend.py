import math

import pytest
import torch

from attn_backend import AttnBackend, AttnConfig, AttnMeta, world_flex_attn_forward
from metal_test_utils import require_metal_attn_ops


def _reference_attention(q, k, v, causal: bool):
    qf, kf, vf = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    if qf.size(1) != kf.size(1):
        if qf.size(1) < kf.size(1) or (qf.size(1) % kf.size(1)) != 0:
            raise RuntimeError("GQA requires q_heads divisible by kv_heads")
        group = qf.size(1) // kf.size(1)
        head_idx = torch.arange(qf.size(1), device=q.device, dtype=torch.long) // group
        kf = kf.index_select(1, head_idx)
        vf = vf.index_select(1, head_idx)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(qf.size(-1))
    if causal:
        t, l = qf.size(-2), kf.size(-2)
        tri = torch.triu(torch.ones((t, l), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(tri[None, None], float("-inf"))
    finite = torch.isfinite(scores).any(dim=-1, keepdim=True)
    scores = torch.where(finite, scores, torch.zeros_like(scores))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.where(finite, probs, torch.zeros_like(probs))
    return torch.matmul(probs, vf).to(q.dtype)


def test_auto_backend_uses_pytorch_flex_on_cpu():
    q = torch.randn(1, 2, 8, 32, device="cpu", dtype=torch.float32)
    k = torch.randn(1, 2, 8, 32, device="cpu", dtype=torch.float32)
    v = torch.randn(1, 2, 8, 32, device="cpu", dtype=torch.float32)
    out = world_flex_attn_forward(
        q, k, v, AttnMeta(flex_block_mask=None, q_len=8, kv_len=8), AttnConfig(causal=True), backend=AttnBackend.AUTO
    )
    expected = world_flex_attn_forward(
        q, k, v, AttnMeta(flex_block_mask=None, q_len=8, kv_len=8), AttnConfig(causal=True), backend=AttnBackend.PYTORCH_FLEX
    )
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("shape", [(1, 4, 4, 16, 16, 32), (1, 8, 2, 12, 20, 64)])
def test_pytorch_flex_matches_reference(shape, causal):
    b, hq, hkv, t, l, dh = shape
    q = torch.randn(b, hq, t, dh, device="cpu", dtype=torch.float32)
    k = torch.randn(b, hkv, l, dh, device="cpu", dtype=torch.float32)
    v = torch.randn(b, hkv, l, dh, device="cpu", dtype=torch.float32)
    out = world_flex_attn_forward(
        q, k, v, AttnMeta(flex_block_mask=None, q_len=t, kv_len=l), AttnConfig(causal=causal, enable_gqa=(hq != hkv)),
        backend=AttnBackend.PYTORCH_FLEX,
    )
    ref = _reference_attention(q, k, v, causal=causal)
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS backend not available")
def test_auto_backend_uses_metal_on_mps(monkeypatch):
    require_metal_attn_ops()
    monkeypatch.setenv("WORLD_METAL_IMPL", "fast")
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")
    q = torch.randn(1, 8, 16, 64, device="mps", dtype=torch.float16)
    k = torch.randn(1, 2, 32, 64, device="mps", dtype=torch.float16)
    v = torch.randn(1, 2, 32, 64, device="mps", dtype=torch.float16)
    meta = AttnMeta(flex_block_mask=None, q_len=16, kv_len=32)
    cfg = AttnConfig(causal=True, enable_gqa=True)
    metal = world_flex_attn_forward(q, k, v, meta, cfg, backend=AttnBackend.METAL)
    auto = world_flex_attn_forward(q, k, v, meta, cfg, backend=AttnBackend.AUTO)
    assert torch.allclose(
        metal.to("cpu", dtype=torch.float32), auto.to("cpu", dtype=torch.float32), atol=1e-4, rtol=1e-4
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA backend not available")
def test_auto_backend_uses_pytorch_flex_on_cuda():
    q = torch.randn(1, 4, 8, 32, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, 8, 32, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 4, 8, 32, device="cuda", dtype=torch.float16)
    out = world_flex_attn_forward(
        q, k, v, AttnMeta(flex_block_mask=None, q_len=8, kv_len=8), AttnConfig(causal=True), backend=AttnBackend.AUTO
    )
    ref = world_flex_attn_forward(
        q, k, v, AttnMeta(flex_block_mask=None, q_len=8, kv_len=8), AttnConfig(causal=True), backend=AttnBackend.PYTORCH_FLEX
    )
    assert torch.allclose(out, ref, atol=2e-3, rtol=2e-3)

