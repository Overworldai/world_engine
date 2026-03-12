from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "model"))

from kv_cache import LayerKVCache
from attn_backend import AttnBackend, AttnConfig, AttnMeta, world_flex_attn_forward


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS backend not available on this system",
)


def _require_metal_ops():
    if not hasattr(torch.ops, "world"):
        pytest.skip("Metal world namespace not registered")
    required = ["flex_attn_metal_ref", "flex_attn_metal_fast", "flex_attn_metal_fast_blocks", "flex_attn_metal_fast_active"]
    if not all(hasattr(torch.ops.world, name) for name in required):
        pytest.skip("Required Metal ops not registered")


def _pos_ids(frame_idx: int, B: int, T: int, device: str):
    return {"f_pos": torch.full((B, T), frame_idx, device=device, dtype=torch.long)}


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("gqa", [False, True])
def test_kv_cache_to_backend_path_matches_ref(causal, gqa, monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B = 1
    T = 8
    Dh = 64
    Hq = 8
    Hkv = 2 if gqa else Hq
    L_hist = 32
    block_size = 4

    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=torch.float16)
    kf = torch.randn(B, Hkv, T, Dh, device="mps", dtype=torch.float16)
    vf = torch.randn(B, Hkv, T, Dh, device="mps", dtype=torch.float16)

    cache = LayerKVCache(B, Hkv, L_hist, Dh, torch.float16, T).to("mps")
    # Write one frame to establish rolling state.
    _ = cache.upsert(torch.stack([kf, vf], dim=0), _pos_ids(0, B, T, "mps"), is_frozen=False)
    # Read/update next frame.
    k, v, _bm, block_written, active_blocks, bs = cache.upsert(
        torch.stack([kf, vf], dim=0), _pos_ids(1, B, T, "mps"), is_frozen=False
    )

    # Direct block-written fast path.
    out_fast_blocks = torch.ops.world.flex_attn_metal_fast_blocks(q, k, v, block_written, int(bs), causal)
    out_fast_active = torch.ops.world.flex_attn_metal_fast_active(q, k, v, active_blocks, int(bs), causal)

    # Dense-mask reference from block-written metadata.
    dense = torch.zeros((k.size(2),), device="mps", dtype=torch.uint8)
    for i in range(block_written.numel()):
        if int(block_written[i].item()) != 0:
            s = i * int(bs)
            e = min(k.size(2), s + int(bs))
            dense[s:e] = 1
    dense_mask = dense.view(1, 1, 1, k.size(2)).expand(B, Hq, T, k.size(2)).contiguous()
    out_ref = torch.ops.world.flex_attn_metal_ref(q, k, v, dense_mask, causal)

    assert out_fast_blocks.shape == out_ref.shape
    assert torch.allclose(
        out_fast_blocks.to("cpu", dtype=torch.float32),
        out_ref.to("cpu", dtype=torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )
    assert torch.allclose(
        out_fast_active.to("cpu", dtype=torch.float32),
        out_ref.to("cpu", dtype=torch.float32),
        atol=3e-2,
        rtol=3e-2,
    )


def test_world_flex_attn_forward_prefers_block_metadata(monkeypatch):
    _require_metal_ops()
    monkeypatch.setenv("WORLD_METAL_IMPL", "fast")
    monkeypatch.setenv("WORLD_METAL_FAST_NO_FALLBACK", "1")

    B, Hq, Hkv, T, L, Dh = 1, 8, 2, 8, 24, 64
    q = torch.randn(B, Hq, T, Dh, device="mps", dtype=torch.float16)
    k = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)
    v = torch.randn(B, Hkv, L, Dh, device="mps", dtype=torch.float16)

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
    out = world_flex_attn_forward(q, k, v, meta, cfg, backend=AttnBackend.METAL)

    direct = torch.ops.world.flex_attn_metal_fast_active(q, k, v, active_blocks, block_size, True)
    assert torch.allclose(
        out.to("cpu", dtype=torch.float32),
        direct.to("cpu", dtype=torch.float32),
        atol=1e-4,
        rtol=1e-4,
    )

