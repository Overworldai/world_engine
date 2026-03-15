import types

import pytest
import torch

from kv_cache import LayerKVCache, StaticKVCache


def _pos_ids(frame_idx: int, b: int, t: int, device: str = "cpu"):
    return {"f_pos": torch.full((b, t), frame_idx, device=device, dtype=torch.long)}


def _new_layer_cache(*, l=16, tpf=4, pd=1):
    # num_buckets = (L/tpf)/pd
    return LayerKVCache(B=1, H=2, L=l, Dh=8, dtype=torch.float32, tokens_per_frame=tpf, pinned_dilation=pd)


def test_layer_kv_cache_returns_active_blocks_on_metal_backend(monkeypatch):
    monkeypatch.setenv("WORLD_ATTENTION_BACKEND", "metal")
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    cache = _new_layer_cache(l=16, tpf=4, pd=1)
    kv = torch.randn(2, 1, 2, 4, 8)

    _k, _v, _bm, block_written, active_blocks, _bs = cache.upsert(
        kv, _pos_ids(0, 1, 4), is_frozen=False, frame_idx_int=0
    )
    assert active_blocks is not None
    expected = torch.nonzero(block_written, as_tuple=False).flatten().to(torch.int32)
    assert torch.equal(active_blocks.cpu(), expected.cpu())


def test_layer_kv_cache_saturated_path_avoids_nonzero_write_step(monkeypatch):
    monkeypatch.setenv("WORLD_ATTENTION_BACKEND", "metal")
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    cache = _new_layer_cache(l=16, tpf=4, pd=1)  # num_buckets=4
    kv = torch.randn(2, 1, 2, 4, 8)

    # Saturate ring by writing each slot once.
    for frame_idx in range(4):
        cache.upsert(kv, _pos_ids(frame_idx, 1, 4), is_frozen=False, frame_idx_int=frame_idx)
    assert len(cache._seen_slots) == cache.num_buckets

    # If this path still calls nonzero, test should fail.
    monkeypatch.setattr(torch, "nonzero", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nonzero called")))
    _k, _v, _bm, _block_written, active_blocks, _bs = cache.upsert(
        kv, _pos_ids(4, 1, 4), is_frozen=False, frame_idx_int=4
    )
    # frame_idx=4 -> slot 0 masked out for this call, so active excludes block 0.
    assert torch.equal(active_blocks.cpu(), torch.tensor([1, 2, 3, 4], dtype=torch.int32))


def test_layer_kv_cache_saturated_path_avoids_nonzero_non_write_step(monkeypatch):
    monkeypatch.setenv("WORLD_ATTENTION_BACKEND", "metal")
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    cache = _new_layer_cache(l=16, tpf=4, pd=2)  # num_buckets=2, write steps at 0,2,4,...
    kv = torch.randn(2, 1, 2, 4, 8)

    cache.upsert(kv, _pos_ids(0, 1, 4), is_frozen=False, frame_idx_int=0)
    cache.upsert(kv, _pos_ids(2, 1, 4), is_frozen=False, frame_idx_int=2)
    assert len(cache._seen_slots) == cache.num_buckets

    monkeypatch.setattr(torch, "nonzero", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nonzero called")))
    _k, _v, _bm, _block_written, active_blocks, _bs = cache.upsert(
        kv, _pos_ids(3, 1, 4), is_frozen=True, frame_idx_int=3
    )
    # For dilation=2, only the pinned ring subset plus tail are active.
    assert torch.equal(active_blocks.cpu(), torch.tensor([0, 1, 4], dtype=torch.int32))


def test_layer_kv_cache_unsaturated_regular_geometry_avoids_nonzero(monkeypatch):
    monkeypatch.setenv("WORLD_ATTENTION_BACKEND", "metal")
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    cache = _new_layer_cache(l=16, tpf=4, pd=1)
    kv = torch.randn(2, 1, 2, 4, 8)

    calls = {"n": 0}
    real_nonzero = torch.nonzero

    def _counting_nonzero(*args, **kwargs):
        calls["n"] += 1
        return real_nonzero(*args, **kwargs)

    monkeypatch.setattr(torch, "nonzero", _counting_nonzero)
    cache.upsert(kv, _pos_ids(0, 1, 4), is_frozen=False, frame_idx_int=0)
    assert calls["n"] == 0


def test_layer_kv_cache_irregular_geometry_falls_back_to_nonzero(monkeypatch):
    monkeypatch.setenv("WORLD_ATTENTION_BACKEND", "metal")
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    # tpf=5 is not divisible by block_size=4, so nonzero fallback is required.
    cache = _new_layer_cache(l=20, tpf=5, pd=1)
    kv = torch.randn(2, 1, 2, 5, 8)

    calls = {"n": 0}
    real_nonzero = torch.nonzero

    def _counting_nonzero(*args, **kwargs):
        calls["n"] += 1
        return real_nonzero(*args, **kwargs)

    monkeypatch.setattr(torch, "nonzero", _counting_nonzero)
    cache.upsert(kv, _pos_ids(0, 1, 5), is_frozen=False, frame_idx_int=0)
    assert calls["n"] >= 1


def test_load_state_rebuilds_seen_slots(monkeypatch):
    monkeypatch.setenv("WORLD_ATTENTION_BACKEND", "metal")
    monkeypatch.setenv("WORLD_METAL_BLOCK_SIZE", "4")
    cfg = types.SimpleNamespace(
        height=2,
        width=2,
        local_window=4,
        global_window=4,
        global_attn_period=2,
        global_attn_offset=0,
        global_pinned_dilation=2,
        n_layers=2,
        n_kv_heads=2,
        n_heads=2,
        d_model=16,
    )
    cache = StaticKVCache(cfg, batch_size=1, dtype=torch.float32)
    kv = torch.randn(1, 2, 4, 8)
    pos = _pos_ids(0, 1, 4)
    cache.set_frozen(False)
    cache.upsert(kv, kv, pos, layer=0)

    state = cache.get_state()
    clone = StaticKVCache(cfg, batch_size=1, dtype=torch.float32)
    clone.load_state(state)
    assert len(clone.layers[0]._seen_slots) > 0
    assert clone.layers[0]._metal_bs_cache == 0

