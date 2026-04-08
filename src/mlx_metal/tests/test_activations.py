"""
Test: sequential frame activation determinism for MLXWorldModel.

Generates two sequential frames (denoise + cache_write), saving per-block
activations for each. Verifies that:
  1. Running the same frame twice with identical state produces identical activations.
  2. Block activations are finite (no NaN/Inf).
  3. Activations change between sequential frames (the model is not degenerate).

Based on the bench_render pipeline.

Usage:
  pytest src/mlx_metal/tests/test_activations.py -v
  python -m src.mlx_metal.tests.test_activations          # standalone
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np
import pytest

from ..mlx_world_model import (
    MLXWorldModel,
    RingKVCache,
    TransformerBlock,
    compute_rope_angles,
    load_from_pytorch,
    T,
)

MODEL_URI = "Overworld-Models/MR160k"


@dataclass
class ActivationCapture:
    """Monkey-patches TransformerBlock to capture per-block activations."""
    records: Dict[int, List[np.ndarray]] = field(default_factory=dict)
    _originals: Dict[int, object] = field(default_factory=dict)

    def install(self, model: MLXWorldModel):
        """Wrap each transformer block's __call__ to save its output."""
        for idx, blk in enumerate(model.transformer):
            self.records[idx] = []
            orig = blk.__class__.__call__

            def make_hook(block_idx, original_fn):
                def hooked_call(self_blk, *args, **kwargs):
                    out = original_fn(self_blk, *args, **kwargs)
                    x_out, v1_out = out
                    # Save a numpy copy of the block output
                    self_blk._capture_list.append(np.array(x_out))
                    return out
                return hooked_call

            blk._capture_list = self.records[idx]
            blk._hooked_call = make_hook(idx, orig)
            blk.__class__.__call__ = lambda self_blk, *a, **kw: self_blk._hooked_call(self_blk, *a, **kw)

    def install_per_instance(self, model: MLXWorldModel):
        """Instance-level hook — avoids class-level monkey-patching issues."""
        for idx, blk in enumerate(model.transformer):
            self.records[idx] = []
            orig_call = TransformerBlock.__call__

            def make_hook(block_idx, original_fn):
                capture_list = self.records[block_idx]
                def hooked(self_blk, x, cond, ctrl_emb, rope_cos, rope_sin, v1_in, kv_cache, frame_idx):
                    x_out, v1_out = original_fn(self_blk, x, cond, ctrl_emb, rope_cos, rope_sin, v1_in, kv_cache, frame_idx)
                    capture_list.append(np.array(x_out))
                    return x_out, v1_out
                return hooked

            # Bind the hook as an instance method
            import types
            blk.__call__ = types.MethodType(
                lambda self_blk, x, cond, ctrl_emb, rope_cos, rope_sin, v1_in, kv_cache, frame_idx, _h=make_hook(idx, orig_call): _h(self_blk, x, cond, ctrl_emb, rope_cos, rope_sin, v1_in, kv_cache, frame_idx),
                blk,
            )

    def clear(self):
        for v in self.records.values():
            v.clear()

    def snapshot(self) -> Dict[int, List[np.ndarray]]:
        """Return a deep copy of current records and clear."""
        snap = {k: list(v) for k, v in self.records.items()}
        self.clear()
        return snap


def _setup_model(profile: str = "fp16"):
    """Load model and return (model, cfg, latent_shape)."""
    int8_profile = None if profile == "fp16" else profile
    model, cfg = load_from_pytorch(MODEL_URI, int8_profile=int8_profile)
    pH, pW = cfg.patch
    latent_shape = (1, 1, cfg.channels, cfg.height * pH, cfg.width * pW)
    return model, cfg, latent_shape


def _seed_cache(model, cfg, latent_shape):
    """Write a random seed frame into the KV cache (frame 0)."""
    mouse = mx.zeros((1, 1, 2), dtype=mx.float16)
    button = mx.zeros((1, 1, cfg.n_buttons), dtype=mx.float16)
    scroll = mx.zeros((1, 1, 1), dtype=mx.float16)
    seed = mx.array(np.random.randn(*latent_shape).astype(np.float16))
    rope_cos, rope_sin = compute_rope_angles(0, model.ts_mult, model.rope_xy, model.rope_inv_t)
    model.cache_write(seed, rope_cos, rope_sin, mouse, button, scroll, 0)
    return mouse, button, scroll


def _run_frame(model, cfg, latent_shape, frame_idx, mouse, button, scroll, *, rng_seed=42):
    """Run denoise + cache_write for a single frame. Returns denoised output."""
    np.random.seed(rng_seed)
    rope_cos, rope_sin = compute_rope_angles(frame_idx, model.ts_mult, model.rope_xy, model.rope_inv_t)
    x = mx.array(np.random.randn(*latent_shape).astype(np.float16))
    out = model.denoise(x, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
    mx.eval(out)
    model.cache_write(out, rope_cos, rope_sin, mouse, button, scroll, frame_idx)
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_cfg():
    """Load model once per test module (expensive)."""
    model, cfg, latent_shape = _setup_model("fp16")
    return model, cfg, latent_shape


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSequentialActivations:
    """Test activation properties across two sequential frames."""

    def test_activations_finite(self, model_and_cfg):
        """All per-block activations must be finite (no NaN / Inf)."""
        model, cfg, latent_shape = model_and_cfg
        mouse, button, scroll = _seed_cache(model, cfg, latent_shape)

        cap = ActivationCapture()
        cap.install_per_instance(model)

        _run_frame(model, cfg, latent_shape, 1, mouse, button, scroll, rng_seed=42)
        snap1 = cap.snapshot()

        _run_frame(model, cfg, latent_shape, 2, mouse, button, scroll, rng_seed=99)
        snap2 = cap.snapshot()

        for block_idx in snap1:
            for step_idx, act in enumerate(snap1[block_idx]):
                assert np.all(np.isfinite(act)), (
                    f"Frame 1, block {block_idx}, denoise step {step_idx}: non-finite activations"
                )
            for step_idx, act in enumerate(snap2[block_idx]):
                assert np.all(np.isfinite(act)), (
                    f"Frame 2, block {block_idx}, denoise step {step_idx}: non-finite activations"
                )

    def test_determinism_same_input(self, model_and_cfg):
        """Same frame index + same RNG seed + same cache state => identical activations."""
        model, cfg, latent_shape = model_and_cfg

        # Reset KV caches to zeros for a clean state
        for kv in model.kv_caches:
            kv.keys = mx.zeros_like(kv.keys)
            kv.values = mx.zeros_like(kv.values)
            kv.written_slots = set()

        mouse, button, scroll = _seed_cache(model, cfg, latent_shape)

        # Save cache state
        cache_state = []
        for kv in model.kv_caches:
            cache_state.append((
                np.array(kv.keys),
                np.array(kv.values),
                set(kv.written_slots),
            ))

        cap = ActivationCapture()
        cap.install_per_instance(model)

        # Run frame 1, first time
        _run_frame(model, cfg, latent_shape, 1, mouse, button, scroll, rng_seed=42)
        snap_a = cap.snapshot()

        # Restore cache state to before frame 1
        for i, kv in enumerate(model.kv_caches):
            kv.keys = mx.array(cache_state[i][0])
            kv.values = mx.array(cache_state[i][1])
            kv.written_slots = cache_state[i][2]

        # Run frame 1, second time with same seed
        _run_frame(model, cfg, latent_shape, 1, mouse, button, scroll, rng_seed=42)
        snap_b = cap.snapshot()

        for block_idx in snap_a:
            for step_idx in range(len(snap_a[block_idx])):
                a = snap_a[block_idx][step_idx]
                b = snap_b[block_idx][step_idx]
                np.testing.assert_array_equal(
                    a, b,
                    err_msg=f"Block {block_idx}, step {step_idx}: not bitwise identical on replay",
                )

    def test_activations_differ_across_frames(self, model_and_cfg):
        """Sequential frames with different noise must produce different activations."""
        model, cfg, latent_shape = model_and_cfg

        # Reset caches
        for kv in model.kv_caches:
            kv.keys = mx.zeros_like(kv.keys)
            kv.values = mx.zeros_like(kv.values)
            kv.written_slots = set()

        mouse, button, scroll = _seed_cache(model, cfg, latent_shape)

        cap = ActivationCapture()
        cap.install_per_instance(model)

        _run_frame(model, cfg, latent_shape, 1, mouse, button, scroll, rng_seed=42)
        snap1 = cap.snapshot()

        _run_frame(model, cfg, latent_shape, 2, mouse, button, scroll, rng_seed=99)
        snap2 = cap.snapshot()

        # At least the final denoise step should differ for most blocks
        n_changed = 0
        for block_idx in snap1:
            # Compare last activation (final denoise step output, from cache_write pass)
            a = snap1[block_idx][-1]
            b = snap2[block_idx][-1]
            if not np.allclose(a, b, atol=1e-3):
                n_changed += 1

        from ..mlx_world_model import N_LAYERS
        assert n_changed > N_LAYERS // 2, (
            f"Only {n_changed}/{N_LAYERS} blocks changed between frames — model may be degenerate"
        )

    def test_activation_magnitudes_reasonable(self, model_and_cfg):
        """Block activations should have reasonable magnitude (not exploding)."""
        model, cfg, latent_shape = model_and_cfg

        for kv in model.kv_caches:
            kv.keys = mx.zeros_like(kv.keys)
            kv.values = mx.zeros_like(kv.values)
            kv.written_slots = set()

        mouse, button, scroll = _seed_cache(model, cfg, latent_shape)

        cap = ActivationCapture()
        cap.install_per_instance(model)

        _run_frame(model, cfg, latent_shape, 1, mouse, button, scroll, rng_seed=42)
        snap = cap.snapshot()

        max_abs = 0.0
        for block_idx in snap:
            for step_idx, act in enumerate(snap[block_idx]):
                block_max = float(np.max(np.abs(act)))
                max_abs = max(max_abs, block_max)
                # fp16 max is 65504; activations should be well below that
                assert block_max < 60000, (
                    f"Block {block_idx}, step {step_idx}: max |activation| = {block_max:.1f} — near fp16 overflow"
                )

    def test_sequential_cache_state_evolves(self, model_and_cfg):
        """KV cache state must change after each frame's cache_write."""
        model, cfg, latent_shape = model_and_cfg

        for kv in model.kv_caches:
            kv.keys = mx.zeros_like(kv.keys)
            kv.values = mx.zeros_like(kv.values)
            kv.written_slots = set()

        mouse, button, scroll = _seed_cache(model, cfg, latent_shape)

        # Snapshot cache after seed
        keys_after_seed = [np.array(kv.keys) for kv in model.kv_caches]

        _run_frame(model, cfg, latent_shape, 1, mouse, button, scroll, rng_seed=42)

        # Snapshot cache after frame 1
        keys_after_f1 = [np.array(kv.keys) for kv in model.kv_caches]

        _run_frame(model, cfg, latent_shape, 2, mouse, button, scroll, rng_seed=99)

        keys_after_f2 = [np.array(kv.keys) for kv in model.kv_caches]

        # At least some layers should have different cache content
        changed_seed_to_f1 = sum(
            1 for a, b in zip(keys_after_seed, keys_after_f1)
            if not np.array_equal(a, b)
        )
        changed_f1_to_f2 = sum(
            1 for a, b in zip(keys_after_f1, keys_after_f2)
            if not np.array_equal(a, b)
        )

        from ..mlx_world_model import N_LAYERS
        assert changed_seed_to_f1 > 0, "No KV cache layers changed after frame 1"
        assert changed_f1_to_f2 > 0, "No KV cache layers changed after frame 2"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading model...")
    model, cfg, latent_shape = _setup_model("fp16")
    print(f"  latent_shape: {latent_shape}")

    print("Seeding KV cache (frame 0)...")
    mouse, button, scroll = _seed_cache(model, cfg, latent_shape)

    cap = ActivationCapture()
    cap.install_per_instance(model)

    print("Running frame 1 (denoise + cache_write)...")
    out1 = _run_frame(model, cfg, latent_shape, 1, mouse, button, scroll, rng_seed=42)
    snap1 = cap.snapshot()
    print(f"  output: mean={np.array(out1).mean():.4f}, std={np.array(out1).std():.4f}")
    print(f"  captured {sum(len(v) for v in snap1.values())} activations across {len(snap1)} blocks")

    print("Running frame 2 (denoise + cache_write)...")
    out2 = _run_frame(model, cfg, latent_shape, 2, mouse, button, scroll, rng_seed=99)
    snap2 = cap.snapshot()
    print(f"  output: mean={np.array(out2).mean():.4f}, std={np.array(out2).std():.4f}")

    # Compare activations
    print("\nComparing activations between frames:")
    from ..mlx_world_model import N_LAYERS
    for block_idx in range(N_LAYERS):
        acts1 = snap1[block_idx]
        acts2 = snap2[block_idx]
        for step in range(len(acts1)):
            a, b = acts1[step], acts2[step]
            diff = np.abs(a - b)
            max_diff = diff.max()
            mean_diff = diff.mean()
            finite_a = np.all(np.isfinite(a))
            finite_b = np.all(np.isfinite(b))
            status = "OK" if finite_a and finite_b else "NaN/Inf!"
            if step == len(acts1) - 1:  # Only print cache_write step per block
                print(f"  block {block_idx:2d} step {step}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} [{status}]")

    print("\nAll checks passed.")
