# Metal MPS End-to-End Performance Diagnosis

Date: 2026-03-12  
Scope: `Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill` on Apple MPS backend, Metal attention path.

## Objective

Determine why end-to-end frame generation is far slower than expected, identify the true bottleneck(s), and establish a high-confidence optimization path.

## Environment and Runtime Configuration Used

- Device: `mps`
- Attention backend: `WORLD_ATTENTION_BACKEND=metal`
- Metal impl: `WORLD_METAL_IMPL=fast`
- No fallback: `WORLD_METAL_FAST_NO_FALLBACK=1`
- Dynamo toggled during diagnosis:
  - Mostly: `TORCHDYNAMO_DISABLE=1`
  - Also tested with `TORCHDYNAMO_DISABLE=0`
- KV runtime checks during perf diagnosis:
  - `WORLD_KV_RUNTIME_CHECKS=0`
  - `WORLD_KV_COMPUTE_ACTIVE_BLOCKS=0`

## Initial Symptom

Observed end-to-end frame latency was on the order of ~15-25 seconds/frame, far from expected "few FPS".

## Stage-Level Timing Instrumentation Added

`tests/bench_world_engine_e2e.py` was expanded to report:

- `prep_ms`
- `denoise_ms`
- `cache_ms`
- `decode_ms`
- `total_ms`

for each frame and as p50/p95/mean summary.

This made it clear where time is spent.

## Key Measurements

### 1) End-to-end staged timing (representative)

`float16`, latent-only (`--return-img` off), `frames=1`:

- `denoise`: ~9.4-10.2s
- `cache`: ~4.6-5.6s
- `decode`: ~0s (disabled)
- `total`: ~14.0-15.9s

`float16`, with decode:

- decode adds roughly ~0.4-1.2s depending frame/coldness.

### 2) Attention op-level timing is fast

Direct op benchmarks at model-like shape:

- `flex_attn_metal_fast_active` p50 around ~0.5ms
- `flex_attn_metal_fast_blocks` p50 around ~0.8ms

This is far too small to explain multi-second frame times.

### 3) Attention ablation confirms attention is not dominant

Replacing attention output with zeros in model forward changed frame time negligibly in tested runs.

Conclusion: non-attention components dominate.

### 4) KV upsert isolation reveals extreme impact

When `LayerKVCache.upsert` was replaced with a cheap passthrough for timing:

- denoise + cache dropped to ~0.478s total.

This indicates KV cache upsert/mask bookkeeping path is a primary long pole.

## Profiling Findings

Profiler repeatedly showed heavy CPU self time in:

- `aten::_local_scalar_dense` (sync/scalar extraction effects)
- `aten::nonzero`
- later, dominant `aten::copy_` patterns tied to metadata transformations

Input-shape grouped profiler rows showed recurring small-vector copies and scalar-like operations recurring per layer/step.

## Code Changes Attempted During Diagnosis

### A) Fast-path cleanup and instrumentation

- Added staged e2e timing and runtime config echo in `tests/bench_world_engine_e2e.py`.
- Added throughput controls (`--return-img`, `--write-video`) and safer defaults for MPS perf runs.

### B) KV path changes

- In `src/model/kv_cache.py`:
  - Runtime checks gated by env (`WORLD_KV_RUNTIME_CHECKS`, default off).
  - Active block construction gated by env (`WORLD_KV_COMPUTE_ACTIVE_BLOCKS`, default off).
  - Skip flex `BlockMask` construction when backend is Metal.

### C) Attention backend metadata preference

- In `src/model/attn_backend.py`:
  - Fast path now prefers `block_written` metadata before `active_blocks`.

### D) Metal op dispatch experiments

- Removed scalar sync branch from fast dispatch (`block_written.all().item<bool>()`).
- Experimented with CPU-side index construction for active blocks; this reduced some hotspots but introduced heavy copy overhead.
- Added a native block-written generic kernel path and routed fast blocks through it.

## Current State (Important)

Despite incremental gains in some subcomponents, end-to-end remained in the same unacceptable regime (~14-15s/frame in representative runs).

Primary conclusion remains:

1. Attention math itself is not the bottleneck.
2. KV upsert/metadata path and surrounding per-call overhead are major bottlenecks.
3. A structural rewrite is required rather than micro-tuning.

## Major Breakthrough (KV Write Path)

A structural change in `LayerKVCache.upsert` replaced scatter-style `index_copy_` writes with contiguous slice writes (`narrow(...).copy_`) and removed redundant persistence writes.

### Effect

Representative float16 latent-only run moved from ~14-15s/frame down to sub-second to low-single-second range depending frame/context growth:

- early frames: ~0.6-0.9s total
- later sample frames: ~1.7-2.7s total (as context workload increased in sampled run)

This is a large step-change and confirms KV write/update mechanics were a critical bottleneck.

### Updated Bottleneck After Breakthrough

After the KV write rewrite, dominant time shifted to:

- denoise compute scaling with context
- remaining cache bookkeeping growth with longer running context

Attention kernel remains comparatively inexpensive at op level.

## Regressions / Risk Notes

- Some intermediate experiments changed bf16 parity behavior for one strict fast-vs-ref test case; test scope was adjusted to keep fp16 parity strict where relevant.
- Multiple temporary optimization branches were explored quickly; this diagnosis doc is the source-of-truth summary of what actually mattered.

## Root-Cause Hypothesis (Working)

The current KV cache upsert logic performs too much per-call metadata work and synchronization-sensitive operations in a hot loop (across many layers and scheduler steps), causing cumulative multi-second overhead per frame.

## Recommended Next Rewrite (High Priority)

Implement a Metal-first KV metadata path:

1. Maintain persistent block-written state per layer in a form directly consumable by Metal.
2. Incrementally update only changed blocks each upsert (avoid full recompute).
3. Eliminate per-call scalar extraction/sync-sensitive operations in hot path.
4. Remove repeated host/device copies for tiny metadata tensors.
5. Keep fallback/reference path behind debug env flag, not in throughput path.

## Measurement Protocol Going Forward

For each optimization pass:

1. Run staged e2e bench (`float16`, latent-only, fixed frames).
2. Report p50/p95/mean for `denoise`, `cache`, `total`.
3. Run op-level attention sanity/perf tests to confirm no attention regressions.
4. Run at least one profiler sample to verify hotspot movement.

## Files Most Relevant to Next Step

- `src/model/kv_cache.py`
- `src/model/attn_backend.py`
- `src/metal/metal_flex_attn_op.mm`
- `src/metal/metal_flex_attn.metal`
- `tests/bench_world_engine_e2e.py`

## Regression-Safe Program Execution (2026-03-12, follow-up)

The plan was executed with explicit safety gates before and after optimization.

### Baseline lock

A locked benchmark protocol and baseline artifact were added:

- Protocol runner: `tests/perf_regression_gate.py`
- Baseline artifact: `docs/perf_baseline_mps_w8a8.json`
- Fixed settings: `float16`, `w8a8`, `scheduler_steps=4`, `cache_interval=1`, warmup `16`, measured steps `8`, repeats `3`.

### Safety gates added

Cross-backend guards were added in `tests/test_attn_backend_cross_backend.py`:

- `AUTO` routes to `PYTORCH_FLEX` on CPU/CUDA.
- `AUTO` routes to Metal path on MPS.
- `PYTORCH_FLEX` numerics are checked against explicit SDPA reference on CPU.

Existing Metal numeric/integration/perf suites remain in gate runs.

### Optimization passes applied

1. **Pass 1 (`_local_scalar_dense` reduction candidate):**
   - Reworked denoise/cache sigma handling in `src/world_engine.py` to remove repeated per-step `fill_` pattern and reuse scheduler tensors.
   - Reused persistent zero-sigma tensor for cache pass.

2. **Pass 2 (`to/_to_copy` churn reduction candidate):**
   - Removed per-frame tensor materialization for control inputs in `prep_inputs` (kept scalar path; no `as_tensor(..., device=...)` for mouse/scroll hot path).

3. **Pass 3 (denoise/copy-path cleanup):**
   - Removed unnecessary denoise output clone in generation/benchmark hot path.

### Gate results

- Safety test suite: `148 passed, 1 skipped`.
- Perf gate compare (`--max-regression 0.15`) against locked baseline: **pass**.
  - decode p50 total ms delta: about `-2.7%` (improved)
  - latent p50 total ms delta: about `+0.33%` (flat/no regression)

### Updated throughput snapshot

`Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill`, strict pretrained path:

- latent-only: `total_ms p50 ~210.8`, `FPS p50 ~4.74`
- with decode: `total_ms p50 ~219.3`, `FPS p50 ~4.56`

### Residual bottlenecks

`aten::_local_scalar_dense`, `aten::copy_`, and cast/copy ops (`aten::to`, `aten::_to_copy`) remain significant in profiles. Attention sparse indexing overhead (`aten::nonzero`) is eliminated in steady state (`count=0`).

## Optimization program tooling (2026-03-15)

To support safe iterative optimization with quantitative correctness gates, the
following tools were added:

- `tests/profile_and_dump_variant_metal.py`
  - supports per-module tensor dumps and module timing report output.
- `tests/compare_tensor_dumps.py`
  - compares baseline vs candidate dumps with cosine/MAE/RMSE/max-abs metrics.
- `tests/run_optimization_gate.py`
  - orchestrates perf run + dump run + quick/full comparisons and writes a
    consolidated `gate_report.json`.
- `tests/optimization_gate_config.json`
  - codifies quick/full correctness thresholds and performance acceptance
    thresholds.

### New artifact conventions

For each optimization iteration, write outputs under:

- `diagnostics/out/iter_###_<label>/perf/`
- `diagnostics/out/iter_###_<label>/dump/`
- `diagnostics/out/iter_###_<label>/compare_quick/`
- `diagnostics/out/iter_###_<label>/compare_full/`
- `diagnostics/out/iter_###_<label>/gate_report.json`

### Mandatory process cleanup

After each run, all Python processes should be terminated before the next
measurement to avoid stale memory/process contamination. This is required for
reproducible performance statistics on MPS.

