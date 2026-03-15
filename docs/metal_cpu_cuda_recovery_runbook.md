# Metal/CPU/CUDA Recovery Runbook (Ultra Detailed)

This runbook is the execution plan to recover correct temporal coherence and backend parity for `Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill` on this repository.

It is designed to:
- minimize expensive CPU runs,
- isolate regressions quickly,
- enforce strict reproducibility,
- avoid resource leaks/hangs.

---

## Non-Negotiable Rule: Kill Python After Every Run

After **every** command that runs model code (CPU, MPS, CUDA, parity, tests, benchmarks), execute the cleanup command below.

This is mandatory even if the run appears to succeed.

```bash
python3 - <<'PY'
import os, signal, subprocess
me=os.getpid()
out=subprocess.check_output(['ps','-ax','-o','pid=,command='], text=True)
for line in out.splitlines():
    s=line.strip()
    if not s:
        continue
    p=s.split(None,1)
    if len(p)<2:
        continue
    pid=int(p[0]); cmd=p[1].lower()
    if pid==me:
        continue
    if 'python' in cmd:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
print('python_cleanup_done')
PY
```

If a run times out or hangs, run cleanup immediately before anything else.

---

## Scope and Success Criteria

We need to determine why current branch outputs are visually wrong while `wp-1.5` is known-good on CPU/CUDA.

Success criteria:
1. Current branch CPU short rollout matches `wp-1.5` CPU golden latent trajectory.
2. Current branch MPS/Metal short rollout matches current branch CPU within bf16 tolerance.
3. Medium-length outputs (32+ frames) no longer collapse into confetti/noise textures.
4. Regression tests fail on bad semantics and pass on fixed behavior.

---

## Environment and Paths

Repository root:
- `/Users/louiscastricato/overworld/world_engine`

Reference worktree:
- `/Users/louiscastricato/overworld/world_engine_wp15`

Use these defaults for deterministic runs unless a step says otherwise:
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `TORCHDYNAMO_DISABLE=1`
- `WORLD_KV_RUNTIME_CHECKS=0`
- `WORLD_KV_COMPUTE_ACTIVE_BLOCKS=0`

Seed URL (fixed):
- `https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed`

---

## Artifact Layout (Create First)

Create and use this structure:

```text
diagnostics/
  scripts/
  out/
  golden/
  reports/
```

Store all intermediate outputs here (no root-level dump).

Commands:

```bash
mkdir -p diagnostics/scripts diagnostics/out diagnostics/golden diagnostics/reports
```

Then run the mandatory Python cleanup command.

---

## Phase 1: Golden Baseline From `wp-1.5` (Single Expensive CPU Run)

### Objective
Capture one canonical short latent trajectory on `wp-1.5` CPU that all current-branch variants must match.

### Steps
1. Run latent-only short rollout (recommended 8 steps max).
2. Save:
   - latents per step (`.pt`)
   - metadata (`.json`): seed URL, controls, dtype, scheduler steps, cache interval, env vars
   - summary metrics (`.json`): adjacent latent cosine/MAE
3. Cleanup Python.

### Notes
- Do not generate video in this phase unless needed for human sanity check.
- This is the only long CPU run to start.

---

## Phase 2: Current Branch CPU vs Golden (Short Deterministic)

### Objective
Detect whether regression exists before Metal is involved.

### Steps
1. Run same latent-only short rollout on current branch CPU.
2. Compare step-by-step to `diagnostics/golden/wp15_cpu_latents.pt`.
3. Record first failing step and error magnitudes in `diagnostics/reports/current_cpu_vs_wp15.json`.
4. Cleanup Python.

### Interpretation
- If CPU already diverges from golden: regression is in shared model/inference path.
- If CPU matches golden: focus shifts to Metal-specific path.

---

## Phase 3: Fast Ablation Matrix (Current Branch CPU)

### Objective
Find the smallest feature set causing divergence from golden.

### Toggle axes (one at a time first, then combinations)
1. `patch_cached_noise_conditioning` ON/OFF
2. `patch_Attn_merge_qkv` ON/OFF
3. `patch_MLPFusion_split` ON/OFF
4. attention wrapper path:
   - direct `flex_attention`
   - `world_flex_attn_forward`
5. KV metadata optimization paths:
   - active-block arithmetic path
   - fallback/block-written path

### Execution protocol
For each configuration:
1. Run short latent rollout (same seed/controls).
2. Compare to golden.
3. Save row in `diagnostics/reports/ablation_matrix_cpu.json`:
   - config ID
   - first failing step
   - stepwise cos/mae
4. Cleanup Python.

### Exit condition
Stop when one toggle or minimal toggle set restores golden parity.

---

## Phase 4: Metal Parity Once CPU Path Is Re-Grounded

### Objective
After CPU path is fixed, ensure Metal matches CPU and stays stable over rollout.

### Steps
1. Run short latent rollout on current branch MPS/Metal with same controls/noise.
2. Compare CPU vs Metal stepwise latents.
3. Save `diagnostics/reports/cpu_vs_metal_short.json`.
4. Generate medium video (32 frames) only after short parity passes.
5. Cleanup Python after each run.

### Required metrics
- Per-step latent cosine
- Per-step latent MAE
- Optional decoded-frame cosine for sampled steps

---

## Phase 5: Attention and Mask Semantics Verification

### Objective
Prove mask semantics are aligned across wrapper paths and backends.

### Checks
1. Compare `world_flex_attn_forward` vs direct `flex_attention` on CPU with identical `q/k/v/meta`.
2. Verify `block_written` and `active_blocks` invariants against expected `mask_written`.
3. Validate causal/non-causal behavior intentionally with explicit flags in diagnostic script.
4. Cleanup Python after each check.

### Output
- `diagnostics/reports/attn_semantics_checks.json`

---

## Phase 6: Temporal Conditioning and Cache State Invariants

### Objective
Ensure temporal conditioning inputs and cache evolution are not drifting unexpectedly.

### Invariants to check per step
1. `frame_idx` monotonic increment
2. `frame_timestamp` monotonic and scaled correctly
3. `kv_cache._is_frozen` state transitions:
   - denoise pass: frozen
   - cache pass: unfrozen
4. `written` mask evolves as expected for ring/tail
5. `block_written` consistency with `written`
6. No invalid empty attention windows for active queries

### Output
- `diagnostics/reports/cache_temporal_invariants.json`

Cleanup Python after each run.

---

## Phase 7: Fix Application and Verification Gate

### Objective
Apply minimal fix and prove it.

### Gate sequence
1. Current CPU vs golden short parity: pass
2. Current Metal vs current CPU short parity: pass
3. 32-frame Metal video sanity: pass
4. 120-frame Metal stress sanity: pass
5. Cleanup Python after each gate run

If any gate fails, do not proceed to cleanup/docs finalization.

---

## Phase 8: Regression Tests to Add

Create tests that must pass before claiming resolution:

1. `tests/test_golden_short_rollout_cpu.py`
   - compares latent trajectory to stored golden artifact
2. `tests/test_attention_wrapper_semantics.py`
   - wrapper vs direct flex equivalence
3. `tests/test_kv_cache_state_trajectory.py`
   - ring/tail/written/block metadata invariants
4. `tests/test_metal_cpu_short_parity.py` (MPS-gated)
   - short latent parity thresholds

Each test command run must be followed by Python cleanup.

---

## Repo Cleanup Plan (After Fix Verified)

1. Move all ad-hoc debug scripts into `diagnostics/scripts/`
2. Move generated media to `diagnostics/out/`
3. Keep only essential reference artifacts in repo
4. Add `.gitignore` entries for transient diagnostics outputs
5. Remove dead toggles and one-off monkeypatch logic used during debugging
6. Update:
   - `README_metal_hybrid.md` (runtime behavior + validated config)
   - `docs/metal_mps_full_diagnosis.md` (root cause + fix evidence)

Cleanup Python after any validation runs performed during cleanup.

---

## Standard Command Template for Runs

Use this template for every scripted run:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
TORCHDYNAMO_DISABLE=1 \
WORLD_KV_RUNTIME_CHECKS=0 \
WORLD_KV_COMPUTE_ACTIVE_BLOCKS=0 \
PYTHONPATH=. \
./.venv/bin/python diagnostics/scripts/<script>.py <args...>
```

Then immediately execute the mandatory Python cleanup command.

---

## Execution Discipline Checklist

For every single run:
1. Confirm output path is under `diagnostics/out/`
2. Run command
3. Save machine-readable report (`.json`) if applicable
4. **Kill all Python tasks**
5. Record status in a run log (command, elapsed, pass/fail, key metrics)

Do not skip step 4.

---

## Current Working Hypotheses (Ranked)

1. Shared inference patch path regression (CPU+Metal), not purely Metal kernel bug.
2. Attention semantic mismatch in wrapper path under current metadata usage.
3. Temporal conditioning/cache update interaction over multi-step rollout.
4. Secondary backend-specific numeric drift amplifying an upstream semantic issue.

This ranking should be updated in `diagnostics/reports/hypothesis_status.json` as evidence changes.

