## Hybrid Metal attention backend

This repository includes an experimental **hybrid Metal backend** for attention.
The high-level model and inference loop remain in PyTorch, while the hottest
attention path can be routed through custom Metal ops on Apple M-series GPUs.

### Selecting attention backend

Attention backends are controlled via the `WORLD_ATTENTION_BACKEND` environment
variable:

- `flex` (default): use PyTorch `flex_attention` everywhere.
- `metal`: use custom `world.flex_attn_metal_*` ops on MPS devices.
- `auto`: choose based on availability/device.

Example:

```bash
WORLD_ATTENTION_BACKEND=metal WORLD_METAL_IMPL=fast python examples/gen_sample.py
```

### Implementation overview

- Python-side wrappers:
  - `src/model/attn_backend.py` defines:
    - `AttnBackend`: backend selector (`pytorch-flex`, `metal-op`, `auto`).
    - `AttnConfig` / `AttnMeta`: small structs describing behavior and KV
      geometry.
    - `world_flex_attn_forward(...)`: single entry point used by attention
      modules.
- Call sites:
  - `Attn`, `MergedQKVAttn`, and `CrossAttention` now call
    `world_flex_attn_forward` instead of `flex_attention` directly.
- Metal custom op:
  - `src/metal/metal_flex_attn_op.mm` registers
    `torch.ops.world.flex_attn_metal` on the MPS backend and wires it to the
    `metal_flex_attn_forward` Metal kernel in
    `src/metal/metal_flex_attn.metal`.
- Tests:
  - `tests/test_metal_attn_numeric.py` compares Metal vs flex attention on
    small random inputs (when the Metal op is available).
  - `tests/test_metal_attn_perf.py` provides a basic throughput sanity check on
    M‑series devices.

### Status

Attention Metal kernels include fast sparse/block-aware paths and a reference
path.

Known limitations:

- Attention Metal path is inference-only.
- Fast specialized kernels are tuned for float16; bfloat16 is supported via native generic kernel when available (otherwise fp16 boundary fallback).

### End-to-end benchmark

Use this to track actual generation latency/FPS on MPS:

```bash
python tests/bench_world_engine_e2e.py --model-uri <your_model_uri> --attention-backend metal --dtype float16 --quant w8a8 --scheduler-steps 4 --cache-interval 1
```

Add `--return-img` to include VAE decode in the benchmarked path.

### Regression-safe performance gate

Capture a locked baseline (3 repeats):

```bash
python tests/perf_regression_gate.py --output docs/perf_baseline_mps_w8a8.json --repeats 3 --warmup 16 --steps 8
```

Compare current code to baseline (fails on regression beyond threshold):

```bash
python tests/perf_regression_gate.py --output docs/perf_baseline_mps_w8a8.json --compare-only --repeats 3 --warmup 16 --steps 8 --max-regression 0.15
```

### Current validated throughput (strict pretrained path)

`Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill`, `scheduler_steps=4`, `cache_interval=1`, `float16`, `w8a8`:

- latent-only: `total_ms p50 ~210.8`, `FPS p50 ~4.74`
- with decode: `total_ms p50 ~219.3`, `FPS p50 ~4.56`

### Optimization gate workflow (baseline-safe)

Use the new optimization gate runner to ensure every speed change is validated
against tensor-dump correctness and performance thresholds:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1 \
WORLD_ATTENTION_BACKEND=metal WORLD_METAL_IMPL=fast WORLD_METAL_FAST_NO_FALLBACK=1 \
WORLD_METAL_PREFER_ACTIVE_DISPATCH=1 WORLD_KV_RUNTIME_CHECKS=0 WORLD_KV_COMPUTE_ACTIVE_BLOCKS=0 \
PYTHONPATH=. ./.venv/bin/python tests/run_optimization_gate.py \
  --model-uri Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill \
  --device mps --dtype bfloat16 --profile-steps 16 \
  --baseline-dump-dir diagnostics/out/metal_profile_baseline \
  --baseline-perf-report diagnostics/out/metal_profile_perf_only/profile_report.json \
  --output-dir diagnostics/out/optimization_gate_run
```

Artifacts written:

- `gate_report.json` (overall decision)
- perf run reports under `perf/`
- dump run reports under `dump/`
- quick/full comparisons under `compare_quick/` and `compare_full/`

### Tensor-dump regression comparison

You can compare any candidate dump run against baseline directly:

```bash
PYTHONPATH=. ./.venv/bin/python tests/compare_tensor_dumps.py \
  --baseline-dir diagnostics/out/metal_profile_baseline \
  --candidate-dir diagnostics/out/optimization_gate_run/dump \
  --phase all --strict \
  --out-dir diagnostics/out/optimization_gate_run/manual_compare
```

The comparison emits:

- `comparison_summary.json`
- `comparison_worst_modules.json`
- `comparison_full.json`

