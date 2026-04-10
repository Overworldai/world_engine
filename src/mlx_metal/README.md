# mlx_metal — Apple Silicon Inference

MLX model, custom Metal kernels, and Apple Neural Engine (ANE) integration for
world model inference on Apple Silicon.

## Contents

```
src/mlx_metal/
  __init__.py              package root
  engine.py                MLXWorldEngine — WorldEngine subclass for Apple Silicon
  mlx_world_model.py       MLX WorldModel with selective W8A8 linear layers
  ane/                     Apple Neural Engine (ANE) TAEHV codec
    ae_ane.py              CoreMLTAEHV — stateful encoder/decoder on ANE via CoreML
    test_ane.py            ANE validation test
  ext/                     C++ MLX extension (native Metal 4 / NAX path)
    CMakeLists.txt
    setup.py               built automatically via uv sync
    bindings.cpp            nanobind -> Python
    kernels/
      w8a8_gemm.h           MLX Primitive headers (GEMM + fused quant)
      w8a8_gemm.cpp         host-side dispatch (tile selection, Metal buffer binding)
      w8a8_gemm.metal        v1 (both-staged) + v2 (A-direct) tiled NAX GEMM
      w8a8_matvec.metal      SIMD dot-product matvec for M<5 (decode path)
      w8a8_fused_silu_quant.metal        ZeroQuant: fused SiLU + int8 quant
      w8a8_fused_rmsnorm_quant.metal     ZeroQuant: fused RMSNorm (+AdaLN) + int8 quant
    we_kernels/
      __init__.py            Python wrappers
  benchmarks/
    bench_gemm.py          GEMM kernel timing: fp16 vs W8A16 vs W8A8
    bench_e2e.py           end-to-end operator chains (activation + GEMM) across quant strategies
    bench_fused_quant.py   ZeroQuant fused kernel benchmarks
    bench_mlx.py           full model benchmark (fp16 / speed / max_qat profiles)
    bench_render.py        full render pipeline: MLX model + TAEHV decode → PNG frames
    bench_engine.py        MLXWorldEngine end-to-end benchmark
```

## MLXWorldEngine

`MLXWorldEngine` subclasses `WorldEngine` for Apple Silicon. It runs the world
model on MLX (Metal GPU) and the TAEHV video codec on the Neural Engine via CoreML.

```python
from world_engine import MLXWorldEngine, CtrlInput

engine = MLXWorldEngine("Overworld/Waypoint-1.5-1B", ane_vae=True)
engine.set_prompt("A fun game")
engine.append_frame(seed_img)  # [4, 720, 1280, 3] uint8

for ctrl in controls:
    img = engine.gen_frame_pipelined(ctrl=ctrl)
    if img is not None:
        display(img)  # [4, 720, 1280, 3] uint8
img = engine.flush_pipeline()
```

### Hardware mapping

| Component | Hardware | Latency |
|-----------|----------|---------|
| World model (denoise + cache) | Metal GPU via MLX | ~30-80ms |
| TAEHV encoder | ANE via CoreML | ~12ms |
| TAEHV decoder (stateful) | ANE via CoreML | ~22ms |

`gen_frame_pipelined()` overlaps GPU denoise with ANE decode in a background
thread. `gen_frame()` runs synchronously.

### ANE TAEHV decoder

The TAEHV decoder maintains temporal state (MemBlock memories) between frames.
CoreML's `StateType` doesn't compile on ANE (error -14), so state is passed as
explicit model inputs/outputs — the caller manages the state ring between
`predict()` calls. Output state is built with `torch.cat` (not `zeros + scatter`,
which is ~40ms slower on ANE).

Exported models are cached in `diagnostics/taehv_ane/` and auto-generated on
first use.

## Architecture

### M5 Neural Accelerators (NAX)

The M5 GPU embeds a dedicated matrix-multiply unit (Neural Accelerator) in every
GPU core. The `matmul2d` operation supports `char x char -> int` (int8 x int8 -> int32)
as documented in Table 7.3 of the
[Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
(Section 7.2.1). The NAX hardware runs int8 MMA at ~2x the throughput of fp16
([benchmark](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)).

### Kernel implementation

The tiled GEMM kernel (`ext/kernels/w8a8_gemm.metal`) uses MLX's Steel NAX
abstractions (`NAXTile`, `BaseNAXFrag`, `tile_matmad_nax`) for the int8 MMA path.
Both v1 (both A+B staged) and v2 (A-direct, B staged) variants live in one file,
selected at dispatch time based on N/K ratio.

Key design:
- **v1 (both-staged)**: A and B through threadgroup. Wins for wide N (≥4096).
- **v2 (A-direct)**: A loads from device, B staged. Halves TG usage. Wins for square shapes.
- **int8 threadgroup memory** — 2x denser than fp16, enabling larger BK tiles
- **MLX NAXTile fragment loads** with compile-time `Int<1>` column stride
- **MPP `matmul2d` cooperative tensors** — int8 × int8 → int32 on NAX hardware
- **Per-row scale epilogue** — `out = (int32_accum * x_scale * w_scale + bias)` cast to fp16
- **Multiple tile specializations** selected at runtime based on M, N, K
- **Matvec kernel** for M<5 (decode path): SIMD dot-product with 128-bit vector loads

### ZeroQuant fused activation quantization

Fuses dynamic activation quantization into the preceding operator (RMSNorm, SiLU),
eliminating a separate kernel dispatch and memory round-trip per GEMM:

```
Unfused:  RMSNorm → fp16 → [quant kernel] → int8 + scale → W8A8 GEMM → fp16
Fused:    RMSNorm+Quant → int8 + scale → W8A8 GEMM → fp16
```

Fused Metal kernels:
- **`fused_rmsnorm_quant`** / **`fused_rmsnorm_adaln_quant`** — RMSNorm (+ optional
  AdaLN `*(1+s)+b` modulation) + per-row int8 quantization in 3 phases: sum-of-squares
  reduction → normalize + absmax → quantize. Feeds QKV and MLP fc1 projections.
- **`fused_rmsnorm_adaln_smooth_quant`** / **`fused_rmsnorm_smooth_quant`** — Same as
  above but with per-channel SmoothQuant scale applied after normalization/modulation
  and before quantization: `v = (rms_norm(x) * (1+s) + b) * smooth_scale[k]`.
- **`fused_silu_quant`** — SiLU activation + per-row int8 quantization in 2 phases:
  SiLU + absmax → quantize. Feeds MLP fc2 (via `Int8NaxSiLULinear`).

### SmoothQuant integration

Loads pre-calibrated per-channel smooth scales from any applicable smoothquant model checkpoint.
96 smooth scales total: per block × 4 (q_proj, k_proj, v_proj, mlp.fc1).

For merged QKV projections, the three separate q/k/v scales are unified via element-wise
max (matching PyTorch's `merge_qkv_smoothscales`), with weight compensation applied.
Smooth scales are fused directly into the Metal quantization kernels — no fallback path.

## Setup

### Requirements

- macOS 26+ with Apple Silicon M5+
- Python 3.10+
- MLX (built from source via `uv sync`)
- Xcode (for Metal compiler) + Metal 4.0

### Running benchmarks

```bash
# Engine (end-to-end via MLXWorldEngine — ANE decode by default)
uv run python -m src.mlx_metal.benchmarks.bench_engine
uv run python -m src.mlx_metal.benchmarks.bench_engine --save-frames
uv run python -m src.mlx_metal.benchmarks.bench_engine --no-ane        # CPU decode

# Render (detailed model/decode split timing, stability analysis)
uv run python -m src.mlx_metal.benchmarks.bench_render --ane --save-frames
uv run python -m src.mlx_metal.benchmarks.bench_render --smoothquant --save-frames
uv run python -m src.mlx_metal.benchmarks.bench_render --stability --frames 60

# ANE TAEHV validation
uv run python -m src.mlx_metal.ane.test_ane

# Kernels
uv run python -m src.mlx_metal.benchmarks.bench_gemm
uv run python -m src.mlx_metal.benchmarks.bench_gemm --shapes sweep --accuracy
uv run python -m src.mlx_metal.benchmarks.bench_e2e
uv run python -m src.mlx_metal.benchmarks.bench_fused_quant
uv run python -m src.mlx_metal.benchmarks.bench_mlx
uv run python -m src.mlx_metal.benchmarks.bench_mlx --smoothquant
```
