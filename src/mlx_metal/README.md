# mlx_metal — W8A8 Quantized Inference on Apple Silicon

Custom Metal kernels and MLX model code for W8A8 symmetric quantized inference
on Apple Silicon GPUs.

## Contents

```
src/mlx_metal/
  __init__.py              package root
  mlx_world_model.py       MLX WorldModel with selective W8A8 linear layers
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
```

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

Loads pre-calibrated per-channel smooth scales from `Overworld-Models/MR160k-smoothquant`.
96 smooth scales total: per block × 4 (q_proj, k_proj, v_proj, mlp.fc1).

For merged QKV projections, the three separate q/k/v scales are unified via element-wise
max (matching PyTorch's `merge_qkv_smoothscales`), with weight compensation applied.
Smooth scales are fused directly into the Metal quantization kernels — no fallback path.

Benchmark (standalone kernel, M=512):
| Kernel | Shape | Separate | Fused | Speedup |
|--------|-------|----------|-------|---------|
| SiLU+Quant | 512×8192 | 525μs | 203μs | **2.59×** |
| RMSNorm+AdaLN+Quant | 512×2048 | 258μs | 174μs | **1.48×** |

End-to-end (fused quant + GEMM vs fp16 baseline):
| Operation | fp16 | W8A8 fused | Speedup |
|-----------|------|-----------|---------|
| QKV proj (2048→6144) | 484μs | 348μs | **0.72×** |
| MLP fc2 (8192→2048) | 593μs | 478μs | **0.81×** |
| **Forward pass total** | **44.3ms** | **36.8ms** | **0.83×** |

## Known Issues

### fp16 overflow with AdaLN-gated models

The model was trained at **bfloat16** (exponent range up to 3.4e38). The AdaLN gates
have trained magnitudes of ~20× per layer. Through the `x = y * gate + residual`
accumulation across 24 residual layers, activations grow to 500–1800 absmax. This is
by design and works at bfloat16/fp32.

At fp16 (max 65504), these values are within range individually, but **intermediate
products overflow** during element-wise computation when the lazy evaluation graph
spans multiple transformer blocks. The issue manifests as NaN during multi-frame
generation with real image context in the KV cache:

- **fp32**: stable for unlimited frames
- **fp16 without seed context**: stable (activations stay smaller with zero KV history)
- **fp16 with real seed image**: NaN deterministically at frame 2+

The int8 W8A8 path inherits this limitation since all non-GEMM operations (attention,
RMSNorm, gate multiplication, residual adds) run at fp16. The int8 GEMM itself has
plenty of precision (int32 accumulator, fp32 epilogue).

## Setup

### Requirements

- macOS 26+ with Apple Silicon M5+
- Python 3.10+
- MLX (built from source via `uv sync`)
- Xcode (for Metal compiler) + Metal 4.0

### Running benchmarks

```bash
uv run python -m src.mlx_metal.benchmarks.bench_gemm
uv run python -m src.mlx_metal.benchmarks.bench_gemm --shapes sweep --accuracy
uv run python -m src.mlx_metal.benchmarks.bench_e2e
uv run python -m src.mlx_metal.benchmarks.bench_e2e --accuracy
uv run python -m src.mlx_metal.benchmarks.bench_fused_quant
uv run python -m src.mlx_metal.benchmarks.bench_fused_quant --accuracy
uv run python -m src.mlx_metal.benchmarks.bench_mlx
uv run python -m src.mlx_metal.benchmarks.bench_mlx --smoothquant
uv run python -m src.mlx_metal.benchmarks.bench_mlx --model-uri Overworld-Models/MR160k-smoothquant
uv run python -m src.mlx_metal.benchmarks.bench_render --save-frames
uv run python -m src.mlx_metal.benchmarks.bench_render --smoothquant --save-frames
```
