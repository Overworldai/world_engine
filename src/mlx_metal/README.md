# mlx_metal — W8A8 Quantized Inference on Apple Silicon

Custom Metal kernels and MLX model code for W8A8 symmetric quantized inference
on Apple Silicon GPUs.

## Contents

```
src/mlx_metal/
  __init__.py              package root (re-exports we_kernels.w8a8_gemm_nax)
  mlx_world_model.py       MLX WorldModel with selective W8A8 linear layers
  bench_w8a8.py            benchmark: fp16 vs W8A16 vs W8A8
  bench_mlx.py             end-to-end model benchmark
  ext/                     C++ MLX extension (native Metal 4 / NAX path)
    CMakeLists.txt
    setup.py               built automatically via uv sync
    bindings.cpp            nanobind -> Python
    kernels/
      w8a8_gemm.h           MLX Primitive header
      w8a8_gemm.cpp         host-side dispatch (tile selection, Metal buffer binding)
      w8a8_gemm.metal       NAX kernel (MLX Steel NAXTile, int8 MMA via MPP)
    we_kernels/
      __init__.py            Python wrapper (dynamic activation quant + reshape)
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

The C++ extension kernel (`ext/kernels/w8a8_gemm.metal`) uses MLX's own Steel NAX
abstractions (`NAXTile`, `BaseNAXFrag`, `tile_matmad_nax`) for the int8 MMA path (frankly,
because MLX has better code optimisation than anything I could come up with).

Key design:
- **Both A and B staged through threadgroup** with coalesced 128-bit device reads
- **int8 threadgroup memory** — 2x denser than fp16, enabling larger BK tiles
- **MLX NAXTile fragment loads** with compile-time `Int<1>` column stride for
  vectorized reads and `const_for_loop` for static register allocation
- **MPP `matmul2d` cooperative tensors** — int8x int8 -> int32 on NAX hardware
- **Per-row scale epilogue** — `out = (int32_accum * x_scale * w_scale + bias)` cast to fp16
- **Multiple tile specializations** selected at runtime based on M, N, K

## Setup

### Requirements

- macOS 26+ with Apple Silicon M5+
- Python 3.10+
- MLX (built from source via `uv sync`)
- Xcode (for Metal compiler) + Metal 4.0

### Running benchmarks

```bash
python -m src.mlx_metal.bench_w8a8
python -m src.mlx_metal.bench_w8a8 --shapes sweep --accuracy
```
