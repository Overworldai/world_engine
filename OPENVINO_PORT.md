# OpenVINO Port

Port of world_engine to Intel Xe3 iGPU via OpenVINO. Targets the Panther Lake Arc B390 (96 EUs, shared DDR5 memory).

## Architecture

Hybrid approach: Python orchestrates the denoising loop + KV cache state, OpenVINO runs the heavy compute (transformer forward, VAE encode/decode) on the iGPU.

The key challenge was **externalizing the KV cache** — the original code mutates KV buffers in-place inside `Attn.forward()` via FlexAttention + BlockMask. The portable version makes the model a pure function: KV tensors go in, updated KV tensors come out.

## Files

| File | Purpose |
|------|---------|
| `src/portable_attn.py` | `PortableAttn` / `PortableCrossAttention` — replaces FlexAttention with SDPA. RoPE rewritten to avoid `aten::unfold`. GQA via `repeat_interleave`. Per-layer KV geometry computed from config. |
| `src/stateless_kv.py` | `upsert_stateless()` — pure function KV cache. No `nn.Buffer`, no in-place mutation. Uses arithmetic instead of `torch.where` (Intel GPU OpenCL compiler bug workaround). `StatelessKVManager` holds KV state as plain tensor lists. |
| `src/portable_model.py` | `PortableWorldModel` — mirrors `WorldModel` but uses portable attention + stateless KV. `from_original()` copies weights from a loaded `WorldModel`. |
| `src/openvino_engine.py` | `OpenVINOWorldEngine` — drop-in replacement for `WorldEngine`. Pre-allocated I/O buffers, async inference, INT4 quantized model on GPU. |
| `scripts/export_openvino.py` | Exports VAE encoder, VAE decoder, and transformer to OpenVINO IR. Supports `--frozen-only` and `--denoise-loop` variants. |
| `scripts/quantize_openvino.py` | NNCF INT4/INT8 weight compression for exported IR models. |

## CUDA-Specific Things That Were Replaced

| Original | Portable Replacement |
|----------|---------------------|
| `flex_attention()` + `BlockMask` | `F.scaled_dot_product_attention()` + dense float mask |
| `torch.compile(fullgraph=True, mode="max-autotune")` | Removed (OpenVINO compilation replaces this) |
| `torch.autocast("cuda")` | Removed |
| `torch._dynamo` config | Removed |
| `FlashInfer nvfp4` / `torch._scaled_mm` FP8 | NNCF INT4 on OV IR |
| `TensorDict` for pos_ids | Plain dict |
| `nn.Buffer` + `index_copy_()` in KV cache | Arithmetic + broadcast comparison (no `torch.where`) |
| `Tensor.unfold()` in RoPE | `view(..., -1, 2)` + indexing |

## Intel GPU Workarounds

**OpenCL Select bug**: The Intel GPU OpenCL compiler has a bug where the `MASK` macro in generated `select` kernels expands incorrectly. This affects all `torch.where` ops. Fixed by replacing every `torch.where` in model code with arithmetic equivalents (bool-to-float multiply, logical ops).

**Per-channel INT4**: `group_size=-1` with `ratio=1.0` crashes oneDNN. Use `group_size=128, ratio=0.8` (80% INT4, 20% INT8 backup).

## Performance

Waypoint-1-Small (2560d, 22 layers, 40 heads, 256 tokens/frame) on Intel Arc B390 iGPU (96 EUs):

| Configuration | Time/frame | FPS |
|--------------|-----------|-----|
| FP32 portable, frozen-only GPU, cache on CPU | 83s | 0.012 |
| INT4 weights, FP16 KV, all GPU | 6.66s | 0.15 |
| + async inference + pre-allocated I/O buffers | **4.34s** | **0.23** |

**Profiling breakdown** (best config):
- Denoise (4 steps): ~3.6s (0.9s/step, of which 0.65s is GPU compute)
- Cache pass: ~0.5s
- VAE decode: ~0.2s

**Theoretical limit**: ~1.0s/frame (1 FPS) at INT4 compute-bound. The ~4x gap is OV runtime overhead per inference call (kernel dispatch, shared memory staging, synchronization).

## How to Run

```bash
# Install dependencies
pip install openvino nncf

# Export model to OpenVINO IR
python scripts/export_openvino.py --model-uri Overworld/Waypoint-1-Small --output-dir exported_models/

# Quantize to INT4
python scripts/quantize_openvino.py --ir-dir exported_models/ --mode int4_sym --skip-vae

# Run inference
python -c "
from world_engine.openvino_engine import OpenVINOWorldEngine
engine = OpenVINOWorldEngine('Overworld/Waypoint-1-Small', mode='openvino', ir_dir='exported_models/', device='GPU')
engine.reset()
frame = engine.gen_frame()  # returns [H, W, 3] uint8
"
```

## Next Steps

1. **Bake denoise loop into single OV graph** — eliminates 3 of 4 per-step Python round-trips (~1.5s saved). Currently OOMs during trace (5 passes through 10B model); needs memory-efficient tracing or graph stitching.
2. **OV remote tensors** — keep KV state on GPU between frames, avoid CPU-side copies entirely.
3. **Profile OV runtime overhead** — the 0.25s/step gap between GPU compute (0.65s) and wall clock (0.9s) is OV kernel dispatch + data staging.
4. **INT8 KV cache** — reduce KV buffers from 1.37GB (FP16) to 0.69GB. Requires quantization-aware attention.
