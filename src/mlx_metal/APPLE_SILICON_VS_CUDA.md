# Apple Silicon (MLX/Metal) vs NVIDIA (CUDA): Lessons Learned

Notes accumulated while writing custom Metal kernels for a transformer-based world
model on M5 Max — W8A8 GEMMs, fused QKV+RoPE+norm, KV cache management, and
SageAttention-style int8 attention. Many CUDA optimization patterns don't
translate; some are counterproductive. This doc captures what we discovered.

## TL;DR

- **NAX (Neural Accelerator eXtensions):** Apple's analog to NVIDIA tensor cores,
  but with very different programming model, flexibility, and pipelining behavior.
- **Unified memory:** No host↔device copies. Same physical RAM, different cache
  behavior.
- **Hardware:** M5 Max has lower compute:bandwidth ratio than NVIDIA — bandwidth
  matters more, compute throughput matters less.
- **Execution model:** No warp-level multithreading. ALU ops serialize with NAX
  MMA more than on NVIDIA, but partial overlap exists.
- **Compiler quirks (observed):** same kernel source can compile to different
  perf based on what else is in the `.metal` file — likely some form of
  whole-compilation-unit optimization decision, but we haven't verified.
  Workaround: isolate critical kernels in their own files.
- **Static-size template templates matter:** `Int<1>` compile-time strides and
  `const_for_loop<>` consistently produce better code than runtime-shape
  equivalents in our experience. We haven't characterized the exact gap but
  it's been worth the template machinery every time.
- **Lazy eval:** Isolated `mx.eval()` benchmarks are misleading. Always validate
  end-to-end. Per-call timing ≠ pipeline timing.
- **Kernel launch overhead matters:** Each MLX kernel launch is ~50us amortized.
  Fusing 96 launches/frame saves ~5ms.
- **No async copy:** No `cp.async` equivalent. Memory loads block compute.
- **Tile size ceiling:** ~96 registers/lane (vs NVIDIA's 256) caps MMA tiles.
- **Vectorization is the dominant lever** for device↔TG/register
  transfers. Explicit `half4` / `int4` / `char4` is worth 10-45% on
  streaming kernels; thread→chunk mapping (cross-lane vs per-thread
  sequential) is a secondary 1-6% that flips with TG size. See the
  kernel-by-kernel audit in the Memory Loading section.
- **Flag:** half4 threadgroup stores from a precompiled `.metallib`
  produce nondeterministic output under adversarial load on M5 Max
  (specific rows in one dispatch wave get stale bytes). A
  structurally-equivalent JIT kernel via `mx.fast.metal_kernel` does
  NOT reproduce, so the bug is in the metallib toolchain / primitive
  dispatch path, not the Metal spec. Workaround: keep TG writes scalar.
  See Memory Loading section + `tests/repro_half4_tg_race.py`.

---

## Apple Silicon GPU Architecture (vs NVIDIA SM)

| Concept | Apple Silicon | NVIDIA |
|---|---|---|
| Execution unit | **GPU core** (40 on M5 Max) | **SM** (108 on A100) |
| Subdivision | **simdgroup** (32 lanes, like a warp) | **warp** (32 threads) |
| MMA hardware | **NAX** (one per core) [ref][apple-m5] | **tensor core** (multiple per SM) |
| ALU hardware | inline with simdgroup pipeline | **CUDA cores** (separate from tensor cores) |
| Address spaces | `device`, `threadgroup`, `constant`, `thread` [ref][msl-spec] | `__device__`, `__shared__`, `__constant__`, `__local__` |
| Shared/TG memory | 32KB/core (shared) | 164KB/SM (A100) [ref][ampere-guide], 228KB/SM (H100) [ref][hopper-guide] |
| Registers/lane | ~96 (estimated, undocumented) | 256/warp lane (A100 register file 256KB/SM) [ref][ampere-guide] |
| Async copy | **none** | `cp.async` (Ampere) [ref][ampere-async], TMA (Hopper) [ref][hopper-tma] |
| L1 cache | per-core | per-SM |
| L2 cache | shared, large | shared, large |
| Memory model | **unified** with CPU [ref][apple-uma] | discrete VRAM |
| Programming | MSL (C++17 dialect) [ref][msl-spec] | CUDA (C++) |
| Driver API | Metal (objc/cpp) | CUDA Runtime/Driver |

**Single biggest difference**: NVIDIA's separate tensor and CUDA cores allow the
warp scheduler to dispatch ALU and tensor MMA in parallel across different warps.
On Apple Silicon we observe ALU ops serializing with NAX MMA more than the
CUDA-side picture would suggest, though not fully — there's partial overlap.
We don't have public documentation describing the exact issue-slot model; the
behavior we infer from timing is in the NAX deep dive below.

**Unified memory consequence** [ref][apple-uma]: GEMM weight matrices don't need
to be uploaded to GPU VRAM — they're just RAM. But cache pressure is still real.
The L2 cache is the practical "GPU memory" for hot data.

---

## NAX Deep Dive

NAX (Neural Accelerator, sometimes called "Neural Accelerator eXtensions") is
Apple Silicon's hardware MMA unit introduced in M5 [ref][apple-m5], analogous
to NVIDIA's tensor cores. Accessed via the `MetalPerformancePrimitives`
framework's `mpp::tensor_ops::matmul2d` API [ref][mpp-guide].

### Architecture

- **One NAX unit per GPU core** (40 cores on M5 Max) [ref][apple-m5]
- Each NAX does **1024 FP16 FMA operations per cycle** [ref][creative-m5]
- Operates at the **simdgroup** level (32 lanes = 1 simdgroup, analogous to
  NVIDIA warp) [ref][msl-spec]
- Throughput: **~70 TFLOPS FP16 in aggregate across 40 cores** [ref][creative-m5];
  int8 approximately 2× the FP16 rate based on our measurements
- Hardware fragment size: **16×16** is the atomic MMA fragment (see NAXFrag below)
- Cooperative tensor distribution: 32 lanes × 8 elements/lane = 256 = 16×16 ✓
- Developers can program NAX directly using Tensor APIs in Metal 4 [ref][apple-m5]

### NAX MMA primitive: `matmul2d_descriptor(M, N, K)`

Apple's `mpp::tensor_ops::matmul2d` allows running matmul at either simdgroup
scope (`execution_simdgroup`) or threadgroup scope (`execution_simdgroups<N>`)
[ref][mpp-guide]. The tile size and options are configured via the
`matmul2d_descriptor`.

Hardware accepts these descriptor sizes (verified via static_assert in our
probe kernels):

| M | N | K | Execution scope | Per-lane elements (A) | (B) | (C) |
|---|---|---|---|---|---|---|
| 16 | 32 | 16 | `execution_simdgroup` | 8 | 16 | 16 |
| 32 | 16 | 16 | `execution_simdgroup` | 16 | 8 | 16 |
| 32 | 32 | 16 | `execution_simdgroup` | 16 | 16 | 32 |
| 64 | 32 | 16 | `execution_simdgroups<4>` | — | — | — |

`M=64` with single simdgroup: **fails** with `"M must be 16 or 32 if both inputs
are cooperative tensors"`. Larger M requires cooperating across 4 SGs.

`N=64` with `execution_simdgroup`: also fails. N must be ≤ 32 for single-SG.

K is fixed at **16** for our use case (default for int8). K can be larger for
half×half but we didn't explore.

### Type combinations supported by NAX MMA

From `MPPTensorOpsMatMul2d.h`:

| Left (A) | Right (B) | Destination (C) | Throughput | Notes |
|---|---|---|---|---|
| half | half | half | 70 TFLOPS | standard fp16 path |
| half | half | float | 70 TFLOPS | fp16 inputs, fp32 accum |
| half | int8_t | half | 70 TFLOPS | mixed — runs at fp16 rate |
| int8_t | int8_t | **int** | **130 TOPS** | only int32 accum allowed |
| int8_t | int8_t | float | ❌ | `static_assert "Unsupported type"` |
| float | float | float | ~30 TFLOPS | rarely used |
| bfloat | bfloat | float | ~50 TFLOPS | not tested |
| uint4b | uint4b | int | ? | int4 path, not explored |

**Critical constraint**: int8 × int8 MMA requires int32 accumulator —
int8×int8→float fails with `static_assert "Unsupported type"`. The int→float
conversion must happen in a separate ALU step afterward. This is the source
of the ~1% int8 SDPA overhead we measured; we haven't found a way around it.

**Mixed precision is a trap**: `half × int8 → half` compiles cleanly, but we
measured it **~10% slower** than int8×int8 MMA. One plausible explanation:
the hardware falls back to the fp16 rate (70 TOPS) rather than the int8 rate
(~130 TOPS estimated) for mixed inputs. We didn't verify this — it's just a
guess consistent with the timing.

### MLX's NAXTile / NAXFrag abstraction

MLX Steel wraps NAX in `NAXFrag` (one 16×16 fragment) and `NAXTile<T, TR, TC>`
(grid of `TR × TC` fragments). Found in:

```
mlx/include/mlx/backend/metal/kernels/steel/attn/nax.h
mlx/include/mlx/backend/metal/kernels/steel/gemm/nax.h
```

Per-lane element layout for the standard `BaseNAXFrag` (M=16, N=32, K=16):
```
kElemRows = 2          // each lane owns 2 rows of the 16×16 fragment
kElemCols = 4          // each lane owns 4 cols
kElemRowsJump = 8      // rows are at offsets {sc.y, sc.y + 8}
kElemsPerFrag = 8      // 2 * 4 = 8 elements per lane

get_coord() returns (col_base, row_base) where:
  row_base ∈ {0..7}
  col_base ∈ {0, 4, 8, 12}
```

Lane (`sc.x`, `sc.y`) owns elements at:
- A[sc.y + i*8, sc.x + j] for i ∈ {0,1}, j ∈ {0..3} → 8 elements

### Custom NAXFrag for M=32

We reverse-engineered the M=32 layout (Apple doesn't document it). Probe kernel:
set `ct_a[i] = i+1` for lane 0 only, B = all 1s, run MMA, decode output:

```
ct_a[0..3]   → A row 0,  cols {sc.x..sc.x+3}  (sum 1+2+3+4 = 10) ✓
ct_a[4..7]   → A row 8,  cols {sc.x..sc.x+3}  (sum 5+6+7+8 = 26) ✓
ct_a[8..11]  → A row 16                        (sum 9+10+11+12 = 42) ✓
ct_a[12..15] → A row 24                        (sum 13+14+15+16 = 58) ✓
```

So M=32 = same `get_coord()` as M=16, with kElemRows=4, kElemRowsJump=8, just
extending row offsets to {0, 8, 16, 24}.

C output (32×32) splits as: `Cn0` covers M-rows 0-15, `Cn1` covers M-rows 16-31.
Each contains two BaseNAXFrag(M=16) sub-tiles for two N-halves (N-cols 0-15
and 16-31).

Custom abstraction lives in `kernels/nax_m32.h`.

### NAX vs NVIDIA tensor cores: pipelining

The most important architectural difference. On NVIDIA:

```
[tensor core MMA]  ‖  [CUDA core ALU]
       ‖                    ‖
   warp scheduler distributes work across both
```

Tensor cores and CUDA cores are **separate execution units** with separate
register/instruction issue. Different warps can use them in parallel — the warp
scheduler dispatches a tensor core MMA in warp 0 and a CUDA core multiply in
warp 1 in the same cycle.

On Apple Silicon:

```
NAX MMA → [waits in pipeline] → ALU op → [waits in pipeline] → NAX MMA
```

We don't have public documentation on how NAX and ALU are scheduled, but the
picture above is what our measurements are consistent with.

**Observation**: the ALU overhead cost scales **sub-linearly** with ALU op
count. Comparing two SDPA variants on the same shape:
- int8 MMA path: 4096 ALU ops per kernel → +0.7% overhead
- fp16 MMA + int8→fp16 cast-on-load path: 524K ALU ops → +5.2% overhead

If ALU and MMA were fully serialized, 128× more ops should yield ~128× more
overhead. We actually measured only 7.4× more. That's consistent with **some**
overlap happening somewhere in the pipeline, but we can't say exactly what
from timing alone. Best guess: NAX MMA is multi-cycle and some ALU can be
issued during its tail, but we haven't verified this.

**Practical implication from the measurements**: kernels with *small* amounts
of ALU per MMA (like int8 SDPA with ~16 int→float conversions per block) pay
almost no penalty for that ALU. Kernels with *large* amounts of ALU per MMA
(like the fp16-MMA variant with per-element int8→fp16 casts) don't get the
same hiding benefit. There seems to be a saturation point somewhere between
4K and 500K ALU ops per kernel for our shapes — we didn't try to characterize
it more precisely.

### NAX async behavior (commit/resolve pattern)

Apple's MPP API uses a cooperative_tensor pattern that LOOKS async:
```metal
gemm_op.run(ct_a, ct_b, ct_c);    // possibly initiates MMA
// theoretically: do other work here
auto val = ct_c[0];                // implicit wait for MMA result
```

If `run()` is async, this enables software pipelining — issue MMA for block N+1
while doing softmax on block N's results.

But MLX's `NAXFrag::mma()` wrapper does immediate readback inside the function:
```metal
gemm_op.run(ct_a, ct_b, ct_c);
for (i...) Cn0[i] = ct_c[i];      // immediate readback
```

Eliminates any async benefit. To exploit pipelining, you'd need to keep the
cooperative_tensor alive across iterations and split commit from readback.
We didn't pursue this because the residual overhead is already small (~1%).

### NAX register pressure

NAX cooperative tensors live in **register space** (per-lane). Their footprint:

- `NAXTile<int, 1, 2>` (Stile_int for BK=32): 16 int32/lane = 64 bytes/lane
- `NAXTile<float, 1, 2>` (Stile for softmax): 16 float/lane = 64 bytes
- `NAXTile<float, 1, 4>` (Otile for D=64): 32 float/lane = 128 bytes
- Q/K/V fragments (transient per MMA): ~24 bytes total

Peak during int32→float conversion (Stile_int + Stile both alive): ~64 + 64 +
128 = **256 bytes = 64 registers/lane**.

Apple Silicon has ~96 registers/lane (estimated, undocumented). At 64 we're
close to the limit. BK=64 doubles Stile_int and Stile to 32 elements each —
peak goes to 96+. In our measurements BK=64 often regresses sharply at this
point in a way consistent with register spill, though we haven't inspected
generated code to confirm.

**Workaround we used**: explicit C++ scope around Stile_int, hoping the
compiler can reclaim its registers before the softmax block:
```metal
stile_t Stile;  // float, kept alive through softmax
{
    stile_int_t Stile_int;  // dies at end of scope
    // MMA fills Stile_int
    // convert + scale → Stile
}  // Stile_int's registers reclaimable here
// Softmax on Stile (Otile + Stile = 96 regs, fits)
```

This made BK=64 work (sometimes — see compilation context quirk above).

### NAX MMA call accounting

For our SDPA kernel (BQ=32, BK=32, BD=64, M=16 NAXFrag):

Q@K^T per BK-block:
- TQ=1, TK=2, TD=4 → inner loop: 1 × (TK/2)=1 × 4 = 4 paired MMA calls
- Each call: M=16 × N=32 × K=16 = 8K ops
- Total per block: 32K ops in 4 calls

P@V per block:
- TQ=1, TD=2 (paired N-stride), TK=2 → 1 × 2 × 2 = 4 paired MMA calls
- Total per block: 32K ops in 4 calls

**Per BK block: 8 MMA calls, 64K ops**. At 8192-token KV: 256 blocks ×
8 calls = 2048 MMA calls per SDPA kernel invocation.

With M=32 instead, output covers 32×32 per call → halved call count for the
same work. We measured this improves wide-N GEMMs (+3-8%) but not SDPA
(MMA throughput isn't the bottleneck — register pressure is).

### NAX-specific gotchas summary

1. **No fp8** — `metal::float8_*` types don't exist. SageAttention2's FP8 P@V
   path is not implementable.
2. **Int8 MMA needs int32 destination** — can't accumulate into float directly,
   must convert in ALU.
3. **Threadgroup tensors not supported** — `tensor<threadgroup, ...>` fails
   static_assert. Only device/constant address spaces.
4. **Mixed-precision is slow** — `half × int8` runs at fp16 rate, not int8 rate.
5. **Layout opaque** — cooperative tensor element ordering is hardware-defined
   and not in any docs. Reverse engineering required for non-standard descriptors.
6. **NAXFrag's wrapper hides async potential** — readback is immediate.
7. **N=64 doesn't work** with `execution_simdgroup`. Need multi-SG.
8. **Compilation context affects register allocation** — same kernel can spill
   based on which other kernels are in the file.

---

## Memory Loading: device → registers and threadgroup

Memory access patterns differ enough from CUDA that several CUDA "best
practices" are wrong on Apple Silicon.

### Load patterns and vectorization: kernel-by-kernel audit

CUDA's golden rule is warp-level coalescing: adjacent threads in a warp
access adjacent scalar addresses so the hardware can fuse 32 narrow loads
into one wide transaction. We don't rely on scalar-level fusion on Apple
Silicon — when we want a wide transaction we issue one explicitly (`int4`
/ `half4`). Two axes matter for device↔TG/register transfers:

- **Vectorization (`half4` / `int4` / `char4`)** — the biggest lever.
  Explicit wide loads are consistently faster than scalar loops with the
  same access pattern. This is the dominant effect we measured.
- **Thread→chunk mapping** — either "cross-lane" (adjacent lanes hit
  adjacent wide chunks, thread strides by TG size across iterations) or
  "per-thread sequential" (each thread owns a contiguous strip, iterates
  through it one chunk at a time). Secondary effect: 1-6%, and it flips
  based on TG size.

We swept every hot kernel. Results below. All outputs bit-identical or
within fp-rounding tolerance of the original.

| kernel | change | before (ms/call) | after (ms/call) | speedup |
|---|---|---|---|---|
| `fused_quant_upsert` | vector half4 + per-thread-seq | 1.624 | 0.853 | **47% faster** |
| `fused_silu_quant` | vector half4 + cross-lane | 1.824 | 1.561 | 14.5% faster |
| `fused_quant` | vector half4 + cross-lane | — | ~0.85 | kept, baseline not retested |
| `fused_rmsnorm_*_quant` (4 variants) | vector Phase 2/3 only (Phase 1 scalar — see flag below) | 1.128 | 0.973 | 13.8% faster |
| W8A8 GEMM V1 | per-thread-seq loader (already using int4) | — | — | 1-2% microbench, ~5% bench_steady, much tighter variance |
| `fused_qkv_rope` | skipped | — | — | simd_shuffle RoPE tightly coupled to lane layout, D_HEAD=64 too small per-simdgroup to benefit from vectorization |
| `fused_transpose_quant` | skipped | — | — | scattered read (transpose); vectorizing would require rewriting the index mapping |

**End-to-end** (`bench_steady`, 15 frames, speed profile): median denoise
**154.2 ms** (down from ~162 ms baseline), std 1.1 ms, max 155.5 ms — no
tail-latency blowups.

#### Vectorization is the dominant lever (10-45%)

Scalar `half` → vectorized `half4` / `int4` / `char4` accounts for almost
all of the 47% win on `fused_quant_upsert` and the 14.5% on
`fused_silu_quant`. Same bytes moved, ~4× fewer load instructions, less
LSU pressure. Issue wide loads explicitly via
`*reinterpret_cast<device half4*>` — the MSL spec doesn't promise
lane-coalescing (CUDA's HW does, [ref][cuda-coalesce]; Apple's might but
isn't documented), and even on CUDA where coalescing is guaranteed,
explicit `LDG.128` still beats 4 coalesced scalar loads on instruction
count alone. Same guidance on both platforms.

#### Thread mapping is secondary (1-6%) and flips with TG size

Within a vectorized loader the thread→chunk mapping is a small knob:

- **Per-thread sequential** — thread `tid` owns a contiguous strip;
  per-iter the simdgroup's collective footprint scatters across the tile.
- **Cross-lane** — at iter `i` all threads hit `tid + i*TG_SIZE`;
  per-iter the simdgroup hits one contiguous chunk; per-thread, each
  thread strides far between iters.

Measured on M5 Max:

| kernel | TG size | per-TG data | winner | margin |
|---|---|---|---|---|
| `fused_quant_upsert` | 64 threads | 4 KB | per-thread seq | ~6% |
| W8A8 GEMM V1 | 128 threads | 4 KB tile | per-thread seq | 1-2% |
| `fused_silu_quant` | 256 threads | 16 KB row | cross-lane | ~3% |
| `fused_rmsnorm_*_quant` | 256 threads | 4 KB row | cross-lane | small |

Rule: ≤128 threads → per-thread-seq, ≥256 threads → cross-lane. Guess at
why: 256 per-thread-seq streams scatter the instantaneous footprint
across the whole tile, overloading prefetch streams; cross-lane keeps
it compact (one ~512-byte window advancing through the row). At 64
threads only 64 in-flight streams — manageable either way.

#### Practical takeaway

1. Vectorize first (`half4` / `int4` / `char4`) — this is the 10-45% knob.
2. Pick thread mapping by TG size: per-thread-seq ≤128, cross-lane ≥256.
3. Don't measure with per-call `mx.eval` — the ~50µs fence buries 1-2%
   kernel wins. Batch many ops per eval, or just use bench_steady.

**⚠ Flag — narrowed reproducer for a half4-TG-store race on M5 Max.**

Vectorizing the `fused_rmsnorm_*_quant` Phase 1 writes (half4 device
load + 4 contiguous scalar TG stores, which the Metal compiler fuses to
a half4 TG store) produces reproducible corruption under load: ~0.6% of
elements wrong on specific rows, clustered in a single 40-wide dispatch
wave (= the 40 GPU cores on M5 Max). Absmax reads back 2-3× too large,
dequantized output off by 20%+.

**Minimal reproducer narrowed to a 4-kernel A/B/C/D probe sweep** — see
`ext/kernels/repro_half4_tg.metal` + `tests/repro_half4_tg_race.py`
(`--mode probe`). Each kernel dispatches through our `.metallib` +
custom `mx::Primitive` path. Results on M5 Max, 30 runs each:

| kernel | structure | corrupt runs |
|---|---|---|
| A `repro_half4_tg`        | Phase 1 half4 TG writes → barrier → Phase 2 scalar TG→device copy | 0 / 30 |
| B `repro_half4_tg_reduce` | A + sum_sq reduction (simd_sum + sg_reduce TG writes)              | 0 / 30 |
| C `repro_half4_tg_rmw`    | B + Phase 2 TG read-modify-write (x_cache *= rms_inv)              | 0 / 30 |
| D `repro_half4_tg_adaln`  | C + Phase 2 per-column device reads (`adaln_s[k]`, `adaln_b[k]`)   | **14-18 / 30** |

**The trigger is the combination in D** — Phase 2 that does BOTH
(a) read-modify-write to the TG cache AND (b) per-element device reads
from auxiliary buffers, interleaved in the same loop. Taking away either
ingredient makes the race disappear.

**Per MSL spec v4 this should not happen.**
`threadgroup_barrier(mem_flags::mem_threadgroup)` is supposed to make
all pre-barrier TG writes visible to all post-barrier reads across the
threadgroup [ref][msl-spec]. Variants A/B/C confirm the barrier works
when Phase 2's memory pattern is simpler.

**What we ruled out** (via the same probe harness):

- Explicit `half4` TG stores via `reinterpret_cast<threadgroup half4*>`
- 16-byte-aligned backing (`threadgroup uint4 storage[MAX_K/8]`)
- `dot(v,v)` vs `v.x*v.x + ...` for sum_sq
- Extra `threadgroup_barrier(mem_threadgroup)` after Phase 1 writes
- Zero-init of the TG cache before Phase 1 (masks it in isolation but
  not under pytest-suite load)
- Switching `dispatch_threadgroups` → `dispatch_threads`
- The JIT path: a structurally-identical kernel compiled via
  `mx.fast.metal_kernel` and dispatched through its generic
  `CustomKernel` primitive does NOT reproduce — so the bug isn't in
  the MSL source, it's somewhere in the compiled-metallib path.

**Working hypothesis — Dynamic Caching on Apple Family 9+.** Apple's
tech talks ["Explore GPU advancements in M3 and A17 Pro"][m3-dynamic-caching]
and ["Learn performance best practices for Metal shaders"][metal-perf-111373]
describe a major microarchitectural change in the Family 9 shader core:
**Dynamic Caching**. Quoting the performance talk: *"threadgroup device
constant memory types are using the same cache hierarchy"*, and from
the M3 talk: *"register, threadgroup, tile, stack, and buffer data are
all cached on chip"* in one unified on-chip memory. Before Family 9,
threadgroup memory was a separate on-chip SRAM; on M3/M4/M5 it shares a
cache with device memory.

That's a clean explanation for why the race exists only on variant D:

- D is the only probe that mixes Phase 1 TG writes with Phase 2 per-element
  **device** reads (`adaln_s[k]`, `adaln_b[k]`). Under Dynamic Caching
  both hit the same cache lines.
- MSL's `mem_threadgroup` barrier only orders TG memory; it's not
  specified to affect speculative loads that go through the "device"
  path. On pre-Family-9 hardware (separate SRAM) that distinction was
  moot. On unified cache, if a Phase-2 device load is forwarded from a
  cache line that Phase 1's TG write hasn't yet fully committed, the
  TG-scoped barrier wouldn't prevent it.

Consistent with the data: only variant D (unified-cache mixing) trips
it; timing-dependent (depends on which cache lines overlap); disappears
on A/B/C where Phase 2 has no device reads.

This is still a spec violation in spirit — MSL v4 says `mem_threadgroup`
barriers synchronize TG memory visibility across the threadgroup, which
ought to mean flushing whatever cache backs TG memory regardless of
whether a subsequent read is TG-coded or device-coded. But it's a
new-to-Family-9 gap the spec language didn't anticipate.

Why the JIT path (`mx.fast.metal_kernel` via MLX's `CustomKernel`) doesn't
trip it is still unclear — possibly different AIR scheduling / prefetch
hinting chosen by the JIT vs the offline metallib compiler. That's the
remaining unknown, and also why the repro sits in-tree rather than being
closed.

**Related Metal barrier bugs in the wild.** Our pattern is distinct but
sits in the same family as [wgpu#3181][wgpu-3181] ("Apparently
miscompiled barrier in compute shader" — barrier *elided* after
atomics + TG write pattern; workaround was a storage buffer +
`storageBarrier`), [wgpu#4500][wgpu-4500] (dynamic TG memory as
entry-point parameter causes miscompiled barriers), and
[MLX#2205][mlx-2205] (same kernel produces different results on M1 Max
vs M3 Max — hardware-generation-dependent correctness is a known shape).

[m3-dynamic-caching]: https://developer.apple.com/videos/play/tech-talks/111375/
[metal-perf-111373]: https://developer.apple.com/videos/play/tech-talks/111373/
[wgpu-3181]: https://github.com/gfx-rs/wgpu/issues/3181
[mlx-2205]: https://github.com/ml-explore/mlx/issues/2205

**Workaround in production:** RMSNorm Phase 1 uses the scalar path
(`half` load + scalar TG store), which is known safe. Phase 2/3 keep
the vectorized device I/O (still safe — no cross-thread visibility
dependency on half4 TG writes). Net: partial vectorization still
delivers 13.8% vs the all-scalar baseline.

**Reproducing:**

```bash
cd src/mlx_metal/ext && python setup.py build_ext --inplace
uv run python src/mlx_metal/tests/repro_half4_tg_race.py --mode probe
# expected: A=0, B=0, C=0, D≈15/30 corrupt
```

Next debugging step for Apple: reduce variant D to a pure `MTLDevice` +
`MTLCommandBuffer` test case outside MLX entirely, using `xcrun metal`
to build a standalone `.metallib`. The probe kernel is already ~150 lines
of MSL that triggers the bug through `mlx_build_metallib`, so porting it
to an `xcrun metal` build is the obvious next step.

[wgpu-4500]: https://github.com/gfx-rs/wgpu/issues/4500

### 128-bit aligned vector loads are critical

For high-throughput device → threadgroup copies, use `int4` (4×int32 = 16 bytes)
or `half4` (4×half = 8 bytes) loads. Each lane issues one wide load instead of
multiple narrow ones.

Pattern from `w8a8_gemm.metal` (cooperative load):
```metal
// 128-bit (int4) coalesced load: 16 int8 values per transaction
*reinterpret_cast<threadgroup int4*>(dst) =
    *reinterpret_cast<device const int4*>(x_q + gm * K + gk);
```

This issues **one-quarter the load instructions** of the equivalent scalar
loop. In our testing, explicit `int4` has been consistently faster than
scalar loops for cross-address-space copies (device → threadgroup), even
where we'd have expected the compiler to auto-vectorize. We don't have a
verified reason for this — it might be that the compiler doesn't always
vectorize across address spaces, or it might be something else. We just
default to explicit vector loads for these copies because they work.

### TG memory padding to avoid bank conflicts

Threadgroup memory has banks (we estimate 32 banks of 4 bytes each, like NVIDIA
shared mem). Stride-K accesses where K is a power-of-2 cause bank conflicts.
Apple's optimization guidance recommends aligning threadgroup memory to
**16 bytes** and reordering access patterns to mitigate bank issues
[ref][metal-perf-wwdc20].

Standard fix used everywhere in the GEMM kernels: pad rows by 16 bytes:
```metal
constexpr short _A_PAD = 16;
constexpr short _LDA_TG = _BK + _A_PAD;  // never a power of 2
threadgroup int8_t As[_BM * _LDA_TG];
```

### Threadgroup staging often hurts

CUDA kernels stage K/V into shared memory aggressively [ref][ampere-async],
relying on `cp.async` for overlapped loads. We tested this for SDPA:
cooperative TG-stage K/V into shared, then load fragments from TG into NAX
cooperative tensors.

**Result: 3.6× slower** than direct device → cooperative tensor loads. We
don't have a verified decomposition of where the slowdown comes from; some
plausible contributors:
1. Apple Silicon's L2 cache may already be serving multi-SG reads cheaply, so
   the TG stage adds a copy without reducing bandwidth pressure
2. TG barriers add fixed cost, and the TG-staged version requires one before
   reading staged data
3. Direct device → cooperative tensor loads seem to be fast on NAX in our
   measurements

On the CUDA side, TG staging is well-documented as benefitting from `cp.async`
overlap [ref][ampere-async] and tensor-core shared-mem load paths. Apple
has no `cp.async` equivalent, so any "overlap the load with compute" benefit
from TG staging isn't available.

So **direct device loads are the default for Apple Silicon SDPA-style kernels**.
TG-staging is reserved for repeated reuse (Q in SDPA, A or B in GEMM) where the
data is read many times.

### Q-staging in SDPA: TG memory pays off when reuse is high

For SDPA, Q is loaded once and reused across all KV blocks (256 blocks for our
8192-token cache). Staging Q in TG memory pays off:
```metal
threadgroup half Q_smem[32 * 64];  // 4KB per simdgroup
// One-time load, BUT also pre-multiply by scale2 here (saves per-block scale)
for (uint i = tid; i < total; i += stride) {
    Q_smem[i] = (half)(Q_dev[i] * scale2);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
// Now MMA reuses Q_smem 256 times across the KV loop
```

Reuse factor of 256 amortizes the staging cost. K and V have reuse factor of 1
(read once per kernel) so direct device load wins for them.

### The `int4` packing trick for int8 storage

For our int8 KV cache, we store data as `char` (int8) but load as `char4` from
device for vectorized transfers:
```metal
const device char4* src_q = reinterpret_cast<const device char4*>(...);
// Each load reads 4 int8 values in one transaction
```

NAXTile's int8 load uses `static_cast<half>(src[i])` per element internally.
We haven't inspected the emitted code, but this path is fast in practice for
sequential access patterns — fast enough that we haven't needed to replace it.

---

## NAXTile / Static-Size Templates: Why They're Fast

MLX's `NAXTile<T, TR, TC>` and `NAXFrag` aren't just abstractions — in our
experience they're **performance-critical**, and our working theory is that
encoding dimensions into the type system gives the compiler more to work
with than runtime values of the same dimensions. The rest of this section
walks through the patterns; the causal story is a hypothesis we haven't
directly verified against generated code.

### `Int<1>` compile-time stride

NAXFrag's load function:
```metal
template <typename T, typename SrcPtrType, typename StrX, typename StrY>
METAL_FUNC static constexpr void load(
    thread dtype_frag_t<T>& dst,
    SrcPtrType src,
    StrX str_x, StrY str_y, ...)
{
    // ...
    if constexpr (metal::is_same_v<StrY, Int<1>>) {
        // Optimized path: stride 1 → contiguous loads → vectorize
        for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = src[r * str_x + c + j];
        }
    } else {
        // Generic path: arbitrary stride → scalar loads
        for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = src[r * str_x + (c + j) * str_y];
        }
    }
}
```

When you call `load<half, BD, 1>(...)` with `Int<1>` for the column stride,
what we expect (based on how `if constexpr` and template specialization are
defined in C++) is:
1. The `if constexpr (metal::is_same_v<StrY, Int<1>>)` branch is selected at
   compile time, so only the stride-1 body is emitted
2. `str_y` never appears in the generated path, so no `* str_y` multiply is
   present to eliminate

What the compiler does *beyond* that — vectorization decisions, address
folding — is up to it. We haven't inspected generated code. Empirically,
passing `Int<1>` has been reliably faster than passing a runtime `int = 1`,
and our hypothesis is that the template path gives the compiler a
guaranteed-known-stride body to work with, rather than needing to propagate
a literal argument value across inlining boundaries.

We measured this difference indirectly: switching `Q_tg` loads from runtime
stride to `Int<BD>, Int<1>` template params gave noticeable improvements in
the SDPA inner loop (now baked into our kernels).

#### Why `Int<>` and not just `constexpr int`?

This is an observation without a verified root cause. What we see:

- **On CUDA (anecdotally)**: writing `int stride = 1` as a function argument
  or `constexpr int stride = 1` in a calling scope is often enough. nvcc
  tends to specialize across inlining, so `if (stride == 1)` branches often
  get eliminated in practice.
- **On Metal (ours)**: doing the same thing doesn't reliably trigger branch
  elimination or address-arithmetic simplification. We've hit cases where
  switching from `int str_y` to `Int<1>` (passing the value as a type) gave
  measurable speedups in the SDPA inner loop.

We don't have evidence for a specific reason — the Metal compiler is
proprietary and we haven't inspected generated code. One step of speculation:
Apple's compiler may be more conservative about propagating literal argument
values across inlining boundaries than nvcc is. That's a guess, not a claim
about its internals.

What we can say with confidence:

- `Int<N>` encodes the value **into the type**, so `metal::is_same_v<StrY,
  Int<1>>` is a compile-time check that works via template specialization,
  independently of whatever the compiler does with runtime values.
- The same pattern exists in standard C++ as `std::integral_constant<int, N>`.
  Both CUDA and Metal support it.

**Practical rule we follow**: when porting CUDA kernels with `constexpr`
values that flow across function boundaries, wrap them in
`std::integral_constant` / `Int<>` before porting to Metal. We've had better
results with explicit type-level encoding than assuming the compiler will
propagate a value.

### `const_for_loop<>` with constexpr indices

NAXTile loops over its fragments using:
```metal
const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
    const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),  // constexpr access
            src, ...,
            idx_row * Int<kFragRows>{},  // compile-time offset
            idx_col * Int<kFragCols>{});
    });
});
```

`idx_row.value` is `constexpr`, so `frag_at<R, C>()` is a different template
instantiation per iteration — that part is language-defined. Beyond that,
the lambda-based `const_for_loop` pattern has held up well for us where
plain runtime-index loops have regressed, and our working theory is that
compile-time indices give the compiler a better shot at register allocation
and scheduling. We haven't verified that against emitted code. A runtime-
indexed loop would at minimum prevent the template-level fragment selection;
whether it also serializes loads or forces stack allocation is something
we've observed effects of but not confirmed mechanically.

### `STEEL_PRAGMA_UNROLL`: forcing the unroll

When templates can't enforce constexpr context, we use `STEEL_PRAGMA_UNROLL`
(`#pragma clang loop unroll(full)`) on inner loops:
```metal
STEEL_PRAGMA_UNROLL
for (short i = 0; i < FragF::kElemRows * FragF::kElemCols; i++) {
    fd[i] = (float)fs[i] * s;
}
```

In our observed behavior, explicit unroll makes a noticeable difference for
tight inner loops over small fragments. A guess at why: an unrolled loop
accesses distinct indices `fd[0]`, `fd[1]`, ..., which can be resolved to
specific registers at compile time, while a non-unrolled loop over runtime
`i` requires either indexed register access (limited support) or stack spill.
We haven't verified this by inspecting generated SASS/AIR — it's the
simplest explanation for the speedup we see.

### Why this matters more than on CUDA (in our experience)

Based on our porting work: CUDA kernels that use `constexpr` / literal values
flowing across function boundaries, with no explicit unroll pragma, often
run fine. Porting the same code to Metal without `STEEL_PRAGMA_UNROLL` and
`Int<>` template params regularly produces slower code than we'd expect.

We don't have a mechanical explanation for this — the Metal compiler is
closed-source and we haven't done deep analysis. Our working model is that
it's more conservative about cross-function specialization, but that's a
hypothesis based on porting experience, not a claim about the toolchain.

**Practical takeaway**: use explicit `STEEL_PRAGMA_UNROLL` and `Int<>` on
Metal even where the equivalent CUDA code worked without them. The pragmas
cost nothing if the compiler was going to do it anyway; they help when it
wouldn't.

### Cooperative tensor lifetime tricks

In our observations, the compiler respects lexical scope for register
liveness. Putting a temporary tile in a `{ ... }` block appears to help the
compiler reclaim its registers earlier:
```metal
stile_t Stile;  // float, kept alive through softmax
{
    stile_int_t Stile_int;  // int32, dies at end of scope
    // MMA fills Stile_int
    // convert + scale → Stile
}  // Stile_int's registers reclaimable here
// Softmax on Stile (Otile + Stile = 96 regs, fits)
// Without the scope: Stile_int + Stile + Otile = 128 regs, spills
```

This was the difference between BK=64 spilling (+56% slower) and not spilling
(+0.9%).

---

## GEMM Lessons (W8A8 int8 GEMM specifically)

The W8A8 GEMM kernel (`w8a8_gemm.metal`) is the most performance-critical kernel
in the world model. Here's what we learned tuning it.

### Tile selection: smallest wins, not largest

Counterintuitive finding from `select_tile()`: **`bm64_bn64_bk64_wm2_wn2` wins
for ALL tested shapes**, not the larger `bm128_bn128_bk64_wm4_wn4`.

Plausible contributors (we haven't isolated which dominates):
1. **Occupancy**: 40 GPU cores can host more small TGs (4 simdgroups each)
   than fewer large TGs (16 simdgroups each)
2. **Load balancing**: many small TGs may hide latency from one another better
3. **Bandwidth**: large TGs compete more for L2 cache
4. **Register pressure**: BM=128 forces TM=2 per simdgroup → more accumulator
   registers per lane

NVIDIA CUTLASS generally picks larger tiles. For our shapes on Apple Silicon,
more-smaller-TGs has consistently won.

### V1 (both-staged) vs V2 (A-direct) variants

We have two GEMM variants:
- **V1**: stage both A and B through threadgroup memory
- **V2**: load A directly from device, stage only B in TG

V2 is theoretically better when A has low reuse (M dimension is small). But in
practice **V1 wins end-to-end**, even though V2 wins in isolated benchmarks.

This is the classic isolated-vs-pipeline divergence: V2's reduced TG memory
pressure helps when nothing else is competing, but in the lazy-eval pipeline
where many GEMMs run back-to-back, V1's predictable memory pattern interleaves
better.

### Matvec path for small M

For `M < 5` (decode path: 1-4 tokens), we dispatch a separate `w8a8_matvec`
kernel instead of the tiled GEMM. The matvec kernel:
- Has 4 simdgroups × 32 threads = 128 threads per TG
- Each TG produces 32 output values (BN=32)
- Single K-loop, no tiling

The crossover at M=5 was benchmarked. Above M=5, the tiled GEMM amortizes its
overhead better; below M=5, the matvec kernel wins.

### Cooperative coalesced loads

Pattern repeated throughout the GEMMs:
```metal
const uint flat = sgid * 32 + lane;
constexpr uint _TG_SIZE = _WM * _WN * 32;

// Cooperative load A [BM, BK] -> threadgroup, 16 elements per thread
for (uint t = flat; t < uint(_BM) * uint(_BK) / 16u; t += _TG_SIZE) {
    short r = t / (_BK / 16);
    short c = (t % (_BK / 16)) * 16;
    threadgroup int8_t* dst = As + r * _LDA_TG + c;
    *reinterpret_cast<threadgroup int4*>(dst) =
        *reinterpret_cast<device const int4*>(x_q + gm * K + gk);
}
```

Each thread loads 16 int8 values per transaction (`int4` = 16 bytes). The
threadgroup as a whole loads `_TG_SIZE * 16` bytes per iteration, distributed
sequentially across the M×K tile.

### Epilogue: per-row scaling is ~free

After int8 × int8 → int32 MMA, we have to dequant. The epilogue:
```metal
float val = float(Dtile[i,j]) * x_scales[gm] * w_scales[gn] + bias[gn];
out[gm * N + gn] = half(val);
```

This is a tiny amount of work compared to the MMA (4 float ops per output
element vs 2K MACs to produce it), and in our profiling the epilogue has
never shown up as a bottleneck.

### `int4` accumulator type for int8 inputs

NAX's int8 MMA writes to `NAXTile<int, TM, TN>` (int32 accumulator). We can't
accumulate into float directly (`static_assert "Unsupported type"`). The
int → float conversion happens in the epilogue, which is fine because it runs
once per MMA group (not per element load).

### M=32 for wide N: shape-dependent dispatch

Custom M=32 MMA descriptor (vs default M=16) halves MMA call count. We
auto-dispatch:
```cpp
static const TileConfig& select_tile(uint32_t M, uint32_t N, uint32_t K) {
    static int gemm_mode = std::getenv("WE_GEMM_M32") ? std::atoi(...) : 3;
    if (gemm_mode == 1) return M32_SMALL;
    // Mode 3: M=32 only when wins (wide N)
    if (gemm_mode == 3 && N >= 6000) return M32_SMALL;
    return V1_SMALL;
}
```

Wins at N=6144 (QKV proj) and N=8192 (gate_up). Loses at N=2048 (out_proj,
mlp_down) where the M=32 register pressure hurts more than fewer MMA calls
help.

---

## MLX Dispatch and Lazy Evaluation

MLX uses lazy evaluation — graph nodes are accumulated until `mx.eval()` (or an
implicit eval like `numpy()` or `print()`) forces execution [ref][mlx-lazy].
MLX also supports `mx.compile` to fuse separate GPU kernels into single fused
kernels [ref][mlx-lazy]. This has major implications for kernel design.

### Each kernel launch has fixed overhead

A kernel launch in MLX involves:
1. Building the GPU command buffer entry
2. Setting argument buffers (`set_input_array`, `set_output_array`, `set_bytes`)
3. Setting the compute pipeline state
4. Dispatching threadgroups

Empirically: ~50us per launch amortized in the lazy executor. (Per-call sync
with `mx.eval()` is much slower — many milliseconds — because it forces a
GPU-CPU roundtrip per call.)

For our world model: 24 transformer layers × 4 denoise steps × 2 quant calls =
192 launches per frame for KV quantization alone. At 50us each = 9.6ms just
in launch overhead.

**Optimization that worked**: fuse K and V quantization into a single kernel
launch (`fused_quant_upsert` does quant + cache write in one dispatch). Saved
~5ms/frame.

### Lazy evaluation re-batches kernels

When you queue 100 operations before `mx.eval()`, MLX:
1. Builds the dependency graph
2. Schedules kernels to maximize GPU occupancy
3. Co-schedules kernels that don't conflict on memory
4. Issues them in batched command buffers

This means **isolated benchmarks that call `mx.eval()` after each operation
produce different timings than the lazy pipeline**. Two examples we hit:

**V2_DEEP_K GEMM**: appeared 20% faster in isolation, was 10% slower end-to-end.
Reason: V2's larger BK occupies threadgroup memory for longer, blocking other
kernels in the lazy pipeline from co-scheduling.

**M=32 GEMM**: appeared 2.58× faster on QKV in isolation. Real win is ~3-8%
end-to-end. The "2.58×" was a warmup artifact — the first shape benchmarked
takes the cold-cache hit, looking artificially slow.

**Rule**: always validate with `bench_steady` (runs many denoise iterations
back-to-back). Never trust `mx.eval()` per-call timings.

### Concat / reshape ops aren't free

MLX's `mx.reshape()` is metadata-only (no data movement). But `mx.concatenate`
schedules a copy kernel even if the data is contiguous. We tried batching K+V
quantization via concat:
```python
kv_flat = mx.concatenate([K_flat, V_flat], axis=0)  # adds a copy kernel
quant_result = fused_quant(kv_flat)  # 1 kernel instead of 2
```

This was faster than 2 separate calls but the concat overhead partially
cancelled the savings. The clean solution was to write a custom Metal kernel
(`fused_quant_upsert`) that takes K and V as separate inputs, eliminating
both the concat AND the separate quant call AND the separate cache write.

### Static variables in C++ dispatch are a footgun

We initially tried env-var dispatch like:
```cpp
static int gemm_mode = std::getenv("WE_GEMM_M32") ?
    std::atoi(std::getenv("WE_GEMM_M32")) : 0;
```

The `static` initialization happens **once per process**. If you change the env
var across `importlib.reload()` calls in the same Python process, the C++
static doesn't update. We hit this when benchmarking M=16 vs M=32 in one
process and got identical numbers (both running the same kernel from initial
import).

**Workaround (weak)**: separate Python processes for A/B benchmarks. This
means every variant flip costs a subprocess spawn + full MLX import +
first-call JIT, which on our machine is several seconds per run.

**Fix (strong)**: don't use env vars for A/B variant selection at all.
Expose each variant as its own `mx::Primitive` subclass with its own
`eval_gpu` that hard-codes the kernel name, and its own Python entry
point. Two reasons this is better:

- Zero hidden global state — the call site selects the variant, the
  primitive type carries the choice through MLX's graph.
- Can't accidentally co-dispatch default and variant with "which one
  gets used" depending on which Python process set the env var first.

We used this pattern during the load-pattern investigation
(`W8A8GemmSeq` alongside `W8A8Gemm`, `KVCacheUpsertXlane` alongside
`KVCacheUpsert`), with distinct primitive classes so MLX's graph
deduplicator couldn't collapse them. Once the A/B concluded (per-thread
sequential won on both) the loser was deleted and the winner promoted
to the default kernel body.

The one env var we still have is `WE_GEMM_M32`, because it's a runtime
tile-selection heuristic, not a variant for A/B benchmarking. A future
refactor could replace it with a `w8a8_gemm_m32()` entry point; the
footgun still applies if anyone adds a new `static getenv()` inside a
primitive.

### Command encoder reuse

MLX reuses the Metal command encoder across multiple kernel dispatches in the
same lazy-eval batch. From our `eval_gpu` implementations:
```cpp
auto& enc = mx::metal::get_command_encoder(s);  // get current encoder
enc.set_compute_pipeline_state(kernel);
enc.set_input_array(...);
enc.dispatch_threadgroups(grid, group);
```

The encoder is shared — there's no "create encoder, dispatch, end encoder" per
kernel like in raw Metal. This reduces overhead but means you can't easily
capture per-kernel command buffers for profiling.

### `set_input_array` vs `set_output_array`

```cpp
enc.set_input_array(arr, slot);   // for read-only buffers
enc.set_output_array(arr, slot);  // for write/read-write buffers
```

The distinction matters for MLX's dependency tracking. `set_input_array` adds
the array as a read dependency; `set_output_array` adds it as a write
dependency. Get this wrong and you get race conditions in the lazy schedule
(or worse, the optimizer may eliminate "unused" output writes).

### In-place mutation via `copy_shared_buffer`

For kernels that mutate inputs (like KV cache upserts), we use:
```cpp
outputs[0].copy_shared_buffer(cache_k_q);  // donate input buffer to output
enc.set_output_array(outputs[0], 0);
```

This tells MLX the output IS the input, sharing the same physical buffer. Saves
allocation and lets the kernel do in-place writes.

---

## Compiler Quirks: Same Kernel, Different Perf Across Edits

**Observation**: the same kernel can compile to noticeably different
performance — including what looks like register spill behavior — depending
on what other kernels are in the same `.metal` file.

Concrete case with the BK=64 SDPA variant:
- During active development: `seq_sdpa_int8block` with BK=64 → +0.9% overhead
- After removing 5 unused experimental kernels from the same file: same
  kernel source → +56% overhead, which looks like a register spill
- Workaround: moved BK=64 to its own file `scatter_sdpa_bk64.metal`. Perf
  restored.

We don't know exactly what the Metal compiler is doing — it's closed-source.
Our guess is that whole-file compilation is doing some allocation decision
that depends on the other kernels present (liveness analysis budget?
register pressure heuristics across kernels?). We haven't verified this.

**Pragmatic takeaway**: don't assume a kernel's perf is stable under
unrelated edits to its `.metal` file. If a kernel is performance-critical
and close to a resource limit, putting it in its own compilation unit is
cheap insurance.

### Other compiler observations (without causal explanations):

These are patterns we've seen that worked for us. We haven't validated the
underlying compiler behavior.

- **Explicit `METAL_FUNC` on hot helpers** (`__attribute__((__always_inline__))`):
  we've had cases where removing it caused obvious perf regressions, so we
  leave it on anywhere it was present in the source we copied from MLX.
- **Explicit `STEEL_PRAGMA_UNROLL` on tight inner loops**: removing these
  has regressed our kernels more than once. Keep them.
- **Template-time constants** (`Int<>`, `constexpr short kElemRows = 2`):
  these propagate through template expansions reliably in our testing.
- **Lambda-based loops** (`const_for_loop<>`): inline without overhead in
  every case we've checked, probably because the lambdas are constexpr and
  the iteration count is known.

---

## SDPA-Specific Findings

A few results that are specific to the int8 SDPA work but don't fit the
general categories above. These complement the project memory file
`project_int8_kv_sdpa.md`.

### Bandwidth vs compute trade: int8 V wins on Apple Silicon

SageAttention [ref][sageattention] keeps V as fp16 (only K is int8). We
tested both approaches on our kernel:

| Config | Apple Silicon SDPA |
|---|---|
| int8 K + fp16 V (SageAttention) | +3.4% slower |
| **int8 K + int8 V (ours)** | **+1.1% slower** |

Plausible reason: V is loaded once per P@V MMA. Reading 1 byte (int8) vs
2 bytes (fp16) halves the V memory traffic. On a platform where memory
bandwidth is the tighter constraint (see BW numbers vs NVIDIA above), that
bandwidth savings appears to outweigh the int8→fp16 cast overhead. We
didn't confirm which factor dominates — we just measured the end-to-end
result and kept the faster option. **Opposite of what SageAttention
recommends for NVIDIA**, which makes sense if their platform's constraint
profile is different.

### Quantization granularity: per-block wins

| Granularity | SDPA overhead | Quality (MAE) |
|---|---|---|
| Per-token (one scale per K row) | +10% | best |
| Per-block (32 tokens share a scale) | **+1.1%** | -0.2% from per-token |
| Per-cache (one global scale) | +5% (cast cost) | poor (clipping) |

Per-token requires column-wise scaling on Stile after MMA — every column needs
a different scale. NAXTile's `row_bin_op<MulOp>` doesn't fit this pattern, so
we hand-roll it, and it costs ~10% overhead.

Per-block (SageAttention's recommendation) reduces this to ONE scalar multiply
per BK-block on the entire Stile. Essentially free.

### Online softmax fusion: counterproductive (for us)

SageAttention [ref][sageattention] fuses `convert + scale + max + exp + sum`
into one pass. We tried the same on Apple Silicon:
- 5 separate NAXTile passes → +1.5% overhead
- Manually fused 2 passes → +2.1% overhead (slower!)

We don't know why the fused version is slower. One guess: the fused loop body
has more simultaneously-live variables, which may be pressuring register
allocation. But we haven't inspected the generated code to confirm. What we
do know is that the separate-pass version consistently wins in our
measurements.

CUDA compilers and hardware may handle fused bodies better (more registers,
different pipelining), which would explain why SageAttention's fusion is a
win there. Either way, on Metal the decomposed form has worked better for us.

### Smooth-Q doesn't help diffusion models

SageAttention [ref][sageattention] subtracts per-token Q mean before
quantization to remove outliers (a.k.a. SmoothQuant-style [ref][smoothquant]).
Measured improvement on our model: **0.8%** quantization MAE.

Reason: post-RMSNorm Q is already centered (mean ≈ 0). Smooth-Q matters for
LLMs where attention has per-head/per-token biases. Diffusion model Q doesn't
have this issue.

### Final overhead breakdown

The 1.1% residual overhead vs fp16 SDPA:
- ~0.7% from int32→float conversion in NAX MMA epilogue (irreducible without
  hardware changes)
- ~0.4% from int8→fp16 cast on V loads during P@V (could be eliminated with
  int8 P@V MMA, but quality hit is too large per SageAttention guidance)

Both are intrinsic to the int8 design choice. Net: 50% KV memory savings at
near-parity speed.

---

## Production results (M5 Max, world model, 8192-token KV cache)

Final config: `WE_INT8_KV=1` + auto M=32 GEMM dispatch.

| Metric | fp16 baseline | int8 production |
|---|---|---|
| Denoise (per frame) | ~160ms | ~161ms |
| KV cache memory | 1.6GB | 0.8GB (-50%) |
| Variance (std) | 2.5ms | 0.5ms (3-5× lower) |
| Quality (latent MAE) | — | 0.91% |

The 50% memory savings is the substantive win. Speed parity (±1%) means int8
is "free" — no quality/speed tradeoff for the memory benefit.

---

## Key files in this codebase

- `kernels/scatter_sdpa.metal` — fp16 + int8 block SDPA kernels
- `kernels/scatter_sdpa_bk64.metal` — isolated BK=64 (compile context fix)
- `kernels/kv_cache_upsert.metal` — fused quant+upsert (eliminates separate launches)
- `kernels/nax_m32.h` — custom M=32 NAXFrag for wide-N GEMMs
- `kernels/we_ops.cpp` — primitive dispatchers, including `select_tile()` auto-dispatch

## References

### Apple Silicon / Metal

- [Apple M5 announcement — "the next big leap in AI performance"][apple-m5] —
  introduces Neural Accelerators in each GPU core, Tensor APIs in Metal 4
- [Apple M5 Pro and M5 Max announcement][apple-m5-pro] — detailed specs including
  memory bandwidth
- [Apple Unified Memory Architecture (WWDC20)][apple-uma] — how the SoC shares
  memory between CPU and GPU with no copy overhead
- [Creative Strategies: M5 Apple Silicon — It's All About the Cache And Tensors][creative-m5]
  — Neural Accelerator details (1024 FP16 FMA/cycle, ~70 TFLOPS FP16 aggregate
  on M5 Max)
- [Metal Shading Language Specification v4][msl-spec] — address spaces,
  simdgroup primitives, compiler attributes
- [Metal Performance Primitives (MPP) Programming Guide][mpp-guide] —
  `tensor_ops::matmul2d`, cooperative tensors, execution scopes
- [WWDC25: Combine Metal 4 machine learning and graphics][wwdc25-ml] —
  tensor APIs, neural graphics integration
- [WWDC20: Optimize Metal Performance for Apple Silicon Macs][metal-perf-wwdc20]
  — threadgroup memory banking, 16-byte alignment
- [Metal 4 matmul tensor op example (liuliu)][liuliu-matmul4] — community
  reference example
- [PR: MLX experiment using Metal Performance Primitives][mlx-mpp-pr] — active
  work integrating MPP into MLX

### MLX

- [MLX lazy evaluation docs][mlx-lazy] — how the graph executor works,
  `mx.compile` for kernel fusion
- [MLX on GitHub][mlx-github] — Apple's array framework for Apple Silicon
- [MLX Computation Graph model (DeepWiki)][mlx-deepwiki] — lazy graph internals

### NVIDIA (for comparison)

- [NVIDIA Ampere GPU Architecture Tuning Guide][ampere-guide] — A100 specs:
  256KB register file, 164KB shared memory per SM
- [NVIDIA Hopper Tuning Guide][hopper-guide] — H100 specs, TMA, 228KB shared
  memory per SM
- [CUDA Programming Guide: Asynchronous Data Copies][ampere-async] — `cp.async`
  and `cuda::memcpy_async` on Ampere+
- [NVIDIA blog: Controlling Data Movement on Ampere][ampere-async-blog] —
  overlapping compute with memory via `cp.async`
- [CUTLASS Tutorial: Hopper Tensor Memory Accelerator (Colfax)][hopper-tma] —
  TMA for bulk async tensor copies
- [RTX 4090 specs (Runpod)][rtx4090-specs]

### Papers

- [SageAttention (ICLR 2025)][sageattention] — per-block int8 Q/K, fp16 V
- [SageAttention2 (ICML 2025 NeurIPS 2025 Spotlight)][sageattention2] —
  INT4 Q/K + FP8 P/V
- [KIVI (ICML 2024): A Tuning-Free Asymmetric 2bit KV Cache Quantization][kivi]
  — per-channel K quant, per-token V quant
- [INT-FlashAttention (2024)][intflash] — INT8 FlashAttention, per-token K scales
- [SmoothQuant (ICML 2023)][smoothquant] — per-channel activation smoothing
  before quantization

### Project-local references

- MLX Steel attention header: `mlx/include/mlx/backend/metal/kernels/steel/attn/nax.h`
- MLX Steel GEMM header: `mlx/include/mlx/backend/metal/kernels/steel/gemm/nax.h`
- Apple MPP header: `MPPTensorOpsMatMul2d.h` in MacOSX SDK
- Our custom M=32 NAXFrag: `kernels/nax_m32.h`
- Our SDPA variants: `kernels/scatter_sdpa.metal`, `kernels/scatter_sdpa_bk64.metal`

<!-- Reference links -->
[apple-m5]: https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/
[apple-m5-pro]: https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
[apple-uma]: https://developer.apple.com/videos/play/wwdc2020/10686/
[creative-m5]: https://creativestrategies.com/research/m5-apple-silicon-its-all-about-the-cache-and-tensors/
[msl-spec]: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
[mpp-guide]: https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf
[wwdc25-ml]: https://developer.apple.com/videos/play/wwdc2025/262/
[metal-perf-wwdc20]: https://developer.apple.com/videos/play/wwdc2020/10632/
[liuliu-matmul4]: https://github.com/liuliu/example_matmul_metal4
[mlx-mpp-pr]: https://github.com/ml-explore/mlx/pull/2687
[mlx-lazy]: https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html
[mlx-github]: https://github.com/ml-explore/mlx
[mlx-deepwiki]: https://deepwiki.com/ml-explore/mlx/3.1-computation-graph-model
[ampere-guide]: https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html
[hopper-guide]: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
[ampere-async]: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html
[cuda-coalesce]: https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
[ampere-async-blog]: https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/
[hopper-tma]: https://research.colfax-intl.com/tutorial-hopper-tma/
[rtx4090-specs]: https://www.runpod.io/articles/guides/nvidia-rtx-4090
[sageattention]: https://arxiv.org/abs/2410.02367
[sageattention2]: https://arxiv.org/abs/2411.10958
[kivi]: https://arxiv.org/abs/2402.02750
[intflash]: https://arxiv.org/abs/2409.16997
[smoothquant]: https://arxiv.org/abs/2211.10438

Sources:
- [Apple unleashes M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)
- [Apple M5 Pro and M5 Max](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)
- [Creative Strategies — M5 Cache and Tensors](https://creativestrategies.com/research/m5-apple-silicon-its-all-about-the-cache-and-tensors/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Performance Primitives Programming Guide](https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf)
- [WWDC25: Combine Metal 4 ML and graphics](https://developer.apple.com/videos/play/wwdc2025/262/)
- [WWDC20: Optimize Metal Performance](https://developer.apple.com/videos/play/wwdc2020/10632/)
- [WWDC20: Explore Apple Silicon System Architecture](https://developer.apple.com/videos/play/wwdc2020/10686/)
- [MLX Lazy Evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX DeepWiki — Computation Graph](https://deepwiki.com/ml-explore/mlx/3.1-computation-graph-model)
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
- [CUDA Programming Guide — Async Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [NVIDIA blog — Ampere data movement](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)
- [CUTLASS Hopper TMA Tutorial (Colfax)](https://research.colfax-intl.com/tutorial-hopper-tma/)
- [SageAttention (arXiv 2410.02367)](https://arxiv.org/abs/2410.02367)
- [SageAttention2 (arXiv 2411.10958)](https://arxiv.org/abs/2411.10958)
- [KIVI (arXiv 2402.02750)](https://arxiv.org/abs/2402.02750)
- [INT-FlashAttention (arXiv 2409.16997)](https://arxiv.org/abs/2409.16997)
- [SmoothQuant (arXiv 2211.10438)](https://arxiv.org/abs/2211.10438)
