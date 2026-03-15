"""
Grouped GEMM kernel adapted from FBGEMM's Triton implementation.
"""


from __future__ import annotations

import inspect
import warnings
from typing import Optional

import torch
from torch.library import triton_op, wrap_triton
import triton
import triton.language as tl
from triton.runtime import driver


_NV_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=1,
    )
    for block_size_m in [64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
]

_AMD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "waves_per_eu": waves_per_cu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [32, 64, 128]
    for block_size_n in [32, 64, 128, 256]
    for block_size_k in [128, 256]
    for num_stages in [1, 2]
    for num_warps, waves_per_cu in [(4, 1), (8, 2), (16, 4)]
    for matrix_instr_nonkdim in [16]
]

HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)
_HAS_WS_SUPPORT = None


def _check_ws_support() -> bool:
    if not hasattr(tl, "async_task"):
        return False
    config_signature = inspect.signature(triton.Config).parameters
    if (
        "num_consumer_groups" not in config_signature
        or "num_buffers_warp_spec" not in config_signature
    ):
        return False
    return HAS_TMA_DESC


def _set_ws_support() -> None:
    global _HAS_WS_SUPPORT
    if _HAS_WS_SUPPORT is None:
        _HAS_WS_SUPPORT = _check_ws_support()


_set_ws_support()


class TmaAutoTuneHelper:
    class KernelParamWrapper:
        def __init__(self, desc: torch.Tensor):
            self.desc = desc

        def tma_desc_cpu_ptr(self) -> int:
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = driver.active.utils.fill_1d_tma_descriptor
        self.fill_2d_tma_descriptor_inner = driver.active.utils.fill_2d_tma_descriptor
        if HAS_TMA_DESC:
            self.descriptors: dict[str, torch.Tensor] = {}
        else:
            self.cuda_descriptors: dict[str, torch.Tensor] = {}

    def init_tma_descriptor(self, name: str) -> None:
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(self.TMA_SIZE, device="cpu", dtype=torch.int8)
        else:
            self.cuda_descriptors[name] = torch.empty(self.TMA_SIZE, device="cuda", dtype=torch.int8)

    def fill_1d_tma_descriptor(self, name: str, ptr: int, dim: int, block_dim: int, element_size: int) -> None:
        if HAS_TMA_DESC:
            desc = self.descriptors[name]
            assert desc.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, desc.data_ptr())
        else:
            desc = self.cuda_descriptors[name]
            buf = torch.empty_like(desc, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(ptr, dim, block_dim, element_size, buf.data_ptr())
            desc.copy_(buf, non_blocking=True)

    def fill_2d_tma_descriptor(
        self,
        name: str,
        ptr: int,
        dim1: int,
        dim0: int,
        block_dim1: int,
        block_dim0: int,
        element_size: int,
    ) -> None:
        if HAS_TMA_DESC:
            desc = self.descriptors[name]
            assert desc.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc.data_ptr()
            )
        else:
            desc = self.cuda_descriptors[name]
            buf = torch.empty_like(desc, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf.data_ptr()
            )
            desc.copy_(buf, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name: str):
        if HAS_TMA_DESC:
            return self.KernelParamWrapper(self.descriptors[name])
        return self.cuda_descriptors[name]


@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    restore_value=["c_ptr"],
)
@triton.jit
def _fbgemm_grouped_gemm(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    scatter_add_indices,
    m_sizes,
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    tl.static_assert(not (FUSE_SCATTER_ADD and USE_TMA_STORE), "Cannot fuse scatter add with TMA store!")

    tidx = tl.program_id(0)
    dtype: tl.dtype = c_ptr.dtype.element_ty

    m_end_offset = tl.zeros((), dtype=tl.int64)
    iterated_tiles = tl.zeros((), dtype=tl.int32)
    for g in tl.range(G):
        m_size = tl.load(m_sizes + g)
        if m_size > 0:
            m_start_offset = m_end_offset
            m_end_offset = m_start_offset + m_size
            n_start_offset = g.to(tl.int64) * N
            n_size = N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                c_desc_ptr = tl.make_tensor_descriptor(
                    c_ptr + m_start_offset * N,
                    shape=[m_size, n_size],
                    strides=[n_size, 1],
                    block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                )

            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

                if USE_TMA_LOAD:
                    tl.static_assert(K % BLOCK_SIZE_K == 0)
                    m_offset = (m_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (n_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl._experimental_descriptor_load(
                            a_desc_ptr, [m_offset, k_offset], [BLOCK_SIZE_M, BLOCK_SIZE_K], dtype
                        )
                        b = tl._experimental_descriptor_load(
                            b_desc_ptr, [n_offset, k_offset], [BLOCK_SIZE_N, BLOCK_SIZE_K], dtype
                        )
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, acc=accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    a_ptrs = a_desc_ptr + (m_start_offset + offs_am[:, None]) * K + offs_k[None, :]
                    b_ptrs = b_desc_ptr + (n_start_offset + offs_bn[:, None]) * K + offs_k[None, :]
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        updated_k_offset = k_offset + offs_k
                        updated_k_offset_mask = updated_k_offset[None, :] < K
                        a = tl.load(a_ptrs, mask=(offs_am[:, None] < m_size) & updated_k_offset_mask, other=0.0)
                        b = tl.load(b_ptrs, mask=(offs_bn[:, None] < n_size) & updated_k_offset_mask, other=0.0)
                        if a.dtype != b.dtype:
                            if a.dtype == tl.float32 or b.dtype == tl.float32:
                                a = a.to(tl.float32)
                                b = b.to(tl.float32)
                            else:
                                b = b.to(a.dtype)
                        accumulator += tl.dot(a, b.T)
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                if USE_TMA_STORE:
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    c_desc_ptr.store([m_offset, n_offset], accumulator.to(c_ptr.dtype.element_ty))
                elif FUSE_SCATTER_ADD:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    mask = offs_am < m_size
                    m_offsets = tl.load(scatter_add_indices + m_start_offset + offs_am, mask=mask, cache_modifier=".ca")
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c = accumulator.to(c_ptr.dtype.element_ty)
                    tl.atomic_add(
                        c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                        c,
                        mask=mask[:, None] & (offs_bn[None, :] < n_size),
                        sem="relaxed",
                    )
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c = accumulator.to(c_ptr.dtype.element_ty)
                    tl.store(
                        c_ptr + (m_start_offset + offs_am[:, None]) * N + offs_bn[None, :],
                        c,
                        mask=(offs_am[:, None] < m_size) & (offs_bn[None, :] < n_size),
                    )
                tidx += NUM_SMS

            iterated_tiles += num_tiles

@triton_op(
    "world_engine::grouped_gemm",
    mutates_args=(),
)
def grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_fast_accum: bool = True,
    use_warp_specialization: bool = True,
    fp32_output: bool = False,
    scatter_add_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    use_tma_load = not torch.version.hip
    use_tma_store = False

    if use_tma_load and not HAS_TMA_DESC:
        use_tma_load = False

    if use_warp_specialization and torch.version.hip:
        use_warp_specialization = False

    if use_warp_specialization and not _HAS_WS_SUPPORT:
        use_warp_specialization = False

    if use_warp_specialization:
        use_tma_store = True

    groups = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    m, k = x.shape
    n = w.shape[0] // groups
    assert k == w.shape[1]

    if k % 8 != 0 or n % 8 != 0:
        use_warp_specialization = False
        use_tma_load = False
        use_tma_store = False
        assert scatter_add_indices is None, "Fused scatter add path disabled when K or N is not a multiple of 8."

    fuse_scatter_add = scatter_add_indices is not None
    y_dtype = torch.float32 if fp32_output else torch.bfloat16
    y = torch.empty((m, n), device=x.device, dtype=y_dtype)

    if fuse_scatter_add:
        assert scatter_add_indices.is_contiguous()
        assert scatter_add_indices.shape == (m,)

    if m == 0 or n == 0:
        return y

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    desc_helper = None
    desc_x = x
    desc_w = w

    if use_tma_load:
        desc_helper = TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("x")
        desc_helper.init_tma_descriptor("w")
        desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
        desc_w = desc_helper.get_tma_descriptor_kernel_param("w")

    if use_tma_store:
        def alloc_fn(size: int, alignment: int, stream: Optional[int]):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

    def grid(meta):
        if use_tma_load:
            desc_helper.fill_2d_tma_descriptor(
                "x",
                x.data_ptr(),
                m,
                k,
                meta["BLOCK_SIZE_M"] // meta["NUM_CONSUMER_GROUPS"],
                meta["BLOCK_SIZE_K"],
                x.element_size(),
            )
            desc_helper.fill_2d_tma_descriptor(
                "w",
                w.data_ptr(),
                n * groups,
                k,
                meta["BLOCK_SIZE_N"],
                meta["BLOCK_SIZE_K"],
                w.element_size(),
            )
        return (num_sms,)

    m_bucket_cap = 16384
    m_bucket = min(triton.next_power_of_2(m), m_bucket_cap)
    # Keep the non-warp-specialized path available; it is sufficient for the
    # local MoE fallback path and much easier to reason about while debugging.
    wrap_triton(_fbgemm_grouped_gemm)[grid](
        desc_x,
        desc_w,
        y,
        scatter_add_indices,
        m_sizes,
        groups,
        m_bucket,
        n,
        k,
        num_sms,
        fuse_scatter_add,
        use_tma_load,
        use_tma_store,
        use_fast_accum,
    )
    return y
