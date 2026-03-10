"""
Code here all yoinked from https://github.com/pytorch/FBGEMM
"""


from __future__ import annotations

import torch
import triton
import triton.language as tl


# based on https://github.com/pytorch/FBGEMM/blob/2364886be020b45246eacf8a3eb294194bb08b84/fbgemm_gpu/experimental/gen_ai/src/moe/index_shuffling.cu
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["n_experts"],
    reset_to_zero=["token_counts_ptr"],
)
@triton.jit
def _index_shuffling_topk_kernel(
    scores_ptr,
    out_experts_ptr,
    out_token_offsets_ptr,
    token_counts_ptr,
    stride_t,
    stride_e,
    n_experts,
    TOP_K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    token_idx = tl.program_id(0)
    expert_offsets = tl.arange(0, BLOCK_E)
    mask = expert_offsets < n_experts
    scores = tl.load(
        scores_ptr + token_idx * stride_t + expert_offsets * stride_e,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    expert_ids = expert_offsets.to(tl.int32)

    for k in tl.static_range(TOP_K):
        _, idx = tl.max(scores, axis=0, return_indices=True, return_indices_tie_break_left=True)
        offset = token_idx * TOP_K + k
        token_offset = tl.atomic_add(token_counts_ptr + idx, 1, sem="relaxed")
        tl.store(out_experts_ptr + offset, idx)
        tl.store(out_token_offsets_ptr + offset, token_offset)
        scores = tl.where(expert_ids == idx, -float("inf"), scores)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
    ],
    key=["n_elements", "n_experts"],
)
@triton.jit
def _index_shuffling_scatter_kernel(
    expert_indices_ptr,
    token_offsets_ptr,
    token_counts_ptr,
    shuffled_expert_indices_ptr,
    shuffled_token_indices_ptr,
    n_elements,
    n_experts,
    top_k,
    BLOCK_E: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    expert_indices = tl.load(expert_indices_ptr + offsets, mask=mask, other=0)
    token_offsets = tl.load(token_offsets_ptr + offsets, mask=mask, other=0)

    expert_offsets = tl.arange(0, BLOCK_E)
    expert_mask = expert_offsets < n_experts
    token_counts = tl.load(token_counts_ptr + expert_offsets, mask=expert_mask, other=0)
    token_starts = tl.sum(
        tl.where(expert_offsets[None, :] < tl.expand_dims(expert_indices, 1), token_counts[None, :], 0),
        axis=1,
    )
    shuffled_offsets = token_starts + token_offsets

    tl.store(shuffled_expert_indices_ptr + shuffled_offsets, expert_indices, mask=mask)
    tl.store(shuffled_token_indices_ptr + shuffled_offsets, offsets // top_k, mask=mask)


def index_shuffling(scores: torch.Tensor, top_k: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert scores.ndim == 2
    assert top_k > 0

    n_tokens, n_experts = scores.shape
    assert top_k <= n_experts
    assert scores.is_cuda

    block_e = triton.next_power_of_2(n_experts)
    assert top_k <= 8
    assert block_e <= 1024

    n_elements = n_tokens * top_k
    expert_indices = torch.empty(n_elements, device=scores.device, dtype=torch.int32)
    token_counts = torch.zeros(n_experts, device=scores.device, dtype=torch.int32)
    token_offsets = torch.empty(n_elements, device=scores.device, dtype=torch.int32)
    _index_shuffling_topk_kernel[(n_tokens,)](
        scores,
        expert_indices,
        token_offsets,
        token_counts,
        scores.stride(0),
        scores.stride(1),
        n_experts,
        TOP_K=top_k,
        BLOCK_E=block_e,
    )

    # can't combine the two kernels cleanly. there needs to be grid wide sync at this point for the
    # topk kernel to complete its atomics into token_counts. unfortunately triton has no explicit global 
    # sync primitives, so that's not possible.
    shuffled_expert_indices = torch.empty(n_elements, device=scores.device, dtype=torch.int32)
    shuffled_token_indices = torch.empty(n_elements, device=scores.device, dtype=torch.int32)

    def scatter_grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _index_shuffling_scatter_kernel[scatter_grid](
        expert_indices,
        token_offsets,
        token_counts,
        shuffled_expert_indices,
        shuffled_token_indices,
        n_elements,
        n_experts,
        top_k,
        BLOCK_E=block_e,
    )

    return token_counts, shuffled_expert_indices, shuffled_token_indices


# yoinked from https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gen_ai/gen_ai/moe/gather_scatter.py
@triton.jit
def _fbgemm_scatter_add_dense_tokens(
    out_tokens,
    in_tokens,
    token_indices,
    valid_token_count,
    D: tl.constexpr,
    BLOCK_D_OUTER: tl.constexpr,
    BLOCK_D_INNER: tl.constexpr,
):
    input_token_index = tl.program_id(0).to(tl.int64)
    feature_offset = tl.program_id(1) * BLOCK_D_OUTER + tl.arange(0, BLOCK_D_INNER)[:]

    if valid_token_count is not None:
        valid_token_count = tl.load(
            valid_token_count, None, eviction_policy="evict_last"
        )
        if input_token_index >= valid_token_count:
            return

    output_token_index = tl.load(
        token_indices + input_token_index, None, eviction_policy="evict_last"
    ).to(tl.int64)

    for _ in range(0, BLOCK_D_OUTER // BLOCK_D_INNER):
        input_token_value = tl.load(
            in_tokens + input_token_index * D + feature_offset,
            None,
            eviction_policy="evict_first",
        )

        tl.atomic_add(
            out_tokens + output_token_index * D + feature_offset,
            input_token_value,
            None,
            sem="relaxed",
        )
        feature_offset += BLOCK_D_INNER


def scatter_add_dense_tokens(
    out_tokens: torch.Tensor,
    in_tokens: torch.Tensor,
    token_indices: torch.Tensor,
    valid_token_count: torch.Tensor | None = None,
) -> None:
    assert torch.version.hip is not None or (
        torch.version.cuda is not None and torch.version.cuda >= "12.4"
    ), "Requires CUDA version 12.4 or later on Nvidia GPUs!"

    assert out_tokens.ndim == 2 and in_tokens.ndim == 2
    assert in_tokens.shape[1] == out_tokens.shape[1]
    assert token_indices.ndim == 1
    assert out_tokens.is_cuda and in_tokens.is_cuda and token_indices.is_cuda
    assert out_tokens.is_contiguous()
    assert in_tokens.is_contiguous()
    assert token_indices.is_contiguous()
    assert out_tokens.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert in_tokens.dtype == out_tokens.dtype

    a, D = in_tokens.shape
    if a == 0:
        return
    assert token_indices.shape == (a,)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    if a >= NUM_SMS:
        BLOCK_D_OUTER = D
        BLOCK_D_INNER = 1024
    else:
        BLOCK_D_OUTER = 512
        BLOCK_D_INNER = 256
    while D % BLOCK_D_OUTER != 0:
        BLOCK_D_OUTER //= 2
    while D % BLOCK_D_INNER != 0:
        BLOCK_D_INNER //= 2

    grid = (a, D // BLOCK_D_OUTER)
    _fbgemm_scatter_add_dense_tokens[grid](
        out_tokens,
        in_tokens,
        token_indices,
        valid_token_count,
        D,  # pyre-ignore
        BLOCK_D_OUTER,  # pyre-ignore
        BLOCK_D_INNER,  # pyre-ignore
    )
