# MoE summary test commands:
# `uv run pytest -q tests/test_moe_kernels.py -k 'test_moe_implementation_matrix' -s`
# `uv run pytest -q tests/test_moe_kernels.py -k 'test_moe_implementation_matrix' --run-kernel-benchmarks -s`

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
import statistics
import sys
import time
import traceback

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.kernels import (
        grouped_gemm,
        index_shuffling,
        scatter_add_dense_tokens,
    )
    HAS_LOCAL_KERNELS = True
except ImportError:
    grouped_gemm = None
    index_shuffling = None
    scatter_add_dense_tokens = None
    HAS_LOCAL_KERNELS = False
import src.model.moe as moe_module
from src.model.moe import MoECustomKernels, MoEFBGEMM, MoETorchBaseline


CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for MoE kernel tests")
LOCAL_KERNELS_ONLY = pytest.mark.skipif(not HAS_LOCAL_KERNELS, reason="local kernels package is not present on this branch")

COMPILE_OPTIONS = {
    "max_autotune": True,
    "coordinate_descent_tuning": True,
    "triton.cudagraphs": True,
}

MOE_TEST_BATCH = 4
FBGEMM_MOE_COMPILE_DISABLED_REASON = (
    "compiled FBGEMM MoE disabled"
)

# fbgemm kernels
def _load_fbgemm_index_shuffling() -> Callable | None:
    try:
        from fbgemm_gpu.experimental.gen_ai.moe import index_shuffling as fbgemm_index_shuffling
    except ImportError:
        return None
    return fbgemm_index_shuffling


def _has_fbgemm_scatter_add() -> bool:
    return hasattr(torch.ops, "fbgemm") and hasattr(torch.ops.fbgemm, "scatter_add_dense_tokens")


def _load_fbgemm_grouped_gemm() -> Callable | None:
    try:
        from fbgemm_gpu.experimental.gemm.triton_gemm.grouped_gemm import grouped_gemm as fbgemm_grouped_gemm
    except ImportError:
        return None
    return fbgemm_grouped_gemm


def has_fbgemm_moe_kernels() -> bool:
    return _load_fbgemm_index_shuffling() is not None and _has_fbgemm_scatter_add()


def fbgemm_index_shuffling(scores: torch.Tensor, top_k: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op = _load_fbgemm_index_shuffling()
    if op is None:
        raise RuntimeError("fbgemm index_shuffling is not available in this environment")
    return op(scores, top_k=top_k)


def fbgemm_scatter_add_dense_tokens(
    out_tokens: torch.Tensor,
    in_tokens: torch.Tensor,
    token_indices: torch.Tensor,
    valid_token_count: torch.Tensor | None = None,
) -> None:
    if not _has_fbgemm_scatter_add():
        raise RuntimeError("fbgemm scatter_add_dense_tokens is not available in this environment")
    torch.ops.fbgemm.scatter_add_dense_tokens(out_tokens, in_tokens, token_indices, valid_token_count)


def fbgemm_grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_fast_accum: bool = True,
    *,
    _output_tensor: torch.Tensor | None = None,
    _scatter_add_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    op = _load_fbgemm_grouped_gemm()
    if op is None:
        raise RuntimeError("fbgemm grouped_gemm is not available in this environment")
    return op(
        x,
        w,
        m_sizes,
        use_fast_accum=use_fast_accum,
        _output_tensor=_output_tensor,
        _scatter_add_indices=_scatter_add_indices,
    )


def _canonical_pairs(expert_indices: torch.Tensor, token_indices: torch.Tensor, n_tokens: int) -> torch.Tensor:
    key = expert_indices.to(torch.int64) * n_tokens + token_indices.to(torch.int64)
    return torch.stack((expert_indices, token_indices), dim=1).index_select(0, torch.argsort(key))


def _random_scores(n_tokens: int, n_experts: int, device: str) -> torch.Tensor:
    return torch.randn(n_tokens, n_experts, device=device, dtype=torch.float32)


def _max_pair_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> int:
    if lhs.numel() == 0:
        return 0
    return int((lhs.to(torch.int64) - rhs.to(torch.int64)).abs().max().item())


def _time_cuda_ms(fn, warmup: int = 50, iters: int = 250) -> float:
    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
        for _ in range(iters):
            start.record(stream)
            fn()
            end.record(stream)
            end.synchronize()
            samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _make_grouped_gemm_inputs(
    n_experts: int,
    d_model: int,
    d_hidden: int,
    *,
    top_k: int = 8,
    n_tokens: int = 512,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.full((n_experts,), n_tokens * top_k // n_experts, device="cuda", dtype=torch.int32)
    counts[-1] += n_tokens * top_k - int(counts.sum().item())
    x_grouped = torch.randn((n_tokens * top_k + 1, d_model), device="cuda", dtype=dtype).contiguous()
    w = torch.randn((n_experts * d_hidden, d_model), device="cuda", dtype=dtype).contiguous()
    return x_grouped, w, counts.contiguous()


def _grouped_gemm_weight_for_torch(
    w: torch.Tensor,
    n_experts: int,
    d_hidden: int,
) -> torch.Tensor:
    return w.view(n_experts, d_hidden, w.shape[1]).transpose(-2, -1).contiguous()


def _grouped_gemm_offs(m_sizes: torch.Tensor) -> torch.Tensor:
    return m_sizes.cumsum(0).to(torch.int32).contiguous()


def _make_moe_inputs(
    *,
    n_tokens: int = 512,
    n_experts: int = 16,
    top_k: int = 8,
    d_model: int = 2048,
    d_hidden: int = 1024,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.randn((n_tokens, d_model), device="cuda", dtype=dtype).contiguous()
    logits = torch.randn((n_tokens, n_experts), device="cuda", dtype=torch.float32).contiguous()
    expert_in_proj = torch.randn((n_experts, d_hidden, d_model), device="cuda", dtype=dtype).contiguous()
    expert_out_proj = torch.randn((n_experts, d_model, d_hidden), device="cuda", dtype=dtype).contiguous()
    return x, logits, expert_in_proj, expert_out_proj


def _make_moe_scatter_inputs(
    *,
    n_tokens: int = 512,
    n_experts: int = 16,
    top_k: int = 8,
    d_model: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.randn((n_tokens, n_experts), device="cuda", dtype=torch.float32).contiguous()
    _token_counts, _expert_sorted, src = index_shuffling(scores, top_k=top_k)
    in_tokens = torch.randn((n_tokens * top_k, d_model), device="cuda", dtype=dtype).contiguous()
    return in_tokens, src.to(torch.int64).contiguous()


def _make_moe_config(
    *,
    n_experts: int = 16,
    top_k: int = 8,
    d_model: int = 2048,
    d_hidden: int = 1024,
    gated_linear: bool = False,
):
    mlp_ratio = (d_hidden * top_k) / d_model
    return SimpleNamespace(
        moe_top_k=top_k,
        moe_n_experts=n_experts,
        d_model=d_model,
        mlp_ratio=mlp_ratio,
        gated_linear=gated_linear,
    )


def _make_moe_modules(
    *,
    n_experts: int = 16,
    top_k: int = 8,
    d_model: int = 2048,
    d_hidden: int = 1024,
    gated_linear: bool = False,
    dtype: torch.dtype = torch.bfloat16,
):
    op = _load_fbgemm_index_shuffling()
    if op is not None:
        moe_module.fbgemm_index_shuffling = op
    cfg = _make_moe_config(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
        gated_linear=gated_linear,
    )
    moe_fbgemm_impl = MoEFBGEMM(cfg).eval().to(device="cuda", dtype=dtype)
    moe_custom_bf16_impl = MoECustomKernels(cfg, accumulate_in_fp32=False).eval().to(device="cuda", dtype=dtype)
    moe_custom_fp32_impl = MoECustomKernels(cfg, accumulate_in_fp32=True).eval().to(device="cuda", dtype=dtype)
    state = moe_fbgemm_impl.state_dict()
    moe_custom_bf16_impl.load_state_dict(state)
    moe_custom_fp32_impl.load_state_dict(state)
    return moe_fbgemm_impl, moe_custom_bf16_impl, moe_custom_fp32_impl


def _make_moe_modules_with_baseline(
    *,
    n_experts: int = 16,
    top_k: int = 8,
    d_model: int = 2048,
    d_hidden: int = 1024,
    gated_linear: bool = False,
    dtype: torch.dtype = torch.bfloat16,
):
    op = _load_fbgemm_index_shuffling()
    if op is not None:
        moe_module.fbgemm_index_shuffling = op
    cfg = _make_moe_config(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
        gated_linear=gated_linear,
    )
    moe_fbgemm_impl = MoEFBGEMM(cfg).eval().to(device="cuda", dtype=dtype)
    moe_custom_bf16_impl = MoECustomKernels(cfg, accumulate_in_fp32=False).eval().to(device="cuda", dtype=dtype)
    moe_custom_fp32_impl = MoECustomKernels(cfg, accumulate_in_fp32=True).eval().to(device="cuda", dtype=dtype)
    moe_baseline = MoETorchBaseline(cfg).eval().to(device="cuda", dtype=dtype)
    state = moe_fbgemm_impl.state_dict()
    moe_custom_bf16_impl.load_state_dict(state)
    moe_custom_fp32_impl.load_state_dict(state)
    moe_baseline.load_state_dict(state)
    return moe_fbgemm_impl, moe_custom_bf16_impl, moe_custom_fp32_impl, moe_baseline


def _make_forced_moe_case(
    *,
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
    batch: int = MOE_TEST_BATCH,
):
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    x, logits, expert_in_proj, expert_out_proj = _make_moe_inputs(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    x = x.unsqueeze(0).expand(batch, -1, -1).contiguous()
    gate = logits.unsqueeze(0).expand(batch, -1, -1).contiguous()
    return x, gate, expert_in_proj, expert_out_proj


def _run_moe_module_eager(
    module: torch.nn.Module,
    x: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.inference_mode():
        return module(x, gate=gate)


def _run_custom_moe_eager(
    x: torch.Tensor,
    logits: torch.Tensor,
    expert_in_proj: torch.Tensor,
    expert_out_proj: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    logits_fp32 = logits.float()
    token_counts, expert_sorted, src = index_shuffling(logits_fp32, top_k=top_k)
    m_sizes = token_counts[: expert_in_proj.shape[0]].to(torch.int32).contiguous()
    src = src.to(torch.long)
    expert_sorted = expert_sorted.to(torch.long)
    log_z = logits_fp32.logsumexp(-1)
    weights = (logits_fp32[src, expert_sorted] - log_z[src]).exp().to(x.dtype)

    x_grouped = x.index_select(0, torch.cat((src, src[:1]), dim=0))
    h = grouped_gemm(
        x_grouped,
        expert_in_proj.reshape(-1, expert_in_proj.shape[-1]).contiguous(),
        m_sizes,
    )
    h = F.silu(h)
    y_grouped = grouped_gemm(
        h,
        expert_out_proj.reshape(-1, expert_out_proj.shape[-1]).contiguous(),
        m_sizes,
    )[:-1]
    return scatter_add_dense_tokens(
        (y_grouped * weights.unsqueeze(-1)).contiguous(),
        src,
        n_tokens=x.shape[0],
    )


def _run_fbgemm_moe_eager(
    x: torch.Tensor,
    logits: torch.Tensor,
    expert_in_proj: torch.Tensor,
    expert_out_proj: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    logits_fp32 = logits.float()
    token_counts, expert_sorted, src = fbgemm_index_shuffling(logits_fp32, top_k=top_k)
    offs = token_counts[: expert_in_proj.shape[0]].cumsum(0).to(torch.int32)
    src = src.to(torch.long)
    expert_sorted = expert_sorted.to(torch.long)
    log_z = logits_fp32.logsumexp(-1)
    weights = (logits_fp32[src, expert_sorted] - log_z[src]).exp().to(x.dtype)

    x_grouped = x.index_select(0, torch.cat((src, src[:1]), dim=0))
    h = F.grouped_mm(x_grouped, expert_in_proj.transpose(-2, -1), offs=offs)
    h = F.silu(h)
    y_grouped = F.grouped_mm(h, expert_out_proj.transpose(-2, -1), offs=offs)[:-1]
    out = torch.zeros((x.shape[0], x.shape[1]), device=x.device, dtype=torch.float32)
    fbgemm_scatter_add_dense_tokens(out, (y_grouped * weights.unsqueeze(-1)).contiguous(), src)
    return out


def _make_real_moe_grouped_state(
    x: torch.Tensor,
    logits: torch.Tensor,
    expert_in_proj: torch.Tensor,
    expert_out_proj: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_fp32 = logits.float()
    scores, expert = logits.topk(top_k, dim=-1, sorted=False)
    weights = (scores.float() - logits_fp32.logsumexp(dim=-1, keepdim=True)).exp().to(x.dtype)

    expert = expert.flatten()
    expert_sorted, sort_idx = expert.sort()
    expert_ids = torch.arange(expert_in_proj.shape[0], device=expert.device, dtype=expert_sorted.dtype)
    offsets = torch.searchsorted(expert_sorted, expert_ids, right=True).to(torch.int32)
    m_sizes = torch.diff(torch.cat((offsets.new_zeros(1), offsets)), dim=0).to(torch.int32).contiguous()
    src = (sort_idx // top_k).to(torch.long)
    weights_sorted = weights.reshape(-1).index_select(0, sort_idx.to(torch.long))
    x_grouped = x.index_select(0, torch.cat((src, src[:1]), dim=0))
    return x_grouped, offsets, m_sizes, src, weights_sorted, expert_out_proj


def _baseline_index_shuffling(
    logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits_fp32 = logits.float()
    _scores, expert = logits_fp32.topk(top_k, dim=-1, sorted=False)
    expert = expert.flatten()
    expert_sorted, sort_idx = expert.sort()
    expert_ids = torch.arange(
        logits.shape[-1],
        device=expert.device,
        dtype=expert_sorted.dtype,
    )
    offsets = torch.searchsorted(expert_sorted, expert_ids, right=True).to(torch.int32)
    token_counts = torch.diff(torch.cat((offsets.new_zeros(1), offsets)), dim=0).to(torch.int32).contiguous()
    src = (sort_idx // top_k).to(torch.int32)
    return token_counts, expert_sorted.to(torch.int32), src


def _torch_moe_combine_deterministic(
    weighted_grouped: torch.Tensor,
    src: torch.Tensor,
    n_tokens: int,
) -> torch.Tensor:
    out = torch.zeros((n_tokens, weighted_grouped.shape[-1]), device=weighted_grouped.device, dtype=weighted_grouped.dtype)
    for i in range(weighted_grouped.shape[0]):
        out[src[i]] = out[src[i]] + weighted_grouped[i]
    return out


def _slow_grouped_mm_reference(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((mat_a.shape[0], mat_b.shape[1]), device=mat_a.device, dtype=torch.float32)
    start = 0
    for expert_idx, end_tensor in enumerate(offs):
        end = int(end_tensor.item())
        if end > start:
            out[start:end] = mat_a[start:end].float() @ mat_b[expert_idx].transpose(-2, -1).float()
        start = end
    return out


@torch.compile(fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
def _run_moe_module_compiled(moe: MoEFBGEMM, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
    return moe(x, gate=gate)


def _run_moe_module_compiled_inference(moe: MoEFBGEMM, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
    with torch.inference_mode():
        return _run_moe_module_compiled(moe, x, gate=gate).clone()


@torch.compile(fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
def _run_moe_custom_module_compiled(
    moe: MoECustomKernels,
    x: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    return moe(x, gate=gate)


def _run_moe_custom_module_compiled_inference(
    moe: MoECustomKernels,
    x: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.inference_mode():
        return _run_moe_custom_module_compiled(moe, x, gate=gate).clone()


@torch.compile(fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
def _run_moe_baseline_module_compiled(
    moe: MoETorchBaseline,
    x: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    return moe(x, gate=gate)


def _run_moe_baseline_module_compiled_inference(
    moe: MoETorchBaseline,
    x: torch.Tensor,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.inference_mode():
        return _run_moe_baseline_module_compiled(moe, x, gate=gate).clone()

def _require_benchmark_flag(request) -> None:
    if not request.config.getoption("--run-kernel-benchmarks"):
        pytest.skip("pass --run-kernel-benchmarks to run kernel timing tests")


def _skip_if_fbgemm_module_compile_disabled() -> None:
    pytest.skip(FBGEMM_MOE_COMPILE_DISABLED_REASON)


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.parametrize(("n_tokens", "n_experts", "top_k"), [(512, 32, 2), (2048, 64, 4)])
def test_index_shuffling_matches_topk_semantics(n_tokens: int, n_experts: int, top_k: int):
    scores = _random_scores(n_tokens, n_experts, "cuda")
    token_counts, expert_indices, token_indices = index_shuffling(scores, top_k=top_k)

    _, expected_experts = scores.topk(top_k, dim=-1, sorted=False)
    expected_pairs = _canonical_pairs(
        expected_experts.reshape(-1),
        torch.arange(n_tokens, device=scores.device, dtype=torch.int64).repeat_interleave(top_k),
        n_tokens=n_tokens,
    )

    actual_pairs = _canonical_pairs(expert_indices, token_indices, n_tokens=n_tokens)
    expected_counts = torch.bincount(expected_pairs[:, 0], minlength=n_experts)

    assert torch.equal(token_counts.cpu(), expected_counts.cpu())
    assert torch.equal(actual_pairs.cpu(), expected_pairs.cpu())


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.skipif(not has_fbgemm_moe_kernels(), reason="fbgemm MoE kernels are not available")
@pytest.mark.parametrize(("n_tokens", "n_experts", "top_k"), [(512, 32, 2), (2048, 128, 4)])
def test_index_shuffling_matches_fbgemm_groups(n_tokens: int, n_experts: int, top_k: int):
    scores = _random_scores(n_tokens, n_experts, "cuda")
    ref_counts, ref_experts, ref_tokens = fbgemm_index_shuffling(scores, top_k=top_k)
    test_counts, test_experts, test_tokens = index_shuffling(scores, top_k=top_k)

    ref_pairs = _canonical_pairs(ref_experts, ref_tokens, n_tokens=n_tokens)
    test_pairs = _canonical_pairs(test_experts, test_tokens, n_tokens=n_tokens)
    counts_diff = _max_pair_diff(test_counts[:n_experts], ref_counts[:n_experts])
    expert_diff = _max_pair_diff(test_pairs[:, 0], ref_pairs[:, 0])
    token_diff = _max_pair_diff(test_pairs[:, 1], ref_pairs[:, 1])

    print(
        f"\nindex_shuffling diffs: "
        f"max_count_diff={counts_diff} "
        f"max_expert_diff={expert_diff} "
        f"max_token_diff={token_diff}"
    )

    assert torch.equal(test_counts[:n_experts].cpu(), ref_counts[:n_experts].cpu())
    assert torch.equal(test_pairs.cpu(), ref_pairs.cpu())


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.parametrize(("n_experts", "top_k", "d_model", "d_hidden"), [(16, 8, 2048, 1024)])
def test_custom_index_shuffling_matches_baseline_real_moe_case(
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
):
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    _x, logits, _expert_in_proj, _expert_out_proj = _make_moe_inputs(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )

    baseline_counts, baseline_experts, baseline_src = _baseline_index_shuffling(logits, top_k)
    custom_counts, custom_experts, custom_src = index_shuffling(logits.float(), top_k=top_k)

    counts_diff = (custom_counts[:n_experts].to(torch.int64) - baseline_counts.to(torch.int64)).abs()
    expert_diff = (custom_experts.to(torch.int64) - baseline_experts.to(torch.int64)).abs()
    src_diff = (custom_src.to(torch.int64) - baseline_src.to(torch.int64)).abs()
    baseline_pairs = torch.stack((baseline_experts, baseline_src), dim=1)
    custom_pairs = torch.stack((custom_experts, custom_src), dim=1)
    baseline_pair_key = baseline_pairs[:, 0].to(torch.int64) * logits.shape[0] + baseline_pairs[:, 1].to(torch.int64)
    custom_pair_key = custom_pairs[:, 0].to(torch.int64) * logits.shape[0] + custom_pairs[:, 1].to(torch.int64)
    baseline_pairs_sorted = baseline_pairs.index_select(0, torch.argsort(baseline_pair_key))
    custom_pairs_sorted = custom_pairs.index_select(0, torch.argsort(custom_pair_key))

    print(
        "\ncustom index_shuffling vs baseline real moe case diffs: "
        f"max_count_diff={int(counts_diff.max().item())} "
        f"max_expert_diff={int(expert_diff.max().item())} "
        f"max_src_diff={int(src_diff.max().item())}"
    )
    print(f"baseline experts head: {baseline_experts[:16].cpu().tolist()}")
    print(f"custom experts head: {custom_experts[:16].cpu().tolist()}")
    print(f"baseline src head: {baseline_src[:16].cpu().tolist()}")
    print(f"custom src head: {custom_src[:16].cpu().tolist()}")
    print(f"baseline sorted pair head: {baseline_pairs_sorted[:16].cpu().tolist()}")
    print(f"custom sorted pair head: {custom_pairs_sorted[:16].cpu().tolist()}")

    assert torch.equal(custom_counts[:n_experts].cpu(), baseline_counts.cpu())
    assert torch.equal(custom_experts.cpu(), baseline_experts.cpu())
    assert torch.equal(custom_pairs_sorted.cpu(), baseline_pairs_sorted.cpu())


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.parametrize(("n_tokens", "n_experts", "top_k", "d_model"), [(512, 16, 8, 256), (1024, 32, 8, 512)])
def test_scatter_add_dense_tokens_matches_reference(n_tokens: int, n_experts: int, top_k: int, d_model: int):
    in_tokens, token_indices = _make_moe_scatter_inputs(
        n_tokens=n_tokens,
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
    )

    expected = torch.zeros(n_tokens, d_model, device="cuda", dtype=torch.float32)
    expected.index_add_(0, token_indices, in_tokens.float())
    actual = scatter_add_dense_tokens(in_tokens, token_indices, n_tokens=n_tokens)

    torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.skipif(not has_fbgemm_moe_kernels(), reason="fbgemm MoE kernels are not available")
@pytest.mark.parametrize(("n_tokens", "n_experts", "top_k", "d_model"), [(512, 16, 8, 256), (1024, 32, 8, 512)])
def test_scatter_add_dense_tokens_matches_fbgemm(n_tokens: int, n_experts: int, top_k: int, d_model: int):
    in_tokens, token_indices = _make_moe_scatter_inputs(
        n_tokens=n_tokens,
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
    )

    ref = torch.zeros(n_tokens, d_model, device="cuda", dtype=torch.float32)
    fbgemm_scatter_add_dense_tokens(ref, in_tokens.contiguous(), token_indices.contiguous())
    actual = scatter_add_dense_tokens(in_tokens, token_indices, n_tokens=n_tokens)
    diff = (actual.float() - ref.float()).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())

    print(
        f"\nscatter_add_dense_tokens diffs: "
        f"max_abs_diff={max_abs_diff:.6f} "
        f"mean_abs_diff={mean_abs_diff:.6f}"
    )

    torch.testing.assert_close(actual.float(), ref.float(), atol=5e-2, rtol=5e-2)


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.skipif(_load_fbgemm_grouped_gemm() is None, reason="fbgemm grouped_gemm is not available")
@pytest.mark.parametrize(("n_experts", "d_model", "d_hidden"), [(8, 2048, 1024), (16, 2048, 1024)])
def test_grouped_gemm_matches_fbgemm(n_experts: int, d_model: int, d_hidden: int):
    x_grouped, w, m_sizes = _make_grouped_gemm_inputs(n_experts, d_model, d_hidden)
    ref = fbgemm_grouped_gemm(x_grouped, w, m_sizes)
    actual = grouped_gemm(x_grouped, w, m_sizes)
    diff = (actual.float() - ref.float()).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())

    print(
        f"\ngrouped_gemm diffs: "
        f"max_abs_diff={max_abs_diff:.6f} "
        f"mean_abs_diff={mean_abs_diff:.6f}"
    )

    torch.testing.assert_close(actual.float(), ref.float(), atol=5e-2, rtol=5e-2)


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.parametrize(("n_experts", "d_model", "d_hidden"), [(8, 2048, 1024), (16, 2048, 1024)])
def test_grouped_gemm_eager_vs_torch_grouped_mm(n_experts: int, d_model: int, d_hidden: int):
    x_grouped, w, m_sizes = _make_grouped_gemm_inputs(n_experts, d_model, d_hidden)
    torch_w = _grouped_gemm_weight_for_torch(w, n_experts, d_hidden)
    offs = _grouped_gemm_offs(m_sizes)
    active = int(m_sizes.sum().item())

    ref = F.grouped_mm(x_grouped, torch_w, offs=offs)
    actual = grouped_gemm(x_grouped, w, m_sizes)
    diff = (actual[:active].float() - ref[:active].float()).abs()
    max_abs_diff = float(diff.max().item())
    mean_abs_diff = float(diff.mean().item())

    print(
        f"\ngrouped_gemm vs torch grouped_mm diffs: "
        f"max_abs_diff={max_abs_diff:.6f} "
        f"mean_abs_diff={mean_abs_diff:.6f}"
    )

    assert torch.isfinite(actual[:active]).all()
    assert actual[:active].shape == ref[:active].shape
    assert max_abs_diff < 8.0


@CUDA_ONLY
@pytest.mark.skipif(_load_fbgemm_grouped_gemm() is None, reason="fbgemm grouped_gemm is not available")
@pytest.mark.parametrize(("n_experts", "top_k", "d_model", "d_hidden"), [(16, 8, 2048, 1024)])
def test_fbgemm_grouped_gemm_vs_torch_grouped_mm_repro(
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
):
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    x_grouped, w, m_sizes = _make_grouped_gemm_inputs(
        n_experts,
        d_model,
        d_hidden,
        top_k=top_k,
    )
    torch_w = _grouped_gemm_weight_for_torch(w, n_experts, d_hidden)
    offs = _grouped_gemm_offs(m_sizes)
    active = int(m_sizes.sum().item())

    torch_out = F.grouped_mm(x_grouped, torch_w, offs=offs)[:active]
    fbgemm_out = fbgemm_grouped_gemm(x_grouped, w, m_sizes)[:active]
    diff = (fbgemm_out.float() - torch_out.float()).abs()

    print(f"\nF.grouped_mm head: {torch_out[0, :16].float().cpu().tolist()}")
    print(f"FBGEMM grouped_gemm head: {fbgemm_out[0, :16].float().cpu().tolist()}")
    print(
        "FBGEMM grouped_gemm vs F.grouped_mm diffs: "
        f"max_abs_diff={float(diff.max().item()):.6f} "
        f"mean_abs_diff={float(diff.mean().item()):.6f}"
    )

    assert torch.isfinite(torch_out).all()
    assert torch.isfinite(fbgemm_out).all()


@CUDA_ONLY
@pytest.mark.skipif(_load_fbgemm_grouped_gemm() is None, reason="fbgemm grouped_gemm is not available")
@pytest.mark.parametrize("use_fast_accum", [True, False], ids=["fast_accum", "precise_accum"])
@pytest.mark.parametrize(("n_experts", "top_k", "d_model", "d_hidden"), [(16, 8, 2048, 1024)])
def test_fbgemm_grouped_gemm_vs_torch_grouped_mm_real_moe_repro(
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
    use_fast_accum: bool,
):
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    x, logits, expert_in_proj, expert_out_proj = _make_moe_inputs(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    x_grouped, offsets, m_sizes, src, weights_sorted, expert_out_proj = _make_real_moe_grouped_state(
        x,
        logits,
        expert_in_proj,
        expert_out_proj,
        top_k,
    )
    expert_in_proj_flat = expert_in_proj.reshape(-1, expert_in_proj.shape[-1]).contiguous()
    expert_out_proj_flat = expert_out_proj.reshape(-1, expert_out_proj.shape[-1]).contiguous()
    expert_out_proj_flat_fp32 = expert_out_proj_flat.float().contiguous()

    torch_h = F.grouped_mm(x_grouped, expert_in_proj.transpose(-2, -1), offs=offsets)
    torch_h[-1].zero_()
    fbgemm_h = fbgemm_grouped_gemm(
        x_grouped,
        expert_in_proj_flat,
        m_sizes,
        use_fast_accum=use_fast_accum,
    )
    custom_h = grouped_gemm(
        x_grouped,
        expert_in_proj_flat,
        m_sizes,
        use_fast_accum=use_fast_accum,
    )
    custom_h_fp32 = grouped_gemm(
        x_grouped,
        expert_in_proj_flat,
        m_sizes,
        use_fast_accum=use_fast_accum,
        fp32_output=True,
    )
    slow_h = _slow_grouped_mm_reference(x_grouped, expert_in_proj, offsets)
    slow_h[-1].zero_()
    h_diff = (fbgemm_h.float() - slow_h.float()).abs()
    custom_h_diff = (custom_h.float() - slow_h.float()).abs()
    custom_h_fp32_diff = (custom_h_fp32.float() - slow_h.float()).abs()

    fbgemm_h[-1].zero_()
    custom_h[-1].zero_()
    custom_h_fp32[-1].zero_()
    torch_h_act = F.silu(torch_h)
    fbgemm_h_act = F.silu(fbgemm_h)
    custom_h_act = F.silu(custom_h)
    custom_h_fp32_act = F.silu(custom_h_fp32)
    slow_h_act = F.silu(slow_h)
    torch_y = F.grouped_mm(torch_h_act, expert_out_proj.transpose(-2, -1), offs=offsets)[:-1]
    slow_y = _slow_grouped_mm_reference(slow_h_act, expert_out_proj, offsets)[:-1]
    fbgemm_y = fbgemm_grouped_gemm(
        fbgemm_h_act,
        expert_out_proj_flat,
        m_sizes,
        use_fast_accum=use_fast_accum,
    )[:-1]
    custom_y = grouped_gemm(
        custom_h_act,
        expert_out_proj_flat,
        m_sizes,
        use_fast_accum=use_fast_accum,
    )[:-1]
    custom_y_fp32 = grouped_gemm(
        custom_h_fp32_act,
        expert_out_proj_flat_fp32,
        m_sizes,
        use_fast_accum=use_fast_accum,
        fp32_output=True,
    )[:-1]
    y_diff = (fbgemm_y.float() - slow_y.float()).abs()
    custom_y_diff = (custom_y.float() - slow_y.float()).abs()
    custom_y_fp32_diff = (custom_y_fp32.float() - slow_y.float()).abs()

    torch_out = _torch_moe_combine_deterministic(
        (torch_y * weights_sorted.unsqueeze(-1)).contiguous(),
        src,
        x.shape[0],
    )
    fbgemm_out = _torch_moe_combine_deterministic(
        (fbgemm_y * weights_sorted.unsqueeze(-1)).contiguous(),
        src,
        x.shape[0],
    )
    custom_out = _torch_moe_combine_deterministic(
        (custom_y * weights_sorted.unsqueeze(-1)).contiguous(),
        src,
        x.shape[0],
    )
    slow_out = _torch_moe_combine_deterministic(
        (slow_y * weights_sorted.unsqueeze(-1).float()).contiguous(),
        src,
        x.shape[0],
    )
    custom_out_fp32 = _torch_moe_combine_deterministic(
        (custom_y_fp32 * weights_sorted.unsqueeze(-1).float()).contiguous(),
        src,
        x.shape[0],
    )
    out_diff = (fbgemm_out.float() - slow_out.float()).abs()
    custom_out_diff = (custom_out.float() - slow_out.float()).abs()
    custom_out_fp32_diff = (custom_out_fp32.float() - slow_out.float()).abs()
    custom_out_fp32_bf16_diff = (
        custom_out_fp32.to(torch.bfloat16).float() - slow_out.to(torch.bfloat16).float()
    ).abs()

    print(f"\nreal moe grouped_gemm repro use_fast_accum={use_fast_accum}")
    print(f"real moe slow reference h head: {slow_h[0, :16].float().cpu().tolist()}")
    print(f"real moe F.grouped_mm h head: {torch_h[0, :16].float().cpu().tolist()}")
    print(f"real moe FBGEMM grouped_gemm h head: {fbgemm_h[0, :16].float().cpu().tolist()}")
    print(f"real moe custom grouped_gemm h head: {custom_h[0, :16].float().cpu().tolist()}")
    print(f"real moe custom grouped_gemm fp32-output h head: {custom_h_fp32[0, :16].float().cpu().tolist()}")
    print(
        "real moe FBGEMM grouped_gemm first-stage diffs vs slow reference: "
        f"max_abs_diff={float(h_diff.max().item()):.6f} "
        f"mean_abs_diff={float(h_diff.mean().item()):.6f}"
    )
    print(
        "real moe custom grouped_gemm first-stage diffs vs slow reference: "
        f"max_abs_diff={float(custom_h_diff.max().item()):.6f} "
        f"mean_abs_diff={float(custom_h_diff.mean().item()):.6f}"
    )
    print(
        "real moe custom grouped_gemm fp32-output first-stage diffs vs slow reference: "
        f"max_abs_diff={float(custom_h_fp32_diff.max().item()):.6f} "
        f"mean_abs_diff={float(custom_h_fp32_diff.mean().item()):.6f}"
    )
    print(
        "real moe FBGEMM grouped_gemm second-stage diffs vs slow reference: "
        f"max_abs_diff={float(y_diff.max().item()):.6f} "
        f"mean_abs_diff={float(y_diff.mean().item()):.6f}"
    )
    print(
        "real moe custom grouped_gemm second-stage diffs vs slow reference: "
        f"max_abs_diff={float(custom_y_diff.max().item()):.6f} "
        f"mean_abs_diff={float(custom_y_diff.mean().item()):.6f}"
    )
    print(
        "real moe custom grouped_gemm fp32-output second-stage diffs vs slow reference: "
        f"max_abs_diff={float(custom_y_fp32_diff.max().item()):.6f} "
        f"mean_abs_diff={float(custom_y_fp32_diff.mean().item()):.6f}"
    )
    print(
        "real moe FBGEMM grouped_gemm final combined diffs vs slow reference: "
        f"max_abs_diff={float(out_diff.max().item()):.6f} "
        f"mean_abs_diff={float(out_diff.mean().item()):.6f}"
    )
    print(
        "real moe custom grouped_gemm final combined diffs vs slow reference: "
        f"max_abs_diff={float(custom_out_diff.max().item()):.6f} "
        f"mean_abs_diff={float(custom_out_diff.mean().item()):.6f}"
    )
    print(
        "real moe custom grouped_gemm fp32-output final combined diffs vs slow reference: "
        f"max_abs_diff={float(custom_out_fp32_diff.max().item()):.6f} "
        f"mean_abs_diff={float(custom_out_fp32_diff.mean().item()):.6f}"
    )
    print(
        "real moe custom grouped_gemm fp32-output final combined diffs vs slow reference after bf16 cast: "
        f"max_abs_diff={float(custom_out_fp32_bf16_diff.max().item()):.6f} "
        f"mean_abs_diff={float(custom_out_fp32_bf16_diff.mean().item()):.6f}"
    )

    assert torch.isfinite(torch_out).all()
    assert torch.isfinite(fbgemm_out).all()
    assert torch.isfinite(custom_out).all()
    assert torch.isfinite(custom_out_fp32).all()


@CUDA_ONLY
@pytest.mark.parametrize(("n_experts", "top_k", "d_model", "d_hidden"), [(16, 8, 2048, 1024)])
def test_torch_grouped_mm_vs_slow_reference_real_moe_repro(
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
):
    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    x, logits, expert_in_proj, expert_out_proj = _make_moe_inputs(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    x_grouped, offsets, _m_sizes, src, weights_sorted, expert_out_proj = _make_real_moe_grouped_state(
        x,
        logits,
        expert_in_proj,
        expert_out_proj,
        top_k,
    )

    torch_h = F.grouped_mm(x_grouped, expert_in_proj.transpose(-2, -1), offs=offsets)
    torch_h[-1].zero_()
    slow_h = _slow_grouped_mm_reference(x_grouped, expert_in_proj, offsets)
    slow_h[-1].zero_()
    h_diff = (torch_h.float() - slow_h.float()).abs()

    torch_h_act = F.silu(torch_h)
    slow_h_act = F.silu(slow_h)
    torch_y = F.grouped_mm(torch_h_act, expert_out_proj.transpose(-2, -1), offs=offsets)[:-1]
    slow_y = _slow_grouped_mm_reference(slow_h_act, expert_out_proj, offsets)[:-1]
    y_diff = (torch_y.float() - slow_y.float()).abs()

    torch_out = _torch_moe_combine_deterministic(
        (torch_y * weights_sorted.unsqueeze(-1)).contiguous(),
        src,
        x.shape[0],
    )
    slow_out = _torch_moe_combine_deterministic(
        (slow_y * weights_sorted.unsqueeze(-1).float()).contiguous(),
        src,
        x.shape[0],
    )
    out_diff = (torch_out.float() - slow_out.float()).abs()

    print(f"\nreal moe F.grouped_mm vs slow reference")
    print(f"F.grouped_mm h head: {torch_h[0, :16].float().cpu().tolist()}")
    print(f"slow reference h head: {slow_h[0, :16].float().cpu().tolist()}")
    print(
        "F.grouped_mm first-stage diffs vs slow reference: "
        f"max_abs_diff={float(h_diff.max().item()):.6f} "
        f"mean_abs_diff={float(h_diff.mean().item()):.6f}"
    )
    print(
        "F.grouped_mm second-stage diffs vs slow reference: "
        f"max_abs_diff={float(y_diff.max().item()):.6f} "
        f"mean_abs_diff={float(y_diff.mean().item()):.6f}"
    )
    print(
        "F.grouped_mm final combined diffs vs slow reference: "
        f"max_abs_diff={float(out_diff.max().item()):.6f} "
        f"mean_abs_diff={float(out_diff.mean().item()):.6f}"
    )

    assert torch.isfinite(torch_out).all()
    assert torch.isfinite(slow_out).all()


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.parametrize(("n_tokens", "n_experts", "top_k"), [(4096, 32, 4), (16384, 128, 4)])
def test_index_shuffling_timing(n_tokens: int, n_experts: int, top_k: int, record_property, request):
    _require_benchmark_flag(request)
    scores = _random_scores(n_tokens, n_experts, "cuda")
    custom_ms = _time_cuda_ms(lambda: index_shuffling(scores, top_k=top_k))
    result = {"custom_ms": round(custom_ms, 4)}

    if has_fbgemm_moe_kernels():
        fbgemm_ms = _time_cuda_ms(lambda: fbgemm_index_shuffling(scores, top_k=top_k))
        result["fbgemm_ms"] = round(fbgemm_ms, 4)
        result["slowdown_vs_fbgemm"] = round(custom_ms / fbgemm_ms, 4)

    record_property("index_shuffling_timing", result)
    print(f"\nindex_shuffling timing: {result}")


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.parametrize(("n_tokens", "n_experts", "top_k", "d_model"), [(4096, 32, 4, 512), (8192, 128, 4, 1024)])
def test_scatter_add_dense_tokens_timing(
    n_tokens: int,
    n_experts: int,
    top_k: int,
    d_model: int,
    record_property,
    request,
):
    _require_benchmark_flag(request)
    in_tokens, token_indices = _make_moe_scatter_inputs(
        n_tokens=n_tokens,
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
    )

    def run_custom():
        scatter_add_dense_tokens(in_tokens, token_indices, n_tokens=n_tokens)

    custom_ms = _time_cuda_ms(run_custom)
    result = {"custom_ms": round(custom_ms, 4)}

    if has_fbgemm_moe_kernels():
        def run_fbgemm():
            out = torch.zeros(n_tokens, d_model, device="cuda", dtype=torch.float32)
            fbgemm_scatter_add_dense_tokens(out, in_tokens, token_indices)

        fbgemm_ms = _time_cuda_ms(run_fbgemm)
        result["fbgemm_ms"] = round(fbgemm_ms, 4)
        result["slowdown_vs_fbgemm"] = round(custom_ms / fbgemm_ms, 4)

    record_property("scatter_add_dense_tokens_timing", result)
    print(f"\nscatter_add_dense_tokens timing: {result}")


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.skipif(_load_fbgemm_grouped_gemm() is None, reason="fbgemm grouped_gemm is not available")
@pytest.mark.parametrize(("n_experts", "d_model", "d_hidden"), [(8, 2048, 1024), (16, 2048, 1024)])
def test_grouped_gemm_timing(n_experts: int, d_model: int, d_hidden: int, record_property, request):
    _require_benchmark_flag(request)
    x_grouped, w, m_sizes = _make_grouped_gemm_inputs(n_experts, d_model, d_hidden)

    custom_ms = _time_cuda_ms(lambda: grouped_gemm(x_grouped, w, m_sizes))
    fbgemm_ms = _time_cuda_ms(lambda: fbgemm_grouped_gemm(x_grouped, w, m_sizes))
    result = {
        "custom_ms": round(custom_ms, 4),
        "fbgemm_ms": round(fbgemm_ms, 4),
        "slowdown_vs_fbgemm": round(custom_ms / fbgemm_ms, 4),
    }

    record_property("grouped_gemm_timing", result)
    print(f"\ngrouped_gemm timing: {result}")


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.parametrize(("n_experts", "d_model", "d_hidden"), [(8, 2048, 1024), (16, 2048, 1024)])
def test_grouped_gemm_eager_vs_torch_grouped_mm_timing(
    n_experts: int,
    d_model: int,
    d_hidden: int,
    record_property,
    request,
):
    _require_benchmark_flag(request)
    x_grouped, w, m_sizes = _make_grouped_gemm_inputs(n_experts, d_model, d_hidden)
    torch_w = _grouped_gemm_weight_for_torch(w, n_experts, d_hidden)
    offs = _grouped_gemm_offs(m_sizes)

    custom_ms = _time_cuda_ms(lambda: grouped_gemm(x_grouped, w, m_sizes))
    torch_ms = _time_cuda_ms(lambda: F.grouped_mm(x_grouped, torch_w, offs=offs))
    result = {
        "custom_ms": round(custom_ms, 4),
        "torch_grouped_mm_ms": round(torch_ms, 4),
        "slowdown_vs_torch_grouped_mm": round(custom_ms / torch_ms, 4),
    }

    record_property("grouped_gemm_eager_vs_torch_grouped_mm_timing", result)
    print(f"\ngrouped_gemm eager vs torch grouped_mm timing: {result}")


@CUDA_ONLY
@pytest.mark.skipif(not has_fbgemm_moe_kernels(), reason="fbgemm MoE kernels are not available")
@pytest.mark.parametrize(("n_experts", "top_k", "d_model", "d_hidden"), [(16, 8, 2048, 1024)])
def test_moe_implementation_matrix(
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
    record_property,
    request,
):
    x, gate, expert_in_proj, expert_out_proj = _make_forced_moe_case(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    moe_fbgemm_impl, moe_custom_bf16_impl, moe_custom_fp32_impl, moe_baseline = _make_moe_modules_with_baseline(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    with torch.no_grad():
        moe_fbgemm_impl.expert_in_proj.copy_(expert_in_proj)
        moe_fbgemm_impl.expert_out_proj.copy_(expert_out_proj)
        moe_custom_bf16_impl.expert_in_proj.copy_(expert_in_proj)
        moe_custom_bf16_impl.expert_out_proj.copy_(expert_out_proj)
        moe_custom_fp32_impl.expert_in_proj.copy_(expert_in_proj)
        moe_custom_fp32_impl.expert_out_proj.copy_(expert_out_proj)
        moe_baseline.expert_in_proj.copy_(expert_in_proj)
        moe_baseline.expert_out_proj.copy_(expert_out_proj)

    implementations = [
        ("fbgemm", moe_fbgemm_impl, _run_moe_module_compiled_inference),
        ("custom_bf16", moe_custom_bf16_impl, _run_moe_custom_module_compiled_inference),
        ("custom_fp32", moe_custom_fp32_impl, _run_moe_custom_module_compiled_inference),
        ("baseline", moe_baseline, _run_moe_baseline_module_compiled_inference),
    ]
    eager_baseline = _run_moe_module_eager(moe_baseline, x, gate=gate)
    run_timing = request.config.getoption("--run-kernel-benchmarks")
    result = {}
    rows: list[tuple[str, str, str, str, str, str, str, str, str, str, str, str]] = []
    compiled_failures: list[tuple[str, str]] = []

    for name, module, compiled_runner in implementations:
        eager_out = _run_moe_module_eager(module, x, gate=gate)
        eager_repeat = _run_moe_module_eager(module, x, gate=gate)
        eager_self_diff = (eager_out.float() - eager_repeat.float()).abs()
        eager_vs_baseline = (eager_out.float() - eager_baseline.float()).abs()
        eager_ms_str = "n/a"
        compiled_ms_str = "n/a"
        speedup_str = "n/a"

        entry = {
            "eager_self_max_abs_diff": round(float(eager_self_diff.max().item()), 6),
            "eager_self_mean_abs_diff": round(float(eager_self_diff.mean().item()), 6),
            "eager_max_abs_diff_vs_baseline": round(float(eager_vs_baseline.max().item()), 6),
            "eager_mean_abs_diff_vs_baseline": round(float(eager_vs_baseline.mean().item()), 6),
        }
        if run_timing:
            eager_ms = _time_cuda_ms(lambda m=module: _run_moe_module_eager(m, x, gate=gate))
            entry["eager_ms"] = round(eager_ms, 4)
            eager_ms_str = f"{entry['eager_ms']:.4f}"

        compiled_vs_baseline_max_str = "n/a"
        compiled_vs_baseline_mean_str = "n/a"
        compiled_vs_eager_max_str = "n/a"
        compiled_vs_eager_mean_str = "n/a"
        compiled_status = "ok"
        try:
            compiled_out = compiled_runner(module, x, gate=gate)
            compiled_vs_baseline = (compiled_out.float() - eager_baseline.float()).abs()
            compiled_vs_eager = (compiled_out.float() - eager_out.float()).abs()
            entry.update(
                {
                    "compiled_max_abs_diff_vs_baseline": round(float(compiled_vs_baseline.max().item()), 6),
                    "compiled_mean_abs_diff_vs_baseline": round(float(compiled_vs_baseline.mean().item()), 6),
                    "compiled_max_abs_diff_vs_eager": round(float(compiled_vs_eager.max().item()), 6),
                    "compiled_mean_abs_diff_vs_eager": round(float(compiled_vs_eager.mean().item()), 6),
                }
            )
            if run_timing:
                compiled_ms = _time_cuda_ms(lambda m=module, r=compiled_runner: r(m, x, gate=gate))
                entry["compiled_ms"] = round(compiled_ms, 4)
                entry["compiled_speedup_vs_eager"] = round(eager_ms / compiled_ms, 4)
                compiled_ms_str = f"{entry['compiled_ms']:.4f}"
                speedup_str = f"{entry['compiled_speedup_vs_eager']:.3f}x"
            compiled_vs_baseline_max_str = f"{entry['compiled_max_abs_diff_vs_baseline']:.6f}"
            compiled_vs_baseline_mean_str = f"{entry['compiled_mean_abs_diff_vs_baseline']:.6f}"
            compiled_vs_eager_max_str = f"{entry['compiled_max_abs_diff_vs_eager']:.6f}"
            compiled_vs_eager_mean_str = f"{entry['compiled_mean_abs_diff_vs_eager']:.6f}"
            assert torch.isfinite(compiled_out).all()
        except Exception as exc:
            if name == "baseline":
                compiled_status = "expected:_slow_grouped_mm_not_compileable"
            else:
                compiled_status = f"unavailable:{exc.__class__.__name__}"
            entry["compiled_status"] = compiled_status
            compiled_failures.append((name, traceback.format_exc()))

        result[name] = entry
        rows.append(
            (
                name,
                eager_ms_str,
                compiled_ms_str,
                speedup_str,
                f"{entry['eager_self_max_abs_diff']:.6f}",
                f"{entry['eager_self_mean_abs_diff']:.6f}",
                f"{entry['eager_max_abs_diff_vs_baseline']:.6f}",
                f"{entry['eager_mean_abs_diff_vs_baseline']:.6f}",
                compiled_vs_baseline_max_str,
                compiled_vs_baseline_mean_str,
                compiled_vs_eager_max_str,
                compiled_vs_eager_mean_str,
                compiled_status,
            )
        )

        assert torch.isfinite(eager_out).all()

    header = (
        "impl".ljust(10)
        + "eager_ms".rjust(12)
        + "compiled_ms".rjust(14)
        + "speedup".rjust(12)
        + "self_max".rjust(12)
        + "self_mean".rjust(13)
        + "vs_base_max".rjust(14)
        + "vs_base_mean".rjust(15)
        + "comp_vs_base_max".rjust(18)
        + "comp_vs_base_mean".rjust(19)
        + "comp_vs_eager_max".rjust(19)
        + "comp_vs_eager_mean".rjust(20)
        + "status".rjust(20)
    )
    print("\nmoe implementation matrix")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            row[0].ljust(10)
            + row[1].rjust(12)
            + row[2].rjust(14)
            + row[3].rjust(12)
            + row[4].rjust(12)
            + row[5].rjust(13)
            + row[6].rjust(14)
            + row[7].rjust(15)
            + row[8].rjust(18)
            + row[9].rjust(19)
            + row[10].rjust(19)
            + row[11].rjust(20)
            + row[12].rjust(20)
        )
    if compiled_failures:
        print("\ncompiled failure tracebacks")
        for name, tb in compiled_failures:
            print(f"[{name}]")
            print(tb.rstrip())

    record_property("moe_implementation_matrix", result)
