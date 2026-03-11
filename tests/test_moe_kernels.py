from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
import statistics
import sys
import time

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
import src.model.world_model as world_model_module
from src.model.world_model import MoE, MoEWithoutFBGEMM


CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for MoE kernel tests")
LOCAL_KERNELS_ONLY = pytest.mark.skipif(not HAS_LOCAL_KERNELS, reason="local kernels package is not present on this branch")

COMPILE_OPTIONS = {
    "max_autotune": True,
    "coordinate_descent_tuning": True,
    "triton.cudagraphs": True,
}

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
) -> torch.Tensor:
    op = _load_fbgemm_grouped_gemm()
    if op is None:
        raise RuntimeError("fbgemm grouped_gemm is not available in this environment")
    return op(x, w, m_sizes, use_fast_accum=use_fast_accum)


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
    cfg = _make_moe_config(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
        gated_linear=gated_linear,
    )
    moe_fbgemm = MoE(cfg).eval().to(device="cuda", dtype=dtype)
    moe_fallback = MoEWithoutFBGEMM(cfg).eval().to(device="cuda", dtype=dtype)
    moe_fallback.load_state_dict(moe_fbgemm.state_dict())
    return moe_fbgemm, moe_fallback


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
    out = torch.zeros_like(x)
    scatter_add_dense_tokens(out, (y_grouped * weights.unsqueeze(-1)).contiguous(), src)
    return out


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
    out = torch.zeros_like(x)
    fbgemm_scatter_add_dense_tokens(out, (y_grouped * weights.unsqueeze(-1)).contiguous(), src)
    return out


def _run_loaded_moe_eager(
    x: torch.Tensor,
    logits: torch.Tensor,
    expert_in_proj: torch.Tensor,
    expert_out_proj: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    logits_fp32 = logits.float()
    token_counts, expert_sorted, src = world_model_module.fbgemm_index_shuffling(logits_fp32, top_k=top_k)
    offs = token_counts[: expert_in_proj.shape[0]].cumsum(0).to(torch.int32)
    src = src.to(torch.long)
    expert_sorted = expert_sorted.to(torch.long)
    log_z = logits_fp32.logsumexp(-1)
    weights = (logits_fp32[src, expert_sorted] - log_z[src]).exp().to(x.dtype)

    x_grouped = x.index_select(0, torch.cat((src, src[:1]), dim=0))
    h = F.grouped_mm(x_grouped, expert_in_proj.transpose(-2, -1), offs=offs)
    h = F.silu(h)
    y_grouped = F.grouped_mm(h, expert_out_proj.transpose(-2, -1), offs=offs)[:-1]
    out = torch.zeros_like(x)
    torch.ops.fbgemm.scatter_add_dense_tokens(out, (y_grouped * weights.unsqueeze(-1)).contiguous(), src)
    return out


@torch.compile(fullgraph=True, dynamic=False, options=COMPILE_OPTIONS)
def _run_moe_module_compiled(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    return moe(x)


def _run_moe_module_compiled_inference(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return _run_moe_module_compiled(moe, x)

def _require_benchmark_flag(request) -> None:
    if not request.config.getoption("--run-kernel-benchmarks"):
        pytest.skip("pass --run-kernel-benchmarks to run kernel timing tests")


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
@pytest.mark.parametrize(("n_out", "n_in", "width"), [(1024, 2048, 256), (2048, 4096, 512)])
def test_scatter_add_dense_tokens_matches_reference(n_out: int, n_in: int, width: int):
    in_tokens = torch.randn(n_in, width, device="cuda", dtype=torch.bfloat16)
    token_indices = torch.randint(0, n_out, (n_in,), device="cuda", dtype=torch.int64)

    expected = torch.zeros(n_out, width, device="cuda", dtype=torch.float32)
    actual = torch.zeros(n_out, width, device="cuda", dtype=in_tokens.dtype)

    expected.index_add_(0, token_indices, in_tokens.float())
    scatter_add_dense_tokens(actual, in_tokens, token_indices)

    torch.testing.assert_close(actual.float(), expected, atol=5e-2, rtol=5e-2)


@CUDA_ONLY
@LOCAL_KERNELS_ONLY
@pytest.mark.skipif(not has_fbgemm_moe_kernels(), reason="fbgemm MoE kernels are not available")
@pytest.mark.parametrize(("n_out", "n_in", "width"), [(1024, 2048, 256), (2048, 4096, 512)])
def test_scatter_add_dense_tokens_matches_fbgemm(n_out: int, n_in: int, width: int):
    in_tokens = torch.randn(n_in, width, device="cuda", dtype=torch.bfloat16)
    token_indices = torch.randint(0, n_out, (n_in,), device="cuda", dtype=torch.int64)

    ref = torch.zeros(n_out, width, device="cuda", dtype=in_tokens.dtype)
    actual = torch.zeros_like(ref)

    fbgemm_scatter_add_dense_tokens(ref, in_tokens.contiguous(), token_indices.contiguous())
    scatter_add_dense_tokens(actual, in_tokens, token_indices)
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
@pytest.mark.parametrize(("n_out", "n_in", "width"), [(4096, 16384, 512), (8192, 32768, 1024)])
def test_scatter_add_dense_tokens_timing(n_out: int, n_in: int, width: int, record_property, request):
    _require_benchmark_flag(request)
    in_tokens = torch.randn(n_in, width, device="cuda", dtype=torch.bfloat16).contiguous()
    token_indices = torch.randint(0, n_out, (n_in,), device="cuda", dtype=torch.int64).contiguous()

    def run_custom():
        out = torch.zeros(n_out, width, device="cuda", dtype=in_tokens.dtype)
        scatter_add_dense_tokens(out, in_tokens, token_indices)

    custom_ms = _time_cuda_ms(run_custom)
    result = {"custom_ms": round(custom_ms, 4)}

    if has_fbgemm_moe_kernels():
        def run_fbgemm():
            out = torch.zeros(n_out, width, device="cuda", dtype=in_tokens.dtype)
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
@LOCAL_KERNELS_ONLY
@pytest.mark.skipif(not has_fbgemm_moe_kernels(), reason="fbgemm MoE kernels are not available")
@pytest.mark.parametrize(("n_experts", "top_k", "d_model", "d_hidden"), [(16, 8, 2048, 1024)])
def test_moe_eager_timing_vs_fbgemm(
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
    record_property,
    request,
):
    _require_benchmark_flag(request)
    x, logits, expert_in_proj, expert_out_proj = _make_moe_inputs(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )

    custom_out = _run_custom_moe_eager(x, logits, expert_in_proj, expert_out_proj, top_k)
    moe_fbgemm, moe_fallback = _make_moe_modules(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    with torch.no_grad():
        moe_fbgemm.expert_in_proj.copy_(expert_in_proj)
        moe_fbgemm.expert_out_proj.copy_(expert_out_proj)
        moe_fallback.expert_in_proj.copy_(expert_in_proj)
        moe_fallback.expert_out_proj.copy_(expert_out_proj)
    fbgemm_out = _run_moe_module_eager(moe_fbgemm, x.unsqueeze(0), gate=logits.unsqueeze(0)).squeeze(0)
    diff = (custom_out.float() - fbgemm_out.float()).abs()

    custom_ms = _time_cuda_ms(lambda: _run_custom_moe_eager(x, logits, expert_in_proj, expert_out_proj, top_k))
    fbgemm_ms = _time_cuda_ms(lambda: _run_moe_module_eager(moe_fbgemm, x.unsqueeze(0), gate=logits.unsqueeze(0)))
    result = {
        "custom_ms": round(custom_ms, 4),
        "fbgemm_ms": round(fbgemm_ms, 4),
        "slowdown_vs_fbgemm": round(custom_ms / fbgemm_ms, 4),
        "max_abs_diff": round(float(diff.max().item()), 6),
        "mean_abs_diff": round(float(diff.mean().item()), 6),
    }

    record_property("moe_eager_timing_vs_fbgemm", result)
    print(f"\nmoe eager timing vs fbgemm: {result}")


@CUDA_ONLY
@pytest.mark.skipif(not has_fbgemm_moe_kernels(), reason="fbgemm MoE kernels are not available")
@pytest.mark.parametrize(("n_experts", "top_k", "d_model", "d_hidden"), [(16, 8, 2048, 1024)])
def test_fbgemm_moe_module_eager_matches_handwritten_eager(
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
):
    moe_fbgemm, _ = _make_moe_modules(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    x, logits, expert_in_proj, expert_out_proj = _make_moe_inputs(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    with torch.no_grad():
        moe_fbgemm.expert_in_proj.copy_(expert_in_proj)
        moe_fbgemm.expert_out_proj.copy_(expert_out_proj)

    module_out = _run_moe_module_eager(moe_fbgemm, x.unsqueeze(0), gate=logits.unsqueeze(0)).squeeze(0)
    module_out_repeat = _run_moe_module_eager(moe_fbgemm, x.unsqueeze(0), gate=logits.unsqueeze(0)).squeeze(0)
    loaded_out = _run_loaded_moe_eager(x, logits, expert_in_proj, expert_out_proj, top_k)
    loaded_out_repeat = _run_loaded_moe_eager(x, logits, expert_in_proj, expert_out_proj, top_k)
    fbgemm_out = _run_fbgemm_moe_eager(x, logits, expert_in_proj, expert_out_proj, top_k)
    module_self_diff = (module_out.float() - module_out_repeat.float()).abs()
    loaded_self_diff = (loaded_out.float() - loaded_out_repeat.float()).abs()
    loaded_diff = (module_out.float() - loaded_out.float()).abs()
    fbgemm_diff = (module_out.float() - fbgemm_out.float()).abs()

    print(
        f"\nworld_model symbols: "
        f"fbgemm_index_shuffling={world_model_module.fbgemm_index_shuffling!r} "
        f"custom_index_shuffling={world_model_module.custom_index_shuffling!r} "
        f"scatter_add={torch.ops.fbgemm.scatter_add_dense_tokens!r}"
    )
    print(
        f"module self diffs: "
        f"max_abs_diff={float(module_self_diff.max().item()):.6f} "
        f"mean_abs_diff={float(module_self_diff.mean().item()):.6f}"
    )
    print(
        f"loaded handwritten self diffs: "
        f"max_abs_diff={float(loaded_self_diff.max().item()):.6f} "
        f"mean_abs_diff={float(loaded_self_diff.mean().item()):.6f}"
    )
    print(
        f"module vs loaded handwritten diffs: "
        f"max_abs_diff={float(loaded_diff.max().item()):.6f} "
        f"mean_abs_diff={float(loaded_diff.mean().item()):.6f}"
    )
    print(
        f"module vs old fbgemm handwritten diffs: "
        f"max_abs_diff={float(fbgemm_diff.max().item()):.6f} "
        f"mean_abs_diff={float(fbgemm_diff.mean().item()):.6f}"
    )

    max_self_max = max(
        float(module_self_diff.max().item()),
        float(loaded_self_diff.max().item()),
    )
    max_self_mean = max(
        float(module_self_diff.mean().item()),
        float(loaded_self_diff.mean().item()),
    )
    assert float(loaded_diff.max().item()) <= max_self_max + 8.0
    assert float(loaded_diff.mean().item()) <= max_self_mean + 0.05


@CUDA_ONLY
# @pytest.mark.skipif(not has_fbgemm_moe_kernels(), reason="fbgemm MoE kernels are not available")
@pytest.mark.parametrize(("n_experts", "top_k", "d_model", "d_hidden"), [(16, 8, 2048, 1024)])
def test_fbgemm_moe_eager_vs_compiled_timing(
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
    record_property,
    request,
):
    _require_benchmark_flag(request)
    moe_fbgemm, _ = _make_moe_modules(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    x = torch.randn((1, 512, d_model), device="cuda", dtype=torch.bfloat16)

    eager_out = _run_moe_module_eager(moe_fbgemm, x)
    compiled_out = _run_moe_module_compiled_inference(moe_fbgemm, x)
    diff = (compiled_out.float() - eager_out.float()).abs()

    eager_ms = _time_cuda_ms(lambda: _run_moe_module_eager(moe_fbgemm, x))
    compiled_ms = _time_cuda_ms(lambda: _run_moe_module_compiled_inference(moe_fbgemm, x))
    result = {
        "eager_ms": round(eager_ms, 4),
        "compiled_ms": round(compiled_ms, 4),
        "speedup_vs_eager": round(eager_ms / compiled_ms, 4),
        "max_abs_diff": round(float(diff.max().item()), 6),
        "mean_abs_diff": round(float(diff.mean().item()), 6),
    }

    record_property("fbgemm_moe_eager_vs_compiled_timing", result)
    print(f"\nfbgemm moe eager vs compiled timing: {result}")


@CUDA_ONLY
@pytest.mark.skipif(not has_fbgemm_moe_kernels(), reason="fbgemm MoE kernels are not available")
@pytest.mark.parametrize(("n_experts", "top_k", "d_model", "d_hidden"), [(16, 8, 2048, 1024)])
def test_moe_module_forward_matches_fbgemm_eager_and_compiled(
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
):
    moe_fbgemm, moe_fallback = _make_moe_modules(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    x = torch.randn((1, 512, d_model), device="cuda", dtype=torch.bfloat16)

    eager_fbgemm = _run_moe_module_eager(moe_fbgemm, x)
    eager_fallback = _run_moe_module_eager(moe_fallback, x)
    compiled_fbgemm = _run_moe_module_compiled_inference(moe_fbgemm, x)

    fallback_diff = (eager_fallback.float() - eager_fbgemm.float()).abs()
    compiled_diff = (compiled_fbgemm.float() - eager_fbgemm.float()).abs()

    print(
        f"\nmoe module diffs: "
        f"fallback_max_abs_diff={float(fallback_diff.max().item()):.6f} "
        f"fallback_mean_abs_diff={float(fallback_diff.mean().item()):.6f} "
        f"compiled_max_abs_diff={float(compiled_diff.max().item()):.6f} "
        f"compiled_mean_abs_diff={float(compiled_diff.mean().item()):.6f}"
    )

    assert torch.isfinite(eager_fallback).all()
    assert torch.isfinite(compiled_fbgemm).all()


@CUDA_ONLY
@pytest.mark.skipif(not has_fbgemm_moe_kernels(), reason="fbgemm MoE kernels are not available")
@pytest.mark.parametrize(("n_experts", "top_k", "d_model", "d_hidden"), [(16, 8, 2048, 1024)])
def test_moe_module_timing_vs_fbgemm_eager_and_compiled(
    n_experts: int,
    top_k: int,
    d_model: int,
    d_hidden: int,
    record_property,
    request,
):
    _require_benchmark_flag(request)
    moe_fbgemm, moe_fallback = _make_moe_modules(
        n_experts=n_experts,
        top_k=top_k,
        d_model=d_model,
        d_hidden=d_hidden,
    )
    x = torch.randn((1, 512, d_model), device="cuda", dtype=torch.bfloat16)

    eager_fbgemm = _run_moe_module_eager(moe_fbgemm, x)
    eager_fallback = _run_moe_module_eager(moe_fallback, x)
    compiled_fbgemm = _run_moe_module_compiled_inference(moe_fbgemm, x)
    fallback_diff = (eager_fallback.float() - eager_fbgemm.float()).abs()
    compiled_diff = (compiled_fbgemm.float() - eager_fbgemm.float()).abs()

    fallback_ms = _time_cuda_ms(lambda: _run_moe_module_eager(moe_fallback, x))
    eager_fbgemm_ms = _time_cuda_ms(lambda: _run_moe_module_eager(moe_fbgemm, x))
    compiled_fbgemm_ms = _time_cuda_ms(lambda: _run_moe_module_compiled_inference(moe_fbgemm, x))
    result = {
        "fallback_ms": round(fallback_ms, 4),
        "eager_fbgemm_ms": round(eager_fbgemm_ms, 4),
        "compiled_fbgemm_ms": round(compiled_fbgemm_ms, 4),
        "fallback_slowdown_vs_eager_fbgemm": round(fallback_ms / eager_fbgemm_ms, 4),
        "fallback_slowdown_vs_compiled_fbgemm": round(fallback_ms / compiled_fbgemm_ms, 4),
        "compiled_speedup_vs_eager_fbgemm": round(eager_fbgemm_ms / compiled_fbgemm_ms, 4),
        "fallback_max_abs_diff": round(float(fallback_diff.max().item()), 6),
        "fallback_mean_abs_diff": round(float(fallback_diff.mean().item()), 6),
        "compiled_max_abs_diff": round(float(compiled_diff.max().item()), 6),
        "compiled_mean_abs_diff": round(float(compiled_diff.mean().item()), 6),
    }

    record_property("moe_module_timing_vs_fbgemm_eager_and_compiled", result)
    print(f"\nmoe module timing vs fbgemm eager and compiled: {result}")
