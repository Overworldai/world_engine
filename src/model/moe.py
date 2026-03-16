import torch.nn.functional as F
import torch
from torch import nn

# These MoE implementations live separately from the main world model so the
# transformer structure stays focused and the numerical/debugging variants can
# be compared side-by-side.

try:
    from fbgemm_gpu.experimental.gen_ai.moe import index_shuffling as fbgemm_index_shuffling
    import fbgemm_gpu.experimental.gen_ai.moe.gather_scatter  # noqa
    print("FBGEMM found for MoE kernels! :)")
except ImportError:
    fbgemm_index_shuffling = None

from ..kernels import grouped_gemm as custom_grouped_gemm
from ..kernels import index_shuffling as custom_index_shuffling
from ..kernels import scatter_add_dense_tokens as custom_scatter_add_dense_tokens


def _slow_grouped_mm(mat_a: torch.Tensor, mat_b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Deterministic grouped matmul oracle used for correctness comparisons.

    This is intentionally slow: it iterates expert-by-expert and materializes
    fp32 outputs so we can isolate routing/scatter bugs from grouped-GEMM math.
    """
    assert mat_a.dim() == 2
    assert mat_b.dim() == 3
    assert offs.dim() == 1

    out = torch.zeros((mat_a.shape[0], mat_b.shape[1]), device=mat_a.device, dtype=torch.float32)
    start = 0
    for expert_idx, end_tensor in enumerate(offs):
        end = int(end_tensor.item())
        if end > start:
            rows = mat_a[start:end].float()
            weight_t = mat_b[expert_idx].transpose(-2, -1).float()
            out[start:end] = rows @ weight_t
        start = end
    return out


class MoETorchBaseline(nn.Module):
    """Pure Torch reference path.

    This path preserves the baseline stable expert/token ordering contract:
    `topk -> flatten -> expert.sort() -> index_copy_ -> reshape -> sum`.
    This is useful as the baseline, but the more efficient implementations
    below do not have intra-expert ordering enforced by expert.sort().
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.moe_top_k
        moe_mlp_ratio = getattr(config, "moe_mlp_ratio", None) or config.mlp_ratio / config.moe_top_k
        d_intermediate = int(config.d_model * moe_mlp_ratio)
        self.router = nn.Linear(config.d_model, config.moe_n_experts, bias=False)
        self.expert_in_proj = nn.Parameter(
            torch.empty(config.moe_n_experts, d_intermediate * (2 if config.gated_linear else 1), config.d_model)
        )
        self.expert_out_proj = nn.Parameter(torch.empty(config.moe_n_experts, config.d_model, d_intermediate))

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise NotImplementedError("inference only")

        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        logits = self.router(x) if gate is None else gate.reshape(-1, gate.size(-1))

        logits_fp32 = logits.float()
        scores, expert = logits.topk(self.top_k, dim=-1, sorted=False)
        weights = (scores.float() - logits_fp32.logsumexp(dim=-1, keepdim=True)).exp().to(x.dtype)

        expert = expert.flatten()
        expert_sorted, sort_idx = expert.sort()
        expert_ids = torch.arange(self.expert_in_proj.size(0), device=expert.device, dtype=expert_sorted.dtype)
        offsets = torch.searchsorted(expert_sorted, expert_ids, right=True).to(torch.int32)

        # Pad the indices instead of copying-and-growing the activation tensor.
        # grouped_mm-style paths require a trailing padded row because offs[-1]
        # must stay strictly less than the sliced dimension length.
        src = sort_idx // self.top_k
        x_grouped = x.index_select(0, torch.cat((src, src[:1]), dim=0))
        h = _slow_grouped_mm(x_grouped, self.expert_in_proj, offsets)
        h[-1].zero_()

        if self.config.gated_linear:
            gate_act, up = h.chunk(2, dim=-1)
            h = F.silu(gate_act) * up
        else:
            h = F.silu(h)

        y_grouped = _slow_grouped_mm(h, self.expert_out_proj, offsets)[:-1]
        y = torch.empty_like(y_grouped).index_copy_(0, sort_idx, y_grouped).view(x.size(0), self.top_k, -1)
        out = (y * weights.unsqueeze(-1)).sum(dim=1)
        return out.to(x.dtype).reshape(orig_shape)


class MoECustomKernels(nn.Module):
    """Local-kernel fallback with an order-invariant downstream contract.

    `custom_index_shuffling` is intentionally not stable inside each expert
    bucket. The rest of this path therefore has to be order-invariant:
    keep `src`, `expert_sorted`, grouped activations, and route weights aligned,
    then reduce back to tokens with scatter-add.
    """

    def __init__(self, config, *, accumulate_in_fp32: bool = False):
        super().__init__()
        self.config = config
        self.top_k = config.moe_top_k
        self.accumulate_in_fp32 = accumulate_in_fp32
        moe_mlp_ratio = getattr(config, "moe_mlp_ratio", None) or config.mlp_ratio / config.moe_top_k
        d_intermediate = int(config.d_model * moe_mlp_ratio)
        self.router = nn.Linear(config.d_model, config.moe_n_experts, bias=False)
        self.expert_in_proj = nn.Parameter(
            torch.empty(config.moe_n_experts, d_intermediate * (2 if config.gated_linear else 1), config.d_model)
        )
        self.expert_out_proj = nn.Parameter(torch.empty(config.moe_n_experts, config.d_model, d_intermediate))

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise NotImplementedError("inference only")

        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])
        logits = self.router(x) if gate is None else gate.reshape(-1, gate.size(-1))

        logits_fp32 = logits.float()
        token_counts, expert_sorted, src = custom_index_shuffling(logits_fp32, top_k=self.top_k)
        expert_sorted = expert_sorted.to(torch.long)
        src = src.to(torch.long)
        expert_counts = token_counts[: self.expert_in_proj.size(0)].to(torch.int32).contiguous()
        log_z = logits_fp32.logsumexp(-1)
        weights = (logits_fp32[src, expert_sorted] - log_z[src]).exp().to(x.dtype)

        # preserve the custom routing order. This is only correct if all
        # per-route tensors stay aligned to the arbitrary order returned by the
        # routing kernel.
        x_grouped = x.index_select(0, torch.cat((src, src[:1]), dim=0))
        # Writing grouped GEMM outputs into fp32 buffers is the main accuracy
        # improvement over the default bf16-materialized path. We keep this
        # configurable so benchmark runs can compare the bf16 and fp32-accum
        # tradeoff directly.
        h = custom_grouped_gemm(
            x_grouped,
            self.expert_in_proj.reshape(-1, self.expert_in_proj.shape[-1]).contiguous(),
            expert_counts,
            fp32_output=self.accumulate_in_fp32,
        )

        if self.config.gated_linear:
            gate_act, up = h.chunk(2, dim=-1)
            h = F.silu(gate_act) * up
        else:
            h = F.silu(h)

        expert_out_proj = self.expert_out_proj.reshape(-1, self.expert_out_proj.shape[-1]).contiguous()
        if self.accumulate_in_fp32:
            expert_out_proj = expert_out_proj.float().contiguous()
        y_grouped = custom_grouped_gemm(
            h,
            expert_out_proj,
            expert_counts,
            fp32_output=self.accumulate_in_fp32,
        )[:-1]
        weighted = y_grouped * weights.unsqueeze(-1)
        # scatter accumulation must stay fp32. lower-precision accumulation is
        # too noisy for MoE, so the kernel itself allocates an fp32 destination.
        out = custom_scatter_add_dense_tokens(weighted, src, n_tokens=x.size(0))
        # recast fp32 out buffer back to input dtype
        return out.to(x.dtype).reshape(orig_shape)


class MoEFBGEMM(nn.Module):
    """BROKEN reference wrapper around the current FBGEMM/PyTorch inference path.

    The eager path is useful for side-by-side accuracy and throughput
    comparisons. The compiled path inherits the known schema issues from the
    FBGEMM's scatter-add operator, so compiled results should be treated with care.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.moe_top_k
        moe_mlp_ratio = getattr(config, "moe_mlp_ratio", None) or (config.mlp_ratio / config.moe_top_k)
        d_int = int(config.d_model * moe_mlp_ratio)

        self.router = nn.Linear(config.d_model, config.moe_n_experts, bias=False)
        self.expert_in_proj = nn.Parameter(
            torch.empty(config.moe_n_experts, d_int * (2 if config.gated_linear else 1), config.d_model)
        )
        self.expert_out_proj = nn.Parameter(torch.empty(config.moe_n_experts, config.d_model, d_int))

    def forward(self, x: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            raise NotImplementedError("inference only")

        orig = x.shape
        x = x.reshape(-1, orig[-1])
        logits = self.router(x) if gate is None else gate.reshape(-1, gate.size(-1))

        logits32 = logits.float()
        if fbgemm_index_shuffling is None:
            raise RuntimeError("FBGEMM index_shuffling is not available in this environment")
        token_counts, expert_sorted, src = fbgemm_index_shuffling(logits32, top_k=self.top_k)
        offs = token_counts[: self.expert_in_proj.size(0)].cumsum(0).to(torch.int32)

        src = src.to(torch.long)
        expert_sorted = expert_sorted.to(torch.long)
        logZ = logits32.logsumexp(-1)
        w = (logits32[src, expert_sorted] - logZ[src]).exp().to(x.dtype)

        # grouped_mm shares the same padded-row constraint as the reference path
        xg = x.index_select(0, torch.cat((src, src[:1]), 0))
        h = F.grouped_mm(xg, self.expert_in_proj.transpose(-2, -1), offs=offs)
        if self.config.gated_linear:
            ga, up = h.chunk(2, -1)
            h = F.silu(ga) * up
        else:
            h = F.silu(h)

        yg = F.grouped_mm(h, self.expert_out_proj.transpose(-2, -1), offs=offs)[:-1]
        # the accumulator must stay fp32 here too, the FBGEMM scatter op writes
        # into the provided destination buffer
        out = torch.zeros_like(x, dtype=torch.float32)
        torch.ops.fbgemm.scatter_add_dense_tokens(out, (yg * w.unsqueeze(-1)).contiguous(), src)
        return out.to(x.dtype).reshape(orig)
