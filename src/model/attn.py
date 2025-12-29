import torch
import einops as eo
from torch import nn
import torch.nn.functional as F

from .rope import OrthoRoPE

from torch.nn.attention.flex_attention import flex_attention


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Attn(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.value_residual = getattr(config, "value_residual", False)
        if self.value_residual:
            self.v_lamb = nn.Parameter(torch.tensor(0.5))

        self.n_heads = config.n_heads
        self.n_kv_heads = getattr(config, "n_kv_heads", config.n_heads)
        self.d_head = config.d_model // self.n_heads
        assert config.d_model % self.n_heads == 0

        self.enable_gqa = self.n_heads != self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = OrthoRoPE(config)

        self.gated_attn = getattr(config, "gated_attn", False)
        if self.gated_attn:
            self.gate_proj = nn.Linear(self.n_heads, self.n_heads, bias=False)  # sparse attn gate
            nn.init.zeros_(self.gate_proj.weight)

    def forward(self, x, pos_ids, v1, bm, kv_cache=None):
        # Q, K, V proj -> QK-norm -> RoPE
        q = eo.rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_heads, d=self.d_head)
        k = eo.rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.n_kv_heads, d=self.d_head)
        v = eo.rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.n_kv_heads, d=self.d_head)

        if self.value_residual:
            v1 = v if v1 is None else v1
            v = torch.lerp(v, v1.view_as(v), self.v_lamb)

        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rope(q, pos_ids), self.rope(k, pos_ids)

        if kv_cache is None:
            torch._assert(bm is not None, "bm must be provided when kv_cache is None")
        else:
            # Update KV-cache and K, V in-place
            k, v, bm = kv_cache.upsert(k, v, pos_ids, self.layer_idx)

        # SDPA -> Attention Gate -> Out Proj
        y = flex_attention(q, k, v, block_mask=bm, enable_gqa=self.enable_gqa)
        if self.gated_attn:
            gates = torch.sigmoid(self.gate_proj(x[..., :self.n_heads]))
            y = y * gates.permute(0, 2, 1).unsqueeze(-1)
        y = eo.rearrange(y, "b h t d -> b t (h d)")
        y = self.out_proj(y)
        return y, v1


class CrossAttention(nn.Module):
    def __init__(self, config, context_dim=None):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.d_head = config.d_model // config.n_heads
        self.inner_dim = context_dim or config.d_model
        assert self.inner_dim % self.d_head == 0
        self.n_heads = self.inner_dim // self.d_head
        self.q_proj = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(context_dim or config.d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(context_dim or config.d_model, self.inner_dim, bias=False)

        self.out_proj = nn.Linear(self.inner_dim, config.d_model, bias=False)
        self.out_proj.weight.detach().zero_()

    def forward(self, x, context, context_pad_mask=None):
        q = eo.rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_heads)
        k = eo.rearrange(self.k_proj(context), "b t (h d) -> b h t d", h=self.n_heads)
        v = eo.rearrange(self.v_proj(context), "b t (h d) -> b h t d", h=self.n_heads)
        q, k = rms_norm(q), rms_norm(k)
        out = flex_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().reshape(x.size(0), x.size(1), -1)
        return self.out_proj(out)
