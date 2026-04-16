import torch
import einops as eo
from torch import nn

from torch.nn.attention.flex_attention import flex_attention

from .nn import rms_norm, NoCastModule


class OrthoRoPEAngles(NoCastModule):
    """Functions as a on the fly RoPE angle computer called every fwd pass. Should be setup
    as a module under WordDiT, then each forward pass it computes a shared tuple of (rope_cos, rope_sin)
    tensors that get passed to every block for their underlying RoPE computations."""
    def __init__(self, config):
        super().__init__()
        self.config = config

        d_head = config.d_model // config.n_heads
        torch._assert(d_head % 8 == 0, "d_head must be divisible by 8")
        d_xy, d_t = d_head // 8, d_head // 4

        nyq = float(config.rope_nyquist_frac)
        max_freq = min(self.config.height, self.config.width) * nyq
        n = (d_xy + 1) // 2
        xy = (torch.linspace(1.0, max_freq / 2, n, dtype=torch.float32) * torch.pi).repeat_interleave(2)[:d_xy]

        theta = float(config.rope_theta)
        inv_t = 1.0 / (theta ** (torch.arange(0, d_t, 2, dtype=torch.float32) / d_t))
        inv_t = inv_t.repeat_interleave(2)  # [d_t]

        self.register_buffer("xy", xy, persistent=False)        # [d_xy]
        self.register_buffer("inv_t", inv_t, persistent=False)  # [d_t]

    @torch.autocast("cuda", enabled=False)
    def forward(self, pos_ids):
        if not torch.compiler.is_compiling():
            torch._assert(
                (pos_ids["y_pos"].max() < self.config.height) & (pos_ids["x_pos"].max() < self.config.width),
                f"pos_ids out of bounds, {self.config.height}, {self.config.width}"
            )

        x = (2.0 * pos_ids["x_pos"].float() + 1.0) / self.config.width - 1.0
        y = (2.0 * pos_ids["y_pos"].float() + 1.0) / self.config.height - 1.0
        t = pos_ids["t_pos"].float()

        freqs = torch.cat(
            (x.unsqueeze(-1) * self.xy, y.unsqueeze(-1) * self.xy, t.unsqueeze(-1) * self.inv_t),
            dim=-1,  # [B,T,d_head//2]
        )
        # Returns rope_cos, rope_sin angles of shape [B, 1, T, D/2]
        return freqs.cos()[:, None], freqs.sin()[:, None]


class OrthoRoPE(NoCastModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @torch.autocast("cuda", enabled=False)
    def forward(self, x, rope_angles):
        cos, sin = rope_angles
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)


class Attn(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.value_residual = config.value_residual

        if self.value_residual:
            self.v_lamb = nn.Parameter(torch.tensor(0.5))

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_model // self.n_heads
        assert config.d_model % self.n_heads == 0

        self.enable_gqa = self.n_heads != self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = OrthoRoPE(config)

        self.gated_attn = config.gated_attn
        if self.gated_attn:
            self.gate_proj = nn.Linear(self.n_heads, self.n_heads, bias=False)  # sparse attn gate
            nn.init.zeros_(self.gate_proj.weight)

    def forward(self, x, pos_ids, rope_angles, v1, kv_cache):
        # Q, K, V proj -> QK-norm -> RoPE
        q = eo.rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_heads, d=self.d_head)
        k = eo.rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.n_kv_heads, d=self.d_head)
        v = eo.rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.n_kv_heads, d=self.d_head)

        if self.value_residual:
            v1 = v if v1 is None else v1
            v = torch.lerp(v, v1.view_as(v), self.v_lamb)

        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rope(q, rope_angles), self.rope(k, rope_angles)

        # Update KV-cache in-place
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
