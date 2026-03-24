"""
Portable attention modules that replace FlexAttention with standard SDPA.
Used for OpenVINO export — no Triton, no BlockMask, no CUDA-specific ops.

Updated for wp-1.5: OrthoRoPEAngles computed once per forward pass,
passed as rope_angles to each block. No rotary_embedding_torch dependency.
"""
import torch
import einops as eo
from torch import nn, Tensor
import torch.nn.functional as F

from .model.nn import rms_norm, NoCastModule


class PortableOrthoRoPEAngles(NoCastModule):
    """Computes RoPE angles once per forward pass from pos_ids.
    Portable version of OrthoRoPEAngles — no torch.autocast('cuda')."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        d_head = config.d_model // config.n_heads
        assert d_head % 8 == 0, "d_head must be divisible by 8"
        d_xy, d_t = d_head // 8, d_head // 4

        nyq = float(getattr(config, "rope_nyquist_frac", 0.8))
        max_freq = min(config.height, config.width) * nyq
        n = (d_xy + 1) // 2
        xy = (torch.linspace(1.0, max_freq / 2, n, dtype=torch.float32) * torch.pi).repeat_interleave(2)[:d_xy]

        theta = float(getattr(config, "rope_theta", 10000.0))
        inv_t = 1.0 / (theta ** (torch.arange(0, d_t, 2, dtype=torch.float32) / d_t))
        inv_t = inv_t.repeat_interleave(2)  # [d_t]

        self.register_buffer("xy", xy, persistent=False)       # [d_xy]
        self.register_buffer("inv_t", inv_t, persistent=False)  # [d_t]

    def forward(self, pos_ids):
        x = (2.0 * pos_ids["x_pos"].float() + 1.0) / self.config.width - 1.0
        y = (2.0 * pos_ids["y_pos"].float() + 1.0) / self.config.height - 1.0
        t = pos_ids["t_pos"].float()

        freqs = torch.cat(
            (x.unsqueeze(-1) * self.xy, y.unsqueeze(-1) * self.xy, t.unsqueeze(-1) * self.inv_t),
            dim=-1,  # [B, T, d_head//2]
        )
        # Returns rope_cos, rope_sin of shape [B, 1, T, D/2]
        return freqs.cos()[:, None], freqs.sin()[:, None]


class PortableOrthoRoPE(NoCastModule):
    """Applies RoPE rotation from precomputed angles.
    Portable version — uses view(-1, 2) instead of unfold(-1, 2, 2)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert not getattr(self.config, "has_audio", False)

    def forward(self, x, rope_angles):
        cos, sin = rope_angles
        # Replace unfold(-1, 2, 2) with view — equivalent but OV-compatible
        xf = x.float().reshape(*x.shape[:-1], -1, 2)
        x0, x1 = xf[..., 0], xf[..., 1]
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)


class PortableAttn(nn.Module):
    """Self-attention using SDPA instead of FlexAttention. Stateless — KV cache handled externally."""

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
        self.enable_gqa = self.n_heads != self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope = PortableOrthoRoPE(config)

        self.gated_attn = getattr(config, "gated_attn", False)
        if self.gated_attn:
            self.gate_proj = nn.Linear(self.n_heads, self.n_heads, bias=False)
            nn.init.zeros_(self.gate_proj.weight)

        # Per-layer KV cache geometry
        tpf = config.height * config.width
        period = config.global_attn_period
        off = getattr(config, "global_attn_offset", 0) % period
        is_global = ((layer_idx - off) % period == 0)
        L = (config.global_window if is_global else config.local_window) * tpf
        pd = config.global_pinned_dilation if is_global else 1
        self.kv_L = L
        self.kv_pinned_dilation = pd
        self.kv_num_buckets = (L // tpf) // pd

    def forward(self, x, pos_ids, rope_angles, v1, kv_buf, written, is_frozen):
        """
        Args:
            x: [B, T, D]
            pos_ids: dict with f_pos, t_pos, y_pos, x_pos
            rope_angles: (cos, sin) from PortableOrthoRoPEAngles
            v1: first-layer V for value residual (or None)
            kv_buf: [2, B, H_kv, capacity, Dh] — external KV state
            written: [capacity] bool — which slots are valid
            is_frozen: scalar bool tensor
        Returns:
            y: [B, T, D] — attention output
            v1: updated value residual
            kv_buf_out: [2, B, H_kv, capacity, Dh] — updated KV state
            written_out: [capacity] bool — updated written mask
        """
        from .stateless_kv import upsert_stateless

        B, T, _ = x.shape

        q = eo.rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_heads, d=self.d_head)
        k = eo.rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.n_kv_heads, d=self.d_head)
        v = eo.rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.n_kv_heads, d=self.d_head)

        if self.value_residual:
            v1 = v if v1 is None else v1
            v = torch.lerp(v, v1.view_as(v), self.v_lamb)

        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rope(q, rope_angles), self.rope(k, rope_angles)

        # Stateless KV cache upsert
        k_full, v_full, attn_mask, kv_buf_out, written_out = upsert_stateless(
            kv_buf, written, k, v, pos_ids,
            is_frozen=is_frozen,
            tpf=T,
            L=self.kv_L,
            num_buckets=self.kv_num_buckets,
            pinned_dilation=self.kv_pinned_dilation,
        )

        # GQA: expand KV heads to match Q heads
        if self.enable_gqa:
            repeats = self.n_heads // self.n_kv_heads
            k_full = k_full.repeat_interleave(repeats, dim=1)
            v_full = v_full.repeat_interleave(repeats, dim=1)

        # SDPA with dense mask — cast KV to match Q dtype (KV may be FP16 for bandwidth)
        y = F.scaled_dot_product_attention(q, k_full.to(q.dtype), v_full.to(q.dtype), attn_mask=attn_mask.to(q.dtype))

        if self.gated_attn:
            gates = torch.sigmoid(self.gate_proj(x[..., :self.n_heads]))
            y = y * gates.permute(0, 2, 1).unsqueeze(-1)

        y = eo.rearrange(y, "b h t d -> b t (h d)")
        y = self.out_proj(y)
        return y, v1, kv_buf_out, written_out


class PortableCrossAttention(nn.Module):
    """Cross-attention using SDPA instead of FlexAttention."""

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
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().reshape(x.size(0), x.size(1), -1)
        return self.out_proj(out)
