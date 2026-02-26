import torch
import einops as eo
from torch import nn

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

create_block_mask = torch.compile(create_block_mask)
flex_attention = torch.compile(flex_attention)

from rotary_embedding_torch import RotaryEmbedding

from .nn import rms_norm, NoCastModule


class RoPE(NoCastModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert not getattr(self.config, "has_audio", False)

        freqs = self.get_freqs(config)
        self.cos = nn.Buffer(freqs.cos().contiguous(), persistent=False)
        self.sin = nn.Buffer(freqs.sin().contiguous(), persistent=False)

    def get_angles(self, pos_ids):
        t, y, x = pos_ids["t_pos"], pos_ids["y_pos"], pos_ids["x_pos"]  # [B,T]
        H, W = self.config.height, self.config.width
        if not torch.compiler.is_compiling():
            torch._assert((y.max() < H) & (x.max() < W), f"pos_ids out of bounds, {y.max()}, {x.max()}")
        flat = t * (H * W) + y * W + x                         # [B,T]
        idx = flat.reshape(-1).to(torch.long)
        cos = self.cos.index_select(0, idx).view(*flat.shape, -1)
        sin = self.sin.index_select(0, idx).view(*flat.shape, -1)
        return cos[:, None], sin[:, None]  # add head dim for broadcast

    @torch.autocast("cuda", enabled=False)
    def forward(self, x, pos_ids):
        assert self.cos.dtype == self.sin.dtype == torch.float32
        cos, sin = self.get_angles(pos_ids)
        x0, x1 = x.float().unfold(-1, 2, 2).unbind(-1)
        y0 = x0 * cos - x1 * sin
        y1 = x1 * cos + x0 * sin
        return torch.cat((y0, y1), dim=-1).type_as(x)

    def get_freqs(self, config):
        raise NotImplementedError


class OrthoRoPE(RoPE):
    """
    RoPE for rotation across orthogonal axes: time, height, and width
    Time: Geometric Spectrum -- rotates 1/2 of head dim
    Height / Width: Linear Spectrum -- rotates 1/4th of head dim each (1/2 combined)
    """
    def get_freqs(self, config):
        H, W, T = config.height, config.width, config.n_frames
        head_dim = config.d_model // config.n_heads

        max_freq = min(H, W) * 0.8  # stay below nyquist
        rope_xy = RotaryEmbedding(dim=head_dim // 8, freqs_for='pixel', max_freq=max_freq)
        freqs_x = rope_xy(torch.linspace(-1 + 1 / W, 1 - 1 / W, W))[None, :, :]   # [1,W,D]
        freqs_y = rope_xy(torch.linspace(-1 + 1 / H, 1 - 1 / H, H))[:, None, :]   # [H,1,D]

        freq_t = RotaryEmbedding(dim=head_dim // 4, freqs_for='lang').forward(torch.arange(T))

        return torch.cat([
            eo.repeat(freqs_x.expand(H, W, -1), 'h w d -> (t h w) d', t=T),   # X
            eo.repeat(freqs_y.expand(H, W, -1), 'h w d -> (t h w) d', t=T),   # Y
            eo.repeat(freq_t, 't d -> (t h w) d', h=H, w=W)     # T
        ], dim=-1)

from typing import Optional
from torch import Tensor
from tensordict import TensorDict
def get_block_mask_mod(
    pos_ids: TensorDict,
    window_len: Optional[Tensor] = None,
    doc_id: Optional[Tensor] = None,
    is_causal: bool = True,
    nattn_window: Optional[int] = None,
    dilation: Optional[int] = None,
    pinned_dilation: Optional[int] = None,
    curr_frame_mask: Optional[Tensor] = None,
    context_conditioning: bool = False,
    device="cpu"
):
    t_pos = pos_ids["t_pos"]
    assert not (pinned_dilation and dilation)

    def mask_mod(b, h, q, kv):
        t_q, t_kv = t_pos[b, q], t_pos[b, kv]  # timestamp of q / kv

        base_mask = (t_kv <= t_q) if is_causal else True  # causal / bidirectional
        window_mask = (t_q - t_kv).abs() < window_len if window_len is not None else True  # sliding window
        same_doc_mask = ((doc_id[b, q] >= 0) & (doc_id[b, q] == doc_id[b, kv])) if doc_id is not None else True
        dilation_mask = ((t_q - t_kv) % dilation == 0) if dilation is not None else True
        pinned_dil_mask = (t_kv % pinned_dilation == 0) | (t_q == t_kv) if pinned_dilation is not None else True

        # Neighborhood Attention
        if nattn_window is not None:
            x_q, x_kv = pos_ids["x_pos"][b, q], pos_ids["x_pos"][b, kv]
            y_q, y_kv = pos_ids["y_pos"][b, q], pos_ids["y_pos"][b, kv]
            nattn_mask = ((x_q - x_kv).abs() < nattn_window) & ((y_q - y_kv).abs() < nattn_window)
        else:
            nattn_mask = True

        # Teacher Forcing
        #################
        # current prev attn: previous frames are noised at a contant level, current frames noised at random level
        # matches inference behavior
        if curr_frame_mask is not None:
            cid_q, cid_kv = curr_frame_mask[b, q], curr_frame_mask[b, kv]
            prev_curr_mask = ((cid_kv == 0) & ((cid_q == 0) | (t_kv != t_q))) \
                | ((cid_kv == cid_q) & (cid_q >= 1) & (t_kv == t_q))
        else:
            prev_curr_mask = True

        return base_mask & window_mask & same_doc_mask & prev_curr_mask & dilation_mask & pinned_dil_mask & nattn_mask

    return mask_mod


def make_attn_block_mask(
    config,
    layer_idx: int,
    seq_len: int,
    pos_ids: TensorDict,
    doc_id: Optional[Tensor],
    curr_frame_mask: Optional[Tensor],
    device,
):
    # defaults
    causal = getattr(config, "causal", True)
    global_period = getattr(config, "global_attn_period", 4)
    global_offset = getattr(config, "global_attn_offset", 0)
    local_nattn_window = getattr(config, "local_nattn_window", None)

    t_pos = pos_ids["t_pos"]
    torch._assert(t_pos.shape[-1] - seq_len == 0, "q_offset must be 0")
    torch._assert(doc_id is None or doc_id.size(1) == t_pos.size(1), "doc_id must be token-expanded to S tokens")

    kwargs = dict(
        pos_ids=pos_ids,
        doc_id=doc_id,
        is_causal=causal,
        curr_frame_mask=curr_frame_mask,
        context_conditioning=getattr(config, "context_conditioning", False),
        device=device,
    )
    L = t_pos.shape[-1]

    if getattr(config, "global_local_merged", False):
        local_mask_mod = get_block_mask_mod(window_len=config.local_window, nattn_window=local_nattn_window, **kwargs)
        global_mask_mod = get_block_mask_mod(
            window_len=config.global_window,
            dilation=getattr(config, "global_dilation", None),
            pinned_dilation=getattr(config, "global_pinned_dilation", None),
            **kwargs,
        )
        half = config.n_heads // 2

        def mask_mod(b, h, q, kv):
            global_mask = global_mask_mod(b, h, q, kv)
            local_mask = local_mask_mod(b, h, q, kv)
            is_global = h < half
            return (is_global & global_mask) | ((~is_global) & local_mask)

        return create_block_mask(mask_mod, B=None, H=config.n_heads, Q_LEN=L, KV_LEN=L, device=device)

    off = global_offset % global_period
    use_global = ((layer_idx - off) % global_period) == 0
    mask_mod = get_block_mask_mod(
        window_len=config.global_window if use_global else config.local_window,
        nattn_window=None if use_global else local_nattn_window,
        dilation=getattr(config, "global_dilation", None) if use_global else None,
        pinned_dilation=getattr(config, "global_pinned_dilation", None) if use_global else None,
        **kwargs,
    )
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=L, KV_LEN=L, device=device)


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

    def forward(self, x, context, context_pad_mask):
        torch._assert(context_pad_mask is not None, "context_pad_mask is None")
        torch._assert(context_pad_mask.dtype == torch.bool, "context_pad_mask must be bool")
        torch._assert(context_pad_mask.shape[0] == context.shape[0], "bad mask batch")
        torch._assert(context_pad_mask.shape[1] == context.shape[1], "bad mask seq")
        torch._assert_async((~context_pad_mask).any(dim=1).all(), "all-pad context")

        q = eo.rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.n_heads)
        k = eo.rearrange(self.k_proj(context), "b t (h d) -> b h t d", h=self.n_heads)
        v = eo.rearrange(self.v_proj(context), "b t (h d) -> b h t d", h=self.n_heads)
        q, k = rms_norm(q), rms_norm(k)
        keep = ~context_pad_mask  # True = keep
        out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=keep[:, None, None, :],
            dropout_p=0.0,
        )
        out = out.transpose(1, 2).contiguous().reshape(x.size(0), x.size(1), -1)
        return self.out_proj(out)


