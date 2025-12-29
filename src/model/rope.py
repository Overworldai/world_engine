from rotary_embedding_torch import RotaryEmbedding
import torch
from torch import nn

import einops as eo

from .nocast_module import NoCastModule


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
