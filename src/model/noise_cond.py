import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from .nocast_module import NoCastModule


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class MLP(nn.Module):
    def __init__(self, dim_in, dim_middle, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_middle, bias=False)
        self.fc2 = nn.Linear(dim_middle, dim_out, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 2 * dim, bias=False)

    def forward(self, x, cond):
        # cond: [b, n, d], x: [b, n*m, d]
        b, n, d = cond.shape
        _, nm, _ = x.shape
        m = nm // n

        y = F.silu(cond)
        ab = self.fc(y)                    # [b, n, 2d]
        ab = ab.view(b, n, 1, 2 * d)         # [b, n, 1, 2d]
        ab = ab.expand(-1, -1, m, -1)      # [b, n, m, 2d]
        ab = ab.reshape(b, nm, 2 * d)        # [b, nm, 2d]

        a, b_ = ab.chunk(2, dim=-1)        # [b, nm, d] each
        x = rms_norm(x) * (1 + a) + b_
        return x


def ada_rmsnorm(x, scale, bias):
    x4 = eo.rearrange(x, 'b (n m) d -> b n m d', n=scale.size(1))
    y4 = rms_norm(x4) * (1 + scale.unsqueeze(2)) + bias.unsqueeze(2)
    return eo.rearrange(y4, 'b n m d -> b (n m) d')


def ada_gate(x, gate):
    x4 = eo.rearrange(x, 'b (n m) d -> b n m d', n=gate.size(1))
    return eo.rearrange(x4 * gate.unsqueeze(2), 'b n m d -> b (n m) d')


class NoiseConditioner(NoCastModule):
    """Sigma -> logSNR -> Fourier Features -> Dense"""
    def __init__(self, dim, fourier_dim=512, base=10_000.0):
        super().__init__()
        assert fourier_dim % 2 == 0
        half = fourier_dim // 2
        self.freq = nn.Buffer(torch.logspace(0, -1, steps=half, base=base, dtype=torch.float32), persistent=False)
        self.mlp = MLP(fourier_dim, dim * 4, dim)

    def forward(self, s, eps=torch.finfo(torch.float32).eps):
        assert self.freq.dtype == torch.float32
        orig_dtype, shape = s.dtype, s.shape

        with torch.autocast("cuda", enabled=False):
            s = s.reshape(-1).float()  # fp32 for fourier numerical stability
            s = s * 1000  # expressive rotation range

            # calculate fourier features
            phase = s[:, None] * self.freq[None, :]
            emb = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
            emb = emb * 2**0.5  # Ensure unit variance
            emb = self.mlp(emb)

        return emb.to(orig_dtype).view(*shape, -1)
