import torch
from torch import nn
import torch.nn.functional as F

import warnings

import einops as eo


class NoCastModule(torch.nn.Module):
    def _apply(self, fn):
        def keep_dtype(t):
            old_dtype = t.dtype
            out = fn(t)
            if out.dtype is not old_dtype:
                warnings.warn(
                    f"{self.__class__.__name__}: requested dtype cast ignored; "
                    f"keeping {old_dtype}.",
                    stacklevel=3,
                )
                out = out.to(dtype=old_dtype)
            return out

        return super()._apply(keep_dtype)

    def to(self, *args, **kwargs):
        warn_cast = False

        # m.to(ref_tensor): use ref's device, ignore its dtype
        if args and isinstance(args[0], torch.Tensor):
            ref, *rest = args
            args = (ref.device, *rest)
            base = next(self.parameters(), None) or next(self.buffers(), None)
            if base is not None and ref.dtype is not base.dtype:
                warn_cast = True

        # keyword dtype
        if kwargs.pop("dtype", None) is not None:
            warn_cast = True

        # positional dtype
        args = tuple(a for a in args if not isinstance(a, torch.dtype))

        if warn_cast:
            warnings.warn(
                f"{self.__class__.__name__}.to: requested dtype cast ignored; "
                "keeping existing dtypes.",
                stacklevel=2,
            )

        return super().to(*args, **kwargs)



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
