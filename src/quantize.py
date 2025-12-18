import torch
import torch.nn as nn


class FP8W8A8Linear(nn.Module):
    __constants__ = ("in_features", "out_features")

    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features

        f8 = torch.float8_e4m3fn
        inv = 1.0 / float(torch.finfo(f8).max)
        self._inv = inv

        w = lin.weight.detach()
        ws = (w.abs().amax() * inv).clamp_min(1e-8).float()      # 0-d
        wf8 = (w / ws.to(w.dtype)).to(f8).contiguous()            # row-major
        self.register_buffer("wT", wf8.t())                       # col-major view (no contiguous)
        self.register_buffer("ws", ws)

        if lin.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", lin.bias.detach().to(torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape
        x2 = x.reshape(-1, s[-1])

        xs = (x2.abs().amax() * self._inv).clamp_min(1e-8).float()          # 0-d
        xf8 = (x2 / xs.to(x2.dtype)).to(torch.float8_e4m3fn).contiguous()

        y = torch._scaled_mm(
            xf8, self.wT, xs, self.ws,
            bias=self.bias, out_dtype=torch.float16, use_fast_accum=True
        )
        return y.reshape(*s[:-1], self.out_features).to(x.dtype)


def quantize_model(model: nn.Module, quant: str):
    def eligible(m: nn.Module) -> bool:
        w = getattr(m, "weight", None)
        sh = getattr(w, "shape", ())
        return (
            isinstance(m, nn.Linear) and
            getattr(w, "dtype", None) is torch.bfloat16 and
            all(d % 16 == 0 for d in sh)
        )

    new_linear = {
        "w8a8": FP8W8A8Linear,
    }[quant]

    for name, child in model.named_children():
        setattr(model, name, new_linear(child)) if eligible(child) else quantize_model(child, quant)
    return model
