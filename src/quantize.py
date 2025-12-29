from typing import Optional

import torch
import torch.nn as nn


QUANTS = [None, "w8a8"]


try:
    from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
    QUANTS.append("nvfp4")
except ImportError:
    pass


@torch.library.custom_op("world_engine::fp4_linear", mutates_args=())
def fp4_linear(
    a_bf16: torch.Tensor,
    b_fp4_T: torch.Tensor,
    a_global_sf: torch.Tensor,
    b_sf_T: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    a_fp4, a_sf = nvfp4_quantize(
        a_bf16,
        a_global_sf,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    return mm_fp4(a_fp4, b_fp4_T, a_sf, b_sf_T, alpha, out_dtype=torch.bfloat16, backend="cutlass")


@fp4_linear.register_fake
def _fp4_linear_fake(
    a_bf16: torch.Tensor,
    b_fp4_T: torch.Tensor,
    a_global_sf: torch.Tensor,
    b_sf_T: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.empty((a_bf16.shape[0], b_fp4_T.shape[1]), device=a_bf16.device, dtype=torch.bfloat16)


class FP4Linear(nn.Module):
    """FP4 Linear layer using FlashInfer's NVFP4 quantization."""

    def __init__(self, lin: nn.Linear):
        super().__init__()

        self.in_features = lin.in_features
        self.out_features = lin.out_features

        # Check alignment requirements for NVFP4 TMA
        assert self.in_features % 32 == 0 and self.out_features % 32 == 0, "features % 32 != 0, nvfp4 disallowed"

        # Store weight from original linear layer
        self.weight = nn.Parameter(lin.weight.detach().clone())

        # Cached FP4 weight and scales (populated on first forward)
        self._weight_fp4_T: Optional[torch.Tensor] = None
        self._weight_scales_T: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None
        self._dummy_scale: Optional[torch.Tensor] = None
        self._weight_global_sf = None

        with torch.no_grad():
            # Quantize weights eagerly (no lazy path)
            self._dummy_scale = torch.full((1,), 1.0, device=self.weight.device, dtype=torch.float32)
            weight_bf16 = self.weight.to(torch.bfloat16).to(self.weight.device).contiguous()
            weight_amax = weight_bf16.float().abs().nan_to_num().max()
            self._weight_global_sf = (1.0) / weight_amax
            self._alpha = 1.0 / (self._weight_global_sf * self._dummy_scale)
            w_fp4, w_sf = nvfp4_quantize(
                weight_bf16,
                self._weight_global_sf,
                sfLayout=SfLayout.layout_128x4,
                do_shuffle=False,
            )
            self._weight_fp4_T = w_fp4.t()
            self._weight_scales_T = w_sf.t()

            # Warmup flashinfer fp4 graphs
            assert self.weight.is_cuda, "Weights need to be on GPU before quantization"
            # TODO: test actual shape warmup, might perform better
            lazy_x = torch.zeros((1, lin.in_features), device=self.weight.device, dtype=torch.bfloat16)
            fp4_linear(
                lazy_x,
                self._weight_fp4_T,
                self._dummy_scale,
                self._weight_scales_T,
                self._alpha,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using FP4 quantization and FlashInfer GEMM."""
        x_flat = x.reshape(-1, x.shape[-1])
        y = fp4_linear(
            x_flat.to(torch.bfloat16).contiguous(),
            self._weight_fp4_T,
            self._dummy_scale,
            self._weight_scales_T,
            self._alpha,
        )
        return y.reshape(x.shape[:-1] + (-1,))


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
    if quant is None:
        return model

    def eligible(m: nn.Module) -> bool:
        w = getattr(m, "weight", None)
        if not isinstance(m, nn.Linear):
            return False
        if getattr(w, "dtype", None) != torch.bfloat16:
            return False
        o, k = w.shape
        return (o % 32 == 0) and (k % 32 == 0)

    new_linear = {
        "w8a8": FP8W8A8Linear,
        "nvfp4": FP4Linear,
    }[quant]

    for name, child in model.named_children():
        setattr(model, name, new_linear(child)) if eligible(child) else quantize_model(child, quant)
    return model
