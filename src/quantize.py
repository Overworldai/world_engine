from typing import Optional

import torch
import torch.nn as nn


QUANTS = [None]  # TODO: enable specific quant based on model config, which should specify compatible quants [None, "w8a8", "fp8"]


try:
    from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
    QUANTS.append("nvfp4")
except ImportError:
    pass
try:
    from sgl_kernel import int8_scaled_mm as sgl_int8_scaled_mm
    if "w8a8" not in QUANTS:
        QUANTS.append("w8a8")
except ImportError:
    sgl_int8_scaled_mm = None
try:
    from gemlite.helper import A8W8_INT8_dynamic
    import gemlite
    gemlite.set_autotune("max")
except ImportError:
    A8W8_INT8_dynamic = None
try:
    from lmdeploy.pytorch.models.q_modules import QLinear
except ImportError:
    QLinear = None


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


class FP8Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features

        self.bias = (
            nn.Parameter(lin.bias.data.clone().to(torch.float8_e4m3fn))
            if lin.bias is not None
            else None
        )
        w_amax = lin.weight.data.abs().amax()
        w = lin.weight.data.clone().div(w_amax).to(torch.float8_e4m3fn)
        self.register_buffer("w_amax", w_amax)
        self.register_buffer("weightT", w.t())
        self.dummy_scale = torch.ones((), device=lin.weight.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using FP8 matmul.

        Args:
            x: Input tensor of shape [..., in_features] (flattens if > 2D)

        Returns:
            Output tensor of shape [..., out_features] in BF16 format, unflattened if input is > 2D
        """

        # Convert input to FP8 e4m3
        x_fp8 = x.to(torch.float8_e4m3fn).reshape(-1, x.size(-1)).contiguous()

        result = torch._scaled_mm(
            x_fp8,
            self.weightT,
            bias=self.bias,
            scale_a=self.dummy_scale,
            scale_b=self.w_amax,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )

        return result.reshape(x.shape[:-1] + (-1,))

def _per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Local per-token symmetric int8 quantization matching SGLang's W8A8 flow:
      scale = absmax / 127
      x_q   = round(x / scale)
    Returns:
      x_q:    [..., K] int8
      scales: [..., 1] float32
    """
    x_fp = x.float().nan_to_num()
    scales = (x_fp.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127.0).float()
    x_q = torch.round(x_fp / scales).clamp(-127, 127).to(torch.int8)
    return x_q, scales


@torch.library.custom_op("world_engine::w8a8_int8_linear", mutates_args=())
def w8a8_int8_linear(
    a: torch.Tensor,
    b_int8_T: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    if sgl_int8_scaled_mm is None:
        raise ImportError("sgl-kernel is required for quant='w8a8'")

    assert a.ndim == 2, "expected [M, K] input"
    x_q, x_scale = _per_token_quant_int8(a.contiguous())

    bias_arg = None if bias.numel() == 0 else bias
    return sgl_int8_scaled_mm(
        x_q,                 # [M, K] row-major int8
        b_int8_T,            # [K, N] column-major int8 view
        x_scale,             # [M, 1] float32
        b_scale,             # [N, 1] float32
        out_dtype=a.dtype,
        bias=bias_arg,
    )


@w8a8_int8_linear.register_fake
def _w8a8_int8_linear_fake(
    a: torch.Tensor,
    b_int8_T: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (a.shape[0], b_int8_T.shape[1]),
        device=a.device,
        dtype=a.dtype,
    )


class W8A8Int8LinearSGLang(nn.Module):
    """
    INT8 W8A8 linear using sgl-kernel's int8_scaled_mm.
    Weight path:
      - static per-channel symmetric int8
      - stored as a transposed [K, N] view (column-major for the kernel)
    Activation path:
      - dynamic per-token symmetric int8
    """

    __constants__ = ("in_features", "out_features")

    def __init__(self, lin: nn.Linear):
        super().__init__()

        if sgl_int8_scaled_mm is None:
            raise ImportError("sgl-kernel is required for quant='w8a8'")

        self.in_features = lin.in_features
        self.out_features = lin.out_features

        # Your current eligible() already enforces % 32, which is stricter than needed.
        w = lin.weight.detach()  # [N, K]

        # Per-output-channel symmetric weight quantization.
        w_scale = (
            w.float()
            .abs()
            .nan_to_num()
            .amax(dim=1, keepdim=True)
            .clamp_min(1e-10)
            / 127.0
        ).float()  # [N, 1]

        w_q = torch.round(w.float() / w_scale).clamp(-127, 127).to(torch.int8)  # [N, K]

        # IMPORTANT: keep this as a transpose view, not contiguous().
        # sgl-kernel expects mat_b to be column-major [K, N] with stride(0) == 1.
        self.register_buffer("weight_int8_T", w_q.t())         # [K, N], column-major view
        self.register_buffer("weight_scale", w_scale.contiguous())  # [N, 1]

        if lin.bias is None:
            self.register_buffer(
                "bias",
                torch.empty(0, device=w.device, dtype=lin.weight.dtype),
            )
        else:
            self.register_buffer(
                "bias",
                lin.bias.detach().to(lin.weight.dtype).contiguous(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda, "w8a8 requires CUDA"
        assert x.dtype in (torch.float16, torch.bfloat16), \
            "w8a8 expects fp16/bf16 activations"

        s = x.shape
        x2 = x.reshape(-1, s[-1]).contiguous()
        bias = self.bias
        if bias.numel() != 0 and bias.dtype != x2.dtype:
            bias = bias.to(x2.dtype)
        y = w8a8_int8_linear(
            x2,
            self.weight_int8_T,
            self.weight_scale,
            bias,
        )
        return y.reshape(*s[:-1], self.out_features)


class INT8W8A8GemLite(nn.Module):
    __constants__ = ("in_features", "out_features")

    def __init__(self, lin: nn.Linear):
        super().__init__()
        if A8W8_INT8_dynamic is None:
            raise ImportError("Install gemlite for quant='w8a8_gemlite'")

        self.in_features = lin.in_features
        self.out_features = lin.out_features

        # Minimal wrapper: assumes the layer is already on the target CUDA device.
        self.impl = A8W8_INT8_dynamic(
            device=str(lin.weight.device),
            dtype=lin.weight.dtype,
        ).from_linear(lin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape
        y = self.impl(x.reshape(-1, s[-1]).contiguous())
        return y.reshape(*s[:-1], self.out_features).to(x.dtype)


class INT8W8A8LMDeploy(nn.Module):
    __constants__ = ("in_features", "out_features")

    def __init__(self, lin: nn.Linear):
        super().__init__()
        if QLinear is None:
            raise ImportError("Install lmdeploy for quant='w8a8_lmdeploy'")

        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.impl = QLinear.from_float(lin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape
        y = self.impl(x.reshape(-1, s[-1]).contiguous())
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
        "w8a8_gemlite": INT8W8A8GemLite,
        "w8a8_lmdeploy": INT8W8A8LMDeploy,
        "w8a8_sglang": W8A8Int8LinearSGLang,
        "fp8w8a8": FP8W8A8Linear,
        "nvfp4": FP4Linear,
        "fp8": FP8Linear,
    }[quant]

    for name, child in model.named_children():
        setattr(model, name, new_linear(child)) if eligible(child) else quantize_model(
            child, quant
        )
    return model


from torchao.quantization import (quantize_, 
                                  Int4WeightOnlyConfig, 
                                  Int8WeightOnlyConfig, 
                                  Int8DynamicActivationInt8WeightConfig,
                                  Float8DynamicActivationInt4WeightConfig,
                                  Int8DynamicActivationIntxWeightConfig,
                                  Float8WeightOnlyConfig,
                                  Float8DynamicActivationFloat8WeightConfig,
                                  PerTensor, PerRow)
from torchao.quantization.quantize_.workflows import Int4PackingFormat, Float8PackingFormat
from torchao.quantization.qat import (QATConfig, 
                                      IntxFakeQuantizeConfig, 
                                      Float8FakeQuantizeConfig)

_LAYER_FILTERS = {
    "mlp":       lambda mod, fqn: isinstance(mod, torch.nn.Linear) and "transformer.blocks" in fqn and ".mlp." in fqn,
    "attn": lambda mod, fqn: isinstance(mod, torch.nn.Linear) and ".attn." in fqn,\
    "mlp_and_attn": lambda mod, fqn: isinstance(mod, torch.nn.Linear) and "transformer.blocks" in fqn and (".mlp." in fqn or ".attn." in fqn),
    "all":       lambda mod, fqn: isinstance(mod, torch.nn.Linear),
}

def apply_ptq_model(model, config: str, layers: str = "mlp"):
    """Apply PTQ in-place. layers: 'mlp', 'attention', or None for all Linear layers."""
    filter_fn = _LAYER_FILTERS.get(layers) if layers else None

    if config == "int4_weights":
        qconfig = Int4WeightOnlyConfig(group_size=32, int4_packing_format="tile_packed_to_4d", int4_choose_qparams_algorithm="hqq")
    elif config == "int8_weights":
        qconfig = Int8WeightOnlyConfig()
    elif config == "int_w8a8":
        qconfig = Int8DynamicActivationInt8WeightConfig()
    elif config == "int4w_int8a":
        qconfig = Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4)
    elif config == "int4w_fp8a":
        qconfig = Float8DynamicActivationInt4WeightConfig()
    elif config == "fp_w8a8":
        qconfig = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
    # elif config == "fp_w4a4":
    #     qconfig = NVFP4DynamicActivationNVFP4WeightConfig()
    elif config == "fp8_weights":
        qconfig = Float8WeightOnlyConfig()
    else:
        raise ValueError(f"Unknown quant_config: {config!r}")
    quantize_(model, qconfig, filter_fn=filter_fn)

def apply_qat(model, quant_config: str = "fp8_general", layers: str = None, step: str = "prepare"):
    """Apply QAT in-place. layers: 'mlp', 'attention', or None for all Linear layers."""
    filter_fn = _LAYER_FILTERS.get(layers) if layers else None

    if step == "prepare":
        if quant_config == "fp8_general":
            weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerTensor())
            qconfig = QATConfig(weight_config=weight_config, step=step)
        elif quant_config == "int8_weights":
            weight_config = IntxFakeQuantizeConfig(torch.int8, group_size=32, is_symmetric=True)
            qconfig = QATConfig(weight_config=weight_config, step=step)
        elif quant_config == "int4_weights":
            config = Int4WeightOnlyConfig(
                group_size=32,
                int4_packing_format=Int4PackingFormat.PRESHUFFLED
            )
            qconfig = QATConfig(base_config=config, step=step)
        else:
            raise ValueError(f"Unknown quant_config: {quant_config!r}")

    elif step == "convert":
        # convert step requires a real PTQ base_config (not FakeQuantizeConfigBase)
        # or None (which just strips fake-quant wrappers back to plain nn.Linear)
        if quant_config == "fp8_general":
            qconfig = QATConfig(base_config=Float8WeightOnlyConfig(), step=step)
        elif quant_config == "int8_weights":
            quantize_(model, QATConfig(step=step), filter_fn=filter_fn)  # need to run quantize to convert fake quant to real quant for int8, since int8 fake quant is not a simple wrapper around int8 PTQ module
            qconfig = QATConfig(base_config=Int8WeightOnlyConfig(), step=step)
        elif quant_config == "int4_weights":
            qconfig = QATConfig(base_config=Int4WeightOnlyConfig(group_size=32, int4_packing_format=Int4PackingFormat.PRESHUFFLED), step=step)
        elif quant_config == "bf16":
            qconfig = QATConfig(step=step)
        else:
            raise ValueError(f"Unknown quant_config: {quant_config!r}")
    quantize_(model, qconfig, filter_fn=filter_fn)