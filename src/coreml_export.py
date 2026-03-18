"""
Core ML export for the WorldModel with stateful KV cache.

Ring buffer management and stale-slot masking are handled INSIDE the model
to avoid expensive host-side state.read_state()/write_state() round-trips.

The host precomputes ring slot positions (trivial integer math) and passes
them as fp16 inputs, keeping the entire model graph in fp16.

Architecture per frame:
  1. Host: compute cond (fp32), RoPE (fp32), ring positions (int math)
  2. CoreML predict x4: denoise steps (is_cache_write=0)
  3. CoreML predict x1: cache write  (is_cache_write=1)
  4. Host: VAE decode

Usage:
    PYTHONPATH=. .venv312/bin/python -m src.coreml_export \
        --model-uri Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill \
        --out diagnostics/out/world_model.mlpackage
"""
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _rms_norm_d(x: Tensor) -> Tensor:
    return F.rms_norm(x, (2048,))

def _rms_norm_h(x: Tensor) -> Tensor:
    return F.rms_norm(x, (64,))

def _ortho_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    cos_h = cos.to(x.dtype)
    sin_h = sin.to(x.dtype)
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    y0 = x0 * cos_h - x1 * sin_h
    y1 = x1 * cos_h + x0 * sin_h
    return torch.stack((y0, y1), dim=-1).reshape_as(x)


N_LAYERS = 24
D_MODEL = 2048
N_HEADS = 32
N_KV_HEADS = 32
D_HEAD = 64
D_ROPE = D_HEAD // 2  # 32
T = 512
C = 32
HP = 16
WP = 32
PH = 2
PW = 2
N_BUTTONS = 256


class StatefulWorldModel(nn.Module):
    def __init__(self, model, cfg):
        super().__init__()

        self.patchify = model.patchify
        self.unpatchify = model.unpatchify
        self.out_norm_fc = model.out_norm.fc
        self.ctrl_emb = model.ctrl_emb

        local_L = cfg.local_window * T
        global_L = cfg.global_window * T
        period = cfg.global_attn_period
        off = getattr(cfg, "global_attn_offset", 0) % period

        self.blocks = nn.ModuleList()
        for i, blk in enumerate(model.transformer.blocks):
            is_global = ((i - off) % period == 0)
            L = global_L if is_global else local_L
            cap = L + T
            self.blocks.append(_Block(blk, cfg, cap, L, is_global))

    def _ada_ln(self, x: Tensor, cond: Tensor) -> Tensor:
        y = F.silu(cond)
        ab = self.out_norm_fc(y)
        ab = ab.unsqueeze(2).expand(-1, -1, T, -1).reshape(1, T, 2 * D_MODEL)
        a, b_ = ab.chunk(2, dim=-1)
        return _rms_norm_d(x) * (1 + a) + b_

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
        mouse: Tensor,
        button: Tensor,
        scroll: Tensor,
        is_cache_write: Tensor,
        ring_start_local: Tensor,
        ring_start_global: Tensor,
        write_step_global: Tensor,
    ) -> Tensor:
        ctrl_emb = self.ctrl_emb(mouse, button, scroll) + 1e-7

        x_2d = x.reshape(1, C, HP * PH, WP * PW)
        x_pat = self.patchify(x_2d)
        x_seq = x_pat.permute(0, 2, 3, 1).reshape(1, T, D_MODEL)

        v1 = torch.zeros(1, N_KV_HEADS, T, D_HEAD, device=x.device, dtype=x.dtype)
        first = True
        for blk in self.blocks:
            x_seq, v1 = blk(
                x_seq, cond, ctrl_emb,
                rope_cos, rope_sin,
                None if first else v1,
                is_cache_write,
                ring_start_local, ring_start_global, write_step_global,
            )
            first = False

        x_seq = F.silu(self._ada_ln(x_seq, cond))

        v_out = self.unpatchify(x_seq)
        v_out = v_out.view(1, HP, WP, C, PH, PW)
        v_out = v_out.permute(0, 3, 1, 4, 2, 5).reshape(1, 1, C, HP * PH, WP * PW)
        return v_out


class _Block(nn.Module):
    def __init__(self, blk, cfg, kv_capacity: int, L: int, is_global: bool):
        super().__init__()
        self.kv_capacity = kv_capacity
        self.L = L
        self.is_global = is_global

        attn = blk.attn
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.out_proj = attn.out_proj
        self.value_residual = attn.value_residual
        if self.value_residual:
            self.v_lamb = attn.v_lamb

        self.cond_proj = blk.cond_head.cond_proj
        self.cond_bias_in = blk.cond_head.bias_in
        self.mlp = blk.mlp

        self.has_ctrl = blk.ctrl_mlpfusion is not None
        if self.has_ctrl:
            ctrl = blk.ctrl_mlpfusion
            Wx, Wc = ctrl.mlp.fc1.weight.chunk(2, dim=1)
            D = ctrl.mlp.fc2.in_features
            dev, dt = ctrl.mlp.fc2.weight.device, ctrl.mlp.fc2.weight.dtype
            self.ctrl_fc1_x = nn.Linear(D, D, bias=False, device=dev, dtype=dt)
            self.ctrl_fc1_c = nn.Linear(D, D, bias=False, device=dev, dtype=dt)
            self.ctrl_fc2 = nn.Linear(D, D, bias=False, device=dev, dtype=dt)
            with torch.no_grad():
                self.ctrl_fc1_x.weight.copy_(Wx)
                self.ctrl_fc1_c.weight.copy_(Wc)
                self.ctrl_fc2.weight.copy_(ctrl.mlp.fc2.weight)

        self.register_buffer("k_cache", torch.zeros(1, N_KV_HEADS, kv_capacity, D_HEAD))
        self.register_buffer("v_cache", torch.zeros(1, N_KV_HEADS, kv_capacity, D_HEAD))
        written = torch.zeros(kv_capacity)
        written[kv_capacity - T:] = 1.0
        self.register_buffer("written", written)
        self.register_buffer("_frame_offsets", torch.arange(T, dtype=torch.long))
        self.register_buffer("_pos_arange", torch.arange(kv_capacity, dtype=torch.long))

    def _cond(self, cond: Tensor):
        c = cond + self.cond_bias_in if self.cond_bias_in is not None else cond
        h = F.silu(c)
        return tuple(p(h) for p in self.cond_proj)

    def _ada_rmsnorm(self, x: Tensor, scale: Tensor, bias: Tensor) -> Tensor:
        x4 = x.view(1, 1, T, D_MODEL)
        y4 = _rms_norm_d(x4) * (1 + scale.unsqueeze(2)) + bias.unsqueeze(2)
        return y4.reshape(1, T, D_MODEL)

    def _ada_gate(self, x: Tensor, gate: Tensor) -> Tensor:
        x4 = x.view(1, 1, T, D_MODEL)
        return (x4 * gate.unsqueeze(2)).reshape(1, T, D_MODEL)

    def forward(
        self, x: Tensor, cond: Tensor, ctrl_emb: Tensor,
        rope_cos: Tensor, rope_sin: Tensor, v1_in: Tensor,
        is_cache_write: Tensor,
        ring_start_local: Tensor, ring_start_global: Tensor,
        write_step_global: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        s0, b0, g0, s1, b1, g1 = self._cond(cond)

        residual = x
        x_n = self._ada_rmsnorm(x, s0, b0)

        q = self.q_proj(x_n).reshape(1, T, N_HEADS, D_HEAD).transpose(1, 2)
        k_new = self.k_proj(x_n).reshape(1, T, N_KV_HEADS, D_HEAD).transpose(1, 2)
        v_new = self.v_proj(x_n).reshape(1, T, N_KV_HEADS, D_HEAD).transpose(1, 2)

        if self.value_residual and v1_in is not None:
            v1_r = v1_in.view(1, N_KV_HEADS, T, D_HEAD)
            v_new = v_new + self.v_lamb * (v1_r - v_new)
            v1_out = v1_in
        else:
            v1_out = v_new

        q = _rms_norm_h(q)
        k_new = _rms_norm_h(k_new)
        q = _ortho_rope(q, rope_cos, rope_sin)
        k_new = _ortho_rope(k_new, rope_cos, rope_sin)

        # -- Write KV to tail (always) --
        cap = self.kv_capacity
        L = self.L
        self.k_cache[:, :, cap - T:cap, :] = k_new
        self.v_cache[:, :, cap - T:cap, :] = v_new

        # -- Pick precomputed ring_start and write_step for this layer type --
        if self.is_global:
            ring_start = ring_start_global.reshape(-1).long()
            write_step = write_step_global.reshape(-1)
        else:
            ring_start = ring_start_local.reshape(-1).long()
            write_step = torch.ones_like(is_cache_write.reshape(-1))

        ring_idx = self._frame_offsets + ring_start  # [T]

        # -- Effective flags --
        icw = is_cache_write.reshape(-1)
        effective_write = icw * write_step
        stale_factor = write_step * (1 - icw)

        # -- Attention mask --
        ring_pos = ((self._pos_arange >= ring_start) & (self._pos_arange < ring_start + T)).float()
        w_eff = self.written * (1 - ring_pos * stale_factor)
        attn_bias = w_eff.unsqueeze(0).unsqueeze(0).unsqueeze(0) * 1e4 - 1e4

        y = F.scaled_dot_product_attention(q, self.k_cache, self.v_cache, attn_mask=attn_bias)

        # -- Ring buffer copy --
        ring_idx_kv = ring_idx.view(1, 1, T, 1).expand(1, N_KV_HEADS, T, D_HEAD)

        cur_k = self.k_cache.gather(2, ring_idx_kv)
        k_tail = self.k_cache[:, :, L:cap, :]
        self.k_cache.scatter_(2, ring_idx_kv, cur_k + (k_tail - cur_k) * effective_write)

        cur_v = self.v_cache.gather(2, ring_idx_kv)
        v_tail = self.v_cache[:, :, L:cap, :]
        self.v_cache.scatter_(2, ring_idx_kv, cur_v + (v_tail - cur_v) * effective_write)

        w_cur = self.written.gather(0, ring_idx)
        ones_w = torch.ones_like(w_cur)
        self.written.scatter_(0, ring_idx, w_cur + (ones_w - w_cur) * effective_write.expand(T))

        # -- Rest of block --
        y = y.transpose(1, 2).reshape(1, T, N_HEADS * D_HEAD)
        y = self.out_proj(y)
        x = self._ada_gate(y, g0) + residual

        if self.has_ctrl:
            x_n2 = _rms_norm_d(x)
            c_n = _rms_norm_d(ctrl_emb)
            x4 = x_n2.reshape(1, 1, T, D_MODEL)
            h = F.silu(self.ctrl_fc1_x(x4) + self.ctrl_fc1_c(c_n).unsqueeze(2))
            x = self.ctrl_fc2(h).flatten(1, 2) + x

        x = self._ada_gate(self.mlp(self._ada_rmsnorm(x, s1, b1)), g1) + x
        return x, v1_out


DSIGMAS = [-0.1, -0.15, -0.45, -0.3]  # sigma diffs: [0.9-1.0, 0.75-0.9, 0.3-0.75, 0.0-0.3]


class UnrolledDenoiseModel(StatefulWorldModel):
    """Extends StatefulWorldModel to run 4 Euler denoise steps in one forward pass.
    Inherits directly so buffer names (blocks.{i}.k_cache etc.) match the single-step model."""

    def _single_step(self, x, cond, rope_cos, rope_sin, mouse, button, scroll,
                     is_cache_write, ring_start_local, ring_start_global, write_step_global):
        return super().forward(x, cond, rope_cos, rope_sin, mouse, button, scroll,
                               is_cache_write, ring_start_local, ring_start_global, write_step_global)

    def forward(
        self,
        x: Tensor,
        cond0: Tensor, cond1: Tensor, cond2: Tensor, cond3: Tensor,
        rope_cos: Tensor, rope_sin: Tensor,
        mouse: Tensor, button: Tensor, scroll: Tensor,
        is_cache_write: Tensor,
        ring_start_local: Tensor, ring_start_global: Tensor, write_step_global: Tensor,
    ) -> Tensor:
        args = (rope_cos, rope_sin, mouse, button, scroll,
                is_cache_write, ring_start_local, ring_start_global, write_step_global)
        v = self._single_step(x, cond0, *args); x = x + DSIGMAS[0] * v
        v = self._single_step(x, cond1, *args); x = x + DSIGMAS[1] * v
        v = self._single_step(x, cond2, *args); x = x + DSIGMAS[2] * v
        v = self._single_step(x, cond3, *args); x = x + DSIGMAS[3] * v
        return x


def _patch_rms_norm_globally():
    from .model import nn as model_nn
    from .model import world_model as wm_mod
    from .model import attn as attn_mod
    model_nn.rms_norm = _rms_norm_d
    wm_mod.rms_norm = _rms_norm_d
    attn_mod.rms_norm = _rms_norm_d


def build_model(model_uri: str, device="cpu", dtype=torch.float32):
    """Build model in fp32. coremltools handles fp16 conversion via compute_precision."""
    from .model import WorldModel
    _patch_rms_norm_globally()
    cfg = WorldModel.load_config(model_uri)
    model = WorldModel.from_pretrained(model_uri, cfg=cfg, device=device, dtype=dtype).eval()
    return StatefulWorldModel(model, cfg).eval(), cfg


def convert_to_coreml(model_uri: str, out_path: str, device: str = "cpu"):
    import coremltools as ct
    import numpy as np

    print("[export] Building model...")
    model, cfg = build_model(model_uri, device=device)

    local_L = cfg.local_window * T
    global_L = cfg.global_window * T
    period = cfg.global_attn_period
    off = getattr(cfg, "global_attn_offset", 0) % period

    # Trace in fp32; compute_precision=FLOAT16 handles all conversion uniformly
    x = torch.randn(1, 1, C, HP * PH, WP * PW)
    cond = torch.randn(1, 1, D_MODEL)
    rope_cos = torch.randn(1, 1, T, D_ROPE)
    rope_sin = torch.randn(1, 1, T, D_ROPE)
    mouse = torch.zeros(1, 1, 2)
    button = torch.zeros(1, 1, N_BUTTONS)
    scroll = torch.zeros(1, 1, 1)
    is_cache_write = torch.zeros(1)
    ring_start_local = torch.zeros(1)
    ring_start_global = torch.zeros(1)
    write_step_global = torch.ones(1)

    print("[export] Forward test...")
    with torch.no_grad():
        out = model(x, cond, rope_cos, rope_sin, mouse, button, scroll,
                    is_cache_write, ring_start_local, ring_start_global, write_step_global)
    print(f"[export] Output: {out.shape}")

    print("[export] Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (x, cond, rope_cos, rope_sin, mouse, button, scroll,
             is_cache_write, ring_start_local, ring_start_global, write_step_global),
            strict=False,
        )
    print("[export] Trace done.")

    inputs = [
        ct.TensorType(name="x", shape=(1, 1, C, HP * PH, WP * PW)),
        ct.TensorType(name="cond", shape=(1, 1, D_MODEL)),
        ct.TensorType(name="rope_cos", shape=(1, 1, T, D_ROPE)),
        ct.TensorType(name="rope_sin", shape=(1, 1, T, D_ROPE)),
        ct.TensorType(name="mouse", shape=(1, 1, 2)),
        ct.TensorType(name="button", shape=(1, 1, N_BUTTONS)),
        ct.TensorType(name="scroll", shape=(1, 1, 1)),
        ct.TensorType(name="is_cache_write", shape=(1,)),
        ct.TensorType(name="ring_start_local", shape=(1,)),
        ct.TensorType(name="ring_start_global", shape=(1,)),
        ct.TensorType(name="write_step_global", shape=(1,)),
    ]

    states = []
    for i in range(N_LAYERS):
        is_global = ((i - off) % period == 0)
        cap = (global_L if is_global else local_L) + T
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, N_KV_HEADS, cap, D_HEAD)),
            name=f"blocks.{i}.k_cache",
        ))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, N_KV_HEADS, cap, D_HEAD)),
            name=f"blocks.{i}.v_cache",
        ))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(cap,)),
            name=f"blocks.{i}.written",
        ))

    print(f"[export] {len(inputs)} inputs, {len(states)} states")
    print("[export] Converting to Core ML...")
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        states=states,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    print("[export] Done.")
    mlmodel.save(out_path)
    print(f"[export] Saved to {out_path}")
    return mlmodel


def convert_unrolled_to_coreml(model_uri: str, out_path: str, device: str = "cpu"):
    """Export the 4-step unrolled denoise model. Use alongside the single-step model for cache writes."""
    import coremltools as ct
    import numpy as np

    print("[export-unrolled] Building model...")
    from .model import WorldModel
    _patch_rms_norm_globally()
    cfg = WorldModel.load_config(model_uri)
    model = WorldModel.from_pretrained(model_uri, cfg=cfg, device=device, dtype=torch.float32).eval()
    unrolled = UnrolledDenoiseModel(model, cfg).eval()

    local_L = cfg.local_window * T
    global_L = cfg.global_window * T
    period = cfg.global_attn_period
    off = getattr(cfg, "global_attn_offset", 0) % period

    x = torch.randn(1, 1, C, HP * PH, WP * PW)
    conds = [torch.randn(1, 1, D_MODEL) for _ in range(4)]
    rope_cos = torch.randn(1, 1, T, D_ROPE)
    rope_sin = torch.randn(1, 1, T, D_ROPE)
    mouse = torch.zeros(1, 1, 2)
    button = torch.zeros(1, 1, N_BUTTONS)
    scroll = torch.zeros(1, 1, 1)
    is_cache_write = torch.zeros(1)
    ring_start_local = torch.zeros(1)
    ring_start_global = torch.zeros(1)
    write_step_global = torch.ones(1)

    trace_inputs = (x, *conds, rope_cos, rope_sin, mouse, button, scroll,
                    is_cache_write, ring_start_local, ring_start_global, write_step_global)

    print("[export-unrolled] Forward test...")
    with torch.no_grad():
        out = unrolled(*trace_inputs)
    print(f"[export-unrolled] Output: {out.shape}")

    print("[export-unrolled] Tracing (this may use significant memory)...")
    with torch.no_grad():
        traced = torch.jit.trace(unrolled, trace_inputs, strict=False)
    print("[export-unrolled] Trace done.")

    inputs = [
        ct.TensorType(name="x", shape=(1, 1, C, HP * PH, WP * PW)),
        ct.TensorType(name="cond0", shape=(1, 1, D_MODEL)),
        ct.TensorType(name="cond1", shape=(1, 1, D_MODEL)),
        ct.TensorType(name="cond2", shape=(1, 1, D_MODEL)),
        ct.TensorType(name="cond3", shape=(1, 1, D_MODEL)),
        ct.TensorType(name="rope_cos", shape=(1, 1, T, D_ROPE)),
        ct.TensorType(name="rope_sin", shape=(1, 1, T, D_ROPE)),
        ct.TensorType(name="mouse", shape=(1, 1, 2)),
        ct.TensorType(name="button", shape=(1, 1, N_BUTTONS)),
        ct.TensorType(name="scroll", shape=(1, 1, 1)),
        ct.TensorType(name="is_cache_write", shape=(1,)),
        ct.TensorType(name="ring_start_local", shape=(1,)),
        ct.TensorType(name="ring_start_global", shape=(1,)),
        ct.TensorType(name="write_step_global", shape=(1,)),
    ]

    states = []
    for i in range(N_LAYERS):
        is_global = ((i - off) % period == 0)
        cap = (global_L if is_global else local_L) + T
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, N_KV_HEADS, cap, D_HEAD)),
            name=f"blocks.{i}.k_cache",
        ))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(1, N_KV_HEADS, cap, D_HEAD)),
            name=f"blocks.{i}.v_cache",
        ))
        states.append(ct.StateType(
            wrapped_type=ct.TensorType(shape=(cap,)),
            name=f"blocks.{i}.written",
        ))

    print(f"[export-unrolled] {len(inputs)} inputs, {len(states)} states")
    print("[export-unrolled] Converting to Core ML...")
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        states=states,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
    )

    print("[export-unrolled] Done.")
    mlmodel.save(out_path)
    print(f"[export-unrolled] Saved to {out_path}")
    return mlmodel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True)
    parser.add_argument("--out", default="diagnostics/out/world_model.mlpackage")
    parser.add_argument("--mode", choices=["single", "unrolled", "both"], default="both")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.mode in ("single", "both"):
        convert_to_coreml(args.model_uri, args.out, device=args.device)
    if args.mode in ("unrolled", "both"):
        unrolled_out = args.out.replace(".mlpackage", "_unrolled.mlpackage")
        convert_unrolled_to_coreml(args.model_uri, unrolled_out, device=args.device)
