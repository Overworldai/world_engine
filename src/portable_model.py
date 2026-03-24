"""
Stateless portable WorldModel for OpenVINO export.
Replaces CUDA-specific ops (FlexAttention, torch.compile, BlockMask)
with standard PyTorch ops (SDPA, dense masks).
KV cache state is externalized as explicit input/output tensors.

Updated for wp-1.5: OrthoRoPEAngles computed once in PortableDiT,
f_pos for KV cache indexing, frame_idx parameter, ts_mult timing.
"""
from typing import Optional, List, Dict

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import einops as eo

from .portable_attn import PortableAttn, PortableCrossAttention, PortableOrthoRoPEAngles
from .stateless_kv import upsert_stateless, make_dense_mask, StatelessKVManager
from .model.nn import rms_norm, AdaLN, NoiseConditioner, NoCastModule, MLP, ada_rmsnorm, ada_gate
from .model.world_model import (
    ControllerInputEmbedding, MLPFusion, CFG, WorldModel, PromptEncoder,
)
from .model.base_model import BaseModel
from .ae import InferenceAE, get_ae


class PortableNoiseConditioner(NoCastModule):
    """NoiseConditioner without torch.autocast('cuda')."""
    def __init__(self, dim, fourier_dim=512, base=10_000.0):
        super().__init__()
        assert fourier_dim % 2 == 0
        half = fourier_dim // 2
        self.freq = nn.Buffer(torch.logspace(0, -1, steps=half, base=base, dtype=torch.float32), persistent=False)
        self.mlp = MLP(fourier_dim, dim * 4, dim)

    def forward(self, s, eps=torch.finfo(torch.float32).eps):
        orig_dtype, shape = s.dtype, s.shape
        s = s.reshape(-1).float()
        s = s * 1000

        phase = s[:, None] * self.freq[None, :]
        emb = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        emb = emb * 2**0.5
        emb = self.mlp(emb)

        return emb.to(orig_dtype).view(*shape, -1)


class PortableCondHead(nn.Module):
    """CondHead — identical to original, no CUDA-specific ops."""
    n_cond = 6

    def __init__(self, config):
        super().__init__()
        self.bias_in = nn.Parameter(torch.zeros(config.d_model)) if config.noise_conditioning == "wan" else None
        self.cond_proj = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model, bias=False) for _ in range(self.n_cond)]
        )

    def forward(self, cond):
        cond = cond + self.bias_in if self.bias_in is not None else cond
        h = F.silu(cond)
        return tuple(p(h) for p in self.cond_proj)


class PortableDiTBlock(nn.Module):
    """DiT block using PortableAttn (SDPA) instead of FlexAttention."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attn = PortableAttn(config, layer_idx)
        self.mlp = MLP(config.d_model, config.d_model * config.mlp_ratio, config.d_model)
        self.cond_head = PortableCondHead(config)

        do_prompt_cond = config.prompt_conditioning is not None and layer_idx % config.prompt_conditioning_period == 0
        self.prompt_cross_attn = PortableCrossAttention(config, config.prompt_embedding_dim) if do_prompt_cond else None
        do_ctrl_cond = config.ctrl_conditioning_period is not None and layer_idx % config.ctrl_conditioning_period == 0
        self.ctrl_mlpfusion = MLPFusion(config) if do_ctrl_cond else None

    def forward(self, x, pos_ids, rope_angles, cond, ctx, v, kv_buf, written, is_frozen):
        """
        Returns: x, v, kv_buf_out, written_out
        """
        s0, b0, g0, s1, b1, g1 = self.cond_head(cond)

        # Self / Causal Attention
        residual = x
        x = ada_rmsnorm(x, s0, b0)
        x, v, kv_buf_out, written_out = self.attn(
            x, pos_ids, rope_angles, v,
            kv_buf=kv_buf, written=written, is_frozen=is_frozen
        )
        x = ada_gate(x, g0) + residual

        # Cross Attention Prompt Conditioning
        if self.prompt_cross_attn is not None:
            x = self.prompt_cross_attn(
                rms_norm(x),
                context=rms_norm(ctx["prompt_emb"]),
                context_pad_mask=ctx["prompt_pad_mask"],
            ) + x

        # MLPFusion Controller Conditioning
        if self.ctrl_mlpfusion is not None:
            x = self.ctrl_mlpfusion(rms_norm(x), rms_norm(ctx["ctrl_emb"])) + x

        # MLP
        x = ada_gate(self.mlp(ada_rmsnorm(x, s1, b1)), g1) + x

        return x, v, kv_buf_out, written_out


class PortableDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([PortableDiTBlock(config, idx) for idx in range(config.n_layers)])
        self.rope_angles = PortableOrthoRoPEAngles(config)

        # Share noise conditioning weights across layers (like original)
        if config.noise_conditioning in ("dit_air", "wan"):
            ref_proj = self.blocks[0].cond_head.cond_proj
            for blk in self.blocks[1:]:
                for blk_mod, ref_mod in zip(blk.cond_head.cond_proj, ref_proj):
                    blk_mod.weight = ref_mod.weight

    def forward(self, x, pos_ids, cond, ctx, kv_bufs, written_bufs, is_frozen):
        """
        Args:
            kv_bufs: list of [2, B, H_kv, capacity_i, Dh] per layer
            written_bufs: list of [capacity_i] bool per layer
            is_frozen: scalar bool tensor
        Returns:
            x, kv_bufs_out, written_bufs_out
        """
        rope_angles = self.rope_angles(pos_ids)
        v = None
        kv_bufs_out = []
        written_bufs_out = []
        for i, block in enumerate(self.blocks):
            x, v, kv_out, written_out = block(
                x, pos_ids, rope_angles, cond, ctx, v,
                kv_buf=kv_bufs[i], written=written_bufs[i], is_frozen=is_frozen
            )
            kv_bufs_out.append(kv_out)
            written_bufs_out.append(written_out)
        return x, kv_bufs_out, written_bufs_out


class PortableWorldModel(nn.Module):
    """
    Stateless WorldModel for OpenVINO export.
    Takes KV cache as explicit input, returns updated KV as output.
    No FlexAttention, no torch.compile, no CUDA autocast.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.tokens_per_frame == config.height * config.width

        self.denoise_step_emb = PortableNoiseConditioner(config.d_model)
        self.ctrl_emb = ControllerInputEmbedding(config)

        if config.ctrl_conditioning is not None:
            self.ctrl_cfg = CFG(config.d_model, config.ctrl_cond_dropout)
        if config.prompt_conditioning is not None:
            self.prompt_cfg = CFG(config.prompt_embedding_dim, config.prompt_cond_dropout)

        self.transformer = PortableDiT(config)

        self.patch = tuple(config.patch)

        C, D = config.channels, config.d_model
        self.patchify = nn.Conv2d(C, D, kernel_size=self.patch, stride=self.patch, bias=False)
        self.unpatchify = nn.Linear(D, C * math.prod(self.patch), bias=True)
        self.out_norm = AdaLN(config.d_model)

        # Cached 1-frame pos_ids
        T = config.tokens_per_frame
        idx = torch.arange(T, dtype=torch.long)
        self.register_buffer("_t_pos_1f", torch.empty(T, dtype=torch.long), persistent=False)
        self.register_buffer("_y_pos_1f", idx.div(config.width, rounding_mode="floor"), persistent=False)
        self.register_buffer("_x_pos_1f", idx.remainder(config.width), persistent=False)

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        frame_timestamp: Tensor,
        kv_bufs: List[Tensor],
        written_bufs: List[Tensor],
        is_frozen: Tensor,
        frame_idx: Optional[Tensor] = None,
        prompt_emb: Optional[Tensor] = None,
        prompt_pad_mask: Optional[Tensor] = None,
        mouse: Optional[Tensor] = None,
        button: Optional[Tensor] = None,
        scroll: Optional[Tensor] = None,
    ):
        """
        Stateless forward pass.

        Args:
            x: [B, N, C, H, W]
            sigma: [B, N]
            frame_timestamp: [B, N] — RoPE time (= frame_idx * ts_mult)
            kv_bufs: list of [2, B, H_kv, capacity_i, Dh] per layer
            written_bufs: list of [capacity_i] bool per layer
            is_frozen: [1] bool tensor
            frame_idx: [B, N] — raw frame counter for KV cache (if None, uses frame_timestamp)
            prompt_emb, prompt_pad_mask: text conditioning
            mouse, button, scroll: controller inputs

        Returns:
            v_pred: [B, N, C, H, W]
            kv_bufs_out: list of updated KV tensors
            written_bufs_out: list of updated written masks
        """
        B, N, C, H, W = x.shape
        ph, pw = self.patch
        Hp, Wp = H // ph, W // pw

        self._t_pos_1f.copy_(frame_timestamp[0, 0].expand_as(self._t_pos_1f))
        f_pos_val = (frame_timestamp if frame_idx is None else frame_idx)[0, 0]
        pos_ids = {
            "f_pos": f_pos_val.expand_as(self._t_pos_1f)[None],
            "t_pos": self._t_pos_1f[None],
            "y_pos": self._y_pos_1f[None],
            "x_pos": self._x_pos_1f[None],
        }

        cond = self.denoise_step_emb(sigma)

        ctx = {
            "ctrl_emb": self.ctrl_emb(mouse, button, scroll),
            "prompt_emb": prompt_emb,
            "prompt_pad_mask": prompt_pad_mask,
        }

        D = self.unpatchify.in_features
        x = self.patchify(x.reshape(B * N, C, H, W))
        x = eo.rearrange(x.view(B, N, D, Hp, Wp), 'b n d hp wp -> b (n hp wp) d')
        x, kv_bufs_out, written_bufs_out = self.transformer(
            x, pos_ids, cond, ctx, kv_bufs, written_bufs, is_frozen
        )
        x = F.silu(self.out_norm(x, cond))
        x = eo.rearrange(
            self.unpatchify(x),
            'b (n hp wp) (c ph pw) -> b n c (hp ph) (wp pw)',
            n=N, hp=Hp, wp=Wp, ph=ph, pw=pw
        )

        return x, kv_bufs_out, written_bufs_out

    @classmethod
    def from_original(cls, original: WorldModel):
        """
        Create a PortableWorldModel from a loaded WorldModel,
        copying all weights.
        """
        config = original.config
        portable = cls(config)

        # Copy denoise step embedder weights
        portable.denoise_step_emb.freq.copy_(original.denoise_step_emb.freq)
        portable.denoise_step_emb.mlp.load_state_dict(original.denoise_step_emb.mlp.state_dict())

        # Copy controller embedding
        portable.ctrl_emb.load_state_dict(original.ctrl_emb.state_dict())

        # Copy CFG modules
        if hasattr(original, 'ctrl_cfg'):
            portable.ctrl_cfg.load_state_dict(original.ctrl_cfg.state_dict())
        if hasattr(original, 'prompt_cfg'):
            portable.prompt_cfg.load_state_dict(original.prompt_cfg.state_dict())

        # Copy patchify / unpatchify
        portable.patchify.load_state_dict(original.patchify.state_dict())
        portable.unpatchify.load_state_dict(original.unpatchify.state_dict())
        portable.out_norm.load_state_dict(original.out_norm.state_dict())

        # Copy rope_angles buffers from original transformer
        portable.transformer.rope_angles.xy.copy_(original.transformer.rope_angles.xy)
        portable.transformer.rope_angles.inv_t.copy_(original.transformer.rope_angles.inv_t)

        # Copy transformer blocks
        for port_blk, orig_blk in zip(portable.transformer.blocks, original.transformer.blocks):
            # Attention weights
            port_blk.attn.q_proj.load_state_dict(orig_blk.attn.q_proj.state_dict())
            port_blk.attn.k_proj.load_state_dict(orig_blk.attn.k_proj.state_dict())
            port_blk.attn.v_proj.load_state_dict(orig_blk.attn.v_proj.state_dict())
            port_blk.attn.out_proj.load_state_dict(orig_blk.attn.out_proj.state_dict())

            if port_blk.attn.value_residual:
                port_blk.attn.v_lamb.data.copy_(orig_blk.attn.v_lamb.data)
            if port_blk.attn.gated_attn:
                port_blk.attn.gate_proj.load_state_dict(orig_blk.attn.gate_proj.state_dict())

            # Cond head
            for port_cp, orig_cp in zip(port_blk.cond_head.cond_proj, orig_blk.cond_head.cond_proj):
                port_cp.load_state_dict(orig_cp.state_dict())
            if port_blk.cond_head.bias_in is not None:
                port_blk.cond_head.bias_in.data.copy_(orig_blk.cond_head.bias_in.data)

            # MLP
            port_blk.mlp.load_state_dict(orig_blk.mlp.state_dict())

            # Cross attention
            if port_blk.prompt_cross_attn is not None:
                port_blk.prompt_cross_attn.q_proj.load_state_dict(orig_blk.prompt_cross_attn.q_proj.state_dict())
                port_blk.prompt_cross_attn.k_proj.load_state_dict(orig_blk.prompt_cross_attn.k_proj.state_dict())
                port_blk.prompt_cross_attn.v_proj.load_state_dict(orig_blk.prompt_cross_attn.v_proj.state_dict())
                port_blk.prompt_cross_attn.out_proj.load_state_dict(orig_blk.prompt_cross_attn.out_proj.state_dict())

            # Controller MLPFusion
            if port_blk.ctrl_mlpfusion is not None:
                port_blk.ctrl_mlpfusion.load_state_dict(orig_blk.ctrl_mlpfusion.state_dict())

        # Copy position buffers
        portable._t_pos_1f.copy_(original._t_pos_1f)
        portable._y_pos_1f.copy_(original._y_pos_1f)
        portable._x_pos_1f.copy_(original._x_pos_1f)

        return portable
