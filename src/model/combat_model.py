from typing import Optional, List
from torch import Tensor

import einops as eo
from tensordict import TensorDict
import math

import torch
from torch import nn
import torch.nn.functional as F

from .. import nn as owl_nn
from .base_model import BaseModel
from .world_model import WorldDiT

class ControllerInputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = owl_nn.MLPCustom(
            config.n_buttons,
            config.d_model * config.mlp_ratio,
            config.d_model
        )

    def forward(self, button: Tensor):
        return self.mlp(button)


class CFG(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.null_emb = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor, is_conditioned: Optional[bool] = None) -> torch.Tensor:
        """
        x: [B, L, D]
        is_conditioned:
          - None: training-style random dropout
          - bool: whole batch conditioned / unconditioned at sampling
        """
        B, L, _ = x.shape
        null = self.null_emb.expand(B, L, -1)

        # training-style dropout OR unspecified
        if self.training or is_conditioned is None:
            if self.dropout == 0.0:
                return x
            drop = torch.rand(B, 1, 1, device=x.device) < self.dropout  # [B,1,1]
            return torch.where(drop, null, x)

        # sampling-time switch
        return x if is_conditioned else null


class MLPFusion(nn.Module):
    """Fuses per-group conditioning into tokens by applying an MLP to cat([x, cond])"""
    def __init__(self, config):
        super().__init__()
        self.D = config.d_model
        self.mlp = owl_nn.MLPCustom(2 * self.D, self.D, self.D)
        self.mlp.fc2.weight.detach().zero_()  # same init behavior as your original

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, _, D = x.shape
        L = cond.shape[1]

        Wx, Wc = self.mlp.fc1.weight.chunk(2, dim=1)  # each [D, D]

        x = x.view(B, L, -1, D)
        h = F.linear(x, Wx) + F.linear(cond, Wc).unsqueeze(2)  # broadcast, no repeat/cat
        h = F.silu(h)
        y = F.linear(h, self.mlp.fc2.weight)
        return y.flatten(1, 2)


class CondHead(nn.Module):
    """Per-layer conditioning head: bias_in → SiLU → Linear → chunk(n_cond)."""
    n_cond = 6

    def __init__(self, config):
        super().__init__()
        self.bias_in = nn.Parameter(torch.zeros(config.d_model)) if config.noise_conditioning == "wan" else None
        self.cond_proj = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model, bias=False) for _ in range(self.n_cond)]
        )

        # AdaLN-Zero
        if self.bias_in is not None:
            self.bias_in.detach().zero_()
        for p in self.cond_proj:
            p.weight.detach().zero_()

    def forward(self, cond):
        cond = cond + self.bias_in if self.bias_in is not None else cond
        h = F.silu(cond)
        return tuple(p(h) for p in self.cond_proj)


class WorldDiTBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.attn = owl_nn.Attn(config, layer_idx)
        self.mlp = owl_nn.MLP(config)
        self.cond_head = CondHead(config)

        # cross attn
        do_prompt_cond = (config.prompt_conditioning is not None) and (layer_idx % config.prompt_conditioning_period == 0)
        self.prompt_cross_attn = owl_nn.CrossAttention(config, config.prompt_embedding_dim) if do_prompt_cond else None

        self.ctrl_cross_attn = self.ctrl_mlpfusion = None
        if layer_idx % config.ctrl_conditioning_period == 0:
            if config.ctrl_conditioning == "cross_attention_same_frame":
                self.ctrl_cross_attn = owl_nn.CrossAttentionSameFrame(config)
            elif config.ctrl_conditioning == "cross_attn_lookback":
                self.ctrl_cross_attn = owl_nn.CrossAttentionLookBack(config)
            elif config.ctrl_conditioning == "mlp_fusion":
                self.ctrl_mlpfusion = MLPFusion(config)

    def forward(self, x, pos_ids, cond, ctx, v, kv_cache=None, bm=None):
        """
        0) Causal Frame Attention
        1) Frame->CTX
        2) MLP
        """
        do_mlp_ckpt = self.training and getattr(self.config, "mlp_gradient_checkpointing", False)

        s0, b0, g0, s1, b1, g1 = self.cond_head(cond)

        residual = x
        x = owl_nn.ada_rmsnorm(x, s0, b0)
        x, v = self.attn(x, pos_ids, v, bm, kv_cache=kv_cache)
        x = owl_nn.ada_gate(x, g0) + residual

        if self.prompt_cross_attn is not None:
            def prompt_xattn(xm, pe, pm):
                residual = xm
                per_frame = ctx.get("prompt_per_frame", False)
                if per_frame:
                    B, S, D = xm.shape
                    tpf = self.config.tokens_per_frame
                    torch._assert(S % tpf == 0, "bad tpf")
                    xm = xm.reshape(B * (S // tpf), tpf, D)
                out = self.prompt_cross_attn(
                    owl_nn.rms_norm(xm),
                    context=owl_nn.rms_norm(pe),
                    context_pad_mask=pm,
                )
                if per_frame:
                    out = out.reshape(B, S, D)
                return out + residual

            x = owl_nn.maybe_ckpt(do_mlp_ckpt, prompt_xattn, x, ctx["prompt_emb"], ctx["prompt_pad_mask"])

        if self.ctrl_cross_attn is not None:
            inference_mode = kv_cache is not None
            if self.config.ctrl_conditioning == "cross_attn_lookback":
                x = self.ctrl_cross_attn(
                    owl_nn.rms_norm(x), owl_nn.rms_norm(ctx["ctrl_emb"]), pos_ids, inference_mode
                ) + x
            else:
                x = self.ctrl_cross_attn(owl_nn.rms_norm(x), owl_nn.rms_norm(ctx["ctrl_emb"]), inference_mode) + x
        if self.ctrl_mlpfusion is not None:
            def ctrl_mlpfusion(x_raw, ctrl_raw):
                return self.ctrl_mlpfusion(owl_nn.rms_norm(x_raw), owl_nn.rms_norm(ctrl_raw)) + x_raw
            x = owl_nn.maybe_ckpt(do_mlp_ckpt, ctrl_mlpfusion, x, ctx["ctrl_emb"])

        def cond_mlp(xm, sm, bm, gm):
            residual = xm
            xm = self.mlp(owl_nn.ada_rmsnorm(xm, sm, bm))
            return owl_nn.ada_gate(xm, gm) + residual

        x = owl_nn.maybe_ckpt(do_mlp_ckpt, cond_mlp, x, s1, b1, g1)

        return x, v


class CombatDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([WorldDiTBlock(config, idx) for idx in range(config.n_layers)])

        if self.config.noise_conditioning in ("dit_air", "wan"):
            ref_proj = self.blocks[0].cond_head.cond_proj
            for blk in self.blocks[1:]:
                for blk_mod, ref_mod in zip(blk.cond_head.cond_proj, ref_proj):
                    blk_mod.weight = ref_mod.weight

        # Shared RoPE buffers
        ref_rope = self.blocks[0].attn.rope
        for blk in self.blocks[1:]:
            blk.attn.rope = ref_rope

    def get_block_masks(self, x, pos_ids, doc_id=None, curr_frame_mask=None, kv_cache=None):
        """Return (local_bm, global_bm). If kv_cache, both None. If merged, both same."""
        if kv_cache is not None:
            return None, None

        global_period = getattr(self.config, "global_attn_period", 4)
        global_offset = getattr(self.config, "global_attn_offset", 0)
        off = global_offset % global_period

        # merged => one mask reused everywhere
        if getattr(self.config, "global_local_merged", False):
            bm = owl_nn.make_attn_block_mask(
                self.config, 0, x.size(1), pos_ids, doc_id, curr_frame_mask, device=x.device
            )
            return bm, bm

        # not merged => build exactly two masks once
        bm_global = owl_nn.make_attn_block_mask(
            self.config, off, x.size(1), pos_ids, doc_id, curr_frame_mask, device=x.device
        )
        bm_local = bm_global if global_period == 1 else owl_nn.make_attn_block_mask(
            self.config, off + 1, x.size(1), pos_ids, doc_id, curr_frame_mask, device=x.device
        )
        return bm_local, bm_global

    def forward(self, x, pos_ids, cond, ctx, doc_id=None, curr_frame_mask=None, kv_cache=None):
        ckpt_mode = getattr(self.config, "block_gradient_checkpointing", False)
        ckpt_mode = ckpt_mode if (self.training and kv_cache is None) else False

        bm_local, bm_global = self.get_block_masks(x, pos_ids, doc_id, curr_frame_mask, kv_cache)

        global_period = getattr(self.config, "global_attn_period", 4)
        global_offset = getattr(self.config, "global_attn_offset", 0)
        off = global_offset % global_period

        v = None
        for i, block in enumerate(self.blocks):
            bm = None if kv_cache is not None else (bm_global if ((i - off) % global_period) == 0 else bm_local)

            def run_block(xm, vm, cm, block=block, bm=bm):
                return block(
                    xm, pos_ids, cm, ctx, vm,
                    bm=bm,
                    kv_cache=kv_cache,
                )
            x, v = owl_nn.maybe_ckpt(ckpt_mode, run_block, x, v, cond)

        return x


class CombatModel(BaseModel):
    """
    WORLD: Wayfarer Operator-driven Rectified-flow Long-context Diffuser

    Denoise a frame given
    - All previous frames
    - The prompt embedding
    - The controller input embedding
    - The current noise level
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        assert config.tokens_per_frame == config.height * config.width

        self.denoise_step_emb = owl_nn.NoiseConditioner(config.d_model)
        self.ctrl_emb = ControllerInputEmbedding(config)

        if config.ctrl_conditioning is not None:
            self.ctrl_cfg = CFG(config.d_model, config.ctrl_cond_dropout)
        if config.prompt_conditioning is not None:
            self.prompt_cfg = CFG(config.prompt_embedding_dim, config.prompt_cond_dropout)

        self.transformer = CombatDiT(config)

        self.patch = tuple(getattr(config, "patch", (1, 1)))

        C, D = config.channels, config.d_model
        self.patchify = nn.Conv2d(C, D, kernel_size=self.patch, stride=self.patch, bias=False)  # TODO
        self.unpatchify = nn.Linear(D, C * math.prod(self.patch), bias=True)
        self.out_norm = owl_nn.AdaLN(config.d_model)

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        frame_timestamp: Tensor,
        button: Optional[Tensor] = None,
        mouse: Optional[Tensor] = None,
        doc_id: Optional[Tensor] = None,
        kv_cache=None,
        curr_frame_mask: Optional[Tensor] = None,
        ctrl_cond: Optional[bool] = None,
        prompt_cond: Optional[bool] = None,
        frame_rgbd_mask: Optional[Tensor] = None
    ):
        """
        x: [B, N, C, H, W],
        sigma: [B, N]
        frame_timestamp: [B, N]
        prompt_emb: [B, P, D]
        controller_inputs: [B, N, I]
        doc_id: [B, N]
        ctrl_cond: Inference only, whether to apply controller conditioning
        """
        B, N, Cin, H, W = x.shape
        C = self.config.channels
        ph, pw = self.patch
        assert (H % ph == 0) and (W % pw == 0), "H, W must be divisible by patch"
        Hp, Wp = H // ph, W // pw
        torch._assert(Hp * Wp == self.config.tokens_per_frame, f"{Hp} * {Wp} != {self.config.tokens_per_frame}")

        pos_ids = self.get_pos_ids(frame_timestamp, Hp, Wp)

        if curr_frame_mask is not None:
            torch._assert(curr_frame_mask.size(1) == N, "curr_frame_mask must be frame-length")
            curr_frame_mask = curr_frame_mask.repeat_interleave(Hp * Wp, 1)

        assert doc_id is None or kv_cache is None, "Cannot use sequence packing with kv caching"
        if doc_id is not None:
            doc_id = doc_id.repeat_interleave(Hp * Wp, dim=1)

        # embed
        cond = self.denoise_step_emb(sigma)  # [B, N, d]

        ctx = {}
        if button is not None:
            ctx["ctrl_emb"] = self.ctrl_cfg(self.ctrl_emb(button), ctrl_cond)
        x_in = x.reshape(B * N, Cin, H, W)
        # if self.videojam_depth:
        #     torch._assert((Cin == C) or (Cin == 2 * C), "videojam_depth=True: expected Cin==C (rgb) or Cin==2C (rgbd)")
        #     if Cin == C:
        #         x = self.patchify(x_in)
        #     else:
        #         rgb = x_in[:, :C]
        #         dep = x_in[:, C:2 * C]
        #         x_rgb = self.patchify(rgb)
        #         x_rgbd = self.patchify_rgb(rgb) + self.patchify_depth(dep)
        #         m = 1.0 if frame_rgbd_mask is None else frame_rgbd_mask.to(dtype=x_rgbd.dtype, device=x_rgbd.device).reshape(B * N, 1, 1, 1)
        #         x = x_rgbd * m + x_rgb * (1.0 - m)
        # else:
        torch._assert(Cin == C, "Expected RGB-only input (Cin==C)")
        x = self.patchify(x_in)

        D = self.unpatchify.in_features
        x = eo.rearrange(x.view(B, N, D, Hp, Wp), 'b n d hp wp -> b (n hp wp) d')
        x = self.transformer(x, pos_ids, cond, ctx, doc_id, curr_frame_mask, kv_cache)
        x = F.silu(self.out_norm(x, cond))
        x_rgb = eo.rearrange(
            self.unpatchify(x),
            'b (n hp wp) (c ph pw) -> b n c (hp ph) (wp pw)',
            n=N, hp=Hp, wp=Wp, ph=ph, pw=pw
        )

        # if self.videojam_depth:
        #     x_dep = eo.rearrange(
        #         self.unpatchify_depth(x),
        #         'b (n hp wp) (c ph pw) -> b n c (hp ph) (wp pw)',
        #         n=N, hp=Hp, wp=Wp, ph=ph, pw=pw
        #     )
        #     return torch.cat([x_rgb, x_dep], dim=2)

        return x_rgb

    def get_frame_timestamps(self, fps: torch.Tensor, num_frames: int, device):
        assert fps.dim() == 1 and fps.dtype == torch.long
        if not (self.config.base_fps % fps).eq(0).all():
            raise ValueError(f"base_fps={int(self.config.base_fps)} seen_fps={torch.unique(fps.detach().cpu()).tolist()}")
        scale = (self.config.base_fps // fps).unsqueeze(1)
        return torch.arange(num_frames, device=device).unsqueeze(0) * scale.to(device=device)

    @staticmethod
    def get_pos_ids(seq_ts: torch.Tensor, H: int, W: int) -> TensorDict:
        """Positions for [B, F*H*W]; seq_ts is [B,F] (long)."""
        B, F = seq_ts.shape
        device = seq_ts.device
        y = torch.arange(H, device=device, dtype=torch.long).repeat_interleave(W)  # [H*W]
        x = torch.arange(W, device=device, dtype=torch.long).repeat(H)             # [H*W]
        return TensorDict(
            {
                "t_pos": seq_ts.repeat_interleave(H * W, 1),  # [B,F*H*W]
                "y_pos": y.repeat(F).expand(B, -1),           # [B,F*H*W]
                "x_pos": x.repeat(F).expand(B, -1),           # [B,F*H*W]
            },
            batch_size=[B, F * H * W],
        )

    def fwd_cfg(self, *args, cfg_scales: Optional[dict[str, float]] = None, novel_cfg=False, **kwargs):
        fn = self.fwd_novel_cfg if novel_cfg else self.fwd_basic_cfg
        return fn(*args, cfg_scales=cfg_scales, **kwargs)

    def fwd_basic_cfg(self, *args, cfg_scales: Optional[dict[str, float]] = None, **kwargs):
        if cfg_scales is None:
            return self.forward(*args, **kwargs)

        # don't let the caller manually pass the modality flags we're going to control
        if any(k in kwargs for k in cfg_scales):
            raise ValueError("cfg_scales keys must be controlled by fwd_cfg")

        # null / unconditional: all modalities off
        null_flags = {k: False for k in cfg_scales}
        v_uncond = self.forward(*args, **null_flags, **kwargs)

        v = v_uncond
        for name, scale in cfg_scales.items():
            cond_flags = dict(null_flags)
            cond_flags[name] = True
            v_cond = self.forward(*args, **cond_flags, **kwargs)
            v = v + (v_cond - v_uncond) * scale  # CFG contribution for this modality

        return v

    def fwd_novel_cfg(self, *args, cfg_scales: Optional[dict[str, float]] = None, **kwargs):
        if cfg_scales is None:
            return self.forward(*args, **kwargs)

        # don't let the caller manually pass the modality flags we're going to control
        if any(k in kwargs for k in cfg_scales):
            raise ValueError("cfg_scales keys must be controlled by fwd_novel_cfg")

        # null / unconditional: all modalities off
        null_flags = {k: False for k in cfg_scales}
        v_uncond = self.forward(*args, **null_flags, **kwargs)

        # all modalities on
        all_flags = {k: True for k in cfg_scales}
        v_all = self.forward(*args, **all_flags, **kwargs)

        v = v_uncond
        for name, scale in cfg_scales.items():
            not_flags = dict(all_flags)
            not_flags[name] = False
            v_not = self.forward(*args, **not_flags, **kwargs)
            v = v + (v_all - v_not) * scale  # novel CFG contribution for this modality

        return v
