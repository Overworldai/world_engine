from typing import Optional, List
from torch import Tensor

import einops as eo
from tensordict import TensorDict
import math

import torch
from torch import nn
import torch.nn.functional as F

from .attn import Attn, CrossAttention, OrthoRoPEAngles
from .nn import AdaLN, ada_gate, ada_rmsnorm, NoiseConditioner
from .base_model import BaseModel


class PromptEncoder(nn.Module):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    """Callable for text -> UMT5 embedding"""
    def __init__(self, model_id="google/umt5-xl", dtype=torch.bfloat16):
        from transformers import AutoTokenizer, UMT5EncoderModel
        super().__init__()
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.encoder = UMT5EncoderModel.from_pretrained(model_id, torch_dtype=dtype).eval()

    @torch.compile
    def encode(self, inputs):
        return self.encoder(**inputs).last_hidden_state

    @torch.inference_mode()
    def forward(self, texts: List[str]):
        import ftfy
        texts = [ftfy.fix_text(t) for t in texts]
        inputs = self.tok(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(self.encoder.device)
        attn = inputs["attention_mask"]                      # [B,S] 1=token 0=pad
        emb = self.encode(inputs).to(self.dtype)
        emb = emb * attn.unsqueeze(-1).type_as(emb)          # preserve old behavior
        pad_mask = attn.eq(0)
        return emb, pad_mask


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


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


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.moe_top_k

        d_intermediate = int(config.d_model * config.mlp_ratio / self.top_k)
        self.router = nn.Linear(config.d_model, config.moe_n_experts, bias=False)
        self.expert_in = nn.Parameter(torch.empty(config.moe_n_experts, d_intermediate, config.d_model))
        self.expert_out = nn.Parameter(torch.empty(config.moe_n_experts, config.d_model, d_intermediate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # grouped_mm requires bf16 inputs and breaks w/ autocast
        assert x.is_cuda and not torch.is_autocast_enabled(x.device.type)
        assert x.dtype == self.expert_in.dtype == self.expert_out.dtype == torch.bfloat16

        orig_shape = x.shape
        x = x.reshape(-1, orig_shape[-1])

        # Route tokens to top_k experts and norm weights (must add to 1.0)
        scores, expert = self.router(x).topk(self.top_k, dim=-1, sorted=False)
        weights = (scores - scores.logsumexp(-1, True)).exp()

        # Sort token/expert into expert-major order for grouped_mm
        expert_sorted, sort_idx = expert.flatten().sort()
        counts = expert_sorted.new_zeros(self.expert_in.size(0), dtype=torch.int32)
        counts.scatter_add_(0, expert_sorted, torch.ones_like(expert_sorted, dtype=torch.int32))
        offsets = torch.cumsum(counts, 0, dtype=torch.int32)

        # Pack tokens by expert + append one dummy row for grouped_mm.
        src = sort_idx // self.top_k
        x_grouped = F.pad(x.index_select(0, src), (0, 0, 0, 1))

        # Run the expert MLPs as two grouped GEMMs
        h = F.grouped_mm(x_grouped, self.expert_in.transpose(-2, -1), offs=offsets)
        h[-1].zero_()  # ensure last row initialized
        h = F.silu(h)
        y_grp = F.grouped_mm(h, self.expert_out.transpose(-2, -1), offs=offsets)[:-1]

        # Restore token-major order and combine expert outputs
        y = torch.empty_like(y_grp).index_copy_(0, sort_idx, y_grp).view(x.size(0), self.top_k, -1)
        return (y.float() * weights.unsqueeze(-1)).sum(dim=1).type_as(x).reshape(orig_shape)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_middle, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_middle, bias=False)
        self.fc2 = nn.Linear(dim_middle, dim_out, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class ControllerInputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(
            config.n_buttons + 3,  # mouse velocity (x,y) + scroll sign
            config.d_model * config.mlp_ratio,
            config.d_model
        )

    def forward(self, mouse: Tensor, button: Tensor, scroll: Tensor):
        x = torch.cat((mouse, button, scroll), dim=-1)
        return self.mlp(x)


class MLPFusion(nn.Module):
    """Fuses per-group conditioning into tokens by applying an MLP to cat([x, cond])"""
    def __init__(self, config):
        super().__init__()
        self.mlp = MLP(2 * config.d_model, config.d_model, config.d_model)

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

    def forward(self, cond):
        cond = cond + self.bias_in if self.bias_in is not None else cond
        h = F.silu(cond)
        return tuple(p(h) for p in self.cond_proj)


class WorldDiTBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.attn = Attn(config, layer_idx)
        if getattr(config, "moe", False):
            self.mlp = MoE(config)
        else:
            self.mlp = MLP(config.d_model, config.d_model * config.mlp_ratio, config.d_model)
        self.cond_head = CondHead(config)

        do_prompt_cond = config.prompt_conditioning and layer_idx % config.prompt_conditioning_period == 0
        self.prompt_cross_attn = CrossAttention(config, config.prompt_embedding_dim) if do_prompt_cond else None
        do_ctrl_cond = config.ctrl_conditioning_period is not None and layer_idx % config.ctrl_conditioning_period == 0
        self.ctrl_mlpfusion = MLPFusion(config) if do_ctrl_cond else None

    def forward(self, x, pos_ids, rope_angles, cond, ctx, v, kv_cache=None):
        """
        0) Causal Frame Attention
        1) Frame->CTX Cross Attention
        2) MLP
        """
        s0, b0, g0, s1, b1, g1 = self.cond_head(cond)

        # Self / Causal Attention
        residual = x
        x = ada_rmsnorm(x, s0, b0)
        x, v = self.attn(x, pos_ids, rope_angles, v, kv_cache=kv_cache)
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

        return x, v


class WorldDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([WorldDiTBlock(config, idx) for idx in range(config.n_layers)])
        self.rope_angles = OrthoRoPEAngles(config)

        if self.config.noise_conditioning in ("dit_air", "wan"):
            ref_proj = self.blocks[0].cond_head.cond_proj
            for blk in self.blocks[1:]:
                for blk_mod, ref_mod in zip(blk.cond_head.cond_proj, ref_proj):
                    blk_mod.weight = ref_mod.weight

    def forward(self, x, pos_ids, cond, ctx, kv_cache=None):
        rope_angles = self.rope_angles(pos_ids)
        v = None
        for i, block in enumerate(self.blocks):
            x, v = block(x, pos_ids, rope_angles, cond, ctx, v, kv_cache=kv_cache)
        return x


class WorldModel(BaseModel):
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

        self.denoise_step_emb = NoiseConditioner(config.d_model)
        self.ctrl_emb = ControllerInputEmbedding(config)

        if config.ctrl_conditioning is not None:
            self.ctrl_cfg = CFG(config.d_model, config.ctrl_cond_dropout)
        if config.prompt_conditioning is not None:
            self.prompt_cfg = CFG(config.prompt_embedding_dim, config.prompt_cond_dropout)

        self.transformer = WorldDiT(config)

        self.patch = tuple(config.patch)

        C, D = config.channels, config.d_model
        self.patchify = nn.Conv2d(C, D, kernel_size=self.patch, stride=self.patch, bias=False)
        self.unpatchify = nn.Linear(D, C * math.prod(self.patch), bias=True)
        self.out_norm = AdaLN(config.d_model)

        # Cached 1-frame pos_ids (buffers + cached TensorDict view)
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
        frame_idx: Optional[Tensor] = None,
        prompt_emb: Optional[Tensor] = None,
        prompt_pad_mask: Optional[Tensor] = None,
        mouse: Optional[Tensor] = None,
        button: Optional[Tensor] = None,
        scroll: Optional[Tensor] = None,
        kv_cache=None,
        ctrl_cond: Optional[bool] = None,
        prompt_cond: Optional[bool] = None,
    ):
        """
        x: [B, N, C, H, W],
        sigma: [B, N]
        frame_timestamp: [B, N]
        prompt_emb: [B, P, D]
        controller_inputs: [B, N, I]
        ctrl_cond: Inference only, whether to apply controller conditioning
        """
        B, N, C, H, W = x.shape
        ph, pw = self.patch
        assert (H % ph == 0) and (W % pw == 0), "H, W must be divisible by patch"
        Hp, Wp = H // ph, W // pw
        torch._assert(Hp * Wp == self.config.tokens_per_frame, f"{Hp} * {Wp} != {self.config.tokens_per_frame}")

        torch._assert(B == 1 and N == 1, "WorldModel.forward currently supports B==1, N==1")
        self._t_pos_1f.copy_(frame_timestamp[0, 0].expand_as(self._t_pos_1f))
        pos_ids = TensorDict(
            {
                "f_pos": (frame_timestamp if frame_idx is None else frame_idx)[0, 0].expand_as(self._t_pos_1f)[None],
                "t_pos": self._t_pos_1f[None],
                "y_pos": self._y_pos_1f[None],
                "x_pos": self._x_pos_1f[None],
            },
            batch_size=[1, self._t_pos_1f.numel()],
        )

        cond = self.denoise_step_emb(sigma)  # [B, N, d]

        assert button is not None
        ctx = {
            "ctrl_emb": self.ctrl_emb(mouse, button, scroll),
            "prompt_emb": prompt_emb,
            "prompt_pad_mask": prompt_pad_mask,
        }

        D = self.unpatchify.in_features
        x = self.patchify(x.reshape(B * N, C, H, W))
        x = eo.rearrange(x.view(B, N, D, Hp, Wp), 'b n d hp wp -> b (n hp wp) d')
        x = self.transformer(x, pos_ids, cond, ctx, kv_cache)
        x = F.silu(self.out_norm(x, cond))
        x = eo.rearrange(
            self.unpatchify(x),
            'b (n hp wp) (c ph pw) -> b n c (hp ph) (wp pw)',
            n=N, hp=Hp, wp=Wp, ph=ph, pw=pw
        )

        return x

    def get_active_parameters(self) -> int:
        total = sum(p.numel() for p in self.parameters())
        c = self.config
        if getattr(c, "moe", False):
            moe_mlp_ratio = getattr(c, "moe_mlp_ratio", None) or c.mlp_ratio / c.moe_top_k
            hidden, top_k = int(c.d_model * moe_mlp_ratio), min(c.moe_top_k, c.moe_n_experts)
            total -= (c.moe_n_experts - top_k) * c.n_layers * c.d_model * hidden * 2
        return total

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.config.model_type != "waypoint-1.5":
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        state_dict = dict(state_dict)

        if "unpatchify.weight" in state_dict and state_dict["unpatchify.weight"].ndim == 4:
            w = state_dict["unpatchify.weight"]  # [D, C, ph, pw]
            state_dict["unpatchify.weight"] = w.permute(1, 2, 3, 0).reshape(-1, w.shape[0])
        if "unpatchify.bias" in state_dict and state_dict["unpatchify.bias"].numel() != self.unpatchify.bias.numel():
            ph, pw = self.patch
            state_dict["unpatchify.bias"] = state_dict["unpatchify.bias"][:, None, None].expand(-1, ph, pw).reshape(-1)

        for i in range(self.config.n_layers):
            p = f"transformer.blocks.{i}."
            for name in ("fc1.weight", "fc2.weight", "expert_in", "expert_out", "router.weight"):
                old = p + "dit_mlp." + name
                if old in state_dict:
                    state_dict.setdefault(p + "mlp." + name, state_dict.pop(old))

            attn_bias = state_dict.pop(p + "attn_cond_head.bias_in", None)
            mlp_bias = state_dict.pop(p + "mlp_cond_head.bias_in", None)
            if attn_bias is not None or mlp_bias is not None:
                state_dict.setdefault(p + "cond_head.bias_in", mlp_bias if mlp_bias is not None else attn_bias)

            for j in range(3):
                attn = state_dict.pop(p + f"attn_cond_head.cond_proj.{j}.weight", None)
                mlp = state_dict.pop(p + f"mlp_cond_head.cond_proj.{j}.weight", None)
                if attn is not None:
                    state_dict.setdefault(p + f"cond_head.cond_proj.{j}.weight", attn)
                if mlp is not None:
                    state_dict.setdefault(p + f"cond_head.cond_proj.{j + 3}.weight", mlp)

            x = state_dict.pop(p + "ctrl_mlpfusion.fc1_x.weight", None)
            c = state_dict.pop(p + "ctrl_mlpfusion.fc1_c.weight", None)
            if x is not None and c is not None:
                state_dict.setdefault(p + "ctrl_mlpfusion.mlp.fc1.weight", torch.cat((x, c), dim=1))
            old = state_dict.pop(p + "ctrl_mlpfusion.fc2.weight", None)
            if old is not None:
                state_dict.setdefault(p + "ctrl_mlpfusion.mlp.fc2.weight", old)

        ref = "transformer.blocks.0.cond_head.cond_proj."
        for i in range(1, self.config.n_layers):
            p = f"transformer.blocks.{i}.cond_head.cond_proj."
            for j in range(6):
                k, rk = p + f"{j}.weight", ref + f"{j}.weight"
                if k not in state_dict and rk in state_dict:
                    state_dict[k] = state_dict[rk]
        state_dict = {k: v for k, v in state_dict.items() if ".cond_heads." not in k}

        return super().load_state_dict(state_dict, strict=strict, assign=assign)
