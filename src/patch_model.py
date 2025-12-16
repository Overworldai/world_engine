import torch
from torch import nn, Tensor

from owl_wms.nn import Attn, rms_norm
from torch.nn.attention.flex_attention import flex_attention


def _bf16_u16(x: Tensor) -> Tensor:
    # reinterpret bf16 storage as int16 -> unsigned 0..65535 in int32
    return x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


class CachedDenoiseStepEmb(nn.Module):
    """bf16 sigma -> bf16 embedding via 64k LUT; invalid sigma => OOB index error (no silent wrong)."""
    def __init__(self, base: nn.Module, sigmas: list[float]):
        super().__init__()
        device = next(base.parameters()).device

        levels = torch.tensor(sigmas, device=device, dtype=torch.bfloat16)          # [S]
        bits = _bf16_u16(levels)                                                   # [S]
        if torch.unique(bits).numel() != bits.numel():
            raise ValueError("scheduler_sigmas collide in bf16; caching would be ambiguous")

        with torch.no_grad():
            table = base(levels[:, None]).squeeze(1).to(torch.bfloat16).contiguous()  # [S,D]

        lut = torch.full((65536,), -1, device=device, dtype=torch.int32)
        lut[bits] = torch.arange(bits.numel(), device=device, dtype=torch.int32)

        self.register_buffer("table", table, persistent=False)                     # [S,D] bf16
        self.register_buffer("lut", lut, persistent=False)                         # [65536] int32
        self.register_buffer("oob", torch.tensor(bits.numel(), device=device, dtype=torch.int32), persistent=False)

    def forward(self, sigma: Tensor) -> Tensor:
        if sigma.dtype is not torch.bfloat16:
            raise RuntimeError("CachedDenoiseStepEmb expects sigma bf16")
        idx = self.lut[_bf16_u16(sigma)]
        idx = torch.where(idx >= 0, idx, self.oob)                                 # invalid -> S (OOB)
        return self.table[idx.to(torch.int64)]                                     # [...,D] bf16


class CachedCondHead(nn.Module):
    """bf16 cond -> cached (s0,b0,g0,s1,b1,g1); invalid cond => OOB index error (no silent wrong)."""
    def __init__(self, base, cached_denoise_step_emb: CachedDenoiseStepEmb, max_key_dims: int = 8):
        super().__init__()
        table = cached_denoise_step_emb.table                                      # [S,D] bf16
        S, D = table.shape

        with torch.no_grad():
            emb = table[:, None, :]                                                # [S,1,D]
            cache = torch.stack([t.squeeze(1) for t in base(emb)], 0).to(torch.bfloat16).contiguous()  # [6,S,D]

        # pick a single embedding dimension whose bf16 bits uniquely identify sigma
        key_dim = None
        for d in range(min(D, max_key_dims)):
            b = _bf16_u16(table[:, d])
            if torch.unique(b).numel() == S:
                key_dim = d
                key_bits = b
                break
        if key_dim is None:
            raise ValueError("Could not find a unique bf16 key dim for cond->sigma mapping; increase max_key_dims")

        lut = torch.full((65536,), -1, device=table.device, dtype=torch.int32)
        lut[key_bits] = torch.arange(S, device=table.device, dtype=torch.int32)

        self.key_dim = int(key_dim)
        self.register_buffer("cache", cache, persistent=False)                     # [6,S,D] bf16
        self.register_buffer("lut", lut, persistent=False)                         # [65536] int32
        self.register_buffer("oob", torch.tensor(S, device=table.device, dtype=torch.int32), persistent=False)

    def forward(self, cond: Tensor):
        if cond.dtype is not torch.bfloat16:
            raise RuntimeError("CachedCondHead expects cond bf16")
        idx = self.lut[_bf16_u16(cond[..., self.key_dim])]
        idx = torch.where(idx >= 0, idx, self.oob)                                 # invalid -> S (OOB)
        g = self.cache[:, idx.to(torch.int64)]                                     # [6,...,D] bf16 (or errors)
        return tuple(g.unbind(0))                                                  # (s0,b0,g0,s1,b1,g1)


def patch_cached_noise_conditioning(model) -> None:
    # Call AFTER: model.to(device="cuda", dtype=torch.bfloat16).eval()
    cached_denoise_step_emb = CachedDenoiseStepEmb(model.denoise_step_emb, model.config.scheduler_sigmas)
    model.denoise_step_emb = cached_denoise_step_emb
    for blk in model.transformer.blocks:
        blk.cond_head = CachedCondHead(blk.cond_head, cached_denoise_step_emb)


class MergedQKVAttn(Attn):
    def __init__(self, src: Attn, config):
        super().__init__(config, src.layer_idx)          # makes fresh q/k/v/out/etc
        self.to(device=src.q_proj.weight.device, dtype=src.q_proj.weight.dtype)
        self.load_state_dict(src.state_dict(), strict=False)  # copies trained weights/buffers
        self.train(src.training)                          # preserve train/eval mode

        self.q_out = self.n_heads * self.d_head
        self.kv_out = self.n_kv_heads * self.d_head

        self.qkv_proj = nn.Linear(
            self.q_proj.in_features,
            self.q_out + 2 * self.kv_out,
            bias=False,
            device=self.q_proj.weight.device,
            dtype=self.q_proj.weight.dtype,
        )
        with torch.no_grad():
            self.qkv_proj.weight.copy_(torch.cat(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0
            ))

        del self.q_proj, self.k_proj, self.v_proj

    def forward(self, x, pos_ids, bm, v1, kv_cache=None):
        q, k, v = self.qkv_proj(x).split((self.q_out, self.kv_out, self.kv_out), dim=-1)

        B, T = x.shape[:2]
        q = q.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.value_residual:
            v1 = v if v1 is None else v1
            v = torch.lerp(v, v1.view_as(v), self.v_lamb)

        q, k = rms_norm(q), rms_norm(k)
        q, k = self.rope(q, pos_ids), self.rope(k, pos_ids)

        if kv_cache is not None:
            k, v, bm = kv_cache.upsert(k, v, pos_ids["t_pos"], self.layer_idx)

        y = flex_attention(q, k, v, block_mask=bm, enable_gqa=self.enable_gqa)

        if self.gated_attn:
            gates = torch.sigmoid(self.gate_proj(x[..., : self.n_heads]))
            y = y * gates.permute(0, 2, 1).unsqueeze(-1)

        y = y.transpose(1, 2).reshape(B, T, -1)
        y = self.out_proj(y)
        return y, v1


def patch_Attn_merge_qkv(model) -> None:
    for name, mod in list(model.named_modules()):
        if isinstance(mod, Attn) and not isinstance(mod, MergedQKVAttn):
            model.set_submodule(name, MergedQKVAttn(mod, model.config))


def apply_inference_patches(model) -> None:
    patch_cached_noise_conditioning(model)
    patch_Attn_merge_qkv(model)
