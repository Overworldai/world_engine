import torch
from torch import Tensor


"""
WARNING:
- Always assumes scale=1, shift=0
"""


def bake_weight_norm_(module) -> int:
    """
    Removes weight parametrizations (from torch.nn.utils.parametrizations.weight_norm)
    and leaves the current parametrized weight as a plain Parameter.
    Returns how many modules were de-parametrized.
    """
    import torch.nn.utils.parametrize as parametrize

    n = 0
    for m in module.modules():
        # weight_norm registers a parametrization on "weight"
        if hasattr(m, "parametrizations") and "weight" in getattr(m, "parametrizations", {}):
            parametrize.remove_parametrizations(m, "weight", leave_parametrized=True)
            n += 1
    return n


class InferenceAE:
    def __init__(self, ae_model, device=None, dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.ae_model = ae_model.eval().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(cls, model_uri: str, **kwargs):
        import pathlib

        import huggingface_hub
        from omegaconf import OmegaConf
        from safetensors.torch import load_file
        from .ae_nn import AutoEncoder

        try:
            base = pathlib.Path(huggingface_hub.snapshot_download(model_uri))
        except Exception:
            base = pathlib.Path(model_uri)

        enc_cfg = OmegaConf.load(base / "encoder_conf.yml").model
        dec_cfg = OmegaConf.load(base / "decoder_conf.yml").model
        model = AutoEncoder(enc_cfg, dec_cfg)

        enc_sd = load_file(base / "encoder.safetensors", device="cpu")
        dec_sd = load_file(base / "decoder.safetensors", device="cpu")
        model.encoder.load_state_dict(enc_sd, strict=True)
        model.decoder.load_state_dict(dec_sd, strict=True)

        bake_weight_norm_(model)

        return cls(model, **kwargs)

    def encode(self, img: Tensor):
        """RGB -> RGB+D -> latent"""
        assert img.dim() == 3, "Expected [H, W, C] image tensor"
        img = img.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        rgb = img.permute(0, 3, 1, 2).contiguous().div(255).mul(2).sub(1)

        ####
        # Match AE input channel count (e.g. pad RGB -> RGB0 if model expects 4ch)
        in_ch = self.ae_model.encoder.conv_in.proj.in_channels
        if rgb.shape[1] < in_ch:
            pad = torch.zeros((rgb.shape[0], in_ch - rgb.shape[1], rgb.shape[2], rgb.shape[3]),
                              device=rgb.device, dtype=rgb.dtype)
            rgb = torch.cat([rgb, pad], dim=1)
        elif rgb.shape[1] > in_ch:
            rgb = rgb[:, :in_ch]
        ####

        return self.ae_model.encoder(rgb)

    @torch.inference_mode()
    @torch.compile(mode="max-autotune", dynamic=False, fullgraph=True)
    @torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    def decode(self, latent: Tensor):
        decoded = self.ae_model.decoder(latent)
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = (decoded * 255).round().to(torch.uint8)
        return decoded.squeeze(0).permute(1, 2, 0)[..., :3]
