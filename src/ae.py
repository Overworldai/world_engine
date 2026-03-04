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


class ChunkedStreamingTAEHV(InferenceAE):
    def __init__(self, ae_model, device=None, dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.ae_model = ae_model.eval().to(device=device, dtype=dtype)

    @classmethod
    def from_pretrained(cls, model_uri: str, **kwargs):
        import pathlib

        import huggingface_hub
        from taehv import TAEHV

        try:
            base = pathlib.Path(huggingface_hub.snapshot_download(model_uri))
        except Exception:
            base = pathlib.Path(model_uri)

        ckpt = base if base.is_file() else base / "taehv1_5.pth"
        return cls(TAEHV(str(ckpt)), **kwargs)

    def reset(self):
        from taehv import StreamingTAEHV

        self._encoder = StreamingTAEHV(self.ae_model)
        self._decoder = StreamingTAEHV(self.ae_model)
        self._is_first_encode = True

    @torch.inference_mode()
    def encode(self, img: Tensor):
        """
        First call:
            img: [H, W, C] uint8
        Later calls:
            img: [T, H, W, C] uint8 with T == self.ae_model.t_downscale

        Returns:
            latent: [B, C, h, w]
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)

        assert img.dim() == 4 and img.shape[-1] == 3, (
            "Expected [H, W, C] or [T, H, W, C] uint8 image tensor"
        )

        expected_t = 1 if self._is_first_encode else self.ae_model.t_downscale
        if img.shape[0] != expected_t:
            raise ValueError(
                f"Expected {expected_t} frame(s), got {img.shape[0]}"
            )

        rgb = img.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        rgb = rgb.permute(0, 1, 4, 2, 3).contiguous().div(255)

        if self._is_first_encode:
            rgb = rgb.repeat(1, self.ae_model.t_downscale, 1, 1, 1)
            self._is_first_encode = False

        latent = self._encoder.encode(rgb)
        if latent is None:
            raise RuntimeError("Expected a latent after a full chunk")

        return latent[:, 0]

    @torch.inference_mode()
    def decode(self, latent: Tensor):
        """
        Input:
            latent: [B, C, h, w]

        Returns:
            frames: [T, H, W, C] uint8
        """
        latent = latent.unsqueeze(1).to(device=self.device, dtype=self.dtype)

        first = self._decoder.decode(latent)
        if first is None:
            raise RuntimeError("Expected decoded output after a latent")

        frames = [first]
        while True:
            frame = self._decoder.decode()
            if frame is None:
                break
            frames.append(frame)

        decoded = torch.cat(frames, dim=1)
        decoded = (decoded.clamp(0, 1) * 255).round().to(torch.uint8)
        return decoded.squeeze(0).permute(0, 2, 3, 1)[..., :3]


def get_ae(ae_uri, is_taehv_ae=False, **kwargs):
    if is_taehv_ae:
        return ChunkedStreamingTAEHV.from_pretrained(ae_uri, **kwargs)
    else:
        return InferenceAE.from_pretrained(ae_uri, **kwargs)
