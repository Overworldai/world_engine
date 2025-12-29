from typing import Optional
from torch import Tensor

import torch



class InferenceAE:
    def __init__(self, ae_model, device=None, dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.ae_model = ae_model.eval().to(device=device, dtype=dtype)
        # self.depth_model = BatchedDepthPipe(input_mode="bfloat16", batch_size=1)

        self.scale = 1.0  # TODO: dont hardcode. AE should keep internal scale buffer

    @classmethod
    def from_pretrained(cls, model_uri: str, subdir: Optional[str] = "ae", **kwargs):
        import huggingface_hub, pathlib
        from owl_vaes import from_pretrained
        base_path = huggingface_hub.snapshot_download(model_uri)
        base_path = pathlib.Path(base_path) / (subdir)
        model = from_pretrained(base_path / "config.yaml", base_path / "ckpt.pt")
        return cls(model, **kwargs)

    def encode(self, img: Tensor):
        """RGB -> RGB+D -> latent"""
        assert img.dim() == 3
        img = img.unsqueeze(0)  # [H,W,C] -> [1,H,W,C]
        img = img.to(device=self.device, dtype=self.dtype)
        img = img.permute(0, 3, 1, 2).contiguous()
        rgb = img.div(255.0).mul(2.0).sub(1.0)

        # TODO: fix hack
        """
        depth = self.depth_model(rgb)
        x = torch.cat([rgb, depth], dim=1)
        """
        x = rgb

        lat = self.ae_model.encoder(x)
        return lat / self.scale

    @torch.compile
    @torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    def decode(self, latent: Tensor):
        # Decode single latent: [C, H, W]
        decoded = self.ae_model.decoder(latent * self.scale)
        decoded = (decoded / 2 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
        decoded = (decoded * 255).round().to(torch.uint8)  # uint8 [0,255]
        decoded = decoded.squeeze(0).permute(1, 2, 0)  # [H, W, 4] (RGBD)
        decoded = decoded[..., :3]  # strip depth
        return decoded
