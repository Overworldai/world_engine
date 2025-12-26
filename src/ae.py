from typing import Optional
from torch import Tensor

import os
import torch
from torch.nn import functional as F

# from depth_anything_v2.dpt import DepthAnythingV2


torch.backends.cuda.enable_flash_sdp(True)


def download_ckpt_if_not_exists():
    os.makedirs("checkpoints/depth", exist_ok=True)
    ckpt_path = "checkpoints/depth/depth_anything_v2_vitl.pth"
    if not os.path.exists(ckpt_path):
        import urllib.request
        url = "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
        print(f"Downloading checkpoint from {url} to {ckpt_path}...")
        urllib.request.urlretrieve(url, ckpt_path)
        print("Download complete.")


def normalize_depth(x, percentile_clip=2.0, gamma=1.2, output_uint8=True):
    """
    Enhanced depth map normalization with contrast boosting.
    Uses global statistics across all frames to prevent flickering.

    Args:
        x: [b,1,h,w] depth map tensor
        percentile_clip: percentage to clip from each tail for robust normalization (default: 2.0)
        gamma: gamma correction factor for contrast enhancement (>1 increases contrast, default: 1.2)

    Returns:
        [b,1,h,w] uint8 tensor with enhanced contrast depth map
    """
    b, c, h, w = x.shape
    x = x.float()

    # Flatten ALL dimensions for global percentile calculation
    # This ensures consistent normalization across all frames
    x_flat = x.reshape(-1)

    # Robust normalization using percentile clipping to handle outliers
    # Computing globally across all frames prevents flickering
    lower_percentile = torch.quantile(x_flat, percentile_clip / 100.0)
    upper_percentile = torch.quantile(x_flat, 1.0 - percentile_clip / 100.0)

    # Clip and normalize to [0, 1]
    x = torch.clamp(x, lower_percentile, upper_percentile)
    x = (x - lower_percentile) / (upper_percentile - lower_percentile + 1e-8)

    # Apply gamma correction for contrast enhancement
    # gamma > 1 stretches the middle values, increasing contrast
    x = torch.pow(x, gamma)

    # Apply contrast stretching using global mean
    global_mean = x.mean()
    x = (x - global_mean) * 1.5 + 0.5
    x = torch.clamp(x, 0.0, 1.0)

    if output_uint8:
        x = (x*255).to(torch.uint8)
    else:
        return (x.bfloat16() * 2) - 1
    return x


class BatchedDepthPipe:
    def __init__(self, input_mode = "uint8", batch_size = 10):
        self.model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        download_ckpt_if_not_exists()

        self.model.load_state_dict(torch.load('checkpoints/depth/depth_anything_v2_vitl.pth', map_location='cpu'))
        self.model = self.model.cuda().bfloat16().eval()
        self.model = torch.compile(self.model,dynamic=False,fullgraph=True, mode = 'max-autotune')
        self.batch_size = batch_size

        self.input_mode = input_mode # "uint8" or "bfloat16"

    @torch.compile()
    def preprocess(self, x, target_size=(518, 518)):
        """
        x is assumed [b,c,h,w] [-1,1] bfloat16 tensor (RGB)
        Returns: [b,c,h,w] float32 tensor, normalized as expected by DepthAnythingV2
        """

        x = x.cuda(non_blocking=True)

        if self.input_mode == "bfloat16":
            # Convert from [-1,1] to [0,1]
            x = (x + 1.0) / 2.0
        else:
            x = (x.bfloat16() / 255.0)

        # Normalize using ImageNet mean/std for RGB
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype)[None, :, None, None]
        x = (x - mean) / std

        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

        return x

    def forward(self, batch):
        # Assume batch is input to preprocess
        _,_,h,w = batch.shape
        batch = self.preprocess(batch)
        depth = self.model(batch)
        depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, size=(h,w), mode='bilinear', align_corners=True)
        depth = normalize_depth(depth, output_uint8=(self.input_mode == "uint8"))
        return depth # b1hw

    @torch.no_grad()
    def __call__(self, x):
        n = x.shape[0]
        batches = []
        for i in range(0, n, self.batch_size):
            batch = x[i:i+self.batch_size]
            if batch.shape[0] < self.batch_size:
                # Pad to full batch size
                pad_size = self.batch_size - batch.shape[0]
                pad_shape = (pad_size, *batch.shape[1:])
                pad = torch.zeros(pad_shape, device=batch.device, dtype=batch.dtype)
                batch = torch.cat([batch, pad], dim=0)
            batches.append(batch)
        outputs = []
        for i, batch in enumerate(batches):
            out = self.forward(batch)
            # If last batch and was padded, remove padding
            if i == len(batches) - 1 and n % self.batch_size != 0:
                out = out[:n % self.batch_size]
            outputs.append(out)
        return torch.cat(outputs, dim=0)


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
