"""
CoreML-accelerated TAEHV encoder/decoder on the Apple Neural Engine.

Encoder: stateless, ANE (CPU_AND_NE) — 12ms, 0% GPU.
Decoder: stateful via explicit I/O state, ANE — 22ms, 0% GPU, exact streaming quality.

CoreML's StateType doesn't compile on ANE (error -14 regardless of tensor
count). The workaround: pass MemBlock state as regular model inputs/outputs
and manage the state on the Python side between predict() calls.
"""
import os

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Runtime wrapper (used at inference time)
# ---------------------------------------------------------------------------

class CoreMLTAEHV:
    """TAEHV via CoreML on ANE. Drop-in replacement for ChunkedStreamingTAEHV."""

    _ENCODE_SIZES = {(720, 1280): (512, 1024), (360, 640): (256, 512)}
    _DECODE_SIZES = {v: k for k, v in _ENCODE_SIZES.items()}

    def __init__(
        self,
        encoder_path: str,
        decoder_path: str,
        compute_units: str = "CPU_AND_NE",
        auto_aspect_ratio: bool = True,
        dtype=torch.bfloat16,
    ):
        cu = getattr(ct.ComputeUnit, compute_units)
        self.encoder_ml = ct.models.MLModel(encoder_path, compute_units=cu)
        self.decoder_ml = ct.models.MLModel(decoder_path, compute_units=cu)
        self.auto_aspect_ratio = auto_aspect_ratio
        self.dtype = dtype

        self._enc_key = None
        self._frames_key = None
        self._state_lo = None
        self._state_mid = None
        self._state_hi = None
        self._primed = False

        self._warmup()

    def _warmup(self):
        enc_pred = self.encoder_ml.predict({"x": np.zeros((4, 12, 256, 512), dtype=np.float16)})
        self._enc_key = list(enc_pred.keys())[0]

        dec_pred = self.decoder_ml.predict({
            "x": np.zeros((1, 32, 32, 64), dtype=np.float16),
            "state_lo": np.zeros((3, 256, 32, 64), dtype=np.float16),
            "state_mid": np.zeros((3, 128, 64, 128), dtype=np.float16),
            "state_hi": np.zeros((3, 64, 128, 256), dtype=np.float16),
        })
        for k, v in dec_pred.items():
            if v.shape[0] == 4:
                self._frames_key = k
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        model_uri: str,
        auto_aspect_ratio: bool = True,
        compute_units: str = "CPU_AND_NE",
        dtype=torch.bfloat16,
        device=None,
        export_dir: str = "diagnostics/taehv_ane",
        **kwargs,
    ):
        enc_path = os.path.join(export_dir, "taehv_encoder.mlpackage")
        dec_path = os.path.join(export_dir, "taehv_decoder_ane.mlpackage")

        if not os.path.exists(enc_path) or not os.path.exists(dec_path):
            print(f"[CoreMLTAEHV] Exporting models to {export_dir}...")
            _export_taehv(model_uri, export_dir)

        return cls(enc_path, dec_path, compute_units=compute_units,
                   auto_aspect_ratio=auto_aspect_ratio, dtype=dtype)

    def reset(self):
        self._state_lo = np.zeros((3, 256, 32, 64), dtype=np.float16)
        self._state_mid = np.zeros((3, 128, 64, 128), dtype=np.float16)
        self._state_hi = np.zeros((3, 64, 128, 256), dtype=np.float16)
        self._primed = False

    def _resize(self, x: Tensor, size: tuple[int, int]) -> Tensor:
        return F.interpolate(x[0], size=size, mode="bilinear", align_corners=False)[None]

    def _decode_raw(self, lat_np):
        """Run decoder, update state, return raw frames."""
        pred = self.decoder_ml.predict({
            "x": lat_np,
            "state_lo": self._state_lo,
            "state_mid": self._state_mid,
            "state_hi": self._state_hi,
        })
        for k, v in pred.items():
            if v.shape == (3, 256, 32, 64):
                self._state_lo = v
            elif v.shape == (3, 128, 64, 128):
                self._state_mid = v
            elif v.shape == (3, 64, 128, 256):
                self._state_hi = v
        return pred[self._frames_key]

    @torch.inference_mode()
    def encode(self, img: Tensor):
        t = 4
        assert img.dim() == 4 and img.shape[-1] == 3 and img.shape[0] == t

        rgb = img.unsqueeze(0).to(dtype=torch.float32).permute(0, 1, 4, 2, 3).contiguous().div(255)
        if self.auto_aspect_ratio:
            rgb = self._resize(rgb, self._ENCODE_SIZES[img.shape[1:3]])

        rgb_4d = rgb.reshape(4, 3, rgb.shape[3], rgb.shape[4])
        enc_input = F.pixel_unshuffle(rgb_4d, 2)

        pred = self.encoder_ml.predict({"x": enc_input.numpy().astype(np.float16)})
        return torch.from_numpy(pred[self._enc_key].astype(np.float32)).to(dtype=self.dtype)

    @torch.inference_mode()
    def decode(self, latent: Tensor):
        assert latent.dim() == 4
        lat_np = latent.to(dtype=torch.float32).numpy().astype(np.float16)

        # Prime on first decode (matches streaming TAEHV's frames_to_trim warmup)
        if not self._primed:
            for _ in range(3):  # frames_to_trim = 3 for taehv1_5
                self._decode_raw(lat_np)
            self._primed = True

        frames_np = self._decode_raw(lat_np)  # [4, 3, 512, 1024]

        decoded = torch.from_numpy(frames_np.astype(np.float32)).unsqueeze(0)
        if self.auto_aspect_ratio:
            decoded = self._resize(decoded, self._DECODE_SIZES[decoded.shape[-2:]])
        decoded = (decoded.clamp(0, 1) * 255).round().to(torch.uint8)
        return decoded.squeeze(0).permute(0, 2, 3, 1)[..., :3]


# ---------------------------------------------------------------------------
# Export: nn.Module wrappers traced for CoreML conversion (used once)
# ---------------------------------------------------------------------------

class _EncoderStatic(nn.Module):
    """Stateless encoder. Hardcoded for taehv1_5: T=4, patch_size=2, 512x1024.

    Input:  [4, 12, 256, 512]  (4 frames after pixel_unshuffle)
    Output: [1, 32, 32, 64]    (1 latent)
    """

    def __init__(self, taehv):
        super().__init__()
        self.blocks = nn.ModuleList(list(taehv.encoder))

    def forward(self, x):
        x = self.blocks[0](x)  # Conv [4, 64, 256, 512]
        x = self.blocks[1](x)  # ReLU

        x = self.blocks[2].conv(x.reshape(2, 128, 256, 512))  # TPool(2): 4→2
        x = self.blocks[3](x)  # Conv stride=2: [2, 64, 128, 256]

        for i in [4, 5, 6]:
            past = torch.cat([torch.zeros_like(x[:1]), x[:-1]], dim=0)
            x = self.blocks[i](x, past)

        x = self.blocks[7].conv(x.reshape(1, 128, 128, 256))  # TPool(2): 2→1
        x = self.blocks[8](x)  # Conv stride=2: [1, 64, 64, 128]

        for i in [9, 10, 11]:
            x = self.blocks[i](x, torch.zeros_like(x))

        x = self.blocks[12].conv(x)  # TPool(1): no change
        x = self.blocks[13](x)       # Conv stride=2: [1, 64, 32, 64]

        for i in [14, 15, 16]:
            x = self.blocks[i](x, torch.zeros_like(x))

        return self.blocks[17](x)  # Conv 64→32: [1, 32, 32, 64]


class _DecoderExplicitState(nn.Module):
    """Stateful decoder with state as explicit inputs/outputs (ANE-compatible).

    CoreML's StateType fails on ANE (error -14). This passes MemBlock
    memories as regular inputs and returns updated memories as outputs.
    Uses torch.cat (not zeros+scatter) to build output state — scatter
    ops are ~40ms slower on ANE.

    Input:  x [1, 32, 32, 64], state_lo [3, 256, 32, 64],
            state_mid [3, 128, 64, 128], state_hi [3, 64, 128, 256]
    Output: frames [4, 3, 512, 1024], new_state_lo, new_state_mid, new_state_hi
    """

    def __init__(self, taehv):
        super().__init__()
        self.blocks = nn.ModuleList(list(taehv.decoder))

    def forward(self, x, state_lo, state_mid, state_hi):
        x = self.blocks[0](x)   # Clamp
        x = self.blocks[1](x)   # Conv 32→256
        x = self.blocks[2](x)   # ReLU

        # Group 1 (blocks 3-5, T=1): save INPUT to each block as new state
        save_3 = x
        x = self.blocks[3](x, state_lo[0:1])
        save_4 = x
        x = self.blocks[4](x, state_lo[1:2])
        save_5 = x
        x = self.blocks[5](x, state_lo[2:3])
        new_lo = torch.cat([save_3, save_4, save_5], dim=0)

        x = self.blocks[6](x)          # Upsample(2)
        x = self.blocks[7].conv(x)     # TGrow(1)
        x = self.blocks[8](x)          # Conv 256→128

        # Group 2 (blocks 9-11, T=1)
        save_9 = x
        x = self.blocks[9](x, state_mid[0:1])
        save_10 = x
        x = self.blocks[10](x, state_mid[1:2])
        save_11 = x
        x = self.blocks[11](x, state_mid[2:3])
        new_mid = torch.cat([save_9, save_10, save_11], dim=0)

        x = self.blocks[12](x)         # Upsample(2)
        x = self.blocks[13].conv(x)    # TGrow(2): 1→2
        x = x.reshape(2, 128, 128, 256)
        x = self.blocks[14](x)         # Conv 128→64

        # Group 3 (blocks 15-17, T=2): save LAST frame's input
        past = torch.cat([state_hi[0:1], x[:1]], dim=0)
        save_15 = x[1:2]
        x = self.blocks[15](x, past)

        past = torch.cat([state_hi[1:2], x[:1]], dim=0)
        save_16 = x[1:2]
        x = self.blocks[16](x, past)

        past = torch.cat([state_hi[2:3], x[:1]], dim=0)
        save_17 = x[1:2]
        x = self.blocks[17](x, past)
        new_hi = torch.cat([save_15, save_16, save_17], dim=0)

        x = self.blocks[18](x)         # Upsample(2)
        x = self.blocks[19].conv(x)    # TGrow(2): 2→4
        x = x.reshape(4, 64, 256, 512)
        x = self.blocks[20](x)         # Conv 64→64
        x = self.blocks[21](x)         # ReLU
        x = self.blocks[22](x)         # Conv 64→12

        x = F.pixel_shuffle(x, 2)      # [4, 3, 512, 1024]
        x = x.clamp(0, 1)

        return x, new_lo, new_mid, new_hi


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------

def _export_taehv(model_uri: str, out_dir: str):
    """Export TAEHV encoder (stateless) and decoder (explicit-state) for ANE."""
    import pathlib
    import huggingface_hub
    from taehv import TAEHV

    try:
        base = pathlib.Path(huggingface_hub.snapshot_download(model_uri))
    except Exception:
        base = pathlib.Path(model_uri)
    ckpt = base if base.is_file() else base / "taehv1_5.pth"

    taehv = TAEHV(str(ckpt)).eval().to(torch.float32)
    os.makedirs(out_dir, exist_ok=True)

    # Encoder — stateless
    enc_path = os.path.join(out_dir, "taehv_encoder.mlpackage")
    if not os.path.exists(enc_path):
        enc = _EncoderStatic(taehv).eval()
        with torch.no_grad():
            traced = torch.jit.trace(enc, torch.randn(4, 12, 256, 512))
        ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=(4, 12, 256, 512))],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        ).save(enc_path)

    # Decoder — explicit I/O state (no StateType, ANE-compatible)
    dec_path = os.path.join(out_dir, "taehv_decoder_ane.mlpackage")
    if not os.path.exists(dec_path):
        dec = _DecoderExplicitState(taehv).eval()
        dummy = (
            torch.randn(1, 32, 32, 64),
            torch.zeros(3, 256, 32, 64),
            torch.zeros(3, 128, 64, 128),
            torch.zeros(3, 64, 128, 256),
        )
        with torch.no_grad():
            traced = torch.jit.trace(dec, dummy, strict=False)
        ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="x", shape=(1, 32, 32, 64)),
                ct.TensorType(name="state_lo", shape=(3, 256, 32, 64)),
                ct.TensorType(name="state_mid", shape=(3, 128, 64, 128)),
                ct.TensorType(name="state_hi", shape=(3, 64, 128, 256)),
            ],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        ).save(dec_path)

    print(f"[CoreMLTAEHV] Exported to {out_dir}")
