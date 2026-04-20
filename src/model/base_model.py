import huggingface_hub
import os

from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch import nn
import torch

MODEL_CONFIG_DEFAULTS = OmegaConf.create(
    {
        "auto_aspect_ratio": True,
        "gated_attn": False,
        "inference_fps": "${base_fps}",
        "model_type": "waypoint-1",
        "n_kv_heads": "${n_heads}",
        "patch": [1, 1],
        "prompt_conditioning": None,
        "prompt_encoder_uri": "google/umt5-xl",
        "rope_nyquist_frac": 0.8,
        "rope_theta": 10000.0,
        "taehv_ae": False,
        "temporal_compression": 1,
        "value_residual": False,
    }
)


class BaseModel(nn.Module):
    @classmethod
    def from_pretrained(cls, path: str, cfg=None, device=None, dtype=None, load_weights: bool = True):
        """Load weights and OmegaConf YAML."""
        device = torch.get_default_device() if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype

        try:
            path = huggingface_hub.snapshot_download(
                path,
                allow_patterns=[
                    "config.yaml",
                    "model.safetensors",
                ],
            )
        except Exception:
            pass

        if cfg is None:
            cfg = cls.load_config(path)
        model = cls(cfg).to(dtype=dtype, device=device)

        if load_weights:
            safetensors_path = os.path.join(path, "model.safetensors")
            assert device.type != "cuda" or device.index is not None
            load_device = device.index if device.type == "cuda" else device.type
            model.load_state_dict(load_file(safetensors_path, device=load_device), strict=True)

        return model

    @staticmethod
    def load_config(path):
        if os.path.isdir(path):
            cfg_path = os.path.join(path, "config.yaml")
        else:
            cfg_path = huggingface_hub.hf_hub_download(repo_id=path, filename="config.yaml")
        cfg = OmegaConf.load(cfg_path)
        return OmegaConf.merge(MODEL_CONFIG_DEFAULTS, cfg)
