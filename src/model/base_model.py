import huggingface_hub
import os

from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch import nn


class BaseModel(nn.Module):
    @classmethod
    def from_pretrained(cls, path: str, cfg=None, device=None, dtype=None):
        """Load weights and OmegaConf YAML."""
        device = device or "cpu"

        try:
            path = huggingface_hub.snapshot_download(path)
        except Exception:
            pass

        if cfg is None:
            cfg = cls.load_config(path)
        model = cls(cfg).to(device=device, dtype=dtype)

        # Stream weights straight into `model` (no CPU state_dict first)
        safetensors_path = os.path.join(path, "model.safetensors")
        model.load_state_dict(load_file(safetensors_path, device=device), strict=True)

        return model

    @staticmethod
    def load_config(path):
        if os.path.isdir(path):
            cfg_path = os.path.join(path, "config.yaml")
        else:
            cfg_path = huggingface_hub.hf_hub_download(repo_id=path, filename="config.yaml")
        return OmegaConf.load(cfg_path)
