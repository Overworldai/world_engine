import huggingface_hub
import os

from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch import nn
import torch

from ..quantize import apply_qat

class BaseModel(nn.Module):
    @classmethod
    def from_pretrained(cls, path: str, cfg=None, device=None, dtype=None, load_weights: bool = True):
        """Load weights and OmegaConf YAML."""
        device = torch.get_default_device() if device is None else device
        dtype = torch.get_default_dtype() if dtype is None else dtype

        try:
            path = huggingface_hub.snapshot_download(path)
        except Exception:
            pass

        if cfg is None:
            cfg = cls.load_config(path)
        model = cls(cfg).to(dtype=dtype, device=device)

        if cfg.quant is not None:
            apply_qat(model, quant_config=cfg.quant, layers="mlp", step="prepare")

        if load_weights:
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
