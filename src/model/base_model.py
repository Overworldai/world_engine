import huggingface_hub
import os
import tempfile

from omegaconf import OmegaConf
from safetensors.torch import save_file, load_file
from torch import nn


class BaseModel(nn.Module):
    def save_pretrained(self, path: str) -> None:
        """Save weights (.safetensors) and OmegaConf YAML."""
        os.makedirs(path, exist_ok=True)
        save_file(
            {k: v.detach().cpu() for k, v in self.state_dict().items()},
            os.path.join(path, "model.safetensors"),
        )
        OmegaConf.save(self.config, os.path.join(path, "config.yaml"))

    @classmethod
    def from_pretrained(cls, path: str, cfg=None, device=None):
        """Load weights and OmegaConf YAML."""
        device = device or "cpu"

        try:
            path = huggingface_hub.snapshot_download(path)
        except Exception:
            pass

        if cfg is None:
            cfg = cls.load_config(path)
        model = cls(cfg)

        if device != "cpu":
            model = model.to(device)

        # Stream weights straight into `model` (no CPU state_dict first)
        safetensors_path = os.path.join(path, "model.safetensors")
        model.load_state_dict(load_file(safetensors_path, device=device), strict=True)

        return model

    def push_to_hub(self, uri: str, **kwargs):
        huggingface_hub.create_repo(uri, repo_type="model", exist_ok=True, private=True)
        with tempfile.TemporaryDirectory() as d:
            self.save_pretrained(d)
            huggingface_hub.upload_folder(folder_path=d, repo_id=uri, **kwargs)

    @staticmethod
    def load_config(path):
        if os.path.isdir(path):
            cfg_path = os.path.join(path, "config.yaml")
        else:
            cfg_path = huggingface_hub.hf_hub_download(repo_id=path, filename="config.yaml")
        return OmegaConf.load(cfg_path)
