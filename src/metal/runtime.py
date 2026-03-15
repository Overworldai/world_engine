from __future__ import annotations

from pathlib import Path
import os

import torch
from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "metal"
_BUILD_DIR = _ROOT.parent / ".build" / "torch_extensions"

_ATTN_EXT_NAME = "world_metal_attn_ext"

_ATTN_READY = False


def _ensure_build_env() -> None:
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_BUILD_DIR))

    venv_bin = _ROOT.parent / ".venv" / "bin"
    if venv_bin.exists():
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
        os.environ.setdefault("NINJA", str(venv_bin / "ninja"))


def _try_load_extension(name: str, source: Path) -> bool:
    if not torch.backends.mps.is_available():
        return False
    if not source.exists():
        return False

    _ensure_build_env()
    try:
        load(
            name=name,
            sources=[str(source)],
            extra_cflags=["-std=c++17"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            with_cuda=False,
            is_python_module=False,
            verbose=False,
        )
        return True
    except Exception:
        return False


def ensure_metal_attention_op_loaded() -> bool:
    global _ATTN_READY
    if _ATTN_READY:
        return True
    _ATTN_READY = _try_load_extension(_ATTN_EXT_NAME, _SRC / "metal_flex_attn_op.mm")
    return _ATTN_READY


def metal_attention_available() -> bool:
    if hasattr(torch.ops, "world") and hasattr(torch.ops.world, "flex_attn_metal_ref"):
        return True
    return ensure_metal_attention_op_loaded()
