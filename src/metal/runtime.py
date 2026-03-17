from __future__ import annotations

from pathlib import Path
import os
import sys

import torch
from torch.utils.cpp_extension import load

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "metal"
_BUILD_DIR = _ROOT.parent / ".build" / "torch_extensions"

_ATTN_EXT_NAME = "world_metal_attn_ext"

_ATTN_READY = False
_ATTN_FAKE_READY = False


def _ensure_build_env() -> None:
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_BUILD_DIR))

    # Prefer the currently active interpreter's bin directory (works for worktrees),
    # then fall back to local repo .venv/bin if present.
    bin_candidates: list[Path] = [
        Path(sys.executable).parent,
        Path(sys.prefix) / "bin",
        _ROOT.parent / ".venv" / "bin",
    ]
    for bin_dir in bin_candidates:
        if not bin_dir.exists():
            continue
        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
        ninja_bin = bin_dir / "ninja"
        if ninja_bin.exists():
            os.environ.setdefault("NINJA", str(ninja_bin))
            break


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
            extra_cflags=["-std=c++17", "-O3"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            with_cuda=False,
            is_python_module=False,
            verbose=False,
        )
        return True
    except Exception as exc:
        if os.environ.get("WORLD_METAL_RUNTIME_DEBUG", "0") == "1":
            print(f"[metal.runtime] failed to load {name}: {type(exc).__name__}: {exc}")
        return False


def _register_attention_fake_kernels() -> None:
    global _ATTN_FAKE_READY
    if _ATTN_FAKE_READY:
        return

    def _out_like_q(q: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(q)

    try:
        @torch.library.register_fake("world::flex_attn_metal")
        def _fake_metal(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor | None = None,
            causal: bool = True,
        ) -> torch.Tensor:
            return _out_like_q(q)

        @torch.library.register_fake("world::flex_attn_metal_ref")
        def _fake_metal_ref(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor | None = None,
            causal: bool = True,
        ) -> torch.Tensor:
            return _out_like_q(q)

        @torch.library.register_fake("world::flex_attn_metal_fast")
        def _fake_metal_fast(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            mask: torch.Tensor | None = None,
            causal: bool = True,
        ) -> torch.Tensor:
            return _out_like_q(q)

        @torch.library.register_fake("world::flex_attn_metal_fast_blocks")
        def _fake_metal_fast_blocks(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            block_written: torch.Tensor,
            block_size: int,
            causal: bool = True,
        ) -> torch.Tensor:
            return _out_like_q(q)

        @torch.library.register_fake("world::flex_attn_metal_fast_blocks_direct")
        def _fake_metal_fast_blocks_direct(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            block_written: torch.Tensor,
            block_size: int,
            causal: bool = True,
        ) -> torch.Tensor:
            return _out_like_q(q)

        @torch.library.register_fake("world::flex_attn_metal_fast_active")
        def _fake_metal_fast_active(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            active_blocks: torch.Tensor,
            block_size: int,
            causal: bool = True,
        ) -> torch.Tensor:
            return _out_like_q(q)

        @torch.library.register_fake("world::flex_attn_metal_fast_active_counted")
        def _fake_metal_fast_active_counted(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            active_blocks: torch.Tensor,
            active_count: torch.Tensor,
            block_size: int,
            causal: bool = True,
        ) -> torch.Tensor:
            return _out_like_q(q)
    except RuntimeError:
        # Fake implementations can only be registered once per process.
        pass

    _ATTN_FAKE_READY = True


def ensure_metal_attention_op_loaded() -> bool:
    global _ATTN_READY
    if _ATTN_READY:
        _register_attention_fake_kernels()
        return True
    _ATTN_READY = _try_load_extension(_ATTN_EXT_NAME, _SRC / "metal_flex_attn_op.mm")
    if _ATTN_READY:
        _register_attention_fake_kernels()
    return _ATTN_READY


def metal_attention_available() -> bool:
    if hasattr(torch.ops, "world") and hasattr(torch.ops.world, "flex_attn_metal_ref"):
        _register_attention_fake_kernels()
        return True
    return ensure_metal_attention_op_loaded()
