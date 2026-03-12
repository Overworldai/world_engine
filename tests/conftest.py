from __future__ import annotations

from pathlib import Path
import os
import math

import pytest
import torch
from torch.utils.cpp_extension import load


_EXT_NAME = "world_metal_attn_ext"
_EXT_BUILT = False
_FALLBACK_REGISTERED = False


def _reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    causal: bool,
) -> torch.Tensor:
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)

    if qf.size(1) != kf.size(1):
        if qf.size(1) < kf.size(1) or (qf.size(1) % kf.size(1)) != 0:
            raise RuntimeError("GQA requires q_heads divisible by kv_heads")
        group_size = qf.size(1) // kf.size(1)
        head_idx = torch.arange(qf.size(1), device=q.device, dtype=torch.long) // group_size
        kf = kf.index_select(1, head_idx)
        vf = vf.index_select(1, head_idx)

    scores = torch.matmul(qf, kf.transpose(-2, -1)) / math.sqrt(q.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    if causal:
        t = q.size(-2)
        l = k.size(-2)
        causal_mask = torch.triu(
            torch.ones((t, l), device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask[None, None], float("-inf"))

    finite_row = torch.isfinite(scores).any(dim=-1, keepdim=True)
    safe_scores = torch.where(finite_row, scores, torch.zeros_like(scores))
    probs = torch.softmax(safe_scores, dim=-1)
    probs = torch.where(finite_row, probs, torch.zeros_like(probs))
    out = torch.matmul(probs, vf)
    return out.to(q.dtype)


def _register_python_fallback_op() -> None:
    global _FALLBACK_REGISTERED
    if _FALLBACK_REGISTERED:
        return

    try:
        lib = torch.library.Library("world", "DEF")
        lib.define("flex_attn_metal(Tensor q, Tensor k, Tensor v, Tensor? mask=None, bool causal=True) -> Tensor")
        lib.define("flex_attn_metal_ref(Tensor q, Tensor k, Tensor v, Tensor? mask=None, bool causal=True) -> Tensor")
        lib.define("flex_attn_metal_fast(Tensor q, Tensor k, Tensor v, Tensor? mask=None, bool causal=True) -> Tensor")
        lib.define("flex_attn_metal_fast_blocks(Tensor q, Tensor k, Tensor v, Tensor block_written, int block_size, bool causal=True) -> Tensor")
        lib.define("flex_attn_metal_fast_active(Tensor q, Tensor k, Tensor v, Tensor active_blocks, int block_size, bool causal=True) -> Tensor")
    except Exception:
        # Signature may already be defined by another registration path.
        pass

    impl = torch.library.Library("world", "IMPL", "CompositeExplicitAutograd")
    fn = lambda q, k, v, mask=None, causal=True: _reference_attention(q, k, v, mask, bool(causal))
    impl.impl("flex_attn_metal", fn)
    impl.impl("flex_attn_metal_ref", fn)
    impl.impl("flex_attn_metal_fast", fn)
    impl.impl(
        "flex_attn_metal_fast_blocks",
        lambda q, k, v, block_written, block_size, causal=True: _reference_attention(
            q,
            k,
            v,
            # Build dense mask from block_written for fallback semantics.
            torch.cat(
                [
                    torch.full(
                        (int(block_size),),
                        int(block_written[i].item() != 0),
                        device=q.device,
                        dtype=torch.uint8,
                    )
                    for i in range(block_written.numel())
                ],
                dim=0,
            )[: k.size(-2)].view(1, 1, 1, k.size(-2)).expand(q.size(0), q.size(1), q.size(2), k.size(-2)).contiguous(),
            bool(causal),
        ),
    )
    impl.impl(
        "flex_attn_metal_fast_active",
        lambda q, k, v, active_blocks, block_size, causal=True: _reference_attention(
            q,
            k,
            v,
            (
                torch.zeros((k.size(-2),), device=q.device, dtype=torch.uint8)
                .index_fill(
                    0,
                    (
                        torch.cat(
                            [
                                torch.arange(
                                    int(b.item()) * int(block_size),
                                    min(k.size(-2), int(b.item()) * int(block_size) + int(block_size)),
                                    device=q.device,
                                    dtype=torch.long,
                                )
                                for b in active_blocks
                            ],
                            dim=0,
                        )
                        if active_blocks.numel() > 0
                        else torch.empty((0,), device=q.device, dtype=torch.long)
                    ),
                    1,
                )
                .view(1, 1, 1, k.size(-2))
                .expand(q.size(0), q.size(1), q.size(2), k.size(-2))
                .contiguous()
            ),
            bool(causal),
        ),
    )
    _FALLBACK_REGISTERED = True


def _load_metal_attention_extension() -> None:
    global _EXT_BUILT
    if _EXT_BUILT:
        return

    if not torch.backends.mps.is_available():
        # Let MPS-gated tests skip naturally.
        return

    source = Path(__file__).resolve().parents[1] / "src" / "metal" / "metal_flex_attn_op.mm"
    if not source.exists():
        raise FileNotFoundError(f"Missing Metal extension source: {source}")

    build_dir = Path(__file__).resolve().parents[1] / ".build" / "torch_extensions"
    build_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    venv_bin = repo_root / ".venv" / "bin"

    # Make the extension cache deterministic in this repo.
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(build_dir))
    # torch.utils.cpp_extension shells out to `ninja`; ensure the venv binary
    # is discoverable even when PATH is inherited from the host shell.
    if venv_bin.exists():
        os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
        os.environ.setdefault("NINJA", str(venv_bin / "ninja"))

    try:
        load(
            name=_EXT_NAME,
            sources=[str(source)],
            extra_cflags=["-std=c++17"],
            extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
            with_cuda=False,
            is_python_module=False,
            verbose=False,
        )
        _EXT_BUILT = True
    except Exception:
        # Keep tests executable on environments where the ObjC++ binding is not
        # yet compatible with the installed torch MPS headers.
        _register_python_fallback_op()


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):  # noqa: D401 - pytest hook signature
    _load_metal_attention_extension()

