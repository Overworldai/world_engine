# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "world_engine",
#   "numpy",
#   "pygame-ce",  # community fork; ships a newer SDL2 with better Wayland support
#   "pillow",
# ]
#
# [tool.uv.sources]
# world_engine = { path = "..", editable = true }
# ///
#
# Minimal interactive client for the Overworld World Engine.
#
#   uv run examples/interactive.py Overworld/Waypoint-1.5-1B
#
# Controls:
#   WASD / mouse / buttons : forwarded as CtrlInput to the model
#   ESC                    : pause (freeze last frame, release mouse)
#   U                      : reset (re-seed, continues playing)
#   Left-click (on pause)  : resume
#   Close window / Ctrl+C  : quit
#
# Supports both Waypoint-1 / 1.1 (single-frame output) and Waypoint-1.5
# (4-frame temporally-compressed output). The only model-dependent branches
# live in `prime_seed` and `render`, keyed off `engine.model_cfg.model_type`.

import argparse
import io
import json
import logging
import random
import time
import urllib.request

import numpy as np
import pygame
import torch
from PIL import Image

from world_engine import CtrlInput, WorldEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("interactive")


# GitHub contents API for the Biome `seeds/` directory, pinned to a known ref.
# Same source as examples/gen_sample.py.
BIOME_SEEDS_API = (
    "https://api.github.com/repos/Overworldai/Biome/contents/seeds?ref=14343a6"
)

WINDOW_SIZE = (1280, 720)
# Aspect ratio for the center-crop applied to seed images.
CROP_ASPECT_W, CROP_ASPECT_H = 16, 9

# Map pygame keys / mouse buttons to the Windows VK integers that CtrlInput.button
# expects (see https://github.com/Overworldai/owl-control keycode table). Covers
# the main ANSI rows, space, shift, and the three mouse buttons — enough for
# WASD / spacebar / look-around gameplay without being exhaustive.
# Uses `pygame.K_*` int constants directly so this dict can be built at import
# time (before `pygame.init()`).
PYGAME_TO_VK: dict[int, int] = (
    {getattr(pygame, f"K_{ch}"): ord(ch) for ch in "1234567890"}
    | {pygame.K_MINUS: 0xBD, pygame.K_EQUALS: 0xBB}
    | {getattr(pygame, f"K_{ch}"): ord(ch.upper()) for ch in "qwertyuiop"}
    | {pygame.K_LEFTBRACKET: 0xDB, pygame.K_RIGHTBRACKET: 0xDD, pygame.K_BACKSLASH: 0xDC}
    | {getattr(pygame, f"K_{ch}"): ord(ch.upper()) for ch in "asdfghjkl"}
    | {pygame.K_SEMICOLON: 0xBA, pygame.K_QUOTE: 0xDE}
    | {getattr(pygame, f"K_{ch}"): ord(ch.upper()) for ch in "zxcvbnm"}
    | {pygame.K_COMMA: 0xBC, pygame.K_PERIOD: 0xBE, pygame.K_SLASH: 0xBF}
    | {pygame.K_SPACE: 0x20, pygame.K_LSHIFT: 0x10, pygame.K_RSHIFT: 0x10}
)
# pygame mouse button ids: 1=left, 2=middle, 3=right. VK: 0x01 LBUTTON, 0x04 MBUTTON, 0x02 RBUTTON.
MOUSE_TO_VK: dict[int, int] = {1: 0x01, 2: 0x04, 3: 0x02}


# --- seed loading -----------------------------------------------------------

def center_crop(img: Image.Image) -> np.ndarray:
    """Center-crop to CROP_ASPECT_W:CROP_ASPECT_H. Returns uint8 (H, W, 3)."""
    w, h = img.size
    # Pick whichever dimension is the limiting factor and derive the other.
    if w * CROP_ASPECT_H > h * CROP_ASPECT_W:
        new_w, new_h = h * CROP_ASPECT_W // CROP_ASPECT_H, h
    else:
        new_w, new_h = w, w * CROP_ASPECT_H // CROP_ASPECT_W
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    # `.copy()` — PIL's buffer is read-only and torch.from_numpy requires writable.
    return np.asarray(img.crop((left, top, left + new_w, top + new_h)).convert("RGB")).copy()


def load_seed_from_path(path: str) -> np.ndarray:
    """Load a local image as uint8 (H, W, 3), center-cropped."""
    log.info("loading seed from local file: %s", path)
    return center_crop(Image.open(path))


def load_seed_from_github() -> np.ndarray:
    """Download a random seed from the pinned Biome `seeds/` directory."""
    log.info("fetching Biome seeds index")
    with urllib.request.urlopen(BIOME_SEEDS_API) as res:
        entries = [e for e in json.load(res) if e["type"] == "file"]
    url = random.choice(entries)["download_url"]
    log.info("downloading random Biome seed: %s", url)
    with urllib.request.urlopen(url) as res:
        img_bytes = res.read()
    return center_crop(Image.open(io.BytesIO(img_bytes)))


def prime_seed(engine: WorldEngine, seed: np.ndarray) -> None:
    """Encode the seed frame into the engine's KV cache."""
    t = torch.from_numpy(seed).to(engine.device)  # uint8 (H, W, 3)
    tc = engine.model_cfg.temporal_compression
    if tc > 1:
        # Multi-frame models (e.g. Waypoint-1.5) consume/produce a stack of
        # `temporal_compression` frames per step.
        t = t.unsqueeze(0).expand(tc, -1, -1, -1).contiguous()
    log.info("priming engine with seed shape=%s", tuple(t.shape))
    engine.append_frame(t)


# --- rendering --------------------------------------------------------------

def _blit_frame(screen: pygame.Surface, frame: np.ndarray) -> pygame.Surface:
    """Blit a single (H, W, 3) uint8 numpy frame, scaled to the window. Returns the scaled surface."""
    # pygame.surfarray expects (W, H, 3), so swap the first two axes.
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, screen.get_size())
    screen.blit(surf, (0, 0))
    return surf


def _draw_hud(
    screen: pygame.Surface,
    font: pygame.font.Font | None,
    model_uri: str,
    batch_dt: float,
) -> None:
    """Draw FPS / frametime and model name at the top-right corner. No-op if font is None."""
    if font is None:
        return
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if batch_dt > 0:
        lines.append((f"{1.0 / batch_dt:.1f} fps / {batch_dt * 1000:.1f} ms", (255, 255, 255)))
    lines.append((model_uri, (160, 160, 160)))
    for i, (text, color) in enumerate(lines):
        label = font.render(text, True, color)
        x = screen.get_width() - label.get_width() - 12
        y = 12 + i * (label.get_height() + 4)
        screen.blit(label, (x, y))


def render(
    screen: pygame.Surface,
    frame_cpu: torch.Tensor,
    batch_dt: float,
    hud_font: pygame.font.Font | None = None,
    model_uri: str = "",
) -> pygame.Surface:
    """Display an already-on-CPU frame; return the last surface for pause caching.

    For multi-frame models the tensor is (T, H, W, 3) — we spread the T
    sub-frames evenly across `batch_dt` (per README "Waypoint-1.5 Behavior").
    The sleeps are what let the pipeline overlap: while we pace here, the GPU
    is already computing the next batch.
    """
    arr = frame_cpu.numpy()
    if arr.ndim == 3:  # single-frame model: (H, W, 3)
        last = _blit_frame(screen, arr)
        _draw_hud(screen, hud_font, model_uri, batch_dt)
        pygame.display.flip()
        return last

    # Multi-frame model: (T, H, W, 3)
    step_ms = max(0, int(batch_dt * 1000 / arr.shape[0]))
    last: pygame.Surface | None = None
    for i, sub in enumerate(arr):
        if i > 0 and step_ms:
            pygame.time.wait(step_ms)
        last = _blit_frame(screen, sub)
        _draw_hud(screen, hud_font, model_uri, batch_dt)
        pygame.display.flip()
    assert last is not None
    return last


def draw_pause_overlay(screen: pygame.Surface, last: pygame.Surface, font: pygame.font.Font) -> None:
    """Redraw the cached last frame with a dimmed overlay and centered pause text."""
    screen.blit(last, (0, 0))
    dim = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 128))  # 50% black
    screen.blit(dim, (0, 0))
    label = font.render("Paused — click to resume", True, (255, 255, 255))
    rect = label.get_rect(center=screen.get_rect().center)
    screen.blit(label, rect)
    pygame.display.flip()


def draw_status(screen: pygame.Surface, font: pygame.font.Font, text: str) -> None:
    """Clear to black and draw a status line in the bottom-left corner."""
    screen.fill((0, 0, 0))
    label = font.render(text, True, (220, 220, 220))
    screen.blit(label, (16, screen.get_height() - label.get_height() - 16))
    pygame.display.flip()


# --- mouse helpers -----------------------------------------------------------

def grab_mouse(screen: pygame.Surface) -> None:
    """Confine + hide the cursor for FPS-style gameplay."""
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()  # discard any pre-grab accumulated delta


def release_mouse() -> None:
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)


# --- engine init -------------------------------------------------------------

def init_engine(
    screen: pygame.Surface,
    font: pygame.font.Font,
    model_uri: str,
    quant: str | None,
    device: str,
    seed_path: str | None,
) -> tuple[WorldEngine, np.ndarray, torch.Tensor]:
    """Load the model, seed, and produce the first frame (torch.compile warmup).

    Draws status to the screen before each blocking step. The window won't pump
    events during the heavy steps (model load, compile), but the status text
    gives the user a progress indication.

    Returns (engine, seed, first_frame_cpu).
    """
    draw_status(screen, font, "Loading model…")
    log.info("loading model %s (quant=%s, device=%s)", model_uri, quant, device)
    engine = WorldEngine(model_uri, quant=quant, device=device)
    log.info(
        "model loaded: type=%s, temporal_compression=%d",
        engine.model_cfg.model_type, engine.model_cfg.temporal_compression,
    )

    draw_status(screen, font, "Loading seed…")
    seed = load_seed_from_path(seed_path) if seed_path else load_seed_from_github()

    draw_status(screen, font, "Priming engine…")
    engine.reset()
    prime_seed(engine, seed)

    # The first gen_frame triggers torch.compile — the most expensive step.
    draw_status(screen, font, "Warming up (torch.compile)…")
    log.info("warming up torch.compile")
    w0 = time.perf_counter()
    first = engine.gen_frame(ctrl=CtrlInput()).cpu()
    log.info("warmup complete in %.1fs", time.perf_counter() - w0)

    return engine, seed, first


# --- gameplay ----------------------------------------------------------------

def gameplay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    hud_font: pygame.font.Font,
    clock: pygame.time.Clock,
    engine: WorldEngine,
    seed: np.ndarray,
    first_frame: torch.Tensor,
    model_uri: str,
    mouse_sensitivity: float,
) -> None:
    """Interactive generation loop. Starts auto-paused on the first frame.

    Uses the pipelining pattern from the README: gen_frame() queues GPU kernels
    and returns immediately; we render the *previous* batch (with pacing sleeps)
    while the GPU works; then .cpu() syncs and transfers the result.
    """
    last_surface: pygame.Surface = render(screen, first_frame, 0.0)
    paused = True
    log.info("ready")

    held_vks: set[int] = set()
    # Pipeline state: `pending` holds the not-yet-rendered CPU frame from the
    # previous gen_frame call. We render it while the GPU computes the next one.
    pending: torch.Tensor | None = None
    batch_dt = 0.0

    while True:
        scroll = 0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return

            # Auto-pause when the cursor leaves the window. Safety net for
            # WMs where `set_grab` is advisory and the cursor can escape.
            elif e.type == pygame.WINDOWLEAVE and not paused:
                if pending is not None:
                    last_surface = render(screen, pending, batch_dt, hud_font, model_uri)
                    pending = None
                paused = True
                release_mouse()
                continue

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE and not paused:
                    if pending is not None:
                        last_surface = render(screen, pending, batch_dt, hud_font, model_uri)
                        pending = None
                    paused = True
                    release_mouse()
                    continue
                if e.key == pygame.K_u and not paused:
                    # reset() clears the KV cache and all state, so the model
                    # must be re-seeded with append_frame before it can produce
                    # coherent output again.
                    pending = None
                    engine.reset()
                    prime_seed(engine, seed)
                    continue
                vk = PYGAME_TO_VK.get(e.key)
                if vk is not None:
                    held_vks.add(vk)

            elif e.type == pygame.KEYUP:
                vk = PYGAME_TO_VK.get(e.key)
                if vk is not None:
                    held_vks.discard(vk)

            elif e.type == pygame.MOUSEBUTTONDOWN:
                if paused and e.button == 1:
                    paused = False
                    grab_mouse(screen)
                    continue
                vk = MOUSE_TO_VK.get(e.button)
                if vk is not None:
                    held_vks.add(vk)
                if e.button == 4:
                    scroll += 1
                elif e.button == 5:
                    scroll -= 1

            elif e.type == pygame.MOUSEBUTTONUP:
                vk = MOUSE_TO_VK.get(e.button)
                if vk is not None:
                    held_vks.discard(vk)

        if paused:
            draw_pause_overlay(screen, last_surface, font)
            clock.tick(60)
            continue

        dx, dy = pygame.mouse.get_rel()
        ctrl = CtrlInput(
            button=set(held_vks),
            mouse=(dx * mouse_sensitivity, dy * mouse_sensitivity),
            scroll_wheel=scroll,
        )

        # Pipeline: kick off generation (GPU kernels queued, returns fast),
        # then render the *previous* batch while the GPU works. Finally .cpu()
        # syncs and transfers the just-computed batch to CPU.
        t0 = time.perf_counter()
        next_frames = engine.gen_frame(ctrl=ctrl)
        if pending is not None:
            last_surface = render(screen, pending, batch_dt, hud_font, model_uri)
        pending = next_frames.cpu()
        batch_dt = time.perf_counter() - t0


# --- entry point -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_uri", help="HF model id, e.g. Overworld/Waypoint-1.5-1B")
    ap.add_argument("-s", "--seed", help="Path to a local seed image (defaults to a random Biome seed)")
    ap.add_argument("-q", "--quant", choices=["intw8a8", "fp8w8a8", "nvfp4"], default=None)
    ap.add_argument("-d", "--device", default="cuda")
    ap.add_argument("-m", "--mouse-sensitivity", type=float, default=1.5)
    args = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption(args.model_uri)
    font = pygame.font.SysFont(None, 36)
    hud_font = pygame.font.SysFont(None, 22)
    status_font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    try:
        engine, seed, first = init_engine(
            screen, status_font, args.model_uri, args.quant, args.device, args.seed,
        )
        gameplay(
            screen, font, hud_font, clock,
            engine, seed, first,
            args.model_uri, args.mouse_sensitivity,
        )
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
