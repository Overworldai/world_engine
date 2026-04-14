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
# (4-frame temporally-compressed output).
#
# Frame pacing strategy
# ---------------------
# gen_frame() dispatches GPU kernels and returns a not-yet-ready tensor. We
# render the *previous* batch's sub-frames (with pacing sleeps) while the GPU
# computes, then call .cpu() to sync + transfer the result.
#
# For multi-frame models (temporal_compression > 1), each gen_frame produces T
# sub-frames that must be spread over a pacing interval. The interval is:
#
#   pace_s = max(batch_dt * SLEEP_RATIO, target_s - overhead)
#
# where:
# - batch_dt is the previous cycle's wall-clock time.
# - overhead is the measured non-render portion of the cycle (dispatch +
#   .cpu() + events).
# - target_s = T / fps_cap. When fps_cap is 0 (--uncap-fps), target_s is 0
#   and pace_s falls back to `batch_dt * SLEEP_RATIO` — pure GPU-bound pacing.
#
# - The `target_s - overhead` term ensures the *total* cycle (render + overhead)
#   hits the model's target framerate when a cap is active.
# - The `batch_dt * SLEEP_RATIO` floor prevents the render from filling the
#   entire cycle, which would create a diverging feedback loop (batch_dt
#   includes render time, so pacing to 100% of it grows without bound).
# - Each sub-frame is presented immediately, then we sleep for SLEEP_RATIO of
#   the remaining time until the next deadline (yielding CPU), then busy-wait
#   the rest for precise timing. See the SLEEP_RATIO constant for details.

import argparse
import io
import json
import logging
import random
import time
import urllib.request
from dataclasses import InitVar, dataclass, field
from typing import ClassVar

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


WINDOW_SIZE = (1280, 720)
# Fraction of a sleep interval that is yielded to the OS via pygame.time.wait;
# the remainder is covered by a busy-wait spin for precise timing. OS schedulers
# are free to extend sleeps beyond their stated duration, so we deliberately
# undershoot and spin the rest. 0.8 was selected ad-hoc and could be raised
# toward 1.0 on platforms with more accurate sleep granularity.
SLEEP_RATIO = 0.8


# --- rendering --------------------------------------------------------------


@dataclass
class Renderer:
    """Owns the screen surface and fonts; handles all drawing."""

    model_uri: str
    screen: pygame.Surface = field(init=False)
    font: pygame.font.Font = field(init=False)
    hud_font: pygame.font.Font = field(init=False)
    status_font: pygame.font.Font = field(init=False)
    _last_surface: pygame.Surface | None = field(init=False, default=None, repr=False)
    """Cached surface of the most recently rendered frame, used for pause."""

    def __post_init__(self) -> None:
        self.screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption(self.model_uri)
        self.font = pygame.font.SysFont(None, 36)
        self.hud_font = pygame.font.SysFont(None, 22)
        self.status_font = pygame.font.SysFont(None, 24)

    def _present(
        self,
        frame: np.ndarray,
        batch_dt: float,
        temporal_compression: int,
    ) -> None:
        """Blit a single (H, W, 3) frame, draw HUD, flip, and cache for pause."""
        # pygame.surfarray expects (W, H, 3), so swap the first two axes.
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, self.screen.get_size())
        self.screen.blit(surf, (0, 0))
        self._last_surface = surf

        lines: list[tuple[str, tuple[int, int, int]]] = []
        if batch_dt > 0:
            lfps = 1.0 / batch_dt
            if temporal_compression > 1:
                lines.append(
                    (f"{lfps * temporal_compression:.1f} FPS", (255, 255, 255)),
                )
                lines.append(
                    (f"{lfps:.1f} LFPS / {batch_dt * 1000:.1f} ms", (255, 255, 255)),
                )
            else:
                lines.append(
                    (f"{lfps:.1f} FPS / {batch_dt * 1000:.1f} ms", (255, 255, 255)),
                )
        lines.append((self.model_uri, (160, 160, 160)))
        for i, (text, color) in enumerate(lines):
            label = self.hud_font.render(text, True, color)
            x = self.screen.get_width() - label.get_width() - 12
            y = 12 + i * (label.get_height() + 4)
            self.screen.blit(label, (x, y))

        pygame.display.flip()

    def render_frame(
        self,
        frame_cpu: torch.Tensor,
        batch_dt: float,
        temporal_compression: int,
        pace_s: float,
    ) -> None:
        """Display an already-on-CPU frame and cache it for the pause overlay.

        Sub-frames are spread evenly across `pace_s`. The caller computes
        `pace_s` to balance GPU overlap headroom with the FPS cap.
        """
        arr = frame_cpu.numpy()
        # Treat single-frame (H, W, 3) as a batch of one.
        frames = [arr] if arr.ndim == 3 else list(arr)
        step_s = pace_s / len(frames)
        start = time.perf_counter()
        for i, sub in enumerate(frames):
            self._present(sub, batch_dt, temporal_compression)
            # Hybrid sleep+spin: yield CPU for SLEEP_RATIO of the remaining
            # time, then busy-wait the rest for precise timing.
            deadline = start + step_s * (i + 1)
            remaining_ms = int((deadline - time.perf_counter()) * SLEEP_RATIO * 1000)
            if remaining_ms > 0:
                pygame.time.wait(remaining_ms)
            while time.perf_counter() < deadline:
                pass

    def draw_pause(self) -> None:
        """Redraw cached last frame with a dimmed overlay and pause text."""
        assert self._last_surface is not None
        self.screen.blit(self._last_surface, (0, 0))
        dim = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 128))  # 50% black
        self.screen.blit(dim, (0, 0))
        label = self.font.render("Paused — click to resume", True, (255, 255, 255))
        rect = label.get_rect(center=self.screen.get_rect().center)
        self.screen.blit(label, rect)
        pygame.display.flip()

    def draw_status(self, text: str) -> None:
        """Clear to black and draw a status line in the bottom-left corner."""
        self.screen.fill((0, 0, 0))
        label = self.status_font.render(text, True, (220, 220, 220))
        self.screen.blit(
            label,
            (16, self.screen.get_height() - label.get_height() - 16),
        )
        pygame.display.flip()


# --- engine ------------------------------------------------------------------


@dataclass
class Engine:
    """Wraps WorldEngine with seed management and the generation pipeline.

    Frames are produced by `next_frame()` and should be `.cpu()`'d by the
    caller before the next `next_frame()` call (GPU buffers may be reused).
    """

    model_uri: str
    quant: InitVar[str | None]
    device: InitVar[str]
    inner: WorldEngine = field(init=False, repr=False)
    seed: np.ndarray | None = field(init=False, default=None, repr=False)
    """Center-cropped uint8 (H, W, 3) numpy array, set via `set_seed()`."""

    def __post_init__(self, quant: str | None, device: str) -> None:
        log.info(
            "loading model %s (quant=%s, device=%s)",
            self.model_uri,
            quant,
            device,
        )
        self.inner = WorldEngine(self.model_uri, quant=quant, device=device)
        log.info(
            "model loaded: type=%s, temporal_compression=%d",
            self.inner.model_cfg.model_type,
            self.inner.model_cfg.temporal_compression,
        )

    @property
    def temporal_compression(self) -> int:
        return getattr(self.inner.model_cfg, "temporal_compression", 1)

    @property
    def inference_fps(self) -> int:
        """Visual framerate the model targets."""
        return getattr(self.inner.model_cfg, "inference_fps", 60)

    def set_seed(self, img: Image.Image) -> None:
        """Center-crop the image to the expected aspect ratio and store as seed."""
        # TODO: Calculate aspect ratio automatically once
        # https://github.com/Overworldai/world_engine/issues/43 lands
        crop_w, crop_h = 16, 9
        w, h = img.size
        if w * crop_h > h * crop_w:
            new_w, new_h = h * crop_w // crop_h, h
        else:
            new_w, new_h = w, w * crop_h // crop_w
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        cropped = img.crop((left, top, left + new_w, top + new_h)).convert("RGB")
        # .copy() — PIL's buffer is read-only and torch.from_numpy requires writable.
        self.seed = np.asarray(cropped).copy()

    def prime(self) -> None:
        """Reset state and encode the seed frame into the KV cache."""
        assert self.seed is not None, "call set_seed() first"
        self.inner.reset()
        t = torch.from_numpy(self.seed).to(self.inner.device)  # uint8 (H, W, 3)
        tc = self.inner.model_cfg.temporal_compression
        if tc > 1:
            # Multi-frame models (e.g. Waypoint-1.5) consume/produce a stack of
            # `temporal_compression` frames per step.
            t = t.unsqueeze(0).expand(tc, -1, -1, -1).contiguous()
        log.info("priming engine with seed shape=%s", tuple(t.shape))
        self.inner.append_frame(t)

    def warmup(self) -> torch.Tensor:
        """Run one gen_frame to trigger torch.compile. Returns the first frame (CPU)."""
        log.info("warming up torch.compile")
        w0 = time.perf_counter()
        first = self.next_frame(ctrl=CtrlInput()).cpu()
        log.info("warmup complete in %.1fs", time.perf_counter() - w0)
        return first

    def next_frame(self, ctrl: CtrlInput) -> torch.Tensor:
        """Generate the next frame. Returns a GPU tensor.

        Caller must .cpu() before the next call (GPU buffers may be reused).
        """
        return self.inner.gen_frame(ctrl=ctrl)

    def reset(self) -> None:
        """Reset all state and re-prime the seed.

        reset() clears the KV cache and all state, so the model must be
        re-seeded with append_frame before it can produce coherent output.
        """
        self.prime()


# --- gameplay ----------------------------------------------------------------


@dataclass
class GameState:
    """Interactive generation loop state. Call `run()` to enter the main loop."""

    renderer: Renderer
    engine: Engine
    clock: pygame.time.Clock
    mouse_sensitivity: float
    uncap_fps: bool = False
    paused: bool = True
    held_vks: set[int] = field(default_factory=set)
    """Currently pressed Windows VK codes, forwarded as `CtrlInput.button`."""
    scroll: int = 0
    pending: torch.Tensor | None = None
    """Not-yet-rendered CPU frame from the previous `gen_frame` call.
    Rendered while the GPU computes the next batch (pipeline overlap)."""
    batch_dt: float = 0.0
    """Wall-clock seconds the last full cycle (dispatch + render + sync) took."""
    _overhead: float = 0.0
    """Non-render portion of the previous cycle (dispatch + .cpu() + events).
    Subtracted from the FPS-cap target so pacing compensates for it."""
    _pace_s: float = 0.0
    """Pacing interval used by the last render_frame call."""

    @property
    def temporal_compression(self) -> int:
        return self.engine.temporal_compression

    @property
    def fps_cap(self) -> int:
        """0 = uncapped; otherwise the model's inference_fps."""
        return 0 if self.uncap_fps else self.engine.inference_fps

    def _compute_pace(self) -> float:
        """Compute pacing interval for render_frame, accounting for overhead."""
        # Target interval from the model's intended visual framerate.
        target_s = self.temporal_compression / self.fps_cap if self.fps_cap > 0 else 0.0
        # Subtract measured non-render overhead so the *total* cycle hits target_s.
        # Use SLEEP_RATIO of batch_dt as a floor so the render never fills the entire
        # overlap window (batch_dt includes render time — pacing to 100% diverges).
        return max(self.batch_dt * SLEEP_RATIO, target_s - self._overhead)

    def _enter_pause(self) -> None:
        """Flush any in-flight batch and enter paused state."""
        if self.pending is not None:
            self.renderer.render_frame(
                self.pending,
                self.batch_dt,
                self.temporal_compression,
                self._compute_pace(),
            )
            self.pending = None
        self.paused = True
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)

    def _exit_pause(self) -> None:
        """Re-grab the cursor and resume gameplay."""
        self.paused = False
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
        pygame.mouse.get_rel()  # discard accumulated delta during pause

    # Map pygame keys / mouse buttons to the Windows VK integers that
    # CtrlInput.button expects (see https://github.com/Overworldai/owl-control
    # keycode table). Covers the main ANSI rows, space, shift, and three
    # mouse buttons — enough for WASD / spacebar / look-around gameplay.
    _KEY_TO_VK: ClassVar[dict[int, int]] = (
        {getattr(pygame, f"K_{ch}"): ord(ch) for ch in "1234567890"}
        | {pygame.K_MINUS: 0xBD, pygame.K_EQUALS: 0xBB}
        | {getattr(pygame, f"K_{ch}"): ord(ch.upper()) for ch in "qwertyuiop"}
        | {
            pygame.K_LEFTBRACKET: 0xDB,
            pygame.K_RIGHTBRACKET: 0xDD,
            pygame.K_BACKSLASH: 0xDC,
        }
        | {getattr(pygame, f"K_{ch}"): ord(ch.upper()) for ch in "asdfghjkl"}
        | {pygame.K_SEMICOLON: 0xBA, pygame.K_QUOTE: 0xDE}
        | {getattr(pygame, f"K_{ch}"): ord(ch.upper()) for ch in "zxcvbnm"}
        | {pygame.K_COMMA: 0xBC, pygame.K_PERIOD: 0xBE, pygame.K_SLASH: 0xBF}
        | {pygame.K_SPACE: 0x20, pygame.K_LSHIFT: 0x10, pygame.K_RSHIFT: 0x10}
    )
    _MOUSE_TO_VK: ClassVar[dict[int, int]] = {1: 0x01, 2: 0x04, 3: 0x02}
    """pygame button ids 1/2/3 → VK 0x01 LBUTTON / 0x04 MBUTTON / 0x02 RBUTTON."""

    def _process_events(self) -> bool:
        """Drain pygame events and update state. Returns False to quit."""
        self.scroll = 0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False

            # Auto-pause when the cursor leaves the window. Safety net for
            # WMs where `set_grab` is advisory and the cursor can escape.
            elif e.type == pygame.WINDOWLEAVE and not self.paused:
                self._enter_pause()

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE and not self.paused:
                    self._enter_pause()
                elif e.key == pygame.K_u and not self.paused:
                    self.pending = None
                    self.engine.reset()
                else:
                    vk = self._KEY_TO_VK.get(e.key)
                    if vk is not None:
                        self.held_vks.add(vk)

            elif e.type == pygame.KEYUP:
                vk = self._KEY_TO_VK.get(e.key)
                if vk is not None:
                    self.held_vks.discard(vk)

            elif e.type == pygame.MOUSEBUTTONDOWN:
                if self.paused and e.button == 1:
                    self._exit_pause()
                else:
                    vk = self._MOUSE_TO_VK.get(e.button)
                    if vk is not None:
                        self.held_vks.add(vk)

            elif e.type == pygame.MOUSEWHEEL:
                self.scroll += e.y

            elif e.type == pygame.MOUSEBUTTONUP:
                vk = self._MOUSE_TO_VK.get(e.button)
                if vk is not None:
                    self.held_vks.discard(vk)

        return True

    def run(self) -> None:
        """Interactive generation loop. Starts auto-paused on the first frame.

        See "Frame pacing strategy" at the top of the file for the full
        pipeline design. The first iteration after (un)pause has no previous
        batch to render, so there is no GPU overlap on that frame.
        """
        self.renderer.render_frame(self.pending, 0.0, self.temporal_compression, 0.0)
        self.pending = None
        log.info("ready")

        while True:
            if not self._process_events():
                return

            if self.paused:
                self.renderer.draw_pause()
                self.clock.tick(self.engine.inference_fps)
                continue

            dx, dy = pygame.mouse.get_rel()
            ctrl = CtrlInput(
                button=set(self.held_vks),
                mouse=(dx * self.mouse_sensitivity, dy * self.mouse_sensitivity),
                scroll_wheel=self.scroll,
            )

            # Pipeline (see "Frame pacing strategy" at top of file):
            # 1. Dispatch gen_frame — GPU kernels are queued, returns fast.
            # 2. Render the *previous* batch with pacing sleeps while GPU works.
            # 3. .cpu() syncs the GPU and transfers the just-computed batch.
            # 4. Measure overhead (non-render time) to feed back into pacing.
            t0 = time.perf_counter()
            next_frames = self.engine.next_frame(ctrl=ctrl)
            if self.pending is not None:
                self._pace_s = self._compute_pace()
                self.renderer.render_frame(
                    self.pending,
                    self.batch_dt,
                    self.temporal_compression,
                    self._pace_s,
                )
            self.pending = next_frames.cpu()
            self.batch_dt = time.perf_counter() - t0
            self._overhead = self.batch_dt - self._pace_s


# --- entry point -------------------------------------------------------------


def get_seed(path: str | None) -> Image.Image:
    """Load a seed image from a local path, or download a random one from Biome."""
    if path is not None:
        log.info("loading seed from local file: %s", path)
        return Image.open(path)
    # GitHub contents API for the Biome `seeds/` directory, pinned to a known ref.
    biome_api = (
        "https://api.github.com/repos/Overworldai/Biome/contents/seeds?ref=14343a6"
    )
    log.info("fetching Biome seeds index")
    with urllib.request.urlopen(biome_api) as res:
        entries = [e for e in json.load(res) if e["type"] == "file"]
    url = random.choice(entries)["download_url"]
    log.info("downloading random Biome seed: %s", url)
    with urllib.request.urlopen(url) as res:
        return Image.open(io.BytesIO(res.read()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_uri", help="HF model id, e.g. Overworld/Waypoint-1.5-1B")
    ap.add_argument(
        "-s",
        "--seed",
        help="Path to a local seed image (defaults to a random Biome seed)",
    )
    ap.add_argument(
        "-q",
        "--quant",
        choices=["intw8a8", "fp8w8a8", "nvfp4"],
        default=None,
    )
    ap.add_argument("-d", "--device", default="cuda")
    ap.add_argument("-m", "--mouse-sensitivity", type=float, default=1.5)
    ap.add_argument(
        "--uncap-fps",
        action="store_true",
        help="Disable the inference_fps framerate cap (run as fast as the GPU allows)",
    )
    args = ap.parse_args()

    pygame.init()
    renderer = Renderer(args.model_uri)
    clock = pygame.time.Clock()

    try:
        renderer.draw_status("Loading model…")
        engine = Engine(args.model_uri, args.quant, args.device)
        renderer.draw_status("Loading seed…")
        engine.set_seed(get_seed(args.seed))
        renderer.draw_status("Priming engine…")
        engine.prime()
        renderer.draw_status("Warming up (torch.compile)…")
        first = engine.warmup()
        GameState(
            renderer,
            engine,
            clock,
            args.mouse_sensitivity,
            uncap_fps=args.uncap_fps,
            pending=first,
        ).run()
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
