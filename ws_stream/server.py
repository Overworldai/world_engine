#!/usr/bin/env python3
"""
Async WebSocket video streamer for real-time browser rendering (no X11).

- Runs on the remote GPU box; streams compressed frames over WebSocket.
- Encodes frames as JPEG or WebP using OpenCV.
- Targets ~30 FPS with non-blocking asyncio.
- Easy integration: publish frames from your model via `publish_frame(np.ndarray)` or
  run the built-in Tekken pipeline loop (see --use-tekken).
- Optional input channel: text messages like `{"type":"action","id":123}` update current action.

Usage examples:
  python -m ws_stream.server --host 0.0.0.0 --port 8765 --codec jpeg --quality 80
  # With Tekken pipeline producing frames continuously
  python -m ws_stream.server --use-tekken --fps 30

Then open index.html in your browser and connect to ws://<SERVER_IP>:8765

SSH port forward example (from your local machine):
  ssh -N -L 8765:localhost:8765 <user>@<server>

Notes:
- Frames are expected as HxWx3 uint8 in RGB. If torch tensors are provided, they will be moved to CPU.
- Slow clients are pruned automatically to keep latency low.
"""

from __future__ import annotations
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Set, Deque, Union

import numpy as np
import cv2

# from src.world_engine import WorldEngine
from world_engine import WorldEngine

# Optional torch import (lazy-loaded only when actually needed)
torch = None  # type: ignore

# Tekken pipeline (optional, lazy-loaded)
global frame_count
frame_count = 0

import websockets


@dataclass
class StreamConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    fps: int = 30
    codec: str = "jpeg"  # "jpeg" or "webp"
    quality: int = 80     # 1-100 JPEG quality or WebP quality
    send_timeout_s: float = 0.25  # per-client send timeout
    heartbeat_s: float = 15.0     # ping interval
    frame_format: str = "rgb"     # incoming frame format: "rgb" or "bgr"


class FrameEncoder:
    def __init__(self, codec: str = "jpeg", quality: int = 80, frame_format: str = "rgb") -> None:
        codec = codec.lower()
        if codec not in ("jpeg", "webp"):
            raise ValueError("codec must be 'jpeg' or 'webp'")
        self.codec = codec
        self.quality = int(quality)
        ff = frame_format.lower()
        if ff not in ("rgb", "bgr"):
            raise ValueError("frame_format must be 'rgb' or 'bgr'")
        self.frame_format = ff
        if self.codec == "jpeg":
            self._fourcc = ".jpg"
            self._params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            self.mime = "image/jpeg"
        else:
            self._fourcc = ".webp"
            self._params = [cv2.IMWRITE_WEBP_QUALITY, self.quality]
            self.mime = "image/webp"

    def encode(self, frame: np.ndarray) -> bytes:
        # Expect HxWx3 RGB uint8
        if frame is None:
            raise ValueError("frame is None")
        if isinstance(frame, np.ndarray):
            arr = frame
        else:
            # Lazy import torch only when needed
            global torch  # type: ignore
            if torch is None:
                try:
                    import torch as _torch  # type: ignore
                    torch = _torch  # type: ignore
                except Exception:
                    torch = None  # type: ignore
            if torch is not None and isinstance(frame, torch.Tensor):  # type: ignore
                t = frame
                if t.device.type != 'cpu':
                    t = t.detach().to('cpu')
                if t.dtype != torch.uint8:
                    t = t.to(torch.uint8)
                arr = t.numpy()
            else:
                raise TypeError("frame must be numpy array or torch tensor")

        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape[-1] == 4:  # RGBA -> RGB (drop alpha)
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        # Ensure contiguous memory to avoid intermittent OpenCV encode failures
        arr = np.ascontiguousarray(arr)

        # OpenCV expects BGR for encoding
        if self.frame_format == "rgb":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            # already BGR
            bgr = arr
        ok, buf = cv2.imencode(self._fourcc, bgr, self._params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return buf.tobytes()


class StreamHub:
    """Manages clients and broadcasting of the latest frame efficiently."""
    def __init__(self, cfg: StreamConfig, encoder: FrameEncoder) -> None:
        self.cfg = cfg
        self.encoder = encoder
        # Use forward reference for type to avoid importing deprecated class at runtime
        self.clients: Set["websockets.WebSocketServerProtocol"] = set()
        self._latest_bytes: Optional[bytes] = None
        self._latest_mime: str = encoder.mime
        self._lock = asyncio.Lock()
        self._last_broadcast_ts = 0.0
        self._last_frame_ts = 0.0
        self._last_stale_log_ts = 0.0
        self._frame_event = asyncio.Event()
        self._current_action_id: int = 0  # optional control input
        self._last_logged_action: int = -1
        self.log_actions: bool = False
        # raw frame queue (latest wins) for async encoding
        self._latest_raw: Optional[Union[np.ndarray, "torch.Tensor"]] = None
        self._encode_event = asyncio.Event()

    async def register(self, ws: "websockets.WebSocketServerProtocol") -> None:
        async with self._lock:
            self.clients.add(ws)
        # send a tiny header as JSON for mime (send as text, not bytes)
        await self._safe_send(ws, json.dumps({"type": "meta", "mime": self._latest_mime}))
        try:
            peer = getattr(ws, "remote_address", None)
            print(f"[ws] client connected: {peer}")
        except Exception:
            pass

    async def unregister(self, ws: "websockets.WebSocketServerProtocol") -> None:
        async with self._lock:
            self.clients.discard(ws)
        try:
            peer = getattr(ws, "remote_address", None)
            print(f"[ws] client disconnected: {peer}")
        except Exception:
            pass

    @property
    def current_action(self) -> int:
        return self._current_action_id

    def set_action(self, action_id: int) -> None:
        self._current_action_id = int(action_id)
        if self.log_actions and self._current_action_id != self._last_logged_action:
            print(f"[input] action={self._current_action_id}")
            self._last_logged_action = self._current_action_id

    async def _safe_send(self, ws: "websockets.WebSocketServerProtocol", data: "bytes | str") -> None:
        try:
            await asyncio.wait_for(ws.send(data), timeout=self.cfg.send_timeout_s)
        except Exception:
            # Drop on send errors/timeouts; caller will clean up
            raise

    async def broadcast_latest(self) -> None:
        """Broadcast task that runs at target FPS and sends the most recent frame."""
        frame_interval = 1.0 / float(self.cfg.fps)
        while True:
            start = time.perf_counter()
            if self._latest_bytes is not None:
                # snapshot clients to avoid holding the lock during sends
                async with self._lock:
                    targets = list(self.clients)
                if targets:
                    results = await asyncio.gather(
                        *(self._safe_send(ws, self._latest_bytes) for ws in targets),
                        return_exceptions=True,
                    )
                    # prune failed clients
                    for ws, res in zip(targets, results):
                        if isinstance(res, Exception):
                            try:
                                await ws.close()
                            except Exception:
                                pass
                            await self.unregister(ws)
            # Log if frames are stale (helps debug producer stalls)
            now_mon = time.monotonic()
            if self._last_frame_ts > 0 and (now_mon - self._last_frame_ts) > max(2.0, 2.0 * frame_interval):
                # throttle logs to at most 1 per second
                if now_mon - self._last_stale_log_ts > 1.0:
                    delay = now_mon - self._last_frame_ts
                    print(f"[ws] warning: no new frame for {delay:.1f}s (streaming last frame)")
                    self._last_stale_log_ts = now_mon
            # maintain FPS pacing
            elapsed = time.perf_counter() - start
            await asyncio.sleep(max(0.0, frame_interval - elapsed))

    def publish_frame(self, frame: np.ndarray | "torch.Tensor") -> None:
        """Publish a raw frame; encoding is offloaded to a background task.
        Latest frame wins to keep latency low.
        """
        global frame_count
        # print(frame.shape)
        if frame.ndim >= 3 and frame.shape[-1] > 3:
            frame = frame[:, :, -3:]
        frame_count += 1
        # print(f"Published frame count: {frame_count}")
        self._latest_raw = frame
        self._encode_event.set()

    async def encode_worker(self) -> None:
        """Background encoder: converts raw frames to compressed bytes using a thread."""
        while True:
            await self._encode_event.wait()
            self._encode_event.clear()
            raw = self._latest_raw
            if raw is None:
                continue
            try:
                # Offload CPU-bound encode to thread to avoid blocking the event loop
                data = await asyncio.to_thread(self.encoder.encode, raw)
                self._latest_bytes = data
                self._last_frame_ts = time.monotonic()
                self._frame_event.set()
            except Exception as e:
                print(f"[encoder] failed: {e}")


async def tekken_producer_loop(
    hub: StreamHub,
    model_path: Optional[str] = None,
    debug_overlay: bool = False,
) -> None:
    
    print("Initializing WorldEngine for Tekken pipeline...")
    pipe = WorldEngine("/mnt/data/laplace/models/combat_sfpp/step_1408", model_config_overrides={"n_frames": 6400}, device="cuda")
    print("WorldEngine initialized.")

    try:
        frame_interval = 1.0 / float(hub.cfg.fps) if hub.cfg.fps > 0 else 0.0
        while True:
            start = time.perf_counter()
            try:
                action_id = hub.current_action
                print(f"Generating frame for action {action_id}...")
                frame = pipe.gen_frame()
                frame = frame.cpu().numpy()[:, :, ::-1]  # RGB -> BGR for OpenCV

                # Tekken returns BGR uint8 already; encoder can skip conversion if configured
                if debug_overlay:
                    # Draw a small live overlay with time and action id
                    try:
                        # Ensure we have a numpy array for cv2.putText
                        arr = None
                        try:
                            # torch may or may not be loaded; handle both
                            from torch import Tensor as _TorchTensor  # type: ignore
                            if isinstance(frame, _TorchTensor):
                                arr = frame.numpy()
                        except Exception:
                            arr = None
                        if arr is None:
                            arr = frame  # assume numpy
                        tstr = time.strftime('%H:%M:%S')
                        cv2.putText(arr, f"LIVE {tstr} action={action_id}", (10, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                        hub.publish_frame(arr)
                    except Exception:
                        # Fallback to publishing original frame on any overlay error
                        hub.publish_frame(frame)
                else:
                    hub.publish_frame(frame)
            except Exception as e:
                # Log and continue; do not kill the producer loop on transient errors
                print(f"[tekken] frame error: {e}")
                await asyncio.sleep(0.01)
            # Pace the producer to roughly match target FPS so simulation time is real-time
            if frame_interval > 0.0:
                elapsed = time.perf_counter() - start
                await asyncio.sleep(max(0.0, frame_interval - elapsed))
            else:
                await asyncio.sleep(0)  # yield to event loop
    except asyncio.CancelledError:
        pass


async def ws_handler(hub: StreamHub, ws: "websockets.WebSocketServerProtocol") -> None:
    await hub.register(ws)
    try:
        async for msg in ws:
            # We accept simple control messages from client
            if isinstance(msg, bytes):
                # ignore binary from client for now
                continue
            # text -> try parse json
            try:
                data = json.loads(msg)
                if data.get("type") == "action" and "id" in data:
                    hub.set_action(int(data["id"]))
                # Could extend with other message types
            except Exception:
                # also accept simple "action:123" text
                if msg.startswith("action:"):
                    try:
                        hub.set_action(int(msg.split(":", 1)[1]))
                    except Exception:
                        pass
    finally:
        await hub.unregister(ws)


async def heartbeat_task(hub: StreamHub) -> None:
    while True:
        await asyncio.sleep(hub.cfg.heartbeat_s)
        # Optionally could ping clients if needed; websockets lib has built-in pings.


async def main_async(args: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Async WebSocket video streamer")
    parser.add_argument("--host", default=os.environ.get("WS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WS_PORT", 8765)))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("WS_FPS", 30)))
    parser.add_argument("--codec", choices=["jpeg", "webp"], default=os.environ.get("WS_CODEC", "jpeg"))
    parser.add_argument("--quality", type=int, default=int(os.environ.get("WS_QUALITY", 80)))
    parser.add_argument("--use-tekken", action="store_true", help="Use TekkenPipeline as frame source")
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo frames if no pipeline")
    parser.add_argument("--frame-format", choices=["rgb","bgr"], default=os.environ.get("WS_FRAME_FORMAT", "rgb"), help="Incoming frame format for encoder.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional model path")
    parser.add_argument("--debug-overlay", action="store_true", help="Draw LIVE timestamp and action id on frames (debug)")
    parser.add_argument("--log-actions", action="store_true", help="Log action id changes received from client")
    parser.add_argument("--exit-after", type=float, default=None, help="Optional seconds to run then exit (for quick tests)")
    ns = parser.parse_args(args)

    cfg = StreamConfig(host=ns.host, port=ns.port, fps=ns.fps, codec=ns.codec, quality=ns.quality, frame_format=ns.frame_format)
    # If using Tekken, default to BGR frames for faster path (skip color convert)
    if ns.use_tekken:
        cfg.frame_format = "bgr"
    encoder = FrameEncoder(cfg.codec, cfg.quality, frame_format=cfg.frame_format)
    hub = StreamHub(cfg, encoder)
    hub.log_actions = bool(ns.log_actions)

    async def connection_handler(*args):
        """Compatible handler for websockets 10.x-12.x (optional path arg)."""
        if len(args) == 1:
            ws = args[0]
        elif len(args) == 2:
            ws, _path = args
        else:
            raise RuntimeError(f"Unexpected connection handler signature: {len(args)} args")
        await ws_handler(hub, ws)

    print(f"Starting WebSocket server on {cfg.host}:{cfg.port} (codec={cfg.codec}, quality={cfg.quality}, frame_format={cfg.frame_format})")
    # Disable permessage-deflate since frames are already compressed (avoids extra CPU cost)
    async with websockets.serve(connection_handler, cfg.host, cfg.port, max_size=None, compression=None, ping_interval=cfg.heartbeat_s):
        tasks = [
            asyncio.create_task(hub.broadcast_latest(), name="broadcast"),
            asyncio.create_task(heartbeat_task(hub), name="heartbeat"),
            asyncio.create_task(hub.encode_worker(), name="encoder"),
        ]
        print("WebSocket server started. Waiting for clients to connect...")
        model_path = ns.model_path if ns.model_path else ("/mnt/data/laplace/models/combat_sfpp/step_640" if ns.use_tekken else None)
        # Producer selection
        if ns.use_tekken:
            tasks.append(asyncio.create_task(tekken_producer_loop(hub, model_path, debug_overlay=ns.debug_overlay), name="tekken"))
        elif ns.demo:
            tasks.append(asyncio.create_task(demo_producer(hub), name="demo"))
        else:
            print("No producer started. Call hub.publish_frame(frame) from your code or run with --demo/--use-tekken.")

        # Graceful shutdown
        stop = asyncio.Future()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_running_loop().add_signal_handler(sig, stop.set_result, None)
            except NotImplementedError:
                pass

        if ns.exit_after is not None:
            try:
                await asyncio.wait_for(stop, timeout=ns.exit_after)
            except asyncio.TimeoutError:
                pass
        else:
            await stop
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return 0


async def demo_producer(hub: StreamHub) -> None:
    """Synthetic frame generator to validate the pipeline."""
    H, W = 448, 736
    t = 0.0
    try:
        while True:
            # simple moving gradient pattern
            x = np.linspace(0, 1, W, dtype=np.float32)
            y = np.linspace(0, 1, H, dtype=np.float32)
            X, Y = np.meshgrid(x, y)
            r = (np.sin(2*np.pi*(X + t)) * 0.5 + 0.5)
            g = (np.sin(2*np.pi*(Y + t*0.8)) * 0.5 + 0.5)
            b = (np.sin(2*np.pi*(X+Y + t*1.2)) * 0.5 + 0.5)
            frame = np.dstack([(r*255).astype(np.uint8), (g*255).astype(np.uint8), (b*255).astype(np.uint8)])
            hub.publish_frame(frame)
            t += 0.01
            await asyncio.sleep(0)  # yield
    except asyncio.CancelledError:
        pass


def main() -> None:
    rc = asyncio.run(main_async(sys.argv[1:]))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
