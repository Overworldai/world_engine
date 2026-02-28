#!/usr/bin/env python3
"""
Async WebSocket video streamer for real-time browser rendering (no X11).

Streams frames from the WorldEngine combat pipeline over WebSocket.
Optional input channel: text messages like {"type":"action","id":123} update current action.

Usage:
  python -m ws_stream.server --host 0.0.0.0 --port 8765 --fps 30

SSH port forward example (from your local machine):
  ssh -N -L 8765:localhost:8765 <user>@<server>
"""

from __future__ import annotations
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional, Set, Union

import cv2
import numpy as np
import torch
import websockets
from PIL import Image

from world_engine import WorldEngine, CtrlInput

frame_count = 0


def actionid_to_multihot(action_id: int, num_buttons=8) -> np.ndarray:
    buttons = np.zeros(num_buttons, dtype=bool)
    for i in range(num_buttons):
        if action_id & (1 << i):
            buttons[i] = 1
    return buttons


@dataclass
class StreamConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    fps: int = 30
    codec: str = "jpeg"       # "jpeg" or "webp"
    quality: int = 80         # 1-100
    send_timeout_s: float = 0.25
    heartbeat_s: float = 15.0
    frame_format: str = "bgr"


class FrameEncoder:
    def __init__(self, codec: str = "jpeg", quality: int = 80, frame_format: str = "bgr") -> None:
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

    def encode(self, frame: Union[np.ndarray, torch.Tensor]) -> bytes:
        if frame is None:
            raise ValueError("frame is None")
        if isinstance(frame, torch.Tensor):
            t = frame.detach().cpu()
            if t.dtype != torch.uint8:
                t = t.to(torch.uint8)
            arr = t.numpy()
        elif isinstance(frame, np.ndarray):
            arr = frame
        else:
            raise TypeError("frame must be numpy array or torch tensor")

        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)

        arr = np.ascontiguousarray(arr)

        if self.frame_format == "rgb":
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
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
        self.clients: Set["websockets.WebSocketServerProtocol"] = set()
        self._latest_bytes: Optional[bytes] = None
        self._latest_mime: str = encoder.mime
        self._lock = asyncio.Lock()
        self._last_frame_ts = 0.0
        self._last_stale_log_ts = 0.0
        self._frame_event = asyncio.Event()
        self._current_action_id: int = 0
        self._last_logged_action: int = -1
        self.log_actions: bool = False
        self._latest_raw: Optional[Union[np.ndarray, torch.Tensor]] = None
        self._encode_event = asyncio.Event()

    async def register(self, ws: "websockets.WebSocketServerProtocol") -> None:
        async with self._lock:
            self.clients.add(ws)
        await self._safe_send(ws, json.dumps({"type": "meta", "mime": self._latest_mime}))
        try:
            print(f"[ws] client connected: {ws.remote_address}")
        except Exception:
            pass

    async def unregister(self, ws: "websockets.WebSocketServerProtocol") -> None:
        async with self._lock:
            self.clients.discard(ws)
        try:
            print(f"[ws] client disconnected: {ws.remote_address}")
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
            raise

    async def broadcast_latest(self) -> None:
        frame_interval = 1.0 / float(self.cfg.fps)
        while True:
            start = time.perf_counter()
            if self._latest_bytes is not None:
                async with self._lock:
                    targets = list(self.clients)
                if targets:
                    results = await asyncio.gather(
                        *(self._safe_send(ws, self._latest_bytes) for ws in targets),
                        return_exceptions=True,
                    )
                    for ws, res in zip(targets, results):
                        if isinstance(res, Exception):
                            try:
                                await ws.close()
                            except Exception:
                                pass
                            await self.unregister(ws)
            now_mon = time.monotonic()
            if self._last_frame_ts > 0 and (now_mon - self._last_frame_ts) > max(2.0, 2.0 * frame_interval):
                if now_mon - self._last_stale_log_ts > 1.0:
                    delay = now_mon - self._last_frame_ts
                    print(f"[ws] warning: no new frame for {delay:.1f}s (streaming last frame)")
                    self._last_stale_log_ts = now_mon
            elapsed = time.perf_counter() - start
            await asyncio.sleep(max(0.0, frame_interval - elapsed))

    def publish_frame(self, frame: Union[np.ndarray, torch.Tensor]) -> None:
        global frame_count
        if frame.ndim >= 3 and frame.shape[-1] > 3:
            frame = frame[:, :, -3:]
        frame_count += 1
        self._latest_raw = frame
        self._encode_event.set()

    async def encode_worker(self) -> None:
        while True:
            await self._encode_event.wait()
            self._encode_event.clear()
            raw = self._latest_raw
            if raw is None:
                continue
            try:
                data = await asyncio.to_thread(self.encoder.encode, raw)
                self._latest_bytes = data
                self._last_frame_ts = time.monotonic()
                self._frame_event.set()
            except Exception as e:
                print(f"[encoder] failed: {e}")


async def tekken_producer_loop(hub: StreamHub, debug_overlay: bool = False) -> None:
    print("Initializing WorldEngine...")
    pipe = WorldEngine("/mnt/data/laplace/models/combat_sfpp/step_1408", model_config_overrides={"n_frames": 6400}, device="cuda")
    print("WorldEngine initialized.")

    img = Image.open("assets/seed_Frames/seed_data_orig/round_968_frame_0000.jpeg").convert("RGB")
    image_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    pipe.append_frame(image_tensor)
    print("Initial frame appended. Starting producer loop...")

    try:
        frame_interval = 1.0 / float(hub.cfg.fps) if hub.cfg.fps > 0 else 0.0
        while True:
            start = time.perf_counter()
            try:
                action_id = hub.current_action
                ctrl = CtrlInput(button=action_id, mouse=(0.0, 0.0))
                frame = pipe.gen_frame(ctrl=ctrl)
                frame = frame.cpu().numpy()[:, :, ::-1]  # RGB -> BGR

                if debug_overlay:
                    try:
                        arr = frame.numpy() if isinstance(frame, torch.Tensor) else frame
                        tstr = time.strftime('%H:%M:%S')
                        cv2.putText(arr, f"LIVE {tstr} action={action_id}", (10, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                        hub.publish_frame(arr)
                    except Exception:
                        hub.publish_frame(frame)
                else:
                    hub.publish_frame(frame)
            except Exception as e:
                import traceback
                print(f"[tekken] frame error: {e}")
                traceback.print_exc()
                await asyncio.sleep(0.01)
            if frame_interval > 0.0:
                elapsed = time.perf_counter() - start
                await asyncio.sleep(max(0.0, frame_interval - elapsed))
            else:
                await asyncio.sleep(0)
    except asyncio.CancelledError:
        pass


async def ws_handler(hub: StreamHub, ws: "websockets.WebSocketServerProtocol") -> None:
    await hub.register(ws)
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                continue
            try:
                data = json.loads(msg)
                if data.get("type") == "action" and "id" in data:
                    hub.set_action(int(data["id"]))
            except Exception:
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


async def demo_producer(hub: StreamHub) -> None:
    """Synthetic frame generator to validate the pipeline."""
    H, W = 448, 736
    t = 0.0
    try:
        while True:
            x = np.linspace(0, 1, W, dtype=np.float32)
            y = np.linspace(0, 1, H, dtype=np.float32)
            X, Y = np.meshgrid(x, y)
            r = (np.sin(2*np.pi*(X + t)) * 0.5 + 0.5)
            g = (np.sin(2*np.pi*(Y + t*0.8)) * 0.5 + 0.5)
            b = (np.sin(2*np.pi*(X+Y + t*1.2)) * 0.5 + 0.5)
            frame = np.dstack([(r*255).astype(np.uint8), (g*255).astype(np.uint8), (b*255).astype(np.uint8)])
            hub.publish_frame(frame)
            t += 0.01
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        pass


async def main_async(args: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Async WebSocket video streamer")
    parser.add_argument("--host", default=os.environ.get("WS_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("WS_PORT", 8765)))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("WS_FPS", 30)))
    parser.add_argument("--codec", choices=["jpeg", "webp"], default=os.environ.get("WS_CODEC", "jpeg"))
    parser.add_argument("--quality", type=int, default=int(os.environ.get("WS_QUALITY", 80)))
    parser.add_argument("--demo", action="store_true", help="Run with synthetic demo frames instead of WorldEngine")
    parser.add_argument("--debug-overlay", action="store_true", help="Draw live timestamp and action id on frames")
    parser.add_argument("--log-actions", action="store_true", help="Log action id changes received from client")
    parser.add_argument("--exit-after", type=float, default=None, help="Seconds to run then exit (for quick tests)")
    ns = parser.parse_args(args)

    cfg = StreamConfig(host=ns.host, port=ns.port, fps=ns.fps, codec=ns.codec, quality=ns.quality)
    encoder = FrameEncoder(cfg.codec, cfg.quality, frame_format=cfg.frame_format)
    hub = StreamHub(cfg, encoder)
    hub.log_actions = bool(ns.log_actions)

    async def connection_handler(*args):
        if len(args) == 1:
            ws = args[0]
        elif len(args) == 2:
            ws, _path = args
        else:
            raise RuntimeError(f"Unexpected connection handler signature: {len(args)} args")
        await ws_handler(hub, ws)

    print(f"Starting WebSocket server on {cfg.host}:{cfg.port} (codec={cfg.codec}, quality={cfg.quality}, frame_format={cfg.frame_format})")
    async with websockets.serve(connection_handler, cfg.host, cfg.port, max_size=None, compression=None, ping_interval=cfg.heartbeat_s):
        tasks = [
            asyncio.create_task(hub.broadcast_latest(), name="broadcast"),
            asyncio.create_task(heartbeat_task(hub), name="heartbeat"),
            asyncio.create_task(hub.encode_worker(), name="encoder"),
        ]
        print("WebSocket server started. Waiting for clients to connect...")
        if ns.demo:
            tasks.append(asyncio.create_task(demo_producer(hub), name="demo"))
        else:
            tasks.append(asyncio.create_task(tekken_producer_loop(hub, debug_overlay=ns.debug_overlay), name="tekken"))

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


def main() -> None:
    rc = asyncio.run(main_async(sys.argv[1:]))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
