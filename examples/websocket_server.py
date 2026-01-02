"""
Low-latency WebSocket server for WorldEngine frame streaming.

Optimizations for latency:
- Pre-allocated JPEG encoding buffer
- Minimal JSON overhead in message protocol
- Direct tensor-to-bytes pipeline
- No unnecessary copies or conversions
- Configurable JPEG quality for bandwidth/quality tradeoff

Usage:
    python examples/websocket_server.py

Client connects via WebSocket to ws://localhost:8080/ws
"""

import asyncio
import base64
import io
import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("websocket_server")

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

# ============================================================================
# Configuration
# ============================================================================

MODEL_URI = "OpenWorldLabs/Medium-0-NoCaption-SF-Shift8"
QUANT = "w8a8"
N_FRAMES = 4096
DEVICE = "cuda"
JPEG_QUALITY = 80  # Lower = faster encoding, smaller size, lower quality

BUTTON_CODES = {
    "W": ord("W"),
    "A": ord("A"),
    "S": ord("S"),
    "D": ord("D"),
    "R": ord("R"),
    "SPACE": ord(" "),
    "SHIFT": 0x10,
    "MOUSE_LEFT": 0x01,
    "MOUSE_RIGHT": 0x02,
    "MOUSE_MIDDLE": 0x04,
}

SEED_URL = "https://gist.github.com/user-attachments/assets/5d91c49a-2ae9-418f-99c0-e93ae387e1de"

# ============================================================================
# Engine Setup
# ============================================================================

engine = None
seed_frame = None


def load_seed_frame(target_size: tuple[int, int] = (360, 640)) -> torch.Tensor:
    """Load and preprocess the seed frame."""
    urllib.request.urlretrieve(SEED_URL, "/tmp/seed.png")
    img = torchvision.io.read_image("/tmp/seed.png")
    img = img[:3].unsqueeze(0).float()
    frame = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)[0]
    return frame.to(dtype=torch.uint8, device=DEVICE).permute(1, 2, 0).contiguous()


def load_engine():
    """Initialize the WorldEngine and seed frame."""
    global engine, seed_frame
    from world_engine import WorldEngine

    engine = WorldEngine(
        MODEL_URI,
        device=DEVICE,
        model_config_overrides={"n_frames": N_FRAMES, "ae_uri": "OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan"},
        quant=QUANT,
    )
    seed_frame = load_seed_frame()


# ============================================================================
# Frame Encoding (Latency Optimized)
# ============================================================================

_jpeg_buffer = io.BytesIO()


def frame_to_jpeg(frame: torch.Tensor, quality: int = JPEG_QUALITY) -> bytes:
    """Convert frame tensor to JPEG bytes with minimal latency."""
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0, 255).to(torch.uint8)

    _jpeg_buffer.seek(0)
    _jpeg_buffer.truncate()

    img = Image.fromarray(frame.cpu().numpy(), mode="RGB")
    img.save(_jpeg_buffer, format="JPEG", quality=quality, optimize=False)
    return _jpeg_buffer.getvalue()


def frame_to_base64(frame: torch.Tensor) -> str:
    """Convert frame to base64-encoded JPEG string."""
    jpeg_bytes = frame_to_jpeg(frame)
    return base64.b64encode(jpeg_bytes).decode("ascii")


# ============================================================================
# Session Management
# ============================================================================

@dataclass
class Session:
    """Tracks state for a single WebSocket connection."""
    frame_count: int = 0
    max_frames: int = N_FRAMES - 2

    def should_reset(self) -> bool:
        return self.frame_count >= self.max_frames

    def reset(self):
        self.frame_count = 0

    def increment(self):
        self.frame_count += 1


# ============================================================================
# Message Protocol
# ============================================================================

class MessageType:
    """WebSocket message types."""
    STATUS = "status"
    FRAME = "frame"
    ERROR = "error"
    CONTROL = "control"
    RESET = "reset"


def parse_buttons(button_names: list[str]) -> Set[int]:
    """Convert button name strings to button code set."""
    return {
        BUTTON_CODES[name.upper()]
        for name in button_names
        if name.upper() in BUTTON_CODES
    }


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="WorldEngine WebSocket Server")


@app.on_event("startup")
async def startup():
    load_engine()


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "healthy",
        "model": MODEL_URI,
        "quant": QUANT,
        "engine_loaded": engine is not None,
    })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for frame streaming.

    Protocol:
        Server -> Client:
            {"type": "status", "message": str}
            {"type": "frame", "data": base64_jpeg, "id": int}
            {"type": "error", "message": str}

        Client -> Server:
            {"type": "control", "buttons": [str], "mouse_dx": float, "mouse_dy": float}
            {"type": "reset"}
    """
    from world_engine import CtrlInput

    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"Client connected: {client_host}")

    await websocket.accept()
    session = Session()

    async def send(msg_type: str, **kwargs):
        await websocket.send_text(json.dumps({"type": msg_type, **kwargs}))

    async def send_frame(frame: torch.Tensor):
        b64_data = await asyncio.to_thread(frame_to_base64, frame)
        await send(MessageType.FRAME, data=b64_data, id=session.frame_count)

    async def do_reset():
        await asyncio.to_thread(engine.reset)
        await asyncio.to_thread(engine.append_frame, seed_frame)
        session.reset()
        await send(MessageType.STATUS, message="reset")

    try:
        await send(MessageType.STATUS, message="initializing")
        await asyncio.to_thread(engine.reset)

        await send(MessageType.STATUS, message="loading seed frame")
        await asyncio.to_thread(engine.append_frame, seed_frame)

        await send(MessageType.STATUS, message="ready")

        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                msg = json.loads(raw)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break

            msg_type = msg.get("type", MessageType.CONTROL)
            logger.info(f"[{client_host}] Received: {msg_type}")

            if msg_type == MessageType.RESET:
                logger.info(f"[{client_host}] Reset requested")
                await do_reset()
                continue

            if msg_type == MessageType.CONTROL:
                if session.should_reset():
                    logger.info(f"[{client_host}] Auto-reset at frame limit")
                    await do_reset()

                buttons_raw = msg.get("buttons", [])
                buttons = parse_buttons(buttons_raw)
                mouse_dx = float(msg.get("mouse_dx", 0))
                mouse_dy = float(msg.get("mouse_dy", 0))

                logger.info(
                    f"[{client_host}] Frame {session.frame_count}: "
                    f"buttons={buttons_raw}, mouse=({mouse_dx:.1f}, {mouse_dy:.1f})"
                )

                ctrl = CtrlInput(button=buttons, mouse=(mouse_dx, mouse_dy))
                frame = await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)
                session.increment()

                await send_frame(frame)

    except Exception as e:
        logger.error(f"[{client_host}] Error: {e}")
        try:
            await send(MessageType.ERROR, message=str(e))
        except Exception:
            pass
    finally:
        logger.info(f"[{client_host}] Disconnected (frames: {session.frame_count})")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WorldEngine WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_ping_interval=300,
        ws_ping_timeout=300,
    )
