"""
Low-latency WebSocket server for WorldEngine frame streaming.

Usage:
    python examples/websocket_server.py

Client connects via WebSocket to ws://localhost:8080/ws
"""

import asyncio
import base64
import io
import json
import logging
import time
import urllib.request
from dataclasses import dataclass

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
JPEG_QUALITY = 85

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
    logger.info("Downloading seed frame...")
    urllib.request.urlretrieve(SEED_URL, "/tmp/seed.png")
    logger.info("Reading seed image...")
    img = torchvision.io.read_image("/tmp/seed.png")
    img = img[:3].unsqueeze(0).float()
    frame = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)[0]
    result = frame.to(dtype=torch.uint8, device=DEVICE).permute(1, 2, 0).contiguous()
    logger.info(f"Seed frame ready: {result.shape}, {result.dtype}, {result.device}")
    return result


def load_engine():
    """Initialize the WorldEngine and seed frame."""
    global engine, seed_frame
    from world_engine import WorldEngine

    logger.info(f"Loading WorldEngine: {MODEL_URI} (quant={QUANT})")
    engine = WorldEngine(
        MODEL_URI,
        device=DEVICE,
        model_config_overrides={"n_frames": N_FRAMES, "ae_uri": "OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan"},
        quant=QUANT,
    )
    logger.info("WorldEngine loaded")
    seed_frame = load_seed_frame()
    logger.info("All initialization complete")


# ============================================================================
# Frame Encoding
# ============================================================================

def frame_to_jpeg(frame: torch.Tensor, quality: int = JPEG_QUALITY) -> bytes:
    """Convert frame tensor to JPEG bytes."""
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0, 255).to(torch.uint8)
    img = Image.fromarray(frame.cpu().numpy(), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ============================================================================
# Session Management
# ============================================================================

@dataclass
class Session:
    """Tracks state for a single WebSocket connection."""
    frame_count: int = 0
    max_frames: int = N_FRAMES - 2


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


# Status codes (client maps these to display text)
class Status:
    INIT = "init"          # Engine resetting
    LOADING = "loading"    # Loading seed frame
    READY = "ready"        # Ready for game loop
    RESET = "reset"        # Session reset


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for frame streaming.

    Protocol:
        Server -> Client:
            {"type": "status", "code": str}
            {"type": "frame", "data": base64_jpeg, "frame_id": int, "client_ts": float, "gen_ms": float}
            {"type": "error", "message": str}

        Client -> Server:
            {"type": "control", "buttons": [str], "mouse_dx": float, "mouse_dy": float, "ts": float}
            {"type": "reset"}

    Status codes: init, loading, ready, reset
    """
    from world_engine import CtrlInput

    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"Client connected: {client_host}")

    await websocket.accept()
    session = Session()

    async def send_json(data: dict):
        await websocket.send_text(json.dumps(data))

    try:
        await send_json({"type": "status", "code": Status.INIT})

        logger.info(f"[{client_host}] Calling engine.reset()...")
        await asyncio.to_thread(engine.reset)

        await send_json({"type": "status", "code": Status.LOADING})

        logger.info(f"[{client_host}] Calling append_frame...")
        await asyncio.to_thread(engine.append_frame, seed_frame)

        # Send initial frame so client has something to display
        jpeg = await asyncio.to_thread(frame_to_jpeg, seed_frame)
        await send_json({
            "type": "frame",
            "data": base64.b64encode(jpeg).decode("ascii"),
            "frame_id": 0,
            "client_ts": 0,
            "gen_ms": 0,
        })

        await send_json({"type": "status", "code": Status.READY})
        logger.info(f"[{client_host}] Ready for game loop")

        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                msg = json.loads(raw)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                logger.info(f"[{client_host}] Client disconnected")
                break

            msg_type = msg.get("type", "control")

            if msg_type == "reset":
                logger.info(f"[{client_host}] Reset requested")
                await asyncio.to_thread(engine.reset)
                await asyncio.to_thread(engine.append_frame, seed_frame)
                session.frame_count = 0
                await send_json({"type": "status", "code": Status.RESET})
                continue

            if msg_type == "control":
                buttons = {BUTTON_CODES[b.upper()] for b in msg.get("buttons", []) if b.upper() in BUTTON_CODES}
                mouse_dx = float(msg.get("mouse_dx", 0))
                mouse_dy = float(msg.get("mouse_dy", 0))
                client_ts = msg.get("ts", 0)

                if session.frame_count >= session.max_frames:
                    logger.info(f"[{client_host}] Auto-reset at frame limit")
                    await asyncio.to_thread(engine.reset)
                    await asyncio.to_thread(engine.append_frame, seed_frame)
                    session.frame_count = 0
                    await send_json({"type": "status", "code": Status.RESET})

                ctrl = CtrlInput(button=buttons, mouse=(mouse_dx, mouse_dy))

                t0 = time.perf_counter()
                frame = await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)
                gen_time = (time.perf_counter() - t0) * 1000

                session.frame_count += 1

                # Encode and send frame with timing info
                jpeg = await asyncio.to_thread(frame_to_jpeg, frame)
                await send_json({
                    "type": "frame",
                    "data": base64.b64encode(jpeg).decode("ascii"),
                    "frame_id": session.frame_count,
                    "client_ts": client_ts,
                    "gen_ms": gen_time,
                })

                # Logging
                logger.info(f"[{client_host}] Received control (buttons={buttons}, mouse=({mouse_dx},{mouse_dy})) -> Sent frame {session.frame_count} (gen={gen_time:.1f}ms)")
                if session.frame_count % 60 == 0:
                    logger.info(f"[{client_host}] Frame {session.frame_count} (gen={gen_time:.1f}ms)")

    except Exception as e:
        logger.error(f"[{client_host}] Error: {e}", exc_info=True)
        try:
            await send_json({"type": "error", "message": str(e)})
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
