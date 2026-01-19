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
from contextlib import asynccontextmanager
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

MODEL_URI = "OverWorld/Waypoint-Medium-Beta-2026-01-16"
QUANT = "w8a8"
N_FRAMES = 4096
DEVICE = "cuda"
JPEG_QUALITY = 85

BUTTON_CODES = {}
# A-Z keys
for i in range(65, 91):
    BUTTON_CODES[chr(i)] = i
# 0-9 keys
for i in range(10):
    BUTTON_CODES[str(i)] = ord(str(i))
# Special keys
BUTTON_CODES["UP"] = 0x26
BUTTON_CODES["DOWN"] = 0x28
BUTTON_CODES["LEFT"] = 0x25
BUTTON_CODES["RIGHT"] = 0x27
BUTTON_CODES["SHIFT"] = 0x10
BUTTON_CODES["CTRL"] = 0x11
BUTTON_CODES["SPACE"] = 0x20
BUTTON_CODES["TAB"] = 0x09
BUTTON_CODES["ENTER"] = 0x0D
BUTTON_CODES["MOUSE_LEFT"] = 0x01
BUTTON_CODES["MOUSE_RIGHT"] = 0x02
BUTTON_CODES["MOUSE_MIDDLE"] = 0x04


# SEED_URL = "https://gist.github.com/user-attachments/assets/5d91c49a-2ae9-418f-99c0-e93ae387e1de"
SEED_URL = "https://images.gamebanana.com/img/ss/mods/5aaaf43065f65.jpg"

# Default prompt - describes the expected visual style
DEFAULT_PROMPT = "First-person shooter gameplay footage from a true POV perspective, "
"the camera locked to the player's eyes as assault rifles, carbines, "
"machine guns, laser-sighted firearms, bullet-fed weapons, magazines, "
"barrels, muzzles, tracers, ammo, and launchers dominate the frame, "
"with constant gun handling, recoil, muzzle flash, shell ejection, "
"and ballistic impacts. Continuous real-time FPS motion with no cuts, "
"weapon-centric framing, realistic gun physics, authentic firearm "
"materials, high-caliber ammunition, laser optics, iron sights, and "
"relentless gun-driven action, rendered in ultra-realistic 4K at 60fps."


# ============================================================================
# Engine Setup
# ============================================================================

engine = None
seed_frame = None
CtrlInput = None
current_prompt = DEFAULT_PROMPT
engine_warmed_up = False

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

def load_seed_from_url(url, target_size=(360, 640)):
    """Load a seed frame from URL (used for prompt_with_seed)"""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            img_data = response.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        import numpy as np
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
        frame = F.interpolate(img_tensor, size=target_size, mode="bilinear", align_corners=False)[0]
        return frame.to(dtype=torch.uint8, device=DEVICE).permute(1, 2, 0).contiguous()
    except Exception as e:
        print(f"[ERROR] Failed to load seed from URL: {e}")
        return None

def load_engine():
    """Initialize the WorldEngine with configured model."""
    global engine, seed_frame, CtrlInput
    from world_engine import WorldEngine, CtrlInput as CI
    CtrlInput = CI

    logger.info(f"Loading WorldEngine: {MODEL_URI}")
    logger.info(f"N_FRAMES: {N_FRAMES}")
    logger.info(f"Default prompt: {DEFAULT_PROMPT[:80]}...")

    # Model config overrides
    # scheduler_sigmas: diffusion denoising schedule (MUST end with 0.0)
    # ae_uri: VAE model for encoding/decoding frames
    engine = WorldEngine(
        MODEL_URI,
        device=DEVICE,
        model_config_overrides={
            "n_frames": N_FRAMES,
            "ae_uri": "OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan",
            "scheduler_sigmas": [1.0, 0.8, 0.2, 0.0],
        },
        quant=QUANT,
        dtype=torch.bfloat16,
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    load_engine()
    yield
    # Shutdown (if needed in the future)

app = FastAPI(title="WorldEngine WebSocket Server", lifespan=lifespan)


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
    global seed_frame, current_prompt, engine_warmed_up
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"Client connected: {client_host}")

    await websocket.accept()
    session = Session()

    async def send_json(data: dict):
        await websocket.send_text(json.dumps(data))

    # Warmup on first connection (CUDA graphs will complain if its not all called in the same thread context)
    if not engine_warmed_up:
        logger.info("First connection - warming up model...")
        await send_json({"type": "status", "code": "warmup"})

        def do_warmup():
            warmup_start = time.perf_counter()
            engine.reset()
            engine.append_frame(seed_frame)
            engine.set_prompt(current_prompt)
            _ = engine.gen_frame(ctrl=CtrlInput(button=set(), mouse=(0.0, 0.0)))
            return time.perf_counter() - warmup_start

        warmup_time = await asyncio.to_thread(do_warmup)
        logger.info(f"Warmup complete ({warmup_time:.2f}s)")
        engine_warmed_up = True

    async def reset_engine():
        await asyncio.to_thread(engine.reset)
        await asyncio.to_thread(engine.append_frame, seed_frame)
        await asyncio.to_thread(engine.set_prompt, current_prompt)
        session.frame_count = 0
        await send_json({"type": "status", "code": Status.RESET})
        logger.info(f"[{client_host}] Engine Reset")

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
        paused = False

        # Helper to drain all pending messages and return only the latest control input
        async def get_latest_control():
            """Drain the message queue and return only the most recent control input."""
            latest_control_msg = None
            skipped_count = 0

            while True:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                    msg = json.loads(raw)

                    # Handle non-control messages immediately
                    msg_type = msg.get("type", "control")
                    if msg_type != "control":
                        return msg  # Return special messages immediately

                    # For control messages, keep only the latest
                    if latest_control_msg is not None:
                        skipped_count += 1
                    latest_control_msg = msg

                except asyncio.TimeoutError:
                    # No more messages in queue
                    # if skipped_count > 0:
                    #     logger.info(f"[{client_host}] Skipped {skipped_count} queued inputs, using latest")
                    return latest_control_msg
                except WebSocketDisconnect:
                    raise

        while True:
            try:
                msg = await get_latest_control()
                if msg is None:
                    continue
            except WebSocketDisconnect:
                logger.info(f"[{client_host}] Client disconnected")
                break

            msg_type = msg.get("type", "control")

            match msg_type:
                case "reset":
                    logger.info(f"[{client_host}] Reset requested")
                    await reset_engine()
                    continue
                case "pause":
                    # don't really have to do anything special for pausing
                    paused = True
                    logger.info("[RECV] Paused")
                case "resume":
                    # don't really have to do anything special for resuming
                    paused = False
                    logger.info("[RECV] Resumed")
                case "prompt":
                    new_prompt = msg.get("prompt", "").strip()
                    logger.info(f"[RECV] Prompt received: '{new_prompt[:50]}...'")
                    try:
                        current_prompt = new_prompt if new_prompt else DEFAULT_PROMPT
                        await reset_engine()
                    except Exception as e:
                        logger.info(f"[GEN] Failed to set prompt: {e}")
                case "prompt_with_seed":
                    new_prompt = msg.get("prompt", "").strip()
                    seed_url = msg.get("seed_url")
                    logger.info(f"[RECV] Prompt with seed: '{new_prompt}', URL: {seed_url}")
                    try:
                        if seed_url:
                            url_frame = load_seed_from_url(seed_url)
                            if url_frame is not None:
                                seed_frame = url_frame
                                logger.info("[RECV] Seed frame loaded from URL")
                        current_prompt = new_prompt if new_prompt else DEFAULT_PROMPT
                        logger.info("[RECV] Seed frame prompt loaded from URL, resetting engine")
                        await reset_engine()
                    except Exception as e:
                        logger.info(f"[GEN] Failed to set prompt: {e}")
                case "control":
                    if paused: continue
                    buttons = {BUTTON_CODES[b.upper()] for b in msg.get("buttons", []) if b.upper() in BUTTON_CODES}
                    mouse_dx = float(msg.get("mouse_dx", 0))
                    mouse_dy = float(msg.get("mouse_dy", 0))
                    client_ts = msg.get("ts", 0)

                    if session.frame_count >= session.max_frames:
                        logger.info(f"[{client_host}] Auto-reset at frame limit")
                        await reset_engine()

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
                    if session.frame_count % 60 == 0:
                        logger.info(f"[{client_host}] Received control (buttons={buttons}, mouse=({mouse_dx},{mouse_dy})) -> Sent frame {session.frame_count} (gen={gen_time:.1f}ms)")

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
