"""
WorldEngine WebSocket Server (Headless)
No pygame, no popups - just WebSocket streaming

Run locally: python server.py
"""

import asyncio
import base64
import io
import json
import os
import time
import urllib.request
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

# =============================================================================
# Configuration - hardcoded for deployment
# =============================================================================
MODEL_URI = "OpenWorldLabs/Medium-0-NoCaption-SF-Shift8"
QUANT = "w8a8"
N_FRAMES = 4096
DEVICE = "cuda"

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


# =============================================================================
# Utilities
# =============================================================================

def load_seed_frame(target_size: tuple[int, int] = (360, 640)) -> torch.Tensor:
    """Load default seed frame as (H,W,3) uint8 on GPU"""
    print("[INIT] Downloading seed frame...")
    urllib.request.urlretrieve(SEED_URL, "/tmp/seed.png")
    print("[INIT] Reading seed image...")
    img = torchvision.io.read_image("/tmp/seed.png")
    img = img[:3].unsqueeze(0).float()
    frame = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)[0]
    result = frame.to(dtype=torch.uint8, device=DEVICE).permute(1, 2, 0).contiguous()
    print(f"[INIT] Seed frame ready: {result.shape}, {result.dtype}, {result.device}")
    return result


def frame_to_jpeg(frame: torch.Tensor, quality: int = 85) -> bytes:
    """Convert (H,W,3) tensor to JPEG bytes"""
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0, 255).to(torch.uint8)
    img = Image.fromarray(frame.cpu().numpy(), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# =============================================================================
# Global engine (loaded once)
# =============================================================================

engine = None
seed_frame = None


def load_engine():
    """Load WorldEngine and seed frame"""
    global engine, seed_frame
    
    from world_engine import WorldEngine
    
    print(f"[INIT] Loading WorldEngine: {MODEL_URI} (quant={QUANT})")
    engine = WorldEngine(
        MODEL_URI,
        device=DEVICE,
        model_config_overrides={"n_frames": N_FRAMES, "ae_uri": "OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan"},
        quant=QUANT,
    )
    print("[INIT] WorldEngine loaded")
    seed_frame = load_seed_frame()
    print("[INIT] All initialization complete")


# =============================================================================
# Session state
# =============================================================================

@dataclass
class Session:
    frame_count: int = 0
    max_frames: int = N_FRAMES - 2


# =============================================================================
# HTML Client (embedded)
# =============================================================================

HTML_CLIENT = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WorldEngine</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: #0a0a0a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        #game-container {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        #game-canvas {
            max-width: 100%;
            max-height: 100%;
            cursor: none;
        }
        #overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.85);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 100;
        }
        #overlay.hidden { display: none; }
        #overlay h1 { font-size: 3rem; margin-bottom: 1rem; }
        #overlay p { font-size: 1.2rem; opacity: 0.7; margin-bottom: 2rem; }
        #play-btn {
            padding: 16px 48px;
            font-size: 1.2rem;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            background: #2563eb;
            color: #fff;
            cursor: pointer;
        }
        #play-btn:hover { background: #1d4ed8; }
        #play-btn:disabled { background: #374151; cursor: not-allowed; }
        #hud {
            position: absolute;
            top: 16px;
            left: 16px;
            font-family: monospace;
            font-size: 14px;
            background: rgba(0,0,0,0.6);
            padding: 12px;
            border-radius: 8px;
            pointer-events: none;
        }
        #hud div { margin-bottom: 4px; }
        #controls-hint {
            position: absolute;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 13px;
            opacity: 0.5;
            pointer-events: none;
        }
        #status-bar {
            background: #111;
            padding: 8px 16px;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
            border-top: 1px solid #222;
        }
        .connected { color: #22c55e; }
        .disconnected { color: #ef4444; }
        .connecting { color: #eab308; }
    </style>
</head>
<body>
    <div id="game-container">
        <canvas id="game-canvas" width="1280" height="720"></canvas>
        <div id="overlay">
            <h1>🌍 WorldEngine</h1>
            <p>Real-time world model inference</p>
            <button id="play-btn">Click to Play</button>
        </div>
        <div id="hud">
            <div>FPS: <span id="fps">--</span></div>
            <div>Frame: <span id="frame-id">0</span></div>
            <div>Keys: <span id="active-keys">none</span></div>
        </div>
        <div id="controls-hint">
            WASD = Move | Mouse = Look | Space = Jump | Shift = Sprint | R = Action | U = Reset | ESC = Menu
        </div>
    </div>
    <div id="status-bar">
        <span>Status: <span id="status" class="disconnected">Disconnected</span></span>
        <span id="info">--</span>
    </div>
<script>
const canvas = document.getElementById('game-canvas');
const ctx = canvas.getContext('2d');
const overlay = document.getElementById('overlay');
const playBtn = document.getElementById('play-btn');
const statusEl = document.getElementById('status');
const infoEl = document.getElementById('info');

let ws = null;
let isConnected = false;
let isReady = false;  // Wait for server to be ready
let isPointerLocked = false;
let waitingForFrame = false;  // Prevent message queue buildup

const keys = new Set();
let mouseDx = 0, mouseDy = 0;
const sensitivity = 1.5;

let frameCount = 0, lastFpsUpdate = performance.now(), fps = 0;

const keyMap = {
    'KeyW': 'W', 'KeyA': 'A', 'KeyS': 'S', 'KeyD': 'D',
    'KeyR': 'R', 'Space': 'SPACE',
    'ShiftLeft': 'SHIFT', 'ShiftRight': 'SHIFT'
};

function getWsUrl() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    return proto + '//' + location.host + '/ws';
}

function connect() {
    statusEl.textContent = 'Connecting...';
    statusEl.className = 'connecting';
    playBtn.disabled = true;
    
    ws = new WebSocket(getWsUrl());
    
    ws.onopen = () => {
        isConnected = true;
        statusEl.textContent = 'Initializing...';
        statusEl.className = 'connecting';
        // Don't start game loop yet - wait for "connected" status
    };
    
    ws.onclose = () => {
        isConnected = false;
        isReady = false;
        waitingForFrame = false;
        statusEl.textContent = 'Disconnected';
        statusEl.className = 'disconnected';
        overlay.classList.remove('hidden');
        playBtn.disabled = false;
        if (document.pointerLockElement) document.exitPointerLock();
    };
    
    ws.onerror = () => {
        statusEl.textContent = 'Error';
        statusEl.className = 'disconnected';
    };
    
    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'frame') {
            waitingForFrame = false;
            
            // Calculate round-trip time
            if (msg.client_ts) {
                const rtt = performance.now() - msg.client_ts;
                document.getElementById('info').textContent = 
                    `RTT: ${rtt.toFixed(0)}ms | Gen: ${msg.gen_ms?.toFixed(0) || '?'}ms`;
            }
            
            const img = new Image();
            img.onload = () => {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                document.getElementById('frame-id').textContent = msg.frame_id;
                frameCount++;
                const now = performance.now();
                if (now - lastFpsUpdate >= 1000) {
                    document.getElementById('fps').textContent = frameCount;
                    frameCount = 0;
                    lastFpsUpdate = now;
                }
            };
            img.src = 'data:image/jpeg;base64,' + msg.data;
        } else if (msg.type === 'status') {
            infoEl.textContent = msg.message;
            if (msg.message === 'connected' && !isReady) {
                isReady = true;
                waitingForFrame = false;
                statusEl.textContent = 'Connected';
                statusEl.className = 'connected';
                overlay.classList.add('hidden');
                canvas.requestPointerLock();
                sendControl();
                requestAnimationFrame(gameLoop);
            }
        } else if (msg.type === 'error') {
            infoEl.textContent = 'Error: ' + msg.message;
        }
    };
}

function sendControl() {
    if (!ws || ws.readyState !== WebSocket.OPEN || !isReady || waitingForFrame) return;
    const buttonArr = Array.from(keys);
    waitingForFrame = true;
    ws.send(JSON.stringify({
        type: 'control',
        buttons: buttonArr,
        mouse_dx: 0,  // Disabled - causes latency
        mouse_dy: 0,
        ts: performance.now()
    }));
    mouseDx = mouseDy = 0;
    document.getElementById('active-keys').textContent = keys.size ? Array.from(keys).join(' ') : 'none';
}

function gameLoop() {
    if (!isConnected || !isReady) return;
    sendControl();
    requestAnimationFrame(gameLoop);
}

// playBtn handler is set below with overlay handling

document.addEventListener('keydown', (e) => {
    if (!isConnected || !isReady) return;
    if (e.code === 'Escape') {
        if (document.pointerLockElement) document.exitPointerLock();
        overlay.classList.remove('hidden');
        return;
    }
    if (e.code === 'KeyU') {
        ws.send(JSON.stringify({ type: 'reset' }));
        return;
    }
    if (keyMap[e.code]) { 
        keys.add(keyMap[e.code]); 
        e.preventDefault(); 
    }
});

document.addEventListener('keyup', (e) => {
    if (keyMap[e.code]) keys.delete(keyMap[e.code]);
});

document.addEventListener('mousedown', (e) => {
    if (!isConnected || !isReady || !isPointerLocked) return;
    if (e.button === 0) keys.add('MOUSE_LEFT');
    if (e.button === 1) keys.add('MOUSE_MIDDLE');
    if (e.button === 2) keys.add('MOUSE_RIGHT');
});

document.addEventListener('mouseup', (e) => {
    if (e.button === 0) keys.delete('MOUSE_LEFT');
    if (e.button === 1) keys.delete('MOUSE_MIDDLE');
    if (e.button === 2) keys.delete('MOUSE_RIGHT');
});

document.addEventListener('mousemove', (e) => {
    if (!isPointerLocked) return;
    mouseDx += e.movementX;
    mouseDy += e.movementY;
});

document.addEventListener('pointerlockchange', () => {
    isPointerLocked = !!document.pointerLockElement;
    if (!isPointerLocked && isConnected) overlay.classList.remove('hidden');
});

canvas.onclick = () => {
    if (isConnected && !isPointerLocked) {
        overlay.classList.add('hidden');
        canvas.requestPointerLock();
    }
};

// Click overlay to resume game
overlay.onclick = (e) => {
    if (isConnected && isReady && e.target === overlay) {
        overlay.classList.add('hidden');
        canvas.requestPointerLock();
    }
};

// Also make play button resume if already connected
playBtn.onclick = () => {
    if (isConnected && isReady) {
        overlay.classList.add('hidden');
        canvas.requestPointerLock();
    } else if (!isConnected) {
        connect();
    }
};

canvas.oncontextmenu = (e) => e.preventDefault();
</script>
</body>
</html>
'''


# =============================================================================
# FastAPI Application
# =============================================================================

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="WorldEngine")


@app.on_event("startup")
async def startup():
    load_engine()


@app.get("/")
async def index():
    return HTMLResponse(content=HTML_CLIENT)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_URI,
        "quant": QUANT,
        "engine_loaded": engine is not None,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    from world_engine import CtrlInput
    
    print("[WS] Connection attempt...")
    await websocket.accept()
    print("[WS] Accepted")
    session = Session()
    
    async def send_json(data: dict):
        await websocket.send_text(json.dumps(data))
    
    async def send_frame(frame: torch.Tensor):
        jpeg = await asyncio.to_thread(frame_to_jpeg, frame)
        await send_json({
            "type": "frame",
            "data": base64.b64encode(jpeg).decode("ascii"),
            "frame_id": session.frame_count,
        })
    
    try:
        # Send status immediately so client knows we're alive
        await send_json({"type": "status", "message": "initializing..."})
        
        print("[WS] Calling engine.reset()...")
        await asyncio.to_thread(engine.reset)
        
        await send_json({"type": "status", "message": "loading frame..."})
        
        print("[WS] Reset done. Calling append_frame...")
        await asyncio.to_thread(engine.append_frame, seed_frame)
        
        print("[WS] Append done. Sending status...")
        await send_json({"type": "status", "message": "connected"})
        print("[WS] Ready for game loop")
        
        msg_count = 0
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                msg = json.loads(raw)
                msg_count += 1
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                print("[WS] Client disconnected")
                break
            
            msg_type = msg.get("type", "control")
            
            if msg_type == "reset":
                print("[WS] Reset requested")
                await asyncio.to_thread(engine.reset)
                await asyncio.to_thread(engine.append_frame, seed_frame)
                session.frame_count = 0
                await send_json({"type": "status", "message": "reset"})
                continue
            
            if msg_type == "control":
                buttons = {BUTTON_CODES[b.upper()] for b in msg.get("buttons", []) if b.upper() in BUTTON_CODES}
                mouse_dx = float(msg.get("mouse_dx", 0))
                mouse_dy = float(msg.get("mouse_dy", 0))
                client_ts = msg.get("ts", 0)  # Client timestamp
                
                if session.frame_count >= session.max_frames:
                    await asyncio.to_thread(engine.reset)
                    await asyncio.to_thread(engine.append_frame, seed_frame)
                    session.frame_count = 0
                    await send_json({"type": "status", "message": "reset"})
                
                ctrl = CtrlInput(button=buttons, mouse=(mouse_dx, mouse_dy))
                
                t0 = time.perf_counter()
                frame = await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)
                gen_time = (time.perf_counter() - t0) * 1000
                
                session.frame_count += 1
                
                # Send frame with timing info
                jpeg = await asyncio.to_thread(frame_to_jpeg, frame)
                await send_json({
                    "type": "frame",
                    "data": base64.b64encode(jpeg).decode("ascii"),
                    "frame_id": session.frame_count,
                    "client_ts": client_ts,  # Echo back for RTT calculation
                    "gen_ms": gen_time,
                })
                
                # Log FPS every 60 frames
                if session.frame_count % 60 == 0:
                    print(f"[WS] Frame {session.frame_count} (gen_time={gen_time:.1f}ms)")
    
    except Exception as e:
        import traceback
        print(f"[WS] ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            await send_json({"type": "error", "message": str(e)})
        except:
            pass


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, ws_ping_interval=300, ws_ping_timeout=300)
