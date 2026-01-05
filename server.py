"""
WorldEngine WebSocket Server - Debug timing
"""

import asyncio
import base64
import io
import json
import time
import urllib.request
from queue import Queue, Empty
import threading

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

# =============================================================================
# Configuration
# =============================================================================
MODEL_URI = "OverWorld/Waypoint-1-Medium-Beta-2026-01-03"
QUANT = None
N_FRAMES = 4096
DEVICE = "cuda"

BUTTON_CODES = {
    "W": ord("W"), "A": ord("A"), "S": ord("S"), "D": ord("D"),
    "R": ord("R"), "SPACE": ord(" "), "SHIFT": 0x10,
    "MOUSE_LEFT": 0x01, "MOUSE_RIGHT": 0x02, "MOUSE_MIDDLE": 0x04,
}

SEED_URL = "https://gist.github.com/user-attachments/assets/5d91c49a-2ae9-418f-99c0-e93ae387e1de"

# =============================================================================
# Utilities
# =============================================================================

def load_seed_frame(target_size=(360, 640)):
    print("[INIT] Downloading seed frame...")
    urllib.request.urlretrieve(SEED_URL, "/tmp/seed.png")
    img = torchvision.io.read_image("/tmp/seed.png")[:3].unsqueeze(0).float()
    frame = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)[0]
    return frame.to(dtype=torch.uint8, device=DEVICE).permute(1, 2, 0).contiguous()


def frame_to_jpeg_cpu(cpu_array, quality=70):
    """Takes numpy array, returns JPEG bytes"""
    img = Image.fromarray(cpu_array, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# =============================================================================
# Global state
# =============================================================================
engine = None
seed_frame = None
CtrlInput = None


def load_engine():
    global engine, seed_frame, CtrlInput
    from world_engine import WorldEngine, CtrlInput as CI
    CtrlInput = CI
    
    print(f"[INIT] Loading WorldEngine: {MODEL_URI}")
    engine = WorldEngine(
        MODEL_URI, device=DEVICE,
        model_config_overrides={"n_frames": N_FRAMES, "ae_uri": "OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan"},
        quant=QUANT, dtype=torch.bfloat16,
    )
    seed_frame = load_seed_frame()
    print("[INIT] Ready")


# =============================================================================
# HTML Client
# =============================================================================
HTML_CLIENT = '''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><title>WorldEngine</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#fff;font-family:system-ui;overflow:hidden;height:100vh;display:flex;flex-direction:column}
#game-container{flex:1;display:flex;align-items:center;justify-content:center;position:relative}
#game-canvas{max-width:100%;max-height:100%;cursor:none}
#overlay{position:absolute;inset:0;background:rgba(0,0,0,0.85);display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:100}
#overlay.hidden{display:none}
#overlay h1{font-size:3rem;margin-bottom:1rem}
#overlay p{opacity:0.7;margin-bottom:2rem}
#play-btn{padding:16px 48px;font-size:1.2rem;border:none;border-radius:8px;background:#2563eb;color:#fff;cursor:pointer}
#hud{position:absolute;top:16px;left:16px;font-family:monospace;font-size:14px;background:rgba(0,0,0,0.6);padding:12px;border-radius:8px}
#hud div{margin-bottom:4px}
#status-bar{background:#111;padding:8px 16px;font-size:13px;display:flex;justify-content:space-between;border-top:1px solid #222}
.connected{color:#22c55e}.disconnected{color:#ef4444}.connecting{color:#eab308}
</style>
</head>
<body>
<div id="game-container">
<canvas id="game-canvas" width="1280" height="720"></canvas>
<div id="overlay"><h1>WorldEngine</h1><p>Real-time world model</p><button id="play-btn">Click to Play</button></div>
<div id="hud"><div>FPS: <span id="fps">--</span></div><div>Frame: <span id="frame-id">0</span></div><div>Keys: <span id="active-keys">none</span></div></div>
</div>
<div id="status-bar"><span>Status: <span id="status" class="disconnected">Disconnected</span></span><span id="info">--</span></div>
<script>
const canvas=document.getElementById('game-canvas'),ctx=canvas.getContext('2d'),overlay=document.getElementById('overlay'),playBtn=document.getElementById('play-btn'),statusEl=document.getElementById('status'),infoEl=document.getElementById('info');
let ws=null,isConnected=false,isReady=false,isPointerLocked=false,controlInterval=null;
const keys=new Set();let mouseDx=0,mouseDy=0;const sensitivity=1.5;
let frameCount=0,lastFpsUpdate=performance.now();
const keyMap={'KeyW':'W','KeyA':'A','KeyS':'S','KeyD':'D','KeyR':'R','Space':'SPACE','ShiftLeft':'SHIFT','ShiftRight':'SHIFT'};

function connect(){
if(ws)ws.close();
statusEl.textContent='Connecting...';statusEl.className='connecting';
ws=new WebSocket((location.protocol==='https:'?'wss:':'ws:')+'//'+location.host+'/ws');
ws.onopen=()=>{isConnected=true;statusEl.textContent='Initializing...';};
ws.onclose=()=>{isConnected=false;isReady=false;if(controlInterval)clearInterval(controlInterval);statusEl.textContent='Disconnected';statusEl.className='disconnected';overlay.classList.remove('hidden');};
ws.onmessage=(e)=>{
const msg=JSON.parse(e.data);
if(msg.type==='frame'){
infoEl.textContent=msg.timing;
const img=new Image();
img.onload=()=>{ctx.drawImage(img,0,0,canvas.width,canvas.height);document.getElementById('frame-id').textContent=msg.frame_id;frameCount++;const now=performance.now();if(now-lastFpsUpdate>=1000){document.getElementById('fps').textContent=frameCount;frameCount=0;lastFpsUpdate=now;}};
img.src='data:image/jpeg;base64,'+msg.data;
}else if(msg.type==='status'&&msg.message==='connected'&&!isReady){
isReady=true;statusEl.textContent='Connected';statusEl.className='connected';overlay.classList.add('hidden');canvas.requestPointerLock();
controlInterval=setInterval(sendControl,16);
}};
}

function sendControl(){
if(!ws||ws.readyState!==1||!isReady)return;
ws.send(JSON.stringify({type:'control',buttons:Array.from(keys),mouse_dx:mouseDx*sensitivity,mouse_dy:mouseDy*sensitivity}));
mouseDx=mouseDy=0;
document.getElementById('active-keys').textContent=keys.size?Array.from(keys).join(' '):'none';
}

document.addEventListener('keydown',e=>{if(!isReady)return;if(e.code==='Escape'){document.exitPointerLock();overlay.classList.remove('hidden');return;}if(e.code==='KeyU'){ws.send(JSON.stringify({type:'reset'}));return;}if(keyMap[e.code]){keys.add(keyMap[e.code]);e.preventDefault();}});
document.addEventListener('keyup',e=>{if(keyMap[e.code])keys.delete(keyMap[e.code]);});
document.addEventListener('mousemove',e=>{if(document.pointerLockElement){mouseDx+=e.movementX;mouseDy+=e.movementY;}});
document.addEventListener('pointerlockchange',()=>{isPointerLocked=!!document.pointerLockElement;if(!isPointerLocked&&isConnected)overlay.classList.remove('hidden');});
canvas.onclick=()=>{if(isConnected&&!isPointerLocked){overlay.classList.add('hidden');canvas.requestPointerLock();}};
playBtn.onclick=()=>{if(!isConnected)connect();else if(isReady){overlay.classList.add('hidden');canvas.requestPointerLock();}};
canvas.oncontextmenu=e=>e.preventDefault();
</script>
</body></html>'''


# =============================================================================
# FastAPI
# =============================================================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

@app.on_event("startup")
async def startup():
    load_engine()

@app.get("/")
async def index():
    return HTMLResponse(HTML_CLIENT)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[WS] Connected")
    
    # Shared state
    ctrl_state = {"buttons": set(), "mouse_dx": 0.0, "mouse_dy": 0.0}
    ctrl_lock = threading.Lock()
    frame_queue = Queue(maxsize=3)
    running = True
    reset_flag = False
    
    async def receiver():
        nonlocal running, reset_flag
        while running:
            try:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "reset":
                    reset_flag = True
                elif msg.get("type") == "control":
                    with ctrl_lock:
                        ctrl_state["buttons"] = {BUTTON_CODES[b.upper()] for b in msg.get("buttons", []) if b.upper() in BUTTON_CODES}
                        ctrl_state["mouse_dx"] += float(msg.get("mouse_dx", 0))
                        ctrl_state["mouse_dy"] += float(msg.get("mouse_dy", 0))
            except:
                running = False
                break
    
    async def sender():
        nonlocal running
        while running:
            try:
                try:
                    frame_data = frame_queue.get(timeout=0.01)
                except Empty:
                    await asyncio.sleep(0.001)
                    continue
                
                t0 = time.perf_counter()
                await ws.send_text(json.dumps(frame_data))
                send_ms = (time.perf_counter() - t0) * 1000
                if send_ms > 5:
                    print(f"[SEND] slow: {send_ms:.1f}ms")
                    
            except Exception as e:
                print(f"[SEND] Error: {e}")
                running = False
                break
    
    def generator():
        nonlocal running, reset_flag
        frame_count = 0
        max_frames = N_FRAMES - 2
        
        # Init
        engine.reset()
        engine.append_frame(seed_frame)
        frame_queue.put({"type": "status", "message": "connected"})
        
        loop_start = time.perf_counter()
        loop_count = 0
        
        while running:
            try:
                if reset_flag or frame_count >= max_frames:
                    engine.reset()
                    engine.append_frame(seed_frame)
                    frame_count = 0
                    reset_flag = False
                
                # Snapshot controls
                with ctrl_lock:
                    buttons = ctrl_state["buttons"].copy()
                    mdx, mdy = ctrl_state["mouse_dx"], ctrl_state["mouse_dy"]
                    ctrl_state["mouse_dx"] = ctrl_state["mouse_dy"] = 0.0
                
                # === DETAILED TIMING ===
                t0 = time.perf_counter()
                
                # Generate frame
                if buttons or abs(mdx) > 0.1 or abs(mdy) > 0.1: print(f"[CTRL] buttons={buttons} mouse=({mdx:.1f},{mdy:.1f})")
                frame = engine.gen_frame(ctrl=CtrlInput(button=buttons, mouse=(mdx, mdy)))
                t1 = time.perf_counter()
                
                # Sync GPU
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                
                # GPU -> CPU
                cpu_array = frame.cpu().numpy()
                t3 = time.perf_counter()
                
                # JPEG encode
                jpeg = frame_to_jpeg_cpu(cpu_array)
                t4 = time.perf_counter()
                
                # Base64
                b64 = base64.b64encode(jpeg).decode('ascii')
                t5 = time.perf_counter()
                
                gen_ms = (t1 - t0) * 1000
                sync_ms = (t2 - t1) * 1000
                cpu_ms = (t3 - t2) * 1000
                jpg_ms = (t4 - t3) * 1000
                b64_ms = (t5 - t4) * 1000
                total_ms = (t5 - t0) * 1000
                
                frame_count += 1
                loop_count += 1
                
                timing_str = f"gen={gen_ms:.1f} sync={sync_ms:.1f} cpu={cpu_ms:.1f} jpg={jpg_ms:.1f} b64={b64_ms:.1f}"
                
                # Queue frame
                frame_data = {
                    "type": "frame", "data": b64, "frame_id": frame_count,
                    "timing": timing_str
                }
                try:
                    frame_queue.put_nowait(frame_data)
                except:
                    pass
                
                # Log
                elapsed = time.perf_counter() - loop_start
                if elapsed >= 1.0:
                    print(f"[GEN] {loop_count/elapsed:.1f} FPS | {timing_str} | total={total_ms:.1f}ms")
                    loop_count = 0
                    loop_start = time.perf_counter()
                    
            except Exception as e:
                print(f"[GEN] Error: {e}")
                running = False
                break
    
    try:
        await ws.send_text(json.dumps({"type": "status", "message": "initializing..."}))
        
        gen_thread = threading.Thread(target=generator, daemon=True)
        gen_thread.start()
        
        recv_task = asyncio.create_task(receiver())
        send_task = asyncio.create_task(sender())
        
        await asyncio.wait([recv_task, send_task], return_when=asyncio.FIRST_COMPLETED)
        running = False
        
    except Exception as e:
        print(f"[WS] Error: {e}")
    
    running = False
    print("[WS] Disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, ws_ping_interval=300, ws_ping_timeout=300)
