"""
WebRTC server for WorldEngine frame streaming using aiortc.

Architecture:
- Signaling: WebSocket endpoint for SDP/ICE exchange only (lightweight)
- Peer: Server acts as WebRTC peer, generating video frames from WorldEngine
- Video track: Server pushes frames via VP8-encoded video stream
- Data channel: Client sends inputs at 60Hz (independent of video)

Flow:
1. Client connects to /signaling WebSocket
2. Server sends status updates (init, loading, ready)
3. Client sends SDP offer
4. Server creates peer connection, adds tracks, sends SDP answer
5. ICE completes, DTLS handshake, media flows
6. Frame generation starts only when peer connection is established

Usage:
    python examples/webrtc_server.py

Client connects via WebSocket to ws://localhost:8080/signaling for signaling,
then establishes WebRTC peer connection for media/data.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import fractions
import json
import logging
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("webrtc_server")

# Reduce noise from libraries
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

import torch
import torch.nn.functional as F
import torchvision
from av import VideoFrame

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
)
from aiortc.mediastreams import MediaStreamTrack, MediaStreamError

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

# ============================================================================
# Configuration
# ============================================================================

MODEL_URI = "OverWorld/Waypoint-Medium-Beta-2026-01-11"
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

# ICE configuration with STUN for NAT traversal
RTC_CONFIG = RTCConfiguration(iceServers=[
    RTCIceServer(urls=["stun:stun.l.google.com:19302"])
])

# ============================================================================
# Engine Setup
# ============================================================================

engine = None
seed_frame = None

# Single-thread executor for all engine operations
# PyTorch dynamo doesn't support being called from multiple threads
engine_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")


async def run_in_engine_thread(func, *args):
    """Run a function in the dedicated engine thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(engine_executor, func, *args)


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
# Status Codes (sent to client via signaling)
# ============================================================================

class Status:
    INIT = "init"          # Server initializing
    LOADING = "loading"    # Engine warming up
    READY = "ready"        # Ready to receive offer
    STREAMING = "streaming"  # Actively streaming frames
    RESET = "reset"        # Session reset


# ============================================================================
# WorldEngine Video Track
# ============================================================================

class WorldEngineVideoTrack(MediaStreamTrack):
    """
    Video track that generates frames from WorldEngine.

    Lifecycle:
    1. warm() - Reset engine and prepare for streaming (keeps engine "hot")
    2. start_streaming() - Begin background frame generation (called when connected)
    3. recv() - Called by aiortc to get frames for encoding
    4. stop() - Stop frame generation

    Decouples input reception from frame generation:
    - Inputs arrive via DataChannel at ~60Hz
    - Frames are generated in a background loop
    - Video track pulls latest frame when aiortc requests

    Uses latest-only input policy: if multiple inputs arrive during
    frame generation, only the newest is used.
    """

    kind = "video"

    def __init__(self, client_id: str):
        super().__init__()
        self.client_id = client_id

        # Latest control input (updated by DataChannel handler)
        self._latest_ctrl = None
        self._ctrl_lock = asyncio.Lock()

        # Frame timing for video track
        self._pts = 0
        self._time_base = fractions.Fraction(1, 30)  # 30 FPS nominal

        # Session state
        self.frame_count = 0
        self.max_frames = N_FRAMES - 2
        self._warmed = False
        self._streaming = False
        self._paused = False

        # Frame queue (size 2: one generating, one ready)
        self._frame_queue: asyncio.Queue[VideoFrame] = asyncio.Queue(maxsize=2)
        self._generator_task: Optional[asyncio.Task] = None

        # Stats
        self.last_gen_time = 0.0
        self._stats_callback = None  # Set by session to send stats via DataChannel
        self._last_client_ts = 0  # Last client timestamp for RTT calculation

    async def warm(self):
        """
        Warm up the engine - reset and prepare for streaming.
        Runs a warmup frame to trigger JIT compilation before actual streaming.
        """
        if self._warmed:
            return

        from world_engine import CtrlInput

        logger.info(f"[{self.client_id}] Warming engine...")
        await run_in_engine_thread(engine.reset)
        await run_in_engine_thread(engine.append_frame, seed_frame)

        # Generate a warmup frame to trigger JIT compilation
        logger.info(f"[{self.client_id}] Running warmup frame (JIT compile)...")
        await run_in_engine_thread(engine.gen_frame, CtrlInput())

        self._warmed = True
        logger.info(f"[{self.client_id}] Engine warm")

    def start_streaming(self):
        """
        Start background frame generation.
        Called when peer connection is established.
        """
        if self._streaming:
            return

        if not self._warmed:
            logger.warning(f"[{self.client_id}] start_streaming() called before warm()")

        self._streaming = True
        self._generator_task = asyncio.create_task(self._generate_frames())
        logger.info(f"[{self.client_id}] Started streaming")

    def stop(self):
        """Stop frame generation (called by aiortc, must be sync)."""
        logger.info(f"[{self.client_id}] Stopping video track...")
        self._streaming = False

        if self._generator_task:
            self._generator_task.cancel()
            self._generator_task = None

        logger.info(f"[{self.client_id}] Video track stopped")

    async def stop_async(self):
        """Async version for manual cleanup."""
        logger.info(f"[{self.client_id}] Stopping video track (async)...")
        self._streaming = False

        if self._generator_task:
            self._generator_task.cancel()
            try:
                await self._generator_task
            except asyncio.CancelledError:
                pass
            self._generator_task = None

        logger.info(f"[{self.client_id}] Video track stopped")

    def set_paused(self, paused: bool):
        """Pause or resume frame generation."""
        self._paused = paused

    async def _generate_frames(self):
        """Background loop generating frames continuously."""
        from world_engine import CtrlInput

        logger.info(f"[{self.client_id}] Frame generator started")

        while self._streaming:
            try:
                # Skip frame generation when paused
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                # Get latest control input (atomic read + clear)
                async with self._ctrl_lock:
                    ctrl = self._latest_ctrl
                    self._latest_ctrl = None

                # Default to no-op if no input
                if ctrl is None:
                    ctrl = CtrlInput()

                # Check frame limit, auto-reset if needed
                if self.frame_count >= self.max_frames:
                    logger.info(f"[{self.client_id}] Auto-reset at frame limit")
                    await run_in_engine_thread(engine.reset)
                    await run_in_engine_thread(engine.append_frame, seed_frame)
                    self.frame_count = 0

                # Generate frame (40-80ms on GPU)
                t0 = time.perf_counter()
                frame_tensor = await run_in_engine_thread(engine.gen_frame, ctrl)
                self.last_gen_time = (time.perf_counter() - t0) * 1000

                self.frame_count += 1

                # Convert torch tensor to av.VideoFrame
                # frame_tensor is (H, W, 3) uint8 on GPU
                frame_np = frame_tensor.cpu().numpy()
                video_frame = VideoFrame.from_ndarray(frame_np, format="rgb24")
                video_frame.pts = self._pts
                video_frame.time_base = self._time_base
                self._pts += 1

                # Put in queue (drop oldest if full - keep fresh)
                try:
                    self._frame_queue.put_nowait(video_frame)
                except asyncio.QueueFull:
                    try:
                        self._frame_queue.get_nowait()  # Discard oldest
                    except asyncio.QueueEmpty:
                        pass
                    self._frame_queue.put_nowait(video_frame)

                # Send stats to client via data channel
                if self._stats_callback:
                    stats_msg = {
                        "type": "stats",
                        "gentime": round(self.last_gen_time, 1),
                        "frame": self.frame_count,
                        "client_ts": self._last_client_ts  # Echo client timestamp for RTT calc
                    }
                    await self._stats_callback(stats_msg)
                    if self.frame_count % 60 == 0:
                        logger.info(f"[{self.client_id}] Stats sent: {stats_msg}")

                # Log first frame and then periodically
                if self.frame_count == 1 or self.frame_count % 60 == 0:
                    logger.info(f"[{self.client_id}] Frame {self.frame_count} (gen={self.last_gen_time:.1f}ms)")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.client_id}] Frame generation error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info(f"[{self.client_id}] Frame generator stopped")

    async def recv(self):
        """Called by aiortc to get next frame for encoding/sending."""
        if not self._streaming:
            logger.debug(f"[{self.client_id}] recv() called but not streaming")
            raise MediaStreamError("Track not streaming")

        # Wait for frame from generator (no timeout - handles pause and slow JIT)
        # When paused, generator sleeps but doesn't produce frames
        # We poll the queue with short timeouts to stay responsive to stop()
        while self._streaming:
            try:
                frame = await asyncio.wait_for(self._frame_queue.get(), timeout=0.5)
                return frame
            except asyncio.TimeoutError:
                # No frame yet, keep waiting if still streaming
                continue

        # Streaming stopped while waiting
        raise MediaStreamError("Track stopped")

    async def update_control(self, buttons: set, mouse_dx: float, mouse_dy: float, client_ts: float = 0):
        """
        Update latest control input (called by DataChannel handler).
        Uses latest-only policy: overwrites any pending input.
        """
        from world_engine import CtrlInput

        ctrl = CtrlInput(button=buttons, mouse=(mouse_dx, mouse_dy))
        async with self._ctrl_lock:
            self._latest_ctrl = ctrl
            self._last_client_ts = client_ts

    async def reset_session(self):
        """Reset the engine and frame counter."""
        logger.info(f"[{self.client_id}] Reset requested")
        await run_in_engine_thread(engine.reset)
        await run_in_engine_thread(engine.append_frame, seed_frame)
        self.frame_count = 0


# ============================================================================
# Session Management
# ============================================================================

@dataclass
class Session:
    """Tracks state for a client session."""
    client_id: str
    pc: Optional[RTCPeerConnection] = None
    video_track: Optional[WorldEngineVideoTrack] = None
    data_channel: Optional[object] = None

active_sessions: dict[str, Session] = {}


async def cleanup_session(client_id: str):
    """Clean up session resources."""
    if client_id not in active_sessions:
        return

    session = active_sessions.pop(client_id)
    if session.video_track:
        await session.video_track.stop_async()
    if session.pc:
        await session.pc.close()
    logger.info(f"[{client_id}] Session cleaned up")


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_engine()
    yield

app = FastAPI(title="WorldEngine WebRTC Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "healthy",
        "model": MODEL_URI,
        "quant": QUANT,
        "engine_loaded": engine is not None,
        "active_sessions": len(active_sessions),
    })


# ============================================================================
# Signaling Endpoint
# ============================================================================

@app.websocket("/signaling")
async def signaling_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for WebRTC signaling.

    This is a lightweight signaling relay that also manages peer connection
    lifecycle since the server is itself a WebRTC peer.

    Protocol:
        Client -> Server:
            {"type": "offer", "sdp": "..."}
            {"type": "ice", "candidate": {...}}

        Server -> Client:
            {"type": "answer", "sdp": "..."}
            {"type": "ice", "candidate": {...}}
            {"type": "status", "code": "init|loading|ready|connected|streaming|reset"}
            {"type": "error", "message": "..."}
    """
    client_host = websocket.client.host if websocket.client else "unknown"
    client_id = f"{client_host}_{id(websocket)}"
    logger.info(f"[{client_id}] Signaling connection")

    await websocket.accept()

    async def send_status(code: str):
        await websocket.send_text(json.dumps({"type": "status", "code": code}))

    async def send_json(data: dict):
        await websocket.send_text(json.dumps(data))

    # Create session with video track (but no peer connection yet)
    video_track = WorldEngineVideoTrack(client_id)
    session = Session(client_id=client_id, video_track=video_track)
    active_sessions[client_id] = session

    try:
        # Phase 1: Warm up engine while client prepares
        await send_status(Status.INIT)
        await send_status(Status.LOADING)
        await video_track.warm()
        await send_status(Status.READY)

        # Phase 2: Wait for signaling messages
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(raw)
            except asyncio.TimeoutError:
                # Keepalive
                try:
                    await send_json({"type": "ping"})
                except Exception:
                    break
                continue
            except WebSocketDisconnect:
                logger.info(f"[{client_id}] Signaling disconnected")
                break

            msg_type = msg.get("type")

            if msg_type == "offer":
                # Client sent offer - now create peer connection
                logger.info(f"[{client_id}] Received offer, creating peer connection")

                pc = RTCPeerConnection(configuration=RTC_CONFIG)
                session.pc = pc

                # Add video track to peer connection
                pc.addTrack(video_track)
                logger.info(f"[{client_id}] Added video track to peer connection")

                @pc.on("connectionstatechange")
                async def on_conn_state():
                    state = pc.connectionState
                    logger.info(f"[{client_id}] Connection state: {state}")
                    if state in ("failed", "closed", "disconnected"):
                        await cleanup_session(client_id)

                @pc.on("iceconnectionstatechange")
                async def on_ice_state():
                    logger.debug(f"[{client_id}] ICE connection state: {pc.iceConnectionState}")

                # Handle data channel from client
                @pc.on("datachannel")
                def on_datachannel(channel):
                    logger.info(f"[{client_id}] DataChannel received: {channel.label}, state: {channel.readyState}")
                    session.data_channel = channel

                    # Set up stats callback to send via this channel
                    def setup_stats_callback():
                        logger.info(f"[{client_id}] Setting up stats callback")
                        async def send_stats(stats):
                            if channel.readyState == "open":
                                try:
                                    channel.send(json.dumps(stats))
                                except Exception as e:
                                    logger.warning(f"[{client_id}] Failed to send stats: {e}")
                        video_track._stats_callback = send_stats
                        logger.info(f"[{client_id}] Stats callback set")

                    # Channel might already be open when we receive it
                    if channel.readyState == "open":
                        setup_stats_callback()

                    @channel.on("open")
                    def on_dc_open():
                        logger.info(f"[{client_id}] DataChannel open event")
                        setup_stats_callback()

                    @channel.on("close")
                    def on_dc_close():
                        logger.info(f"[{client_id}] DataChannel closed")
                        video_track._stats_callback = None

                    @channel.on("message")
                    def on_dc_message(message):
                        asyncio.create_task(_handle_dc_message(message, video_track, client_id, channel))

                # Process the offer
                offer = RTCSessionDescription(sdp=msg["sdp"], type="offer")
                await pc.setRemoteDescription(offer)

                # Create answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                # Start streaming now - aiortc calls recv() immediately after DTLS completes,
                # which can race with connectionstatechange. By starting here, we ensure
                # frames are ready before RTP sender needs them.
                video_track.start_streaming()

                # Wait for ICE gathering to complete (aiortc doesn't support trickle ICE)
                while pc.iceGatheringState != "complete":
                    await asyncio.sleep(0.01)

                # Send answer with embedded ICE candidates
                final_sdp = pc.localDescription.sdp
                logger.debug(f"[{client_id}] ICE candidates in SDP: {sum(1 for l in final_sdp.split(chr(10)) if 'candidate:' in l)}")

                await send_json({
                    "type": "answer",
                    "sdp": final_sdp
                })
                logger.info(f"[{client_id}] Sent answer")

                # Send streaming status immediately - aiortc's connection state tracking
                # is unreliable, but if we get here the connection will work
                await send_status(Status.STREAMING)

            elif msg_type == "ice":
                # Handle trickle ICE candidates from client
                if not session.pc:
                    continue

                candidate_data = msg.get("candidate")
                if candidate_data and candidate_data.get("candidate"):
                    try:
                        from aiortc.sdp import candidate_from_sdp
                        candidate_str = candidate_data.get("candidate")
                        if candidate_str.startswith("candidate:"):
                            candidate_str = candidate_str[10:]
                        candidate = candidate_from_sdp(candidate_str)
                        candidate.sdpMid = candidate_data.get("sdpMid")
                        candidate.sdpMLineIndex = candidate_data.get("sdpMLineIndex")
                        await session.pc.addIceCandidate(candidate)
                        logger.info(f"[{client_id}] Added ICE candidate")
                    except Exception as e:
                        logger.warning(f"[{client_id}] Failed to add ICE candidate: {e}")

            elif msg_type == "pong":
                # Client responded to keepalive
                pass

    except Exception as e:
        logger.error(f"[{client_id}] Error: {e}", exc_info=True)
        try:
            await send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        await cleanup_session(client_id)
        logger.info(f"[{client_id}] Disconnected")


async def _handle_dc_message(message, video_track, client_id, data_channel):
    """Handle control messages from data channel."""
    try:
        msg = json.loads(message)
        msg_type = msg.get("type", "control")

        if msg_type == "control":
            buttons = {
                BUTTON_CODES[b.upper()]
                for b in msg.get("buttons", [])
                if b.upper() in BUTTON_CODES
            }
            mouse_dx = float(msg.get("mouse_dx", 0))
            mouse_dy = float(msg.get("mouse_dy", 0))
            client_ts = float(msg.get("ts", 0))  # Client timestamp for RTT
            if video_track._last_client_ts == 0 and client_ts > 0:
                logger.info(f"[{client_id}] First control with ts: {client_ts}")
            await video_track.update_control(buttons, mouse_dx, mouse_dy, client_ts)

        elif msg_type == "pause":
            paused = msg.get("paused", True)
            video_track.set_paused(paused)
            logger.info(f"[{client_id}] Pause: {paused}")

        elif msg_type == "reset":
            await video_track.reset_session()
            # Send reset confirmation via data channel
            if data_channel and data_channel.readyState == "open":
                try:
                    data_channel.send(json.dumps({"type": "reset", "success": True}))
                    logger.info(f"[{client_id}] Reset complete, confirmation sent")
                except Exception as e:
                    logger.warning(f"[{client_id}] Failed to send reset confirmation: {e}")

    except Exception as e:
        logger.error(f"[{client_id}] DataChannel message error: {e}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WorldEngine WebRTC Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )
