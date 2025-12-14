from typing import AsyncIterable, AsyncIterator

import asyncio
import contextlib
import cv2
import os
import random
import sys
import torch
import torch.nn.functional as F
from torchvision.io import read_video

from world_engine import WorldEngine, CtrlInput


# Mouse sensitivity multiplier for velocity
MOUSE_SENSITIVITY = 2.5
MODEL_URI = "OpenWorldLabs/CoD-V2-L20-MLP5-Patch5-5-Self-Forcing-Shift1-2.1Step"
MAX_FRAMES = 4096


def load_random_video_frame(video_dir: str = "../../../video_clips", target_size: tuple = (360, 640)) -> torch.Tensor | None:
    """Load a random frame from a random video in the specified directory. Returns None if directory doesn't exist."""
    if not os.path.exists(video_dir):
        print(f"Video directory '{video_dir}' not found - skipping seed frame")
        return None

    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    if not video_files:
        print(f"No video files found in {video_dir} - skipping seed frame")
        return None

    chosen_video = os.path.join(video_dir, random.choice(video_files))
    print(f"Loading seed frame from: {os.path.basename(chosen_video)}")

    # Read first second of video
    video, _, _ = read_video(chosen_video, start_pts=0, end_pts=1, pts_unit="sec")

    # video: (T, H, W, C) with C=4 → keep RGB, go to (T, C, H, W)
    video = video[..., :3].permute(0, 3, 1, 2)  # (T, 3, H, W)

    # Pick random frame from the first second
    frame_idx = random.randint(0, len(video) - 1)
    print(f"Selected frame {frame_idx}/{len(video)-1}")

    # Resize to target size (default 360x640 for latent compatibility)
    frame = F.interpolate(video[frame_idx:frame_idx+1], size=target_size, mode="bilinear", align_corners=False)
    frame = frame[0].to(device='cuda', dtype=torch.uint8)  # (3, H, W)
    frame = frame.permute(1, 2, 0)  # (H, W, 3) - channels last for append_frame

    return frame


async def render(frames: AsyncIterable[torch.Tensor], win_name="World Engine", mouse_state=None) -> None:
    """Render stream of RGB tensor images."""
    # WINDOW_GUI_NORMAL disables the right-click dropdown menu (requires Qt backend)
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(win_name, 640, 360)  # Set initial window size

    # Set up mouse callback if mouse_state is provided
    if mouse_state is not None:
        def mouse_callback(event, x, y, flags, param):
            # Only update position, don't clear buttons on every movement
            mouse_state['pos'] = (x, y)

            # Update button state based on current flags
            mouse_state['buttons'].clear()
            if flags & cv2.EVENT_FLAG_LBUTTON:
                mouse_state['buttons'].add(0x01)  # LMB
            if flags & cv2.EVENT_FLAG_RBUTTON:
                mouse_state['buttons'].add(0x02)  # RMB
            if flags & cv2.EVENT_FLAG_MBUTTON:
                mouse_state['buttons'].add(0x04)  # MMB

        cv2.setMouseCallback(win_name, mouse_callback)

    frame_num = 0
    async for t in frames:
        frame_num += 1
        print(f"Rendering frame {frame_num}", end='\r', flush=True)
        cv2.setWindowTitle(win_name, f"Frame {frame_num} - Y=new seed, U=restart, ESC=exit")
        # Convert RGB to BGR for OpenCV display
        frame_bgr = t.cpu().numpy()[..., ::-1]
        cv2.imshow(win_name, frame_bgr)
        await asyncio.sleep(0)
    cv2.destroyAllWindows()


async def frame_stream(engine: WorldEngine, ctrls: AsyncIterable[CtrlInput], seed_frame: torch.Tensor | None, max_frames: int = MAX_FRAMES-2) -> AsyncIterator[torch.Tensor]:
    """Generate frame by calling Engine for each ctrl. Resets context after max_frames to avoid positional encoding crash."""
    frame_count = 0
    current_seed = seed_frame

    # Initialize with seed frame if available
    if current_seed is not None:
        print("Appending seed frame to engine...")
        await asyncio.to_thread(engine.append_frame, current_seed)

    print("Generating first frame...")
    yield await asyncio.to_thread(engine.gen_frame)
    frame_count += 1
    print("First frame generated!")

    async for ctrl in ctrls:
        # Check for reset commands
        if hasattr(ctrl, 'reset_command'):
            if ctrl.reset_command == 'new_seed':
                print("\n[Y pressed - Loading new random seed frame]")
                current_seed = await asyncio.to_thread(load_random_video_frame)
                await asyncio.to_thread(engine.reset)
                if current_seed is not None:
                    await asyncio.to_thread(engine.append_frame, current_seed)
                frame_count = 0
                yield await asyncio.to_thread(engine.gen_frame)
                frame_count += 1
                continue
            elif ctrl.reset_command == 'restart_seed':
                print("\n[U pressed - Restarting with current seed frame]")
                await asyncio.to_thread(engine.reset)
                if current_seed is not None:
                    await asyncio.to_thread(engine.append_frame, current_seed)
                frame_count = 0
                yield await asyncio.to_thread(engine.gen_frame)
                frame_count += 1
                continue

        if frame_count >= max_frames:
            print(f"\n[Resetting context after {frame_count} frames to avoid positional encoding crash]")
            await asyncio.to_thread(engine.reset)
            if current_seed is not None:
                await asyncio.to_thread(engine.append_frame, current_seed)
            frame_count = 0

        yield await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)
        frame_count += 1


async def ctrl_stream(delay: int = 1, mouse_state=None) -> AsyncIterator[CtrlInput]:
    """Accumulate key presses asyncronously. Yield CtrlInput once next() is called.
    Special keys: Y (121) = new random seed, U (117) = restart current seed, ESC = exit"""
    q: asyncio.Queue[int] = asyncio.Queue()
    last_mouse_pos = [0, 0]

    async def producer() -> None:
        while True:
            k = cv2.waitKey(delay)
            if k != -1:
                print(f"\n[Key pressed: {k} ({chr(k) if 32 <= k < 127 else 'special'})]")
                await q.put(k)
            await asyncio.sleep(0)

    prod_task = asyncio.create_task(producer())
    while True:
        buttons: set[int] = set()
        reset_command = None

        # Drain everything currently in the queue into this batch
        with contextlib.suppress(asyncio.QueueEmpty):
            while True:
                k = q.get_nowait()
                if k == 27:  # ESC
                    print("\n[ESC pressed - Exiting]")
                    prod_task.cancel()
                    return
                elif k == 121:  # 'y' key (lowercase)
                    print("\n[Y key detected - Will load new seed]")
                    reset_command = 'new_seed'
                elif k == 117:  # 'u' key (lowercase)
                    print("\n[U key detected - Will restart with current seed]")
                    reset_command = 'restart_seed'
                else:
                    # Convert lowercase letters to uppercase for model compatibility
                    if 97 <= k <= 122:  # lowercase a-z
                        k_converted = k - 32
                        #print(f"\n[Button input: {k} ('{chr(k)}') -> sending {k_converted} ('{chr(k_converted)}') to model]")
                        buttons.add(k_converted)
                    # Map platform-specific keycodes to Windows VK codes (used in training)
                    elif k == 225:  # OpenCV Linux LShift
                        k_converted = 0xA0  # Windows VK_LSHIFT (160)
                        #print(f"\n[Button input: {k} (LShift) -> sending {k_converted} (0xA0/VK_LSHIFT) to model]")
                        buttons.add(k_converted)
                    elif k == 32:  # Space (already correct)
                        #print(f"\n[Button input: {k} (Space/0x20)]")
                        buttons.add(k)
                    else:
                        #print(f"\n[Button input: {k}]")
                        buttons.add(k)

        # Add mouse buttons if mouse_state is provided
        mouse_velocity = (0.0, 0.0)
        if mouse_state is not None:
            # Add mouse button inputs
            buttons.update(mouse_state['buttons'])
            #if mouse_state['buttons']:
            #    print(f"\n[Mouse buttons: {mouse_state['buttons']}]")

            # Calculate mouse velocity (position delta)
            curr_pos = mouse_state['pos']
            delta_x = float(curr_pos[0] - last_mouse_pos[0])
            delta_y = float(curr_pos[1] - last_mouse_pos[1])

            # Clamp to prevent huge spikes (max 100 pixels per frame)
            max_delta = 100.0
            delta_x = max(-max_delta, min(max_delta, delta_x))
            delta_y = max(-max_delta, min(max_delta, delta_y))

            last_mouse_pos[0], last_mouse_pos[1] = curr_pos

            # Normalize to window size (640x360) and apply sensitivity multiplier
            if delta_x != 0.0 or delta_y != 0.0:
                normalized_vel = (
                    (delta_x / 640.0) * MOUSE_SENSITIVITY,
                    (delta_y / 360.0) * MOUSE_SENSITIVITY
                )
                #print(f"\n[Mouse velocity: {normalized_vel}]")
                mouse_velocity = normalized_vel

        ctrl = CtrlInput(button=buttons, mouse=mouse_velocity)
        if reset_command:
            ctrl.reset_command = reset_command
        yield ctrl


async def main() -> None:
    uri = sys.argv[1] if len(sys.argv) > 1 else MODEL_URI
    video_dir = sys.argv[2] if len(sys.argv) > 2 else "../../../video_clips"

    print("Loading initial seed frame...")
    seed_frame = await asyncio.to_thread(load_random_video_frame, video_dir)

    if seed_frame is None:
        print("No seed frame loaded - starting from random noise")
    else:
        print("Seed frame loaded successfully")

    print("Initializing WorldEngine...")
    engine = WorldEngine(uri, device="cuda", model_config_overrides={"n_frames" : MAX_FRAMES})

    print("Starting interactive session...")
    if seed_frame is not None:
        print("Controls: Y = new random seed, U = restart current seed, ESC = exit")
    else:
        print("Controls: ESC = exit (seed frame controls disabled - no video directory)")
    print("Mouse: LMB/RMB/MMB supported, movement tracked as velocity")

    # Shared mouse state between render and ctrl_stream
    mouse_state = {'pos': (0, 0), 'buttons': set()}

    ctrls = ctrl_stream(mouse_state=mouse_state)
    frames = frame_stream(engine, ctrls, seed_frame)
    await render(frames, mouse_state=mouse_state)


if __name__ == "__main__":
    asyncio.run(main())
