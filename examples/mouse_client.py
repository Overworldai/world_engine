from typing import AsyncIterable, AsyncIterator

import asyncio
import contextlib
import os
import random
import sys
import torch
import torch.nn.functional as F
from torchvision.io import read_video
import time
import pygame
import numpy as np

from world_engine import WorldEngine, CtrlInput


# Mouse sensitivity multiplier for velocity
MOUSE_SENSITIVITY = 1.5
MODEL_URI = "OpenWorldLabs/CoD-V3-30K-SF"
ROOT_CLIPS_DIR = "../../video_clips"
MAX_FRAMES = 4096
START_TIME = None
GLOBAL_DTYPE = None


def load_random_video_frame(video_dir: str = ROOT_CLIPS_DIR, target_size: tuple = (360, 640)) -> torch.Tensor | None:
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


def set_mouse_lock(screen, locked: bool):
    pygame.event.set_grab(locked)
    pygame.mouse.set_visible(not locked)
    pygame.mouse.get_rel()  # flush any big jump
    if locked:
        pygame.mouse.set_pos(screen.get_width() // 2, screen.get_height() // 2)

async def render(frames: AsyncIterable[torch.Tensor], mouse_state=None) -> None:
    """Render stream of RGB tensor images using pygame."""
    pygame.init()

    # Set up display
    width, height = 1920, 1080
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    set_mouse_lock(screen, True)
    pygame.display.set_caption("World Engine - Initializing...")

    frame_num = 0
    last_time = None

    async for t in frames:
        frame_num += 1

        # Calculate FPS
        crnt_time = time.time()
        if last_time is not None:
            delay = crnt_time - last_time
            fps_num = round(1.0 / delay, 1)
        else:
            fps_num = 0.0
        print(f"FPS: {fps_num}")

        pygame.display.set_caption(f"Y=new seed, U=restart, ESC=exit")
        last_time = time.time()

        # Convert tensor to numpy array (already RGB, no flip needed!)
        frame_rgb = t.cpu().numpy()

        # Convert to pygame surface
        # pygame expects (width, height, 3) but numpy is (height, width, 3)
        surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))

        # Scale to window size
        scaled_surface = pygame.transform.scale(surface, screen.get_size())

        # Blit to screen
        screen.blit(scaled_surface, (0, 0))
        pygame.display.flip()
        #set_mouse_lock(screen, True)

        await asyncio.sleep(0)

    pygame.quit()


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
                current_seed = await asyncio.to_thread(load_random_video_frame)
                await asyncio.to_thread(engine.reset)
                if current_seed is not None:
                    await asyncio.to_thread(engine.append_frame, current_seed)
                frame_count = 0
                yield await asyncio.to_thread(engine.gen_frame)
                frame_count += 1
                continue
            elif ctrl.reset_command == 'restart_seed':
                await asyncio.to_thread(engine.reset)
                if current_seed is not None:
                    await asyncio.to_thread(engine.append_frame, current_seed)
                frame_count = 0
                yield await asyncio.to_thread(engine.gen_frame)
                frame_count += 1
                continue

        if frame_count >= max_frames:
            await asyncio.to_thread(engine.reset)
            if current_seed is not None:
                await asyncio.to_thread(engine.append_frame, current_seed)
            frame_count = 0

        yield await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)
        frame_count += 1

async def ctrl_stream(delay: int = 1, mouse_state=None) -> AsyncIterator[CtrlInput]:
    """Poll key states and mouse inputs each frame.
    Special keys: Y = new random seed, U = restart current seed, ESC = exit
    Arrow keys simulate mouse movement."""

    # Arrow key velocity for mouse simulation
    ARROW_KEY_VELOCITY = MOUSE_SENSITIVITY

    # Track previous key states for edge detection
    prev_y_pressed = False
    prev_u_pressed = False

    while True:
        # Process pygame events to keep window responsive and handle mouse
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if mouse_state is not None:
                    if event.button == 1:
                        mouse_state['buttons'].add(0x01)  # LMB
                    elif event.button == 2:
                        mouse_state['buttons'].add(0x04)  # MMB
                    elif event.button == 3:
                        mouse_state['buttons'].add(0x02)  # RMB
            elif event.type == pygame.MOUSEBUTTONUP:
                if mouse_state is not None:
                    if event.button == 1:
                        mouse_state['buttons'].discard(0x01)
                    elif event.button == 2:
                        mouse_state['buttons'].discard(0x04)
                    elif event.button == 3:
                        mouse_state['buttons'].discard(0x02)

        # Poll key states directly
        keys = pygame.key.get_pressed()

        # Check for exit
        if keys[pygame.K_ESCAPE]:
            return

        # Check for reset commands (edge-triggered)
        reset_command = None
        if keys[pygame.K_y] and not prev_y_pressed:
            reset_command = 'new_seed'
        elif keys[pygame.K_u] and not prev_u_pressed:
            reset_command = 'restart_seed'
        prev_y_pressed = keys[pygame.K_y]
        prev_u_pressed = keys[pygame.K_u]

        # Build button set from held keys
        buttons: set[int] = set()

        # Letters A-Z
        # Only WASD keys
        if keys[pygame.K_w]:
            buttons.add(ord('W'))
        if keys[pygame.K_a]:
            buttons.add(ord('A'))
        if keys[pygame.K_s]:
            buttons.add(ord('S'))
        if keys[pygame.K_d]:
            buttons.add(ord('D'))

        # Numbers 0-9
        #for i in range(10):
        #    if keys[pygame.K_0 + i]:
        #        buttons.add(48 + i)  # ASCII for '0'-'9'

        # Space
        if keys[pygame.K_SPACE]:
            buttons.add(32)

        # Shift keys
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            buttons.add(0xA0)  # VK_LSHIFT

        # Arrow keys control mouse movement
        mouse_velocity_x, mouse_velocity_y = pygame.mouse.get_rel()
        #mouse_velocity_x = 0.0
        #mouse_velocity_y = 0.0
        #if keys[pygame.K_UP]:
        #    mouse_velocity_y -= ARROW_KEY_VELOCITY
        #if keys[pygame.K_DOWN]:
        #    mouse_velocity_y += ARROW_KEY_VELOCITY
        #if keys[pygame.K_LEFT]:
        ##    mouse_velocity_x -= ARROW_KEY_VELOCITY
        #if keys[pygame.K_RIGHT]:
        #    mouse_velocity_x += ARROW_KEY_VELOCITY

        # Add mouse buttons
        if mouse_state is not None:
            buttons.update(mouse_state['buttons'])

        mouse_velocity = [mouse_velocity_x * MOUSE_SENSITIVITY, mouse_velocity_y * MOUSE_SENSITIVITY]
        mouse_velocity = torch.tensor(mouse_velocity, dtype=GLOBAL_DTYPE, device = 'cuda')

        ctrl = CtrlInput(button=buttons, mouse=mouse_velocity)
        if reset_command:
            ctrl.reset_command = reset_command

        yield ctrl
        await asyncio.sleep(0)


async def main() -> None:
    uri = sys.argv[1] if len(sys.argv) > 1 else MODEL_URI
    video_dir = sys.argv[2] if len(sys.argv) > 2 else ROOT_CLIPS_DIR

    print("Loading initial seed frame...")
    seed_frame = await asyncio.to_thread(load_random_video_frame, video_dir)

    if seed_frame is None:
        print("No seed frame loaded - starting from random noise")
    else:
        print("Seed frame loaded successfully")

    print("Initializing WorldEngine...")
    engine = WorldEngine(uri, device="cuda", model_config_overrides={"n_frames" : MAX_FRAMES}, apply_patches=True, quant = None)
    global GLOBAL_DTYPE
    GLOBAL_DTYPE = engine.dtype
    print("Starting interactive session...")
    if seed_frame is not None:
        print("Controls: Y = new random seed, U = restart current seed, ESC = exit")
    else:
        print("Controls: ESC = exit (seed frame controls disabled - no video directory)")
    print("Mouse clicks: LMB/RMB/MMB supported")
    print("Movement: Use arrow keys (simulates mouse movement)")

    # Shared mouse state (just buttons, no position tracking)
    mouse_state = {'buttons': set()}

    ctrls = ctrl_stream(mouse_state=mouse_state)
    frames = frame_stream(engine, ctrls, seed_frame)
    await render(frames, mouse_state=mouse_state)


if __name__ == "__main__":
    asyncio.run(main())
