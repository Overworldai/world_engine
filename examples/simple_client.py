from typing import AsyncIterable, AsyncIterator

import asyncio
import contextlib
import cv2
import sys
import torch

from world_engine import WorldEngine, CtrlInput


async def render(frames: AsyncIterable[torch.Tensor], win_name="Hello (Over)World (ESC to exit)") -> None:
    """Render stream of RGB tensor images."""
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    async for t in frames:
        cv2.imshow(win_name, t.cpu().numpy())
        await asyncio.sleep(0)
    cv2.destroyAllWindows()


async def frame_stream(engine: WorldEngine, ctrls: AsyncIterable[CtrlInput]) -> AsyncIterator[torch.Tensor]:
    """Generate frame by calling Engine for each ctrl."""
    yield await asyncio.to_thread(engine.gen_frame)
    async for ctrl in ctrls:
        yield await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)


async def ctrl_stream(delay: int = 1) -> AsyncIterator[CtrlInput]:
    """Accumulate key presses asyncronously. Yield CtrlInput once next() is called."""
    q: asyncio.Queue[int] = asyncio.Queue()

    async def producer() -> None:
        while True:
            k = cv2.waitKey(delay)
            if k != -1:
                await q.put(k)
            await asyncio.sleep(0)

    prod_task = asyncio.create_task(producer())
    while True:
        buttons: set[int] = set()
        # Drain everything currently in the queue into this batch
        with contextlib.suppress(asyncio.QueueEmpty):
            while True:
                k = q.get_nowait()
                if k == 27:
                    # End if ESC pressed
                    prod_task.cancel()
                    return
                buttons.add(k)

        yield CtrlInput(button=buttons)


async def main() -> None:
    uri = sys.argv[1] if len(sys.argv) > 1 else "OpenWorldLabs/CoD-Img-Base"
    engine = WorldEngine(uri, device="cuda")
    ctrls = ctrl_stream()
    frames = frame_stream(engine, ctrls)
    await render(frames)


if __name__ == "__main__":
    asyncio.run(main())
