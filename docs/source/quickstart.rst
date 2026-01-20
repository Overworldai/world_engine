Quickstart
==========

Get started with World Engine in minutes.

Installation
------------

First, set up a virtual environment (recommended):

.. code-block:: bash

   python3 -m venv .env
   source .env/bin/activate

Install World Engine with PyTorch CUDA support:

.. code-block:: bash

   pip install \
     --index-url https://download.pytorch.org/whl/test/cu128 \
     --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
     --extra-index-url https://pypi.org/simple \
     --upgrade --ignore-installed \
     "world_engine @ git+https://github.com/Overworldai/world_engine.git"

Set your HuggingFace token to access Waypoint models:

.. code-block:: bash

   export HF_TOKEN=<your access token>

Get your token at https://huggingface.co/settings/tokens

Basic Usage
-----------

Understanding CtrlInput
^^^^^^^^^^^^^^^^^^^^^^^

The ``CtrlInput`` class represents controller input for a single frame. It encapsulates button presses and mouse/pointer movement:

.. code-block:: python

   from world_engine import CtrlInput

   # Create control input with button presses and mouse movement
   ctrl = CtrlInput(
       button={48, 42},      # Set of pressed button IDs
       mouse=[0.4, 0.3]      # (x, y) mouse velocity vector
   )

   # Empty control input (no buttons, no mouse movement)
   ctrl = CtrlInput()

**Button Keycodes:** Button IDs are defined by `Owl-Control <https://github.com/Overworldai/owl-control/blob/main/src/system/keycode.rs>`_

**Mouse Input:** The ``mouse`` parameter is a raw velocity vector ``(x, y)`` representing pointer movement.

Using WorldEngine
^^^^^^^^^^^^^^^^^

The ``WorldEngine`` class is the main interface for generating frames. It maintains state across frames, including the visual history and current prompt.

Loading a Model
"""""""""""""""

Create an engine instance by specifying a Waypoint model from HuggingFace:

.. code-block:: python

   from world_engine import WorldEngine

   engine = WorldEngine(
       "OpenWorldLabs/CoDCtl-Causal-SelfForcing-UniformSigma",
       device="cuda"
   )

Setting the Prompt
""""""""""""""""""

Use ``set_prompt()`` to specify a text prompt that conditions frame generation. The prompt persists until you call ``set_prompt()`` again:

.. code-block:: python

   engine.set_prompt("A fun game")
   # All subsequent frames will be conditioned on this prompt

Appending Frames
""""""""""""""""

Use ``append_frame()`` to manually add a frame to the sequence without generating. This is useful for:

- Setting an initial frame
- Forcing a specific image at any point in the sequence
- Seeding generation with reference images

**Single Frame:**

.. code-block:: python

   import torch

   # Append a single frame (H, W, 3) uint8 tensor
   img = torch.randint(0, 256, (512, 512, 3), dtype=torch.uint8)
   returned_img = engine.append_frame(img)

**Multiple Frames:**

.. code-block:: python

   # Append multiple frames at once
   frames = torch.randint(0, 256, (4, 512, 512, 3), dtype=torch.uint8)
   engine.append_frame(frames)

**Frames with Controls:**

.. code-block:: python

   # Append frames with corresponding control inputs
   frames = torch.randint(0, 256, (2, 512, 512, 3), dtype=torch.uint8)
   ctrls = [
       CtrlInput(button={48}),
       CtrlInput(button={42})
   ]
   engine.append_frame(frames, ctrls=ctrls)

.. note::
   ``append_frame()`` returns the appended image(s) on the same device as ``engine.device``

Generating Frames
"""""""""""""""""

Use ``gen_frame()`` to generate a new frame conditioned on:

- The current prompt (set via ``set_prompt()``)
- Visual history (all previously generated/appended frames)
- Controller input (optional)

.. code-block:: python

   from world_engine import CtrlInput

   # Generate with control input
   ctrl = CtrlInput(button={48, 42}, mouse=[0.4, 0.3])
   img = engine.gen_frame(ctrl=ctrl)

   # Generate without control input
   img = engine.gen_frame()

   # Generate multiple frames in sequence
   for ctrl_input in [
       CtrlInput(button={48, 42}, mouse=[0.4, 0.3]),
       CtrlInput(mouse=[0.1, 0.2]),
       CtrlInput(button={95, 32, 105}),
   ]:
       img = engine.gen_frame(ctrl=ctrl_input)

Resetting State
"""""""""""""""

Use ``reset()`` to clear the visual history and start fresh. The prompt is preserved:

.. code-block:: python

   engine.reset()
   # Visual history cleared, but prompt remains set

Complete Example: Interactive Pygame Application
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's a complete example using OpenCV to create an interactive world model experience:

.. code-block:: python

   from typing import AsyncIterable, AsyncIterator
   import asyncio
   import contextlib
   import cv2
   import sys
   import torch
   from world_engine import WorldEngine, CtrlInput

   async def render(frames: AsyncIterable[torch.Tensor],
                    win_name="World Engine Demo (ESC to exit)") -> None:
       """Render stream of RGB tensor images."""
       cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
       async for t in frames:
           cv2.imshow(win_name, t.cpu().numpy())
           await asyncio.sleep(0)
       cv2.destroyAllWindows()

   async def frame_stream(engine: WorldEngine,
                         ctrls: AsyncIterable[CtrlInput]) -> AsyncIterator[torch.Tensor]:
       """Generate frame by calling Engine for each ctrl."""
       yield await asyncio.to_thread(engine.gen_frame)
       async for ctrl in ctrls:
           yield await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)

   async def ctrl_stream(delay: int = 1) -> AsyncIterator[CtrlInput]:
       """Accumulate key presses asynchronously."""
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
                   if k == 27:  # ESC key
                       prod_task.cancel()
                       return
                   buttons.add(k)

           yield CtrlInput(button=buttons)

   async def main() -> None:
       uri = sys.argv[1] if len(sys.argv) > 1 else \
             "OpenWorldLabs/CoDCtl-Causal-Flux-SelfForcing"

       # Create engine
       engine = WorldEngine(uri, device="cuda")

       # Set initial prompt
       engine.set_prompt("A fun platformer game")

       # Stream controls and frames
       ctrls = ctrl_stream()
       frames = frame_stream(engine, ctrls)
       await render(frames)

   if __name__ == "__main__":
       asyncio.run(main())

This example creates a real-time interactive experience where:

1. OpenCV captures keyboard input asynchronously
2. Inputs are batched into ``CtrlInput`` objects
3. Each frame is generated based on input and visual history
4. Frames are rendered to an OpenCV window
5. Press ESC to exit

Next Steps
----------

- Explore the :doc:`api_reference` for detailed API documentation
- Check out :doc:`quantization` for model compression options
- Review :doc:`modeling_specifics` if you want to understand internals
