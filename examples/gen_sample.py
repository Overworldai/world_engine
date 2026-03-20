"""
Additional Dependencies: opencv-python imageio[pyav]
Run: `python3 examples/gen_sample.py Overworld/Waypoint-1.5-1B`
Run: `python3 examples/gen_sample.py <WP1.5 model URI>`
"""
import cv2
import imageio.v3 as iio
import random
import sys
import urllib.request
import numpy as np
import torch

from world_engine import WorldEngine, CtrlInput


# Create inference engine
engine = WorldEngine(sys.argv[1], device="cuda")


# Set seed frame
url = random.choice([
    "https://gist.github.com/user-attachments/assets/d81c6d26-a838-4afe-9d13-fd67677043c3",
    "https://gist.github.com/user-attachments/assets/b6d18c38-098e-43b0-8e61-66a16e5d8946",
    "https://gist.github.com/user-attachments/assets/0734a8c1-3eb4-4ffe-8c37-5665c45ab559",
    "https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed",
    "https://gist.github.com/user-attachments/assets/68c943a4-008a-4c25-948c-c81ab4c47d21",
])
frame = cv2.imdecode(np.frombuffer(urllib.request.urlopen(url).read(), np.uint8), cv2.IMREAD_COLOR)
engine.append_frame(torch.from_numpy(np.repeat(frame[None], 4, axis=0)))


# Define sequence of controller inputs applied
controller_sequence = [
    # move mouse, jump, do nothing, trigger, do nothing, trigger+jump, do nothing
    CtrlInput(mouse=[0.2, 0.2]), CtrlInput(button={32}), CtrlInput(), CtrlInput(), CtrlInput(),
    CtrlInput(button={1}), CtrlInput(), CtrlInput(), CtrlInput(button={1, 32}),
    CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(), CtrlInput(),
] * 4
controller_sequence += [CtrlInput()] * 8
controller_sequence += (
    [CtrlInput(button={32})] * 10 +  # forward
    [CtrlInput(button={65})] * 10 +  # left
    [CtrlInput(button={68})] * 10 +  # right
    [CtrlInput(button={83})] * 10   # backwards
)
controller_sequence += [CtrlInput()] * 10


# Generate frames conditioned on controller inputs
with iio.imopen("out.mp4", "w", plugin="pyav") as out:
    four_frames = engine.gen_frame().cpu().numpy()  # int8 [4, H, W, 3]
    out.write(four_frames, fps=60, codec="libx264")
    for ctrl in controller_sequence:
        four_frames = engine.gen_frame(ctrl=ctrl).cpu().numpy()
        out.write(four_frames)
