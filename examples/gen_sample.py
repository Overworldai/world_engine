# uv run --dev examples/gen_sample.py Overworld/Waypoint-1.5-1B

import cv2
import imageio.v3 as iio
import random
import sys
import urllib.request
import json
import numpy as np
import torch

from world_engine import WorldEngine, CtrlInput


# Create inference engine
engine = WorldEngine(sys.argv[1], device="cuda")


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


# Set seed frame
with urllib.request.urlopen("https://api.github.com/repos/Overworldai/Biome/contents/seeds?ref=14343a6") as res:
    urls = [item["download_url"] for item in json.load(res) if item["type"] == "file"]
url = random.choice(urls)

seed_frame = cv2.imdecode(np.frombuffer(urllib.request.urlopen(url).read(), np.uint8), cv2.IMREAD_COLOR)
seed_frame_x4 = torch.from_numpy(np.repeat(seed_frame[None], 4, axis=0))


# Generate frames conditioned on controller inputs
with iio.imopen("out.mp4", "w", plugin="pyav") as out:
    engine.append_frame(seed_frame_x4)
    out.write(seed_frame_x4, fps=60, codec="libx264")
    for ctrl in controller_sequence:
        four_frames = engine.gen_frame(ctrl=ctrl).cpu().numpy()
        out.write(four_frames)
