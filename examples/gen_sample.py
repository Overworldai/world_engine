# uv run --dev examples/gen_sample.py Overworld/Waypoint-1.5-1B

import cv2
import imageio.v3 as iio
import random
import sys
import urllib.request
import numpy as np
import torch

from world_engine import WorldEngine, CtrlInput


# Create inference engine
model_config_overrides = {"ae_uri": "Overworld-Models/taehv1_5",
                          "patch": [2, 2],
                          "temporal_compression": 4,
                          "inference_fps": 60,
                          "taehv_ae": True}

model_config_overrides.update({})
engine = WorldEngine(sys.argv[1], 
                     model_config_overrides=model_config_overrides,
                     quant="w8a8_gemlite",
                     device="cuda")

total_linear_params = sum(mod.weight.numel() for _, mod in engine.model.named_modules() if isinstance(mod, torch.nn.Linear))
print(f"Total linear layer parameters: {total_linear_params:,}")

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
url = random.choice([
    "https://gist.github.com/user-attachments/assets/d81c6d26-a838-4afe-9d13-fd67677043c3",
    "https://gist.github.com/user-attachments/assets/b6d18c38-098e-43b0-8e61-66a16e5d8946",
    "https://gist.github.com/user-attachments/assets/0734a8c1-3eb4-4ffe-8c37-5665c45ab559",
    "https://gist.github.com/user-attachments/assets/f9c20d4d-7565-452d-8b02-42a85ea175ed",
    "https://gist.github.com/user-attachments/assets/68c943a4-008a-4c25-948c-c81ab4c47d21",
])
seed_frame = cv2.imdecode(np.frombuffer(urllib.request.urlopen(url).read(), np.uint8), cv2.IMREAD_COLOR)
seed_frame_x4 = torch.from_numpy(np.repeat(seed_frame[None], 4, axis=0))


# Generate frames conditioned on controller inputs
with iio.imopen("out.mp4", "w", plugin="pyav") as out:
    engine.append_frame(seed_frame_x4)
    out.write(seed_frame_x4, fps=60, codec="libx264")
    for ctrl in controller_sequence:
        four_frames = engine.gen_frame(ctrl=ctrl).cpu().numpy()
        out.write(four_frames)
