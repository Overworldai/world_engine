[![Documentation Status](https://readthedocs.com/projects/overworld-world-engine/badge/?version=latest)](https://overworld-world-engine.readthedocs-hosted.com/en/latest/index.html)

<div align="center">

# 🌐 Overworld Inference Engine

**Core library for World Model inference**

[📖 Documentation](https://overworld-world-engine.readthedocs-hosted.com/en/latest/index.html) ·
[⚡ Quickstart](#quick-start) ·
[✨ Showcase and Examples](#showcase-and-examples)

</div>


<p align="center">
  <img src="./assets/diagram.svg" alt="Diagram" width="420" />
</p>

## Overview

Core library for world model inference:

- Simple API to load models and generate image frames from text, control inputs, and prior frames
- Encapsulates the frame-generation stack (DiT, autoencoder, text encoder, KV cache)
- Optimized backends for Nvidia, AMD, Apple Silicon, etc., on consumer and data center GPUs
- Loading base World Models and LoRA adapters

### Out of scope

Not a full client:

- No rendering/display of video or images
- No reading controller/keyboard/mouse input
- No external integrations

Out-of-scope pieces can go in `examples/`, which is **not** part of the `world_engine.*` package.

## Quick Start

#### Setup

```sh
# Recommended: set up venv
python3 -m venv .env
source .env/bin/activate
```

```sh
# Install
pip install --upgrade --ignore-installed "world_engine @ git+https://github.com/Overworldai/world_engine.git"
```

```sh
# Specify HuggingFace Token (https://huggingface.co/settings/tokens)
export HF_TOKEN=<your access token>
```

#### Run

```py
from world_engine import WorldEngine, CtrlInput

# Create inference engine
engine = WorldEngine("Overworld/Waypoint-1.5-1B", device="cuda")

# Specify a prompt
engine.set_prompt("A fun game")

# Optional: Force the next frame to be a specific image
img = pipeline.append_frame(uint8_img)  # (H, W, 3)

# Generate 3 video frames conditioned on controller inputs
for controller_input in [
		CtrlInput(button={48, 42}, mouse=[0.4, 0.3]),
		CtrlInput(mouse=[0.1, 0.2]),
		CtrlInput(button={95, 32, 105}),
]:
	img = engine.gen_frame(ctrl=controller_input)  # see section below for img shape explanation
```

## Waypoint-1.5 Behavior

All interfaces and handling for Waypoint-1 (or 1.1) and Waypoint-1.5 remain the same **except** the following:

In Waypoint-1.5, the `img` passed to `append_frame(...)` and returned by `gen_frame(...)` is now a sequence of 4 frames. Waypoint-1.5 applies temporal compression and generates 4 frames for every controller input.

Whereas previously, `img` was a uint8 rgb array of shape `[Height, Width, 3]`, **in Waypoint-1.5 it is of shape `[4, Height, Width, 3]`**.

Additionally, Waypoint-1.5 expects 720p inputs / outputs, therefore `img` is `[4, 720, 1280, 3]`.

See [examples/gen_sample.py](./examples/gen_sample.py) for reference.

Space each 4-frame batch evenly across the time until the next batch is ready, while the next batch is generated in parallel to keep playback smooth and latency low. Example code to accomplish this is below.

```py
def render_batch(frames, batch_dt):
    step = batch_dt / len(frames)
    render(frames[0])
    for frame in frames[1:]:
        time.sleep(step)
        render(frame)


def generation_loop(engine, ctrl_input_generator):
    frames, batch_dt = None, 0.0
    for ctrl in ctrl_input_generator:
        start = time.perf_counter()
        next_frames = engine.gen_frame(ctrl=ctrl)
        if frames is not None:
            render_batch(frames, batch_dt)
        frames, batch_dt = next_frames.cpu(), time.perf_counter() - start
```

## Usage

```py
from world_engine import WorldEngine, CtrlInput
```

Load model to GPU

```py
engine = WorldEngine("Overworld/Waypoint-1.5-1B", device="cuda")
```

Specify a prompt which will be used until this function is called again

```py
engine.set_prompt("A fun game")
```

Generate an image conditioned on current controller input (explicit) and history / prompt (implicit)

```py
controller_input = CtrlInput(button={48, 42}, mouse=[0.4, 0.3])
img = engine.gen_frame(ctrl=controller_input)
```

Instead of generating, **set** the next frame as a specific image. Typically done as a step before generating.

```py
# example: random noise image
uint8_img = torch.randint(0, 256, (512, 512, 3), dtype=torch.uint8)
img = pipeline.append_frame(uint8_img)  # returns passed image
```

Note: returned `img` is always on the same device as `engine.device`

## Quantization 

Model can be quantized by passing `quant` argument to WorldEngine

```py
engine = WorldEngine("Overworld/Waypoint-1.5-1B", quant="intw8a8", device="cuda")
```
Supported inference quantization schemes are:

| Config | Description | Supported GPUs |
|--------|-------------|----------------|
| `intw8a8` | INT8 weights + INT8 dynamic per-token activations | NVIDIA (30xx, 40xx, Ampere+) |
| `fp8w8a8` | FP8 (e4m3) weights + FP8 per-tensor activations via `torch._scaled_mm` | NVIDIA Ada Lovelace / Hopper+ (RTX 40xx, H100) |
| `nvfp4` | NVFP4 weights + FP4 activations via FlashInfer/CUTLASS | NVIDIA Blackwell (B100, B200, RTX 5090) |


### WorldEngine

`WorldEngine` computes each new frame from past frames, the controls, and the current prompt, then appends it to the sequence so later frames stay aligned with what has already been generated.


### CtrlInput

```py
@dataclass
class CtrlInput:
    button: Set[int] = field(default_factory=set)  # pressed button IDs
    mouse: Tuple[float, float] = (0.0, 0.0)  # (dx, dy) position change
	scroll_wheel: int = 0  # down, stationary, or up -> (-1, 0, 1)
```

- `button` keycodes are defined by [Owl-Control](https://github.com/Overworldai/owl-control/blob/main/src/system/keycode.rs)
- `mouse` is the amount of change in mouse since the last frame
- `scroll_wheel` is the ternary scroll wheel movement identifier


## Showcase and Examples

### Tools and clients integrating `world_engine`

- [Overworld.stream](https://overworld.stream)
- [Overworld Biome](https://github.com/Overworldai/Biome/)
- [World Engine Zero (HF Space)](https://huggingface.co/spaces/Overworld/waypoint-1-small)
- [Daydream scope-overworld](https://github.com/daydreamlive/scope-overworld)
- [Hypnagogia](https://github.com/philpax/hypnagogia/)
- [LocalWorld](https://github.com/Overworldai/local_world)

### Examples and Reference Code

- ["Generate MP4 Sample Given Controller Inputs](./examples/gen_sample.py)
- [Run Performance Benchmarks (`pytest examples/benchmark.py`)](./examples/benchmark.py)
