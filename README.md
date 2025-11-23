<div align="center">

# 🌐 OverWorld Inference Engine

  Inference Engine for World Models
</div>


## Overview

Core library for model and inference plumbing:

- Simple API to load models and generate image frames from text, control inputs, and prior frames
- Encapsulates the frame-generation stack (DiT, autoencoder, text encoder, KV cache)
- Optimized backends for Nvidia, AMD, Apple Silicon, etc., on consumer and data center GPUs
- Loading base World Models and LoRA adapters

### Out of scope

Not a full client:

- No rendering/display of video or images
- No reading controller/keyboard/mouse input
- No FAL or other external integrations

Out-of-scope pieces can go in `examples/`, which is **not** part of the `world_engine.*` package.

## Quick Start

#### Setup
```
# Install
pip install "world_engine @ git+https://github.com/Wayfarer-Labs/world_engine.git"

# Specify HuggingFace Token (https://huggingface.co/settings/tokens)
export HF_TOKEN=<your access token>
```

#### Run
```
from world_engine import WorldEngine, CtrlInput

# Create inference engine
engine = WorldEngine("OpenWorldLabs/CoD-Img-Base", device="cuda")

# Specify a prompt
engine.set_prompt("A fun game")

# Optional: Force the next frame to be a specific image
img = pipeline.append_frame(uint8_img)

# Generate 3 video frames conditioned on controller inputs
for controller_input in [
		CtrlInput(button={48, 42}, mouse=[0.4, 0.3]),
		CtrlInput(mouse=[0.1, 0.2]),
		CtrlInput(button={95, 32, 105}),
]:
	img = engine.gen_frame(ctrl=controller_input)
```

## Docs

### WorldEngine

`WorldEngine` computes each new frame from past frames, the controls, and the current prompt, then appends it to the sequence so later frames stay aligned with what has already been generated.

## Usage
```
from world_engine import WorldEngine, CtrlInput
```

Load model to GPU
```
engine = WorldEngine("OpenWorldLabs/CoD-Img-Base", device="cuda")
```

Avoid expensive recompilation through caching
```
engine = WorldEngine(
    "OpenWorldLabs/CoD-Img-Base",
    device="cuda",
    compile_cache_path="~/.cache/world_engine/"
)
```

Specify a prompt which will be used until this function is called again
```
engine.set_prompt("A fun game")
```

Generate a image conditioned on current controller input (explicit) and history / prompt (implicit)
```
controller_input = CtrlInput(button={48, 42}, mouse=[0.4, 0.3])
img = engine.gen_frame(ctrl=controller_input)
```

Instead of generating, **set** the next frame as a specific image. Typically done as a step before generating.
```
# example: random noise image
uint8_img = torch.randint(0, 256, (512, 512, 3), dtype=torch.uint8)
img = pipeline.append_frame(uint8_img)  # returns passed image
```

Note: returned `img` is always on the same device as `engine.device`


### CtrlInput

```
@dataclass
class CtrlInput:
    button: Set[int] = field(default_factory=set)  # pressed button IDs
    mouse: Tuple[float, float] = (0.0, 0.0)  # (x, y) position
```

Keycodes are defined by [Owl-Control](https://github.com/Wayfarer-Labs/owl-control/blob/main/src/system/keycode.rs)


## Examples
- ["Hello (Over)World" client](./examples/simple_client.py)
- [Deploy to skypilot](./examples/skypilot.yaml)
- [Benchmark](./examples/benchmark.py)
