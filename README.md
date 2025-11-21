<div align="center">

# 🌐 OverWorld Inference Engine

  Inference Engine for World Models
</div>


## Usage

#### Setup
```
# Install
pip install "world_engine@git+https://github.com/Wayfarer-Labs/world_engine.git"

# Specify HuggingFace Token (https://huggingface.co/settings/tokens)
export HF_TOKEN=<your access token>
```

#### Run
```
from world_engine import WorldEngine, CtrlInput

# Create inference engine
engine = WorldEngine("OpenWorldLabs/CoD-Img-Base", device=device)

# Specify a prompt
engine.set_prompt("A fun game")

# Generate 3 video frames conditioned on controller inputs
for controller_input in [
		CtrlInput(button={48, 42}, mouse=[0.4, 0.3]),
		CtrlInput(mouse=[0.1, 0.2]),
		CtrlInput(button={95, 32, 105}),
]:
	img = engine.gen_frame(ctrl=controller_input)
```

## Scope

### What this package is
A library which abstracts away model implementation details inference optimizations
- Encapsulates handling of DiT, autoencoder, and text encoder
- Library consumer specifies model URI and handles controller inputs, text prompts, and images

In scope:
- Optimized implementations for Nvidia, AMD, Apple Silicon, etc
- Consumer and data center hardware
- Loading base World Models and LoRA adapters
- Performing all steps necessary to create a frame image conditioned on history, controls, and a text string

### What this package isn't
This isn't a fully featured client, it's a core library.

Out of scope:
- Rendering / displaying video and images
- Reading controllers or keyboard inputs
- FAL integration, other integrations

Anything out of scope can be added to `examples/`, which isn't part of the `world_engine.*` package.
