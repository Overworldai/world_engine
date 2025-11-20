# World Engine

Wayfarers Inference Library

# Usage

Install
```
pip install git+https://github.com/Wayfarer-Labs/world_engine
```

Specify [HuggingFace Token](https://huggingface.co/settings/tokens)
```
export HF_TOKEN=...
```

Run:
```
from world_engine import WorldEngine, CtrlInput

# Create inference engine
engine = WorldEngine("OpenWorldLabs/CoD-Img-Base", device=device)

# Specify a prompt
engine.set_prompt("A fun game")

# Generate 3 video frames using controller inputs
for controller_input in [
		CtrlInput(button={48, 42}, mouse=[0.4, 0.3]),
		CtrlInput(mouse=[0.1, 0.2]),
		CtrlInput(button={95, 32, 105}),
]:
	img = engine.gen_frame(ctrl=controller_input)
```

# Scope

### What this package is
A tool which does one thing and does it well

In scope:
- An abstraction away from the WM:
  - No knowledge of DiT, autoencoder, or text encoder needed
  - user works in the space of controller inputs, text prompts, and images
- Optimized implementations for Nvidia, AMD, Apple Silicon, etc
- Consumer and data center hardware
- Loading base World Models and LoRA adapters

### What this package isn't
This isn't a full featured client, it's a core library.

Out of scope:
- Rendering video
- Reading controllers / keyboard inputs
- FAL integration, other integrations

Anything out of scope can be added to `examples/`, which isn't importable or part of the `world_model` package.
