#!/usr/bin/env python3
"""
Export world_engine models to OpenVINO IR format.

Usage:
    python scripts/export_openvino.py --model-uri Overworld/Waypoint-1-Small --output-dir exported_models/

Exports three separate IR models:
    1. VAE Encoder  (vae_encoder.xml/bin)
    2. VAE Decoder  (vae_decoder.xml/bin)
    3. Transformer  (transformer.xml/bin) — stateless, with explicit KV I/O

Updated for wp-1.5: uses get_ae() factory, adds frame_idx parameter.
"""
import argparse
import sys
import os
from pathlib import Path

import torch
import openvino as ov

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from world_engine.model.world_model import WorldModel
from world_engine.ae import get_ae, InferenceAE
from world_engine.portable_model import PortableWorldModel
from world_engine.stateless_kv import StatelessKVManager


def export_vae_encoder(ae, output_dir: Path, dtype=torch.float32):
    """Export VAE encoder to OpenVINO IR."""
    print("Exporting VAE encoder...")
    encoder = ae.ae_model.encoder.to("cpu").to(dtype).eval()

    in_ch = encoder.conv_in.proj.in_channels
    example_input = torch.randn(1, in_ch, 360, 640, dtype=dtype)

    ov_model = ov.convert_model(encoder, example_input=example_input)
    ov.save_model(ov_model, str(output_dir / "vae_encoder.xml"), compress_to_fp16=True)
    print(f"  Saved to {output_dir / 'vae_encoder.xml'}")


def export_vae_decoder(ae, output_dir: Path, dtype=torch.float32):
    """Export VAE decoder to OpenVINO IR."""
    print("Exporting VAE decoder...")
    decoder = ae.ae_model.decoder.to("cpu").to(dtype).eval()

    latent_ch = decoder.conv_in.proj.in_channels
    encoder = ae.ae_model.encoder.to("cpu").to(dtype).eval()
    in_ch = encoder.conv_in.proj.in_channels
    with torch.no_grad():
        dummy_in = torch.randn(1, in_ch, 360, 640, dtype=dtype)
        latent = encoder(dummy_in)
    latent_shape = latent.shape
    print(f"  Latent shape: {latent_shape}")

    example_input = torch.randn(latent_shape, dtype=dtype)
    ov_model = ov.convert_model(decoder, example_input=example_input)
    ov.save_model(ov_model, str(output_dir / "vae_decoder.xml"), compress_to_fp16=True)
    print(f"  Saved to {output_dir / 'vae_decoder.xml'}")

    return latent_shape


def export_transformer(model_uri: str, config, output_dir: Path, latent_shape, dtype=torch.float32):
    """Export stateless transformer to OpenVINO IR."""
    print("Exporting transformer...")

    # Load original model and convert to portable
    original = WorldModel.from_pretrained(model_uri, cfg=config, device="cpu", dtype=dtype)
    original = original.eval()

    portable = PortableWorldModel.from_original(original).to(dtype).eval()
    del original  # free memory

    # Create KV manager to get buffer shapes
    kv_mgr = StatelessKVManager(config, batch_size=1, dtype=torch.float16, device=torch.device("cpu"))

    B, N = 1, 1
    C = config.channels
    pH, pW = tuple(config.patch)
    H = config.height * pH
    W = config.width * pW

    # Build example inputs
    x = torch.randn(B, N, C, H, W, dtype=dtype)
    sigma = torch.tensor([[0.5]], dtype=dtype)
    frame_timestamp = torch.tensor([[0]], dtype=torch.long)
    frame_idx = torch.tensor([[0]], dtype=torch.long)
    is_frozen = torch.tensor([True], dtype=torch.bool)

    # Controller inputs
    n_buttons = config.n_buttons
    mouse = torch.zeros(B, N, 2, dtype=dtype)
    button = torch.zeros(B, N, n_buttons, dtype=dtype)
    scroll = torch.zeros(B, N, 1, dtype=dtype)

    # Prompt conditioning
    prompt_emb = None
    prompt_pad_mask = None
    if config.prompt_conditioning is not None:
        prompt_dim = config.prompt_embedding_dim
        prompt_emb = torch.zeros(B, 512, prompt_dim, dtype=dtype)
        prompt_pad_mask = torch.ones(B, 512, dtype=torch.bool)  # all padding

    kv_bufs, written_bufs = kv_mgr.get_state()

    print(f"  Model config: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, tokens_per_frame={config.tokens_per_frame}")
    print(f"  Input shape: x={list(x.shape)}")
    print(f"  KV layers: {len(kv_bufs)}, shapes: {[list(kv.shape) for kv in kv_bufs[:3]]}...")

    # Test forward pass first
    print("  Running test forward pass...")
    with torch.no_grad():
        v_pred, kv_out, written_out = portable(
            x, sigma, frame_timestamp,
            kv_bufs, written_bufs, is_frozen,
            frame_idx=frame_idx,
            prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
            mouse=mouse, button=button, scroll=scroll,
        )
    print(f"  Test output shape: {list(v_pred.shape)}")

    # Export to OpenVINO via FlatTransformer wrapper
    print("  Converting to OpenVINO IR (this may take a while)...")

    class FlatTransformer(torch.nn.Module):
        def __init__(self, model, n_layers):
            super().__init__()
            self.model = model
            self.n_layers = n_layers

        def forward(self, x, sigma, frame_timestamp, frame_idx, is_frozen,
                    mouse, button, scroll, prompt_emb, prompt_pad_mask,
                    *kv_and_written):
            kv_bufs = list(kv_and_written[:self.n_layers])
            written_bufs = list(kv_and_written[self.n_layers:])
            v_pred, kv_out, written_out = self.model(
                x, sigma, frame_timestamp,
                kv_bufs, written_bufs, is_frozen,
                frame_idx=frame_idx,
                prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
                mouse=mouse, button=button, scroll=scroll,
            )
            return (v_pred, *kv_out, *written_out)

    flat_model = FlatTransformer(portable, config.n_layers).eval()

    example_inputs = (
        x, sigma, frame_timestamp, frame_idx, is_frozen,
        mouse, button, scroll,
        prompt_emb if prompt_emb is not None else torch.zeros(B, 1, 1, dtype=dtype),
        prompt_pad_mask if prompt_pad_mask is not None else torch.zeros(B, 1, dtype=torch.bool),
        *kv_bufs, *written_bufs,
    )

    try:
        ov_model = ov.convert_model(flat_model, example_input=example_inputs)
        ov.save_model(ov_model, str(output_dir / "transformer.xml"), compress_to_fp16=True)
        print(f"  Saved to {output_dir / 'transformer.xml'}")
    except Exception as e:
        print(f"  OpenVINO conversion failed: {e}")
        print("  Trying torch.jit.trace fallback...")
        try:
            with torch.no_grad():
                traced = torch.jit.trace(flat_model, example_inputs)
            ov_model = ov.convert_model(traced, example_input=example_inputs)
            ov.save_model(ov_model, str(output_dir / "transformer.xml"), compress_to_fp16=True)
            print(f"  Saved to {output_dir / 'transformer.xml'} (via jit.trace)")
        except Exception as e2:
            print(f"  JIT trace also failed: {e2}")
            print("  Saving portable PyTorch model instead for manual conversion.")
            torch.save(portable.state_dict(), output_dir / "transformer_portable.pt")
            print(f"  Saved PyTorch state dict to {output_dir / 'transformer_portable.pt'}")

    # Save config for runtime
    from omegaconf import OmegaConf
    OmegaConf.save(config, str(output_dir / "config.yaml"))
    print(f"  Config saved to {output_dir / 'config.yaml'}")


def export_transformer_frozen(model_uri: str, config, output_dir: Path, latent_shape, dtype=torch.float32):
    """Export frozen-only transformer (denoise pass) for GPU.

    Bakes is_frozen=True so torch.where on large KV tensors is eliminated.
    Returns only v_pred (no KV outputs). This avoids the Intel GPU OpenCL
    compiler bug with Select on large tensors.
    """
    print("Exporting frozen transformer (GPU-friendly)...")

    original = WorldModel.from_pretrained(model_uri, cfg=config, device="cpu", dtype=dtype)
    original = original.eval()

    portable = PortableWorldModel.from_original(original).to(dtype).eval()
    del original

    kv_mgr = StatelessKVManager(config, batch_size=1, dtype=torch.float16, device=torch.device("cpu"))

    B, N = 1, 1
    C = config.channels
    pH, pW = tuple(config.patch)
    H = config.height * pH
    W = config.width * pW

    x = torch.randn(B, N, C, H, W, dtype=dtype)
    sigma = torch.tensor([[0.5]], dtype=dtype)
    frame_timestamp = torch.tensor([[0]], dtype=torch.long)
    frame_idx = torch.tensor([[0]], dtype=torch.long)

    n_buttons = config.n_buttons
    mouse = torch.zeros(B, N, 2, dtype=dtype)
    button = torch.zeros(B, N, n_buttons, dtype=dtype)
    scroll = torch.zeros(B, N, 1, dtype=dtype)

    prompt_emb = None
    prompt_pad_mask = None
    if config.prompt_conditioning is not None:
        prompt_dim = config.prompt_embedding_dim
        prompt_emb = torch.zeros(B, 512, prompt_dim, dtype=dtype)
        prompt_pad_mask = torch.ones(B, 512, dtype=torch.bool)

    kv_bufs, written_bufs = kv_mgr.get_state()

    print(f"  Model config: d_model={config.d_model}, n_layers={config.n_layers}")
    print(f"  KV layers: {len(kv_bufs)}, shapes: {[list(kv.shape) for kv in kv_bufs[:3]]}...")

    # Test forward pass
    print("  Running test forward pass (frozen)...")
    is_frozen = torch.tensor([True], dtype=torch.bool)
    with torch.no_grad():
        v_pred, _, _ = portable(
            x, sigma, frame_timestamp,
            kv_bufs, written_bufs, is_frozen,
            frame_idx=frame_idx,
            prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
            mouse=mouse, button=button, scroll=scroll,
        )
    print(f"  Test output shape: {list(v_pred.shape)}")

    # Frozen wrapper: bakes is_frozen=True, returns only v_pred
    class FrozenFlatTransformer(torch.nn.Module):
        def __init__(self, model, n_layers):
            super().__init__()
            self.model = model
            self.n_layers = n_layers

        def forward(self, x, sigma, frame_timestamp, frame_idx,
                    mouse, button, scroll, prompt_emb, prompt_pad_mask,
                    *kv_and_written):
            kv_bufs = list(kv_and_written[:self.n_layers])
            written_bufs = list(kv_and_written[self.n_layers:])
            is_frozen = torch.tensor([True], dtype=torch.bool, device=x.device)
            v_pred, _, _ = self.model(
                x, sigma, frame_timestamp,
                kv_bufs, written_bufs, is_frozen,
                frame_idx=frame_idx,
                prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
                mouse=mouse, button=button, scroll=scroll,
            )
            return v_pred

    flat_model = FrozenFlatTransformer(portable, config.n_layers).eval()

    example_inputs = (
        x, sigma, frame_timestamp, frame_idx,
        mouse, button, scroll,
        prompt_emb if prompt_emb is not None else torch.zeros(B, 1, 1, dtype=dtype),
        prompt_pad_mask if prompt_pad_mask is not None else torch.zeros(B, 1, dtype=torch.bool),
        *kv_bufs, *written_bufs,
    )

    print("  Converting to OpenVINO IR (this may take a while)...")
    try:
        ov_model = ov.convert_model(flat_model, example_input=example_inputs)
        ov.save_model(ov_model, str(output_dir / "transformer_frozen.xml"), compress_to_fp16=True)
        print(f"  Saved to {output_dir / 'transformer_frozen.xml'}")
    except Exception as e:
        print(f"  OpenVINO conversion failed: {e}")
        print("  Trying torch.jit.trace fallback...")
        try:
            with torch.no_grad():
                traced = torch.jit.trace(flat_model, example_inputs)
            ov_model = ov.convert_model(traced, example_input=example_inputs)
            ov.save_model(ov_model, str(output_dir / "transformer_frozen.xml"), compress_to_fp16=True)
            print(f"  Saved to {output_dir / 'transformer_frozen.xml'} (via jit.trace)")
        except Exception as e2:
            print(f"  JIT trace also failed: {e2}")

    from omegaconf import OmegaConf
    OmegaConf.save(config, str(output_dir / "config.yaml"))
    print(f"  Config saved to {output_dir / 'config.yaml'}")


def export_denoise_loop(model_uri: str, config, output_dir: Path, latent_shape, dtype=torch.float32):
    """Export the full denoise loop + cache pass as a single OV graph.

    This bakes the 4-step scheduler loop into the model, eliminating
    per-step Python overhead (~0.5s per call × 3 saved = ~1.5s).
    Returns denoised x0 and updated KV state.
    """
    print("Exporting denoise loop (full frame pipeline)...")

    original = WorldModel.from_pretrained(model_uri, cfg=config, device="cpu", dtype=dtype)
    portable = PortableWorldModel.from_original(original.eval()).to(dtype).eval()
    del original

    kv_mgr = StatelessKVManager(config, batch_size=1, dtype=torch.float16, device=torch.device("cpu"))

    B, N = 1, 1
    C = config.channels
    pH, pW = tuple(config.patch)
    H, W = config.height * pH, config.width * pW

    scheduler_sigmas = torch.tensor(config.scheduler_sigmas, dtype=dtype)

    x = torch.randn(B, N, C, H, W, dtype=dtype)
    frame_timestamp = torch.tensor([[0]], dtype=torch.long)
    frame_idx = torch.tensor([[0]], dtype=torch.long)
    mouse = torch.zeros(B, N, 2, dtype=dtype)
    button = torch.zeros(B, N, config.n_buttons, dtype=dtype)
    scroll = torch.zeros(B, N, 1, dtype=dtype)

    prompt_emb = None
    prompt_pad_mask = None
    if config.prompt_conditioning is not None:
        prompt_emb = torch.zeros(B, 512, config.prompt_embedding_dim, dtype=dtype)
        prompt_pad_mask = torch.ones(B, 512, dtype=torch.bool)

    kv_bufs, written_bufs = kv_mgr.get_state()
    n_layers = config.n_layers

    print(f"  Scheduler: {len(config.scheduler_sigmas)} sigmas ({len(config.scheduler_sigmas)-1} denoise steps + 1 cache)")

    class DenoiseLoopModel(torch.nn.Module):
        """Full frame pipeline: denoise loop (frozen) + cache pass (unfrozen).
        Single OV graph = single GPU submission per frame."""
        def __init__(self, model, sigmas, n_layers):
            super().__init__()
            self.model = model
            self.n_layers = n_layers
            self.register_buffer("sigmas", sigmas)
            self.register_buffer("dsigmas", sigmas.diff())

        def forward(self, x, frame_timestamp, frame_idx,
                    mouse, button, scroll, prompt_emb, prompt_pad_mask,
                    *kv_and_written):
            kv_bufs = list(kv_and_written[:self.n_layers])
            written_bufs = list(kv_and_written[self.n_layers:])
            is_frozen = torch.tensor([True], dtype=torch.bool, device=x.device)

            # Denoise loop (frozen KV)
            for i in range(self.dsigmas.shape[0]):
                sigma = self.sigmas[i].view(1, 1)
                v_pred, _, _ = self.model(
                    x, sigma, frame_timestamp,
                    kv_bufs, written_bufs, is_frozen,
                    frame_idx=frame_idx,
                    prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
                    mouse=mouse, button=button, scroll=scroll,
                )
                x = x + self.dsigmas[i] * v_pred

            # Cache pass (unfrozen KV)
            is_frozen = torch.tensor([False], dtype=torch.bool, device=x.device)
            sigma_zero = torch.zeros(1, 1, dtype=x.dtype, device=x.device)
            _, kv_out, written_out = self.model(
                x, sigma_zero, frame_timestamp,
                kv_bufs, written_bufs, is_frozen,
                frame_idx=frame_idx,
                prompt_emb=prompt_emb, prompt_pad_mask=prompt_pad_mask,
                mouse=mouse, button=button, scroll=scroll,
            )

            return (x, *kv_out, *written_out)

    loop_model = DenoiseLoopModel(portable, scheduler_sigmas, n_layers).eval()

    # Test
    print("  Running test...")
    with torch.no_grad():
        example_inputs = (
            x, frame_timestamp, frame_idx,
            mouse, button, scroll,
            prompt_emb if prompt_emb is not None else torch.zeros(B, 1, 1, dtype=dtype),
            prompt_pad_mask if prompt_pad_mask is not None else torch.zeros(B, 1, dtype=torch.bool),
            *kv_bufs, *written_bufs,
        )
        results = loop_model(*example_inputs)
    print(f"  Test output shape: {list(results[0].shape)}")

    print("  Converting to OpenVINO IR...")
    try:
        ov_model = ov.convert_model(loop_model, example_input=example_inputs)
        ov.save_model(ov_model, str(output_dir / "denoise_loop.xml"), compress_to_fp16=True)
        print(f"  Saved to {output_dir / 'denoise_loop.xml'}")
    except Exception as e:
        print(f"  OV conversion failed: {e}")
        print("  Trying torch.jit.trace...")
        try:
            with torch.no_grad():
                traced = torch.jit.trace(loop_model, example_inputs)
            ov_model = ov.convert_model(traced, example_input=example_inputs)
            ov.save_model(ov_model, str(output_dir / "denoise_loop.xml"), compress_to_fp16=True)
            print(f"  Saved to {output_dir / 'denoise_loop.xml'} (via jit.trace)")
        except Exception as e2:
            print(f"  JIT trace also failed: {e2}")

    from omegaconf import OmegaConf
    OmegaConf.save(config, str(output_dir / "config.yaml"))


def main():
    parser = argparse.ArgumentParser(description="Export world_engine to OpenVINO IR")
    parser.add_argument("--model-uri", type=str, required=True,
                        help="HuggingFace model URI or local path")
    parser.add_argument("--output-dir", type=str, default="exported_models",
                        help="Output directory for IR files")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16"],
                        help="Export precision (float32 recommended, compress_to_fp16 handles the rest)")
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE export")
    parser.add_argument("--skip-transformer", action="store_true", help="Skip transformer export")
    parser.add_argument("--frozen-only", action="store_true",
                        help="Export frozen-only transformer (GPU-friendly, no KV updates)")
    parser.add_argument("--denoise-loop", action="store_true",
                        help="Export full denoise loop + cache as single graph")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    print(f"Model URI: {args.model_uri}")
    print(f"Output dir: {output_dir}")

    # Load config
    config = WorldModel.load_config(args.model_uri)
    print(f"Loaded config: {config.d_model}d, {config.n_layers}L, {config.n_heads}H")

    latent_shape = None

    if not args.skip_vae:
        # Load VAE via factory
        ae = get_ae(
            config.ae_uri,
            is_taehv_ae=getattr(config, "taehv_ae", False),
            auto_aspect_ratio=getattr(config, "auto_aspect_ratio", True),
            device="cpu",
            dtype=dtype,
        )
        export_vae_encoder(ae, output_dir, dtype)
        latent_shape = export_vae_decoder(ae, output_dir, dtype)
        del ae

    if not args.skip_transformer:
        if latent_shape is None:
            ae = get_ae(
                config.ae_uri,
                is_taehv_ae=getattr(config, "taehv_ae", False),
                auto_aspect_ratio=getattr(config, "auto_aspect_ratio", True),
                device="cpu",
                dtype=dtype,
            )
            encoder = ae.ae_model.encoder.to("cpu").to(dtype).eval()
            in_ch = encoder.conv_in.proj.in_channels
            with torch.no_grad():
                latent_shape = encoder(torch.randn(1, in_ch, 360, 640, dtype=dtype)).shape
            del ae, encoder

        if args.denoise_loop:
            export_denoise_loop(args.model_uri, config, output_dir, latent_shape, dtype)
        elif args.frozen_only:
            export_transformer_frozen(args.model_uri, config, output_dir, latent_shape, dtype)
        else:
            export_transformer(args.model_uri, config, output_dir, latent_shape, dtype)

    print("\nExport complete!")


if __name__ == "__main__":
    main()
