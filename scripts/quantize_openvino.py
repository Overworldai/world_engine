#!/usr/bin/env python3
"""
Quantize OpenVINO IR models to INT4 for faster inference on Intel GPU.

Usage:
    python scripts/quantize_openvino.py --ir-dir exported_models/
    python scripts/quantize_openvino.py --ir-dir exported_models/ --mode int4_asym --group-size 64
"""
import argparse
from pathlib import Path
import time

import openvino as ov
from nncf import compress_weights, CompressWeightsMode


MODE_MAP = {
    "int4_sym": CompressWeightsMode.INT4_SYM,
    "int4_asym": CompressWeightsMode.INT4_ASYM,
    "int8_sym": CompressWeightsMode.INT8_SYM,
    "int8_asym": CompressWeightsMode.INT8_ASYM,
}


def quantize_model(input_path: Path, output_path: Path, mode: str, group_size: int, ratio: float):
    core = ov.Core()

    print(f"Loading model: {input_path}")
    model = core.read_model(str(input_path))
    print(f"  Inputs: {len(model.inputs)}, Outputs: {len(model.outputs)}")

    compress_mode = MODE_MAP[mode]
    print(f"Compressing weights: mode={mode}, group_size={group_size}, ratio={ratio}")
    t0 = time.perf_counter()

    compressed = compress_weights(
        model,
        mode=compress_mode,
        group_size=group_size,
        ratio=ratio,
    )

    elapsed = time.perf_counter() - t0
    print(f"  Compression done in {elapsed:.1f}s")

    print(f"Saving to: {output_path}")
    ov.save_model(compressed, str(output_path))

    # Report size reduction
    orig_size = sum(f.stat().st_size for f in input_path.parent.glob(input_path.stem + ".*"))
    new_size = sum(f.stat().st_size for f in output_path.parent.glob(output_path.stem + ".*"))
    print(f"  Original: {orig_size / 1e9:.2f} GB")
    print(f"  Quantized: {new_size / 1e9:.2f} GB")
    print(f"  Reduction: {orig_size / new_size:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Quantize OpenVINO IR to INT4/INT8")
    parser.add_argument("--ir-dir", type=str, default="exported_models",
                        help="Directory containing IR models")
    parser.add_argument("--mode", type=str, default="int4_sym",
                        choices=list(MODE_MAP.keys()),
                        help="Quantization mode (default: int4_sym)")
    parser.add_argument("--group-size", type=int, default=128,
                        help="Group size for weight quantization (default: 128)")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Ratio of INT4 layers vs INT8 backup (default: 0.8)")
    parser.add_argument("--skip-vae", action="store_true",
                        help="Skip VAE quantization")
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)

    # Quantize frozen transformer
    frozen_path = ir_dir / "transformer_frozen.xml"
    if frozen_path.exists():
        out_name = f"transformer_frozen_{args.mode}_g{args.group_size}.xml"
        quantize_model(frozen_path, ir_dir / out_name, args.mode, args.group_size, args.ratio)
    else:
        print(f"No frozen transformer found at {frozen_path}")

    # Optionally quantize VAE
    if not args.skip_vae:
        for vae_name in ["vae_decoder", "vae_encoder"]:
            vae_path = ir_dir / f"{vae_name}.xml"
            if vae_path.exists():
                out_name = f"{vae_name}_{args.mode}.xml"
                quantize_model(vae_path, ir_dir / out_name, args.mode, args.group_size, 1.0)

    print("\nQuantization complete!")


if __name__ == "__main__":
    main()
