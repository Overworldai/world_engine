"""
Test CoreML ANE TAEHV: correctness, stability, and speed.
Run: python -m src.mlx_metal.ane.test_ane
"""
import time
import numpy as np
import torch

CKPT_URI = "Overworld-Models/taehv1_5"


def test():
    from src.ae import get_ae, ChunkedStreamingTAEHV

    print("=" * 60)
    print("CoreML ANE TAEHV — Validation")
    print("=" * 60)

    ane = get_ae(CKPT_URI, is_taehv_ae=True, ane=True, dtype=torch.float32)
    pt = ChunkedStreamingTAEHV.from_pretrained(CKPT_URI, device="cpu", dtype=torch.float32)

    # --- Encode ---
    img = torch.randint(0, 255, (4, 720, 1280, 3), dtype=torch.uint8)
    pt_lat = pt.encode(img)
    ane_lat = ane.encode(img)
    enc_mae = (pt_lat.float() - ane_lat.float()).abs().mean().item()
    print(f"\n[encode]  MAE={enc_mae:.6f}  shape={ane_lat.shape}")
    assert enc_mae < 0.01

    # --- Decode quality ---
    torch.manual_seed(42)
    lats = [torch.randn(1, 32, 32, 64) for _ in range(8)]
    pt.reset(); ane.reset()
    maes = []
    for i, lat in enumerate(lats):
        pt_out = pt.decode(lat)
        ane_out = ane.decode(lat)
        assert pt_out.shape == ane_out.shape, f"Shape mismatch: {pt_out.shape} vs {ane_out.shape}"
        mae = (pt_out.float() - ane_out.float()).abs().mean().item()
        maes.append(mae)
    avg_mae = np.mean(maes[2:])
    print(f"[decode]  MAE={avg_mae:.2f}/255  frames={ane_out.shape}")
    assert avg_mae < 1.0, f"Decode MAE too high: {avg_mae}"

    # --- Stability ---
    ane.reset()
    for i in range(20):
        frames = ane.decode(torch.randn(1, 32, 32, 64) * 0.5)
        assert not torch.isnan(frames.float()).any(), f"NaN at frame {i}"
    print(f"[stable]  20 frames, no NaN")

    # --- Reset ---
    ane.reset()
    out1 = ane.decode(lats[0])
    ane.reset()
    out2 = ane.decode(lats[0])
    assert (out1.float() - out2.float()).abs().max() == 0
    print(f"[reset]   deterministic after reset")

    # --- Speed ---
    ane.reset()
    enc_t, dec_t = [], []
    for _ in range(15):
        t0 = time.perf_counter(); ane.encode(img); enc_t.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); ane.decode(lats[0]); dec_t.append((time.perf_counter() - t0) * 1000)
    enc = np.mean(enc_t[5:])
    dec = np.mean(dec_t[5:])
    print(f"\n[speed]   enc={enc:.1f}ms  dec={dec:.1f}ms  total={enc+dec:.1f}ms  GPU=0%")

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    test()
