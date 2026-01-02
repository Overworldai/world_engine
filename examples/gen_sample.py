import cv2
from world_engine import WorldEngine


def gen_vid():
    overrides = {"ae_uri": "OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan"}
    engine = WorldEngine("OpenWorldLabs/Medium-0-NoCaption-SF-Shift8", device="cuda", model_config_overrides=overrides)
    writer: cv2.VideoWriter | None = None
    for _ in range(240):
        frame = engine.gen_frame().cpu().numpy()[:, :, ::-1]  # RGB -> BGR for OpenCV
        writer = writer or cv2.VideoWriter(
            "out.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            60,
            (frame.shape[1], frame.shape[0])
        )
        writer.write(frame)

    if writer is not None: writer.release()


if __name__ == "__main__":
    gen_vid()
