import cv2
from world_engine import WorldEngine


def gen_vid():
    # engine = WorldEngine("OpenWorldLabs/CoDCtl-Causal-Flux-SelfForcing", device="cuda")
    engine = WorldEngine("/mnt/data/laplace/models/combat_sfpp/step_640", model_config_overrides={"n_frames": 24000}, device="cuda")
    writer = None
    for i in range(2400):
        print(f"Generating frame {i}")
        frame = engine.gen_frame()
        print(frame.shape, frame.dtype, frame.device)
        frame = frame.cpu().numpy()[:, :, ::-1]  # RGB -> BGR for OpenCV
        writer = writer or cv2.VideoWriter(
            "out.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            60,
            (frame.shape[1], frame.shape[0])
        )
        writer.write(frame)

    writer.release()


if __name__ == "__main__":
    gen_vid()
