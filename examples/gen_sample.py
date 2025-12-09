import cv2
from world_engine import WorldEngine


def gen_vid():
    engine = WorldEngine("OpenWorldLabs/CoDCtl-Causal-Flux-SelfForcing", device="cuda")
    writer = None
    for _ in range(240):
        frame = engine.gen_frame().cpu().numpy()[:, :, ::-1]  # RGB -> BGR for OpenCV
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
