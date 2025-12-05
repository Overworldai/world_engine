import torch
from torch.profiler import profile, ProfilerActivity

from world_engine import WorldEngine


def do_profile(n_frames=64, row_limit=20):
    engine = WorldEngine("OpenWorldLabs/CoDCtl-Causal-Flux-SelfForcing", device="cuda")
    # warmup
    for _ in range(4):
        engine.gen_frame()

    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for step in range(n_frames):
            engine.gen_frame(return_img=False)
        torch.cuda.synchronize()

    print("\n===== Top ops by CUDA time =====")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=row_limit,
    ))

    print("\n===== Top ops by CPU time =====")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=row_limit,
    ))


if __name__ == "__main__":
    do_profile()
