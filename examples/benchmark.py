import pytest
import torch

from world_engine import WorldEngine


# TODO
# - benchmark encode img
# - benchmark encode prompt


def version_with_commit(pkg):
    import json
    from importlib.metadata import distribution
    dist = distribution(pkg.__name__.split('.')[0])
    version = dist.version
    try:
        data = dist.read_text("direct_url.json")
        commit = (data and json.loads(data).get("vcs_info", {}).get("commit_id"))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        commit = None
    return f"{version} @ {commit[:7]}" if commit else version


@pytest.fixture(scope="session", autouse=True)
def print_env_info():
    import platform
    import world_engine as world_engine_pkg
    print(
        "\n=== Environment ===\n"
        f"torch:        {torch.__version__}\n"
        f"torch.cuda:   {torch.version.cuda}\n"
        f"world_engine: {version_with_commit(world_engine_pkg)}\n\n"
        "=== Hardware ===\n"
        f"OS:   {platform.system()} {platform.release()} ({platform.machine()})\n"
        f"CPU:  {platform.processor() or 'unknown'}"
    )

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(
            f"GPU:  {props.name}\n"
            f"      capability {props.major}.{props.minor}\n"
            f"      total memory: {props.total_memory / 1e9:.1f} GB"
        )
    else:
        print("GPU:  none (CUDA not available)")


@pytest.fixture(scope="session")
def engine(model_uri="OpenWorldLabs/Codbuland"):
    engine = WorldEngine(model_uri, device="cuda")
    # global warmup
    for _ in range(3):
        engine.gen_frame()
    return engine


@pytest.fixture(scope="session")
def last_latent(engine):
    return engine.gen_frame(return_img=False).detach()


def test_img_decoder_only(benchmark, engine, last_latent):
    def run():
        with torch.amp.autocast("cuda", torch.bfloat16):
            engine.vae.decode(last_latent)
        torch.cuda.synchronize()

    benchmark(run)


@pytest.mark.parametrize("dit_only", [True])
@pytest.mark.parametrize("n_frames", [1, 4, 16, 64, 256])
def test_ar_rollout(benchmark, engine, dit_only, n_frames):
    def setup():
        engine.reset()
        engine.gen_frame(return_img=not dit_only)
        torch.cuda.synchronize()

    def target():
        for _ in range(n_frames):
            engine.gen_frame(return_img=not dit_only)
        torch.cuda.synchronize()

    benchmark.pedantic(target, setup=setup, rounds=20)
