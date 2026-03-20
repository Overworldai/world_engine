# MODEL_URI="Overworld/Waypoint-1.5-1B" uv run --dev pytest examples/benchmark.py

import os
import pytest
import torch
import random

from world_engine import WorldEngine, CtrlInput


MODEL_URI = os.environ.get("MODEL_URI", "Overworld/Waypoint-1-Small")


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


def get_warm_engine(model_uri, model_overrides=None):
    model_config_overrides = {}
    model_config_overrides.update(model_overrides or {})
    engine = WorldEngine(
        model_uri,
        model_config_overrides=model_config_overrides,
        quant="intw8a8",
        device="cuda",
        load_weights=False
    )

    # global warmup
    for _ in range(3):
        engine.gen_frame()
    return engine


@pytest.fixture(scope="session")
def engine():
    return get_warm_engine(MODEL_URI)


@pytest.fixture(scope="session")
def last_latent(engine):
    return engine.gen_frame(return_img=False).detach()


def test_img_decoder_only(benchmark, engine, last_latent):
    def run():
        with torch.amp.autocast("cuda", torch.bfloat16):
            engine.vae.decode(last_latent)
        torch.cuda.synchronize()

    benchmark(run)


MODEL_OVERRIDES = [None]


@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.parametrize("dit_only", [True])
@pytest.mark.parametrize("n_frames", [256])
@pytest.mark.parametrize(
    "model_overrides", MODEL_OVERRIDES,
    ids=lambda d: (",".join(f"{k}={v}" for k, v in d.items()) or "") if d else ""
)
def test_ar_rollout(benchmark, dit_only, n_frames, model_overrides, blocking):
    engine = get_warm_engine(MODEL_URI, model_overrides=model_overrides)

    try:
        total_params = sum(p.numel() for p in engine.model.parameters())
        active_params = int(engine.model.get_active_parameters())
        benchmark.name = f"{benchmark.name} | params={total_params:,} | active={active_params:,}"
    except Exception:
        pass

    def setup():
        engine.reset()
        engine.gen_frame(return_img=not dit_only)
        torch.cuda.synchronize()

    def target():
        ctrls = [
            CtrlInput(
                button=set(random.sample(range(1, 65), random.randint(0, 10))),
                mouse=(random.random(), random.random()),
                scroll_wheel=random.choice((-1, 0, 1))
            )
            for _ in range(n_frames)
        ]
        for ctrl in ctrls:
            engine.gen_frame(return_img=not dit_only)
        if blocking:
            torch.cuda.synchronize()

    benchmark.pedantic(target, setup=setup, rounds=20)