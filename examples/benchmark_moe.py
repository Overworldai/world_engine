import pytest
import torch

from world_engine import WorldEngine


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


@pytest.fixture(scope="session", autouse=True)
def _compact_benchmark_table(request):
    request.config._benchmarksession.columns = ["median", "max", "mean", "stddev"]


def get_warm_engine(model_uri, model_overrides=None):
    model_config_overrides = {"prompt_conditioning": None}
    model_config_overrides.update(model_overrides or {})
    engine = WorldEngine(
        model_uri,
        model_config_overrides=model_config_overrides,
        device="cuda",
        load_weights=False
    )

    # global warmup
    for _ in range(3):
        engine.gen_frame()
    return engine


@pytest.fixture(scope="session")
def engine(model_uri="Overworld-Models/MR160k"):
    return get_warm_engine(model_uri)


default_cfg = {"prompt_conditioning": None}
dense_moe = {**default_cfg, "moe": True, "moe_n_experts": 8, "moe_top_k": 8}
sparse_moe = {**default_cfg, "moe": True, "moe_n_experts": 16, "moe_top_k": 1, "shared_frame_experts": False}
target_moe = {**default_cfg, "moe": True, "moe_n_experts": 16, "moe_top_k": 4}

moes = [target_moe]

MODEL_OVERRIDES = [default_cfg, *moes]


@pytest.mark.parametrize("dit_only", [True])
@pytest.mark.parametrize("n_frames", [256])
@pytest.mark.parametrize(
    "model_overrides", MODEL_OVERRIDES,
    ids=lambda d: (",".join(f"{k}={v}" for k, v in d.items()) or "") if d else ""
)
def test_ar_rollout(benchmark, dit_only, n_frames, model_overrides):
    engine = get_warm_engine("Overworld-Models/MR160k", model_overrides=model_overrides)
    total_params = sum(p.numel() for p in engine.model.parameters())
    active_params = int(engine.model.get_active_parameters())
    benchmark.name = f"{benchmark.name} | params={total_params:,} | active={active_params:,}"

    def setup():
        engine.reset()
        engine.gen_frame(return_img=not dit_only)
        torch.cuda.synchronize()

    def target():
        for _ in range(n_frames):
            engine.gen_frame(return_img=not dit_only)
        torch.cuda.synchronize()

    benchmark.pedantic(target, setup=setup, rounds=20)
