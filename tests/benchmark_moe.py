import pytest
import torch
from time import perf_counter
from pathlib import Path

from world_engine import WorldEngine


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _frame_debug_summary(frame: torch.Tensor) -> str:
    flat = frame.detach().float().reshape(-1)
    head = flat[:8].cpu().tolist()
    head_fmt = ", ".join(f"{v:.4f}" for v in head)
    return (
        f"shape={tuple(frame.shape)} "
        f"head=[{head_fmt}] "
        f"mean={flat.mean().item():.6f} "
        f"std={flat.std().item():.6f}"
    )


def prewarm_custom_moe_kernels(engine: WorldEngine) -> None:
    from world_engine.kernels import index_shuffling, scatter_add_dense_tokens

    cfg = engine.model_cfg
    has_fbgemm = False
    try:
        import fbgemm_gpu.experimental.gen_ai.moe.gather_scatter  # noqa
        from fbgemm_gpu.experimental.gen_ai.moe import index_shuffling as _fbgemm_index_shuffling  # noqa
        has_fbgemm = _fbgemm_index_shuffling is not None
    except ImportError:
        has_fbgemm = False

    if has_fbgemm or not getattr(cfg, "moe", False):
        return

    n_tokens = cfg.tokens_per_frame
    n_experts = cfg.moe_n_experts
    top_k = min(cfg.moe_top_k, n_experts)
    d_model = cfg.d_model

    scores = torch.randn((n_tokens, n_experts), device=engine.device, dtype=torch.float32)
    _, _, src = index_shuffling(scores, top_k=top_k)

    in_tokens = torch.randn((n_tokens * top_k, d_model), device=engine.device, dtype=engine.dtype)
    scatter_add_dense_tokens(in_tokens, src, n_tokens=n_tokens)
    torch.cuda.synchronize()


def log_moe_runtime_path(engine: WorldEngine) -> None:
    from world_engine.model import world_model as world_model_module

    mlp_types = sorted(
        {
            type(block.mlp).__name__
            for block in engine.model.transformer.blocks
            if hasattr(block, "mlp")
        }
    )
    print(
        "moe_runtime_path "
        f"moe={getattr(engine.model_cfg, 'moe', False)} "
        f"moe_n_experts={getattr(engine.model_cfg, 'moe_n_experts', None)} "
        f"moe_top_k={getattr(engine.model_cfg, 'moe_top_k', None)} "
        f"mlp_types={mlp_types} "
        f"fbgemm_index_shuffling={getattr(world_model_module, 'fbgemm_index_shuffling', None)!r} "
        f"custom_index_shuffling={getattr(world_model_module, 'custom_index_shuffling', None)!r}"
    )


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


@pytest.fixture(scope="session", autouse=False)
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
    benchmark_session = getattr(request.config, "_benchmarksession", None)
    if benchmark_session is None:
        return

    benchmark_session.columns = ["median", "max", "mean", "stddev", "rounds", "iterations"]

    import pytest_benchmark.session as pb_session
    import pytest_benchmark.table as pb_table

    if not getattr(pb_session, "_vram_cols", False):
        orig = pb_session.BenchmarkSession.prepare_benchmarks

        def prepare_benchmarks(self):
            for bench in orig(self):
                info = bench.get("extra_info") or {}
                if "max_vram_alloc" in info:
                    bench["rounds"] = info["max_vram_alloc"]
                if "max_vram_reserved" in info:
                    bench["iterations"] = info["max_vram_reserved"]
                yield bench

        pb_session.BenchmarkSession.prepare_benchmarks = prepare_benchmarks
        pb_session._vram_cols = True

    if not getattr(pb_table, "_vram_labels", False):
        code = pb_table.TableResults.display.__code__
        mapping = {"Rounds": "max_vram_alloc", "Iterations": "max_vram_reserved"}
        pb_table.TableResults.display.__code__ = code.replace(
            co_consts=tuple(mapping.get(c, c) for c in code.co_consts)
        )
        pb_table._vram_labels = True


def get_warm_engine(model_uri, model_overrides=None):
    model_config_overrides = {"ae_uri": "Overworld-Models/taehv1_5", "ae_type": "taehv"}
    model_config_overrides.update(model_overrides or {})
    engine = WorldEngine(
        model_uri,
        model_config_overrides=model_config_overrides,
        device="cuda",
        load_weights=False
    )

    log_moe_runtime_path(engine)

    # prewarm_custom_moe_kernels(engine)

    # global warmup
    for _ in range(3):
        engine.gen_frame()
    return engine


@pytest.fixture(scope="session")
def engine(model_uri="Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill"):
    return get_warm_engine(model_uri)

@pytest.fixture(scope="session")
def last_latent(engine):
    return engine.gen_frame(return_img=False).detach()



def test_img_decoder_only(benchmark, engine, last_latent):
    def run():
        with torch.amp.autocast("cuda", torch.bfloat16):
            engine.vae.decode(last_latent)
        torch.cuda.synchronize()

    benchmark(run)


default_cfg = {
    "taehv_ae": True,
    "ae_uri": "Overworld-Models/taehv1_5",
    "gated_linear": False,
    #"ctrl_conditioning_period": None,
    #"scheduler_sigmas": [1.0, 0.75, 0.5, 0.25, 0.0],
    #"channels": 16,
    #"gated_linear": False,
    #"ctrl_conditioning_period": 3,
    #"prompt_conditioning": None,
    #"global_window": 256,
    "n_kv_heads": 16,
    "n_heads": 32,
}
dense_moe = {**default_cfg, "moe": True, "moe_n_experts": 8, "moe_top_k": 8, "shared_frame_experts": False}
xs_moe = {**default_cfg, "moe": True, "moe_n_experts": 16, "moe_top_k": 8, "shared_frame_experts": False, "n_layers": 8}
sparse_moe = {**default_cfg, "moe": True, "moe_n_experts": 16, "moe_top_k": 1, "shared_frame_experts": False}
target_moe = {**default_cfg, "moe": True, "moe_n_experts": 32, "moe_top_k": 4, "shared_frame_experts": False}

moes = [dense_moe, sparse_moe, target_moe]
shared_frame_moes = [{**x, "shared_frame_experts": True} for x in moes]

MODEL_OVERRIDES = [
    default_cfg,
    *moes,
    #*shared_frame_moes,
]
MOE_MODEL_OVERRIDES = [cfg for cfg in MODEL_OVERRIDES if cfg.get("moe", False)]
DENSE_MOE_MODEL_OVERRIDES = [dense_moe]
XS_MOE_MODEL_OVERRIDES = [xs_moe]

@pytest.mark.parametrize(
    "model_overrides", XS_MOE_MODEL_OVERRIDES,
    ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items())
)
def test_moe_gen_frame_once(model_overrides):
    engine = WorldEngine(
        "Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill",
        model_config_overrides=model_overrides,
        device="cuda",
        load_weights=False,
    )
    prewarm_custom_moe_kernels(engine)
    frame = engine.gen_frame(return_img=False)
    torch.cuda.synchronize()

    assert frame is not None
    assert frame.is_cuda
    assert frame.dtype == engine.dtype


@pytest.mark.parametrize(
    "model_overrides", XS_MOE_MODEL_OVERRIDES,
    ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items())
)
def test_moe_gen_frame_rollout_repro(model_overrides):
    _set_seed(0)
    warm_start = perf_counter()
    engine = get_warm_engine(
        "Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill",
        model_overrides=model_overrides,
    )
    warm_ms = (perf_counter() - warm_start) * 1000

    frame = None
    print(f"rollout_repro warmup_ms={warm_ms:.2f}")
    for _ in range(3):
        setup_start = perf_counter()
        engine.reset()
        engine.gen_frame(return_img=False)
        torch.cuda.synchronize()
        setup_ms = (perf_counter() - setup_start) * 1000

        rollout_start = perf_counter()
        for _ in range(64):
            frame = engine.gen_frame(return_img=False)
        torch.cuda.synchronize()
        rollout_ms = (perf_counter() - rollout_start) * 1000
        print(f"rollout_repro setup_ms={setup_ms:.2f} rollout_ms={rollout_ms:.2f} per_frame_ms={rollout_ms / 64:.2f}")
        print(f"rollout_repro output {_frame_debug_summary(frame)}")

    assert frame is not None
    assert frame.is_cuda


@pytest.mark.parametrize(
    "model_overrides", DENSE_MOE_MODEL_OVERRIDES,
    ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items())
)
def test_dump_moe_fbgemm_fx_graph(model_overrides):
    from torch._dynamo import export

    engine = WorldEngine(
        "Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill",
        model_config_overrides={"ae_uri": "Overworld-Models/taehv1_5", "ae_type": "taehv", **model_overrides},
        device="cuda",
        load_weights=False,
    )

    class SingleDenoiseStep(torch.nn.Module):
        def __init__(self, engine: WorldEngine):
            super().__init__()
            self.engine = engine

        def forward(self, x):
            ctx = self.engine.prep_inputs(x)
            self.engine.kv_cache.set_frozen(True)
            sigma = x.new_empty((x.size(0), x.size(1)))
            step_sig = self.engine.scheduler_sigmas[0]
            return self.engine.model(x, sigma.fill_(step_sig), **ctx, kv_cache=self.engine.kv_cache)

    x = torch.randn(engine.frm_shape, device=engine.device, dtype=engine.dtype)
    module = SingleDenoiseStep(engine).eval()
    with torch.no_grad():
        exported, guards = export(module, x)

    out = Path("tests/generated_kernels/fbgemm_moe_fx_graph.txt")
    out.write_text(
        "\n".join(
            [
                f"guards={guards}",
                "",
                "=== graph ===",
                str(exported.graph),
                "",
                "=== code ===",
                exported.code,
            ]
        )
    )
    print(f"wrote FX graph to {out}")


@pytest.mark.parametrize("dit_only", [True])
@pytest.mark.parametrize("n_frames", [256])
@pytest.mark.parametrize(
    "model_overrides", XS_MOE_MODEL_OVERRIDES,
    ids=lambda d: (",".join(f"{k}={v}" for k, v in d.items()) or "") if d else ""
)
def test_ar_rollout(benchmark, dit_only, n_frames, model_overrides):
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    engine = get_warm_engine("Overworld-Models/Lapp0-WP-Mini-1.4.5-BL-Distill", model_overrides=model_overrides)
    total_params = sum(p.numel() for p in engine.model.parameters())
    active_params = int(engine.model.get_active_parameters())
    benchmark.name = f"{benchmark.name} | params={total_params:,} | active={active_params:,}"

    max_alloc = 0
    max_reserved = 0

    def setup():
        engine.reset()
        engine.gen_frame(return_img=not dit_only)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def target():
        for _ in range(n_frames):
            engine.gen_frame(return_img=not dit_only)
        torch.cuda.synchronize()

    def teardown():
        nonlocal max_alloc, max_reserved
        max_alloc = max(max_alloc, torch.cuda.max_memory_allocated())
        max_reserved = max(max_reserved, torch.cuda.max_memory_reserved())

    benchmark.pedantic(target, setup=setup, teardown=teardown, rounds=10)
    benchmark.extra_info["max_vram_alloc"] = round(max_alloc / 2**30, 2)
    benchmark.extra_info["max_vram_reserved"] = round(max_reserved / 2**30, 2)
