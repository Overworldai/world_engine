import os

# some of the kernel recompilation changes require fresh torchinductor cache, as stale
# cache hits will cause the wrong compiled variant of a kernel to be used 
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"


def pytest_addoption(parser):
    parser.addoption(
        "--run-kernel-benchmarks",
        action="store_true",
        default=False,
        help="run opt-in MoE kernel timing benchmarks",
    )
