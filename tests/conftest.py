def pytest_addoption(parser):
    parser.addoption(
        "--run-kernel-benchmarks",
        action="store_true",
        default=False,
        help="run opt-in MoE kernel timing benchmarks",
    )
