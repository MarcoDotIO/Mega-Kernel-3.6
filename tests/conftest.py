def pytest_addoption(parser):
    parser.addoption(
        "--run-runtime-parity",
        action="store_true",
        default=False,
        help="run optional vLLM/SGLang numerical parity tests",
    )
