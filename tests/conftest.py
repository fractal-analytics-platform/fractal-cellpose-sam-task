def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help=(
            "Run a faster version of the tests, monkeypatching cellpose with a mock "
            "that validates call arguments against the installed Cellpose API"
        ),
    )
