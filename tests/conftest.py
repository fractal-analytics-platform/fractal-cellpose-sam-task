import pytest


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


@pytest.fixture
def is_fast(request):
    """Whether the Cellpose model should be mocked (`--fast` option)."""
    return request.config.getoption("--fast")
