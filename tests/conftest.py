import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run a faster version of the tests, monkeypatching cellpose",
    )


@pytest.fixture
def is_github_or_fast(request):
    """Fixture to determine if tests are running in GitHub Actions or in fast mode."""
    is_git_actions = "GITHUB_ACTIONS" in os.environ
    is_fast = request.config.getoption("--fast")
    return is_git_actions or is_fast
