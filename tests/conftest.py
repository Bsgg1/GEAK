import threading

import pytest

from minisweagent.models import GLOBAL_MODEL_STATS


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-fire",
        action="store_true",
        default=False,
        help="Run fire tests (real API calls that cost money)",
    )


@pytest.fixture(autouse=True)
def _no_git_prompt(monkeypatch):
    """Prevent git from prompting for credentials during tests."""
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")


@pytest.fixture(autouse=True)
def _unset_data_dir_env(monkeypatch):
    """Ensure bundled-data discovery resolves in-package during tests.

    A leaked GEAK_ROOT / GEAK_SUBAGENTS_ROOT (common in container
    envs) would steer get_data_dir() away from the package and make the
    subagent/skill discovery tests fail spuriously.
    """
    for var in ("GEAK_ROOT", "GEAK_SUBAGENTS_ROOT", "GEAK_USE_SKILLS"):
        monkeypatch.delenv(var, raising=False)


_global_stats_lock = threading.Lock()


@pytest.fixture
def reset_global_stats():
    """Reset global model stats and ensure exclusive access for tests that need it."""
    with _global_stats_lock:
        # Reset at start (pylint: disable=protected-access)
        GLOBAL_MODEL_STATS._cost = 0.0
        GLOBAL_MODEL_STATS._n_calls = 0
        yield
        GLOBAL_MODEL_STATS._cost = 0.0
        GLOBAL_MODEL_STATS._n_calls = 0
