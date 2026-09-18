"""Deterministic Hypothesis profiles for FrequenSolve contract tests."""

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def close_test_figures():
    """Release figures created by this test without importing optional plotting."""
    pyplot = sys.modules.get("matplotlib.pyplot")
    original = set(pyplot.get_fignums()) if pyplot is not None else set()
    yield
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is not None:
        for number in set(pyplot.get_fignums()) - original:
            pyplot.close(number)


try:
    from hypothesis import HealthCheck, settings
except ModuleNotFoundError:
    # Installed-package contracts intentionally omit development dependencies.
    pass
else:
    settings.register_profile(
        "pr",
        max_examples=50,
        deadline=500,
        derandomize=True,
        suppress_health_check=(HealthCheck.too_slow,),
    )
    settings.register_profile(
        "campaign",
        max_examples=500,
        deadline=1000,
        suppress_health_check=(HealthCheck.too_slow,),
    )
    settings.load_profile(os.environ.get("FREQUENSOLVE_HYPOTHESIS_PROFILE", "pr"))
