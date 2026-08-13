"""Deterministic Hypothesis profiles for FrequenSolve contract tests."""

import os

from hypothesis import HealthCheck, settings

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
