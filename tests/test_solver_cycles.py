"""Public multigrid settings preserve backend defaults and serialize explicit choices."""

from types import MappingProxyType

import pytest

from frequensolve import SolverConfig


@pytest.mark.parametrize("cycle", ["auto", "V", "W", "log", 2, {"visits": [1, 2, 2]}])
def test_cycle_round_trip(cycle):
    config = SolverConfig(cycle=cycle, pre_smooth=0, post_smooth=2)
    payload = config.to_fs()
    assert payload["pre_smooth"] == 0
    assert payload["post_smooth"] == 2
    assert payload.get("cycle", "auto") == cycle
    assert SolverConfig.from_fs(payload).to_fs() == payload


def test_defaults_preserve_backend_policy_and_copy_schedule():
    assert "cycle" not in SolverConfig(mode="fast").to_fs()
    assert "pre_smooth" not in SolverConfig().to_fs()
    schedule = {"visits": [1, 2], "anchor": "fine"}
    config = SolverConfig(cycle=MappingProxyType(schedule))
    schedule["visits"][0] = 9
    assert config.to_fs()["cycle"]["visits"] == [1, 2]
    config.pre_smooth = 0
    assert config.to_fs()["pre_smooth"] == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"cycle": "invalid"},
        {"cycle": 0},
        {"cycle": True},
        {"cycle": {"visits": []}},
        {"cycle": {"visits": [1, -1]}},
        {"cycle": {"coarse": 0}},
        {"cycle": {"anchor": "invalid"}},
        {"pre_smooth": -1},
        {"post_smooth": 1.5},
        {"post_smooth": True},
    ],
)
def test_invalid_cycle_settings(settings):
    with pytest.raises(ValueError):
        SolverConfig(**settings)
