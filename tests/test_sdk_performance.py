from __future__ import annotations

from copy import deepcopy

import h5py
import pytest

from frequensolve.seismic.trace_store import TraceStore
from scripts.run_sdk_performance import (
    BASELINE_SCHEMA,
    SCHEMA,
    Scenario,
    _write_trace_product,
    compare_to_baseline,
    comparison_dependency_versions,
    comparison_runner_identity,
    measure_scenario,
    validate_evidence,
)


def _evidence() -> dict:
    return {
        "schema": SCHEMA,
        "runner": {
            "system": "Linux",
            "release": "reviewed-kernel",
            "machine": "x86_64",
            "processor": "reviewed-cpu",
            "cpuCount": 4,
            "pythonImplementation": "CPython",
            "pythonVersion": "3.12.1",
            "runnerName": "ephemeral-job-name",
            "runnerImageOs": "ubuntu24",
            "runnerImageVersion": "20260801.1",
        },
        "dependencies": {"frequensolve": "1.2.3", "numpy": "2.4.0"},
        "scenarios": [
            {
                "name": "representative-scenario",
                "samples": [
                    {
                        "sample": 1,
                        "wallTimeSeconds": 0.25,
                        "pythonHeapPeakBytes": 1024,
                    }
                ],
                "wallTimeSeconds": {"median": 0.25},
                "pythonHeapPeakBytes": {"median": 1024.0},
            }
        ],
    }


def _baseline(evidence: dict) -> dict:
    return {
        "schema": BASELINE_SCHEMA,
        "runner": comparison_runner_identity(evidence["runner"]),
        "dependencies": comparison_dependency_versions(evidence["dependencies"]),
        "thresholds": {
            "representative-scenario": {
                "maxMedianWallTimeSeconds": 0.5,
                "maxMedianPythonHeapBytes": 2048,
            }
        },
    }


def test_measurement_retains_raw_samples_and_variance_statistics():
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        return {"calls": calls}

    scenario = Scenario(
        name="deterministic",
        category="unit",
        size="small",
        operation=operation,
    )

    with pytest.raises(RuntimeError, match="changed across samples"):
        measure_scenario(scenario, warmups=1, samples=2)

    assert calls == 3

    measured = measure_scenario(
        Scenario("stable", "unit", "small", lambda: {"items": 1}),
        warmups=1,
        samples=3,
    )
    assert len(measured["samples"]) == 3
    assert measured["wallTimeSeconds"]["median"] > 0
    assert measured["pythonHeapPeakBytes"]["median"] > 0
    assert measured["wallTimeSeconds"]["coefficientOfVariation"] >= 0


def test_trace_fixture_selects_indexed_packed_access(tmp_path):
    path = tmp_path / "traces.h5"
    _write_trace_product(path, frequencies=3, receivers=4)

    with h5py.File(path, "r") as h5:
        assert TraceStore._is_indexed_packed_h5(h5)
        assert "surface" not in h5
        assert len(h5["trace_index/datasets/packed_path"]) == 3


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda evidence: evidence.update(scenarios=[]), "no scenarios"),
        (
            lambda evidence: evidence["scenarios"][0].update(samples=[]),
            "no measurements",
        ),
        (
            lambda evidence: evidence["scenarios"][0]["samples"][0].update(
                wallTimeSeconds=0
            ),
            "empty wall-time",
        ),
    ],
)
def test_evidence_validation_fails_closed(mutation, message):
    evidence = _evidence()
    mutation(evidence)

    with pytest.raises(ValueError, match=message):
        validate_evidence(evidence, ["representative-scenario"])


def test_baseline_comparison_ignores_only_ephemeral_runner_name():
    evidence = _evidence()
    baseline = _baseline(evidence)

    assert compare_to_baseline(evidence, baseline) == []
    evidence["runner"]["runnerName"] = "another-ephemeral-job-name"
    assert compare_to_baseline(evidence, baseline) == []

    evidence["runner"]["processor"] = "different-cpu"
    with pytest.raises(ValueError, match="runner identity drifted"):
        compare_to_baseline(evidence, baseline)


def test_baseline_comparison_rejects_dependency_drift_and_regressions():
    evidence = _evidence()
    baseline = _baseline(evidence)
    drifted = deepcopy(evidence)
    drifted["dependencies"]["numpy"] = "2.5.0"

    with pytest.raises(ValueError, match="dependency versions drifted"):
        compare_to_baseline(drifted, baseline)

    new_package_commit = deepcopy(evidence)
    new_package_commit["dependencies"]["frequensolve"] = "1.2.4"
    assert compare_to_baseline(new_package_commit, baseline) == []

    evidence["scenarios"][0]["wallTimeSeconds"]["median"] = 0.75
    evidence["scenarios"][0]["pythonHeapPeakBytes"]["median"] = 4096
    violations = compare_to_baseline(evidence, baseline)

    assert len(violations) == 2
    assert "median wall time" in violations[0]
    assert "median Python heap" in violations[1]


def test_baseline_comparison_rejects_missing_scenarios():
    evidence = _evidence()
    baseline = _baseline(evidence)
    baseline["thresholds"]["missing-scenario"] = {
        "maxMedianWallTimeSeconds": 1.0,
        "maxMedianPythonHeapBytes": 4096,
    }

    with pytest.raises(ValueError, match="scenario set differs"):
        compare_to_baseline(evidence, baseline)
