"""Bounded, output-based tolerance pilots using independent solver configurations."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import frequensolve as fs
from frequensolve.simulation.tuning import _relative_error

from .test_transient import job_at


def test_pilots_preserve_parent_and_numerical_problem(tmp_path):
    job = job_at(tmp_path)
    before = job.to_fs()
    study = job.tolerance_study(steps=5)
    assert job.to_fs() == before
    assert list(study.jobs) == [1e-5, 1e-4, 1e-3]
    for tolerance, pilot in study.jobs.items():
        assert pilot.simulation is not job.simulation
        assert pilot.time.dt == job.time.dt and pilot.time.steps == 5
        assert pilot.time.method == job.time.method
        assert pilot.source == job.source
        assert pilot.simulation.solver.tolerance == tolerance
        assert pilot.checkpoint == fs.CheckpointSchedule()
        assert pilot.outputs is not job.outputs
        assert pilot.job_file != job.job_file
        pilot.save()
        restored = fs.BaseJob.load(pilot.job_file)
        assert restored.simulation.solver.tolerance == tolerance
    assert job.simulation.solver.tolerance is None
    assert len({pilot.to_fs()["simulation"] for pilot in study.jobs.values()}) == 3
    with pytest.raises(ValueError, match="exceed"):
        job.tolerance_study(steps=11)
    with pytest.raises(ValueError, match="reference"):
        job.tolerance_study(steps=5, reference_tolerance=1e-3)


def samples(values):
    return xr.DataArray(
        values, dims=["source", "time"], coords={"source": [1, 2], "time": [0, 1]}
    )


def test_comparison_checks_each_source_and_alignment():
    ref = samples([[1000, 1000], [1, 1]])
    perturbed = samples([[1000, 1000], [1.1, 1.1]])
    assert _relative_error(ref, perturbed) == pytest.approx(0.1)
    for invalid in [
        ref.assign_coords(time=[1, 2]),
        ref * np.nan,
        samples([[0, 0], [1, 1]]),
    ]:
        with pytest.raises(ValueError):
            _relative_error(invalid, ref)


def test_study_selects_measured_candidate_without_mutation(tmp_path, monkeypatch):
    job = job_at(tmp_path)
    study = job.tolerance_study(steps=5, tolerances=[1e-4, 3e-4, 1e-3])
    before = job.to_fs()
    calls = []

    def run(pilot, **options):
        assert options == {"check": True}
        calls.append(pilot.simulation.solver.tolerance)
        if pilot.simulation.solver.tolerance == 3e-4:
            raise ValueError("rejected diagnostic candidate")

    def output(pilot):
        data = samples(np.ones((2, 2)) * (1 + pilot.simulation.solver.tolerance))
        return SimpleNamespace(open=lambda: SimpleNamespace(read=lambda group: data))

    monkeypatch.setattr(fs.TransientJob, "wavefields", property(output))
    monkeypatch.setattr(
        fs.TransientJob,
        "operation_result",
        lambda _: SimpleNamespace(
            timings={"elapsed": 1},
            solver={"convergence": {"solves": [{"iterations": 3}]}},
        ),
    )
    result = study.run(
        SimpleNamespace(run=run),
        groups=["velocity"],
        output="wavefields",
        max_relative_error=2e-4,
    )
    assert calls == [1e-5, 1e-4, 3e-4, 1e-3]
    assert result.recommended_tolerance == 1e-4
    assert result.trials[1]["mean_iterations"] == 3
    assert "failure" in result.trials[2]
    assert not result.trials[3]["passed"]
    assert job.to_fs() == before

    def fail_reference(*args, **kwargs):
        raise ValueError("failed reference")

    with pytest.raises(ValueError, match="failed reference"):
        study.run(
            SimpleNamespace(run=fail_reference),
            groups=["velocity"],
            max_relative_error=0.001,
        )
