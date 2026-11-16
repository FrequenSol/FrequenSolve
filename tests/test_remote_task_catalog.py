"""Remote catalog refusals name why a task is not current for the staged job."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.hpc.site import SlurmSite
from frequensolve.orchestrator.sites.remote_catalog import (
    RemoteTaskError,
    RemoteTaskFailedError,
    RemoteTaskMissingError,
    RemoteTaskPartitionError,
    RemoteTaskStaleError,
    validate_remote_task_catalog,
)
from frequensolve.simulation.artifact_contract import (
    ArtifactCatalog,
    task_result_path,
)
from frequensolve.simulation.task_index import TaskIndex, TaskIndexEntry

_REMOTE = PurePosixPath("/scratch/fs/jobs/s00_linearize_0001/results")
_FREQUENCIES = (complex(9.0, -0.05), complex(10.0, -0.05), complex(11.0, -0.05))


def _fingerprint(digit: str) -> str:
    return f"sha256:{digit * 64}"


_STAGED = {
    "job": _fingerprint("1"),
    "simulation": _fingerprint("2"),
    "outputs": _fingerprint("3"),
}

# Shape of the Stampede3 Gulf_1D linearize task 3 result: the adjoint solve
# hit FS_MG max_iter and Sauce committed a failed task for the current staging.
_CONVERGENCE = {
    "converged": False,
    "failure_count": 1,
    "solve_count": 3,
    "status": "failed",
    "worst_code": 1,
    "solves": [
        {
            "code": 0,
            "context": "forward",
            "converged": True,
            "grid": 3,
            "iterations": 195,
            "residual": 4.89708019680461e-07,
            "rhs_count": 1,
            "solver": "FS_MG",
            "status": "converged",
            "tolerance": 1e-06,
        },
        {
            "code": 1,
            "context": "adjoint",
            "converged": False,
            "grid": 3,
            "iterations": 2000,
            "residual": 2.8122526899816325e-05,
            "rhs_count": 1,
            "solver": "FS_MG",
            "status": "max_iter",
            "tolerance": 1e-06,
        },
        {
            "code": 0,
            "context": "adjoint",
            "converged": True,
            "grid": 3,
            "iterations": 533,
            "residual": 4.926787699357875e-07,
            "rhs_count": 1,
            "solver": "FS_MG",
            "status": "converged",
            "tolerance": 1e-06,
        },
    ],
}


def _job():
    return SimpleNamespace(
        n_tasks=len(_FREQUENCIES),
        f_list=_FREQUENCIES,
        staged_task_fingerprints=lambda site: dict(_STAGED),
        staged_artifact_fingerprints=lambda site: dict(_STAGED),
    )


def _task_result(task, **overrides):
    frequency = _FREQUENCIES[task - 1]
    value = {
        "schema": "fs-task-result-2",
        "partition": {
            "task": task,
            "task_count": len(_FREQUENCIES),
            "frequency": {"real": frequency.real, "imag": frequency.imag},
        },
        "fingerprints": dict(_STAGED),
        "status": {"state": "success", "code": 0},
        "solver": {"convergence": {"status": "converged", "solves": []}},
        "artifacts": [],
    }
    value.update(copy.deepcopy(overrides))
    return value


def _failed_result(task=3, **overrides):
    return _task_result(
        task,
        status={"state": "failed", "code": 2},
        solver={"convergence": _CONVERGENCE},
        **overrides,
    )


def _catalog(root: Path, results) -> ArtifactCatalog:
    for task, value in results.items():
        path = task_result_path(root, task)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    return ArtifactCatalog.read_task_results(root, tasks=range(1, 4))


def _validate(catalog, *, job=None, remote_root=_REMOTE):
    validate_remote_task_catalog(
        job or _job(),
        catalog,
        range(1, 4),
        site_name="SlurmSite",
        remote_root=remote_root,
    )


def _index(root: Path, **rows) -> TaskIndex:
    entries = {
        task: TaskIndexEntry(
            task_id=task,
            status="success",
            frequency=_FREQUENCIES[task - 1],
            fingerprints=dict(_STAGED),
            artifact_offset=0,
            artifact_count=0,
        )
        for task in range(1, 4)
    }
    for name, row in rows.items():
        task = int(name.removeprefix("task"))
        if row is None:
            entries.pop(task)
        else:
            entries[task] = replace(entries[task], **row)
    return TaskIndex(
        result_path=root,
        path=root / "_fs_run" / "tasks.h5",
        tasks=entries,
        artifacts=(),
    )


def test_current_results_and_index_rows_are_accepted(tmp_path):
    _validate(_catalog(tmp_path, {task: _task_result(task) for task in (1, 2, 3)}))
    _validate(_index(tmp_path))


def test_missing_task_result_is_not_called_a_staging_mismatch(tmp_path):
    catalog = _catalog(tmp_path, {1: _task_result(1), 2: _task_result(2)})

    with pytest.raises(RemoteTaskMissingError) as error:
        _validate(catalog)

    message = str(error.value)
    assert error.value.task == 3
    assert "Remote task 3 (11-0.05j Hz) has not committed a result" in message
    assert f"no result at {_REMOTE}/_fs_run/tasks/task_000003/result.json" in message
    assert "/scratch/fs/jobs/s00_linearize_0001/logs/task_3.log" in message
    assert "does not match" not in message


def test_result_from_another_generation_does_not_match_the_staged_job(tmp_path):
    stale = _failed_result(fingerprints={**_STAGED, "job": _fingerprint("9")})
    catalog = _catalog(tmp_path, {1: _task_result(1), 2: _task_result(2), 3: stale})

    with pytest.raises(RemoteTaskStaleError) as error:
        _validate(catalog)

    message = str(error.value)
    assert "Remote task 3 (11-0.05j Hz) does not match the current staged job" in (
        message
    )
    assert "another staged generation" in message
    assert "job sha256:999999999999 != staged sha256:111111111111" in message
    assert "simulation" not in message
    # A stale record's own failure is context, not the diagnosis.
    assert "that run reported status state 'failed', code 2" in message
    assert not isinstance(error.value, RemoteTaskFailedError)


def test_result_for_another_frequency_is_a_partition_error(tmp_path):
    shifted = _task_result(3)
    shifted["partition"]["frequency"] = {"real": 12.0, "imag": -0.05}
    catalog = _catalog(tmp_path, {1: _task_result(1), 2: _task_result(2), 3: shifted})

    with pytest.raises(RemoteTaskPartitionError) as error:
        _validate(catalog)

    assert str(error.value) == (
        "Remote task 3 (11-0.05j Hz) has the current fingerprints but reports "
        "frequency 12-0.05j Hz instead."
    )


def test_angular_result_for_another_mode_is_a_partition_error(tmp_path):
    job = SimpleNamespace(**vars(_job()), n_modes=3)
    results = {
        task: _task_result(
            task,
            partition={"task": task, "task_count": 3, "azimuthal_mode": task - 1},
        )
        for task in (1, 2, 3)
    }
    results[2]["partition"]["task_count"] = 4
    catalog = _catalog(tmp_path, results)

    with pytest.raises(RemoteTaskPartitionError) as error:
        _validate(catalog, job=job)

    assert str(error.value) == (
        "Remote angular task 2 (azimuthal mode 1) has the current fingerprints "
        "but reports azimuthal mode 1 of 4 tasks instead of mode 1 of 3."
    )


def test_current_failed_task_names_the_failing_solves_and_log(tmp_path):
    catalog = _catalog(
        tmp_path, {1: _task_result(1), 2: _task_result(2), 3: _failed_result()}
    )

    with pytest.raises(RemoteTaskFailedError) as error:
        _validate(catalog)

    assert error.value.task == 3
    assert str(error.value).splitlines() == [
        "Remote task 3 (11-0.05j Hz) ran for the current staged job and failed: "
        "status state 'failed', code 2.",
        "Failed solves (1 of 3):",
        "  solves[1] adjoint: max_iter after 2000 iterations, residual 2.812e-05, "
        "tolerance 1e-06 (FS_MG grid 3, code 1)",
        f"Task result: {_REMOTE}/_fs_run/tasks/task_000003/result.json",
        "Task log: /scratch/fs/jobs/s00_linearize_0001/logs/task_3.log "
        "(site.fetch_logs(job, task=3) downloads it).",
    ]


def test_failed_task_without_a_failed_solve_says_so(tmp_path):
    failed = _task_result(3, status={"state": "failed", "code": 3})
    catalog = _catalog(tmp_path, {1: _task_result(1), 2: _task_result(2), 3: failed})

    with pytest.raises(RemoteTaskFailedError) as error:
        _validate(catalog, remote_root=None)

    message = str(error.value)
    assert "status state 'failed', code 3" in message
    assert "records no failed solve; the failure happened outside" in message
    assert "_fs_run/tasks/task_000003/result.json in the remote result" in message
    assert "Task log: site.fetch_logs(job, task=3) downloads it." in message


@pytest.mark.parametrize(
    ("row", "kind", "expected"),
    [
        (None, RemoteTaskMissingError, "the remote task index has no row for it"),
        (
            {"fingerprints": {**_STAGED, "outputs": _fingerprint("8")}},
            RemoteTaskStaleError,
            "does not match the current staged job: its task index row was "
            "written by another staged generation (outputs sha256:888888888888 "
            "!= staged sha256:333333333333; that run reported task index "
            "status 'success')",
        ),
        (
            {"frequency": complex(11.5, -0.05)},
            RemoteTaskPartitionError,
            "reports frequency 11.5-0.05j Hz instead.",
        ),
        (
            {"status": "failed", "iterations": 2000, "residual": 2.8e-05},
            RemoteTaskFailedError,
            "ran for the current staged job and failed: task index status "
            "'failed'.\nTask index summary: 2000 iterations, residual 2.8e-05.\n"
            "Per-solve contexts and tolerances: "
            f"{_REMOTE}/_fs_run/tasks/task_000003/result.json\n"
            "Task log: /scratch/fs/jobs/s00_linearize_0001/logs/task_3.log",
        ),
    ],
    ids=["missing", "stale", "frequency", "failed"],
)
def test_task_index_refusals_are_classified(tmp_path, row, kind, expected):
    index = _index(tmp_path, task3=row)
    assert not index.is_task_current(3, frequency=_FREQUENCIES[2], fingerprints=_STAGED)

    with pytest.raises(kind) as error:
        _validate(index)

    assert error.value.task == 3
    assert expected in str(error.value)


def test_task_index_angular_jobs_require_task_results(tmp_path):
    job = SimpleNamespace(**vars(_job()), n_modes=3)

    with pytest.raises(RemoteTaskError, match="cannot be validated from the task"):
        _validate(_index(tmp_path), job=job)


def test_slurm_fetch_reports_the_failed_task_with_remote_paths(tmp_path):
    remote = {
        f"_fs_run/tasks/task_{task:06d}/result.json": json.dumps(value).encode()
        for task, value in {
            1: _task_result(1),
            2: _task_result(2),
            3: _failed_result(),
        }.items()
    }

    class Transfer:
        def get_files(self, remote_root, local, paths, **kwargs):
            for relative in paths:
                if relative in remote:
                    target = Path(local) / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(remote[relative])

    class Site:
        fetch_artifact_catalog = SlurmSite.fetch_artifact_catalog
        _validate_remote_catalog = SlurmSite._validate_remote_catalog

        def __init__(self):
            self._transfer = Transfer()

        def _remote_result_dir(self, job):
            return Path(_REMOTE)

    job = SimpleNamespace(**vars(_job()), _result_path=tmp_path / "results")

    with pytest.raises(RemoteTaskFailedError) as error:
        Site().fetch_artifact_catalog(job)

    message = str(error.value)
    assert "Remote task 3 (11-0.05j Hz) ran for the current staged job" in message
    assert "solves[1] adjoint: max_iter after 2000 iterations" in message
    assert f"{_REMOTE}/_fs_run/tasks/task_000003/result.json" in message
    assert "/scratch/fs/jobs/s00_linearize_0001/logs/task_3.log" in message
    assert not (tmp_path / "results" / "_fs_run").exists()
