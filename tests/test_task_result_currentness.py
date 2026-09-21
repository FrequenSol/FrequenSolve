"""Frequency currentness through authoritative Sauce task results."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.project.project import Project
from frequensolve.simulation.artifact_contract import ArtifactCatalog
from frequensolve.simulation.jobs import FrequencyDomainJob


def _saved_job(
    tmp_path: Path,
    frequencies: list[complex | float],
) -> FrequencyDomainJob:
    project = Project(name="project", path=tmp_path / "project")
    simulation = project.new_simulation(
        name="simple",
        physics="acoustic",
        dimension=2,
    )
    simulation.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1])
    )
    job = FrequencyDomainJob(
        name="frequency",
        simulation=simulation,
        f_list=frequencies,
    )
    job.save()
    return job


def _write_task_result(
    job: FrequencyDomainJob,
    task: int,
    *,
    status: str = "success",
    fingerprints: dict[str, str] | None = None,
    write_payload: bool = True,
    declared_bytes: int | None = None,
) -> tuple[Path, Path]:
    relative_payload = Path("payload") / f"generation-a-task-{task}.bin"
    payload = job._result_path / relative_payload
    data = f"opaque task {task} payload".encode()
    if write_payload:
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(data)
    result = job._result_path / "_fs_run" / "tasks" / f"task_{task:06d}" / "result.json"
    result.parent.mkdir(parents=True, exist_ok=True)
    frequency = complex(job.f_list[task - 1])
    result.write_text(
        json.dumps(
            {
                "schema": "fs-task-result-2",
                "partition": {
                    "task": task,
                    "task_count": job.n_tasks,
                    "frequency": {
                        "real": frequency.real,
                        "imag": frequency.imag,
                    },
                },
                "fingerprints": (
                    job._artifact_contract_fingerprints()
                    if fingerprints is None
                    else fingerprints
                ),
                "status": {
                    "state": status,
                    "code": 0 if status in {"success", "skipped"} else 1,
                },
                "artifacts": [
                    {
                        "id": "traces",
                        "role": "simulated_traces",
                        "representation": "hdf5_shard",
                        "schema": "fs-trace-shard-2",
                        "path": relative_payload.as_posix(),
                        "retention": "durable",
                        "generation": "generation-a",
                        "bytes": (
                            len(data) if declared_bytes is None else declared_bytes
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return result, payload


def _write_task_index(job: FrequencyDomainJob) -> Path:
    """Materialize the frozen index schema from this test's task results."""

    rows = [
        json.loads(
            (
                job._result_path
                / "_fs_run"
                / "tasks"
                / f"task_{task:06d}"
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        for task in range(1, job.n_tasks + 1)
    ]
    path = job._result_path / "_fs_run" / "tasks.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "fs-task-index-1"
        tasks = h5.create_group("tasks")
        artifacts = h5.create_group("artifacts")
        dependencies = h5.create_group("dependencies")
        tasks.create_dataset(
            "task_id", data=np.arange(1, job.n_tasks + 1, dtype=np.int64)
        )
        tasks.create_dataset(
            "status",
            data=np.asarray([row["status"]["state"] for row in rows], dtype=object),
            dtype=strings,
        )
        for component in ("real", "imag"):
            tasks.create_dataset(
                f"frequency_{component}",
                data=np.asarray(
                    [row["partition"]["frequency"][component] for row in rows],
                    dtype=np.float64,
                ),
            )
        for name in ("job", "simulation", "outputs"):
            tasks.create_dataset(
                f"fingerprint_{name}",
                data=np.asarray(
                    [row["fingerprints"][name] for row in rows], dtype=object
                ),
                dtype=strings,
            )
        tasks.create_dataset(
            "artifact_offset", data=np.arange(job.n_tasks, dtype=np.int64)
        )
        tasks.create_dataset(
            "artifact_count", data=np.ones(job.n_tasks, dtype=np.int64)
        )
        tasks.create_dataset(
            "iterations", data=np.full(job.n_tasks, -1, dtype=np.int64)
        )
        tasks.create_dataset(
            "residual", data=np.full(job.n_tasks, np.nan, dtype=np.float64)
        )
        for name in (
            "mesh",
            "setup",
            "assembly",
            "solve_forward",
            "solve_adjoint",
            "imaging",
        ):
            tasks.create_dataset(
                f"timing_{name}", data=np.full(job.n_tasks, np.nan, dtype=np.float64)
            )
        for name in (
            "id",
            "role",
            "schema",
            "representation",
            "path",
            "retention",
            "generation",
        ):
            artifacts.create_dataset(
                name,
                data=np.asarray(
                    [row["artifacts"][0].get(name, "") for row in rows],
                    dtype=object,
                ),
                dtype=strings,
            )
        artifacts.create_dataset(
            "bytes",
            data=np.asarray(
                [row["artifacts"][0]["bytes"] for row in rows], dtype=np.int64
            ),
        )
        artifacts.create_dataset(
            "dependency_offset", data=np.zeros(job.n_tasks, dtype=np.int64)
        )
        artifacts.create_dataset(
            "dependency_count", data=np.zeros(job.n_tasks, dtype=np.int64)
        )
        dependencies.create_dataset(
            "id", data=np.asarray([], dtype=object), dtype=strings
        )
    return path


def test_currentness_distinguishes_equal_real_laplace_tasks(tmp_path):
    job = _saved_job(tmp_path, [5.0 - 0.2j, 5.0 - 0.05j])
    _write_task_result(job, 1)
    _write_task_result(job, 2, status="skipped")

    assert job.current_tasks() == [1, 2]
    assert job.is_run_current()
    assert job.task_run_plan()["pending_indices"] == []


def test_raw_input_fingerprints_detect_public_file_rewrites(tmp_path):
    job = _saved_job(tmp_path, [5.0])
    _write_task_result(job, 1)
    job_payload = json.loads(job._file.read_text())
    simulation_payload = json.loads(job.simulation._file.read_text())
    job._file.write_text(json.dumps(job_payload, indent=7), encoding="utf-8")
    job.simulation._file.write_text(
        json.dumps(simulation_payload, indent=5),
        encoding="utf-8",
    )

    assert not job.is_task_current(1)


def test_effective_output_request_has_explicit_contract_scope(tmp_path):
    job = _saved_job(tmp_path, [5.0])

    request = job.effective_output_request_payload()

    assert request == {
        "schema": "fs-effective-output-request-1",
        "workflow": "forward",
        "Outputs": job.to_fs(job.export_context())["Outputs"],
    }


def test_direct_job_without_output_digest_uses_raw_job_fallback(tmp_path):
    job = _saved_job(tmp_path, [5.0])
    payload = json.loads(job._file.read_text(encoding="utf-8"))
    payload.pop("artifact_contract")
    job._file.write_text(json.dumps(payload), encoding="utf-8")

    fingerprints = job._artifact_contract_fingerprints()

    assert fingerprints is not None
    assert fingerprints["outputs"] == fingerprints["job"]
    _write_task_result(job, 1, fingerprints=fingerprints)
    assert job.is_task_current(1)


@pytest.mark.parametrize("field", ["job", "simulation", "outputs"])
def test_stale_task_fingerprint_is_not_current(tmp_path, field):
    job = _saved_job(tmp_path, [3.0])
    fingerprints = job._artifact_contract_fingerprints()
    assert fingerprints is not None
    fingerprints[field] = "f" * 64
    _write_task_result(job, 1, fingerprints=fingerprints)

    assert not job.is_task_current(1)
    assert job.current_tasks() == []
    assert job.task_run_plan()["pending_indices"] == [0]


def test_missing_task_result_is_pending(tmp_path):
    job = _saved_job(tmp_path, [3.0])

    assert not job.is_task_current(1)
    assert not job.is_run_current()
    assert job.task_run_plan()["pending_indices"] == [0]


def test_missing_declared_payload_is_pending(tmp_path):
    job = _saved_job(tmp_path, [3.0])
    _write_task_result(job, 1, write_payload=False)

    assert not job.is_task_current(1)
    assert job.task_run_plan()["pending_indices"] == [0]


def test_declared_payload_size_must_match(tmp_path):
    job = _saved_job(tmp_path, [3.0])
    _write_task_result(job, 1, declared_bytes=1)

    assert not job.is_task_current(1)
    assert job.task_run_plan()["pending_indices"] == [0]


def test_failed_task_result_is_pending(tmp_path):
    job = _saved_job(tmp_path, [3.0])
    _write_task_result(job, 1, status="failed")

    assert not job.is_task_current(1)
    assert job.frequency_status()[0]["status"] == "failed"


def test_currentness_never_scans_or_opens_hdf5(tmp_path, monkeypatch):
    job = _saved_job(tmp_path, [2.0, 4.0])
    _write_task_result(job, 1)
    _write_task_result(job, 2)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("currentness must not discover or open payloads")

    monkeypatch.setattr(Path, "glob", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    monkeypatch.setattr(h5py, "File", forbidden)

    assert job.current_tasks() == [1, 2]
    assert job.task_run_plan()["pending_indices"] == []


def test_currentness_loads_the_consolidated_index_once(tmp_path, monkeypatch):
    job = _saved_job(tmp_path, [2.0 - 0.2j, 4.0 - 0.05j])
    _write_task_result(job, 1)
    _write_task_result(job, 2)
    _write_task_index(job)

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("a current index must not re-read task JSON payloads")

    monkeypatch.setattr(
        ArtifactCatalog,
        "read_task_results",
        classmethod(forbidden_fallback),
    )

    assert job.current_tasks() == [1, 2]
    assert job.task_run_plan()["pending_indices"] == []
