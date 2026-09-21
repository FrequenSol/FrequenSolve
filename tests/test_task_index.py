from __future__ import annotations

import json
import os

import h5py
import numpy as np
import pytest

from frequensolve.simulation.artifact_contract import (
    ArtifactCatalog,
    ArtifactContractError,
    ArtifactRequest,
    task_result_path,
)
from frequensolve.simulation.task_index import (
    TASK_INDEX_VERSION,
    TaskIndex,
    load_task_catalog,
    task_index_path,
)

_STRING = h5py.string_dtype(encoding="utf-8")
_TIMINGS = (
    "timing_mesh",
    "timing_setup",
    "timing_assembly",
    "timing_solve_forward",
    "timing_solve_adjoint",
    "timing_imaging",
)


def _fingerprint(digit: str) -> str:
    return f"sha256:{digit * 64}"


def _strings(group, name, values):
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=_STRING)


def _write_index(result_path, *, count=2):
    path = task_index_path(result_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    task_ids = np.arange(1, count + 1, dtype=np.int64)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = TASK_INDEX_VERSION
        tasks = h5.create_group("tasks")
        artifacts = h5.create_group("artifacts")
        dependencies = h5.create_group("dependencies")
        tasks.create_dataset("task_id", data=task_ids)
        _strings(tasks, "status", ["success"] * count)
        tasks.create_dataset(
            "frequency_real", data=np.arange(1, count + 1, dtype=np.float64)
        )
        tasks.create_dataset(
            "frequency_imag", data=np.full(count, 0.05, dtype=np.float64)
        )
        _strings(tasks, "fingerprint_job", [_fingerprint("1")] * count)
        _strings(tasks, "fingerprint_simulation", [_fingerprint("2")] * count)
        _strings(tasks, "fingerprint_outputs", [_fingerprint("3")] * count)
        tasks.create_dataset("artifact_offset", data=np.arange(count, dtype=np.int64))
        tasks.create_dataset("artifact_count", data=np.ones(count, dtype=np.int64))
        tasks.create_dataset("iterations", data=np.arange(10, 10 + count))
        tasks.create_dataset(
            "residual", data=np.linspace(1.0e-5, 2.0e-5, count, dtype=np.float64)
        )
        for position, name in enumerate(_TIMINGS):
            tasks.create_dataset(
                name,
                data=np.full(count, position / 10.0, dtype=np.float64),
            )

        _strings(artifacts, "id", [f"traces-{task}" for task in task_ids])
        _strings(artifacts, "role", ["simulated_traces"] * count)
        _strings(artifacts, "schema", ["trace-shard-1"] * count)
        _strings(artifacts, "representation", ["hdf5_shard"] * count)
        _strings(
            artifacts,
            "path",
            [f"opaque/task-{task}.h5" for task in task_ids],
        )
        _strings(artifacts, "retention", ["durable"] * count)
        _strings(artifacts, "generation", ["generation-1"] * count)
        artifacts.create_dataset("bytes", data=np.full(count, 7, dtype=np.int64))
        artifacts.create_dataset(
            "dependency_offset", data=np.zeros(count, dtype=np.int64)
        )
        artifacts.create_dataset(
            "dependency_count", data=np.zeros(count, dtype=np.int64)
        )
        _strings(dependencies, "id", [])
    return path


def _write_task_result(result_path, task):
    path = task_result_path(result_path, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "fs-task-result-2",
                "partition": {
                    "task": task,
                    "task_count": 2,
                    "frequency": {"real": float(task), "imag": 0.05},
                },
                "fingerprints": {
                    "job": _fingerprint("1"),
                    "simulation": _fingerprint("2"),
                    "outputs": _fingerprint("3"),
                },
                "status": {"state": "success", "code": 0},
                "solver": {
                    "convergence": {"iterations": 9 + task, "residual": task * 1e-5}
                },
                "timings": {
                    name.removeprefix("timing_"): position / 10.0
                    for position, name in enumerate(_TIMINGS)
                },
                "artifacts": [
                    {
                        "id": f"traces-{task}",
                        "role": "simulated_traces",
                        "schema": "trace-shard-1",
                        "representation": "hdf5_shard",
                        "path": f"opaque/task-{task}.h5",
                        "retention": "durable",
                        "generation": "generation-1",
                        "bytes": 7,
                    }
                ],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return path


def test_task_index_reads_task_summaries_and_queries_artifacts(tmp_path):
    result_path = tmp_path / "results"
    _write_index(result_path)
    payload = result_path / "opaque/task-2.h5"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"1234567")

    index = TaskIndex.read(result_path)

    assert list(index.tasks) == [1, 2]
    assert index.tasks[2].frequency == complex(2.0, 0.05)
    assert index.tasks[2].iterations == 11
    assert index.tasks[2].residual == pytest.approx(2.0e-5)
    assert index.tasks[2].timings["solve_forward"] == pytest.approx(0.3)
    assert [item.id for item in index.query(role="simulated_traces")] == [
        "traces-1",
        "traces-2",
    ]
    request = ArtifactRequest(
        role="simulated_traces",
        representations=("hdf5_shard",),
        retention="durable",
    )
    assert index.require_one(request, task=2).path == payload.resolve()
    assert index.is_task_current(
        2,
        frequency=complex(2.0, 0.05),
        fingerprints={"job": _fingerprint("1")},
        required_artifact=request,
    )
    assert not index.is_task_current(2, frequency=complex(2.0, 0.0))


def test_task_index_and_json_catalog_queries_agree(tmp_path):
    result_path = tmp_path / "results"
    for task in (1, 2):
        _write_task_result(result_path, task)
    _write_index(result_path)

    index = TaskIndex.read(result_path)
    json_catalog = ArtifactCatalog.read_task_results(result_path, tasks=(1, 2))
    request = ArtifactRequest(role="simulated_traces", representations=("hdf5_shard",))

    assert index.query(role="simulated_traces") == json_catalog.query(
        role="simulated_traces"
    )
    assert index.select(request, task=2) == json_catalog.select(request, task=2)


def test_task_index_reconstructs_dependency_closure(tmp_path):
    result_path = tmp_path / "results"
    path = _write_index(result_path)
    with h5py.File(path, "a") as h5:
        h5["tasks/artifact_count"][...] = [2, 0]
        h5["tasks/artifact_offset"][...] = [0, 2]
        h5["artifacts/dependency_offset"][...] = [0, 1]
        h5["artifacts/dependency_count"][...] = [1, 0]
        del h5["dependencies/id"]
        _strings(h5["dependencies"], "id", ["traces-2"])

    index = TaskIndex.read(result_path)

    assert index.artifacts_for_task(1)[0].dependencies == ("traces-2",)


def test_task_index_rejects_invalid_dependency_slices_and_unresolved_ids(tmp_path):
    result_path = tmp_path / "results"
    path = _write_index(result_path)
    with h5py.File(path, "a") as h5:
        h5["artifacts/dependency_offset"][...] = [0, 2]
        h5["artifacts/dependency_count"][...] = [1, 0]
        del h5["dependencies/id"]
        _strings(h5["dependencies"], "id", ["traces-1"])
    with pytest.raises(ArtifactContractError, match="contiguous"):
        TaskIndex.read(result_path)

    path = _write_index(result_path)
    with h5py.File(path, "a") as h5:
        h5["artifacts/dependency_offset"][...] = [0, 1]
        h5["artifacts/dependency_count"][...] = [1, 0]
        del h5["dependencies/id"]
        _strings(h5["dependencies"], "id", ["missing"])
    with pytest.raises(ArtifactContractError, match="unresolved"):
        TaskIndex.read(result_path)


@pytest.mark.parametrize(
    ("offsets", "counts", "match"),
    [
        ([1, 1], [1, 1], "contiguous"),
        ([0, 2], [1, 1], "contiguous"),
        ([0, 2], [2, 1], "exceeds"),
        ([0, 0], [0, 1], "do not cover"),
    ],
)
def test_task_index_rejects_invalid_artifact_slices(tmp_path, offsets, counts, match):
    path = _write_index(tmp_path / "results")
    with h5py.File(path, "a") as h5:
        h5["tasks/artifact_offset"][...] = offsets
        h5["tasks/artifact_count"][...] = counts

    with pytest.raises(ArtifactContractError, match=match):
        TaskIndex.read(tmp_path / "results")


def test_task_index_rejects_duplicate_tasks_and_nonportable_paths(tmp_path):
    result_path = tmp_path / "results"
    path = _write_index(result_path)
    with h5py.File(path, "a") as h5:
        h5["tasks/task_id"][...] = [1, 1]
    with pytest.raises(ArtifactContractError, match="unique"):
        TaskIndex.read(result_path)

    path = _write_index(result_path)
    with h5py.File(path, "a") as h5:
        h5["artifacts/path"][0] = "../escape.h5"
    with pytest.raises(ArtifactContractError, match="artifact.path"):
        TaskIndex.read(result_path)


def test_task_index_rejects_duplicate_artifact_keys_within_a_task(tmp_path):
    result_path = tmp_path / "results"
    path = _write_index(result_path)
    with h5py.File(path, "a") as h5:
        h5["tasks/artifact_count"][...] = [2, 0]
        h5["tasks/artifact_offset"][...] = [0, 2]
        h5["artifacts/id"][1] = "traces-1"

    with pytest.raises(ArtifactContractError, match="unique within task 1"):
        TaskIndex.read(result_path)


def test_catalog_falls_back_only_to_known_results_when_index_is_missing_or_stale(
    tmp_path, monkeypatch
):
    result_path = tmp_path / "results"
    first = _write_task_result(result_path, 1)

    def reject_scan(*_args, **_kwargs):
        raise AssertionError("task index fallback must not scan the result tree")

    monkeypatch.setattr(type(result_path), "glob", reject_scan)
    monkeypatch.setattr(type(result_path), "rglob", reject_scan)

    missing = load_task_catalog(result_path, tasks=(1, 2))
    assert isinstance(missing, ArtifactCatalog)
    assert list(missing.results) == [1]

    index_path = _write_index(result_path)
    future = index_path.stat().st_mtime_ns + 1_000_000
    os.utime(first, ns=(future, future))
    stale = load_task_catalog(result_path, tasks=(1, 2))
    assert isinstance(stale, ArtifactCatalog)
    assert list(stale.results) == [1]


def test_catalog_uses_index_when_it_is_newer_than_known_task_results(tmp_path):
    result_path = tmp_path / "results"
    _write_task_result(result_path, 1)
    index_path = _write_index(result_path)
    newer = task_result_path(result_path, 1).stat().st_mtime_ns + 1_000_000
    os.utime(index_path, ns=(newer, newer))

    catalog = load_task_catalog(result_path, tasks=(1, 2))

    assert isinstance(catalog, TaskIndex)
    assert [item.id for item in catalog.artifacts_for_task(2)] == ["traces-2"]


def test_catalog_falls_back_from_malformed_index_but_direct_read_is_strict(tmp_path):
    result_path = tmp_path / "results"
    _write_task_result(result_path, 1)
    path = _write_index(result_path)
    with h5py.File(path, "a") as h5:
        h5.attrs["schema"] = "unsupported-index"

    with pytest.raises(ArtifactContractError, match="unsupported"):
        TaskIndex.read(result_path)
    catalog = load_task_catalog(result_path, tasks=(1, 2))

    assert isinstance(catalog, ArtifactCatalog)
    assert list(catalog.results) == [1]


def test_catalog_rejects_nonpositive_requested_task(tmp_path):
    _write_index(tmp_path / "results")

    with pytest.raises(ArtifactContractError, match="one-based"):
        load_task_catalog(tmp_path / "results", tasks=(0,))
