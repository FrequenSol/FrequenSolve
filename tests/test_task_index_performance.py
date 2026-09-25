from __future__ import annotations

import time
import tracemalloc

import h5py
import numpy as np

from frequensolve.simulation.artifact_contract import ArtifactRequest
from frequensolve.simulation.task_index import (
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
_TRACE_REQUEST = ArtifactRequest(
    role="simulated_traces",
    representations=("hdf5_shard",),
    retention="durable",
)


def _strings(group, name, values):
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=_STRING)


def _write_index(result_path, count):
    path = task_index_path(result_path)
    path.parent.mkdir(parents=True)
    zeros = np.zeros(count, dtype=np.float64)
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "fs-task-index-1"
        tasks = h5.create_group("tasks")
        artifacts = h5.create_group("artifacts")
        dependencies = h5.create_group("dependencies")
        tasks.create_dataset("task_id", data=np.arange(1, count + 1, dtype=np.int64))
        _strings(tasks, "status", ["success"] * count)
        tasks.create_dataset("frequency_real", data=zeros)
        tasks.create_dataset("frequency_imag", data=zeros)
        digest = [f"sha256:{'1' * 64}"] * count
        _strings(tasks, "fingerprint_job", digest)
        _strings(tasks, "fingerprint_simulation", digest)
        _strings(tasks, "fingerprint_outputs", digest)
        tasks.create_dataset("artifact_offset", data=np.arange(count, dtype=np.int64))
        tasks.create_dataset("artifact_count", data=np.ones(count, dtype=np.int64))
        tasks.create_dataset("iterations", data=np.full(count, -1, dtype=np.int64))
        tasks.create_dataset("residual", data=np.full(count, np.nan))
        for name in _TIMINGS:
            tasks.create_dataset(name, data=np.full(count, np.nan))
        _strings(artifacts, "id", [f"trace-{task}" for task in range(count)])
        _strings(artifacts, "role", ["simulated_traces"] * count)
        _strings(artifacts, "schema", ["trace-shard-1"] * count)
        _strings(artifacts, "representation", ["hdf5_shard"] * count)
        _strings(artifacts, "path", [f"opaque/{task}.h5" for task in range(count)])
        _strings(artifacts, "retention", ["durable"] * count)
        _strings(artifacts, "generation", [""] * count)
        artifacts.create_dataset("bytes", data=np.zeros(count, dtype=np.int64))
        artifacts.create_dataset(
            "dependency_offset", data=np.zeros(count, dtype=np.int64)
        )
        artifacts.create_dataset(
            "dependency_count", data=np.zeros(count, dtype=np.int64)
        )
        _strings(dependencies, "id", [])
    payloads = result_path / "opaque"
    payloads.mkdir()
    for task in range(count):
        (payloads / f"{task}.h5").touch()
    return path


def _load_and_check(result_path):
    catalog = load_task_catalog(result_path, tasks=range(1, 1001))
    assert isinstance(catalog, TaskIndex)
    assert (
        sum(
            catalog.is_task_current(task, required_artifact=_TRACE_REQUEST)
            for task in range(1, 1001)
        )
        == 1000
    )
    return catalog


def _best_currentness_time(result_path):
    measurements = []
    for _ in range(3):
        started = time.perf_counter()
        index = _load_and_check(result_path)
        measurements.append(time.perf_counter() - started)
        assert len(index.tasks) == 1000
    return min(measurements)


def test_thousand_task_index_time_memory_and_metadata_size(tmp_path, record_property):
    result_path = tmp_path / "results"
    path = _write_index(result_path, 1000)

    elapsed = _best_currentness_time(result_path)
    tracemalloc.start()
    index = _load_and_check(result_path)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    record_property("task_index_elapsed_seconds", elapsed)
    record_property("task_index_peak_bytes", peak_bytes)
    record_property("task_index_file_bytes", path.stat().st_size)
    assert len(index.tasks) == 1000
    assert elapsed < 2.0
    assert path.stat().st_size < 10 * 1024 * 1024
