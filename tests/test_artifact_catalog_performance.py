from __future__ import annotations

import json
import time

from frequensolve.simulation.artifact_contract import (
    ArtifactCatalog,
    task_result_path,
)


def _fingerprint(digit: str) -> str:
    return f"sha256:{digit * 64}"


def _write_task_results(result_path, count: int) -> None:
    for task in range(1, count + 1):
        path = task_result_path(result_path, task)
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "fs-task-result-2",
                    "partition": {
                        "task": task,
                        "task_count": count,
                        "frequency": {"real": task / 10.0, "imag": 0.05},
                    },
                    "fingerprints": {
                        "job": _fingerprint("1"),
                        "simulation": _fingerprint("2"),
                        "outputs": _fingerprint("3"),
                    },
                    "status": {"state": "success", "code": 0},
                    "artifacts": [
                        {
                            "id": "receivers",
                            "role": "simulated_traces",
                            "representation": "hdf5_shard",
                            "schema": "fs_trace_payload_shard_v1",
                            "path": f"opaque/task-{task}.h5",
                            "retention": "durable",
                            "bytes": 128,
                        }
                    ],
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )


def _best_catalog_time(result_path, count: int) -> float:
    measurements = []
    tasks = range(1, count + 1)
    for _ in range(3):
        started = time.perf_counter()
        catalog = ArtifactCatalog.read_task_results(result_path, tasks=tasks)
        measurements.append(time.perf_counter() - started)
        assert len(catalog.results) == count
    return min(measurements)


def test_warm_thousand_task_catalog_is_linear_and_subsecond(tmp_path):
    result_path = tmp_path / "results"
    _write_task_results(result_path, 1000)

    ArtifactCatalog.read_task_results(result_path, tasks=range(1, 1001))
    elapsed_500 = _best_catalog_time(result_path, 500)
    elapsed_1000 = _best_catalog_time(result_path, 1000)

    assert elapsed_1000 < 1.0
    assert elapsed_1000 < 2.5 * elapsed_500
