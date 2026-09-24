"""Receiver trace resolution through the strict Sauce artifact contract."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest

from frequensolve.simulation.artifact_contract import ArtifactContractError
from frequensolve.simulation.jobs.artifacts import TraceManifest, TraceOutputSpec

_SHA256 = "0" * 64


def _job(tmp_path: Path, frequencies: list[complex | float]) -> SimpleNamespace:
    result_path = tmp_path / "results"
    simulation = SimpleNamespace(name="simulation", _file=tmp_path / "simulation.json")
    return SimpleNamespace(
        simulation=simulation,
        project_path=tmp_path,
        _result_path=result_path,
        trace_outputs=TraceOutputSpec(
            path=result_path / "configured-trace-directory",
            frequencies=frequencies,
            groups=["hydrophones"],
        ),
        _file=None,
    )


def _write_task_result(
    result_path: Path,
    task: int,
    frequency: complex | float,
    artifacts: list[dict[str, object]],
) -> Path:
    frequency = complex(frequency)
    path = result_path / "_fs_run" / "tasks" / f"task_{task:06d}" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "fs-task-result-2",
                "partition": {
                    "task": task,
                    "frequency": {
                        "real": frequency.real,
                        "imag": frequency.imag,
                    },
                },
                "fingerprints": {
                    "job": _SHA256,
                    "simulation": _SHA256,
                    "outputs": _SHA256,
                },
                "status": {"state": "success", "code": 0},
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return path


def _trace_artifact(path: str) -> dict[str, object]:
    return {
        "id": "traces",
        "role": "simulated_traces",
        "representation": "hdf5_shard",
        "schema": "fs-trace-shard-2",
        "path": path,
        "retention": "durable",
        "bytes": 12,
    }


def test_receiver_manifest_uses_opaque_producer_paths(tmp_path):
    job = _job(tmp_path, [2.0, 4.0])
    relative_paths = [
        "payloads/7f/opaque-a.data",
        "elsewhere/generation-12/opaque-b.bin",
    ]
    for task, (frequency, relative_path) in enumerate(
        zip(job.trace_outputs.frequencies, relative_paths), start=1
    ):
        _write_task_result(
            job._result_path,
            task,
            frequency,
            [_trace_artifact(relative_path)],
        )

    manifest = TraceManifest.from_job(job)

    assert manifest.files == [job._result_path / path for path in relative_paths]
    assert [artifact.id for artifact in manifest.artifacts] == [
        "traces",
        "traces",
    ]
    assert [artifact.representation for artifact in manifest.artifacts] == [
        "hdf5_shard",
        "hdf5_shard",
    ]
    assert manifest.artifact_contract == "fs-task-result-2"


def test_receiver_manifest_distinguishes_equal_real_frequency_laplace_tasks(tmp_path):
    frequencies = [5.0 - 0.2j, 5.0 - 0.05j]
    job = _job(tmp_path, frequencies)
    for task, frequency in enumerate(frequencies, start=1):
        _write_task_result(
            job._result_path,
            task,
            frequency,
            [_trace_artifact(f"opaque/task-{task}-laplace.h5")],
        )

    manifest = TraceManifest.from_job(job)

    assert manifest.frequencies == {1: 5.0, 2: 5.0}
    assert manifest.laplace == {1: -0.2, 2: -0.05}
    assert manifest.files == [
        job._result_path / "opaque/task-1-laplace.h5",
        job._result_path / "opaque/task-2-laplace.h5",
    ]


def test_receiver_manifest_rejects_missing_task_result(tmp_path):
    job = _job(tmp_path, [2.0, 4.0])
    _write_task_result(
        job._result_path,
        1,
        2.0,
        [_trace_artifact("opaque/only-first-task.h5")],
    )

    with pytest.raises(ArtifactContractError, match="task 2 has no committed result"):
        TraceManifest.from_job(job)


def test_unstarted_receiver_manifest_has_no_inferred_files(tmp_path):
    job = _job(tmp_path, [2.0, 4.0])

    manifest = TraceManifest.from_job(job)

    assert manifest.files == []
    assert manifest.artifacts == []
    assert not manifest.complete


def test_receiver_manifest_rejects_ambiguous_trace_records(tmp_path):
    job = _job(tmp_path, [2.0])
    _write_task_result(
        job._result_path,
        1,
        2.0,
        [
            _trace_artifact("opaque/first.h5"),
            _trace_artifact("opaque/second.h5"),
        ],
    )

    with pytest.raises(ArtifactContractError, match="must be unique within a task"):
        TraceManifest.from_job(job)


def test_receiver_resolution_never_scans_or_opens_payload_hdf5(tmp_path, monkeypatch):
    job = _job(tmp_path, [3.0])
    _write_task_result(
        job._result_path,
        1,
        3.0,
        [_trace_artifact("opaque/not-even-an-hdf5-name.payload")],
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("receiver artifact resolution must not inspect payloads")

    monkeypatch.setattr(Path, "glob", fail)
    monkeypatch.setattr(Path, "rglob", fail)
    monkeypatch.setattr(h5py, "File", fail)

    manifest = TraceManifest.from_job(job)

    assert manifest.files == [job._result_path / "opaque/not-even-an-hdf5-name.payload"]


def test_v2_receiver_manifest_does_not_fall_back_to_legacy_receiver_name(tmp_path):
    job = _job(tmp_path, [3.0])
    exact = "opaque/traces_generation_4.h5"
    _write_task_result(
        job._result_path,
        1,
        3.0,
        [_trace_artifact(exact)],
    )
    legacy = job._result_path / "opaque/receivers_generation_4.h5"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.touch()

    manifest = TraceManifest.from_job(job)

    assert manifest.files == [job._result_path / exact]
    assert manifest.existing_files == []
    assert not manifest.complete


def test_trace_manifest_accepts_validated_downloaded_identity(tmp_path):
    job = _job(tmp_path, [4.0])
    remote = {key: _SHA256 for key in ("job", "simulation", "outputs")}
    job._task_reuse_fingerprints = lambda: {key: "sha256:" + "1" * 64 for key in remote}
    job.downloaded_task_fingerprints = lambda: [remote]
    _write_task_result(job._result_path, 1, 4.0, [_trace_artifact("payloads/trace.h5")])
    assert TraceManifest.from_job(job).files == [job._result_path / "payloads/trace.h5"]
    job.downloaded_task_fingerprints = lambda: []
    assert TraceManifest.from_job(job).files == []
