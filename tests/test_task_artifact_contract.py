from __future__ import annotations

import json
from copy import deepcopy

import pytest

from frequensolve.simulation.artifact_contract import (
    ArtifactCatalog,
    ArtifactContractError,
    ArtifactRequest,
    TaskResult,
    task_result_path,
)


def _fingerprint(digit: str) -> str:
    return f"sha256:{digit * 64}"


def _task_result(*, task: int = 1, real: float = 2.5, imag: float = 0.05):
    return {
        "schema": "fs-task-result-2",
        "partition": {
            "task": task,
            "task_count": 2,
            "frequency": {"real": real, "imag": imag},
        },
        "fingerprints": {
            "job": _fingerprint("1"),
            "simulation": _fingerprint("2"),
            "outputs": _fingerprint("3"),
        },
        "status": {"state": "success", "code": 0},
        "solver": {"convergence": {"iterations": 12, "residual": 1.0e-6}},
        "timings": {"assembly": 0.25, "solve": 1.5},
        "artifacts": [
            {
                "id": "traces",
                "role": "simulated_traces",
                "representation": "hdf5_shard",
                "schema": "fs_trace_payload_shard_v1",
                "path": f"traces/opaque-{task}-{imag}.h5",
                "retention": "durable",
                "generation": "1",
                "bytes": 128,
            }
        ],
    }


def _write_result(result_path, payload):
    path = task_result_path(result_path, payload["partition"]["task"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_task_result_reads_exact_partition_and_opaque_artifact_path(tmp_path):
    result_path = tmp_path / "results"
    path = _write_result(result_path, _task_result())

    result = TaskResult.read(path, result_path=result_path)

    assert result.partition.frequency == complex(2.5, 0.05)
    assert result.successful
    assert result.artifacts[0].relative_path == "traces/opaque-1-0.05.h5"
    assert (
        result.artifacts[0].path == (result_path / "traces/opaque-1-0.05.h5").resolve()
    )
    assert result.artifacts[0].to_fs() == _task_result()["artifacts"][0]
    assert result.to_fs() == _task_result()


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/escape.h5",
        "../escape.h5",
        "traces/../../escape.h5",
        r"C:\\escape.h5",
        "C:escape.h5",
        r"traces\\escape.h5",
        "./traces/part.h5",
        "traces//part.h5",
        "traces/",
    ],
)
def test_task_result_rejects_nonportable_or_escaping_artifact_paths(tmp_path, path):
    payload = _task_result()
    payload["artifacts"][0]["path"] = path
    result_path = tmp_path / "results"
    result_file = _write_result(result_path, payload)

    with pytest.raises(ArtifactContractError, match="artifact.path"):
        TaskResult.read(result_file, result_path=result_path)


def test_task_result_rejects_legacy_contract(tmp_path):
    payload = _task_result()
    payload["schema"] = "fs-run-1"
    result_path = tmp_path / "results"
    result_file = _write_result(result_path, payload)

    with pytest.raises(ArtifactContractError, match="unsupported"):
        TaskResult.read(result_file, result_path=result_path)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_task_result_requires_finite_imaginary_frequency(tmp_path, value):
    payload = _task_result()
    if value is None:
        del payload["partition"]["frequency"]["imag"]
    else:
        payload["partition"]["frequency"]["imag"] = value
    result_path = tmp_path / "results"
    result_file = _write_result(result_path, payload)

    with pytest.raises(ArtifactContractError, match="frequency.imag"):
        TaskResult.read(result_file, result_path=result_path)


def test_task_result_rejects_duplicate_artifact_identity(tmp_path):
    payload = _task_result()
    payload["artifacts"].append(deepcopy(payload["artifacts"][0]))
    result_path = tmp_path / "results"
    result_file = _write_result(result_path, payload)

    with pytest.raises(ArtifactContractError, match="unique"):
        TaskResult.read(result_file, result_path=result_path)


def test_task_result_preserves_resolved_dependencies(tmp_path):
    payload = _task_result()
    payload["artifacts"].append(
        {
            "id": "visualization",
            "role": "visualization",
            "representation": "xmf",
            "schema": "xdmf-2.1",
            "path": "visualization/model.xmf",
            "retention": "durable",
            "bytes": 42,
            "dependencies": ["traces"],
        }
    )
    result_path = tmp_path / "results"
    result_file = _write_result(result_path, payload)

    result = TaskResult.read(result_file, result_path=result_path)

    assert result.artifacts[1].dependencies == ("traces",)
    assert result.to_fs()["artifacts"][1]["dependencies"] == ["traces"]


def test_task_result_rejects_unresolved_dependencies(tmp_path):
    payload = _task_result()
    payload["artifacts"][0]["dependencies"] = ["missing"]
    result_path = tmp_path / "results"
    result_file = _write_result(result_path, payload)

    with pytest.raises(ArtifactContractError, match="resolve"):
        TaskResult.read(result_file, result_path=result_path)


def test_task_result_rejects_ambiguous_dependencies(tmp_path):
    payload = _task_result()
    alternate = deepcopy(payload["artifacts"][0])
    alternate["representation"] = "alternate"
    payload["artifacts"].append(alternate)
    payload["artifacts"][0]["dependencies"] = ["traces"]
    result_path = tmp_path / "results"
    result_file = _write_result(result_path, payload)

    with pytest.raises(ArtifactContractError, match="uniquely"):
        TaskResult.read(result_file, result_path=result_path)


def test_task_result_rejects_uncommitted_artifact_state(tmp_path):
    payload = _task_result()
    payload["artifacts"][0]["state"] = "writing"
    result_path = tmp_path / "results"
    result_file = _write_result(result_path, payload)

    with pytest.raises(ArtifactContractError, match="unsupported fields.*state"):
        TaskResult.read(result_file, result_path=result_path)


def test_artifact_catalog_uses_only_known_task_result_paths(tmp_path, monkeypatch):
    result_path = tmp_path / "results"
    _write_result(result_path, _task_result(task=1, imag=0.2))
    _write_result(result_path, _task_result(task=2, imag=0.0))

    def reject_discovery(*_args, **_kwargs):
        raise AssertionError("artifact catalog must not recursively discover files")

    monkeypatch.setattr(type(result_path), "rglob", reject_discovery)
    monkeypatch.setattr(type(result_path), "glob", reject_discovery)

    catalog = ArtifactCatalog.read_task_results(result_path, tasks=[1, 2, 3])

    assert list(catalog.results) == [1, 2]
    assert catalog.results[1].partition.frequency == complex(2.5, 0.2)
    assert catalog.results[2].partition.frequency == complex(2.5, 0.0)
    assert catalog.query(task=1, role="simulated_traces") == [
        catalog.results[1].artifacts[0]
    ]
    assert catalog.query(task=3) == []


def test_catalog_rejects_task_file_with_mismatched_partition(tmp_path):
    result_path = tmp_path / "results"
    payload = _task_result(task=2)
    path = task_result_path(result_path, 1)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactContractError, match="expected 1"):
        ArtifactCatalog.read_task_results(result_path, tasks=[1])


def test_typed_artifact_request_uses_representation_preference(tmp_path):
    payload = _task_result()
    payload["artifacts"].insert(
        0,
        {
            **payload["artifacts"][0],
            "representation": "collection_manifest",
            "path": "traces/manifest.json",
            "bytes": 64,
        },
    )
    result_path = tmp_path / "results"
    _write_result(result_path, payload)
    catalog = ArtifactCatalog.read_task_results(result_path, tasks=[1])

    selected = catalog.select(
        ArtifactRequest(
            role="simulated_traces",
            representations=("hdf5_shard", "collection_manifest"),
            retention="durable",
        ),
        task=1,
    )

    assert [item.representation for item in selected] == [
        "hdf5_shard",
        "collection_manifest",
    ]
    assert (
        catalog.require_one(
            ArtifactRequest(
                role="simulated_traces",
                representations=("hdf5_shard", "collection_manifest"),
            ),
            task=1,
        ).representation
        == "hdf5_shard"
    )


def test_typed_artifact_request_requires_unique_result(tmp_path):
    result_path = tmp_path / "results"
    _write_result(result_path, _task_result())
    catalog = ArtifactCatalog.read_task_results(result_path, tasks=[1])

    artifact = catalog.require_one(
        ArtifactRequest(
            id="traces",
            role="simulated_traces",
            representations=("hdf5_shard",),
        ),
        task=1,
    )

    assert artifact.relative_path == "traces/opaque-1-0.05.h5"
    with pytest.raises(ArtifactContractError, match="found 0"):
        catalog.require_one(ArtifactRequest(role="image"), task=1)


def test_typed_artifact_request_validates_retention_and_preferences():
    with pytest.raises(ArtifactContractError, match="retention"):
        ArtifactRequest(role="image", retention="forever")
    with pytest.raises(ArtifactContractError, match="representations must be unique"):
        ArtifactRequest(role="image", representations=("hdf5", "hdf5"))
