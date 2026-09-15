import json

import pytest

from frequensolve.simulation.artifact_catalog import load_artifact_catalog
from frequensolve.simulation.artifact_contract import (
    ArtifactContractError,
    ArtifactRequest,
    OperationResult,
    load_operation_result,
    operation_result_path,
    task_result_path,
)
from frequensolve.simulation.artifact_transfer import select_transfer_artifacts


def _artifact(name, *, role="image", dependencies=()):
    result = {
        "id": name,
        "role": role,
        "schema": "test-1",
        "representation": "hdf5",
        "path": f"payloads/{name}.h5",
        "retention": "durable",
        "bytes": 0,
    }
    if dependencies:
        result["dependencies"] = list(dependencies)
    return result


def _operation(name="smooth", *, operation=None, artifacts=None):
    return {
        "schema": "fs-operation-result-1",
        "operation": {
            "name": name if operation is None else operation,
            "generation": "generation-7",
        },
        "fingerprints": {
            "job": "a" * 64,
            "simulation": "b" * 64,
            "outputs": "c" * 64,
        },
        "status": {"state": "success", "code": 0},
        "timings": {"assembly": 0.5},
        "artifacts": [_artifact("gradient")] if artifacts is None else artifacts,
    }


def _task():
    return {
        "schema": "fs-task-result-2",
        "partition": {
            "task": 1,
            "frequency": {"real": 2.0, "imag": 0.1},
        },
        "fingerprints": {
            "job": "a" * 64,
            "simulation": "b" * 64,
            "outputs": "c" * 64,
        },
        "status": {"state": "success", "code": 0},
        "artifacts": [_artifact("trace", role="trace")],
    }


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "name", ["pack", "smooth", "raytrace", "size", "validate", "init"]
)
def test_operation_result_loads_only_canonical_fixed_names(tmp_path, name):
    path = operation_result_path(tmp_path, name)
    _write(path, _operation(name))

    result = load_operation_result(tmp_path, name)

    assert result.name == name
    assert result.generation == "generation-7"
    assert result.successful
    assert result.to_fs() == _operation(name)


def test_operation_result_rejects_alias_and_mismatched_fixed_path(tmp_path):
    with pytest.raises(ArtifactContractError, match="canonical operations"):
        operation_result_path(tmp_path, "ray_trace")

    path = operation_result_path(tmp_path, "smooth")
    _write(path, _operation("smooth", operation="pack"))
    with pytest.raises(ArtifactContractError, match="requested 'smooth'"):
        OperationResult.read(path, result_path=tmp_path, workflow="smooth")


def test_operation_result_rejects_unresolved_dependency(tmp_path):
    path = operation_result_path(tmp_path, "smooth")
    _write(
        path,
        _operation(artifacts=[_artifact("view", dependencies=("missing-data",))]),
    )

    with pytest.raises(ArtifactContractError, match="resolve uniquely"):
        load_operation_result(tmp_path, "smooth")


def test_combined_catalog_loads_only_explicit_fixed_operation_paths(tmp_path):
    _write(task_result_path(tmp_path, 1), _task())
    _write(operation_result_path(tmp_path, "smooth"), _operation("smooth"))
    _write(operation_result_path(tmp_path, "pack"), _operation("pack"))
    _write(
        tmp_path / "unrelated" / "operations" / "validate" / "result.json",
        _operation("validate"),
    )

    catalog = load_artifact_catalog(
        tmp_path,
        tasks=[1],
        operations=["smooth", "validate"],
    )

    assert set(catalog.operations) == {"smooth"}
    assert [item.id for item in catalog.query()] == ["trace", "gradient"]
    assert catalog.query(operation="pack") == []
    assert (
        catalog.require_one(ArtifactRequest(role="image"), operation="smooth").id
        == "gradient"
    )


def test_operation_dependency_closure_stays_within_operation(tmp_path):
    _write(task_result_path(tmp_path, 1), _task())
    artifacts = [
        _artifact("view", role="visualization", dependencies=("view-data",)),
        _artifact("view-data", role="visualization_data"),
    ]
    _write(
        operation_result_path(tmp_path, "smooth"),
        _operation("smooth", artifacts=artifacts),
    )
    catalog = load_artifact_catalog(tmp_path, tasks=[1], operations=["smooth"])

    selected = select_transfer_artifacts(
        catalog,
        requests=[ArtifactRequest(role="visualization")],
        include_defaults=False,
    )

    assert [item.id for item in selected] == ["view", "view-data"]


def test_missing_operation_result_does_not_trigger_discovery(tmp_path):
    _write(task_result_path(tmp_path, 1), _task())
    _write(
        tmp_path / "somewhere" / "smooth" / "result.json",
        _operation("smooth"),
    )

    catalog = load_artifact_catalog(tmp_path, tasks=[1], operations=["smooth"])

    assert catalog.operations == {}
