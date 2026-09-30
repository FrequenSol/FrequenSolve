# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from frequensolve import SeismicSimulation
from frequensolve.mesh._root_patch import RootPatchDescriptor
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.local.site import (
    _attach_task_result,
    _read_task_result,
)
from frequensolve.simulation.jobs._patches import PatchPreparationJob, _validate_request
from frequensolve.simulation.jobs.base import BaseJob
from tests.test_imaging_jobs import _assert_valid
from tests.test_operation_artifact_contract import _operation, _write


def test_descriptor_is_immutable_and_roundtrips():
    descriptor = RootPatchDescriptor("abcdef0123456789", [4, 1, 7])
    assert descriptor.roots == (1, 4, 7)
    with pytest.raises(FrozenInstanceError):
        descriptor.roots = (2,)
    mesh = MeshManager(file="parent.gmp", format="gmp", root_patch=descriptor)
    payload = mesh.to_fs()
    restored = MeshManager.from_fs(payload)
    assert restored.root_patch == descriptor
    assert restored.to_fs()["root_patch"] == payload["root_patch"]
    assert restored.to_fs()["file"] == payload["file"]


@pytest.mark.parametrize("roots", [[], [0], [1, 1], [True], [1.5]])
def test_descriptor_rejects_invalid_roots(roots):
    with pytest.raises(ValueError):
        RootPatchDescriptor("0" * 16, roots)


def test_descriptor_requires_parent_file():
    mesh = MeshManager(root_patch=RootPatchDescriptor("0" * 16, (1,)))
    with pytest.raises(ValueError, match="parent mesh file"):
        mesh.to_fs()


@pytest.mark.parametrize(
    "updates",
    [
        {"padding": None},
        {"padding": -1},
        {"padding": float("nan")},
        {"roots": [1, 1]},
        {"lower": [0, 0], "upper": [1, 1]},
    ],
)
def test_preparation_rejects_ambiguous_or_invalid_requests(updates):
    request = {
        "units": "m",
        "patches": [{"name": "one", "roots": [1], "padding": 0, **updates}],
    }
    with pytest.raises(ValueError):
        _validate_request(request, 2)


def test_preparation_rejects_duplicate_names_and_dimension_mismatch():
    patch = {"name": "one", "lower": [0, 0], "upper": [1, 2], "padding": 0}
    with pytest.raises(ValueError, match="unique"):
        _validate_request({"units": "m", "patches": [patch, patch]}, 2)
    with pytest.raises(ValueError, match="3 finite"):
        _validate_request({"units": "m", "patches": [patch]}, 3)


@pytest.mark.parametrize(
    "points",
    [
        None,
        [None],
        [{"kind": "source", "id": True, "coordinates": [0, 0]}],
        [{"kind": "receiver", "id": 2, "coordinates": [0, 0]}],
        [{"kind": "source", "id": 1, "group": "p", "coordinates": [0, 0]}],
        [{"kind": "source", "id": 1, "coordinates": [0, float("nan")]}],
        [{"kind": "source", "id": 1, "coordinates": [0, 0, 0]}],
        [{"kind": "source", "id": 1, "coordinates": [0, 0]}] * 2,
    ],
)
def test_preparation_rejects_invalid_acquisition_points(points):
    request = {
        "units": "m",
        "patches": [{"name": "one", "roots": [1], "padding": 0, "points": points}],
    }
    with pytest.raises(ValueError):
        _validate_request(request, 2)


def test_job_roundtrip_has_one_task_and_tracks_entire_stage(tmp_path):
    simulation = SeismicSimulation(
        name="parent", physics="acoustic", dimension=2, project_path=tmp_path
    )
    request = {"units": "m", "patches": [{"name": "one", "roots": [1], "padding": 100}]}
    job = PatchPreparationJob("prepare", simulation, [3, 7.5], request)
    saved = BaseJob.load(job.save())
    assert isinstance(saved, PatchPreparationJob)
    assert saved.n_tasks == 1
    assert not saved.supports_trace_packing
    assert saved.request == job.request
    assert saved.to_fs()["f_list"] == job.to_fs()["f_list"]
    _assert_valid(saved.to_fs())
    fingerprint = job.task_fingerprint(1)
    job.f_list[1] = 8
    assert job.task_fingerprint(1) != fingerprint
    job.request["patches"][0]["padding"] = 200
    assert saved.fingerprint() != job.fingerprint()
    assert saved.request["patches"][0]["padding"] == 100


def test_local_worker_reads_preparation_operation(tmp_path):
    payload = _operation("patch_prepare", artifacts=[])
    result_file = tmp_path / "_fs_run/operations/patch_prepare/result.json"
    _write(result_file, payload)
    job_file = tmp_path / "job.json"
    job_file.write_text(
        json.dumps({"result_path": str(tmp_path), "workflow": "patch_prepare"})
    )
    operation = _read_task_result(job_file, 0)
    row = {}
    _attach_task_result(row, operation)
    assert row["operation"]["name"] == "patch_prepare"
    assert "partition" not in row
    with pytest.raises(ValueError, match="exactly one"):
        _read_task_result(job_file, 1)


def test_job_rejects_child_and_nonfinite_frequencies():
    simulation = SimpleNamespace(dimension=2, mesh=SimpleNamespace(root_patch=None))
    with pytest.raises(ValueError, match="finite"):
        PatchPreparationJob("p", simulation, [3 + float("nan") * 1j], {"units": "m"})
    simulation.mesh.root_patch = RootPatchDescriptor("0" * 16, (1,))
    with pytest.raises(ValueError, match="parent"):
        PatchPreparationJob("p", simulation, [3], {"units": "m"})


def test_material_artifact_content_invalidates_preparation(tmp_path):
    from frequensolve.model.layered import LayeredModel
    from frequensolve.model.parameterization import MeshPropertySpace

    simulation = SeismicSimulation(
        name="parent", physics="acoustic", dimension=2, project_path=tmp_path
    )
    simulation.model = LayeredModel(dimension=2, x_limits=[0, 1])
    simulation.model.add_surface(name="top", depth=0)
    simulation.model.add_layer(
        name="fluid", physics="acoustic", properties={"Vp": 1.5, "Rho": 1}
    )
    simulation.model.add_surface(name="bottom", depth=1)
    simulation.model.property_spaces["vp"] = MeshPropertySpace(
        artifact="parent.h5", frequency=3, epw=1
    )
    request = {"units": "m", "patches": [{"name": "one", "roots": [1], "padding": 0}]}
    selection = PatchPreparationJob("select", simulation, [3], request)
    inventory = PatchPreparationJob("inventory", simulation, [3], {"units": "m"})
    assert inventory._input_fingerprint_payload() == {}
    with pytest.raises(FileNotFoundError, match="parent.h5"):
        selection._input_fingerprint_payload()
    artifact = tmp_path / "parent.h5"
    artifact.write_bytes(b"basis-one")
    simulation.save()
    first = selection.fingerprint()
    restored = BaseJob.load(selection.save())
    assert restored.fingerprint() == first
    inventory_first = inventory.fingerprint()
    artifact.write_bytes(b"basis-two")  # Same path and byte count.
    assert selection.fingerprint() != first
    assert restored.fingerprint() == selection.fingerprint()
    assert inventory.fingerprint() == inventory_first
