# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import json
import os
import time
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
from tests.test_operation_artifact_contract import _operation, _task, _write


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


@pytest.mark.parametrize("edge_samples", [None, 1, "true", [True]])
def test_preparation_edge_samples_must_be_boolean(edge_samples):
    with pytest.raises(ValueError, match="edge_samples"):
        _validate_request({"units": "m", "edge_samples": edge_samples}, 2)


def test_preparation_edge_samples_roundtrip_matches_contract(tmp_path):
    simulation = SeismicSimulation(
        name="parent", physics="acoustic", dimension=2, project_path=tmp_path
    )
    request = {"units": "m", "edge_samples": True}
    job = PatchPreparationJob("inventory", simulation, [3], request)
    assert job.request == request
    saved = BaseJob.load(job.save())
    assert saved.request == request
    _assert_valid(saved.to_fs())
    plain = PatchPreparationJob("plain", simulation, [3], {"units": "m"})
    assert plain.request == {"units": "m"}
    assert plain.fingerprint() != job.fingerprint()


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
    with pytest.raises(FileNotFoundError, match="Save the parent simulation"):
        selection._input_fingerprint_payload()
    artifact = tmp_path / "parent.h5"
    artifact.write_bytes(b"basis-one")
    simulation.save()
    # An unrecorded artifact is not trusted: a run must have resolved it.
    with pytest.raises(FileNotFoundError, match="Run a job that builds"):
        selection._input_fingerprint_payload()
    _record(tmp_path, "basis", [_material("vp", artifact, project=tmp_path)])
    first = selection.fingerprint()
    restored = BaseJob.load(selection.save())
    assert restored.fingerprint() == first
    inventory_first = inventory.fingerprint()
    artifact.write_bytes(b"basis-two")  # Same path and byte count.
    assert selection.fingerprint() != first
    assert restored.fingerprint() == selection.fingerprint()
    assert inventory.fingerprint() == inventory_first


def _material(
    space, path, *, project=None, key=None, frequency=5.0, identity="basis-1"
):
    record = {
        "id": f"material_space:{space}",
        "role": "material_space",
        "representation": "hdf5",
        "schema": "fs-property-space-1",
        "space": space,
        "path": str(path),
        "mesh_keyed": key is not None,
        "basis_identity": identity,
        "built": True,
        "retention": "durable",
        "bytes": 9,
    }
    if key is not None:
        record["mesh_key"] = key
        record["sizing_frequency"] = frequency
    if project is not None:
        record["project_relative_path"] = path.relative_to(project).as_posix()
    return record


def _record(project, job, records, *, simulation="b" * 64, state="success", op=None):
    results = project / "jobs" / "parent" / job / "results" / "_fs_run"
    if op is None:
        path, payload = results / "tasks" / "task_000001" / "result.json", _task()
    else:
        path, payload = results / "operations" / op / "result.json", _operation(op)
    payload["fingerprints"]["simulation"] = simulation
    payload["status"] = {"state": state, "code": 0 if state == "success" else 3}
    payload["artifacts"] = []
    payload["execution"] = {"material_artifacts": records}
    _write(path, payload)
    # Distinct, increasing modification times order the records newest last.
    stamp = time.time_ns() + len(list(project.glob("jobs/*/*"))) * 10**9
    os.utime(path, ns=(stamp, stamp))
    return path


def test_mesh_sizing_frequency_matches_the_solver_rule():
    from frequensolve.simulation.jobs._patches import mesh_sizing_frequency

    assert mesh_sizing_frequency([3, 5 + 1j], {}) == 5.0
    assert mesh_sizing_frequency([3, -6], {"Mesh": {"adapt": {}}}) == 6.0
    assert mesh_sizing_frequency([3], {"Mesh": {"adapt": {"f_low": 7}}}) == 7.0


def test_preparation_selects_the_recorded_keyed_artifact_exactly(tmp_path):
    from frequensolve.model.layered import LayeredModel
    from frequensolve.model.parameterization import MeshPropertySpace
    from frequensolve.simulation.artifact_contract import OperationResult
    from frequensolve.simulation.jobs._patches import recorded_material_artifact

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
    selection = PatchPreparationJob("select", simulation, [3, 5], request)
    with pytest.raises(FileNotFoundError, match="Save the parent simulation"):
        selection._input_fingerprint_payload()
    simulation.save()
    current = selection._sha256_file(selection._simulation_path())
    with pytest.raises(FileNotFoundError, match="sizing frequency 5"):
        selection._input_fingerprint_payload()

    def keyed(key, **kwargs):
        path = tmp_path / f"parent.{key * 16}.h5"
        path.write_bytes(key.encode() * 9)
        return path, _material("vp", path, project=tmp_path, key=key * 16, **kwargs)

    low, low_record = keyed("3", frequency=3.0, identity="f3")
    band, band_record = keyed("5", identity="f5")
    other, other_record = keyed("7", identity="other-simulation")
    failed, failed_record = keyed("9", identity="failed")
    _record(tmp_path, "low", [low_record], simulation=current)
    _record(tmp_path, "band", [band_record], simulation=current, op="init")
    # Newer, but another simulation file, a failed run, or another space.
    _record(tmp_path, "other", [other_record])
    _record(tmp_path, "failed", [failed_record], simulation=current, state="failed")
    _record(
        tmp_path,
        "rho",
        [_material("rho", other, key="7" * 16, identity="rho")],
        simulation=current,
    )
    payload = selection._input_fingerprint_payload()
    assert payload["vp"]["basis_identity"] == "f5"
    assert payload["vp"] == {
        **selection._path_content_fingerprint(band),
        "basis_identity": "f5",
    }
    # A job with a lower band reads the other keyed artifact.
    lower = PatchPreparationJob("lower", simulation, [3], request)
    assert lower._input_fingerprint_payload()["vp"]["basis_identity"] == "f3"
    first = selection.fingerprint()
    band.write_bytes(b"rebuilt-5")
    assert selection.fingerprint() != first

    # The backstop compares what patch_prepare recorded with the selection.
    def prepared(record):
        path = _record(tmp_path, "select", [record], simulation=current, op="init")
        return OperationResult.read(path, result_path=path.parents[3], workflow="init")

    selection._verify_recorded_materials(prepared({**band_record, "built": False}))
    with pytest.raises(RuntimeError, match="different material artifacts"):
        selection._verify_recorded_materials(prepared(low_record))

    # Records of the same path agree; a second keyed path for one band is ambiguous.
    _record(tmp_path, "again", [{**band_record, "built": False}], simulation=current)
    assert recorded_material_artifact(
        tmp_path,
        "vp",
        "parent.h5",
        simulation_fingerprint=current,
        sizing_frequency=5.0,
    ) == (band.resolve(), selection._material_selection()["vp"][1])
    clash, clash_record = keyed("c", identity="clash")
    _record(tmp_path, "clash", [clash_record], simulation=current)
    with pytest.raises(RuntimeError, match="several material artifacts"):
        selection._input_fingerprint_payload()
    clash.unlink()
    _record(tmp_path, "clash", [], simulation=current)
    band.unlink()  # A unique record whose file is gone is reported, not skipped.
    with pytest.raises(FileNotFoundError, match="missing"):
        selection._input_fingerprint_payload()
    band.write_bytes(b"rebuilt-5")

    # A relocated project resolves the portable project-relative path.
    moved = {
        **band_record,
        "path": f"/elsewhere/{band.name}",
        "basis_identity": "moved",
    }
    _record(tmp_path, "moved", [moved], simulation=current)
    assert selection._input_fingerprint_payload()["vp"]["basis_identity"] == "moved"

    # An existing declared path wins, as in Sauce, but still needs a record.
    literal = tmp_path / "parent.h5"
    literal.write_bytes(b"literal")
    with pytest.raises(FileNotFoundError, match="no run has recorded"):
        selection._input_fingerprint_payload()
    _record(tmp_path, "literal", [_material("vp", literal, identity="literal")])
    assert selection._input_fingerprint_payload()["vp"]["basis_identity"] == "literal"
