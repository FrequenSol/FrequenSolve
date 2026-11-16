# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import frequensolve as fs
from frequensolve.imaging._artifacts import ControlStateFile
from frequensolve.mesh._stage_snapshot import PatchStageSnapshot, _digest
from frequensolve.mesh.patches import PreparedPatchSet
from frequensolve.model.property import Property
from tests.test_root_patch_integration import _parent


def _fixture(tmp_path):
    simulation = _parent(tmp_path, 2, "acoustic")
    root = Path(simulation.project_path)
    with h5py.File(root / "reference.h5", "w") as h5:
        h5["data"] = np.arange(4.0)
    simulation.model.layers["fluid"].properties["Vp"] = Property.file(
        root / "reference.h5", dataset="data", units="km/s"
    )
    # Exercise the same file locator keys used by native material grids and named spaces.
    simulation.extra["stage_fixture"] = {
        "file": "reference.h5:/data",
        "artifact": "reference.h5",
    }
    result = root / "prepared"
    result.mkdir()
    (result / "parent.gmp").write_bytes(b"immutable parent geometry")
    geometry = {
        "parent_file": "parent.gmp",
        "parent_fingerprint": "1234567890ABCDEF",
        "dimension": 2,
        "patches": [
            {
                "name": "west",
                "material_coverage": [
                    {"space": "vp", "basis_identity": "sha256:" + "1" * 64}
                ],
            }
        ],
    }
    job = SimpleNamespace(
        _result_path=result,
        geometry_report=deepcopy(geometry),
        simulation=simulation,
        f_list=[3, 7.5],
    )
    prepared = PreparedPatchSet(
        geometry,
        [{"name": "west", "sources": [1]}],
        [job],
        fs.BoundaryCondition(conditions=["pml"], boundaries=["patch_cut"]),
    )
    state = ControlStateFile(
        {"model.vp": [0.1, 0.3]}, control_spaces={"model.vp": "sha256:" + "1" * 64}
    )
    return simulation, prepared, state


def _publish(tmp_path, **updates):
    simulation, prepared, state = _fixture(tmp_path)
    args = dict(name="stage-one", directory=tmp_path / "stage")
    args.update(updates)
    snapshot = prepared.freeze_stage(state, **args)
    return snapshot, simulation, prepared, state


@pytest.mark.parametrize(
    "band", [[], [[0, 0]], [[-3, 0]], [[3, float("inf")]], [[3, 0, 1]]]
)
def test_reader_rejects_invalid_stage_frequency_band(tmp_path, band):
    snapshot, *_ = _publish(tmp_path)
    payload = json.loads(snapshot.manifest.read_text())
    payload["frequencies"] = band
    snapshot.manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="frequency band"):
        PatchStageSnapshot.read(snapshot.manifest, identity=_digest(snapshot.manifest))


def test_stage_copies_inputs_and_survives_source_mutation_and_relocation(tmp_path):
    snapshot, simulation, prepared, state = _publish(tmp_path)
    original = simulation.to_fs()
    payload = json.loads(snapshot.simulation_file.read_text())
    material = payload["Model"]["subdomains"][0]["properties"]["vp"]
    assert "reference.h5" not in json.dumps(material)
    assert payload["stage_fixture"]["file"].endswith(":/data")
    pinned = snapshot.manifest.parent / payload["stage_fixture"]["artifact"]
    with h5py.File(pinned) as h5:
        np.testing.assert_array_equal(h5["data"], np.arange(4.0))
    with h5py.File(Path(simulation.project_path) / "reference.h5", "r+") as h5:
        h5["data"][:] = 9
    state.blocks["model.vp"][:] = 8
    (prepared.jobs[-1]._result_path / "parent.gmp").write_bytes(b"new parent")
    np.testing.assert_allclose(
        ControlStateFile.read(snapshot.control_state).blocks["model.vp"], [0.1, 0.3]
    )
    assert simulation.to_fs() == original
    moved = tmp_path / "moved"
    snapshot.manifest.parent.rename(moved)
    restored = PatchStageSnapshot.read(
        moved / "manifest.json", identity=snapshot.identity
    )
    assert restored.to_fs()["identity"] == snapshot.identity
    assert (
        json.loads(restored.manifest.read_text())["material_basis"]["vp"]
        == "sha256:" + "1" * 64
    )


@pytest.mark.parametrize(
    "changed",
    ["control_state.h5", "simulation.json", "parent.gmp", "manifest.json", "inputs"],
)
def test_stage_rejects_tampering_even_with_same_length(tmp_path, changed):
    snapshot, *_ = _publish(tmp_path)
    path = snapshot.manifest.parent / changed
    if path.is_dir():
        path = next(path.iterdir())
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match="identity mismatch|input changed"):
        snapshot.verify()


def test_stage_never_overwrites_an_existing_stage(tmp_path):
    snapshot, simulation, prepared, state = _publish(tmp_path)
    state.blocks["model.vp"][:] = 4
    with pytest.raises(FileExistsError, match="already published"):
        prepared.freeze_stage(
            state,
            name="stage-one",
            directory=snapshot.manifest.parent,
        )
    snapshot.verify()


def test_missing_input_does_not_publish_partial_stage(tmp_path):
    simulation, prepared, state = _fixture(tmp_path)
    (Path(simulation.project_path) / "reference.h5").unlink()
    with pytest.raises(FileNotFoundError):
        prepared.freeze_stage(
            state,
            name="stage",
            directory=tmp_path / "stage",
        )
    assert not (tmp_path / "stage").exists()
    assert not list(tmp_path.glob(".patch-stage-*"))


def test_stage_pins_rsf_sidecar(tmp_path):
    simulation, prepared, state = _fixture(tmp_path)
    project = Path(simulation.project_path)
    (project / "volume.bin").write_bytes(b"physical array")
    (project / "volume.rsf").write_text(
        'n1=2 in="volume.bin" data_format="native_float"\n'
    )
    simulation.extra["stage_fixture"] = {"file": "volume.rsf"}
    snapshot = prepared.freeze_stage(state, name="stage", directory=tmp_path / "stage")
    from frequensolve.model.property import rsf_binary_path

    payload = json.loads(snapshot.simulation_file.read_text())
    header = snapshot.manifest.parent / payload["stage_fixture"]["file"]
    assert rsf_binary_path(header).read_bytes() == b"physical array"
    (project / "volume.bin").unlink()
    snapshot.verify()


def test_external_hdf_links_cannot_escape_stage_freeze(tmp_path):
    simulation, prepared, state = _fixture(tmp_path)
    with h5py.File(Path(simulation.project_path) / "reference.h5", "r+") as h5:
        h5["external"] = h5py.ExternalLink("other.h5", "data")
    with pytest.raises(ValueError, match="external links"):
        prepared.freeze_stage(
            state,
            name="stage",
            directory=tmp_path / "stage",
        )
    assert not (tmp_path / "stage").exists()


def test_manifest_rejects_paths_outside_bundle(tmp_path):
    snapshot, *_ = _publish(tmp_path)
    payload = json.loads(snapshot.manifest.read_text())
    payload["files"][0]["file"] = "../outside"
    snapshot.manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="bundle-relative"):
        PatchStageSnapshot.read(snapshot.manifest, identity=_digest(snapshot.manifest))


def test_published_bundle_matches_native_contract(tmp_path):
    from jsonschema import Draft202012Validator

    from tests.test_imaging_jobs import CONTRACT_ROOT

    snapshot, *_ = _publish(tmp_path)
    schema = json.loads(
        (CONTRACT_ROOT / "internal/fs-patch-stage-1/schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(snapshot.manifest.read_text()))


@pytest.mark.integration
@pytest.mark.parametrize("dimension", [2, 3])
def test_native_reader_consumes_sdk_bundle(tmp_path, dimension):
    import os
    import subprocess

    probe = os.environ.get(f"FS_PATCH_STAGE_PROBE_{dimension}D")
    if not probe:
        pytest.skip("Set FS_PATCH_STAGE_PROBE_2D/3D to the native stage reader probe")
    snapshot, *_ = _publish(tmp_path)
    result = subprocess.run(
        [probe, str(snapshot.manifest), snapshot.identity],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "PML_STAGE_PROBE=PASS" in result.stdout
    assert str(snapshot.control_state) in result.stdout

    # Re-export the loaded simulation through the public SDK before native comparison.
    pinned_simulation = fs.SeismicSimulation.load(
        snapshot.simulation_file, project_path=snapshot.manifest.parent
    )
    current = snapshot.manifest.parent / "current.json"
    current.write_text(pinned_simulation.as_json())
    result = subprocess.run(
        [probe, str(snapshot.manifest), snapshot.identity, str(current)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PML_STAGE_PROBE=PASS" in result.stdout


def test_stale_preparation_cannot_publish_stage(tmp_path):
    simulation, prepared, state = _fixture(tmp_path)
    prepared.jobs[-1].geometry_report["parent_fingerprint"] = "FFFFFFFFFFFFFFFF"
    with pytest.raises(ValueError, match="current committed preparation"):
        prepared.freeze_stage(state, name="stage", directory=tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_candidate_mutation_during_copy_does_not_change_captured_state(
    tmp_path, monkeypatch
):
    from frequensolve.mesh._stage_snapshot import _InputCopier

    simulation, prepared, state = _fixture(tmp_path)
    copy_file = _InputCopier._copy_file

    def mutate_candidate(copier, source, destination):
        state.blocks["model.vp"][:] = 99
        copy_file(copier, source, destination)

    monkeypatch.setattr(_InputCopier, "_copy_file", mutate_candidate)
    snapshot = prepared.freeze_stage(state, name="stage", directory=tmp_path / "stage")
    np.testing.assert_allclose(
        ControlStateFile.read(snapshot.control_state).blocks["model.vp"], [0.1, 0.3]
    )


def test_input_mutation_during_copy_does_not_publish_stage(tmp_path, monkeypatch):
    from frequensolve.mesh import _stage_snapshot

    simulation, prepared, state = _fixture(tmp_path)
    copy_file = _stage_snapshot.shutil.copyfile

    def mutate_source(source, destination):
        copy_file(source, destination)
        Path(source).write_bytes(b"changed input")

    monkeypatch.setattr(_stage_snapshot.shutil, "copyfile", mutate_source)
    with pytest.raises(ValueError, match="changed while copying"):
        prepared.freeze_stage(state, name="stage", directory=tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_stage_cannot_recursively_copy_its_own_destination(tmp_path):
    simulation, prepared, state = _fixture(tmp_path)
    simulation.extra["stage_fixture"]["artifact"] = str(tmp_path)
    with pytest.raises(ValueError, match="inside a copied input directory"):
        prepared.freeze_stage(state, name="stage", directory=tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_stage_job_roundtrip_and_input_identity(tmp_path):
    from frequensolve.imaging.jobs import FWIOperatorJob
    from tests.test_imaging_jobs import _assert_valid

    snapshot, simulation, prepared, state = _publish(tmp_path)
    candidate = state.write(tmp_path / "candidate.h5")
    job = FWIOperatorJob(
        "stage_job",
        simulation,
        [3],
        action="linearize",
        active=["model.vp"],
        state="linearization.json",
        control_state=candidate,
        pml_stage=snapshot,
    )
    payload = job.to_fs()
    _assert_valid(payload)
    assert (
        payload["fwi_operator"]["controls"]["pml_stage"]["identity"]
        == snapshot.identity
    )
    job.save()
    restored = FWIOperatorJob.from_fs(json.loads(job.job_file.read_text()))
    assert restored.pml_stage.identity == snapshot.identity
    assert restored._input_fingerprint_payload()["pml_stage"] == snapshot.identity
    with pytest.raises(ValueError, match="explicit candidate"):
        FWIOperatorJob(
            "bad_stage",
            simulation,
            [3],
            action="linearize",
            active=[],
            state="linearization.json",
            pml_stage=snapshot,
        )
    with h5py.File(snapshot.control_state, "r+") as h5:
        h5["controls/model.vp"][:] = 7
    with pytest.raises(ValueError, match="input changed"):
        restored.to_fs()
