# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import json
import shutil
from pathlib import Path

import pytest

from frequensolve.imaging.jobs import FWIOperatorJob
from frequensolve.mesh._stage_mesh import PatchStageMesh
from frequensolve.mesh._stage_snapshot import PatchStageSnapshot, _digest
from tests.test_imaging_jobs import _assert_valid
from tests.test_patch_stage_snapshot import _publish


def _capture(root, stage):
    root.mkdir()
    files = {}
    for name in ("initial.h5", "refinements.h5", "final.h5"):
        path = root / name
        path.write_bytes(name.encode())
        files[name] = {"sha256": _digest(path), "bytes": path.stat().st_size}
    manifest = root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "fs-stage-mesh-1",
                "execution_identity": "sha256:" + "1" * 64,
                "context": {"stage_identity": stage.identity},
                "files": files,
            }
        )
    )
    return manifest


def test_publication_relocation_and_job_roundtrip(tmp_path):
    stage, simulation, _, _ = _publish(tmp_path)
    source = _capture(tmp_path / "capture", stage)
    mesh = stage.publish_mesh(source, tmp_path / "mesh")
    original = mesh.identity
    source.parent.joinpath("initial.h5").write_bytes(b"modified")
    moved = tmp_path / "relocated"
    mesh.manifest.parent.rename(moved)
    mesh = PatchStageMesh.read(moved / "manifest.json", identity=original, stage=stage)
    job = FWIOperatorJob(
        "replay",
        simulation,
        [3],
        action="linearize",
        active=[],
        state="linearization.json",
        control_state=stage.control_state,
        pml_stage=stage,
        stage_mesh=mesh,
    )
    payload = job.to_fs()
    _assert_valid(payload)
    job.save()
    restored = FWIOperatorJob.from_fs(json.loads(job.job_file.read_text()))
    assert restored.stage_mesh.identity == original
    assert restored._input_fingerprint_payload()["stage_mesh"] == original
    capture = FWIOperatorJob(
        "capture",
        simulation,
        [3],
        action="linearize",
        active=[],
        state="linearization.json",
        control_state=stage.control_state,
        pml_stage=stage,
        stage_mesh="capture",
    )
    _assert_valid(capture.to_fs())
    capture.save()
    assert (
        FWIOperatorJob.from_fs(json.loads(capture.job_file.read_text())).stage_mesh
        == "capture"
    )


@pytest.mark.parametrize(
    "name", ["initial.h5", "refinements.h5", "final.h5", "manifest.json"]
)
def test_committed_mutation_is_rejected(tmp_path, name):
    stage, _, _, _ = _publish(tmp_path)
    source = _capture(tmp_path / "capture", stage)
    mesh = stage.publish_mesh(source, tmp_path / "mesh")
    path = mesh.manifest.parent / name
    original = path.read_bytes()
    path.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError, match="identity mismatch|file changed"):
        mesh.verify(stage=stage)


def test_publication_refuses_overwrite_and_invalid_stage(tmp_path):
    stage, _, _, _ = _publish(tmp_path)
    source = _capture(tmp_path / "capture", stage)
    mesh = stage.publish_mesh(source, tmp_path / "mesh")
    with pytest.raises(FileExistsError):
        stage.publish_mesh(source, mesh.manifest.parent)
    payload = json.loads(source.read_text())
    payload["context"]["stage_identity"] = "sha256:" + "0" * 64
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different input stage"):
        stage.publish_mesh(source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_capture_requires_baseline_and_single_frequency(tmp_path):
    stage, simulation, _, state = _publish(tmp_path)
    options = dict(
        action="linearize",
        active=[],
        state="linearization.json",
        pml_stage=stage,
        stage_mesh="capture",
    )
    with pytest.raises(ValueError, match="one frequency"):
        FWIOperatorJob(
            "many", simulation, [3, 7.5], control_state=stage.control_state, **options
        )
    state.blocks["model.vp"][:] = 8
    candidate = Path(simulation.project_path) / "candidate.h5"
    state.write(candidate)
    with pytest.raises(ValueError, match="canonical stage baseline"):
        FWIOperatorJob("candidate", simulation, [3], control_state=candidate, **options)


def test_capture_rejects_frequency_outside_stage(tmp_path):
    stage, simulation, _, _ = _publish(tmp_path)
    with pytest.raises(ValueError, match="outside the pinned stage band"):
        FWIOperatorJob(
            "outside",
            simulation,
            [9],
            action="linearize",
            active=[],
            state="unused.json",
            control_state=stage.control_state,
            pml_stage=stage,
            stage_mesh="capture",
        )


def test_failed_publication_keeps_destination_absent(tmp_path, monkeypatch):
    from frequensolve.mesh import _stage_mesh

    stage, _, _, _ = _publish(tmp_path)
    source = _capture(tmp_path / "capture", stage)
    copy = _stage_mesh.shutil.copyfile

    def mutate(original, target):
        result = copy(original, target)
        if Path(original).name == "final.h5":
            Path(original).write_bytes(b"changed during copy")
        return result

    monkeypatch.setattr(_stage_mesh.shutil, "copyfile", mutate)
    with pytest.raises(ValueError, match="file changed"):
        stage.publish_mesh(source, tmp_path / "mesh")
    assert not (tmp_path / "mesh").exists()
    assert not list(tmp_path.glob(".stage-mesh-*"))


@pytest.mark.parametrize("mode", ["capture", "replay"])
def test_remote_staging_preserves_complete_stage_bundles(tmp_path, mode):
    stage, simulation, _, _ = _publish(
        tmp_path, directory=tmp_path / "project" / "stage"
    )
    project = Path(simulation.project_path)
    mesh = stage.publish_mesh(_capture(tmp_path / "capture", stage), project / "mesh")
    # Generated jobs and unrelated files must not become immutable inputs.
    (stage.manifest.parent / "unrelated.txt").write_text("do not transfer")
    candidate = project / "candidate.h5"
    shutil.copyfile(stage.control_state, candidate)
    job = FWIOperatorJob(
        "remote_stage",
        simulation,
        [3],
        action="linearize",
        active=[],
        state="linearization.json",
        control_state=candidate,
        pml_stage=stage,
        stage_mesh="capture" if mode == "capture" else mesh,
    )
    destination = tmp_path / "remote_project"
    staged, _ = job.save_for_remote("fixture", destination)
    payload = json.loads(staged.read_text())
    pairs = job.remote_input_files(destination)
    assert len(pairs) == len(set(pairs))
    transferred = {Path(local).resolve() for local, _ in pairs}
    expected = set(stage.input_files()) | {candidate.resolve()}
    if mode == "replay":
        expected.update(mesh.input_files())
    assert expected <= transferred
    assert stage.manifest.parent / "unrelated.txt" not in transferred
    for local, remote in pairs:
        remote.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, remote)
    restored = PatchStageSnapshot.read(
        destination / "stage" / "manifest.json", identity=stage.identity
    )
    assert payload["fwi_operator"]["controls"]["pml_stage"] == restored.to_fs()
    assert (
        Path(payload["fwi_operator"]["controls"]["state"])
        == destination / "candidate.h5"
    )
    if mode == "replay":
        replay = PatchStageMesh.read(
            destination / "mesh" / "manifest.json",
            identity=mesh.identity,
            stage=restored,
        )
        assert payload["fwi_operator"]["controls"]["stage_mesh"] == replay.to_fs()
    else:
        assert not (destination / "mesh").exists()
    stage.manifest.parent.joinpath("parent.gmp").write_bytes(b"changed")
    with pytest.raises(ValueError, match="input changed"):
        job.remote_input_files(destination)


def test_remote_staging_requires_project_contained_bundle(tmp_path):
    stage, simulation, _, _ = _publish(tmp_path)
    job = FWIOperatorJob(
        "remote_stage",
        simulation,
        [3],
        action="linearize",
        active=[],
        state="linearization.json",
        control_state=stage.control_state,
        pml_stage=stage,
        stage_mesh="capture",
    )
    with pytest.raises(ValueError, match="inside the job project"):
        job.remote_input_files(tmp_path / "remote_project")
