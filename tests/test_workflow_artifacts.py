import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from frequensolve.workflow.analysis import analyze_pressure
from frequensolve.workflow.client import create_project, staged_inputs
from frequensolve.workflow.native import pressure_snapshot
from frequensolve.workflow.results import _publish, snapshot_from_bundle
from tests.test_workflow import snapshot


def native_files(tmp_path):
    h5py = pytest.importorskip("h5py")
    a, b = tmp_path / "trace.h5", tmp_path / "survey.h5"
    with h5py.File(a, "w") as f:
        d = f.create_dataset("reciprocal", data=np.ones((2, 1, 2, 2)))
        for name, value in {
            "representation": "coordinate",
            "coordinate_units": "Pa",
            "components": "p",
            "roles": "primal",
            "rhs_norm_dataset": "/reciprocal_rhs_norm",
        }.items():
            d.attrs["value_frame_" + name] = np.asarray(
                [value], dtype=h5py.string_dtype()
            )
        f["reciprocal_rhs_norm"] = [1.0, 1.0]
        f["frequency"], f["laplace"], f["task_id"] = 2.0, 0.0, 1
        for key in (
            "fs_solver_version",
            "fs_build_id",
            "fs_physics",
            "fs_dimension",
            "fs_workflow",
        ):
            f.attrs[key] = {
                "fs_physics": "acoustic",
                "fs_dimension": "3",
                "fs_workflow": "forward",
            }.get(key, "fake-" + key)
    with h5py.File(b, "w") as f:
        f["survey/receiver_groups/_catalog/group_name"] = np.asarray(
            ["reciprocal"], dtype=h5py.string_dtype()
        )
        for name in (
            "survey/receiver_groups/reciprocal/receivers/coordinates",
            "survey/source_encoding/reference_coordinates",
        ):
            f[name] = np.array([[0, 1], [0, 0], [0, 0]], dtype=float)
            f[name].attrs["units"] = np.asarray(["m"], dtype=h5py.string_dtype())
        f["survey/source_encoding/field_id"] = [1, 2]
        f["survey/receiver_groups/reciprocal/receivers/receiver_id"] = [1, 2]
        f["survey/source_encoding/source_scale"] = [1.0, 1.0]
        f["survey/source_encoding/counts"] = [2, 2]
        f["survey/source_encoding/encoding_kind"] = np.asarray(
            ["identity"], dtype=h5py.string_dtype()
        )
        for key in (
            "fs_solver_version",
            "fs_build_id",
            "fs_physics",
            "fs_dimension",
            "fs_workflow",
        ):
            f.attrs[key] = {
                "fs_physics": "acoustic",
                "fs_dimension": "3",
                "fs_workflow": "forward",
            }.get(key, "fake-" + key)
    return a, b


def test_curated_inputs_new_identity_and_deterministic_remote_serialization(tmp_path):
    first = create_project(tmp_path, "first", "First")
    second = create_project(tmp_path, "second", "Second")
    assert first["project_id"] != second["project_id"]
    identity, files, job = staged_inputs(tmp_path / "first")
    assert identity.storage_prefix == first["project_key"]
    assert files == staged_inputs(tmp_path / "first")[1]
    assert len(files) == 2
    assert json.loads(files[job])["f_list"] == [2.0, 20.0]
    simulation = json.loads(files["simulations/acoustic_3d/acoustic_3d.json"])
    assert (
        simulation["dimension"] == 3 and simulation["Solver"]["precision"] == "double"
    )
    with pytest.raises(FileExistsError):
        create_project(tmp_path, "first", "Replacement")


def test_native_units_and_external_link_rejection(tmp_path):
    h5py = pytest.importorskip("h5py")
    a, b = native_files(tmp_path)
    row = {
        "frequency_index": 0,
        "frequency_value": ["2", "0"],
        "scientific": {"converged": True},
    }
    result = pressure_snapshot(
        a.read_bytes(), b.read_bytes(), row, {"origin": "synthetic"}
    )
    assert result["metadata"]["units"] == "Pa"
    assert result["metadata"]["coordinate_units"] == "m"
    assert result["metadata"]["source_encoding"]["encoding_kind"] == ["identity"]
    with pytest.raises(ValueError, match="identity"):
        pressure_snapshot(
            a.read_bytes(), b.read_bytes(), {**row, "frequency_index": 1}, {}
        )
    with h5py.File(a, "a") as f:
        del f["reciprocal"]
        f["reciprocal"] = h5py.ExternalLink(str(b), "/private")
    with pytest.raises(ValueError, match="links"):
        pressure_snapshot(a.read_bytes(), b.read_bytes(), row, {})


def test_bundle_hash_verification_and_nonoverwriting_publication(tmp_path):
    a, b = native_files(tmp_path)
    files = [
        {
            "path": "payload.h5",
            "schema": "fs_trace_payload_shard_v1",
            "data": a.read_bytes(),
        },
        {
            "path": "metadata.h5",
            "schema": "fs_trace_metadata_v1",
            "data": b.read_bytes(),
        },
    ]
    manifest = {
        "version": "workflow-native-bundle.v1",
        "results": {
            "plan_id": "fake-plan",
            "execution_id": "fake-run",
            "plan_revision": 1,
            "digest": "fake-digest",
        },
        "selected": [
            {
                "frequency": {
                    "frequency_index": 0,
                    "frequency_value": ["2", "0"],
                    "scientific": {"converged": True},
                },
                "files": [
                    {
                        "path": f["path"],
                        "schema": f["schema"],
                        "size": len(f["data"]),
                        "sha256": hashlib.sha256(f["data"]).hexdigest(),
                    }
                    for f in files
                ],
            }
        ],
    }

    def bundle(corrupt=False):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("manifest.json", json.dumps(manifest))
            for f in files:
                z.writestr(f["path"], b"changed" if corrupt else f["data"])
        return buf.getvalue()

    _publish(tmp_path, "result.zip", bundle())
    assert (
        snapshot_from_bundle(tmp_path, "result.zip", 0)["evidence"][
            "service_attestation"
        ]
        is False
    )
    with pytest.raises(FileExistsError):
        _publish(tmp_path, "result.zip", b"replacement")
    _publish(tmp_path, "corrupt.zip", bundle(True))
    with pytest.raises(ValueError, match="changed"):
        snapshot_from_bundle(tmp_path, "corrupt.zip", 0)


def test_complex_difference_is_not_difference_of_magnitudes():
    result = analyze_pressure(snapshot([1j]), snapshot([1]))
    assert result["comparison"]["complex_rms_difference"] == pytest.approx(2**0.5)


def test_clean_notebook_and_failed_execution_evidence(tmp_path, monkeypatch):
    pytest.importorskip("nbclient")
    pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    import nbclient

    import frequensolve.workflow.notebooks as operations
    from frequensolve.workflow.notebooks import (
        create_notebook,
        replay_notebook,
        verify_replay,
    )

    source = tmp_path / "pressure.json"
    source.write_text(json.dumps(snapshot([3 + 4j])))
    notebook = create_notebook(tmp_path / "complete", source)
    receipt = replay_notebook(notebook, timeout=30)
    assert receipt["status"] == "complete" and receipt["origin"] == "local"
    assert verify_replay(notebook)["valid"]
    original = notebook.read_bytes()
    notebook.write_bytes(original + b"\n")
    assert "notebook_changed" in verify_replay(notebook)["mismatches"]
    notebook.write_bytes(original)
    local_input = notebook.parent / "pressure-0.json"
    original_input = local_input.read_bytes()
    local_input.write_text("{}")
    assert "input_changed" in verify_replay(notebook)["mismatches"]
    local_input.write_bytes(original_input)
    with monkeypatch.context() as changed:
        changed.setattr(
            operations.importlib.metadata, "version", lambda name: "changed-environment"
        )
        assert "environment_changed" in verify_replay(notebook)["mismatches"]
    assert json.loads((notebook.parent / "report.json").read_text())["magnitude"] == [
        [5.0]
    ]
    with pytest.raises(FileExistsError):
        replay_notebook(notebook)
    failed = create_notebook(tmp_path / "failed", source)

    def fail(*args, **kwargs):
        raise RuntimeError("fake kernel failure")

    monkeypatch.setattr(nbclient.NotebookClient, "execute", fail)
    with pytest.raises(RuntimeError, match="kernel failure"):
        replay_notebook(failed)
    assert (
        json.loads((failed.parent / "replay.json").read_text())["status"]
        == "incomplete"
    )
    assert not (failed.parent / "report.html").exists()
