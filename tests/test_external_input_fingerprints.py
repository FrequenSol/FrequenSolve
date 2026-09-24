import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from frequensolve.imaging.jobs import ControlGradientJob
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.orchestrator.sites.hpc.site import SlurmSite
from frequensolve.simulation.artifact_contract import (
    ArtifactCatalog,
    TaskPartition,
    TaskResult,
)
from frequensolve.simulation.jobs import BaseJob
from frequensolve.simulation.jobs.serialization import JobSerializationMixin
from frequensolve.simulation.simulation import SeismicSimulation


def _born_job(tmp_path, *, n_tasks=1):
    simulation = SeismicSimulation(
        name="controlled",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    direction = tmp_path / "direction.h5"
    direction.write_bytes(b"first direction")
    job = ControlGradientJob(
        "born",
        simulation,
        [float(index + 1) for index in range(n_tasks)],
        kind="born",
        direction=direction,
    )
    return job, direction


def test_serialized_external_inputs_are_hashed_once_for_one_thousand_tasks(
    tmp_path,
    monkeypatch,
):
    job, _direction = _born_job(tmp_path, n_tasks=1000)
    calls = 0
    original = JobSerializationMixin._sha256_file

    def count_sha256(path):
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(
        JobSerializationMixin,
        "_sha256_file",
        staticmethod(count_sha256),
    )

    BaseSite().prepare_job(job, validate=False)
    expected = job.fingerprint_payload()["inputs"]
    for task in range(1, job.n_tasks + 1):
        assert job.task_fingerprint_payload(task)["inputs"] == expected

    assert calls == 1


def test_saved_job_contains_compact_artifact_provenance(tmp_path):
    job, direction = _born_job(tmp_path)

    BaseSite().prepare_job(job, validate=False)
    payload = json.loads(job.job_file.read_text())

    assert payload["artifact_contract"] == {
        "schema": "fs-job-artifact-metadata-1",
        "versions": {
            "task_result": "fs-task-result-2",
            "operation_result": "fs-operation-result-1",
            "task_index": "fs-task-index-1",
            "collection": "fs-sharded-array-1",
        },
        "external_inputs": {
            "schema": "fs-external-input-set-1",
            "digest": job.fingerprint_payload()["inputs"]["digest"],
        },
        "output_request": {
            "schema": "fs-output-request-fingerprint-1",
            "digest": job.effective_output_request_fingerprint(),
        },
    }
    serialized = json.dumps(payload["artifact_contract"])
    assert str(direction) not in serialized
    assert "first direction" not in serialized


def test_fresh_prepare_refreshes_external_input_digest(tmp_path):
    job, direction = _born_job(tmp_path)
    site = BaseSite()

    site.prepare_job(job, validate=False)
    first = job.task_fingerprint_payload(1)["inputs"]["digest"]
    direction.write_bytes(b"second direction")

    # A prepared job is a stable staging snapshot until the next boundary.
    assert job.task_fingerprint_payload(1)["inputs"]["digest"] == first
    site.prepare_job(job, validate=False)
    second = job.task_fingerprint_payload(1)["inputs"]["digest"]

    assert second != first
    assert (
        json.loads(job.job_file.read_text())["artifact_contract"]["external_inputs"][
            "digest"
        ]
        == second
    )


def test_saved_fingerprints_refresh_external_inputs_at_save_boundary(tmp_path):
    job, direction = _born_job(tmp_path)
    job.save()

    whole = job.fingerprint()
    task = job.task_fingerprint(1)
    direction.write_bytes(b"second direction")

    assert job.fingerprint() == whole
    assert job.task_fingerprint(1) == task
    job.save()
    assert job.fingerprint() != whole
    assert job.task_fingerprint(1) != task


def test_direct_save_refreshes_serialized_external_input_digest(tmp_path):
    job, direction = _born_job(tmp_path)

    job.save()
    first = json.loads(job.job_file.read_text())["artifact_contract"][
        "external_inputs"
    ]["digest"]
    direction.write_bytes(b"second direction")
    job.save()
    second = json.loads(job.job_file.read_text())["artifact_contract"][
        "external_inputs"
    ]["digest"]

    assert second != first


def test_remote_staging_hashes_external_inputs_once(tmp_path, monkeypatch):
    job, direction = _born_job(tmp_path)
    calls = 0
    original = JobSerializationMixin._sha256_file

    def count_sha256(path):
        nonlocal calls
        if Path(path) == direction:
            calls += 1
        return original(path)

    monkeypatch.setattr(
        JobSerializationMixin,
        "_sha256_file",
        staticmethod(count_sha256),
    )

    staged, _remote = job.save_for_remote("test", "/remote/project")

    assert calls == 1
    assert (
        json.loads(staged.read_text())["artifact_contract"]
        == json.loads(job.job_file.read_text())["artifact_contract"]
    )


def test_staged_provenance_is_compact_atomic_generation_state(tmp_path):
    job, _direction = _born_job(tmp_path)

    staged_job, _ = job.save_for_remote("test", "/remote/project")
    provenance = job._staged_provenance_path("test")
    first = json.loads(provenance.read_text())

    assert first == {
        "schema": "fs-staged-provenance-1",
        "job": {"digest": job._sha256_file(staged_job)},
    }
    staged_simulation, _ = job.save_simulation_for_remote("test", "/remote/project")
    complete = json.loads(provenance.read_text())
    assert complete == {
        **first,
        "simulation": {"digest": job._sha256_file(staged_simulation)},
    }
    assert job.staged_artifact_fingerprints("test") == {
        "job": complete["job"]["digest"],
        "simulation": complete["simulation"]["digest"],
        "outputs": job._output_request_fingerprint_cache["digest"],
    }

    job.save_for_remote("test", "/different/remote/project")
    replacement = json.loads(provenance.read_text())
    assert set(replacement) == {"schema", "job"}
    assert job.staged_artifact_fingerprints("test") is None


def test_staged_fingerprints_make_remote_currentness_relocation_stable(tmp_path):
    (tmp_path / "first").mkdir()
    first, _ = _born_job(tmp_path / "first")
    first.save_for_remote("SimpleNamespace", "/remote/project")
    first.save_simulation_for_remote("SimpleNamespace", "/remote/project")

    relocated_root = tmp_path / "relocated"
    shutil.copytree(first.project_path, relocated_root)
    copied_job = relocated_root / first.job_file.relative_to(first.project_path)
    relocated = BaseJob.load(copied_job, project_path=relocated_root)

    fingerprints = first.staged_artifact_fingerprints("SimpleNamespace")
    assert relocated.staged_artifact_fingerprints("SimpleNamespace") == fingerprints
    result = TaskResult(
        path=Path("result.json"),
        result_path=tmp_path,
        partition=TaskPartition(task=1, task_count=1, frequency=complex(1.0)),
        fingerprints=fingerprints,
        state="success",
        code=0,
        artifacts=(),
    )
    catalog = ArtifactCatalog(result_path=tmp_path, results={1: result})

    SlurmSite._validate_remote_catalog(
        SimpleNamespace(), relocated, catalog, tasks=(1,)
    )


def test_directory_input_fingerprint_is_compact_and_content_sensitive(tmp_path):
    input_dir = tmp_path / "observed"
    (input_dir / "nested").mkdir(parents=True)
    (input_dir / "a.h5").write_bytes(b"first")
    (input_dir / "nested" / "b.h5").write_bytes(b"second")

    first = JobSerializationMixin._path_content_fingerprint(input_dir)
    repeated = JobSerializationMixin._path_content_fingerprint(input_dir)
    (input_dir / "nested" / "b.h5").write_bytes(b"changed")
    changed = JobSerializationMixin._path_content_fingerprint(input_dir)

    assert first == repeated
    assert first == {
        "kind": "directory",
        "sha256": first["sha256"],
        "files": 2,
    }
    assert changed["sha256"] != first["sha256"]


def test_downloaded_fingerprints_validate_rewritten_inputs_and_reload(tmp_path):
    job, _ = _born_job(tmp_path)
    job.save_for_remote("SlurmSite", "/work2/example/project")
    job.save_simulation_for_remote("SlurmSite", "/work2/example/project")
    expected = job.staged_task_fingerprints("SlurmSite")
    assert job.downloaded_task_fingerprints() == [expected]
    loaded = BaseJob.load(job.job_file)
    assert loaded.downloaded_task_fingerprints() == [expected]
    loaded.save()
    assert loaded.downloaded_task_fingerprints() == [expected]
    # Submission records the scheduler id after the inputs were staged.
    loaded._job_id = "3532821"
    loaded.save()
    assert "job_id" in json.loads(loaded.job_file.read_text())
    assert loaded.downloaded_task_fingerprints() == [expected]
    payload = json.loads(loaded.simulation._file.read_text())
    payload["scaling"] = "robust"
    loaded.simulation._file.write_text(json.dumps(payload))
    assert loaded.downloaded_task_fingerprints() == []


def test_downloaded_fingerprints_reject_tampered_staging(tmp_path):
    job, _ = _born_job(tmp_path)
    staged, _ = job.save_for_remote("SlurmSite", "/work2/example/project")
    job.save_simulation_for_remote("SlurmSite", "/work2/example/project")
    staged.write_text(staged.read_text() + "\n")
    assert job.downloaded_task_fingerprints() == []
