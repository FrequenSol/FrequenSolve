"""Retention policy preserves successful provenance and physical invalidation."""

import json

import pytest

from frequensolve.simulation.artifact_contract import TaskResult
from frequensolve.simulation.jobs.forward import FrequencyDomainJob
from tests.test_task_result_currentness import _saved_job, _write_task_result


def test_retry_policy_defaults_false_and_roundtrips(tmp_path):
    job = _saved_job(tmp_path, [2.0])
    assert job.preserve_task_outputs is False
    assert "preserve_task_outputs" not in job.to_fs()
    job.preserve_task_outputs = True
    job.save()
    loaded = FrequencyDomainJob.from_fs(
        json.loads(job.job_file.read_text()), project_path=job.project_path
    )
    assert loaded.preserve_task_outputs is True
    job.preserve_task_outputs = "true"
    with pytest.raises(TypeError, match="boolean"):
        job.to_fs()


def test_successful_task_reuse_keeps_full_producer_hashes(tmp_path):
    job = _saved_job(tmp_path, [2.0, 3.0])
    job.preserve_task_outputs = True
    job.save()
    original = job._artifact_contract_fingerprints()
    fingerprints = {**original, "compatibility": job._retry_compatibility_hash()}
    first, _ = _write_task_result(job, 1, fingerprints=fingerprints)
    _write_task_result(job, 2, fingerprints=fingerprints, status="failed")
    producer_bytes = first.read_bytes()
    simfile = job.simulation._file
    simulation = json.loads(simfile.read_text())
    simulation.setdefault("Solver", {})["max_iter"] = 999
    simulation["Solver"]["tolerance"] = 1e-10
    simfile.write_text(json.dumps(simulation))
    assert job.current_tasks() == [1]
    assert first.read_bytes() == producer_bytes
    assert (
        TaskResult.read(first, result_path=job._result_path).fingerprints
        == fingerprints
    )
    simulation["Solver"]["grids"] = 99
    simfile.write_text(json.dumps(simulation))
    assert job.current_tasks() == []


def test_staged_retry_identity_matches_staged_payloads(tmp_path):
    job = _saved_job(tmp_path, [2.0])
    job.preserve_task_outputs = True
    staged_job, _ = job.save_for_remote("test", "/remote/project")
    staged_sim, _ = job.save_simulation_for_remote("test", "/remote/project")
    expected = job._compatibility_hash_payloads(
        json.loads(staged_job.read_text()), json.loads(staged_sim.read_text())
    )
    assert job.staged_task_fingerprints("test") == {"compatibility": expected}
    assert set(job.staged_artifact_fingerprints("test")) == {
        "job",
        "simulation",
        "outputs",
    }
    job.preserve_task_outputs = False
    assert job.staged_task_fingerprints("test") == job.staged_artifact_fingerprints(
        "test"
    )


def test_task_result_roundtrip_preserves_detailed_provenance(tmp_path):
    job = _saved_job(tmp_path, [2.0])
    file, _ = _write_task_result(job, 1)
    document = json.loads(file.read_text())
    document.update(
        build={"git_commit": "abc"},
        execution={"exports": [{"path": "/requested/state.json"}]},
        license={"license_id": "test"},
        misc={"active_solve_dofs": 10},
        workflow={"name": "forward"},
    )
    file.write_text(json.dumps(document))
    parsed = TaskResult.read(file, result_path=job._result_path)
    assert parsed.to_fs() == document
