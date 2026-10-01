import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "operation,includes_context",
    [("prepare", False), ("gradient", True), ("proximal", True)],
)
def test_regularization_remote_scan_distinguishes_context_input(
    operation, includes_context
):
    from frequensolve.simulation.jobs.remote import JobRemoteMixin

    payload = {
        "control_sensitivities": {
            "input": "input.h5",
            "Regularization": {
                "operation": operation,
                "metric": "metric.h5",
                "lower": "lower.h5",
                "upper": "upper.h5",
                "context": "context.h5",
                "result": "output.h5",
            },
        }
    }
    refs = set(JobRemoteMixin._iter_file_references(payload))
    assert refs == {"input.h5", "metric.h5", "lower.h5", "upper.h5"} | (
        {"context.h5"} if includes_context else set()
    )


@pytest.mark.parametrize("schema", ["fs-objective-vector-3", "fs-objective-vector-4"])
def test_direct_remote_submission_stages_nested_and_absolute_vector_payloads(
    tmp_path, schema
):
    from frequensolve.imaging.data import DataSpace, DataVector, canonical_json_sha256
    from frequensolve.imaging.jobs import FWIOperatorJob
    from frequensolve.orchestrator.sites.hpc.site import SlurmSite
    from frequensolve.project import Project
    from frequensolve.seismic import Acquisition, ReceiverNode

    root = (tmp_path / "project").resolve()
    project = Project(name="direct", path=root)
    sim = project.new_simulation(name="sim", physics="acoustic", dimension=2)
    acq = Acquisition()
    acq.add_sources(kind="scalar", coords=[[0, 0]])
    node = ReceiverNode(name="hydrophone")
    node.add_component(name="p", field="pressure")
    acq.add_receiver_group(name="surface", device=node, coords=[[0, 0]])
    sim.acquisition = acq
    space = DataSpace.from_simulation(sim, [4.0])
    inputs = root / "jobs" / "client_inputs"
    for task in (1, 2):
        manifest = space.ones().write_objective_vector(
            inputs / f"dual_{task}.json",
            state_fingerprint="sha256:" + "a" * 64,
            term_layout=space.term_layouts(),
            schema=schema,
        )
        payload = json.loads(manifest.read_text())
        data_path = (
            manifest.with_suffix(".h5")
            if schema.endswith("-4")
            else DataVector.shard_path(manifest)
        )
        nested = inputs / "data" / data_path.name
        nested.parent.mkdir(exist_ok=True)
        data_path.rename(nested)
        reference = str(nested) if task == 1 else f"data/{nested.name}"
        if schema.endswith("-4"):
            payload["file"] = reference
        else:
            payload["shards"][0]["file"] = reference
        payload.pop("manifest_fingerprint")
        payload["manifest_fingerprint"] = canonical_json_sha256(payload)
        manifest.write_text(json.dumps(payload))
    job = FWIOperatorJob(
        "vjp",
        sim,
        [4.0, 6.0],
        action="vjp",
        active=["model.vp"],
        state="state.json",
        objective_vector=inputs / "dual.json",
        covector="vjp.h5",
    )
    job.save()
    remote = (tmp_path / "remote").resolve()
    # Exercise the actual site's input-transfer entry point without a connection.
    uploads = []

    class Site:
        work_dir = remote
        _transfer_remote_simulation_inputs = (
            SlurmSite._transfer_remote_simulation_inputs
        )

        def put(self, local, target):
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(local).read_bytes())
            uploads.append(target)

    Site()._transfer_remote_simulation_inputs(job)
    for task in (1, 2):
        staged = remote / "jobs" / "client_inputs" / f"dual_{task}.json"
        if schema.endswith("-3"):
            reference = json.loads(staged.read_text())["shards"][0]["file"]
            assert reference.startswith(str(remote)) and Path(reference).is_file()
        np.testing.assert_array_equal(
            DataVector.read_objective_vector(staged, space).values, space.ones().values
        )
    assert len({p for p in uploads if p.suffix == ".h5"}) == 2
    assert not (remote / "jobs" / "client_inputs" / "dual.json").exists()


def _vp_coefficients(path):
    """Return the inline sediment ``vp`` control coefficients of a saved simulation."""

    def find(node):
        if isinstance(node, dict):
            parameterized = node.get("parameterized")
            if isinstance(parameterized, dict) and parameterized.get("id") == "vp":
                return parameterized["control"]["coefficients"]
            nodes = node.values()
        else:
            nodes = node if isinstance(node, list) else ()
        for value in nodes:
            found = find(value)
            if found is not None:
                return found
        return None

    return find(json.loads(Path(path).read_text()))


def test_job_saves_write_each_simulation_content_once(tmp_path, monkeypatch):
    from frequensolve.imaging.jobs import FWIOperatorJob
    from frequensolve.model.parameterization import (
        ParameterizedProperty,
        TensorHatControl,
    )
    from frequensolve.simulation import simulation as module
    from tests.imaging_fakes import layered_simulation

    simulation = layered_simulation(tmp_path / "project", save=False)
    control = TensorHatControl(
        axes=("x", "z"),
        shape=(4, 3),
        origin=(0.0, 200.0),
        spacing=(1000.0, 500.0),
        coefficients=np.arange(12.0),
    )
    sediment = next(s for s in simulation.model.subdomains if s.name == "sediment")
    sediment.properties["vp"] = ParameterizedProperty(1900.0, id="vp", control=control)
    written = []
    write = module.atomic_write_json
    monkeypatch.setattr(
        module,
        "atomic_write_json",
        lambda path, *args, **kwargs: written.append(path)
        or write(path, *args, **kwargs),
    )

    def job(name):
        return FWIOperatorJob(
            name,
            simulation,
            [4.0],
            action="linearize",
            active=["model.vp"],
            state="s.json",
            covector="g.h5",
        )

    file = simulation.save()
    identity = (file.stat().st_ino, file.stat().st_mtime_ns)
    digest = JobSerializationMixin._sha256_file(file)
    # Every job saves its simulation; unchanged content is not written again.
    jobs = [job(name) for name in ("first", "second", "third")]
    for each in jobs:
        each.save()
    assert written == [file]
    assert (file.stat().st_ino, file.stat().st_mtime_ns) == identity
    assert jobs[-1]._artifact_contract_fingerprints()["simulation"] == digest
    staged, _ = jobs[-1].save_simulation_for_remote("test", "/remote/project")
    assert _vp_coefficients(staged) == list(range(12))
    assert json.loads(staged.read_text())["project_path"] == "/remote/project"
    # Any content change is written, including an in-place coefficient edit.
    sediment.properties["vp"].control.coefficients[0] = 7.0
    job("edited").save()
    assert len(written) == 2 and _vp_coefficients(file)[0] == 7.0
    sediment.properties["vp"] = sediment.properties["vp"].with_coefficients(
        np.zeros(12)
    )
    job("replaced").save()
    assert len(written) == 3 and not any(_vp_coefficients(file))
    simulation.save()
    assert len(written) == 3
    # Other JSON options, or a file changed or removed by anyone else, are
    # written again.
    simulation.save(indent=None)
    assert len(written) == 4 and "\n" not in file.read_text()
    simulation.save()
    assert len(written) == 5
    file.write_text("{}")
    simulation.save()
    assert len(written) == 6 and not any(_vp_coefficients(file))
    file.unlink()
    simulation.save()
    assert len(written) == 7 and file.is_file()

    class Note:
        def __init__(self, text):
            self.text = text

        def to_fs(self, ctx=None):
            return {"text": self.text}

    # Content without a reliable identity is always written.
    simulation.extra["note"] = Note("first")
    simulation.save()
    simulation.save()
    assert len(written) == 9
    assert json.loads(file.read_text())["note"] == {"text": "first"}


def test_jobs_share_one_remote_staging_of_an_unchanged_simulation(
    tmp_path, monkeypatch
):
    from frequensolve.imaging.jobs import FWIOperatorJob
    from frequensolve.model.parameterization import (
        ParameterizedProperty,
        TensorHatControl,
    )
    from frequensolve.simulation.jobs import remote as module
    from tests.imaging_fakes import layered_simulation

    simulation = layered_simulation(tmp_path / "project", save=False)
    sediment = next(s for s in simulation.model.subdomains if s.name == "sediment")
    sediment.properties["vp"] = ParameterizedProperty(
        1900.0,
        id="vp",
        control=TensorHatControl(
            axes=("x", "z"),
            shape=(4, 3),
            origin=(0.0, 200.0),
            spacing=(1000.0, 500.0),
            coefficients=np.arange(12.0),
        ),
    )
    direction = tmp_path / "project" / "inputs" / "direction.h5"
    direction.parent.mkdir(parents=True)
    direction.write_bytes(b"direction")
    loads = []
    load = module.json.load
    monkeypatch.setattr(
        module.json, "load", lambda f, *a, **k: loads.append(f.name) or load(f, *a, **k)
    )

    def staged(name, remote="/remote/project"):
        job = FWIOperatorJob(
            name,
            simulation,
            [4.0],
            action="jvp",
            active=["model.vp"],
            state="s.json",
            objective_vector="jvp.json",
            direction=direction,
        )
        job.save_for_remote("SlurmSite", remote)
        path, _ = job.save_simulation_for_remote("SlurmSite", remote)
        return job, path

    first, first_path = staged("first")
    simulation_file = str(simulation._file)
    assert loads.count(simulation_file) == 1
    inputs = first.remote_input_files("/remote/project")
    second, second_path = staged("second")
    # The second job copies the first mapping; nothing is reloaded or remapped.
    assert loads.count(simulation_file) == 2  # the first job's input scan
    assert second_path != first_path
    assert second_path.read_bytes() == first_path.read_bytes()
    assert second_path.stat().st_mtime_ns == first_path.stat().st_mtime_ns
    assert second.staged_artifact_fingerprints("SlurmSite")["simulation"] == (
        first.staged_artifact_fingerprints("SlurmSite")["simulation"]
    )
    assert json.loads(second_path.read_text())["project_path"] == "/remote/project"
    assert second.downloaded_task_fingerprints() == [
        second.staged_task_fingerprints("SlurmSite")
    ]
    # Input scans read each simulation content once and find the same files.
    assert [Path(local) for local, _ in inputs] == [direction]
    assert second.remote_input_files("/remote/project") == inputs
    assert loads.count(simulation_file) == 2
    # Another remote project, a new simulation content, or a staged copy
    # changed since, are mapped again.
    _, other = staged("other", "/elsewhere")
    assert json.loads(other.read_text())["project_path"] == "/elsewhere"
    assert loads.count(simulation_file) == 3
    sediment.properties["vp"] = sediment.properties["vp"].with_coefficients(
        np.zeros(12)
    )
    _, changed = staged("changed")
    assert loads.count(simulation_file) == 4
    assert changed.read_bytes() != first_path.read_bytes()
    changed.write_text(changed.read_text() + "\n")
    third, third_path = staged("third")
    assert loads.count(simulation_file) == 5
    assert third_path.read_text() == changed.read_text()[:-1]
    assert third.downloaded_task_fingerprints() == [
        third.staged_task_fingerprints("SlurmSite")
    ]


def test_input_scans_of_a_simulation_skeleton_match_the_full_payload(tmp_path):
    from frequensolve.simulation.jobs.remote import (
        JobRemoteMixin,
        _reference_skeleton,
    )

    project = tmp_path / "project"
    payload = {
        "project_path": str(project),
        "Model": {
            "coefficients": [float(i) for i in range(1000)],
            "nested": [[1.0, 2.0], [3, {"file": "grid.rsf"}], True, None],
            "Subdomains": [
                {"name": "a", "properties": {"vp": {"file": str(project / "vp.h5")}}},
                {"value": 2.5, "receiver_file": "receivers.csv"},
            ],
        },
        "Acquisition": {"observed": "observed.h5", "weights": [0.5, 1.5]},
        "control_sensitivities": {"input": str(project / "jobs" / "x" / "in.h5")},
    }
    skeleton = _reference_skeleton(payload)
    assert "coefficients" in skeleton["Model"] and not skeleton["Model"]["coefficients"]
    assert list(JobRemoteMixin._iter_file_references(skeleton)) == list(
        JobRemoteMixin._iter_file_references(payload)
    )
    assert JobRemoteMixin._payload_project_roots(
        skeleton
    ) == JobRemoteMixin._payload_project_roots(payload)


def test_simulation_fingerprints_hash_each_saved_content_once(tmp_path, monkeypatch):
    from frequensolve.simulation.jobs import serialization
    from frequensolve.simulation.jobs.run_state import SkipPolicy

    job, _ = _born_job(tmp_path, n_tasks=3)
    BaseSite().prepare_job(job, validate=False)
    simulation = Path(job.simulation._file)
    loads = []
    load = serialization.json.load
    monkeypatch.setattr(
        serialization.json,
        "load",
        lambda f, *a, **k: loads.append(Path(f.name)) or load(f, *a, **k),
    )
    whole = job.fingerprint()
    tasks = [job.task_fingerprint(task) for task in (1, 2, 3)]
    compatible = [
        job.task_policy_fingerprint(task, SkipPolicy.compatible()) for task in (1, 2, 3)
    ]
    job.write_run_state(status="completed")
    # One read of the saved simulation per hashed view (whole, without Solver).
    assert loads.count(simulation) == 2
    job.save()
    assert job.fingerprint() == whole
    assert [job.task_fingerprint(task) for task in (1, 2, 3)] == tasks
    assert loads.count(simulation) == 2
    # A changed file is hashed again.
    payload = json.loads(simulation.read_text())
    payload["scaling"] = "robust"
    simulation.write_text(json.dumps(payload))
    assert job.fingerprint() != whole
    assert job.task_policy_fingerprint(1, SkipPolicy.compatible()) != compatible[0]
    assert loads.count(simulation) == 4
