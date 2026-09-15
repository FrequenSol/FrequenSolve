import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import sympy as sp

import frequensolve as fs
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.base import JobStatus, RunResult
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project.project import Project
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.artifact_contract import ArtifactRecord, task_result_path
from frequensolve.simulation.jobs import (
    BaseJob,
    FrequencyDomainJob,
    ImagingJob,
    JobLayout,
    TimeDomainJob,
)
from frequensolve.simulation.jobs.artifacts import RunMetadata, TraceManifest
from frequensolve.simulation.solver import SolverConfig


def _run_artifact(result_path, relative_path, *, representation):
    return ArtifactRecord.from_fs(
        {
            "id": str(relative_path),
            "role": "visualization" if representation == "vtk" else "trace",
            "representation": representation,
            "schema": "test-artifact-1",
            "path": str(relative_path),
            "retention": "durable",
            "bytes": 0,
        },
        result_path=result_path,
    )


def _project_with_trace_simulation(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    return project, sim


def _commit_task_result(
    job,
    task,
    *,
    state="success",
    code=0,
    artifact=True,
    solver=None,
    timings=None,
    resources=None,
):
    """Commit one exact v2 task result for lifecycle tests."""

    fingerprints = job._artifact_contract_fingerprints()
    assert fingerprints is not None
    records = []
    if artifact:
        relative = f"opaque/task-{task}.h5"
        payload = job._result_path / relative
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.touch()
        records.append(
            {
                "id": "traces",
                "role": "simulated_traces",
                "representation": "hdf5_shard",
                "schema": "fs-trace-shard-2",
                "path": relative,
                "retention": "durable",
                "bytes": payload.stat().st_size,
            }
        )
    result = task_result_path(job._result_path, task)
    result.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "fs-task-result-2",
        "partition": {
            "task": task,
            "task_count": job.n_tasks,
            "frequency": {
                "real": float(np.real(job.f_list[task - 1])),
                "imag": float(np.imag(job.f_list[task - 1])),
            },
        },
        "fingerprints": fingerprints,
        "status": {"state": state, "code": code},
        "artifacts": records,
    }
    for key, value in (
        ("solver", solver),
        ("timings", timings),
        ("resources", resources),
    ):
        if value:
            document[key] = value
    result.write_text(json.dumps(document))
    return job._result_path / records[0]["path"] if records else None


def test_project_save_load_uses_relative_simulation_paths(tmp_path):
    project, sim = _project_with_trace_simulation(tmp_path)

    project_file = project.save()
    payload = json.loads(project_file.read_text())
    loaded = Project.load(project_file)

    assert payload["simulations"] == ["simulations/simple/simple.json"]
    assert loaded.path == project.path
    assert loaded.simulations["simple"].name == sim.name
    assert loaded.simulations["simple"]._project is loaded


def test_job_layout_uses_base_only_for_relative_paths():
    relative = JobLayout.from_payload(
        {
            "project_path": "/shared/frequensolve",
            "simulation": "simulations/simple/simple.json",
            "result_path": "jobs/simple/freq/results",
            "name": "freq",
        }
    )
    absolute = JobLayout.from_payload(
        {
            "project_path": "/shared/frequensolve",
            "simulation": "/models/simple/simple.json",
            "result_path": "/scratch/results/simple/freq",
            "name": "freq",
        }
    )

    assert relative.simulation_file == Path(
        "/shared/frequensolve/simulations/simple/simple.json"
    )
    assert relative.result_dir == Path("/shared/frequensolve/jobs/simple/freq/results")
    assert absolute.simulation_file == Path("/models/simple/simple.json")
    assert absolute.result_dir == Path("/scratch/results/simple/freq")


def test_generic_load_dispatches_saved_project_job_and_simulation(tmp_path):
    project, sim = _project_with_trace_simulation(tmp_path)
    project_file = project.save()
    sim_file = sim.save()
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job_file = job.save()

    loaded_project = fs.load(project_file)
    loaded_sim = fs.load(sim_file)
    loaded_job = fs.load(job_file)

    assert isinstance(loaded_project, Project)
    assert loaded_project.path == project.path
    assert loaded_sim.name == sim.name
    assert isinstance(loaded_job, FrequencyDomainJob)
    assert loaded_job.name == job.name
    assert loaded_job.simulation.name == sim.name


def test_generic_load_infers_trace_store(tmp_path):
    trace_file = tmp_path / "traces.h5"
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(trace_file, "w") as h5:
        h5.create_dataset("frequency", data=np.array([10.0]))
        dset = h5.create_dataset(
            "surface",
            data=np.zeros((1, 1, 1, 1), dtype=np.float32),
        )
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([1], dtype=np.int32)

    traces = fs.load(trace_file)

    assert isinstance(traces, TraceDataset)
    assert traces.manifest.groups == ["surface"]
    assert traces.manifest.frequencies == {1: 10.0}


def test_successful_run_result_opens_images_through_its_site():
    expected = object()
    job = object.__new__(ImagingJob)
    site = SimpleNamespace(fetch_image=lambda requested: expected)
    result = RunResult(
        job=job,
        status=JobStatus(state="completed", return_code=0),
        site=site,
    )

    assert result.images() is expected


def test_project_save_serializes_solver_hp_sympy_policy(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    epw = sp.Symbol("epw")
    sim += SolverConfig(
        refinements=["h", "h", "h", "a"],
        hp={
            "order": sp.Piecewise(
                (4, epw < 2.0),
                (3, True),
            ),
            "overrides": [
                {
                    "classifier": "physics",
                    "value": "acoustic",
                    "order": sp.Piecewise(
                        (5, epw < 1.5),
                        (4, True),
                    ),
                },
            ],
            "order_x": {"min": 2, "max": 4},
        },
    )

    project_file = project.save()
    payload = json.loads(project_file.read_text())
    sim_payload = json.loads((project.path / payload["simulations"][0]).read_text())

    assert sim_payload["Solver"]["hp"] == {
        "order": {
            "op": "case",
            "branches": [
                {
                    "if": {"op": "<", "args": [{"var": "epw"}, {"value": 2.0}]},
                    "then": 4,
                }
            ],
            "else": 3,
        },
        "overrides": [
            {
                "classifier": "physics",
                "value": "acoustic",
                "order": {
                    "op": "case",
                    "branches": [
                        {
                            "if": {
                                "op": "<",
                                "args": [{"var": "epw"}, {"value": 1.5}],
                            },
                            "then": 5,
                        }
                    ],
                    "else": 4,
                },
            },
        ],
        "order_x": {"min": 2, "max": 4},
    }


def test_project_api_rejects_auto_migrate_option(tmp_path):
    project, _ = _project_with_trace_simulation(tmp_path)
    project_file = project.save()

    with pytest.raises(TypeError):
        Project.load(project_file, auto_migrate=True)
    with pytest.raises(TypeError):
        Project(name="project", path=tmp_path / "other", auto_migrate=True)


def test_loaded_copied_job_uses_explicit_project_override(tmp_path):
    original = Project(name="project", path=tmp_path / "original")
    sim = original.new_simulation(name="simple", physics="acoustic", dimension=2)
    mesh_file = original.path / "simulations" / "simple" / "mesh.gmp"
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    mesh_file.write_text("mesh")
    sim.mesh = MeshManager(file=mesh_file, format="Gmsh")
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()

    copied_root = tmp_path / "copied"
    shutil.copytree(original.path, copied_root)
    copied_job_file = copied_root / "jobs" / "simple" / "freq" / "freq.json"
    copied_sim_file = copied_root / "simulations" / "simple" / "simple.json"

    job_payload = json.loads(copied_job_file.read_text())
    job_payload["project_path"] = str(original.path)
    job_payload["simulation"] = str(
        original.path / "simulations" / "simple" / "simple.json"
    )
    copied_job_file.write_text(json.dumps(job_payload))

    sim_payload = json.loads(copied_sim_file.read_text())
    sim_payload["project_path"] = str(original.path)
    copied_sim_file.write_text(json.dumps(sim_payload))

    loaded = BaseJob.load(copied_job_file, project_path=copied_root)
    assert loaded.project_path == copied_root.resolve()
    assert loaded.simulation._file == copied_sim_file.resolve()

    loaded.save()
    job_payload = json.loads(copied_job_file.read_text())
    sim_payload = json.loads(copied_sim_file.read_text())

    assert job_payload["project_path"] == str(copied_root.resolve())
    assert job_payload["simulation"] == "simulations/simple/simple.json"
    assert sim_payload["project_path"] == str(copied_root.resolve())
    assert str(original.path) not in json.dumps(job_payload)
    assert str(original.path) not in json.dumps(sim_payload)

    staged_job, remote_job = loaded.save_for_remote("Dummy", Path("/scratch/run"))
    staged_sim, remote_sim = loaded.save_simulation_for_remote(
        "Dummy", Path("/scratch/run")
    )
    staged_job_payload = json.loads(Path(staged_job).read_text())
    staged_sim_payload = json.loads(Path(staged_sim).read_text())

    assert remote_job == Path("/scratch/run/jobs/simple/freq/freq.json")
    assert remote_sim == Path("/scratch/run/simulations/simple/simple.json")
    assert staged_job_payload["project_path"] == "/scratch/run"
    assert staged_job_payload["simulation"] == str(remote_sim)
    assert staged_sim_payload["project_path"] == "/scratch/run"
    assert str(original.path) not in json.dumps(staged_job_payload)
    assert str(original.path) not in json.dumps(staged_sim_payload)
    assert str(copied_root.resolve()) not in json.dumps(staged_job_payload)
    assert str(copied_root.resolve()) not in json.dumps(staged_sim_payload)


def test_remote_staging_rewrites_stale_absolute_artifact_roots(tmp_path, monkeypatch):
    original = Project(name="project", path=tmp_path / "original")
    sim = original.new_simulation(name="simple", physics="acoustic", dimension=2)
    mesh_file = original.path / "simulations" / "simple" / "mesh.gmp"
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    mesh_file.write_text("mesh")
    sim.mesh = MeshManager(file=mesh_file, format="Gmsh")
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()

    copied_root = tmp_path / "copied"
    shutil.copytree(original.path, copied_root)
    copied_job_file = copied_root / "jobs" / "simple" / "freq" / "freq.json"
    copied_sim_file = copied_root / "simulations" / "simple" / "simple.json"
    loaded = BaseJob.load(copied_job_file, project_path=copied_root)
    loaded.save()

    sim_payload = json.loads(copied_sim_file.read_text())
    sim_payload["Mesh"]["file"] = str(mesh_file)
    copied_sim_file.write_text(json.dumps(sim_payload))
    monkeypatch.setattr(loaded.simulation, "save", lambda: copied_sim_file)
    staged_sim, _ = loaded.save_simulation_for_remote("Dummy", Path("/scratch/run"))
    staged_payload = json.loads(Path(staged_sim).read_text())

    assert staged_payload["Mesh"]["file"] == "/scratch/run/simulations/simple/mesh.gmp"
    assert str(original.path) not in json.dumps(staged_payload)
    assert str(copied_root.resolve()) not in json.dumps(staged_payload)


def test_remote_input_files_maps_stale_absolute_refs_to_copied_inputs(tmp_path):
    original = Project(name="project", path=tmp_path / "original")
    sim = original.new_simulation(name="simple", physics="acoustic", dimension=2)
    mesh_file = original.path / "simulations" / "simple" / "mesh.gmp"
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    mesh_file.write_text("mesh")
    sim.mesh = MeshManager(file=mesh_file, format="Gmsh")
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()

    copied_root = tmp_path / "copied"
    shutil.copytree(original.path, copied_root)
    copied_job_file = copied_root / "jobs" / "simple" / "freq" / "freq.json"
    copied_sim_file = copied_root / "simulations" / "simple" / "simple.json"
    loaded = BaseJob.load(copied_job_file, project_path=copied_root)
    loaded.save()

    sim_payload = json.loads(copied_sim_file.read_text())
    sim_payload["Mesh"]["file"] = str(mesh_file)
    copied_sim_file.write_text(json.dumps(sim_payload))
    shutil.rmtree(original.path)

    pairs = loaded.remote_input_files(Path("/scratch/run"))

    assert (
        copied_root.resolve() / "simulations" / "simple" / "mesh.gmp",
        Path("/scratch/run/simulations/simple/mesh.gmp"),
    ) in pairs


def test_project_copy_rewrites_saved_job_and_simulation_roots(tmp_path):
    original = Project(name="project", path=tmp_path / "original")
    sim = original.new_simulation(name="simple", physics="acoustic", dimension=2)
    mesh_file = original.path / "simulations" / "simple" / "mesh.gmp"
    mesh_file.parent.mkdir(parents=True, exist_ok=True)
    mesh_file.write_text("mesh")
    sim.mesh = MeshManager(file=mesh_file, format="Gmsh")
    FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0]).save()
    original.save()
    stale_root = tmp_path / "stale"
    original_job_file = original.path / "jobs" / "simple" / "freq" / "freq.json"
    original_sim_file = original.path / "simulations" / "simple" / "simple.json"
    job_payload = json.loads(original_job_file.read_text())
    job_payload["project_path"] = str(stale_root)
    job_payload["simulation"] = str(stale_root / "simulations/simple/simple.json")
    job_payload["result_path"] = str(stale_root / "jobs/simple/freq/results")
    original_job_file.write_text(json.dumps(job_payload))
    sim_payload = json.loads(original_sim_file.read_text())
    sim_payload["project_path"] = str(stale_root)
    original_sim_file.write_text(json.dumps(sim_payload))

    copied = Project.copy(original.path, tmp_path / "copied")
    copied_job_file = copied.path / "jobs" / "simple" / "freq" / "freq.json"
    copied_sim_file = copied.path / "simulations" / "simple" / "simple.json"
    job_payload = json.loads(copied_job_file.read_text())
    sim_payload = json.loads(copied_sim_file.read_text())

    assert job_payload["project_path"] == str(copied.path)
    assert job_payload["simulation"] == "simulations/simple/simple.json"
    assert job_payload["result_path"] == "jobs/simple/freq/results"
    assert sim_payload["project_path"] == str(copied.path)
    assert str(original.path) not in json.dumps(job_payload)
    assert str(original.path) not in json.dumps(sim_payload)
    assert str(stale_root) not in json.dumps(job_payload)
    assert str(stale_root) not in json.dumps(sim_payload)


def test_run_metadata_filters_output_files(tmp_path):
    result_path = tmp_path / "results"
    metadata = RunMetadata(
        result_path=result_path,
        artifacts=tuple(
            _run_artifact(
                result_path,
                relative,
                representation=("vtk" if relative.endswith(".vtu") else "hdf5"),
            )
            for relative in (
                "ParaView/pv_00000.vtu",
                "ParaView/pv_coarse_00000.vtu",
                "ParaView/pv_coarse_00001.vtu",
                "ParaView/pv_fine_00000.vtu",
                "ParaView/pressure_1.vtu",
                "ParaView/pressure_2.vtu",
                "traces/traces_1.h5",
            )
        ),
    )

    assert metadata.output_files(kind="vtk", suffix=".vtu") == [
        result_path / "ParaView/pv_00000.vtu",
        result_path / "ParaView/pv_coarse_00000.vtu",
        result_path / "ParaView/pv_coarse_00001.vtu",
        result_path / "ParaView/pv_fine_00000.vtu",
        result_path / "ParaView/pressure_1.vtu",
        result_path / "ParaView/pressure_2.vtu",
    ]
    assert metadata.output_files(kind="vtk", suffix=".vtu", base="pv_coarse") == [
        result_path / "ParaView/pv_coarse_00000.vtu",
        result_path / "ParaView/pv_coarse_00001.vtu",
    ]
    assert metadata.output_files(kind="vtk", suffix=".vtu", base="pv_fine") == [
        result_path / "ParaView/pv_fine_00000.vtu"
    ]
    assert metadata.output_files(kind="vtk", suffix=".vtu", base="pv") == [
        result_path / "ParaView/pv_00000.vtu"
    ]
    assert metadata.output_files(kind="vtk", suffix=".vtu", base="pressure_2") == [
        result_path / "ParaView/pressure_2.vtu"
    ]
    assert metadata.output_files(
        kind="vtk",
        suffix=".vtu",
        base="pressure_2.vtu",
    ) == [result_path / "ParaView/pressure_2.vtu"]
    result = RunResult(
        job=object(),
        status=JobStatus(state="completed", return_code=0),
        run_metadata=metadata,
    )
    assert result.output_files(kind="vtk", suffix=".vtu", base="pressure_1") == [
        result_path / "ParaView/pressure_1.vtu"
    ]


def test_trace_manifest_prefers_producer_reported_task_artifact(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.save()
    reported = _commit_task_result(job, 1)

    manifest = TraceManifest.from_job(job)

    assert manifest.files == [reported]
    assert manifest.artifacts[0].path == reported


def test_run_metadata_does_not_discover_unregistered_vtu_files(tmp_path):
    result_path = tmp_path / "results"
    paraview = result_path / "ParaView"
    paraview.mkdir(parents=True)
    (paraview / "pv_coarse_00000.vtu").touch()
    (paraview / "pv_fine_00000.vtu").touch()
    metadata = RunMetadata(
        result_path=result_path,
        artifacts=(
            _run_artifact(
                result_path,
                "traces/traces.h5",
                representation="hdf5",
            ),
        ),
    )

    assert metadata.output_files(base="pv_coarse", suffix=".vtu") == []
    assert metadata.output_files(kind="vtu", base="pv_fine") == []
    assert metadata.output_files(kind="vtk", suffix=".vtu", base="pv_coarse") == []


def test_run_metadata_deduplicates_existing_output_file_aliases(tmp_path):
    result_path = tmp_path / "results"
    paraview = result_path / "ParaView"
    lower_paraview = result_path / "paraview"
    paraview.mkdir(parents=True)
    canonical = paraview / "pv_00000.vtu"
    alias = lower_paraview / "pv_00000.vtu"
    canonical.write_text("<VTKFile></VTKFile>")
    if not alias.exists():
        lower_paraview.mkdir(parents=True, exist_ok=True)
        try:
            os.link(canonical, alias)
        except OSError as exc:
            pytest.skip(f"filesystem does not support hard links: {exc}")

    metadata = RunMetadata(
        result_path=result_path,
        artifacts=(
            _run_artifact(
                result_path,
                "ParaView/pv_00000.vtu",
                representation="vtk",
            ),
            _run_artifact(
                result_path,
                "paraview/pv_00000.vtu",
                representation="vtk",
            ),
        ),
    )

    assert metadata.output_files(base="pv", suffix=".vtu", existing=True) == [canonical]


def test_job_vtk_outputs_resolve_under_result_directory(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0])
    job.vtk("quicklook", path="paraview/qc", fields="pressure")
    job.save()

    assert job.vtk_outputs == {"quicklook": job._result_path / "paraview/qc"}
    assert job.paraview_outputs == job.vtk_outputs


def test_job_save_load_persists_required_simulation_inputs(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])

    job_file = job.save()
    payload = json.loads(job_file.read_text())
    loaded = BaseJob.load(job_file)

    assert sim._file.exists()
    assert job_file.exists()
    assert payload["simulation"] == "simulations/simple/simple.json"
    assert payload["Outputs"]["traces"]["path"] == "traces"
    assert "overwrite" not in payload
    assert "max_versions" not in payload
    assert not (job_file.parent / "manifest.json").exists()
    assert loaded.name == "freq"
    assert loaded.simulation.name == "simple"
    assert loaded.f_list == [1.0, 2.0]
    assert loaded._file == job_file


def test_job_loading_accepts_job_object_and_job_directory(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = TimeDomainJob(name="time", simulation=sim, f_max=5.0, T_max=1.0)

    job_file = job.save()
    loaded_from_object = BaseJob.load(job)
    loaded_from_method = job.load_saved()
    loaded_from_dir = BaseJob.load(job_file.parent)

    assert job.job_file == job_file
    assert loaded_from_object._file == job_file
    assert loaded_from_method._file == job_file
    assert loaded_from_dir._file == job_file
    assert loaded_from_object.name == "time"
    assert loaded_from_object.simulation.name == sim.name


def test_project_load_job_finds_saved_job_by_simulation_and_unique_name(tmp_path):
    project, sim = _project_with_trace_simulation(tmp_path)
    time_job = TimeDomainJob(name="time", simulation=sim, f_max=5.0, T_max=1.0)
    freq_job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])

    time_file = time_job.save()
    freq_file = freq_job.save()

    assert project.job_file("time") == time_file
    assert project.job_file("freq", simulation=sim) == freq_file
    assert project.job_file("freq", simulation="simple") == freq_file
    assert project.load_job("time")._file == time_file
    loaded = project.load_job("freq", simulation="simple")
    assert loaded._file == freq_file
    assert loaded.simulation is project.simulations["simple"]
    assert loaded.simulation is sim
    assert loaded.simulation._project is project
    loaded.simulation.dimension = 3
    project.save()
    assert json.loads(sim._file.read_text())["dimension"] == 3


def test_project_list_jobs_reports_result_status(tmp_path):
    project, sim = _project_with_trace_simulation(tmp_path)
    freq_job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])

    job_file = freq_job.save()
    [row] = project.list_jobs()

    assert row["name"] == "freq"
    assert row["simulation"] == "simple"
    assert row["job_type"] == "FrequencyDomainJob"
    assert row["workflow"] == "forward"
    assert row["n_tasks"] == 1
    assert row["job_file"] == str(job_file)
    assert row["relative_job_file"] == "jobs/simple/freq/freq.json"
    assert row["loaded"] is True
    assert row["results_exist"] is False
    assert row["results_current"] is False

    _commit_task_result(freq_job, 1)
    freq_job.write_run_state(status="completed")

    [row] = project.list_jobs(simulation=sim)
    assert row["results_exist"] is True
    assert row["trace_outputs_exist"] is True
    assert row["results_current"] is True
    assert row["run_status"] == "completed"
    assert row["task_summary"]["complete"] == 1


def test_frequency_domain_job_normalizes_laplace_sign(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)

    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0 + 0.25j])

    assert job.f_list == [1.0 - 0.25j]


def test_time_domain_job_requires_valid_sampling(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)

    with pytest.raises(ValueError, match="either df or T_max"):
        TimeDomainJob(name="td", simulation=sim, f_max=5.0)
    with pytest.raises(ValueError, match="df must be positive"):
        TimeDomainJob(name="td", simulation=sim, f_max=5.0, df=0.0)
    with pytest.raises(ValueError, match="f_max must be greater"):
        TimeDomainJob(name="td", simulation=sim, f_min=5.0, f_max=5.0, df=1.0)
    with pytest.raises(ValueError, match="damping_factor"):
        TimeDomainJob(
            name="td", simulation=sim, f_max=5.0, T_max=1.0, damping_factor=0.5
        )
    with pytest.raises(ValueError, match="only one of damping_factor or laplace"):
        TimeDomainJob(
            name="td",
            simulation=sim,
            f_max=5.0,
            T_max=1.0,
            damping_factor=10.0,
            laplace=-0.1,
        )


def test_time_domain_job_supports_damping_factor_and_direct_laplace(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)

    job = TimeDomainJob(
        name="td",
        simulation=sim,
        f_min=0.0,
        f_max=1.0,
        T_max=2.0,
        damping_factor=10.0,
    )
    expected_laplace = -math.log(10.0) / (2.0 * math.pi * 2.0)

    assert [freq.real for freq in job.f_list] == [0.5, 1.0]
    assert [freq.imag for freq in job.f_list] == pytest.approx(
        [expected_laplace, expected_laplace]
    )

    direct = TimeDomainJob(
        name="td_direct",
        simulation=sim,
        f_min=0.0,
        f_max=1.0,
        df=0.5,
        laplace=0.25,
    )

    assert [freq.real for freq in direct.f_list] == [0.5, 1.0]
    assert [freq.imag for freq in direct.f_list] == pytest.approx([-0.25, -0.25])


def test_time_domain_job_supports_sparse_hermite_reconstruction(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = TimeDomainJob(
        name="td_hermite",
        simulation=sim,
        f_min=1.0,
        f_max=25.0,
        df=1.0,
        reconstruction="hermite",
        sample_every=4,
        high_frequency_taper=4.0,
        interpolation_time_shift=0.35,
    )

    assert job.workflow == "forward_df"
    assert job.phase_derivatives == 1
    assert job.f_list == [1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 25.0]
    assert len(job.f_list) < 0.3 * 25
    assert job.time_reconstruction == {
        "method": "hermite",
        "target_df": 1.0,
        "sample_every": 4,
        "high_frequency_taper": 4.0,
        "interpolation_time_shift": 0.35,
    }

    job_file = job.save()
    payload = json.loads(job_file.read_text())
    loaded = BaseJob.load(job_file)

    assert payload["workflow"] == "forward_df"
    assert payload["derivative_order"] == 1
    assert payload["time_reconstruction"] == job.time_reconstruction
    assert loaded.f_list == job.f_list
    assert loaded.phase_derivatives == 1
    assert loaded.time_reconstruction == job.time_reconstruction

    legacy_payload = json.loads(job_file.read_text())
    legacy_reconstruction = legacy_payload["time_reconstruction"]
    legacy_reconstruction["frequency_reduction"] = legacy_reconstruction.pop(
        "sample_every"
    )
    job_file.write_text(json.dumps(legacy_payload))
    legacy_loaded = BaseJob.load(job_file)

    assert legacy_loaded.sample_every == 4
    assert legacy_loaded.time_reconstruction["sample_every"] == 4
    assert "frequency_reduction" not in legacy_loaded.time_reconstruction


def test_time_domain_job_hermite_frequencies_are_exact_fine_grid_subset(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    fine = TimeDomainJob(
        name="td_fine",
        simulation=sim,
        f_min=0.0,
        f_max=35.0,
        T_max=3.0,
    )
    quarter = TimeDomainJob(
        name="td_quarter",
        simulation=sim,
        f_min=0.0,
        f_max=25.0,
        T_max=3.0,
        reconstruction="hermite",
        sample_every=4,
    )

    fine_through_25 = [frequency for frequency in fine.f_list if frequency <= 25.0]
    expected = fine_through_25[::4]
    if expected[-1] != fine_through_25[-1]:
        expected.append(fine_through_25[-1])

    assert quarter.f_list == expected
    assert all(frequency in fine.f_list for frequency in quarter.f_list)


def test_local_submit_autosaves_job_and_simulation(monkeypatch, tmp_path):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    site = LocalSite()

    monkeypatch.setattr(site, "_submit_local_tasks", lambda job, **kwargs: [])

    run = site.submit(job)

    assert run.job is job
    assert sim._file.exists()
    assert job._file.exists()


def test_local_fetch_logs_selects_task_and_frequency(monkeypatch, tmp_path):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/bin/echo")
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0, 20.0])
    job.save()
    job._stdout_path.mkdir(parents=True)
    first = job._stdout_path / "task_1.log"
    second = job._stdout_path / "task_2.log"
    first.write_text("ten hertz")
    second.write_text("twenty hertz")

    site = LocalSite()

    assert site.fetch_logs(job) == job._stdout_path
    assert site.fetch_logs(job, task=2) == second
    assert site.fetch_logs(job, frequency=10.0) == first
    assert site.fetch_logs([job], frequency=20.0) == {"freq": second}


def test_job_frequency_status_summary_and_task_timings(tmp_path, capsys):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0, 3.0])
    job.save()

    trace_file = _commit_task_result(
        job,
        1,
        timings={"solve_forward": 1.25},
    )
    _commit_task_result(
        job,
        2,
        state="failed",
        code=1,
        artifact=False,
        timings={"solve_forward": 2.5},
    )

    rows = job.frequency_status()

    assert [row["status"] for row in rows] == ["succeeded", "failed", "not_run"]
    assert rows[0]["task"] == 1
    assert rows[0]["frequency"] == 1.0
    assert rows[1]["duration_seconds"] == 2.5
    assert job.frequency_summary() == {
        "total": 3,
        "succeeded": 1,
        "failed": 1,
        "not_run": 1,
    }

    returned = job.print_frequency_summary()
    captured = capsys.readouterr()

    assert returned["failed"] == 1
    assert "Job freq: 1/3 frequencies succeeded; 1 failed; 1 not run." in captured.out
    assert job.task_timings() == [
        {
            "task": 1,
            "frequency": 1.0,
            "duration_seconds": 1.25,
            "core_count": None,
            "core_hours": None,
            "status": "succeeded",
            "trace_file": trace_file,
        },
        {
            "task": 2,
            "frequency": 2.0,
            "duration_seconds": 2.5,
            "core_count": None,
            "core_hours": None,
            "status": "failed",
            "trace_file": None,
        },
    ]


def test_job_failed_tasks_reports_reasons(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0, 3.0])
    job.save()

    job.write_run_state(
        status="completed",
        tasks=[
            {
                "task_id": 0,
                "status": "error",
                "complete": True,
                "error": "mesh generation failed",
            },
            {
                "task_id": 1,
                "status": "success",
                "complete": True,
                "solver": {
                    "convergence": {
                        "converged": True,
                        "status": "converged",
                        "solve_count": 1,
                        "failure_count": 0,
                        "worst_code": 0,
                        "solves": [
                            {
                                "converged": True,
                                "iterations": 24,
                                "residual": 2.0e-3,
                                "status": "converged",
                            }
                        ],
                    }
                },
            },
        ],
    )

    failures = job.failed_tasks()

    assert [row["task"] for row in failures] == [1, 2]
    assert job.list_failed_tasks() == failures
    assert failures[0]["frequency"] == 1.0
    assert failures[0]["reason"] == "mesh generation failed"
    assert failures[1]["frequency"] == 2.0
    assert failures[1]["reason"] == (
        "Solver residual 0.002 exceeded failure threshold 0.001 after 24 iterations."
    )
    assert failures[1]["solver"]["convergence"]["residual"] == 0.002


def test_job_plot_task_timings(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    job.save()
    job.write_run_state(
        status="completed",
        tasks=[
            {"task_id": 0, "status": "success", "duration_seconds": 1.0},
            {"task_id": 1, "status": "success", "duration_seconds": 2.0},
        ],
    )

    ax = job.plot_task_timings()

    assert ax.get_xlabel() == "Frequency (Hz)"
    assert ax.get_ylabel() == "Runtime (s)"
    assert len(ax.patches) == 2


def test_job_task_timings_accepts_native_elapsed_s(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[5.0, 10.0])
    job.save()
    for task, elapsed in enumerate((0.5, 1.25), 1):
        file = job._result_path / f"_fs_run/tasks/task_{task:06d}/result.json"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(
            json.dumps(
                {
                    "schema": "fs-task-result-2",
                    "partition": {
                        "task": task,
                        "task_count": 2,
                        "frequency": {"real": job.f_list[task - 1], "imag": 0.0},
                    },
                    "fingerprints": job._artifact_contract_fingerprints(),
                    "status": {"state": "success", "code": 0},
                    "timings": {"elapsed": elapsed},
                    "artifacts": [],
                }
            )
        )

    assert [row["duration_seconds"] for row in job.task_timings()] == [0.5, 1.25]


def test_job_plot_task_timings_uses_sparse_ticks(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(
        name="freq",
        simulation=sim,
        f_list=[float(i) for i in range(1, 41)],
    )
    job.save()
    job.write_run_state(
        status="completed",
        tasks=[
            {
                "task_id": index,
                "status": "success",
                "duration_seconds": 10.0,
            }
            for index in range(40)
        ],
    )

    ax = job.plot_task_timings(max_xticks=6)

    assert ax.get_xlabel() == "Frequency (Hz)"
    assert len(ax.get_xticks()) <= 8


def test_job_plot_task_timings_switches_to_lines_for_large_sweeps(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(
        name="freq",
        simulation=sim,
        f_list=[float(i) for i in range(1, 121)],
    )
    job.save()
    job.write_run_state(
        status="completed",
        tasks=[
            {"task_id": index, "status": "success", "duration_seconds": index + 1.0}
            for index in range(120)
        ],
    )

    ax = job.plot_task_timings(max_xticks=6)

    assert len(ax.lines) == 1
    assert len(ax.patches) == 0
    assert len(ax.get_xticks()) <= 8


def test_job_plot_task_timings_supports_core_hours(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0, 20.0])
    job.save()
    job.write_run_state(
        status="completed",
        tasks=[
            {
                "task_id": 0,
                "status": "success",
                "duration_seconds": 1800.0,
                "n_ranks": 2,
                "threads_per_rank": 4,
            },
            {
                "task_id": 1,
                "status": "success",
                "duration_seconds": 3600.0,
                "n_ranks": 1,
                "threads_per_rank": 16,
            },
        ],
    )

    timings = job.task_timings()
    assert timings[0]["core_count"] == 8.0
    assert timings[0]["core_hours"] == 4.0
    assert timings[1]["core_count"] == 16.0
    assert timings[1]["core_hours"] == 16.0

    ax = job.plot_task_timings(unit="core-hours")

    assert ax.get_ylabel() == "Runtime (core-hours)"
    assert [patch.get_height() for patch in ax.patches] == [4.0, 16.0]


def test_job_phase_timings_and_plot(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[5.0, 10.0])
    job.save()
    _commit_task_result(
        job,
        1,
        timings={"setup": 0.2, "assembly": 1.0, "solve_forward": 2.0},
    )
    _commit_task_result(
        job,
        2,
        timings={"setup": 0.3, "assembly": 1.5, "solve_forward": 3.0},
    )

    rows = job.phase_timings(phases=["setup", "assembly", "solve_forward"])

    assert rows == [
        {
            "task": 1,
            "frequency": 5.0,
            "status": "succeeded",
            "total_seconds": 3.2,
            "setup": 0.2,
            "assembly": 1.0,
            "solve_forward": 2.0,
        },
        {
            "task": 2,
            "frequency": 10.0,
            "status": "succeeded",
            "total_seconds": 4.8,
            "setup": 0.3,
            "assembly": 1.5,
            "solve_forward": 3.0,
        },
    ]

    ax = job.plot_phase_timings(phases=["assembly", "solve_forward"])

    assert ax.get_xlabel() == "Frequency (Hz)"
    assert ax.get_ylabel() == "Runtime (s)"
    assert len(ax.patches) == 4
