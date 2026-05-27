import json
import math
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.base import JobStatus, RunResult
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project.project import Project
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.jobs import (
    BaseJob,
    FrequencyDomainJob,
    TimeDomainJob,
)
from frequensolve.simulation.jobs.artifacts import RunMetadata


def _project_with_trace_simulation(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.mesh = MeshManager(HexMeshGenerator(l_bound=[0, 0], u_bound=[1, 1], n=[1, 1]))
    return project, sim


def test_project_save_load_uses_relative_simulation_paths(tmp_path):
    project, sim = _project_with_trace_simulation(tmp_path)

    project_file = project.save()
    payload = json.loads(project_file.read_text())
    loaded = Project.load(project_file)

    assert payload["simulations"] == ["simulations/simple/simple.json"]
    assert loaded.path == project.path
    assert loaded.simulations["simple"].name == sim.name
    assert loaded.simulations["simple"]._project is loaded


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
        outputs={
            "files": [
                {
                    "relative_path": "ParaView/pv_00000.vtu",
                    "kind": "vtk",
                },
                {
                    "relative_path": "ParaView/pv_coarse_00000.vtu",
                    "kind": "vtk",
                },
                {
                    "relative_path": "ParaView/pv_coarse_00001.vtu",
                    "kind": "vtk",
                },
                {
                    "relative_path": "ParaView/pv_fine_00000.vtu",
                    "kind": "vtk",
                },
                {
                    "relative_path": "ParaView/pressure_1.vtu",
                    "kind": "vtk",
                },
                {
                    "relative_path": "ParaView/pressure_2.vtu",
                    "kind": "vtk",
                },
                {
                    "relative_path": "traces/traces_1.h5",
                    "kind": "hdf5",
                },
            ]
        },
        result_path=result_path,
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


def test_run_metadata_discovers_unregistered_vtu_files(tmp_path):
    result_path = tmp_path / "results"
    paraview = result_path / "ParaView"
    paraview.mkdir(parents=True)
    coarse = paraview / "pv_coarse_00000.vtu"
    fine = paraview / "pv_fine_00000.vtu"
    coarse.touch()
    fine.touch()
    metadata = RunMetadata(
        outputs={
            "files": [
                {
                    "relative_path": "traces/traces.h5",
                    "kind": "hdf5",
                },
            ]
        },
        result_path=result_path,
    )

    assert metadata.output_files(base="pv_coarse", suffix=".vtu") == [coarse]
    assert metadata.output_files(kind="vtu", base="pv_fine") == [fine]
    assert metadata.output_files(kind="vtk", suffix=".vtu", base="pv_coarse") == [
        coarse
    ]


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
        outputs={
            "files": [
                {
                    "relative_path": "ParaView/pv_00000.vtu",
                    "kind": "vtk",
                },
                {
                    "relative_path": "paraview/pv_00000.vtu",
                    "kind": "vtk",
                },
            ]
        },
        result_path=result_path,
    )

    assert metadata.output_files(base="pv", suffix=".vtu", existing=True) == [canonical]


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
    assert project.load_job("freq", simulation="simple")._file == freq_file


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

    trace_file = freq_job.expected_trace_files()[0]
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.touch()
    freq_job.write_run_state(status="completed")

    [row] = project.list_jobs(simulation=sim)
    assert row["results_exist"] is True
    assert row["trace_outputs_exist"] is True
    assert row["results_current"] is True
    assert row["run_status"] == "completed"
    assert row["task_summary"]["complete"] == 1


def test_job_traces_open_prefers_existing_packed_trace_file(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()

    trace_dir = job._result_path / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "packed": {"path": "traces/traces.h5"},
                "frequencies": [{"task_id": 1, "frequency": 1.0}],
            }
        )
    )
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(trace_dir / "traces.h5", "w") as h5:
        h5.create_dataset("frequency", data=np.array([1.0]))
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        dset = h5.create_dataset(
            "surface",
            data=np.zeros((1, 1, 1, 1, 2), dtype=np.float32),
        )
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["layout_kind"] = ["dense_trace_v1"]
        dset.attrs["receiver"] = np.array([101], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([7], dtype=np.int32)

    traces = job.traces.open()

    assert traces.files == [str(trace_dir / "traces.h5")]
    assert traces.groups == ["surface"]


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


def test_local_submit_autosaves_job_and_simulation(monkeypatch, tmp_path):
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

    trace_file = job.trace_manifest.files[0]
    trace_file.parent.mkdir(parents=True)
    trace_file.touch()
    job.write_run_state(
        status="failed",
        tasks=[
            {"task_id": 0, "status": "success", "duration_seconds": 1.25},
            {"task_id": 1, "status": "error", "duration_seconds": 2.5},
        ],
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
            "trace_file": job.trace_manifest.files[1],
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
        "Solver residual 0.002 exceeded failure threshold 0.001 after " "24 iterations."
    )
    assert failures[1]["solver"]["convergence"]["residual"] == 0.002


def test_job_run_state_summarizes_solver_convergence(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0, 3.0])
    job.save()
    for trace_file in job.trace_manifest.files[:2]:
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.touch()
    solver_manifest = job._result_path / "_fs_run" / "run_manifest.json"
    solver_manifest.parent.mkdir(parents=True, exist_ok=True)
    solver_manifest.write_text(
        json.dumps(
            {
                "schema": "fs-run-manifest-1",
                "solver": {
                    "name": "solver",
                    "convergence": {
                        "converged": True,
                        "failure_count": 0,
                        "solve_count": 0,
                        "status": "not_run",
                        "worst_code": 0,
                    },
                },
            }
        )
    )

    job.write_run_state(
        status="completed",
        tasks=[
            {
                "task_id": 0,
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
                                "context": "forward",
                                "converged": True,
                                "iterations": 16,
                                "residual": 8.665320086047467e-05,
                                "solver": "FS_MG",
                                "status": "converged",
                                "tolerance": 1.0e-4,
                            }
                        ],
                    }
                },
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
                                "context": "forward",
                                "converged": True,
                                "iterations": 24,
                                "residual": 1.1e-3,
                                "solver": "FS_MG",
                                "status": "converged",
                                "tolerance": 1.0e-4,
                            }
                        ],
                    }
                },
            },
        ],
    )

    payload = json.loads(job.run_state_file.read_text())

    assert payload["status"] == "completed"
    assert payload["task_summary"] == {
        "total": 3,
        "complete": 2,
        "succeeded": 1,
        "failed": 1,
        "not_run": 1,
    }
    assert [row["status"] for row in payload["tasks"]] == [
        "succeeded",
        "failed",
        "not_run",
    ]
    first = payload["tasks"][0]["solver"]["convergence"]
    second = payload["tasks"][1]["solver"]["convergence"]
    assert first["iterations"] == 16
    assert first["residual"] == 8.665e-05
    assert first["failed"] is False
    assert second["residual"] == 0.0011
    assert second["failed"] is True
    convergence = payload["solver"]["convergence"]
    assert convergence["status"] == "failed"
    assert convergence["converged"] is False
    assert convergence["solve_count"] == 2
    assert convergence["failure_count"] == 1
    assert convergence["worst_code"] == 0
    assert convergence["iterations"] == 40
    assert convergence["residual"] == 0.0011
    assert convergence["tasks"] == [
        {
            "task": 1,
            "frequency": 1.0,
            "converged": True,
            "iterations": 16,
            "residual": 8.665e-05,
            "status": "converged",
        },
        {
            "task": 2,
            "frequency": 2.0,
            "converged": True,
            "iterations": 24,
            "residual": 0.0011,
            "status": "failed",
        },
    ]
    mirrored = json.loads(solver_manifest.read_text())
    assert mirrored["task_summary"] == payload["task_summary"]
    assert [row["status"] for row in mirrored["tasks"]] == [
        "succeeded",
        "failed",
        "not_run",
    ]
    assert mirrored["solver"]["name"] == "solver"
    assert mirrored["solver"]["convergence"] == convergence


def test_job_collects_task_run_manifests_into_job_manifest(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    job.save()
    solver_manifest = job._result_path / "_fs_run" / "run_manifest.json"
    solver_manifest.parent.mkdir(parents=True, exist_ok=True)
    solver_manifest.write_text(
        json.dumps(
            {
                "schema": "fs-run-manifest-1",
                "solver": {"name": "solver"},
            }
        )
    )
    for task, residual in ((1, 8.2e-5), (2, 2.0e-3)):
        task_manifest = job.task_run_manifest_path(task)
        task_manifest.parent.mkdir(parents=True, exist_ok=True)
        task_manifest.write_text(
            json.dumps(
                {
                    "exit_status": {"code": 0, "status": "success"},
                    "execution": {"mpi": {"ranks": task}, "openmp": {"threads": 8}},
                    "solver": {
                        "convergence": {
                            "converged": True,
                            "status": "converged",
                            "solve_count": 1,
                            "failure_count": 0,
                            "worst_code": 0,
                            "solves": [
                                {
                                    "context": "forward",
                                    "converged": True,
                                    "iterations": 4 + task,
                                    "residual": residual,
                                    "status": "converged",
                                }
                            ],
                        }
                    },
                }
            )
        )

    path = job.collect_task_run_manifests()

    assert path == job.run_state_file
    payload = json.loads(job.run_state_file.read_text())
    assert payload["task_summary"] == {
        "total": 2,
        "complete": 2,
        "succeeded": 1,
        "failed": 1,
        "not_run": 0,
    }
    mirrored = json.loads(solver_manifest.read_text())
    assert mirrored["task_summary"] == payload["task_summary"]
    assert [row["status"] for row in mirrored["tasks"]] == ["succeeded", "failed"]
    assert mirrored["tasks"][0]["returncode"] == 0
    assert mirrored["tasks"][1]["n_ranks"] == 2
    assert mirrored["tasks"][1]["threads_per_rank"] == 8
    assert mirrored["solver"]["convergence"]["status"] == "failed"
    assert mirrored["solver"]["convergence"]["residual"] == 0.002


def test_job_collects_skipped_task_run_manifests_as_successful(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    job.save()
    solver_manifest = job._result_path / "_fs_run" / "run_manifest.json"
    solver_manifest.parent.mkdir(parents=True, exist_ok=True)
    solver_manifest.write_text(json.dumps({"schema": "fs-run-manifest-1"}))

    task_manifest = job.task_run_manifest_path(1)
    task_manifest.parent.mkdir(parents=True, exist_ok=True)
    task_manifest.write_text(
        json.dumps(
            {
                "exit_status": {"code": 1, "status": "failed"},
                "execution": {"skipped": True},
                "solver": {
                    "convergence": {
                        "converged": False,
                        "status": "failed",
                        "solve_count": 0,
                        "failure_count": 1,
                        "worst_code": 1,
                    }
                },
            }
        )
    )

    job.collect_task_run_manifests()

    payload = json.loads(job.run_state_file.read_text())
    assert payload["task_summary"] == {
        "total": 1,
        "complete": 1,
        "succeeded": 1,
        "failed": 0,
        "not_run": 0,
    }
    mirrored = json.loads(solver_manifest.read_text())
    assert mirrored["task_summary"] == payload["task_summary"]
    assert mirrored["tasks"][0]["status"] == "succeeded"
    assert "solver" not in mirrored
    assert job.failed_tasks() == []


def test_job_run_state_reads_solver_convergence_from_manifest_path(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[5.0])
    job.save()
    trace_file = job.trace_manifest.files[0]
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    trace_file.touch()
    task_manifest = tmp_path / "task_run_manifest.json"
    task_manifest.write_text(
        json.dumps(
            {
                "solver": {
                    "convergence": {
                        "converged": True,
                        "status": "converged",
                        "solve_count": 0,
                        "failure_count": 0,
                        "worst_code": 0,
                        "solves": [
                            {
                                "converged": True,
                                "iterations": 7,
                                "residual": 9.1e-5,
                                "status": "converged",
                            }
                        ],
                    }
                }
            }
        )
    )

    job.write_run_state(
        status="completed",
        tasks=[
            {
                "task_id": 0,
                "status": "success",
                "complete": True,
                "run_manifest": str(task_manifest),
            }
        ],
    )

    payload = json.loads(job.run_state_file.read_text())

    convergence = payload["solver"]["convergence"]
    assert convergence["status"] == "converged"
    assert convergence["solve_count"] == 1
    assert convergence["iterations"] == 7
    assert convergence["residual"] == 9.1e-05


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


def test_job_task_timings_preserves_skipped_task_runtime(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[10.0, 20.0])
    job.save()
    for file in job.expected_trace_files():
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    job.write_run_state(
        status="completed",
        tasks=[
            {
                "task_id": 0,
                "status": "success",
                "duration_seconds": 12.0,
                "core_count": 8,
            },
            {
                "task_id": 1,
                "status": "success",
                "duration_seconds": 24.0,
                "core_count": 8,
            },
        ],
    )

    job.write_run_state(status="skipped")

    assert [row["duration_seconds"] for row in job.task_timings()] == [12.0, 24.0]
    assert [row["core_count"] for row in job.task_timings()] == [8.0, 8.0]


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
    run_dir = job._result_path / "_fs_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "timings.json").write_text(
        json.dumps(
            {
                "schema": "fs-timings-1",
                "tasks": [
                    {
                        "task": 1,
                        "phases": {
                            "setup": 0.2,
                            "assembly": 1.0,
                            "solve_forward": 2.0,
                        },
                    },
                    {
                        "task": 2,
                        "phases": {
                            "setup": 0.3,
                            "assembly": 1.5,
                            "solve_forward": 3.0,
                        },
                    },
                ],
            }
        )
    )

    rows = job.phase_timings(phases=["setup", "assembly", "solve_forward"])

    assert rows == [
        {
            "task": 1,
            "frequency": 5.0,
            "status": "not_run",
            "total_seconds": 3.2,
            "setup": 0.2,
            "assembly": 1.0,
            "solve_forward": 2.0,
        },
        {
            "task": 2,
            "frequency": 10.0,
            "status": "not_run",
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


def test_job_task_plan_only_runs_new_frequencies_when_range_expands(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    job.save()
    for file in job.expected_trace_files():
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(file.name)
    job.write_run_state(status="completed")

    expanded = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0, 3.0])
    expanded.save()

    plan = expanded.task_run_plan()

    assert plan["current_tasks"] == [1, 2]
    assert plan["pending_indices"] == [2]
    assert expanded.frequency_summary() == {
        "total": 3,
        "succeeded": 2,
        "failed": 0,
        "not_run": 1,
    }


def test_job_task_plan_force_runs_all_frequencies(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    job.save()
    for file in job.expected_trace_files():
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(file.name)
    job.write_run_state(status="completed")

    plan = job.task_run_plan(force=True)

    assert plan == {
        "pending_indices": [0, 1],
        "current_tasks": [],
        "reused_tasks": [],
    }
    assert not any(file.exists() for file in job.expected_trace_files())


def test_job_task_plan_skips_current_packed_trace_product(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    job.save()
    trace_dir = job.trace_outputs.path
    trace_dir.mkdir(parents=True, exist_ok=True)
    packed = trace_dir / "traces.h5"
    packed.touch()
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
                "frequencies": [
                    {"task_id": 1, "frequency": 1.0, "status": "packed"},
                    {"task_id": 2, "frequency": 2.0, "status": "packed"},
                ],
            }
        )
    )

    job.write_run_state(
        status="completed",
        tasks=[
            {"task_id": 0, "status": "success", "duration_seconds": 1.0},
            {"task_id": 1, "status": "success", "duration_seconds": 2.0},
        ],
    )

    assert len(job.expected_trace_files()) == 2
    assert job.trace_manifest.packed_file == packed
    assert job.is_run_current()
    assert job.task_run_plan() == {
        "pending_indices": [],
        "current_tasks": [1, 2],
        "reused_tasks": [],
    }
    assert job.frequency_summary() == {
        "total": 2,
        "succeeded": 2,
        "failed": 0,
        "not_run": 0,
    }


def test_failed_frequency_prevents_local_skip_with_packed_trace(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    job.save()
    trace_dir = job.trace_outputs.path
    trace_dir.mkdir(parents=True, exist_ok=True)
    packed = trace_dir / "traces.h5"
    packed.touch()
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
                "frequencies": [
                    {"task_id": 1, "frequency": 1.0, "status": "packed"},
                    {"task_id": 2, "frequency": 2.0, "status": "packed"},
                ],
            }
        )
    )

    job.write_run_state(
        status="completed",
        tasks=[
            {"task_id": 0, "status": "success", "duration_seconds": 1.0},
            {"task_id": 1, "status": "error", "duration_seconds": 2.0},
        ],
    )

    assert not job.is_run_current()
    assert job.current_tasks() == [1]
    assert job.task_run_plan() == {
        "pending_indices": [1],
        "current_tasks": [1],
        "reused_tasks": [],
    }


def test_incomplete_packed_trace_product_warns_and_is_not_current(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(
        name="freq",
        simulation=sim,
        f_list=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    job.save()
    trace_dir = job.trace_outputs.path
    trace_dir.mkdir(parents=True, exist_ok=True)
    packed = trace_dir / "traces.h5"
    packed.touch()
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
                "frequencies": [
                    {"task_id": 1, "frequency": 1.0, "status": "packed"},
                    {"task_id": 2, "frequency": 2.0, "status": "packed"},
                    {"task_id": 5, "frequency": 5.0, "status": "packed"},
                ],
            }
        )
    )

    assert job.trace_manifest.packed_file == packed
    assert job.trace_manifest.missing_packed_frequencies == {
        3: 3.0,
        4: 4.0,
        6: 6.0,
    }
    assert not job.trace_manifest.complete

    with pytest.warns(RuntimeWarning) as caught:
        assert not job.is_run_current()
    message = str(caught[0].message)
    assert "missing 3 of 6 expected frequencies" in message
    assert "tasks 3-4: 3 Hz-4 Hz" in message
    assert "task 6: 6 Hz" in message
    assert "traces_3.h5" not in message

    with pytest.warns(RuntimeWarning, match="missing 3 of 6 expected frequencies"):
        traces = TraceDataset.from_job(job)
    assert traces.manifest.files == [packed]
    assert traces.manifest.frequencies == {1: 1.0, 2: 2.0, 5: 5.0}


def test_frequency_named_trace_shard_counts_as_current_when_pack_is_stale(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[50.0])
    job.save()
    trace_dir = job.trace_outputs.path
    shard_dir = trace_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / "f_50.00000_hz.h5"
    with h5py.File(shard, "w") as h5:
        h5.create_dataset("frequency", data=50.0)
        h5.create_dataset("laplace", data=-0.5)

    packed = trace_dir / "traces.h5"
    packed.touch()
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
                "frequencies": [
                    {"task_id": 1, "frequency": 100.0, "status": "packed"},
                ],
            }
        )
    )

    with pytest.warns(RuntimeWarning, match="missing 1 of 1 expected frequencies"):
        assert job.trace_outputs_exist()

    job.write_run_state(status="completed")
    state = job.run_state()

    assert state["task_summary"] == {
        "total": 1,
        "complete": 1,
        "succeeded": 1,
        "failed": 0,
        "not_run": 0,
    }
    assert state["tasks"][0]["path"].endswith("traces/shards/f_50.00000_hz.h5")
    assert job.current_tasks() == [1]
    assert job.frequency_summary() == {
        "total": 1,
        "succeeded": 1,
        "failed": 0,
        "not_run": 0,
    }
    assert job.task_run_plan()["pending_indices"] == []


def test_job_task_plan_reruns_packed_product_when_fingerprint_changes(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    job.save()
    trace_dir = job.trace_outputs.path
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "traces.h5").touch()
    (trace_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fs-trace-manifest-1",
                "packed": {
                    "format": "hdf5",
                    "schema": "fs-traces-packed-1",
                    "relative_path": "traces/traces.h5",
                },
                "frequencies": [
                    {"task_id": 1, "frequency": 1.0, "status": "packed"},
                    {"task_id": 2, "frequency": 2.0, "status": "packed"},
                ],
            }
        )
    )
    job.write_run_state(
        status="completed",
        tasks=[
            {"task_id": 0, "status": "success"},
            {"task_id": 1, "status": "success"},
        ],
    )

    expanded = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0, 3.0])
    expanded.save()

    with pytest.warns(RuntimeWarning, match="missing 1 of 3 expected frequencies"):
        assert not expanded.is_run_current()
    assert expanded.task_run_plan()["pending_indices"] == [0, 1, 2]


def test_job_task_plan_reuses_frequency_outputs_when_sampling_interleaves(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = TimeDomainJob(name="time", simulation=sim, f_min=0.0, f_max=1.0, T_max=2.0)
    job.save()
    old_files = job.expected_trace_files()
    old_files[0].parent.mkdir(parents=True, exist_ok=True)
    old_files[0].write_text("0.5 Hz")
    old_files[1].write_text("1.0 Hz")
    job.write_run_state(status="completed")

    expanded = TimeDomainJob(
        name="time",
        simulation=sim,
        f_min=0.0,
        f_max=1.0,
        T_max=4.0,
    )
    expanded.save()

    plan = expanded.task_run_plan(reuse=True)
    new_files = expanded.expected_trace_files()

    assert plan["pending_indices"] == [0, 2]
    assert plan["current_tasks"] == [2, 4]
    assert [record["task"] for record in plan["reused_tasks"]] == [2, 4]
    assert new_files[1].read_text() == "0.5 Hz"
    assert new_files[3].read_text() == "1.0 Hz"
    assert expanded.frequency_summary() == {
        "total": 4,
        "succeeded": 2,
        "failed": 0,
        "not_run": 2,
    }


def test_job_task_plan_preserves_exact_matches_when_reusing_shifted_outputs(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 3.0])
    job.save()
    old_files = job.expected_trace_files()
    old_files[0].parent.mkdir(parents=True, exist_ok=True)
    old_files[0].write_text("1 Hz")
    old_files[1].write_text("3 Hz")
    job.write_run_state(status="completed")

    expanded = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0, 3.0])
    expanded.save()

    plan = expanded.task_run_plan(reuse=True)
    new_files = expanded.expected_trace_files()

    assert plan["pending_indices"] == [1]
    assert plan["current_tasks"] == [1, 3]
    assert new_files[0].read_text() == "1 Hz"
    assert new_files[2].read_text() == "3 Hz"


def test_job_task_plan_stages_reused_traces_and_removes_stale_pending_slots(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])
    job.save()
    old_files = job.expected_trace_files()
    old_files[0].parent.mkdir(parents=True, exist_ok=True)
    old_files[0].write_text("1 Hz")
    old_files[1].write_text("2 Hz")
    job.write_run_state(status="completed")

    cache = job._result_path / "_fs_run" / "cache" / "traces_vds.h5"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("stale")

    expanded = FrequencyDomainJob(
        name="freq",
        simulation=sim,
        f_list=[0.5, 1.0, 2.0],
    )
    expanded.save()

    plan = expanded.task_run_plan(reuse=True)
    new_files = expanded.expected_trace_files()

    assert plan["pending_indices"] == [0]
    assert plan["current_tasks"] == [2, 3]
    assert not new_files[0].exists()
    assert new_files[1].read_text() == "1 Hz"
    assert new_files[2].read_text() == "2 Hz"
    assert not cache.exists()

    expanded.write_run_state(status="completed", tasks=plan["reused_tasks"])

    assert expanded.frequency_summary() == {
        "total": 3,
        "succeeded": 2,
        "failed": 0,
        "not_run": 1,
    }
