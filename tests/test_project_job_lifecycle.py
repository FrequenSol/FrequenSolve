import json

import pytest

from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project.project import Project
from frequensolve.simulation.jobs import (
    FrequencyDomainJob,
    SimulationJob,
    TimeDomainJob,
)
from frequensolve.simulation.output_manager import OutputManager, TraceOutput


def _project_with_trace_simulation(tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    sim = project.new_simulation(name="simple", physics="acoustic", dimension=2)
    sim.outputs = OutputManager()
    sim.outputs += TraceOutput()
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


def test_job_save_load_persists_required_simulation_inputs(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])

    job_file = job.save()
    payload = json.loads(job_file.read_text())
    loaded = SimulationJob.load(job_file)

    assert sim._file.exists()
    assert job_file.exists()
    assert payload["simulation"] == "simulations/simple/simple.json"
    assert "overwrite" not in payload
    assert "max_versions" not in payload
    assert not (job_file.parent / "manifest.json").exists()
    assert loaded.name == "freq"
    assert loaded.simulation.name == "simple"
    assert loaded.f_list == [1.0, 2.0]
    assert loaded._file == job_file


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


def test_local_submit_autosaves_job_and_simulation(monkeypatch, tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0])
    site = LocalSite()

    monkeypatch.setattr(site, "_submit_local_tasks", lambda job, **kwargs: [])

    run = site.submit(job)

    assert run.job is job
    assert sim._file.exists()
    assert job._file.exists()


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
            "status": "succeeded",
            "trace_file": trace_file,
        },
        {
            "task": 2,
            "frequency": 2.0,
            "duration_seconds": 2.5,
            "status": "failed",
            "trace_file": job.trace_manifest.files[1],
        },
    ]


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

    assert ax.get_xlabel() == "Frequency task"
    assert ax.get_ylabel() == "Runtime (s)"
    assert len(ax.patches) == 2
