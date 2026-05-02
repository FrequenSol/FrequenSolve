import json

import pytest

from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.orchestrator.sites.base import JobStatus, RunResult
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project.project import Project
from frequensolve.simulation.artifacts import RunMetadata
from frequensolve.simulation.jobs import (
    FrequencyDomainJob,
    SimulationJob,
    TimeDomainJob,
)


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


def test_job_save_load_persists_required_simulation_inputs(tmp_path):
    _, sim = _project_with_trace_simulation(tmp_path)
    job = FrequencyDomainJob(name="freq", simulation=sim, f_list=[1.0, 2.0])

    job_file = job.save()
    payload = json.loads(job_file.read_text())
    loaded = SimulationJob.load(job_file)

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
