import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from frequensolve.orchestrator.sites.hpc import site as hpc


def _load_scheduler_module():
    path = Path(hpc.__file__).parent / "templates" / "sweep" / "adaptive_scheduler.py"
    spec = importlib.util.spec_from_file_location("adaptive_scheduler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adaptive_scheduler_interval_allocation_and_coalescing():
    scheduler = _load_scheduler_module()

    offset, remaining = scheduler._allocate_interval([(0, 4), (8, 2)], 3)

    assert offset == 0
    assert remaining == [(3, 1), (8, 2)]
    assert scheduler._free_interval(remaining, 4, 4) == [(3, 7)]


def test_adaptive_scheduler_reads_structured_config_without_environment(
    tmp_path,
):
    scheduler = _load_scheduler_module()
    sizing = tmp_path / "sizing.json"
    sizing.write_text(json.dumps({"task": [{"memory": "1.5 GB"}]}))
    config = {
        "executable": "/remote/bin/solver",
        "total_ranks": 8,
        "omp_threads": 2,
        "mem_per_rank_gib": 4,
        "job_task_count": 1,
        "task_indices": [1],
        "sizing_json": str(sizing),
    }

    instance = scheduler.AdaptiveScheduler(
        config,
        job_file="job.json",
        output=str(tmp_path),
        status=str(tmp_path / "status.json"),
    )

    assert instance._load_task_memory() == [1.5]
    assert instance._choose_base_ranks(1.5) == 1
    assert instance.task_indices == [1]


def test_adaptive_scheduler_passes_configured_mpi_arguments(monkeypatch, tmp_path):
    scheduler = _load_scheduler_module()
    launched = {}

    class Process:
        pass

    def fake_popen(command, **kwargs):
        launched["command"] = command
        return Process()

    monkeypatch.setattr(scheduler.subprocess, "Popen", fake_popen)
    instance = scheduler.AdaptiveScheduler(
        {
            "executable": "/remote/bin/solver",
            "mpi": "srun",
            "mpi_args": ["--kill-on-bad-exit=1", "--wait=30"],
            "total_ranks": 2,
            "omp_threads": 1,
            "mem_per_rank_gib": 1,
            "job_task_count": 1,
            "task_indices": [1],
        },
        job_file="job.json",
        output=str(tmp_path),
        status=str(tmp_path / "status.json"),
    )

    instance._launch(task_id=1, offset=0, ranks=2, memory=1.0)

    assert launched["command"][:6] == [
        "srun",
        "--kill-on-bad-exit=1",
        "--wait=30",
        "--exclusive",
        "--ntasks",
        "2",
    ]
    instance.running[-1][-1].close()


def test_sizing_checkpoint_validation_uses_scheduler_memory_field(tmp_path):
    scheduler = _load_scheduler_module()
    sizing = tmp_path / "sizing.json"
    sizing.write_text(
        json.dumps(
            {
                "schema": "fs-sizing-2",
                "sweep_status": "forward_sweep_checkpoint",
                "task": [{"memory": "512 MB"}, {"memory": "1.5 GB"}],
            }
        )
    )

    assert scheduler.validate_sizing_checkpoint(str(sizing), 2) == [0.5, 1.5]
    result = subprocess.run(
        [sys.executable, scheduler.__file__, "--validate-sizing", str(sizing), "2"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_sizing_checkpoint_validation_rejects_memory_bytes_only(tmp_path):
    scheduler = _load_scheduler_module()
    sizing = tmp_path / "sizing.json"
    sizing.write_text(
        json.dumps(
            {
                "schema": "fs-sizing-2",
                "sweep_status": "forward_sweep_checkpoint",
                "task": [{"memory_bytes": 1024}],
            }
        )
    )

    with pytest.raises(SystemExit, match="task 1 missing valid memory estimate"):
        scheduler.validate_sizing_checkpoint(str(sizing), 1)


def test_adaptive_scheduler_requires_one_task_when_sizing_is_skipped(tmp_path):
    scheduler = _load_scheduler_module()
    instance = scheduler.AdaptiveScheduler(
        {
            "executable": "/remote/bin/solver",
            "total_ranks": 2,
            "omp_threads": 1,
            "mem_per_rank_gib": 1,
            "job_task_count": 2,
            "task_indices": [1, 2],
            "skip_sizing": True,
        },
        job_file="job.json",
        output=str(tmp_path),
        status=str(tmp_path / "status.json"),
    )

    with pytest.raises(SystemExit, match="requires exactly one submitted task"):
        instance.run()


@pytest.mark.parametrize("skip_sizing", [False, True])
@pytest.mark.parametrize("limit", [1, 3, 8])
def test_scheduler_caps_rounding_and_single_task_boost(
    monkeypatch, tmp_path, skip_sizing, limit
):
    module = _load_scheduler_module()
    instance = module.AdaptiveScheduler(
        {
            "executable": "unused-solver",
            "total_ranks": 8,
            "omp_threads": 1,
            "mem_per_rank_gib": 1,
            "job_task_count": 1,
            "task_indices": [1],
            "skip_sizing": skip_sizing,
            "round_to": 4,
            "max_ranks_per_task": limit,
        },
        job_file="job.json",
        output=str(tmp_path),
        status=str(tmp_path / "status.json"),
    )
    monkeypatch.setattr(instance, "_load_task_memory", lambda: [100.0])
    launches = []

    def launch(task, offset, ranks, memory):
        launches.append(ranks)
        instance.successful_tasks.append(task)

    monkeypatch.setattr(instance, "_launch", launch)
    instance.run()
    assert launches == [limit]


@pytest.mark.parametrize(
    ("launcher", "expected_prefix"),
    [
        ("ibrun", ["ibrun", "-n", "4", "-o", "2", "task_affinity"]),
        (
            "srun",
            ["srun", "--exclusive", "--ntasks", "4", "--cpus-per-task", "2"],
        ),
    ],
)
def test_adaptive_scheduler_builds_launcher_specific_task_commands(
    tmp_path, launcher, expected_prefix
):
    scheduler = _load_scheduler_module()
    instance = scheduler.AdaptiveScheduler(
        {
            "executable": "/remote/bin/solver",
            "mpi": launcher,
            "total_ranks": 8,
            "omp_threads": 2,
            "mem_per_rank_gib": 4,
            "job_task_count": 1,
            "task_indices": [1],
            "sizing_json": str(tmp_path / "sizing.json"),
        },
        job_file="job.json",
        output=str(tmp_path),
        status=str(tmp_path / "status.json"),
    )

    command = instance._launch_command(task_id=1, offset=2, ranks=4)

    assert command[: len(expected_prefix)] == expected_prefix
    assert command[-2:] == ["--task", "1"]


@pytest.mark.parametrize("rank_limit", [None, 1, 3])
@pytest.mark.parametrize("launcher", ["mpirun", "mpiexec"])
def test_adaptive_scheduler_mpirun_uses_full_allocation_without_ibrun_flags(
    tmp_path, rank_limit, launcher
):
    scheduler = _load_scheduler_module()
    instance = scheduler.AdaptiveScheduler(
        {
            "executable": "/remote/bin/solver",
            "mpi": f"/usr/bin/{launcher}",
            "total_ranks": 4,
            "omp_threads": 1,
            "mem_per_rank_gib": 4,
            "job_task_count": 1,
            "task_indices": [1],
            "sizing_json": str(tmp_path / "sizing.json"),
            **({"max_ranks_per_task": rank_limit} if rank_limit else {}),
        },
        job_file="job.json",
        output=str(tmp_path),
        status=str(tmp_path / "status.json"),
    )

    ranks = rank_limit or 4
    assert instance._choose_base_ranks(0.25) == ranks
    assert instance.free_intervals == [(0, ranks)]
    command = instance._launch_command(task_id=1, offset=0, ranks=ranks)
    assert command[:3] == [f"/usr/bin/{launcher}", "-n", str(ranks)]
    assert "-o" not in command
    assert "task_affinity" not in command

    with pytest.raises(SystemExit, match="requires exclusive use"):
        instance._launch_command(task_id=1, offset=1, ranks=3)
