import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

from frequensolve.orchestrator.sites.hpc import site as hpc


def _slurm_allocation(monkeypatch, module, counts="8", hosts=("node0",)):
    for key in list(os.environ):
        if key.startswith("SLURM_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "allocation-nodes")
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", counts)
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **k: "\n".join(hosts)
    )


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


def _idle_scheduler(tmp_path, **options):
    module = _load_scheduler_module()
    return module.AdaptiveScheduler(
        {
            "executable": "unused-solver",
            "mpi": "srun",
            "total_ranks": 64,
            "omp_threads": 14,
            "mem_per_rank_gib": 10,
            "job_task_count": 3,
            "task_indices": [1, 2, 3],
            **options,
        },
        job_file="job.json",
        output=str(tmp_path),
        status=str(tmp_path / "status.json"),
    )


def test_idle_sweep_balances_equal_tasks_before_launch(monkeypatch, tmp_path):
    instance = _idle_scheduler(tmp_path)
    # Previously two ceil(64/3) boosts left the third task at its three-rank
    # memory minimum: 22/22/3 instead of a balanced, fully occupied allocation.
    launches = _launched_ranks(monkeypatch, instance, [20.0, 20.0, 20.0])
    assert launches == {1: 22, 2: 21, 3: 21}


@pytest.mark.parametrize("minimum", [0, -1])
@pytest.mark.parametrize("skip_sizing", [False, True])
def test_scheduler_rejects_nonpositive_minimum(tmp_path, minimum, skip_sizing):
    with pytest.raises(ValueError, match="min_ranks must be positive"):
        _idle_scheduler(tmp_path, min_ranks=minimum, skip_sizing=skip_sizing)


@pytest.mark.parametrize("skip_sizing", [False, True])
def test_zero_memory_task_still_launches_with_positive_ranks(
    monkeypatch, tmp_path, skip_sizing
):
    instance = _idle_scheduler(
        tmp_path, job_task_count=1, task_indices=[1], skip_sizing=skip_sizing
    )
    assert instance._choose_base_ranks(0.0) == 1
    assert _launched_ranks(monkeypatch, instance, [0.0]) == {1: 64}


def test_idle_sweep_weights_work_and_redistributes_after_caps(monkeypatch, tmp_path):
    instance = _idle_scheduler(tmp_path, total_ranks=60)
    assert _launched_ranks(monkeypatch, instance, [20.0, 30.0, 50.0]) == {
        1: 12,
        2: 18,
        3: 30,
    }
    instance = _idle_scheduler(tmp_path, total_ranks=60, max_ranks_per_task=24)
    assert _launched_ranks(monkeypatch, instance, [20.0, 30.0, 50.0]) == {
        1: 14,
        2: 22,
        3: 24,
    }


def test_idle_sweep_preserves_pins_and_boost_caps(monkeypatch, tmp_path):
    instance = _idle_scheduler(tmp_path, total_ranks=60, task_ranks={"1": 3})
    launches = _launched_ranks(monkeypatch, instance, [20.0, 30.0, 50.0])
    assert launches == {1: 3, 2: 21, 3: 36}
    instance = _idle_scheduler(tmp_path, boost_max_factor=2)
    assert _launched_ranks(monkeypatch, instance, [20.0] * 3) == {1: 6, 2: 6, 3: 6}


@pytest.mark.parametrize("ranks, expected", [(64, [24, 20, 20]), (63, [20, 20, 20])])
def test_idle_sweep_obeys_rank_quanta(monkeypatch, tmp_path, ranks, expected):
    instance = _idle_scheduler(tmp_path, total_ranks=ranks, round_to=4)
    assert sorted(
        _launched_ranks(monkeypatch, instance, [20.0] * 3).values()
    ) == sorted(expected)


def test_idle_sweep_zero_cost_and_exact_minimum_budget(tmp_path):
    instance = _idle_scheduler(tmp_path, total_ranks=12)
    plan = instance._balanced_idle_plan(
        [(1, 4, 8, 0.0), (2, 4, 8, 0.0), (3, 4, 8, 0.0)]
    )
    assert plan == [(1, 0, 4, 0.0), (2, 4, 4, 0.0), (3, 8, 4, 0.0)]
    plan = instance._balanced_idle_plan([(1, 1, 8, 0.0), (2, 1, 8, 0.0)])
    assert [entry[2] for entry in plan] == [6, 6]


def test_busy_and_overfull_sweeps_keep_existing_packing(tmp_path):
    instance = _idle_scheduler(tmp_path)
    candidates = [(1, 3, 22, 20.0), (2, 3, 22, 20.0), (3, 3, 22, 20.0)]
    instance.running = [object()]
    assert instance._balanced_idle_plan(candidates) is None
    assert [entry[2] for entry in instance._plan_launches(candidates)] == [22, 22, 3]
    instance.running = []
    instance.free_intervals = [(0, 20), (40, 24)]
    assert instance._balanced_idle_plan(candidates) is None
    assert instance._plan_launches(candidates) == [
        (1, 40, 22, 20.0),
        (2, 0, 3, 20.0),
        (3, 3, 3, 20.0),
    ]
    instance.free_intervals = [(0, 64)]
    assert (
        instance._balanced_idle_plan([(1, 40, 50, 100.0), (2, 40, 50, 100.0)]) is None
    )
    assert instance._balanced_idle_plan([]) is None


@pytest.mark.parametrize("launcher", ["mpiexec", "mpirun"])
def test_idle_balance_does_not_partition_exclusive_launchers(tmp_path, launcher):
    instance = _idle_scheduler(tmp_path, mpi=launcher)
    assert instance._balanced_idle_plan([(1, 32, 64, 20.0), (2, 32, 64, 20.0)]) is None


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
    _slurm_allocation(monkeypatch, scheduler, "2")
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
        "--exact",
        "--distribution=arbitrary",
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
            ["srun", "--exclusive", "--exact", "--distribution=arbitrary"],
        ),
    ],
)
def test_adaptive_scheduler_builds_launcher_specific_task_commands(
    monkeypatch, tmp_path, launcher, expected_prefix
):
    scheduler = _load_scheduler_module()
    _slurm_allocation(monkeypatch, scheduler)
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


def _write_job(tmp_path):
    job = tmp_path / "job.json"
    job.write_text(
        json.dumps({"project_path": str(tmp_path), "result_path": "results"})
    )
    return job


def _write_error_json(tmp_path, relative, message, *, mtime=None):
    path = tmp_path / "results" / "_fs_run" / relative / "error.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "fs-error-1", "message": message}))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_step_failure_reason_prefers_error_json_then_log_line(tmp_path):
    scheduler = _load_scheduler_module()
    job = _write_job(tmp_path)
    log = tmp_path / "task_2.log"
    log.write_text("Error: Objective needs observed data\n   at: x.f90:3\n")

    _write_error_json(tmp_path, "tasks/task_000002", "error.json reason")
    assert scheduler.step_failure_reason(job, 2, log, 1, not_before=0.0) == (
        "solver exited with status 1: error.json reason"
    )

    # An error.json from an earlier attempt must not explain this one.
    stale = time.time() - 3600
    _write_error_json(tmp_path, "tasks/task_000002", "old reason", mtime=stale)
    assert scheduler.step_failure_reason(job, 2, log, 1, not_before=time.time()) == (
        "solver exited with status 1: Objective needs observed data"
    )


@pytest.mark.parametrize("return_code", [-11, 139])
def test_step_failure_reason_names_signal_deaths(tmp_path, return_code):
    scheduler = _load_scheduler_module()

    reason = scheduler.step_failure_reason(
        tmp_path / "missing.json", 1, None, return_code, not_before=0.0
    )

    assert reason == "solver terminated by SIGSEGV"


def test_record_failure_names_operation_step_and_keeps_existing_reason(tmp_path):
    scheduler = _load_scheduler_module()
    job = _write_job(tmp_path)
    _write_error_json(tmp_path, "operations/init", "mesh file is unreadable")
    status = tmp_path / "scheduler_status.json"
    status.write_text(json.dumps({"state": "running", "total": 3}))

    result = subprocess.run(
        [
            sys.executable,
            scheduler.__file__,
            "--record-failure",
            "--status",
            str(status),
            "--job",
            str(job),
            "--tasks",
            "3",
            "--step",
            "init",
            "--log",
            str(tmp_path / "init.log"),
            "--return-code",
            "1",
            "--not-before",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(status.read_text())
    assert payload["state"] == "failed"
    assert payload["phase"] == "init"
    assert payload["return_code"] == 1
    assert payload["abort_reason"] == (
        "Mesh preparation failed: solver exited with status 1: "
        "mesh file is unreadable; see init.log"
    )

    payload["abort_reason"] = "MPI health check failed"
    status.write_text(json.dumps(payload))
    scheduler.record_step_failure(
        status,
        job_file=str(job),
        n_tasks=3,
        step="smooth",
        log_file="",
        return_code=1,
        not_before=0.0,
    )
    assert json.loads(status.read_text())["abort_reason"] == "MPI health check failed"


def _pinned_scheduler(tmp_path, *, mpi="srun", task_ranks, tasks, **extra):
    scheduler = _load_scheduler_module()
    return scheduler.AdaptiveScheduler(
        {
            "executable": "/remote/bin/solver",
            "mpi": mpi,
            "total_ranks": 8,
            "omp_threads": 2,
            "mem_per_rank_gib": 1,
            "job_task_count": len(tasks),
            "task_indices": tasks,
            "task_ranks": {str(task): ranks for task, ranks in task_ranks.items()},
            **extra,
        },
        job_file="job.json",
        output=str(tmp_path),
        status=str(tmp_path / "status.json"),
    )


def _launched_ranks(monkeypatch, instance, memory):
    """Run the scheduler with tasks that finish as soon as they launch."""

    scheduler = _load_scheduler_module()
    monkeypatch.setattr(instance, "_load_task_memory", lambda: memory)
    launches = {}

    def launch(task, offset, ranks, task_memory):
        launches[task] = ranks
        instance.successful_tasks.append(task)
        instance.free_intervals = scheduler._free_interval(
            instance.free_intervals, offset, ranks
        )

    monkeypatch.setattr(instance, "_launch", launch)
    instance.run()
    return launches


@pytest.mark.parametrize("skip_sizing", [False, True])
def test_pinned_tasks_run_on_their_ranks_without_boosts(
    monkeypatch, tmp_path, skip_sizing
):
    """A background checkpoint restores only on the rank count that wrote it."""

    instance = _pinned_scheduler(
        tmp_path,
        task_ranks={1: 3, 2: 2, 3: 5},
        tasks=[1, 2, 3],
        skip_sizing=skip_sizing,
    )

    # Memory-based sizing and boosts would give task 1 far more ranks.
    launches = _launched_ranks(monkeypatch, instance, [100.0, 0.1, 0.1])

    assert launches == {1: 3, 2: 2, 3: 5}


def test_pins_leave_other_tasks_adaptive_and_single_pins_unboosted(
    monkeypatch, tmp_path
):
    mixed = _pinned_scheduler(tmp_path, task_ranks={2: 3}, tasks=[1, 2])
    assert _launched_ranks(monkeypatch, mixed, [2.0, 2.0])[2] == 3

    single = _pinned_scheduler(tmp_path, task_ranks={1: 3}, tasks=[1])
    assert _launched_ranks(monkeypatch, single, [0.0]) == {1: 3}

    with pytest.raises(SystemExit, match="pinned ranks for every task"):
        _pinned_scheduler(
            tmp_path, task_ranks={1: 3}, tasks=[1, 2], skip_sizing=True
        ).run()


@pytest.mark.parametrize("ranks", [0, 9])
def test_pins_outside_the_allocation_are_rejected(tmp_path, ranks):
    with pytest.raises(ValueError, match="pinned to"):
        _pinned_scheduler(tmp_path, task_ranks={1: ranks}, tasks=[1])


@pytest.mark.parametrize("binding", ["cores", "none"])
def test_mpirun_pinned_task_holds_the_allocation_and_maps_cores(tmp_path, binding):
    scheduler = _load_scheduler_module()
    instance = scheduler.AdaptiveScheduler(
        {
            "executable": "/remote/bin/solver",
            "mpi": "/usr/bin/mpirun",
            "total_ranks": 4,
            "omp_threads": 3,
            "mem_per_rank_gib": 1,
            "job_task_count": 2,
            "task_indices": [1, 2],
            "task_ranks": {"1": 2},
        },
        job_file="job.json",
        output=str(tmp_path),
        status=str(tmp_path / "status.json"),
        rank_binding=binding,
    )

    assert instance._choose_base_ranks(1.0, 1) == 4  # Exclusive allocation.
    pinned = instance._launch_command(task_id=1, offset=0, ranks=4)
    sized = instance._launch_command(task_id=2, offset=0, ranks=4)

    placement = ["--map-by", "slot:PE=3", "--bind-to", "core"]
    expected = placement if binding == "cores" else []
    assert pinned[: 3 + len(expected)] == ["/usr/bin/mpirun", "-n", "2", *expected]
    assert sized[: 3 + len(expected)] == ["/usr/bin/mpirun", "-n", "4", *expected]


def test_status_records_the_ranks_every_task_ran_on(monkeypatch, tmp_path):
    scheduler = _load_scheduler_module()
    _slurm_allocation(monkeypatch, scheduler)
    instance = _pinned_scheduler(tmp_path, task_ranks={1: 3}, tasks=[1])

    class Process:
        pass

    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda *a, **k: Process())
    monkeypatch.setattr(instance, "launch_delay", 0)
    instance._launch(task_id=1, offset=0, ranks=3, memory=1.0)
    instance.running[-1][-1].close()

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["task_ranks"] == {"1": 3}


def test_slurm_places_full_cohort_without_fragmenting_hosts(monkeypatch, tmp_path):
    module = _load_scheduler_module()
    hosts = [f"node{i:03d}" for i in range(100)]
    _slurm_allocation(monkeypatch, module, "8(x100)", hosts)
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "112(x100)")
    instance = _idle_scheduler(
        tmp_path,
        total_ranks=800,
        mpi_args=["--distribution=block:block", "--cpu-bind=cores", "--wait=30"],
    )
    occupied = []
    offset = 0
    for task, ranks in enumerate([203, 174, 147, 123, 101, 41, 11], 1):
        command = instance._launch_command(task, offset, ranks)
        selected = (tmp_path / f"task_{task}.hosts").read_text().split()
        assert selected == [hosts[i // 8] for i in range(offset, offset + ranks)]
        assert command[command.index("--ntasks") + 1] == str(ranks)
        assert command[command.index("--cpus-per-task") + 1] == "14"
        assert "--cpu-bind=cores" in command
        assert "--exact" in command
        assert "--overlap" not in command
        assert [x for x in command if x.startswith("--distribution=")][
            -1
        ] == "--distribution=arbitrary"
        occupied.extend(selected)
        offset += ranks
    assert Counter(occupied) == Counter({host: 8 for host in hosts})
    # A completed interior task returns the exact same host capacities.
    free = module._free_interval([], 203, 174)
    start, remaining = module._allocate_interval(free, 101)
    assert instance._slurm_rank_hosts(start, 101) == occupied[203:304]
    assert remaining == [(304, 73)]


def test_slurm_unequal_capacities_pins_and_partial_nodes(monkeypatch, tmp_path):
    module = _load_scheduler_module()
    _slurm_allocation(monkeypatch, module, "3,1,4", ["a", "b", "c"])
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "6,2,8")
    instance = _pinned_scheduler(tmp_path, task_ranks={1: 2, 2: 3}, tasks=[1, 2])
    assert instance._choose_base_ranks(1000, 1) == 2
    assert instance._slurm_rank_hosts(0, 2) == ["a", "a"]
    assert instance._slurm_rank_hosts(2, 3) == ["a", "b", "c"]
    assert instance._slurm_rank_hosts(5, 3) == ["c"] * 3
    for offset, ranks in [(-1, 1), (0, 0), (7, 2)]:
        with pytest.raises(ValueError, match="outside"):
            instance._slurm_rank_hosts(offset, ranks)


@pytest.mark.parametrize("value", ["8(x0)", "8(x4)", "8,8", "garbage", "-1", ""])
def test_slurm_invalid_counts_fail_closed(value):
    module = _load_scheduler_module()
    with pytest.raises(ValueError, match="Slurm"):
        module._slurm_counts(value, 3)


def test_slurm_allocation_mismatch_and_cpu_shortfall_fail_closed(monkeypatch, tmp_path):
    module = _load_scheduler_module()
    _slurm_allocation(monkeypatch, module, "4")
    instance = _idle_scheduler(tmp_path, total_ranks=8)
    with pytest.raises(ValueError, match="rank slots"):
        instance._slurm_rank_hosts(0, 1)
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "8")
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "16")
    with pytest.raises(ValueError, match="insufficient CPUs"):
        instance._slurm_rank_hosts(0, 1)
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE")
    monkeypatch.delenv("SLURM_TASKS_PER_NODE")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "8")
    assert instance._slurm_rank_hosts(0, 1) == ["node0"]


@pytest.mark.parametrize(
    "option", ["--nodes=100", "-N100", "--ntasks-per-node=8", "--overlap"]
)
def test_slurm_rejects_conflicting_explicit_placement(monkeypatch, tmp_path, option):
    instance = _idle_scheduler(tmp_path, mpi_args=[option])
    with pytest.raises(ValueError, match="conflicts"):
        instance._launch_command(1, 0, 1)


def test_slurm_child_environment_does_not_inherit_allocation_placement(
    monkeypatch, tmp_path
):
    instance = _idle_scheduler(tmp_path)
    constraints = {
        "SLURM_NNODES": "100",
        "SLURM_JOB_NUM_NODES": "100",
        "SLURM_NTASKS_PER_NODE": "8",
        "SLURM_HOSTFILE": "old-hosts",
        "SLURM_NTASKS": "800",
        "SLURM_NPROCS": "800",
        "SLURM_NODELIST": "all-nodes",
        "SLURM_DISTRIBUTION": "block:block",
        "SLURM_OVERLAP": "1",
        "SLURM_TASKS_PER_NODE": "8(x100)",
    }
    preserved = {
        "SLURM_JOB_ID": "123",
        "SLURM_MEM_PER_CPU": "1000",
        "OMP_NUM_THREADS": "14",
        "SLURM_CPU_BIND": "cores",
        "I_MPI_DEBUG": "0",
    }
    for key, value in {**constraints, **preserved}.items():
        monkeypatch.setenv(key, value)
    env = instance._launch_environment()
    assert not constraints.keys() & env.keys()
    assert all(env[key] == value for key, value in preserved.items())
    assert all(os.environ[key] == value for key, value in constraints.items())


def test_slurm_status_waits_for_worker_start_signal(monkeypatch, tmp_path, capsys):
    module = _load_scheduler_module()
    _slurm_allocation(monkeypatch, module, "2")
    instance = _idle_scheduler(
        tmp_path, total_ranks=2, task_indices=[1, 2], launch_delay_seconds=0
    )
    clock = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: object())
    instance._launch(1, 0, 1, 1.0)
    try:
        payload = json.loads(instance.status_file.read_text())
        assert (payload["running"], payload["launching"], payload["pending"]) == (
            0,
            1,
            1,
        )
        assert payload["waiting_for_step_tasks"] == []
        clock[0] += 61
        instance._write_status("running")
        payload = json.loads(instance.status_file.read_text())
        assert payload["waiting_for_step_tasks"] == [1]
        assert "waiting for Slurm step startup" in capsys.readouterr().out
        assert (
            "1 launching (1 waiting for Slurm step)"
            in hpc.SlurmSite._format_scheduler_status(payload)
        )
        instance._start_files[1].write_text("4\n")
        instance._write_status("running")
        payload = json.loads(instance.status_file.read_text())
        assert (payload["running"], payload["launching"], payload["pending"]) == (
            1,
            0,
            1,
        )
        assert payload["step_ids"] == {"1": "4"}
        assert payload["waiting_for_step_tasks"] == []
    finally:
        instance.running[-1][-1].close()


def test_slurm_worker_wrapper_signals_rank_zero_and_preserves_argv(
    monkeypatch, tmp_path
):
    module = _load_scheduler_module()
    _slurm_allocation(monkeypatch, module, "2")
    instance = _idle_scheduler(tmp_path, total_ranks=2, executable=sys.executable)
    command = instance._launch_command(1, 0, 2)
    prefix = command[command.index("/bin/sh") : command.index(sys.executable)]
    # Exercise the actual POSIX wrapper without an MPI or Slurm installation.
    for rank in [1, 0]:
        result = subprocess.run(
            [
                *prefix,
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1]); sys.exit(7)",
                "space ' quote",
            ],
            env={**os.environ, "SLURM_PROCID": str(rank), "SLURM_STEP_ID": "9"},
            capture_output=True,
            text=True,
        )
        assert result.returncode == 7
        assert result.stdout.strip() == "space ' quote"
        assert instance._start_files[1].exists() == (rank == 0)
    # Old startup signals cannot turn a fresh launch into a running task.
    fresh = _idle_scheduler(tmp_path, total_ranks=2)
    fresh._launch_command(1, 0, 2)
    assert fresh._start_files[1] != instance._start_files[1]
    assert not fresh._start_files[1].exists()


def test_failed_launch_cancels_existing_steps(monkeypatch, tmp_path):
    module = _load_scheduler_module()
    _slurm_allocation(monkeypatch, module, "2")
    instance = _idle_scheduler(
        tmp_path,
        total_ranks=2,
        task_indices=[1, 2],
        job_task_count=2,
        launch_delay_seconds=0,
    )
    monkeypatch.setattr(instance, "_load_task_memory", lambda: [0, 0])

    class Process:
        terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            return 0

    process = Process()

    def popen(*args, **kwargs):
        if instance.running:
            raise OSError("cannot start launcher")
        return process

    monkeypatch.setattr(module.subprocess, "Popen", popen)
    with pytest.raises(SystemExit, match="cannot start launcher"):
        instance.run()
    assert process.terminated
    status = json.loads(instance.status_file.read_text())
    assert status["state"] == "failed"
    assert status["aborted_tasks"] == [1]
    assert status["running"] == status["launching"] == 0
