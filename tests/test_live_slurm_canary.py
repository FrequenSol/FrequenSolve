from pathlib import Path
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.hpc import live_canary as canary
from frequensolve.orchestrator.sites.hpc.live_canary import (
    CANARY_ACKNOWLEDGEMENT,
    LiveSlurmCanaryError,
    build_synthetic_live_slurm_job,
    require_live_slurm_acknowledgement,
)

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]


def site():
    return SimpleNamespace(
        config=SimpleNamespace(
            hostname="login.example.invalid",
            known_hosts_file="/synthetic/known_hosts",
            known_hosts_name="login.example.invalid",
            queue="synthetic",
        ),
        run_config=SimpleNamespace(
            queue="synthetic",
            nodes=1,
            duration="00:20:00",
            poll_interval=0,
            slurm_args=[],
            run_path=None,
        ),
        work_dir=Path("/approved/staging"),
        scratch_dir=Path("/approved/scratch"),
    )


def test_canary_requires_manual_acknowledgement():
    require_live_slurm_acknowledgement(CANARY_ACKNOWLEDGEMENT)
    with pytest.raises(LiveSlurmCanaryError, match="disabled"):
        require_live_slurm_acknowledgement(None)


def test_synthetic_fixture_is_small_and_uniquely_named(tmp_path):
    project_path = tmp_path / "project"
    project, success, cancellation = build_synthetic_live_slurm_job(
        project_path, "run-123456"
    )

    assert project.name.startswith("fs_canary_run_123456")
    assert success.name.endswith("_success")
    assert cancellation.name.endswith("_cancel")
    assert success.simulation is cancellation.simulation
    assert success.f_list == [20.0]
    with pytest.raises(LiveSlurmCanaryError, match="must not already exist"):
        build_synthetic_live_slurm_job(project_path, "run-654321")


def test_live_canary_uses_existing_public_submission_calls(monkeypatch, tmp_path):
    enterprise_site = site()
    calls = []
    scheduler = {"cancelled": False, "probes": 0}
    success_job = SimpleNamespace(name="fs_canary_success", _job_id=None)
    cancel_job = SimpleNamespace(name="fs_canary_cancel", _job_id=None)
    success_result = SimpleNamespace(
        status=SimpleNamespace(state="complete"),
        output_files=lambda existing: [Path("synthetic-output")],
    )
    cancel_result = SimpleNamespace(status=SimpleNamespace(state="cancelled"))
    success_handle = SimpleNamespace(
        id="10",
        watch=lambda **kwargs: [
            SimpleNamespace(state="pending"),
            SimpleNamespace(state="running"),
            SimpleNamespace(state="complete"),
        ],
        wait=lambda **kwargs: success_result,
        cancel=lambda: None,
    )

    def cancel():
        scheduler["cancelled"] = True

    cancel_handle = SimpleNamespace(
        id="20",
        wait=lambda **kwargs: cancel_result,
        cancel=cancel,
    )
    handles = iter((success_handle, cancel_handle))

    def submit(job, **kwargs):
        calls.append(kwargs)
        handle = next(handles)
        job._job_id = handle.id
        return handle

    def run_login(command):
        if "squeue -h -n " in command or "squeue -h -j 10" in command:
            return ""
        if "squeue -h -j 20" in command:
            if not scheduler["cancelled"]:
                return "20"
            scheduler["probes"] += 1
            return "" if scheduler["probes"] > 1 else "20"
        raise AssertionError(command)

    enterprise_site.submit = submit
    enterprise_site.run_login = run_login
    enterprise_site.enterprise_hpc_preflight = lambda: None
    fetched_logs = []
    enterprise_site.fetch_logs = lambda job: fetched_logs.append(job) or tmp_path
    monkeypatch.setattr(
        canary,
        "build_synthetic_live_slurm_job",
        lambda *args, **kwargs: (SimpleNamespace(), success_job, cancel_job),
    )
    monkeypatch.setattr(canary, "_trace_evidence", lambda result: {"finite": True})
    monkeypatch.setattr(
        canary,
        "_solver_evidence",
        lambda *args, **kwargs: {"mpiRanks": 4, "threadsPerRank": 2},
    )

    result = canary.run_live_slurm_canary(
        site=enterprise_site,
        project_path=tmp_path / "project",
        run_token="run-123456",
        ranks_per_node=4,
        timeout=1200,
        cancel_timeout=30,
    )

    assert scheduler["cancelled"] is True
    assert result["success"]["states"] == ["pending", "running", "complete"]
    assert result["cancellation"]["terminalState"] == "cancelled"
    assert calls[0]["fetch"] is True
    assert calls[1]["fetch"] is False
    assert fetched_logs == [success_job]
    assert all("threads_per_rank" not in call for call in calls)
    assert all(call["mode"] == "batch" for call in calls)


def test_solver_evidence_uses_public_fetched_logs(tmp_path):
    (tmp_path / "init.log").write_text(
        " -- Parallelism --\n Ranks  :    4\n Threads:    1\n"
    )
    (tmp_path / "task_1.log").write_text(
        "  Iter     ||r||_2         step\n"
        "       1   2.5339E-01   2.5043E+01\n"
        "      18   8.7280E-05   2.0403E-03\n"
        " -- Timing --\n"
    )

    assert canary._solver_evidence(tmp_path / "task_1.log", 4) == {
        "mpiRanks": 4,
        "threadsPerRank": 1,
        "iterations": 18,
        "residual": 8.728e-05,
    }


def test_live_canary_recovers_and_cancels_after_interrupted_submit(
    monkeypatch, tmp_path
):
    enterprise_site = site()
    job = SimpleNamespace(name="fs_canary_recover", _job_id=None)
    other = SimpleNamespace(name="fs_canary_other", _job_id=None)
    scheduler = {"active": False, "cancelled": []}

    def submit(selected, **kwargs):
        selected._job_id = "30"
        scheduler["active"] = True
        raise RuntimeError("synthetic post-submit transport loss")

    def run_login(command):
        if "squeue -h -n fs_canary_recover" in command:
            return "30" if scheduler["active"] else ""
        if "squeue -h -n fs_canary_other" in command:
            return ""
        if "squeue -h -j 30" in command:
            return "30" if scheduler["active"] else ""
        raise AssertionError(command)

    def cancel_job(job_id):
        scheduler["cancelled"].append(job_id)
        scheduler["active"] = False

    enterprise_site.submit = submit
    enterprise_site.run_login = run_login
    enterprise_site.cancel_job = cancel_job
    enterprise_site.enterprise_hpc_preflight = lambda: None
    monkeypatch.setattr(
        canary,
        "build_synthetic_live_slurm_job",
        lambda *args, **kwargs: (SimpleNamespace(), job, other),
    )

    with pytest.raises(RuntimeError, match="post-submit transport loss"):
        canary.run_live_slurm_canary(
            site=enterprise_site,
            project_path=tmp_path / "project",
            run_token="run-654321",
            ranks_per_node=4,
            timeout=1200,
            cancel_timeout=30,
        )

    assert scheduler == {"active": False, "cancelled": ["30"]}
