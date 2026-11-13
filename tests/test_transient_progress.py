"""Progress-file interoperability and lifecycle-safe time-step monitoring."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.base import JobStatus, RunHandle
from frequensolve.orchestrator.sites.hpc.site import SlurmSite
from frequensolve.orchestrator.sites.local.site import LocalSite
from frequensolve.orchestrator.utils.progress import (
    RunMonitor,
    status_table_html,
    status_text,
)
from frequensolve.orchestrator.utils.transient_progress import read_local_progress


@pytest.fixture
def run(tmp_path):
    job = SimpleNamespace(
        name="pulse", workflow="transient", time=SimpleNamespace(steps=100), n_tasks=1
    )
    job.task_manifest_file = lambda task, result_path=None: (
        Path(result_path or tmp_path) / f"task_{task}" / "result.json"
    )
    site = SimpleNamespace(
        site_name="local",
        _read_transient_progress=lambda run: read_local_progress(run.job),
    )
    return RunHandle(
        site=site,
        job=job,
        _status_fn=lambda run: JobStatus(state="running"),
        backend={"progress_not_before": 1000},
    )


def publish(run, completed=20, task=1, **overrides):
    path = run.job.task_manifest_file(task).with_name("progress.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        schema="fs-transient-progress-1",
        completed_steps=completed,
        total_steps=100,
        started_at=datetime.fromtimestamp(1000, timezone.utc).isoformat(),
        generation="attempt-1",
    )
    payload.update(overrides)
    path.write_text(json.dumps(payload))
    return path


def test_progress_changes_without_task_completion(run, capsys):
    monitor = RunMonitor([run], is_notebook=False)
    for completed in (20, 22):
        publish(run, completed)
        status = run.status()
        assert status.state == "running"
        assert status.progress == completed / 100
        monitor._emit([run], {0: status})
        panel = status_table_html([run], {0: status})
        assert f"{completed}/100 time steps completed" in panel
        assert "Succeeded" not in panel
    text = capsys.readouterr().out
    assert "20/100 time steps completed" in text
    assert "22/100 time steps completed" in text


@pytest.mark.parametrize(
    "override",
    [
        {"schema": "other"},
        {"completed_steps": -1},
        {"completed_steps": 101},
        {"completed_steps": True},
        {"completed_steps": 1.5},
        {"total_steps": 200},
        {"started_at": "1970-01-01T00:00:00+00:00"},
        {"started_at": "invalid"},
        {"started_at": "2026-10-29T00:00:00"},
    ],
)
def test_invalid_or_old_progress_is_ignored(run, override):
    publish(run, **override)
    status = run.status()
    assert status.progress is None
    assert "waiting for solver progress (100 time steps)" in status.message


def test_missing_and_malformed_files_preserve_last_good_progress(run):
    assert run.status().progress is None
    path = publish(run, 40)
    assert run.status().progress == 0.4
    path.write_text('{"completed_steps":')
    assert run.status().progress == 0.4
    path.unlink()
    assert run.status().progress == 0.4


def test_solver_start_in_submission_second_is_current(run):
    run.backend["progress_not_before"] = 1000.9
    publish(run, 20)
    assert run.status().progress == 0.2


def test_multimode_totals_include_reused_tasks_and_restart_steps(run):
    run.job.n_tasks = 3
    run.backend["task_plan"] = {"current_tasks": [1]}
    publish(run, 60, task=2)  # absolute accepted step after checkpoint restoration
    publish(run, 20, task=3)
    status = run.status()
    assert status.progress == 0.6
    assert "180/300 time steps completed" in status.message


def test_completed_steps_do_not_complete_a_running_or_failed_job(run):
    publish(run, 100)
    assert not run.status().is_complete
    run._status_fn = lambda run: JobStatus(
        state="failed", return_code=1, message="output failed"
    )
    status = run.status()
    assert not status.is_successful
    assert "output failed" in status_text([run], {0: status})
    assert "100/100 time steps completed" in status.message


def test_successful_old_solver_needs_no_progress_file(run):
    run._status_fn = lambda run: JobStatus(state="completed", return_code=0)
    assert run.status().progress == 1


def test_queued_job_does_not_read_previous_progress(run):
    publish(run, 100)
    run._status_fn = lambda run: JobStatus(state="pending")
    assert run.status().progress == 0


def test_finalizer_replacement_keeps_step_display(run):
    run._finalize_fn = lambda run, status: run._make_result(
        JobStatus(state="completed", return_code=0)
    )
    result = run._complete_from_status(JobStatus(state="completed", return_code=0))
    assert result.status.message == "100/100 time steps completed"


def test_local_reader_uses_task_paths(run):
    publish(run, 32)
    assert LocalSite._read_transient_progress(None, run)[1]["completed_steps"] == 32


def test_remote_reader_uses_remote_result_root(run, tmp_path):
    paths = []
    site = object.__new__(SlurmSite)
    site._remote_result_dir = lambda job: tmp_path / "remote results"
    site._read_remote_json = lambda path: paths.append(path) or {"completed_steps": 44}
    assert site._read_transient_progress(run)[1]["completed_steps"] == 44
    assert paths == [tmp_path / "remote results/task_1/progress.json"]


def test_frequency_jobs_keep_existing_counts(run):
    run.job.workflow = "forward"
    run._status_fn = lambda run: JobStatus(
        state="running",
        raw={
            "task_status": {
                "successful": 2,
                "failed": 0,
                "running": 1,
                "pending": 0,
                "total": 3,
            }
        },
    )
    status = run.status()
    assert "time_progress" not in status.raw
    assert "2 successful" in status_text([run], {0: status})
    assert "Succeeded" in status_table_html([run], {0: status})


def test_eta_excludes_checkpoint_steps_and_setup_time(run):
    publish(
        run,
        60,
        initial_step=50,
        elapsed_seconds=20,
        updated_at="2026-10-29T12:00:00+00:00",
    )
    status = run.status()
    estimate = datetime.fromisoformat(
        status.raw["time_progress"]["estimated_completion"]
    )
    assert (
        estimate.timestamp()
        == datetime(2026, 10, 29, 12, 1, 20, tzinfo=timezone.utc).timestamp()
    )
    assert "estimated finish ~" in status.message
    assert "estimated finish ~" in status_table_html([run], {0: status})
    # Polling an unchanged file must not move the predicted finish into the future.
    assert run.status().raw["time_progress"] == status.raw["time_progress"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"initial_step": 60, "elapsed_seconds": 2},
        {"initial_step": 50, "elapsed_seconds": 0},
        {"initial_step": 50, "elapsed_seconds": float("nan")},
        {"initial_step": 50, "elapsed_seconds": float("inf")},
        {"initial_step": 50, "elapsed_seconds": 2, "updated_at": "invalid"},
    ],
)
def test_eta_waits_for_valid_measured_advances(run, overrides):
    publish(run, 60, **overrides)
    status = run.status()
    assert status.progress == 0.6
    assert "estimated_completion" not in status.raw["time_progress"]


def test_multimode_eta_uses_latest_finish_and_waits_for_unstarted_tasks(run):
    run.job.n_tasks = 2
    publish(
        run,
        60,
        task=1,
        initial_step=50,
        elapsed_seconds=20,
        updated_at="2026-10-29T12:00:00+00:00",
    )
    assert "estimated_completion" not in run.status().raw["time_progress"]
    publish(
        run,
        20,
        task=2,
        initial_step=0,
        elapsed_seconds=60,
        updated_at="2026-10-29T12:00:00+00:00",
    )
    status = run.status()
    estimate = datetime.fromisoformat(
        status.raw["time_progress"]["estimated_completion"]
    )
    assert (
        estimate.timestamp()
        == datetime(2026, 10, 29, 12, 4, tzinfo=timezone.utc).timestamp()
    )


def test_sauce_compact_timezone_is_supported(run):
    publish(
        run,
        20,
        initial_step=0,
        elapsed_seconds=5,
        started_at="2026-10-29T13:00:00-0500",
        updated_at="2026-10-29T13:01:00-0500",
    )
    status = run.status()
    assert status.progress == 0.2
    stamp = datetime.fromisoformat(status.raw["time_progress"]["estimated_completion"])
    assert (
        stamp.timestamp()
        == datetime(2026, 10, 29, 18, 1, 20, tzinfo=timezone.utc).timestamp()
    )


def test_session_async_wait_reports_intermediate_progress(run, capsys):
    import asyncio

    from frequensolve.orchestrator.sites.hpc.session import AllocationSession

    async def exercise():
        session = object.__new__(AllocationSession)
        session.base_site = SimpleNamespace(
            _read_transient_progress=lambda run: read_local_progress(run.job),
        )
        session._is_notebook = False
        session._emit_status = lambda status, **kwargs: print(status.message)
        session.generation = "session-generation"
        session.allocation_id = "12345"
        state = {"state": "running", "generation": session.generation}
        session._read = lambda path: dict(state)
        session.root = Path("/session")
        session.status = lambda: {"state": "ready"}
        run.site = session
        run.id = "a" * 32
        run._status_fn = session._poll
        publish(run, 20)
        asyncio.get_running_loop().call_later(
            0.05, lambda: state.update(state="complete")
        )
        return await run.wait_async(poll_interval=0.01)

    result = asyncio.run(exercise())
    assert result.successful
    assert "20/100 time steps completed" in capsys.readouterr().out
