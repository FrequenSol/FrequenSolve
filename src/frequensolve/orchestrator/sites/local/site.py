"""Local execution site using Dask workers and local filesystem artifacts."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import signal
import subprocess
import time
import warnings
from dataclasses import dataclass, field
from logging import ERROR
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
from urllib.parse import urlparse

from frequensolve._optional import optional_dependency_error

try:
    from dask import config as dask_config
    from dask.distributed import Client, Future, LocalCluster, wait
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "LocalSite",
        extra="parallel",
        dependencies=("dask", "distributed"),
        error=exc,
    ) from exc

from numpy.typing import ArrayLike

from frequensolve.orchestrator.sites.base import (
    BaseSite,
    JobStatus,
    RunHandle,
    RunResult,
    _merge_task_status_with_plan,
    _wait_for_path,
)
from frequensolve.orchestrator.sites.local.config import LocalSiteConfig
from frequensolve.orchestrator.sites.local.dask_logging import (
    configure_dependency_logging,
)
from frequensolve.orchestrator.utils.environment import (
    build_subprocess_environment,
    validate_environment,
)
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.jobs import BaseJob, SkipPolicy
from frequensolve.simulation.jobs.imaging import ImageDatabase, ImagingJob

logger = logging.getLogger(__name__)

__all__ = ["LocalSite", "run_task"]

PACK_TASK_ID = -3
SMOOTH_TASK_ID = -2
MESH_TASK_ID = -1
DASK_LOGGING_PRELOAD = "frequensolve.orchestrator.sites.local.dask_logging"

# TODO: print this when running job
# print("Dask dashboard:", site.dashboard_url or "not available")


def _task_log_name(task_id: int) -> str:
    if task_id == PACK_TASK_ID:
        return "pack.log"
    if task_id == SMOOTH_TASK_ID:
        return "smooth.log"
    if task_id == MESH_TASK_ID:
        return "init.log"
    return f"task_{task_id + 1}.log"


def _job_result_path(job_file: Union[str, Path]) -> Optional[Path]:
    try:
        job_file = Path(job_file)
        data = json.loads(job_file.read_text())
    except Exception:
        return None
    result_path = data.get("result_path")
    if not result_path:
        return None
    path = Path(str(result_path))
    if path.is_absolute():
        return path
    project_path = data.get("project_path")
    if project_path:
        return Path(str(project_path)) / path
    return job_file.parent / path


def _task_run_manifest_path(job_file: Union[str, Path], task_id: int) -> Optional[Path]:
    if task_id < 0:
        return None
    result_path = _job_result_path(job_file)
    if result_path is None:
        return None
    return (
        result_path
        / "_fs_run"
        / "tasks"
        / f"task_{task_id + 1:06d}"
        / "run_manifest.json"
    )


def _read_task_solver_convergence(
    job_file: Union[str, Path], task_id: int
) -> tuple[Optional[Path], Optional[Dict[str, object]]]:
    manifest_path = _task_run_manifest_path(job_file, task_id)
    if manifest_path is None or not manifest_path.exists():
        return manifest_path, None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return manifest_path, None
    return manifest_path, BaseJob.solver_convergence_summary(manifest)


def _fallback_task_summary(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    summary = {
        "total": 0,
        "complete": 0,
        "succeeded": 0,
        "failed": 0,
        "not_run": 0,
    }
    for record in records:
        summary["total"] += 1
        status = BaseJob._normalized_task_status(record.get("status"))
        if record.get("complete") or status in {"succeeded", "current", "skipped"}:
            summary["complete"] += 1
        if status == "succeeded":
            summary["succeeded"] += 1
        elif status == "failed":
            summary["failed"] += 1
        elif status == "not_run":
            summary["not_run"] += 1
    return summary


def _job_task_summary(
    job: BaseJob, task_records: Iterable[Mapping[str, Any]]
) -> Dict[str, int]:
    if hasattr(job, "run_state"):
        try:
            state = job.run_state()
        except Exception:
            state = {}
        summary = state.get("task_summary") if isinstance(state, Mapping) else None
        if isinstance(summary, Mapping):
            return {
                key: int(summary.get(key, 0))
                for key in ("total", "complete", "succeeded", "failed", "not_run")
            }
    return _fallback_task_summary(task_records)


def _task_summary_message(summary: Mapping[str, int]) -> str:
    parts = [
        f"{int(summary.get('succeeded', 0))} succeeded",
        f"{int(summary.get('failed', 0))} failed",
        f"{int(summary.get('complete', 0))} complete",
    ]
    not_run = int(summary.get("not_run", 0))
    if not_run:
        parts.append(f"{not_run} not run")
    total = int(summary.get("total", 0))
    if total:
        parts.append(f"{total} total")
    return "tasks: " + ", ".join(parts)


def _local_task_state(status: Any) -> str:
    status = str(status).strip().lower().replace(" ", "_")
    if status in {
        "accepted",
        "accepted_failed",
        "finished",
        "success",
        "successful",
        "succeeded",
        "complete",
        "completed",
        "done",
        "current",
        "reused",
        "skipped",
    }:
        return "successful"
    if status in {
        "error",
        "failed",
        "failure",
        "cancelled",
        "canceled",
        "killed",
        "timeout",
    }:
        return "failed"
    if status in {"not_run", "not-run", "created"}:
        return "pending"
    return "running"


def _task_result_successful(result: Mapping[str, Any]) -> bool:
    return _local_task_state(result.get("status")) == "successful"


def _collect_future_results(
    futures: Iterable[Future],
) -> tuple[List[Mapping[str, Any]], List[Dict[str, Any]]]:
    results: List[Mapping[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for index, future in enumerate(futures):
        try:
            result = future.result()
        except Exception as exc:
            errors.append(
                {
                    "future_index": index,
                    "status": getattr(future, "status", None),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            )
        else:
            if isinstance(result, Mapping):
                results.append(dict(result))
            else:
                results.append({"status": "success", "result": result})
    return results, errors


def _local_task_status(statuses: Iterable[Any]) -> Dict[str, int]:
    states = [_local_task_state(status) for status in statuses]
    total = len(states)
    failed = sum(state == "failed" for state in states)
    successful = sum(state == "successful" for state in states)
    running = sum(state == "running" for state in states)
    pending = sum(state == "pending" for state in states)
    return {
        "successful": successful,
        "failed": failed,
        "running": running,
        "pending": pending,
        "total": total,
    }


def _local_task_status_message(status: Mapping[str, int]) -> str:
    return (
        "tasks: "
        f"{int(status.get('successful', 0))} successful, "
        f"{int(status.get('failed', 0))} failed, "
        f"{int(status.get('running', 0))} running, "
        f"{int(status.get('pending', 0))} pending, "
        f"{int(status.get('total', 0))} total"
    )


def _summary_task_status(summary: Mapping[str, int]) -> Dict[str, int]:
    return {
        "successful": int(summary.get("succeeded", 0)),
        "failed": int(summary.get("failed", 0)),
        "running": 0,
        "pending": int(summary.get("not_run", 0)),
        "total": int(summary.get("total", 0)),
    }


def _normalize_failure_tolerance(value: Any, *, default: Optional[int] = 4):
    if value is None:
        return None
    if isinstance(value, bool):
        return default if value else 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            return default
        if text in {"none", "unlimited", "inf", "infinite"}:
            return None
        if text in {"false", "off", "no"}:
            return 0
        if text in {"true", "on", "yes"}:
            return default
    count = int(value)
    if count < 0:
        return None
    return count


def _failure_tolerance_exceeded(
    summary: Mapping[str, int],
    tolerate_failures: Optional[int],
) -> bool:
    return tolerate_failures is not None and int(summary.get("failed", 0)) > int(
        tolerate_failures
    )


def run_task(
    job_file: str,
    task_id: int,
    executable: str,
    env: dict,
    n_ranks: int = 1,
    n_threads: int = 1,
    stdout_dir: str = None,
    fresh: bool = False,
) -> dict:
    """Run a single task and return its results.

    Args:
        job_file: Path to the job file
        task_id: Task ID to run
        executable: Path to the solver executable
        env: Environment variables
        n_ranks: Number of MPI ranks
        n_threads: Number of threads per rank
        stdout_dir: Directory to store task logs
        fresh: Pass --fresh to the solver to disable solver-side output reuse

    Returns:
        Dict containing task results
    """
    _wait_for_path(job_file)
    if n_ranks < 1:
        raise ValueError("n_ranks must be at least 1")
    threads_per_rank = max(1, n_threads // n_ranks)
    core_count = n_ranks * threads_per_rank
    if n_ranks > 1:
        args = [
            "mpirun",
            "-np",
            f"{n_ranks}",
        ]
    else:
        args = []

    args += [
        executable,
        "-nthreads",
        f"{threads_per_rank}",
        "--job",
        f"{job_file}",
    ]
    if fresh:
        args += ["--fresh"]
    if task_id == PACK_TASK_ID:
        args += ["--pack"]
    elif task_id == SMOOTH_TASK_ID:
        args += ["--smooth"]
    elif task_id == MESH_TASK_ID:
        args += ["--init", "--map"]
    else:
        args += ["--task", f"{task_id + 1}"]
    command = shlex.join(args)
    logger.info("Executing: %s", command)

    if stdout_dir:
        os.makedirs(stdout_dir, exist_ok=True)
        stdout_file = os.path.join(stdout_dir, _task_log_name(task_id))
    else:
        stdout_file = None
    started = time.perf_counter()
    manifest_path: Optional[Path] = None
    solver_convergence: Optional[Dict[str, object]] = None
    try:
        stdout_path = stdout_file if stdout_file else os.devnull
        with open(stdout_path, "w") as stdout:
            if stdout_file:
                stdout.write(f"[INFO] {logger.name}: Executing: {command}\n")
                stdout.flush()
            proc = subprocess.Popen(
                args, stdout=stdout, stderr=stdout, env=env, text=True
            )
            return_code = proc.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, args)

        manifest_path, solver_convergence = _read_task_solver_convergence(
            job_file, task_id
        )
        solver_failed = bool(
            solver_convergence is not None and solver_convergence.get("failed")
        )

        result = {
            "task_id": task_id,
            "status": "error" if solver_failed else "success",
            "complete": True,
            "returncode": return_code,
            "duration_seconds": time.perf_counter() - started,
            "n_ranks": n_ranks,
            "threads_per_rank": threads_per_rank,
            "core_count": core_count,
            "stdout": (
                os.path.join(stdout_dir, _task_log_name(task_id))
                if stdout_dir
                else None
            ),
        }
        if manifest_path is not None:
            result["run_manifest"] = str(manifest_path)
        if solver_convergence is not None:
            result["solver"] = {"convergence": solver_convergence}
        if solver_failed:
            residual = solver_convergence.get(
                "residual", solver_convergence.get("final_residual")
            )
            result["error"] = "Solver convergence failed" + (
                f"; residual {residual}" if residual is not None else ""
            )
        return result
    except Exception as e:
        manifest_path, solver_convergence = _read_task_solver_convergence(
            job_file, task_id
        )
        result = {
            "task_id": task_id,
            "status": "error",
            "complete": False,
            "error": str(e),
            "duration_seconds": time.perf_counter() - started,
            "n_ranks": n_ranks,
            "threads_per_rank": threads_per_rank,
            "core_count": core_count,
            "stdout": (
                os.path.join(stdout_dir, _task_log_name(task_id))
                if stdout_dir
                else None
            ),
        }
        if manifest_path is not None:
            result["run_manifest"] = str(manifest_path)
        if solver_convergence is not None:
            result["solver"] = {"convergence": solver_convergence}
        return result


@dataclass
class LocalTaskSubmission:
    """Dask futures and task-plan metadata for one local job submission.

    Args:
        futures: Dask futures representing submitted mesh/frequency tasks.
        task_plan: Serialized task plan used by status reporting and run
            metadata.
    """

    futures: List["Future"]
    task_plan: Dict[str, object]


@dataclass(kw_only=True)
class LocalSite(BaseSite):
    """Site for local execution using a Dask local cluster.

    Args:
        verbose: Whether to print site status messages in addition to logging.
        n_workers: Optional Dask worker count.
        threads_per_worker: Optional thread count per Dask worker.
        memory_per_worker: Optional worker memory limit.
        solver: Path to the local solver executable.
        environment: Non-secret environment values added to worker and solver
            subprocesses.
        shutdown_on_completion: Whether to close the Dask cluster after a run
            completes.
        dashboard_host: Hostname used for the Dask dashboard.
        dashboard_port: Dashboard port, or ``0`` to let Dask choose.
    """

    config: LocalSiteConfig = field(init=False)
    executable: str = field(init=False)
    env: dict = field(default_factory=dict)
    solver: Optional[Union[str, Path]] = None
    environment: Mapping[str, object] = field(default_factory=dict)
    n_workers: Optional[int] = None
    threads_per_worker: Optional[int] = None
    memory_per_worker: Optional[int] = None
    shutdown_on_completion: bool = True
    dashboard_host: str = "localhost"
    dashboard_port: int = 0

    _dask_client: Optional[Client] = field(default=None, init=False)
    _dask_cluster: Optional[LocalCluster] = field(default=None, init=False)
    _futures: List["Future"] = field(default_factory=list, init=False)
    _dashboard_port: Optional[str] = field(default=None, init=False)
    _dashboard_url: Optional[str] = field(default=None, init=False)
    _active_n_workers: Optional[int] = field(default=None, init=False)
    _active_threads_per_worker: Optional[int] = field(default=None, init=False)
    _active_memory_per_worker: Optional[int] = field(default=None, init=False)
    _closed: bool = field(default=True, init=False)

    # ----------------- lifecycle -----------------

    def __post_init__(self):
        self.config = LocalSiteConfig()
        self.executable = self._get_solver_path()
        explicit_environment = {
            **self.env,
            **validate_environment(self.environment),
        }
        self.env = build_subprocess_environment(
            defaults={"VECLIB_MAXIMUM_THREADS": "1"},
            overrides=explicit_environment,
        )
        self._quiet_dependency_loggers()

    def submit(self, job: BaseJob, **kwargs) -> RunHandle:
        """Submit job and return an awaitable run handle.

        Args:
            job: The simulation job to run
            **kwargs: Additional arguments for task configuration. Pass
                ``check=True`` to make ``wait()`` raise by default for failed
                runs, or ``validate=False`` to skip SDK pre-run validation.

        Returns:
            RunHandle for the submitted tasks
        """
        check = bool(kwargs.pop("check", False))
        force_run = bool(
            kwargs.pop("force_run", False)
            or kwargs.pop("force", False)
            or kwargs.pop("rerun", False)
        )
        skip_policy_value = kwargs.pop("skip", kwargs.pop("skip_policy", None))
        residual = kwargs.pop("residual", None)
        ignore_solver_options = kwargs.pop("ignore_solver_options", None)
        reuse = bool(kwargs.pop("reuse", True))
        skip_policy = SkipPolicy.from_value(
            skip_policy_value,
            residual=residual,
            ignore_solver_options=ignore_solver_options,
            reuse=reuse and not force_run,
        )
        plan_skip_policy = (
            skip_policy
            if (
                skip_policy_value is not None
                or residual is not None
                or ignore_solver_options is not None
            )
            else None
        )
        force_run = bool(force_run or skip_policy.force)
        validate = kwargs.pop("validate", True)
        pack = bool(kwargs.pop("pack", True))
        tolerate_failures = _normalize_failure_tolerance(
            kwargs.pop("tolerate_failures", 4),
            default=4,
        )
        shutdown_on_completion = bool(
            kwargs.pop("shutdown_on_completion", self.shutdown_on_completion)
        )
        self.prepare_job(job, validate=validate)
        if not force_run and job.is_run_current():
            logger.info(
                "Skipping job %s; fingerprint matches and expected trace outputs exist.",
                job.name,
            )
            job.write_run_state(status="skipped")
            return RunHandle.skipped(self, job)

        try:
            submission = self._submit_local_tasks(
                job,
                force_run=force_run,
                skip_policy=plan_skip_policy,
                reuse=reuse,
                **kwargs,
            )
        except Exception:
            if shutdown_on_completion:
                self.close(wait=False, retire=False)
            raise
        if isinstance(submission, LocalTaskSubmission):
            futures = submission.futures
            task_plan = submission.task_plan
        else:
            futures = submission
            task_plan = {}
        smooth_only = (
            isinstance(job, ImagingJob)
            and not futures
            and task_plan
            and job.needs_image_smoothing()
        )
        if not futures and task_plan and not smooth_only:
            skipped = task_plan.get(
                "skipped_task_records",
                task_plan.get("reused_tasks", []),
            )
            job.write_run_state(status="skipped", tasks=skipped)
            return RunHandle.skipped(self, job, "No frequency tasks need to run")
        handle = RunHandle(
            site=self,
            job=job,
            id=f"local:{job.name}",
            mode="local",
            poll_interval=0.5,
            check=check,
            _status_fn=self._poll_local_run,
            _wait_fn=self._wait_local_run,
            _finalize_fn=self._finalize_local_run,
            _timeout_fn=self._timeout_local_run,
            _cancel_fn=self._cancel_local_run,
        )
        handle.backend["futures"] = futures
        handle.backend["task_plan"] = task_plan
        handle.backend["pack_after_tasks"] = pack
        handle.backend["fresh"] = force_run
        handle.backend["smooth_only"] = smooth_only
        handle.backend["tolerate_failures"] = tolerate_failures
        handle.backend["shutdown_on_completion"] = shutdown_on_completion
        return handle

    def _submit_local_smooth_task(
        self,
        run: RunHandle,
        all_futures: Optional[List[Future]] = None,
    ) -> Future:
        """Submit the imaging smooth/stack task once and return its future."""

        smooth_future = run.backend.get("smooth_future")
        if smooth_future is not None:
            if all_futures is not None and smooth_future not in all_futures:
                all_futures.append(smooth_future)
            return smooth_future

        if self._dask_client is None:
            self._ensure_dask_for_tasks(1)
        smooth_future = self._dask_client.submit(
            run_task,
            run.job._file,
            SMOOTH_TASK_ID,
            self.executable,
            self.env,
            n_ranks=1,
            n_threads=self._current_threads_per_worker(),
            stdout_dir=str(run.job._stdout_path),
            fresh=bool(run.backend.get("fresh", False)),
            resources={"CPU": self._current_threads_per_worker()},
        )
        run.backend["smooth_future"] = smooth_future
        self._futures.append(smooth_future)
        if all_futures is not None:
            all_futures.append(smooth_future)
        return smooth_future

    def _poll_local_smooth_task(
        self,
        run: RunHandle,
        smooth_future: Future,
        *,
        raw: Optional[Dict[str, Any]] = None,
    ) -> JobStatus:
        """Poll an imaging smooth/stack task as part of local run status."""

        raw = dict(raw or {})
        smooth_state = _local_task_state(getattr(smooth_future, "status", "unknown"))
        if smooth_state == "successful":
            smooth_result = smooth_future.result()
            if isinstance(smooth_result, Mapping):
                smooth_result = dict(smooth_result)
            else:
                smooth_result = {"status": "success", "result": smooth_result}
            run.backend["smooth_result"] = smooth_result
            raw["smooth"] = smooth_result
            if not _task_result_successful(smooth_result):
                return JobStatus(
                    state="failed",
                    return_code=1,
                    job_id=run.id,
                    message="Image smoothing task failed",
                    raw=raw,
                )
            return JobStatus(
                state="completed",
                return_code=0,
                job_id=run.id,
                message="Image smoothing task completed",
                raw=raw,
            )
        if smooth_state == "failed":
            try:
                raw["smooth"] = smooth_future.result()
            except Exception as exc:
                raw["smooth_error"] = str(exc)
                run.backend["smooth_error"] = {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            return JobStatus(
                state="failed",
                return_code=1,
                job_id=run.id,
                message="Image smoothing task failed",
                raw=raw,
            )
        return JobStatus(
            state="running",
            job_id=run.id,
            message="Image smoothing task running",
            raw=raw,
        )

    def _poll_local_run(self, run: RunHandle) -> JobStatus:
        futures = run.backend.get("futures", [])
        if not futures:
            if run.backend.get("smooth_only", False):
                smooth_future = self._submit_local_smooth_task(run)
                return self._poll_local_smooth_task(run, smooth_future)
            return JobStatus(state="skipped", return_code=0, job_id=run.id)
        statuses = [getattr(future, "status", "unknown") for future in futures]
        states = [_local_task_state(status) for status in statuses]
        task_status = _merge_task_status_with_plan(
            _local_task_status(statuses),
            run.backend.get("task_plan"),
            job=run.job,
        )
        terminal = {"successful", "failed"}
        if all(state in terminal for state in states) and isinstance(
            run.job, ImagingJob
        ):
            task_results, task_result_errors = _collect_future_results(futures)
            run.backend["task_results"] = task_results
            if task_result_errors:
                run.backend["task_result_errors"] = task_result_errors
                raw = {
                    "statuses": statuses,
                    "task_states": states,
                    "task_status": task_status,
                    "tasks": task_results,
                    "task_result_errors": task_result_errors,
                }
                return JobStatus(
                    state="failed",
                    return_code=1,
                    job_id=run.id,
                    message="One or more local task results could not be read",
                    raw=raw,
                )
            smooth_future = self._submit_local_smooth_task(run)
            return self._poll_local_smooth_task(
                run,
                smooth_future,
                raw={
                    "statuses": statuses,
                    "task_states": states,
                    "task_status": task_status,
                    "tasks": task_results,
                },
            )
        if all(state in terminal for state in states) and any(
            state == "failed" for state in states
        ):
            task_results, task_result_errors = _collect_future_results(futures)
            run.backend["task_results"] = task_results
            if task_result_errors:
                run.backend["task_result_errors"] = task_result_errors
            return JobStatus(
                state="failed",
                return_code=1,
                job_id=run.id,
                message=_local_task_status_message(task_status),
                raw={
                    "statuses": statuses,
                    "task_states": states,
                    "task_status": task_status,
                    "tasks": task_results,
                    **(
                        {"task_result_errors": task_result_errors}
                        if task_result_errors
                        else {}
                    ),
                },
            )
        if all(state == "successful" for state in states):
            task_results, task_result_errors = _collect_future_results(futures)
            run.backend["task_results"] = task_results
            if task_result_errors:
                run.backend["task_result_errors"] = task_result_errors
                return JobStatus(
                    state="failed",
                    return_code=1,
                    job_id=run.id,
                    message="One or more local task results could not be read",
                    raw={
                        "statuses": statuses,
                        "task_states": states,
                        "task_status": task_status,
                        "tasks": task_results,
                        "task_result_errors": task_result_errors,
                    },
                )
            return JobStatus(
                state="completed",
                return_code=0,
                job_id=run.id,
                message=_local_task_status_message(task_status),
                raw={
                    "statuses": statuses,
                    "task_states": states,
                    "task_status": task_status,
                    "tasks": task_results,
                },
            )
        return JobStatus(
            state="running",
            job_id=run.id,
            message=_local_task_status_message(task_status),
            raw={
                "statuses": statuses,
                "task_states": states,
                "task_status": task_status,
            },
        )

    def _wait_local_run(
        self,
        run: RunHandle,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        futures = run.backend.get("futures", [])
        if not futures:
            if run.backend.get("smooth_only", False):
                return self._wait_local_smooth_only(run)
            skipped = run.backend.get("task_plan", {}).get(
                "skipped_task_records",
                run.backend.get("task_plan", {}).get("reused_tasks", []),
            )
            if skipped:
                run.job.write_run_state(status="skipped", tasks=skipped)
            status = JobStatus(state="skipped", return_code=0, job_id=run.id)
            self._emit_status(status)
            return run._make_result(status)

        smooth_future = None
        pack_future = None
        all_futures = list(futures)

        try:
            cached_task_results = run.backend.get("task_results")
            if cached_task_results is None:
                wait_result = wait(futures, timeout=timeout)
                not_done = list(getattr(wait_result, "not_done", []))
                if not_done:
                    self._cancel_futures(not_done)
                    status = JobStatus(
                        state="timeout",
                        return_code=1,
                        job_id=run.id,
                        message=(
                            f"Timed out waiting for local run after {timeout} seconds"
                        ),
                        raw={"unfinished": len(not_done)},
                    )
                    run.job.write_run_state(status="timeout")
                    return run._make_result(status)

                task_results, task_result_errors = _collect_future_results(futures)
                run.backend["task_results"] = task_results
                if task_result_errors:
                    run.backend["task_result_errors"] = task_result_errors
                    status = JobStatus(
                        state="failed",
                        return_code=1,
                        job_id=run.id,
                        message="One or more local task results could not be read",
                        raw={
                            "tasks": task_results,
                            "task_result_errors": task_result_errors,
                        },
                    )
                    run.job.write_run_state(
                        status="failed",
                        tasks=task_results,
                        errors=task_result_errors,
                    )
                    return run._make_result(status)
            else:
                task_results = [dict(result) for result in cached_task_results]
                task_result_errors = list(run.backend.get("task_result_errors", []))
                if task_result_errors:
                    status = JobStatus(
                        state="failed",
                        return_code=1,
                        job_id=run.id,
                        message="One or more local task results could not be read",
                        raw={
                            "tasks": task_results,
                            "task_result_errors": task_result_errors,
                        },
                    )
                    run.job.write_run_state(
                        status="failed",
                        tasks=task_results,
                        errors=task_result_errors,
                    )
                    return run._make_result(status)
            skipped_tasks = run.backend.get("task_plan", {}).get(
                "skipped_task_records",
                run.backend.get("task_plan", {}).get("reused_tasks", []),
            )
            state_tasks = [*task_results, *skipped_tasks]
            if hasattr(run.job, "invalidate_trace_cache"):
                run.job.invalidate_trace_cache()
            task_errors = [
                result for result in task_results if not _task_result_successful(result)
            ]

            smooth_result = None
            if isinstance(run.job, ImagingJob):
                smooth_future = self._submit_local_smooth_task(run, all_futures)
                cached_smooth_result = run.backend.get("smooth_result")
                if cached_smooth_result is None:
                    smooth_result = smooth_future.result()
                    if isinstance(smooth_result, Mapping):
                        smooth_result = dict(smooth_result)
                    else:
                        smooth_result = {"status": "success", "result": smooth_result}
                    run.backend["smooth_result"] = smooth_result
                else:
                    smooth_result = dict(cached_smooth_result)
                if not _task_result_successful(smooth_result):
                    status = JobStatus(
                        state="failed",
                        return_code=1,
                        job_id=run.id,
                        message="Image smoothing task failed",
                        raw={"smooth": smooth_result, "tasks": task_results},
                    )
                    run.job.write_run_state(
                        status="failed", tasks=state_tasks, smooth=smooth_result
                    )
                    return run._make_result(status)

            pack_result = None
            if run.backend.get("pack_after_tasks", False):
                if self._dask_client is None:
                    raise RuntimeError("Cannot run solver packing task without Dask")
                if hasattr(run.job, "remove_packed_trace_products"):
                    run.job.remove_packed_trace_products()
                pack_future = self._dask_client.submit(
                    run_task,
                    run.job._file,
                    PACK_TASK_ID,
                    self.executable,
                    self.env,
                    n_ranks=1,
                    n_threads=self._current_threads_per_worker(),
                    stdout_dir=str(run.job._stdout_path),
                    fresh=bool(run.backend.get("fresh", False)),
                    resources={"CPU": self._current_threads_per_worker()},
                )
                all_futures.append(pack_future)
                pack_result = pack_future.result()
                if isinstance(pack_result, Mapping):
                    pack_result = dict(pack_result)
                else:
                    pack_result = {"status": "success", "result": pack_result}
                if not _task_result_successful(pack_result):
                    outputs_exist = False
                    if hasattr(run.job, "trace_outputs_exist"):
                        try:
                            outputs_exist = bool(run.job.trace_outputs_exist())
                        except Exception:
                            outputs_exist = False
                    if outputs_exist and not task_errors:
                        run.job.write_run_state(
                            status="completed",
                            tasks=state_tasks,
                            pack=pack_result,
                            pack_error=pack_result,
                        )
                        task_summary = _job_task_summary(run.job, state_tasks)
                        message = _task_summary_message(task_summary)
                        pack_error = pack_result.get("error") or pack_result.get(
                            "stderr"
                        )
                        if pack_error:
                            message = f"{message}; packing failed: {pack_error}"
                        else:
                            message = f"{message}; packing failed"
                        raw = {
                            "tasks": task_results,
                            "task_summary": task_summary,
                            "task_status": _summary_task_status(task_summary),
                            "pack": pack_result,
                            "pack_error": pack_result,
                        }
                        if smooth_result is not None:
                            raw["smooth"] = smooth_result
                        status = JobStatus(
                            state="completed",
                            return_code=0,
                            job_id=run.id,
                            message=message,
                            raw=raw,
                        )
                        return run._make_result(status)
                    raw = {"pack": pack_result, "tasks": task_results}
                    if smooth_result is not None:
                        raw["smooth"] = smooth_result
                    status = JobStatus(
                        state="failed",
                        return_code=1,
                        job_id=run.id,
                        message="Packing task failed",
                        raw=raw,
                    )
                    run.job.write_run_state(
                        status="failed",
                        tasks=state_tasks,
                        pack=pack_result,
                        **(
                            {"smooth": smooth_result}
                            if smooth_result is not None
                            else {}
                        ),
                    )
                    return run._make_result(status)

            run.job.write_run_state(
                status="completed",
                tasks=state_tasks,
                **({"errors": task_errors} if task_errors else {}),
                **({"smooth": smooth_result} if smooth_result is not None else {}),
                **({"pack": pack_result} if pack_result is not None else {}),
            )
            task_summary = _job_task_summary(run.job, state_tasks)
            tolerate_failures = run.backend.get("tolerate_failures", 4)
            if _failure_tolerance_exceeded(task_summary, tolerate_failures):
                run.job.write_run_state(
                    status="failed",
                    tasks=state_tasks,
                    **({"errors": task_errors} if task_errors else {}),
                    **({"smooth": smooth_result} if smooth_result is not None else {}),
                    **({"pack": pack_result} if pack_result is not None else {}),
                    failure_tolerance=tolerate_failures,
                )
                task_summary = _job_task_summary(run.job, state_tasks)
                failed = int(task_summary.get("failed", 0))
                status = JobStatus(
                    state="failed",
                    return_code=1,
                    job_id=run.id,
                    message=(
                        f"{_task_summary_message(task_summary)}; failure tolerance "
                        f"exceeded ({failed} failed, tolerate_failures="
                        f"{tolerate_failures})"
                    ),
                    raw={
                        "tasks": task_results,
                        "task_summary": task_summary,
                        "task_status": _summary_task_status(task_summary),
                        "tolerate_failures": tolerate_failures,
                        **({"errors": task_errors} if task_errors else {}),
                        **(
                            {"smooth": smooth_result}
                            if smooth_result is not None
                            else {}
                        ),
                        **({"pack": pack_result} if pack_result is not None else {}),
                    },
                )
                return run._make_result(status)
            raw = {
                "tasks": task_results,
                "task_summary": task_summary,
                "task_status": _summary_task_status(task_summary),
                **({"errors": task_errors} if task_errors else {}),
                **({"pack": pack_result} if pack_result is not None else {}),
            }
            if smooth_result is not None:
                raw["smooth"] = smooth_result
            status = JobStatus(
                state="completed",
                return_code=0,
                job_id=run.id,
                message=_task_summary_message(task_summary),
                raw=raw,
            )
            return run._make_result(status)
        except Exception as exc:
            self._cancel_futures(all_futures)
            try:
                cached_tasks = run.backend.get("task_results")
                run.job.write_run_state(
                    status="failed",
                    **({"tasks": cached_tasks} if cached_tasks is not None else {}),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            except Exception:
                logger.debug("Could not write failed run state", exc_info=True)
            status = JobStatus(
                state="failed",
                return_code=1,
                job_id=run.id,
                message=str(exc),
                raw={"error_type": type(exc).__name__},
            )
            return run._make_result(status)
        finally:
            self._release_futures(all_futures)
            if run.backend.get("shutdown_on_completion", self.shutdown_on_completion):
                self.close(wait=True, retire=True)

    def _wait_local_smooth_only(self, run: RunHandle) -> RunResult:
        """Run only the imaging smooth/stack postprocess for current shards."""

        all_futures = []
        try:
            smooth_future = self._submit_local_smooth_task(run, all_futures)
            cached_smooth_result = run.backend.get("smooth_result")
            if cached_smooth_result is None:
                smooth_result = smooth_future.result()
                if isinstance(smooth_result, Mapping):
                    smooth_result = dict(smooth_result)
                else:
                    smooth_result = {"status": "success", "result": smooth_result}
                run.backend["smooth_result"] = smooth_result
            else:
                smooth_result = dict(cached_smooth_result)
            skipped = run.backend.get("task_plan", {}).get(
                "skipped_task_records",
                run.backend.get("task_plan", {}).get("reused_tasks", []),
            )
            if not _task_result_successful(smooth_result):
                status = JobStatus(
                    state="failed",
                    return_code=1,
                    job_id=run.id,
                    message="Image smoothing task failed",
                    raw={"smooth": smooth_result},
                )
                run.job.write_run_state(
                    status="failed",
                    tasks=skipped,
                    smooth=smooth_result,
                )
                return run._make_result(status)

            run.job.write_run_state(
                status="completed",
                tasks=skipped,
                smooth=smooth_result,
            )
            status = JobStatus(
                state="completed",
                return_code=0,
                job_id=run.id,
                message="Image smoothing task completed",
                raw={"smooth": smooth_result},
            )
            return run._make_result(status)
        except Exception as exc:
            self._cancel_futures(all_futures)
            try:
                run.job.write_run_state(
                    status="failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            except Exception:
                logger.debug("Could not write failed smooth run state", exc_info=True)
            status = JobStatus(
                state="failed",
                return_code=1,
                job_id=run.id,
                message=str(exc),
                raw={"error_type": type(exc).__name__},
            )
            return run._make_result(status)
        finally:
            self._release_futures(all_futures)
            if run.backend.get("shutdown_on_completion", self.shutdown_on_completion):
                self.close(wait=True, retire=True)

    def _finalize_local_run(self, run: RunHandle, status: JobStatus) -> RunResult:
        return self._wait_local_run(run, timeout=None, poll_interval=0)

    def _timeout_local_run(self, run: RunHandle, status: JobStatus) -> RunResult:
        futures = run.backend.get("futures", [])
        self._cancel_futures(futures)
        try:
            run.job.write_run_state(status="timeout", error=status.message)
        except Exception:
            logger.debug("Could not write timed-out run state", exc_info=True)
        self._release_futures(futures)
        if run.backend.get("shutdown_on_completion", self.shutdown_on_completion):
            self.close(wait=False, retire=False)
        return run._make_result(status)

    def _cancel_local_run(self, run: RunHandle) -> None:
        self._cancel_futures(run.backend.get("futures", []))
        self._release_futures(run.backend.get("futures", []))
        if run.backend.get("shutdown_on_completion", self.shutdown_on_completion):
            self.close(wait=False, retire=False)

    def _submit_local_tasks(
        self,
        job: BaseJob,
        force_run: bool = False,
        *,
        skip_policy: Optional[Any] = None,
        reuse: bool = True,
        residual: Optional[float] = None,
        ignore_solver_options: Optional[bool] = None,
        **kwargs,
    ) -> LocalTaskSubmission:
        """Submit job tasks to the local Dask executor.

        Args:
            job: The simulation job to run
            **kwargs: Additional arguments for task configuration

        Returns:
            LocalTaskSubmission containing Dask futures and the task plan.
        """
        if not self.executable:
            raise RuntimeError("Solver executable not found, cannot submit job")

        job_file = job.save()
        plan_kwargs = {}
        if skip_policy is not None:
            plan_kwargs["skip_policy"] = skip_policy
        if residual is not None:
            plan_kwargs["residual"] = residual
        if ignore_solver_options is not None:
            plan_kwargs["ignore_solver_options"] = ignore_solver_options
        task_plan = job.task_run_plan(
            reuse=bool(reuse) and not force_run,
            force=force_run,
            **plan_kwargs,
        )
        pending_indices = list(task_plan["pending_indices"])
        if not pending_indices:
            return LocalTaskSubmission(futures=[], task_plan=task_plan)

        self._ensure_dask_for_tasks(1)

        n_ranks = kwargs.get("procs_per_job", 1)

        stdout_dir = str(job._stdout_path)
        os.makedirs(stdout_dir, exist_ok=True)
        for log_name in [_task_log_name(MESH_TASK_ID)]:
            try:
                os.remove(os.path.join(stdout_dir, log_name))
            except FileNotFoundError:
                pass
        for index in pending_indices:
            try:
                os.remove(os.path.join(stdout_dir, f"task_{index + 1}.log"))
            except FileNotFoundError:
                pass

        futures = []

        # Mesh and size first
        init_threads = self._current_threads_per_worker()
        future = self._dask_client.submit(
            run_task,
            job_file,
            MESH_TASK_ID,
            self.executable,
            self.env,
            n_ranks=1,
            n_threads=init_threads,
            stdout_dir=stdout_dir,
            fresh=force_run,
            resources={"CPU": init_threads},
        )
        try:
            mesh_result = future.result()
            if not _task_result_successful(mesh_result):
                log_path = mesh_result.get("stdout")
                message = f"Mesh task failed: {mesh_result.get('error')}"
                if log_path:
                    message = f"{message}. Log: {log_path}"
                try:
                    job.write_run_state(
                        status="failed",
                        mesh=mesh_result,
                        errors=[mesh_result],
                    )
                except Exception:
                    logger.debug("Could not write failed mesh run state", exc_info=True)
                raise RuntimeError(message)
        finally:
            self._release_futures([future])

        self._ensure_dask_for_tasks(len(pending_indices))

        # Loop tasks in reverse order for improved load balancing.
        for i in sorted(pending_indices, reverse=True):
            try:
                future = self._dask_client.submit(
                    run_task,
                    job_file,
                    i,
                    self.executable,
                    self.env,
                    n_ranks=n_ranks,
                    n_threads=self._current_threads_per_worker(),
                    stdout_dir=stdout_dir,
                    fresh=force_run,
                    retries=0,
                    priority=i,
                    actor=False,
                    pure=True,
                    resources={"CPU": self._current_threads_per_worker()},
                )
                futures.append(future)
            except Exception as e:
                logger.error("Failed to submit task %s: %s", i, str(e))
                raise

        self._futures.extend(futures)
        return LocalTaskSubmission(futures=futures, task_plan=task_plan)

    def fetch_traces(
        self,
        job: Union[BaseJob, List[BaseJob]],
        upscale: int = 1,
        path: Optional[Union[str, Path]] = None,
        combine: bool = False,
    ) -> Union[TraceDataset, Dict[str, TraceDataset]]:
        """Open trace outputs from completed local jobs.

        Args:
            job: Single job or list of jobs.
            upscale: Optional time/frequency upscaling factor for trace reads.
            path: Optional project path used to resolve relative artifacts.
            combine: Whether to combine multiple jobs into one dataset.

        Returns:
            ``TraceDataset`` for a single job or combined reads, otherwise a
            mapping keyed by job name.
        """

        project_path = Path(path).resolve() if path is not None else None
        if isinstance(job, BaseJob):
            db = TraceDataset.from_job(job, upscale, project_path=project_path)
            return db
        if combine:
            return TraceDataset.from_jobs(job, upscale, project_path=project_path)
        else:
            db_map = {}
            for j in job:
                db = TraceDataset.from_job(j, upscale, project_path=project_path)
                db_map[j.name] = db
            return db_map

    def fetch_wavefields(
        self,
        job: Union[BaseJob, List[BaseJob]],
        upscale: int = 1,
        path: Optional[Union[str, Path]] = None,
    ) -> Union[TraceDataset, Dict[str, TraceDataset]]:
        """Open wavefield outputs from completed local jobs.

        Args:
            job: Single job or list of jobs.
            upscale: Optional upscaling factor for wavefield trace reads.
            path: Optional project path used to resolve relative artifacts.

        Returns:
            Wavefield dataset for a single job or a mapping keyed by job name.
        """

        project_path = Path(path).resolve() if path is not None else None
        if isinstance(job, BaseJob):
            return job.wavefields.open(upscale=upscale, project_path=project_path)
        db_map = {}
        for j in job:
            db_map[j.name] = j.wavefields.open(
                upscale=upscale,
                project_path=project_path,
            )
        return db_map

    def fetch_image(
        self,
        job: Union[ImagingJob, List[ImagingJob]],
    ) -> ArrayLike:
        """Open and accumulate imaging outputs.

        Args:
            job: Imaging job or list of imaging jobs.

        Returns:
            ``ImageDatabase`` for a single job, or a mapping keyed by job name.
        """

        if isinstance(job, ImagingJob):
            jobs = [job]
        else:
            jobs = job
        images = {}
        for job in jobs:
            local = job._local_image_path
            image = ImageDatabase(
                path=local,
                shape=job.grid.shape,
                parts=job.n_tasks,
            )
            image.require_aggregate()
            images[job.name] = image

        if len(images) == 1:
            return images[jobs[0].name]
        else:
            return images

    def fetch_paraview(self, job: BaseJob, path: Optional[Union[str, Path]] = None):
        """Return local ParaView output paths for a job.

        Args:
            job: Completed job.
            path: Accepted for API compatibility; local paths are already
                resolved by the job artifact handle.

        Returns:
            Mapping/list of ParaView output paths recorded by the job.
        """
        return job.paraview_outputs

    @property
    def provisioned(self) -> bool:
        """Local execution is always immediately available."""
        return True

    def sync(self, project):
        """Return ``project`` because local execution needs no synchronization.

        Args:
            project: Project object.

        Returns:
            The same project object.
        """
        return project

    def _sync_project(self, project):
        """Local execution does not require explicit synchronization."""
        return project

    def get(
        self,
        remote_path: Union[str, Path],
        local_path: Union[str, Path],
        overwrite: bool = False,
    ):
        """Copy a local file or directory.

        Args:
            remote_path: Source path in the local-site namespace.
            local_path: Destination path.
            overwrite: Whether to replace an existing destination.

        Returns:
            Destination path.

        Raises:
            FileNotFoundError: If ``remote_path`` does not exist.
        """
        remote_path = Path(remote_path)
        local_path = Path(local_path)
        if not remote_path.exists():
            raise FileNotFoundError(remote_path)
        if local_path.exists() and not overwrite:
            return local_path
        if remote_path.is_dir():
            if local_path.exists():
                shutil.rmtree(local_path)
            shutil.copytree(remote_path, local_path)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(remote_path, local_path)
        return local_path

    def put(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        """Copy a file or directory into the local-site namespace.

        Args:
            local_path: Source path.
            remote_path: Destination path in the local-site namespace.

        Returns:
            Destination path.
        """
        return self.get(local_path, remote_path, overwrite=True)

    @property
    def dashboard_url(self) -> Optional[str]:
        """Get the URL for the Dask dashboard.

        Returns:
            URL string if dashboard is available, None otherwise
        """
        return self._dashboard_url

    def cancel_job(self, job_id: str):
        """Cancel a running job.

        Args:
            job_id: The ID of the job to cancel
        """
        try:
            os.kill(int(job_id), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except ValueError:
            raise ValueError(f"Invalid process ID: {job_id}")

    def _initialize_dask(self, n_workers: Optional[int] = None):
        """Initialize Dask client and cluster."""

        if self._dask_client is not None:
            return
        self._quiet_dependency_loggers()

        n_workers, threads_per_worker, memory_per_worker = self._cluster_settings(
            n_workers
        )

        total_threads = n_workers * threads_per_worker
        total_memory = n_workers * memory_per_worker
        if total_threads > self.config.cores:
            raise ValueError(
                f"Total threads ({total_threads}) exceed available cores ({self.config.cores})"
            )
        if self.config.memory:
            if total_memory > self.config.memory:
                raise ValueError(
                    f"Total memory ({total_memory}MB) exceed available memory ({self.config.memory}MB)"
                )

        logger.info(
            "Initializing Dask with %d workers, %d threads per worker, %dMB memory per worker",
            n_workers,
            threads_per_worker,
            memory_per_worker,
        )

        try:
            self._active_n_workers = n_workers
            self._active_threads_per_worker = threads_per_worker
            self._active_memory_per_worker = memory_per_worker
            dependency_log_level = self._dependency_log_level()
            dependency_log_name = logging.getLevelName(dependency_log_level)
            worker_preload = self._with_dask_logging_preload(
                dask_config.get("distributed.worker.preload", default=[])
            )
            nanny_preload = self._with_dask_logging_preload(
                dask_config.get("distributed.nanny.preload", default=[])
            )
            scheduler_preload = self._with_dask_logging_preload(
                dask_config.get("distributed.scheduler.preload", default=[])
            )
            dask_config.update(
                dask_config.config,
                {
                    "distributed": {
                        "worker": {
                            "memory": {"target": 0.6, "pause": 0.8},
                            "threads": threads_per_worker,
                            "preload": worker_preload,
                        },
                        "scheduler": {
                            "work-stealing": True,
                            "work-stealing-interval": "1s",
                            "bandwidth": 1,
                            "preload": scheduler_preload,
                            "dashboard": {
                                "bokeh-application": {"allow_websocket_origin": []}
                            },
                        },
                        "nanny": {"preload": nanny_preload},
                        "comm": {"timeouts": {"connect": "10s", "tcp": "30s"}},
                        "logging": {
                            "distributed": dependency_log_name,
                            "distributed.client": dependency_log_name,
                            "distributed.core": dependency_log_name,
                            "distributed.nanny": dependency_log_name,
                            "distributed.scheduler": dependency_log_name,
                            "distributed.worker": dependency_log_name,
                            "tornado": "CRITICAL",
                            "tornado.application": "ERROR",
                            "bokeh": "ERROR",
                        },
                    },
                },
                priority="new",
            )
            self._quiet_dependency_loggers()

            dashboard_address = f"{self.dashboard_host}:{self.dashboard_port}"
            self._dask_cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=f"{memory_per_worker}MB",
                dashboard_address=dashboard_address,
                host=self.dashboard_host,
                local_directory="/tmp/dask-worker-space",
                scheduler_port=0,
                silence_logs=dependency_log_level,
                preload=worker_preload,
                preload_nanny=nanny_preload,
                scheduler_kwargs={"preload": scheduler_preload},
                processes=True,
                resources={"CPU": threads_per_worker},
            )
            self._dask_client = Client(self._dask_cluster, timeout="20s")
            self._closed = False
            self._quiet_dependency_loggers()

            try:
                dashboard_url = str(self._dask_cluster.dashboard_link)
                parsed = urlparse(dashboard_url)
                if parsed.port and self.dashboard_host in {"localhost", "127.0.0.1"}:
                    parsed = parsed._replace(
                        netloc=f"{self.dashboard_host}:{parsed.port}"
                    )
                    dashboard_url = parsed.geturl()
                self._dashboard_url = dashboard_url
                self._dashboard_port = str(parsed.port) if parsed.port else None
                logger.info("Dask dashboard available at %s", self.dashboard_url)
            except Exception:
                self._dashboard_port = None
                self._dashboard_url = None

        except Exception as e:
            logger.error("Failed to initialize Dask cluster: %s", str(e))
            self.close(wait=False, retire=False)
            raise

    def _ensure_dask_for_tasks(self, task_count: int) -> None:
        n_workers = self._worker_count_for_task_count(task_count)
        if self._dask_client is not None:
            self._prune_futures()
            if self._active_cluster_matches(n_workers):
                return
            if self._futures:
                logger.debug(
                    "Keeping active Dask cluster while %d futures are still active",
                    len(self._futures),
                )
                return
            logger.info("Refreshing Dask cluster for %d pending tasks", task_count)
            self.close(wait=True, retire=True)
        self._initialize_dask(n_workers)

    def _worker_count_for_task_count(self, task_count: int) -> int:
        if self.n_workers is not None:
            return max(1, int(self.n_workers))

        n_workers = self.config.cores
        n_workers = max(1, int(n_workers))
        if task_count < n_workers:
            while n_workers > task_count and n_workers > 1:
                n_workers = n_workers // 2
        return max(1, n_workers)

    def _cluster_settings(
        self, n_workers: Optional[int] = None
    ) -> tuple[int, int, int]:
        if n_workers is None:
            n_workers = (
                self.n_workers if self.n_workers is not None else self.config.cores
            )
        n_workers = max(1, int(n_workers))
        if self.threads_per_worker is None:
            threads_per_worker = max(1, self.config.cores // n_workers)
        else:
            threads_per_worker = int(self.threads_per_worker)
        if self.memory_per_worker is None:
            if self.config.memory:
                memory_per_worker = int((0.9 * self.config.memory) / n_workers)
            else:
                memory_per_worker = 4096
        else:
            memory_per_worker = int(self.memory_per_worker)
        return n_workers, threads_per_worker, memory_per_worker

    def _active_cluster_matches(self, n_workers: int) -> bool:
        expected = self._cluster_settings(n_workers)
        active = (
            self._active_n_workers,
            self._active_threads_per_worker,
            self._active_memory_per_worker,
        )
        return active == expected

    def _current_threads_per_worker(self) -> int:
        if self._active_threads_per_worker is not None:
            return self._active_threads_per_worker
        if self.threads_per_worker is not None:
            return int(self.threads_per_worker)
        return 1

    @staticmethod
    def _with_dask_logging_preload(value) -> List[str]:
        if value is None:
            modules = []
        elif isinstance(value, str):
            modules = [value]
        else:
            modules = list(value)
        if DASK_LOGGING_PRELOAD not in modules:
            modules.append(DASK_LOGGING_PRELOAD)
        return modules

    @staticmethod
    def _dependency_log_level() -> int:
        for name in ("distributed", "dask", "tornado", "bokeh"):
            level = getattr(
                logging.getLogger(name), "_frequensolve_dependency_level", None
            )
            if level is not None:
                return int(level)
        return ERROR

    @classmethod
    def _quiet_dependency_loggers(cls) -> None:
        level = cls._dependency_log_level()
        configure_dependency_logging(level)

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.close(wait=False, retire=False, timeout=2.0)
        except Exception:
            pass

    def __enter__(self) -> "LocalSite":
        """Enter a context manager without changing site state."""

        return self

    def __exit__(self, exc_type, exc, tb):
        """Close the local cluster when leaving a context manager."""

        self.close()
        return False

    def close(self, *, wait: bool = True, retire: bool = True, timeout: float = 30.0):
        """Close Dask client/cluster resources owned by this site.

        Args:
            wait: Whether to wait for orderly cluster shutdown.
            retire: Accepted for compatibility with older shutdown semantics.
            timeout: Maximum Dask close timeout in seconds.
        """

        if self._closed and self._dask_client is None and self._dask_cluster is None:
            return
        self._closed = True

        # Cancel outstanding futures
        if self._dask_client is not None and self._futures:
            try:
                self._dask_client.cancel(self._futures, force=True)
            except Exception:
                pass
            self._futures.clear()

        # LocalCluster.close() already owns orderly worker/scheduler teardown.
        # Explicit retire/remove or scale-to-zero can race with worker heartbeat
        # callbacks and emit noisy "unregistered worker" shutdown warnings.
        try:
            if self._dask_client is not None:
                self._dask_client.close(timeout=timeout)
        except Exception:
            pass
        finally:
            self._dask_client = None

        try:
            if self._dask_cluster is not None:
                self._dask_cluster.close(timeout=timeout, fast=not wait)
        except Exception:
            pass
        finally:
            self._dask_cluster = None
            self._dashboard_port = None
            self._dashboard_url = None
            self._active_n_workers = None
            self._active_threads_per_worker = None
            self._active_memory_per_worker = None

    def stop(self):
        """Alias for ``close()`` used by interactive workflows."""

        self.close()

    def _cancel_futures(self, futures: Iterable[Future]) -> None:
        futures = list(futures)
        if not futures:
            return
        if self._dask_client is not None:
            try:
                self._dask_client.cancel(futures, force=True)
                return
            except Exception:
                logger.debug("Dask future cancellation failed", exc_info=True)
        for future in futures:
            try:
                future.cancel()
            except Exception:
                logger.debug("Future cancellation failed", exc_info=True)

    def _release_futures(self, futures: Iterable[Future]) -> None:
        future_set = set(futures)
        for future in future_set:
            try:
                future.release()
            except Exception:
                logger.debug("Future release failed", exc_info=True)
        if future_set:
            self._futures = [
                future for future in self._futures if future not in future_set
            ]
        self._prune_futures()

    def _prune_futures(self) -> None:
        terminal = {"finished", "error", "cancelled", "lost"}
        active = []
        for future in self._futures:
            if getattr(future, "status", None) in terminal:
                try:
                    future.release()
                except Exception:
                    logger.debug("Future release failed", exc_info=True)
            else:
                active.append(future)
        self._futures = active

    def _get_solver_path(self) -> str:
        """Get the solver path."""
        executable = self.solver
        if not executable:
            warnings.warn(
                "Solver executable not configured; set solver in site.toml "
                "or pass solver= explicitly",
                stacklevel=2,
            )
            return None
        executable = str(Path(executable).expanduser())
        if not Path(executable).exists():
            warnings.warn(f"Solver executable not found at {executable}", stacklevel=2)
            return None
        return executable
