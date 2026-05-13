from __future__ import annotations

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
from typing import Dict, Iterable, List, Optional, Union
from urllib.parse import urlparse

from frequensolve._optional import optional_dependency_error

try:
    from dask import config as dask_config
    from dask.distributed import Client, Future, LocalCluster, wait
    from dotenv import load_dotenv
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "LocalSite",
        extra="parallel",
        dependencies=("dask", "distributed", "python-dotenv"),
        error=exc,
    ) from exc

from numpy.typing import ArrayLike

from frequensolve.orchestrator.config.local import LocalSiteConfig
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    JobStatus,
    RunHandle,
    RunResult,
    _wait_for_path,
)
from frequensolve.orchestrator.sites.dask_logging import configure_dependency_logging
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.imaging import ImageDatabase, ImagingJob
from frequensolve.simulation.jobs import SimulationJob

logger = logging.getLogger(__name__)

__all__ = ["LocalSite"]

PACK_TASK_ID = -3
SMOOTH_TASK_ID = -2
MESH_TASK_ID = -1
DASK_LOGGING_PRELOAD = "frequensolve.orchestrator.sites.dask_logging"

# TODO: print this when running job
# print("Dask dashboard:", site.dashboard_url or "not available")


def _task_log_name(task_id: int) -> str:
    if task_id == PACK_TASK_ID:
        return "pack.log"
    if task_id == SMOOTH_TASK_ID:
        return "smooth.log"
    if task_id == MESH_TASK_ID:
        return "mesh.log"
    return f"task_{task_id + 1}.log"


def run_task(
    job_file: str,
    task_id: int,
    executable: str,
    env: dict,
    n_ranks: int = 1,
    n_threads: int = 1,
    stdout_dir: str = None,
) -> dict:
    """Run a single task and return its results.

    Args:
        job_file: Path to the job file
        task_id: Task ID to run
        executable: Path to the solver executable
        env: Environment variables
        n_ranks: Number of MPI ranks
        n_threads: Number of threads per rank
        output_dir: Directory to store completed output files
        active_dir: Directory to store active output files

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
        "-j",
        f"{job_file}",
    ]
    if task_id == PACK_TASK_ID:
        args += ["--pack"]
    elif task_id == SMOOTH_TASK_ID:
        args += ["--smooth"]
    elif task_id == MESH_TASK_ID:
        args += ["--mesh"]
    else:
        args += ["-i", f"{task_id + 1}"]
    command = shlex.join(args)
    logger.info("Executing: %s", command)

    if stdout_dir:
        os.makedirs(stdout_dir, exist_ok=True)
        stdout_file = os.path.join(stdout_dir, _task_log_name(task_id))
    else:
        stdout_file = None
    started = time.perf_counter()
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

        return {
            "task_id": task_id,
            "status": "success",
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
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "error",
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


@dataclass
class LocalTaskSubmission:
    futures: List["Future"]
    task_plan: Dict[str, object]


@dataclass(kw_only=True)
class LocalSite(BaseSite):
    """Site for local execution."""

    config: LocalSiteConfig = field(init=False)
    executable: str = field(init=False)
    env: dict = field(default_factory=dict)
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
    _closed: bool = field(default=True, init=False)

    # ----------------- lifecycle -----------------

    def __post_init__(self):
        self.config = LocalSiteConfig()
        self.executable = self._get_solver_path()
        self.env = os.environ.copy()
        self.env["FS_SOLVER_PATH"] = os.getenv("FS_SOLVER_PATH")
        self._quiet_dependency_loggers()

    def submit(self, job: SimulationJob, **kwargs) -> RunHandle:
        """Submit job and return an awaitable run handle.

        Args:
            job: The simulation job to run
            **kwargs: Additional arguments for task configuration

        Returns:
            RunHandle for the submitted tasks
        """
        force = bool(kwargs.pop("force", False) or kwargs.pop("rerun", False))
        pack = bool(kwargs.pop("pack", True))
        shutdown_on_completion = bool(
            kwargs.pop("shutdown_on_completion", self.shutdown_on_completion)
        )
        self.prepare_job(job)
        if not force and job.is_run_current():
            logger.info(
                "Skipping job %s; fingerprint matches and expected trace outputs exist.",
                job.name,
            )
            job.write_run_state(status="skipped")
            return RunHandle.skipped(self, job)

        try:
            submission = self._submit_local_tasks(job, **kwargs)
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
        if not futures and task_plan:
            reused = task_plan.get("reused_tasks", [])
            job.write_run_state(status="skipped", tasks=reused)
            return RunHandle.skipped(self, job, "No frequency tasks need to run")
        handle = RunHandle(
            site=self,
            job=job,
            id=f"local:{job.name}",
            mode="local",
            poll_interval=0.5,
            _status_fn=self._poll_local_run,
            _wait_fn=self._wait_local_run,
            _cancel_fn=self._cancel_local_run,
        )
        handle.backend["futures"] = futures
        handle.backend["task_plan"] = task_plan
        handle.backend["pack_after_tasks"] = pack
        handle.backend["shutdown_on_completion"] = shutdown_on_completion
        return handle

    def _poll_local_run(self, run: RunHandle) -> JobStatus:
        futures = run.backend.get("futures", [])
        if not futures:
            return JobStatus(state="skipped", return_code=0, job_id=run.id)
        statuses = [getattr(future, "status", "unknown") for future in futures]
        if any(status == "error" for status in statuses):
            return JobStatus(
                state="failed", return_code=1, job_id=run.id, raw={"statuses": statuses}
            )
        if all(status == "finished" for status in statuses):
            return JobStatus(
                state="completed",
                return_code=0,
                job_id=run.id,
                raw={"statuses": statuses},
            )
        return JobStatus(state="running", job_id=run.id, raw={"statuses": statuses})

    def _wait_local_run(
        self,
        run: RunHandle,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        futures = run.backend.get("futures", [])
        if not futures:
            reused = run.backend.get("task_plan", {}).get("reused_tasks", [])
            if reused:
                run.job.write_run_state(status="skipped", tasks=reused)
            status = JobStatus(state="skipped", return_code=0, job_id=run.id)
            self._emit_status(status)
            return run._make_result(status)

        pbar = None
        smooth_future = None
        pack_future = None
        all_futures = list(futures)

        try:
            self._emit_status(JobStatus(state="running", job_id=run.id))
            if self._is_notebook:
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm

            pbar = tqdm(
                total=len(futures),
                desc=f"Running: {run.job.name}",
                bar_format=(
                    "{desc} {n_fmt}/{total_fmt} |{bar}| Elapsed time: {elapsed}s"
                ),
                colour="#4ec9b0",
            )

            def update_progress(future):
                pbar.update(1)

            for future in futures:
                future.add_done_callback(update_progress)

            wait_result = wait(futures, timeout=timeout)
            not_done = list(getattr(wait_result, "not_done", []))
            if not_done:
                self._cancel_futures(not_done)
                status = JobStatus(
                    state="timeout",
                    return_code=1,
                    job_id=run.id,
                    message=f"Timed out waiting for local run after {timeout} seconds",
                    raw={"unfinished": len(not_done)},
                )
                run.job.write_run_state(status="timeout")
                self._emit_status(status)
                return run._make_result(status)

            task_results = [future.result() for future in futures]
            reused_tasks = run.backend.get("task_plan", {}).get("reused_tasks", [])
            state_tasks = [*task_results, *reused_tasks]
            if hasattr(run.job, "invalidate_trace_cache"):
                run.job.invalidate_trace_cache()
            task_errors = [
                result for result in task_results if result.get("status") != "success"
            ]
            if task_errors:
                status = JobStatus(
                    state="failed",
                    return_code=1,
                    job_id=run.id,
                    message=f"{len(task_errors)} local tasks failed",
                    raw={"tasks": task_results},
                )
                run.job.write_run_state(
                    status="failed", tasks=state_tasks, errors=task_errors
                )
                self._emit_status(status)
                return run._make_result(status)

            if isinstance(run.job, ImagingJob):
                smooth_future = self._dask_client.submit(
                    run_task,
                    run.job._file,
                    SMOOTH_TASK_ID,
                    self.executable,
                    self.env,
                    n_ranks=1,
                    n_threads=self.threads_per_worker,
                    stdout_dir=str(run.job._stdout_path),
                    resources={"CPU": self.threads_per_worker},
                )
                all_futures.append(smooth_future)
                smooth_result = smooth_future.result()
                if smooth_result.get("status") != "success":
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
                    self._emit_status(status)
                    return run._make_result(status)

            pack_result = None
            if run.backend.get("pack_after_tasks", False):
                if self._dask_client is None:
                    raise RuntimeError("Cannot run solver packing task without Dask")
                pack_future = self._dask_client.submit(
                    run_task,
                    run.job._file,
                    PACK_TASK_ID,
                    self.executable,
                    self.env,
                    n_ranks=1,
                    n_threads=self.threads_per_worker,
                    stdout_dir=str(run.job._stdout_path),
                    resources={"CPU": self.threads_per_worker},
                )
                all_futures.append(pack_future)
                pack_result = pack_future.result()
                if pack_result.get("status") != "success":
                    status = JobStatus(
                        state="failed",
                        return_code=1,
                        job_id=run.id,
                        message="Packing task failed",
                        raw={"pack": pack_result, "tasks": task_results},
                    )
                    run.job.write_run_state(
                        status="failed", tasks=state_tasks, pack=pack_result
                    )
                    self._emit_status(status)
                    return run._make_result(status)

            run.job.write_run_state(
                status="completed",
                tasks=state_tasks,
                **({"pack": pack_result} if pack_result is not None else {}),
            )
            status = JobStatus(
                state="completed",
                return_code=0,
                job_id=run.id,
                raw={
                    "tasks": task_results,
                    **({"pack": pack_result} if pack_result is not None else {}),
                },
            )
            self._emit_status(status)
            return run._make_result(status)
        except Exception as exc:
            self._cancel_futures(all_futures)
            try:
                run.job.write_run_state(status="failed", error=str(exc))
            except Exception:
                logger.debug("Could not write failed run state", exc_info=True)
            status = JobStatus(
                state="failed",
                return_code=1,
                job_id=run.id,
                message=str(exc),
            )
            self._emit_status(status)
            return run._make_result(status)
        finally:
            if pbar is not None:
                pbar.close()
            self._release_futures(all_futures)
            if run.backend.get("shutdown_on_completion", self.shutdown_on_completion):
                self.close(wait=True, retire=True)

    def _cancel_local_run(self, run: RunHandle) -> None:
        self._cancel_futures(run.backend.get("futures", []))
        self._release_futures(run.backend.get("futures", []))
        if run.backend.get("shutdown_on_completion", self.shutdown_on_completion):
            self.close(wait=False, retire=False)

    def _submit_local_tasks(self, job: SimulationJob, **kwargs) -> LocalTaskSubmission:
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
        task_plan = job.task_run_plan(reuse=True)
        pending_indices = list(task_plan["pending_indices"])
        if not pending_indices:
            return LocalTaskSubmission(futures=[], task_plan=task_plan)

        if self._dask_client is None:
            if self.n_workers is None:
                self.n_workers = self.config.cores
            n_workers = self.n_workers
            if len(pending_indices) < n_workers:
                while n_workers > len(pending_indices) and n_workers > 1:
                    n_workers = n_workers // 2
            self._initialize_dask(max(1, n_workers))
        else:
            self._prune_futures()

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
        future = self._dask_client.submit(
            run_task,
            job_file,
            MESH_TASK_ID,
            self.executable,
            self.env,
            n_ranks=1,
            n_threads=1,
            stdout_dir=stdout_dir,
            resources={"CPU": 1},
        )
        try:
            mesh_result = future.result()
            if mesh_result.get("status") != "success":
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
                    n_threads=self.threads_per_worker,
                    stdout_dir=stdout_dir,
                    retries=0,
                    priority=i,
                    actor=False,
                    pure=True,
                    resources={"CPU": self.threads_per_worker},
                )
                futures.append(future)
            except Exception as e:
                logger.error("Failed to submit task %s: %s", i, str(e))
                raise

        self._futures.extend(futures)
        return LocalTaskSubmission(futures=futures, task_plan=task_plan)

    def fetch_traces(
        self,
        job: Union[SimulationJob, List[SimulationJob]],
        upscale: int = 1,
        path: Optional[Union[str, Path]] = None,
        combine: bool = False,
    ) -> Union[TraceDataset, Dict[str, TraceDataset]]:
        project_path = Path(path).resolve() if path is not None else None
        if isinstance(job, SimulationJob):
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

    def fetch_image(
        self,
        job: Union[ImagingJob, List[ImagingJob]],
    ) -> ArrayLike:
        """Gets and accumulates images."""

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
            images[job.name] = image

        if len(images) == 1:
            return images[jobs[0].name]
        else:
            return images

    def fetch_paraview(
        self, job: SimulationJob, path: Optional[Union[str, Path]] = None
    ):
        """Return local ParaView output paths for a job."""
        return job.paraview_outputs

    @property
    def provisioned(self) -> bool:
        """Local execution is always immediately available."""
        return True

    def sync(self, project):
        """Local execution does not require explicit synchronization."""
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
        """Copy a local file or directory."""
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
        """Send a file or directory to the site."""
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

        if n_workers is None:
            self.n_workers = self.config.cores
        else:
            self.n_workers = n_workers

        if self.threads_per_worker is None:
            self.threads_per_worker = self.config.cores // self.n_workers
        if self.memory_per_worker is None:
            if self.config.memory:
                self.memory_per_worker = int(
                    (0.9 * self.config.memory) / self.n_workers
                )
            else:
                self.memory_per_worker = 4096

        total_threads = self.n_workers * self.threads_per_worker
        total_memory = self.n_workers * self.memory_per_worker
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
            self.n_workers,
            self.threads_per_worker,
            self.memory_per_worker,
        )

        try:
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
                            "threads": self.threads_per_worker,
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
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=f"{self.memory_per_worker}MB",
                dashboard_address=dashboard_address,
                host=self.dashboard_host,
                local_directory="/tmp/dask-worker-space",
                scheduler_port=0,
                silence_logs=dependency_log_level,
                preload=worker_preload,
                preload_nanny=nanny_preload,
                scheduler_kwargs={"preload": scheduler_preload},
                processes=True,
                resources={"CPU": self.threads_per_worker},
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
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self, *, wait: bool = True, retire: bool = True, timeout: float = 30.0):
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

        # Politely ask workers to go away before closing cluster
        if retire and self._dask_client is not None:
            try:
                # Migrate data off workers and then close them
                self._dask_client.retire_workers(
                    workers=None, close_workers=True, remove=True
                )
            except Exception:
                # Fall back to scaling to zero
                try:
                    if self._dask_cluster is not None:
                        self._dask_cluster.scale(0)
                except Exception:
                    pass

        # Close client then cluster
        try:
            if self._dask_client is not None:
                self._dask_client.close(timeout=timeout)
        except Exception:
            pass
        finally:
            self._dask_client = None

        try:
            if self._dask_cluster is not None:
                try:
                    self._dask_cluster.scale(0)
                except Exception:
                    pass
                self._dask_cluster.close(timeout=timeout, fast=not wait)
        except Exception:
            pass
        finally:
            self._dask_cluster = None
            self._dashboard_port = None
            self._dashboard_url = None

    def stop(self):
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
        load_dotenv()
        executable = os.getenv("LOCAL_SOLVER_EXECUTABLE")
        if not executable:
            warnings.warn(
                "LOCAL_SOLVER_EXECUTABLE not set in environment",
                stacklevel=2,
            )
            return None
        if not Path(executable).exists():
            warnings.warn(f"Solver executable not found at {executable}", stacklevel=2)
            return None
        return executable
