"""Owned and borrowed persistent SLURM allocation executors."""

from __future__ import annotations

import json
import math
import shlex
import threading
import time
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Optional
from uuid import uuid4

from frequensolve.orchestrator.sites.base import BaseSite, JobStatus, RunHandle
from frequensolve.orchestrator.sites.curvature import (
    FILE_KEYS,
    positive_count,
    read_request,
    single_rank,
)
from frequensolve.orchestrator.sites.hpc.curvature import SlurmCurvatureRunner
from frequensolve.orchestrator.sites.hpc.site import (
    _job_requires_postprocess,
    _safe_slurm_directive,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import temporary_text_file
from frequensolve.simulation.jobs.run_state import SkipPolicy

__all__ = ["AdaptiveWorkers", "AllocationSession"]


@dataclass(frozen=True)
class AdaptiveWorkers:
    """Memory-sized worker rank limits inside a persistent allocation.

    ``max_ranks_per_task=None`` permits a task to use the whole allocation.
    Legacy partitioned artifacts retain their producer's rank count.
    """

    min_ranks: int = 1
    max_ranks_per_task: Optional[int] = None
    mem_cushion: float = 1.5

    def __post_init__(self):
        positive_count(self.min_ranks, "min_ranks")
        if self.max_ranks_per_task is not None:
            positive_count(self.max_ranks_per_task, "max_ranks_per_task")
            if self.max_ranks_per_task < self.min_ranks:
                raise ValueError("max_ranks_per_task must be >= min_ranks")
        if not math.isfinite(self.mem_cushion) or self.mem_cushion < 1:
            raise ValueError("mem_cushion must be finite and >= 1")


class AllocationSession(BaseSite):
    """Site-compatible executor backed by one allocation-resident scheduler.

    Construct with ``site.session(...)`` and enter the context to start and
    await readiness. ``site.attach_session(id)`` borrows an existing session.
    Closing an owned session drains its work and releases the allocation;
    closing a borrowed session only disconnects this client. Remote request
    and result records are retained for inspection and reattachment.
    """

    supports_curvature = True
    supports_background_reuse = True
    shutdown_on_completion = False

    def __init__(
        self,
        site,
        *,
        workers=None,
        lease_timeout=300.0,
        startup_timeout=1800.0,
        cleanup_timeout=30.0,
        session_id=None,
        **resources,
    ):
        self.base_site = site
        self.session_id = self._id(session_id or uuid4().hex)
        self.root = site.work_dir / ".fs_sessions" / self.session_id
        self.owned = session_id is None
        self.workers = workers or AdaptiveWorkers()
        if not isinstance(self.workers, AdaptiveWorkers):
            raise TypeError("workers must be an AdaptiveWorkers configuration")
        self.lease_timeout = self._seconds(lease_timeout, "lease_timeout")
        self.startup_timeout = self._seconds(startup_timeout, "startup_timeout")
        self.cleanup_timeout = self._seconds(cleanup_timeout, "cleanup_timeout")
        self.resources = resources
        self.generation = None
        self.allocation_id = None
        self._config = None
        self._handles = {}
        self._jobs = {}
        self._closed = False
        self._ready = False
        self._client_id = uuid4().hex
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._heartbeat = None
        self._slurm_checked_at = 0.0
        self._slurm_state = "unknown"
        self._controller_state = None
        self.verbose = getattr(site, "verbose", False)

    def __getattr__(self, name):
        # Transfers, fetches, cache cleanup and configuration keep the base
        # site's paths and credentials. Execution methods are overridden here.
        return getattr(self.base_site, name)

    @staticmethod
    def _id(value):
        value = str(value)
        if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(
                "session/request id must be 32 lowercase hexadecimal characters"
            )
        return value

    @staticmethod
    def _seconds(value, name):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")
        return value

    def _login(self, command):
        return self.base_site._run_login_checked(
            command, purpose="operate allocation session"
        )

    def _read(self, path):
        with self._lock:
            return self.base_site._read_remote_json(path)

    def _text(self, text, destination):
        with temporary_text_file(text, suffix=".json", prefix="fs-session-") as local:
            self.base_site.put(local, destination)

    def _publish(self, path, payload, *, immutable=False):
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            temporary = path.parent / f".{uuid4().hex}.partial"
            self._text(text, temporary)
            if immutable:
                # Hard-link publication is atomic and never replaces a request
                # already acknowledged after an uncertain transport outcome.
                self._login(
                    f"ln {shlex.quote(str(temporary))} {shlex.quote(str(path))} 2>/dev/null || test -f {shlex.quote(str(path))}; rm -f -- {shlex.quote(str(temporary))}"
                )
                if self._read(path) != payload:
                    raise ValueError("request id already belongs to different work")
            else:
                self._login(
                    f"mv -f -- {shlex.quote(str(temporary))} {shlex.quote(str(path))}"
                )

    def _renew(self):
        self._publish(
            self.root / "leases" / f"{self._client_id}.json",
            {
                "generation": self.generation,
                "expires_at": time.time() + self.lease_timeout,
            },
        )

    def _keep_alive(self):
        while not self._stop.wait(min(10.0, self.lease_timeout / 3)):
            try:
                self._renew()
            except Exception:
                # The durable lease tolerates transient transport errors. Polling
                # detects a dead controller; do not hide it behind another job.
                continue

    def _start_heartbeat(self):
        self._renew()
        self._heartbeat = threading.Thread(
            target=self._keep_alive,
            name=f"fs-session-{self.session_id[:8]}",
            daemon=True,
        )
        self._heartbeat.start()

    def start(self):
        """Submit once (or attach), renew the client lease, return this session."""
        with self._lock:
            if self._closed:
                raise RuntimeError("allocation session is closed")
            if self.allocation_id is not None:
                return self
            if not self.owned:
                self._config = self._read(self.root / "config.json")
                metadata = self._read(self.root / "allocation.json")
                if not self._config or not metadata:
                    raise RuntimeError("allocation session was not found")
                self.generation = self._config["generation"]
                self.allocation_id = str(metadata["allocation_id"])
                self.lease_timeout = float(self._config["lease_timeout"])
                scheduler = self._config["scheduler"]
                self.workers = AdaptiveWorkers(
                    min_ranks=scheduler["min_ranks"],
                    max_ranks_per_task=scheduler["max_ranks_per_task"],
                    mem_cushion=scheduler["mem_cushion"],
                )
                self._start_heartbeat()
                return self
            site = self.base_site
            config, extra = site.run_config.resolved(site.config, **self.resources)
            if extra:
                raise TypeError(
                    "Unexpected allocation resource option(s): "
                    + ", ".join(sorted(extra))
                )
            if config.mpi_async_progress or config.mpi_health_check_timeout:
                raise NotImplementedError(
                    "persistent sessions do not support MPI async progress or health-check options"
                )
            nodes = positive_count(config.nodes, "nodes")
            per_node = positive_count(config.ranks_per_node or 8, "ranks_per_node")
            shape = site.config_for_partition(str(config.queue))
            if per_node > shape.cores_per_node:
                raise ValueError("ranks_per_node exceeds available CPU cores")
            if shape.memory_per_node <= 0:
                raise ValueError(
                    "persistent sessions require memory_per_node in the site profile"
                )
            duration = shape.validate_request(
                nodes, nodes * per_node, config.duration or shape.max_duration
            )
            limit = self.workers.max_ranks_per_task or nodes * per_node
            if self.workers.min_ranks > limit or limit > nodes * per_node:
                raise ValueError("worker rank limits exceed the allocation")
            if site.enterprise_hpc is not None:
                site._enterprise_hpc_preflight(run_config=config)
            else:
                site.check_solver_compatibility()
            self.generation = uuid4().hex
            threads = shape.cores_per_node // per_node
            self._config = {
                "schema": "fs-allocation-session-1",
                "generation": self.generation,
                "nodes": nodes,
                "ranks_per_node": per_node,
                "total_ranks": nodes * per_node,
                "queue": config.queue,
                "duration": duration,
                "account": config.account or site.config.account,
                "lease_timeout": self.lease_timeout,
                "rank_binding": "explicit",
                "scheduler": {
                    "executable": str(site.executable),
                    "mpi": str(site.mpi_cmd),
                    "mpi_args": list(site.mpi_args),
                    "total_ranks": nodes * per_node,
                    "omp_threads": threads,
                    "mem_per_rank_gib": shape.memory_per_node / per_node / 1024.0,
                    "min_ranks": self.workers.min_ranks,
                    "max_ranks_per_task": limit,
                    "mem_cushion": self.workers.mem_cushion,
                },
            }
            # A unique directory prevents accidental reuse after controller loss.
            self._login(
                f"mkdir -p -- {shlex.quote(str(self.root.parent))} && mkdir -- {shlex.quote(str(self.root))} && mkdir -- "
                + " ".join(
                    shlex.quote(str(self.root / name))
                    for name in ("requests", "commands", "runs", "leases")
                )
            )
            self._text(json.dumps(self._config), self.root / "config.json")
            directory = files("frequensolve.orchestrator.sites.hpc").joinpath(
                "templates", "sweep"
            )
            for name in ("adaptive_scheduler.py", "session_controller.py"):
                with as_file(directory.joinpath(name)) as source:
                    site.put(source, self.root / name)
            script = site._render_template(
                "session/session_SLURM.sh",
                name=_safe_slurm_directive("FS_session", "name"),
                queue=_safe_slurm_directive(config.queue, "queue"),
                account=_safe_slurm_directive(
                    config.account or site.config.account, "account"
                ),
                duration=_safe_slurm_directive(duration, "duration"),
                nodes=nodes,
                ranks=nodes * per_node,
                ranks_per_node=per_node,
                cpus_per_task=(
                    None if Path(str(site.mpi_cmd)).name == "ibrun" else threads
                ),
                root=shlex.quote(str(self.root)),
                log_path=_safe_slurm_directive(
                    str(self.root / "controller.log"), "log_path"
                ),
                runtime_setup=site._runtime_setup_lines(),
            )
            self._text(script, self.root / "allocation.sh")
            self._renew()
            self.allocation_id = site._submit_sbatch(
                "sbatch "
                + " ".join(shlex.quote(str(arg)) for arg in config.slurm_args)
                + " "
                + shlex.quote(str(self.root / "allocation.sh"))
            )
            self._text(
                json.dumps(
                    {"allocation_id": self.allocation_id, "generation": self.generation}
                ),
                self.root / "allocation.json",
            )
            self._start_heartbeat()
            site._emit(
                f"Persistent allocation {self.allocation_id}; session {self.session_id}"
            )
            return self

    def wait_ready(self, timeout=None):
        """Wait for controller readiness, not merely SLURM RUNNING status."""
        self.start()
        deadline = time.monotonic() + (
            self.startup_timeout
            if timeout is None
            else self._seconds(timeout, "timeout")
        )
        while time.monotonic() < deadline:
            status = self.status()
            if status.get("state") == "ready":
                self._ready = True
                return self
            if status.get("state") in {
                "failed",
                "closed",
                "cancelled",
                "timeout",
                "complete",
            }:
                raise RuntimeError(f"allocation session did not become ready: {status}")
            time.sleep(0.2)
        raise TimeoutError("timed out waiting for persistent allocation readiness")

    def __enter__(self):
        try:
            return self.wait_ready()
        except BaseException:
            self.close(drain=False)
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close(drain=exc_type is None)
        except Exception:
            if exc_type is None:
                raise
        return False

    def status(self):
        """Return allocation and controller state, including its heartbeat."""
        if self.allocation_id is None:
            return {"state": "new"}
        payload = self._read(self.root / "status.json")
        if payload and payload.get("generation") != self.generation:
            raise RuntimeError("allocation generation changed")
        if payload:
            self._controller_state = payload
        else:
            # SSH/shared-filesystem reads can fail transiently between actions.
            # Keep the last heartbeat only for its existing freshness window;
            # Slurm's terminal state and heartbeat expiry still detect loss.
            payload = self._controller_state
        if payload and payload.get("state") in {"closed", "failed"}:
            return payload
        if time.monotonic() - self._slurm_checked_at > 5:
            self._slurm_state = self.base_site.update_status(self.allocation_id)
            self._slurm_checked_at = time.monotonic()
        if self._slurm_state in {"complete", "failed", "cancelled", "timeout"}:
            return {"state": self._slurm_state}
        if payload and time.time() - float(payload["updated_at"]) < 60:
            return payload
        state = self._slurm_state
        if state == "running":
            if payload:
                return {
                    "state": "failed",
                    "reason": "allocation controller heartbeat expired",
                }
            return {"state": "starting"}
        return {"state": state}

    def _require_ready(self):
        if self._closed or not self._ready:
            raise RuntimeError(
                "enter the allocation session or call wait_ready() before submitting"
            )
        status = self.status()
        if status.get("state") != "ready":
            raise RuntimeError(f"allocation session is unavailable: {status}")

    def _command(self, kind, **payload):
        self._publish(
            self.root / "commands" / f"{uuid4().hex}.json",
            {"generation": self.generation, "kind": kind, **payload},
            immutable=True,
        )

    def _submit_request(
        self, request, job, *, request_id=None, fetch=False, check=False, task_plan=None
    ):
        run_id = self._id(request_id or uuid4().hex)
        self._publish(
            self.root / "requests" / f"{run_id}.json",
            {
                "schema": "fs-session-request-1",
                "generation": self.generation,
                **request,
            },
            immutable=True,
        )
        handle = RunHandle(
            site=self,
            job=job,
            id=run_id,
            mode="session",
            poll_interval=0.2,
            check=check,
            backend={
                "allocation_id": self.allocation_id,
                "session_id": self.session_id,
                "generation": self.generation,
                "task_plan": task_plan,
            },
            _status_fn=self._poll,
            _cancel_fn=lambda run: self.cancel_job(run.id),
            _finalize_fn=self._finalize,
            _pending_fetch_fn=(
                (lambda run: self.base_site.fetch_outputs(run.job)) if fetch else None
            ),
        )
        self._handles[run_id] = handle
        return handle

    def submit(
        self, job, *, check=False, fetch=False, force=False, request_id=None, **options
    ):
        """Queue work in this allocation; return an independently cancellable run."""
        with self._lock:
            self._require_ready()
            mode = options.pop("mode", "auto")
            if mode not in {"auto", "session", "attached"}:
                raise ValueError("an explicit session cannot submit new batch jobs")
            for key in (
                "nodes",
                "ranks_per_node",
                "queue",
                "account",
                "duration",
                "slurm_args",
            ):
                value = options.pop(key, None)
                if value is not None and value != self._config.get(key):
                    raise ValueError(
                        f"{key} cannot change inside an allocation session"
                    )
            validate = options.pop("validate", True)
            options.pop("solver_policy", None)  # allocation preflight already ran
            postprocess = bool(
                options.pop("postprocess_only", False)
                or getattr(job, "postprocess_only", False)
            )
            fresh = bool(force or options.pop("rerun", False))
            skip = SkipPolicy.from_value(
                options.pop("skip", options.pop("skip_policy", None)),
                residual=options.pop("residual", None),
                ignore_solver_options=options.pop("ignore_solver_options", None),
                reuse=bool(options.pop("reuse", True)) and not fresh,
            )
            fresh = fresh or skip.force
            rank_override = options.pop("ranks_per_task", None)
            pack = bool(options.pop("pack", True)) and bool(
                getattr(job, "supports_trace_packing", True)
            )
            if options:
                raise TypeError(
                    "Unsupported session submission option(s): "
                    + ", ".join(sorted(options))
                )
            site = self.base_site
            site.prepare_job(job, validate=validate)
            if postprocess and not _job_requires_postprocess(job):
                raise ValueError("postprocess_only requires solver postprocessing")
            if not fresh and site.is_run_current(job):
                job.write_run_state(status="skipped")
                handle = RunHandle.skipped(self, job)
                if fetch:
                    site.fetch_outputs(job)
                return handle
            inflight = self._jobs.get(str(getattr(job, "job_file", job.name)))
            if inflight and not inflight.status().is_complete:
                if fresh:
                    raise RuntimeError("this job already has an active session run")
                return inflight
            plan = job.task_run_plan(reuse=skip.reuse, force=fresh, skip_policy=skip)
            tasks = (
                []
                if postprocess
                else [int(index) + 1 for index in plan["pending_indices"]]
            )
            smooth = _job_requires_postprocess(job)
            if not tasks and not smooth:
                return RunHandle.skipped(self, job, "No frequency tasks need to run")
            site.prepare_job(job, sync_project=True, validate=False)
            pins = site._task_partitions(job, tasks)
            total = self._config["total_ranks"]
            job_limit = getattr(job, "max_ranks_per_task", None) or total
            limit = min(job_limit, self._config["scheduler"]["max_ranks_per_task"])
            minimum = min(self.workers.min_ranks, job_limit)
            if rank_override is not None:
                rank_override = positive_count(rank_override, "ranks_per_task")
                if rank_override > limit:
                    raise ValueError("ranks_per_task exceeds the worker rank limit")
                minimum = rank_override
            task_ranks = site._fitting_task_ranks(pins, tasks, limit=limit)
            local, remote = job.save_for_remote(site.__class__.__name__, site.work_dir)
            site._transfer_remote_simulation_inputs(job)
            site.put(Path(local), Path(remote))
            logs = Path(remote).parent / "logs"
            scheduler = {
                "job_task_count": job.n_tasks,
                "task_indices": tasks,
                "fresh": fresh,
                "min_ranks": minimum,
                "max_ranks_per_task": limit,
                "task_ranks": {str(task): ranks for task, ranks in task_ranks.items()},
                "skip_sizing": len(tasks) <= 1 or set(tasks) <= set(task_ranks),
                "sizing_json": str(logs / "FS_sizing.json"),
            }
            run = self._submit_request(
                {
                    "kind": "job",
                    "job_file": str(remote),
                    "logs": str(logs),
                    "cwd": str(
                        site._remote_run_path(site.run_config.run_path, job=job)
                    ),
                    "scheduler": scheduler,
                    "postprocess": smooth,
                    "postprocess_ranks": min(job_limit, total),
                    "pack": pack,
                },
                job,
                request_id=request_id,
                fetch=fetch,
                check=check,
                task_plan=plan,
            )
            job._job_id = self.allocation_id
            record = site._record_site_run(job, scheduler_id=self.allocation_id)
            if record is not None:
                updated = record.with_updates(
                    metadata={
                        **record.metadata,
                        "session_id": self.session_id,
                        "session_run_id": run.id,
                        "generation": self.generation,
                    }
                )
                job.write_run_record(updated)
                site._store_remote_run_records(job, updated)
            self._jobs[str(getattr(job, "job_file", job.name))] = run
            return run

    def _poll(self, run):
        payload = self._read(self.root / "runs" / str(run.id) / "status.json")
        if (
            payload
            and payload.get("generation") == self.generation
            and payload.get("state") in {"complete", "failed", "cancelled", "timeout"}
        ):
            state = payload["state"]
        else:
            allocation = self.status()
            if allocation.get("state") not in {"ready", "draining"}:
                state = (
                    "timeout" if allocation.get("reason") == "walltime" else "failed"
                )
                return JobStatus(
                    state=state,
                    return_code=1,
                    job_id=run.id,
                    message=f"allocation session unavailable: {allocation}",
                    raw={"allocation_id": self.allocation_id},
                )
            state = payload.get("state", "pending") if payload else "pending"
        raw = dict(payload or {}, allocation_id=self.allocation_id)
        raw["task_summary"] = {
            key: raw.get(key, 0)
            for key in ("total", "successful", "failed", "running", "pending")
        }
        return JobStatus(
            state=state,
            return_code=(
                0
                if state == "complete"
                else (1 if state in {"failed", "cancelled", "timeout"} else -1)
            ),
            job_id=run.id,
            message=raw.get("reason", ""),
            raw=raw,
        )

    def _finalize(self, run, status):
        self.base_site._finalize_run_record(run, status)
        return run._make_result(status)

    def cancel_job(self, job_id):
        self._command("cancel", run_id=self._id(job_id))
        return True

    def handle(self, job, job_id=None, *, mode="session"):
        """Reattach a handle by its session run id, including after reconnect."""
        if job_id is None:
            record = job.latest_run(site=self.site_name)
            job_id = record.metadata.get("session_run_id") if record else None
        run_id = self._id(job_id)
        return RunHandle(
            site=self,
            job=job,
            id=run_id,
            mode="session",
            poll_interval=0.2,
            backend={"allocation_id": self.allocation_id},
            _status_fn=self._poll,
            _cancel_fn=lambda run: self.cancel_job(run.id),
            _finalize_fn=self._finalize,
        )

    def remove_result_files(self, job, relative_paths):
        return self.base_site.remove_result_files(job, relative_paths)

    def fetch_logs(self, job, **options):
        return self.base_site.fetch_logs(job, **options)

    def fetch_wavefields(self, job, **options):
        return self.base_site.fetch_wavefields(job, **options)

    def remove_curvature_files(self, *args, **kwargs):
        return self.base_site.remove_curvature_files(*args, **kwargs)

    def run_curvature(self, request, **options):
        """Stage curvature inputs and schedule through the same resource ledger."""
        self._require_ready()
        mode = options.pop("mode", "auto")
        if mode not in {"auto", "session", "attached"} or options:
            raise ValueError("curvature uses the session's fixed allocation resources")
        site = self.base_site
        runner = SlurmCurvatureRunner(site)
        path = Path(request).absolute()
        local = read_request(path)
        runner.workdir = path.parent.parent
        runner.root = runner.remote_root(runner.workdir)
        directory = runner._remote(path.parent)
        staged = {}
        for key in FILE_KEYS:
            if key != "output" and key in local:
                runner._collect(
                    Path(local[key]).absolute(), staged, fresh=key == "input"
                )
        resolved = runner._upload(directory, list(staged.values()))
        remote = dict(local)
        remote_output = directory / Path(local["output"]).name
        for key in FILE_KEYS:
            if key in local:
                remote[key] = str(
                    remote_output
                    if key == "output"
                    else resolved[Path(local[key]).absolute()]
                )
        runner._put_text(json.dumps(remote), directory / "request.json", ".json")
        run = self._submit_request(
            {
                "kind": "curvature",
                "request_file": str(directory / "request.json"),
                "logs": str(directory),
                "cwd": str(directory),
                "ranks": 1 if single_rank(local) else self._config["total_ranks"],
            },
            None,
            check=True,
        )
        try:
            run.wait(check=True)
        except BaseException:
            self.cancel_job(run.id)
            try:
                site.get(directory / "solver.log", path.parent / "solver.log")
            except Exception:
                pass
            raise
        site.get(directory / "solver.log", path.parent / "solver.log")
        site.get(remote_output, Path(local["output"]).absolute())

    def close(self, *, drain=True):
        """Bounded, idempotent cleanup; borrowed sessions only detach."""
        if self._closed:
            return
        error = None
        try:
            if self.owned and self.allocation_id:
                try:
                    self._command("close", drain=drain)
                    deadline = time.monotonic() + self.cleanup_timeout
                    while time.monotonic() < deadline:
                        if self.status().get("state") in {
                            "closed",
                            "failed",
                            "complete",
                            "cancelled",
                            "timeout",
                        }:
                            break
                        time.sleep(0.2)
                    else:
                        if drain:
                            error = TimeoutError(
                                "session drain exceeded cleanup_timeout; unfinished work was cancelled"
                            )
                finally:
                    self.base_site.cancel_job(self.allocation_id)
        finally:
            self._closed = True
            self._ready = False
            self._stop.set()
            if self._heartbeat is not None:
                self._heartbeat.join(timeout=1)
            if self.generation:
                try:
                    self._publish(
                        self.root / "leases" / f"{self._client_id}.json",
                        {"generation": self.generation, "expires_at": 0},
                    )
                except Exception:
                    pass
        if error:
            raise error
