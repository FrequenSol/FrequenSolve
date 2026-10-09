"""Owned and borrowed persistent SLURM allocation executors."""

from __future__ import annotations

import json
import logging
import math
import shlex
import threading
import time
import warnings
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

__all__ = [
    "AdaptiveWorkers",
    "AllocationSession",
    "DEFAULT_HEARTBEAT_TIMEOUT",
    "DEFAULT_SESSION_COMMAND_TIMEOUT",
]

logger = logging.getLogger(__name__)

# The controller heartbeat lives on the same shared work filesystem as the
# solvers' output. Metadata stalls of one to several minutes occur on busy
# Lustre/GPFS scratch during large checkpoint and trace writes, so a 60 s
# window declares healthy allocations dead. Five minutes (also the
# ``scheduler_heartbeat_timeout`` default) rides out such stalls while still
# detecting a hung controller long before a typical multi-hour walltime ends.
# Loss also requires the heartbeat to stop *changing* (client monotonic clock)
# while Slurm reports the job running. Used when the site has no run config.
DEFAULT_HEARTBEAT_TIMEOUT = 300.0
# Bound on one session filesystem command on the login node (directory
# creation, lease/request/command publication). Used without a run config.
DEFAULT_SESSION_COMMAND_TIMEOUT = 120.0
# Windows below this misfire on slow shared filesystems; warn at start.
_SHORT_HEARTBEAT_WINDOW = 120.0
# Bound on waiting for the controller to cancel work before ``scancel``.
_CANCEL_TIMEOUT = 120.0
# After the controller reports closed, let the batch job end by itself (so it
# is accounted COMPLETED, not CANCELLED) before falling back to ``scancel``.
_EXIT_GRACE = 10.0
_SLURM_ENDED = {"complete", "failed", "cancelled", "timeout"}
_POLL_INTERVAL = 0.2
_PROCESSED = "processed"
_ENDED = {"closed", "failed", "complete", "cancelled", "timeout"}


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

    ``heartbeat_timeout`` is how long the controller heartbeat may stop
    advancing, measured on this client's monotonic clock while Slurm reports
    the allocation running, before the controller is declared lost. ``"auto"``
    uses the site's ``run_config.scheduler_heartbeat_timeout`` as configured
    (default 300 s); seconds override it and ``None`` disables the check.
    Windows below 120 s log a warning when the session starts.
    ``cleanup_timeout`` caps an owned session's drain on normal exit; ``None``
    (default) waits while the controller is alive. A drain that reaches the
    cap or loses the heartbeat warns and leaves the allocation to its lease
    and walltime instead of cancelling work mid-write.
    ``session_command_timeout`` bounds each login-node filesystem command the
    session runs (directory creation and the ``mv``/``ln`` that publish lease
    renewals, requests and commands); ``"auto"`` uses the site's
    ``run_config.session_command_timeout`` (default 120 s). A timeout names
    the command and remote path.
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
        cleanup_timeout=None,
        heartbeat_timeout="auto",
        session_command_timeout="auto",
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
        self.cleanup_timeout = (
            None
            if cleanup_timeout is None
            else self._seconds(cleanup_timeout, "cleanup_timeout")
        )
        self.heartbeat_timeout = self._heartbeat_window(heartbeat_timeout)
        if session_command_timeout == "auto":
            session_command_timeout = getattr(
                getattr(site, "run_config", None),
                "session_command_timeout",
                DEFAULT_SESSION_COMMAND_TIMEOUT,
            )
        self.command_timeout = self._seconds(
            session_command_timeout, "session_command_timeout"
        )
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
        # Serializes submissions only. Lease renewal and status reads never
        # take it, so a long project sync or upload cannot expire the lease.
        self._lock = threading.RLock()
        self._lease_lock = threading.Lock()
        self._lease_sequence = 0
        self._lease_revoked = False
        self._lease_thread = None
        self._state_lock = threading.Lock()
        self._slurm_checked_at = None
        self._slurm_state = "unknown"
        self._controller_state = None
        self._heartbeat_token = None
        self._heartbeat_seen_at = None
        self._stale_observations = 0
        self._stale_observed_at = None
        self._last_close_status = None
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

    def _heartbeat_window(self, value):
        if isinstance(value, str):
            if value != "auto":
                raise ValueError("heartbeat_timeout must be seconds, None or 'auto'")
            run_config = getattr(self.base_site, "run_config", None)
            configured = getattr(
                run_config, "scheduler_heartbeat_timeout", DEFAULT_HEARTBEAT_TIMEOUT
            )
            if configured is None:
                return None
            return self._seconds(configured, "scheduler_heartbeat_timeout")
        return None if value is None else self._seconds(value, "heartbeat_timeout")

    def _warn_if_short_window(self):
        window = self.heartbeat_timeout
        if window is not None and window < _SHORT_HEARTBEAT_WINDOW:
            logger.warning(
                "Allocation session %s uses a %g s controller heartbeat window. "
                "Windows below %g s misfire when the shared work filesystem "
                "stalls and fail healthy runs; raise heartbeat_timeout (or the "
                "site's run_config.scheduler_heartbeat_timeout, default 300 s).",
                self.session_id,
                window,
                _SHORT_HEARTBEAT_WINDOW,
            )

    def _identity(self):
        return f"session {self.session_id} (Slurm allocation {self.allocation_id})"

    @property
    def controller_log(self):
        """Remote path of the allocation controller's log (stdout and stderr)."""
        return self.root / "controller.log"

    def _login(self, command, *, action, path):
        return self.base_site._run_session_command(
            command, timeout=self.command_timeout, action=action, path=path
        )

    def _read(self, path):
        return self.base_site._read_remote_json(path)

    def _text(self, text, destination):
        with temporary_text_file(text, suffix=".json", prefix="fs-session-") as local:
            self.base_site.put(local, destination)

    def _publish(self, path, payload, *, immutable=False):
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        temporary = path.parent / f".{uuid4().hex}.partial"
        self._text(text, temporary)
        if not immutable:
            self._login(
                f"mv -f -- {shlex.quote(str(temporary))} {shlex.quote(str(path))}",
                action="publication (mv)",
                path=path,
            )
            return
        # Hard-link publication is atomic and never replaces a request already
        # acknowledged after an uncertain transport outcome. The controller
        # moves processed files to ``processed/``; an id archived there is
        # never published again and the archived copy is authoritative.
        processed = path.parent / _PROCESSED / path.name
        self._login(
            f"test -e {shlex.quote(str(processed))} || ln {shlex.quote(str(temporary))} {shlex.quote(str(path))} 2>/dev/null || test -f {shlex.quote(str(path))}; rm -f -- {shlex.quote(str(temporary))}",
            action="publication (ln)",
            path=path,
        )
        existing = self._read(processed) or self._read(path) or self._read(processed)
        if existing is None:
            raise RuntimeError(f"could not confirm publication of {path.name}")
        if existing != payload:
            raise ValueError("request id already belongs to different work")

    def _renew(self, *, revoke=False):
        # The controller times renewals with its own monotonic clock: any new
        # ``sequence`` counts, so client/controller clock skew is irrelevant.
        # ``expires_at`` is informational (and 0 revokes for older readers).
        with self._lease_lock:
            if self._lease_revoked:
                return
            self._lease_sequence += 1
            self._publish(
                self.root / "leases" / f"{self._client_id}.json",
                {
                    "generation": self.generation,
                    "sequence": self._lease_sequence,
                    "timeout": self.lease_timeout,
                    "revoked": revoke,
                    "expires_at": 0 if revoke else time.time() + self.lease_timeout,
                },
            )
            self._lease_revoked = revoke

    def _keep_alive(self):
        while not self._stop.wait(min(10.0, self.lease_timeout / 3)):
            try:
                self._renew()
            except Exception as error:
                # The durable lease tolerates transient transport errors. Polling
                # detects a dead controller; do not hide it behind another job.
                logger.debug("Lease renewal for %s failed: %s", self._identity(), error)
                continue

    def _start_lease(self):
        self._renew()
        self._lease_thread = threading.Thread(
            target=self._keep_alive,
            name=f"fs-session-{self.session_id[:8]}",
            daemon=True,
        )
        self._lease_thread.start()

    def start(self):
        """Submit once (or attach), renew the client lease, return this session."""
        with self._lock:
            if self._closed:
                raise RuntimeError("allocation session is closed")
            if self.allocation_id is not None:
                return self
            self._warn_if_short_window()
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
                self._start_lease()
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
                "heartbeat_timeout": self.heartbeat_timeout,
                # Several heartbeats per window, without rewriting status.json
                # more than once a second on the shared filesystem.
                "heartbeat_interval": (
                    1.0
                    if self.heartbeat_timeout is None
                    else min(1.0, self.heartbeat_timeout / 10)
                ),
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
                ),
                action="session directory creation (mkdir)",
                path=self.root,
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
            self._start_lease()
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
            if status.get("state") in _ENDED:
                raise RuntimeError(f"allocation session did not become ready: {status}")
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError("timed out waiting for persistent allocation readiness")

    def __enter__(self):
        try:
            return self.wait_ready()
        except BaseException:
            try:
                self.close(drain=False)
            except Exception as cleanup:
                # Report the readiness failure, not the cleanup failure.
                logger.warning(
                    "Releasing %s after a failed start also failed: %s",
                    self._identity(),
                    cleanup,
                    exc_info=cleanup,
                )
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close(drain=exc_type is None)
        except Exception as error:
            if exc_type is None:
                # The work in the context completed; cleanup failure must not
                # discard its results. Records stay on the remote filesystem.
                logger.warning(
                    "Closing %s failed: %s", self._identity(), error, exc_info=error
                )
                warnings.warn(
                    f"Closing allocation {self._identity()} failed after its work "
                    f"completed: {error}. Inspect it with "
                    f"site.attach_session({self.session_id!r}) and release it "
                    f"with scancel {self.allocation_id} if it is still running.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                logger.warning(
                    "Releasing %s after %s also failed: %s",
                    self._identity(),
                    exc_type.__name__,
                    error,
                    exc_info=error,
                )
        return False

    def _observe_heartbeat(self, payload):
        """Record one successful heartbeat read (caller holds ``_state_lock``)."""
        now = time.monotonic()
        token = (
            payload.get("sequence"),
            payload.get("updated_at"),
            payload.get("state"),
        )
        if token != self._heartbeat_token:
            self._heartbeat_token = token
            self._heartbeat_seen_at = now
            self._stale_observations = 0
            self._stale_observed_at = None
            return
        if (
            self.heartbeat_timeout is None
            or now - self._heartbeat_seen_at < self.heartbeat_timeout
        ):
            return
        if (
            self._stale_observed_at is None
            or now - self._stale_observed_at >= _POLL_INTERVAL
        ):
            self._stale_observations += 1
            self._stale_observed_at = now

    def status(self):
        """Return allocation and controller state, including its heartbeat.

        The controller is reported lost (``state="failed"``,
        ``heartbeat_expired=True``) only after two successful reads, at least
        one poll interval apart, show a heartbeat unchanged for longer than
        ``heartbeat_timeout`` on this client's monotonic clock while Slurm
        reports the allocation running. Wall-clock timestamps written on the
        compute node are never compared with this host's clock.
        """
        if self.allocation_id is None:
            return {"state": "new"}
        payload = self._read(self.root / "status.json")
        if payload and payload.get("generation") != self.generation:
            raise RuntimeError("allocation generation changed")
        with self._state_lock:
            if payload:
                self._controller_state = payload
                self._observe_heartbeat(payload)
            else:
                # SSH/shared-filesystem reads can fail transiently. A failed
                # read is not evidence of loss and does not count as stale.
                payload = self._controller_state
            stale = (
                time.monotonic() - self._heartbeat_seen_at
                if self._stale_observations >= 2
                else None
            )
        if payload and payload.get("state") == "failed":
            return dict(payload, controller_log=str(self.controller_log))
        if payload and payload.get("state") == "closed":
            return payload
        if (
            self._slurm_checked_at is None
            or time.monotonic() - self._slurm_checked_at > 5
        ):
            self._slurm_state = self.base_site.update_status(self.allocation_id)
            self._slurm_checked_at = time.monotonic()
        state = self._slurm_state
        if state in _SLURM_ENDED:
            return {"state": state, "controller_log": str(self.controller_log)}
        if not payload:
            return {"state": "starting" if state == "running" else state}
        if stale is not None and state == "running":
            return {
                "state": "failed",
                "heartbeat_expired": True,
                "reason": (
                    f"allocation controller heartbeat expired: unchanged for "
                    f"{stale:.0f} s (heartbeat_timeout {self.heartbeat_timeout:g} s) "
                    f"while Slurm reports allocation {self.allocation_id} running; "
                    f"see the controller log {self.controller_log} on the login node"
                ),
                "controller_log": str(self.controller_log),
                "last_heartbeat": payload,
            }
        return payload

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
            poll_interval=_POLL_INTERVAL,
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
            poll_interval=_POLL_INTERVAL,
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
        """Idempotent cleanup; borrowed sessions only detach.

        An owned session publishes ``close`` and waits for its controller.
        ``drain=True`` waits while the controller is alive (up to
        ``cleanup_timeout`` when set) so in-flight runs finish their writes;
        the allocation is cancelled only after the controller has closed. If
        the cap is reached or the heartbeat is lost during the drain, a
        ``RuntimeWarning`` names the session and allocation, which are left
        to the lease and walltime. ``drain=False`` cancels outstanding work
        and the allocation. Transport errors raise after cleanup, naming the
        session so it can be reattached.
        """
        if self._closed:
            return
        self._closed = True
        self._ready = False
        keep_lease = False
        try:
            if self.owned and self.allocation_id:
                keep_lease = self._release(drain)
        finally:
            self._stop.set()
            if self._lease_thread is not None:
                self._lease_thread.join(timeout=1)
            if self.generation and not keep_lease:
                try:
                    self._renew(revoke=True)
                except Exception as error:
                    logger.debug(
                        "Could not revoke the lease of %s: %s", self._identity(), error
                    )

    def _release(self, drain):
        """Close the owned allocation; return True when it is left running."""
        outcome, error = None, None
        try:
            self._command("close", drain=drain)
            outcome = self._await_close(drain)
        except Exception as caught:
            error = caught
        except BaseException:
            # Interrupted while waiting: the caller asked to stop now.
            self._cancel_allocation(best_effort=True)
            raise
        if outcome in {"closed", "unstarted"} or not drain:
            # A closed controller leaves nothing to protect; a cancellation
            # (drain=False) must not leave the allocation running.
            if outcome == "closed" and error is None and self._allocation_ended():
                return False
            self._cancel_allocation(
                best_effort=error is not None or outcome in {"closed", "unstarted"}
            )
            if error is not None:
                raise error
            return False
        # Never cancel a drain mid-write: leave the allocation to its lease.
        guidance = (
            f"The allocation was not cancelled, so in-flight runs are not killed "
            f"mid-write; its controller cancels unfinished work once this "
            f"client's lease lapses ({self.lease_timeout:g} s) and Slurm ends it "
            f"at walltime. Inspect it with site.attach_session({self.session_id!r}) "
            f"or release it with scancel {self.allocation_id}."
        )
        if error is not None:
            logger.warning(
                "Could not confirm that %s drained: %s", self._identity(), error
            )
            raise RuntimeError(
                f"Could not confirm that allocation {self._identity()} drained: "
                f"{error}. {guidance}"
            ) from error
        if outcome == "lost":
            reason = (self._last_close_status or {}).get("reason", "heartbeat expired")
            message = (
                f"Allocation {self._identity()} lost its controller heartbeat while "
                f"draining ({reason}). {guidance}"
            )
        else:
            active = (self._controller_state or {}).get("active_runs", "unknown")
            message = (
                f"Allocation {self._identity()} was still draining {active} run(s) "
                f"when cleanup_timeout ({self.cleanup_timeout:g} s) elapsed. "
                f"{guidance}"
            )
        logger.warning(message)
        warnings.warn(message, RuntimeWarning, stacklevel=4)
        return True

    def _await_close(self, drain):
        """Wait for the controller: ``closed``, ``unstarted``, ``cap`` or ``lost``."""
        cap = self.cleanup_timeout
        if cap is None and not drain:
            # Cancellation is quick unless the controller is stuck; then the
            # allocation is cancelled anyway.
            cap = _CANCEL_TIMEOUT
        started = time.monotonic()
        announced = False
        self._last_close_status = None
        while True:
            status = self.status()
            self._last_close_status = status
            if status.get("heartbeat_expired"):
                return "lost"
            if self._controller_state is None:
                return "unstarted"  # nothing was submitted, nothing to drain
            if status.get("state") in _ENDED:
                return "closed"
            elapsed = time.monotonic() - started
            if cap is not None and elapsed >= cap:
                return "cap"
            if drain and not announced and status.get("active_runs"):
                logger.info(
                    "Waiting for %s to drain %s active run(s)",
                    self._identity(),
                    status["active_runs"],
                )
                announced = True
            # Long drains poll gently; each poll is a remote read.
            time.sleep(min(2.0, _POLL_INTERVAL + elapsed / 30))

    def _allocation_ended(self):
        """Wait briefly for Slurm to report the closed controller's job ended."""
        deadline = time.monotonic() + _EXIT_GRACE
        while True:
            try:
                if self.base_site.update_status(self.allocation_id) in _SLURM_ENDED:
                    return True
            except Exception as error:
                logger.debug("Could not query %s: %s", self._identity(), error)
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.5)

    def _cancel_allocation(self, *, best_effort):
        """``scancel``; ``best_effort`` logs failures instead of raising."""
        try:
            self.base_site.cancel_job(self.allocation_id)
        except Exception as failure:
            if best_effort:
                logger.warning(
                    "Could not cancel %s: %s",
                    self._identity(),
                    failure,
                    exc_info=failure,
                )
                return
            raise RuntimeError(
                f"Could not cancel allocation {self._identity()}: {failure}. Release "
                f"it with scancel {self.allocation_id}."
            ) from failure
