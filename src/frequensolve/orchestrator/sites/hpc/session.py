"""Owned and borrowed persistent SLURM allocation executors."""

from __future__ import annotations

import getpass
import hashlib
import json
import logging
import math
import os
import re
import shlex
import socket
import sys
import threading
import time
import warnings
from dataclasses import dataclass, field
from importlib.resources import as_file, files
from pathlib import Path
from types import SimpleNamespace
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
    partition_gpus_per_node,
    rank_memory_gib,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    SESSION_JOB_NAME,
    slurm_time_seconds,
    temporary_text_file,
)
from frequensolve.simulation.jobs.run_state import SkipPolicy

__all__ = [
    "AdaptiveWorkers",
    "AllocationSession",
    "DEFAULT_HEARTBEAT_TIMEOUT",
    "DEFAULT_MIN_TIME_LEFT",
    "DEFAULT_SESSION_COMMAND_TIMEOUT",
    "SessionInfo",
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
# Reuse skips sessions with less walltime left than this, by default.
DEFAULT_MIN_TIME_LEFT = 900.0
# How long an adopting client waits for the controller to list its lease.
_ADOPT_CONFIRM_TIMEOUT = 15.0
# A lingering session this close to the end of its linger is not adopted.
_LINGER_MARGIN = 30.0
# Settings a session must share with a request to be reused for it.
_COMPATIBLE = {
    "schema": "its controller version",
    "nodes": "nodes",
    "ranks_per_node": "ranks_per_node",
    "total_ranks": "total ranks",
    "queue": "queue",
    "account": "account",
    "gpus_per_node": "GPUs per node",
    "rank_binding": "rank binding",
    "environment_digest": "module and environment setup",
}
_COMPATIBLE_SCHEDULER = (
    "executable",
    "mpi",
    "mpi_args",
    "omp_threads",
    "mem_per_rank_gib",
    "min_ranks",
    "max_ranks_per_task",
    "mem_cushion",
)
_DURATION = re.compile(
    r"(?:(?P<h>\d+(?:\.\d*)?)h)?(?:(?P<m>\d+(?:\.\d*)?)m)?(?:(?P<s>\d+(?:\.\d*)?)s)?"
)


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


def _duration(value, name):
    """Seconds from a number or a duration such as ``"90s"``, ``"30m"``,
    ``"1h30m"`` or Slurm's ``[D-]HH:MM:SS``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be seconds or a duration such as '30m'")
    if isinstance(value, str):
        text = value.strip().lower()
        match = _DURATION.fullmatch(text)
        if text and match and any(match.groupdict().values()):
            seconds = sum(
                float(match[unit] or 0) * scale
                for unit, scale in (("h", 3600.0), ("m", 60.0), ("s", 1.0))
            )
        else:
            seconds = slurm_time_seconds(text)
            if seconds is None:
                raise ValueError(
                    f"{name} must be seconds or a duration such as '30m', '2h' "
                    "or '01:30:00'"
                )
    else:
        seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be finite and >= 0")
    return seconds


def _describe(seconds):
    for unit, size in (("h", 3600.0), ("min", 60.0)):
        if seconds >= size:
            return f"{seconds / size:.3g} {unit}"
    return f"{seconds:.3g} s"


def reuse_key(reuse):
    """Return ``(key, label)`` for a ``reuse`` scope: a string or a project.

    A project's key combines its name with a digest of its resolved path, so
    unrelated projects that share a name never share allocations. A string
    names a scope directly, for example to share across projects.
    """
    if reuse is None:
        return None, None
    if isinstance(reuse, str):
        if not reuse.strip():
            raise ValueError("reuse must be a non-empty string or a Project")
        return f"key:{reuse}", reuse
    name, path = getattr(reuse, "name", None), getattr(reuse, "path", None)
    if isinstance(name, str) and path is not None:
        resolved = str(Path(path).expanduser().resolve())
        digest = hashlib.sha256(resolved.encode()).hexdigest()[:16]
        return f"project:{name}:{digest}", name
    raise TypeError("reuse must be None, a string or a Project")


def _creator():
    """Who started a session, shown in listings; never used to grant reuse."""
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover - no login name in the environment
        user = None
    return {
        "user": user,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "script": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else None,
    }


def _revoked(lease):
    return bool(lease.get("revoked")) or float(lease.get("expires_at", 1) or 0) <= 0


def _gone_here(lease):
    """Whether a lease names a process on this host that has exited."""
    pid = lease.get("pid")
    if (
        lease.get("host") != socket.gethostname()
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or pid == os.getpid()
    ):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False  # alive, but owned by someone else
    return False


def _live_leases(leases, status):
    """Return the ids of the clients whose leases are live.

    Revoked leases, and leases of exited processes on this host, are not.
    With a running controller (``status``), a lease is live if the
    controller lists it or if it renewed since the controller last read it;
    before the controller runs, every other lease counts as live.
    """
    live = {
        client
        for client, lease in leases.items()
        if not _revoked(lease) and not _gone_here(lease)
    }
    if not status or "client_ids" not in status:
        return live
    listed = set(status.get("client_ids") or ())
    observed = status.get("lease_sequences") or {}
    return {
        client
        for client in live
        if client in listed or observed.get(client) != leases[client].get("sequence")
    }


def _command_timeout(site):
    return getattr(
        getattr(site, "run_config", None),
        "session_command_timeout",
        DEFAULT_SESSION_COMMAND_TIMEOUT,
    )


def _session_dirs(site, allocation_ids, timeout):
    """Map live allocation ids to the ids of the sessions that recorded them."""
    if not allocation_ids:
        return {}
    sessions = site.work_dir / ".fs_sessions"
    needles = " ".join(
        "-e " + shlex.quote(f'"allocation_id": "{job}"') for job in allocation_ids
    )
    output = site._run_session_command(
        f"grep -H -o -F {needles} -- {shlex.quote(str(sessions))}/*/allocation.json "
        "2>/dev/null; true",
        timeout=timeout,
        action="session lookup (grep)",
        path=sessions,
    )
    found = {}
    for line in output.splitlines():
        directory, separator, match = line.partition("/allocation.json:")
        fields = match.split('"')
        if not separator or len(fields) < 4:
            continue
        session_id, allocation_id = directory.rsplit("/", 1)[-1], fields[3]
        if allocation_id not in allocation_ids or len(session_id) != 32:
            continue
        # An allocation recorded twice is ambiguous; never guess.
        found[allocation_id] = None if allocation_id in found else session_id
    return {job: session for job, session in found.items() if session is not None}


def _read_leases(site, root, timeout):
    """Return ``{client id: lease}`` for the client leases of a session."""
    leases = root / "leases"
    output = site._run_session_command(
        f"for f in {shlex.quote(str(leases))}/*.json; do "
        '[ -f "$f" ] || continue; printf "%s\t" "${f##*/}"; head -n 1 -- "$f"; '
        "echo; done; true",
        timeout=timeout,
        action="lease listing",
        path=leases,
    )
    found = {}
    for line in output.splitlines():
        name, _, text = line.partition("\t")
        if not name.endswith(".json") or not text.strip():
            continue
        try:
            lease = json.loads(text)
        except ValueError:
            continue
        if isinstance(lease, dict):
            found[name[: -len(".json")]] = lease
    return found


@dataclass(frozen=True)
class SessionInfo:
    """A queued or running persistent session of this user on a site.

    ``state`` is ``"pending"`` while Slurm queues the allocation, then the
    controller's state (``"starting"`` before its first heartbeat).
    ``clients`` counts attached clients (``None`` before the controller
    runs); ``idle_seconds`` is how long nothing has been attached or running.
    ``time_left`` is Slurm's remaining walltime in seconds (the time limit
    while pending). ``reuse`` names the reuse scope the session was started
    with, if any.
    """

    session_id: str
    allocation_id: str
    state: str
    clients: Optional[int]
    idle_seconds: Optional[float]
    linger: float
    time_left: Optional[float]
    nodes: Optional[int]
    ranks_per_node: Optional[int]
    queue: Optional[str]
    reuse: Optional[str]
    created_by: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict, repr=False, compare=False)
    status: Optional[dict] = field(default=None, repr=False, compare=False)


def list_sessions(site, reuse=None):
    """List the user's queued and running sessions on ``site``.

    Only Slurm jobs of the current user named like session allocations, and
    recorded by a session directory under the site's work directory, are
    considered. ``reuse`` keeps the sessions started with that reuse scope.
    """
    return _list_sessions(site, reuse_key(reuse)[0])


def _list_sessions(site, key):
    jobs = site._session_allocations()
    sessions = []
    for allocation_id, session_id in sorted(
        _session_dirs(site, sorted(jobs), _command_timeout(site)).items()
    ):
        root = site.work_dir / ".fs_sessions" / session_id
        config = site._read_remote_json(root / "config.json")
        if not isinstance(config, dict) or "generation" not in config:
            continue
        reuse = config.get("reuse") or {}
        if key is not None and reuse.get("key") != key:
            continue
        job = jobs[allocation_id]
        status = None
        if job["state"] == "running":
            status = site._read_remote_json(root / "status.json")
            if (
                not isinstance(status, dict)
                or status.get("generation") != config["generation"]
            ):
                status = None
        if job["state"] == "running":
            state = (status or {}).get("state") or "starting"
        else:
            state = job["state"]  # "pending", or a state reuse never adopts
        if state in _ENDED:
            continue  # its controller closed; Slurm is still completing the job
        sessions.append(
            SessionInfo(
                session_id=session_id,
                allocation_id=allocation_id,
                state=state,
                clients=None if status is None else status.get("clients"),
                idle_seconds=None if status is None else status.get("idle_seconds"),
                linger=float(
                    (status or {}).get("linger", config.get("linger") or 0.0) or 0.0
                ),
                time_left=job.get("time_left"),
                nodes=config.get("nodes"),
                ranks_per_node=config.get("ranks_per_node"),
                queue=config.get("queue"),
                reuse=reuse.get("label"),
                created_by=dict(config.get("created_by") or {}),
                config=config,
                status=status,
            )
        )
    return sessions


def release_sessions(site, reuse=None, *, include_attached=False):
    """End the user's idle sessions on ``site``; return the released ones.

    ``reuse`` limits this to sessions started with that reuse scope; ``None``
    releases every idle session of this user on the site. Sessions with an
    attached client are kept unless ``include_attached=True``, which cancels
    their work.
    """
    released = []
    for info in list_sessions(site, reuse):
        session = AllocationSession(site, session_id=info.session_id)
        if not include_attached:
            live, _ = session._attached(info, observe=False)
            if live:
                logger.info(
                    "Keeping allocation %s (session %s): %d client(s) attached",
                    info.allocation_id,
                    info.session_id,
                    len(live),
                )
                continue
        session._load(info.config)
        session.release(drain=False)
        released.append(info)
    return released


class AllocationSession(BaseSite):
    """Site-compatible executor backed by one allocation-resident scheduler.

    Construct with ``site.session(...)`` and enter the context to start and
    await readiness. ``site.attach_session(id)`` borrows an existing session.
    Closing an owned session drains its work and releases the allocation;
    closing a borrowed session only disconnects this client. Remote request
    and result records are retained for inspection and reattachment.

    ``reuse`` (a project or a string) lets a later client adopt this session
    instead of queueing a new allocation, and lets this one adopt an earlier
    session started with the same ``reuse``: only the user's own running or
    queued sessions with that scope, the same resources, solver and
    environment, at least ``min_time_left`` of walltime and no attached
    client qualify (``share=True`` also joins sessions other clients use).
    Non-FrequenSolve jobs never qualify. ``linger`` (seconds or a duration
    such as ``"30m"``; default 0) keeps the allocation that long once no
    client is attached and nothing runs, including after a normal exit, so
    closing only detaches; :meth:`release` ends it now. With ``linger``, a
    client that disappears leaves its runs to finish, and resubmitting such
    a job after adopting the session waits on the existing run.

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
        reuse=None,
        linger=0,
        share=False,
        min_time_left=DEFAULT_MIN_TIME_LEFT,
        **resources,
    ):
        self.base_site = site
        self.session_id = self._id(session_id or uuid4().hex)
        self._new_session_id = self.session_id
        self.root = self._session_root(self.session_id)
        self.owned = session_id is None
        self.reuse_key, self.reuse_label = reuse_key(reuse)
        self.linger = _duration(linger, "linger")
        self.share = bool(share)
        self.min_time_left = _duration(min_time_left, "min_time_left")
        if not self.owned and (reuse is not None or self.share or self.linger):
            raise ValueError(
                "attach_session(session_id) borrows a session; reuse, share and "
                "linger apply to sessions this client starts or adopts"
            )
        if self.share and self.reuse_key is None:
            raise ValueError("share=True requires reuse")
        # Whether this client adopted an earlier session instead of starting one.
        self.reused = False
        self._adopt_only = False
        self._released = False
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

    def _session_root(self, session_id):
        return self.base_site.work_dir / ".fs_sessions" / session_id

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
                    # Lets a later client on this host see that we exited.
                    "client": self._client_id,
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    # The last clients attached decide how long it lingers.
                    "linger": self.linger if self.owned else None,
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

    def _end_lease(self, *, revoke):
        self._stop.set()
        if self._lease_thread is not None:
            self._lease_thread.join(timeout=1)
        # A client that never wrote a lease has nothing to revoke.
        if self.generation and revoke and self._lease_sequence:
            try:
                self._renew(revoke=True)
            except Exception as error:
                logger.debug(
                    "Could not revoke the lease of %s: %s", self._identity(), error
                )

    def start(self):
        """Submit once, adopt a reusable session, or attach; renew the lease."""
        with self._lock:
            if self._closed:
                raise RuntimeError("allocation session is closed")
            if self.allocation_id is not None:
                return self
            self._warn_if_short_window()
            if not self.owned:
                self._load()
                self.lease_timeout = float(self._config["lease_timeout"])
                self._start_lease()
                return self
            planned, plan = self._plan()
            site = self.base_site
            if site.enterprise_hpc is not None:
                site._enterprise_hpc_preflight(run_config=plan.run_config)
            if self.reuse_key is not None and self._adopt(planned):
                return self
            if self._adopt_only:
                raise RuntimeError(
                    f"No running or queued allocation session for "
                    f"{self.reuse_label!r} can be reused with these resources; "
                    "start one with site.session(..., reuse=...)"
                )
            if site.enterprise_hpc is None:
                site.check_solver_compatibility()
            self._submit_allocation(planned, plan)
            return self

    def _load(self, config=None):
        """Read an existing session's configuration and allocation id."""
        self._config = config or self._read(self.root / "config.json")
        metadata = self._read(self.root / "allocation.json")
        if not self._config or not metadata:
            raise RuntimeError("allocation session was not found")
        self.generation = self._config["generation"]
        self.allocation_id = str(metadata["allocation_id"])
        scheduler = self._config["scheduler"]
        self.workers = AdaptiveWorkers(
            min_ranks=scheduler["min_ranks"],
            max_ranks_per_task=scheduler["max_ranks_per_task"],
            mem_cushion=scheduler["mem_cushion"],
        )

    def _plan(self):
        """Resolve and validate a new allocation's configuration, locally."""
        site = self.base_site
        config, extra = site.run_config.resolved(site.config, **self.resources)
        if extra:
            raise TypeError(
                "Unexpected allocation resource option(s): " + ", ".join(sorted(extra))
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
        threads = shape.cores_per_node // per_node
        runtime_setup = site._runtime_setup_lines(
            getattr(shape, "partition_environment", None)
        )
        planned = {
            "schema": "fs-allocation-session-1",
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
            # Ranks of concurrent steps select the GPU under their slot.
            "gpus_per_node": partition_gpus_per_node(shape, config.queue),
            # Reuse requires the same modules and environment.
            "environment_digest": hashlib.sha256(
                "\n".join(runtime_setup).encode()
            ).hexdigest(),
            "linger": self.linger,
            "reuse": (
                None
                if self.reuse_key is None
                else {"key": self.reuse_key, "label": self.reuse_label}
            ),
            "created_by": _creator(),
            "scheduler": {
                "executable": str(site.executable),
                "mpi": str(site.mpi_cmd),
                "mpi_args": list(site.mpi_args),
                "total_ranks": nodes * per_node,
                "omp_threads": threads,
                "mem_per_rank_gib": rank_memory_gib(shape, per_node),
                "min_ranks": self.workers.min_ranks,
                "max_ranks_per_task": limit,
                "mem_cushion": self.workers.mem_cushion,
            },
        }
        plan = SimpleNamespace(
            run_config=config,
            duration=duration,
            per_node=per_node,
            threads=threads,
            runtime_setup=runtime_setup,
        )
        return planned, plan

    def _submit_allocation(self, planned, plan):
        """Create the session directory and submit its batch job."""
        site = self.base_site
        self.generation = uuid4().hex
        self._config = {**planned, "generation": self.generation}
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
        config = plan.run_config
        script = site._render_template(
            "session/session_SLURM.sh",
            name=_safe_slurm_directive(SESSION_JOB_NAME, "name"),
            queue=_safe_slurm_directive(config.queue, "queue"),
            account=_safe_slurm_directive(
                config.account or site.config.account, "account"
            ),
            duration=_safe_slurm_directive(plan.duration, "duration"),
            nodes=planned["nodes"],
            ranks=planned["total_ranks"],
            ranks_per_node=plan.per_node,
            cpus_per_task=(
                None if Path(str(site.mpi_cmd)).name == "ibrun" else plan.threads
            ),
            root=shlex.quote(str(self.root)),
            log_path=_safe_slurm_directive(
                str(self.root / "controller.log"), "log_path"
            ),
            runtime_setup=plan.runtime_setup,
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

    # -- reuse ----------------------------------------------------------------

    def _adopt(self, planned):
        """Take over a compatible session started with the same reuse scope."""
        site = self.base_site
        try:
            candidates = _list_sessions(site, self.reuse_key)
        except Exception as error:
            logger.warning(
                "Could not look up reusable allocation sessions for %r; starting "
                "a new allocation: %s",
                self.reuse_label,
                error,
            )
            return False
        # Running sessions first, then those with the most walltime left.
        candidates.sort(
            key=lambda info: (info.state == "pending", -(info.time_left or 0.0))
        )
        for info in candidates:
            reason = self._unsuitable(info, planned)
            seen = set()
            if reason is None:
                live, seen = self._attached(info)
                if live and not self.share:
                    reason = (
                        f"{len(live)} other client(s) attached; pass share=True to "
                        "join it"
                    )
            if reason is not None:
                site._emit(
                    f"Not reusing allocation {info.allocation_id} (session "
                    f"{info.session_id}): {reason}"
                )
                continue
            if self._claim(info, seen):
                return True
        return False

    def _unsuitable(self, info, planned):
        """Return why a candidate session cannot take this work, or None."""
        if info.state not in {"pending", "starting", "ready"}:
            return f"its controller is {info.state}"
        if info.time_left is not None and info.time_left < self.min_time_left:
            return (
                f"{info.time_left:.0f} s of walltime left, below min_time_left "
                f"({self.min_time_left:g} s)"
            )
        if (
            info.idle_seconds is not None
            and info.linger - info.idle_seconds < _LINGER_MARGIN
        ):
            return "its linger period is about to end"
        config = info.config
        for key, label in _COMPATIBLE.items():
            if config.get(key) != planned.get(key):
                if key == "environment_digest":
                    return f"its {label} differs"
                return (
                    f"{label} differs (session {config.get(key)!r}, requested "
                    f"{planned.get(key)!r})"
                )
        scheduler = config.get("scheduler") or {}
        for key in _COMPATIBLE_SCHEDULER:
            if scheduler.get(key) != planned["scheduler"].get(key):
                return (
                    f"{key} differs (session {scheduler.get(key)!r}, requested "
                    f"{planned['scheduler'].get(key)!r})"
                )
        return None

    def _attached(self, info, *, observe=True):
        """Return ``(live, seen)`` client ids of a candidate session's leases.

        With a running controller, liveness is as in :func:`_live_leases`.
        Before it runs, a lease is live while its sequence advances, observed
        over one renewal period when ``observe``.
        """
        site = self.base_site
        root = self._session_root(info.session_id)
        leases = _read_leases(site, root, self.command_timeout)
        seen = set(leases)
        if info.status is not None and "client_ids" in info.status:
            return _live_leases(leases, info.status), seen
        waiting = {client: leases[client] for client in _live_leases(leases, None)}
        if not waiting or not observe:
            return set(waiting), seen
        period = max(
            min(10.0, float(lease.get("timeout") or 300.0) / 3)
            for lease in waiting.values()
        )
        site._emit(
            f"Checking whether a client still waits for queued allocation "
            f"{info.allocation_id} (session {info.session_id})"
        )
        time.sleep(period + 5.0)
        again = _read_leases(site, root, self.command_timeout)
        live = {
            client
            for client, lease in waiting.items()
            if client in again
            and not _revoked(again[client])
            and again[client].get("sequence") != lease.get("sequence")
        }
        return live, seen | set(again)

    def _claim(self, info, seen):
        """Adopt ``info``; back off if another client adopts it at once."""
        self.session_id = info.session_id
        self.root = self._session_root(info.session_id)
        self._config = info.config
        self.generation = info.config["generation"]
        self.allocation_id = info.allocation_id
        try:
            self._start_lease()
            reason = None
            if not self.share:
                # Each adopter writes its lease before reading the others', so
                # of two concurrent adopters at least one sees the other.
                current = _read_leases(self.base_site, self.root, self.command_timeout)
                if any(
                    client != self._client_id
                    and client not in seen
                    and not _revoked(lease)
                    for client, lease in current.items()
                ):
                    reason = "another client is adopting it"
            if reason is None and not self._confirm_adoption():
                reason = "it ended while being adopted"
        except BaseException:
            self._abandon()
            raise
        if reason is not None:
            self.base_site._emit(
                f"Not reusing allocation {info.allocation_id} (session "
                f"{info.session_id}): {reason}"
            )
            self._abandon()
            return False
        self.reused = True
        scheduler = self._config["scheduler"]
        self.workers = AdaptiveWorkers(
            min_ranks=scheduler["min_ranks"],
            max_ranks_per_task=scheduler["max_ranks_per_task"],
            mem_cushion=scheduler["mem_cushion"],
        )
        detail = info.state
        if info.idle_seconds is not None:
            detail += f", idle {_describe(round(info.idle_seconds))}"
        if info.time_left is not None and math.isfinite(info.time_left):
            detail += f", {_describe(round(info.time_left))} left"
        self.base_site._emit(
            f"Reusing allocation {self.allocation_id} (session {self.session_id}; "
            f"{detail})",
            force=True,
        )
        return True

    def _confirm_adoption(self):
        """Wait for a running controller to list this client; False if it ended."""
        deadline = time.monotonic() + _ADOPT_CONFIRM_TIMEOUT
        while True:
            status = self._read(self.root / "status.json")
            if not status or status.get("generation") != self.generation:
                return True  # not started: the controller reads leases at start
            if status.get("state") in _ENDED or status.get("reason"):
                return False
            if self._client_id in (status.get("client_ids") or ()):
                return True
            if time.monotonic() >= deadline:
                # A slow filesystem; wait_ready() keeps checking liveness.
                return True
            time.sleep(0.5)

    def _abandon(self):
        """Drop an unsuccessful claim, leaving the candidate session as it was."""
        self._end_lease(revoke=True)
        self._stop = threading.Event()
        self._lease_thread = None
        self._lease_revoked = False
        self.session_id = self._new_session_id
        self.root = self._session_root(self.session_id)
        self._config = self.generation = self.allocation_id = None

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
        except BaseException as error:
            try:
                if (
                    self.owned
                    and not self.reused
                    and not (self.linger and isinstance(error, KeyboardInterrupt))
                ):
                    # A new allocation that never became usable. Interrupting
                    # a lingering one keeps its place in the queue for reuse.
                    self.release(drain=False)
                else:
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
        handle = self._run_handle(
            job, run_id, fetch=fetch, check=check, task_plan=task_plan
        )
        self._handles[run_id] = handle
        return handle

    def _run_handle(self, job, run_id, *, fetch=False, check=False, task_plan=None):
        return RunHandle(
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

    def _resume(self, job, *, fresh, fetch, check):
        """Return a handle to this job's run that an earlier client left running.

        The job's run record names the session run. ``fresh`` cancels that run
        (and waits for it to stop) so the job is submitted again.
        """
        latest = getattr(job, "latest_run", None)
        record = latest(site=self.site_name) if callable(latest) else None
        metadata = getattr(record, "metadata", None) or {}
        run_id = metadata.get("session_run_id")
        if (
            not run_id
            or run_id in self._handles
            or metadata.get("session_id") != self.session_id
            or metadata.get("generation") != self.generation
        ):
            return None
        handle = self._run_handle(job, self._id(run_id), fetch=fetch, check=check)
        if handle.status().is_complete:
            return None
        if fresh:
            handle.cancel()
            handle.wait(check=False)
            return None
        self.base_site._emit(
            f"Resuming run {run_id} of {getattr(job, 'name', 'job')} in "
            f"{self._identity()}"
        )
        self._handles[handle.id] = handle
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
            resumed = self._resume(job, fresh=fresh, fetch=fetch, check=check)
            if resumed is not None:
                self._jobs[str(getattr(job, "job_file", job.name))] = resumed
                return resumed
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

        An owned session with ``linger``, or with other clients attached,
        only detaches: ``drain=True`` waits for this client's runs (up to
        ``cleanup_timeout``) and ``drain=False`` cancels them, while the
        allocation stays up for the other clients, then for ``linger``
        seconds. :meth:`release` ends it now.
        """
        if self._closed:
            return
        self._closed = True
        self._ready = False
        keep_lease = False
        try:
            if self.owned and self.allocation_id:
                others = self._other_clients()
                if self.linger > 0 or others:
                    self._detach(drain, others)
                else:
                    keep_lease = self._release(drain)
        finally:
            self._end_lease(revoke=not keep_lease)

    def release(self, *, drain=True):
        """End the allocation now, including work other clients submitted.

        Also works after :meth:`close`, for example on a lingering session
        once its ``with`` block has exited. ``drain`` is as for :meth:`close`.
        """
        if self.allocation_id is None:
            if self.owned:
                self.close(drain=drain)
                return
            self._load()
        if self._released:
            return
        self._released = True
        self._closed = True
        self._ready = False
        keep_lease = False
        try:
            keep_lease = self._release(drain)
        finally:
            self._end_lease(revoke=not keep_lease)

    def _other_clients(self):
        """Count other attached clients, as the controller or leases report."""
        try:
            status = self._read(self.root / "status.json")
            leases = _read_leases(self.base_site, self.root, self.command_timeout)
        except Exception as error:
            logger.debug(
                "Could not list the clients of %s: %s", self._identity(), error
            )
            return 0
        if not status or status.get("generation") != self.generation:
            status = None
        return len(_live_leases(leases, status) - {self._client_id})

    def _detach(self, drain, others):
        """Leave the allocation to other clients or to its ``linger``."""
        unfinished = [
            handle
            for handle in self._handles.values()
            if handle._result is None and not handle._last_status.is_complete
        ]
        if unfinished and drain:
            unfinished = self._await_own_runs(unfinished)
        elif unfinished:
            for handle in unfinished:
                try:
                    self.cancel_job(handle.id)
                except Exception as error:
                    logger.warning(
                        "Could not cancel run %s in %s: %s",
                        handle.id,
                        self._identity(),
                        error,
                    )
            unfinished = []
        if others:
            kept = f"it stays up for {others} other attached client(s)"
            if self.linger:
                kept += f", then {_describe(self.linger)} once idle (linger)"
        else:
            kept = f"it stays up {_describe(self.linger)} once idle (linger)"
        again = (
            "site.session(..., reuse=<the same reuse>)"
            if self.reuse_key is not None
            else f"site.attach_session({self.session_id!r})"
        )
        message = (
            f"Detached from allocation {self.allocation_id} (session "
            f"{self.session_id}); {kept}. Use it again with {again}, or end it "
            f"with session.release(), site.release_sessions(...) or scancel "
            f"{self.allocation_id}."
        )
        if unfinished:
            message += (
                f" {len(unfinished)} run(s) of this client were still running and "
                "continue."
            )
        self.base_site._emit(message, force=True)

    def _await_own_runs(self, handles):
        """Wait (up to ``cleanup_timeout``) for runs; return those still running."""
        started = time.monotonic()
        announced = False
        while True:
            handles = [
                handle for handle in handles if not self._poll(handle).is_complete
            ]
            elapsed = time.monotonic() - started
            if not handles or (
                self.cleanup_timeout is not None and elapsed >= self.cleanup_timeout
            ):
                return handles
            if not announced:
                logger.info(
                    "Waiting for %d run(s) in %s before detaching",
                    len(handles),
                    self._identity(),
                )
                announced = True
            time.sleep(min(2.0, _POLL_INTERVAL + elapsed / 30))

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
