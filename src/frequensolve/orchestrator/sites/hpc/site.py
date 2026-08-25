"""Generic SSH/SLURM HPC site support.

This module contains the reusable mechanics for SLURM-backed remote sites.
Built-in presets and user configuration provide partition shapes, scheduler
limits, credentials, and default paths without requiring site subclasses.
"""

import asyncio
import json
import logging
import os
import re
import shlex
import signal
import socket
import subprocess
import time
from asyncio import Future
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from datetime import datetime, timezone
from importlib.resources import as_file, files
from pathlib import Path
from select import select
from typing import Any, Dict, List, Literal, Mapping, Optional, Type, Union

from frequensolve._optional import optional_dependency_error

try:
    from paramiko import SSHClient
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "SlurmSite",
        extra="hpc",
        dependencies=("paramiko",),
        error=exc,
    ) from exc

from jinja2 import Environment, PackageLoader

from frequensolve.frequensolver import (
    IDENTITY_QUERY_TIMEOUT_SECONDS,
    FrequenSolverCompatibility,
    check_frequensolver_compatibility,
    resolve_frequensolver_policy,
)
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    JobStatus,
    RunHandle,
    RunResult,
    _check_if_notebook,
    _merge_task_status_with_plan,
)
from frequensolve.orchestrator.sites.config import BaseSiteConfig
from frequensolve.orchestrator.sites.config_file import _host_tmp_path_for_config
from frequensolve.orchestrator.sites.hpc.auth import SlurmAuthenticator
from frequensolve.orchestrator.sites.hpc.slurm_helpers import as_list as _as_list
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    hms_to_seconds as _hms_to_seconds,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    normalize_slurm_state as _normalize_slurm_state,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    parse_sbatch_job_id as _parse_sbatch_job_id,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    read_stream as _read_stream,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    seconds_to_hms as _seconds_to_hms,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    ssh_exit_status as _ssh_exit_status,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    temporary_text_file as _temporary_text_file,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    validate_slurm_job_id as _validate_slurm_job_id,
)
from frequensolve.orchestrator.sites.hpc.transfer import SlurmTransferManager
from frequensolve.orchestrator.utils.credential_store import CredentialStore
from frequensolve.orchestrator.utils.credentials import Credentials
from frequensolve.orchestrator.utils.environment import (
    NUMERIC_RUNTIME_DEFAULTS,
    validate_environment,
)
from frequensolve.orchestrator.utils.pool import PoolInfo
from frequensolve.orchestrator.utils.ssh import SSHClientClass
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.jobs import BaseJob, SkipPolicy
from frequensolve.simulation.jobs.imaging import ImagingJob
from frequensolve.util.setup_logger import init_logger

__all__ = [
    "SlurmSiteConfig",
    "SlurmPartitionConfig",
    "SlurmLoginCredentials",
    "SlurmRunConfig",
    "SlurmSite",
]

# Initialize the logger
logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/hpc.log")

_HPC_RUNTIME_DEFAULTS = {
    **NUMERIC_RUNTIME_DEFAULTS,
    "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "20ms",
    "KMP_STACKSIZE": "20M",
}

_ADAPTIVE_SCHEDULER_HEARTBEAT_TIMEOUT = 60.0

_SHELL_ENV_REFERENCE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _quote_runtime_environment_value(value: str) -> str:
    """Quote an environment value while expanding simple ``${NAME}`` refs."""

    references = list(_SHELL_ENV_REFERENCE.finditer(value))
    if not references:
        return shlex.quote(value)

    def escape_literal(fragment: str) -> str:
        return (
            fragment.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("$", "\\$")
            .replace("`", "\\`")
        )

    pieces = []
    position = 0
    for reference in references:
        pieces.append(escape_literal(value[position : reference.start()]))
        pieces.append(reference.group(0))
        position = reference.end()
    pieces.append(escape_literal(value[position:]))
    return f'"{"".join(pieces)}"'


# ----------------------------------
# Generic SLURM Config
# ----------------------------------
@dataclass(frozen=True)
class SlurmPartitionConfig:
    """Resource shape and scheduler limits for one SLURM partition."""

    max_duration: str = "00-02:00:00"
    min_nodes: int = 1
    max_nodes: int = 1
    cores_per_node: int = 1
    sockets_per_node: int = 1
    memory_per_node: int = 0
    gpus_per_node: int = 0


@dataclass
class SlurmSiteConfig(BaseSiteConfig):
    """Minimal reusable configuration for a SLURM-backed site.

    Site-specific config classes can expose the same attributes and do not need
    to inherit from this class.
    """

    hostname: str
    queue: str = "normal"
    scheduler: str = "SLURM"
    mpi_wrapper: str = "srun"
    poll_interval: int = 5
    account: str = ""
    tmp_dir: Optional[Union[str, Path]] = None
    max_duration: str = "00-02:00:00"
    min_nodes: int = 1
    max_nodes: int = 1
    cores_per_node: int = 1
    sockets_per_node: int = 1
    memory_per_node: int = 0
    gpus_per_node: int = 0
    partitions: Mapping[str, Union[SlurmPartitionConfig, Mapping[str, Any]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        normalized: Dict[str, SlurmPartitionConfig] = {}
        for name, value in self.partitions.items():
            if isinstance(value, SlurmPartitionConfig):
                partition = value
            elif isinstance(value, Mapping):
                try:
                    partition = SlurmPartitionConfig(**value)
                except TypeError as exc:
                    raise ValueError(
                        f"Invalid SLURM partition config for {name!r}: {exc}"
                    ) from exc
            else:
                raise ValueError(f"SLURM partition config for {name!r} must be a table")
            normalized[str(name)] = partition
        self.partitions = normalized
        if normalized:
            self._apply_partition(self.queue)

    @property
    def default_partition(self) -> str:
        """Return the partition selected when a run does not override it."""

        return self.queue

    @property
    def cores_per_socket(self) -> int:
        """Return the number of CPU cores in each socket."""

        return self.cores_per_node // self.sockets_per_node

    @property
    def memory_per_core(self) -> float:
        """Return memory per CPU core in megabytes."""

        return self.memory_per_node / self.cores_per_node

    def for_partition(self, partition: str) -> "SlurmSiteConfig":
        """Return this site config resolved for a partition."""

        if self.partitions and partition not in self.partitions:
            known = ", ".join(sorted(self.partitions))
            raise ValueError(
                f"Unknown SLURM partition {partition!r}. Known partitions: {known}"
            )
        return replace(self, queue=partition)

    def _apply_partition(self, partition: str) -> None:
        try:
            values = self.partitions[partition]
        except KeyError as exc:
            known = ", ".join(sorted(self.partitions))
            raise ValueError(
                f"Unknown SLURM partition {partition!r}. Known partitions: {known}"
            ) from exc
        for item in dataclass_fields(SlurmPartitionConfig):
            setattr(self, item.name, getattr(values, item.name))

    def validate_request(self, nodes: int, tasks: int, duration: Optional[str]) -> str:
        """Validate a SLURM allocation request and return a usable duration."""

        duration = duration or self.max_duration
        if nodes < self.min_nodes:
            raise ValueError(f"Minimum number of nodes is {self.min_nodes}")
        if nodes > self.max_nodes:
            raise ValueError(f"Maximum number of nodes is {self.max_nodes}")
        if tasks < nodes:
            raise ValueError("Number of tasks must be at least the number of nodes")
        if _hms_to_seconds(duration) > _hms_to_seconds(self.max_duration):
            logger.warning(
                "Requested duration %s exceeds maximum %s; using maximum",
                duration,
                self.max_duration,
            )
            return self.max_duration
        return duration


# ----------------------------------
# Generic Login Credentials
# ----------------------------------
class SlurmLoginCredentials(Credentials):
    """Generic SSH credentials for SLURM-backed HPC sites."""

    user_env: str = "HPC_USERNAME"
    pw_env: str = "HPC_PASSWORD"
    ssh_key_env: str = "SSH_PASSPHRASE"


_RANK_ALIASES = {
    "procs_per_node": "ranks_per_node",
    "procs_per_task": "ranks_per_task",
}


def _normalize_rank_aliases(values: Mapping[str, Any]) -> Dict[str, Any]:
    """Return keyword values with legacy process names normalized to ranks."""

    normalized = dict(values)
    for old, new in _RANK_ALIASES.items():
        if old not in normalized:
            continue
        old_value = normalized.pop(old)
        if old_value is None:
            continue
        new_value = normalized.get(new)
        if new_value is not None and new_value != old_value:
            raise ValueError(f"Pass either {new!r} or {old!r}, not both")
        normalized[new] = old_value
    return normalized


def _normalize_failure_tolerance(value: Any, *, default: Optional[int] = 4):
    """Normalize failure-tolerance inputs accepted by submit/run helpers."""

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


@dataclass(init=False)
class SlurmRunConfig:
    """Default resource request for SLURM job submissions.

    Args:
        queue: Queue/partition name.
        nodes: Requested node count.
        duration: Requested wall time.
        ranks_per_node: MPI ranks per node.
        ranks_per_task: MPI ranks used by each FrequenSolve task.
        mpi_async_progress: Reserved compatibility option. Enabling it is
            temporarily unsupported and raises :class:`NotImplementedError`.
        tolerate_failures: Number of failed tasks tolerated before the adaptive
            scheduler aborts. ``None`` disables early aborts.
        account: Allocation account.
        notify_on: Optional SLURM mail notification trigger.
        notify_email: Optional notification email address.
        poll_interval: Polling interval in seconds.
        scheduler_heartbeat_timeout: Maximum seconds without a new adaptive
            scheduler heartbeat before the run is reported failed. ``None``
            disables heartbeat enforcement.
        run_path: Remote run directory override.
        slurm_args: Additional raw ``sbatch`` arguments.
    """

    queue: Optional[str] = None
    nodes: int = 1
    duration: Optional[str] = None
    ranks_per_node: Optional[int] = None
    ranks_per_task: Optional[int] = None
    mpi_async_progress: bool = False
    tolerate_failures: Optional[int] = 4
    account: Optional[str] = None
    notify_on: Optional[Literal["begin", "end", "fail", "all", "none"]] = None
    notify_email: Optional[str] = None
    poll_interval: Optional[int] = None
    scheduler_heartbeat_timeout: Optional[float] = _ADAPTIVE_SCHEDULER_HEARTBEAT_TIMEOUT
    run_path: Optional[Union[str, Path]] = None
    slurm_args: List[str] = field(default_factory=list)

    def __init__(
        self,
        queue: Optional[str] = None,
        nodes: int = 1,
        duration: Optional[str] = None,
        ranks_per_node: Optional[int] = None,
        ranks_per_task: Optional[int] = None,
        mpi_async_progress: bool = False,
        tolerate_failures: Optional[int] = 4,
        account: Optional[str] = None,
        notify_on: Optional[Literal["begin", "end", "fail", "all", "none"]] = None,
        notify_email: Optional[str] = None,
        poll_interval: Optional[int] = None,
        scheduler_heartbeat_timeout: Optional[float] = (
            _ADAPTIVE_SCHEDULER_HEARTBEAT_TIMEOUT
        ),
        run_path: Optional[Union[str, Path]] = None,
        slurm_args: Optional[List[str]] = None,
        **aliases,
    ):
        values = _normalize_rank_aliases(
            {
                "ranks_per_node": ranks_per_node,
                "ranks_per_task": ranks_per_task,
                **aliases,
            }
        )
        unexpected = sorted(set(values) - {"ranks_per_node", "ranks_per_task"})
        if unexpected:
            names = ", ".join(unexpected)
            raise TypeError(f"Unexpected SlurmRunConfig option(s): {names}")

        self.queue = queue
        self.nodes = nodes
        self.duration = duration
        self.ranks_per_node = values.get("ranks_per_node")
        self.ranks_per_task = values.get("ranks_per_task")
        if not isinstance(mpi_async_progress, bool):
            raise ValueError("mpi_async_progress must be true or false")
        if mpi_async_progress:
            raise NotImplementedError(
                "mpi_async_progress=True is temporarily unavailable because it "
                "can race during MPI initialization and is not yet fully "
                "supported. Leave mpi_async_progress unset or set it to false."
            )
        self.mpi_async_progress = mpi_async_progress
        self.tolerate_failures = _normalize_failure_tolerance(
            tolerate_failures,
            default=4,
        )
        self.account = account
        self.notify_on = notify_on
        self.notify_email = notify_email
        self.poll_interval = poll_interval
        if (
            scheduler_heartbeat_timeout is not None
            and float(scheduler_heartbeat_timeout) <= 0
        ):
            raise ValueError("scheduler_heartbeat_timeout must be greater than zero")
        self.scheduler_heartbeat_timeout = (
            None
            if scheduler_heartbeat_timeout is None
            else float(scheduler_heartbeat_timeout)
        )
        self.run_path = run_path
        self.slurm_args = list(slurm_args or [])

    @property
    def procs_per_node(self) -> Optional[int]:
        """Compatibility alias for ``ranks_per_node``."""

        return self.ranks_per_node

    @procs_per_node.setter
    def procs_per_node(self, value: Optional[int]) -> None:
        self.ranks_per_node = value

    @property
    def procs_per_task(self) -> Optional[int]:
        """Compatibility alias for ``ranks_per_task``."""

        return self.ranks_per_task

    @procs_per_task.setter
    def procs_per_task(self, value: Optional[int]) -> None:
        self.ranks_per_task = value

    @classmethod
    def field_names(cls) -> set[str]:
        """Return dataclass field names accepted as run-config overrides."""

        return {item.name for item in dataclass_fields(cls)} | set(_RANK_ALIASES)

    def merged(self, **overrides) -> "SlurmRunConfig":
        """Return a copy with non-``None`` overrides applied.

        Args:
            **overrides: Field values to override.

        Returns:
            New ``SlurmRunConfig`` instance.
        """

        overrides = _normalize_rank_aliases(overrides)
        values = {
            "queue": self.queue,
            "nodes": self.nodes,
            "duration": self.duration,
            "ranks_per_node": self.ranks_per_node,
            "ranks_per_task": self.ranks_per_task,
            "mpi_async_progress": self.mpi_async_progress,
            "tolerate_failures": self.tolerate_failures,
            "account": self.account,
            "notify_on": self.notify_on,
            "notify_email": self.notify_email,
            "poll_interval": self.poll_interval,
            "scheduler_heartbeat_timeout": self.scheduler_heartbeat_timeout,
            "run_path": self.run_path,
            "slurm_args": list(self.slurm_args),
        }
        for key, value in overrides.items():
            if key in values and value is not None:
                values[key] = value
        return SlurmRunConfig(**values)

    def resolved(
        self,
        site_config: Any,
        **overrides,
    ) -> tuple["SlurmRunConfig", Dict[str, Any]]:
        """Resolve defaults against a site configuration.

        Args:
            site_config: Site configuration providing queue and poll defaults.
            **overrides: Submission keyword arguments.

        Returns:
            ``(run_config, extra_kwargs)`` where extra kwargs were not recognized
            as run-config fields.
        """

        overrides = _normalize_rank_aliases(overrides)
        run_keys = self.field_names()
        config = self.merged(**{k: v for k, v in overrides.items() if k in run_keys})
        if config.queue is None:
            config.queue = site_config.queue
        if config.poll_interval is None:
            config.poll_interval = site_config.poll_interval
        extra = {k: v for k, v in overrides.items() if k not in run_keys}
        return config, extra


# ----------------------------------
# Generic SLURM Site
# ----------------------------------
@dataclass(kw_only=True, init=False)
class SlurmSite(BaseSite):
    """Generic SLURM HPC site.

    Manages authentication, transfer, provisioning, and job execution for SLURM-backed HPC systems.

    Args:
        transfer_method: File transfer backend, either ``"rsync"`` or
            ``"sftp"``.
        default_queue: Deprecated compatibility alias for ``default_partition``.
        default_partition: Partition used when no run config supplies one.
        config: Optional site configuration object.
        credentials: Optional login credentials object.
        username: SSH username. The configured environment variable remains a
            fallback for direct-constructor compatibility.
        credential: Keyring lookup name used to keep credentials for different
            sites separate.
        ssh_key: Optional private-key path.
        solver: Remote path to the ``FS_seismic`` solver router executable.
        work_dir: Remote base directory used to resolve relative FrequenSolve
            project, simulation, and job paths. It may be on any writable
            remote filesystem and defaults to ``$WORK/frequensolve``.
        scratch_dir: Optional remote scratch directory reserved for future
            model and high-I/O storage.
        modules: Environment modules loaded before remote solver execution.
        environment: Non-secret environment values exported before remote
            solver execution.
        frequensolver_policy: Compatibility behavior: ``"warn"`` (default),
            ``"strict"``, or ``"off"``.
        run_config: Default SLURM resource request.
        config.tmp_dir: Optional remote directory for transient transfer
            tarballs and provisioning scripts. Defaults to ``/tmp``.
        verbose: Whether to print site status messages in addition to logging.
    """

    credentials: SlurmLoginCredentials
    config: Any
    run_config: SlurmRunConfig
    pool: PoolInfo
    transfer_method: Literal["rsync", "sftp"] = "rsync"
    _executable: str
    _login_client: SSHClientClass
    _compute_client: Optional[SSHClientClass] = None
    _work_dir: Path
    solver: Optional[Union[str, Path]]
    _configured_work_dir: Optional[Union[str, Path]]
    _rel_proj_path: Optional[Path]
    _scratch_dir: Optional[Path]
    modules: List[str]
    environment: Dict[str, str]
    frequensolver_policy: Optional[str]
    _frequensolver_compatibility_result: Optional[FrequenSolverCompatibility]
    _frequensolver_compatibility_policy: Optional[str]

    site_name: str = "SLURM"
    credentials_cls: Type["SlurmLoginCredentials"] = None
    config_cls: Optional[Type[Any]] = None
    default_partition: Optional[str] = None
    default_queue: Optional[str] = None
    default_host: Optional[str] = None
    default_solver_executable: Optional[str] = None

    def _job_validation_options(self, job: Any) -> Dict[str, Any]:
        """Allow absolute paths whose contents are owned by the remote site."""

        return {"allow_unverified_remote_files": True}

    def __init__(
        self,
        rel_path: Optional[Union[str, Path]] = None,
        transfer_method: Literal["rsync", "sftp"] = "rsync",
        default_queue: Optional[str] = None,
        default_partition: Optional[str] = None,
        config: Optional[Any] = None,
        credentials: Optional["SlurmLoginCredentials"] = None,
        username: Optional[str] = None,
        credential: Optional[str] = None,
        ssh_key: Optional[Union[str, Path]] = None,
        credential_store: Optional[CredentialStore] = None,
        solver: Optional[Union[str, Path]] = None,
        work_dir: Optional[Union[str, Path]] = None,
        scratch_dir: Optional[Union[str, Path]] = None,
        modules: Optional[List[str]] = None,
        environment: Optional[Mapping[str, object]] = None,
        run_config: Optional[SlurmRunConfig] = None,
        verbose: bool = False,
        frequensolver_policy: Optional[str] = None,
    ):
        if (
            default_partition is not None
            and default_queue is not None
            and default_partition != default_queue
        ):
            raise ValueError(
                "Pass either default_partition or default_queue, not conflicting values"
            )
        partition = (
            default_partition
            if default_partition is not None
            else (
                default_queue
                if default_queue is not None
                else (
                    self.default_partition
                    if self.default_partition is not None
                    else self.default_queue
                )
            )
        )
        self.verbose = verbose
        logger.debug(
            "Initializing %s with work_dir=%s, rel_path=%s, partition=%s",
            self.site_name,
            work_dir,
            rel_path,
            partition,
        )

        if config is not None:
            self.config = config
            if (
                partition is not None
                and getattr(config, "queue", partition) != partition
                and callable(getattr(config, "for_partition", None))
            ):
                self.config = config.for_partition(partition)
        elif self.config_cls is not None:
            self.config = (
                self.config_cls(queue=partition)
                if partition is not None
                else self.config_cls()
            )
        else:
            raise ValueError("SlurmSite requires either a config object or config_cls")
        if partition is None:
            partition = self.config.queue
        self.default_partition = partition
        self.default_queue = partition
        if self.credentials_cls is None:
            self.credentials_cls = SlurmLoginCredentials
        host = getattr(self.config, "hostname", None) or self.default_host
        self.credentials = credentials or self.credentials_cls(
            username=username,
            credential=credential or host or self.site_name,
            ssh_key=ssh_key,
            credential_store=credential_store,
        )
        self.solver = solver
        self._configured_work_dir = work_dir
        self._rel_proj_path = Path(rel_path) if rel_path not in {None, ""} else None
        self._scratch_dir = self._normalize_optional_remote_dir(
            scratch_dir,
            name="scratch_dir",
        )
        if isinstance(modules, str):
            raise ValueError("modules must be an array of module names")
        self.modules = [str(module) for module in (modules or [])]
        self.environment = validate_environment(environment)
        self.frequensolver_policy = frequensolver_policy
        self._frequensolver_compatibility_result = None
        self._frequensolver_compatibility_policy = None
        self.transfer_method = transfer_method
        self.run_config = run_config or SlurmRunConfig(queue=partition)
        self._authenticator = SlurmAuthenticator(self)
        self._transfer = SlurmTransferManager(self)

        self._login_client = SSHClientClass(self.authenticate())
        logger.info("SSH client authenticated successfully")

        self._work_dir = self._get_work_dir()
        self._executable = self._get_solver_path()

        self.pool = PoolInfo()
        self._is_notebook = _check_if_notebook()
        self._compute_client = None

        self._emit(f"{self.site_name} initialized with work_dir: {self._work_dir}")

    @property
    def executable(self) -> str:
        """Get the configured solver executable."""

        if self._executable is None:
            raise ValueError(
                "Solver executable not specified; configure solver in "
                "site.toml or pass solver= explicitly."
            )
        return self._executable

    @property
    def compute_client(self) -> SSHClient:
        """Get the compute client."""
        if self._compute_client is None:
            raise RuntimeError(
                "No compute client is attached; call provision/attach first"
            )
        return self._compute_client.client

    @property
    def compute_host(self) -> str:
        """Get the compute host."""
        if self._compute_client is None:
            raise RuntimeError(
                "No compute client is attached; call provision/attach first"
            )
        return self._compute_client.hostname

    @property
    def login_client(self) -> SSHClient:
        """Get the login client."""
        return self._login_client.client

    @property
    def login_host(self) -> str:
        """Get the login host."""
        return self._login_client.hostname

    @property
    def mpi_cmd(self) -> str:
        """Get the MPI launch command."""
        return f"{self.config.mpi_wrapper}"

    @property
    def pool_host(self) -> str:
        """Get the resource pool host node."""
        return self._authenticator.get_job_host(self.pool.id)

    @property
    def work_dir(self) -> Path:
        """Return the remote base directory for relative FrequenSolve paths."""
        return self._work_dir

    @property
    def scratch_dir(self) -> Optional[Path]:
        """Return the configured future-facing remote scratch directory."""

        return self._scratch_dir

    @property
    def remote_tmp_dir(self) -> Path:
        """Remote directory used for transient staging on the login host."""

        tmp_dir = getattr(self.config, "tmp_dir", None)
        if tmp_dir is None or str(tmp_dir).strip() == "":
            return Path("/tmp")
        path = Path(str(tmp_dir))
        if not path.is_absolute():
            raise ValueError("SLURM tmp_dir must be an absolute remote path")
        return path

    @property
    def provisioned(self):
        """Check if the site is provisioned."""
        self.update_status()
        return self.pool.is_running

    def __enter__(self):
        """Enter a context manager without changing site state."""

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close SSH clients when leaving a context manager."""

        self.close()
        return False

    def close(self) -> None:
        """Close active SSH clients held by the site."""

        if getattr(self, "_compute_client", None):
            self._compute_client.close()
            self._compute_client = None
        if getattr(self, "_login_client", None):
            self._login_client.close()
            self._login_client = None

    def authenticate(self, host: Optional[str] = None):
        """Connect to the login node.

        Args:
            host: Optional login host override.

        Returns:
            Paramiko SSH client or local SSH control-socket proxy.
        """

        client = self._authenticator.authenticate(host)
        persist_pending = getattr(self.credentials, "persist_pending", None)
        if callable(persist_pending):
            persist_pending()
        return client

    def submit(
        self,
        job: BaseJob,
        *,
        queue: Optional[str] = None,
        nodes: Optional[int] = None,
        ranks_per_node: Optional[int] = None,
        duration: Optional[str] = None,
        mpi_async_progress: Optional[bool] = None,
        force: bool = False,
        mode: Literal["auto", "attached", "batch"] = "auto",
        fetch: bool = False,
        check: bool = False,
        **overrides,
    ) -> RunHandle:
        """Submit a job and return an awaitable run handle.

        Args:
            job: Job to submit.
            queue: Queue or partition for this submission.
            nodes: Number of nodes for this submission.
            ranks_per_node: MPI ranks per node for this submission.
            duration: Wall time for this submission.
            mpi_async_progress: Reserved compatibility option. Passing ``True``
                raises :class:`NotImplementedError` while asynchronous MPI
                progress support is unavailable.
            force: Force a new run even when current results exist.
            mode: Submission mode: ``"auto"``, ``"attached"``, or ``"batch"``.
            fetch: Whether to fetch outputs after completion.
            check: Whether the returned handle raises by default when waited
                and the run reaches an unsuccessful terminal status.
            **overrides: Resource-request or site-specific submission
                overrides. Pass ``validate=False`` to skip SDK pre-run
                validation.

        Returns:
            ``RunHandle`` for the submitted or attached run.
        """

        overrides.update(
            {
                name: value
                for name, value in {
                    "queue": queue,
                    "nodes": nodes,
                    "ranks_per_node": ranks_per_node,
                    "duration": duration,
                    "mpi_async_progress": mpi_async_progress,
                }.items()
                if value is not None
            }
        )
        frequensolver_policy = overrides.pop(
            "frequensolver_policy", self.frequensolver_policy
        )
        fresh_run = bool(force or overrides.pop("rerun", False))
        skip_policy_value = overrides.pop("skip", overrides.pop("skip_policy", None))
        residual = overrides.pop("residual", None)
        ignore_solver_options = overrides.pop("ignore_solver_options", None)
        reuse = bool(overrides.pop("reuse", True))
        skip_policy = SkipPolicy.from_value(
            skip_policy_value,
            residual=residual,
            ignore_solver_options=ignore_solver_options,
            reuse=reuse and not fresh_run,
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
        fresh_run = bool(fresh_run or skip_policy.force)
        validate = overrides.pop("validate", True)
        self.check_frequensolver_compatibility(policy=frequensolver_policy)
        self.prepare_job(job, validate=validate)
        if mode not in {"auto", "attached", "batch"}:
            raise ValueError("mode must be 'auto', 'attached', or 'batch'")

        run_config, extra_kwargs = self.run_config.resolved(self.config, **overrides)

        if not fresh_run:
            handle = self._reattach_inflight_run(
                job,
                poll_interval=run_config.poll_interval,
                scheduler_heartbeat_timeout=run_config.scheduler_heartbeat_timeout,
                fetch=fetch,
                check=check,
            )
            if handle is not None:
                return handle

            if self.is_run_current(job):
                job.write_run_state(status="skipped")
                self._emit(f"Skipping {job.name}; run is current")
                handle = RunHandle.skipped(self, job)
                if fetch:
                    self.fetch_outputs(job)
                return handle

        self.prepare_job(job, sync_project=True, validate=False)

        active_allocation = self.provisioned if mode in {"auto", "attached"} else False
        use_attached = mode == "attached" or (mode == "auto" and active_allocation)
        if use_attached:
            if not active_allocation:
                raise RuntimeError(
                    "No active compute allocation is attached; use mode='batch' "
                    "or allow mode='auto' to submit a batch job."
                )
            pool_id = self.pool.id
            if not pool_id:
                raise RuntimeError(
                    "Active SLURM allocation is missing a valid scheduler job id"
                )
            try:
                allocation_id = _validate_slurm_job_id(pool_id)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Active SLURM allocation is missing a valid scheduler job id"
                ) from exc
            pack = bool(extra_kwargs.pop("pack", True))
            future = self._submit_attached(
                job,
                ranks_per_task=run_config.ranks_per_task or 2,
                mpi_async_progress=run_config.mpi_async_progress,
                fresh=fresh_run,
                **({"pack": pack} if not pack else {}),
            )
            job._job_id = allocation_id
            self._emit(f"Submitted {job.name} to active {self.site_name} allocation")
            self._record_site_run(job, status="running")
            handle = RunHandle(
                site=self,
                job=job,
                id=allocation_id,
                mode="attached",
                poll_interval=run_config.poll_interval,
                check=check,
                _status_fn=self._poll_attached_run,
                _wait_fn=self._wait_attached_run,
                _wait_async_fn=self._wait_attached_run_async,
                _timeout_fn=self._timeout_slurm_run,
                _generic_wait=False,
                _cancel_fn=lambda run: self.cancel_job(str(run.id)),
                _fetch_fn=(lambda run: self.fetch_outputs(run.job)) if fetch else None,
                _fetch_on_complete=fetch,
            )
            handle.backend["future"] = future
            handle.backend["mpi_async_progress"] = run_config.mpi_async_progress
            return handle

        task_plan = None
        if hasattr(job, "task_run_plan"):
            task_plan = job.task_run_plan(
                reuse=reuse and not fresh_run,
                force=fresh_run,
                skip_policy=plan_skip_policy,
            )
            pending_indices = list(task_plan["pending_indices"])
        else:
            pending_indices = list(range(int(getattr(job, "n_tasks", 0))))
        smooth_only = (
            isinstance(job, ImagingJob)
            and not pending_indices
            and self._remote_image_smoothing_needed(job)
        )
        if task_plan is not None and not pending_indices and not smooth_only:
            job.write_run_state(
                status="skipped",
                tasks=task_plan.get(
                    "skipped_task_records",
                    task_plan.get("reused_tasks", []),
                ),
            )
            self._emit(f"Skipping {job.name}; no frequency tasks need to run")
            handle = RunHandle.skipped(
                self,
                job,
                "No frequency tasks need to run",
            )
            if fetch:
                self.fetch_outputs(job)
            return handle

        job_id = self._submit_slurm_batch(
            job,
            run_config,
            fresh=fresh_run,
            **({"task_plan": task_plan} if task_plan is not None else {}),
            **(
                {"skip_policy": plan_skip_policy}
                if plan_skip_policy is not None
                else {}
            ),
            reuse=reuse,
            **({"smooth_only": smooth_only} if smooth_only else {}),
            **extra_kwargs,
        )
        handle = self.handle(job, job_id=job_id, mode="batch")
        handle.poll_interval = run_config.poll_interval or self.config.poll_interval
        handle.backend["scheduler_heartbeat_timeout"] = (
            run_config.scheduler_heartbeat_timeout
        )
        handle.backend["mpi_async_progress"] = run_config.mpi_async_progress
        handle.check = check
        handle._fetch_fn = (lambda run: self.fetch_outputs(run.job)) if fetch else None
        handle._fetch_on_complete = fetch
        if task_plan is not None:
            handle.backend["task_plan"] = task_plan
        return handle

    def check_frequensolver_compatibility(
        self,
        *,
        policy: Optional[str] = None,
        force: bool = False,
    ) -> FrequenSolverCompatibility:
        """Check the remote solver once before this site submits work."""

        selected = resolve_frequensolver_policy(
            policy if policy is not None else self.frequensolver_policy
        )
        if (
            not force
            and self._frequensolver_compatibility_result is not None
            and self._frequensolver_compatibility_policy == selected
        ):
            return self._frequensolver_compatibility_result
        result = check_frequensolver_compatibility(
            self.executable,
            policy=selected,
            remote_runner=lambda command: self.run_login(
                command,
                timeout=IDENTITY_QUERY_TIMEOUT_SECONDS,
            ),
            setup_commands=self._runtime_setup_lines(),
        )
        self._frequensolver_compatibility_result = result
        self._frequensolver_compatibility_policy = selected
        return result

    def handle(
        self, job, job_id: Optional[str] = None, mode: str = "attached"
    ) -> RunHandle:
        """Create a run handle and attach SLURM-specific wait behavior."""

        handle = super().handle(job, job_id=job_id, mode=mode)
        handle._timeout_fn = self._timeout_slurm_run
        return handle

    def _timeout_slurm_run(self, run: RunHandle, status: JobStatus) -> RunResult:
        """Cancel a still-running SLURM job before publishing a timeout result."""

        self.cancel_job(str(run.id))
        self._finalize_run_record(run, status)
        return run._make_result(status)

    def provision(
        self, nodes: int, tasks: int, duration: Optional[str] = None, **kwargs
    ) -> RunHandle:
        """Submit an interactive SLURM allocation and return a run handle.

        The returned handle completes when the allocation is running and the
        compute client is attached.
        """

        nhost = nodes
        nproc = tasks
        logger.info(
            "Provisioning SLURM job with nhost=%d, nproc=%d, duration=%s",
            nhost,
            nproc,
            duration,
        )
        duration = duration or getattr(self.config, "max_duration", "00-02:00:00")
        self._emit(
            f"Submitting {self.site_name} allocation: nodes={nhost}, "
            f"tasks={nproc}, duration={duration}"
        )
        duration = self.config.validate_request(nhost, nproc, duration)
        script = self._generate_provision_script(nhost, nproc, duration, **kwargs)

        with _temporary_text_file(
            script,
            suffix=".sh",
            prefix="slurm_",
            directory=_host_tmp_path_for_config(
                getattr(self, "_site_config_path", None)
            ),
        ) as script_path:
            logger.debug("Temporary SLURM script created at %s", script_path)
            remote_path = self.remote_tmp_dir / os.path.basename(script_path)
            try:
                self.put(script_path, remote_path)

                self.pool.id = self._submit_sbatch(
                    f"sbatch {shlex.quote(str(remote_path))}"
                )
                logger.debug("Job submitted successfully with job ID: %s", self.pool.id)
                self.pool._status.status = "pending"
                self._emit(f"Allocation submitted: {self.pool.id}")

            except Exception as e:
                logger.exception("Exception occurred during provisioning: %s", str(e))
                self.pool._status.status = "failed"
                self.pool._status.stderr = str(e)
                raise

        return self._allocation_handle(self.pool.id)

    def attach_allocation(self, job_id: Optional[str] = None) -> RunHandle:
        """Create a handle for an existing SLURM allocation.

        If job_id is not provided, queued jobs will be listed and the user will
        be prompted to select a job.
        """
        if job_id is None:
            job_id = self._select_job()
        self.pool.id = job_id
        self._emit(f"Tracking existing allocation: {self.pool.id}")
        return self._allocation_handle(str(job_id))

    def run_cmd(self, client, cmd: str, *, timeout: Optional[float] = None):
        """Run a command using the connected SSH client."""
        if client is None:
            raise RuntimeError("SSH client is not connected")
        logger.debug("Executing on %s: %s", client.hostname, cmd)
        if timeout is not None:
            return client.exec_command(cmd, timeout=timeout)
        return client.client.exec_command(cmd)

    def run_compute_cmd(self, cmd: str):
        """Run a command on compute node using exec_command."""
        return self.run_cmd(self._compute_client, cmd)

    def run_login_cmd(self, cmd: str, *, timeout: Optional[float] = None):
        """Run a command on login node using exec_command."""
        return self.run_cmd(self._login_client, cmd, timeout=timeout)

    def run_compute(self, cmd: str) -> str:
        """Run a command on compute node and return its stdout as a stripped string."""
        _, stdout, _ = self.run_compute_cmd(cmd)
        return stdout.read().decode().strip()

    def run_login(self, cmd: str, *, timeout: Optional[float] = None) -> str:
        """Run a command on login node and return its stdout as a stripped string."""
        stdout = None
        try:
            _, stdout, _ = self.run_login_cmd(cmd, timeout=timeout)
            return _read_stream(stdout)
        except TimeoutError as exc:
            channel = getattr(stdout, "channel", None)
            close_channel = getattr(channel, "close", None)
            if callable(close_channel):
                close_channel()
            suffix = f" after {timeout} seconds" if timeout is not None else ""
            raise TimeoutError(f"SSH login command timed out{suffix}") from exc

    def is_run_current(self, job: BaseJob) -> bool:
        """Return True when this site has current successful results for a job."""

        if not isinstance(job, BaseJob):
            return bool(job.is_run_current())
        record = job.latest_run(site=self.site_name)
        if record is None:
            return False
        try:
            if record.fingerprint != job.fingerprint():
                return False
        except Exception as exc:
            logger.debug("Could not fingerprint job %s: %s", job.name, exc)
            return False
        if not self._remote_run_successful(record):
            return False
        if isinstance(job, ImagingJob):
            return self._remote_image_output_exists(job)
        return True

    def _remote_file_exists(self, path: Union[str, Path]) -> bool:
        """Return whether a regular file exists on the remote login node."""

        quoted = shlex.quote(str(path))
        try:
            return self.run_login(f"test -f {quoted} && printf 1 || printf 0") == "1"
        except Exception as exc:
            logger.debug("Could not stat remote file %s: %s", path, exc)
            return False

    def _remote_image_file(self, job: ImagingJob, part: Optional[int] = None) -> Path:
        image_dir = job._remote_image_path(self.work_dir)
        if part is None:
            return image_dir / "image.h5"
        return image_dir / f"image_{part}.h5"

    def _remote_image_output_exists(self, job: ImagingJob) -> bool:
        """Return whether the remote aggregate image file exists."""

        return self._remote_file_exists(self._remote_image_file(job))

    def _remote_image_part_outputs_exist(self, job: ImagingJob) -> bool:
        """Return whether all remote per-frequency image shards exist."""

        return all(
            self._remote_file_exists(self._remote_image_file(job, part))
            for part in range(1, job.n_tasks + 1)
        )

    def _remote_image_smoothing_needed(self, job: ImagingJob) -> bool:
        """Return whether remote image shards need the final smooth/stack step."""

        return self._remote_image_part_outputs_exist(
            job
        ) and not self._remote_image_output_exists(job)

    def update_status(self, job_id: Optional[str] = None):
        """Check the status of the resource request."""

        if job_id is None:
            job_id = self.pool.id
            job_specified = False  # Checking status of pool
        else:
            job_specified = True  # Checking status of a specific job

        if not job_id:
            self.pool._status.status = "unknown"
            return "unknown"
        job_id = _validate_slurm_job_id(job_id)

        # Get job status
        queue_status = self.run_login(f"squeue -j {job_id} -h -o %t").strip()
        logger.debug("Job %s status from squeue: '%s'", job_id, queue_status)

        # If no status returned, job is not in queue - check sacct for completion status
        if not queue_status:
            sacct_cmd = f"sacct -j {job_id} -n -o State%20"
            completion_status = self.run_login(sacct_cmd).strip()
            logger.debug(
                "Job %s completion status from sacct: '%s'", job_id, completion_status
            )
            status = "unknown"
            for line in completion_status.splitlines():
                status = _normalize_slurm_state(line)
                if status != "unknown":
                    break
        else:
            status = _normalize_slurm_state(queue_status)

        if not job_specified:
            self.pool._status.status = status
        return status

    def put(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        """Transfer files from a local path to a remote path."""

        return self._transfer.put(local_path, remote_path)

    def fetch_traces(
        self,
        job: Union[BaseJob, List[BaseJob]],
        upscale: int = 1,
    ) -> Union[TraceDataset, Dict[str, TraceDataset]]:
        """Get trace results from the remote site.

        Args:
            job: A BaseJob object.
        """

        jobs, single = _as_list(job, BaseJob)

        db_map = {}

        for j in jobs:
            try:
                trace_dir_name = Path(j.trace_outputs.path).name
                remote_dir = self._remote_result_dir(j) / trace_dir_name
                local_dir = j._local_path / "results" / trace_dir_name
                local_dir.mkdir(parents=True, exist_ok=True)
                self.get(remote_dir, local_dir)

                db = TraceDataset.from_job(j, upscale)
                db_map[j.name] = db

            except Exception as e:
                logger.exception("Error downloading traces: %s", str(e))
                raise

        if single:
            return db_map[jobs[0].name]
        else:
            return db_map

    def fetch_wavefields(
        self,
        job: Union[BaseJob, List[BaseJob]],
        upscale: int = 1,
    ) -> Union[TraceDataset, Dict[str, TraceDataset]]:
        """Get wavefield trace results from the remote site."""

        jobs, single = _as_list(job, BaseJob)
        db_map = {}

        for j in jobs:
            try:
                wavefield_outputs = j.wavefield_trace_outputs
                if not wavefield_outputs.groups:
                    raise ValueError("Job has no wavefield outputs")
                wavefield_dir_name = Path(wavefield_outputs.path).name
                remote_dir = self._remote_result_dir(j) / wavefield_dir_name
                local_dir = j._local_path / "results" / wavefield_dir_name
                local_dir.mkdir(parents=True, exist_ok=True)
                self.get(remote_dir, local_dir)

                db_map[j.name] = j.wavefields.open(upscale=upscale)

            except Exception as e:
                logger.exception("Error downloading wavefields: %s", str(e))
                raise

        if single:
            return db_map[jobs[0].name]
        return db_map

    def fetch_outputs(self, job: BaseJob):
        """Fetch common result metadata and trace outputs for a completed job."""

        local_results = job._local_path / "results"
        local_results.mkdir(parents=True, exist_ok=True)

        self.fetch_run_metadata(job)
        try:
            self.get(
                self._remote_logs_dir(job),
                job._local_path / "logs",
            )
        except Exception as exc:
            logger.debug("Could not fetch logs for job %s: %s", job.name, exc)

        try:
            self.fetch_traces(job)
        except Exception as exc:
            logger.debug("Could not fetch traces for job %s: %s", job.name, exc)

        if getattr(job.outputs, "wavefields", None):
            try:
                self.fetch_wavefields(job)
            except Exception as exc:
                logger.debug(
                    "Could not fetch wavefields for job %s: %s",
                    job.name,
                    exc,
                )

        if getattr(job.outputs, "vtk", None):
            try:
                self.fetch_vtk(job)
            except Exception as exc:
                logger.debug(
                    "Could not fetch VTK outputs for job %s: %s",
                    job.name,
                    exc,
                )

        return local_results

    @staticmethod
    def _fetch_message(
        label: str,
        remote: Union[str, Path],
        local: Union[str, Path],
    ) -> str:
        return f"{label}\n\tFrom: {remote}\n\tTo: {local}"

    def fetch_run_metadata(self, job: BaseJob) -> Optional[Path]:
        """Fetch ``_fs_run`` metadata and aggregate task manifests locally."""

        remote_run_dir = self._remote_result_dir(job) / "_fs_run"
        local_run_dir = job._result_path / "_fs_run"
        try:
            self.get(remote_run_dir, local_run_dir)
        except Exception as exc:
            logger.debug("Could not fetch _fs_run for job %s: %s", job.name, exc)
            return None
        return job.collect_task_run_manifests()

    def fetch_vtk(self, job: BaseJob):
        """Get configured VTK/visualization files from the remote site.

        Args:
            job: A BaseJob object.
        """

        self.fetch_run_metadata(job)
        job.outputs.ensure_unique_names()
        seen = set()
        for output in job.outputs.vtk:
            path = Path(output.path)
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            remote_dir = self._remote_result_dir(job) / path
            local_dir = job._local_path / "results" / path
            local_dir.mkdir(parents=True, exist_ok=True)
            self._emit(
                self._fetch_message("Fetching VTK outputs", remote_dir, local_dir)
            )
            self.get(remote_dir, local_dir)
        return job.vtk_outputs

    def fetch_paraview(self, job: BaseJob):
        """Fetch visualization files using the historical method name."""

        return self.fetch_vtk(job)

    def fetch_image(
        self,
        job: Union[ImagingJob, List[ImagingJob]],
    ):
        """Get image files from the remote site.

        Args:
            job: An ImagingJob object.
        """

        jobs, single = _as_list(job, ImagingJob)
        images = {}
        for job in jobs:
            try:
                remote = job._remote_image_path(self.work_dir)
                local = job._local_image_path
                self.get(remote, local)

                images[job.name] = job.load_images()

            except Exception as e:
                logger.exception("Error retrieving payload: %s", str(e))
                raise

        if single:
            return images[jobs[0].name]
        else:
            return images

    def fetch_logs(
        self,
        job: Union[BaseJob, List[BaseJob]],
        *,
        local_dir: Optional[Union[str, Path]] = None,
        task: Optional[int] = None,
        frequency: Optional[Union[float, complex]] = None,
        include_batch: bool = False,
        show: bool = False,
    ) -> Union[Path, dict]:
        """Fetch log files from the remote site to the local machine.

        Downloads the job log directory (e.g. task_1.txt, task_2.txt, ... and
        ``batch/job_<id>.o``) from the remote job run. ``include_batch=True``
        also supports fetching batch logs created in the legacy shared
        ``jobs/batch`` directory.

        Args:
            job: A BaseJob or list of BaseJobs whose logs to fetch.
            local_dir: Optional local directory to write logs into. If None,
                logs are written to job._local_path / "logs" for each job.
            task: Optional one-based frequency task number. When provided,
                returns that task's log file instead of the log directory.
            frequency: Optional physical frequency. When provided, selects the
                matching frequency task and returns that task's log file.
            include_batch: If True, ensure the current job's SLURM batch logs
                are fetched into ``logs/batch``. Falls back to the legacy
                shared ``run_path/jobs/batch`` location when needed (requires
                ``job._job_id``).
            show: If True, print log contents after fetching. When
                include_batch is True, prints in order: batch .o file, batch
                .e file, then the selected or latest task log file. Otherwise
                prints only the selected or latest task log file.

        Returns:
            If a single job: Path to the local log directory or selected log
            file. If a list of jobs: dict mapping job name to Path.
        """
        jobs, single = _as_list(job, BaseJob)
        requested_local_dir = Path(local_dir) if local_dir is not None else None
        result = {}

        for j in jobs:
            if requested_local_dir is None:
                log_dir = j._local_path / "logs"
            elif single:
                log_dir = requested_local_dir
            else:
                log_dir = requested_local_dir / j.name
            remote_run_dir = self._remote_result_dir(j) / "_fs_run"
            try:
                local_run_dir = j._result_path / "_fs_run"
                self._emit(
                    self._fetch_message(
                        "Fetching run metadata",
                        remote_run_dir,
                        local_run_dir,
                    )
                )
                self.fetch_run_metadata(j)
            except Exception as e:
                logger.debug(
                    "Could not aggregate _fs_run metadata for job %s from %s: %s",
                    j.name,
                    remote_run_dir,
                    e,
                )
            remote_log_dir = self._remote_logs_dir(j)
            try:
                self._emit(
                    self._fetch_message("Fetching logs", remote_log_dir, log_dir)
                )
                self.get(remote_log_dir, log_dir)
                selected_log = self._select_log_path(
                    j,
                    log_dir,
                    task=task,
                    frequency=frequency,
                )
                result[j.name] = selected_log

                if include_batch and getattr(j, "_job_id", None):
                    batch_remote = remote_log_dir / "batch"
                    legacy_batch_remote = self.work_dir / "jobs" / "batch"
                    batch_local = log_dir / "batch"
                    self._emit(
                        self._fetch_message(
                            "Fetching batch logs",
                            batch_remote,
                            batch_local,
                        )
                    )
                    for suffix in (".o", ".e"):
                        remote_batch = batch_remote / f"job_{j._job_id}{suffix}"
                        local_batch = batch_local / f"job_{j._job_id}{suffix}"
                        if local_batch.exists():
                            continue
                        for candidate in (
                            remote_batch,
                            legacy_batch_remote / remote_batch.name,
                        ):
                            try:
                                self.get(candidate, local_batch)
                                break
                            except Exception as e:
                                logger.debug(
                                    "Could not fetch batch log %s: %s",
                                    candidate,
                                    e,
                                )
                elif include_batch:
                    logger.warning(
                        "Job %s has no _job_id; skipping SLURM batch logs",
                        j.name,
                    )

                if show:
                    if include_batch:
                        self._print_batch_logs(
                            job_name=j.name,
                            log_dir=log_dir,
                            job_id=getattr(j, "_job_id", None),
                        )
                    self._show_logs(selected_log, job_name=j.name)
            except Exception as e:
                logger.exception("Error fetching logs for job %s: %s", j.name, e)
                raise
        if single:
            return result[jobs[0].name]
        return result

    def get(
        self,
        remote_path: Union[str, Path],
        local_path: Union[str, Path],
        overwrite: bool = False,
    ):
        """Transfer files from a remote path to a local path."""

        return self._transfer.get(remote_path, local_path, overwrite=overwrite)

    def cancel_job(self, job_id: Optional[str] = None) -> bool:
        """Cancel a job."""
        if job_id is None:
            job_id = self._select_job()
        if not job_id:
            logger.debug("No SLURM job id supplied; skipping cancellation")
            return False
        job_id = _validate_slurm_job_id(job_id)
        cancelled_ids: set[str] = getattr(self, "_cancelled_job_ids", set())
        if job_id in cancelled_ids:
            logger.debug("SLURM job %s was already cancelled by this site", job_id)
            return False
        _, stdout, stderr = self.run_login_cmd(f"scancel {job_id}")
        error_output = _read_stream(stderr)
        exit_status = _ssh_exit_status(stdout, stderr)
        if error_output or exit_status not in {None, 0}:
            raise RuntimeError("SLURM cancellation failed")
        cancelled_ids.add(job_id)
        self._cancelled_job_ids = cancelled_ids
        logger.info("SLURM job %s cancellation requested", job_id)
        return True

    def deprovision(self, **kwargs):
        """Release HPC resources."""
        return self.cancel_job(self.pool.id)

    def sync(self, project):
        """Sync the project to the site."""
        self._sync_project(project)

    def config_for_partition(self, partition: str):
        """Return the site configuration for a SLURM partition name."""

        if self.config_cls is not None:
            return self.config_cls(partition)
        resolver = getattr(self.config, "for_partition", None)
        if callable(resolver):
            return resolver(partition)
        return self.config

    def config_for_queue(self, queue: str):
        """Compatibility alias for :meth:`config_for_partition`."""

        return self.config_for_partition(queue)

    def _reattach_inflight_run(
        self,
        job: BaseJob,
        *,
        poll_interval: Optional[float] = None,
        scheduler_heartbeat_timeout: Optional[float] = (
            _ADAPTIVE_SCHEDULER_HEARTBEAT_TIMEOUT
        ),
        fetch: bool = False,
        check: bool = False,
    ) -> Optional[RunHandle]:
        """Return a handle for a matching active scheduler job, if one exists."""

        record_status = self._matching_inflight_run_record(job)
        if record_status is None:
            return None
        record, scheduler_status = record_status
        job._job_id = record.scheduler_id
        updated = record.with_updates(
            status=scheduler_status,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        job.write_run_record(updated)
        self._emit(
            f"Reattached {job.name} to running {self.site_name} job "
            f"{record.scheduler_id}"
        )
        handle = self.handle(job, job_id=record.scheduler_id, mode="batch")
        handle.poll_interval = poll_interval or self.config.poll_interval
        handle.backend["scheduler_heartbeat_timeout"] = scheduler_heartbeat_timeout
        handle.check = check
        handle._fetch_fn = (lambda run: self.fetch_outputs(run.job)) if fetch else None
        handle._fetch_on_complete = fetch
        handle.backend["reattached"] = True
        return handle

    def _matching_inflight_run_record(self, job: BaseJob):
        if not isinstance(job, BaseJob):
            return None
        try:
            current_payload = job.fingerprint_payload()
            current_fingerprint = job._hash_payload(current_payload)
        except Exception as exc:
            logger.debug("Could not fingerprint job %s: %s", job.name, exc)
            return None

        records = [
            record
            for record in job.run_records()
            if record.site == self.site_name and record.scheduler_id is not None
        ]
        records = sorted(
            records,
            key=lambda record: record.updated_at or record.submitted_at or "",
            reverse=True,
        )
        for record in records:
            if not self._run_record_matches_fingerprint(
                record,
                fingerprint=current_fingerprint,
                fingerprint_payload=current_payload,
            ):
                continue
            scheduler_status = self.update_status(record.scheduler_id)
            if scheduler_status in {"pending", "running"}:
                return record, scheduler_status
        return None

    @staticmethod
    def _run_record_matches_fingerprint(
        record,
        *,
        fingerprint: str,
        fingerprint_payload: Dict[str, Any],
    ) -> bool:
        if record.fingerprint != fingerprint:
            return False
        record_payload = record.fingerprint_payload or {}
        if not record_payload:
            return True
        return SlurmSite._fingerprint_section_matches(
            record_payload, fingerprint_payload, "simulation"
        ) and SlurmSite._fingerprint_section_matches(
            record_payload, fingerprint_payload, "job"
        )

    @staticmethod
    def _fingerprint_section_matches(
        record_payload: Dict[str, Any],
        fingerprint_payload: Dict[str, Any],
        section: str,
    ) -> bool:
        if section not in record_payload or section not in fingerprint_payload:
            return False
        return BaseJob._hash_payload(record_payload[section]) == BaseJob._hash_payload(
            fingerprint_payload[section]
        )

    def _remote_run_successful(self, record) -> bool:
        if not self._record_status_successful(record.status):
            return False

        manifest = self._read_remote_json(
            record.result_dir / "_fs_run" / "run_manifest.json"
        )
        if not isinstance(manifest, dict):
            return False

        task_summary = manifest.get("task_summary")
        if not isinstance(task_summary, dict):
            return False
        try:
            failed = int(task_summary.get("failed") or 0)
            complete = int(task_summary.get("complete") or 0)
            total = int(task_summary.get("total") or 0)
        except (TypeError, ValueError):
            return False
        expected_total = self._record_expected_task_count(record)
        if expected_total is not None and total != expected_total:
            return False
        if failed != 0 or total <= 0 or complete != total:
            return False
        succeeded = task_summary.get("succeeded", task_summary.get("successful"))
        if succeeded is not None:
            try:
                if int(succeeded) != total:
                    return False
            except (TypeError, ValueError):
                return False
        return self._remote_path_has_files(record.logs_dir)

    @staticmethod
    def _record_status_successful(status: Optional[str]) -> bool:
        return str(status or "").lower() in {
            "complete",
            "completed",
            "success",
            "successful",
            "succeeded",
        }

    @staticmethod
    def _record_expected_task_count(record) -> Optional[int]:
        payload = record.fingerprint_payload or {}
        job_payload = payload.get("job") if isinstance(payload, Mapping) else None
        if not isinstance(job_payload, Mapping):
            return None
        frequencies = job_payload.get("f_list")
        if frequencies is None:
            return None
        try:
            return len(frequencies)
        except TypeError:
            return None

    def _remote_path_has_files(self, path: Union[str, Path]) -> bool:
        quoted = shlex.quote(str(path))
        try:
            text = self.run_login(
                f"find {quoted} -type f -print -quit 2>/dev/null || true"
            ).strip()
        except Exception as exc:
            logger.debug("Could not inspect remote path %s: %s", path, exc)
            return False
        return bool(text)

    def _read_remote_json(self, path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        quoted = shlex.quote(str(path))
        try:
            text = self.run_login(f"test -f {quoted} && cat {quoted} || true").strip()
        except Exception as exc:
            logger.debug("Could not read remote JSON %s: %s", path, exc)
            return None
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.debug("Could not parse remote JSON %s: %s", path, exc)
            return None
        return payload if isinstance(payload, dict) else None

    def _record_site_run(
        self,
        job: BaseJob,
        *,
        scheduler_id: Optional[str] = None,
        status: str = "submitted",
    ):
        if not isinstance(job, BaseJob):
            return None
        metadata = {}
        config_path = getattr(self, "_site_config_path", None)
        profile = getattr(self, "_site_profile", None)
        if config_path is not None and profile is not None:
            metadata = {
                "site_config_path": str(config_path),
                "site_profile": str(profile),
            }
        record = job.record_site_run(
            site=self.site_name,
            work_dir=self.work_dir,
            scheduler_id=scheduler_id,
            status=status,
            site_module=self.__class__.__module__,
            site_class=self.__class__.__name__,
            rel_path=self._rel_proj_path,
            metadata=metadata,
        )
        self._store_remote_run_records(job, record)
        return record

    def _store_remote_run_records(self, job: BaseJob, record=None) -> None:
        record = record or job.latest_run(site=self.site_name)
        if record is None:
            return
        try:
            self.put(job.run_records_file, record.result_dir / "_fs_run" / "runs.json")
        except Exception as exc:
            logger.debug(
                "Could not write remote run record for job %s: %s",
                job.name,
                exc,
            )

    def _finalize_run_record(self, run: RunHandle, status: JobStatus) -> None:
        job = getattr(run, "job", None)
        if not isinstance(job, BaseJob):
            return
        record = job.latest_run(site=self.site_name)
        if record is None:
            return
        updated = record.with_updates(
            status=status.state,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        job.write_run_record(updated)
        self._store_remote_run_records(job, updated)

    def _remote_job_dir(self, job: BaseJob) -> Path:
        record = (
            job.latest_run(site=self.site_name) if isinstance(job, BaseJob) else None
        )
        return record.job_dir if record is not None else job._remote_path(self.work_dir)

    def _remote_result_dir(self, job: BaseJob) -> Path:
        record = (
            job.latest_run(site=self.site_name) if isinstance(job, BaseJob) else None
        )
        if record is not None:
            return record.result_dir
        return job._remote_path(self.work_dir) / "results"

    def _remote_logs_dir(self, job: BaseJob) -> Path:
        record = (
            job.latest_run(site=self.site_name) if isinstance(job, BaseJob) else None
        )
        if record is not None:
            return record.logs_dir
        return job._remote_path(self.work_dir) / "logs"

    def _poll_run(self, run: RunHandle) -> JobStatus:
        status = self.update_status(str(run.id))
        scheduler_status = (
            None if status == "pending" else self._read_scheduler_status(run)
        )
        return_code = (
            0
            if status == "complete"
            else (1 if status in {"failed", "cancelled", "timeout"} else -1)
        )
        message = ""
        raw: Dict[str, Any] = {"scheduler": "slurm"}
        if scheduler_status is not None:
            scheduler_failure = self._adaptive_scheduler_failure(
                run,
                scheduler_status,
                slurm_status=status,
            )
            scheduler_status = _merge_task_status_with_plan(
                scheduler_status,
                run.backend.get("task_plan"),
                job=run.job,
            )
            raw["task_status"] = scheduler_status
            message = self._format_scheduler_status(scheduler_status)
            if scheduler_failure is not None:
                status = "failed"
                return_code = 1
                raw["scheduler_liveness"] = scheduler_failure["raw"]
                message = f"{scheduler_failure['message']}; {message}"
            elif status == "unknown" and self._adaptive_scheduler_complete(
                scheduler_status
            ):
                status = "complete"
                return_code = 0
                raw["scheduler_terminal_fallback"] = {
                    "state": "complete",
                    "reason": "slurm-accounting-unavailable",
                }
        job_status = JobStatus(
            state=status,
            return_code=return_code,
            job_id=str(run.id),
            message=message,
            raw=raw,
        )
        return job_status

    @staticmethod
    def _adaptive_scheduler_failure(
        run: RunHandle,
        payload: Dict[str, Any],
        *,
        slurm_status: str,
    ) -> Optional[Dict[str, Any]]:
        """Return failure details for a dead adaptive scheduler heartbeat."""

        scheduler_state = str(payload.get("state") or "").strip().lower()
        if scheduler_state in {"failed", "cancelled", "canceled", "timeout"}:
            reason = payload.get("abort_reason")
            detail = f": {reason}" if reason else ""
            return {
                "message": f"Adaptive scheduler reported {scheduler_state}{detail}",
                "raw": {
                    "state": scheduler_state,
                    "stale": False,
                },
            }
        if slurm_status != "running":
            return None
        if scheduler_state != "running":
            return None

        heartbeat = payload.get("updated_at")
        if not heartbeat:
            return None

        timeout = run.backend.get(
            "scheduler_heartbeat_timeout",
            _ADAPTIVE_SCHEDULER_HEARTBEAT_TIMEOUT,
        )
        if timeout is None:
            return None
        timeout = float(timeout)
        now = time.monotonic()
        previous = run.backend.get("_adaptive_scheduler_heartbeat")
        if previous != heartbeat:
            run.backend["_adaptive_scheduler_heartbeat"] = heartbeat
            run.backend["_adaptive_scheduler_heartbeat_seen_at"] = now
            return None

        last_seen = float(
            run.backend.setdefault("_adaptive_scheduler_heartbeat_seen_at", now)
        )
        age = max(0.0, now - last_seen)
        if age < timeout:
            return None

        return {
            "message": (
                "Adaptive scheduler heartbeat stopped advancing for "
                f"{age:.1f} seconds (timeout {timeout:.1f} seconds) while the "
                f"SLURM job {run.id} is still running"
            ),
            "raw": {
                "state": scheduler_state,
                "heartbeat": heartbeat,
                "age_seconds": age,
                "timeout_seconds": timeout,
                "stale": True,
            },
        }

    @staticmethod
    def _adaptive_scheduler_complete(payload: Mapping[str, Any]) -> bool:
        """Confirm terminal success when SLURM accounting is unavailable."""

        if str(payload.get("state") or "").strip().lower() != "complete":
            return False
        try:
            total = int(payload.get("total") or 0)
            successful = int(payload.get("successful") or payload.get("succeeded") or 0)
            failed = int(payload.get("failed") or 0)
            running = int(payload.get("running") or 0)
            pending = int(payload.get("pending") or 0)
        except (TypeError, ValueError):
            return False
        return (
            total >= 0
            and successful >= total
            and failed == 0
            and running == 0
            and pending == 0
        )

    def _scheduler_status_path(self, job: BaseJob) -> Path:
        """Return the remote scheduler progress file for a batch simulation job."""

        return self._remote_logs_dir(job) / "scheduler_status.json"

    def _read_scheduler_status(self, run: RunHandle) -> Optional[Dict[str, Any]]:
        """Read the remote adaptive scheduler status payload if it exists."""

        job = getattr(run, "job", None)
        if job is None:
            return None
        status_path = self._scheduler_status_path(job)
        cmd = f"test -f {shlex.quote(str(status_path))} && cat {shlex.quote(str(status_path))}"
        try:
            text = self.run_login(cmd).strip()
        except Exception as exc:
            logger.debug("Could not read scheduler status %s: %s", status_path, exc)
            return None
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.debug("Could not parse scheduler status %s: %s", status_path, exc)
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _format_scheduler_status(payload: Dict[str, Any]) -> str:
        """Format task counts from an adaptive scheduler status payload."""

        total = int(payload.get("total") or 0)
        successful = int(payload.get("successful") or payload.get("succeeded") or 0)
        failed = int(payload.get("failed") or 0)
        running = int(payload.get("running") or 0)
        pending = int(
            payload.get("pending") or max(0, total - successful - failed - running)
        )
        return (
            "tasks: "
            f"{successful} successful, {failed} failed, {running} running, "
            f"{pending} pending, {total} total"
        )

    def _allocation_handle(self, job_id: str) -> RunHandle:
        return RunHandle(
            site=self,
            job=None,
            id=str(job_id),
            mode="allocation",
            poll_interval=self.config.poll_interval,
            _status_fn=self._poll_allocation,
            _wait_fn=self._wait_allocation,
            _wait_async_fn=self._wait_allocation_async,
            _finalize_fn=self._finalize_allocation,
            _timeout_fn=self._timeout_slurm_run,
            _cancel_fn=lambda run: self.cancel_job(str(run.id)),
        )

    def _poll_allocation(self, run: RunHandle) -> JobStatus:
        status = self.update_status(str(run.id))
        if status == "running":
            return JobStatus(
                state="completed",
                return_code=0,
                job_id=str(run.id),
                message="Allocation is running",
                raw={"scheduler_state": status},
            )
        return_code = (
            1 if status in {"failed", "cancelled", "timeout", "complete"} else -1
        )
        return JobStatus(
            state=status,
            return_code=return_code,
            job_id=str(run.id),
            raw={"scheduler_state": status},
        )

    def _finalize_allocation(self, run: RunHandle, status: JobStatus) -> RunResult:
        if status.is_successful:
            self._attach_compute_client()
        return run._make_result(status)

    def _wait_allocation(
        self,
        run: RunHandle,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        interval = self.config.poll_interval if poll_interval is None else poll_interval
        start = time.monotonic()
        last_state = object()
        while True:
            status = self._poll_allocation(run)
            if status.state != last_state:
                self._emit_status(status)
                last_state = status.state
            if status.is_successful:
                self._attach_compute_client()
                return run._make_result(status)
            if status.is_complete:
                return run._make_result(status)
            if timeout is not None and time.monotonic() - start > timeout:
                return run._make_result(
                    JobStatus(
                        state="timeout",
                        job_id=str(run.id),
                        message=f"Timed out waiting for allocation after {timeout} seconds",
                    )
                )
            time.sleep(interval)

    async def _wait_allocation_async(
        self,
        run: RunHandle,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        interval = self.config.poll_interval if poll_interval is None else poll_interval
        start = time.monotonic()
        last_state = object()
        while True:
            status = await asyncio.to_thread(self._poll_allocation, run)
            if status.state != last_state:
                self._emit_status(status)
                last_state = status.state
            if status.is_successful:
                await asyncio.to_thread(self._attach_compute_client)
                return run._make_result(status)
            if status.is_complete:
                return run._make_result(status)
            if timeout is not None and time.monotonic() - start > timeout:
                return run._make_result(
                    JobStatus(
                        state="timeout",
                        job_id=str(run.id),
                        message=f"Timed out waiting for allocation after {timeout} seconds",
                    )
                )
            await asyncio.sleep(interval)

    def _sync_project(self, project):
        """Sync the project to the site."""
        project._transfer(self)

    def _sync_result(self, result):
        """Sync a result with the site."""
        raise NotImplementedError(
            f"Syncing results is not implemented for {self.site_name}"
        )

    def _sync_simulation(self, simulation):
        """Sync the simulation to the site."""
        raise NotImplementedError(
            f"Syncing simulations is not implemented for {self.site_name}"
        )

    def _submit_slurm_batch(
        self,
        job: BaseJob,
        config: SlurmRunConfig,
        *,
        task_plan: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> str:
        duration = config.duration or getattr(
            self.config, "max_duration", "00-02:00:00"
        )
        skip_policy = kwargs.pop("skip_policy", kwargs.pop("skip", None))
        residual = kwargs.pop("residual", None)
        ignore_solver_options = kwargs.pop("ignore_solver_options", None)
        reuse = bool(kwargs.pop("reuse", True))
        if task_plan is None:
            fresh = bool(kwargs.get("fresh", False))
            task_plan = job.task_run_plan(
                reuse=reuse and not fresh,
                force=fresh,
                skip_policy=skip_policy,
                residual=residual,
                ignore_solver_options=ignore_solver_options,
            )
        task_indices = [int(index) + 1 for index in task_plan["pending_indices"]]
        kwargs.setdefault("skip_sizing", len(task_indices) == 1)
        run_path = self._remote_run_path(config.run_path, job=job)
        script = self._sweep_SLURM_script(
            n_tasks=len(task_indices),
            n_job_tasks=job.n_tasks,
            task_indices=task_indices,
            n_nodes=config.nodes,
            stdout=str(job._remote_path(self.work_dir) / "logs"),
            duration=duration,
            imaging_job=isinstance(job, ImagingJob),
            mpi_async_progress=config.mpi_async_progress,
            **(
                {"ranks_per_task": config.ranks_per_task}
                if config.ranks_per_task is not None
                else {}
            ),
            **(
                {"ranks_per_node": config.ranks_per_node}
                if config.ranks_per_node is not None
                else {}
            ),
            **(
                {"tolerate_failures": config.tolerate_failures}
                if config.tolerate_failures is not None
                else {}
            ),
            **({"queue": config.queue} if config.queue is not None else {}),
            **({"account": config.account} if config.account is not None else {}),
            **({"notify_on": config.notify_on} if config.notify_on is not None else {}),
            **(
                {"notify_email": config.notify_email}
                if config.notify_email is not None
                else {}
            ),
            run_path=run_path,
            **kwargs,
        )

        remote_script, remote_job = self._transfer_SLURM_job(script, job)

        logs_path = job._remote_path(self.work_dir) / "logs"
        cmd = f"mkdir -p {logs_path}/batch && "
        cmd += f"rm -f {logs_path}/scheduler_status.json && "
        cmd += "sbatch "
        if config.slurm_args:
            for arg in config.slurm_args:
                cmd += f"{arg} "
        cmd += f"{remote_script} {remote_job}"

        job_id = self._submit_sbatch(cmd)

        logger.info(
            "Job %s submitted successfully to %s:%s",
            job_id,
            self.site_name,
            config.queue or self.config.queue,
        )
        self._emit(
            f"Submitted {job.name} to {self.site_name}:{config.queue or self.config.queue} "
            f"as job {job_id}"
        )
        job._job_id = job_id
        self._record_site_run(job, scheduler_id=job_id, status="submitted")
        return job_id

    def _poll_attached_run(self, run: RunHandle) -> JobStatus:
        future = run.backend.get("future")
        if future is None:
            return JobStatus(state="unknown", job_id=run.id)
        if future.done():
            if future.exception() is not None:
                return JobStatus(
                    state="failed",
                    return_code=1,
                    job_id=run.id,
                    message=str(future.exception()),
                )
            return JobStatus(state="completed", return_code=0, job_id=run.id)
        return JobStatus(state="running", job_id=run.id)

    def _wait_attached_run(
        self,
        run: RunHandle,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        future = run.backend.get("future")
        if future is None:
            status = JobStatus(state="unknown", job_id=run.id)
            return RunResult(job=run.job, status=status, site=self)

        if self._is_notebook:
            import nest_asyncio

            nest_asyncio.apply()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._get_or_create_event_loop()
        else:
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call run.wait() for an attached SLURM run while an "
                    "asyncio loop is already running; use 'await run' instead."
                )
        if timeout is not None:
            try:
                loop.run_until_complete(asyncio.wait_for(future, timeout=timeout))
            except asyncio.TimeoutError:
                if not future.cancelled():
                    raise
                status = JobStatus(
                    state="timeout",
                    job_id=str(run.id),
                    message=f"Timed out waiting for run after {timeout} seconds",
                )
                return self._timeout_slurm_run(run, status)
        else:
            loop.run_until_complete(future)
        status = self._poll_attached_run(run)
        self._emit_status(status)
        return run._make_result(status)

    async def _wait_attached_run_async(
        self,
        run: RunHandle,
        timeout: Optional[float] = None,
        poll_interval: Optional[float] = None,
    ) -> RunResult:
        future = run.backend.get("future")
        if future is None:
            status = JobStatus(state="unknown", job_id=run.id)
            return RunResult(job=run.job, status=status, site=self)

        if timeout is not None:
            try:
                await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                if not future.cancelled():
                    raise
                status = JobStatus(
                    state="timeout",
                    job_id=str(run.id),
                    message=f"Timed out waiting for run after {timeout} seconds",
                )
                return self._timeout_slurm_run(run, status)
        else:
            await future
        status = self._poll_attached_run(run)
        self._emit_status(status)
        return run._make_result(status)

    def _submit_attached(
        self,
        job: BaseJob,
        ranks_per_task: int = 2,
        *,
        pack: bool = True,
        fresh: bool = False,
        mpi_async_progress: bool = False,
        **aliases,
    ) -> Future:
        """Submit a job into an already attached compute allocation."""

        alias_values = _normalize_rank_aliases(
            {"ranks_per_task": ranks_per_task, **aliases}
        )
        ranks_per_task = int(alias_values.get("ranks_per_task") or 2)

        loop = self._get_or_create_event_loop()
        future = loop.create_future()
        if self._compute_client is None:
            self._attach_compute_client()

        remote_script, remote_job = self._transfer_job(
            job,
            pack=pack,
            fresh=fresh,
            mpi_async_progress=mpi_async_progress,
        )
        ntasks_per_item = max(ranks_per_task, self.pool.nproc // job.n_tasks)

        if self._compute_client.is_proxy():
            interactive = self.compute_client.invoke_shell()
            cmd = (
                f"cd {self.work_dir} && "
                f"{remote_script} {remote_job} {ntasks_per_item}\n"
            )
            interactive.stdin.write(cmd.encode())
            interactive.stdin.flush()
            monitor = self._monitor_command_output(future, job, interactive)
        else:
            cmd = (
                f"cd {self.work_dir} && {remote_script} {remote_job} {ntasks_per_item}"
            )
            interactive = self.login_client.invoke_shell()
            interactive.send(f"ssh {self.compute_host}\n")
            time.sleep(1)
            interactive.send(cmd)
            monitor = self._monitor_command_output(future, job, interactive)

        loop.create_task(monitor)
        return future

    @staticmethod
    def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
        """Return the current event loop, creating one for sync contexts if needed."""

        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _template_env(self, *, keep_trailing_newline: bool = False) -> Environment:
        """Return the Jinja environment used for remote execution scripts."""

        return Environment(
            loader=PackageLoader("frequensolve.orchestrator.sites.hpc", "templates"),
            keep_trailing_newline=keep_trailing_newline,
        )

    def _render_template(
        self,
        template_name: str,
        *,
        keep_trailing_newline: bool = False,
        **context,
    ) -> str:
        """Render a scheduler template from the FrequenSolve template directory."""

        return (
            self._template_env(keep_trailing_newline=keep_trailing_newline)
            .get_template(template_name)
            .render(**context)
        )

    def _runtime_setup_lines(self) -> List[str]:
        """Return safely quoted module and environment setup commands."""

        lines = [f"module load {shlex.quote(module)}" for module in self.modules]
        if self.modules:
            lines.append("module list")
        runtime_environment = {
            **_HPC_RUNTIME_DEFAULTS,
            **self.environment,
        }
        lines.extend(
            f"export {name}={_quote_runtime_environment_value(value)}"
            for name, value in runtime_environment.items()
        )
        return lines

    def _attach_compute_client(self):
        """Connect to the current pool's compute host and populate pool metadata."""

        compute_client = self._authenticator.connect_to_job_host(self.pool.id)
        self._compute_client = SSHClientClass(compute_client)
        self._set_pool_info()

    def _submit_sbatch(self, cmd: str) -> str:
        """Run an sbatch command and return the submitted job id."""

        _, stdout, stderr = self.run_login_cmd(cmd)
        output = _read_stream(stdout)
        err = _read_stream(stderr)
        logger.debug("sbatch output: %s", output)
        try:
            job_id = _parse_sbatch_job_id(output)
        except ValueError:
            if err:
                logger.error("sbatch error: %s", err)
            raise
        if err:
            logger.debug("sbatch stderr: %s", err)
        return job_id

    def _get_solver_path(self) -> str:
        """Get the configured solver executable path on the remote system."""
        executable = self.solver
        if executable is None or executable == "":
            executable = self.default_solver_executable
        if executable is None or executable == "":
            raise ValueError(
                "Solver executable not specified; configure solver in "
                "site.toml or pass solver= explicitly."
            )
        return executable

    def _set_pool_info(self):
        """Get information about the pool."""
        logger.debug("Getting pool info for job %s", self.pool.id)

        # Get SLURM job details using scontrol
        stdout = self.run_login(f"scontrol show job {self.pool.id}")
        entries = stdout.split()

        deets = {}
        for entry in entries:
            try:
                key, value = entry.split("=")
                deets[key] = value
            except ValueError:
                continue

        host = deets["BatchHost"]
        nproc = int(deets["NumTasks"])
        nhost = int(deets["NumNodes"])
        ncore = int(deets["NumCPUs"])
        start_time = deets["StartTime"]
        end_time = deets["EndTime"]
        time_limit = deets["TimeLimit"]
        run_time = deets["RunTime"]
        seconds = _hms_to_seconds(time_limit) - _hms_to_seconds(run_time)
        _seconds_to_hms(seconds)

        logger.info(
            "Pool info - host: %s, nodes: %d, tasks: %d, cores: %d, start: %s, end: %s",
            host,
            nhost,
            nproc,
            ncore,
            start_time,
            end_time,
        )

        self.pool.hostnode = host
        self.pool.nhost = nhost
        self.pool.nproc = nproc
        self.pool.ncore = ncore
        self.pool.start_time = start_time
        self.pool.end_time = end_time
        logger.info("Current status of pool %s: %s", self.pool.id, self.pool.status)

    def _list_jobs(self):
        """List all queued jobs."""
        jobs = self.run_login(
            f'squeue -u {self.credentials.username} -h --format="%.10i %.10B %.5D %.4t %.10L"'
        )
        return jobs

    def _select_job(self):
        """Select a queued job when exactly one is available.

        Interactive prompts are intentionally avoided in the FrequenSolve Python API core. Callers
        should pass a job id explicitly when more than one allocation exists.
        """
        jobs = self._list_jobs()

        job_lines = [line for line in jobs.splitlines() if line.strip()]
        if not job_lines:
            raise RuntimeError("No jobs found")
        if len(job_lines) == 1:
            return job_lines[0].split()[0]
        job_ids = ", ".join(line.split()[0] for line in job_lines)
        raise RuntimeError(
            "Multiple SLURM jobs are available; pass job_id explicitly. "
            f"Available job ids: {job_ids}"
        )

    def _transfer_SLURM_job(self, script: str, job: BaseJob):
        """Transfer a SLURM job to the remote site."""
        remote_script = (self.work_dir / "sweep").with_suffix(".slurm")
        remote_runner = self._adaptive_scheduler_remote_path()
        with _temporary_text_file(
            script,
            suffix=".slurm",
            prefix="sweep",
            directory=_host_tmp_path_for_config(
                getattr(self, "_site_config_path", None)
            ),
        ) as script_path:
            logger.debug("Temporary sweep script created at %s", script_path)
            self.put(script_path, remote_script)

        local_job, remote_job = job.save_for_remote(
            self.__class__.__name__, self.work_dir
        )

        self._transfer_remote_simulation_inputs(job)
        logger.debug("Transferring job file to remote path: %s", remote_job)
        self.put(Path(local_job), Path(remote_job))
        runner_resource = (
            files("frequensolve.orchestrator.sites.hpc")
            .joinpath("templates")
            .joinpath("sweep")
            .joinpath("adaptive_scheduler.py")
        )
        with as_file(runner_resource) as local_runner:
            self.put(local_runner, remote_runner)
        self.run_login(
            f"chmod 700 {shlex.quote(str(remote_script))} "
            f"{shlex.quote(str(remote_runner))}"
        )

        return remote_script, remote_job

    def _adaptive_scheduler_remote_path(self) -> Path:
        """Return the remote path for the transferred adaptive scheduler."""

        return self.work_dir / "adaptive_scheduler.py"

    def _remote_run_path(
        self,
        run_path: Optional[Union[str, Path]],
        *,
        job: Optional[BaseJob] = None,
    ) -> Path:
        """Return the remote directory where SLURM scripts should run."""

        if run_path is None:
            return self.work_dir

        path = Path(run_path)
        if job is None:
            return path

        try:
            local_project = Path(job.project_path).resolve()
            relative = path.expanduser().resolve().relative_to(local_project)
        except Exception:
            return path
        return self.work_dir / relative

    def _transfer_job(
        self,
        job: BaseJob,
        *,
        pack: bool = True,
        fresh: bool = False,
        mpi_async_progress: bool = False,
    ):
        """Submit a simulation job to the remote site.

        Args:
            job (BaseJob): The simulation job to submit
        """
        if self._compute_client is None:
            raise NotImplementedError("Batch sweep job not implemented yet.")

        # Note: job must be saved for remote **before** script is generated
        local_job, remote_job = job.save_for_remote(
            self.__class__.__name__, self.work_dir
        )
        script = self._sweep_script(
            job,
            pack=pack,
            fresh=fresh,
            mpi_async_progress=mpi_async_progress,
        )

        self._transfer_remote_simulation_inputs(job)
        logger.debug("Transferring job file to remote path: %s", remote_job)
        self.put(Path(local_job), Path(remote_job))

        remote_script = (self.work_dir / "sweep").with_suffix(".sh")
        with _temporary_text_file(
            script,
            suffix=".sh",
            prefix="sweep",
            directory=_host_tmp_path_for_config(
                getattr(self, "_site_config_path", None)
            ),
        ) as script_path:
            logger.debug("Temporary sweep script created at %s", script_path)
            self.put(script_path, remote_script)

        self.run_login(f"chmod 700 {remote_script}")

        return remote_script, remote_job

    def _transfer_remote_simulation_inputs(self, job: BaseJob) -> None:
        """Transfer the simulation JSON and direct file inputs for a staged job."""

        local_sim, remote_sim = job.save_simulation_for_remote(
            self.__class__.__name__, self.work_dir
        )
        logger.debug("Transferring simulation file to remote path: %s", remote_sim)
        self.put(Path(local_sim), Path(remote_sim))

        for local_file, remote_file in job.remote_input_files(self.work_dir):
            logger.debug("Transferring input file to remote path: %s", remote_file)
            self.put(Path(local_file), Path(remote_file))

    def _is_running(self, job_id: int) -> bool:
        """Check if a job is running."""
        validated_job_id = _validate_slurm_job_id(job_id)
        status = self.run_login(f"squeue -j {validated_job_id} -h -o %t")
        return status == "R"

    @staticmethod
    def _mpi_async_progress_layout(
        *,
        cores_per_node: int,
        ranks_per_node: int,
    ) -> tuple[int, List[str]]:
        """Return solver threads and exports for MPI asynchronous progress."""

        if cores_per_node <= 0 or ranks_per_node <= 0:
            raise ValueError(
                "mpi_async_progress requires positive cores_per_node and "
                "ranks_per_node"
            )
        if cores_per_node % ranks_per_node:
            raise ValueError(
                "mpi_async_progress requires cores_per_node to be evenly "
                "divisible by ranks_per_node; "
                f"got {cores_per_node} cores and {ranks_per_node} ranks"
            )

        cores_per_rank = cores_per_node // ranks_per_node
        solver_threads = cores_per_rank - 1
        if solver_threads < 1:
            raise ValueError(
                "mpi_async_progress requires at least two cores per MPI rank "
                "so one core can be reserved for progress"
            )

        progress_pins = ",".join(
            str((rank + 1) * cores_per_rank - 1) for rank in range(ranks_per_node)
        )
        return solver_threads, [
            "export OMP_PLACES=cores",
            "export OMP_PROC_BIND=close",
            "export I_MPI_ASYNC_PROGRESS=1",
            "export I_MPI_ASYNC_PROGRESS_THREADS=1",
            f"export I_MPI_ASYNC_PROGRESS_PIN={progress_pins}",
        ]

    def _sweep_script(self, job: BaseJob, **kwargs) -> str:
        """Generate a script for sweeping through tasks on pre-provisioned resources."""

        n_tasks = job.n_tasks
        dir_out = str(job._remote_path(self.work_dir) / "logs")
        pack_job = kwargs.pop("pack", None)
        if pack_job is None:
            pack_job = kwargs.pop("pack_job", True)
        mpi_async_progress = kwargs.pop("mpi_async_progress", False)
        kwargs.pop("executable", None)
        n_threads = self.pool.ncore // self.pool.nproc
        mpi_async_progress_setup: List[str] = []
        if mpi_async_progress:
            nodes = int(self.pool.nhost or 1)
            if self.pool.ncore % nodes or self.pool.nproc % nodes:
                raise ValueError(
                    "mpi_async_progress requires allocation cores and ranks to "
                    "divide evenly across nodes"
                )
            n_threads, mpi_async_progress_setup = self._mpi_async_progress_layout(
                cores_per_node=self.pool.ncore // nodes,
                ranks_per_node=self.pool.nproc // nodes,
            )
        return self._render_template(
            "sweep/sweep_SLURM.sh",
            batch_job=False,
            n_tasks=n_tasks,
            n_procs=self.pool.nproc,
            n_threads=n_threads,
            mpi=self.mpi_cmd,
            dir_out=dir_out,
            executable=self.executable,
            imaging_job=isinstance(job, ImagingJob),
            pack_job=bool(pack_job),
            runtime_setup=self._runtime_setup_lines(),
            mpi_async_progress_setup=mpi_async_progress_setup,
            **kwargs,
        )

    def _sweep_SLURM_script(
        self,
        n_tasks: int,
        n_nodes: int,
        stdout: str,
        name: str = "FrequenSolve",
        duration: str = "00-02:00:00",
        ranks_per_node: Optional[int] = None,
        notify_on: Optional[Literal["begin", "end", "fail", "all", "none"]] = None,
        notify_email: Optional[str] = None,
        imaging_job: bool = False,
        n_job_tasks: Optional[int] = None,
        task_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> str:
        """Generate a SLURM sweep script.

        Args:
            n_tasks:        Number of tasks (frequencies) to run
            duration:       Duration of the job (DD-HH:MM:SS)
            n_nodes:        Number of nodes to run on
            ranks_per_node: Number of MPI ranks per node
            ranks_per_task: Number of MPI ranks per task
            queue:          Queue/partition to run on (optional, defaults to site queue)
            account:        Account/allocation to run on
            notify_on:      Notify on event (optional)
            notify_email:   Email address to notify (optional)
            **kwargs:       Additional keyword arguments
        """

        # Unpack keyword arguments
        queue = str(kwargs.pop("queue", self.config.queue))
        config = self.config_for_partition(queue)
        account = str(kwargs.pop("account", self.config.account))
        run_path = str(kwargs.pop("run_path", self.work_dir))
        rank_values = _normalize_rank_aliases(
            {
                "ranks_per_node": ranks_per_node,
                **{
                    key: kwargs.pop(key)
                    for key in tuple(_RANK_ALIASES)
                    if key in kwargs
                },
            }
        )
        ranks_per_node = int(rank_values.get("ranks_per_node") or 8)
        mem_cushion = float(kwargs.pop("mem_cushion", 1.5))
        min_ranks = int(kwargs.pop("min_ranks", 1))
        round_to = int(kwargs.pop("round_to", 1))
        cap_fraction = float(kwargs.pop("cap_fraction", 1.0))
        kwargs.pop("tail_threshold", None)
        boost_max_factor = float(kwargs.pop("boost_max_factor", 8.0))
        tolerate_failures = _normalize_failure_tolerance(
            kwargs.pop("tolerate_failures", 4),
            default=4,
        )
        sizing_json = kwargs.pop("sizing_json", None)
        if not sizing_json:
            sizing_json = str(Path(stdout).parent / "FS_sizing.json")
        launch_delay_seconds = float(kwargs.pop("launch_delay_seconds", 0.25))
        pack_job = bool(kwargs.pop("pack", True))
        mpi_async_progress = bool(kwargs.pop("mpi_async_progress", False))
        kwargs.pop("executable", None)
        n_job_tasks = int(n_tasks if n_job_tasks is None else n_job_tasks)
        if task_indices is None:
            task_indices = list(range(1, n_job_tasks + 1))
        else:
            task_indices = [int(index) for index in task_indices]
        skip_sizing = bool(kwargs.pop("skip_sizing", n_tasks == 1))
        proc_memory = (config.memory_per_node / ranks_per_node) / 1024.0
        duration = config.validate_request(n_nodes, n_nodes * ranks_per_node, duration)
        n_threads = config.cores_per_node // ranks_per_node
        mpi_async_progress_setup: List[str] = []
        if mpi_async_progress:
            n_threads, mpi_async_progress_setup = self._mpi_async_progress_layout(
                cores_per_node=config.cores_per_node,
                ranks_per_node=ranks_per_node,
            )
        scheduler_config = {
            "executable": str(self.executable),
            "mpi": str(self.mpi_cmd),
            "fresh": bool(kwargs.get("fresh", False)),
            "total_ranks": n_nodes * ranks_per_node,
            "omp_threads": n_threads,
            "mem_per_rank_gib": proc_memory,
            "job_task_count": n_job_tasks,
            "task_indices": task_indices,
            "skip_sizing": skip_sizing,
            "min_ranks": min_ranks,
            "round_to": round_to,
            "cap_fraction": cap_fraction,
            "mem_cushion": mem_cushion,
            "boost_max_factor": boost_max_factor,
            "failure_tolerance": tolerate_failures,
            "sizing_json": str(sizing_json),
            "launch_delay_seconds": launch_delay_seconds,
        }

        return self._render_template(
            "sweep/adaptive_sweep.sh",
            batch_job=True,
            name=name,
            dir_out=stdout,
            proc_memory=proc_memory,
            mem_cushion=mem_cushion,
            min_ranks=min_ranks,
            round_to=round_to,
            cap_fraction=cap_fraction,
            boost_max_factor=boost_max_factor,
            tolerate_failures=(
                "none" if tolerate_failures is None else int(tolerate_failures)
            ),
            skip_sizing=1 if skip_sizing else 0,
            n_nodes=n_nodes,
            n_procs=n_nodes * ranks_per_node,
            n_threads=n_threads,
            n_tasks=n_tasks,
            n_job_tasks=n_job_tasks,
            smooth_only=bool(kwargs.pop("smooth_only", False)),
            duration=duration,
            queue=queue,
            account=account,
            imaging_job=imaging_job,
            pack_job=pack_job,
            mpi=self.mpi_cmd,
            executable=self.executable,
            runtime_setup=self._runtime_setup_lines(),
            mpi_async_progress_setup=mpi_async_progress_setup,
            scheduler_config_shell=shlex.quote(json.dumps(scheduler_config, indent=2)),
            sizing_json_shell=shlex.quote(str(sizing_json)),
            scheduler_runner=shlex.quote(str(self._adaptive_scheduler_remote_path())),
            **({"sizing_json": sizing_json} if sizing_json is not None else {}),
            **({"run_path": run_path} if run_path is not None else {}),
            **({"notify_on": notify_on.upper()} if notify_on is not None else {}),
            **({"notify_email": notify_email} if notify_email is not None else {}),
            **kwargs,
        )

    def _generate_provision_script(
        self,
        n_nodes: int,
        ranks_per_node: int,
        duration: str = "00-02:00:00",
        queue: Optional[str] = None,
        account: Optional[str] = None,
        notify_email: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate a script for provisioning a SLURM allocation.

        Args:
            n_nodes:        Number of nodes to provision
            ranks_per_node: Number of MPI ranks per node
            duration:       Duration of the job (DD-HH:MM:SS)
            queue:          Queue/partition to run on (optional, defaults to site queue)
            account:        Account/allocation to run on
            notify_email:   Email address to notify (optional)
        """
        if "procs_per_node" in kwargs:
            values = _normalize_rank_aliases(
                {
                    "ranks_per_node": ranks_per_node,
                    "procs_per_node": kwargs.pop("procs_per_node"),
                }
            )
            ranks_per_node = int(values["ranks_per_node"])
        name = kwargs.get("name", "FS_cluster")
        account = account or self.run_config.account or self.config.account

        return self._render_template(
            "provision/provision_SLURM.sh",
            keep_trailing_newline=True,
            name=name,
            n_nodes=n_nodes,
            nhost=n_nodes,
            nproc=n_nodes * ranks_per_node,
            ranks_per_node=ranks_per_node,
            queue=queue,
            account=account,
            duration=duration,
            work_dir=self.work_dir,
            mpi=self.config.mpi_wrapper,
            **({"notify_email": notify_email} if notify_email is not None else {}),
        )

    def _print_batch_logs(
        self,
        job_name: str,
        log_dir: Path,
        job_id: Optional[str],
    ) -> None:
        """Print SLURM batch stdout/stderr logs if they were fetched."""
        log_path = Path(log_dir)

        if job_id:
            batch_dir = log_path / "batch"
            for suffix, label in ((".o", "batch stdout"), (".e", "batch stderr")):
                f = batch_dir / f"job_{job_id}{suffix}"
                if f.exists():
                    self._print_file_with_header(str(f), f"[{job_name}] {label}")

    @staticmethod
    def _latest_task_log_path(log_dir: Union[str, Path]) -> Optional[Path]:
        """Return path to the task log file with the highest index (task_N.txt or task_N.out)."""
        log_path = Path(log_dir)
        if not log_path.is_dir():
            return None
        best_index = -1
        best_path = None
        for pattern in ("task_*.log", "task_*.txt", "task_*.out"):
            for f in log_path.glob(pattern):
                try:
                    n = int(f.stem.rsplit("_", 1)[-1])
                    if n > best_index:
                        best_index = n
                        best_path = f
                except (ValueError, IndexError):
                    continue
        return best_path

    @staticmethod
    def _print_file_with_header(path: str, header: str) -> None:
        """Print a file's contents with a clear header."""
        try:
            text = Path(path).read_text(errors="replace")
        except OSError as e:
            print(f"--- {header} (could not read: {e}) ---")
            return
        print(f"\n{'=' * 60}\n{header}\n{path}\n{'=' * 60}\n{text}\n")

    @staticmethod
    def _normalize_optional_remote_dir(
        value: Optional[Union[str, Path]],
        *,
        name: str,
    ) -> Optional[Path]:
        """Normalize an optional absolute directory on the remote system."""

        if value is None or str(value).strip() == "":
            return None
        path = Path(value)
        if not path.is_absolute():
            raise ValueError(f"SLURM {name} must be an absolute remote path")
        return path

    def _get_work_dir(self) -> Path:
        """Resolve the remote base directory for relative FrequenSolve paths."""

        configured = self._normalize_optional_remote_dir(
            self._configured_work_dir,
            name="work_dir",
        )
        legacy_rel_path = self._rel_proj_path

        if configured is not None:
            work_dir = configured
            if legacy_rel_path is not None:
                work_dir = work_dir / legacy_rel_path
        else:
            _, stdout, stderr = self._login_client.client.exec_command("echo $WORK")
            work_root = stdout.read().decode().strip()
            if not work_root:
                raise RuntimeError(
                    f"Failed to determine $WORK for {self.site_name}; configure "
                    "an absolute work_dir base in site.toml or pass work_dir= explicitly"
                )
            work_dir = Path(work_root) / (legacy_rel_path or "frequensolve")

        self._work_dir = Path(work_dir)
        logger.info("Work directory: %s", self._work_dir)
        return self._work_dir

    def _get_free_port(self) -> int:
        """Find a free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            return s.getsockname()[1]

    async def _monitor_command_output(self, future, job, interactive=None):
        """Monitor command output and update future accordingly."""
        try:
            output_buffer = ""
            error_buffer = ""

            while True:
                # Handle subprocess case
                if isinstance(interactive, subprocess.Popen):
                    # Check if process has ended
                    if interactive.poll() is not None:
                        if not future.done():
                            future.set_exception(
                                RuntimeError("Process ended unexpectedly")
                            )
                        return

                    # Read from stdout/stderr
                    reads, _, _ = select(
                        [interactive.stdout, interactive.stderr], [], [], 0.1
                    )
                    for fd in reads:
                        data = fd.read(4096).decode("utf-8", errors="replace")
                        if fd == interactive.stdout:
                            output_buffer += data
                        else:
                            error_buffer += data

                # Process output buffer
                while "\n" in output_buffer:
                    line, output_buffer = output_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._emit(line)
                    if "Sweep Complete" in line:
                        future.set_result(job.traces)
                        logger.info("Sweep job completed successfully")
                        if interactive:
                            if isinstance(interactive, subprocess.Popen):
                                try:
                                    pgid = os.getpgid(interactive.pid)
                                    os.killpg(pgid, signal.SIGKILL)
                                except Exception:
                                    try:
                                        interactive.kill()
                                    except Exception:
                                        pass
                            else:
                                interactive.close()
                        return

                # Process error buffer
                while "\n" in error_buffer:
                    line, error_buffer = error_buffer.split("\n", 1)
                    line = line.strip()
                    if line:
                        self._emit(line, level=logging.ERROR)
                        future.set_exception(RuntimeError(f"Sweep job failed: {line}"))
                        if interactive:
                            if isinstance(interactive, subprocess.Popen):
                                try:
                                    pgid = os.getpgid(interactive.pid)
                                    os.killpg(pgid, signal.SIGKILL)
                                except Exception:
                                    try:
                                        interactive.kill()
                                    except Exception:
                                        pass
                            else:
                                interactive.close()
                        return

                await asyncio.sleep(0.2)

        except Exception as e:
            logger.exception("Error in monitor task")
            future.set_exception(e)
            if interactive:
                if isinstance(interactive, subprocess.Popen):
                    try:
                        pgid = os.getpgid(interactive.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except Exception:
                        try:
                            interactive.kill()
                        except Exception:
                            pass
                else:
                    interactive.close()
