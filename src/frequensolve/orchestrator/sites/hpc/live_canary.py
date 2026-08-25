"""Bounded, manually invoked live Slurm acceptance canary.

This module deliberately has no scheduled or CI entry point.  Operators must
provide a local policy file and an explicit acknowledgement for every run.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import stat
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from frequensolve import (
    Q_,
    Acquisition,
    BoundaryCondition,
    FrequencyDomainJob,
    LayeredModel,
    Project,
    ReceiverNode,
    VtkOutput,
    __version__,
    ureg,
)
from frequensolve.orchestrator.sites import Site, load_site_config
from frequensolve.orchestrator.sites.hpc.enterprise import EnterpriseHPCProfile
from frequensolve.orchestrator.sites.hpc.slurm_helpers import validate_slurm_job_id

CANARY_POLICY_SCHEMA = "frequensolve-live-slurm-canary-policy/v1"
CANARY_EVIDENCE_SCHEMA = "frequensolve-live-slurm-canary-evidence/v1"
CANARY_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_BOUNDED_LIVE_SLURM_SPEND"
SYNTHETIC_FIXTURE_ID = "synthetic-acoustic-2d-v1"

_POLICY_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]*\Z")
_KNOWN_HOST_NAME = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9.:-]*|\[[A-Fa-f0-9:.]+\]:[0-9]+)\Z"
)
_WALL_TIME = re.compile(r"[0-9]{2}:[0-5][0-9]:[0-5][0-9]\Z")
_RUN_TOKEN = re.compile(r"[a-z0-9][a-z0-9-]{5,31}\Z")


class LiveSlurmCanaryError(RuntimeError):
    """Raised when the live canary is unsafe, incomplete, or unsuccessful."""


def _required_text(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"live Slurm canary {label} has an invalid value")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"live Slurm canary {label} must be a positive integer")
    return value


def _remote_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"live Slurm canary {label} must be an absolute path")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ValueError(
            f"live Slurm canary {label} must be an absolute traversal-free path"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"live Slurm canary {label} contains a control character")
    return str(path)


def _wall_time_seconds(value: str) -> int:
    _required_text(value, "maxWallTime", _WALL_TIME)
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


@dataclass(frozen=True)
class LiveSlurmCanaryPolicy:
    """Closed local execution policy required before any scheduler mutation."""

    policy_id: str
    fixture_id: str
    allowed_host: str
    known_hosts_file: Path
    known_hosts_name: str
    allowed_partition: str
    allowed_work_dir: str
    allowed_scratch_dir: str
    max_nodes: int
    max_ranks: int
    max_threads_per_rank: int
    max_wall_time: str
    completion_timeout_seconds: int
    cancel_timeout_seconds: int
    cleanup_owner: str
    log_retention: str
    emergency_cancel_command: str
    schema: str = CANARY_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != CANARY_POLICY_SCHEMA:
            raise ValueError(
                f"live Slurm canary schema must be {CANARY_POLICY_SCHEMA!r}"
            )
        _required_text(self.policy_id, "policyId", _POLICY_ID)
        if self.fixture_id != SYNTHETIC_FIXTURE_ID:
            raise ValueError(
                f"live Slurm canary fixtureId must be {SYNTHETIC_FIXTURE_ID!r}"
            )
        _required_text(self.allowed_host, "allowedHost", _HOST)
        _required_text(self.known_hosts_name, "knownHostsName", _KNOWN_HOST_NAME)
        _required_text(self.allowed_partition, "allowedPartition", _TOKEN)
        _remote_path(self.allowed_work_dir, "allowedWorkDir")
        _remote_path(self.allowed_scratch_dir, "allowedScratchDir")
        for label, value in (
            ("maxNodes", self.max_nodes),
            ("maxRanks", self.max_ranks),
            ("maxThreadsPerRank", self.max_threads_per_rank),
            ("completionTimeoutSeconds", self.completion_timeout_seconds),
            ("cancelTimeoutSeconds", self.cancel_timeout_seconds),
        ):
            _positive_int(value, label)
        if _wall_time_seconds(self.max_wall_time) > self.completion_timeout_seconds:
            raise ValueError(
                "live Slurm canary completion timeout must cover maxWallTime"
            )
        if not self.cleanup_owner.strip() or len(self.cleanup_owner) > 128:
            raise ValueError("live Slurm canary cleanupOwner is required")
        if not self.log_retention.strip() or len(self.log_retention) > 128:
            raise ValueError("live Slurm canary logRetention is required")
        if self.emergency_cancel_command != "scancel {job_id}":
            raise ValueError(
                "live Slurm canary emergencyCancelCommand must be " "'scancel {job_id}'"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LiveSlurmCanaryPolicy":
        """Parse a closed policy mapping and reject unknown controls."""

        fields = {
            "schema",
            "policyId",
            "fixtureId",
            "allowedHost",
            "knownHostsFile",
            "knownHostsName",
            "allowedPartition",
            "allowedWorkDir",
            "allowedScratchDir",
            "maxNodes",
            "maxRanks",
            "maxThreadsPerRank",
            "maxWallTime",
            "completionTimeoutSeconds",
            "cancelTimeoutSeconds",
            "cleanupOwner",
            "logRetention",
            "emergencyCancelCommand",
        }
        unknown = sorted(set(values) - fields)
        if unknown:
            raise ValueError(
                "unsupported live Slurm canary policy key(s): " + ", ".join(unknown)
            )
        missing = sorted(fields - set(values))
        if missing:
            raise ValueError(
                "missing live Slurm canary policy key(s): " + ", ".join(missing)
            )
        return cls(
            schema=str(values["schema"]),
            policy_id=str(values["policyId"]),
            fixture_id=str(values["fixtureId"]),
            allowed_host=str(values["allowedHost"]),
            known_hosts_file=Path(str(values["knownHostsFile"])).expanduser(),
            known_hosts_name=str(values["knownHostsName"]),
            allowed_partition=str(values["allowedPartition"]),
            allowed_work_dir=str(values["allowedWorkDir"]),
            allowed_scratch_dir=str(values["allowedScratchDir"]),
            max_nodes=_positive_int(values["maxNodes"], "maxNodes"),
            max_ranks=_positive_int(values["maxRanks"], "maxRanks"),
            max_threads_per_rank=_positive_int(
                values["maxThreadsPerRank"], "maxThreadsPerRank"
            ),
            max_wall_time=str(values["maxWallTime"]),
            completion_timeout_seconds=_positive_int(
                values["completionTimeoutSeconds"], "completionTimeoutSeconds"
            ),
            cancel_timeout_seconds=_positive_int(
                values["cancelTimeoutSeconds"], "cancelTimeoutSeconds"
            ),
            cleanup_owner=str(values["cleanupOwner"]),
            log_retention=str(values["logRetention"]),
            emergency_cancel_command=str(values["emergencyCancelCommand"]),
        )


def load_live_slurm_canary_policy(path: Path) -> LiveSlurmCanaryPolicy:
    """Load a duplicate-key-safe local canary policy."""

    def closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate live Slurm canary policy key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(), object_pairs_hook=closed_object)
    except OSError as exc:
        raise LiveSlurmCanaryError("live Slurm canary policy is unavailable") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise LiveSlurmCanaryError("live Slurm canary policy is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise LiveSlurmCanaryError("live Slurm canary policy must contain an object")
    try:
        return LiveSlurmCanaryPolicy.from_mapping(payload)
    except ValueError as exc:
        raise LiveSlurmCanaryError(str(exc)) from exc


def require_live_slurm_acknowledgement(value: Optional[str]) -> None:
    """Fail closed unless the operator explicitly accepts bounded live spend."""

    if value != CANARY_ACKNOWLEDGEMENT:
        raise LiveSlurmCanaryError(
            "live Slurm canary is disabled; pass the exact acknowledgement "
            f"{CANARY_ACKNOWLEDGEMENT!r}"
        )


def _selected_site_table(
    config: Mapping[str, Any], profile: Optional[str]
) -> Mapping[str, Any]:
    sites = config.get("sites")
    selected = profile or config.get("default")
    if not isinstance(sites, Mapping) or not isinstance(selected, str):
        raise LiveSlurmCanaryError("site config has no selected profile")
    table = sites.get(selected)
    if not isinstance(table, Mapping):
        raise LiveSlurmCanaryError("selected site profile is unavailable")
    return table


def validate_live_slurm_local_inputs(
    *,
    site_config_path: Path,
    profile: Optional[str],
    policy: LiveSlurmCanaryPolicy,
) -> None:
    """Validate non-interactive credentials and trusted-host material."""

    config = load_site_config(site_config_path)
    site = _selected_site_table(config, profile)
    site_type = str(site.get("type", "")).replace("_", "").replace("-", "").lower()
    if site_type not in {"slurm", "slurmsite"}:
        raise LiveSlurmCanaryError("selected site profile is not generic Slurm")
    if site.get("hostname") != policy.allowed_host:
        raise LiveSlurmCanaryError("site host is not allowlisted by the canary policy")
    if not isinstance(site.get("username"), str) or not str(site["username"]).strip():
        raise LiveSlurmCanaryError(
            "site config must provide a non-interactive username"
        )
    key_value = site.get("ssh_key")
    if not isinstance(key_value, str) or not key_value.strip():
        raise LiveSlurmCanaryError("site config must provide an explicit SSH key path")
    key_path = Path(key_value).expanduser()
    try:
        key_stat = key_path.stat()
    except OSError as exc:
        raise LiveSlurmCanaryError("configured SSH key is unavailable") from exc
    if not stat.S_ISREG(key_stat.st_mode) or key_stat.st_mode & 0o077:
        raise LiveSlurmCanaryError(
            "configured SSH key must be a regular file without group/world access"
        )
    try:
        from paramiko import HostKeys

        host_keys = HostKeys(str(policy.known_hosts_file))
    except (OSError, ValueError) as exc:
        raise LiveSlurmCanaryError("approved known-hosts file is unavailable") from exc
    if not host_keys.lookup(policy.known_hosts_name):
        raise LiveSlurmCanaryError(
            "approved known-hosts file has no key for knownHostsName"
        )


def validate_live_slurm_site(
    site: Any,
    policy: LiveSlurmCanaryPolicy,
    *,
    ranks: int,
    threads_per_rank: int,
) -> EnterpriseHPCProfile:
    """Bind the instantiated site and immutable bundle to the approved policy."""

    profile = getattr(site, "enterprise_hpc", None)
    if not isinstance(profile, EnterpriseHPCProfile):
        raise LiveSlurmCanaryError("site has no Enterprise HPC compatibility profile")
    observed = {
        "host": str(getattr(site.config, "hostname", "")),
        "partition": str(getattr(site.config, "queue", "")),
        "work_dir": str(site.work_dir),
        "scratch_dir": str(site.scratch_dir or ""),
    }
    expected = {
        "host": policy.allowed_host,
        "partition": policy.allowed_partition,
        "work_dir": policy.allowed_work_dir,
        "scratch_dir": policy.allowed_scratch_dir,
    }
    mismatches = [name for name in expected if observed[name] != expected[name]]
    if mismatches:
        raise LiveSlurmCanaryError(
            "site is outside the live canary policy: " + ", ".join(mismatches)
        )
    for label, actual, maximum in (
        ("nodes", int(site.run_config.nodes), policy.max_nodes),
        ("ranks", ranks, policy.max_ranks),
        ("threads per rank", threads_per_rank, policy.max_threads_per_rank),
        ("profile nodes", profile.max_nodes, policy.max_nodes),
        ("profile ranks", profile.max_ranks, policy.max_ranks),
        (
            "profile threads per rank",
            profile.max_threads_per_rank,
            policy.max_threads_per_rank,
        ),
    ):
        if actual < 1 or actual > maximum:
            raise LiveSlurmCanaryError(f"live Slurm canary exceeds approved {label}")
    duration = str(site.run_config.duration or "")
    if not _WALL_TIME.fullmatch(duration):
        raise LiveSlurmCanaryError(
            "site run config must set a bounded HH:MM:SS duration"
        )
    if _wall_time_seconds(duration) > _wall_time_seconds(policy.max_wall_time):
        raise LiveSlurmCanaryError("live Slurm canary exceeds approved wall time")
    if profile.max_wall_time != policy.max_wall_time:
        raise LiveSlurmCanaryError("Enterprise HPC profile wall time is outside policy")
    if profile.allowed_partitions != (policy.allowed_partition,):
        raise LiveSlurmCanaryError(
            "Enterprise HPC profile partitions are outside policy"
        )
    return profile


def build_synthetic_live_slurm_job(
    project_path: Path, run_token: str
) -> tuple[Any, Any, Any]:
    """Build the small, visibly synthetic public-API canary fixture."""

    _required_text(run_token, "run token", _RUN_TOKEN)
    if project_path.exists():
        raise LiveSlurmCanaryError("canary project path must not already exist")
    prefix = f"fs_canary_{run_token.replace('-', '_')}"
    project = Project(name=f"{prefix}_project", path=project_path)
    simulation = project.new_simulation(
        name=f"{prefix}_simulation",
        physics="acoustic",
        dimension=2,
        units={"length": "km", "velocity": "km/s", "density": "g/cm^3"},
    )
    model = LayeredModel(name="synthetic_model", dimension=2, x_limits=[0.0, 0.4])
    model.add_surface(name="top", depth=0.0 * ureg.m)
    model.add_layer(
        name="synthetic_upper",
        properties={
            "Vp": 2000 * ureg.m / ureg.s,
            "Rho": 2.2 * ureg.g / ureg.cm**3,
        },
    )
    model.add_surface(name="interface", depth=100 * ureg.m)
    model.add_layer(
        name="synthetic_lower",
        properties={
            "Vp": 2600 * ureg.m / ureg.s,
            "Rho": 2.4 * ureg.g / ureg.cm**3,
        },
    )
    model.add_surface(name="bottom", depth=200 * ureg.m)
    simulation += model
    simulation += model.hex_mesh_generator([4, 2])
    simulation.mesh.set_adapt(elems_per_wave=2.0, order=4, f_low=5.0)
    simulation.mesh.set_source_grading(d1=0.05, factor=2.0)
    simulation += BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    simulation += BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min", "x_max", "z_max"],
        pml_wavelengths=0.5,
        pml_reflection=1e-3,
    )
    acquisition = Acquisition()
    acquisition.add_sources(kind="scalar", coords=Q_([[200, 10]], "m"))
    receiver = ReceiverNode(name="synthetic_hydrophone")
    receiver.add_component(name="p", field="pressure")
    acquisition.add_receiver_group(
        name="surface",
        device=receiver,
        coords=Q_([[x, 25] for x in np.linspace(0, 400, 21)], "m"),
    )
    simulation += acquisition
    output = VtkOutput.domain(
        name="synthetic_field",
        properties=["Vp"],
        fields=["pressure"],
        show_pml=False,
        upscale=0,
        order=1,
    )
    success = FrequencyDomainJob(
        name=f"{prefix}_success",
        simulation=simulation,
        f_list=[20.0],
        outputs=[output],
    )
    cancellation = FrequencyDomainJob(
        name=f"{prefix}_cancel",
        simulation=simulation,
        f_list=[20.0],
        outputs=[output],
    )
    return project, success, cancellation


def _manifest_evidence(job: Any, ranks: int, threads_per_rank: int) -> dict[str, Any]:
    manifest_path = Path(job._result_path) / "_fs_run" / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        task = manifest["tasks"][0]
        convergence = task["solver"]["convergence"]
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise LiveSlurmCanaryError("fetched solver run manifest is incomplete") from exc
    if task.get("n_ranks") != ranks or task.get("threads_per_rank") != threads_per_rank:
        raise LiveSlurmCanaryError("solver run manifest has an unexpected MPI layout")
    if not convergence.get("converged"):
        raise LiveSlurmCanaryError("synthetic solver case did not converge")
    return {
        "mpiRanks": task["n_ranks"],
        "threadsPerRank": task["threads_per_rank"],
        "iterations": convergence.get("iterations"),
        "residual": convergence.get("residual"),
        "solverStatus": convergence.get("status"),
    }


def _trace_evidence(result: Any) -> dict[str, Any]:
    traces = result.traces()
    sources = traces.sources("surface")
    if len(sources) != 1:
        raise LiveSlurmCanaryError("synthetic trace identity is not stable")
    values = np.asarray(traces.fd("surface", "p", source=int(sources[0])))
    if not values.size or not np.isfinite(values).all() or not np.any(values != 0):
        raise LiveSlurmCanaryError("synthetic trace invariant failed")
    return {
        "group": "surface",
        "component": "p",
        "sourceCount": 1,
        "shape": list(values.shape),
        "finite": True,
        "nonzero": True,
    }


def _process_count(site: Any, needle: str) -> int:
    encoded = base64.b64encode(needle.encode()).decode()
    probe = (
        "import base64,glob,os,sys\n"
        "needle=base64.b64decode(sys.argv[1])\n"
        "own={os.getpid(),os.getppid()}\n"
        "matches=[]\n"
        "for path in glob.glob('/proc/[0-9]*/cmdline'):\n"
        "    if int(path.split('/')[2]) in own:\n"
        "        continue\n"
        "    try:\n"
        "        data=open(path,'rb').read()\n"
        "    except OSError:\n"
        "        continue\n"
        "    if needle in data:\n"
        "        matches.append(path)\n"
        "print(len(matches))"
    )
    output = site.run_login(
        f"python3 -c {shlex.quote(probe)} {shlex.quote(encoded)}"
    ).strip()
    try:
        return int(output)
    except ValueError as exc:
        raise LiveSlurmCanaryError("remote process cleanup probe failed") from exc


def _owned_remote_paths(site: Any, project: Any, jobs: Sequence[Any]) -> list[str]:
    root = PurePosixPath(str(site.work_dir))
    paths = [str(site._remote_job_dir(job)) for job in jobs]
    remote_simulation = root / "jobs" / project.simulations[0].name
    paths.extend(
        [
            str(remote_simulation),
            str(root / "simulations" / project.simulations[0].name),
            str(root / f"{project.name}.json"),
        ]
    )
    for value in paths:
        path = PurePosixPath(value)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise LiveSlurmCanaryError(
                "cleanup target is outside the work directory"
            ) from exc
        if not path.name.startswith("fs_canary_"):
            raise LiveSlurmCanaryError("cleanup target is not canary-owned")
    return paths


def _cleanup_remote_paths(site: Any, paths: Sequence[str]) -> None:
    cleaner = (
        "import pathlib,shutil,sys;"
        "[(p.unlink() if p.is_symlink() or p.is_file() else shutil.rmtree(p)) "
        "for p in map(pathlib.Path,sys.argv[1:]) if p.exists() or p.is_symlink()]"
    )
    arguments = " ".join(shlex.quote(path) for path in paths)
    site.run_login(f"python3 -c {shlex.quote(cleaner)} {arguments}")
    remaining = site.run_login(
        "python3 -c "
        + shlex.quote(
            "import pathlib,sys;print(sum(pathlib.Path(p).exists() or "
            "pathlib.Path(p).is_symlink() for p in sys.argv[1:]))"
        )
        + " "
        + arguments
    ).strip()
    if remaining != "0":
        raise LiveSlurmCanaryError("canary-owned remote cleanup is incomplete")


def _scheduler_job_present(site: Any, job_id: Any) -> bool:
    validated_job_id = validate_slurm_job_id(job_id)
    return bool(site.run_login(f"squeue -h -j {validated_job_id} -o %i").strip())


def _cancel_active_scheduler_job(
    site: Any,
    handle: Any,
    *,
    timeout: float,
    poll_interval: float,
) -> None:
    """Cancel an active scheduler job even if the public handle is terminal."""

    if not _scheduler_job_present(site, handle.id):
        return
    handle.cancel()
    deadline = time.monotonic() + timeout
    while _scheduler_job_present(site, handle.id):
        if time.monotonic() >= deadline:
            raise LiveSlurmCanaryError(
                "canary scheduler job remained active after cancellation"
            )
        time.sleep(poll_interval)


def run_live_slurm_canary(
    *,
    site: Any,
    policy: LiveSlurmCanaryPolicy,
    project_path: Path,
    run_token: str,
    ranks: int,
    threads_per_rank: int,
) -> dict[str, Any]:
    """Run success, fetch/load, cancellation, and owned-path cleanup checks."""

    profile = validate_live_slurm_site(
        site,
        policy,
        ranks=ranks,
        threads_per_rank=threads_per_rank,
    )
    preflight = site.enterprise_hpc_preflight()
    project, success_job, cancel_job = build_synthetic_live_slurm_job(
        project_path, run_token
    )
    started = time.monotonic()
    success_handle = None
    cancel_handle = None
    remote_paths: list[str] = []
    try:
        success_handle = site.submit(
            success_job,
            force=True,
            fetch=True,
            check=True,
            mode="batch",
            queue=policy.allowed_partition,
            nodes=1,
            ranks_per_node=ranks,
            threads_per_rank=threads_per_rank,
            duration=policy.max_wall_time,
        )
        success_states = [
            status.state
            for status in success_handle.watch(
                timeout=policy.completion_timeout_seconds,
                poll_interval=site.run_config.poll_interval,
            )
        ]
        success_result = success_handle.wait(check=True)
        if success_result.status.state != "complete":
            raise LiveSlurmCanaryError("success case did not reach complete")
        trace_evidence = _trace_evidence(success_result)
        solver_evidence = _manifest_evidence(
            success_job, ranks=ranks, threads_per_rank=threads_per_rank
        )

        cancel_handle = site.submit(
            cancel_job,
            force=True,
            fetch=False,
            check=False,
            mode="batch",
            queue=policy.allowed_partition,
            nodes=1,
            ranks_per_node=ranks,
            threads_per_rank=threads_per_rank,
            duration=policy.max_wall_time,
        )
        cancel_handle.cancel()
        cancel_result = cancel_handle.wait(
            timeout=policy.cancel_timeout_seconds,
            poll_interval=site.run_config.poll_interval,
            check=False,
        )
        if cancel_result.status.state != "cancelled":
            raise LiveSlurmCanaryError("cancellation case did not reach cancelled")
        scheduler_remaining = site.run_login(
            f"squeue -h -j {shlex.quote(str(cancel_handle.id))} -o %i"
        ).strip()
        if scheduler_remaining:
            raise LiveSlurmCanaryError("cancelled job remains in the Slurm queue")
        process_count = _process_count(site, cancel_job.name)
        if process_count:
            raise LiveSlurmCanaryError(
                "cancelled canary process remains on the cluster"
            )
        remote_paths = _owned_remote_paths(site, project, [success_job, cancel_job])
        _cleanup_remote_paths(site, remote_paths)

        elapsed = time.monotonic() - started
        return {
            "schema": CANARY_EVIDENCE_SCHEMA,
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "policyId": policy.policy_id,
            "fixtureId": policy.fixture_id,
            "frequensolveVersion": __version__,
            "identities": preflight.to_evidence(),
            "request": {
                "partition": policy.allowed_partition,
                "nodes": 1,
                "ranks": ranks,
                "threadsPerRank": threads_per_rank,
                "maxWallTime": policy.max_wall_time,
            },
            "success": {
                "schedulerJobId": str(success_handle.id),
                "states": success_states,
                "terminalState": success_result.status.state,
                "outputs": len(success_result.output_files(existing=True)),
                "traces": trace_evidence,
                **solver_evidence,
            },
            "cancellation": {
                "schedulerJobId": str(cancel_handle.id),
                "terminalState": cancel_result.status.state,
                "schedulerJobsRemaining": 0,
                "remoteProcessesRemaining": process_count,
            },
            "cleanup": {
                "ownedPathsRemoved": len(remote_paths),
                "unownedPathsRemoved": 0,
                "ownerRecorded": bool(policy.cleanup_owner),
            },
            "retentionPolicyRecorded": bool(policy.log_retention),
            "elapsedSeconds": round(elapsed, 3),
            "licensingBehaviorEvaluated": False,
        }
    finally:
        all_terminal = True
        for handle in (cancel_handle, success_handle):
            if handle is None:
                continue
            try:
                _cancel_active_scheduler_job(
                    site,
                    handle,
                    timeout=policy.cancel_timeout_seconds,
                    poll_interval=site.run_config.poll_interval,
                )
            except Exception:
                all_terminal = False
            else:
                all_terminal = all_terminal and not _scheduler_job_present(
                    site, handle.id
                )
        if all_terminal and not remote_paths and success_handle is not None:
            try:
                remote_paths = _owned_remote_paths(
                    site, project, [success_job, cancel_job]
                )
                _cleanup_remote_paths(site, remote_paths)
            except Exception:
                pass


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--site-config", required=True, type=Path)
    parser.add_argument("--site-profile")
    parser.add_argument("--project-path", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--ranks", required=True, type=int)
    parser.add_argument("--threads-per-rank", default=1, type=int)
    parser.add_argument(
        "--acknowledge",
        help="Exact per-run acknowledgement; it is intentionally not persisted",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the manual canary and write sanitized evidence atomically."""

    args = _argument_parser().parse_args(argv)
    require_live_slurm_acknowledgement(
        args.acknowledge or os.environ.get("FREQUENSOLVE_LIVE_SLURM_ACKNOWLEDGEMENT")
    )
    policy = load_live_slurm_canary_policy(args.policy)
    validate_live_slurm_local_inputs(
        site_config_path=args.site_config,
        profile=args.site_profile,
        policy=policy,
    )
    evidence_path = args.evidence.expanduser().resolve()
    if evidence_path.exists():
        raise LiveSlurmCanaryError("evidence path already exists")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    site = Site(config_path=args.site_config, profile=args.site_profile)
    try:
        evidence = run_live_slurm_canary(
            site=site,
            policy=policy,
            project_path=args.project_path.expanduser().resolve(),
            run_token=args.run_token,
            ranks=args.ranks,
            threads_per_rank=args.threads_per_rank,
        )
    finally:
        site.close()
    temporary = evidence_path.with_suffix(evidence_path.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    temporary.replace(evidence_path)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
