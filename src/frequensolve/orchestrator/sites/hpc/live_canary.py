"""Manually exercise FrequenSolve's public SSH/SFTP/Slurm behavior.

The Deployment acceptance harness supplies the generated Enterprise profile,
execution policy, cleanup, and evidence retention. This module is intentionally
limited to one synthetic success case and one bounded cancellation case.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

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
    ureg,
)
from frequensolve.orchestrator.sites import Site
from frequensolve.orchestrator.sites.hpc.slurm_helpers import validate_slurm_job_id

CANARY_ACKNOWLEDGEMENT = "I_ACKNOWLEDGE_BOUNDED_LIVE_SLURM_SPEND"
SYNTHETIC_FIXTURE_ID = "synthetic-acoustic-2d-v1"

_RUN_TOKEN = re.compile(r"[a-z0-9][a-z0-9-]{5,31}\Z")
_JOB_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class LiveSlurmCanaryError(RuntimeError):
    """Raised when the opt-in live canary cannot finish safely."""


class _RecoveredHandle:
    """Cancel a scheduler job recovered after an interrupted submission."""

    def __init__(self, site: Any, job_id: str):
        self.site = site
        self.id = job_id

    def cancel(self) -> None:
        self.site.cancel_job(self.id)


def require_live_slurm_acknowledgement(value: Optional[str]) -> None:
    """Require an explicit acknowledgement for every live invocation."""

    if value != CANARY_ACKNOWLEDGEMENT:
        raise LiveSlurmCanaryError(
            "live Slurm canary is disabled; pass the exact acknowledgement "
            f"{CANARY_ACKNOWLEDGEMENT!r}"
        )


def build_synthetic_live_slurm_job(
    project_path: Path, run_token: str
) -> tuple[Any, Any, Any]:
    """Build a small, visibly synthetic public-API fixture."""

    if not isinstance(run_token, str) or not _RUN_TOKEN.fullmatch(run_token):
        raise LiveSlurmCanaryError("live Slurm canary run token is invalid")
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
    jobs = tuple(
        FrequencyDomainJob(
            name=f"{prefix}_{suffix}",
            simulation=simulation,
            f_list=[20.0],
            outputs=[output],
        )
        for suffix in ("success", "cancel")
    )
    return project, jobs[0], jobs[1]


def _solver_evidence(log_path: Path, expected_ranks: int) -> dict[str, Any]:
    log_dir = log_path if log_path.is_dir() else log_path.parent
    try:
        init_log = (log_dir / "init.log").read_text()
        task_log = sorted(log_dir.glob("task_*.log"))[0].read_text()
        ranks_match = re.search(r"(?m)^\s*Ranks\s*:\s*(\d+)\s*$", init_log)
        threads_match = re.search(r"(?m)^\s*Threads\s*:\s*(\d+)\s*$", init_log)
        if ranks_match is None or threads_match is None:
            raise LiveSlurmCanaryError("fetched solver logs are incomplete")
        ranks = int(ranks_match.group(1))
        threads = int(threads_match.group(1))
        iterations = re.findall(r"(?m)^\s*(\d+)\s+([0-9.]+E[+-]\d+)\s+", task_log)
    except (IndexError, OSError, ValueError) as exc:
        raise LiveSlurmCanaryError("fetched solver logs are incomplete") from exc
    if (
        ranks != expected_ranks
        or threads < 1
        or not iterations
        or "-- Timing --" not in task_log
    ):
        raise LiveSlurmCanaryError("solver MPI/convergence invariant failed")
    return {
        "mpiRanks": ranks,
        "threadsPerRank": threads,
        "iterations": int(iterations[-1][0]),
        "residual": float(iterations[-1][1]),
    }


def _trace_evidence(result: Any) -> dict[str, Any]:
    traces = result.traces()
    sources = traces.sources("surface")
    if len(sources) != 1:
        raise LiveSlurmCanaryError("synthetic trace identity is not stable")
    values = np.asarray(traces.fd("surface", "p", source=int(sources[0])))
    if not values.size or not np.isfinite(values).all() or not np.any(values != 0):
        raise LiveSlurmCanaryError("synthetic trace invariant failed")
    return {"shape": list(values.shape), "finite": True, "nonzero": True}


def _scheduler_job_present(site: Any, job_id: Any) -> bool:
    job_id = validate_slurm_job_id(job_id)
    return bool(site.run_login(f"squeue -h -j {job_id} -o %i").strip())


def _scheduler_job_ids_for_name(site: Any, name: str) -> tuple[str, ...]:
    if not isinstance(name, str) or not _JOB_NAME.fullmatch(name):
        raise LiveSlurmCanaryError("canary scheduler job name is invalid")
    output = site.run_login(f"squeue -h -n {name} -o %i").strip()
    if not output:
        return ()
    return tuple(validate_slurm_job_id(line.strip()) for line in output.splitlines())


def _recover_handles(
    site: Any, handles: Sequence[Any], jobs: Sequence[Any]
) -> tuple[Any, ...]:
    by_id = {
        validate_slurm_job_id(handle.id): handle
        for handle in handles
        if handle is not None
    }
    for job in jobs:
        if getattr(job, "_job_id", None) is not None:
            job_id = validate_slurm_job_id(job._job_id)
            by_id.setdefault(job_id, _RecoveredHandle(site, job_id))
        for job_id in _scheduler_job_ids_for_name(site, str(job.name)):
            by_id.setdefault(job_id, _RecoveredHandle(site, job_id))
    return tuple(by_id.values())


def _cancel_active_job(
    site: Any, handle: Any, *, timeout: float, poll_interval: float
) -> None:
    if not _scheduler_job_present(site, handle.id):
        return
    handle.cancel()
    deadline = time.monotonic() + timeout
    while _scheduler_job_present(site, handle.id):
        if time.monotonic() >= deadline:
            job_id = validate_slurm_job_id(handle.id)
            raise LiveSlurmCanaryError(
                f"scheduler job {job_id} remains active; run scancel {job_id}"
            )
        time.sleep(poll_interval)


def _cancel_all(site: Any, handles: Sequence[Any], *, timeout: float) -> None:
    failures = []
    for handle in handles:
        try:
            _cancel_active_job(
                site,
                handle,
                timeout=timeout,
                poll_interval=float(site.run_config.poll_interval or 1),
            )
        except Exception as exc:
            failures.append((str(getattr(handle, "id", "unknown")), exc))
    if failures:
        job_id, error = failures[0]
        raise LiveSlurmCanaryError(
            f"canary cleanup could not cancel job {job_id}; run scancel {job_id}"
        ) from error


def run_live_slurm_canary(
    *,
    site: Any,
    project_path: Path,
    run_token: str,
    timeout: float,
    cancel_timeout: float,
) -> dict[str, Any]:
    """Exercise submit, observation, fetch/load, and cancellation."""

    if timeout <= 0 or cancel_timeout <= 0:
        raise LiveSlurmCanaryError("canary timeouts must be positive")
    ranks_per_node = site.run_config.ranks_per_node
    if (
        isinstance(ranks_per_node, bool)
        or not isinstance(ranks_per_node, int)
        or ranks_per_node < 1
    ):
        raise LiveSlurmCanaryError(
            "site run_config.ranks_per_node must be a positive integer"
        )
    _, success_job, cancel_job = build_synthetic_live_slurm_job(project_path, run_token)
    success_handle = None
    cancel_handle = None
    try:
        for job in (success_job, cancel_job):
            if _scheduler_job_ids_for_name(site, job.name):
                raise LiveSlurmCanaryError(
                    "canary scheduler name is already active; choose a fresh run token"
                )
        success_handle = site.submit(
            success_job,
            force=True,
            fetch=True,
            check=True,
            mode="batch",
            queue=site.run_config.queue,
            nodes=site.run_config.nodes,
            ranks_per_node=ranks_per_node,
            duration=site.run_config.duration,
            name=success_job.name,
        )
        states = [
            status.state
            for status in success_handle.watch(
                timeout=timeout,
                poll_interval=site.run_config.poll_interval,
            )
        ]
        success_result = success_handle.wait(check=True)
        if success_result.status.state != "complete":
            raise LiveSlurmCanaryError("success case did not reach complete")
        traces = _trace_evidence(success_result)
        expected_ranks = site.run_config.nodes * ranks_per_node
        solver = _solver_evidence(site.fetch_logs(success_job), expected_ranks)

        cancel_handle = site.submit(
            cancel_job,
            force=True,
            fetch=False,
            check=False,
            mode="batch",
            queue=site.run_config.queue,
            nodes=site.run_config.nodes,
            ranks_per_node=ranks_per_node,
            duration=site.run_config.duration,
            name=cancel_job.name,
        )
        _cancel_active_job(
            site,
            cancel_handle,
            timeout=cancel_timeout,
            poll_interval=float(site.run_config.poll_interval or 1),
        )
        cancel_result = cancel_handle.wait(
            timeout=cancel_timeout,
            poll_interval=site.run_config.poll_interval,
            check=False,
        )
        if cancel_result.status.state != "cancelled":
            raise LiveSlurmCanaryError("cancellation case did not reach cancelled")
        return {
            "fixtureId": SYNTHETIC_FIXTURE_ID,
            "success": {
                "schedulerJobId": str(success_handle.id),
                "states": states,
                "terminalState": success_result.status.state,
                "outputs": len(success_result.output_files(existing=True)),
                "traces": traces,
                **solver,
            },
            "cancellation": {
                "schedulerJobId": str(cancel_handle.id),
                "terminalState": cancel_result.status.state,
            },
        }
    finally:
        handles = _recover_handles(
            site,
            (cancel_handle, success_handle),
            (cancel_job, success_job),
        )
        _cancel_all(site, handles, timeout=cancel_timeout)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-config", required=True, type=Path)
    parser.add_argument("--site-profile")
    parser.add_argument("--project-path", required=True, type=Path)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--cancel-timeout", default=120.0, type=float)
    parser.add_argument("--acknowledge")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the canary and print one sanitized result for Deployment to retain."""

    args = _argument_parser().parse_args(argv)
    require_live_slurm_acknowledgement(
        args.acknowledge or os.environ.get("FREQUENSOLVE_LIVE_SLURM_ACKNOWLEDGEMENT")
    )
    site = Site(config_path=args.site_config, profile=args.site_profile)
    try:
        result = run_live_slurm_canary(
            site=site,
            project_path=args.project_path.expanduser().resolve(),
            run_token=args.run_token,
            timeout=args.timeout,
            cancel_timeout=args.cancel_timeout,
        )
    finally:
        site.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
