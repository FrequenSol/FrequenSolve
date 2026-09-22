"""Supported, provider-independent access to the direct adaptive engine.

Adapters own transport, credentials, staging and publication. The packaged runner
and sweep template remain the single implementation used by direct SSH jobs.
"""

from dataclasses import dataclass
from importlib.resources import files
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from importlib.abc import Traversable

ENGINE_VERSION = "adaptive-scheduler.v1"


@dataclass(frozen=True)
class AdaptivePool:
    nodes: int
    ranks_per_node: int
    threads_per_rank: int
    memory_mib_per_node: int
    wall_time_seconds: int
    partition: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AdaptivePool":
        import re

        fields = {
            "nodes",
            "ranks_per_node",
            "threads_per_rank",
            "memory_mib_per_node",
            "wall_time_seconds",
        }
        if not isinstance(value, Mapping) or set(value) != {*fields, "partition"}:
            raise ValueError(
                "Adaptive pool requires an explicit uniform resource envelope"
            )
        for key in fields:
            if type(value[key]) is not int or value[key] < 1:
                raise ValueError(f"Invalid adaptive pool {key}")
        if not isinstance(value["partition"], str) or not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", value["partition"]
        ):
            raise ValueError("Invalid adaptive pool partition")
        return cls(**dict(value))

    def sbatch_arguments(self) -> list[str]:
        return [
            "--nodes",
            str(self.nodes),
            "--ntasks-per-node",
            str(self.ranks_per_node),
            "--cpus-per-task",
            str(self.threads_per_rank),
            "--mem",
            f"{self.memory_mib_per_node}M",
            "--partition",
            self.partition,
            "--time",
            str((self.wall_time_seconds + 59) // 60),
        ]

    @property
    def memory_per_rank_gib(self) -> float:
        return self.memory_mib_per_node / self.ranks_per_node / 1024


def scheduler_source() -> "Traversable":
    """Return the exact packaged script transferred by direct SSH submission."""
    source = files("frequensolve")
    for part in (
        "orchestrator",
        "sites",
        "hpc",
        "templates",
        "sweep",
        "adaptive_scheduler.py",
    ):
        source = source.joinpath(part)
    return source


def render_sweep(*, scheduler_version: str, **context: Any) -> str:
    """Render the existing initialization/sweep/imaging/packing implementation."""
    from jinja2 import Environment

    if scheduler_version != ENGINE_VERSION:
        raise ValueError("Unsupported adaptive scheduler version")
    template = files("frequensolve")
    for part in (
        "orchestrator",
        "sites",
        "hpc",
        "templates",
        "sweep",
        "adaptive_sweep.sh",
    ):
        template = template.joinpath(part)
    return Environment().from_string(template.read_text()).render(**context)


def main() -> None:
    # Run the same file rather than maintaining a second scheduler entrypoint.
    import runpy
    from importlib.resources import as_file

    with as_file(scheduler_source()) as path:
        runpy.run_path(str(path), run_name="__main__")


def render_allocation(
    *,
    pool: AdaptivePool,
    job_file: str,
    run_path: str,
    output: str,
    executable: str,
    mpi: str,
    mpi_args: list[str],
    task_count: int,
    imaging: bool = False,
    pack: bool = True,
    fresh: bool = True,
    health_check_timeout: str | None = None,
) -> str:
    """Render the direct engine inside an already-granted uniform allocation.

    Transport adapters stage the inputs and supply launchers. Defaults match the
    direct SSH sweep: single-frequency sizing skip, 1.5 memory cushion, boosting
    up to eightfold, and up to four failed frequencies before aborting the sweep.
    """
    import json
    import shlex
    from pathlib import Path

    if type(task_count) is not int or task_count < 1:
        raise ValueError("Adaptive sweep requires a positive task count")
    ranks = pool.nodes * pool.ranks_per_node
    sizing = str(Path(output).parent / "FS_sizing.json")
    config = {
        "version": ENGINE_VERSION,
        "executable": executable,
        "mpi": mpi,
        "mpi_args": list(mpi_args),
        "fresh": fresh,
        "total_ranks": ranks,
        "omp_threads": pool.threads_per_rank,
        "mem_per_rank_gib": pool.memory_per_rank_gib,
        "job_task_count": task_count,
        "task_indices": list(range(1, task_count + 1)),
        "skip_sizing": task_count == 1,
        "min_ranks": 1,
        "round_to": 1,
        "cap_fraction": 1.0,
        "max_ranks_per_task": ranks,
        "mem_cushion": 1.5,
        "boost_max_factor": 8.0,
        "failure_tolerance": 4,
        "sizing_json": sizing,
        "launch_delay_seconds": 0.25,
    }
    return render_sweep(
        scheduler_version=ENGINE_VERSION,
        batch_job=False,
        job_json=shlex.quote(job_file),
        run_path=run_path,
        run_path_shell=shlex.quote(run_path),
        dir_out_shell=shlex.quote(output),
        sizing_json_shell=shlex.quote(sizing),
        skip_sizing=int(task_count == 1),
        runtime_setup=[],
        mpi_async_progress_setup=[],
        mpi_shell=shlex.quote(mpi),
        mpi_args_shell=shlex.join(mpi_args),
        n_procs=ranks,
        init_ranks=ranks,
        n_threads=pool.threads_per_rank,
        n_tasks=task_count,
        n_job_tasks=task_count,
        executable_shell=shlex.quote(executable),
        fresh=fresh,
        scheduler_config_shell=shlex.quote(json.dumps(config, indent=2)),
        scheduler_runner=shlex.quote(str(scheduler_source())),
        smooth_only=False,
        imaging_job=imaging,
        pack_job=pack,
        mpi_health_check_timeout=health_check_timeout,
        mpi_health_check_timeout_shell=shlex.quote(health_check_timeout or ""),
    )
