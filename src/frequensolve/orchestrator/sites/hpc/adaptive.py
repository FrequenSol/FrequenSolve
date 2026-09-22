"""Supported, provider-independent access to the direct adaptive engine.

Adapters own transport, credentials, staging and publication. The packaged runner
and sweep template remain the single implementation used by direct SSH jobs.
"""

from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Mapping

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


def scheduler_source():
    """Return the exact packaged script transferred by direct SSH submission."""
    return files("frequensolve.orchestrator.sites.hpc").joinpath(
        "templates", "sweep", "adaptive_scheduler.py"
    )


def render_sweep(*, scheduler_version: str, **context: Any) -> str:
    """Render the existing initialization/sweep/imaging/packing implementation."""
    from jinja2 import Environment, PackageLoader

    if scheduler_version != ENGINE_VERSION:
        raise ValueError("Unsupported adaptive scheduler version")
    environment = Environment(
        loader=PackageLoader("frequensolve.orchestrator.sites.hpc", "templates")
    )
    return environment.get_template("sweep/adaptive_sweep.sh").render(**context)


def main():
    # Run the same file rather than maintaining a second scheduler entrypoint.
    import runpy
    from importlib.resources import as_file

    with as_file(scheduler_source()) as path:
        runpy.run_path(str(path), run_name="__main__")
