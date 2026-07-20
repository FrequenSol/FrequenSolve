"""Compatibility configuration objects for named HPC sites."""

from typing import Optional

from frequensolve.orchestrator.sites.config_file import load_site_presets
from frequensolve.orchestrator.sites.hpc.site import SlurmSiteConfig

__all__ = ["Stampede3Config"]


class Stampede3Config(SlurmSiteConfig):
    """Backward-compatible view of the built-in Stampede3 SLURM preset.

    New configuration should use ``type = "slurm"`` with
    ``preset = "stampede3"`` in ``site.toml``. This class remains available for
    direct-constructor and persisted-run compatibility.
    """

    def __init__(self, queue: Optional[str] = None):
        catalog = load_site_presets().get("presets", {})
        preset = catalog.get("stampede3")
        if not isinstance(preset, dict):  # pragma: no cover - packaged invariant
            raise ValueError("Built-in Stampede3 site preset was not found")
        partition = queue or preset["default_partition"]
        super().__init__(
            hostname=preset["hostname"],
            queue=partition,
            scheduler=preset.get("scheduler", "SLURM"),
            mpi_wrapper=preset.get("mpi_wrapper", "srun"),
            poll_interval=preset.get("poll_interval", 5),
            account=preset.get("account", ""),
            partitions=preset.get("partitions", {}),
        )

    def for_partition(self, partition: str) -> "Stampede3Config":
        """Return the compatibility config resolved for one partition."""

        return type(self)(partition)
