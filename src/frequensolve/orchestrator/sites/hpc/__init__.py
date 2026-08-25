"""SLURM/HPC site backends."""

from frequensolve.orchestrator.sites.hpc.config import Stampede3Config
from frequensolve.orchestrator.sites.hpc.enterprise import (
    EnterpriseHPCPreflightError,
    EnterpriseHPCPreflightResult,
    EnterpriseHPCProfile,
)
from frequensolve.orchestrator.sites.hpc.site import (
    SlurmLoginCredentials,
    SlurmPartitionConfig,
    SlurmRunConfig,
    SlurmSite,
    SlurmSiteConfig,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    normalize_slurm_state as _normalize_slurm_state,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    parse_sbatch_job_id as _parse_sbatch_job_id,
)
from frequensolve.orchestrator.sites.hpc.stampede3 import (
    Stampede3Site,
    TACCLoginCredentials,
)

__all__ = [
    "EnterpriseHPCPreflightError",
    "EnterpriseHPCPreflightResult",
    "EnterpriseHPCProfile",
    "SlurmLoginCredentials",
    "SlurmPartitionConfig",
    "SlurmRunConfig",
    "SlurmSite",
    "SlurmSiteConfig",
    "Stampede3Config",
    "Stampede3Site",
    "TACCLoginCredentials",
    "_normalize_slurm_state",
    "_parse_sbatch_job_id",
]
