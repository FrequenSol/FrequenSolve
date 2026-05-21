"""Execution site backends.

Optional backend dependencies are guarded at import time. If an optional
dependency is unavailable, the exported class remains present but raises an
install hint when constructed or otherwise used.
"""

from frequensolve._optional import optional_class
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    JobStatus,
    RunHandle,
    RunResult,
)
from frequensolve.orchestrator.sites.config_file import (
    DEFAULT_SITE_CONFIG_NAME,
    SITE_CONFIG_ENV_VAR,
    Site,
    load_site_config,
    site_config_path,
)

LocalSite = optional_class(
    "LocalSite",
    "frequensolve.orchestrator.sites.local.LocalSite",
    extra="parallel",
    dependencies=("dask", "distributed", "python-dotenv"),
    module=__name__,
)

SlurmLoginCredentials = optional_class(
    "SlurmLoginCredentials",
    "frequensolve.orchestrator.sites.hpc.SlurmLoginCredentials",
    extra="hpc",
    dependencies=("paramiko", "python-dotenv"),
    module=__name__,
)
SlurmRunConfig = optional_class(
    "SlurmRunConfig",
    "frequensolve.orchestrator.sites.hpc.SlurmRunConfig",
    extra="hpc",
    dependencies=("paramiko", "python-dotenv"),
    module=__name__,
)
SlurmSite = optional_class(
    "SlurmSite",
    "frequensolve.orchestrator.sites.hpc.SlurmSite",
    extra="hpc",
    dependencies=("paramiko", "python-dotenv"),
    module=__name__,
)
SlurmSiteConfig = optional_class(
    "SlurmSiteConfig",
    "frequensolve.orchestrator.sites.hpc.SlurmSiteConfig",
    extra="hpc",
    dependencies=("paramiko", "python-dotenv"),
    module=__name__,
)
Stampede3Site = optional_class(
    "Stampede3Site",
    "frequensolve.orchestrator.sites.stampede3.Stampede3Site",
    extra="hpc",
    dependencies=("paramiko", "python-dotenv"),
    module=__name__,
)
TACCLoginCredentials = optional_class(
    "TACCLoginCredentials",
    "frequensolve.orchestrator.sites.stampede3.TACCLoginCredentials",
    extra="hpc",
    dependencies=("paramiko", "python-dotenv"),
    module=__name__,
)

AWSSite = optional_class(
    "AWSSite",
    "frequensolve.orchestrator.sites.aws.AWSSite",
    extra="cloud",
    dependencies=("boto3", "botocore", "requests", "python-dotenv"),
    module=__name__,
)
AWSSiteConfig = optional_class(
    "AWSSiteConfig",
    "frequensolve.orchestrator.sites.aws.AWSSiteConfig",
    extra="cloud",
    dependencies=("boto3", "botocore", "requests", "python-dotenv"),
    module=__name__,
)
CognitoAuth = optional_class(
    "CognitoAuth",
    "frequensolve.orchestrator.sites.aws.CognitoAuth",
    extra="cloud",
    dependencies=("boto3", "botocore", "requests", "python-dotenv"),
    module=__name__,
)
GraphQLClient = optional_class(
    "GraphQLClient",
    "frequensolve.orchestrator.sites.aws.GraphQLClient",
    extra="cloud",
    dependencies=("boto3", "botocore", "requests", "python-dotenv"),
    module=__name__,
)

__all__ = [
    "AWSSite",
    "AWSSiteConfig",
    "BaseSite",
    "CognitoAuth",
    "GraphQLClient",
    "JobStatus",
    "LocalSite",
    "RunHandle",
    "RunResult",
    "DEFAULT_SITE_CONFIG_NAME",
    "SITE_CONFIG_ENV_VAR",
    "Site",
    "SlurmLoginCredentials",
    "SlurmRunConfig",
    "SlurmSite",
    "SlurmSiteConfig",
    "Stampede3Site",
    "TACCLoginCredentials",
    "load_site_config",
    "site_config_path",
]
