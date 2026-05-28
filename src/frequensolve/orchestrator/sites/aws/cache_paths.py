"""Filesystem locations for FrequenSol cloud cache files."""

from pathlib import Path

from frequensolve.storage import frequensolve_home


def cloud_cache_dir() -> Path:
    """Return the directory for cloud-specific cache and credentials."""
    path = frequensolve_home() / "cloud"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_domain(domain: str) -> str:
    return domain.replace(":", "_").replace("/", "_")


def cloud_credentials_path() -> Path:
    return cloud_cache_dir() / "credentials"


def legacy_credentials_path() -> Path:
    return frequensolve_home() / "credentials"


def cloud_config_cache_path(domain: str) -> Path:
    return cloud_cache_dir() / f"config_{_safe_domain(domain)}.json"


def legacy_config_cache_path(domain: str) -> Path:
    return frequensolve_home() / f"config_{_safe_domain(domain)}.json"
