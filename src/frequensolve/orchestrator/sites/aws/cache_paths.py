"""Filesystem locations for FrequenSol cloud cache files."""

from pathlib import Path

from frequensolve._cloud_credentials import profile_credentials_filename
from frequensolve.storage import frequensolve_home


def cloud_cache_dir() -> Path:
    """Return the directory for cloud-specific cache and credentials."""
    path = frequensolve_home() / "cloud"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_domain(domain: str) -> str:
    return domain.replace(":", "_").replace("/", "_")


def cloud_credentials_path(profile_name: str | None = None) -> Path:
    """Return the current cloud credentials cache path.

    Profile-aware callers receive an isolated cache entry. The unscoped path is
    retained for direct ``CognitoAuth`` compatibility, but configured sites do
    not use or migrate it.

    Returns:
        Path to the credentials file under the cloud-specific cache directory.
    """

    cache_dir = cloud_cache_dir()
    if profile_name is not None:
        return cache_dir / profile_credentials_filename(profile_name)
    return cache_dir / "credentials"


def legacy_credentials_path() -> Path:
    """Return the pre-cloud-cache credentials path.

    Returns:
        Path used by older SDK versions for cloud credentials.
    """

    return frequensolve_home() / "credentials"


def cloud_config_cache_path(domain: str) -> Path:
    """Return the current cached cloud configuration path for a domain.

    Args:
        domain: Cloud API domain or host name.

    Returns:
        Path to the domain-specific configuration cache file.
    """

    return cloud_cache_dir() / f"config_{_safe_domain(domain)}.json"


def legacy_config_cache_path(domain: str) -> Path:
    """Return the legacy cached cloud configuration path for a domain.

    Args:
        domain: Cloud API domain or host name.

    Returns:
        Path used by older SDK versions for the domain-specific configuration
        cache file.
    """

    return frequensolve_home() / f"config_{_safe_domain(domain)}.json"
