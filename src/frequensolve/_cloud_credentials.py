"""Dependency-light helpers for profile-bound Cloud credential caches."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

CREDENTIAL_CACHE_BINDING_KEY = "_frequensolve_cache_binding"
CREDENTIAL_CACHE_VERSION = 1


def profile_credentials_filename(profile_name: str) -> str:
    """Return a filesystem-safe, stable credential filename for a profile."""

    digest = hashlib.sha256(profile_name.encode("utf-8")).hexdigest()
    return f"credentials-profile-{digest}.json"


def profile_credentials_path(root: Path, profile_name: str) -> Path:
    """Return the profile-scoped credential path below a FrequenSolve home."""

    return root / "cloud" / profile_credentials_filename(profile_name)


def credential_cache_binding(
    *,
    profile_name: str,
    domain: str,
    region: str,
    user_pool_id: str,
    client_id: str,
    identity_pool_id: str,
) -> dict[str, Any]:
    """Build the non-secret identity boundary stored beside cached tokens."""

    return {
        "version": CREDENTIAL_CACHE_VERSION,
        "profile": profile_name,
        "domain": domain.casefold(),
        "region": region,
        "user_pool_id": user_pool_id,
        "client_id": client_id,
        "identity_pool_id": identity_pool_id,
    }


def credential_cache_binding_matches(
    document: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    """Return whether a cache document is bound to exactly ``expected``."""

    return document.get(CREDENTIAL_CACHE_BINDING_KEY) == expected
