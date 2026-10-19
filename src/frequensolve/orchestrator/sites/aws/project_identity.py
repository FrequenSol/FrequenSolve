"""Opt-in project bindings. Never fall back to legacy after identity failure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from frequensolve.project_identity import ProjectIdentity, local_identity


class ProjectIdentityMixin:
    def project_storage_key(self, root: str | Path) -> str:
        identity = local_identity(root)
        return identity.storage_prefix if identity else Path(root).name

    def _ensure_project_identity(
        self, root: str | Path, display_name: str
    ) -> ProjectIdentity | None:
        identity = local_identity(root)
        if identity is None:
            return None
        client = getattr(self, "graphql_client", None)
        if client is None:
            raise RuntimeError(
                "Immutable projects require an authenticated Cloud connection"
            )
        value: Any = client.request_project_identity(
            {
                "action": "bind",
                "manifest": identity.to_dict(),
                "display_name": display_name,
                "request_id": "bind_" + identity.project_id,
            }
        )
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != "project-identity.v1"
            or value.get("project_id") != identity.project_id
            or value.get("storage_prefix") != identity.storage_prefix
        ):
            raise RuntimeError(
                "Cloud returned a different project binding; no project inputs were uploaded"
            )
        return identity
