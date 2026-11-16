"""Opt-in local project identity; this document is not Cloud access authority.

Legacy projects are unchanged. Opt-in Projects bind Cloud storage and saved plans.
Import this module explicitly while the coordinated Cloud capability is gated.
"""

from __future__ import annotations

import json
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4

__all__ = ["ProjectIdentity", "legacy_reference", "reference_token"]

SCHEMA_VERSION = "project-identity.v1"
DIRECTORY = ".frequensolve"
FILENAME = "project-identity.json"
MAX_BYTES = 4096
COPY_RECOVERY = (
    "Finish the project copy before uploading or loading it as a completed copy. "
    "The destination has an active or incomplete copy claim. Wait for its writer; "
    "if it failed, keep the partial folder for inspection and retry Project.copy "
    "from the intact source into a different new, empty destination. Do not "
    "remove the claim or replace its project ID to bypass this check."
)
_UUID4 = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)


def _project_id(value: Any) -> str:
    if not isinstance(value, str) or not _UUID4.fullmatch(value):
        raise ValueError("INVALID_PROJECT_IDENTITY")
    return value


def legacy_reference(key: str) -> dict[str, str]:
    """Wrap an existing key without normalizing it or guessing from its shape."""
    if not isinstance(key, str) or not key or re.search(r"[\x00-\x1f\x7f]", key):
        raise ValueError("INVALID_PROJECT_IDENTITY")
    try:
        if len(key.encode("utf-16-le")) > 512:
            raise ValueError("INVALID_PROJECT_IDENTITY")
    except UnicodeEncodeError as exc:
        raise ValueError("INVALID_PROJECT_IDENTITY") from exc
    return {"kind": "legacy", "project_key": key}


def reference_token(value: Mapping[str, Any]) -> str:
    """Versioned reference token; never substitute it for an old storage key."""
    if not isinstance(value, dict):
        raise ValueError("INVALID_PROJECT_IDENTITY")
    if value.get("kind") == "immutable" and set(value) == {"kind", "project_id"}:
        parts = ["immutable", _project_id(value["project_id"])]
    elif value.get("kind") == "legacy" and set(value) == {"kind", "project_key"}:
        parts = ["legacy", legacy_reference(value["project_key"])["project_key"]]
    else:
        raise ValueError("INVALID_PROJECT_IDENTITY")
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


@contextmanager
def _directory(root: Path, *, create: bool) -> Iterator[int]:
    # Relative directory descriptors keep a replaced/symlinked metadata path
    # from redirecting the operation. Unsupported platforms fail closed.
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise OSError(
            "Safe project identity file operations are unavailable on this platform"
        )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if create:
            try:
                os.mkdir(DIRECTORY, mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
        directory_fd = os.open(
            DIRECTORY, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd
        )
        try:
            yield directory_fd
        finally:
            os.close(directory_fd)
    finally:
        os.close(root_fd)


def _unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("INVALID_PROJECT_IDENTITY")
        result[key] = value
    return result


@dataclass(frozen=True)
class ProjectIdentity:
    """Immutable local UUID candidate, independent of display name and folder.

    A copied/exported identity never proves tenant membership or ownership.
    Cloud registration must authenticate the actor and conditionally claim it.
    """

    project_id: str

    def __post_init__(self) -> None:
        _project_id(self.project_id)

    @classmethod
    def new(cls) -> ProjectIdentity:
        return cls(str(uuid4()))

    @classmethod
    def from_dict(cls, value: Any) -> ProjectIdentity:
        if (
            not isinstance(value, dict)
            or set(value) != {"schema_version", "project_id"}
            or value["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError("INVALID_PROJECT_IDENTITY")
        return cls(_project_id(value["project_id"]))

    def to_dict(self) -> dict[str, str]:
        return {"schema_version": SCHEMA_VERSION, "project_id": self.project_id}

    @property
    def storage_prefix(self) -> str:
        return "pi_" + self.project_id

    def reference(self) -> dict[str, str]:
        return {"kind": "immutable", "project_id": self.project_id}

    def save_new(self, project_root: str | Path) -> Path:
        """Exclusively create a manifest. Never replace or silently rebind one.

        The root must already exist. A failed/crashed write remains invalid;
        loading it fails rather than silently generating a replacement UUID.
        """
        root = Path(project_root).expanduser().absolute()
        payload = (json.dumps(self.to_dict(), separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        with _directory(root, create=True) as directory_fd:
            fd = os.open(
                FILENAME,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(directory_fd)
        return root / DIRECTORY / FILENAME

    @classmethod
    def load(cls, project_root: str | Path) -> ProjectIdentity:
        """Read a bounded regular manifest; missing/corrupt state is an error."""
        root = Path(project_root).expanduser().absolute()
        with _directory(root, create=False) as directory_fd:
            fd = os.open(
                FILENAME,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory_fd,
            )
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
                    raise ValueError("INVALID_PROJECT_IDENTITY")
                payload = stream.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES:
            raise ValueError("INVALID_PROJECT_IDENTITY")
        try:
            return cls.from_dict(
                json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_fields)
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("INVALID_PROJECT_IDENTITY") from exc

    def fork_to(self, project_root: str | Path) -> ProjectIdentity:
        """Create a distinct identity for a new project, leaving this one intact."""
        fork = type(self).new()
        fork.save_new(project_root)
        return fork


def local_identity(root: str | Path) -> ProjectIdentity | None:
    root = Path(root)
    manifest = root / ".frequensolve" / "project-identity.json"
    if os.path.lexists(manifest):
        return ProjectIdentity.load(root)
    if os.path.lexists(root / DIRECTORY):
        raise ValueError(
            "Project identity manifest is missing; restore it before uploading"
        )
    # Saved projects retain the expectation even if a manifest was lost.
    for path in root.glob("*.json"):
        if path.stat().st_size > 1048576:
            continue
        try:
            value = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        if isinstance(value, dict) and value.get("project_id") is not None:
            raise ValueError(
                "Project identity manifest is missing; restore it before uploading"
            )
    return None
