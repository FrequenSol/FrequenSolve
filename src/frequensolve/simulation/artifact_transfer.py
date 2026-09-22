"""Exact transfer planning for typed Sauce artifacts and collections."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, List, Mapping, Protocol, Sequence

from frequensolve.simulation.artifact_contract import (
    COLLECTION_CONTRACT_VERSION,
    ArtifactContractError,
    ArtifactRecord,
    ArtifactRequest,
)

__all__ = [
    "CollectionMember",
    "collection_parts",
    "fetch_artifact_payloads",
    "select_transfer_artifacts",
]

_COLLECTION_MANIFEST_MAX_BYTES = 16 * 1024 * 1024
_EXPLICIT_ROLES = frozenset(
    {
        "wavefield",
        "wavefields",
        "forward_wavefield",
        "adjoint_wavefield",
        "restart",
        "restart_checkpoint",
        "checkpoint",
        "debug",
        "diagnostic",
    }
)


class _Catalog(Protocol):
    def query(
        self,
        *,
        id: str | None = None,
        role: str | None = None,
        representation: str | None = None,
        retention: str | None = None,
        task: int | None = None,
    ) -> List[ArtifactRecord]: ...

    def select(
        self,
        request: ArtifactRequest,
        *,
        task: int | None = None,
    ) -> List[ArtifactRecord]: ...


@dataclass(frozen=True)
class CollectionMember:
    """One exact physical payload owned by a collection manifest."""

    relative_path: str
    path: Path
    bytes: int


def _portable_relative_path(value: object, *, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ArtifactContractError(f"{name} must be a non-empty string")
    if "\\" in value or "\0" in value or re.match(r"^[A-Za-z]:", value):
        raise ArtifactContractError(f"{name} must use portable '/' separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ArtifactContractError(f"{name} must remain relative and normalized")
    return path


def _requires_explicit_selection(record: ArtifactRecord) -> bool:
    return record.role.lower() in _EXPLICIT_ROLES


def select_transfer_artifacts(
    catalog: _Catalog,
    *,
    requests: Sequence[ArtifactRequest] = (),
    include_defaults: bool = True,
) -> tuple[ArtifactRecord, ...]:
    """Select exact records, excluding retained auxiliary data by default.

    Default transfer contains durable final products. Wavefields, restart
    checkpoints, cache/transient data, and debug/diagnostic artifacts require
    an explicit typed request.
    """

    records: Iterable[ArtifactRecord] = ()
    if include_defaults:
        records = (
            record
            for record in catalog.query(retention="durable")
            if not _requires_explicit_selection(record)
        )
    selected = list(records)
    for request in requests:
        selected.extend(catalog.select(request))

    result = []
    seen = set()
    for record in selected:
        key = (
            record.id,
            record.role,
            record.representation,
            record.relative_path,
            record.generation,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return _dependency_closure(catalog, result)


def _catalog_scopes(
    catalog: _Catalog,
) -> Mapping[object, tuple[ArtifactRecord, ...]]:
    """Return dependency-local producer ordering for tasks and operations."""

    artifact_scopes = getattr(catalog, "artifact_scopes", None)
    if callable(artifact_scopes):
        scopes = artifact_scopes()
        if not isinstance(scopes, Mapping):
            raise ArtifactContractError("artifact catalog scopes must be a mapping")
        return {scope: tuple(records) for scope, records in scopes.items()}

    raw_tasks = getattr(catalog, "tasks", None)
    if raw_tasks is None:
        raw_tasks = getattr(catalog, "results", None)
    if not isinstance(raw_tasks, Mapping):
        return {}
    return {
        ("task", int(task)): tuple(catalog.query(task=int(task))) for task in raw_tasks
    }


def _dependency_closure(
    catalog: _Catalog,
    selected: Sequence[ArtifactRecord],
) -> tuple[ArtifactRecord, ...]:
    """Expand JSON dependencies and reject dependency-blind indexed XMF."""

    scopes = _catalog_scopes(catalog)
    scope_by_record = {
        id(record): scope for scope, records in scopes.items() for record in records
    }
    result = list(selected)
    selected_keys = {
        (scope_by_record.get(id(record)), record.id, record.representation)
        for record in result
    }

    position = 0
    while position < len(result):
        record = result[position]
        position += 1
        scope = scope_by_record.get(id(record))
        if record.dependencies:
            candidates = scopes.get(scope, ())
            for dependency in record.dependencies:
                matches = [item for item in candidates if item.id == dependency]
                if len(matches) != 1:
                    raise ArtifactContractError(
                        f"artifact {record.id!r} dependency {dependency!r} "
                        f"resolved to {len(matches)} records in scope {scope!r}"
                    )
                match = matches[0]
                key = (scope, match.id, match.representation)
                if key not in selected_keys:
                    selected_keys.add(key)
                    result.append(match)

    for record in result:
        if record.role != "visualization" or record.representation != "xmf":
            continue
        scope = scope_by_record.get(id(record))
        if not any(
            scope_by_record.get(id(item)) == scope and item.role == "visualization_data"
            for item in result
        ):
            raise ArtifactContractError(
                "XMF transfer requires its visualization_data companion"
            )
    return tuple(result)


def collection_parts(record: ArtifactRecord) -> tuple[CollectionMember, ...]:
    """Read exact member paths from a supported local collection manifest."""

    if record.representation != "collection_manifest":
        raise ArtifactContractError(
            f"artifact {record.id!r} is not a collection manifest"
        )
    try:
        size = record.path.stat().st_size
    except OSError:
        raise
    if size != record.bytes:
        raise ArtifactContractError(
            f"collection manifest {record.relative_path!r} has {size} bytes; "
            f"expected {record.bytes}"
        )
    if size > _COLLECTION_MANIFEST_MAX_BYTES:
        raise ArtifactContractError(
            f"collection manifest exceeds {_COLLECTION_MANIFEST_MAX_BYTES} bytes"
        )
    try:
        payload = json.loads(record.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactContractError(
            f"invalid collection manifest JSON: {record.relative_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise ArtifactContractError("collection manifest must be an object")
    attributes = payload.get("attributes")
    storage = payload.get("storage")
    if not isinstance(attributes, dict) or attributes.get("schema") != record.schema:
        raise ArtifactContractError(
            "collection manifest schema does not match its artifact record"
        )
    if record.schema != COLLECTION_CONTRACT_VERSION or not isinstance(storage, dict):
        raise ArtifactContractError(
            f"unsupported collection manifest schema {record.schema!r}"
        )
    raw_parts = storage.get("parts")
    if not isinstance(raw_parts, list):
        raise ArtifactContractError(
            "collection manifest storage.parts must be an array"
        )
    part_count = storage.get("part_count")
    if (
        isinstance(part_count, bool)
        or not isinstance(part_count, int)
        or part_count != len(raw_parts)
    ):
        raise ArtifactContractError(
            "collection manifest part_count does not match storage.parts"
        )

    manifest = PurePosixPath(record.relative_path)
    result = []
    seen = set()

    def append_member(raw_member: object, name: str) -> None:
        if not isinstance(raw_member, dict):
            raise ArtifactContractError(f"collection manifest {name} must be an object")
        relative = _portable_relative_path(
            raw_member.get("path"),
            name=f"collection manifest {name}.path",
        )
        combined = manifest.parent / relative
        normalized = _portable_relative_path(
            combined.as_posix(),
            name=f"collection manifest {name}.path",
        )
        raw_bytes = raw_member.get("bytes")
        if isinstance(raw_bytes, bool) or not isinstance(raw_bytes, int):
            raise ArtifactContractError(
                f"collection manifest {name}.bytes must be an integer"
            )
        if raw_bytes < 0:
            raise ArtifactContractError(
                f"collection manifest {name}.bytes must be nonnegative"
            )
        path_text = normalized.as_posix()
        if path_text in seen:
            raise ArtifactContractError(
                "collection manifest member paths must be unique"
            )
        seen.add(path_text)
        result.append(
            CollectionMember(
                relative_path=path_text,
                path=record.path.parent / Path(*relative.parts),
                bytes=raw_bytes,
            )
        )

    metadata = storage.get("metadata")
    if metadata is not None:
        append_member(metadata, "storage.metadata")
    for index, raw_part in enumerate(raw_parts):
        append_member(raw_part, f"storage.parts[{index}]")
    return tuple(result)


def fetch_artifact_payloads(
    catalog: _Catalog,
    *,
    fetch_files: Callable[[Sequence[str]], object],
    requests: Sequence[ArtifactRequest] = (),
    include_defaults: bool = True,
) -> list[Path]:
    """Fetch exact selected payloads, expanding collections after manifests."""

    records = select_transfer_artifacts(
        catalog,
        requests=requests,
        include_defaults=include_defaults,
    )
    manifests = [
        record for record in records if record.representation == "collection_manifest"
    ]
    if manifests:
        fetch_files([record.relative_path for record in manifests])

    members = [member for record in manifests for member in collection_parts(record)]
    direct = [
        record for record in records if record.representation != "collection_manifest"
    ]
    payloads = [record.relative_path for record in direct]
    payloads.extend(member.relative_path for member in members)
    if payloads:
        fetch_files(payloads)

    fetched = []
    expected = [(record.path, record.bytes) for record in records]
    expected.extend((member.path, member.bytes) for member in members)
    for path, byte_count in expected:
        try:
            actual = path.stat().st_size
        except OSError as exc:
            raise FileNotFoundError(f"Fetched artifact is missing: {path}") from exc
        if actual != byte_count:
            raise ArtifactContractError(
                f"Fetched artifact {path} has {actual} bytes; expected {byte_count}"
            )
        fetched.append(path)
    return fetched
