"""Strict reader for immutable packed-trace segment manifests."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from frequensolve.simulation.artifact_catalog import CombinedArtifactCatalog
from frequensolve.simulation.artifact_contract import (
    ArtifactContractError,
    ArtifactRecord,
)

__all__ = [
    "TRACE_MANIFEST_VERSION",
    "TracePackEntry",
    "TracePackManifest",
    "TracePackSegment",
]

TRACE_MANIFEST_VERSION = "fs-trace-manifest-2"
_PACKED_SCHEMA = "fs-traces-packed-1"
_FAMILY_KINDS = frozenset({"traces", "wavefields"})
_FAMILY_ROLES = frozenset({"simulated_traces", "sampled_wavefield"})


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactContractError(f"{name} must be an object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactContractError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArtifactContractError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ArtifactContractError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactContractError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ArtifactContractError(f"{name} must be a finite number")
    return result


def _relative_path(value: Any, name: str, root: Path) -> tuple[str, Path]:
    text = _string(value, name)
    if "\\" in text or "\0" in text or re.match(r"^[A-Za-z]:", text):
        raise ArtifactContractError(f"{name} must use portable '/' separators")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or "." in relative.parts
        or ".." in relative.parts
        or relative.as_posix() != text
    ):
        raise ArtifactContractError(f"{name} must remain relative and normalized")
    return text, root / Path(*relative.parts)


@dataclass(frozen=True)
class TracePackSegment:
    """One immutable packed HDF5 segment declared by a trace manifest."""

    id: str
    relative_path: str
    path: Path
    schema: str
    generation: str
    bytes: int
    artifact: ArtifactRecord


@dataclass(frozen=True)
class TracePackEntry:
    """Exact task/frequency location inside one immutable segment."""

    task: int
    frequency: complex
    segment_id: str
    dataset_number: int
    source_path: str
    source_generation: str


@dataclass(frozen=True)
class TracePackManifest:
    """Validated trace family and its explicit segment index."""

    artifact: ArtifactRecord
    generation: str
    family_id: str
    family_kind: str
    family_role: str
    trace_data_root: str
    index_path: str
    segments: tuple[TracePackSegment, ...]
    entries: tuple[TracePackEntry, ...]

    @classmethod
    def from_tasks(cls, catalog, frequencies, *, family_id="traces", fingerprints=None):
        """Resolve packed datasets from authoritative per-task artifact records."""
        segments = {}
        entries = []
        first = None
        for task, frequency in enumerate(frequencies, 1):
            task_catalog = catalog.task_catalog
            indexed = getattr(task_catalog, "tasks", None)
            row = (indexed if indexed is not None else task_catalog.results).get(task)
            if row is None or not row.successful:
                return None
            actual_frequency = (
                row.frequency if indexed is not None else row.partition.frequency
            )
            if actual_frequency != complex(frequency):
                return None
            if fingerprints is not None and any(
                row.fingerprints.get(key) != value
                for key, value in fingerprints.items()
            ):
                return None
            records = catalog.artifacts_for_task(task)
            matches = [
                record
                for record in records
                if record.id == family_id
                and record.representation == "packed_trace"
                and record.retention == "durable"
            ]
            if not matches:
                return None
            if len(matches) != 1:
                raise ArtifactContractError(f"task {task} has ambiguous packed traces")
            record = matches[0]
            if record.dataset_number is None:
                raise ArtifactContractError("packed task requires dataset_number")
            if record.path.stat().st_size != record.bytes:
                raise ArtifactContractError("packed task payload size changed")
            first = first or record
            segment_id = record.relative_path
            segments.setdefault(
                segment_id,
                TracePackSegment(
                    segment_id,
                    record.relative_path,
                    record.path,
                    record.schema,
                    record.generation or "unknown",
                    record.bytes,
                    record,
                ),
            )
            entries.append(
                TracePackEntry(
                    task,
                    complex(frequency),
                    segment_id,
                    record.dataset_number,
                    record.relative_path,
                    record.generation or "unknown",
                )
            )
        if first is None:
            return None
        return cls(
            first,
            first.generation or "unknown",
            family_id,
            "traces" if first.role == "simulated_traces" else "wavefields",
            first.role,
            "/trace_data",
            "/trace_index",
            tuple(segments.values()),
            tuple(entries),
        )

    @classmethod
    def read(
        cls,
        artifact: ArtifactRecord,
        *,
        catalog: CombinedArtifactCatalog,
        operation: str = "pack",
    ) -> "TracePackManifest":
        """Read one manifest and validate it against its operation catalog."""

        if artifact.representation != "packed_manifest":
            raise ArtifactContractError("trace pack artifact must be packed_manifest")
        if artifact.schema != TRACE_MANIFEST_VERSION:
            raise ArtifactContractError(
                f"unsupported trace manifest schema {artifact.schema!r}"
            )
        try:
            size = artifact.path.stat().st_size
        except OSError:
            raise
        if size != artifact.bytes:
            raise ArtifactContractError(
                f"trace manifest {artifact.relative_path!r} has {size} bytes; "
                f"expected {artifact.bytes}"
            )
        try:
            raw = json.loads(artifact.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactContractError(
                f"invalid trace manifest JSON: {artifact.relative_path}"
            ) from exc
        raw = _mapping(raw, "trace manifest")
        unknown = set(raw).difference(
            {
                "schema",
                "generation",
                "family",
                "trace_data_root",
                "index_path",
                "segments",
                "entries",
                "grid",
                "properties",
            }
        )
        if unknown:
            raise ArtifactContractError(
                "trace manifest contains unsupported fields: "
                + ", ".join(sorted(unknown))
            )
        if raw.get("schema") != TRACE_MANIFEST_VERSION:
            raise ArtifactContractError(
                f"trace manifest schema must be {TRACE_MANIFEST_VERSION!r}"
            )
        generation = _string(raw.get("generation"), "trace manifest generation")
        if artifact.generation != generation:
            raise ArtifactContractError(
                "trace manifest generation does not match its artifact record"
            )
        family = _mapping(raw.get("family"), "trace manifest family")
        if set(family) != {"id", "kind", "role"}:
            raise ArtifactContractError(
                "trace manifest family must contain exactly id, kind, and role"
            )
        family_id = _string(family.get("id"), "trace manifest family.id")
        family_kind = _string(family.get("kind"), "trace manifest family.kind")
        family_role = _string(family.get("role"), "trace manifest family.role")
        if family_kind not in _FAMILY_KINDS or family_role not in _FAMILY_ROLES:
            raise ArtifactContractError("trace manifest family kind or role is invalid")
        if (family_kind == "traces") != (family_role == "simulated_traces"):
            raise ArtifactContractError("trace manifest family kind and role disagree")
        if family_id != artifact.id or family_role != artifact.role:
            raise ArtifactContractError(
                "trace manifest family does not match its artifact record"
            )
        trace_data_root = _string(
            raw.get("trace_data_root"), "trace manifest trace_data_root"
        )
        index_path = _string(raw.get("index_path"), "trace manifest index_path")
        if trace_data_root != "/trace_data" or index_path != "/trace_index":
            raise ArtifactContractError(
                "trace manifest must use /trace_data and /trace_index roots"
            )

        operation_records = catalog.artifacts_for_operation(operation)
        if artifact not in operation_records:
            raise ArtifactContractError(
                "trace manifest artifact is not owned by the requested operation"
            )
        records_by_id: dict[str, list[ArtifactRecord]] = {}
        for record in operation_records:
            records_by_id.setdefault(record.id, []).append(record)

        raw_segments = raw.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ArtifactContractError(
                "trace manifest segments must be a nonempty array"
            )
        root = catalog.result_path
        segments = []
        segment_ids = set()
        for index, value in enumerate(raw_segments):
            value = _mapping(value, f"trace manifest segments[{index}]")
            if set(value) != {"id", "path", "schema", "generation", "bytes"}:
                raise ArtifactContractError(
                    f"trace manifest segments[{index}] has invalid fields"
                )
            segment_id = _string(value.get("id"), f"segments[{index}].id")
            if segment_id in segment_ids:
                raise ArtifactContractError("trace manifest segment ids must be unique")
            segment_ids.add(segment_id)
            relative_path, path = _relative_path(
                value.get("path"), f"segments[{index}].path", root
            )
            schema = _string(value.get("schema"), f"segments[{index}].schema")
            segment_generation = _string(
                value.get("generation"), f"segments[{index}].generation"
            )
            byte_count = _integer(value.get("bytes"), f"segments[{index}].bytes")
            matches = records_by_id.get(segment_id, [])
            if len(matches) != 1:
                raise ArtifactContractError(
                    f"trace segment {segment_id!r} resolves to {len(matches)} "
                    "operation artifacts"
                )
            record = matches[0]
            if (
                record.role != "trace_data"
                or record.representation != "packed_hdf5"
                or record.schema != _PACKED_SCHEMA
                or schema != record.schema
                or relative_path != record.relative_path
                or segment_generation != record.generation
                or byte_count != record.bytes
            ):
                raise ArtifactContractError(
                    f"trace segment {segment_id!r} disagrees with its operation artifact"
                )
            segments.append(
                TracePackSegment(
                    id=segment_id,
                    relative_path=relative_path,
                    path=path,
                    schema=schema,
                    generation=segment_generation,
                    bytes=byte_count,
                    artifact=record,
                )
            )
        if tuple(artifact.dependencies) != tuple(segment.id for segment in segments):
            raise ArtifactContractError(
                "trace manifest dependencies must match segment producer order"
            )

        raw_entries = raw.get("entries")
        if not isinstance(raw_entries, list):
            raise ArtifactContractError("trace manifest entries must be an array")
        entries = []
        identities = set()
        locations = set()
        for index, value in enumerate(raw_entries):
            value = _mapping(value, f"trace manifest entries[{index}]")
            required = {
                "task",
                "frequency",
                "segment_id",
                "dataset_number",
                "source_path",
                "source_generation",
            }
            if set(value) != required:
                raise ArtifactContractError(
                    f"trace manifest entries[{index}] has invalid fields"
                )
            task = _integer(value.get("task"), f"entries[{index}].task", minimum=1)
            frequency = _mapping(value.get("frequency"), f"entries[{index}].frequency")
            if set(frequency) != {"real", "imag"}:
                raise ArtifactContractError(
                    f"entries[{index}].frequency must contain real and imag"
                )
            complex_frequency = complex(
                _number(frequency.get("real"), f"entries[{index}].frequency.real"),
                _number(frequency.get("imag"), f"entries[{index}].frequency.imag"),
            )
            segment_id = _string(
                value.get("segment_id"), f"entries[{index}].segment_id"
            )
            if segment_id not in segment_ids:
                raise ArtifactContractError(
                    f"trace entry references unknown segment {segment_id!r}"
                )
            dataset_number = _integer(
                value.get("dataset_number"),
                f"entries[{index}].dataset_number",
                minimum=1,
            )
            source_path, _ = _relative_path(
                value.get("source_path"), f"entries[{index}].source_path", root
            )
            source_generation = _string(
                value.get("source_generation"),
                f"entries[{index}].source_generation",
            )
            identity = (task, complex_frequency)
            location = (segment_id, dataset_number)
            if identity in identities:
                raise ArtifactContractError(
                    "trace manifest task/frequency entries must be unique"
                )
            if location in locations:
                raise ArtifactContractError(
                    "trace manifest segment dataset locations must be unique"
                )
            identities.add(identity)
            locations.add(location)
            entries.append(
                TracePackEntry(
                    task=task,
                    frequency=complex_frequency,
                    segment_id=segment_id,
                    dataset_number=dataset_number,
                    source_path=source_path,
                    source_generation=source_generation,
                )
            )
        return cls(
            artifact=artifact,
            generation=generation,
            family_id=family_id,
            family_kind=family_kind,
            family_role=family_role,
            trace_data_root=trace_data_root,
            index_path=index_path,
            segments=tuple(segments),
            entries=tuple(entries),
        )

    def entries_for(
        self, expected: Sequence[complex | float]
    ) -> tuple[TracePackEntry, ...]:
        """Return exact entries in requested task order, or an empty tuple."""

        by_identity = {(entry.task, entry.frequency): entry for entry in self.entries}
        selected = tuple(
            by_identity.get((task, complex(frequency)))
            for task, frequency in enumerate(expected, start=1)
        )
        if any(entry is None for entry in selected):
            return ()
        return tuple(entry for entry in selected if entry is not None)

    def selected_segments(
        self, entries: Sequence[TracePackEntry]
    ) -> tuple[TracePackSegment, ...]:
        """Return referenced segments in manifest producer order."""

        wanted = {entry.segment_id for entry in entries}
        return tuple(segment for segment in self.segments if segment.id in wanted)
