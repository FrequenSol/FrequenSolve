"""Observed data references, data spaces, and objective-side vectors.

This module binds the objective side of an imaging problem:

* :class:`TraceStoreRef` is one ``HDF5TraceStore`` descriptor from the
  ``fs-imaging-1`` contract.
* :class:`ObservedGroup` is the observed side of one misfit receiver group.
* :class:`ObservedData` normalizes the public ways of naming observed data
  (a forward job, a path stem, a mapping, or a :class:`TraceDataset`) and
  resolves them against a simulation's receiver groups.
* :class:`DataSpace` fixes the vector layout of complex frequency-domain
  trace data and :class:`DataVector` wraps one such vector.
* :meth:`DataVector.write_objective_vector` and
  :meth:`DataVector.read_objective_vector` implement the
  ``fs-objective-vector-3`` manifest and shard format used by Sauce's
  ``fwi_operator`` ``jvp``/``vjp`` actions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr

__all__ = [
    "DataSpace",
    "DataVector",
    "ObservedData",
    "ObservedGroup",
    "TermLayout",
    "TraceStoreRef",
    "canonical_json_sha256",
    "file_sha256",
    "objective_layout_fingerprint",
]

MissingPolicy = Literal["error", "zero", "zeros", "warn", "warning"]
SourceBasis = Literal["source_encoding", "source_geometry"]

_MISSING_POLICIES = {"error", "zero", "zeros", "warn", "warning"}
_SOURCE_BASES = {"source_encoding", "source_geometry"}
_DATA_DIMS = ("frequency", "source", "component", "receiver")
_PACKED_TRACE_FILE = "traces.h5"
_OBJECTIVE_VECTOR_SCHEMA = "fs-objective-vector-3"
_LAYOUT_CHUNK = 4096


# ---------------------------------------------------------------------------
# Hash helpers shared by the objective-vector contract
# ---------------------------------------------------------------------------


def _normalize_hash(value: str) -> str:
    text = str(value).strip().lower()
    if len(text) == 64:
        text = f"sha256:{text}"
    return text


def canonical_json_sha256(payload: Any) -> str:
    """Return ``sha256:<hex>`` of the canonical compact sorted JSON of ``payload``.

    Sauce hashes ``nlohmann::json::dump()`` output, which is compact UTF-8
    with object keys in sorted order. ``json.dumps`` with sorted keys and no
    separators produces the same bytes for the integer and string payloads
    used by the objective contracts.
    """

    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Union[str, Path]) -> str:
    """Return ``sha256:<hex>`` over the exact bytes of ``path``."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def objective_layout_fingerprint(
    n_rows: int, row_ids: np.ndarray, coordinate_keys: np.ndarray
) -> str:
    """Reproduce Sauce's global objective layout fingerprint.

    Sauce splits the canonical row catalog into 4096-row chunks by row id,
    hashes each chunk as ``{"keys": [k1, k2, k3, ...]}`` in row order, and
    hashes the catalog ``{"n_rows": n, "<first row id>": "<chunk hash>"}``.
    Missing rows inside a chunk hash as zero keys, exactly as the Fortran
    implementation zero-fills its chunk buffer.
    """

    n_rows = int(n_rows)
    ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
    keys = np.asarray(coordinate_keys, dtype=np.int64).reshape(-1, 3)
    if ids.size != keys.shape[0]:
        raise ValueError("row_ids and coordinate_keys must have the same length")
    n_chunks = 0 if n_rows == 0 else (n_rows - 1) // _LAYOUT_CHUNK + 1
    catalog: Dict[str, Any] = {"n_rows": n_rows}
    chunk_index = (ids - 1) // _LAYOUT_CHUNK
    for chunk in range(n_chunks):
        first = chunk * _LAYOUT_CHUNK + 1
        last = min(first + _LAYOUT_CHUNK - 1, n_rows)
        buffer = np.zeros((last - first + 1, 3), dtype=np.int64)
        selected = np.nonzero(chunk_index == chunk)[0]
        if selected.size:
            buffer[ids[selected] - first, :] = keys[selected]
        part = canonical_json_sha256({"keys": buffer.reshape(-1).tolist()})
        catalog[str(first)] = part
    return canonical_json_sha256(catalog)


# ---------------------------------------------------------------------------
# Observed data references
# ---------------------------------------------------------------------------


def _clean_optional(value: Optional[str], allowed: set, label: str) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text not in allowed:
        raise ValueError(f"unsupported {label} {value!r}")
    return text


@dataclass(frozen=True)
class TraceStoreRef:
    """One ``HDF5TraceStore`` descriptor in the ``fs-imaging-1`` contract.

    Args:
        file: Trace-store file or packed FrequenSolve trace root.
        dataset: Optional dataset name. For a packed FrequenSolve trace root
            this names a receiver group (``<group>`` or ``<group>_df``) and
            Sauce resolves the active-frequency dataset before reading.
        missing: Policy applied when the store lacks data for a source or
            receiver: ``error`` (default), ``zero``/``zeros``, or
            ``warn``/``warning``.
        source_basis: ``source_encoding`` (observed traces are encoded RHS
            gathers, the default) or ``source_geometry`` (physical source
            gathers encoded during misfit reads).
    """

    file: Path
    dataset: Optional[str] = None
    missing: Optional[MissingPolicy] = None
    source_basis: Optional[SourceBasis] = None

    def __post_init__(self) -> None:
        file = str(self.file).strip()
        if not file:
            raise ValueError("trace-store file must be non-empty")
        object.__setattr__(self, "file", Path(file))
        if self.dataset is not None:
            dataset = str(self.dataset).strip().strip("/")
            if not dataset:
                raise ValueError("trace-store dataset must be non-empty")
            object.__setattr__(self, "dataset", dataset)
        object.__setattr__(
            self,
            "missing",
            _clean_optional(
                self.missing, _MISSING_POLICIES, "trace-store missing policy"
            ),
        )
        object.__setattr__(
            self,
            "source_basis",
            _clean_optional(
                self.source_basis, _SOURCE_BASES, "trace-store source basis"
            ),
        )

    @classmethod
    def packed(
        cls,
        trace_root: Union[str, Path],
        *,
        receiver_group: str,
        suffix: str = "",
        missing: Optional[MissingPolicy] = None,
        source_basis: Optional[SourceBasis] = None,
    ) -> "TraceStoreRef":
        """Reference one receiver group inside a packed FrequenSolve trace root.

        Args:
            trace_root: Trace output directory or its ``traces.h5`` file.
            receiver_group: Receiver group name.
            suffix: Dataset suffix such as ``"_df"`` for the first
                frequency-derivative channel.
            missing: Optional missing-data policy.
            source_basis: Optional observed source basis.
        """

        group = str(receiver_group).strip()
        if not group:
            raise ValueError("receiver_group must be non-empty")
        trace_file = Path(trace_root)
        if trace_file.suffix.lower() not in {".h5", ".hdf5"}:
            trace_file = trace_file / _PACKED_TRACE_FILE
        return cls(
            file=trace_file,
            dataset=f"{group}{suffix}",
            missing=missing,
            source_basis=source_basis,
        )

    def replace(self, **changes: Any) -> "TraceStoreRef":
        """Return a copy with the given fields replaced."""

        values = {
            "file": self.file,
            "dataset": self.dataset,
            "missing": self.missing,
            "source_basis": self.source_basis,
        }
        values.update(changes)
        return TraceStoreRef(**values)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this reference as an ``HDF5TraceStore`` descriptor."""

        return {
            "_type": "HDF5TraceStore",
            "file": str(self.file),
            **({"dataset": self.dataset} if self.dataset is not None else {}),
            **({"missing": self.missing} if self.missing is not None else {}),
            **(
                {"source_basis": self.source_basis}
                if self.source_basis is not None
                else {}
            ),
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "TraceStoreRef":
        """Deserialize an ``HDF5TraceStore``/``SeismicStore`` descriptor."""

        allowed = {"_type", "file", "dataset", "missing", "source_basis"}
        unknown = sorted(set(data).difference(allowed))
        if unknown:
            raise ValueError(f"unsupported trace-store option(s): {', '.join(unknown)}")
        if data.get("_type") not in {"HDF5TraceStore", "SeismicStore"}:
            raise ValueError("expected an HDF5TraceStore or SeismicStore descriptor")
        return cls(
            file=data["file"],
            dataset=data.get("dataset"),
            missing=data.get("missing"),
            source_basis=data.get("source_basis"),
        )


ObservedRef = Union[Path, TraceStoreRef]


def _observed_ref_from_value(value: Any) -> ObservedRef:
    if isinstance(value, TraceStoreRef):
        return value
    if isinstance(value, Mapping):
        return TraceStoreRef.from_fs(value)
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if not text:
            raise ValueError("observed trace reference must be non-empty")
        return Path(text)
    raise TypeError(
        "observed trace references must be a path stem, a TraceStoreRef, or an "
        "HDF5TraceStore mapping"
    )


def _observed_ref_to_fs(ref: ObservedRef) -> Any:
    return ref.to_fs() if isinstance(ref, TraceStoreRef) else str(ref)


@dataclass(frozen=True)
class ObservedGroup:
    """Observed data bound to one misfit receiver group.

    Args:
        name: Receiver group name in the acquisition.
        observed: Observed path stem or :class:`TraceStoreRef`; ``None`` asks
            Sauce for zero observed data (sensitivity kernels).
        derivatives: Mapping from derivative name (``"df"``) to a path stem
            or :class:`TraceStoreRef` holding the matching observed
            derivative traces.
        source_basis: Observed source basis for a path-stem ``observed``
            (``observed_source_basis`` in the contract).
    """

    name: str
    observed: Optional[ObservedRef] = None
    derivatives: Mapping[str, ObservedRef] = field(default_factory=dict)
    source_basis: Optional[SourceBasis] = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("receiver group name must be non-empty")
        object.__setattr__(self, "name", name)
        if self.observed is not None:
            object.__setattr__(
                self, "observed", _observed_ref_from_value(self.observed)
            )
        derivatives = {}
        for key, value in dict(self.derivatives or {}).items():
            axis = str(key).strip()
            if axis != "df":
                raise ValueError("observed derivatives support only the 'df' axis")
            derivatives[axis] = _observed_ref_from_value(value)
        object.__setattr__(self, "derivatives", derivatives)
        object.__setattr__(
            self,
            "source_basis",
            _clean_optional(self.source_basis, _SOURCE_BASES, "observed source basis"),
        )

    @property
    def df(self) -> Optional[ObservedRef]:
        """Return the first frequency-derivative reference when present."""

        return self.derivatives.get("df")

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the observed part of one misfit receiver-group entry."""

        payload: Dict[str, Any] = {"name": self.name}
        if self.observed is not None:
            payload["observed"] = _observed_ref_to_fs(self.observed)
        if self.derivatives:
            payload["observed_derivatives"] = {
                axis: _observed_ref_to_fs(ref) for axis, ref in self.derivatives.items()
            }
        if self.source_basis is not None:
            payload["observed_source_basis"] = self.source_basis
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ObservedGroup":
        """Deserialize the observed part of one misfit receiver-group entry."""

        return cls(
            name=data["name"],
            observed=data.get("observed"),
            derivatives=dict(data.get("observed_derivatives") or {}),
            source_basis=data.get("observed_source_basis"),
        )

    def resolved(self, resolve_path: Any) -> "ObservedGroup":
        """Return a copy with every file path passed through ``resolve_path``."""

        def _resolve(ref: ObservedRef) -> ObservedRef:
            if isinstance(ref, TraceStoreRef):
                return ref.replace(file=resolve_path(ref.file))
            return Path(resolve_path(ref))

        return ObservedGroup(
            name=self.name,
            observed=None if self.observed is None else _resolve(self.observed),
            derivatives={axis: _resolve(ref) for axis, ref in self.derivatives.items()},
            source_basis=self.source_basis,
        )


def _is_trace_dataset(value: Any) -> bool:
    try:
        from frequensolve.seismic.traces import TraceDataset
    except ImportError:  # pragma: no cover - seismic extras always present in tests
        return False
    return isinstance(value, TraceDataset)


def _is_job(value: Any) -> bool:
    return hasattr(value, "trace_path") and hasattr(value, "f_list")


def _sorted_f_map(metadata: Mapping[str, Any]) -> List[Any]:
    return [
        freq
        for _, freq in sorted(metadata["f_map"].items(), key=lambda item: int(item[0]))
    ]


@dataclass(frozen=True)
class _ObservedSource:
    """Normalized form of one observed-data argument."""

    kind: Literal["job", "stem", "mapping", "dataset", "ref"]
    stem: Optional[Path] = None
    ref: Optional[TraceStoreRef] = None
    groups: Mapping[str, ObservedRef] = field(default_factory=dict)
    known_groups: Optional[Tuple[str, ...]] = None
    frequencies: Optional[Tuple[Any, ...]] = None

    def ref_for(
        self,
        group: str,
        *,
        suffix: str,
        missing: Optional[str],
        source_basis: Optional[str],
    ) -> Optional[ObservedRef]:
        """Return the reference for ``group`` or ``None`` when unbound."""

        if self.kind == "mapping":
            if group not in self.groups:
                return None
            ref = self.groups[group]
            if isinstance(ref, TraceStoreRef):
                return ref.replace(
                    missing=ref.missing if ref.missing is not None else missing,
                    source_basis=(
                        ref.source_basis
                        if ref.source_basis is not None
                        else source_basis
                    ),
                )
            return ref
        if self.kind == "ref":
            assert self.ref is not None
            return self.ref.replace(
                missing=self.ref.missing if self.ref.missing is not None else missing,
                source_basis=(
                    self.ref.source_basis
                    if self.ref.source_basis is not None
                    else source_basis
                ),
            )
        assert self.stem is not None
        if self.known_groups is not None and group not in self.known_groups:
            return None
        if suffix or missing is not None:
            return TraceStoreRef.packed(
                self.stem,
                receiver_group=group,
                suffix=suffix,
                missing=missing,
                source_basis=source_basis,
            )
        return self.stem


def _normalize_source(value: Any, *, label: str) -> _ObservedSource:
    if isinstance(value, ObservedData):
        raise TypeError(f"{label} cannot be another ObservedData")
    if _is_job(value):
        groups = tuple(
            str(name)
            for name in getattr(value.trace_outputs, "groups", ())
            if not _is_derivative_group(str(name))
        )
        return _ObservedSource(
            kind="job",
            stem=Path(value.trace_path),
            known_groups=groups or None,
            frequencies=tuple(value.f_list),
        )
    if _is_trace_dataset(value):
        metadata = value.metadata
        packed = [entry["path"] for entry in metadata.get("packed_entries", [])]
        if packed:
            stem = Path(packed[0]).parent
        elif value.files:
            stem = Path(value.files[0]).parent
        else:
            raise ValueError("TraceDataset has no files")
        groups = tuple(str(name) for name in getattr(value.manifest, "groups", ()))
        return _ObservedSource(
            kind="dataset",
            stem=stem,
            known_groups=groups or None,
            frequencies=tuple(_sorted_f_map(metadata)),
        )
    if isinstance(value, TraceStoreRef):
        return _ObservedSource(kind="ref", ref=value)
    if isinstance(value, Mapping):
        if "_type" in value:
            return _ObservedSource(kind="ref", ref=TraceStoreRef.from_fs(value))
        groups: Dict[str, ObservedRef] = {}
        for name, item in value.items():
            if isinstance(item, ObservedGroup):
                raise TypeError(
                    f"{label} mapping values must be paths or trace-store references"
                )
            groups[str(name)] = _observed_ref_from_value(item)
        if not groups:
            raise ValueError(f"{label} mapping must name at least one receiver group")
        return _ObservedSource(kind="mapping", groups=groups)
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        if not text:
            raise ValueError(f"{label} path must be non-empty")
        return _ObservedSource(kind="stem", stem=Path(text))
    raise TypeError(
        f"{label} must be a forward job, a path stem, a mapping of receiver group "
        "to path or TraceStoreRef, a TraceDataset, or a TraceStoreRef"
    )


def _is_derivative_group(name: str) -> bool:
    if name.endswith("_df") or name.endswith("_ds"):
        return True
    stem, _, tail = name.rpartition("_d")
    return bool(stem) and len(tail) == 2 and tail[0].isdigit() and tail[1] in "fs"


class ObservedData:
    """Observed receiver data for an imaging problem.

    Accepted ``source`` forms:

    * a FrequenSolve forward job: its trace output directory is the observed
      path stem and its ``f_list`` gives the frequencies;
    * a path stem (directory or packed trace root) shared by every receiver
      group;
    * a mapping from receiver group name to a path stem, ``.h5`` file,
      :class:`TraceStoreRef`, or ``HDF5TraceStore`` mapping;
    * a :class:`~frequensolve.seismic.traces.TraceDataset` whose packed trace
      root and frequency map are used;
    * a :class:`TraceStoreRef` shared by every receiver group.

    Args:
        source: Observed data in one of the forms above.
        derivatives: Mapping from derivative axis (``"df"``) to a source in
            the same forms. A forward job or path stem is read as a packed
            trace root and the ``<group>_df`` dataset is referenced.
        source_basis: Observed source basis applied to every group.
        missing: Missing-data policy applied to trace-store references.
        frequencies: Explicit frequencies when they cannot be inferred.
    """

    def __init__(
        self,
        source: Any,
        *,
        derivatives: Optional[Mapping[str, Any]] = None,
        source_basis: Optional[SourceBasis] = None,
        missing: Optional[MissingPolicy] = None,
        frequencies: Optional[Iterable[Any]] = None,
    ) -> None:
        self._source = _normalize_source(source, label="observed")
        self._derivatives: Dict[str, _ObservedSource] = {}
        for axis, value in dict(derivatives or {}).items():
            key = str(axis).strip()
            if key != "df":
                raise ValueError("observed derivatives support only the 'df' axis")
            self._derivatives[key] = _normalize_source(value, label="derivatives['df']")
        self._source_basis = _clean_optional(
            source_basis, _SOURCE_BASES, "observed source basis"
        )
        self._missing = _clean_optional(missing, _MISSING_POLICIES, "missing policy")
        if frequencies is not None:
            self._frequencies: Optional[Tuple[Any, ...]] = tuple(frequencies)
        else:
            self._frequencies = self._source.frequencies
            if self._frequencies is None:
                for derivative in self._derivatives.values():
                    if derivative.frequencies is not None:
                        self._frequencies = derivative.frequencies
                        break
        if self._frequencies is not None and len(self._frequencies) == 0:
            raise ValueError("observed data requires at least one frequency")

    @property
    def frequencies(self) -> Optional[List[Any]]:
        """Return the inferred or explicit frequencies, if known."""

        return None if self._frequencies is None else list(self._frequencies)

    @property
    def source_basis(self) -> Optional[str]:
        """Return the observed source basis applied to every group."""

        return self._source_basis

    @property
    def missing(self) -> Optional[str]:
        """Return the missing-data policy applied to trace-store references."""

        return self._missing

    @property
    def group_names(self) -> Optional[Tuple[str, ...]]:
        """Return receiver group names known before resolution.

        ``None`` means the data applies to every receiver group of the
        simulation it is resolved against.
        """

        if self._source.kind == "mapping":
            return tuple(self._source.groups)
        return self._source.known_groups

    @property
    def groups(self) -> Dict[str, ObservedGroup]:
        """Return the observed groups known before resolution.

        A bare path stem or trace-store reference yields an empty mapping;
        use :meth:`resolve` to bind it to a simulation's receiver groups.
        """

        names = self.group_names
        if names is None:
            return {}
        return {name: self._group(name) for name in names}

    def _group(self, name: str) -> ObservedGroup:
        observed = self._source.ref_for(
            name, suffix="", missing=self._missing, source_basis=self._source_basis
        )
        derivatives: Dict[str, ObservedRef] = {}
        for axis, derivative in self._derivatives.items():
            ref = derivative.ref_for(
                name,
                suffix=f"_{axis}",
                missing=self._missing,
                source_basis=self._source_basis,
            )
            if ref is None:
                raise KeyError(
                    f"observed derivatives['{axis}'] do not cover receiver group {name!r}"
                )
            derivatives[axis] = ref
        return ObservedGroup(
            name=name,
            observed=observed,
            derivatives=derivatives,
            source_basis=(
                self._source_basis if not isinstance(observed, TraceStoreRef) else None
            ),
        )

    def resolve(self, simulation: Any) -> List[ObservedGroup]:
        """Return one :class:`ObservedGroup` per simulation receiver group, in order.

        Raises:
            KeyError: If explicit groups do not cover a simulation receiver
                group.
            ValueError: If explicit groups name receiver groups the simulation
                does not define.
        """

        names = [str(group.name) for group in simulation.acquisition.receiver_groups]
        if not names:
            raise ValueError("simulation has no receiver groups")
        known = self.group_names
        if known is not None:
            extra = sorted(set(known).difference(names))
            if extra and self._source.kind == "mapping":
                raise ValueError(
                    "observed data names receiver groups absent from the simulation: "
                    + ", ".join(extra)
                )
            missing = [name for name in names if name not in known]
            if missing:
                raise KeyError(
                    "observed data does not cover receiver group(s): "
                    + ", ".join(missing)
                )
        return [self._group(name) for name in names]

    def __repr__(self) -> str:
        names = self.group_names
        return (
            f"ObservedData(kind={self._source.kind!r}, groups="
            f"{'all' if names is None else list(names)}, frequencies={self.frequencies})"
        )


# ---------------------------------------------------------------------------
# Data space and vectors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DataSegment:
    group: str
    components: Tuple[str, ...]
    sources: Tuple[int, ...]
    receivers: Tuple[int, ...]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return ``(sources, components, receivers)`` for one frequency."""

        return (len(self.sources), len(self.components), len(self.receivers))

    def size(self, n_frequencies: int) -> int:
        """Return the packed scalar count for this segment."""

        return n_frequencies * int(np.prod(self.shape))


@dataclass(frozen=True)
class TermLayout:
    """Row identity of one objective term inside a :class:`DataSpace`.

    Args:
        id: Objective term id.
        n_global_rows: Global row count of the term.
        row_ids: One-based canonical row ids.
        coordinate_keys: ``(n, 3)`` keys ``(encoded RHS, receiver/sample id,
            component)`` per row.
        indices: Flat positions of each row inside the packed data vector,
            or ``None`` for a layout read from a manifest without a space.
        layout_fingerprint: Sauce layout fingerprint; computed when omitted.
    """

    id: str
    n_global_rows: int
    row_ids: np.ndarray
    coordinate_keys: np.ndarray
    indices: Optional[np.ndarray] = None
    layout_fingerprint: Optional[str] = None

    def __post_init__(self) -> None:
        row_ids = np.asarray(self.row_ids, dtype=np.int64).reshape(-1)
        keys = np.asarray(self.coordinate_keys, dtype=np.int64).reshape(-1, 3)
        if keys.shape[0] != row_ids.size:
            raise ValueError("coordinate_keys must have one row per row id")
        if row_ids.size and (row_ids.min() < 1 or row_ids.max() > self.n_global_rows):
            raise ValueError("row ids must lie in 1..n_global_rows")
        if np.unique(row_ids).size != row_ids.size:
            raise ValueError("row ids must be unique")
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "coordinate_keys", keys)
        object.__setattr__(self, "n_global_rows", int(self.n_global_rows))
        if self.indices is not None:
            indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
            if indices.size != row_ids.size:
                raise ValueError("indices must have one entry per row id")
            object.__setattr__(self, "indices", indices)
        if self.layout_fingerprint is None:
            object.__setattr__(
                self,
                "layout_fingerprint",
                objective_layout_fingerprint(self.n_global_rows, row_ids, keys),
            )

    @property
    def complete(self) -> bool:
        """Return whether every global row is present."""

        return self.row_ids.size == self.n_global_rows

    def manifest_entry(self) -> Dict[str, Any]:
        """Return the ``terms[]`` manifest entry for this layout."""

        return {
            "id": self.id,
            "n_global_rows": self.n_global_rows,
            "layout_fingerprint": self.layout_fingerprint,
        }


class DataSpace:
    """Vectorization rules for complex frequency-domain trace data.

    Entries are ordered by receiver group segment, then by
    ``(frequency, source, component, receiver)`` with receiver varying
    fastest, which matches Sauce's dense ``source_component_receiver``
    trace layout.

    Args:
        frequencies: Frequencies packed into the data vector.
        segments: Receiver-group segments describing sources, components,
            and receiver indices.
        dtype: NumPy dtype for packed vectors.
    """

    dims: Tuple[str, ...] = _DATA_DIMS

    def __init__(
        self,
        frequencies: Iterable[Any],
        segments: Iterable[_DataSegment],
        dtype: Any = np.complex128,
    ) -> None:
        self.frequencies = tuple(frequencies)
        self.segments = tuple(segments)
        self.dtype = np.dtype(dtype)
        if not self.frequencies:
            raise ValueError("DataSpace requires at least one frequency")
        if not self.segments:
            raise ValueError("DataSpace requires at least one receiver group segment")
        names = [segment.group for segment in self.segments]
        if len(set(names)) != len(names):
            raise ValueError("DataSpace receiver groups must be unique")

    @classmethod
    def from_simulation(
        cls,
        simulation: Any,
        frequencies: Iterable[Any],
        dtype: Any = np.complex128,
    ) -> "DataSpace":
        """Build a data space from a simulation's sources and receivers.

        Raises:
            ValueError: If the source field count is unknown or a receiver
                group has no components.
        """

        source_count = simulation.acquisition.known_source_field_count()
        if source_count is None:
            raise ValueError(
                "DataSpace requires a known source field count for external "
                "source geometry or encoding"
            )
        if source_count < 1:
            raise ValueError("DataSpace requires at least one source field")
        sources = tuple(range(1, int(source_count) + 1))

        segments = []
        for group in simulation.acquisition.receiver_groups:
            components = tuple(
                component.name for component in group.device.output_components()
            )
            if not components:
                raise ValueError(f"Receiver group '{group.name}' has no components")
            receivers = tuple(range(1, group.output_size + 1))
            segments.append(
                _DataSegment(
                    group=group.name,
                    components=components,
                    sources=sources,
                    receivers=receivers,
                )
            )
        return cls(frequencies=frequencies, segments=segments, dtype=dtype)

    # -- layout -----------------------------------------------------------

    @property
    def groups(self) -> Tuple[str, ...]:
        """Return receiver group names in packing order."""

        return tuple(segment.group for segment in self.segments)

    def segment(self, group: str) -> _DataSegment:
        """Return the segment for ``group``."""

        for segment in self.segments:
            if segment.group == group:
                return segment
        raise KeyError(f"DataSpace has no receiver group {group!r}")

    def offset(self, group: str) -> int:
        """Return the flat offset of ``group``'s block."""

        offset = 0
        for segment in self.segments:
            if segment.group == group:
                return offset
            offset += segment.size(len(self.frequencies))
        raise KeyError(f"DataSpace has no receiver group {group!r}")

    def frequency_index(self, frequency: Optional[Any] = None) -> int:
        """Return the index of ``frequency`` (or the only frequency)."""

        if frequency is None:
            if len(self.frequencies) != 1:
                raise ValueError(
                    "frequency is required when the data space spans several "
                    "frequencies"
                )
            return 0
        if isinstance(frequency, (int, np.integer)) and not isinstance(frequency, bool):
            index = int(frequency)
            if 0 <= index < len(self.frequencies):
                return index
        values = np.asarray(self.frequencies, dtype=complex)
        matches = np.nonzero(np.isclose(values, complex(frequency)))[0]
        if matches.size != 1:
            raise ValueError(f"frequency {frequency!r} is not in the data space")
        return int(matches[0])

    @property
    def size(self) -> int:
        """Return the total number of scalar entries in a data vector."""

        return sum(segment.size(len(self.frequencies)) for segment in self.segments)

    @property
    def shape(self) -> Tuple[int]:
        """Return the one-dimensional shape expected by linear operators."""

        return (self.size,)

    def __len__(self) -> int:
        return self.size

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataSpace):
            return NotImplemented
        return (
            self.frequencies == other.frequencies
            and self.segments == other.segments
            and self.dtype == other.dtype
        )

    def __hash__(self) -> int:
        return hash((self.frequencies, self.segments, self.dtype.str))

    def __repr__(self) -> str:
        groups = ", ".join(
            f"{segment.group}{segment.shape}" for segment in self.segments
        )
        return f"DataSpace(frequencies={list(self.frequencies)}, groups=[{groups}])"

    # -- vectors ----------------------------------------------------------

    def zeros(self) -> "DataVector":
        """Return the zero data vector."""

        return DataVector(np.zeros(self.size, dtype=self.dtype), self)

    def ones(self) -> "DataVector":
        """Return the all-ones data vector."""

        return DataVector(np.ones(self.size, dtype=self.dtype), self)

    def random(self, seed: Optional[int] = None) -> "DataVector":
        """Return a standard complex normal random data vector."""

        rng = np.random.default_rng(seed)
        values = rng.standard_normal(self.size)
        if np.issubdtype(self.dtype, np.complexfloating):
            values = values + 1j * rng.standard_normal(self.size)
        return DataVector(np.asarray(values, dtype=self.dtype), self)

    def vector(self, data: Any) -> "DataVector":
        """Pack ``data`` into a :class:`DataVector`."""

        return DataVector(self.pack(data), self)

    def pack(self, data: Any) -> np.ndarray:
        """Pack trace data into ``(frequency, source, component, receiver)`` order.

        Args:
            data: :class:`DataVector`, ``TraceDataset``, xarray dataset,
                mapping of group data, or an already-packed vector.

        Raises:
            KeyError: If a required receiver group is missing.
            ValueError: If group dimensions or vector size do not match.
        """

        if isinstance(data, DataVector):
            if data.space != self:
                raise ValueError("DataVector belongs to a different DataSpace")
            return np.array(data.values, dtype=self.dtype, copy=True)
        if _is_trace_dataset(data):
            return self.pack(self._dataset_from_trace_dataset(data))
        if isinstance(data, (xr.Dataset, Mapping)):
            chunks = []
            for segment in self.segments:
                if segment.group not in data:
                    raise KeyError(
                        f"Trace data is missing receiver group '{segment.group}'"
                    )
                chunks.append(self._coerce_group(data[segment.group], segment))
            return np.concatenate(chunks).astype(self.dtype, copy=False)

        vector = np.asarray(data, dtype=self.dtype).reshape(-1)
        if vector.size != self.size:
            raise ValueError(
                f"Data vector has size {vector.size}; expected {self.size}"
            )
        return vector

    def _coerce_group(self, data: Any, segment: _DataSegment) -> np.ndarray:
        expected = (len(self.frequencies), *segment.shape)
        if isinstance(data, xr.DataArray):
            da = data.rename({"shot": "source"}) if "shot" in data.dims else data
            missing = [dim for dim in _DATA_DIMS if dim not in da.dims]
            if missing:
                raise ValueError(
                    f"Trace group '{segment.group}' is missing dimensions {missing}"
                )
            values = da.transpose(*_DATA_DIMS).data
        else:
            values = data
        values = np.asarray(values, dtype=self.dtype)
        if tuple(values.shape) != expected:
            raise ValueError(
                f"Trace group '{segment.group}' has shape {values.shape}; "
                f"expected {expected}"
            )
        return values.reshape(-1)

    def _dataset_from_trace_dataset(self, traces: Any) -> xr.Dataset:
        data_vars = {}
        for segment in self.segments:
            values = np.zeros((len(self.frequencies), *segment.shape), dtype=self.dtype)
            for isource, source in enumerate(segment.sources):
                for icomp, component in enumerate(segment.components):
                    fd = traces.fd(segment.group, component, source=source)
                    if "frequency" in fd.dims:
                        fd = fd.interp(
                            frequency=list(self.frequencies),
                            kwargs={"fill_value": 0},
                        )
                    fd_values = (
                        fd.data.compute() if hasattr(fd.data, "compute") else fd.data
                    )
                    values[:, isource, icomp, :] = np.asarray(fd_values).reshape(
                        len(self.frequencies), len(segment.receivers)
                    )
            data_vars[segment.group] = self._group_array(values, segment)
        return xr.Dataset(data_vars)

    def _group_array(self, values: np.ndarray, segment: _DataSegment) -> xr.DataArray:
        return xr.DataArray(
            values,
            dims=_DATA_DIMS,
            coords={
                "frequency": list(self.frequencies),
                "source": list(segment.sources),
                "component": list(segment.components),
                "receiver": list(segment.receivers),
            },
            name=segment.group,
        )

    def unpack(self, vector: Any) -> xr.Dataset:
        """Unpack a data vector into an ``xarray.Dataset`` by receiver group."""

        values = self.pack(vector) if isinstance(vector, DataVector) else vector
        values = np.asarray(values, dtype=self.dtype).reshape(-1)
        if values.size != self.size:
            raise ValueError(
                f"Data vector has size {values.size}; expected {self.size}"
            )
        data_vars = {}
        offset = 0
        for segment in self.segments:
            shape = (len(self.frequencies), *segment.shape)
            n = int(np.prod(shape))
            data_vars[segment.group] = self._group_array(
                values[offset : offset + n].reshape(shape), segment
            )
            offset += n
        return xr.Dataset(data_vars)

    # -- objective rows ----------------------------------------------------

    def term_layout(
        self,
        group: str,
        *,
        frequency: Optional[Any] = None,
        id: Optional[str] = None,
    ) -> TermLayout:
        """Return the dense Sauce row layout of ``group`` at one frequency.

        Sauce numbers dense rows ``rhs + n_rhs * ((c - 1) + n_c * (r - 1))``
        with keys ``(rhs, r, c)``; the returned layout maps every row to its
        position inside this space's ``(frequency, source, component,
        receiver)`` block.
        """

        segment = self.segment(group)
        ifreq = self.frequency_index(frequency)
        n_src, n_comp, n_rcv = segment.shape
        src, comp, rcv = np.meshgrid(
            np.arange(1, n_src + 1),
            np.arange(1, n_comp + 1),
            np.arange(1, n_rcv + 1),
            indexing="ij",
        )
        src = src.reshape(-1)
        comp = comp.reshape(-1)
        rcv = rcv.reshape(-1)
        row_ids = src + n_src * ((comp - 1) + n_comp * (rcv - 1))
        keys = np.stack([src, rcv, comp], axis=1)
        block = n_src * n_comp * n_rcv
        indices = self.offset(group) + ifreq * block + np.arange(block)
        return TermLayout(
            id=group if id is None else str(id),
            n_global_rows=block,
            row_ids=row_ids,
            coordinate_keys=keys,
            indices=indices,
        )

    def term_layouts(
        self,
        *,
        frequency: Optional[Any] = None,
        ids: Optional[Mapping[str, str]] = None,
    ) -> List[TermLayout]:
        """Return one dense :class:`TermLayout` per receiver group.

        Args:
            frequency: Frequency (value or index) selecting the block.
            ids: Optional mapping from receiver group to term id.
        """

        ids = dict(ids or {})
        return [
            self.term_layout(group, frequency=frequency, id=ids.get(group))
            for group in self.groups
        ]


def _as_values(other: Any) -> Any:
    return other.values if isinstance(other, DataVector) else other


class DataVector:
    """One complex data vector over a :class:`DataSpace`.

    The wrapper supports NumPy conversion, arithmetic with other vectors,
    arrays, and scalars, Hermitian products, and the objective-vector file
    format. Plain arrays are accepted everywhere a vector is.
    """

    __array_priority__ = 20

    def __init__(self, values: Any, space: DataSpace) -> None:
        if not isinstance(space, DataSpace):
            raise TypeError("DataVector requires a DataSpace")
        self.space = space
        vector = np.array(_as_values(values), dtype=space.dtype, copy=True).reshape(-1)
        if vector.size != space.size:
            raise ValueError(
                f"Data vector has size {vector.size}; expected {space.size}"
            )
        self.values = vector

    # -- conversions -------------------------------------------------------

    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> np.ndarray:
        if dtype is None:
            return self.values if not copy else self.values.copy()
        return self.values.astype(dtype, copy=bool(copy))

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the packed values."""

        return self.values.copy()

    def to_dataset(self) -> xr.Dataset:
        """Return the vector as an ``xarray.Dataset`` by receiver group."""

        return self.space.unpack(self.values)

    def __getitem__(self, group: str) -> xr.DataArray:
        return self.to_dataset()[group]

    @property
    def shape(self) -> Tuple[int]:
        return self.values.shape  # type: ignore[return-value]

    @property
    def size(self) -> int:
        return int(self.values.size)

    @property
    def dtype(self) -> np.dtype:
        return self.values.dtype

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[Any]:
        return iter(self.values)

    def __repr__(self) -> str:
        return f"DataVector(size={self.size}, norm={self.norm():.6g})"

    def copy(self) -> "DataVector":
        """Return a copy."""

        return DataVector(self.values, self.space)

    def conj(self) -> "DataVector":
        """Return the complex conjugate."""

        return DataVector(np.conj(self.values), self.space)

    # -- arithmetic --------------------------------------------------------

    def _wrap(self, values: np.ndarray) -> "DataVector":
        return DataVector(values, self.space)

    def _check(self, other: Any) -> Any:
        if isinstance(other, DataVector) and other.space != self.space:
            raise ValueError("DataVector operands belong to different spaces")
        return _as_values(other)

    def __add__(self, other: Any) -> "DataVector":
        return self._wrap(self.values + self._check(other))

    __radd__ = __add__

    def __sub__(self, other: Any) -> "DataVector":
        return self._wrap(self.values - self._check(other))

    def __rsub__(self, other: Any) -> "DataVector":
        return self._wrap(self._check(other) - self.values)

    def __mul__(self, other: Any) -> "DataVector":
        return self._wrap(self.values * self._check(other))

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "DataVector":
        return self._wrap(self.values / self._check(other))

    def __rtruediv__(self, other: Any) -> "DataVector":
        return self._wrap(self._check(other) / self.values)

    def __neg__(self) -> "DataVector":
        return self._wrap(-self.values)

    def __pos__(self) -> "DataVector":
        return self.copy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataVector):
            return NotImplemented
        return self.space == other.space and bool(
            np.array_equal(self.values, other.values)
        )

    __hash__ = None  # type: ignore[assignment]

    def vdot(self, other: Any) -> complex:
        """Return the Hermitian product ``conj(self) . other``."""

        return complex(np.vdot(self.values, self._check(other)))

    def dot(self, other: Any) -> float:
        """Return the real pairing ``Re(conj(self) . other)``."""

        return float(np.real(self.vdot(other)))

    def norm(self) -> float:
        """Return the Euclidean norm."""

        return float(np.linalg.norm(self.values))

    # -- objective-vector format -------------------------------------------

    @staticmethod
    def shard_path(manifest_path: Union[str, Path], rank: int = 0) -> Path:
        """Return Sauce's shard path for ``manifest_path`` and ``rank``."""

        manifest_path = Path(manifest_path)
        return manifest_path.with_name(f"{manifest_path.stem}_rank_{int(rank)}.h5")

    def write_objective_vector(
        self,
        path: Union[str, Path],
        *,
        state_fingerprint: str,
        term_layout: Union[TermLayout, Sequence[TermLayout]],
        n_ranks: int = 1,
    ) -> Path:
        """Write this vector as an ``fs-objective-vector-3`` manifest and shard.

        Every row of every term is written into one shard
        (``<stem>_rank_0.h5``); the manifest still declares ``n_ranks`` so a
        multi-rank Sauce partition can consume it after redistribution.

        Args:
            path: Manifest JSON path.
            state_fingerprint: ``sha256:`` fingerprint of the frozen objective
                state the dual belongs to.
            term_layout: One or more :class:`TermLayout` objects whose
                ``indices`` select the vector entries of each term.
            n_ranks: Partition rank count declared in the manifest.

        Returns:
            The manifest path.
        """

        import h5py

        layouts = (
            [term_layout] if isinstance(term_layout, TermLayout) else list(term_layout)
        )
        if not layouts:
            raise ValueError("write_objective_vector requires at least one term")
        ids = [layout.id for layout in layouts]
        if len(set(ids)) != len(ids):
            raise ValueError("objective term ids must be unique")
        state = _normalize_hash(state_fingerprint)
        if not state.startswith("sha256:") or len(state) != 71:
            raise ValueError("state_fingerprint must be a sha256 fingerprint")

        manifest_path = Path(path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        shard = self.shard_path(manifest_path, 0)
        with h5py.File(shard, "w") as h5:
            for index, layout in enumerate(layouts):
                if layout.indices is None:
                    raise ValueError(
                        f"term {layout.id!r} has no data-space indices; build it "
                        "from DataSpace.term_layout"
                    )
                if not layout.complete:
                    raise ValueError(
                        f"term {layout.id!r} does not cover all "
                        f"{layout.n_global_rows} rows"
                    )
                values = self.values[layout.indices]
                group = h5.create_group(f"/terms/{index}")
                group.create_dataset("row_ids", data=layout.row_ids.astype(np.int32))
                group.create_dataset(
                    "coordinate_keys",
                    data=layout.coordinate_keys.astype(np.int32),
                )
                packed = np.empty((values.size, 2), dtype=np.float64)
                packed[:, 0] = np.real(values)
                packed[:, 1] = np.imag(values)
                group.create_dataset("values", data=packed)

        manifest: Dict[str, Any] = {
            "schema": _OBJECTIVE_VECTOR_SCHEMA,
            "state_fingerprint": state,
            "partition": {
                "n_ranks": int(n_ranks),
                "compatibility": "same_mesh_partition",
            },
            "shards": [{"file": str(shard.resolve()), "sha256": file_sha256(shard)}],
            "terms": [layout.manifest_entry() for layout in layouts],
        }
        manifest["manifest_fingerprint"] = canonical_json_sha256(manifest)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest_path

    @staticmethod
    def read_objective_manifest(
        path: Union[str, Path], *, verify: bool = True
    ) -> Dict[str, Any]:
        """Load and validate an ``fs-objective-vector-3`` manifest.

        Args:
            path: Manifest JSON path.
            verify: Recompute the manifest fingerprint and shard hashes.

        Returns:
            The manifest with ``shards[*].file`` resolved to absolute paths.
        """

        manifest_path = Path(path)
        manifest = json.loads(manifest_path.read_text())
        required = {
            "schema",
            "state_fingerprint",
            "manifest_fingerprint",
            "partition",
            "shards",
            "terms",
        }
        missing = sorted(required.difference(manifest))
        if missing:
            raise ValueError(f"objective manifest is missing {', '.join(missing)}")
        if manifest["schema"] != _OBJECTIVE_VECTOR_SCHEMA:
            raise ValueError(
                "obsolete objective vector format; regenerate using version 3"
            )
        if manifest["partition"].get("compatibility") != "same_mesh_partition":
            raise ValueError("unsupported objective partition compatibility")
        if not manifest["shards"] or not manifest["terms"]:
            raise ValueError("objective manifest requires shards and terms")
        if verify:
            basis = {k: v for k, v in manifest.items() if k != "manifest_fingerprint"}
            if canonical_json_sha256(basis) != _normalize_hash(
                manifest["manifest_fingerprint"]
            ):
                raise ValueError("objective manifest fingerprint mismatch")
        resolved = []
        for entry in manifest["shards"]:
            shard = Path(entry["file"])
            if not shard.is_absolute():
                shard = manifest_path.parent / shard
            if not shard.exists():
                candidate = manifest_path.parent / shard.name
                if candidate.exists():
                    shard = candidate
            if not shard.exists():
                raise FileNotFoundError(f"missing objective shard {entry['file']}")
            if verify and file_sha256(shard) != _normalize_hash(entry["sha256"]):
                raise ValueError(f"corrupt objective shard {shard}")
            resolved.append({"file": str(shard), "sha256": entry["sha256"]})
        manifest["shards"] = resolved
        return manifest

    @staticmethod
    def read_term_layouts(
        path: Union[str, Path], *, verify: bool = True
    ) -> List[TermLayout]:
        """Read every term's row ids and keys from an objective vector file.

        The returned layouts carry no data-space ``indices``; they document
        Sauce's row identity for a given state.
        """

        manifest = DataVector.read_objective_manifest(path, verify=verify)
        rows = DataVector._gather_rows(manifest)
        layouts = []
        for index, term in enumerate(manifest["terms"]):
            ids, keys, _ = rows[index]
            layouts.append(
                TermLayout(
                    id=term["id"],
                    n_global_rows=int(term["n_global_rows"]),
                    row_ids=ids,
                    coordinate_keys=keys,
                    layout_fingerprint=term["layout_fingerprint"],
                )
            )
        return layouts

    @staticmethod
    def _gather_rows(
        manifest: Mapping[str, Any],
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Collect rows of every term across shards, sorted by row id."""

        import h5py

        n_terms = len(manifest["terms"])
        ids: List[List[np.ndarray]] = [[] for _ in range(n_terms)]
        keys: List[List[np.ndarray]] = [[] for _ in range(n_terms)]
        values: List[List[np.ndarray]] = [[] for _ in range(n_terms)]
        for entry in manifest["shards"]:
            with h5py.File(entry["file"], "r") as h5:
                terms = h5.get("terms")
                if terms is None:
                    raise ValueError(f"objective shard {entry['file']} has no terms")
                for index in range(n_terms):
                    group = terms.get(str(index))
                    if group is None:
                        raise ValueError(
                            f"objective shard {entry['file']} lacks term {index}"
                        )
                    row_ids = np.asarray(group["row_ids"][()], dtype=np.int64).reshape(
                        -1
                    )
                    row_keys = np.asarray(
                        group["coordinate_keys"][()], dtype=np.int64
                    ).reshape(-1, 3)
                    raw = np.asarray(group["values"][()], dtype=np.float64).reshape(
                        -1, 2
                    )
                    if not (row_ids.size == row_keys.shape[0] == raw.shape[0]):
                        raise ValueError(
                            f"objective shard {entry['file']} term {index} has "
                            "inconsistent row counts"
                        )
                    ids[index].append(row_ids)
                    keys[index].append(row_keys)
                    values[index].append(raw[:, 0] + 1j * raw[:, 1])
        result = []
        for index, term in enumerate(manifest["terms"]):
            row_ids = np.concatenate(ids[index]) if ids[index] else np.zeros(0, int)
            row_keys = (
                np.concatenate(keys[index]) if keys[index] else np.zeros((0, 3), int)
            )
            row_values = (
                np.concatenate(values[index]) if values[index] else np.zeros(0, complex)
            )
            n_global = int(term["n_global_rows"])
            if row_ids.size != n_global:
                raise ValueError(
                    f"objective term {term['id']!r} has {row_ids.size} rows; "
                    f"expected {n_global}"
                )
            if row_ids.size and (row_ids.min() < 1 or row_ids.max() > n_global):
                raise ValueError(f"objective term {term['id']!r} has out-of-range rows")
            order = np.argsort(row_ids, kind="stable")
            row_ids = row_ids[order]
            if np.any(np.diff(row_ids) == 0):
                raise ValueError(f"objective term {term['id']!r} has duplicate rows")
            result.append((row_ids, row_keys[order], row_values[order]))
        return result

    @classmethod
    def read_objective_vector(
        cls,
        path: Union[str, Path],
        space: DataSpace,
        *,
        term_layout: Optional[Union[TermLayout, Sequence[TermLayout]]] = None,
        frequency: Optional[Any] = None,
        verify: bool = True,
        state_fingerprint: Optional[str] = None,
    ) -> "DataVector":
        """Read an ``fs-objective-vector-3`` file into a vector over ``space``.

        Rows may be spread over any number of shards in any order; they are
        reassembled by canonical row id and their coordinate keys are checked
        against the layout.

        Args:
            path: Manifest JSON path.
            space: Target data space.
            term_layout: Layouts giving the data-space position of each row.
                Defaults to :meth:`DataSpace.term_layouts` at ``frequency``,
                matching terms to receiver groups by id.
            frequency: Frequency (value or index) of the file when the space
                spans several frequencies and no layouts are given.
            verify: Recompute fingerprints and shard hashes.
            state_fingerprint: Optional expected ``state_fingerprint``.

        Returns:
            A :class:`DataVector` whose unreferenced entries are zero.
        """

        manifest = cls.read_objective_manifest(path, verify=verify)
        if state_fingerprint is not None and _normalize_hash(
            manifest["state_fingerprint"]
        ) != _normalize_hash(state_fingerprint):
            raise ValueError("objective vector belongs to another baseline")
        rows = cls._gather_rows(manifest)

        if term_layout is None:
            ids = [term["id"] for term in manifest["terms"]]
            unknown = [term_id for term_id in ids if term_id not in space.groups]
            if unknown:
                raise ValueError(
                    "objective term ids do not name receiver groups of the data "
                    f"space ({', '.join(unknown)}); pass term_layout explicitly"
                )
            layouts = {
                term_id: space.term_layout(term_id, frequency=frequency)
                for term_id in ids
            }
        else:
            candidates = (
                [term_layout]
                if isinstance(term_layout, TermLayout)
                else list(term_layout)
            )
            layouts = {layout.id: layout for layout in candidates}

        values = np.zeros(space.size, dtype=space.dtype)
        for index, term in enumerate(manifest["terms"]):
            layout = layouts.get(term["id"])
            if layout is None:
                raise ValueError(f"no term layout for objective term {term['id']!r}")
            if layout.indices is None:
                raise ValueError(
                    f"term layout {term['id']!r} has no data-space indices"
                )
            if layout.n_global_rows != int(term["n_global_rows"]):
                raise ValueError(
                    f"objective term {term['id']!r} row count "
                    f"{term['n_global_rows']} differs from the layout "
                    f"({layout.n_global_rows})"
                )
            if _normalize_hash(term["layout_fingerprint"]) != _normalize_hash(
                str(layout.layout_fingerprint)
            ):
                raise ValueError(
                    f"objective term {term['id']!r} layout fingerprint differs "
                    "from the data-space layout; pass a term_layout read from the "
                    "Sauce state"
                )
            row_ids, keys, row_values = rows[index]
            order = np.argsort(layout.row_ids, kind="stable")
            layout_ids = layout.row_ids[order]
            if not np.array_equal(layout_ids, row_ids):
                raise ValueError(
                    f"objective term {term['id']!r} rows do not match the layout"
                )
            if not np.array_equal(layout.coordinate_keys[order], keys):
                raise ValueError(
                    f"objective term {term['id']!r} coordinate keys do not match "
                    "the layout"
                )
            values[layout.indices[order]] = row_values
        return cls(values, space)
