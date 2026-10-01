"""Objective rows, residuals and simulated data from Sauce's immutable saved state.

Per the ``fs-objective-linearization-3`` to ``-5`` contracts a term cache stores ``simulated`` (the modeled receiver values in
objective rows) and ``objective_residual``, the weighted **observed-minus-simulated** comparison error; the Jacobian
pullback of ``vjp`` is that residual's, i.e. ``-d(simulated)/dm`` for unit weights.  A writer following the opposite
convention records ``convention = "simulated_minus_observed"`` on the ``objective_residual`` dataset.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

OBSERVED_MINUS_SIMULATED = "observed_minus_simulated"
SIMULATED_MINUS_OBSERVED = "simulated_minus_observed"
_CONVENTIONS = (OBSERVED_MINUS_SIMULATED, SIMULATED_MINUS_OBSERVED)

import h5py
import numpy as np

from ._block_digest import flat_block_digest
from .data import (
    DataSpace,
    DataVector,
    TermLayout,
    _DataSegment,
    canonical_json_sha256,
    file_sha256,
)

_DISTRIBUTED_STATE = "fs-objective-linearization-3"
_DISTRIBUTED_VECTOR = "fs-objective-vector-3"
# The objective-vector schema of each state's duals.
_VECTOR_SCHEMAS = {
    _DISTRIBUTED_STATE: _DISTRIBUTED_VECTOR,
    "fs-objective-linearization-4": "fs-objective-vector-4",
    "fs-objective-linearization-5": "fs-objective-vector-5",
}
# States identifying each cache dataset by its digest rather than file bytes.
_DIGESTED_STATES = frozenset({"fs-objective-linearization-5"})


def _resolve(parent: Path, name: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = parent / path
    if not path.is_file():
        path = parent / Path(name).name
    return path


def _read_dataset(
    group: Any,
    name: str,
    digests: Optional[Mapping[str, Any]],
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """Read ``group[name]`` as stored.

    With ``digests`` (version 5) its digest, which covers the flattened
    elements, and its contract ``shape`` are checked too.
    """

    data = np.asarray(group[name][()])
    if digests is not None:
        expected = digests.get(name)
        if expected is None:
            raise ValueError(f"objective state cache records no digest of {name!r}")
        if flat_block_digest(data) != expected:
            raise ValueError(f"objective state cache digest mismatch for {name!r}")
        if shape is not None and data.shape != shape:
            raise ValueError(
                f"objective state cache {name!r} has shape {data.shape}; "
                f"expected {shape}"
            )
    return data


class ObjectiveState:
    """Read saved coordinates independently of authored receiver group names.

    Reads Sauce's ``fs-objective-linearization-5`` state (one manifest and one
    canonical cache per task; each dataset read is checked against its
    ``fs-block-sha256-1`` digest) and the older ``-4`` (canonical) and ``-3``
    (per-rank shards) states, verified by whole-file SHA-256.

    Attributes:
        schema: The state's schema.
        vector_schema: Objective-vector schema of the duals of this state.
        n_ranks: MPI rank count the state is pinned to (one for ``-4``).
        value_dtype: Stored element type of the cached receiver values
            (``float32`` from single-precision builds); ``-5`` duals are
            written in it.
        terms: Rows of each term by id (``keys``, ``residual``, ``simulated``
            and ``convention``).
        term_configs: Resolved configuration of each term, once per shard.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        manifest = json.loads(self.path.read_text())
        schema = manifest.get("schema", _DISTRIBUTED_STATE)
        if schema not in _VECTOR_SCHEMAS:
            raise ValueError(f"unsupported objective state schema {schema!r}")
        self.schema: str = schema
        self.vector_schema: str = _VECTOR_SCHEMAS[schema]
        canonical = schema != _DISTRIBUTED_STATE
        digested = schema in _DIGESTED_STATES
        partition = manifest.get("partition")
        if not canonical:
            self.n_ranks = int(manifest["partition"]["n_ranks"])
            if self.n_ranks < 1 or len(manifest["shards"]) != self.n_ranks:
                raise ValueError("objective state has an inconsistent rank partition")
        elif isinstance(partition, Mapping) and "n_ranks" in partition:
            self.n_ranks = int(partition["n_ranks"])
        else:
            self.n_ranks = 1
        if digested and "manifest_fingerprint" not in manifest:
            raise ValueError("objective state manifest has no manifest fingerprint")
        if "manifest_fingerprint" in manifest:
            basis = {
                k: v
                for k, v in manifest.items()
                if k not in {"manifest_fingerprint", "state_fingerprint"}
            }
            if canonical_json_sha256(basis) != manifest["manifest_fingerprint"]:
                raise ValueError("objective state manifest fingerprint mismatch")
            # Version 3 identifies the state by its manifest; version 5 by a
            # Sauce-defined fingerprint the SDK treats as opaque.
            if not digested and (
                manifest.get("state_fingerprint", manifest["manifest_fingerprint"])
                != manifest["manifest_fingerprint"]
            ):
                raise ValueError("objective state fingerprint mismatch")
        self.value_dtype = np.dtype(np.float64)
        self.terms: Dict[str, Dict[str, Any]] = {}
        # Keep the resolved objective configuration as well as its rows.
        # Patch restriction must preserve parent normalization across ranks.
        self.term_configs: Dict[str, List[Dict[str, Any]]] = {}
        hashes: Dict[Path, str] = {}
        entries = [None] if canonical else manifest["shards"]
        for entry in entries:
            if entry is None:
                shard, payload = self.path, manifest
            else:
                shard = _resolve(self.path.parent, entry["file"])
                if file_sha256(shard) != entry["sha256"]:
                    raise ValueError("objective state shard hash mismatch")
                payload = json.loads(shard.read_text())
            for term in payload["terms"]:
                self.term_configs.setdefault(term["id"], []).append(term)
                cache = _resolve(shard.parent, term["cache"]["file"])
                expected = term["runtime"]["cache_fingerprint"]
                digests: Optional[Mapping[str, Any]] = None
                if digested:
                    # The cache fingerprint covers the dataset digests, which
                    # identify every dataset without rereading the file.
                    digests = term["cache"].get("digests")
                    if (
                        not isinstance(digests, Mapping)
                        or canonical_json_sha256(digests) != expected
                    ):
                        raise ValueError(
                            "objective state cache digests differ from the cache "
                            "fingerprint"
                        )
                else:
                    if cache not in hashes:
                        hashes[cache] = file_sha256(cache)
                    if hashes[cache] != expected:
                        raise ValueError("objective state cache hash mismatch")
                with h5py.File(cache, "r") as h5:
                    group = h5[term["cache"].get("group", "/")]
                    count = int(
                        np.asarray(
                            _read_dataset(group, "n_global_rows", digests)
                        ).item()
                    )
                    if (
                        digested
                        and term["runtime"].get("n_global_rows", count) != count
                    ):
                        raise ValueError(
                            "objective state row count differs from its cache"
                        )
                    keys = np.asarray(
                        _read_dataset(group, "coordinate_keys", digests, (count, 3)),
                        dtype=int,
                    ).reshape(-1, 3)
                    ids = (
                        np.arange(1, count + 1)
                        if canonical
                        else np.asarray(group["row_ids"], dtype=int).reshape(-1)
                    )
                    residual = None
                    convention = OBSERVED_MINUS_SIMULATED
                    if "objective_residual" in group:
                        dataset = group["objective_residual"]
                        packed = _read_dataset(
                            group, "objective_residual", digests, (count, 2)
                        )
                        if digested:
                            self.value_dtype = packed.dtype.newbyteorder("=")
                        residual = (packed[..., 0] + 1j * packed[..., 1]).reshape(-1)
                        convention = dataset.attrs.get("convention", convention)
                        if isinstance(convention, bytes):
                            convention = convention.decode()
                        if convention not in _CONVENTIONS:
                            raise ValueError(
                                f"unknown objective residual convention {convention!r}"
                            )
                    simulated = None
                    if "simulated" in group:
                        packed = _read_dataset(group, "simulated", digests, (count, 2))
                        if digested:
                            self.value_dtype = packed.dtype.newbyteorder("=")
                        simulated = (packed[..., 0] + 1j * packed[..., 1]).reshape(-1)
                if term["id"] not in self.terms:
                    self.terms[term["id"]] = {
                        "receiver_group": term["receiver_group"],
                        "count": count,
                        "keys": np.zeros((count, 3), int),
                        "seen": np.zeros(count, bool),
                        "residual": np.zeros(count, complex),
                        "has_residual": True,
                        "simulated": np.zeros(count, complex),
                        "has_simulated": True,
                        "convention": convention,
                    }
                item = self.terms[term["id"]]
                if count != item["count"] or keys.shape != (ids.size, 3):
                    raise ValueError("objective state has inconsistent row coordinates")
                if np.any(ids < 1) or np.any(ids > count):
                    raise ValueError("objective state has out-of-range row ids")
                if residual is not None and residual.size != ids.size:
                    raise ValueError("objective residual size differs from saved rows")
                if simulated is not None and simulated.size != ids.size:
                    raise ValueError("simulated values differ in size from saved rows")
                if convention != item["convention"]:
                    raise ValueError(
                        "objective residual conventions differ between shards"
                    )
                indices = ids - 1
                repeated = item["seen"][indices]
                if np.any(item["keys"][indices[repeated]] != keys[repeated]):
                    raise ValueError("replicated objective coordinates disagree")
                if residual is not None:
                    if not np.allclose(
                        item["residual"][indices[repeated]],
                        residual[repeated],
                        rtol=1e-5,
                        atol=1e-7,
                    ):
                        raise ValueError("replicated objective residuals disagree")
                    item["residual"][indices] = residual
                else:
                    item["has_residual"] = False
                if simulated is not None:
                    item["simulated"][indices] = simulated
                else:
                    item["has_simulated"] = False
                item["keys"][indices] = keys
                item["seen"][indices] = True
        if not self.terms or any(not np.all(t["seen"]) for t in self.terms.values()):
            raise ValueError("objective state does not cover every global row")

    def vector_options(self) -> Dict[str, Any]:
        """Return the :meth:`DataVector.write_objective_vector` format of a dual.

        Version-3 duals are partitioned like the state; canonical ones are
        not, and ``-5`` duals store values in the state's precision.
        """

        options: Dict[str, Any] = {"schema": self.vector_schema}
        if self.vector_schema == _DISTRIBUTED_VECTOR:
            options["n_ranks"] = self.n_ranks
        elif self.schema in _DIGESTED_STATES:
            options["dtype"] = self.value_dtype
        return options


class _ObjectiveSpace(DataSpace):
    _dense: Set[str]
    _keys: List[Dict[str, np.ndarray]]

    def term_layout(
        self, group: str, *, frequency: Optional[Any] = None, id: Optional[str] = None
    ) -> TermLayout:
        index = self.frequency_index(frequency)
        keys = self._keys[index][group]
        count = len(keys)
        segment = self.segment(group)
        # Dense blocks retain the public source/component/receiver packing.
        if group in self._dense:
            return super().term_layout(group, frequency=frequency, id=id)
        return TermLayout(
            id=group if id is None else id,
            n_global_rows=count,
            row_ids=np.arange(1, count + 1),
            coordinate_keys=keys,
            indices=self.offset(group)
            + index * np.prod(segment.shape)
            + np.arange(count),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataSpace):
            return NotImplemented
        if not super().__eq__(other):
            return False
        return all(
            a.layout_fingerprint == b.layout_fingerprint
            for f in self.frequencies
            for a, b in zip(
                self.term_layouts(frequency=f), other.term_layouts(frequency=f)
            )
        )

    __hash__ = DataSpace.__hash__


def objective_space(
    simulation: Any, frequencies: Sequence[Any], states: Sequence[ObjectiveState]
) -> DataSpace:
    """Use dense axes when exact; otherwise preserve saved sparse/projected rows."""
    try:
        acquisition = DataSpace.from_simulation(simulation, frequencies)
    except ValueError:
        acquisition = None
    names = tuple(states[0].terms)
    if any(tuple(state.terms) != names for state in states):
        raise ValueError("objective terms differ between frequency tasks")
    segments, dense = [], set()
    for name in names:
        terms = [state.terms[name] for state in states]
        segment = None
        if acquisition is not None:
            source = acquisition.segment(terms[0]["receiver_group"])
            candidate = replace(source, group=name)
            check = DataSpace([frequencies[0]], [candidate]).term_layout(name)
            order = np.argsort(check.row_ids)
            if all(
                np.array_equal(t["keys"], check.coordinate_keys[order]) for t in terms
            ):
                segment = candidate
                dense.add(name)
        if segment is None:
            segment = _DataSegment(
                name,
                ("objective",),
                (1,),
                tuple(range(1, max(t["count"] for t in terms) + 1)),
            )
        segments.append(segment)
    space = _ObjectiveSpace(frequencies, segments)
    space._dense = dense
    space._keys = [
        {name: state.terms[name]["keys"] for name in names} for state in states
    ]
    return space


def objective_residual(
    space: DataSpace, states: Sequence[ObjectiveState]
) -> DataVector:
    values = np.zeros(space.size, dtype=space.dtype)
    for frequency, state in zip(space.frequencies, states):
        for layout in space.term_layouts(frequency=frequency):
            term = state.terms[layout.id]
            if not term["has_residual"]:
                raise NotImplementedError(
                    "saved state has no objective residual; regenerate it with current Sauce"
                )
            values[layout.indices] = term["residual"][layout.row_ids - 1]
    return DataVector(values, space)


def objective_simulated(
    space: DataSpace, states: Sequence[ObjectiveState]
) -> DataVector:
    """Return the modeled receiver values of every objective row."""

    values = np.zeros(space.size, dtype=space.dtype)
    for frequency, state in zip(space.frequencies, states):
        for layout in space.term_layouts(frequency=frequency):
            term = state.terms[layout.id]
            if not term["has_simulated"]:
                raise NotImplementedError(
                    "saved state has no simulated values; regenerate it with current Sauce"
                )
            values[layout.indices] = term["simulated"][layout.row_ids - 1]
    return DataVector(values, space)


def residual_sign(states: Sequence[ObjectiveState]) -> float:
    """Return ``s`` with ``residual = s * (observed - simulated)`` shared by every term."""

    conventions = {
        term["convention"] for state in states for term in state.terms.values()
    }
    if len(conventions) != 1:
        raise ValueError("objective terms mix residual conventions")
    return 1.0 if conventions.pop() == OBSERVED_MINUS_SIMULATED else -1.0
