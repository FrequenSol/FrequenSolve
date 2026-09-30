"""Objective rows, residuals and simulated data from Sauce's immutable saved state.

Per the ``fs-objective-linearization-3`` contract a term cache stores ``simulated`` (the modeled receiver values in
objective rows) and ``objective_residual``, the weighted **observed-minus-simulated** comparison error; the Jacobian
pullback of ``vjp`` is that residual's, i.e. ``-d(simulated)/dm`` for unit weights.  A writer following the opposite
convention records ``convention = "simulated_minus_observed"`` on the ``objective_residual`` dataset.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

OBSERVED_MINUS_SIMULATED = "observed_minus_simulated"
SIMULATED_MINUS_OBSERVED = "simulated_minus_observed"
_CONVENTIONS = (OBSERVED_MINUS_SIMULATED, SIMULATED_MINUS_OBSERVED)

import h5py
import numpy as np

from .data import (
    DataSpace,
    DataVector,
    TermLayout,
    _DataSegment,
    canonical_json_sha256,
    file_sha256,
)


def _resolve(parent: Path, name: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = parent / path
    if not path.is_file():
        path = parent / Path(name).name
    return path


class ObjectiveState:
    """Read saved coordinates independently of authored receiver group names."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        manifest = json.loads(self.path.read_text())
        schema = manifest.get("schema", "fs-objective-linearization-3")
        if schema not in {
            "fs-objective-linearization-3",
            "fs-objective-linearization-4",
        }:
            raise ValueError(f"unsupported objective state schema {schema!r}")
        self.vector_schema = schema.replace("linearization", "vector")
        canonical = schema == "fs-objective-linearization-4"
        self.n_ranks = 1 if canonical else int(manifest["partition"]["n_ranks"])
        if not canonical and (
            self.n_ranks < 1 or len(manifest["shards"]) != self.n_ranks
        ):
            raise ValueError("objective state has an inconsistent rank partition")
        if "manifest_fingerprint" in manifest:
            basis = {
                k: v
                for k, v in manifest.items()
                if k not in {"manifest_fingerprint", "state_fingerprint"}
            }
            if canonical_json_sha256(basis) != manifest["manifest_fingerprint"]:
                raise ValueError("objective state manifest fingerprint mismatch")
            if (
                manifest.get("state_fingerprint", manifest["manifest_fingerprint"])
                != manifest["manifest_fingerprint"]
            ):
                raise ValueError("objective state fingerprint mismatch")
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
                if cache not in hashes:
                    hashes[cache] = file_sha256(cache)
                if hashes[cache] != expected:
                    raise ValueError("objective state cache hash mismatch")
                with h5py.File(cache, "r") as h5:
                    group = h5[term["cache"].get("group", "/")]
                    keys = np.asarray(group["coordinate_keys"], dtype=int).reshape(
                        -1, 3
                    )
                    count = int(np.asarray(group["n_global_rows"]).item())
                    ids = (
                        np.arange(1, count + 1)
                        if canonical
                        else np.asarray(group["row_ids"], dtype=int).reshape(-1)
                    )
                    residual = None
                    convention = OBSERVED_MINUS_SIMULATED
                    if "objective_residual" in group:
                        dataset = group["objective_residual"]
                        packed = np.asarray(dataset)
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
                        packed = np.asarray(group["simulated"])
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
