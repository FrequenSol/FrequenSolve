"""Objective rows and residuals from Sauce's immutable saved state."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import h5py
import numpy as np

from .data import DataSpace, DataVector, TermLayout, _DataSegment, file_sha256


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
        self.n_ranks = int(manifest["partition"]["n_ranks"])
        if self.n_ranks < 1 or len(manifest["shards"]) != self.n_ranks:
            raise ValueError("objective state has an inconsistent rank partition")
        self.terms: Dict[str, Dict[str, Any]] = {}
        hashes: Dict[Path, str] = {}
        for entry in manifest["shards"]:
            shard = _resolve(self.path.parent, entry["file"])
            if file_sha256(shard) != entry["sha256"]:
                raise ValueError("objective state shard hash mismatch")
            payload = json.loads(shard.read_text())
            for term in payload["terms"]:
                cache = _resolve(shard.parent, term["cache"]["file"])
                expected = term["runtime"]["cache_fingerprint"]
                if cache not in hashes:
                    hashes[cache] = file_sha256(cache)
                if hashes[cache] != expected:
                    raise ValueError("objective state cache hash mismatch")
                with h5py.File(cache, "r") as h5:
                    group = h5[term["cache"].get("group", "/")]
                    ids = np.asarray(group["row_ids"], dtype=int).reshape(-1)
                    keys = np.asarray(group["coordinate_keys"], dtype=int).reshape(
                        -1, 3
                    )
                    count = int(np.asarray(group["n_global_rows"]).item())
                    residual = None
                    if "objective_residual" in group:
                        packed = np.asarray(group["objective_residual"])
                        residual = (packed[..., 0] + 1j * packed[..., 1]).reshape(-1)
                if term["id"] not in self.terms:
                    self.terms[term["id"]] = {
                        "receiver_group": term["receiver_group"],
                        "count": count,
                        "keys": np.zeros((count, 3), int),
                        "seen": np.zeros(count, bool),
                        "residual": np.zeros(count, complex),
                        "has_residual": True,
                    }
                item = self.terms[term["id"]]
                if count != item["count"] or keys.shape != (ids.size, 3):
                    raise ValueError("objective state has inconsistent row coordinates")
                if np.any(ids < 1) or np.any(ids > count):
                    raise ValueError("objective state has out-of-range row ids")
                if residual is not None and residual.size != ids.size:
                    raise ValueError("objective residual size differs from saved rows")
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
