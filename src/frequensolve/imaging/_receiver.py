"""Verified physical receiver states and independent base/df dual artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

_FIELDS = (
    "frequency_hz",
    "receiver_group",
    "channel",
    "state_fingerprint",
    "partition",
    "value_convention",
    "physics",
    "dimension",
    "geometry",
    "modeling_mode",
    "control_kind",
    "source_mode",
    "units",
    "source_derivative",
    "candidate_fingerprint",
    "acquisition_fingerprint",
    "observation_fingerprint",
    "preprocessing_fingerprint",
    "control_registry_fingerprint",
    "resolved_context_fingerprint",
    "n_rows",
)
_COMMON = (
    "receiver_group",
    "channel",
    "value_convention",
    "physics",
    "dimension",
    "geometry",
    "modeling_mode",
    "control_kind",
    "source_mode",
    "units",
    "source_derivative",
    "candidate_fingerprint",
    "acquisition_fingerprint",
    "observation_fingerprint",
    "preprocessing_fingerprint",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _member(root: Path, entry: dict) -> Path:
    path = Path(entry["file"])
    if not path.is_absolute():
        path = root.parent / path
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"receiver artifact checksum differs: {path}")
    return path


@dataclass
class ReceiverState:
    path: Path
    manifest: dict
    checksum: str
    keys: np.ndarray
    owners: np.ndarray
    values: np.ndarray

    @classmethod
    def read(cls, path: Path) -> "ReceiverState":
        path = Path(path).resolve()
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != "fs-receiver-state-1":
            raise ValueError("expected fs-receiver-state-1")
        if any(field not in manifest for field in _FIELDS):
            raise ValueError("receiver state is missing required identity fields")
        if (
            manifest["channel"] not in {"base", "base_df"}
            or manifest["value_convention"] != "physical_receiver"
        ):
            raise ValueError("expected physical base or base_df receiver states")
        keys: list[np.ndarray] = []
        owners: list[np.ndarray] = []
        data: list[np.ndarray] = []
        for entry in manifest["shards"]:
            rank = entry["rank"]
            if not 0 <= rank < manifest["partition"]["n_ranks"]:
                raise ValueError("receiver shard has an invalid rank")
            with h5py.File(_member(path, entry), "r") as file:
                k = np.asarray(file["keys"])
                if (
                    k.ndim != 2
                    or k.shape[1] != 3
                    or k.dtype.kind not in "iu"
                    or np.any(k <= 0)
                ):
                    raise ValueError(
                        "receiver keys must be positive integer (shot, receiver, component) rows"
                    )
                columns: list[np.ndarray] = []
                for name in ("predicted", "predicted_df", "observed", "observed_df"):
                    if manifest["channel"] == "base" and name.endswith("_df"):
                        columns.append(np.zeros(len(k), complex))
                        continue
                    raw = np.asarray(file[name], dtype=float)
                    if raw.shape != (len(k), 2) or not np.isfinite(raw).all():
                        raise ValueError(f"invalid receiver values in {name}")
                    columns.append(raw[:, 0] + 1j * raw[:, 1])
                keys.append(k)
                owners.append(np.full(len(k), rank, dtype=np.int64))
                data.append(np.column_stack(columns))
        if not keys or not any(len(k) for k in keys):
            raise ValueError("receiver states need at least one receiver trace")
        key_array = np.concatenate(keys).astype(np.int64, copy=False)
        order = np.lexsort(key_array.T[::-1])
        key_array = key_array[order]
        if len(key_array) != manifest["n_rows"] or np.any(
            np.all(key_array[1:] == key_array[:-1], axis=1)
        ):
            raise ValueError("receiver keys are duplicated or missing")
        return cls(
            path,
            manifest,
            sha256(path),
            key_array,
            np.concatenate(owners)[order],
            np.concatenate(data)[order],
        )

    def write_dual(self, path: Path, base: np.ndarray, df: np.ndarray) -> Path:
        """Keep physical row identity and rank ownership, independent of shard ordering."""
        if sha256(self.path) != self.checksum:
            raise ValueError("receiver state changed after band assembly")
        base, df = np.asarray(base), np.asarray(df)
        if base.shape != (len(self.keys),) or df.shape != base.shape:
            raise ValueError("receiver dual shape differs from the state")
        if not np.isfinite(base).all() or not np.isfinite(df).all():
            raise ValueError("receiver dual must be finite")
        if self.manifest["channel"] == "base" and np.any(df != 0):
            raise ValueError("base receiver states cannot carry df duals")
        shards = []
        for rank in range(self.manifest["partition"]["n_ranks"]):
            selected = self.owners == rank
            target = path.with_name(f"{path.stem}_rank_{rank}.h5")
            with h5py.File(target, "w") as file:
                file["keys"] = self.keys[selected]
                for name, values in (("base", base), ("df", df)):
                    if name == "df" and self.manifest["channel"] == "base":
                        continue
                    file[name] = np.column_stack(
                        (values[selected].real, values[selected].imag)
                    )
            shards.append({"file": str(target), "rank": rank, "sha256": sha256(target)})
        vector = {name: self.manifest[name] for name in _FIELDS}
        vector.update(
            schema="fs-receiver-vector-1",
            role="dual",
            source_state=str(self.path),
            source_state_sha256=self.checksum,
            payload_precision="complex128",
            shards=shards,
        )
        _json(path, vector)
        return path


@dataclass
class ReceiverCollection:
    path: Path
    manifest: dict
    checksum: str
    groups: dict[str, ReceiverState]

    @classmethod
    def read(cls, path: Path) -> "ReceiverCollection":
        path = Path(path).resolve()
        manifest = json.loads(path.read_text())
        if manifest.get("schema") == "fs-receiver-state-1":
            state = ReceiverState.read(path)
            groups = {state.manifest["receiver_group"]: state}
        elif manifest.get("schema") == "fs-receiver-state-bundle-1":
            groups = {}
            for entry in manifest["groups"]:
                state = ReceiverState.read(_member(path, entry))
                name = state.manifest["receiver_group"]
                if name != entry["receiver_group"] or name in groups:
                    raise ValueError(
                        "receiver collection has duplicate or mismatched groups"
                    )
                groups[name] = state
            if len(groups) < 2:
                raise ValueError("receiver collection requires multiple groups")
            first = next(iter(groups.values())).manifest
            for state in groups.values():
                for field in (
                    "candidate_fingerprint",
                    "acquisition_fingerprint",
                    "resolved_context_fingerprint",
                    "control_registry_fingerprint",
                    "channel",
                    "source_derivative",
                    "frequency_hz",
                    "partition",
                    "field_retention",
                    "checkpoints",
                ):
                    if state.manifest.get(field) != first.get(field):
                        raise ValueError(f"receiver collection does not share {field}")
        else:
            raise ValueError("unsupported receiver state schema")
        return cls(path, manifest, sha256(path), groups)

    @property
    def fingerprint(self) -> str:
        return self.manifest["state_fingerprint"]

    def write_dual(
        self, path: Path, values: dict[str, tuple[np.ndarray, np.ndarray]]
    ) -> Path:
        if set(values) != set(self.groups):
            raise ValueError("receiver dual must cover every group")
        if sha256(self.path) != self.checksum:
            raise ValueError("receiver collection changed after band assembly")
        if len(self.groups) == 1:
            name = next(iter(self.groups))
            return self.groups[name].write_dual(path, *values[name])
        entries = []
        for index, (name, state) in enumerate(self.groups.items()):
            member = path.with_name(f"{path.stem}_group_{index + 1}.json")
            state.write_dual(member, *values[name])
            entries.append(
                {"receiver_group": name, "file": str(member), "sha256": sha256(member)}
            )
        _json(
            path,
            {
                "schema": "fs-receiver-vector-bundle-1",
                "role": "dual",
                "state_fingerprint": self.fingerprint,
                "source_state": str(self.path),
                "source_state_sha256": self.checksum,
                "groups": entries,
            },
        )
        return path


def receiver_band(
    states: list[ReceiverCollection],
    frequencies: tuple[complex, ...],
    source_derivative: str,
) -> dict[str, np.ndarray]:
    """Align physical trace keys across frequencies, retaining each state's native registry."""
    if len(states) != len(frequencies):
        raise ValueError("receiver band does not cover the requested frequencies")
    names = set(states[0].groups)
    if any(set(state.groups) != names for state in states):
        raise ValueError("receiver band group coverage differs")
    result = {}
    for name in sorted(names):
        first = states[0].groups[name]
        rows = []
        for collection, frequency in zip(states, frequencies):
            state = collection.groups[name]
            if any(state.manifest[field] != first.manifest[field] for field in _COMMON):
                raise ValueError(f"receiver band identity differs for {name}")
            if not np.array_equal(state.keys, first.keys):
                raise ValueError(f"receiver band key coverage differs for {name}")
            if not np.isclose(
                state.manifest["frequency_hz"], frequency.real, rtol=1e-12, atol=0
            ):
                raise ValueError(
                    "receiver state frequency differs from the requested band"
                )
            if state.manifest["source_derivative"] != source_derivative:
                raise ValueError("receiver source derivative policy differs")
            if state.manifest.get(
                "field_retention"
            ) != "checkpoint" or not state.manifest.get("checkpoints"):
                raise ValueError("receiver band requires retained receiver checkpoints")
            rows.append(state.values)
        result[name] = np.stack(rows, axis=1)
    return result
