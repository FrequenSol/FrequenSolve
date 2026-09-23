"""Readers and writers for the Sauce imaging artifact contracts.

This module owns the file formats exchanged with ``fwi_operator`` and the
native control-sensitivity workflows:

- ``fs-control-vector-1`` / ``fs-control-state-1`` HDF5 vectors
  (:class:`ControlVectorFile`, :class:`ControlStateFile`);
- ``fs-control-registry-1`` JSON manifests (:class:`ControlRegistryManifest`);
- ``fs-extension-vector-1`` HDF5 vectors (:class:`ExtensionVectorFile`);
- ``fs-objective-report-1``, ``fs-objective-balance-1`` and
  ``fs-extension-solve-1`` JSON reports;
- Cartesian image files (:class:`ImageSet`);
- the shared variational smoothing configuration (:class:`SmoothingConfig`).

Everything here is deliberately independent of the job layer so that the
operator and workflow layers can read artifacts without building jobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

__all__ = [
    "CONTROL_VECTOR_SCHEMA",
    "CONTROL_STATE_SCHEMA",
    "CONTROL_REGISTRY_SCHEMA",
    "EXTENSION_VECTOR_SCHEMA",
    "OBJECTIVE_REPORT_SCHEMA",
    "OBJECTIVE_BALANCE_SCHEMA",
    "EXTENSION_SOLVE_SCHEMA",
    "REAL_INTERLEAVED",
    "ControlVectorFile",
    "ControlStateFile",
    "RegistryBlock",
    "ControlRegistryManifest",
    "ExtensionVectorField",
    "ExtensionVectorFile",
    "ObjectiveTermReport",
    "ObjectiveReport",
    "BalanceTerm",
    "BalanceArtifact",
    "ReducedNormalReport",
    "ExtensionSolveReport",
    "ImageSet",
    "SmoothingConfig",
    "pack_support_mask",
    "unpack_support_mask",
    "qualified_block_name",
    "unqualified_block_name",
]

CONTROL_VECTOR_SCHEMA = "fs-control-vector-1"
CONTROL_STATE_SCHEMA = "fs-control-state-1"
CONTROL_REGISTRY_SCHEMA = "fs-control-registry-1"
EXTENSION_VECTOR_SCHEMA = "fs-extension-vector-1"
OBJECTIVE_REPORT_SCHEMA = "fs-objective-report-1"
OBJECTIVE_BALANCE_SCHEMA = "fs-objective-balance-1"
EXTENSION_SOLVE_SCHEMA = "fs-extension-solve-1"
REAL_INTERLEAVED = "real_interleaved"

_QUALIFIED_PREFIXES = ("model.", "source.", "reflectivity.")
_SOURCE_QUANTITIES = {"position", "mechanism", "signature", "signature_df"}


# ---------------------------------------------------------------------------
# Block naming helpers
# ---------------------------------------------------------------------------


def _validate_block_name(name: Any) -> str:
    text = str(name).strip()
    if not text or "/" in text or text in {".", ".."}:
        raise ValueError("control block names must be non-empty HDF5-safe names")
    return text


def qualified_block_name(name: str) -> str:
    """Return ``name`` as a qualified registry block name.

    Unqualified material control IDs receive the ``model.`` prefix. Names
    already carrying a ``model.``, ``source.<id>.<quantity>`` or
    ``reflectivity.`` prefix are validated and returned unchanged.
    """

    text = _validate_block_name(name)
    if text.startswith("source."):
        parts = text.split(".")
        if len(parts) != 3 or not parts[1] or parts[2] not in _SOURCE_QUANTITIES:
            raise ValueError(
                "source blocks must be named source.<id>.<position|mechanism|"
                f"signature|signature_df>; got {text!r}"
            )
        return text
    if text.startswith(("model.", "reflectivity.")):
        if len(text.split(".", 1)[1]) == 0:
            raise ValueError(f"qualified block name {text!r} has no block ID")
        return text
    return f"model.{text}"


def unqualified_block_name(name: str) -> str:
    """Return the unqualified material control ID behind a qualified name.

    Only ``model.<id>`` blocks (and bare IDs) have a native unqualified form.
    """

    text = _validate_block_name(name)
    if text.startswith("model."):
        return text[len("model.") :]
    if text.startswith(("source.", "reflectivity.")):
        raise ValueError(f"block {text!r} has no unqualified material-control spelling")
    return text


def _is_qualified(name: str) -> bool:
    return name.startswith(_QUALIFIED_PREFIXES)


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def _decode(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode(value.item())
        return [_decode(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_string(h5: h5py.File, name: str) -> Optional[str]:
    if name not in h5:
        return None
    value = _decode(h5[name][()])
    return None if value is None else str(value)


def _write_string(h5: h5py.File, name: str, value: str) -> None:
    if name in h5:
        del h5[name]
    h5.create_dataset(name, data=np.bytes_(str(value).encode("utf-8")))


def _real_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued (complex blocks interleave)")
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.array(array, copy=True)


def _interleave(values: Any) -> np.ndarray:
    """Pack a complex or real block as ``[Re c1, Im c1, Re c2, ...]``."""

    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = array.reshape(-1)
        out = np.empty(2 * array.size, dtype=np.float64)
        out[0::2] = array.real
        out[1::2] = array.imag
        return out
    return np.asarray(array, dtype=np.float64).reshape(-1)


def pack_support_mask(mask: Any) -> np.ndarray:
    """Pack a boolean per-DOF mask into uint8 bytes, eight DOFs per byte, LSB first."""

    bits = np.asarray(mask).reshape(-1).astype(bool)
    return np.packbits(bits.astype(np.uint8), bitorder="little")


def unpack_support_mask(packed: Any, size: int) -> np.ndarray:
    """Unpack ``size`` DOF flags from LSB-first uint8 bytes."""

    if isinstance(packed, (bytes, bytearray, memoryview)):
        data = np.frombuffer(bytes(packed), dtype=np.uint8)
    else:
        data = np.asarray(packed, dtype=np.uint8).reshape(-1)
    if 8 * data.size < size:
        raise ValueError(
            f"support mask has {8 * data.size} bits but the block has {size} DOFs"
        )
    return np.unpackbits(data, bitorder="little")[:size].astype(bool)


def _normalize_control_spaces(blocks, identities, key_fn):
    result = {}
    for name, identity in identities.items():
        key = key_fn(name)
        if key not in blocks:
            raise ValueError(f"control space names unknown block {key!r}")
        if key in result or not isinstance(identity, str) or not identity.strip():
            raise ValueError(f"invalid or duplicate control space identity for {key!r}")
        result[key] = identity.rstrip(" \0")
    return result


def _normalize_support(
    blocks: Mapping[str, np.ndarray],
    support: Mapping[str, Any],
    measure: Mapping[str, Any],
    key_of: Callable[[str], str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Validate per-block support masks and quantized measures against ``blocks``."""

    masks: Dict[str, np.ndarray] = {}
    for name, mask in dict(support).items():
        key = key_of(name)
        if key not in blocks:
            raise ValueError(f"support mask names unknown block {key!r}")
        flags = np.asarray(mask).reshape(-1).astype(bool)
        if flags.size != blocks[key].size:
            raise ValueError(f"support mask for {key!r} must have one flag per DOF")
        masks[key] = flags
    measures: Dict[str, np.ndarray] = {}
    for name, values in dict(measure).items():
        key = key_of(name)
        if key not in blocks:
            raise ValueError(f"support measure names unknown block {key!r}")
        quantized = np.asarray(values)
        if quantized.dtype != np.uint8:
            if np.any(quantized < 0) or np.any(quantized > 255):
                raise ValueError("support measures must fit in uint8")
            quantized = quantized.astype(np.uint8)
        quantized = quantized.reshape(-1)
        if quantized.size != blocks[key].size:
            raise ValueError(f"support measure for {key!r} must have one byte per DOF")
        measures[key] = quantized
    return masks, measures


def _normalize_min_support(value: Any) -> Optional[float]:
    if value is None:
        return None
    threshold = float(value)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("support_min_support must be a finite non-negative number")
    return threshold


def _write_support_datasets(
    h5: h5py.File,
    support: Mapping[str, np.ndarray],
    measure: Mapping[str, np.ndarray],
    min_support: Optional[float],
) -> None:
    """Write ``/support``, ``/support_measure`` and ``/support_min_support``."""

    if support:
        group = h5.create_group("support")
        for name, mask in support.items():
            group.create_dataset(name, data=pack_support_mask(mask), dtype=np.uint8)
    if measure:
        group = h5.create_group("support_measure")
        for name, values in measure.items():
            group.create_dataset(name, data=values, dtype=np.uint8)
    if min_support is not None:
        h5.create_dataset(
            "support_min_support", data=float(min_support), dtype=np.float64
        )


def _read_support_datasets(
    h5: h5py.File, blocks: Mapping[str, np.ndarray], path: Union[str, Path]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[float]]:
    """Decode the optional support datasets written next to ``/controls``."""

    support: Dict[str, np.ndarray] = {}
    if "support" in h5:
        for name in h5["support"]:
            key = str(name)
            if key not in blocks:
                raise ValueError(f"{path}: /support/{key} has no /controls block")
            support[key] = unpack_support_mask(
                h5["support"][name][()], blocks[key].size
            )
    measure: Dict[str, np.ndarray] = {}
    if "support_measure" in h5:
        for name in h5["support_measure"]:
            key = str(name)
            if key not in blocks:
                raise ValueError(
                    f"{path}: /support_measure/{key} has no /controls block"
                )
            measure[key] = np.asarray(
                h5["support_measure"][name][()], dtype=np.uint8
            ).reshape(-1)
    min_support = None
    if "support_min_support" in h5:
        min_support = float(np.asarray(h5["support_min_support"][()]).reshape(-1)[0])
    return support, measure, min_support


# ---------------------------------------------------------------------------
# Control vectors and states
# ---------------------------------------------------------------------------


@dataclass
class ControlVectorFile:
    """One ``fs-control-vector-1`` direction or covector.

    Args:
        blocks: Ordered mapping ``qualified block -> real coordinates``. Complex
            arrays are interleaved on assignment; every stored block is a real
            one-dimensional ``float64`` array.
        state_fingerprint: Fingerprint of the frozen objective state the vector
            is bound to. Required unless ``native`` is true.
        control_registry_fingerprint: Fingerprint of the resolved control
            registry. Required unless ``native`` is true.
        native: When true, write the legacy ``control_sensitivities`` layout:
            unqualified ``/controls/<id>`` datasets and no identity strings.
        support: Optional ``block -> bool mask`` of supported DOFs, as written
            by ``linearize``/``vjp``/``normal`` covectors to ``/support/<block>``
            (LSB-first packed bits, eight DOFs per byte). Missing blocks are
            treated as fully supported.
        support_measure: Optional ``block -> uint8`` quantized derivative
            measure (``/support_measure/<block>``).
        support_min_support: Optional relative support threshold the writer
            used (``/support_min_support``).
    """

    blocks: Dict[str, np.ndarray]
    state_fingerprint: Optional[str] = None
    control_registry_fingerprint: Optional[str] = None
    native: bool = False
    support: Dict[str, np.ndarray] = field(default_factory=dict)
    support_measure: Dict[str, np.ndarray] = field(default_factory=dict)
    support_min_support: Optional[float] = None
    control_spaces: Dict[str, str] = field(default_factory=dict)

    def _block_key(self, name: str) -> str:
        key = _validate_block_name(name)
        return unqualified_block_name(key) if self.native else qualified_block_name(key)

    def __post_init__(self) -> None:
        ordered: Dict[str, np.ndarray] = {}
        for name, values in dict(self.blocks).items():
            key = self._block_key(name)
            if key in ordered:
                raise ValueError(f"duplicate control block {key!r}")
            ordered[key] = _real_vector(_interleave(values), f"block {key!r}")
        self.blocks = ordered
        self.control_spaces = _normalize_control_spaces(
            self.blocks, self.control_spaces, self._block_key
        )
        self.support, self.support_measure = _normalize_support(
            self.blocks, self.support, self.support_measure, self._block_key
        )
        self.support_min_support = _normalize_min_support(self.support_min_support)
        if not self.native:
            for label in ("state_fingerprint", "control_registry_fingerprint"):
                value = getattr(self, label)
                if value is not None and not str(value).strip():
                    raise ValueError(f"{label} must be a non-empty string")

    # -- vector-space helpers -------------------------------------------------

    @property
    def names(self) -> Tuple[str, ...]:
        """Return the ordered block names."""

        return tuple(self.blocks)

    @property
    def sizes(self) -> Dict[str, int]:
        """Return the real DOF count of each block."""

        return {name: int(values.size) for name, values in self.blocks.items()}

    @property
    def size(self) -> int:
        """Return the total real DOF count."""

        return int(sum(values.size for values in self.blocks.values()))

    def __getitem__(self, name: str) -> np.ndarray:
        key = _validate_block_name(name)
        candidates = [key, qualified_block_name(key)]
        if key.startswith("model.") or not _is_qualified(key):
            candidates.append(unqualified_block_name(key))
        for candidate in candidates:
            if candidate in self.blocks:
                return self.blocks[candidate]
        raise KeyError(name)

    def support_mask(self, name: str) -> np.ndarray:
        """Return the boolean support mask for ``name`` (all true when absent)."""

        key = self._block_key(name)
        if key in self.support:
            return self.support[key]
        return np.ones(self[name].size, dtype=bool)

    def pack(self, order: Optional[Sequence[str]] = None) -> np.ndarray:
        """Concatenate blocks in ``order`` (default: stored order)."""

        names = list(self.blocks) if order is None else [str(n) for n in order]
        return (
            np.concatenate([self[name] for name in names])
            if names
            else np.zeros(0, dtype=np.float64)
        )

    @classmethod
    def from_packed(
        cls,
        vector: Any,
        sizes: Mapping[str, int],
        *,
        state_fingerprint: Optional[str] = None,
        control_registry_fingerprint: Optional[str] = None,
        native: bool = False,
    ) -> "ControlVectorFile":
        """Split one packed vector into blocks using an ordered size mapping."""

        values = _real_vector(vector, "control vector")
        expected = int(sum(int(size) for size in sizes.values()))
        if values.size != expected:
            raise ValueError(
                f"control vector has {values.size} entries; blocks require {expected}"
            )
        blocks = {}
        offset = 0
        for name, size in sizes.items():
            blocks[name] = values[offset : offset + int(size)]
            offset += int(size)
        return cls(
            blocks,
            state_fingerprint=state_fingerprint,
            control_registry_fingerprint=control_registry_fingerprint,
            native=native,
        )

    # -- I/O ------------------------------------------------------------------

    def write(self, path: Union[str, Path]) -> Path:
        """Write the vector file and return its path."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.native:
            missing = [
                label
                for label in ("state_fingerprint", "control_registry_fingerprint")
                if getattr(self, label) is None
            ]
            if missing:
                raise ValueError(
                    f"{CONTROL_VECTOR_SCHEMA} files require {', '.join(missing)}"
                )
        with h5py.File(path, "w") as h5:
            controls = h5.create_group("controls")
            for name, values in self.blocks.items():
                controls.create_dataset(name, data=values, dtype=np.float64)
            for name, identity in self.control_spaces.items():
                _write_string(h5, "control_spaces/" + self._block_key(name), identity)
            if not self.native:
                _write_string(h5, "schema", CONTROL_VECTOR_SCHEMA)
                _write_string(h5, "packing", REAL_INTERLEAVED)
                _write_string(h5, "state_fingerprint", str(self.state_fingerprint))
                _write_string(
                    h5,
                    "control_registry_fingerprint",
                    str(self.control_registry_fingerprint),
                )
            _write_support_datasets(
                h5, self.support, self.support_measure, self.support_min_support
            )
        return path

    @classmethod
    def read(
        cls, path: Union[str, Path], *, native: Optional[bool] = None
    ) -> "ControlVectorFile":
        """Read a vector file.

        ``native`` defaults to the file's layout: files without ``/schema``
        are treated as native control-sensitivity vectors.
        """

        with h5py.File(path, "r") as h5:
            schema = _read_string(h5, "schema")
            if native is None:
                native = schema is None
            if not native:
                if schema != CONTROL_VECTOR_SCHEMA:
                    raise ValueError(
                        f"{path} has schema {schema!r}; expected {CONTROL_VECTOR_SCHEMA!r}"
                    )
                packing = _read_string(h5, "packing")
                if packing != REAL_INTERLEAVED:
                    raise ValueError(f"{path} has unsupported packing {packing!r}")
            elif schema is not None:
                raise ValueError(
                    f"{path} carries schema {schema!r}; not a native vector"
                )
            blocks = _read_control_blocks(h5, path)
            support, measure, min_support = _read_support_datasets(h5, blocks, path)
            return cls(
                blocks,
                state_fingerprint=(
                    None if native else _read_string(h5, "state_fingerprint")
                ),
                control_registry_fingerprint=(
                    None if native else _read_string(h5, "control_registry_fingerprint")
                ),
                native=native,
                control_spaces={
                    str(n): _read_string(h5["control_spaces"], n)
                    for n in h5.get("control_spaces", {})
                },
                support=support,
                support_measure=measure,
                support_min_support=min_support,
            )


def _read_control_blocks(
    h5: h5py.File, path: Union[str, Path]
) -> Dict[str, np.ndarray]:
    if "controls" not in h5:
        raise ValueError(f"{path} has no /controls group")
    group = h5["controls"]
    blocks: Dict[str, np.ndarray] = {}
    for name in sorted(group):
        item = group[name]
        if not isinstance(item, h5py.Dataset):
            raise ValueError(f"{path}: /controls/{name} is not a dataset")
        blocks[str(name)] = np.asarray(item[()], dtype=np.float64).reshape(-1)
    return blocks


def _normalize_scaling(
    blocks: Mapping[str, np.ndarray],
    scaling: Mapping[str, Any],
    units: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Validate ``/scaling`` and ``/scaling_units`` against the state blocks."""

    out_scaling: Dict[str, float] = {}
    for name, value in dict(scaling or {}).items():
        key = qualified_block_name(name)
        if key not in blocks:
            raise ValueError(f"/scaling/{key} has no /controls block")
        number = float(np.asarray(value, dtype=np.float64).reshape(()))
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"/scaling/{key} must be finite and positive")
        out_scaling[key] = number
    out_units: Dict[str, str] = {}
    for name, value in dict(units or {}).items():
        key = qualified_block_name(name)
        if key not in out_scaling:
            raise ValueError(f"/scaling_units/{key} has no /scaling/{key}")
        text = _decode(value) if isinstance(value, (bytes, np.bytes_)) else value
        text = str(text).strip()
        if not text:
            raise ValueError(f"/scaling_units/{key} must be a non-empty string")
        out_units[key] = text
    return out_scaling, out_units


def _read_scaling_datasets(
    h5: h5py.File, blocks: Mapping[str, np.ndarray], path: Union[str, Path]
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """Read the optional ``/scaling`` and ``/scaling_units`` groups."""

    scaling: Dict[str, float] = {}
    units: Dict[str, str] = {}
    for group_name in ("scaling", "scaling_units"):
        if group_name not in h5:
            continue
        for name in h5[group_name]:
            key = str(name)
            if key not in blocks:
                raise ValueError(f"{path}: /{group_name}/{key} has no /controls block")
            value = h5[group_name][name][()]
            if group_name == "scaling":
                scaling[key] = float(value)
            else:
                units[key] = str(_decode(value))
    return scaling, units


@dataclass
class ControlStateFile:
    """One complete ``fs-control-state-1`` baseline over every registry block.

    Args:
        blocks: Ordered mapping ``qualified block -> real coordinates``.
        support: Optional ``block -> bool mask`` marking supported DOFs. Written
            to ``/support/<block>`` as LSB-first packed bits (eight DOFs per
            byte); missing blocks are treated as fully supported.
        support_measure: Optional ``block -> uint8`` quantized derivative
            measure written to ``/support_measure/<block>``.
        support_min_support: Optional relative support threshold the writer
            used (``/support_min_support``).
        scaling: Optional ``block -> float`` physical strength of one stored
            coordinate (``/scaling/<block>``).  Sauce writes it for every
            ``source.<i>.mechanism`` block because mechanism coordinates are
            the writing task's nondimensional source-load components; on
            import Sauce multiplies the block by ``stored / current`` so the
            physical source ``coordinate * scaling`` is the same in every
            task.  A block without scaling is read in the reading task's
            coordinates.
        scaling_units: Optional ``block -> units`` of ``scaling``
            (``/scaling_units/<block>``, e.g. ``"N"`` or ``"N*m"``); every
            entry needs a ``scaling`` entry.
    """

    blocks: Dict[str, np.ndarray]
    support: Dict[str, np.ndarray] = field(default_factory=dict)
    support_measure: Dict[str, np.ndarray] = field(default_factory=dict)
    support_min_support: Optional[float] = None
    scaling: Dict[str, float] = field(default_factory=dict)
    scaling_units: Dict[str, str] = field(default_factory=dict)
    control_spaces: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ordered: Dict[str, np.ndarray] = {}
        for name, values in dict(self.blocks).items():
            key = qualified_block_name(name)
            if key in ordered:
                raise ValueError(f"duplicate control block {key!r}")
            ordered[key] = _real_vector(_interleave(values), f"block {key!r}")
        self.blocks = ordered
        self.control_spaces = _normalize_control_spaces(
            self.blocks, self.control_spaces, qualified_block_name
        )
        self.support, self.support_measure = _normalize_support(
            self.blocks, self.support, self.support_measure, qualified_block_name
        )
        self.support_min_support = _normalize_min_support(self.support_min_support)
        self.scaling, self.scaling_units = _normalize_scaling(
            self.blocks, self.scaling, self.scaling_units
        )

    @property
    def names(self) -> Tuple[str, ...]:
        """Return the ordered block names."""

        return tuple(self.blocks)

    @property
    def sizes(self) -> Dict[str, int]:
        """Return the real DOF count of each block."""

        return {name: int(values.size) for name, values in self.blocks.items()}

    def __getitem__(self, name: str) -> np.ndarray:
        return self.blocks[qualified_block_name(name)]

    def support_mask(self, name: str) -> np.ndarray:
        """Return the boolean support mask for ``name`` (all true when absent)."""

        key = qualified_block_name(name)
        if key in self.support:
            return self.support[key]
        return np.ones(self.blocks[key].size, dtype=bool)

    def restrict(self, names: Sequence[str]) -> ControlVectorFile:
        """Return the active-subspace slice as an (unbound) control vector.

        Support masks and measures of the selected blocks travel with it.
        """

        keys = [qualified_block_name(name) for name in names]
        return ControlVectorFile(
            {key: self[key] for key in keys},
            control_spaces={
                key: self.control_spaces[key]
                for key in keys
                if key in self.control_spaces
            },
            support={key: self.support[key] for key in keys if key in self.support},
            support_measure={
                key: self.support_measure[key]
                for key in keys
                if key in self.support_measure
            },
            support_min_support=self.support_min_support,
        )

    def with_update(self, vector: ControlVectorFile) -> "ControlStateFile":
        """Return a new state whose blocks are replaced by ``vector``'s blocks."""

        blocks = dict(self.blocks)
        for name, values in vector.blocks.items():
            key = qualified_block_name(name)
            if key not in blocks:
                raise ValueError(f"vector block {key!r} is not in the state")
            if values.size != blocks[key].size:
                raise ValueError(f"vector block {key!r} has the wrong size")
            identity = vector.control_spaces.get(name)
            if identity and key in self.control_spaces:
                if identity != self.control_spaces[key]:
                    raise ValueError(
                        f"vector block {key!r} has a different control basis"
                    )
            blocks[key] = values
        return ControlStateFile(
            blocks,
            support=dict(self.support),
            support_measure=dict(self.support_measure),
            support_min_support=self.support_min_support,
            scaling=dict(self.scaling),
            scaling_units=dict(self.scaling_units),
            control_spaces=dict(self.control_spaces),
        )

    def write(self, path: Union[str, Path]) -> Path:
        """Write the state file and return its path."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5:
            _write_string(h5, "schema", CONTROL_STATE_SCHEMA)
            _write_string(h5, "packing", REAL_INTERLEAVED)
            controls = h5.create_group("controls")
            for name, values in self.blocks.items():
                controls.create_dataset(name, data=values, dtype=np.float64)
            for name, identity in self.control_spaces.items():
                _write_string(
                    h5, "control_spaces/" + qualified_block_name(name), identity
                )
            _write_support_datasets(
                h5, self.support, self.support_measure, self.support_min_support
            )
            if self.scaling:
                group = h5.create_group("scaling")
                for name, value in self.scaling.items():
                    group.create_dataset(name, data=float(value), dtype=np.float64)
            if self.scaling_units:
                group = h5.create_group("scaling_units")
                for name, units in self.scaling_units.items():
                    group.create_dataset(name, data=np.bytes_(units.encode("utf-8")))
        return path

    @classmethod
    def read(cls, path: Union[str, Path]) -> "ControlStateFile":
        """Read a complete baseline, decoding packed support masks when present."""

        with h5py.File(path, "r") as h5:
            schema = _read_string(h5, "schema")
            if schema != CONTROL_STATE_SCHEMA:
                raise ValueError(
                    f"{path} has schema {schema!r}; expected {CONTROL_STATE_SCHEMA!r}"
                )
            packing = _read_string(h5, "packing")
            if packing != REAL_INTERLEAVED:
                raise ValueError(f"{path} has unsupported packing {packing!r}")
            blocks = _read_control_blocks(h5, path)
            support, measure, min_support = _read_support_datasets(h5, blocks, path)
            scaling, units = _read_scaling_datasets(h5, blocks, path)
            identities = {
                str(n): _read_string(h5["control_spaces"], n)
                for n in h5.get("control_spaces", {})
            }
        return cls(
            blocks,
            support=support,
            support_measure=measure,
            support_min_support=min_support,
            scaling=scaling,
            scaling_units=units,
            control_spaces=identities,
        )


# ---------------------------------------------------------------------------
# Control registry manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryBlock:
    """One resolved registry block from ``fs-control-registry-1``."""

    id: int
    name: str
    binding: Tuple[int, int, int]
    layout: Tuple[int, int, int, int]
    units: str
    actions: int
    transform: int
    scaling: Tuple[float, float, float, float]
    basis_identity: str
    distributed: bool
    components: Tuple[str, ...] = ()
    global_dofs: Optional[int] = None
    global_ids: Tuple[int, ...] = ()
    owned_global_ids: Tuple[int, ...] = ()

    @property
    def offset(self) -> int:
        """Return the zero-based offset of this block in the full state."""

        return int(self.layout[0]) - 1

    @property
    def size(self) -> int:
        """Return the real DOF count of this block."""

        return int(self.layout[1])

    @property
    def physical_components(self) -> int:
        """Return the number of physical components per DOF group."""

        return int(self.layout[2])

    @property
    def complex(self) -> bool:
        """Return whether the block interleaves complex coordinates."""

        return int(self.layout[3]) == 2

    @property
    def supports_jvp(self) -> bool:
        """Return whether the block participates in JVP actions."""

        return bool(self.actions & 1)

    @property
    def supports_vjp(self) -> bool:
        """Return whether the block participates in VJP actions."""

        return bool(self.actions & 2)

    @property
    def slice(self) -> slice:
        """Return the full-state slice of this block."""

        return slice(self.offset, self.offset + self.size)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegistryBlock":
        """Build a block from its manifest entry."""

        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            binding=tuple(int(v) for v in data["binding"]),  # type: ignore[arg-type]
            layout=tuple(int(v) for v in data["layout"]),  # type: ignore[arg-type]
            units=str(data["units"]),
            actions=int(data["actions"]),
            transform=int(data["transform"]),
            scaling=tuple(float(v) for v in data["scaling"]),  # type: ignore[arg-type]
            basis_identity=str(data.get("basis_identity", "")),
            distributed=bool(data["distributed"]),
            components=tuple(str(c) for c in data.get("components", ())),
            global_dofs=(
                None if data.get("global_dofs") is None else int(data["global_dofs"])
            ),
            global_ids=tuple(int(v) for v in data.get("global_ids", ())),
            owned_global_ids=tuple(int(v) for v in data.get("owned_global_ids", ())),
        )


@dataclass
class ControlRegistryManifest:
    """Reader for ``fs-control-registry-1`` JSON manifests."""

    fingerprint: str
    blocks: Tuple[RegistryBlock, ...]
    active_blocks: Tuple[int, ...]
    active_offsets: Tuple[int, ...]
    coordinates: np.ndarray
    values: np.ndarray
    packing: str = REAL_INTERLEAVED
    pairing: str = "real_euclidean"
    descriptor_rank: int = 0
    n_ranks: int = 1
    rank_descriptors: Tuple[Dict[str, str], ...] = ()
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ControlRegistryManifest":
        """Build a manifest from its JSON mapping."""

        schema = data.get("schema")
        if schema != CONTROL_REGISTRY_SCHEMA:
            raise ValueError(
                f"control registry has schema {schema!r}; expected "
                f"{CONTROL_REGISTRY_SCHEMA!r}"
            )
        blocks = tuple(RegistryBlock.from_dict(block) for block in data["blocks"])
        ids = [block.id for block in blocks]
        if len(set(ids)) != len(ids):
            raise ValueError("control registry block ids must be unique")
        return cls(
            fingerprint=str(data["fingerprint"]),
            blocks=blocks,
            active_blocks=tuple(int(v) for v in data.get("active_blocks", ())),
            active_offsets=tuple(int(v) for v in data.get("active_offsets", ())),
            coordinates=np.asarray(data.get("coordinates", ()), dtype=np.float64),
            values=np.asarray(data.get("values", ()), dtype=np.float64),
            packing=str(data.get("packing", REAL_INTERLEAVED)),
            pairing=str(data.get("pairing", "real_euclidean")),
            descriptor_rank=int(data.get("descriptor_rank", 0)),
            n_ranks=int(data.get("n_ranks", 1)),
            rank_descriptors=tuple(
                {"file": str(item["file"]), "fingerprint": str(item["fingerprint"])}
                for item in data.get("rank_descriptors", ())
            ),
            raw=dict(data),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ControlRegistryManifest":
        """Read a manifest JSON file."""

        with open(path, "r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    @property
    def names(self) -> Tuple[str, ...]:
        """Return every registered block name in registry order."""

        return tuple(block.name for block in self.blocks)

    @property
    def active_names(self) -> Tuple[str, ...]:
        """Return active block names in the authoritative ordered selection."""

        by_id = {block.id: block for block in self.blocks}
        try:
            return tuple(by_id[block_id].name for block_id in self.active_blocks)
        except KeyError as exc:
            raise ValueError(f"active block id {exc} is not registered") from exc

    def block(self, name: str) -> RegistryBlock:
        """Return the block named ``name`` (qualified or bare material ID)."""

        key = qualified_block_name(name)
        for block in self.blocks:
            if block.name == key:
                return block
        raise KeyError(name)

    def bindings(self) -> Dict[str, Tuple[int, int, int]]:
        """Return ``name -> (provider, quantity, entity)`` bindings."""

        return {block.name: block.binding for block in self.blocks}

    def layout(self) -> Dict[str, slice]:
        """Return the full-state slice of every block."""

        return {block.name: block.slice for block in self.blocks}

    @property
    def state_size(self) -> int:
        """Return the total real DOF count of the full state."""

        return int(sum(block.size for block in self.blocks))

    @property
    def active_size(self) -> int:
        """Return the total real DOF count of the active subspace."""

        return int(sum(self.block(name).size for name in self.active_names))

    def active_layout(self) -> Dict[str, slice]:
        """Return the active-vector slice of every active block.

        ``active_offsets`` is indexed by block id and holds one-based offsets
        into the concatenated active vector, or zero for inactive blocks.
        """

        layout: Dict[str, slice] = {}
        for block in self.blocks:
            if block.id - 1 >= len(self.active_offsets):
                continue
            offset = self.active_offsets[block.id - 1]
            if offset > 0:
                layout[block.name] = slice(offset - 1, offset - 1 + block.size)
        return layout

    def unpack_state(self, vector: Optional[Any] = None) -> Dict[str, np.ndarray]:
        """Split a full-state vector (default: ``values``) into blocks."""

        values = self.values if vector is None else _real_vector(vector, "state vector")
        if values.size != self.state_size:
            raise ValueError(
                f"state vector has {values.size} entries; registry has {self.state_size}"
            )
        return {block.name: values[block.slice] for block in self.blocks}


# ---------------------------------------------------------------------------
# Extension vectors
# ---------------------------------------------------------------------------


@dataclass
class ExtensionVectorField:
    """One extension field: ``spatial_count x n_axis`` real taps.

    ``axis`` is ``"lag"`` for time-lag fields or ``"offset"`` for spatial
    half-offset fields; column ``k`` (zero-based) is stored as
    ``/fields/<p>/<axis>/<k+1>``.
    """

    values: np.ndarray
    axis: str = "lag"
    control: Optional[str] = None

    def __post_init__(self) -> None:
        axis = str(self.axis).strip().lower()
        if axis not in {"lag", "offset"}:
            raise ValueError("extension axis must be 'lag' or 'offset'")
        values = np.asarray(self.values)
        if np.iscomplexobj(values):
            raise ValueError("extension taps must be real")
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("extension taps must be a (spatial_count, n_axis) array")
        if not np.all(np.isfinite(values)):
            raise ValueError("extension taps must be finite")
        self.axis = axis
        self.values = np.array(values, copy=True)
        if self.control is not None:
            self.control = _validate_block_name(self.control)

    @property
    def spatial_count(self) -> int:
        """Return the number of spatial control DOFs."""

        return int(self.values.shape[0])

    @property
    def n_axis(self) -> int:
        """Return the number of lag or offset columns."""

        return int(self.values.shape[1])


@dataclass
class ExtensionVectorFile:
    """One ``fs-extension-vector-1`` tangent or covector."""

    fields: List[ExtensionVectorField]
    fingerprint: Optional[str] = None
    baseline: Optional[str] = None
    role: str = "tangent"

    def __post_init__(self) -> None:
        role = str(self.role).strip().lower()
        if role not in {"tangent", "covector"}:
            raise ValueError("extension vector role must be 'tangent' or 'covector'")
        self.role = role
        self.fields = list(self.fields)
        if not self.fields:
            raise ValueError("extension vectors require at least one field")

    @property
    def size(self) -> int:
        """Return the total tap count."""

        return int(sum(field_.values.size for field_ in self.fields))

    def pack(self) -> np.ndarray:
        """Concatenate fields, spatial index fastest, then axis, then field."""

        return np.concatenate(
            [field_.values.reshape(-1, order="F") for field_ in self.fields]
        )

    @classmethod
    def from_packed(
        cls,
        vector: Any,
        template: "ExtensionVectorFile",
        *,
        role: Optional[str] = None,
    ) -> "ExtensionVectorFile":
        """Rebuild a vector with ``template``'s layout from a packed array."""

        values = _real_vector(vector, "extension vector")
        if values.size != template.size:
            raise ValueError(
                f"extension vector has {values.size} entries; template has {template.size}"
            )
        fields = []
        offset = 0
        for field_ in template.fields:
            count = field_.values.size
            fields.append(
                ExtensionVectorField(
                    values[offset : offset + count].reshape(
                        field_.values.shape, order="F"
                    ),
                    axis=field_.axis,
                    control=field_.control,
                )
            )
            offset += count
        return cls(
            fields,
            fingerprint=template.fingerprint,
            baseline=template.baseline,
            role=template.role if role is None else role,
        )

    def write(self, path: Union[str, Path]) -> Path:
        """Write the vector file and return its path."""

        if self.fingerprint is None or self.baseline is None:
            raise ValueError(
                f"{EXTENSION_VECTOR_SCHEMA} files require fingerprint and baseline"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as h5:
            _write_string(h5, "schema", EXTENSION_VECTOR_SCHEMA)
            _write_string(h5, "fingerprint", str(self.fingerprint))
            _write_string(h5, "baseline", str(self.baseline))
            _write_string(h5, "role", self.role)
            root = h5.create_group("fields")
            for p, field_ in enumerate(self.fields, start=1):
                group = root.create_group(f"{p}/{field_.axis}")
                if field_.control is not None:
                    group.parent.attrs["control"] = field_.control
                for k in range(field_.n_axis):
                    group.create_dataset(
                        str(k + 1), data=field_.values[:, k], dtype=np.float64
                    )
        return path

    @classmethod
    def read(cls, path: Union[str, Path]) -> "ExtensionVectorFile":
        """Read a vector file."""

        with h5py.File(path, "r") as h5:
            schema = _read_string(h5, "schema")
            if schema != EXTENSION_VECTOR_SCHEMA:
                raise ValueError(
                    f"{path} has schema {schema!r}; expected {EXTENSION_VECTOR_SCHEMA!r}"
                )
            if "fields" not in h5:
                raise ValueError(f"{path} has no /fields group")
            fields: List[ExtensionVectorField] = []
            root = h5["fields"]
            for p in sorted(root, key=lambda name: int(name)):
                field_group = root[p]
                axes = [name for name in field_group if name in {"lag", "offset"}]
                if len(axes) != 1:
                    raise ValueError(
                        f"{path}: field {p} must have one lag or offset axis"
                    )
                axis_group = field_group[axes[0]]
                columns = sorted(axis_group, key=lambda name: int(name))
                if [int(c) for c in columns] != list(range(1, len(columns) + 1)):
                    raise ValueError(f"{path}: field {p} axis indices must be 1..n")
                values = np.stack(
                    [
                        np.asarray(axis_group[c][()], dtype=np.float64).reshape(-1)
                        for c in columns
                    ],
                    axis=1,
                )
                control = field_group.attrs.get("control")
                fields.append(
                    ExtensionVectorField(
                        values,
                        axis=axes[0],
                        control=None if control is None else str(_decode(control)),
                    )
                )
            return cls(
                fields,
                fingerprint=_read_string(h5, "fingerprint"),
                baseline=_read_string(h5, "baseline"),
                role=_read_string(h5, "role") or "tangent",
            )


# ---------------------------------------------------------------------------
# JSON reports
# ---------------------------------------------------------------------------


def _load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _require_schema(data: Mapping[str, Any], expected: str) -> None:
    schema = data.get("schema")
    if schema != expected:
        raise ValueError(f"report has schema {schema!r}; expected {expected!r}")


@dataclass(frozen=True)
class ObjectiveTermReport:
    """One objective term of an ``fs-objective-report-1`` report."""

    id: str
    raw_sum: float
    effective_weight_mass: float
    normalized_value: float
    weight: float
    weighted_value: float
    active_samples: int
    scale: Tuple[float, ...]
    robust: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectiveTermReport":
        """Build a term from its JSON mapping."""

        return cls(
            id=str(data["id"]),
            raw_sum=float(data["raw_sum"]),
            effective_weight_mass=float(data["effective_weight_mass"]),
            normalized_value=float(data["normalized_value"]),
            weight=float(data["weight"]),
            weighted_value=float(data["weighted_value"]),
            active_samples=int(data["active_samples"]),
            scale=tuple(float(v) for v in data["scale"]),
            robust=dict(data.get("robust") or {}),
        )


@dataclass
class ObjectiveReport:
    """Reader for ``fs-objective-report-1`` task-local objective reports."""

    total: float
    terms: Tuple[ObjectiveTermReport, ...]
    state_fingerprint: Optional[str] = None
    runtime: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectiveReport":
        """Build a report from its JSON mapping."""

        _require_schema(data, OBJECTIVE_REPORT_SCHEMA)
        terms = tuple(ObjectiveTermReport.from_dict(term) for term in data["terms"])
        ids = [term.id for term in terms]
        if len(set(ids)) != len(ids):
            raise ValueError("objective report term ids must be unique")
        return cls(
            total=float(data["total"]),
            terms=terms,
            state_fingerprint=(
                None
                if data.get("state_fingerprint") is None
                else str(data["state_fingerprint"])
            ),
            runtime=dict(data.get("runtime") or {}),
            raw=dict(data),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ObjectiveReport":
        """Read a report JSON file."""

        return cls.from_dict(_load_json(path))

    def term(self, term_id: str) -> ObjectiveTermReport:
        """Return the term with id ``term_id``."""

        for term in self.terms:
            if term.id == term_id:
                return term
        raise KeyError(term_id)

    @property
    def weighted_values(self) -> Dict[str, float]:
        """Return ``term id -> weighted contribution`` to the total."""

        return {term.id: term.weighted_value for term in self.terms}


@dataclass(frozen=True)
class BalanceTerm:
    """One calibrated term of an ``fs-objective-balance-1`` artifact."""

    id: str
    receiver_group: str
    compatibility_fingerprint: str
    components: Tuple[str, ...]
    dimensions: Tuple[str, ...]
    units: Tuple[str, ...]
    scale: Tuple[float, ...]
    effective_weight_mass: float
    active_samples: Optional[int] = None
    statistics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BalanceTerm":
        """Build a term from its JSON mapping."""

        return cls(
            id=str(data["id"]),
            receiver_group=str(data["receiver_group"]),
            compatibility_fingerprint=str(data["compatibility_fingerprint"]),
            components=tuple(str(v) for v in data["components"]),
            dimensions=tuple(str(v) for v in data["dimensions"]),
            units=tuple(str(v) for v in data["units"]),
            scale=tuple(float(v) for v in data["scale"]),
            effective_weight_mass=float(data["effective_weight_mass"]),
            active_samples=(
                None
                if data.get("active_samples") is None
                else int(data["active_samples"])
            ),
            statistics=dict(data.get("statistics") or {}),
        )


@dataclass
class BalanceArtifact:
    """Reader for ``fs-objective-balance-1`` calibration artifacts."""

    frequency_hz: float
    laplace_damping_hz: float
    terms: Tuple[BalanceTerm, ...]
    fingerprints: Dict[str, str]
    runtime: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BalanceArtifact":
        """Build an artifact from its JSON mapping."""

        _require_schema(data, OBJECTIVE_BALANCE_SCHEMA)
        return cls(
            frequency_hz=float(data["frequency_hz"]),
            laplace_damping_hz=float(data["laplace_damping_hz"]),
            terms=tuple(BalanceTerm.from_dict(term) for term in data["terms"]),
            fingerprints={
                str(k): str(v) for k, v in dict(data["fingerprints"]).items()
            },
            runtime=dict(data.get("runtime") or {}),
            raw=dict(data),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BalanceArtifact":
        """Read an artifact JSON file."""

        return cls.from_dict(_load_json(path))

    def term(self, term_id: str) -> BalanceTerm:
        """Return the term with id ``term_id``."""

        for term in self.terms:
            if term.id == term_id:
                return term
        raise KeyError(term_id)

    @property
    def scales(self) -> Dict[str, Tuple[float, ...]]:
        """Return ``term id -> component scales``."""

        return {term.id: term.scale for term in self.terms}


@dataclass(frozen=True)
class ReducedNormalReport:
    """The nested ``reduced_normal`` block of an extension solve report."""

    method: str
    iterations: int
    normal_actions: int
    converged: bool
    rhs_norm: Optional[float] = None
    residual_norm: Optional[float] = None
    quadratic_change: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReducedNormalReport":
        """Build the block from its JSON mapping."""

        return cls(
            method=str(data.get("method", "gauss_newton_schur")),
            iterations=int(data["iterations"]),
            normal_actions=int(data.get("normal_actions", 0)),
            converged=bool(data["converged"]),
            rhs_norm=_optional_float(data.get("rhs_norm")),
            residual_norm=_optional_float(data.get("residual_norm")),
            quadratic_change=_optional_float(data.get("quadratic_change")),
            raw=dict(data),
        )


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _optional_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> Optional[bool]:
    return None if value is None else bool(value)


@dataclass
class ExtensionSolveReport:
    """Reader for ``fs-extension-solve-1`` inner-solve reports.

    The CG (``method == "cg"``) and robust IRLS (``method == "irls_gn_armijo"``)
    solvers write different diagnostic fields; absent fields are ``None``.
    """

    baseline: str
    fingerprint: str
    damping: float
    method: str
    iterations: int
    converged: bool
    normal_actions: Optional[int] = None
    lag_penalty: Optional[float] = None
    lag_scale_seconds: Optional[float] = None
    offset_penalty: Optional[float] = None
    offset_scale_meters: Optional[float] = None
    field_scales: Tuple[float, ...] = ()
    background_batches: Optional[int] = None
    resident_background_bytes: Optional[int] = None
    regularization: Optional[float] = None
    rhs_norm: Optional[float] = None
    residual_norm: Optional[float] = None
    quadratic_change: Optional[float] = None
    quadratic_objective: Optional[float] = None
    evaluations: Optional[int] = None
    line_search_failed: Optional[bool] = None
    initial_gradient_norm: Optional[float] = None
    gradient_norm: Optional[float] = None
    objective: Optional[float] = None
    data_objective: Optional[float] = None
    reduced_objective: Optional[float] = None
    background_gradient_stationary: Optional[bool] = None
    reduced_normal: Optional[ReducedNormalReport] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExtensionSolveReport":
        """Build a report from its JSON mapping."""

        _require_schema(data, EXTENSION_SOLVE_SCHEMA)
        reduced = data.get("reduced_normal")
        if isinstance(reduced, str):
            reduced = json.loads(reduced)
        field_scales = data.get("field_scales", ())
        if np.ndim(field_scales) == 0:
            field_scales = (field_scales,)
        return cls(
            baseline=str(data["baseline"]),
            fingerprint=str(data["fingerprint"]),
            damping=float(data["damping"]),
            method=str(data.get("method", "cg")),
            iterations=int(data.get("iterations", 0)),
            converged=bool(data.get("converged", False)),
            normal_actions=_optional_int(data.get("normal_actions")),
            lag_penalty=_optional_float(data.get("lag_penalty")),
            lag_scale_seconds=_optional_float(data.get("lag_scale_seconds")),
            offset_penalty=_optional_float(data.get("offset_penalty")),
            offset_scale_meters=_optional_float(data.get("offset_scale_meters")),
            field_scales=tuple(float(v) for v in field_scales),
            background_batches=_optional_int(data.get("background_batches")),
            resident_background_bytes=_optional_int(
                data.get("resident_background_bytes")
            ),
            regularization=_optional_float(data.get("regularization")),
            rhs_norm=_optional_float(data.get("rhs_norm")),
            residual_norm=_optional_float(data.get("residual_norm")),
            quadratic_change=_optional_float(data.get("quadratic_change")),
            quadratic_objective=_optional_float(data.get("quadratic_objective")),
            evaluations=_optional_int(data.get("evaluations")),
            line_search_failed=_optional_bool(data.get("line_search_failed")),
            initial_gradient_norm=_optional_float(data.get("initial_gradient_norm")),
            gradient_norm=_optional_float(data.get("gradient_norm")),
            objective=_optional_float(data.get("objective")),
            data_objective=_optional_float(data.get("data_objective")),
            reduced_objective=_optional_float(data.get("reduced_objective")),
            background_gradient_stationary=_optional_bool(
                data.get("background_gradient_stationary")
            ),
            reduced_normal=(
                None if reduced is None else ReducedNormalReport.from_dict(reduced)
            ),
            raw=dict(data),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExtensionSolveReport":
        """Read a report JSON file."""

        return cls.from_dict(_load_json(path))


# ---------------------------------------------------------------------------
# Cartesian image sets
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ImageSet:
    """Reader for Cartesian image files written by imaging workflows.

    Args:
        path: Directory containing aggregate and per-task image HDF5 files.
        parts: Number of per-task image parts.
        shape: Expected xarray-style image shape.
        artifact_files: Optional explicit ``{None: aggregate, task: part}``
            file mapping resolved from the artifact catalog.
        frequencies: Optional per-part frequencies.
    """

    path: Path
    parts: int
    shape: Tuple[int, ...] = ()
    artifact_files: Optional[Dict[Optional[int], Path]] = None
    frequencies: Optional[Tuple[Any, ...]] = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.shape = tuple(int(n) for n in self.shape)
        if self.artifact_files is None and not self.path.exists():
            raise FileNotFoundError(f"Image path {self.path} does not exist")

    @property
    def f_list(self) -> np.ndarray:
        """Return one frequency per image part."""

        if self.frequencies is not None:
            return np.asarray(self.frequencies)
        f_list = np.zeros(self.parts)
        for i in range(self.parts):
            with h5py.File(self.image_file(i + 1), "r") as h5:
                f_list[i] = h5["frequency"][()]
        return f_list

    def image_file(self, part: Optional[int] = None) -> Path:
        """Return the aggregate (``part=None``) or per-task image file path."""

        if self.artifact_files is not None:
            try:
                return Path(self.artifact_files[part])
            except KeyError as exc:
                raise FileNotFoundError(f"No committed image for task {part}") from exc
        if part is None:
            return self.path / "image.h5"
        return self.path / f"image_{int(part)}.h5"

    def require_aggregate(self) -> Path:
        """Return the aggregate image file or explain why it is missing."""

        file = self.image_file()
        if file.exists():
            return file
        existing = [
            self.image_file(i + 1).name
            for i in range(self.parts)
            if self.image_file(i + 1).exists()
        ]
        detail = (
            f" Found per-frequency image shard(s): {', '.join(existing)}."
            if existing
            else " No per-frequency image shards were found either."
        )
        raise FileNotFoundError(
            f"Aggregate image file {file} is missing. Sauce writes image.h5 "
            "during the imaging --smooth postprocess after image_N.h5 shards "
            f"are produced.{detail}"
        )

    @property
    def raw(self):
        """Return the ``image/raw`` group as an ``xarray.Dataset``."""

        return self.read_images("raw")

    @property
    def smoothed(self):
        """Return the ``image/smoothed`` group as an ``xarray.Dataset``."""

        return self.read_images("smoothed")

    @property
    def incremental(self):
        """Return the ``/incremental`` root group written by Born workflows."""

        return self.read_images("", root="/incremental")

    def read_images(
        self,
        group: str,
        *,
        root: str = "/image",
        part: Optional[int] = None,
    ):
        """Read one HDF5 image group into an ``xarray.Dataset``.

        Args:
            group: Group name below ``root`` (``"raw"``, ``"smoothed"``); an
                empty string reads ``root`` itself.
            root: HDF5 root group holding the image groups.
            part: Optional per-task part instead of the aggregate file.
        """

        import xarray as xr

        images = xr.Dataset()
        file = self.require_aggregate() if part is None else self.image_file(part)
        location = f"{root.rstrip('/')}/{group}" if group else root
        with h5py.File(file, "r") as h5:
            if location.strip("/") not in h5 and location not in h5:
                raise KeyError(f"{file} has no image group {location!r}")
            h5group = h5[location]
            if "properties" in h5group:
                properties = [str(_decode(p)) for p in h5group["properties"][()]]
            else:
                properties = [
                    name
                    for name in h5group
                    if isinstance(h5group[name], h5py.Dataset)
                    and "n_grid" in h5group[name].attrs
                ]
            for prop in properties:
                h5data = h5group[prop]
                attrs = h5data.attrs
                x0 = np.asarray(attrs["x0"])[::-1]
                x1 = np.asarray(attrs["x1"])[::-1]
                n = np.asarray(attrs["n_grid"], dtype=int)[::-1]
                dims = [str(_decode(dim)) for dim in attrs["dims"][::-1]]
                axis_units = self._axis_units(attrs, dims)
                coords = {
                    dim: np.linspace(x0[i], x1[i], n[i]) for i, dim in enumerate(dims)
                }
                array = xr.DataArray(
                    data=h5data[:].reshape(n), dims=dims, coords=coords
                )
                for dim, units in zip(dims, axis_units):
                    if units:
                        array.coords[dim].attrs["units"] = units
                units = _decode_attr(attrs.get("units"))
                if units:
                    array.attrs["units"] = units
                for attr_name in ("coordinate_system", "value_scale", "value_storage"):
                    value = _decode_attr(attrs.get(attr_name))
                    if value is not None:
                        array.attrs[attr_name] = value
                images[prop] = array
        return images

    @staticmethod
    def _axis_units(
        attrs: Mapping[str, Any], dims: Sequence[str]
    ) -> List[Optional[str]]:
        units = _decode_attr(attrs.get("axis_units"))
        if units is None:
            return [None] * len(dims)
        if isinstance(units, str):
            return [units] * len(dims)
        return list(units)[::-1]


def _decode_attr(value: Any) -> Any:
    if value is None:
        return None
    decoded = _decode(value)
    if isinstance(decoded, list):
        return decoded[0] if len(decoded) == 1 else decoded
    return decoded


# ---------------------------------------------------------------------------
# Smoothing configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothingConfig:
    """Representation-independent variational smoothing configuration.

    ``to_control_fs`` emits the ``control_sensitivities.Smoothing`` contract
    (Riesz map on control coefficients); ``to_image_fs`` emits the
    ``Imaging.Smoothing`` contract applied while stacking Cartesian shards.
    """

    kind: str = "tikhonov"
    wavelength_fraction: float = 1.0
    alpha: Optional[float] = None
    alpha1: Optional[float] = None
    alpha2: Optional[float] = None
    tgv_ratio: float = 1.0
    reference_wavelength: Optional[float] = None
    derivative_order: int = 1
    epsilon: float = 1.0e-3
    iterations: int = 5
    input_role: str = "dual"
    normalize_amplitude: Optional[bool] = None
    illumination_normalization: str = "none"

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind == "l2":
            kind = "tikhonov"
        if kind not in {"none", "tikhonov", "tv", "tgv"}:
            raise ValueError(f"unsupported smoothing kind {self.kind!r}")
        wavelength_fraction = float(self.wavelength_fraction)
        alpha = None if self.alpha is None else float(self.alpha)
        alpha1 = None if self.alpha1 is None else float(self.alpha1)
        alpha2 = None if self.alpha2 is None else float(self.alpha2)
        tgv_ratio = float(self.tgv_ratio)
        reference_wavelength = (
            None
            if self.reference_wavelength is None
            else float(self.reference_wavelength)
        )
        epsilon = float(self.epsilon)
        derivative_order = int(self.derivative_order)
        iterations = int(self.iterations)
        input_role = str(self.input_role).strip().lower()
        normalize_amplitude = self.normalize_amplitude
        illumination = str(self.illumination_normalization).strip().lower()
        if not np.isfinite(wavelength_fraction) or wavelength_fraction < 0.0:
            raise ValueError(
                "smoothing wavelength fraction must be finite and nonnegative"
            )
        if alpha is not None and (not np.isfinite(alpha) or alpha < 0.0):
            raise ValueError("smoothing alpha must be finite and nonnegative")
        for name, value in (("alpha1", alpha1), ("alpha2", alpha2)):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"smoothing {name} must be finite and nonnegative")
        if (alpha1 is None) != (alpha2 is None):
            raise ValueError("TGV alpha1 and alpha2 must be supplied together")
        if alpha1 is not None and (alpha1 == 0.0) != (alpha2 == 0.0):
            raise ValueError("TGV alpha1 and alpha2 must both be positive or both zero")
        if not np.isfinite(tgv_ratio) or tgv_ratio <= 0.0:
            raise ValueError("TGV ratio must be finite and positive")
        if kind == "tgv" and alpha is not None:
            raise ValueError("TGV smoothing uses alpha1 and alpha2 rather than alpha")
        if kind != "tgv" and alpha1 is not None:
            raise ValueError("alpha1 and alpha2 are only valid for TGV smoothing")
        if reference_wavelength is not None and (
            not np.isfinite(reference_wavelength) or reference_wavelength <= 0.0
        ):
            raise ValueError(
                "smoothing reference wavelength must be finite and positive"
            )
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("smoothing epsilon must be finite and positive")
        if derivative_order not in {1, 2}:
            raise ValueError("smoothing derivative order must be one or two")
        if iterations < 1:
            raise ValueError("smoothing iterations must be positive")
        if input_role not in {"dual", "primal"}:
            raise ValueError("smoothing input role must be 'dual' or 'primal'")
        if normalize_amplitude is not None and not isinstance(
            normalize_amplitude, (bool, np.bool_)
        ):
            raise ValueError("smoothing amplitude normalization must be boolean")
        illumination = {"linear": "source", "nonlinear": "cross"}.get(
            illumination, illumination
        )
        if illumination not in {"none", "source", "cross"}:
            raise ValueError(
                "illumination normalization must be 'none', 'source' (linear), "
                "or 'cross' (nonlinear)"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "wavelength_fraction", wavelength_fraction)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "alpha1", alpha1)
        object.__setattr__(self, "alpha2", alpha2)
        object.__setattr__(self, "tgv_ratio", tgv_ratio)
        object.__setattr__(self, "reference_wavelength", reference_wavelength)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "derivative_order", derivative_order)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "input_role", input_role)
        object.__setattr__(
            self,
            "normalize_amplitude",
            None if normalize_amplitude is None else bool(normalize_amplitude),
        )
        object.__setattr__(self, "illumination_normalization", illumination)

    def amplitude_normalization_enabled(self) -> bool:
        """Return whether nonlinear smoothing uses a dimensionless input block."""

        if self.normalize_amplitude is not None:
            return self.normalize_amplitude
        if self.kind == "tgv":
            return self.alpha1 is None
        return self.alpha is None

    def resolved_alpha(self) -> float:
        """Return the direct variational coefficient for local application."""

        if self.alpha is not None:
            return self.alpha
        if self.reference_wavelength is None:
            raise ValueError(
                "local control smoothing requires alpha or reference_wavelength; "
                "Sauce --smooth can derive wavelength from the material model"
            )
        length = self.wavelength_fraction * self.reference_wavelength / (2.0 * np.pi)
        return length ** (2 * self.derivative_order)

    def resolved_tgv_weights(self) -> Tuple[float, float]:
        """Return the first- and second-order TGV weights."""

        if self.alpha1 is not None and self.alpha2 is not None:
            return self.alpha1, self.alpha2
        if self.reference_wavelength is None:
            raise ValueError(
                "local TGV smoothing requires alpha1/alpha2 or reference_wavelength; "
                "Sauce --smooth can derive wavelength from the material model"
            )
        length = self.wavelength_fraction * self.reference_wavelength / (2.0 * np.pi)
        return length, self.tgv_ratio * length * length

    def _common_fs(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": self.kind,
            "lambda": self.wavelength_fraction,
            "derivative_order": self.derivative_order,
            "epsilon": self.epsilon,
            "iterations": self.iterations,
        }
        if self.alpha is not None:
            payload["alpha"] = self.alpha
        if self.alpha1 is not None and self.alpha2 is not None:
            payload["alpha1"] = self.alpha1
            payload["alpha2"] = self.alpha2
        if self.kind == "tgv" and self.tgv_ratio != 1.0:
            payload["tgv_ratio"] = self.tgv_ratio
        return payload

    def to_control_fs(self) -> Dict[str, Any]:
        """Serialize the ``control_sensitivities.Smoothing`` Riesz-map contract."""

        payload = self._common_fs()
        payload["input_role"] = self.input_role
        if self.reference_wavelength is not None:
            payload["reference_wavelength"] = self.reference_wavelength
        return payload

    def to_image_fs(self) -> Dict[str, Any]:
        """Serialize the ``Imaging.Smoothing`` Cartesian stacking contract.

        Cartesian image smoothing uses mixed first-order FEM fields, so a
        second derivative order is rejected. A reference wavelength is resolved
        into explicit coefficients because the image smoother has no
        ``reference_wavelength`` field.
        """

        if self.derivative_order != 1:
            raise ValueError(
                "Cartesian image smoothing uses mixed first-order FEM fields"
            )
        payload = self._common_fs()
        if self.reference_wavelength is not None:
            if self.kind == "tgv":
                payload["alpha1"], payload["alpha2"] = self.resolved_tgv_weights()
            elif self.kind in {"tikhonov", "tv"}:
                payload["alpha"] = self.resolved_alpha()
        if self.normalize_amplitude is not None:
            payload["normalize_amplitude"] = self.normalize_amplitude
        payload["illumination_normalization"] = self.illumination_normalization
        return payload

    @classmethod
    def from_value(
        cls, value: Optional[Union["SmoothingConfig", Mapping[str, Any]]]
    ) -> Optional["SmoothingConfig"]:
        """Normalize an optional authored smoothing value or Sauce payload."""

        if value is None or isinstance(value, SmoothingConfig):
            return value
        if hasattr(value, "to_control_fs") and not isinstance(value, Mapping):
            value = value.to_control_fs()  # type: ignore[union-attr]
        payload = dict(value)
        if "type" in payload and "kind" not in payload:
            payload["kind"] = payload.pop("type")
        if "lambda" in payload and "wavelength_fraction" not in payload:
            payload["wavelength_fraction"] = payload.pop("lambda")
        if "strength" in payload:
            payload.setdefault("alpha", payload.pop("strength"))
        if (
            "normalize_illumination" in payload
            and "illumination_normalization" not in payload
        ):
            payload["illumination_normalization"] = (
                "cross" if payload.pop("normalize_illumination") else "none"
            )
        payload.pop("normalize_coordinate", None)
        return cls(**payload)
