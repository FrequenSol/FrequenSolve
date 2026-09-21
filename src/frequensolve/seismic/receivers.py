"""Receiver definitions and coordinate systems.

This module defines the various types of receivers and their locations.
"""

import copy
import json
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import blake3
import h5py
import numpy as np
import xarray as xr

from frequensolve.geometry.frame import CoordinateValue, Direction, direction_to_fs
from frequensolve.geometry.grids import CartesianGrid, Grid
from frequensolve.seismic.sparse_survey import ReceiverSampling
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.units import is_quantity, unit_expression, value_and_units_to_fs
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.fields import canonical_field
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    TypeTaggedMixin,
    merge_extra,
)
from frequensolve.util.store import SimulationStore

__all__ = [
    "CoordsArray",
    "CoordsFromFile",
    "CoordsGrid",
    "CoordsSurfaceCarpet",
    "ReceiverComponent",
    "ReceiverGroup",
    "ReceiverCoords",
    "ReceiverDevice",
    "ReceiverArray",
    "EncodedReceiver",
    "ReceiverWeightTable",
    "ReceiverNodeArray",
    "ReceiverNode",
    "ReceiverFiber",
    "ReceiverSampling",
]


def _is_remote_file_reference(value: Any) -> bool:
    text = str(value)
    return text.startswith("remote:") or "://" in text


def _complex_to_fs(value: Any, *, label: str) -> Union[float, List[float]]:
    """Serialize one finite complex scalar using the solver JSON convention."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, complex):
        scalar = value
    elif isinstance(value, Number) and not isinstance(value, (bool, np.bool_)):
        scalar = complex(np.asarray(value).item())
    elif (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(
            isinstance(item, Number) and not isinstance(item, (bool, np.bool_))
            for item in value
        )
    ):
        scalar = complex(value[0], value[1])
    else:
        raise TypeError(
            f"{label} must be a real scalar, complex scalar, or [real, imag] pair"
        )
    if not np.isfinite(scalar.real) or not np.isfinite(scalar.imag):
        raise ValueError(f"{label} must be finite")
    real = float(scalar.real)
    imag = float(scalar.imag)
    return real if imag == 0.0 else [real, imag]


def _complex_from_value(value: Any, *, label: str) -> complex:
    """Return one validated complex scalar from an authored value."""

    serialized = _complex_to_fs(value, label=label)
    if isinstance(serialized, list):
        return complex(serialized[0], serialized[1])
    return complex(serialized)


def _receiver_weight_array(values: Any, *, ndim: int, label: str) -> np.ndarray:
    """Normalize receiver weights with vectorized NumPy validation.

    Real/complex arrays use their natural shape. A numeric final axis of length
    two is also accepted as split real/imaginary storage when it is one rank
    higher than the requested logical array.
    """

    if isinstance(values, xr.DataArray):
        values = values.data
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a numeric array")
    try:
        authored = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a numeric array") from exc
    if authored.dtype.kind not in {"i", "u", "f", "c"}:
        raise TypeError(f"{label} must be a numeric array")

    if authored.ndim == ndim + 1 and authored.shape[-1] == 2:
        real = np.asarray(authored[..., 0], dtype=np.float32)
        imag = np.asarray(authored[..., 1], dtype=np.float32)
        result = real.astype(np.complex64)
        result.imag = imag
    elif authored.ndim == ndim:
        result = np.asarray(authored, dtype=np.complex64)
    else:
        shape = "vector" if ndim == 1 else "matrix"
        raise ValueError(f"{label} must be a {shape}")

    if result.size == 0:
        raise ValueError(f"{label} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def _conjugated_complex_value(value: Any, *, label: str) -> complex:
    """Return the complex conjugate of one validated authored value."""

    return _complex_from_value(value, label=label).conjugate()


@dataclass(kw_only=True)
class ReceiverComponent:
    """Defines a single component/measurement type for a receiver.

    A receiver component specifies what physical quantity is being measured
    (e.g., pressure, velocity) and in what direction for vector quantities.

    Args:
        name: String identifier for this receiver component.
        field: Physical field being measured.
        direction: Optional measurement direction for vector fields.
        units: Optional output units for this component.
        weight: Optional constant complex weight applied to the component.
    """

    name: str = "name"
    field: str
    direction: Optional[Union[List[float], Direction]] = None
    units: Optional[str] = None
    weight: Optional[Any] = None

    def __post_init__(self) -> None:
        self.field = canonical_field(self.field)
        if self.weight is not None:
            _complex_to_fs(self.weight, label="receiver component weight")

    def to_fs(self, ctx: Optional[ExportContext] = None) -> dict:
        """Serialize scalar component metadata for solver input.

        Pointwise values belong to the enclosing :class:`EncodedReceiver` bulk
        table and are intentionally not embedded in component metadata.
        """

        return {
            "name": self.name,
            "field": canonical_field(self.field),
            **(
                {"direction": direction_to_fs(self.direction)}
                if self.direction is not None
                else {}
            ),
            **({"units": self.units} if self.units is not None else {}),
            **(
                {
                    "weight": _complex_to_fs(
                        self.weight,
                        label="receiver component weight",
                    )
                }
                if self.weight is not None
                else {}
            ),
        }

    def conjugated(self) -> "ReceiverComponent":
        """Return a copy with its constant weight conjugated."""

        component = copy.deepcopy(self)
        if component.weight is not None:
            component.weight = _conjugated_complex_value(
                component.weight,
                label="receiver component weight",
            )
        return component

    time_reversed = conjugated

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverComponent":
        """Deserialize a receiver component payload."""

        data = copy.deepcopy(data)
        if "direction" in data:
            data["direction"] = Direction.from_fs(data["direction"])
        return cls(
            name=data["name"],
            field=canonical_field(data["field"]),
            direction=data.get("direction"),
            units=data.get("units"),
            weight=data.get("weight"),
        )


# ----------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------
_RECEIVER_FIBER_DEGREE_UNITS = {"deg", "degree", "degrees"}
_RECEIVER_FIBER_RADIAN_UNITS = {"rad", "radian", "radians"}


@dataclass(kw_only=True)
class ReceiverWeightTable:
    """External dense complex weights for an :class:`EncodedReceiver`.

    The HDF5 dataset uses h5py shape
    ``(encoding * component, receiver, 2)``. The final axis stores real and
    imaginary parts; rows are encoding-major and component-minor.
    """

    file: Union[str, Path]
    dataset: str
    format: Literal["HDF5", "hdf5"] = "HDF5"
    hash: Optional[str] = None
    names_dataset: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("ReceiverWeightTable dataset must be non-empty")
        if self.names_dataset is not None and not self.names_dataset:
            raise ValueError("ReceiverWeightTable names_dataset must be non-empty")
        if self.format.lower() != "hdf5":
            raise ValueError("ReceiverWeightTable supports only HDF5")

    def to_fs(self, ctx: Optional[ExportContext] = None) -> dict:
        """Serialize the small HDF5 reference without loading its values."""

        file = Path(self.file)
        if ctx is not None:
            file = ctx.relative_to_project(file)
        return {
            "_type": "HDF5Dense",
            "file": str(file),
            "dataset": self.dataset,
            "format": "HDF5",
            **({"hash": self.hash} if self.hash is not None else {}),
            **(
                {"names_dataset": self.names_dataset}
                if self.names_dataset is not None
                else {}
            ),
        }

    def validate_shape(
        self,
        row_count: int,
        receiver_count: int,
        ctx: Optional[ExportContext] = None,
    ) -> None:
        """Validate a local external table without loading its values."""

        if _is_remote_file_reference(self.file):
            return
        file = Path(self.file).expanduser()
        project_path = getattr(ctx, "project_path", None)
        if not file.is_absolute() and project_path is not None:
            file = Path(project_path) / file
        if not file.exists():
            return
        with h5py.File(file, "r") as h5:
            if self.dataset not in h5:
                raise ValueError(
                    f"EncodedReceiver weight dataset {self.dataset!r} is missing "
                    f"from {file}"
                )
            shape = h5[self.dataset].shape
        expected = (row_count, receiver_count, 2)
        if shape != expected:
            raise ValueError(
                f"EncodedReceiver weight table has shape {shape}; expected {expected}"
            )

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ReceiverWeightTable":
        """Deserialize an HDF5 receiver-weight reference."""

        return cls(
            file=data["file"],
            dataset=data["dataset"],
            format=data.get("format", "HDF5"),
            hash=data.get("hash"),
            names_dataset=data.get("names_dataset"),
        )


def _receiver_fiber_angle_degrees(angle: Any) -> float:
    """Validate a fiber winding angle and return its value in degrees."""

    multiplier = 1.0
    if is_quantity(angle):
        try:
            value = angle.to("degree").magnitude
        except Exception as exc:
            raise ValueError(
                "ReceiverFiber angle must be an angular quantity."
            ) from exc
    elif isinstance(angle, Mapping):
        if "value" not in angle:
            raise ValueError("ReceiverFiber angle quantity requires a value.")
        value = angle["value"]
        units = unit_expression(angle.get("units", "deg")).strip()
        if units in _RECEIVER_FIBER_RADIAN_UNITS:
            multiplier = 180.0 / np.pi
        elif units not in _RECEIVER_FIBER_DEGREE_UNITS:
            raise ValueError("ReceiverFiber angle units must be degrees or radians.")
    else:
        value = angle

    if isinstance(value, (str, bytes, bool)):
        raise ValueError("ReceiverFiber angle must be a numeric scalar.")
    try:
        scalar = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ReceiverFiber angle must be a numeric scalar.") from exc
    if scalar.ndim != 0:
        raise ValueError("ReceiverFiber angle must be a numeric scalar.")

    degrees = float(scalar) * multiplier
    if not np.isfinite(degrees) or not 0.0 < degrees < 90.0:
        raise ValueError(
            "ReceiverFiber angle must be strictly between 0 and 90 degrees."
        )
    return degrees


def _receiver_fiber_angle_to_fs(angle: Any) -> Any:
    """Serialize a validated fiber winding angle for the solver."""

    degrees = _receiver_fiber_angle_degrees(angle)
    if not is_quantity(angle):
        return value_and_units_to_fs(angle)

    payload = value_and_units_to_fs(angle)
    units = str(payload["units"]).strip().lower()
    if units in _RECEIVER_FIBER_DEGREE_UNITS | _RECEIVER_FIBER_RADIAN_UNITS:
        return payload
    return {"value": degrees, "units": "deg"}


@register_class
@dataclass(kw_only=True)
class ReceiverDevice(TypeTaggedMixin, ABC):
    """Abstract base class for a receiver device with measured components.

    Args:
        name: Optional identifier for this receiver device.
        components: Components defining measured quantities.
        response: Optional receiver response wavelet.
    """

    name: Optional[str] = None
    components: List[ReceiverComponent] = field(default_factory=list)
    # TODO(receiver-response): replace Wavelet with a receiver-response contract
    # that can evaluate a complex transfer function at job frequencies. Sauce
    # must multiply synthetic channels by that response before trace output.
    response: Optional[Wavelet] = None

    def add_component(
        self,
        name: str,
        field: str,
        direction: Optional[List[float]] = None,
        *,
        units: Optional[str] = None,
        weight: Optional[Any] = None,
    ) -> "ReceiverComponent":
        """Add a measured component to this device.

        Args:
            name: Component name used in trace output.
            field: Physical field to measure.
            direction: Optional measurement direction for vector fields.
            units: Optional output units.
            weight: Optional constant complex component weight.

        Returns:
            Newly added ``ReceiverComponent``.
        """

        component = ReceiverComponent(
            name=name,
            field=canonical_field(field),
            direction=direction,
            units=units,
            weight=weight,
        )
        self.components.append(component)
        return component

    def to_fs(self, ctx: Optional[ExportContext] = None) -> dict:
        """Serialize this receiver device for solver input."""

        if self.response is not None:
            raise NotImplementedError(
                "Receiver spectral response is reserved but not implemented. "
                "A solver contract for complex transfer functions must be added "
                "before response can be exported."
            )

        return {
            **({"name": self.name} if self.name is not None else {}),
            "components": [c.to_fs(ctx) for c in self.components],
        }

    def output_components(self) -> Iterator[ReceiverComponent]:
        """Iterate the component names exposed in receiver trace output."""

        return iter(self.components)

    def output_receiver_count(self, point_count: int) -> int:
        """Return the number of receiver rows emitted for ``point_count`` points."""

        return int(point_count)

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverDevice":
        """Deserialize a registered receiver-device payload."""

        return cls.dispatch_from_fs(data, class_registry)


@register_class
@dataclass(kw_only=True)
class ReceiverArray(ReceiverDevice):
    """A physical array of receiver nodes around every group coordinate.

    ``offsets`` are short model-coordinate displacement vectors. Sauce expands
    each authored receiver coordinate by all offsets and then reduces the
    resulting nodes back to one logical receiver by default.

    Args:
        offsets: Numeric matrix with shape ``(array_node, coordinate)``.
        offset_units: Optional scalar or per-coordinate length units.
        reduction: ``"mean"`` or ``"sum"`` combines the nodes around each
            coordinate; ``"none"`` retains every expanded node.
    """

    offsets: Any
    offset_units: Optional[Any] = None
    reduction: Literal["none", "sum", "mean"] = "mean"

    def __post_init__(self) -> None:
        system = None
        values = self.offsets
        if isinstance(values, CoordinateValue):
            system = values.system
            if self.offset_units is None:
                self.offset_units = values.units
            values = values.value
        if system not in {None, "", "global"}:
            raise ValueError("ReceiverArray offsets use model-coordinate directions")

        quantity_units = _first_quantity_units(values)
        if quantity_units is not None:
            if isinstance(self.offset_units, (list, tuple)):
                raise ValueError(
                    "ReceiverArray quantity offsets require one common offset unit"
                )
            units = (
                self.offset_units if self.offset_units is not None else quantity_units
            )
            values = _strip_coordinate_quantities(values, units)
            self.offset_units = units
        try:
            offsets = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("ReceiverArray offsets must be a numeric matrix") from exc
        if offsets.ndim != 2 or offsets.shape[0] == 0 or offsets.shape[1] not in {2, 3}:
            raise ValueError(
                "ReceiverArray offsets must have shape (array_node, 2 or 3)"
            )
        if not np.all(np.isfinite(offsets)):
            raise ValueError("ReceiverArray offsets must be finite")
        if self.reduction not in {"none", "sum", "mean"}:
            raise ValueError("ReceiverArray reduction must be 'none', 'sum', or 'mean'")
        self.offsets = offsets

    @property
    def node_count(self) -> int:
        """Return the number of physical receiver nodes around each coordinate."""

        return int(self.offsets.shape[0])

    def output_receiver_count(self, point_count: int) -> int:
        """Return logical rows after the per-anchor array reduction."""

        count = int(point_count)
        return count * self.node_count if self.reduction == "none" else count

    def to_fs(self, ctx: Optional[ExportContext] = None) -> dict:
        """Serialize the compact physical receiver-array definition."""

        offset_units = self.offset_units
        if isinstance(offset_units, (list, tuple)):
            offset_units = [unit_expression(units) for units in offset_units]
        elif offset_units is not None:
            offset_units = unit_expression(offset_units)
        return {
            "_type": "ReceiverArray",
            **ReceiverDevice.to_fs(self, ctx),
            "offsets": self.offsets.tolist(),
            **({"offset_units": offset_units} if offset_units is not None else {}),
            "reduction": self.reduction,
        }

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverDevice":
        """Deserialize physical arrays and migrate interim weighted payloads."""

        if "weights" in data and "offsets" not in data:
            warnings.warn(
                "Weighted ReceiverArray payloads are deprecated; use EncodedReceiver.",
                DeprecationWarning,
                stacklevel=2,
            )
            migrated = dict(data)
            migrated["_type"] = "EncodedReceiver"
            migrated.setdefault("encoding_count", 1)
            return EncodedReceiver.from_fs(migrated)
        if "offsets" not in data:
            warnings.warn(
                "ReceiverArray payloads without offsets are deprecated; "
                "loading this payload as ReceiverNode.",
                DeprecationWarning,
                stacklevel=2,
            )
            return ReceiverNode.from_fs(data)
        return cls(
            name=data.get("name"),
            components=[ReceiverComponent.from_fs(c) for c in data["components"]],
            response=data.get("response"),
            offsets=data["offsets"],
            offset_units=data.get("offset_units"),
            reduction=data.get("reduction", "mean"),
        )


@register_class
@dataclass(kw_only=True)
class EncodedReceiver(ReceiverDevice):
    """Complex receiver encodings evaluated on one fixed coordinate geometry.

    In-memory weights use shape ``(encoding, component, receiver)``. For a
    single component, the convenient ``(encoding, receiver)`` form is also
    accepted. Each encoding weights every base component independently, so a
    multi-component device produces one output channel per encoding/component
    pair without duplicating component metadata.

    Args:
        weights: Optional complex weight tensor. Split real/imaginary storage
            is accepted with an additional final axis of length two. Ambiguous
            real arrays shaped ``(encoding, 1, 2)`` use the canonical three-axis
            tensor form (two receivers); use ``(encoding, 1, 1, 2)`` for split
            weights over one receiver.
        encoding_names: Optional output encoding names. Generated names are
            omitted from JSON; large explicit name lists are stored in HDF5.
        encoding_count: Required for external tables when names are omitted.
        reduction: Reduction across the fixed receiver geometry.
        weight_table: Optional external HDF5 table.  Weights always define
            the forward receiver operator; adjoint modeling applies its
            Hermitian transpose automatically.
    """

    weights: Optional[Any] = field(default=None, repr=False)
    encoding_names: Optional[Sequence[str]] = None
    encoding_count: Optional[int] = None
    reduction: Literal["none", "sum", "mean"] = "sum"
    weight_table: Optional[ReceiverWeightTable] = None
    _weight_blocks: List[Tuple[np.ndarray, bool]] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.reduction not in {"none", "sum", "mean"}:
            raise ValueError(
                "EncodedReceiver reduction must be 'none', 'sum', or 'mean'"
            )
        if isinstance(self.weight_table, Mapping):
            self.weight_table = ReceiverWeightTable.from_fs(self.weight_table)
        if self.weights is not None and self.weight_table is not None:
            raise ValueError(
                "EncodedReceiver accepts authored weights or an external table, not both"
            )
        names = self.encoding_names
        if names is not None:
            if isinstance(names, (str, bytes)):
                raise TypeError("EncodedReceiver encoding_names must be a sequence")
            names = [str(name) for name in names]
            if any(not name for name in names):
                raise ValueError("EncodedReceiver encoding names must be non-empty")
            if len(names) != len(set(names)):
                raise ValueError("EncodedReceiver encoding names must be unique")
            self.encoding_names = names

        explicit_count = self.encoding_count
        if explicit_count is None and names is not None:
            explicit_count = len(names)
        if explicit_count is not None:
            if (
                isinstance(explicit_count, (bool, np.bool_))
                or int(explicit_count) != explicit_count
            ):
                raise TypeError("EncodedReceiver encoding_count must be an integer")
            explicit_count = int(explicit_count)
            if explicit_count < 1:
                raise ValueError("EncodedReceiver encoding_count must be positive")
        if (
            self.weights is None
            and self.weight_table is None
            and (names is not None or explicit_count is not None)
        ):
            raise ValueError(
                "EncodedReceiver encoding metadata requires weights or a weight table"
            )

        if self.weights is not None:
            self.weights = self._normalize_tensor(
                self.weights,
                encoding_count=explicit_count,
                label="EncodedReceiver weights",
            )
            inferred_count = int(self.weights.shape[0])
            if explicit_count is not None and explicit_count != inferred_count:
                raise ValueError(
                    "EncodedReceiver encoding_count does not match its weight tensor"
                )
            self.encoding_count = inferred_count
        elif explicit_count is not None:
            self.encoding_count = explicit_count
        elif names is not None:
            self.encoding_count = len(names)
        else:
            self.encoding_count = 0 if self.weight_table is None else 1

        if names is not None and len(names) != self.encoding_count:
            raise ValueError("EncodedReceiver encoding_names must match encoding_count")

    def _normalize_tensor(
        self,
        values: Any,
        *,
        encoding_count: Optional[int],
        label: str,
    ) -> np.ndarray:
        """Normalize one bulk encoding tensor without scalar Python objects."""

        if isinstance(values, xr.DataArray):
            values = values.data
        try:
            authored = np.asarray(values)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label} must be a numeric array") from exc
        if authored.dtype.kind not in {"i", "u", "f", "c"}:
            raise TypeError(f"{label} must be a numeric array")

        component_count = len(self.components)
        if authored.ndim == 4:
            tensor = _receiver_weight_array(values, ndim=3, label=label)
        elif authored.ndim == 3:
            is_split_matrix = (
                authored.dtype.kind != "c"
                and authored.shape[-1] == 2
                and (
                    (component_count == 1 and authored.shape[1] != 1)
                    or (encoding_count == 1 and authored.shape[0] == component_count)
                )
            )
            if is_split_matrix:
                matrix = _receiver_weight_array(values, ndim=2, label=label)
                tensor = (
                    matrix[:, np.newaxis, :]
                    if component_count == 1
                    else matrix[np.newaxis, :, :]
                )
            else:
                tensor = _receiver_weight_array(values, ndim=3, label=label)
        elif authored.ndim == 2:
            matrix = _receiver_weight_array(values, ndim=2, label=label)
            if component_count == 1:
                tensor = matrix[:, np.newaxis, :]
            elif encoding_count == 1 and matrix.shape[0] == component_count:
                tensor = matrix[np.newaxis, :, :]
            else:
                raise ValueError(
                    f"{label} must have shape (encoding, component, receiver)"
                )
        elif authored.ndim == 1 and component_count == 1:
            vector = _receiver_weight_array(values, ndim=1, label=label)
            tensor = vector[np.newaxis, np.newaxis, :]
        else:
            raise ValueError(f"{label} must have shape (encoding, component, receiver)")

        if tensor.shape[1] != component_count:
            raise ValueError(
                "EncodedReceiver weight component dimension must match components"
            )
        return tensor

    def _authored_weight_blocks(self) -> Iterator[Tuple[np.ndarray, bool]]:
        """Iterate encoding-major weight tensors and conjugation states."""

        if self.weights is not None:
            yield self.weights, False
        yield from self._weight_blocks

    @property
    def has_authored_weights(self) -> bool:
        """Return whether this device owns in-memory pointwise weights."""

        return self.weights is not None or bool(self._weight_blocks)

    def conjugated(self) -> "EncodedReceiver":
        """Return a conjugated view sharing authored weight arrays."""

        if self.weight_table is not None:
            raise ValueError("Conjugating a receiver requires authored weights")
        result = copy.copy(self)
        result.weights = None
        result._weight_blocks = [
            (block, not conjugate)
            for block, conjugate in self._authored_weight_blocks()
        ]
        if self.encoding_names is not None:
            result.encoding_names = list(self.encoding_names)
        return result

    time_reversed = conjugated

    def _load_encoding_names(self) -> None:
        """Resolve external channel identities after project paths are relocated."""

        table = self.weight_table
        if (
            self.encoding_names is not None
            or table is None
            or table.names_dataset is None
        ):
            return
        if _is_remote_file_reference(table.file):
            raise ValueError(
                "Download the receiver encoding names before listing fields"
            )
        with h5py.File(table.file, "r") as h5:
            dataset = h5[table.names_dataset]
            if dataset.shape != (self.encoding_count,):
                raise ValueError("Receiver encoding names must match encoding_count")
            names = dataset.asstr()[:].tolist()
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Receiver encoding names must be non-empty and unique")
        self.encoding_names = names

    def _encoding_name(self, index: int) -> str:
        if self.encoding_names is not None:
            return self.encoding_names[index]
        return f"encoded_receiver_{index + 1:06d}"

    def output_components(self) -> Iterator[ReceiverComponent]:
        """Iterate expanded encoding/component trace descriptors lazily."""

        self._load_encoding_names()
        if self.encoding_count == 1 and self.encoding_names is None:
            yield from self.components
            return
        for encoding_index in range(self.encoding_count or 0):
            encoding_name = self._encoding_name(encoding_index)
            for component in self.components:
                output = copy.copy(component)
                output.name = (
                    encoding_name
                    if len(self.components) == 1
                    else f"{encoding_name}:{component.name}"
                )
                yield output

    def output_receiver_count(self, point_count: int) -> int:
        """Return rows after reducing the shared receiver geometry."""

        return int(point_count) if self.reduction == "none" else 1

    def add_encoding(
        self,
        name: str,
        weights: Any,
        *,
        conjugate: bool = False,
    ) -> "EncodedReceiver":
        """Append one encoding over every base component and receiver point."""

        if self.weight_table is not None:
            raise ValueError(
                "Cannot add authored encodings to an external weight table"
            )
        if not isinstance(conjugate, (bool, np.bool_)):
            raise TypeError("EncodedReceiver conjugate must be boolean")
        if not name:
            raise ValueError("EncodedReceiver encoding names must be non-empty")

        component_count = len(self.components)
        authored = np.asarray(weights)
        if component_count == 1 and authored.ndim in {1, 2}:
            row = _receiver_weight_array(
                weights,
                ndim=1,
                label=f"EncodedReceiver weights for {name!r}",
            )
            tensor = row[np.newaxis, np.newaxis, :]
        else:
            matrix = _receiver_weight_array(
                weights,
                ndim=2,
                label=f"EncodedReceiver weights for {name!r}",
            )
            if matrix.shape[0] != component_count:
                raise ValueError(
                    "EncodedReceiver encoding rows must match the component count"
                )
            tensor = matrix[np.newaxis, :, :]

        if self.encoding_names is None:
            self.encoding_names = [
                self._encoding_name(index) for index in range(self.encoding_count or 0)
            ]
        if name in self.encoding_names:
            raise ValueError(f"EncodedReceiver encoding name {name!r} is duplicated")
        self.encoding_names.append(name)
        self._weight_blocks.append((tensor, bool(conjugate)))
        self.encoding_count = (self.encoding_count or 0) + 1
        return self

    def validate_size(
        self, point_count: int, ctx: Optional[ExportContext] = None
    ) -> None:
        """Validate the bulk tensor against its components and shared geometry."""

        encoding_count = 0
        for block, _ in self._authored_weight_blocks():
            if block.shape[2] != point_count:
                raise ValueError(
                    f"EncodedReceiver weights have {block.shape[2]} receivers; "
                    f"the receiver geometry has {point_count} points"
                )
            if block.shape[1] != len(self.components):
                raise ValueError(
                    "EncodedReceiver weight component dimension must match components"
                )
            encoding_count += block.shape[0]
        if self.has_authored_weights and encoding_count != self.encoding_count:
            raise ValueError(
                "EncodedReceiver weight encoding dimension must match encoding_count"
            )
        has_pointwise = self.has_authored_weights or self.weight_table is not None
        if not has_pointwise:
            raise ValueError("EncodedReceiver requires a pointwise weight table")
        if any(component.weight is not None for component in self.components):
            raise ValueError(
                "EncodedReceiver pointwise and component scalar weights cannot be mixed"
            )
        if not self.encoding_count:
            raise ValueError("EncodedReceiver requires at least one encoding")
        if self.weight_table is not None:
            self.weight_table.validate_shape(
                self.encoding_count * len(self.components),
                point_count,
                ctx,
            )

    def _materialize_weight_table(
        self,
        ctx: Optional[ExportContext],
        *,
        group_name: str,
        point_count: int,
    ) -> dict:
        """Stream authored complex weights and large names into the input store."""

        store = getattr(ctx, "store", None) if ctx is not None else None
        if store is None:
            raise ValueError(
                "EncodedReceiver weights require an export context with an HDF5 store"
            )

        blocks = tuple(self._authored_weight_blocks())
        component_count = len(self.components)
        row_count = (self.encoding_count or 0) * component_count

        def weight_chunks() -> Iterator[np.ndarray]:
            target_bytes = 32 * 1024 * 1024
            row_bytes = max(1, point_count * 2 * np.dtype(np.float32).itemsize)
            rows_per_chunk = max(1, target_bytes // row_bytes)
            for block, conjugate_block in blocks:
                rows = block.reshape(-1, point_count)
                for start in range(0, rows.shape[0], rows_per_chunk):
                    stop = min(rows.shape[0], start + rows_per_chunk)
                    values = rows[start:stop, :]
                    split = np.empty((*values.shape, 2), dtype=np.float32)
                    split[..., 0] = values.real
                    split[..., 1] = values.imag
                    if conjugate_block:
                        split[..., 1] *= -1.0
                    yield split

        base = f"inputs/acquisition/receivers/{group_name}"
        ref = store.put_array_chunks(
            f"{base}/weights",
            (row_count, point_count, 2),
            weight_chunks,
            attrs={"fs_kind": "encoded_receiver_weights"},
            dims=("encoded_component", "receiver", "complex"),
            dtype=np.float32,
        )
        payload = {"_type": "HDF5Dense", **ref.to_fs(format="HDF5")}
        if self.encoding_names is not None and len(self.encoding_names) > 64:
            names_ref = store.put_string_array(
                f"{base}/encoding_names",
                self.encoding_names,
                dimension="encoding",
                attrs={"fs_kind": "encoded_receiver_names"},
            )
            payload["names_dataset"] = names_ref.clean_dataset
        return payload

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        *,
        group_name: Optional[str] = None,
        point_count: Optional[int] = None,
    ) -> dict:
        """Serialize the encoded receiver with one compact HDF5 weight table."""

        if group_name is None or point_count is None:
            if self.has_authored_weights:
                raise ValueError(
                    "EncodedReceiver weights must be serialized through ReceiverGroup"
                )
            if self.weight_table is None:
                raise ValueError("EncodedReceiver requires a pointwise weight table")
        if point_count is not None:
            self.validate_size(point_count, ctx)
        if self.has_authored_weights:
            assert group_name is not None and point_count is not None
            weight_payload = self._materialize_weight_table(
                ctx, group_name=group_name, point_count=point_count
            )
        else:
            assert self.weight_table is not None
            weight_payload = self.weight_table.to_fs(ctx)
        inline_names = (
            self.encoding_names
            if self.encoding_names is not None and "names_dataset" not in weight_payload
            else None
        )
        return {
            "_type": "EncodedReceiver",
            **ReceiverDevice.to_fs(self, ctx),
            "encoding_count": self.encoding_count,
            **({"encoding_names": list(inline_names)} if inline_names else {}),
            "reduction": self.reduction,
            "weights": weight_payload,
        }

    @classmethod
    def from_fs(cls, data: dict) -> "EncodedReceiver":
        """Deserialize an encoded-receiver HDF5 reference."""

        weight_table = ReceiverWeightTable.from_fs(data["weights"])
        return cls(
            name=data.get("name"),
            components=[ReceiverComponent.from_fs(c) for c in data["components"]],
            response=data.get("response"),
            encoding_names=data.get("encoding_names"),
            encoding_count=data.get("encoding_count"),
            reduction=data.get("reduction", "sum"),
            weight_table=weight_table,
        )


@register_class
@dataclass(kw_only=True)
class ReceiverFiber(ReceiverDevice):
    """Defines a fiber receiver device (e.g. a DAS fiber) that integrates
    response over a length.

    The response is integrated over the gauge length using either a reusable
    sample spacing or a requested number of points per gauge.

    Args:
        name: Optional device name.
        components: Receiver components measured by each channel.
        gauge_length: Physical gauge length.
        channel_spacing: Physical spacing between fiber channels. Defaults to
            ``gauge_length``.
        sample_spacing: Physical spacing for integration samples along a gauge.
        points_per_gauge: Number of integration samples when sample spacing is
            not provided.
        radius: Optional helical-fiber radius. Required with ``angle`` or
            ``pitch``.
        pitch: Optional helical-fiber pitch, mutually exclusive with ``angle``.
        angle: Optional helical-fiber winding angle from the cable axis,
            mutually exclusive with ``pitch``. Plain numbers are degrees;
            unit-aware angular quantities are also accepted.
        response: Optional receiver response wavelet.

    Raises:
        ValueError: If ``gauge_length`` is omitted or ``points_per_gauge`` is
            not positive, or if the helical-fiber geometry is invalid.
    """

    gauge_length: Any = None
    channel_spacing: Optional[Any] = None
    sample_spacing: Optional[Any] = None
    points_per_gauge: Optional[int] = None
    radius: Optional[Any] = None
    pitch: Optional[Any] = None
    angle: Optional[Any] = None

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        components: Optional[List[ReceiverComponent]] = None,
        gauge_length: Optional[Any] = None,
        channel_spacing: Optional[Any] = None,
        sample_spacing: Optional[Any] = None,
        points_per_gauge: Optional[int] = None,
        radius: Optional[Any] = None,
        pitch: Optional[Any] = None,
        angle: Optional[Any] = None,
        response: Optional[Wavelet] = None,
    ):
        if gauge_length is None:
            raise ValueError("ReceiverFiber requires gauge_length.")
        if points_per_gauge is not None:
            points_per_gauge = int(points_per_gauge)
            if points_per_gauge < 1:
                raise ValueError("ReceiverFiber points_per_gauge must be positive.")

        self.name = name
        self.components = list(components) if components is not None else []
        self.response = response
        self.gauge_length = gauge_length
        self.channel_spacing = (
            channel_spacing if channel_spacing is not None else gauge_length
        )
        self.sample_spacing = sample_spacing
        self.points_per_gauge = points_per_gauge
        self.radius = radius
        self.pitch = pitch
        self.angle = angle
        self._validate_helical_geometry()

    def to_fs(self, ctx: Optional[ExportContext] = None) -> dict:
        """Serialize this fiber receiver device for solver input."""

        self._validate_helical_geometry()
        data = {
            "_type": self.__class__.__name__,
            **super().to_fs(ctx),
            "gauge_length": value_and_units_to_fs(self.gauge_length),
        }
        if self.channel_spacing is not None:
            data["channel_spacing"] = value_and_units_to_fs(self.channel_spacing)
        if self.sample_spacing is not None:
            data["sample_spacing"] = value_and_units_to_fs(self.sample_spacing)
        if self.points_per_gauge is not None:
            data["points_per_gauge"] = self.points_per_gauge
        if self.radius is not None:
            data["radius"] = value_and_units_to_fs(self.radius)
        if self.pitch is not None:
            data["pitch"] = value_and_units_to_fs(self.pitch)
        if self.angle is not None:
            data["angle"] = _receiver_fiber_angle_to_fs(self.angle)
        return data

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverFiber":
        """Deserialize a fiber receiver device payload."""

        return cls(
            name=data.get("name"),
            components=[ReceiverComponent.from_fs(c) for c in data["components"]],
            response=data.get("response"),
            gauge_length=data.get("gauge_length"),
            channel_spacing=data.get("channel_spacing"),
            sample_spacing=data.get("sample_spacing"),
            points_per_gauge=data.get("points_per_gauge"),
            radius=data.get("radius"),
            pitch=data.get("pitch"),
            angle=data.get("angle"),
        )

    def _validate_helical_geometry(self) -> None:
        if self.pitch is not None and self.angle is not None:
            raise ValueError(
                "ReceiverFiber accepts only one of angle or pitch, not both."
            )
        if (self.pitch is not None or self.angle is not None) and self.radius is None:
            raise ValueError(
                "ReceiverFiber radius is required when angle or pitch is specified."
            )
        if self.angle is not None:
            _receiver_fiber_angle_degrees(self.angle)


@register_class
@dataclass(kw_only=True)
class ReceiverNodeArray(ReceiverArray):
    """Deprecated compatibility name for :class:`ReceiverArray`."""

    def __post_init__(self) -> None:
        warnings.warn(
            "ReceiverNodeArray is deprecated; use ReceiverArray.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__post_init__()

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverNodeArray":
        """Deserialize a node-array receiver device payload."""

        return cls(
            name=data.get("name"),
            components=[ReceiverComponent.from_fs(c) for c in data["components"]],
            response=data.get("response"),
            offsets=data["offsets"],
            offset_units=data.get("offset_units"),
            reduction=data.get("reduction", "mean"),
        )


@register_class
@dataclass(kw_only=True)
class ReceiverNode(ReceiverDevice):
    """Point receiver device evaluated at each receiver coordinate."""

    def to_fs(self, ctx: Optional[ExportContext] = None) -> dict:
        """Serialize this point receiver device for solver input."""

        return {"_type": self.__class__.__name__, **super().to_fs(ctx)}

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverNode":
        """Deserialize a point receiver device payload."""

        return cls(
            name=data.get("name"),
            components=[ReceiverComponent.from_fs(c) for c in data["components"]],
            response=data.get("response"),
        )


# ----------------------------------------------------------------------
# Receiver Coordinates
# ----------------------------------------------------------------------
@register_class
@dataclass(kw_only=True)
class ReceiverCoords(TypeTaggedMixin, ABC):
    """Base class for receiver coordinates.

    Enables different ways of specifying receiver locations.

    Attributes:
       name (str): Identifier for this set of coordinates.
    """

    @property
    @abstractmethod
    def size(self) -> int:
        """Get the total number of receivers.

        Returns:
           int: Number of receivers.
        """
        pass

    @property
    @abstractmethod
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinate bounds without loading full dataset.

        Returns:
           Tuple[np.ndarray, np.ndarray]: Min and max coordinates.
        """
        pass

    @abstractmethod
    def get(self, indices) -> np.ndarray:
        """Get coordinates for specified indices.

        Args:
           indices: Integer indices or boolean mask.

        Returns:
           np.ndarray: Coordinate array for requested receivers.
        """
        pass

    @abstractmethod
    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Convert coordinates to a solver payload."""
        pass

    @classmethod
    def from_fs(cls, data: Dict) -> "ReceiverCoords":
        """Deserialize a registered receiver-coordinate payload."""

        return cls.dispatch_from_fs(data, class_registry)


@register_class
@dataclass(kw_only=True)
class CoordsFromFile(ReceiverCoords):
    """Receiver coordinates stored in a file.

    Args:
        file: Coordinate file path. Relative paths are resolved from the export
            or project context when serialized/read.
        format: Coordinate file format. Currently ``"HDF5"``.
        dset: HDF5 dataset name. Defaults to ``"coords"``.
        units: Optional coordinate units.
        system: Optional coordinate-system name.
        hash: Optional content hash for freshness checks.
    """

    file: Path
    format: Literal["HDF5"]
    dset: Optional[str] = None
    units: Optional[str] = None
    system: Optional[str] = None
    hash: Optional[str] = None

    def __init__(
        self,
        file: Union[str, Path] = None,
        format: Literal["HDF5"] = "HDF5",
        dset: Optional[str] = None,
        units: Optional[str] = None,
        system: Optional[str] = None,
        hash: Optional[str] = None,
        **kwargs,
    ):
        if file is None and "path" in kwargs:
            file = kwargs.pop("path")
        if kwargs.pop("remote", False) or _is_remote_file_reference(file):
            raise ValueError(
                "CoordsFromFile does not support remote coordinate files yet; "
                "provide a local file or inline/materialized coordinates."
            )
        self.file = Path(file).expanduser()
        if self.file.is_absolute():
            self.file = self.file.resolve()
        self.format = format
        self.dset = dset
        self.units = units
        self.system = system
        self.hash = hash
        self._hash_cache: Optional[tuple[tuple[Any, ...], str]] = None
        self._bounds_cache: Optional[
            tuple[tuple[Any, ...], Tuple[np.ndarray, np.ndarray]]
        ] = None

    @classmethod
    def from_fs(cls, data: Dict) -> "CoordsFromFile":
        """Deserialize file-backed receiver coordinates."""

        file = data["file"]
        format = data["format"]
        if data.get("remote", False) or _is_remote_file_reference(file):
            raise ValueError(
                "CoordsFromFile does not support remote coordinate files yet; "
                "provide a local file or inline/materialized coordinates."
            )
        if format == "HDF5":
            if ":" in file:
                file, dset = file.split(":", 1)
            else:
                dset = data.get("dataset", "coords")
        return cls(
            file=Path(file),
            format=format,
            dset=dset,
            units=data.get("units"),
            system=data.get("system"),
            hash=data.get("hash"),
        )

    @staticmethod
    def _simulation_rel_path(rel_path: Optional[Path]) -> Optional[Path]:
        if rel_path is None:
            return None
        rel_path = Path(rel_path)
        parts = rel_path.parts
        if "simulations" not in parts:
            return None
        index = parts.index("simulations")
        if len(parts) <= index + 1:
            return None
        return Path(*parts[: index + 2])

    def _contextual_file(
        self, ctx=None, *, source_project_path: Optional[Path] = None
    ) -> Path:
        file = Path(self.file).expanduser()
        project_path = getattr(ctx, "project_path", None)
        rel_path = getattr(ctx, "rel_path", None)
        if project_path is not None:
            project_path = Path(project_path).expanduser().resolve()

        if not file.is_absolute():
            return project_path / file if project_path is not None else file

        if source_project_path is not None and project_path is not None:
            try:
                project_relative = file.resolve().relative_to(
                    Path(source_project_path).expanduser().resolve()
                )
            except ValueError:
                pass
            else:
                return project_path / project_relative

        simulation_rel = self._simulation_rel_path(rel_path)
        if project_path is None or simulation_rel is None:
            return file
        expected = project_path / simulation_rel / f"{simulation_rel.name}.h5"
        is_simulation_store = (
            file.name == expected.name
            and file.parent.name == expected.parent.name
            and file.parent.parent.name == "simulations"
        )
        if is_simulation_store and expected.exists():
            return expected
        return file

    def _relative_file(self, ctx=None) -> Path:
        project_path = getattr(ctx, "project_path", None)
        file = self._contextual_file(ctx)
        if project_path is None:
            return file
        try:
            return file.resolve().relative_to(Path(project_path).resolve())
        except Exception:
            return file

    def _local_file(self, ctx=None) -> Path:
        file = self._contextual_file(ctx)
        if file.is_absolute():
            return file
        project_path = getattr(ctx, "project_path", None)
        if project_path is not None:
            return Path(project_path) / file
        return file

    def _hdf5_metadata(self, ctx=None) -> Tuple[Optional[str], Optional[str]]:
        if self.format != "HDF5":
            return None, None
        file = self._local_file(ctx)
        dataset = self.dset or "coords"
        if not file.exists():
            return None, None
        try:
            with h5py.File(file, "r") as h5:
                if dataset not in h5:
                    return None, None
                attrs = h5[dataset].attrs
                units = _h5_attr_string(attrs.get("units"))
                system = _h5_attr_string(
                    attrs.get("system", attrs.get("coordinate_system"))
                )
                return units, system
        except OSError:
            return None, None

    def _fill_metadata_from_file(self, ctx=None) -> None:
        if self.units is not None and self.system is not None:
            return
        units, system = self._hdf5_metadata(ctx)
        if self.units is None:
            self.units = units
        if self.system is None:
            self.system = system

    def _content_hash(self, ctx=None) -> Optional[str]:
        if self.format != "HDF5":
            return None
        file = self._local_file(ctx)
        if not file.exists():
            return None
        dataset = self.dset or "coords"
        try:
            with h5py.File(file, "r") as h5:
                if dataset not in h5:
                    return None
                dset = h5[dataset]
                signature = (
                    file.stat().st_mtime_ns,
                    file.stat().st_size,
                    dataset,
                    dset.shape,
                    str(dset.dtype),
                    self.units,
                    self.system,
                    getattr(ctx, "default_length_units", None),
                )
                if self._hash_cache is not None and self._hash_cache[0] == signature:
                    return self._hash_cache[1]
                if dset.ndim not in {1, 2}:
                    return None
                width = 1 if dset.ndim == 1 else int(dset.shape[1])
                if width == 2:
                    coordinate = ["x", "z"]
                elif width == 3:
                    coordinate = ["x", "y", "z"]
                else:
                    coordinate = list(range(width))
                file_units, file_system = self._hdf5_metadata(ctx)
                units = (
                    self.units
                    or file_units
                    or getattr(ctx, "default_length_units", None)
                )
                system = self.system or file_system
                metadata = {
                    "dtype": str(dset.dtype),
                    "shape": tuple(int(value) for value in dset.shape),
                    "dims": ["receiver", "coordinate"],
                    "coordinate": coordinate,
                    "units": unit_expression(units) if units is not None else None,
                    "system": system,
                }
                hasher = blake3.blake3()
                hasher.update(json.dumps(metadata, sort_keys=True).encode("utf-8"))
                rows = int(dset.shape[0])
                row_bytes = max(1, width * dset.dtype.itemsize)
                rows_per_chunk = max(1, (32 * 1024 * 1024) // row_bytes)
                for start in range(0, rows, rows_per_chunk):
                    values = np.ascontiguousarray(
                        dset[start : min(rows, start + rows_per_chunk)]
                    )
                    hasher.update(values.data.cast("B"))
        except OSError:
            return None
        result = f"blake3:{hasher.hexdigest()}"
        self._hash_cache = (signature, result)
        return result

    @property
    def size(self) -> int:
        """Get the total number of receivers.

        Returns:
           int: Number of receivers.
        """
        if self.format == "HDF5":
            with h5py.File(self._local_file(), "r") as f:
                return f[self.dset or "coords"].shape[0]
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinate bounds without loading full dataset.

        Returns:
           Tuple[np.ndarray, np.ndarray]: Min and max coordinates.
        """
        if self.format == "HDF5":
            file = self._local_file()
            dataset = self.dset or "coords"
            signature = (file.stat().st_mtime_ns, file.stat().st_size, dataset)
            if self._bounds_cache is not None and self._bounds_cache[0] == signature:
                return self._bounds_cache[1]
            with h5py.File(file, "r") as f:
                coords = f[dataset]
                if coords.shape[0] == 0:
                    raise ValueError("Receiver coordinate dataset must not be empty")
                width = 1 if coords.ndim == 1 else int(coords.shape[1])
                row_bytes = max(1, width * coords.dtype.itemsize)
                rows_per_chunk = max(1, (32 * 1024 * 1024) // row_bytes)
                lower = np.full(width, np.inf)
                upper = np.full(width, -np.inf)
                for start in range(0, int(coords.shape[0]), rows_per_chunk):
                    chunk = np.asarray(
                        coords[start : min(coords.shape[0], start + rows_per_chunk)]
                    ).reshape(-1, width)
                    lower = np.minimum(lower, np.min(chunk, axis=0))
                    upper = np.maximum(upper, np.max(chunk, axis=0))
                result = (lower, upper)
                self._bounds_cache = (signature, result)
                return result
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    def __getitem__(self, key: Union[tuple, slice]):
        """Return receiver coordinates selected from the backing file.

        Args:
            key: Slice, integer index, tuple, or boolean mask accepted by the
                HDF5 coordinate dataset.

        Returns:
            Coordinate array for the selected receivers.
        """

        return self.get(key)

    def get(self, indices: Optional[Union[Tuple, slice]] = None) -> np.ndarray:
        """Get coordinates for specified indices.

        Args:
           indices: Integer indices or boolean mask.

        Returns:
           np.ndarray: Coordinate array for requested receivers.
        """
        if self.format == "HDF5":
            file = self._local_file()
            with h5py.File(file, "r") as f:
                dataset = self.dset or "coords"
                if dataset not in f:
                    raise KeyError(
                        f"Receiver coordinate dataset '{dataset}' is missing from "
                        f"'{file}'. Rebuild and save the simulation acquisition "
                        "inputs before copying or running it."
                    )
                if indices is None:
                    return f[dataset][:]
                else:
                    return f[dataset][indices]
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Serialize file-backed receiver coordinates for solver input."""

        rel_path = self._relative_file(ctx)
        default_units = getattr(ctx, "default_length_units", None)
        units = self.units if self.units is not None else default_units
        system = self.system

        if self.format == "HDF5":
            file = str(rel_path)
            dataset = self.dset or "coords"
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

        file_hash = self.hash or self._content_hash(ctx)
        return {
            "_type": self.__class__.__name__,
            "file": file,
            "dataset": dataset,
            "format": self.format,
            **({"hash": file_hash} if file_hash is not None else {}),
            **({"units": unit_expression(units)} if units is not None else {}),
            **({"system": system} if system is not None else {}),
        }


@register_class
@dataclass(kw_only=True)
class CoordsGrid(ReceiverCoords):
    """Receiver coordinates defined by a Cartesian grid.

    Args:
        grid: Cartesian grid defining receiver locations.
        units: Optional coordinate units override.
        system: Optional coordinate-system name override.
    """

    grid: CartesianGrid
    units: Optional[str] = None
    system: Optional[str] = None

    @property
    def size(self) -> int:
        """Return the number of receiver coordinates in the grid."""

        return int(np.prod(self.grid.n))

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper coordinate bounds for the grid."""

        return self.grid.x0, self.grid.x1

    def get(
        self, indices: Optional[Union[int, slice, List[int], List[slice]]] = None
    ) -> np.ndarray:
        """Get coordinates for specified indices.

        Args:
           indices: Can be:
              - None:        Return all coordinates
              - int:         Single flat index into the coordinate array
              - slice:       Slice of flat indices
              - List[int]:   List of flat indices
              - List[slice]: Tensor indices directly into the grid dimensions

        Returns:
           np.ndarray: Array of coordinates for requested indices
        """
        # Return all coordinates if indices is None
        if indices is None:
            return self.grid.get_coords()

        if isinstance(indices, list) and not indices:
            return np.empty((0, len(self.grid.n)), dtype=float)

        # List[slice] case - pass directly to grid
        elif isinstance(indices, list) and isinstance(indices[0], slice):
            return self.grid.get_coords(indices)

        # List[int] case - convert each index to tensor indices
        elif isinstance(indices, list) and isinstance(indices[0], int):
            coords = []
            for idx in indices:
                tensor_indices = []
                remaining = idx
                for n in reversed(self.grid.n):
                    tensor_indices.insert(0, slice(remaining % n, (remaining % n) + 1))
                    remaining //= n
                coords.append(self.grid.get_coords(tensor_indices)[0])
            return np.array(coords)

        # Single int case - convert to tensor indices
        elif isinstance(indices, int):
            tensor_indices = []
            remaining = indices
            for n in reversed(self.grid.n):
                tensor_indices.insert(0, slice(remaining % n, (remaining % n) + 1))
                remaining //= n
            return self.grid.get_coords(tensor_indices)

        # For slice, get all coords and then slice
        elif isinstance(indices, slice):
            coords = self.grid.get_coords()
            return coords[indices]
        else:
            raise ValueError("Invalid indices type")

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Serialize grid receiver coordinates for solver input."""

        payload = {"_type": self.__class__.__name__, "grid": self.grid.to_fs(ctx)}
        if self.units is not None:
            payload["units"] = unit_expression(self.units)
        if self.system is not None:
            payload["system"] = self.system
        return payload

    @classmethod
    def from_fs(cls, data: Dict) -> "CoordsGrid":
        """Deserialize grid-backed receiver coordinates."""

        return cls(
            grid=CartesianGrid.from_fs(data["grid"]),
            units=data.get("units"),
            system=data.get("system"),
        )


class CoordsSurfaceCarpet(ReceiverCoords):
    """Receiver coordinates on a surface-relative tensor-product carpet.

    The carpet is kept in compact axis form until values are explicitly
    requested or exported to HDF5.
    """

    def __init__(
        self,
        *,
        x: Any,
        y: Optional[Any] = None,
        offset: float = 0.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> None:
        self.x = _surface_carpet_axis("x", x, units)
        self.y = None if y is None else _surface_carpet_axis("y", y, units)
        self.offset = float(offset)
        self.units = units
        self.system = system

    @classmethod
    def try_from_surface(
        cls,
        surface: Any,
        *,
        x: Any,
        y: Optional[Any] = None,
        units: Optional[Any] = None,
        above: Optional[Any] = None,
        below: Optional[Any] = None,
    ) -> Optional["CoordsSurfaceCarpet"]:
        """Return a compact carpet when the surface helper exposes metadata."""

        if above is not None and below is not None:
            raise ValueError("Specify only one of above or below")

        simulation = getattr(surface, "_simulation", None)
        if getattr(simulation, "dimension", None) == 3 and y is None:
            raise ValueError("3D surface points_grid requires x and y axes")

        system = getattr(surface, "coordinate_system", surface)
        system_name = getattr(system, "name", None)
        if system_name is None or getattr(system, "type", None) != "surface":
            return None

        carpet_units = units or _first_quantity_units(x)
        if carpet_units is None and y is not None:
            carpet_units = _first_quantity_units(y)

        offset_value = 0.0
        distance = above if above is not None else below
        if carpet_units is None and distance is not None:
            carpet_units = _first_quantity_units(distance)

        if distance is not None:
            normal = str(getattr(system, "normal", "up") or "up").strip().lower()
            if above is not None:
                sign = -1 if normal == "down" else 1
            else:
                sign = 1 if normal == "down" else -1
            try:
                offset_value = sign * _surface_carpet_scalar(
                    "surface offset", distance, carpet_units
                )
            except ValueError as exc:
                if "must be a scalar" in str(exc):
                    return None
                raise

        return cls(
            x=x,
            y=y,
            offset=offset_value,
            units=carpet_units,
            system=system_name,
        )

    @property
    def dimension(self) -> int:
        """Return the number of coordinate columns."""

        return 2 if self.y is None else 3

    @property
    def axes(self) -> List[str]:
        """Return coordinate-axis labels."""

        return ["x", "z"] if self.y is None else ["x", "y", "z"]

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the dense coordinate table shape."""

        return (self.size, self.dimension)

    @property
    def size(self) -> int:
        """Return the number of receiver coordinates."""

        if self.y is None:
            return int(self.x.size)
        return int(self.x.size * self.y.size)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper coordinate bounds without materializing rows."""

        if self.y is None:
            lower = np.asarray([np.min(self.x), self.offset], dtype=float)
            upper = np.asarray([np.max(self.x), self.offset], dtype=float)
        else:
            lower = np.asarray(
                [np.min(self.x), np.min(self.y), self.offset], dtype=float
            )
            upper = np.asarray(
                [np.max(self.x), np.max(self.y), self.offset], dtype=float
            )
        return lower, upper

    def get(
        self, indices: Optional[Union[int, slice, Sequence[int], np.ndarray]] = None
    ) -> np.ndarray:
        """Return coordinate values for all or selected receivers."""

        if indices is None:
            values = np.empty(self.shape, dtype=np.float64)
            offset = 0
            for chunk in self.iter_chunks():
                stop = offset + chunk.shape[0]
                values[offset:stop] = chunk
                offset = stop
            return values

        if isinstance(indices, slice):
            flat_indices = np.arange(self.size, dtype=np.int64)[indices]
        elif isinstance(indices, (int, np.integer)):
            index = int(indices)
            if index < 0:
                index += self.size
            return self._rows_for_indices(np.asarray([index], dtype=np.int64))[0]
        else:
            flat_indices = np.asarray(indices)
            if flat_indices.dtype == np.dtype(bool):
                flat_indices = np.nonzero(flat_indices.reshape(-1))[0]
            flat_indices = flat_indices.astype(np.int64, copy=False).reshape(-1)
            flat_indices[flat_indices < 0] += self.size
        return self._rows_for_indices(flat_indices)

    def iter_chunks(self, chunk_size: int = 1 << 20) -> Iterator[np.ndarray]:
        """Yield dense coordinate chunks in receiver-major order."""

        chunk_size = max(1, int(chunk_size))
        if self.y is None:
            for start in range(0, self.size, chunk_size):
                x_values = self.x[start : start + chunk_size]
                chunk = np.empty((x_values.size, 2), dtype=np.float64)
                chunk[:, 0] = x_values
                chunk[:, 1] = self.offset
                yield chunk
            return

        for start in range(0, self.size, chunk_size):
            stop = min(start + chunk_size, self.size)
            yield self._rows_for_indices(np.arange(start, stop, dtype=np.int64))

    def to_hdf5_reference(
        self,
        store: Any,
        dataset: str,
        *,
        attrs: Optional[Dict[str, Any]] = None,
        dtype: Any = np.float64,
    ):
        """Write this carpet to a store-backed HDF5 dataset."""

        return store.put_array_chunks(
            dataset,
            self.shape,
            self.iter_chunks,
            attrs=attrs,
            dims=["receiver", "coordinate"],
            coords={"coordinate": np.asarray(self.axes, dtype=str)},
            dtype=dtype,
        )

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Serialize inline when no HDF5 export context is available."""

        units = (
            self.units
            if self.units is not None
            else getattr(ctx, "default_length_units", None)
        )
        return CoordsArray(
            coordinates=self.get(),
            units=units,
            system=self.system,
        ).to_fs(ctx)

    def _rows_for_indices(self, indices: np.ndarray) -> np.ndarray:
        if np.any((indices < 0) | (indices >= self.size)):
            raise IndexError("receiver coordinate index out of range")

        if self.y is None:
            rows = np.empty((indices.size, 2), dtype=np.float64)
            rows[:, 0] = self.x[indices]
            rows[:, 1] = self.offset
            return rows

        nx = self.x.size
        x_index = indices % nx
        y_index = indices // nx
        rows = np.empty((indices.size, 3), dtype=np.float64)
        rows[:, 0] = self.x[x_index]
        rows[:, 1] = self.y[y_index]
        rows[:, 2] = self.offset
        return rows


@register_class
@dataclass(kw_only=True)
class CoordsArray(ReceiverCoords):
    """Receiver coordinates stored as an xarray/numpy array.

    Args:
        coordinates: Coordinate array with shape ``(n_receivers, dimension)``.
            xarray input may carry ``units`` and ``system`` attributes.
        units: Optional coordinate units.
        system: Optional coordinate-system name.

    Raises:
        ValueError: If a NumPy coordinate array is not two-dimensional with two
            or three coordinate columns.
    """

    coordinates: Union[xr.DataArray, np.ndarray]
    units: Optional[str] = None
    system: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.coordinates, xr.DataArray):
            if self.units is None:
                self.units = self.coordinates.attrs.get("units")
            if self.system is None:
                self.system = self.coordinates.attrs.get("system")
            self.coordinates = self.coordinates.astype(np.float64, copy=False)

        if isinstance(self.coordinates, np.ndarray):
            self.coordinates = np.asarray(self.coordinates, dtype=np.float64)
            if self.coordinates.ndim != 2:
                raise ValueError(
                    "Coordinates array must be 2D with shape (n_receivers, <simulation dimension>)"
                )
            if self.coordinates.shape[1] == 2:
                self.coordinates = xr.DataArray(
                    self.coordinates,
                    dims=["receiver", "coordinate"],
                    coords={"coordinate": ["x", "z"]},
                )
            elif self.coordinates.shape[1] == 3:
                self.coordinates = xr.DataArray(
                    self.coordinates,
                    dims=["receiver", "coordinate"],
                    coords={"coordinate": ["x", "y", "z"]},
                )
            else:
                raise ValueError("Coordinates array must have 2 or 3 columns")

    @property
    def size(self) -> int:
        """Return the number of receiver coordinates."""

        return len(self.coordinates)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return minimum and maximum coordinate values by coordinate axis."""

        return (
            self.coordinates.min(dim="receiver").values,
            self.coordinates.max(dim="receiver").values,
        )

    def get(self, indices: Optional[Union[int, slice]] = None) -> np.ndarray:
        """Return coordinate values for all or selected receivers.

        Args:
            indices: Optional integer, slice, or indexer accepted by xarray.

        Returns:
            ``float64`` coordinate array.
        """

        if indices is None:
            return np.asarray(self.coordinates.values, dtype=np.float64)
        else:
            return np.asarray(self.coordinates[indices].values, dtype=np.float64)

    def to_file(
        self, file_name: Union[str, Path], format: Optional[Literal["HDF5"]] = None
    ) -> CoordsFromFile:
        """Write coordinates to file and return CoordsFromFile object.

        Args:
            file_name: Output coordinate file path.
            format: Optional file format. Inferred from ``file_name`` when
                omitted.

        Returns:
            ``CoordsFromFile`` pointing at the written HDF5 dataset.

        Raises:
            ValueError: If the file extension does not identify a supported
                format.
        """
        file = Path(file_name).expanduser()
        if format is None:
            if str(file).endswith(".h5") or str(file).endswith(".hdf5"):
                format = "HDF5"
            else:
                raise ValueError(f"Unknown coordinates file extension: {file_name}")

        if not file.is_absolute():
            file = file.resolve()
        if not file.parent.exists():
            file.parent.mkdir(parents=True)

        if format == "HDF5":
            with h5py.File(file, "w") as f:
                dset = f.create_dataset(
                    "coords", data=(self.coordinates.values).astype(np.float64)
                )
                if self.units is not None:
                    dset.attrs["units"] = unit_expression(self.units)
                if self.system is not None:
                    dset.attrs["system"] = self.system
        else:
            raise NotImplementedError(f"Format {format} not implemented")

        return CoordsFromFile(
            file=file,
            dset="coords",
            format=format,
            units=self.units,
            system=self.system,
        )

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Serialize inline receiver coordinates for solver input."""

        values = np.asarray(self.coordinates.values, dtype=np.float64).tolist()
        payload = {"_type": self.__class__.__name__, "value": values}
        if self.units is not None:
            payload["units"] = unit_expression(self.units)
        if self.system is not None:
            payload["system"] = self.system
        return payload

    @classmethod
    def from_fs(cls, data: Dict) -> "CoordsArray":
        """Deserialize inline receiver coordinates."""

        coords = np.array(data.get("coords", data.get("value")), dtype=np.float64)
        return cls(
            coordinates=coords, units=data.get("units"), system=data.get("system")
        )


def _surface_carpet_axis(name: str, values: Any, units: Optional[Any]) -> np.ndarray:
    if is_quantity(values):
        values = values.to(units).magnitude if units is not None else values.magnitude
    else:
        quantity_units = _first_quantity_units(values)
        if quantity_units is not None:
            target_units = units or quantity_units
            values = _strip_coordinate_quantities(values, target_units)
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size == 0:
        raise ValueError(f"{name} must contain at least one coordinate")
    if not np.isfinite(axis).all():
        raise ValueError(f"{name} coordinates must be finite")
    return axis


def _surface_carpet_scalar(name: str, value: Any, units: Optional[Any]) -> float:
    if is_quantity(value):
        value = value.to(units).magnitude if units is not None else value.magnitude
    else:
        quantity_units = _first_quantity_units(value)
        if quantity_units is not None:
            target_units = units or quantity_units
            value = _strip_coordinate_quantities(value, target_units)
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar for compact carpet coordinates")
    scalar = float(array.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _first_quantity_units(value: Any) -> Optional[Any]:
    if is_quantity(value):
        return value.units
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.flat:
                units = _first_quantity_units(item)
                if units is not None:
                    return units
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            units = _first_quantity_units(item)
            if units is not None:
                return units
    return None


def _h5_attr_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
        return _h5_attr_string(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _h5_attr_string(value[0])
    return str(value)


def _strip_coordinate_quantities(value: Any, units: Any) -> Any:
    if is_quantity(value):
        return value.to(units).magnitude
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            values = [_strip_coordinate_quantities(item, units) for item in value.flat]
            return np.asarray(values, dtype=float).reshape(value.shape)
        return value
    if isinstance(value, (list, tuple)):
        return [_strip_coordinate_quantities(item, units) for item in value]
    return value


def coordinate_array_metadata(
    coords: Any,
) -> Tuple[np.ndarray, Optional[Any], Optional[str]]:
    """Return numeric coordinate values plus units/system metadata.

    Pint quantities intentionally become magnitudes here, with their units
    carried on the receiver/source coordinate object instead of being stripped
    implicitly by NumPy.

    Args:
        coords: Raw coordinate array, Pint quantity, or ``CoordinateValue``.

    Returns:
        ``(values, units, system)`` with numeric values in double precision.
    """

    system = None
    explicit_units = None
    if isinstance(coords, CoordinateValue):
        explicit_units = coords.units
        system = coords.system
        coords = coords.value

    quantity_units = _first_quantity_units(coords)
    units = explicit_units if explicit_units is not None else quantity_units
    if quantity_units is not None:
        coords = _strip_coordinate_quantities(coords, units)

    return np.asarray(coords, dtype=np.float64), units, system


# ----------------------------------------------------------------------
# Receiver Groups
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class ReceiverGroup(ExtraFieldsMixin):
    """A group of multi-component receivers with shared output settings.

    This class represents a collection of receivers that measure one or more physical
    quantities. All receivers in the group share output settings and their data will be
    written to the same output file.

    Args:
        name: String identifier for this receiver group.
        device: Device defining receiver type and components.
        coordinates: Receiver coordinates as an array, grid, file path, or
            ``ReceiverCoords`` object.
        domain: Optional domain where the receiver group is evaluated.
        sampling: Optional sparse survey sampling reference.
        survey: Convenience sparse survey name/reference. Merged with
            ``sampling`` when both are supplied.
        extra: Additional solver-facing receiver group fields.
        **kwargs: Additional solver-facing receiver group fields.

    Raises:
        TypeError: If deprecated frame arguments are supplied.
        ValueError: If ``coordinates`` cannot be interpreted.
    """

    name: str = "group"
    device: ReceiverDevice = field(default_factory=ReceiverDevice)
    domain: Optional[int] = None
    coordinates: ReceiverCoords = field(default_factory=ReceiverCoords)
    sampling: Optional[ReceiverSampling] = None
    extra: Dict = field(default_factory=dict)

    @property
    def size(self):
        """Return the number of receiver locations in this group."""

        return self.coordinates.size

    @property
    def output_size(self) -> int:
        """Return the receiver-row count written by this group's device."""

        return self.device.output_receiver_count(self.size)

    @property
    def grid(self) -> Optional[CartesianGrid]:
        """Return the receiver grid when this group uses ``CoordsGrid``."""

        if isinstance(self.coordinates, CoordsGrid):
            return self.coordinates.grid
        return None

    def __init__(
        self,
        name: str,
        device: ReceiverDevice,
        coordinates: Union[np.ndarray, xr.DataArray, str, Path, Grid, ReceiverCoords],
        domain: Optional[int] = None,
        sampling: Optional[Union[str, Dict, ReceiverSampling]] = None,
        survey: Optional[Union[str, ReceiverSampling]] = None,
        extra: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        deprecated_frame_keys = {"frame", "source_frame", "receiver_frame"} & set(
            kwargs
        )
        if deprecated_frame_keys:
            raise TypeError(
                "ReceiverGroup frame is no longer supported; receiver coordinates are physical"
            )
        coords = self._clean_coordinates(coordinates)
        sampling_obj = ReceiverSampling.from_value(sampling)
        survey_obj = ReceiverSampling.from_value(survey)
        if sampling_obj is None:
            sampling_obj = survey_obj
        elif survey_obj is not None and sampling_obj.survey is None:
            sampling_obj.survey = survey_obj.survey
        self.name = name
        self.device = device
        self.coordinates = coords
        self.domain = domain
        self.sampling = sampling_obj
        self._init_extra(extra, **kwargs)
        deprecated_frame_keys = {"frame", "source_frame", "receiver_frame"} & set(
            self.extra
        )
        if deprecated_frame_keys:
            raise TypeError(
                "ReceiverGroup frame is no longer supported; receiver coordinates are physical"
            )

    @property
    def survey(self) -> Optional[str]:
        """Return the sparse survey name referenced by this receiver group."""

        if self.sampling is None:
            return None
        return self.sampling.survey

    @survey.setter
    def survey(self, value: Optional[Union[str, ReceiverSampling]]) -> None:
        """Set sparse survey sampling from a name or ``ReceiverSampling``."""

        self.sampling = ReceiverSampling.from_value(value)

    @staticmethod
    def _clean_coordinates(coords):
        # Allow coordinates to be defined either as a ReceiverCoords object
        # various other reasonble ways:
        if isinstance(coords, CoordinateValue):
            values, units, system = coordinate_array_metadata(coords)
            out = CoordsArray(coordinates=values, units=units, system=system)
        elif isinstance(coords, ReceiverCoords):
            return coords
        elif isinstance(coords, xr.DataArray):
            out = CoordsArray(coordinates=coords)
        elif (
            is_quantity(coords)
            or isinstance(coords, np.ndarray)
            or isinstance(coords, list)
        ):
            values, units, system = coordinate_array_metadata(coords)
            out = CoordsArray(coordinates=values, units=units, system=system)
        elif isinstance(coords, str) or isinstance(coords, Path):
            suffix = Path(coords).suffix.lower()
            if suffix in [".h5", ".hdf5"]:
                out = CoordsFromFile(file=coords, format="HDF5")
            else:
                raise ValueError(f"Unknown coordinates file extension: {coords}")
        elif isinstance(coords, Grid):
            out = CoordsGrid(grid=coords)
        else:
            raise ValueError(f"Unknown coordinates type: {type(coords)}")
        return out

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Serialize this receiver group for solver input.

        Large inline coordinate arrays may be written to the simulation store or
        export directory when an export context is available.
        """

        coords = self.coordinates
        if isinstance(self.device, EncodedReceiver):
            self.device.validate_size(self.size, ctx)
        if isinstance(coords, CoordsSurfaceCarpet):
            default_units = getattr(ctx, "default_length_units", None)
            coordinate_units = (
                coords.units if coords.units is not None else default_units
            )
            if (
                coords.size > 200
                and ctx is not None
                and getattr(ctx, "store", None) is not None
            ):
                dataset = f"inputs/acquisition/receivers/{self.name}/coordinates"
                attrs = {"fs_kind": "receiver_coordinates"}
                if coordinate_units is not None:
                    attrs["units"] = unit_expression(coordinate_units)
                if coords.system is not None:
                    attrs["system"] = coords.system
                ref = coords.to_hdf5_reference(
                    ctx.store,
                    dataset,
                    attrs=attrs,
                    dtype=np.float64,
                )
                coords_payload = {
                    "_type": "CoordsFromFile",
                    **ref.to_fs(format="HDF5"),
                    **(
                        {"units": unit_expression(coordinate_units)}
                        if coordinate_units is not None
                        else {}
                    ),
                    **({"system": coords.system} if coords.system is not None else {}),
                }
            elif coords.size > 200 and ctx is not None and ctx.path is not None:
                file = ctx.path / self.name / "coords.h5"
                store = SimulationStore(file, project_path=ctx.project_path)
                attrs = {"fs_kind": "receiver_coordinates"}
                if coordinate_units is not None:
                    attrs["units"] = unit_expression(coordinate_units)
                if coords.system is not None:
                    attrs["system"] = coords.system
                ref = coords.to_hdf5_reference(
                    store,
                    "coords",
                    attrs=attrs,
                    dtype=np.float64,
                )
                coords_payload = {
                    "_type": "CoordsFromFile",
                    **ref.to_fs(format="HDF5"),
                    **(
                        {"units": unit_expression(coordinate_units)}
                        if coordinate_units is not None
                        else {}
                    ),
                    **({"system": coords.system} if coords.system is not None else {}),
                }
            else:
                coords_payload = coords.to_fs(ctx)
                if coordinate_units is not None and "units" not in coords_payload:
                    if "coords" in coords_payload:
                        coords_payload["value"] = coords_payload.pop("coords")
                    coords_payload["units"] = unit_expression(coordinate_units)
        elif isinstance(coords, CoordsArray):
            default_units = getattr(ctx, "default_length_units", None)
            coordinate_units = (
                coords.units if coords.units is not None else default_units
            )
            if (
                coords.size > 200
                and ctx is not None
                and getattr(ctx, "store", None) is not None
            ):
                dataset = f"inputs/acquisition/receivers/{self.name}/coordinates"
                attrs = {"fs_kind": "receiver_coordinates"}
                if coordinate_units is not None:
                    attrs["units"] = unit_expression(coordinate_units)
                if coords.system is not None:
                    attrs["system"] = coords.system
                assert isinstance(coords.coordinates, xr.DataArray)
                coordinate_dim = coords.coordinates.dims[1]
                ref = ctx.store.put_dataarray(
                    dataset,
                    coords.coordinates,
                    attrs=attrs,
                    coordinate_dims=(coordinate_dim,),
                    dtype=np.float64,
                )
                coords_payload = {
                    "_type": "CoordsFromFile",
                    **ref.to_fs(format="HDF5"),
                    **(
                        {"units": unit_expression(coordinate_units)}
                        if coordinate_units is not None
                        else {}
                    ),
                    **({"system": coords.system} if coords.system is not None else {}),
                }
            elif coords.size > 200 and ctx is not None and ctx.path is not None:
                coords_for_file = CoordsArray(
                    coordinates=coords.coordinates,
                    units=coordinate_units,
                    system=coords.system,
                )
                dump = coords_for_file.to_file(
                    file_name=ctx.path / self.name / "coords.h5",
                    format="HDF5",
                )
                coords_payload = dump.to_fs(ctx)
            else:
                coords_payload = coords.to_fs(ctx)
                if coordinate_units is not None and "units" not in coords_payload:
                    if "coords" in coords_payload:
                        coords_payload["value"] = coords_payload.pop("coords")
                    coords_payload["units"] = unit_expression(coordinate_units)
        else:
            coords_payload = self.coordinates.to_fs(ctx)

        device_payload = (
            self.device.to_fs(
                ctx,
                group_name=self.name,
                point_count=self.size,
            )
            if isinstance(self.device, EncodedReceiver)
            else self.device.to_fs(ctx)
        )
        payload = {
            "name": self.name,
            "device": device_payload,
            **({"domain": self.domain} if self.domain is not None else {}),
            **(
                {"sampling": self.sampling.to_fs(ctx)}
                if self.sampling is not None
                else {}
            ),
            "coordinates": coords_payload,
        }
        return merge_extra(payload, self.extra, "ReceiverGroup")

    @classmethod
    def from_fs(cls, data: Dict) -> "ReceiverGroup":
        """Deserialize a receiver group payload."""

        data = copy.deepcopy(data)
        coords = ReceiverCoords.from_fs(data.pop("coordinates", None))
        sampling = data.pop("sampling", None)
        data.pop("frame", None)

        return cls(
            name=data.pop("name", None),
            device=ReceiverDevice.from_fs(data.pop("device", None)),
            coordinates=coords,
            domain=data.pop("domain", None),
            sampling=sampling,
            **data,
        )
