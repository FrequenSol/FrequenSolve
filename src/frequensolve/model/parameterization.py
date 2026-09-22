"""Sparse material-property control parameterizations."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from frequensolve.model.property import Property
from frequensolve.units import is_quantity, unit_expression
from frequensolve.util.mixins import ExportContext

__all__ = [
    "BSplineControl",
    "CONTROL_TRANSFORMS",
    "HatControl",
    "MeshControl",
    "MeshPropertySpace",
    "ParameterizedProperty",
    "TensorHatControl",
    "control_from_fs",
]

#: Control compositions accepted by ``ParameterizedProperty.transform``.
#: ``identity`` evaluates ``r + c``, ``log`` evaluates ``r exp(c)``,
#: ``inverse`` evaluates ``1 / (1/r + c)`` and ``logit`` evaluates
#: ``sigmoid(logit(r) + c)`` for a dimensionless reference in ``(0, 1)``.
CONTROL_TRANSFORMS: Tuple[str, ...] = ("identity", "log", "inverse", "logit")


def _coefficient_vector(values: ArrayLike, *, minimum: int) -> np.ndarray:
    """Return a finite one-dimensional float64 coefficient vector."""

    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError("material control coefficients must be real-valued")
    array = np.asarray(array, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum:
        raise ValueError(
            f"control coefficients must be a 1-D array with at least {minimum} values"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("control coefficients must be finite")
    return np.array(array, copy=True)


@dataclass
class HatControl:
    """Uniform one-dimensional nodal grid with piecewise-linear hat functions."""

    axis: str
    spacing: float
    coefficients: ArrayLike
    origin: float = 0.0
    coordinate_system: str = "global"
    units: Optional[str] = None
    kind: str = field(default="hat", init=False)

    def __post_init__(self) -> None:
        self.axis = str(self.axis).strip()
        self.coordinate_system = str(self.coordinate_system).strip()
        self.units = None if self.units is None else str(self.units).strip()
        self.origin = float(self.origin)
        self.spacing = float(self.spacing)
        self.coefficients = _coefficient_vector(self.coefficients, minimum=2)
        if not self.axis:
            raise ValueError("hat control requires a coordinate axis")
        if not self.coordinate_system:
            raise ValueError("hat control requires a coordinate system")
        if self.units == "":
            raise ValueError("hat control units cannot be empty")
        if not np.isfinite(self.origin):
            raise ValueError("hat control origin must be finite")
        if not np.isfinite(self.spacing) or self.spacing <= 0.0:
            raise ValueError("hat control spacing must be finite and positive")

    @property
    def size(self) -> int:
        """Return the number of ordered nodal coefficients."""

        return int(np.asarray(self.coefficients).size)

    @property
    def coordinates(self) -> np.ndarray:
        """Return the physical coordinate of each ordered coefficient."""

        return self.origin + self.spacing * np.arange(self.size, dtype=np.float64)

    def with_coefficients(self, coefficients: ArrayLike) -> "HatControl":
        """Return a copy with a replacement coefficient vector."""

        return replace(self, coefficients=coefficients)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this control using the Sauce material contract."""

        payload = {
            "kind": self.kind,
            "coordinate_system": self.coordinate_system,
            "axis": self.axis,
            "origin": self.origin,
            "spacing": self.spacing,
            "coefficients": np.asarray(self.coefficients).tolist(),
        }
        if self.units is not None:
            payload["units"] = self.units
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "HatControl":
        """Deserialize a uniform hat control."""

        return cls(
            axis=data["axis"],
            spacing=data["spacing"],
            coefficients=data["coefficients"],
            origin=data.get("origin", 0.0),
            coordinate_system=data.get("coordinate_system", "global"),
            units=data.get("units"),
        )


@dataclass
class BSplineControl:
    """Arbitrary-degree one-dimensional B-spline property control."""

    axis: str
    knots: ArrayLike
    coefficients: ArrayLike
    degree: int = 3
    coordinate_system: str = "global"
    units: Optional[str] = None
    kind: str = field(default="bspline", init=False)

    def __post_init__(self) -> None:
        self.axis = str(self.axis).strip()
        self.coordinate_system = str(self.coordinate_system).strip()
        self.units = None if self.units is None else str(self.units).strip()
        self.degree = int(self.degree)
        self.knots = np.asarray(self.knots, dtype=np.float64)
        if self.knots.ndim != 1 or self.knots.size < 2:
            raise ValueError("B-spline knots must be a one-dimensional array")
        if not np.all(np.isfinite(self.knots)):
            raise ValueError("B-spline knots must be finite")
        if np.any(np.diff(self.knots) < 0.0):
            raise ValueError("B-spline knots must be nondecreasing")
        if self.degree < 0:
            raise ValueError("B-spline degree must be nonnegative")
        ncontrol = int(self.knots.size - self.degree - 1)
        if ncontrol < self.degree + 1:
            raise ValueError(
                "B-spline knot vector requires at least degree + 1 controls"
            )
        unique, multiplicity = np.unique(self.knots, return_counts=True)
        if np.any(multiplicity > self.degree + 1):
            knot = unique[np.flatnonzero(multiplicity > self.degree + 1)[0]]
            raise ValueError(f"B-spline knot multiplicity exceeds degree + 1 at {knot}")
        if not self.knots[self.degree] < self.knots[ncontrol]:
            raise ValueError("B-spline active knot interval must have positive length")
        self.coefficients = _coefficient_vector(self.coefficients, minimum=1)
        if self.coefficients.size != ncontrol:
            raise ValueError(
                "B-spline coefficient count must equal len(knots) - degree - 1"
            )
        if not self.axis:
            raise ValueError("B-spline control requires a coordinate axis")
        if not self.coordinate_system:
            raise ValueError("B-spline control requires a coordinate system")
        if self.units == "":
            raise ValueError("B-spline control units cannot be empty")
        self.knots = np.array(self.knots, copy=True)

    @property
    def size(self) -> int:
        """Return the number of ordered spline coefficients."""

        return int(np.asarray(self.coefficients).size)

    @property
    def coordinates(self) -> np.ndarray:
        """Return Greville coordinates for the ordered coefficients."""

        knots = np.asarray(self.knots)
        if self.degree == 0:
            return 0.5 * (knots[:-1] + knots[1:])
        return np.array(
            [np.mean(knots[i + 1 : i + self.degree + 1]) for i in range(self.size)],
            dtype=np.float64,
        )

    def with_coefficients(self, coefficients: ArrayLike) -> "BSplineControl":
        """Return a copy with a replacement coefficient vector."""

        return replace(self, coefficients=coefficients)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this control using the Sauce material contract."""

        payload = {
            "kind": self.kind,
            "coordinate_system": self.coordinate_system,
            "axis": self.axis,
            "degree": self.degree,
            "knots": np.asarray(self.knots).tolist(),
            "coefficients": np.asarray(self.coefficients).tolist(),
        }
        if self.units is not None:
            payload["units"] = self.units
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "BSplineControl":
        """Deserialize a B-spline control."""

        return cls(
            axis=data["axis"],
            knots=data["knots"],
            coefficients=data["coefficients"],
            degree=data.get("degree", 3),
            coordinate_system=data.get("coordinate_system", "global"),
            units=data.get("units"),
        )


def _lattice_vector(
    values: Any, *, name: str, count: int, dtype: Any = np.float64
) -> np.ndarray:
    """Return a finite one-dimensional per-axis lattice vector."""

    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or array.size != count:
        raise ValueError(
            f"tensor hat {name} must list one value per lattice axis ({count})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"tensor hat {name} must be finite")
    return np.array(array, copy=True)


@dataclass
class TensorHatControl:
    """Tensor product of two or three uniform nodal hat axes.

    The lattice is independent of the simulation mesh. Coefficients are stored
    flat in the Sauce contract ordering: the first listed axis varies fastest
    (``i = i_0 + n_0 * (i_1 + n_1 * i_2)``). ``coefficients`` may be supplied
    either as that flat vector or as an array whose shape equals ``shape``; an
    array is flattened with the same first-axis-fastest convention.

    Updates vanish outside the closed lattice box, so zero boundary
    coefficients give a continuous localized perturbation.

    Args:
        axes: Distinct axis names in coefficient storage order.
        shape: Number of nodes per axis (at least two each).
        origin: First node coordinate per axis.
        spacing: Positive node spacing per axis.
        coefficients: Flat coefficient vector of length ``prod(shape)`` or an
            array with shape ``shape``.
        coordinate_system: Coordinate system the axes belong to.
        units: Optional length units shared by all axes; defaults to the
            model coordinate units.
    """

    axes: Sequence[str]
    shape: Sequence[int]
    origin: Sequence[float]
    spacing: Sequence[float]
    coefficients: ArrayLike
    coordinate_system: str = "global"
    units: Optional[str] = None
    kind: str = field(default="tensor_hat", init=False)

    def __post_init__(self) -> None:
        if isinstance(self.axes, str):
            raise ValueError("tensor hat axes must be a sequence of axis names")
        self.axes = tuple(str(axis).strip() for axis in self.axes)
        self.coordinate_system = str(self.coordinate_system).strip()
        self.units = None if self.units is None else str(self.units).strip()
        ndim = len(self.axes)
        if ndim not in (2, 3):
            raise ValueError("tensor hat control requires two or three axes")
        if any(not axis for axis in self.axes):
            raise ValueError("tensor hat axis names cannot be empty")
        if len(set(self.axes)) != ndim:
            raise ValueError("tensor hat axes must be distinct")
        if not self.coordinate_system:
            raise ValueError("tensor hat control requires a coordinate system")
        if self.units == "":
            raise ValueError("tensor hat control units cannot be empty")
        shape = np.asarray(self.shape)
        if shape.ndim != 1 or shape.size != ndim:
            raise ValueError(
                f"tensor hat shape must list one value per lattice axis ({ndim})"
            )
        if not np.issubdtype(shape.dtype, np.integer) or np.any(shape < 2):
            raise ValueError("tensor hat shape must be integers of at least 2")
        self.shape = tuple(int(n) for n in shape)
        self.origin = _lattice_vector(self.origin, name="origin", count=ndim)
        self.spacing = _lattice_vector(self.spacing, name="spacing", count=ndim)
        if np.any(self.spacing <= 0.0):
            raise ValueError("tensor hat spacing must be positive")
        self.coefficients = self._flatten(self.coefficients)

    def _flatten(self, values: ArrayLike) -> np.ndarray:
        array = np.asarray(values)
        if np.iscomplexobj(array):
            raise ValueError("material control coefficients must be real-valued")
        if array.ndim == len(self.shape) and array.ndim > 1:
            if tuple(array.shape) != self.shape:
                raise ValueError(
                    f"tensor hat coefficient array shape {tuple(array.shape)} "
                    f"does not match lattice shape {self.shape}"
                )
            array = np.asarray(array, dtype=np.float64).reshape(-1, order="F")
        expected = int(np.prod(self.shape))
        flat = _coefficient_vector(array, minimum=4)
        if flat.size != expected:
            raise ValueError(
                f"tensor hat coefficient count {flat.size} must equal "
                f"product(shape) = {expected}"
            )
        return flat

    @property
    def ndim(self) -> int:
        """Return the number of lattice axes."""

        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the number of ordered lattice coefficients."""

        return int(np.asarray(self.coefficients).size)

    @property
    def axis_coordinates(self) -> List[np.ndarray]:
        """Return the node coordinates along each lattice axis."""

        return [
            float(self.origin[k]) + float(self.spacing[k]) * np.arange(n, dtype=float)
            for k, n in enumerate(self.shape)
        ]

    @property
    def coordinates(self) -> np.ndarray:
        """Return an ``(size, ndim)`` array of node coordinates.

        Rows follow the flat coefficient ordering, so ``coordinates[i]`` is the
        node owning ``coefficients[i]`` and the first axis varies fastest.
        """

        grids = np.meshgrid(*self.axis_coordinates, indexing="ij")
        return np.stack([grid.reshape(-1, order="F") for grid in grids], axis=1)

    @property
    def grid(self) -> np.ndarray:
        """Return the coefficients as an array with shape ``shape``.

        ``grid[i_0, i_1, ...]`` is the coefficient at node ``(i_0, i_1, ...)``.
        """

        return np.asarray(self.coefficients).reshape(self.shape, order="F")

    def with_coefficients(self, coefficients: ArrayLike) -> "TensorHatControl":
        """Return a copy with a replacement flat or lattice-shaped vector."""

        return replace(self, coefficients=coefficients)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this control using the Sauce material contract."""

        payload: Dict[str, Any] = {
            "kind": self.kind,
            "coordinate_system": self.coordinate_system,
            "axes": list(self.axes),
            "shape": list(self.shape),
            "origin": np.asarray(self.origin).tolist(),
            "spacing": np.asarray(self.spacing).tolist(),
            "coefficients": np.asarray(self.coefficients).tolist(),
        }
        if self.units is not None:
            payload["units"] = self.units
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "TensorHatControl":
        """Deserialize a tensor hat control."""

        return cls(
            axes=data["axes"],
            shape=data["shape"],
            origin=data["origin"],
            spacing=data["spacing"],
            coefficients=data["coefficients"],
            coordinate_system=data.get("coordinate_system", "global"),
            units=data.get("units"),
        )


@dataclass
class MeshControl:
    """First-order nodal controls on a named frozen mesh property space.

    The control basis is generated by Sauce from the initial mesh the first
    time the named space is used and frozen in the space's artifact (see
    :class:`MeshPropertySpace` and ``Model.property_spaces``). Coefficients
    start at zero and are carried in named control checkpoints, never inline,
    so this object has no ``coefficients``; ``size`` and ``coordinates`` are
    ``None`` until Sauce has written the artifact.

    Args:
        space: Name of the ``Model/property_spaces`` entry to use.
    """

    space: str
    kind: str = field(default="mesh", init=False)

    def __post_init__(self) -> None:
        self.space = str(self.space).strip()
        if not self.space:
            raise ValueError("mesh control requires a property-space name")

    @property
    def size(self) -> None:
        """Return ``None``; the coefficient count is fixed by the artifact."""

        return None

    @property
    def coordinates(self) -> None:
        """Return ``None``; nodal coordinates live in the space artifact."""

        return None

    @property
    def coefficients(self) -> None:
        """Return ``None``; mesh coefficients are read from checkpoints."""

        return None

    def with_coefficients(self, coefficients: ArrayLike) -> "MeshControl":
        """Reject inline coefficients, which the contract does not accept."""

        raise ValueError(
            "mesh controls do not carry inline coefficients; use a control "
            "checkpoint written against the property-space artifact"
        )

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this control using the Sauce material contract."""

        return {"kind": self.kind, "space": self.space}

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "MeshControl":
        """Deserialize a mesh control."""

        return cls(space=data["space"])


def _frequency_hz(value: Any) -> float:
    """Return a positive frequency in hertz from a float, quantity or mapping."""

    if isinstance(value, Mapping):
        payload = dict(value)
        unknown = set(payload) - {"value", "units"}
        if unknown or "value" not in payload:
            raise ValueError(
                "property-space frequency mapping accepts only value and units"
            )
        units = payload.get("units")
        value = payload["value"]
        if units is not None:
            from frequensolve.units import Q_

            value = Q_(value, unit_expression(units))
    if is_quantity(value):
        try:
            value = value.to("Hz").magnitude
        except Exception as exc:
            raise ValueError(
                "property-space frequency units must be compatible with hertz"
            ) from exc
    frequency = float(value)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("property-space frequency must be finite and positive")
    return frequency


@dataclass
class MeshPropertySpace:
    """Named material-control topology declared in ``Model/property_spaces``.

    The space sizes a first-order nodal hierarchy from the initial geometry
    and reference material state at ``frequency`` hertz with ``epw`` elements
    per wavelength. Sauce freezes the topology and coefficient identifiers in
    ``artifact`` on first use; later runs must reuse the same artifact or name
    a new space. Solution frequency and solver refinement do not size it.

    Args:
        artifact: HDF5 artifact path (``.h5``) that freezes the space.
        frequency: Control sizing frequency. Bare values are hertz; Pint
            quantities and ``{"value", "units"}`` mappings are converted.
        epw: Elements per wavelength, scalar or one value per spatial
            dimension.
    """

    artifact: str
    frequency: Any
    epw: Any

    def __post_init__(self) -> None:
        self.artifact = str(self.artifact).strip()
        if not self.artifact.endswith(".h5"):
            raise ValueError("property-space artifact must be an .h5 path")
        self.frequency = _frequency_hz(self.frequency)
        epw = np.asarray(self.epw, dtype=np.float64)
        if epw.ndim == 0:
            self.epw = float(epw)
            values = np.array([self.epw])
        elif epw.ndim == 1 and epw.size in (2, 3):
            self.epw = [float(v) for v in epw]
            values = epw
        else:
            raise ValueError(
                "property-space epw must be a scalar or one value per dimension"
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("property-space epw must be finite and positive")

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this space using the Sauce material contract."""

        return {
            "artifact": self.artifact,
            "frequency": self.frequency,
            "epw": self.epw,
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "MeshPropertySpace":
        """Deserialize a property-space declaration."""

        return cls(
            artifact=data["artifact"],
            frequency=data["frequency"],
            epw=data["epw"],
        )

    @staticmethod
    def mapping_from_fs(
        data: Optional[Mapping[str, Any]],
    ) -> Dict[str, "MeshPropertySpace"]:
        """Coerce a ``property_spaces`` mapping into named spaces."""

        spaces: Dict[str, MeshPropertySpace] = {}
        for name, value in dict(data or {}).items():
            key = str(name).strip()
            if not key or len(key) > 64:
                raise ValueError("property-space names must be 1 to 64 characters long")
            spaces[key] = (
                value
                if isinstance(value, MeshPropertySpace)
                else MeshPropertySpace.from_fs(value)
            )
        return spaces


PropertyControl = Union[HatControl, BSplineControl, TensorHatControl, MeshControl]

_CONTROL_KINDS = {
    "hat": HatControl,
    "bspline": BSplineControl,
    "tensor_hat": TensorHatControl,
    "mesh": MeshControl,
}


def control_from_fs(data: Mapping[str, Any]) -> PropertyControl:
    """Deserialize one supported sparse property-control map."""

    kind = str(data.get("kind", "bspline")).lower()
    control_class = _CONTROL_KINDS.get(kind)
    if control_class is None:
        raise ValueError(f"unsupported property-control kind {kind!r}")
    return control_class.from_fs(data)


def _is_dimensionless(units: Optional[str]) -> bool:
    if units is None or units == "":
        return True
    try:
        from frequensolve.units import ureg

        return bool(ureg(units).dimensionless)
    except Exception:
        return False


def _constant_reference_value(reference: Property) -> Optional[float]:
    """Return the scalar value of a constant reference, else ``None``."""

    try:
        if not reference.is_constant:
            return None
        low, high = reference.extrema
        low = float(np.asarray(low))
        high = float(np.asarray(high))
    except Exception:
        return None
    return low if low == high else None


def _validate_transform(transform: str, reference: Property, units: Any) -> str:
    transform = str(transform).lower().strip()
    if transform not in CONTROL_TRANSFORMS:
        choices = ", ".join(CONTROL_TRANSFORMS)
        raise ValueError(f"parameterized property transform must be one of {choices}")
    value = _constant_reference_value(reference)
    if transform == "inverse" and value is not None and value <= 0.0:
        raise ValueError(
            "inverse transform requires a strictly positive reference property"
        )
    if transform == "logit":
        if not _is_dimensionless(units):
            raise ValueError(
                "logit transform requires a dimensionless reference property"
            )
        if value is not None and not 0.0 < value < 1.0:
            raise ValueError(
                "logit transform requires a reference strictly between 0 and 1"
            )
    return transform


class ParameterizedProperty(Property):
    """Decorate an ordinary material property with an ordered control block."""

    def __init__(
        self,
        reference: Any,
        *,
        id: str,
        control: Union[PropertyControl, Mapping[str, Any]],
        transform: str = "identity",
        **extra: Any,
    ):
        reference = Property.from_value(reference)
        wrapper_units = extra.pop("units", None)
        reference_units = reference.units
        if wrapper_units is None:
            wrapper_units = reference_units
        elif reference_units is not None:
            wrapper_units = unit_expression(wrapper_units)
            if wrapper_units != reference_units:
                raise ValueError(
                    "parameterized property units disagree with reference units: "
                    f"{wrapper_units!r} != {reference_units!r}"
                )
        super().__init__(0.0, units=wrapper_units)
        block_id = str(id).strip()
        if not block_id:
            raise ValueError("parameterized property requires a non-empty block id")
        if "/" in block_id or block_id in {".", ".."}:
            raise ValueError("parameterized property block id is not HDF5-safe")
        self.reference = reference
        self.id = block_id
        self.control = (
            control_from_fs(control) if isinstance(control, Mapping) else control
        )
        if not isinstance(self.control, tuple(_CONTROL_KINDS.values())):
            raise TypeError("parameterized property requires a supported control map")
        self.transform = _validate_transform(transform, reference, wrapper_units)
        self.extra = dict(extra)

    @property
    def is_constant(self) -> bool:
        """Return false because the control map is coordinate dependent."""

        return False

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        """Return the ordered coefficient vector for the whole block.

        Mesh controls carry no inline coefficients and return ``None``.
        """

        coefficients = self.control.coefficients
        return None if coefficients is None else np.asarray(coefficients)

    def with_coefficients(self, coefficients: ArrayLike) -> "ParameterizedProperty":
        """Return a deep copy with replacement control coefficients."""

        return ParameterizedProperty(
            copy.deepcopy(self.reference),
            id=self.id,
            control=self.control.with_coefficients(coefficients),
            transform=self.transform,
            units=self.units,
            **copy.deepcopy(self.extra),
        )

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        file: Any = None,
        dataset: Optional[str] = None,
        preserve_inline_coordinates: bool = False,
    ) -> Dict[str, Any]:
        """Serialize the decorator and its ordinary reference property."""

        reference_dataset = f"{dataset}/reference" if dataset else None
        reference = self.reference.to_fs(
            ctx=ctx,
            file=file,
            dataset=reference_dataset,
            preserve_inline_coordinates=preserve_inline_coordinates,
        )
        reference.pop("units", None)
        payload: Dict[str, Any] = {
            "parameterized": {
                "id": self.id,
                "reference": reference,
                "transform": self.transform,
                "control": self.control.to_fs(),
            }
        }
        if self.units is not None:
            payload["units"] = self.units
        payload.update(copy.deepcopy(self.extra))
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ParameterizedProperty":
        """Deserialize a parameterized-property wrapper."""

        payload = dict(data)
        parameterized = dict(payload.pop("parameterized"))
        return cls(
            parameterized["reference"],
            id=parameterized["id"],
            transform=parameterized.get("transform", "identity"),
            control=parameterized["control"],
            **payload,
        )
