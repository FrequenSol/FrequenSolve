"""Sparse material-property control parameterizations."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from frequensolve.model.property import Property
from frequensolve.units import unit_expression
from frequensolve.util.mixins import ExportContext

__all__ = [
    "BSplineControl",
    "HatControl",
    "ParameterizedProperty",
    "control_from_fs",
]


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


PropertyControl = Union[HatControl, BSplineControl]


def control_from_fs(data: Mapping[str, Any]) -> PropertyControl:
    """Deserialize one supported sparse property-control map."""

    kind = str(data.get("kind", "bspline")).lower()
    if kind == "hat":
        return HatControl.from_fs(data)
    if kind == "bspline":
        return BSplineControl.from_fs(data)
    raise ValueError(f"unsupported property-control kind {kind!r}")


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
        transform = str(transform).lower()
        if transform not in {"identity", "log"}:
            raise ValueError("parameterized property transform must be identity or log")
        self.reference = reference
        self.id = block_id
        self.control = (
            control_from_fs(control) if isinstance(control, Mapping) else control
        )
        if not isinstance(self.control, (HatControl, BSplineControl)):
            raise TypeError("parameterized property requires a supported control map")
        self.transform = transform
        self.extra = dict(extra)

    @property
    def is_constant(self) -> bool:
        """Return false because the control map is coordinate dependent."""

        return False

    @property
    def coefficients(self) -> np.ndarray:
        """Return the ordered coefficient vector for the whole block."""

        return np.asarray(self.control.coefficients)

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
