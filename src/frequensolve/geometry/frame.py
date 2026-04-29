"""Coordinate-system authoring objects for FrequenSolve inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

from frequensolve.units import (
    is_quantity,
    quantity_to_fs,
    unit_expression,
    value_and_units_to_fs,
)
from frequensolve.util.mixins import merge_extra

__all__ = [
    "CoordinateSystem",
    "CoordinateValue",
    "Direction",
    "coordinate_value_to_fs",
    "direction_to_fs",
]


@dataclass
class CoordinateValue:
    value: Any
    units: Optional[Any] = None
    system: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(cls, data: Any) -> Union["CoordinateValue", Any]:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        return cls(
            value=payload.pop("value"),
            units=payload.pop("units", None),
            system=payload.pop("system", None),
            extra=payload,
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        payload = value_and_units_to_fs(self.value, self.units)
        if not isinstance(payload, dict):
            payload = {"value": payload}
        if self.system is not None:
            payload["system"] = self.system
        return merge_extra(payload, self.extra, "CoordinateValue")


@dataclass
class Direction:
    type: str = "vector"
    system: Optional[str] = None
    axis: Optional[str] = None
    value: Optional[Any] = None
    components: Optional[List[str]] = None
    units: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def axis_direction(cls, axis: str, system: Optional[str] = None) -> "Direction":
        return cls(type="coordinate_axis", axis=axis, system=system)

    @classmethod
    def vector(
        cls, value: Any, units: Optional[Any] = None, system: Optional[str] = None
    ) -> "Direction":
        return cls(type="vector", value=value, units=units, system=system)

    @classmethod
    def basis(cls, components: List[str], system: Optional[str] = None) -> "Direction":
        return cls(type="coordinate_basis", components=components, system=system)

    @classmethod
    def from_fs(cls, data: Any) -> Union["Direction", Any]:
        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        return cls(
            type=payload.pop("type", "vector"),
            system=payload.pop("system", None),
            axis=payload.pop("axis", None),
            value=payload.pop("value", None),
            components=payload.pop("components", None),
            units=payload.pop("units", None),
            extra=payload,
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": self.type}
        if self.system is not None:
            payload["system"] = self.system
        if self.axis is not None:
            payload["axis"] = self.axis
        if self.value is not None:
            value = value_and_units_to_fs(self.value, self.units)
            if isinstance(value, dict):
                payload.update(value)
            else:
                payload["value"] = value
        elif self.units is not None:
            payload["units"] = unit_expression(self.units)
        if self.components is not None:
            payload["components"] = self.components
        return merge_extra(payload, self.extra, "Direction")


@dataclass
class CoordinateSystem:
    type: str = "cartesian"
    name: Optional[str] = None
    origin: Optional[Union[CoordinateValue, Any]] = None
    axis_alignment: Optional[Dict[str, str]] = None
    earth_radius: Optional[Any] = None
    ndim: Optional[int] = None
    fixed_axis: Optional[str] = None
    fixed_value: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def cartesian(cls, name: str = "global", **kwargs) -> "CoordinateSystem":
        return cls(name=name, type="cartesian", **kwargs)

    @classmethod
    def cylindrical(cls, name: str, **kwargs) -> "CoordinateSystem":
        return cls(name=name, type="cylindrical", **kwargs)

    @classmethod
    def spherical(cls, name: str, **kwargs) -> "CoordinateSystem":
        return cls(name=name, type="spherical", **kwargs)

    @classmethod
    def geographic(cls, name: str = "geo", **kwargs) -> "CoordinateSystem":
        return cls(name=name, type="geographic", **kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "CoordinateSystem":
        payload = dict(data)
        origin = CoordinateValue.from_fs(payload.pop("origin", None))
        return cls(
            type=payload.pop("type"),
            name=payload.pop("name", None),
            origin=origin,
            axis_alignment=payload.pop("axis_alignment", None),
            earth_radius=payload.pop("earth_radius", None),
            ndim=payload.pop("ndim", None),
            fixed_axis=payload.pop("fixed_axis", None),
            fixed_value=payload.pop("fixed_value", None),
            extra=payload,
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": self.type}
        if self.name is not None:
            payload["name"] = self.name
        if self.origin is not None:
            payload["origin"] = coordinate_value_to_fs(self.origin)
        if self.axis_alignment is not None:
            payload["axis_alignment"] = self.axis_alignment
        if self.earth_radius is not None:
            payload["earth_radius"] = value_and_units_to_fs(self.earth_radius)
        if self.ndim is not None:
            payload["ndim"] = self.ndim
        if self.fixed_axis is not None:
            payload["fixed_axis"] = self.fixed_axis
        if self.fixed_value is not None:
            payload["fixed_value"] = value_and_units_to_fs(self.fixed_value)
        return merge_extra(payload, self.extra, "CoordinateSystem")


def coordinate_value_to_fs(value: Any) -> Any:
    if isinstance(value, CoordinateValue):
        return value.to_fs()
    if is_quantity(value):
        return quantity_to_fs(value)
    return value


def direction_to_fs(value: Any) -> Any:
    if isinstance(value, Direction):
        return value.to_fs()
    if is_quantity(value):
        return quantity_to_fs(value)
    return value
