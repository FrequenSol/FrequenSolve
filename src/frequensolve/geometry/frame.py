"""Coordinate-system authoring objects for FrequenSolve inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np

from frequensolve.units import (
    is_quantity,
    quantity_to_fs,
    unit_expression,
    value_and_units_to_fs,
)
from frequensolve.util.mixins import merge_extra

__all__ = [
    "Axis",
    "CoordinateSystem",
    "CoordinateValue",
    "Direction",
    "SurfaceCoordinateSystem",
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
class Axis:
    name: str
    direction: str
    positive: Optional[str] = None
    origin: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(cls, data: Union["Axis", Mapping[str, Any]]) -> "Axis":
        if isinstance(data, Axis):
            return data
        payload = dict(data)
        return cls(
            name=payload.pop("name"),
            direction=payload.pop("direction"),
            positive=payload.pop("positive", None),
            origin=payload.pop("origin", None),
            extra=payload,
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "direction": self.direction,
        }
        if self.positive is not None:
            payload["positive"] = self.positive
        if self.origin is not None:
            payload["origin"] = (
                dict(self.origin)
                if isinstance(self.origin, Mapping)
                else value_and_units_to_fs(self.origin)
            )
        return merge_extra(payload, self.extra, "Axis")


@dataclass
class CoordinateSystem:
    type: str = "cartesian"
    name: Optional[str] = None
    origin: Optional[Union[CoordinateValue, Any]] = None
    axis_alignment: Optional[Dict[str, str]] = None
    axes: Optional[List[Axis]] = None
    inherit_axes: Optional[bool] = None
    surface_ref: Optional[Union[str, int]] = None
    normal: Optional[str] = None
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
    def surface(
        cls,
        name: str,
        surface: Union[str, int],
        *,
        normal: str = "up",
        offset: Optional[Any] = None,
        offset_units: Optional[Any] = None,
        **kwargs,
    ) -> "CoordinateSystem":
        """Coordinate system tied to a named model surface.

        ``normal`` is retained for compatibility and creates a surface-relative
        ``z`` axis when no explicit axes are supplied.
        """
        return SurfaceCoordinateSystem(
            name=name,
            surface=surface,
            normal=normal,
            offset=offset,
            offset_units=offset_units,
            **kwargs,
        )

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "CoordinateSystem":
        payload = dict(data)
        class_name = payload.pop("_type", None)
        if class_name == "SurfaceCoordinateSystem":
            return SurfaceCoordinateSystem.from_fs(payload)
        if class_name is not None:
            raise ValueError(f"Unknown coordinate system type: {class_name}")
        if payload.get("type") == "surface":
            return SurfaceCoordinateSystem.from_fs(payload)
        origin = CoordinateValue.from_fs(payload.pop("origin", None))
        axes = payload.pop("axes", None)
        return cls(
            type=payload.pop("type"),
            name=payload.pop("name", None),
            origin=origin,
            axis_alignment=payload.pop("axis_alignment", None),
            axes=[Axis.from_fs(axis) for axis in axes] if axes is not None else None,
            inherit_axes=payload.pop("inherit_axes", None),
            surface_ref=payload.pop("surface", None),
            normal=payload.pop("normal", payload.pop("normal_direction", None)),
            earth_radius=payload.pop("earth_radius", None),
            ndim=payload.pop("ndim", None),
            fixed_axis=payload.pop("fixed_axis", None),
            fixed_value=payload.pop("fixed_value", None),
            extra=payload,
        )

    def _payload(self, ctx=None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": self.type}
        if self.name is not None:
            payload["name"] = self.name
        if self.origin is not None:
            payload["origin"] = coordinate_value_to_fs(self.origin)
        if self.axis_alignment is not None:
            payload["axis_alignment"] = self.axis_alignment
        if self.axes is not None:
            payload["axes"] = [axis.to_fs(ctx) for axis in self.axes]
        if self.inherit_axes is not None:
            payload["inherit_axes"] = self.inherit_axes
        if self.surface_ref is not None:
            payload["surface"] = self.surface_ref
        if self.earth_radius is not None:
            payload["earth_radius"] = value_and_units_to_fs(self.earth_radius)
        if self.ndim is not None:
            payload["ndim"] = self.ndim
        if self.fixed_axis is not None:
            payload["fixed_axis"] = self.fixed_axis
        if self.fixed_value is not None:
            payload["fixed_value"] = (
                dict(self.fixed_value)
                if isinstance(self.fixed_value, Mapping)
                else value_and_units_to_fs(self.fixed_value)
            )
        return payload

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        payload = self._payload(ctx)
        return merge_extra(payload, self.extra, "CoordinateSystem")

    def with_offset(
        self,
        name: str,
        offset: Any,
        *,
        units: Optional[Any] = None,
        normal: Optional[str] = None,
    ) -> "CoordinateSystem":
        """Return a copy with a fixed signed offset along the surface normal axis."""

        payload = self.to_fs()
        payload["name"] = name
        payload.setdefault("ndim", 2)
        if normal is not None:
            payload["normal"] = normal
        payload["fixed_axis"] = "z"
        payload["fixed_value"] = value_and_units_to_fs(offset, units)
        return self.from_fs(payload)

    def above(
        self,
        values: Any,
        distance: Optional[Any] = None,
        *,
        units: Optional[Any] = None,
    ) -> CoordinateValue:
        """Coordinate-aware points above the model surface."""

        sign = self._above_sign()
        if distance is None:
            return self._signed_surface_points(values, sign=sign, units=units)
        return self.on_surface(values, units=units, offset=sign * distance)

    def below(
        self,
        values: Any,
        distance: Optional[Any] = None,
        *,
        units: Optional[Any] = None,
    ) -> CoordinateValue:
        """Coordinate-aware points below the model surface."""

        sign = self._below_sign()
        if distance is None:
            return self._signed_surface_points(values, sign=sign, units=units)
        return self.on_surface(values, units=units, offset=sign * distance)

    def _above_sign(self) -> int:
        return -1 if str(self.normal or "up").strip().lower() == "down" else 1

    def _below_sign(self) -> int:
        return 1 if str(self.normal or "up").strip().lower() == "down" else -1

    def _signed_surface_points(
        self, values: Any, *, sign: int, units: Optional[Any] = None
    ) -> CoordinateValue:
        coord_values, coord_units = _split_quantity(values, units)
        coords = np.asarray(_surface_coordinate_values(coord_values), dtype=float)
        coords[:, -1] = sign * np.abs(coords[:, -1])
        return CoordinateValue(coords.tolist(), units=coord_units, system=self.name)

    def points(
        self,
        values: Any,
        *,
        units: Optional[Any] = None,
        offset: Optional[Any] = None,
    ) -> CoordinateValue:
        """Coordinate-aware point values in this system."""

        if self.type == "surface":
            return self.on_surface(values, units=units, offset=offset)
        return CoordinateValue(values, units=units, system=self.name)

    def on_surface(
        self,
        lateral: Any,
        *,
        units: Optional[Any] = None,
        offset: Optional[Any] = None,
    ) -> CoordinateValue:
        """Point values on or offset from this surface coordinate system.

        ``lateral`` can be a sequence of x positions, an explicit ``(n, 2)``
        array of ``[x, z]`` coordinates for 2D models, or an explicit
        ``(n, 3)`` array of ``[x, y, z]`` coordinates for 3D models. Passing
        ``offset`` appends a constant or per-point normal offset to one-column
        2D laterals or two-column 3D laterals.
        """

        lateral_values, lateral_units = _split_quantity(lateral, units)
        offset_values, offset_units = _split_quantity(offset, lateral_units)
        if offset_units is not None and lateral_units is None:
            lateral_units = offset_units
        values = _surface_coordinate_values(lateral_values, offset=offset_values)
        return CoordinateValue(values, units=lateral_units, system=self.name)

    on = on_surface


class SurfaceCoordinateSystem(CoordinateSystem):
    """Coordinate system whose selected axes are re-datumed to a model surface."""

    def __init__(
        self,
        name: str,
        surface: Union[str, int],
        *,
        axes: Optional[List[Union[Axis, Mapping[str, Any]]]] = None,
        normal: str = "up",
        inherit_axes: bool = True,
        offset: Optional[Any] = None,
        offset_units: Optional[Any] = None,
        **kwargs,
    ) -> None:
        if axes is None:
            axes = [Axis("z", direction="z", positive=normal)]
        axis_list = [Axis.from_fs(axis) for axis in axes]
        if offset is not None:
            kwargs.setdefault("ndim", 2)
            kwargs.setdefault("fixed_axis", axis_list[-1].name if axis_list else "z")
            kwargs.setdefault(
                "fixed_value", value_and_units_to_fs(offset, offset_units)
            )
        super().__init__(
            type="surface",
            name=name,
            surface_ref=surface,
            axes=axis_list,
            inherit_axes=inherit_axes,
            normal=normal,
            **kwargs,
        )

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SurfaceCoordinateSystem":
        payload = dict(data)
        payload.pop("_type", None)
        payload.pop("type", None)
        axes = payload.pop("axes", None)
        surface = payload.pop("surface", payload.pop("surface_ref", None))
        if surface is None:
            raise ValueError("SurfaceCoordinateSystem requires surface")
        normal = payload.pop("normal", payload.pop("normal_direction", "up"))
        inherit_axes = payload.pop("inherit_axes", True)
        return cls(
            name=payload.pop("name"),
            surface=surface,
            axes=axes,
            normal=normal,
            inherit_axes=inherit_axes,
            origin=CoordinateValue.from_fs(payload.pop("origin", None)),
            axis_alignment=payload.pop("axis_alignment", None),
            earth_radius=payload.pop("earth_radius", None),
            ndim=payload.pop("ndim", None),
            fixed_axis=payload.pop("fixed_axis", None),
            fixed_value=payload.pop("fixed_value", None),
            extra=payload,
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        payload = self._payload(ctx)
        payload.pop("type", None)
        payload["_type"] = "SurfaceCoordinateSystem"
        return merge_extra(payload, self.extra, "SurfaceCoordinateSystem")


def _split_quantity(
    value: Any, units: Optional[Any] = None
) -> tuple[Any, Optional[Any]]:
    if is_quantity(value):
        target_units = units or value.units
        return value.to(target_units).magnitude, target_units
    return value, units


def _surface_coordinate_values(lateral: Any, offset: Optional[Any] = None) -> Any:
    values = np.asarray(lateral, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] not in (1, 2, 3):
        raise ValueError(
            "surface coordinates must have shape (n,), (n, 1), (n, 2), or (n, 3)"
        )

    n = values.shape[0]
    if offset is None:
        if values.shape[1] > 1:
            return values.tolist()
        return np.column_stack([values[:, 0], np.zeros(n)]).tolist()

    if values.shape[1] == 3:
        raise ValueError(
            "surface offset cannot be supplied with explicit 3D coordinates"
        )
    offset_values = np.asarray(offset, dtype=float)
    if offset_values.ndim == 0:
        offset_values = np.full(n, float(offset_values))
    if offset_values.shape != (n,):
        raise ValueError("surface offset must be scalar or one value per point")
    return np.column_stack([values, offset_values]).tolist()


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
