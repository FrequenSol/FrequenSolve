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
    """Coordinate values tagged with units and an optional coordinate system.

    Args:
        value: Scalar, array-like, mapping, or xarray object containing the
            coordinate values.
        units: Optional units for ``value``. Pint units and unit strings are
            accepted.
        system: Optional coordinate-system name that gives meaning to the
            coordinate values.
        extra: Additional solver-facing fields preserved during serialization.
    """

    value: Any
    units: Optional[Any] = None
    system: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(cls, data: Any) -> Union["CoordinateValue", Any]:
        """Deserialize a coordinate value payload.

        Args:
            data: Serialized coordinate mapping or a raw value.

        Returns:
            ``CoordinateValue`` when ``data`` is a mapping; otherwise the raw
            value unchanged.
        """

        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        return cls(
            value=payload.pop("value"),
            units=payload.pop("units", None),
            system=payload.pop("system", None),
            extra=payload,
        )

    def to_fs(self, ctx: Any = None) -> Dict[str, Any]:
        """Serialize the coordinate value for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible coordinate payload containing ``value`` and
            optional ``units``/``system`` fields.
        """

        payload = _coordinate_value_and_units_to_fs(self.value, self.units)
        if not isinstance(payload, dict):
            payload = {"value": payload}
        if self.system is not None:
            payload["system"] = self.system
        return merge_extra(payload, self.extra, "CoordinateValue")


@dataclass
class Direction:
    """Coordinate direction or basis vector used by coordinate-system metadata.

    Args:
        type: Direction representation, such as ``"vector"``,
            ``"coordinate_axis"``, or ``"coordinate_basis"``.
        system: Optional coordinate-system name that owns the direction.
        axis: Optional axis name for coordinate-axis directions.
        value: Optional vector value.
        components: Optional component names for basis directions.
        units: Optional vector units.
        extra: Additional solver-facing fields preserved during serialization.
    """

    type: str = "vector"
    system: Optional[str] = None
    axis: Optional[str] = None
    value: Optional[Any] = None
    components: Optional[List[str]] = None
    units: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def axis_direction(cls, axis: str, system: Optional[str] = None) -> "Direction":
        """Create a direction aligned with a named coordinate-system axis.

        Args:
            axis: Axis name.
            system: Optional coordinate-system name.

        Returns:
            Direction payload of type ``"coordinate_axis"``.
        """

        return cls(type="coordinate_axis", axis=axis, system=system)

    @classmethod
    def vector(
        cls, value: Any, units: Optional[Any] = None, system: Optional[str] = None
    ) -> "Direction":
        """Create an explicit vector direction.

        Args:
            value: Vector components.
            units: Optional component units.
            system: Optional coordinate-system name for the vector components.

        Returns:
            Direction payload of type ``"vector"``.
        """

        return cls(type="vector", value=value, units=units, system=system)

    @classmethod
    def basis(cls, components: List[str], system: Optional[str] = None) -> "Direction":
        """Create a coordinate-basis direction.

        Args:
            components: Component names that define the basis direction.
            system: Optional coordinate-system name.

        Returns:
            Direction payload of type ``"coordinate_basis"``.
        """

        return cls(type="coordinate_basis", components=components, system=system)

    @classmethod
    def from_fs(cls, data: Any) -> Union["Direction", Any]:
        """Deserialize a direction payload.

        Args:
            data: Serialized direction mapping or raw value.

        Returns:
            ``Direction`` when ``data`` is a mapping; otherwise the raw value
            unchanged.
        """

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

    def to_fs(self, ctx: Any = None) -> Dict[str, Any]:
        """Serialize the direction for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible direction payload.
        """

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
    """Named coordinate axis mapped to a physical direction.

    Args:
        name: Axis name exposed to xarray dimensions or user payloads.
        direction: Physical direction or solver-recognized direction that this
            axis follows, such as ``"x"``, ``"y"``, or ``"z"``.
        positive: Optional positive orientation, primarily used by
            surface-relative axes.
        origin: Optional axis origin. Unit-bearing values are serialized with
            ``value``/``units``.
        extra: Additional solver-facing fields preserved during serialization.
    """

    name: str
    direction: str
    positive: Optional[str] = None
    origin: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(cls, data: Union["Axis", Mapping[str, Any]]) -> "Axis":
        """Deserialize an axis payload.

        Args:
            data: Existing ``Axis`` or serialized axis mapping.

        Returns:
            ``Axis`` instance.
        """

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

    def to_fs(self, ctx: Any = None) -> Dict[str, Any]:
        """Serialize this axis for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible axis payload.
        """

        payload: Dict[str, Any] = {
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
    """Named coordinate system used by solver inputs and xarray-backed data.

    Args:
        type: Coordinate-system family, such as ``"cartesian"`` or
            ``"surface"``.
        name: Optional coordinate-system name used by properties, grids, and
            coordinate values.
        origin: Optional coordinate-system origin.
        axis_alignment: Optional mapping between physical directions and axis
            names.
        axes: Optional explicit axes exposed by this coordinate system.
        inherit_axes: Whether unspecified physical axes should be inherited
            from the global physical coordinates.
        surface_ref: Optional model surface name or index for surface-relative
            coordinate systems.
        normal: Surface-normal direction convention for surface systems.
        earth_radius: Optional Earth radius for geographic-style systems.
        ndim: Optional declared coordinate dimension.
        fixed_axis: Optional axis fixed to ``fixed_value``.
        fixed_value: Optional fixed coordinate value for ``fixed_axis``.
        extra: Additional solver-facing fields preserved during serialization.
    """

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
    def cartesian(cls, name: str = "global", **kwargs: Any) -> "CoordinateSystem":
        """Create a Cartesian coordinate system.

        Args:
            name: Coordinate-system name.
            **kwargs: Additional ``CoordinateSystem`` fields.

        Returns:
            Cartesian coordinate system.
        """

        return cls(name=name, type="cartesian", **kwargs)

    @classmethod
    def cylindrical(cls, name: str, **kwargs: Any) -> "CoordinateSystem":
        """Create a cylindrical coordinate system.

        Args:
            name: Coordinate-system name.
            **kwargs: Additional ``CoordinateSystem`` fields.

        Returns:
            Cylindrical coordinate system.
        """

        return cls(name=name, type="cylindrical", **kwargs)

    @classmethod
    def spherical(cls, name: str, **kwargs: Any) -> "CoordinateSystem":
        """Create a spherical coordinate system.

        Args:
            name: Coordinate-system name.
            **kwargs: Additional ``CoordinateSystem`` fields.

        Returns:
            Spherical coordinate system.
        """

        return cls(name=name, type="spherical", **kwargs)

    @classmethod
    def geographic(cls, name: str = "geo", **kwargs: Any) -> "CoordinateSystem":
        """Create a geographic coordinate system.

        Args:
            name: Coordinate-system name.
            **kwargs: Additional ``CoordinateSystem`` fields.

        Returns:
            Geographic coordinate system.
        """

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
        **kwargs: Any,
    ) -> "CoordinateSystem":
        """Coordinate system tied to a named model surface.

        ``normal`` is retained for compatibility and creates a surface-relative
        ``z`` axis when no explicit axes are supplied.

        Args:
            name: Coordinate-system name.
            surface: Surface name or one-based surface index.
            normal: Direction considered positive relative to the surface.
            offset: Optional fixed offset along the surface-normal axis.
            offset_units: Optional units for ``offset``.
            **kwargs: Additional coordinate-system fields.

        Returns:
            ``SurfaceCoordinateSystem`` instance.
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
        """Deserialize a coordinate-system payload.

        Args:
            data: Serialized coordinate-system mapping.

        Returns:
            ``CoordinateSystem`` or ``SurfaceCoordinateSystem`` depending on
            the payload type.

        Raises:
            ValueError: If the payload names an unknown coordinate-system
                subclass.
        """

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

    def _payload(self, ctx: Any = None) -> Dict[str, Any]:
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

    def to_fs(self, ctx: Any = None) -> Dict[str, Any]:
        """Serialize the coordinate system for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible coordinate-system payload.
        """

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
        """Return a copy with a fixed offset along the surface-normal axis.

        Args:
            name: Name for the returned coordinate system.
            offset: Fixed offset value.
            units: Optional units for ``offset``.
            normal: Optional replacement surface-normal convention.

        Returns:
            New coordinate system with ``fixed_axis`` and ``fixed_value`` set.
        """

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
        """Create coordinate-aware points above this surface system.

        Args:
            values: Lateral positions or explicit surface-system point values.
            distance: Optional signed distance from the surface. When omitted,
                the final coordinate column in ``values`` is interpreted as the
                offset magnitude.
            units: Optional units for values and distance.

        Returns:
            ``CoordinateValue`` tagged with this coordinate-system name.
        """

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
        """Create coordinate-aware points below this surface system.

        Args:
            values: Lateral positions or explicit surface-system point values.
            distance: Optional signed distance from the surface. When omitted,
                the final coordinate column in ``values`` is interpreted as the
                offset magnitude.
            units: Optional units for values and distance.

        Returns:
            ``CoordinateValue`` tagged with this coordinate-system name.
        """

        sign = self._below_sign()
        if distance is None:
            return self._signed_surface_points(values, sign=sign, units=units)
        return self.on_surface(values, units=units, offset=sign * distance)

    def points_grid(
        self,
        x: Any,
        y: Optional[Any] = None,
        *,
        units: Optional[Any] = None,
        above: Optional[Any] = None,
        below: Optional[Any] = None,
    ) -> CoordinateValue:
        """Create coordinate-aware points on a lateral tensor grid.

        Args:
            x: Lateral x-axis coordinates.
            y: Optional lateral y-axis coordinates for 3D carpets. When omitted,
                the grid is a 2D line of surface points.
            units: Optional coordinate units for the lateral axes.
            above: Optional distance above the surface.
            below: Optional distance below the surface.

        Returns:
            ``CoordinateValue`` tagged with this coordinate-system name.

        Raises:
            ValueError: If both ``above`` and ``below`` are supplied, or if the
                lateral axes cannot define a grid.
        """

        if above is not None and below is not None:
            raise ValueError("Specify only one of above or below")

        lateral, grid_units = _surface_points_grid_lateral(
            x,
            y,
            units=units,
        )
        if above is not None:
            return self.above(lateral, above, units=grid_units)
        if below is not None:
            return self.below(lateral, below, units=grid_units)
        if self.type == "surface":
            return self.on_surface(lateral, units=grid_units, offset=0.0)
        return self.points(lateral, units=grid_units)

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
        """Create coordinate-aware point values in this system.

        Args:
            values: Point coordinates in this coordinate system.
            units: Optional coordinate units.
            offset: Optional surface-normal offset. Used only for surface
                coordinate systems.

        Returns:
            ``CoordinateValue`` tagged with this coordinate-system name.
        """

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

        Args:
            lateral: Lateral coordinates or explicit surface-system points.
            units: Optional coordinate units.
            offset: Optional scalar or per-point normal offset.

        Returns:
            ``CoordinateValue`` tagged with this coordinate-system name.

        Raises:
            ValueError: If lateral coordinates or offsets have unsupported
                shapes.
        """

        lateral_values, lateral_units = _split_quantity(lateral, units)
        offset_values, offset_units = _split_quantity(offset, lateral_units)
        if offset_units is not None and lateral_units is None:
            lateral_units = offset_units
        values = _surface_coordinate_values(lateral_values, offset=offset_values)
        return CoordinateValue(values, units=lateral_units, system=self.name)

    on = on_surface


class SurfaceCoordinateSystem(CoordinateSystem):
    """Coordinate system whose selected axes are re-datumed to a model surface.

    Args:
        name: Coordinate-system name.
        surface: Surface name or one-based surface index.
        axes: Optional axes to expose. Defaults to one surface-relative ``z``
            axis.
        normal: Direction considered positive relative to the surface.
        inherit_axes: Whether missing physical axes should be inherited from
            the global physical coordinate system.
        offset: Optional fixed surface-normal offset.
        offset_units: Optional units for ``offset``.
        **kwargs: Additional coordinate-system fields.
    """

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
        **kwargs: Any,
    ) -> None:
        """Create a coordinate system whose last axis follows a model surface.

        Parameters mirror :meth:`CoordinateSystem.surface`. If ``offset`` is
        supplied, the generated system is fixed at that signed distance from
        the referenced surface.
        """

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
        """Deserialize a surface coordinate-system payload.

        Args:
            data: Serialized surface coordinate-system mapping.

        Returns:
            ``SurfaceCoordinateSystem`` instance.

        Raises:
            ValueError: If the payload does not identify a surface.
        """

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

    def to_fs(self, ctx: Any = None) -> Dict[str, Any]:
        """Serialize this surface coordinate system for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible surface coordinate-system payload.
        """

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
    quantity_units = _first_quantity_units(value)
    if quantity_units is not None:
        target_units = units or quantity_units
        return _strip_coordinate_quantities(value, target_units), target_units
    return value, units


def _first_quantity_units(value: Any) -> Optional[Any]:
    if is_quantity(value):
        return value.units
    if isinstance(value, np.ndarray) and value.dtype == object:
        for item in value.flat:
            units = _first_quantity_units(item)
            if units is not None:
                return units
    if isinstance(value, (list, tuple)):
        for item in value:
            units = _first_quantity_units(item)
            if units is not None:
                return units
    return None


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


def _surface_points_grid_lateral(
    x: Any,
    y: Optional[Any],
    *,
    units: Optional[Any] = None,
) -> tuple[np.ndarray, Optional[Any]]:
    grid_units = units or _first_quantity_units(x)
    if grid_units is None and y is not None:
        grid_units = _first_quantity_units(y)

    x_values, _ = _split_quantity(x, grid_units)
    x_axis = _grid_coordinate_axis("x", x_values)

    if y is None:
        return x_axis.reshape(-1, 1), grid_units

    y_values, _ = _split_quantity(y, grid_units)
    y_axis = _grid_coordinate_axis("y", y_values)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel()]), grid_units


def _grid_coordinate_axis(name: str, values: Any) -> np.ndarray:
    axis = np.asarray(values, dtype=float).reshape(-1)
    if axis.size == 0:
        raise ValueError(f"{name} must contain at least one coordinate")
    if not np.isfinite(axis).all():
        raise ValueError(f"{name} coordinates must be finite")
    return axis


def _coordinate_plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _coordinate_plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coordinate_plain_value(item) for item in value]
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    ):
        return float(value)
    if hasattr(value, "values") and not isinstance(value, (str, bytes)):
        value = value.values
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"f", "i", "u"}:
            return value.astype(np.float64, copy=False).tolist()
        if value.dtype.kind == "O":
            return [_coordinate_plain_value(item) for item in value.tolist()]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            array = np.asarray(value)
            if array.dtype.kind in {"f", "i", "u"}:
                return array.astype(np.float64, copy=False).tolist()
        except (TypeError, ValueError):
            pass
        return value.tolist()
    return value


def _coordinate_value_and_units_to_fs(value: Any, units: Optional[Any] = None) -> Any:
    if is_quantity(value):
        target_units = units or value.units
        return {
            "value": _coordinate_plain_value(value.to(target_units).magnitude),
            "units": unit_expression(target_units),
        }
    if isinstance(value, Mapping):
        payload = _coordinate_plain_value(value)
        if units is not None and "units" not in payload:
            payload["units"] = unit_expression(units)
        return payload

    detected_units = units
    if detected_units is None and hasattr(value, "attrs"):
        detected_units = value.attrs.get("units")

    plain = _coordinate_plain_value(value)
    if detected_units:
        return {"value": plain, "units": unit_expression(detected_units)}
    return plain


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
    """Serialize coordinate-like values for solver input.

    Args:
        value: ``CoordinateValue``, Pint quantity, array-like value, or
            JSON-compatible scalar/mapping.

    Returns:
        JSON-compatible coordinate value.
    """

    if isinstance(value, CoordinateValue):
        return value.to_fs()
    if is_quantity(value):
        return _coordinate_value_and_units_to_fs(value)
    return _coordinate_plain_value(value)


def direction_to_fs(value: Any) -> Any:
    """Serialize direction-like values for solver input.

    Args:
        value: ``Direction``, Pint quantity, or raw solver-compatible value.

    Returns:
        JSON-compatible direction payload or the raw value unchanged.
    """

    if isinstance(value, Direction):
        return value.to_fs()
    if is_quantity(value):
        return quantity_to_fs(value)
    return value
