from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import xarray as xr

from frequensolve.model.model import ModelSubdomain
from frequensolve.model.property import Property
from frequensolve.units import is_quantity, unit_expression, value_and_units_to_fs
from frequensolve.util.mixins import ExtraFieldsMixin, merge_extra
from frequensolve.util.named_list import NamedList

from ._utils import (
    _convert_units,
    _dataarray_with_property_metadata,
    _inline_dataarray_to_fs,
)
from .surfaces import SimpleSurface

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from .model import LayeredModel

__all__ = [
    "BoreholeAnnularPadding",
    "BoreholeSurface",
    "BoreholeLayer",
    "BoreholePart",
    "BoreholePlug",
    "Borehole",
]


def _borehole_surface_ref(value: Any) -> Dict[str, Any]:
    if isinstance(value, SimpleSurface):
        return {"surface": value.name}
    if isinstance(value, str):
        return {"surface": value}
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    raise TypeError(
        "Borehole extent values must be surface names, one-based surface indices, "
        "SimpleSurface objects, or mappings"
    )


def _validate_borehole_radius_profile(radius: Property) -> None:
    value = radius.darr
    if value is None or radius.is_constant or radius.file_path is not None:
        return
    dims = [str(dim) for dim in value.dims]
    if value.ndim == 1:
        if dims[0] not in {"z", "depth"}:
            raise ValueError("Borehole radius profile dimension must be 'z' or 'depth'")
    elif value.ndim == 2:
        theta_dims = {"theta", "azimuth", "angle"}
        depth_dims = {"z", "depth"}
        if not any(dim in theta_dims for dim in dims) or not any(
            dim in depth_dims for dim in dims
        ):
            raise ValueError(
                "2D borehole radius profiles must include theta/azimuth and z/depth dimensions"
            )
    else:
        raise ValueError("Borehole radius profiles must be one- or two-dimensional")
    for dim in dims:
        if dim not in value.coords:
            raise ValueError(
                "Borehole radius profiles require coordinates for every dimension"
            )


def _borehole_radius_to_fs(
    radius: Property,
    ctx=None,
    *,
    borehole_name: Optional[str] = None,
    radius_name: Optional[str] = None,
    radius_group: str = "surfaces",
) -> Any:
    _validate_borehole_radius_profile(radius)
    if (
        ctx is not None
        and getattr(ctx, "store", None) is not None
        and borehole_name is not None
        and radius_name is not None
        and radius.darr is not None
        and not radius.is_constant
    ):
        dataset = (
            f"inputs/model/boreholes/{borehole_name}/" f"{radius_group}/{radius_name}/r"
        )
        return radius.to_fs(ctx=ctx, dataset=dataset)

    if (
        radius.darr is not None
        and not radius.is_constant
        and radius.file_path is None
        and ctx is not None
        and getattr(ctx, "path", None) is not None
        and borehole_name is not None
        and radius_name is not None
    ):
        file = (
            ctx.path
            / "boreholes"
            / borehole_name
            / radius_group
            / f"{radius_name}_r.bin"
        )
        return radius.to_fs(ctx=ctx, file=file)

    if radius.darr is None or radius.is_constant or radius.file_path is not None:
        return radius.to_fs(ctx=ctx)

    return _inline_dataarray_to_fs(_dataarray_with_property_metadata(radius))


def _scalar_number(value: Any, field_name: str) -> float:
    if is_quantity(value):
        raw = value.magnitude
    elif isinstance(value, Mapping):
        if "value" not in value:
            raise ValueError(f"{field_name} mappings require a value")
        raw = value["value"]
    else:
        raw = value

    if isinstance(raw, (str, bytes)):
        raise TypeError(f"{field_name} must be a scalar number")
    values = np.asarray(raw, dtype=float)
    if values.size != 1:
        raise TypeError(f"{field_name} must be a scalar number")
    return float(values.item())


def _padding_count(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("annular_padding/n must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise TypeError("annular_padding/n must be an integer")
    count = int(value)
    if count < 0:
        raise ValueError("annular_padding/n must be non-negative")
    return count


def _length_to_fs(value: Any) -> Any:
    payload = value_and_units_to_fs(value)
    if isinstance(payload, Mapping):
        payload = copy.deepcopy(dict(payload))
        if "units" in payload:
            payload["units"] = unit_expression(payload["units"])
    return payload


@dataclass(kw_only=True)
class BoreholeAnnularPadding(ExtraFieldsMixin):
    """Formation-domain annular padding around a 3D borehole.

    Args:
        n: Number of annular padding cells. ``0`` disables padding.
        outer_radius: Outer radius of the padded annulus. Required when
            ``n > 0``.
        power: Positive radial spacing exponent. Defaults to uniform spacing.
        extra: Additional solver-facing fields preserved on export.

    Raises:
        TypeError: If ``n`` is not an integer or scalar fields are not scalar.
        ValueError: If ``n`` is negative, ``outer_radius`` is missing or
            non-positive when padding is active, or ``power`` is non-positive.
    """

    n: int = 0
    outer_radius: Optional[Any] = None
    power: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        n: int = 0,
        outer_radius: Optional[Any] = None,
        power: float = 1.0,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        self.n = _padding_count(n)
        self.outer_radius = (
            copy.deepcopy(dict(outer_radius))
            if isinstance(outer_radius, Mapping)
            else outer_radius
        )
        self.power = _scalar_number(power, "annular_padding/power")
        if self.power <= 0.0:
            raise ValueError("annular_padding/power must be positive")
        if self.n == 0:
            if self.outer_radius is not None:
                raise ValueError(
                    "annular_padding/n must be positive when outer_radius is supplied"
                )
        elif self.outer_radius is None:
            raise ValueError("annular_padding/outer_radius must be positive when n > 0")
        elif (
            _scalar_number(
                self.outer_radius,
                "annular_padding/outer_radius",
            )
            <= 0.0
        ):
            raise ValueError("annular_padding/outer_radius must be positive when n > 0")
        self._init_extra(extra, **kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "BoreholeAnnularPadding":
        """Deserialize annular padding from a borehole payload."""

        if not isinstance(data, Mapping):
            raise TypeError(
                "annular_padding must be a BoreholeAnnularPadding or mapping"
            )
        return cls(**copy.deepcopy(dict(data)))

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize annular padding for the solver borehole contract."""

        payload: Dict[str, Any] = {"n": self.n}
        if self.n > 0:
            payload["outer_radius"] = _length_to_fs(self.outer_radius)
            payload["power"] = self.power
        return merge_extra(payload, self.extra, "BoreholeAnnularPadding")


@dataclass(kw_only=True)
class BoreholeSurface(ExtraFieldsMixin):
    """Cumulative-radius boundary in a layered-model borehole.

    Args:
        name: Optional surface name used by borehole layer references.
        r: Cumulative radius as a scalar, Pint quantity, property payload, file
            reference, or one-/two-dimensional ``xarray.DataArray``.
        grid: Optional grid metadata for file-backed or ungridded radius
            profiles.
        scale: Multiplicative scale applied to loaded radius values.
        units: Optional radius units.
        system: Optional coordinate-system name for radius-profile
            coordinates.

    Raises:
        ValueError: If ``r`` is missing or a radius profile does not use
            supported borehole coordinates.
    """

    name: Optional[str]
    r: Property = field(default_factory=Property)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: Optional[str] = None,
        r: Any = None,
        *,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        if r is None and name is not None and not isinstance(name, str):
            r = name
            name = None
        if r is None:
            raise ValueError("BoreholeSurface requires r")
        self.name = name
        self.r = (
            r
            if isinstance(r, Property)
            else Property(
                data=r,
                grid=grid,
                scale=scale,
                units=units,
                system=system,
            )
        )
        _validate_borehole_radius_profile(self.r)
        self._init_extra(extra, **kwargs)

    def is_axis(self) -> bool:
        """Return whether this surface is the implicit borehole axis.

        Returns:
            ``True`` when the radius is a scalar zero value; ``False`` for
            nonzero or depth-varying radii.
        """

        try:
            if self.r.darr is not None and not self.r.is_constant:
                return False
            value = self.r.get()
            if is_quantity(value):
                value = value.magnitude
            return float(np.asarray(value).item()) == 0.0
        except Exception:
            return False

    def to_fs(
        self,
        ctx=None,
        *,
        borehole_name: Optional[str] = None,
        radius_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serialize this borehole wall for the solver contract."""

        name = self.name or radius_name
        if name is None:
            raise ValueError("BoreholeSurface requires a name for serialization")
        payload: Dict[str, Any] = {"name": name}
        payload["r"] = _borehole_radius_to_fs(
            self.r,
            ctx,
            borehole_name=borehole_name,
            radius_name=name,
        )
        payload.update(self.merged_extra(payload))
        return payload


@dataclass(kw_only=True)
class BoreholeLayer:
    """Pending radial material layer for a 2D borehole builder.

    ``BoreholeLayer`` is a user-friendly authoring object. It becomes a
    concrete ``BoreholePart`` once an outer radius is known.

    Args:
        name: Optional radial layer name.
        width: Optional scalar radial width. When supplied, the builder can
            close the layer immediately by deriving the outer radius.
        mesh_block_id: Existing or generated solver mesh-block id.
        physics: Optional physics/material family for generated subdomains.
        properties: Optional material properties for generated subdomains.
        grid: Default grid metadata for material properties.
        units: Default units for material properties.
        system: Optional coordinate-system name for property grids. The legacy
            ``coordinate_system`` keyword is accepted as an alias.
        subdomain_name: Optional generated material subdomain name.
        extra: Additional serialized layer fields.
        **kwargs: Extra fields preserved on the pending layer.

    Raises:
        TypeError: If unsupported legacy ``frame`` metadata is supplied.
        ValueError: If coordinate-system aliases conflict.
    """

    name: Optional[str] = None
    width: Optional[Any] = None
    mesh_block_id: Optional[int] = None
    physics: Optional[str] = None
    properties: Optional[dict] = None
    grid: Optional[xr.DataArray] = None
    units: Optional[Any] = None
    system: Optional[str] = None
    subdomain_name: Optional[str] = None
    inner_surface: Optional[str] = None
    outer_surface: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        width: Optional[Any] = None,
        mesh_block_id: Optional[int] = None,
        physics: Optional[str] = None,
        properties: Optional[dict] = None,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        subdomain_name: Optional[str] = None,
        inner_surface: Optional[str] = None,
        outer_surface: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        if "coordinate_system" in kwargs:
            coordinate_system = kwargs.pop("coordinate_system")
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system
        if "frame" in kwargs:
            raise TypeError(
                "BoreholeLayer frame is no longer supported; coordinates are physical"
            )

        self.name = name
        self.width = width
        self.mesh_block_id = mesh_block_id
        self.physics = physics
        self.properties = properties
        self.grid = grid
        self.units = units
        self.system = system
        self.subdomain_name = subdomain_name
        self.inner_surface = inner_surface
        self.outer_surface = outer_surface
        self.extra = dict(extra or {})
        self.extra.update(kwargs)


@dataclass(kw_only=True)
class BoreholePart:
    """One concentric radial material part of a 2D borehole.

    ``r`` is the cumulative outer radius for this part. The first part starts
    at ``r = 0``; each following part starts at the previous part's ``r``.

    Args:
        name: Optional material part name.
        mesh_block_id: Positive solver mesh-block id for this radial interval.
        r: Cumulative outer radius as a scalar, Pint quantity, property
            payload, file reference, or one-dimensional ``xarray.DataArray``.
        grid: Optional grid metadata for file-backed radius values.
        scale: Multiplicative scale applied to loaded radius values.
        units: Optional radius units.
        system: Optional coordinate-system name for radius-profile
            coordinates.
        extra: Additional serialized part fields.
        **kwargs: Extra fields preserved on export.

    Raises:
        TypeError: If removed radius/role fields are supplied.
        ValueError: If ``mesh_block_id`` or ``r`` is missing, the id is not
            positive, or the radius profile is invalid.
    """

    name: Optional[str]
    mesh_block_id: int
    r: Property = field(default_factory=Property)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: Optional[str] = None,
        mesh_block_id: Optional[int] = None,
        r: Any = None,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        unsupported = {
            "role",
            "cells",
            "radius",
            "inner_radius",
            "outer_radius",
        } & set(kwargs)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"BoreholePart uses r; unsupported field(s): {names}")
        if mesh_block_id is None:
            raise ValueError("BoreholePart requires mesh_block_id")
        if mesh_block_id < 1:
            raise ValueError("BoreholePart mesh_block_id must be positive")
        if r is None:
            raise ValueError("BoreholePart requires r")
        self.name = name or f"part_{mesh_block_id}"
        self.mesh_block_id = mesh_block_id
        self.r = (
            r
            if isinstance(r, Property)
            else Property(
                data=r,
                grid=grid,
                scale=scale,
                units=units,
                system=system,
            )
        )
        _validate_borehole_radius_profile(self.r)
        self.extra = dict(extra or {})
        self.extra.update(kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "BoreholePart":
        """Deserialize one borehole material part from a solver payload.

        Args:
            data: Serialized borehole part mapping.

        Returns:
            A ``BoreholePart`` with radius and extra metadata restored.

        Raises:
            TypeError: If legacy unsupported fields are present.
            ValueError: If the payload is missing ``r``.
        """

        payload = copy.deepcopy(dict(data))
        unsupported = {
            "role",
            "cells",
            "radius",
            "inner_radius",
            "outer_radius",
        } & set(payload)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"BoreholePart uses r; unsupported field(s): {names}")
        if "r" not in payload:
            raise ValueError("BoreholePart requires r")
        return cls(
            mesh_block_id=payload.pop("mesh_block_id"),
            name=payload.pop("name", None),
            r=payload.pop("r"),
            extra=payload,
        )

    def to_fs(
        self,
        ctx=None,
        *,
        borehole_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serialize this radial material part for the solver contract.

        Args:
            ctx: Optional export context used for project-relative paths or
                HDF5-backed storage.
            borehole_name: Optional parent borehole name used to place stored
                radius-profile datasets.

        Returns:
            Solver-ready radial part payload.
        """

        payload = {
            "name": self.name,
            "mesh_block_id": self.mesh_block_id,
            "r": _borehole_radius_to_fs(
                self.r,
                ctx,
                borehole_name=borehole_name,
                radius_name=self.name,
            ),
        }
        return merge_extra(payload, self.extra, "BoreholePart")


@dataclass(kw_only=True)
class BoreholePlug:
    """Local axial obstruction inside a borehole fluid interval.

    Args:
        name: Optional plug name.
        top: Top depth or surface reference for the obstruction.
        bottom: Bottom depth or surface reference for the obstruction.
        mesh_block_id: Positive solver mesh-block id for the plug material.
        r: Plug radius. ``radius`` is accepted as an alias.
        radius: Alias for ``r``.
        grid: Optional grid metadata for radius profiles.
        scale: Multiplicative scale applied to loaded radius values.
        units: Optional radius units.
        system: Optional coordinate-system name for radius-profile
            coordinates.
        extra: Additional serialized plug fields.
        **kwargs: Extra fields preserved on export.

    Raises:
        ValueError: If geometry, radius, or mesh-block id fields are missing or
            invalid.
    """

    name: str
    top: Any
    bottom: Any
    mesh_block_id: int
    r: Property = field(default_factory=Property)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        top: Any = None,
        bottom: Any = None,
        mesh_block_id: Optional[int] = None,
        r: Any = None,
        radius: Any = None,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        if "coordinate_system" in kwargs:
            coordinate_system = kwargs.pop("coordinate_system")
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system
        if "frame" in kwargs:
            raise TypeError(
                "BoreholePlug frame is no longer supported; coordinates are physical"
            )
        if radius is not None:
            if r is not None:
                raise ValueError("Specify only one of r or radius")
            r = radius
        if top is None or bottom is None:
            raise ValueError("BoreholePlug requires top and bottom depths")
        if mesh_block_id is None:
            raise ValueError("BoreholePlug requires mesh_block_id")
        if mesh_block_id < 1:
            raise ValueError("BoreholePlug mesh_block_id must be positive")
        if r is None:
            raise ValueError("BoreholePlug requires radius")

        self.name = name or f"plug_{mesh_block_id}"
        self.top = top
        self.bottom = bottom
        self.mesh_block_id = mesh_block_id
        self.r = (
            r
            if isinstance(r, Property)
            else Property(
                data=r,
                grid=grid,
                scale=scale,
                units=units,
                system=system,
            )
        )
        _validate_borehole_radius_profile(self.r)
        self.extra = dict(extra or {})
        self.extra.update(kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "BoreholePlug":
        """Deserialize one borehole plug from a solver payload.

        Args:
            data: Serialized plug mapping.

        Returns:
            A ``BoreholePlug`` with radius and extra metadata restored.

        Raises:
            ValueError: If both ``r`` and legacy ``radius`` are supplied.
        """

        payload = copy.deepcopy(dict(data))
        if "radius" in payload:
            if "r" in payload:
                raise ValueError("Specify only one of plug r or radius")
            payload["r"] = payload.pop("radius")
        return cls(
            mesh_block_id=payload.pop("mesh_block_id"),
            name=payload.pop("name", None),
            top=payload.pop("top"),
            bottom=payload.pop("bottom"),
            r=payload.pop("r"),
            extra=payload,
        )

    @staticmethod
    def _depth_to_fs(value: Any) -> Any:
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
        return value_and_units_to_fs(value)

    def to_fs(
        self,
        ctx=None,
        *,
        borehole_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Serialize this plug geometry and radius for the solver contract.

        Args:
            ctx: Optional export context used for project-relative paths or
                HDF5-backed storage.
            borehole_name: Optional parent borehole name used to place stored
                radius-profile datasets.

        Returns:
            Solver-ready plug payload.
        """

        payload = {
            "name": self.name,
            "mesh_block_id": self.mesh_block_id,
            "top": self._depth_to_fs(self.top),
            "bottom": self._depth_to_fs(self.bottom),
            "r": _borehole_radius_to_fs(
                self.r,
                ctx,
                borehole_name=borehole_name,
                radius_name=self.name,
                radius_group="plugs",
            ),
        }
        return merge_extra(payload, self.extra, "BoreholePlug")


@dataclass(kw_only=True)
class Borehole:
    """Vertical borehole geometry for a layered model.

    Args:
        name: Borehole name used for references and serialized paths.
        axis: Axis mapping, such as ``{"x": value}`` for 2D models or
            ``{"x": value, "y": value}`` for 3D models.
        extent: Mapping containing ``top`` and ``bottom`` extent references.
        parts: Optional closed radial material parts.
        layers: Alias for ``parts`` in serialized payloads.
        surfaces: Optional cumulative-radius surfaces.
        plugs: Optional axial plug/tool-body intervals.
        annular_padding: Optional 3D formation-domain annular padding.
        model: Optional parent ``LayeredModel``. When present, layer and plug
            authoring can create material subdomains automatically.
        extra: Additional serialized borehole fields.
        **kwargs: Extra fields preserved on export.

    Raises:
        ValueError: If both ``parts`` and ``layers`` are supplied.
    """

    name: str
    axis: Dict[str, Any]
    extent: Dict[str, Any]
    surfaces: NamedList = field(default_factory=NamedList)
    parts: NamedList = field(default_factory=NamedList)
    plugs: NamedList = field(default_factory=NamedList)
    annular_padding: Optional[BoreholeAnnularPadding] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    _model: Optional["LayeredModel"] = field(default=None, repr=False)
    _pending_layer: Optional[Dict[str, Any]] = field(default=None, repr=False)
    _part_outer_surface_indices: List[int] = field(default_factory=list, repr=False)

    def __init__(
        self,
        name: str,
        axis: Mapping[str, Any],
        extent: Mapping[str, Any],
        parts: Optional[List[Union[BoreholePart, Mapping[str, Any]]]] = None,
        layers: Optional[List[Union[BoreholePart, Mapping[str, Any]]]] = None,
        surfaces: Optional[List[Union[BoreholeSurface, Mapping[str, Any]]]] = None,
        plugs: Optional[List[Union[BoreholePlug, Mapping[str, Any]]]] = None,
        annular_padding: Optional[
            Union[BoreholeAnnularPadding, Mapping[str, Any]]
        ] = None,
        model: Optional["LayeredModel"] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        if parts is not None and layers is not None:
            raise ValueError("Specify only one of borehole parts or layers")
        self.name = name
        self.axis = copy.deepcopy(dict(axis))
        self.extent = copy.deepcopy(dict(extent))
        self.surfaces = NamedList()
        for surface in surfaces or []:
            self.surfaces.append(
                surface
                if isinstance(surface, BoreholeSurface)
                else BoreholeSurface(**surface)
            )
        self.parts = NamedList()
        for part in (layers if layers is not None else parts) or []:
            self.parts.append(
                part if isinstance(part, BoreholePart) else BoreholePart.from_fs(part)
            )
        if not self.surfaces and self.parts:
            self.surfaces.extend(self._surfaces_from_parts(self.parts))
        self.plugs = NamedList()
        for plug in plugs or []:
            self.plugs.append(
                plug if isinstance(plug, BoreholePlug) else BoreholePlug.from_fs(plug)
            )
        self.annular_padding = (
            annular_padding
            if isinstance(annular_padding, BoreholeAnnularPadding)
            else (
                BoreholeAnnularPadding.from_fs(annular_padding)
                if annular_padding is not None
                else None
            )
        )
        self._model = model
        self._pending_layer = None
        self._part_outer_surface_indices = self._resolve_part_outer_surface_indices()
        self.extra = dict(extra or {})
        self.extra.update(kwargs)

    @staticmethod
    def _surfaces_from_parts(parts: List[BoreholePart]) -> List[BoreholeSurface]:
        surfaces: List[BoreholeSurface] = []
        if parts:
            inner = parts[0].extra.get("inner_surface")
            if inner:
                surfaces.append(BoreholeSurface(inner, r=0.0))
        for part in parts:
            surfaces.append(BoreholeSurface(part.extra.get("outer_surface"), r=part.r))
        return surfaces

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Borehole":
        """Deserialize a complete borehole geometry from a solver payload.

        Args:
            data: Serialized borehole mapping containing axis, extent, layers
                or parts, optional surfaces, and optional plugs.

        Returns:
            A ``Borehole`` with closed parts and surfaces restored.

        Raises:
            ValueError: If axis fields are missing or ambiguous, or if layers
                without radii cannot be matched to surfaces.
        """

        payload = copy.deepcopy(dict(data))
        axis = payload.pop("axis", None)
        x = payload.pop("x", None)
        if axis is None:
            if x is None:
                raise ValueError("Borehole requires axis/x")
            axis = {"x": x}
        elif x is not None:
            raise ValueError("Specify either borehole axis or x, not both")
        surfaces = payload.pop("surfaces", [])
        parts = payload.pop("layers", payload.pop("parts", []))
        plugs = payload.pop("plugs", [])
        annular_padding = payload.pop("annular_padding", None)
        surface_specs: List[BoreholeSurface] = []
        if surfaces:
            surface_specs = [
                (
                    surface
                    if isinstance(surface, BoreholeSurface)
                    else BoreholeSurface(**surface)
                )
                for surface in surfaces
            ]
            surface_by_name = {
                (
                    surface.name
                    or f"{payload.get('name', 'borehole')}_surface_{index + 1}"
                ): surface
                for index, surface in enumerate(surface_specs)
            }
            for index, part in enumerate(parts):
                if "r" not in part:
                    outer = part.get("outer_surface")
                    if outer is not None and outer in surface_by_name:
                        part["r"] = surface_by_name[outer].r
                    elif index < len(surface_specs):
                        part["r"] = surface_specs[index].r
                    else:
                        raise ValueError(
                            "Borehole layers without r require matching surfaces"
                        )
        elif parts:
            first_inner = parts[0].get("inner_surface")
            if first_inner:
                surface_specs.append(BoreholeSurface(first_inner, r=0.0))
            for part in parts:
                outer = part.get("outer_surface")
                if outer:
                    surface_specs.append(BoreholeSurface(outer, r=part["r"]))
                elif "r" in part:
                    surface_specs.append(BoreholeSurface(r=part["r"]))

        borehole = cls(
            name=payload.pop("name"),
            axis=axis,
            extent=payload.pop("extent"),
            layers=parts,
            surfaces=surface_specs,
            plugs=plugs,
            annular_padding=annular_padding,
            extra=payload,
        )
        return borehole

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize borehole geometry, layers, surfaces, and plugs.

        Args:
            ctx: Optional export context used for project-relative paths or
                HDF5-backed storage.

        Returns:
            Solver-ready borehole payload.

        Raises:
            ValueError: If a pending layer has not been closed or the borehole
                has no material parts.
        """

        if self._pending_layer is not None:
            raise ValueError(
                f"Borehole '{self.name}' has an unclosed layer; call add_surface() "
                "to set its outer radius"
            )
        if not self.parts:
            raise ValueError(f"Borehole '{self.name}' requires at least one layer/part")
        payload = {
            "name": self.name,
            "axis": {
                key: (
                    copy.deepcopy(dict(value))
                    if isinstance(value, Mapping)
                    else value_and_units_to_fs(value)
                )
                for key, value in self.axis.items()
            },
            "extent": {
                "top": _borehole_surface_ref(self.extent["top"]),
                "bottom": _borehole_surface_ref(self.extent["bottom"]),
            },
            "layers": [
                self._layer_to_fs(part, index, ctx)
                for index, part in enumerate(self.parts)
            ],
            "surfaces": [
                self._surface_to_fs(index, surface, ctx)
                for index, surface in enumerate(self.surfaces)
                if not surface.is_axis()
            ],
            **(
                {
                    "plugs": [
                        plug.to_fs(ctx, borehole_name=self.name) for plug in self.plugs
                    ]
                }
                if self.plugs
                else {}
            ),
            **(
                {"annular_padding": self.annular_padding.to_fs(ctx)}
                if self.annular_padding is not None
                else {}
            ),
        }
        return merge_extra(payload, self.extra, "Borehole")

    @property
    def surface_names(self) -> List[str]:
        """Return names of explicit borehole radial surfaces.

        Returns:
            Surface names in radial order, with generated names for unnamed
            surfaces.
        """

        return [self._surface_name(index) for index, _ in enumerate(self.surfaces)]

    @property
    def layer_names(self) -> List[str]:
        """Return names of closed and pending borehole radial layers.

        Returns:
            Closed part names followed by the pending layer name, if one is
            open.
        """

        return [part.name for part in self.parts] + (
            [self._pending_layer["name"]] if self._pending_layer is not None else []
        )

    @property
    def pending_layer(self) -> Optional[str]:
        """Return the unclosed layer waiting for an outer surface.

        Returns:
            Pending layer name, or ``None`` when all layers are closed.
        """

        return None if self._pending_layer is None else self._pending_layer["name"]

    def _surface_name(self, index: int) -> str:
        if index < len(self.surfaces) and self.surfaces[index].name:
            return str(self.surfaces[index].name)
        return f"{self.name}_surface_{index + 1}"

    def _default_outer_surface_index(self, part_index: int) -> int:
        if self.surfaces and self.surfaces[0].is_axis():
            return part_index + 1
        return part_index

    def _surface_index_by_name(self) -> Dict[str, int]:
        return {
            self._surface_name(index): index for index, _ in enumerate(self.surfaces)
        }

    def _resolve_part_outer_surface_indices(self) -> List[int]:
        surface_by_name = self._surface_index_by_name()
        indices: List[int] = []
        for index, part in enumerate(self.parts):
            outer = part.extra.get("outer_surface")
            if outer is not None:
                if outer not in surface_by_name:
                    raise ValueError(
                        f"Borehole layer '{part.name}' references unknown "
                        f"outer_surface '{outer}'"
                    )
                outer_index = surface_by_name[outer]
            else:
                outer_index = self._default_outer_surface_index(index)
            if outer_index >= len(self.surfaces):
                raise ValueError(
                    "Borehole layers require ordered outer surfaces; layer "
                    f"'{part.name}' has no matching surface"
                )
            if indices and outer_index <= indices[-1]:
                raise ValueError(
                    "Borehole layer outer surfaces must increase in radial order"
                )
            indices.append(outer_index)
        return indices

    def _outer_surface_index(self, part_index: int) -> int:
        if part_index < len(self._part_outer_surface_indices):
            return self._part_outer_surface_indices[part_index]
        return self._default_outer_surface_index(part_index)

    def _inner_surface_index(self, part_index: int) -> Optional[int]:
        if part_index == 0:
            return 0 if self.surfaces and self.surfaces[0].is_axis() else None
        return self._outer_surface_index(part_index - 1)

    def _layer_to_fs(self, part: BoreholePart, index: int, ctx=None) -> Dict[str, Any]:
        payload = merge_extra(
            {
                "name": part.name,
                "mesh_block_id": part.mesh_block_id,
            },
            part.extra,
            "BoreholePart",
        )
        outer_index = self._outer_surface_index(index)
        payload.setdefault("outer_surface", self._surface_name(outer_index))
        inner_index = self._inner_surface_index(index)
        if inner_index is not None:
            payload.setdefault("inner_surface", self._surface_name(inner_index))
        return payload

    def _surface_to_fs(
        self,
        index: int,
        surface: BoreholeSurface,
        ctx=None,
    ) -> Dict[str, Any]:
        name = self._surface_name(index)
        return surface.to_fs(
            ctx,
            borehole_name=self.name,
            radius_name=name,
        )

    def _part_to_fs(self, part: BoreholePart, index: int, ctx=None) -> Dict[str, Any]:
        payload = self._layer_to_fs(part, index, ctx)
        surface_index = self._outer_surface_index(index)
        payload["r"] = _borehole_radius_to_fs(
            (
                self.surfaces[surface_index].r
                if surface_index < len(self.surfaces)
                else part.r
            ),
            ctx,
            borehole_name=self.name,
            radius_name=payload["outer_surface"],
        )
        return payload

    def _previous_radius(self) -> Optional[Property]:
        if self.surfaces:
            return self.surfaces[-1].r
        if self.parts:
            return self.parts[-1].r
        return None

    @staticmethod
    def _scalar_width_value(width: Any) -> Tuple[float, Optional[str]]:
        if is_quantity(width):
            values = np.asarray(width.magnitude)
            units = unit_expression(width.units)
        elif isinstance(width, (int, float, np.integer, np.floating)):
            values = np.asarray(width)
            units = None
        else:
            raise TypeError(
                "Borehole layer width must be a scalar number or Pint quantity; "
                "use add_surface(...) for variable-radius surfaces"
            )
        if values.size != 1:
            raise TypeError(
                "Borehole layer width must be scalar; use add_surface(...) for "
                "variable-radius surfaces"
            )
        value = float(values.item())
        if value <= 0.0:
            raise ValueError("Borehole layer width must be positive")
        return value, units

    def _radius_from_width(self, width: Any) -> Any:
        width_value, width_units = self._scalar_width_value(width)
        previous = self._previous_radius()
        if previous is None:
            if width_units is not None:
                return {"value": width_value, "units": width_units}
            return width_value
        if not previous.is_constant:
            raise ValueError(
                "Borehole layer width can only follow scalar radii; use "
                "add_surface(...) for variable-radius surfaces"
            )

        previous_value = float(np.asarray(previous.get()).item())
        previous_units = previous.units
        if previous_units is not None:
            if width_units is not None:
                width_value = _convert_units(width_value, width_units, previous_units)
            return {"value": previous_value + width_value, "units": previous_units}
        if width_units is not None:
            if previous_value != 0.0:
                raise ValueError(
                    "Cannot add a unit-bearing width to a unitless previous "
                    "borehole radius"
                )
            return {"value": width_value, "units": width_units}
        return previous_value + width_value

    def add_layer(
        self,
        name: Optional[str] = None,
        *,
        width: Optional[Any] = None,
        mesh_block_id: Optional[int] = None,
        physics: Optional[str] = None,
        properties: Optional[dict] = None,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        subdomain_name: Optional[str] = None,
        **kwargs,
    ) -> Optional[BoreholeSurface]:
        """Add a pending radial material layer.

        The layer is closed by the next ``add_surface(...)`` call. The borehole
        axis at ``r = 0`` is implicit, so the first call is normally
        ``add_layer(...)`` followed by ``add_surface(...)``. When ``width`` is
        supplied, it must be scalar and the layer is closed immediately by an
        automatically created surface.

        Args:
            name: Optional layer name.
            width: Optional scalar radial width. If supplied, the layer is
                closed immediately.
            mesh_block_id: Existing or generated solver mesh-block id.
            physics: Optional physics/material family for generated
                subdomains.
            properties: Optional material properties for generated subdomains.
            grid: Default grid metadata for generated material properties.
            units: Default units for generated material properties.
            system: Coordinate-system name for generated property grids.
            subdomain_name: Optional generated material subdomain name.
            **kwargs: Additional serialized layer fields. The legacy
                ``coordinate_system`` alias is accepted.

        Returns:
            The automatically created closing surface when ``width`` is
            supplied; otherwise ``None``.

        Raises:
            TypeError: If unsupported legacy ``frame`` metadata is supplied.
            ValueError: If another layer is pending, coordinate-system aliases
                conflict, or a width cannot be converted to a valid radius.
        """

        if self._pending_layer is not None:
            raise ValueError(
                f"Borehole layer '{self._pending_layer['name']}' is missing an "
                "outer surface"
            )
        if "coordinate_system" in kwargs:
            coordinate_system = kwargs.pop("coordinate_system")
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system
        if "frame" in kwargs:
            raise TypeError(
                "Borehole.add_layer frame is no longer supported; coordinates are physical"
            )
        layer_name = name or f"part_{len(self.parts) + 1}"
        auto_radius = self._radius_from_width(width) if width is not None else None
        self._pending_layer = {
            "name": layer_name,
            "mesh_block_id": mesh_block_id,
            "physics": physics,
            "properties": properties,
            "grid": grid,
            "units": units,
            "system": system,
            "subdomain_name": subdomain_name,
            "extra": dict(kwargs),
        }
        if width is None:
            return None
        return self.add_surface(r=auto_radius)

    def _add_layer_object(self, layer: BoreholeLayer) -> Optional[BoreholeSurface]:
        return self.add_layer(
            layer.name,
            width=layer.width,
            mesh_block_id=layer.mesh_block_id,
            physics=layer.physics,
            properties=layer.properties,
            grid=layer.grid,
            units=layer.units,
            system=layer.system,
            subdomain_name=layer.subdomain_name,
            **(
                {"inner_surface": layer.inner_surface}
                if layer.inner_surface is not None
                else {}
            ),
            **(
                {"outer_surface": layer.outer_surface}
                if layer.outer_surface is not None
                else {}
            ),
            **layer.extra,
        )

    def add_surface(
        self,
        name: Optional[str] = None,
        r: Any = None,
        *,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> BoreholeSurface:
        """Add a cumulative-radius surface and close the pending layer.

        Args:
            name: Optional surface name. If omitted and ``name`` is radius-like,
                it is treated as ``r`` for compact authoring.
            r: Cumulative radius for the surface. ``radius`` is accepted as an
                alias.
            grid: Optional grid metadata for file-backed radius values.
            scale: Multiplicative scale applied to loaded radius values.
            units: Optional radius units.
            system: Optional coordinate-system name for radius-profile
                coordinates.
            **kwargs: Extra serialized layer fields. ``coordinate_system`` is
                accepted as an alias for ``system``.

        Returns:
            The created ``BoreholeSurface``.

        Raises:
            TypeError: If unexpected keyword arguments are supplied.
            ValueError: If no radius is provided, the surface name is
                duplicated, or a non-axis surface is added before a layer.
        """

        if r is None and name is not None and not isinstance(name, str):
            r = name
            name = None
        if "radius" in kwargs:
            if r is not None:
                raise ValueError("Specify only one of r or radius")
            r = kwargs.pop("radius")
        if "coordinate_system" in kwargs:
            coordinate_system = kwargs.pop("coordinate_system")
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system
        if "frame" in kwargs:
            raise TypeError(
                "Borehole.add_surface frame is no longer supported; coordinates are physical"
            )
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected Borehole.add_surface arguments: {unexpected}")
        if r is None:
            raise ValueError("Borehole.add_surface requires r")
        if name is not None and any(surface.name == name for surface in self.surfaces):
            raise ValueError(f"Borehole surface name already exists: {name}")

        surface = BoreholeSurface(
            name=name,
            r=r,
            grid=grid,
            scale=scale,
            units=units,
            system=system,
        )

        if self._pending_layer is None:
            if not self.parts and surface.is_axis():
                self.surfaces.append(surface)
                return surface
            if self.parts:
                self.surfaces.append(surface)
                self._part_outer_surface_indices[-1] = len(self.surfaces) - 1
                self.parts[-1].r = surface.r
                return surface
            raise ValueError(
                "Borehole.add_surface closes a pending layer; call add_layer() "
                "before adding the first non-axis surface"
            )

        layer = self._pending_layer
        payload = dict(layer["extra"])
        payload.update(
            {
                "name": layer["name"],
                "mesh_block_id": layer["mesh_block_id"],
                "r": surface.r,
            }
        )
        if layer["physics"] is not None:
            payload["physics"] = layer["physics"]
        if layer["properties"] is not None:
            payload["properties"] = layer["properties"]
        if layer["grid"] is not None:
            payload["grid"] = layer["grid"]
        if layer["units"] is not None:
            payload["units"] = layer["units"]
        if layer["system"] is not None:
            payload["system"] = layer["system"]
        if layer["subdomain_name"] is not None:
            payload["subdomain_name"] = layer["subdomain_name"]

        if self._model is None:
            if payload.get("properties") is not None:
                raise ValueError(
                    "Borehole.add_layer with properties requires a parent LayeredModel"
                )
            part = BoreholePart.from_fs(payload)
        else:
            part, subdomain, _ = self._model._coerce_borehole_part(
                self.name,
                payload,
                self._model._next_mesh_block_id(),
            )
            if subdomain is not None:
                self._model._add_unique_subdomain(subdomain)
            elif not any(
                subdomain.mesh_block_id == part.mesh_block_id
                for subdomain in self._model.subdomains
            ):
                raise ValueError(
                    "Borehole layer mesh_block_id must reference a model subdomain; "
                    f"missing: {part.mesh_block_id}"
                )

        self.parts.append(part)
        self.surfaces.append(surface)
        self._part_outer_surface_indices.append(len(self.surfaces) - 1)
        self._pending_layer = None
        return surface

    def add_plug(
        self,
        name: Optional[str] = None,
        *,
        top: Any,
        bottom: Any,
        radius: Any = None,
        r: Any = None,
        mesh_block_id: Optional[int] = None,
        physics: Optional[str] = None,
        properties: Optional[dict] = None,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        subdomain_name: Optional[str] = None,
        **kwargs,
    ) -> BoreholePlug:
        """Add a local plug or tool body inside the borehole.

        Args:
            name: Optional plug name.
            top: Top depth or surface reference for the plug interval.
            bottom: Bottom depth or surface reference for the plug interval.
            radius: Plug radius; alias for ``r``.
            r: Plug radius as a scalar, Pint quantity, property payload, file
                reference, or one-dimensional ``xarray.DataArray``.
            mesh_block_id: Existing or generated solver mesh-block id.
            physics: Optional physics/material family for generated
                subdomains.
            properties: Optional material properties for generated subdomains.
            grid: Default grid metadata for radius and generated properties.
            units: Optional radius or property units.
            system: Optional coordinate-system name for radius/profile grids.
            subdomain_name: Optional generated material subdomain name.
            **kwargs: Extra serialized plug fields. ``coordinate_system`` is
                accepted as an alias for ``system``.

        Returns:
            The created ``BoreholePlug``.

        Raises:
            ValueError: If radius aliases conflict, geometry is incomplete, or
                plug material domains cannot be resolved.
        """

        if r is not None and radius is not None:
            raise ValueError("Specify only one of r or radius")
        if "coordinate_system" in kwargs:
            coordinate_system = kwargs.pop("coordinate_system")
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system
        payload = dict(kwargs)
        payload.update(
            {
                "name": name,
                "top": top,
                "bottom": bottom,
                "mesh_block_id": mesh_block_id,
                "r": r if r is not None else radius,
            }
        )
        if physics is not None:
            payload["physics"] = physics
        if properties is not None:
            payload["properties"] = properties
        if grid is not None:
            payload["grid"] = grid
        if units is not None:
            payload["units"] = units
        if system is not None:
            payload["system"] = system
        if subdomain_name is not None:
            payload["subdomain_name"] = subdomain_name

        if self._model is None:
            if properties is not None:
                raise ValueError(
                    "Borehole.add_plug with properties requires a parent LayeredModel"
                )
            plug = BoreholePlug.from_fs(payload)
        else:
            plug, subdomain, _ = self._model._coerce_borehole_plug(
                self.name,
                payload,
                self._model._next_mesh_block_id(),
            )
            if subdomain is not None:
                self._model._add_unique_subdomain(subdomain)
            elif not any(
                subdomain.mesh_block_id == plug.mesh_block_id
                for subdomain in self._model.subdomains
            ):
                raise ValueError(
                    "Borehole plug mesh_block_id must reference a model subdomain; "
                    f"missing: {plug.mesh_block_id}"
                )

        self.plugs.append(plug)
        return plug

    def __iadd__(self, other):
        if isinstance(other, BoreholeLayer):
            self._add_layer_object(other)
        elif isinstance(other, BoreholeSurface):
            self.add_surface(other.name, r=other.r)
        elif isinstance(other, BoreholePlug):
            self.plugs.append(other)
        else:
            raise ValueError(f"Cannot add {type(other)} to Borehole")
        return self

    @staticmethod
    def _length_value(value: Any, units: Optional[Any] = None) -> float:
        target_units = unit_expression(units) if units is not None else None
        if isinstance(value, Mapping):
            payload = dict(value)
            raw = payload.get("value")
            if raw is None:
                raise ValueError("Length mappings require a value")
            return float(_convert_units(float(raw), payload.get("units"), target_units))
        if hasattr(value, "to") and hasattr(value, "magnitude"):
            if target_units is not None:
                return float(value.to(target_units).magnitude)
            return float(value.magnitude)
        return float(value)

    def axis_x(self, units: Optional[Any] = None) -> float:
        """Return the borehole axis x-coordinate.

        Args:
            units: Optional target units for the returned coordinate.

        Returns:
            Axis coordinate as a float.

        Raises:
            ValueError: If the borehole has no x-axis coordinate.
        """

        if "x" not in self.axis:
            raise ValueError("Borehole requires axis/x")
        return self._length_value(self.axis["x"], units)

    def axis_y(self, units: Optional[Any] = None) -> float:
        """Return the borehole axis y-coordinate for 3D models.

        Args:
            units: Optional target units for the returned coordinate.

        Returns:
            Axis coordinate as a float.

        Raises:
            ValueError: If the borehole has no y-axis coordinate.
        """

        if "y" not in self.axis:
            raise ValueError("Borehole requires axis/y")
        return self._length_value(self.axis["y"], units)

    def radius_profile(
        self,
        part: BoreholePart,
        z: Optional[Any] = None,
        *,
        units: Optional[Any] = None,
        depth_units: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return depth and cumulative outer-radius arrays for one part.

        Args:
            part: Borehole part whose outer radius should be sampled.
            z: Optional depth used for scalar radii.
            units: Optional target units for radius values.
            depth_units: Optional target units for depth coordinates. Defaults
                to ``units`` when omitted.

        Returns:
            Tuple ``(z_values, r_values)`` as NumPy arrays.

        Raises:
            ValueError: If the part radius profile has invalid dimensions.
        """

        radius = part.r
        target_units = unit_expression(units) if units is not None else None
        target_depth_units = (
            unit_expression(depth_units) if depth_units is not None else target_units
        )
        if radius.darr is None or radius.is_constant or radius.file_path is not None:
            r_value = radius.get()
            source_units = radius.units
            r_value = float(
                _convert_units(
                    float(np.asarray(r_value).item()), source_units, target_units
                )
            )
            if z is None:
                z_value = 0.0
            else:
                z_value = self._length_value(z, target_depth_units)
            return np.asarray([z_value], dtype=float), np.asarray(
                [r_value], dtype=float
            )

        _validate_borehole_radius_profile(radius)
        values = radius.darr
        if values.ndim != 1:
            raise ValueError(
                "Borehole.radius_profile supports scalar and depth-varying "
                "circular radii only"
            )
        dim = values.dims[0]
        coord = values.coords[dim]
        z_values = np.asarray(coord.values, dtype=float)
        r_values = np.asarray(values.values, dtype=float)
        if coord.attrs.get("units"):
            z_values = _convert_units(
                z_values,
                coord.attrs.get("units"),
                target_depth_units,
            )
        if radius.units or values.attrs.get("units"):
            r_values = _convert_units(
                r_values,
                radius.units or values.attrs.get("units"),
                target_units,
            )
        return z_values, r_values

    def radius_at(
        self,
        part: BoreholePart,
        z: Optional[Any] = None,
        *,
        units: Optional[Any] = None,
        depth_units: Optional[Any] = None,
    ) -> float:
        """Evaluate one part's cumulative radius at a depth.

        Args:
            part: Borehole part whose outer radius should be evaluated.
            z: Optional depth. When omitted for a depth-varying profile, the
                midpoint sample is returned.
            units: Optional target units for the radius.
            depth_units: Optional units for ``z``.

        Returns:
            Interpolated cumulative radius as a float.
        """

        z_values, r_values = self.radius_profile(
            part,
            z,
            units=units,
            depth_units=depth_units,
        )
        if len(r_values) == 1:
            return float(r_values[0])
        if z is None:
            return float(r_values[len(r_values) // 2])
        z_value = self._length_value(
            z,
            unit_expression(depth_units) if depth_units is not None else units,
        )
        return float(np.interp(z_value, z_values, r_values))

    def draw(
        self,
        ax: Optional["Axes"] = None,
        *,
        z: Optional[Any] = None,
        units: Optional[Any] = None,
        depth_units: Optional[Any] = None,
        subdomains: Optional[List[ModelSubdomain]] = None,
        annotate: bool = True,
        colors: Optional[List[str]] = None,
        alpha: float = 0.35,
        linewidth: float = 1.4,
        title: Optional[str] = None,
        show: bool = False,
    ) -> "Axes":
        """Draw a borehole radial profile as concentric material circles.

        Args:
            ax: Optional matplotlib axes. A new axes is created when omitted.
            z: Optional depth at which depth-varying radii are evaluated.
            units: Optional radius display units.
            depth_units: Optional units for ``z``.
            subdomains: Optional material subdomains used to label mesh-block
                ids.
            annotate: Whether to draw labels for radial intervals.
            colors: Optional color palette for the radial intervals.
            alpha: Fill alpha for material circles.
            linewidth: Circle outline width.
            title: Optional axes title.
            show: Whether to call ``plt.show()`` before returning.

        Returns:
            Matplotlib axes containing the borehole drawing.

        Raises:
            ModuleNotFoundError: Converted to an optional-dependency error when
                matplotlib is not installed.
        """

        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
        except ModuleNotFoundError as exc:
            from frequensolve._optional import optional_dependency_error

            raise optional_dependency_error(
                "Borehole drawing",
                extra="visual",
                dependencies=("matplotlib",),
                error=exc,
            ) from exc

        if ax is None:
            _, ax = plt.subplots()

        palette = colors or [
            "#76b7b2",
            "#f28e2b",
            "#59a14f",
            "#e15759",
            "#4e79a7",
            "#b07aa1",
        ]
        subdomain_names = {
            subdomain.mesh_block_id: subdomain.name for subdomain in subdomains or []
        }
        radii = [
            self.radius_at(part, z, units=units, depth_units=depth_units)
            for part in self.parts
        ]
        if any(radius <= 0.0 for radius in radii):
            raise ValueError("Borehole radii must be positive to draw a profile")
        for radius, part, color in reversed(
            list(
                zip(radii, self.parts, palette * (len(self.parts) // len(palette) + 1))
            )
        ):
            label = subdomain_names.get(part.mesh_block_id, part.name)
            patch = Circle(
                (0.0, 0.0),
                radius,
                facecolor=color,
                edgecolor="black",
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )
            ax.add_patch(patch)

        if annotate:
            for radius, part in zip(radii, self.parts):
                label = subdomain_names.get(part.mesh_block_id, part.name)
                ax.annotate(
                    f"{label}\nr={radius:g}",
                    xy=(radius, 0.0),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9,
                    ha="left",
                    va="bottom",
                )

        outer = max(radii)
        pad = 0.15 * outer if outer > 0.0 else 1.0
        ax.set_xlim(-outer - pad, outer + pad)
        ax.set_ylim(-outer - pad, outer + pad)
        ax.set_aspect("equal", adjustable="box")
        unit_label = f" ({unit_expression(units)})" if units is not None else ""
        ax.set_xlabel(f"Local x radius{unit_label}")
        ax.set_ylabel(f"Local y radius{unit_label}")
        if title is None:
            title = self.name
            if z is not None:
                title = f"{title} at z={self._length_value(z, depth_units or units):g}"
        ax.set_title(title)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0))
        if show:
            plt.show()
        return ax
