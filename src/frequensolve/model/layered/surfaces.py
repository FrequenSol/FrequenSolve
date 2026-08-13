"""Surface, layer, and fracture objects used by layered models.

These classes are the user-visible building blocks that bound stratigraphic
intervals, attach material properties to those intervals, and represent thin
fracture subdomains in the serialized solver model.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.model import ModelSubdomain
from frequensolve.model.property import Property
from frequensolve.units import is_quantity, unit_expression
from frequensolve.util.mixins import ExportContext, merge_extra

from ._utils import (
    _convert_surface_depth,
    _dataarray_with_property_metadata,
    _inline_dataarray_from_fs,
    _inline_dataarray_to_fs,
)

__all__ = [
    "SimpleSurface",
    "Layer",
    "LayerBounds",
    "Fracture",
    "dipping_plane_2d",
    "dipping_plane_3d",
]


def _angle_degrees(value: Any) -> float:
    if is_quantity(value):
        return float(value.to("degree").magnitude)
    return float(value)


def _sequence_length_units(value: Any) -> Optional[str]:
    if is_quantity(value):
        return unit_expression(value.units)
    if isinstance(value, xr.DataArray):
        units = value.attrs.get("units")
        return unit_expression(units) if units is not None else None
    if isinstance(value, Mapping):
        units = value.get("units")
        if units is not None:
            return unit_expression(units)
        value = value.get("value", value)
    if isinstance(value, (str, bytes)):
        return None
    try:
        iterator = iter(value)
    except TypeError:
        return None
    for item in iterator:
        units = _sequence_length_units(item)
        if units is not None:
            return units
    return None


def _coerce_length_values(value: Any, units: Optional[str]) -> np.ndarray:
    if isinstance(value, Mapping) and "value" in value:
        local_units = value.get("units")
        values = value["value"]
        if local_units is not None:
            values = np.asarray(values, dtype=float)
            source_units = unit_expression(local_units)
            if units is not None and source_units != units:
                from frequensolve.units import ureg

                values = (values * ureg(source_units)).to(units).magnitude
            return np.asarray(values, dtype=float)
        value = values

    if is_quantity(value):
        quantity = value.to(units) if units is not None else value
        return np.asarray(quantity.magnitude, dtype=float)

    if isinstance(value, xr.DataArray):
        array_source_units = value.attrs.get("units")
        values = np.asarray(value.values, dtype=float)
        if array_source_units is not None and units is not None:
            array_source_units = unit_expression(array_source_units)
            if array_source_units != units:
                from frequensolve.units import ureg

                values = (values * ureg(array_source_units)).to(units).magnitude
        return np.asarray(values, dtype=float)

    if (
        isinstance(value, (list, tuple))
        and value
        and any(is_quantity(item) for item in value)
    ):
        target_units = units or _sequence_length_units(value)
        return np.asarray(
            [
                (
                    float(item.to(target_units).magnitude)
                    if is_quantity(item)
                    else float(item)
                )
                for item in value
            ],
            dtype=float,
        )

    return np.asarray(value, dtype=float)


def _coerce_point(
    point: Union[Mapping[str, Any], ArrayLike],
    names: tuple[str, ...],
    units: Optional[str],
) -> np.ndarray:
    if isinstance(point, Mapping):
        values = [point[name] for name in names]
    else:
        converted = _coerce_length_values(point, units).reshape(-1)
        if converted.size != len(names):
            joined = ", ".join(names)
            raise ValueError(f"Plane point must contain ({joined})")
        return converted
    if len(values) != len(names):
        joined = ", ".join(names)
        raise ValueError(f"Plane point must contain ({joined})")
    return _coerce_length_values(values, units).reshape(-1)


def _axis_dataarray(
    values: Any,
    *,
    dim: str,
    units: Optional[str],
) -> xr.DataArray:
    coords = _coerce_length_values(values, units).reshape(-1)
    coord = xr.DataArray(coords, dims=[dim])
    if units is not None:
        coord.attrs["units"] = units
    return coord


def dipping_plane_2d(
    point: Union[Mapping[str, Any], ArrayLike],
    dip: Any,
    x: Any,
    *,
    units: Optional[Any] = None,
    name: Optional[str] = None,
) -> xr.DataArray:
    """Return a 2D dipping graph surface ``z = f(x)``.

    Args:
        point: Point on the plane as ``(x, z)`` or ``{"x": ..., "z": ...}``.
        dip: Dip angle in degrees, positive for increasing depth in ``+x``.
        x: One-dimensional x coordinates used to sample the plane.
        units: Optional length units for coordinates and returned depth.
        name: Optional ``xarray.DataArray`` name.

    Returns:
        A one-dimensional ``DataArray`` with dimension ``"x"``.
    """

    target_units = unit_expression(units) if units is not None else None
    target_units = (
        target_units or _sequence_length_units(point) or _sequence_length_units(x)
    )
    point_values = _coerce_point(point, ("x", "z"), target_units)
    x_coord = _axis_dataarray(x, dim="x", units=target_units)
    z = point_values[1] + math.tan(math.radians(_angle_degrees(dip))) * (
        x_coord.values - point_values[0]
    )
    out = xr.DataArray(
        z,
        dims=["x"],
        coords={"x": x_coord},
        name=name or "dipping_plane",
    )
    if target_units is not None:
        out.attrs["units"] = target_units
    return out


def dipping_plane_3d(
    point: Union[Mapping[str, Any], ArrayLike],
    dip: Any,
    strike: Any,
    x: Any,
    y: Any,
    *,
    units: Optional[Any] = None,
    name: Optional[str] = None,
) -> xr.DataArray:
    """Return a 3D dipping graph surface ``z = f(x, y)``.

    Strike is measured clockwise from the positive y-axis in the x-y plane.
    The down-dip direction is strike plus 90 degrees; positive dip increases
    depth along that direction.

    Args:
        point: Point on the plane as ``(x, y, z)`` or a mapping.
        dip: Dip angle in degrees.
        strike: Strike azimuth in degrees clockwise from positive y.
        x: One-dimensional x coordinates used to sample the plane.
        y: One-dimensional y coordinates used to sample the plane.
        units: Optional length units for coordinates and returned depth.
        name: Optional ``xarray.DataArray`` name.

    Returns:
        A two-dimensional ``DataArray`` with dimensions ``("x", "y")``.
    """

    target_units = unit_expression(units) if units is not None else None
    target_units = (
        target_units
        or _sequence_length_units(point)
        or _sequence_length_units(x)
        or _sequence_length_units(y)
    )
    point_values = _coerce_point(point, ("x", "y", "z"), target_units)
    x_coord = _axis_dataarray(x, dim="x", units=target_units)
    y_coord = _axis_dataarray(y, dim="y", units=target_units)
    X, Y = np.meshgrid(x_coord.values, y_coord.values, indexing="ij")

    dip_direction = math.radians(_angle_degrees(strike) + 90.0)
    dip_vector = np.array([math.sin(dip_direction), math.cos(dip_direction)])
    offset = (X - point_values[0]) * dip_vector[0] + (Y - point_values[1]) * dip_vector[
        1
    ]
    z = point_values[2] + math.tan(math.radians(_angle_degrees(dip))) * offset

    out = xr.DataArray(
        z,
        dims=["x", "y"],
        coords={"x": x_coord, "y": y_coord},
        name=name or "dipping_plane",
    )
    if target_units is not None:
        out.attrs["units"] = target_units
    return out


@dataclass(kw_only=True)
class SimpleSurface:
    """Depth surface used to bound layered-model intervals.

    Args:
        name: Surface name used in serialized model payloads and references.
        interface: Whether this surface separates adjacent stratigraphic
            layers.
        cutting: Whether this surface truncates the model below it. Cutting is
            a meshing role independent of ``interface``.
        depth: Surface depth as a scalar, Pint quantity, file reference,
            serialized property payload, or ``xarray.DataArray``.
        grid: Optional grid metadata used for file-backed or ungridded depth
            values.
        scale: Multiplicative scale applied to loaded depth values.
        units: Optional depth units.
        system: Optional coordinate-system name for depth coordinates.
        **kwargs: Extra arguments are rejected; legacy ``xarr`` is accepted as
            a grid alias and ``frame`` raises a clear error.

    Raises:
        TypeError: If unsupported keyword arguments are supplied.
        ValueError: If ``depth`` is missing or cannot be coerced to a property.
    """

    name: str = "surface"
    interface: bool = True
    cutting: bool = False
    depth: Property = field(default_factory=Property)

    def __init__(
        self,
        name: str,
        interface: bool,
        depth: Optional[
            Union[float, str, Path, xr.DataArray, Mapping[str, Any]]
        ] = None,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        cutting: bool = False,
        **kwargs: Any,
    ) -> None:
        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")
        if "frame" in kwargs:
            raise TypeError(
                "SimpleSurface frame is no longer supported; surface coordinates are physical"
            )
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected SimpleSurface arguments: {unexpected}")
        if depth is None:
            raise ValueError("SimpleSurface requires depth")

        self.name = name
        self.interface = interface
        self.cutting = bool(cutting)
        self.depth = Property(
            data=cast(Any, depth),
            grid=grid,
            scale=scale,
            units=units,
            system=system,
        )

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SimpleSurface":
        """Deserialize a simple surface from a solver payload.

        Args:
            data: Mapping containing ``name``, ``interface``, and ``depth``.

        Returns:
            A ``SimpleSurface`` with depth metadata restored from the payload.
        """

        data = copy.deepcopy(dict(data))
        name = data["name"]
        interface = data["interface"]
        depth = data.get("depth", None)
        grid = None
        if isinstance(depth, dict) and "grid" in depth:
            try:
                grid = CartesianGrid.from_fs(depth["grid"]).as_xarray()
            except Exception:
                grid = None

        return cls(
            name=name,
            interface=interface,
            depth=depth,
            cutting=data.get("cutting", False),
            grid=grid,
        )

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize this surface and its depth property.

        Args:
            ctx: Optional export context used for project-relative paths or
                HDF5-backed storage.

        Returns:
            Solver-ready surface payload.
        """

        data: Dict[str, Any] = {
            "name": self.name,
            "interface": self.interface,
            **({"cutting": True} if self.cutting else {}),
        }
        ctx = ctx or ExportContext()
        use_store = getattr(ctx, "store", None) is not None
        file = (
            None
            if self.depth.is_constant or use_store or ctx.path is None
            else ctx.path / (self.name + ".bin")
        )
        dataset = f"inputs/model/surfaces/{self.name}/depth"
        data["depth"] = self.depth.to_fs(
            ctx=ctx,
            file=file,
            dataset=dataset,
        )
        return data

    @property
    def data(self) -> Optional[xr.DataArray]:
        """Return the materialized surface depth data, when available.

        Returns:
            The underlying ``xarray.DataArray`` or ``None`` for lazy file
            references.
        """

        return self.depth.data

    @property
    def extrema(self) -> Tuple[Any, Any]:
        """Return minimum and maximum surface depths.

        Returns:
            Tuple ``(minimum, maximum)`` containing scalar depth values.

        Raises:
            ValueError: If the depth property is lazy or otherwise not
                materialized.
        """
        min, max = self.depth.extrema
        min = min.values
        max = max.values
        return min, max

    def perturb(
        self,
        std: float,
        grid: Optional[xr.DataArray] = None,
        L0: Union[float, List[float]] = 1.0,
        nu: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Apply an additive von Karman perturbation to surface depth.

        Args:
            std: Standard deviation of the perturbation.
            grid: Optional grid used to generate the stochastic field.
            L0: Characteristic length scale or one value per dimension.
            nu: Smoothness parameter passed to the stochastic-field generator.
            seed: Optional random seed for reproducible perturbations.
            **kwargs: Legacy aliases. ``xarr`` is accepted as a grid alias.

        Raises:
            ValueError: If the depth property is file-backed or otherwise not
                materialized.
        """

        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")

        if isinstance(L0, float):
            L0 = [L0]
        k0 = []
        for l0 in L0:
            k0.append(1 / l0)
        self.depth.stochastic_perturbation(
            std=std,
            method="von_karman",
            grid=grid,
            k0=k0,
            nu=nu,
            seed=seed,
            type="additive",
        )

    def plot(self, limits: Dict[str, ArrayLike], **kwargs: Any) -> None:
        """Plot the surface over the provided coordinate limits.

        Args:
            limits: Coordinate values or limits used to construct a plotting
                grid for constant surfaces.
            **kwargs: Matplotlib plotting options. ``ax`` may provide existing
                axes and ``units`` may request display units for depth.

        Returns:
            ``None``. The method draws on the supplied or newly-created axes.

        Raises:
            ModuleNotFoundError: Converted to an optional-dependency error when
                matplotlib is not installed.
        """

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            from frequensolve._optional import optional_dependency_error

            raise optional_dependency_error(
                "Surface plotting",
                extra="visual",
                dependencies=("matplotlib",),
                error=exc,
            ) from exc

        if self.depth.is_constant:
            dims = sorted(limits.keys())
            grid = xr.DataArray(dims=dims, coords=limits)
            surf = self.depth.get(grid=grid)
        else:
            surf = self.depth.get()
        units = kwargs.pop("units", None)
        surf = _convert_surface_depth(
            surf,
            self,
            unit_expression(units) if units is not None else None,
        )

        show = True
        if len(surf.dims) == 1:
            if "ax" in kwargs:
                ax = kwargs.pop("ax")
                show = False
            else:
                fig = plt.figure()
                ax = fig.gca()
            ax.plot(surf.coords["x"].values, surf.values, **kwargs)
        elif len(surf.dims) == 2:
            x = surf.coords["x"].values
            y = surf.coords["y"].values
            z = surf.values

            if "ax" in kwargs:
                ax = kwargs.pop("ax")
                show = False
            else:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")

            surf = ax.plot_surface(x, y, z, **kwargs)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        if show:
            plt.show()


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class Layer(ModelSubdomain):
    """Material layer subdomain bounded by upper and lower model surfaces.

    Args:
        *args: Positional arguments forwarded to ``ModelSubdomain``.
        **kwargs: Keyword arguments forwarded to ``ModelSubdomain``.

    Attributes:
        upper: Upper bounding ``SimpleSurface`` once the layer is attached to a
            ``LayeredModel``.
        lower: Lower bounding ``SimpleSurface`` once the layer is attached to a
            ``LayeredModel``.
    """

    upper: Optional[SimpleSurface]
    lower: Optional[SimpleSurface]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.lower = None
        self.upper = None

    def perturb(
        self,
        property: Union[str, List[str]],
        std: float,
        L0: Union[float, List[float]] = 1.0,
        nu: float = 0.5,
        anisotropy: Optional[List[float]] = None,
        grid: Optional[xr.DataArray] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Apply multiplicative stochastic perturbations to layer properties.

        Args:
            property: Property name or list of property names to perturb.
            std: Standard deviation of the stochastic field.
            L0: Characteristic length scale or one value per dimension.
            nu: Smoothness parameter passed to the stochastic-field generator.
            anisotropy: Optional anisotropic stretching factor per dimension.
            grid: Optional grid used to generate the perturbation. Defaults to
                the layer's property grid.
            seed: Optional random seed for reproducible perturbations.
            **kwargs: Legacy aliases. ``xarr`` is accepted as a grid alias.

        Raises:
            ValueError: If a named property is missing or cannot be perturbed
                because it is file-backed.
        """

        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")

        if grid is None:
            grid = self.properties.grid

        if isinstance(property, str):
            property_names = [property]
        else:
            property_names = property

        if isinstance(L0, float):
            L0 = [L0]
        k0 = []
        for l0 in L0:
            k0.append(1 / l0)

        for name in property_names:
            if name not in self.properties:
                raise ValueError(f"Property '{name}' not found in layer")

            if anisotropy is None:
                anisotropy = [1.0] * len(self.properties[name].data.dims)
            self.properties[name].stochastic_perturbation(
                std=std,
                method="von_karman",
                grid=grid,
                k0=k0,
                nu=nu,
                anisotropy=anisotropy,
                seed=seed,
                type="multiplicative",
            )

    def set_property(self, key: str, value: Union[float, xr.DataArray]) -> None:
        """Set or replace one material property on the layer.

        Args:
            key: Property name or alias.
            value: Property-like value accepted by ``Property``.
        """

        self.properties[key] = value


# Helper class for LayeredModel
@dataclass(kw_only=True, slots=True)
class LayerBounds:
    """Bounding-surface summary for one material layer.

    Args:
        upper: Optional upper bounding surface.
        lower: Optional lower bounding surface.
        layer: Layer whose bounds are being described.
    """

    upper: Optional[SimpleSurface] = None
    lower: Optional[SimpleSurface] = None
    layer: Layer

    def __str__(self) -> str:
        out = f"Layer {self.layer.name}\n"

        if self.upper is not None:
            out += f"  upper: {self.upper.name}{self._depth_label(self.upper)}\n"
        else:
            out += "  upper: None\n"

        if self.lower is not None:
            out += f"  lower: {self.lower.name}{self._depth_label(self.lower)}\n"
        else:
            out += "  lower: None\n"
        return out

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def _depth_label(surface: SimpleSurface) -> str:
        try:
            if surface.depth.is_constant:
                return f" (depth = {surface.depth.get()})"
        except Exception:
            pass
        return ""


@dataclass
class Fracture(SimpleSurface):
    """Curve-based fracture geometry opened by the layered mesh generator.

    A fracture is defined by a center depth and a one-dimensional aperture/gap
    curve. The mesher can open that curve into an explicit fracture domain with
    ``mesh_block_id``; later reduced fracture models can use the same geometry.

    Args:
        name: Fracture surface name.
        depth: Fracture center depth as a scalar, Pint quantity, property
            payload, file reference, or ``xarray.DataArray``.
        gap: One-dimensional aperture curve.
        grid: Optional grid metadata for ``depth`` and ``gap``.
        units: Optional units for geometry values.
        system: Optional coordinate-system name for geometry coordinates.
        interface: Whether the fracture participates in layer ordering.
        physics: Optional physics/material family for a generated fracture
            subdomain.
        properties: Optional material properties for the generated fracture
            subdomain.
        property_grid: Default grid metadata for fracture material properties.
        property_units: Default units for fracture material properties.
        property_system: Coordinate-system name for fracture material
            properties.
        mesh_block_id: Optional positive mesh-block id for the opened fracture
            interval.
        subdomain_name: Optional name for the generated material subdomain.
        **kwargs: Extra serialized fracture fields.

    Raises:
        ValueError: If ``mesh_block_id`` is not positive or ``gap`` is not
            one-dimensional.
    """

    name: str
    gap: Property = field(default_factory=Property)
    physics: Optional[str] = None
    properties: Optional[dict] = None
    property_grid: Optional[xr.DataArray] = None
    property_units: Optional[Any] = None
    property_system: Optional[str] = None
    subdomain_name: Optional[str] = None
    mesh_block_id: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        depth: Any,
        gap: Union[xr.DataArray, Mapping[str, Any], ArrayLike],
        *,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        interface: bool = True,
        physics: Optional[str] = None,
        properties: Optional[dict] = None,
        property_grid: Optional[xr.DataArray] = None,
        property_units: Optional[Any] = None,
        property_system: Optional[str] = None,
        mesh_block_id: Optional[int] = None,
        subdomain_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            interface=interface,
            depth=depth,
            grid=grid,
            units=units,
            system=system,
        )
        self.mesh_block_id = None if mesh_block_id is None else int(mesh_block_id)
        if self.mesh_block_id is not None and self.mesh_block_id <= 0:
            raise ValueError("Fracture mesh_block_id must be positive")
        self.gap = self._coerce_gap(gap, grid=grid, units=units, system=system)
        self.physics = physics
        self.properties = copy.deepcopy(properties)
        self.property_grid = property_grid
        self.property_units = property_units
        self.property_system = property_system
        self.subdomain_name = subdomain_name
        self.extra = dict(kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Fracture":
        """Deserialize a fracture surface from a solver payload.

        Args:
            data: Serialized fracture mapping containing ``depth`` and ``gap``.

        Returns:
            A ``Fracture`` with geometry and optional metadata restored.
        """

        payload = copy.deepcopy(dict(data))
        payload.pop("_type", None)
        payload.pop("type", None)
        return cls(
            name=payload.pop("name"),
            mesh_block_id=payload.pop("mesh_block_id", None),
            depth=payload.pop("depth"),
            gap=payload.pop("gap"),
            interface=payload.pop("interface", True),
            **payload,
        )

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize fracture geometry, aperture, and optional mesh metadata.

        Args:
            ctx: Optional export context used for project-relative paths or
                HDF5-backed storage.

        Returns:
            Solver-ready fracture surface payload.
        """

        payload = {
            "_type": self.__class__.__name__,
            "name": self.name,
            "interface": self.interface,
            "depth": self._property_to_fs(self.depth, field="depth", ctx=ctx),
            "gap": self._property_to_fs(self.gap, field="gap", ctx=ctx),
        }
        if self.mesh_block_id is not None:
            payload["mesh_block_id"] = self.mesh_block_id
        return merge_extra(payload, self.extra, "Fracture")

    @staticmethod
    def _coerce_gap(
        value: Union[Property, xr.DataArray, Mapping[str, Any], ArrayLike],
        *,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> Property:
        if isinstance(value, Property):
            prop = copy.deepcopy(value)
        elif isinstance(value, Mapping) and {"dims", "data"}.issubset(value):
            prop = Property(_inline_dataarray_from_fs(value))
        else:
            prop = Property(cast(Any, value), grid=grid, units=units, system=system)
        if prop.darr is not None and not prop.is_constant and prop.darr.ndim != 1:
            raise ValueError("Fracture gap must be one-dimensional")
        return prop

    def _property_to_fs(
        self,
        prop: Property,
        *,
        field: str,
        ctx: Optional[ExportContext] = None,
    ) -> Dict[str, Any]:
        ctx = ctx or ExportContext()
        dataset = f"inputs/model/surfaces/{self.name}/{field}"
        use_store = getattr(ctx, "store", None) is not None
        if (
            prop.darr is not None
            and not prop.is_constant
            and prop.file_path is None
            and not use_store
        ):
            return _inline_dataarray_to_fs(_dataarray_with_property_metadata(prop))
        file = None
        if not prop.is_constant and not use_store and ctx.path is not None:
            file = ctx.path / f"{self.name}_{field}.bin"
        return prop.to_fs(ctx=ctx, file=file, dataset=dataset)


def _is_fracture_surface_payload(data: Mapping[str, Any]) -> bool:
    type_name = data.get("_type", data.get("type"))
    return "gap" in data or str(type_name).lower() in {
        "fracture",
        "fracture_surface",
        "fracturesurface",
    }


def _model_surface_from_fs(data: Mapping[str, Any]) -> Union[SimpleSurface, Fracture]:
    return (
        Fracture.from_fs(data)
        if _is_fracture_surface_payload(data)
        else SimpleSurface.from_fs(data)
    )
