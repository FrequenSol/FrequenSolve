import copy
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from frequensolve.geometry.frame import Axis
from frequensolve.geometry.grids import CartesianGrid
from frequensolve.mesh.mesh_generators import HexMeshGenerator, TetMeshGenerator
from frequensolve.model.model import ModelBase, ModelSubdomain
from frequensolve.model.property import Property, canonical_property_name
from frequensolve.units import unit_expression, ureg, value_and_units_to_fs
from frequensolve.util.class_registry import register_class
from frequensolve.util.mixins import merge_extra
from frequensolve.util.named_list import NamedList
from frequensolve.util.physics import model_dimension

__all__ = ["SimpleSurface", "Layer", "BoreholePart", "Borehole", "LayeredModel"]

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _property_units(prop: Property) -> Optional[str]:
    if prop.units:
        return unit_expression(prop.units)
    if prop.data is not None and hasattr(prop.data, "attrs"):
        units = prop.data.attrs.get("units")
        if units:
            return unit_expression(units)
    return None


def _axis_units(units: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    units = dict(units or {})
    out = {"x": None, "y": None, "z": None}
    for key, value in units.items():
        out[key] = unit_expression(value) if value is not None else None
    return out


def _convert_units(
    value: float, source_units: Optional[str], target_units: Optional[str]
):
    if not source_units or not target_units or source_units == target_units:
        return value
    return (value * ureg(source_units)).to(target_units).magnitude


def _convert_surface_value(
    value: float,
    surface: "SimpleSurface",
    target_units: Optional[str],
) -> float:
    return _convert_units(value, _property_units(surface.depth), target_units)


def _convert_dataarray_units(
    data: xr.DataArray,
    source_units: Optional[str],
    target_units: Optional[str],
) -> xr.DataArray:
    if target_units is None:
        if source_units is not None:
            data = data.copy(deep=True)
            data.attrs["units"] = source_units
        return data
    if source_units is None or source_units == target_units:
        data = data.copy(deep=True)
        data.attrs["units"] = target_units
        return data
    converted = data.copy(deep=True)
    converted.data = (converted.data * ureg(source_units)).to(target_units).magnitude
    converted.attrs["units"] = target_units
    return converted


def _convert_surface_depth(
    depth: xr.DataArray,
    surface: "SimpleSurface",
    target_units: Optional[str],
) -> xr.DataArray:
    return _convert_dataarray_units(depth, _property_units(surface.depth), target_units)


# ----------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class SimpleSurface:
    """A base class for defining surfaces in a seismic model.

    Attributes:
       name (str):
          Name identifier for the surface.
       interface (bool):
          Whether this is an interface surface.
       depth (Property):
          Surface depth values.
    """

    name: str = "surface"
    interface: bool = True
    depth: Property = field(default_factory=Property)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __init__(
        self,
        name: str,
        interface: bool,
        depth: Optional[Union[float, str, Path, xr.DataArray, Dict[str, Any]]] = None,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        **kwargs,
    ):
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
        self.depth = Property(
            data=depth,
            grid=grid,
            scale=scale,
            units=units,
            system=system,
        )

    @classmethod
    def from_fs(cls, data: Dict) -> "SimpleSurface":
        data = copy.deepcopy(data)
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
            **({"grid": grid} if grid is not None else {}),
        )

    def to_fs(self, ctx=None):
        data = {
            "name": self.name,
            "interface": self.interface,
        }
        ctx = ctx or self.export_context()
        use_store = getattr(ctx, "store", None) is not None
        file = (
            None
            if self.depth.is_constant or use_store
            else self._path / (self.name + ".bin")
        )
        dataset = f"inputs/model/surfaces/{self.name}/depth"
        data["depth"] = self.depth.to_fs(
            ctx=ctx,
            file=file,
            dataset=dataset,
        )
        return data

    def export_context(self):
        from frequensolve.util.mixins import ExportContext

        return ExportContext(self._proj_path, self._rel_path)

    @property
    def data(self):
        return self.depth.data

    @property
    def extrema(self):
        """Get the extreme values (min, max) of the surface.

        Returns:
           tuple: A tuple containing the minimum and maximum depths.
        """
        min, max = self.depth.extrema
        min = min.values
        max = max.values
        return min, max

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path

    @property
    def _path(self) -> Path:
        if self._proj_path is None or self._rel_path is None:
            raise ValueError("Surface is not attached to a project path")
        return self._proj_path / self._rel_path

    def perturb(
        self,
        std: float,
        grid: Optional[xr.DataArray] = None,
        L0: Union[float, List[float]] = 1.0,
        nu: float = 1.0,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Perturb the dataset by a von Karman stochastic field.

        std (float):
           Standard deviation of the perturbation
        grid (xr.DataArray):
           Xarray defining the grid for the perturbation
        L0 (float):
           Characteristic length scale of the perturbation
        nu (float):
           Stochastic field smoothness parameter
           (nu -> 0: less smooth, nu -> 1 more smoother)
        seed (int):
           Random seed (for reproducibility)
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

    def plot(self, limits: Dict[str, ArrayLike], **kwargs):
        """Plot the surface."""
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

    def __init__(self, *args, **kwargs):
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
        **kwargs,
    ) -> None:
        """Perturb the dataset by a Von Karmanstochastic field.

        This just wraps the stochastic_perturbation method of Property.

        Args:
           property (str):
              Name of the property to perturb
           std (float):
              Standard deviation of the perturbation
           grid (xr.DataArray):
              Xarray defining the grid for the perturbation
           L0 (float):
              Characteristic length scale of the perturbation
           nu (float):
              Stochastic field smoothness parameter
              (nu -> 0: less smooth, nu -> 1 more smoother)
           anisotropy (List[float]):
              Anisotropic stretching factor for each dimension
           seed (int):
              Random seed (for reproducibility)
        """

        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")

        if grid is None:
            grid = self.grid

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

    def set_property(self, key: str, value: Union[float, xr.DataArray]):
        self.properties[key] = value


# Helper class for LayeredModel
@dataclass(kw_only=True, slots=True)
class LayerBounds:
    """Defines the bounding surfaces of a model layer.

    Attributes:
       upper (Surface): The upper bounding surface.
       lower (Surface): The lower bounding surface.
       layer_id (int):  Unique identifier for the layer.
    """

    upper: Optional[SimpleSurface] = None
    lower: Optional[SimpleSurface] = None
    layer: Layer

    def __str__(self):
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

    def __repr__(self):
        return str(self)

    @staticmethod
    def _depth_label(surface: SimpleSurface) -> str:
        try:
            if surface.depth.is_constant:
                return f" (depth = {surface.depth.get()})"
        except Exception:
            pass
        return ""


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
    if value.ndim != 1:
        raise ValueError("Borehole radius profiles must be one-dimensional")
    dim = str(value.dims[0])
    if dim not in {"z", "depth"}:
        raise ValueError("Borehole radius profile dimension must be 'z' or 'depth'")
    if dim not in value.coords:
        raise ValueError(
            "Borehole radius profiles require a coordinate for their dimension"
        )


def _borehole_radius_to_fs(
    radius: Property,
    ctx=None,
    *,
    borehole_name: Optional[str] = None,
    part_name: Optional[str] = None,
) -> Any:
    _validate_borehole_radius_profile(radius)
    if (
        ctx is not None
        and getattr(ctx, "store", None) is not None
        and borehole_name is not None
        and part_name is not None
        and radius.darr is not None
        and not radius.is_constant
    ):
        dataset = f"inputs/model/boreholes/{borehole_name}/parts/{part_name}/r"
        return radius.to_fs(ctx=ctx, dataset=dataset)

    if (
        radius.darr is not None
        and not radius.is_constant
        and radius.file_path is None
        and ctx is not None
        and getattr(ctx, "path", None) is not None
        and borehole_name is not None
        and part_name is not None
    ):
        file = ctx.path / "boreholes" / borehole_name / f"{part_name}_r.bin"
        return radius.to_fs(ctx=ctx, file=file)

    if radius.darr is None or radius.is_constant or radius.file_path is not None:
        return radius.to_fs(ctx=ctx)

    value = radius.darr
    payload: Dict[str, Any] = {
        "value": value.values.tolist(),
        "dims": list(value.dims),
        "coords": {},
    }
    for dim in value.dims:
        coord = value.coords[dim]
        coord_payload = {"value": coord.values.tolist()}
        if coord.attrs.get("units"):
            coord_payload["units"] = unit_expression(coord.attrs["units"])
        payload["coords"][dim] = coord_payload
    units = radius.units or value.attrs.get("units")
    if units:
        payload["units"] = unit_expression(units)
    if radius.system is not None:
        payload["system"] = radius.system
    return payload


@dataclass(kw_only=True)
class BoreholePart:
    """One concentric radial material part of a 2D borehole.

    ``r`` is the cumulative outer radius for this part. The first part starts
    at ``r = 0``; each following part starts at the previous part's ``r``.
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
        payload = {
            "name": self.name,
            "mesh_block_id": self.mesh_block_id,
            "r": _borehole_radius_to_fs(
                self.r,
                ctx,
                borehole_name=borehole_name,
                part_name=self.name,
            ),
        }
        return merge_extra(payload, self.extra, "BoreholePart")


@dataclass(kw_only=True)
class Borehole:
    """Vertical 2D borehole geometry for a layered model."""

    name: str
    axis: Dict[str, Any]
    extent: Dict[str, Any]
    parts: NamedList = field(default_factory=NamedList)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        axis: Mapping[str, Any],
        extent: Mapping[str, Any],
        parts: Optional[List[Union[BoreholePart, Mapping[str, Any]]]] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        self.name = name
        self.axis = copy.deepcopy(dict(axis))
        self.extent = copy.deepcopy(dict(extent))
        self.parts = NamedList()
        for part in parts or []:
            self.parts.append(
                part if isinstance(part, BoreholePart) else BoreholePart.from_fs(part)
            )
        self.extra = dict(extra or {})
        self.extra.update(kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Borehole":
        payload = copy.deepcopy(dict(data))
        axis = payload.pop("axis", None)
        x = payload.pop("x", None)
        if axis is None:
            if x is None:
                raise ValueError("Borehole requires axis/x")
            axis = {"x": x}
        elif x is not None:
            raise ValueError("Specify either borehole axis or x, not both")
        return cls(
            name=payload.pop("name"),
            axis=axis,
            extent=payload.pop("extent"),
            parts=payload.pop("parts", []),
            extra=payload,
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
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
            "parts": [part.to_fs(ctx, borehole_name=self.name) for part in self.parts],
        }
        return merge_extra(payload, self.extra, "Borehole")

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
        """Return the borehole axis x-coordinate."""

        if "x" not in self.axis:
            raise ValueError("Borehole requires axis/x")
        return self._length_value(self.axis["x"], units)

    def radius_profile(
        self,
        part: BoreholePart,
        z: Optional[Any] = None,
        *,
        units: Optional[Any] = None,
        depth_units: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return depth and cumulative outer-radius arrays for one part."""

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
        """Evaluate one part's cumulative radius at a depth."""

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
        """Draw a borehole radial profile as concentric material circles."""

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


@register_class
@dataclass(kw_only=True)
class LayeredModel(ModelBase):
    """A class for representing a layered seismic model.

    Attributes:
       name (str):                         The name of the model.
       dimension (int):                    Dimension of the model.
       subdomains (List[ModelSubdomain]):
          All solver material subdomains.
       layers (List[Layer]):
          Stratigraphic layer intervals bounded by model surfaces.
       surfaces (List[Surface]):           Model (simple) surfaces.
       x_limits (List[float]):             X-limits of the model.
       y_limits (Optional[List[float]]):   Y-limits of the model. (3D only)
       ordering (Literal["top_down", "bottom_up"]):
          Ordering of layers. Defaults to "top_down".

       Note:
          A 'surface' must be added at the top and bottom of the model, and at
          least one surface must be added between 'layers'.
    """

    x_limits: List[float]
    y_limits: Optional[List[float]] = None
    surfaces: NamedList = field(default_factory=NamedList)
    boreholes: NamedList = field(default_factory=NamedList)
    ordering: Literal["top_down", "bottom_up"] = "top_down"
    extra: Dict[str, Any] = field(default_factory=dict)

    _last_added: str = "none"
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None
    _surface_names: Set[str] = field(default_factory=set)
    _layer_names: Set[str] = field(default_factory=set)
    _borehole_names: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.dimension = model_dimension(self.dimension)
        if self.dimension not in {2, 3}:
            raise ValueError("LayeredModel dimension must be 2 or 3")
        if len(self.x_limits) != 2:
            raise ValueError("LayeredModel x_limits must contain [min, max]")
        if self.x_limits[1] <= self.x_limits[0]:
            raise ValueError("LayeredModel x_limits must be increasing")
        if self.dimension == 3:
            if self.y_limits is None or len(self.y_limits) != 2:
                raise ValueError("3D LayeredModel requires y_limits=[min, max]")
            if self.y_limits[1] <= self.y_limits[0]:
                raise ValueError("LayeredModel y_limits must be increasing")
        elif self.y_limits is not None:
            raise ValueError("2D LayeredModel does not accept y_limits")

    def add_layer(
        self,
        name: Optional[str] = None,
        mesh_block_id: int = -1,
        physics: Optional[str] = None,
        properties: Optional[dict] = None,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a layer to the model.

        Args:
           name (str):
              Name of the layer.
           mesh_block_id (int):
              Unique identifier for the layer.
           physics (str, optional):
              Optional material physics family or variant for this layer.
           properties (dict):
              Properties of the layer.
           grid (xr.DataArray, optional):
              Xarray defining the grid for the layer (required for reading file formats
              where grid is not stored with data)
           units:
              Optional default units for layer properties. Pint quantities are
              also accepted directly in ``properties``.
           system:
              Optional coordinate-system name for layer property grids. The
              legacy ``coordinate_system`` keyword is accepted as an alias.
        """

        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")
        if "frame" in kwargs:
            raise TypeError(
                "LayeredModel.add_layer frame is no longer supported; layer coordinates are physical"
            )
        coordinate_system = kwargs.pop("coordinate_system", None)
        if coordinate_system is not None:
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system

        name = self._get_unique_name(name, self._layer_names)
        self._layer_names.add(name)

        layer = Layer(
            name=name,
            mesh_block_id=mesh_block_id,
            physics=physics,
            properties=properties,
            grid=grid,
            units=units,
            system=system,
        )
        self += layer

    def add_surface(
        self,
        depth: Optional[Union[float, str, Path, xr.DataArray, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        **kwargs,
    ):
        """Add a surface to the model.

        Args:
           depth (float, str, Path, xr.DataArray):
              Depth of the surface. Can be defined as a float (constant surface),
              via a file, or as an xr.DataArray (gridded surface).
           name (str, optional):
              Name of the surface.
           grid (xr.DataArray, optional):
              Xarray defining the grid for the surface.
           units:
              Optional units for the surface values. Pint quantities are also
              accepted directly as ``depth``.
           system:
              Optional coordinate-system name for the surface values. The
              legacy ``coordinate_system`` keyword is accepted as an alias.
        """

        interface = len(self.surfaces) == 0

        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")
        if depth is None and "z" in kwargs:
            depth = kwargs.pop("z")
        if "frame" in kwargs:
            raise TypeError(
                "LayeredModel.add_surface frame is no longer supported; surface coordinates are physical"
            )
        coordinate_system = kwargs.pop("coordinate_system", None)
        if coordinate_system is not None:
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected LayeredModel.add_surface arguments: {unexpected}"
            )
        if depth is None:
            raise ValueError("LayeredModel.add_surface requires depth")

        name = self._get_unique_name(name, self._surface_names)
        self._surface_names.add(name)

        surface = SimpleSurface(
            name=name,
            interface=interface,
            depth=depth,
            grid=grid,
            scale=scale,
            units=units,
            system=system,
        )
        self += surface

    def add_borehole(
        self,
        name: Optional[str] = None,
        *,
        axis: Optional[Mapping[str, Any]] = None,
        x: Optional[Any] = None,
        y: Optional[Any] = None,
        top: Optional[Any] = None,
        bottom: Optional[Any] = None,
        parts: Optional[List[Union[BoreholePart, Mapping[str, Any]]]] = None,
        **kwargs,
    ) -> Borehole:
        """Add a 2D vertical borehole and any declared material subdomains.

        ``parts`` may be ``BoreholePart`` objects or dictionaries. A part
        dictionary can also include ``physics`` and ``properties``; those fields
        create the corresponding ``ModelSubdomain`` while the borehole part
        keeps only the geometry and mesh-block fields required by the solver.
        When a part does not include ``properties``, its ``mesh_block_id`` must
        already reference an existing model subdomain.
        """

        if "extent" in kwargs:
            if top is not None or bottom is not None:
                raise ValueError("Specify either extent or top/bottom, not both")
            extent = dict(kwargs.pop("extent"))
            top = extent.get("top")
            bottom = extent.get("bottom")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected LayeredModel.add_borehole arguments: {unexpected}"
            )
        if self.dimension != 2:
            raise ValueError(
                "LayeredModel.add_borehole currently supports 2D models only"
            )
        if not parts:
            raise ValueError("LayeredModel.add_borehole requires at least one part")

        if axis is not None:
            axis_payload = copy.deepcopy(dict(axis))
            if x is not None or y is not None:
                raise ValueError("Specify either axis or x/y coordinates, not both")
        else:
            axis_payload = {}
            if x is not None:
                axis_payload["x"] = x
            if y is not None:
                axis_payload["y"] = y
        if not axis_payload:
            raise ValueError("LayeredModel.add_borehole requires axis or x/y")
        if self.dimension == 2 and "y" in axis_payload:
            raise ValueError("2D boreholes accept x, not y")
        if "x" not in axis_payload:
            raise ValueError("2D boreholes require axis/x")

        if top is None:
            top = self.upper_surface()
        if bottom is None:
            bottom = self.lower_surface()

        name = self._get_unique_name(name or "borehole", self._borehole_names)
        self._borehole_names.add(name)

        next_block = self._next_mesh_block_id()
        borehole_parts = []
        material_subdomains = []
        for spec in parts or []:
            part, subdomain, next_block = self._coerce_borehole_part(
                name, spec, next_block
            )
            borehole_parts.append(part)
            if subdomain is not None:
                material_subdomains.append(subdomain)

        for subdomain in material_subdomains:
            if any(
                existing.mesh_block_id == subdomain.mesh_block_id
                for existing in self.subdomains
            ):
                raise ValueError(
                    f"Mesh block id {subdomain.mesh_block_id} is already used"
                )
            self.add_subdomain(subdomain)
        missing_domains = [
            part.mesh_block_id
            for part in borehole_parts
            if not any(
                subdomain.mesh_block_id == part.mesh_block_id
                for subdomain in self.subdomains
            )
        ]
        if missing_domains:
            missing = ", ".join(str(value) for value in missing_domains)
            raise ValueError(
                "Borehole part mesh_block_id must reference a model subdomain; "
                f"missing: {missing}"
            )

        borehole = Borehole(
            name=name,
            axis=axis_payload,
            extent={"top": top, "bottom": bottom},
            parts=borehole_parts,
        )
        self.boreholes.append(borehole)
        return borehole

    def _next_mesh_block_id(self) -> int:
        used = [subdomain.mesh_block_id for subdomain in self.subdomains]
        return max([value for value in used if value >= 0], default=0) + 1

    def _coerce_borehole_part(
        self,
        borehole_name: str,
        spec: Union[BoreholePart, Mapping[str, Any]],
        next_block: int,
    ) -> Tuple[BoreholePart, Optional[ModelSubdomain], int]:
        if isinstance(spec, BoreholePart):
            return spec, None, max(next_block, spec.mesh_block_id + 1)
        if not isinstance(spec, Mapping):
            raise TypeError("Borehole parts must be BoreholePart objects or mappings")

        payload = copy.deepcopy(dict(spec))
        physics = payload.pop("physics", None)
        properties = payload.pop("properties", None)
        grid = payload.pop("grid", None)
        units = payload.pop("units", None)
        system = payload.pop("system", None)
        subdomain_name = payload.pop("subdomain_name", None)
        if payload.get("mesh_block_id") is None:
            if properties is None:
                raise ValueError(
                    "Borehole parts without properties must provide mesh_block_id"
                )
            payload["mesh_block_id"] = next_block
        part = BoreholePart.from_fs(payload)
        next_block = max(next_block, part.mesh_block_id + 1)

        subdomain = None
        if properties is not None:
            subdomain = ModelSubdomain(
                mesh_block_id=part.mesh_block_id,
                name=subdomain_name or f"{borehole_name}_{part.name}",
                physics=physics,
                properties=properties,
                grid=grid,
                units=units,
                system=system,
            )
        return part, subdomain, next_block

    @property
    def surface_names(self):
        return [surf.name for surf in self.surfaces]

    @property
    def layer_names(self):
        return [layer.name for layer in self.layers]

    @property
    def borehole_names(self):
        return [borehole.name for borehole in self.boreholes]

    @property
    def layers(self):
        return NamedList(
            [subdomain for subdomain in self.subdomains if isinstance(subdomain, Layer)]
        )

    @property
    def z_limits(self):
        if len(self.surfaces) < 2:
            raise ValueError("LayeredModel requires at least two surfaces for z_limits")
        z0, _ = self.surfaces[0].extrema  # extrema returns values already
        _, z1 = self.surfaces[-1].extrema
        return float(z0), float(z1)

    def z_limits_in(self, units: Optional[Any] = None) -> Tuple[float, float]:
        """Return model z-limits converted to the requested display units."""

        if len(self.surfaces) < 2:
            raise ValueError("LayeredModel requires at least two surfaces for z_limits")
        target_units = unit_expression(units) if units is not None else None
        z0, _ = self.surfaces[0].extrema
        _, z1 = self.surfaces[-1].extrema
        return (
            float(_convert_surface_value(float(z0), self.surfaces[0], target_units)),
            float(_convert_surface_value(float(z1), self.surfaces[-1], target_units)),
        )

    def property_units(self, property: str) -> Optional[str]:
        """Return the first declared units for a model property."""

        property = canonical_property_name(property)
        for subdomain in self.subdomains:
            if property not in subdomain.properties:
                continue
            units = _property_units(subdomain.properties[property])
            if units:
                return units
        return None

    def convert_property_units(
        self,
        data: xr.DataArray,
        property: str,
        units: Optional[Any],
    ) -> xr.DataArray:
        """Convert sampled property data to requested display units."""

        target_units = unit_expression(units) if units is not None else None
        source_units = data.attrs.get("units") or self.property_units(property)
        if target_units is None:
            target_units = source_units
        return _convert_dataarray_units(data, source_units, target_units)

    def extreme_values(
        self,
        property: str,
        units: Optional[Any] = None,
    ) -> Tuple[float, float]:
        property = canonical_property_name(property)
        target_units = (
            unit_expression(units)
            if units is not None
            else self.property_units(property)
        )
        vmin = 1.0e8
        vmax = -1.0e8
        for subdomain in self.subdomains:
            if property not in subdomain.properties:
                continue
            prop = subdomain.properties[property]
            prop_min, prop_max = prop.extrema
            source_units = _property_units(prop)
            prop_min = float(prop_min)
            prop_max = float(prop_max)
            prop_min = float(_convert_units(prop_min, source_units, target_units))
            prop_max = float(_convert_units(prop_max, source_units, target_units))
            vmin = min(prop_min, vmin)
            vmax = max(prop_max, vmax)
        if vmin == 1.0e8:
            raise ValueError(f"Property '{property}' not found in any layer")
        return vmin, vmax

    def hex_mesh_generator(self, n: Optional[List[int]] = None) -> HexMeshGenerator:
        self._validate_bounds_for_meshing()
        if self.dimension == 2:
            l_bound = [self.x_limits[0], self.z_limits[0]]
            u_bound = [self.x_limits[1], self.z_limits[1]]
        else:
            l_bound = [self.x_limits[0], self.y_limits[0], self.z_limits[0]]
            u_bound = [self.x_limits[1], self.y_limits[1], self.z_limits[1]]

        return HexMeshGenerator(l_bound=l_bound, u_bound=u_bound, n=n)

    def tet_mesh_generator(self, n: Optional[List[int]] = None) -> TetMeshGenerator:
        self._validate_bounds_for_meshing()
        if self.dimension == 2:
            l_bound = [self.x_limits[0], self.z_limits[0]]
            u_bound = [self.x_limits[1], self.z_limits[1]]
        else:
            l_bound = [self.x_limits[0], self.y_limits[0], self.z_limits[0]]
            u_bound = [self.x_limits[1], self.y_limits[1], self.z_limits[1]]

        return TetMeshGenerator(l_bound=l_bound, u_bound=u_bound, n=n)

    @classmethod
    def from_fs(cls, data: Dict) -> "LayeredModel":
        """Creates a LayeredModel instance from a dictionary representation.

        Args:
           data (Dict): Dictionary containing model data with:
              - surfaces: List of surface dictionaries
              - All parent class dict fields

        Returns:
           LayeredModel: New LayeredModel instance created from dictionary data.
        """
        data = copy.deepcopy(data)
        # Create copy and remove surfaces to pass rest to parent
        surfs = data.pop("surfaces")
        subdomains = data.pop("subdomains")
        boreholes = data.pop("boreholes", [])
        if len(surfs) < 2:
            raise ValueError("LayeredModel requires at least two surfaces")
        if len(subdomains) == 0:
            raise ValueError("LayeredModel requires at least one layer")

        layer_count = sum(1 for surface in surfs if surface.get("interface", True)) - 1
        if len(subdomains) < layer_count:
            raise ValueError("LayeredModel has fewer layer definitions than intervals")
        layers = subdomains[:layer_count]
        extra_subdomains = subdomains[layer_count:]

        name = data.pop("name", None)
        data.pop("_type", None)
        data.pop("schema", None)
        dimension = data.pop("dimension", None)
        x_limits = data.pop("x_limits", None)
        y_limits = data.pop("y_limits", None)
        ordering = data.pop("ordering", "top_down")
        model = LayeredModel(
            name=name,
            dimension=dimension,
            x_limits=x_limits,
            y_limits=y_limits,
            ordering=ordering,
        )
        model.extra = data

        model += SimpleSurface.from_fs(surfs[0])

        n_surf = len(surfs)
        isurf = 1
        ilayer = 0
        while isurf < n_surf:
            if ilayer >= len(layers):
                raise ValueError("LayeredModel has more surfaces than layer intervals")
            # Add layer
            model += Layer.from_fs(layers[ilayer])
            ilayer += 1

            # Add any non-interface surfaces (surfaces not between layers)
            while surfs[isurf].get("interface", True) is False:
                model += SimpleSurface.from_fs(surfs[isurf])
                isurf += 1
                if isurf >= n_surf:
                    raise ValueError("LayeredModel ended with a non-interface surface")

            # Add surface
            model += SimpleSurface.from_fs(surfs[isurf])
            isurf += 1

        if ilayer != len(layers):
            raise ValueError("LayeredModel has unused layer definitions")

        for subdomain in extra_subdomains:
            model += ModelSubdomain.from_fs(subdomain)
        for borehole in boreholes:
            model += Borehole.from_fs(borehole)

        return model

    def to_fs(self, ctx=None) -> Dict:
        self._validate_complete()

        base_dict = super().to_fs(ctx)
        surfaces = []
        for i, surface in enumerate(self.surfaces):
            payload = surface.to_fs(ctx)
            if i == len(self.surfaces) - 1:
                payload["interface"] = True
            surfaces.append(payload)
        base_dict.update(
            {
                "_type": self.__class__.__name__,
                "x_limits": self.x_limits,
                **({"y_limits": self.y_limits} if self.y_limits is not None else {}),
                "ordering": self.ordering,
                "surfaces": surfaces,
                **(
                    {"boreholes": [borehole.to_fs(ctx) for borehole in self.boreholes]}
                    if self.boreholes
                    else {}
                ),
            }
        )
        return merge_extra(base_dict, self.extra, "LayeredModel")

    def __iadd__(self, other):
        if isinstance(other, SimpleSurface):
            self.surfaces.append(other)

            # Build layer-to-surface mapping
            if len(self.surfaces) > 1:
                if self.ordering == "top_down":
                    self.layers[-1].lower = other
                else:
                    self.layers[-1].upper = other
            self._last_added = "surface"

        elif isinstance(other, Layer):
            # Check that surfaces sandwiching layers
            if len(self.surfaces) == 0:
                raise ValueError("Must add at least one surface before adding layers")
            if self._last_added == "layer":
                raise ValueError("Must add a surface between consecutive layers")

            # Add layer
            self.add_subdomain(other)
            if len(self.surfaces) > 1:
                self.surfaces[-1].interface = True

            # Build layer-to-surface mapping
            if self.ordering == "top_down":
                if len(self.layers) == 1:
                    prev_lower = self.upper_surface()
                else:
                    prev_lower = self.lower_surface(self.layers[-2])

                self.layers[-1].upper = prev_lower
            else:
                if len(self.layers) == 1:
                    prev_upper = self.lower_surface()
                else:
                    prev_upper = self.upper_surface(self.layers[-2])

                self.layers[-1].lower = prev_upper
            self._last_added = "layer"
        elif isinstance(other, ModelSubdomain):
            self.add_subdomain(other)
        elif isinstance(other, Borehole):
            self.boreholes.append(other)
            self._borehole_names.add(other.name)
        else:
            raise ValueError(f"Cannot add {type(other)} to LayeredModel")
        return self

    def upper_surface(
        self, layer: Optional[Union[str, int, ModelSubdomain]] = None
    ) -> SimpleSurface:
        """Returns the top surface of the model or specified layer.

        Args:
           layer (str, optional): Name of layer to get top surface for.
                                  If None, returns topmost model surface.

        Returns:
           Surface: The top surface.

        Raises:
           ValueError: If specified layer is not found.
        """
        if layer is None:
            if not self.surfaces:
                raise ValueError("LayeredModel has no surfaces")
            if self.ordering == "top_down":
                return self.surfaces[0]
            else:
                return self.surfaces[-1]

        if isinstance(layer, int):
            return self.layers[layer].upper
        elif isinstance(layer, str):
            return self.layers[layer].upper
        elif isinstance(layer, Layer):
            return layer.upper
        elif isinstance(layer, ModelSubdomain):
            mesh_block_id = layer.mesh_block_id
            for candidate in self.layers:
                if candidate.mesh_block_id == mesh_block_id:
                    return candidate.upper
        raise ValueError(f"Layer not found: {layer}")

    def lower_surface(
        self, layer: Optional[Union[str, int, ModelSubdomain]] = None
    ) -> SimpleSurface:
        """Returns the bottom surface of the model or specified layer.

        Args:
           layer (str, optional): Name of layer to get bottom surface for.
                                  If None, returns bottommost model surface.

        Returns:
           Surface: The bottom surface.

        Raises:
           ValueError: If specified layer is not found.
        """
        if layer is None:
            if not self.surfaces:
                raise ValueError("LayeredModel has no surfaces")
            if self.ordering == "top_down":
                return self.surfaces[-1]
            else:
                return self.surfaces[0]

        if isinstance(layer, int):
            return self.layers[layer].lower
        elif isinstance(layer, str):
            return self.layers[layer].lower
        elif isinstance(layer, Layer):
            return layer.lower
        elif isinstance(layer, ModelSubdomain):
            mesh_block_id = layer.mesh_block_id
            for candidate in self.layers:
                if candidate.mesh_block_id == mesh_block_id:
                    return candidate.lower
        raise ValueError(f"Layer not found: {layer}")

    def _coordinate_system(self, name: Optional[str]) -> Optional[Any]:
        if name is None:
            return None
        for system in getattr(self, "_coordinate_systems", []):
            if getattr(system, "name", None) == name:
                return system
        return None

    def _axis_for_dimension(self, system: Any, dim: str) -> Optional[Axis]:
        for axis in getattr(system, "axes", None) or []:
            if axis.name == dim:
                return axis
        if getattr(system, "inherit_axes", False) and dim in {"x", "y"}:
            return Axis(dim, direction=dim)
        return None

    @staticmethod
    def _axis_origin_value(axis: Axis) -> float:
        origin = axis.origin
        if origin is None:
            return 0.0
        if isinstance(origin, Mapping):
            return float(origin.get("value", 0.0))
        if hasattr(origin, "magnitude"):
            return float(origin.magnitude)
        return float(origin)

    def _axis_coordinate(
        self, system: Any, axis: Axis, samples: xr.DataArray
    ) -> xr.DataArray:
        direction = str(axis.direction).strip().lower()
        if direction == "z" and getattr(system, "type", None) == "surface":
            values = self._surface_relative_coordinate(system, samples, axis=axis)
        else:
            if direction not in samples.coords:
                raise ValueError(
                    f"Coordinate system '{system.name}' axis '{axis.name}' "
                    f"references unavailable direction '{axis.direction}'"
                )
            values = samples.coords[direction]
            if axis.origin is not None:
                values = values - self._axis_origin_value(axis)
        return values.rename(axis.name)

    def _surface_relative_coordinate(
        self, system: Any, samples: xr.DataArray, *, axis: Optional[Axis] = None
    ) -> xr.DataArray:
        surface_ref = getattr(system, "surface_ref", None)
        if surface_ref is None:
            raise ValueError(
                f"Coordinate system '{system.name}' is not tied to a surface"
            )
        try:
            surface = self.surfaces[surface_ref]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Coordinate system '{system.name}' references unknown surface "
                f"'{surface_ref}'"
            ) from exc

        surface_depth = surface.depth.get(samples).transpose(*samples.dims)
        z = xr.DataArray(
            samples.coords["z"].values,
            dims=["z"],
            coords={"z": samples.coords["z"]},
        ).broadcast_like(samples)
        positive = (
            axis.positive
            if axis is not None and axis.positive is not None
            else getattr(system, "normal", "up")
        )
        if str(positive or "up").strip().lower() == "down":
            return z - surface_depth
        return surface_depth - z

    def _surface_relative_samples(self, system: Any, samples: xr.DataArray):
        coords = {
            axis.name: self._axis_coordinate(system, axis, samples)
            for axis in getattr(system, "axes", None) or []
            if str(axis.direction).strip().lower() == "z"
        }
        if "up" not in coords:
            coords["up"] = self._surface_relative_coordinate(system, samples)
        return samples.assign_coords(**coords)

    def _property_sample_grid(self, prop: Property, samples: xr.DataArray):
        system = self._coordinate_system(prop.system)
        if system is None:
            return samples
        if getattr(system, "type", None) == "surface":
            return self._surface_relative_samples(system, samples)
        return samples

    def _surface_relative_property_values(
        self, prop: Property, system: Any, samples: xr.DataArray
    ) -> xr.DataArray:
        data = prop.data
        if data is None:
            return prop.get(self._property_sample_grid(prop, samples))

        coords = {}
        for dim in data.dims:
            axis = self._axis_for_dimension(system, dim)
            if axis is not None:
                coords[dim] = self._axis_coordinate(system, axis, samples)
            elif dim in samples.coords:
                coords[dim] = samples.coords[dim]
            else:
                return prop.get(self._property_sample_grid(prop, samples))

        out = data.interp(coords=coords, method="linear")
        if np.isnan(out.values).any():
            nearest = data.interp(
                coords=coords,
                method="nearest",
                kwargs={"fill_value": "extrapolate"},
            )
            nan_mask = np.isnan(out.values)
            out.values[nan_mask] = nearest.values[nan_mask]
        if set(out.dims) != set(samples.dims):
            out = out.broadcast_like(samples)
        return out

    def _property_values_on_samples(
        self, prop: Property, samples: xr.DataArray
    ) -> xr.DataArray:
        system = self._coordinate_system(prop.system)
        if system is not None and getattr(system, "type", None) == "surface":
            return self._surface_relative_property_values(prop, system, samples)
        return prop.get(self._property_sample_grid(prop, samples))

    def get_1D_log(
        self,
        property: str,
        x: float,
        dz: float,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        units: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get a 1D well-log of a property."""

        if self.dimension == 3:
            raise NotImplementedError("3D sampling not implemented")

        if dz <= 0:
            raise ValueError("dz must be positive")
        if z_min is None:
            z_min = self.z_limits[0]
        if z_max is None:
            z_max = self.z_limits[1]
        depths = np.arange(z_min, z_max, dz)
        samples = xr.DataArray(
            data=np.nan * np.ones((1, len(depths))),
            dims=["x", "z"],
            coords={"x": [x], "z": depths},
        )
        property = canonical_property_name(property)
        target_units = (
            unit_expression(units)
            if units is not None
            else self.property_units(property)
        )

        for layer in self.layers:
            if property not in layer.properties.keys():
                continue
            layer_prop = layer.properties[property]
            prop = self._property_values_on_samples(layer_prop, samples)
            prop = prop.transpose(*samples.dims)
            prop = _convert_dataarray_units(
                prop,
                _property_units(layer_prop),
                target_units,
            )
            mask = self._get_layer_mask(layer, samples)
            data = prop.where(mask)
            samples.data = np.where(~np.isnan(data), data, samples.data)
        if target_units is not None:
            samples.attrs["units"] = target_units

        return depths, samples.data[0]

    def plot_1d_log(
        self,
        ax: "Axes",
        property: str,
        x: float,
        dz: float,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        **kwargs,
    ):
        """Plot a 1D well-log of a property."""

        if self.dimension == 3:
            raise NotImplementedError("3D plotting not implemented")

        property_units = kwargs.pop("property_units", self.property_units(property))
        property_label = kwargs.pop(
            "property_label",
            f"{property} [{property_units}]" if property_units else property,
        )
        aspect = kwargs.pop("aspect", None)
        show_legend = kwargs.pop("show_legend", True)
        legend_coords = kwargs.pop("legend_coords", (1.8, 1))

        depths, data = self.get_1D_log(
            property,
            x,
            dz,
            z_min,
            z_max,
            units=property_units,
        )
        ax.plot(data, depths, label=property_label, **kwargs)

        # Set aspect ratio
        x_range = np.nanmax(data) - np.nanmin(data)
        y_range = depths[-1] - depths[0]
        if aspect:
            aspect *= x_range / y_range
            ax.set_aspect(aspect)
        ax.grid(True, alpha=0.5)

        if show_legend:
            ax.legend(bbox_to_anchor=legend_coords, loc="upper right")

    @property
    def properties(self) -> List[str]:
        """List of properties in the model."""
        props = set()
        for subdomain in self.subdomains:
            props.update(subdomain.properties.keys())
        return sorted(props)

    def _subdomain_by_mesh_block_id(self) -> Dict[int, ModelSubdomain]:
        return {subdomain.mesh_block_id: subdomain for subdomain in self.subdomains}

    def _surface_from_reference(self, ref: Any) -> SimpleSurface:
        if isinstance(ref, SimpleSurface):
            return ref
        if isinstance(ref, Mapping):
            if "surface" in ref:
                ref = ref["surface"]
            elif "index" in ref:
                ref = ref["index"]
        if isinstance(ref, int):
            return self.surfaces[ref - 1]
        if isinstance(ref, str):
            if ref == "top":
                return self.surfaces[0]
            if ref == "bottom":
                return self.surfaces[-1]
            if ref.startswith("surface_"):
                try:
                    return self.surfaces[int(ref.split("_", 1)[1]) - 1]
                except (ValueError, IndexError):
                    pass
            for surface in self.surfaces:
                if surface.name == ref:
                    return surface
        raise ValueError(f"Unknown borehole extent surface reference: {ref!r}")

    def _surface_depth_at_x(
        self,
        surface: SimpleSurface,
        x: float,
        units: Optional[str],
    ) -> float:
        coords = xr.DataArray(dims=["x"], coords={"x": [x]})
        depth = surface.depth.get(coords)
        value = float(np.asarray(depth.values).reshape(-1)[0])
        return float(_convert_units(value, _property_units(surface.depth), units))

    @staticmethod
    def _sample_coordinate_mesh(
        samples: xr.DataArray,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        x = xr.DataArray(
            samples.coords["x"].values,
            dims=["x"],
            coords={"x": samples.coords["x"]},
        ).broadcast_like(samples)
        z = xr.DataArray(
            samples.coords["z"].values,
            dims=["z"],
            coords={"z": samples.coords["z"]},
        ).broadcast_like(samples)
        return x, z

    def _borehole_part_masks(
        self,
        borehole: Borehole,
        samples: xr.DataArray,
    ) -> List[Tuple[BoreholePart, xr.DataArray]]:
        if self.dimension != 2:
            return []
        x_units = samples.coords["x"].attrs.get("units")
        z_units = samples.coords["z"].attrs.get("units")
        axis_x = borehole.axis_x(x_units)
        top_surface = self._surface_from_reference(borehole.extent["top"])
        bottom_surface = self._surface_from_reference(borehole.extent["bottom"])
        z_top = self._surface_depth_at_x(top_surface, axis_x, z_units)
        z_bottom = self._surface_depth_at_x(bottom_surface, axis_x, z_units)
        z0, z1 = sorted([z_top, z_bottom])

        x, z = self._sample_coordinate_mesh(samples)
        radial_distance = abs(x - axis_x)
        extent_mask = (z >= z0) & (z <= z1)
        inner = xr.zeros_like(z, dtype=float)
        masks: List[Tuple[BoreholePart, xr.DataArray]] = []
        for index, part in enumerate(borehole.parts):
            z_profile, r_profile = borehole.radius_profile(
                part,
                units=x_units,
                depth_units=z_units,
            )
            if len(r_profile) == 1:
                outer = xr.full_like(z, float(r_profile[0]), dtype=float)
            else:
                outer_values = np.interp(
                    samples.coords["z"].values,
                    z_profile,
                    r_profile,
                )
                outer = xr.DataArray(
                    outer_values,
                    dims=["z"],
                    coords={"z": samples.coords["z"]},
                ).broadcast_like(samples)
            radial_mask = radial_distance <= outer
            if index > 0:
                radial_mask = radial_mask & (radial_distance > inner)
            masks.append((part, extent_mask & radial_mask))
            inner = outer
        return masks

    def _apply_borehole_overrides(
        self,
        gridded: xr.Dataset,
        samples: xr.DataArray,
    ) -> None:
        if self.dimension != 2 or not self.boreholes:
            return
        subdomains = self._subdomain_by_mesh_block_id()
        for borehole in self.boreholes:
            for part, mask in self._borehole_part_masks(borehole, samples):
                subdomain = subdomains.get(part.mesh_block_id)
                if subdomain is None:
                    continue
                for name in gridded.data_vars:
                    if name not in subdomain.properties:
                        continue
                    prop = subdomain.properties[name]
                    target_units = gridded[name].attrs.get("units")
                    source_units = _property_units(prop)
                    if target_units is None and source_units is not None:
                        target_units = source_units
                        gridded[name].attrs["units"] = target_units
                    if prop.is_constant:
                        data = xr.full_like(samples, prop.get(), dtype=float)
                    else:
                        data = self._property_values_on_samples(
                            prop, samples
                        ).transpose(*samples.dims)
                    data = _convert_dataarray_units(data, source_units, target_units)
                    gridded[name].data = np.where(
                        mask.values,
                        data.values,
                        gridded[name].data,
                    )

    def sample_uniform(
        self,
        n: Union[np.ndarray, List[int]],
        axes_units: Optional[Dict[str, Any]] = None,
    ) -> xr.Dataset:
        """Export model on a uniform grid."""
        n = list(n)
        expected = 2 if self.dimension == 2 else 3
        if len(n) != expected:
            raise ValueError(f"Expected {expected} sample counts for {self.dimension}D")
        if any(count < 2 for count in n):
            raise ValueError("All sample counts must be >= 2")

        axes_units = _axis_units(axes_units)
        xl = np.linspace(self.x_limits[0], self.x_limits[1], n[0])
        z_limits = self.z_limits_in(axes_units["z"])
        zl = np.linspace(z_limits[0], z_limits[1], n[-1])
        if self.dimension == 2:
            samples = xr.DataArray(dims=["x", "z"], coords={"z": zl, "x": xl})
        elif self.dimension == 3:
            yl = np.linspace(self.y_limits[0], self.y_limits[1], n[1])
            samples = xr.DataArray(
                dims=["x", "y", "z"], coords={"z": zl, "y": yl, "x": xl}
            )
        else:
            raise ValueError("Invalid dimension")
        for axis, units in axes_units.items():
            if units is not None and axis in samples.coords:
                samples.coords[axis].attrs["units"] = units

        gridded = xr.Dataset(coords=samples.coords)
        for name in self.properties:
            units = self.property_units(name)
            gridded[name] = xr.DataArray(
                dims=samples.dims,
                coords=samples.coords,
                data=np.nan * np.ones(samples.shape),
                attrs={"units": units} if units is not None else {},
            )
        for layer in self.layers:
            mask = self._get_layer_mask(layer, samples)

            for name in self.properties:
                if name not in layer.properties:
                    continue
                prop = layer.properties[name]
                target_units = gridded[name].attrs.get("units")
                source_units = _property_units(prop)
                if target_units is None and source_units is not None:
                    target_units = source_units
                    gridded[name].attrs["units"] = target_units

                if prop.is_constant:
                    data = xr.full_like(samples, prop.get(), dtype=float)
                else:
                    data = self._property_values_on_samples(prop, samples).transpose(
                        *samples.dims
                    )
                data = _convert_dataarray_units(data, source_units, target_units)
                gridded[name].data = np.where(
                    mask.values,
                    data.values,
                    gridded[name].data,
                )
        self._apply_borehole_overrides(gridded, samples)
        return gridded

    def smooth(self, n: ArrayLike, sigma, **kwargs):
        """Smooth the model."""
        from scipy.ndimage import gaussian_filter

        gridded = self.sample_uniform(n)
        for name in self.properties:
            if np.any(np.isnan(gridded[name].data)):
                filled = gridded[name]
                for dim in filled.dims:
                    filled = filled.bfill(dim=dim).ffill(dim=dim)
                data = filled.data
            else:
                data = gridded[name].data

            gridded[name].data = gaussian_filter(data, sigma=sigma, **kwargs)
        for layer in self.layers:
            for name in layer.properties:
                layer.set_property(name, gridded[name])
        return self

    def update_from_dataset(self, dataset: xr.Dataset):
        """Update the model from an Xarray dataset."""

        missing = set(self.properties).difference(dataset.data_vars)
        if missing:
            raise ValueError(f"Dataset is missing model properties: {sorted(missing)}")

        updated = dataset.copy(deep=True)
        for name in self.properties:
            if np.any(np.isnan(updated[name].data)):
                filled = updated[name]
                for dim in filled.dims:
                    filled = filled.bfill(dim=dim).ffill(dim=dim)
                data = filled.data
            else:
                data = updated[name].data
            updated[name].data = data
        for layer in self.layers:
            for name in layer.properties:
                layer.set_property(name, updated[name])
        return self

    def plot(self, property: str, resolution: Optional[List[int]] = None, **kwargs):
        """Plot the model."""
        from frequensolve.plotting.layered import plot_layered_model

        resolution = resolution or [500, 500]
        return plot_layered_model(self, property, resolution, **kwargs)

    def _get_layer_mask(self, layer, samples):
        xgrid = samples.coords["x"]
        upper = (
            self.upper_surface(layer)
            if self.ordering == "top_down"
            else self.lower_surface(layer)
        )
        lower = (
            self.lower_surface(layer)
            if self.ordering == "top_down"
            else self.upper_surface(layer)
        )
        if upper is None or lower is None:
            raise ValueError(f"Layer '{layer.name}' is missing upper/lower surfaces")
        limits = {}
        limits["x_min"] = self.x_limits[0]
        limits["x_max"] = self.x_limits[1]

        if self.y_limits is not None and "y" in samples.coords:
            limits["y_min"] = self.y_limits[0]
            limits["y_max"] = self.y_limits[1]
            ygrid = samples.coords["y"]
            coords_query = xr.DataArray(
                dims=["x", "y"], coords={"x": xgrid, "y": ygrid}
            )
        else:
            coords_query = xgrid

        z_units = samples.coords["z"].attrs.get("units")
        limits["z_min"] = _convert_surface_depth(
            upper.depth.get(coords_query),
            upper,
            z_units,
        )
        limits["z_max"] = _convert_surface_depth(
            lower.depth.get(coords_query),
            lower,
            z_units,
        )

        mask = (
            (samples.coords["z"] <= limits["z_max"])
            & (samples.coords["z"] >= limits["z_min"])
            & (samples.coords["x"] <= limits["x_max"])
            & (samples.coords["x"] >= limits["x_min"])
        )

        if "y" in samples.coords:
            mask &= samples.coords["y"] <= limits["y_max"]
            mask &= samples.coords["y"] >= limits["y_min"]

        mask = mask.transpose(*samples.dims)
        return mask

    @staticmethod
    def _get_unique_name(name: Optional[str], names: Set[str]) -> str:
        orig_name = name
        warn_flag = True
        if name is None:
            warn_flag = False
            name = "unlabeled"

        if name in names:
            i = 1
            while f"{name}_{i}" in names and i < 1000:
                i += 1
            name = f"{name}_{i}"

        if warn_flag and orig_name != name:
            warnings.warn(
                f"\nSurface name '{orig_name}' was not unique; name was changed to '{name}'\n\n"
            )
        return name

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path / self.name
        for subdomain in self.subdomains:
            subdomain._set_path(proj_path, self._rel_path)
        for surface in self.surfaces:
            surface._set_path(proj_path, self._rel_path)

    @property
    def _path(self) -> Path:
        if self._proj_path is None or self._rel_path is None:
            raise ValueError("LayeredModel is not attached to a project path")
        return self._proj_path / self._rel_path

    def _validate_bounds_for_meshing(self) -> None:
        if len(self.surfaces) < 2:
            raise ValueError("At least two surfaces are required before meshing")

    def _validate_complete(self) -> None:
        if len(self.surfaces) < 2:
            raise ValueError("LayeredModel requires at least two surfaces")
        if not self.layers:
            raise ValueError("LayeredModel requires at least one layer")
        for layer in self.layers:
            if layer.upper is None or layer.lower is None:
                raise ValueError(
                    f"Layer '{layer.name}' must be bounded by upper and lower surfaces"
                )


# TODO: info functions like this:
#     max_elevation()
#     max_layer_thickness()
#     min_elevation()
#     min_layer_thickness()
#     check_intersection()
