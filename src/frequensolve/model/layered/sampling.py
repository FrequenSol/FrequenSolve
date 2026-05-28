"""Sampling and plotting helpers mixed into :class:`LayeredModel`.

The public layered-model API uses this module to evaluate material properties
on regular xarray grids, convert between coordinate frames and physical axes,
smooth sampled properties, and pass gridded models to plotting helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from frequensolve.geometry.frame import Axis
from frequensolve.model.model import ModelSubdomain
from frequensolve.model.property import (
    Property,
    _coords_in_data_units,
    canonical_property_name,
)
from frequensolve.units import unit_expression

from ._utils import (
    _axis_units,
    _convert_dataarray_units,
    _convert_surface_depth,
    _convert_units,
    _property_units,
)
from .borehole import Borehole, BoreholePart
from .surfaces import SimpleSurface

if TYPE_CHECKING:
    from matplotlib.axes import Axes


_PHYSICAL_AXIS_ORDER = ("x", "y", "z")


@dataclass(frozen=True)
class _SampleAxis:
    name: str
    direction: str
    axis: Optional[Axis] = None


class LayeredSamplingMixin:
    """Sampling, plotting, and coordinate-frame helpers for layered models.

    The mixin assumes it is combined with ``LayeredModel`` and uses that
    model's surfaces, layers, boreholes, and optional coordinate systems to
    evaluate material properties on regular xarray grids.
    """

    def sample_uniform(
        self,
        n: Union[np.ndarray, List[int]],
        axes_units: Optional[Dict[str, Any]] = None,
        *,
        frame: Optional[Any] = None,
    ) -> xr.Dataset:
        """Sample all model properties on a uniform xarray grid.

        Args:
            n: Number of samples along each exposed grid axis. The length must
                match the model dimension or the number of axes exposed by
                ``frame``.
            axes_units: Optional units for sample coordinates, keyed by either
                exposed axis name or physical direction. Values may be unit
                strings or Pint units.
            frame: Optional coordinate-system object or registered
                coordinate-system name. When provided, its axis names and order
                define the dataset dimensions while physical ``x``, ``y``, and
                ``z`` coordinates are retained for geometry evaluation.

        Returns:
            Dataset whose data variables are the model's canonical property
            names sampled over the requested grid.

        Raises:
            ValueError: If the sample count length does not match the grid
                axes, any sample count is below two, or the requested frame has
                unsupported or incomplete axis definitions.
        """
        sample_frame = self._resolve_sample_frame(frame)
        sample_axes = self._sample_axes(sample_frame)
        n = list(n)
        expected = len(sample_axes)
        if len(n) != expected:
            raise ValueError(f"Expected {expected} sample counts for {self.dimension}D")
        if any(count < 2 for count in n):
            raise ValueError("All sample counts must be >= 2")

        samples = self._uniform_samples(n, sample_axes, axes_units)
        if sample_frame is not None and getattr(sample_frame, "name", None):
            samples.attrs["frame"] = sample_frame.name

        gridded = xr.Dataset(coords=samples.coords, attrs=dict(samples.attrs))
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
        """Smooth sampled material properties and write them back to each layer.

        Args:
            n: Uniform-grid sample counts passed to :meth:`sample_uniform`.
            sigma: Gaussian-filter standard deviation passed to SciPy.
            **kwargs: Additional keyword arguments forwarded to
                ``scipy.ndimage.gaussian_filter``.

        Returns:
            This model, after each layer property has been replaced by the
            smoothed sampled data.
        """
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
        """Replace layer properties with values from a sampled dataset.

        Args:
            dataset: Dataset containing one data variable for every property
                currently declared by the model.

        Returns:
            This model, after each layer property has been replaced.

        Raises:
            ValueError: If ``dataset`` is missing any property declared by the
                model.
        """

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
        """Plot one sampled model property.

        Args:
            property: Property name or alias to sample and plot.
            resolution: Optional two- or three-dimensional sampling resolution.
                Defaults to ``[500, 500]`` for 2D plotting.
            **kwargs: Plotting options forwarded to
                ``frequensolve.plotting.layered.plot_layered_model``.

        Returns:
            The object returned by the layered-model plotting helper.
        """
        from frequensolve.plotting.layered import plot_layered_model

        resolution = resolution or [500, 500]
        return plot_layered_model(self, property, resolution, **kwargs)

    def get_1D_log(
        self,
        property: str,
        x: float,
        dz: float,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        units: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample one property along a vertical 1D log.

        Args:
            property: Property name or alias to sample.
            x: Horizontal coordinate where the vertical log is sampled.
            dz: Positive depth increment.
            z_min: Optional starting depth. Defaults to the model's minimum
                surface depth.
            z_max: Optional ending depth. Defaults to the model's maximum
                surface depth.
            units: Optional output units for property values.

        Returns:
            A tuple ``(depths, values)`` containing depth coordinates and the
            sampled property values.

        Raises:
            NotImplementedError: If called on a 3D model.
            ValueError: If ``dz`` is not positive.
        """

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
        """Plot a vertical 1D property log on existing matplotlib axes.

        Args:
            ax: Matplotlib axes that receives the plotted log.
            property: Property name or alias to sample.
            x: Horizontal coordinate where the vertical log is sampled.
            dz: Positive depth increment.
            z_min: Optional starting depth.
            z_max: Optional ending depth.
            **kwargs: Plot styling options. The method also consumes
                ``property_units``, ``property_label``, ``aspect``,
                ``show_legend``, and ``legend_coords`` when supplied.

        Raises:
            NotImplementedError: If called on a 3D model.
            ValueError: If the underlying 1D sampling request is invalid.
        """

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
        """Return sorted canonical property names declared by any subdomain.

        Returns:
            Sorted list of property names after alias normalization.
        """
        props = set()
        for subdomain in self.subdomains:
            props.update(subdomain.properties.keys())
        return sorted(props)

    def _coordinate_system(self, name: Optional[str]) -> Optional[Any]:
        if name is None:
            return None
        for system in getattr(self, "_coordinate_systems", []):
            if getattr(system, "name", None) == name:
                return system
        return None

    def _resolve_sample_frame(self, frame: Optional[Any]) -> Optional[Any]:
        if frame is None or not isinstance(frame, str):
            return frame
        resolved = self._coordinate_system(frame)
        if resolved is not None:
            return resolved
        available = [
            getattr(item, "name", None)
            for item in getattr(self, "_coordinate_systems", [])
            if getattr(item, "name", None)
        ]
        suffix = f" Available frames: {available}." if available else ""
        raise ValueError(f"Unknown frame {frame!r}.{suffix}")

    def _physical_sample_axes(self) -> Tuple[str, ...]:
        if self.dimension == 2:
            return ("x", "z")
        if self.dimension == 3:
            return ("x", "y", "z")
        raise ValueError(
            f"LayeredModel.sample_uniform supports dimension 2 or 3; "
            f"got {self.dimension!r}"
        )

    def _coordinate_system_axes(self, system: Any) -> List[Axis]:
        axes = [Axis.from_fs(axis) for axis in getattr(system, "axes", None) or []]
        alignment = getattr(system, "axis_alignment", None)
        if isinstance(alignment, Mapping):
            for name, direction in alignment.items():
                name = str(name)
                direction = str(direction)
                if (
                    name in _PHYSICAL_AXIS_ORDER
                    and direction not in _PHYSICAL_AXIS_ORDER
                ):
                    axes.append(Axis(direction, direction=name))
                else:
                    axes.append(Axis(name, direction=direction))
        return axes

    def _sample_axes(self, system: Optional[Any] = None) -> List[_SampleAxis]:
        physical_axes = self._physical_sample_axes()
        if system is None:
            return [_SampleAxis(axis, axis) for axis in physical_axes]

        system_type = getattr(system, "type", None)
        if system_type == "surface":
            raise ValueError(
                "sample_uniform does not create surface-relative sampling grids. "
                "Use surface coordinate systems on properties, or sample in a "
                "cartesian-like coordinate system aligned with x/y/z."
            )

        axes_by_direction: Dict[str, _SampleAxis] = {}
        ordered_axes: List[_SampleAxis] = []
        for axis in self._coordinate_system_axes(system):
            direction = self._axis_direction(axis)
            if direction not in _PHYSICAL_AXIS_ORDER:
                raise ValueError(
                    f"Coordinate system {getattr(system, 'name', None)!r} axis "
                    f"{axis.name!r} references unsupported direction "
                    f"{axis.direction!r}; expected one of {_PHYSICAL_AXIS_ORDER}."
                )
            if direction not in physical_axes:
                raise ValueError(
                    f"Coordinate system {getattr(system, 'name', None)!r} axis "
                    f"{axis.name!r} references direction {direction!r}, which is "
                    f"not available in a {self.dimension}D layered model."
                )
            if direction in axes_by_direction:
                raise ValueError(
                    f"Coordinate system {getattr(system, 'name', None)!r} defines "
                    f"multiple axes for physical direction {direction!r}."
                )
            axes_by_direction[direction] = _SampleAxis(
                name=str(axis.name),
                direction=direction,
                axis=axis,
            )
            ordered_axes.append(axes_by_direction[direction])

        missing = [axis for axis in physical_axes if axis not in axes_by_direction]
        if missing and axes_by_direction and not getattr(system, "inherit_axes", False):
            raise ValueError(
                f"Coordinate system {getattr(system, 'name', None)!r} is missing "
                f"axes for physical directions {missing}. Define matching Axis "
                "objects or set inherit_axes=True."
            )
        for axis in missing:
            axes_by_direction[axis] = _SampleAxis(axis, axis)
            ordered_axes.append(axes_by_direction[axis])
        return ordered_axes

    def _axis_units_for_sample(
        self,
        axes_units: Mapping[str, Optional[str]],
        spec: _SampleAxis,
    ) -> Optional[str]:
        units = axes_units.get(spec.name, axes_units.get(spec.direction))
        if units is not None:
            return units
        if spec.direction == "x":
            return self._x_units
        if spec.direction == "y":
            return self._y_units
        return None

    def _axis_limits(self, axis: str, units: Optional[str]) -> Tuple[float, float]:
        if axis == "x":
            return self.x_limits_in(units)
        if axis == "y":
            return self.y_limits_in(units)
        if axis == "z":
            return self.z_limits_in(units)
        raise ValueError(f"Unsupported sample axis direction: {axis!r}")

    def _axis_sample_values(
        self,
        spec: _SampleAxis,
        physical_values: np.ndarray,
        units: Optional[str],
    ) -> np.ndarray:
        if spec.axis is None or spec.axis.name == spec.direction:
            return physical_values
        origin = self._axis_origin_value(spec.axis, units=units)
        return physical_values - origin

    def _uniform_samples(
        self,
        counts: List[int],
        axes: List[_SampleAxis],
        axes_units: Optional[Dict[str, Any]],
    ) -> xr.DataArray:
        self._normalize_domain_limits()
        requested_units = _axis_units(axes_units)
        dims = [axis.name for axis in axes]
        coords: Dict[str, Any] = {}
        coord_units: Dict[str, Optional[str]] = {}

        for count, axis in zip(counts, axes):
            units = self._axis_units_for_sample(requested_units, axis)
            limits = self._axis_limits(axis.direction, units)
            physical_values = np.linspace(limits[0], limits[1], count)
            sample_values = self._axis_sample_values(axis, physical_values, units)
            coords[axis.name] = (axis.name, sample_values)
            coord_units[axis.name] = units
            if axis.name != axis.direction:
                coords[axis.direction] = (axis.name, physical_values)
                coord_units[axis.direction] = units

        samples = xr.DataArray(dims=dims, coords=coords)
        for axis, units in coord_units.items():
            if units is not None and axis in samples.coords:
                samples.coords[axis].attrs["units"] = units
        return samples

    def _property_coordinate_system(self, prop: Property) -> Optional[Any]:
        if prop.system is None:
            return None
        system = self._coordinate_system(prop.system)
        if system is not None:
            return system
        available = [
            getattr(system, "name", None)
            for system in getattr(self, "_coordinate_systems", [])
            if getattr(system, "name", None)
        ]
        suffix = (
            f" Available coordinate systems: {available}."
            if available
            else " No coordinate systems are currently bound to this model."
        )
        raise ValueError(
            f"Property declares coordinate system {prop.system!r}, but that "
            "coordinate system is not available while sampling. Add the "
            "CoordinateSystem to the simulation before adding/sampling the model, "
            f"or remove the property coordinate_system reference.{suffix}"
        )

    @staticmethod
    def _property_axis_error(
        *,
        system: Any,
        dim: str,
        samples: xr.DataArray,
    ) -> ValueError:
        axis_names = [str(axis.name) for axis in getattr(system, "axes", None) or []]
        return ValueError(
            f"Property dimension {dim!r} is not available for coordinate system "
            f"{system.name!r}. Define an Axis named {dim!r} on that coordinate "
            "system, use a property dimension that already exists on the sampling "
            f"grid, or rename the property dimension. Coordinate-system axes are "
            f"{axis_names or 'none'}; sampling grid coordinates are "
            f"{list(samples.coords)}."
        )

    @staticmethod
    def _axis_direction(axis: Axis) -> str:
        return str(axis.direction).strip().lower()

    def _axis_for_dimension(self, system: Any, dim: str) -> Optional[Axis]:
        for axis in self._coordinate_system_axes(system):
            if axis.name == dim:
                return axis
        if getattr(system, "inherit_axes", False) and dim in {"x", "y"}:
            return Axis(dim, direction=dim)
        return None

    @staticmethod
    def _axis_origin_value(axis: Axis, *, units: Optional[str] = None) -> float:
        origin = axis.origin
        if origin is None:
            return 0.0
        if isinstance(origin, Mapping):
            return float(
                _convert_units(origin.get("value", 0.0), origin.get("units"), units)
            )
        if hasattr(origin, "magnitude"):
            if units is not None:
                return float(origin.to(units).magnitude)
            return float(origin.magnitude)
        return float(origin)

    def _axis_coordinate(
        self, system: Any, axis: Axis, samples: xr.DataArray
    ) -> xr.DataArray:
        direction = self._axis_direction(axis)
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
                values = values - self._axis_origin_value(
                    axis,
                    units=values.attrs.get("units"),
                )
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

        surface_grid = self._property_sample_grid(surface.depth, samples)
        surface_depth = surface.depth.get(surface_grid).transpose(*samples.dims)
        z = self._physical_coord(samples, "z").broadcast_like(samples)
        positive = (
            axis.positive
            if axis is not None and axis.positive is not None
            else getattr(system, "normal", "up")
        )
        if str(positive or "up").strip().lower() == "down":
            return z - surface_depth
        return surface_depth - z

    def _coordinate_system_samples(self, system: Any, samples: xr.DataArray):
        coords = {
            axis.name: self._axis_coordinate(system, axis, samples)
            for axis in self._coordinate_system_axes(system)
        }
        if getattr(system, "type", None) == "surface" and "up" not in coords:
            coords["up"] = self._surface_relative_coordinate(system, samples)
        return samples.assign_coords(**coords)

    def _property_sample_grid(self, prop: Property, samples: xr.DataArray):
        system = self._property_coordinate_system(prop)
        if system is None:
            return samples
        return self._coordinate_system_samples(system, samples)

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
                raise self._property_axis_error(
                    system=system,
                    dim=str(dim),
                    samples=samples,
                )

        coords = _coords_in_data_units(coords, data.coords, data.dims)
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
        system = self._property_coordinate_system(prop)
        if system is not None and getattr(system, "type", None) == "surface":
            return self._surface_relative_property_values(prop, system, samples)
        return prop.get(self._property_sample_grid(prop, samples))

    @staticmethod
    def _physical_coord(samples: xr.DataArray, axis: str) -> xr.DataArray:
        if axis in samples.coords:
            return samples.coords[axis]
        raise ValueError(
            f"Sampling grid is missing physical coordinate {axis!r}. "
            f"Available coordinates are {list(samples.coords)}."
        )

    def _lateral_query(self, samples: xr.DataArray) -> xr.DataArray:
        lateral_axes = [
            axis for axis in ("x", "y") if axis in samples.coords and axis != "z"
        ]
        if not lateral_axes:
            raise ValueError("Layered sampling requires at least one lateral axis")
        if len(lateral_axes) == 1:
            return self._physical_coord(samples, lateral_axes[0])

        dims: List[str] = []
        coords: Dict[str, Any] = {}
        for axis in lateral_axes:
            coord = self._physical_coord(samples, axis)
            for dim in coord.dims:
                if dim not in dims:
                    dims.append(dim)
                    coords[dim] = samples.coords[dim]
            coords[axis] = coord
        return xr.DataArray(dims=dims, coords=coords)

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
        depth = surface.depth.get(self._property_sample_grid(surface.depth, coords))
        value = float(np.asarray(depth.values).reshape(-1)[0])
        return float(_convert_units(value, _property_units(surface.depth), units))

    @staticmethod
    def _sample_coordinate_mesh(
        samples: xr.DataArray,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        x = LayeredSamplingMixin._physical_coord(samples, "x").broadcast_like(samples)
        z = LayeredSamplingMixin._physical_coord(samples, "z").broadcast_like(samples)
        return x, z

    def _borehole_part_masks(
        self,
        borehole: Borehole,
        samples: xr.DataArray,
    ) -> List[Tuple[BoreholePart, xr.DataArray]]:
        if self.dimension != 2:
            return []
        x_coord = self._physical_coord(samples, "x")
        z_coord = self._physical_coord(samples, "z")
        x_units = x_coord.attrs.get("units")
        z_units = z_coord.attrs.get("units")
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
                    z_coord.values,
                    z_profile,
                    r_profile,
                )
                outer = xr.DataArray(
                    outer_values,
                    dims=z_coord.dims,
                    coords={dim: samples.coords[dim] for dim in z_coord.dims},
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

    def _get_layer_mask(self, layer, samples):
        xgrid = self._physical_coord(samples, "x")
        zgrid = self._physical_coord(samples, "z")
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
        limits["x_min"] = float(xgrid.min())
        limits["x_max"] = float(xgrid.max())

        if self.y_limits is not None and "y" in samples.coords:
            ygrid = self._physical_coord(samples, "y")
            limits["y_min"] = float(ygrid.min())
            limits["y_max"] = float(ygrid.max())
            coords_query = self._lateral_query(samples)
        else:
            coords_query = self._lateral_query(samples)

        z_units = zgrid.attrs.get("units")
        upper_grid = self._property_sample_grid(upper.depth, coords_query)
        lower_grid = self._property_sample_grid(lower.depth, coords_query)
        limits["z_min"] = _convert_surface_depth(
            upper.depth.get(upper_grid),
            upper,
            z_units,
        )
        limits["z_max"] = _convert_surface_depth(
            lower.depth.get(lower_grid),
            lower,
            z_units,
        )

        mask = (
            (zgrid <= limits["z_max"])
            & (zgrid >= limits["z_min"])
            & (xgrid <= limits["x_max"])
            & (xgrid >= limits["x_min"])
        )

        if "y" in samples.coords:
            ygrid = self._physical_coord(samples, "y")
            mask &= ygrid <= limits["y_max"]
            mask &= ygrid >= limits["y_min"]

        mask = mask.transpose(*samples.dims)
        return mask
