import copy
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.mesh.mesh_generators import HexMeshGenerator, TetMeshGenerator
from frequensolve.model.model import ModelBase, ModelSubdomain
from frequensolve.model.property import Property, canonical_property_name
from frequensolve.units import unit_expression, ureg
from frequensolve.util.class_registry import register_class
from frequensolve.util.mixins import merge_extra
from frequensolve.util.named_list import NamedList

__all__ = ["SimpleSurface", "Layer", "LayeredModel"]

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


def _convert_units(
    value: float, source_units: Optional[str], target_units: Optional[str]
):
    if not source_units or not target_units or source_units == target_units:
        return value
    return (value * ureg(source_units)).to(target_units).magnitude


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


@register_class
@dataclass(kw_only=True)
class LayeredModel(ModelBase):
    """A class for representing a layered seismic model.

    Attributes:
       name (str):                         The name of the model.
       dimension (int):                    Dimension of the model.
       subdomains (List[ModelSubdomain]):  Model layers.
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
    ordering: Literal["top_down", "bottom_up"] = "top_down"
    extra: Dict[str, Any] = field(default_factory=dict)

    _last_added: str = "none"
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None
    _surface_names: Set[str] = field(default_factory=set)
    _layer_names: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
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

    @property
    def surface_names(self):
        return [surf.name for surf in self.surfaces]

    @property
    def layer_names(self):
        return [layer.name for layer in self.layers]

    @property
    def layers(self):
        return self.subdomains

    @property
    def z_limits(self):
        if len(self.surfaces) < 2:
            raise ValueError("LayeredModel requires at least two surfaces for z_limits")
        z0, _ = self.surfaces[0].extrema  # extrema returns values already
        _, z1 = self.surfaces[-1].extrema
        return float(z0), float(z1)

    def property_units(self, property: str) -> Optional[str]:
        """Return the first declared units for a model property."""

        property = canonical_property_name(property)
        for layer in self.layers:
            if property not in layer.properties:
                continue
            units = _property_units(layer.properties[property])
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
        for layer in self.layers:
            if property not in layer.properties:
                continue
            prop = layer.properties[property]
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
        layers = data.pop("subdomains")
        if len(surfs) < 2:
            raise ValueError("LayeredModel requires at least two surfaces")
        if len(layers) == 0:
            raise ValueError("LayeredModel requires at least one layer")

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
            prop = layer_prop.get(samples)
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
        for layer in self.layers:
            props.update(layer.properties.keys())
        return sorted(props)

    def sample_uniform(self, n: Union[np.ndarray, List[int]]) -> xr.Dataset:
        """Export model on a uniform grid."""
        n = list(n)
        expected = 2 if self.dimension == 2 else 3
        if len(n) != expected:
            raise ValueError(f"Expected {expected} sample counts for {self.dimension}D")
        if any(count < 2 for count in n):
            raise ValueError("All sample counts must be >= 2")

        xl = np.linspace(self.x_limits[0], self.x_limits[1], n[0])
        zl = np.linspace(self.z_limits[0], self.z_limits[1], n[-1])
        if self.dimension == 2:
            samples = xr.DataArray(dims=["x", "z"], coords={"z": zl, "x": xl})
        elif self.dimension == 3:
            yl = np.linspace(self.y_limits[0], self.y_limits[1], n[1])
            samples = xr.DataArray(
                dims=["x", "y", "z"], coords={"z": zl, "y": yl, "x": xl}
            )
        else:
            raise ValueError("Invalid dimension")

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
                    data = prop.get(samples).transpose(*samples.dims)
                data = _convert_dataarray_units(data, source_units, target_units)
                gridded[name].data = np.where(
                    mask.values,
                    data.values,
                    gridded[name].data,
                )
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

        limits["z_min"] = upper.depth.get(coords_query)
        limits["z_max"] = lower.depth.get(coords_query)

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
