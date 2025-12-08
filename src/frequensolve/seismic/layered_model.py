import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.mesh.mesh_generators import HexMeshGenerator, TetMeshGenerator
from frequensolve.model.model import ModelBase, ModelSubdomain
from frequensolve.model.property import Property
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.util.class_registry import register_class
from frequensolve.util.data_file import save_data_if_new
from frequensolve.util.named_list import NamedList

__all__ = ["SimpleSurface", "Layer", "LayeredModel"]


# TODO: make a way to work in depth or elevation coordinates


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
       elevation_ref (float, optional):
          Reference elevation.
       z_phys (Property):
          Depth of the surface.
    """

    name: str = "surface"
    interface: bool = True
    z_ref: Optional[float] = None
    z_phys: Property = field(default_factory=Property)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __init__(
        self,
        name: str,
        interface: bool,
        z_ref: Optional[float],
        z_phys: Union[float, str, Path],
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        **kwargs,
    ):
        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")

        self.name = name
        self.interface = interface
        self.z_ref = z_ref
        self.z_phys = Property(data=z_phys, grid=grid, scale=scale)

    @classmethod
    def from_dict(cls, dict: Dict) -> "SimpleSurface":
        name = dict["name"]
        interface = dict["interface"]
        z_ref = dict["z_ref"]
        data = dict.get("z_phys", None)
        grid = None
        if "file" in data:
            z_phys = data["file"]
            grid = CartesianGrid.from_dict(data["grid"]).as_xarray()
        else:
            z_phys = data["value"]

        return cls(
            name=name,
            interface=interface,
            z_ref=z_ref,
            z_phys=z_phys,
            **({"grid": grid} if grid is not None else {}),
        )

    def __dict__(self):
        data = {
            "name": self.name,
            "z_ref": self.z_ref,
            "interface": self.interface,
        }
        if self.z_phys.is_constant:
            data["z_phys"] = {"value": self.z_phys.get()}
        else:
            file = self._path / (self.name + ".bin")
            file.parent.mkdir(parents=True, exist_ok=True)
            file = save_data_if_new(self.z_phys.darr, file)
            data["z_phys"] = {
                "file": file.relative_to(self._proj_path),
                "grid": self.z_phys.grid.__dict__(),
            }
        return data

    @property
    def data(self):
        return self.z_phys.data

    @property
    def extrema(self):
        """Get the extreme values (min, max) of the surface.

        Returns:
           tuple: A tuple containing the minimum and maximum z-coordinates.
        """
        min, max = self.z_phys.extrema
        min = min.values
        max = max.values
        return min, max

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path

    @property
    def _path(self) -> Path:
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
        self.z_phys.stochastic_perturbation(
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
        import matplotlib.pyplot as plt

        if self.z_phys.is_constant:
            dims = sorted(limits.keys())
            grid = xr.DataArray(dims=dims, coords=limits)
            surf = self.z_phys.get(grid=grid)
        else:
            surf = self.z_phys.get()

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
            properties = [property]
        else:
            properties = property

        if isinstance(L0, float):
            L0 = [L0]
        k0 = []
        for l0 in L0:
            k0.append(1 / l0)

        for property in properties:
            if property not in self.properties:
                raise ValueError(f"Property '{property}' not found in layer")

            if anisotropy is None:
                anisotropy = [1.0] * len(self.properties[property].data.dims)
            self.properties[property].stochastic_perturbation(
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
        self._properties[key] = Property(data=value)


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
            out += f"  upper: {self.upper.name} (z_ref = {self.upper.z_ref})\n"
        else:
            out += "  upper: None\n"

        if self.lower is not None:
            out += f"  lower: {self.lower.name} (z_ref = {self.lower.z_ref})\n"
        else:
            out += "  lower: None\n"
        return out

    def __repr__(self):
        return str(self)


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
    kwargs: Dict = field(default_factory=dict)

    _last_added: str = "none"
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None
    _surface_names: Set[str] = field(default_factory=set)
    _layer_names: Set[str] = field(default_factory=set)

    def add_layer(
        self,
        name: Optional[str] = None,
        mesh_block_id: int = -1,
        frame: str = "physical",
        properties: Optional[dict] = None,
        grid: Optional[xr.DataArray] = None,
        **kwargs,
    ) -> None:
        """Add a layer to the model.

        Args:
           name (str):
              Name of the layer.
           mesh_block_id (int):
              Unique identifier for the layer.
           frame (str):
              Frame of the layer.
           properties (dict):
              Properties of the layer.
           grid (xr.DataArray, optional):
              Xarray defining the grid for the layer (required for reading file formats
              where grid is not stored with data)
        """

        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")

        name = self._get_unique_name(name, self._layer_names)
        self._layer_names.add(name)

        layer = Layer(
            name=name,
            mesh_block_id=mesh_block_id,
            frame=frame,
            properties=properties,
            grid=grid,
        )
        self += layer

    def add_surface(
        self,
        z: Union[float, str, Path, xr.DataArray],
        name: Optional[str] = None,
        z_ref: Optional[float] = None,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        **kwargs,
    ):
        """Add a surface to the model.

        Args:
           z (float, str, Path, xr.DataArray):
              Coordinates of the surface. Can be defined as a float (constant surface),
              via a file, or as an xr.DataArray (gridded surface).
           name (str, optional):
              Name of the surface.
           z_ref (float):
              Reference z-coordinate of the surface.
           grid (xr.DataArray, optional):
              Xarray defining the grid for the surface.
        """

        interface = len(self.surfaces) == 0

        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")

        name = self._get_unique_name(name, self._surface_names)
        self._surface_names.add(name)

        surface = SimpleSurface(
            name=name,
            interface=interface,
            z_ref=z_ref,
            z_phys=z,
            grid=grid,
            scale=scale,
        )
        if z_ref is None:
            if surface.z_phys.is_constant:
                surface.z_ref = surface.z_phys.get()
            else:
                surface.z_ref = float(surface.z_phys.darr.mean().compute().values)
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
        z0, _ = self.surfaces[0].extrema  # extrema returns values already
        _, z1 = self.surfaces[-1].extrema
        return float(z0), float(z1)

    def extreme_values(self, property: str) -> Tuple[float, float]:
        vmin = 1.0e8
        vmax = -1.0e8
        for layer in self.layers:
            if property not in layer.properties:
                continue
            min, max = layer.properties[property].extrema
            vmin = min if min < vmin else vmin
            vmax = max if max > vmax else vmax
        return vmin, vmax

    def hex_mesh_generator(self, n: Optional[List[int]] = None) -> HexMeshGenerator:

        if self.dimension == 2:
            l_bound = [self.x_limits[0], self.z_limits[0]]
            u_bound = [self.x_limits[1], self.z_limits[1]]
        else:
            l_bound = [self.x_limits[0], self.y_limits[0], self.z_limits[0]]
            u_bound = [self.x_limits[1], self.y_limits[1], self.z_limits[1]]

        return HexMeshGenerator(l_bound=l_bound, u_bound=u_bound, n=n)

    def tet_mesh_generator(self, n: Optional[List[int]] = None) -> HexMeshGenerator:

        if self.dimension == 2:
            l_bound = [self.x_limits[0], self.z_limits[0]]
            u_bound = [self.x_limits[1], self.z_limits[1]]
        else:
            l_bound = [self.x_limits[0], self.y_limits[0], self.z_limits[0]]
            u_bound = [self.x_limits[1], self.y_limits[1], self.z_limits[1]]

        return TetMeshGenerator(l_bound=l_bound, u_bound=u_bound, n=n)

    @classmethod
    def from_dict(cls, data: Dict) -> "LayeredModel":
        """Creates a LayeredModel instance from a dictionary representation.

        Args:
           data (Dict): Dictionary containing model data with:
              - surfaces: List of surface dictionaries
              - All parent class dict fields

        Returns:
           LayeredModel: New LayeredModel instance created from dictionary data.
        """
        # Create copy and remove surfaces to pass rest to parent
        surfs = data.pop("surfaces")
        layers = data.pop("subdomains")

        name = data.pop("name", None)
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
        model.kwargs = data

        model += SimpleSurface.from_dict(surfs[0])

        n_surf = len(surfs)
        isurf = 1
        ilayer = 0
        while isurf < n_surf:
            # Add layer
            model += Layer.from_dict(layers[ilayer])
            ilayer += 1

            # Add any non-interface surfaces (surfaces not between layers)
            while surfs[isurf].get("interface", True) == False:
                model += SimpleSurface.from_dict(surfs[isurf])
                isurf += 1

            # Add surface
            model += SimpleSurface.from_dict(surfs[isurf])
            isurf += 1

        return model

    def __dict__(self) -> Dict:
        # Mark bottom surface as interface
        self.surfaces[-1].interface = True

        base_dict = super().__dict__()
        base_dict.update(
            {
                "_type": self.__class__.__name__,
                "x_limits": self.x_limits,
                **({"y_limits": self.y_limits} if self.y_limits is not None else {}),
                "ordering": self.ordering,
                "surfaces": [surface.__dict__() for surface in self.surfaces],
            }
        )
        base_dict.update(self.kwargs)
        return base_dict

    def __iadd__(self, other):
        if isinstance(other, SimpleSurface):

            # TODO: check that surfaces are added in monotone order

            # Add surface
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
                self.layers[-1].lower = other
            else:
                if len(self.layers) == 1:
                    prev_upper = self.lower_surface()
                else:
                    prev_upper = self.upper_surface(self.layers[-2])

                self.layers[-1].lower = prev_upper
                self.layers[-1].upper = other
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
            if self.ordering == "top_down":
                return self.surfaces[0]
            else:
                return self.surfaces[-1]

        surf = None
        if isinstance(layer, int):
            surf = self.layers[layer].upper
        elif isinstance(layer, str):
            surf = self.layers[layer].upper
        elif isinstance(layer, Layer):
            surf = layer.upper
        elif isinstance(layer, ModelSubdomain):
            for layer in self.layers:
                if layer.mesh_block_id == layer.mesh_block_id:
                    return layer.upper
        return surf

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
            if self.ordering == "top_down":
                return self.surfaces[-1]
            else:
                return self.surfaces[0]

        # TODO: add flag and method to "complete" model like this
        if self.ordering == "top_down":
            self.layers[-1].lower = self.lower_surface()
        else:
            self.layers[-1].upper = self.upper_surface()

        surf = None
        if isinstance(layer, int):
            surf = self.layers[layer].lower
        elif isinstance(layer, str):
            surf = self.layers[layer].lower
        elif isinstance(layer, Layer):
            surf = layer.lower
        elif isinstance(layer, ModelSubdomain):
            for layer in self.layers:
                if layer.mesh_block_id == layer.mesh_block_id:
                    return layer.lower
        return surf

    def get_1D_log(
        self,
        property: str,
        x: float,
        dz: float,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
    ) -> xr.DataArray:
        """Get a 1D well-log of a property."""

        if self.dimension == 3:
            raise NotImplementedError("3D sampling not implemented")

        if z_min is None:
            z_min = self.z_limits[0]
        if z_max is None:
            z_max = self.z_limits[1]
        depths = np.arange(z_min, z_max, dz)
        samples = xr.DataArray(dims=["x", "z"], coords={"z": depths, "x": [x]})
        samples = self._physical_to_reference(samples.coords)

        for layer in self.layers:
            layer.properties.keys()
            if property not in layer.properties:
                continue
            prop = layer.properties[property].get(samples)
            dims = sorted(prop.dims)
            prop = prop.transpose(*dims[::-1])
            mask = self._get_layer_mask(layer, samples)
            data = prop.where(mask)
            samples.data = np.where(~np.isnan(data), data, samples.data)

        return depths, samples.data

    def plot_1d_log(
        self,
        ax: Axes,
        property: str,
        x: float,
        dz: float,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        **kwargs,
    ):
        """Plot a 1D well-log of a property."""
        import matplotlib.pyplot as plt

        if self.dimension == 3:
            raise NotImplementedError("3D plotting not implemented")
        if isinstance(property, str):
            property = [property]

        property_units = kwargs.pop("property_units", "km/s")
        property_label = kwargs.pop("property_label", f"{property} [{property_units}]")
        aspect = kwargs.pop("aspect", None)
        show_legend = kwargs.pop("show_legend", True)
        legend_coords = kwargs.pop("legend_coords", (1.8, 1))

        depths, data = self.get_1D_log(property, x, dz, z_min, z_max)
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
        return list(props)

    def sample_uniform(self, n: Union[np.ndarray, List[int]]) -> xr.Dataset:
        """Export model on a uniform grid."""

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
        for property in self.properties:
            gridded[property] = xr.DataArray(
                dims=samples.dims,
                coords=samples.coords,
                # data=np.nan * np.ones(samples.shape),
            )
        samples = self._physical_to_reference_2d(samples)

        for layer in self.layers:
            mask = self._get_layer_mask_2d(layer, samples)

            for property in self.properties:
                if property not in layer.properties:
                    continue
                prop = layer.properties[property]

                if prop.is_constant:
                    gridded[property].data[mask] = prop.get()
                elif layer.frame == "physical":
                    gridded[property].data[mask] = prop.get(samples).data[mask]
                else:
                    for ix, x in enumerate(samples.coords["x"].values):
                        z = samples.data[ix, mask[ix, :]]
                        samp = xr.DataArray(dims=["x", "z"], coords={"z": z, "x": x})
                        gridded[property].data[ix, mask[ix, :]] = prop.get(samp).data
        return gridded

    def smooth(self, n: ArrayLike, sigma, **kwargs):
        """Smooth the model."""
        from scipy.ndimage import gaussian_filter

        gridded = self.sample_uniform(n)
        for property in self.properties:
            if np.any(np.isnan(gridded[property].data)):
                filled = gridded[property]
                for dim in filled.dims:
                    filled = filled.bfill(dim=dim).ffill(dim=dim)
                data = filled.data
            else:
                data = gridded[property].data

            gridded[property].data = gaussian_filter(data, sigma=sigma, **kwargs)
        for layer in self.layers:
            for property in layer.properties:
                layer.set_property(property, gridded[property])
        return self

    def update_from_dataset(self, dataset: xr.Dataset):
        """Update the model from an Xarray dataset."""

        for property in self.properties:
            if np.any(np.isnan(dataset[property].data)):
                filled = dataset[property]
                for dim in filled.dims:
                    filled = filled.bfill(dim=dim).ffill(dim=dim)
                data = filled.data
            else:
                data = dataset[property].data
            dataset[property].data = data
        for layer in self.layers:
            for property in layer.properties:
                layer.set_property(property, dataset[property])
        return self

    def plot(self, property: str, resolution: List[int] = [500, 500], **kwargs):
        """Plot the model."""
        import matplotlib.pyplot as plt

        if self.dimension == 2:
            x = np.linspace(self.x_limits[0], self.x_limits[1], resolution[0])
            z = np.linspace(self.z_limits[0], self.z_limits[1], resolution[1])
            samples = xr.DataArray(dims=["x", "z"], coords={"z": z, "x": x})
        elif self.dimension == 3:
            raise NotImplementedError("3D plotting not implemented")
        if self.ordering == "top_down":
            self.layers[-1].lower = self.lower_surface()
        else:
            self.layers[-1].upper = self.upper_surface()

        # Process kwargs
        units = kwargs.pop("units", "km/s")
        label = kwargs.pop("label", property)
        origin = kwargs.pop("origin", "upper")
        aspect = kwargs.pop("aspect", None)
        axes_names = kwargs.pop("axes_names", {"x": "X", "z": "Depth"})
        axes_units = kwargs.pop("axes_units", {"x": "km", "z": "km"})
        add_colorbar = kwargs.pop("add_colorbar", True)

        # Get surface plotting kwargs
        show_surfs = kwargs.pop("surfaces", True)
        surf_kwargs = {}
        surf_kwargs["color"] = kwargs.pop("linecolor", "k")
        surf_kwargs["linestyle"] = kwargs.pop("linestyle", "-")
        surf_kwargs["linewidth"] = kwargs.pop("linewidth", 1)

        # Get acquisition plotting kwargs
        acq = kwargs.pop("acquisition", None)
        scatter_kwargs = kwargs.pop("scatter_kwargs", {})

        # General kwargs
        figsize = kwargs.pop("figsize", None)
        fontsize = kwargs.pop("fontsize", 12)
        save = kwargs.pop("save", None)
        dpi = kwargs.pop("dpi", None)

        plt.rcParams.update({"font.size": fontsize})

        show = True
        if "ax" in kwargs:
            ax = kwargs.pop("ax")
            show = False
        else:
            fig = plt.figure(**({"figsize": figsize} if figsize is not None else {}))
            ax = fig.gca()

        if units is not None:
            samples.attrs["units"] = units
        if label is not None:
            samples.attrs["long_name"] = label
        if axes_names is not None:
            samples.coords["x"].attrs["long_name"] = axes_names["x"]
            samples.coords["z"].attrs["long_name"] = axes_names["z"]
            if axes_units is not None:
                samples.coords["x"].attrs["units"] = axes_units["x"]
                samples.coords["z"].attrs["units"] = axes_units["z"]

        vmin, vmax = self.extreme_values(property)
        vmin = kwargs.pop("vmin", vmin)
        vmax = kwargs.pop("vmax", vmax)

        for layer in self.layers:
            if property not in layer.properties:
                continue

            # Plot layer in reference frame
            if layer.frame == "reference":
                samp = self._get_layer_samples_ref(layer, samples, property)
                dims = sorted(samp.dims)
                plt.pcolormesh(
                    samp.coords["Xcoord"].values,
                    samp.coords["Zcoord"].values,
                    samp[property].transpose(*dims).values,
                    shading="gouraud",
                    vmin=vmin,
                    vmax=vmax,
                    **kwargs,
                )
            # Plot layer in physical frame
            else:
                prop = layer.properties[property].get(samples)
                mask = self._get_layer_mask(layer, samples)
                data = prop.where(mask)
                data = data.transpose(*samples.dims)
                samples.data = np.where(~np.isnan(data), data, samples.data)

        samples = samples.clip(min=vmin, max=vmax)
        im = samples.plot.imshow(
            ax=ax,
            x="x",
            vmin=vmin,
            vmax=vmax,
            extend="neither",
            add_colorbar=add_colorbar,
            **kwargs,
        )
        # Plot surfaces
        if show_surfs:
            for surf in self.surfaces:
                limits = {}
                limits["x"] = self.x_limits
                if self.y_limits is not None:
                    limits["y"] = self.y_limits
                surf.plot(limits=limits, ax=ax, **surf_kwargs)

        if acq is not None:
            self._plot_acquisition(acq, ax, **scatter_kwargs)
        if aspect == "equal":
            ax.set_aspect("equal")
        if origin == "upper":
            ax.invert_yaxis()
        if save is not None:
            plt.savefig(
                save, bbox_inches="tight", **({"dpi": dpi} if dpi is not None else {})
            )
            plt.close()
        else:
            if show:
                plt.show()

    def _plot_acquisition(self, acquisition: Acquisition, ax, **kwargs):
        from matplotlib.pyplot import cm

        colors = ["b", "g", "orange", "y", "c", "m"]

        plot_sources = kwargs.pop("plot_sources", True)
        groups = kwargs.pop("groups", None)
        if groups is None:
            groups = acquisition.receiver_groups

        # Plot receivers
        for igrp, group in enumerate(groups):
            coords = group.coordinates.get()
            if group.frame == "reference":
                coords = self._reference_to_physical(coords)

            # Plot receiver coordinates as scatter points with higher saturation
            ax.scatter(
                coords[:, 0],
                coords[:, -1],
                marker=".",
                s=30,
                label=f"Receivers ({group.name})",
                zorder=6,
                color=colors[igrp],
                **kwargs,
            )
            # ax.legend(bbox_to_anchor=(0, 1.02), loc="lower left")

        if not plot_sources:
            return

        # Plot sources
        for igrp, group in enumerate(acquisition.source_groups):

            coords = group.get_coordinates()
            color = colors[len(acquisition.receiver_groups)]

            # Map reference coordinates to physical coordinates
            ilist = []
            xlist = []
            if group.source.frame == "reference":
                ilist.append(i)
                xlist.append(coords[i, :])

            if len(xlist) > 0:
                xlist = np.array(xlist)
                xlist = self._reference_to_physical(xlist)
                for i in ilist:
                    coords[i, :] = xlist[i, :]

            # Plot source coordinates as scatter points
            label = f"Sources" if igrp == 0 else None
            ax.scatter(
                coords[:, 0],
                coords[:, -1],
                marker="*",
                s=120,
                label=label,
                zorder=7,
                facecolors="#fff700",
                edgecolors="r",
                linewidths=0.5,
                **kwargs,
            )
            ax.legend(bbox_to_anchor=(0, 1.02), loc="lower left")

    def _get_layer_samples_ref(self, layer, samples, property):
        prop = layer.properties[property]
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

        z_min = upper.z_ref
        z_max = lower.z_ref
        z0 = upper.z_phys.get(xgrid)
        z1 = lower.z_phys.get(xgrid)

        mask = samples.coords["z"].values > z_min
        mask &= samples.coords["z"].values < z_max
        zgrid = samples.coords["z"].values[mask]

        tmp = xr.DataArray(
            dims=["z", "x"],
            coords={"x": xgrid, "z": [z_min, z_max]},
            data=[z0, z1],
        )
        zvals = tmp.interp(coords={"x": xgrid, "z": zgrid})

        samp = xr.Dataset({property: prop.get(zvals)})
        samp = samp.assign_coords(
            {
                "Xcoord": (("x", "z"), xgrid.broadcast_like(zvals).values.T),
                "Zcoord": (("x", "z"), zvals.values.T),
            }
        )
        return samp

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

        limits = {}
        limits["x_min"] = self.x_limits[0]
        limits["x_max"] = self.x_limits[1]
        if self.y_limits is not None:
            limits["y_min"] = self.y_limits[0]
            limits["y_max"] = self.y_limits[1]
        if layer.frame == "reference":
            limits["z_min"] = upper.z_ref
            limits["z_max"] = lower.z_ref
        else:
            limits["z_min"] = upper.z_phys.get(xgrid)
            limits["z_max"] = lower.z_phys.get(xgrid)

        mask = (
            (samples.coords["z"] <= limits["z_max"])
            & (samples.coords["z"] >= limits["z_min"])
            & (samples.coords["x"] <= limits["x_max"])
            & (samples.coords["x"] >= limits["x_min"])
        )
        mask = mask.transpose(*samples.dims)
        return mask

    def _reference_to_physical(self, coords: np.ndarray) -> np.ndarray:
        """Map coordinates from reference to physical coordinates.

        Args:
            coords (np.ndarray): Coordinates to map.

        Returns:
            np.ndarray: Mapped coordinates.
        """
        x_coords = xr.DataArray(dims=["x"], coords={"x": coords[:, 0]})
        z_coords = coords[:, -1]
        z_phys = np.zeros_like(z_coords)
        found = np.zeros_like(z_coords, dtype=bool)

        for layer in self.layers:
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

            z_min = upper.z_ref
            z_max = lower.z_ref
            zl = upper.z_phys.get(x_coords).values
            zu = lower.z_phys.get(x_coords).values

            layer_mask = (z_coords >= z_min) & (z_coords <= z_max) & ~found

            if np.any(layer_mask):
                if layer.frame == "reference":
                    alpha = (z_coords[layer_mask] - z_min) / (z_max - z_min)
                    z_phys[layer_mask] = zl[layer_mask] + alpha * (
                        zu[layer_mask] - zl[layer_mask]
                    )
                else:
                    z_phys[layer_mask] = z_coords[layer_mask]
                found[layer_mask] = True

        if not np.all(found):
            raise ValueError("Some points were not found in any layer")

        coords = coords.copy()
        coords[:, -1] = z_phys
        return coords

    def _get_layer_mask_2d(self, layer, samples):
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

        limits = {}
        limits["x_min"] = self.x_limits[0]
        limits["x_max"] = self.x_limits[1]
        if self.y_limits is not None:
            limits["y_min"] = self.y_limits[0]
            limits["y_max"] = self.y_limits[1]
        if layer.frame == "reference":
            limits["z_min"] = upper.z_ref
            limits["z_max"] = lower.z_ref
        else:
            z_min = upper.z_phys.get(xgrid).values  # shape (nx,)
            z_max = lower.z_phys.get(xgrid).values  # shape (nx,)
            limits["z_min"] = np.broadcast_to(
                z_min[:, np.newaxis], samples.shape
            )  # shape (nx, nz)
            limits["z_max"] = np.broadcast_to(
                z_max[:, np.newaxis], samples.shape
            )  # shape (nx, nz)

        x_mask = (
            (samples.coords["x"] <= limits["x_max"])
            & (samples.coords["x"] >= limits["x_min"])
        ).values
        x_mask = np.broadcast_to(x_mask[:, np.newaxis], samples.shape)
        mask = (
            x_mask
            & (samples.data <= limits["z_max"])
            & (samples.data >= limits["z_min"])
        )
        return mask

    def _physical_to_reference_2d(self, samples: xr.DataArray) -> xr.DataArray:
        """Map coordinates from physical to reference coordinates for a uniform grid.

        Args:
            samples (xr.DataArray): DataArray with x and z coordinates defining a uniform grid.

        Returns:
            xr.DataArray: DataArray with same coords and dims, but z values mapped to reference frame.
        """
        x_coords = samples.coords["x"]
        z_coords = samples.coords["z"]
        Z_coords = np.broadcast_to(z_coords, (len(x_coords), len(z_coords)))  # (nx, nz)
        z_ref = np.zeros((len(x_coords), len(z_coords)))
        found = np.zeros((len(x_coords), len(z_coords)), dtype=bool)

        for layer in self.layers:
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

            zl = upper.z_ref
            zu = lower.z_ref
            z_min = upper.z_phys.get(x_coords).values
            z_max = lower.z_phys.get(x_coords).values

            Z_min = z_min[:, np.newaxis]  # (nx, 1)
            Z_max = z_max[:, np.newaxis]  # (nx, 1)
            layer_mask = (Z_coords >= Z_min) & (Z_coords <= Z_max) & ~found  # (nx, nz)

            if np.any(layer_mask):
                if layer.frame == "reference":
                    alpha = (Z_coords - Z_min) / (Z_max - Z_min)  # (nx, nz)
                    z_ref[layer_mask] = zl + alpha[layer_mask] * (zu - zl)
                else:
                    z_ref[layer_mask] = Z_coords[layer_mask]
                found[layer_mask] = True

        # if not np.all(found):
        #     raise ValueError("Some points were not found in any layer")

        return xr.DataArray(
            dims=["x", "z"], coords={"x": x_coords, "z": z_coords}, data=z_ref
        )

    def _physical_to_reference(self, coords: Dict[str, np.ndarray]) -> np.ndarray:
        """Map coordinates from physical to reference coordinates.

        Args:
            coords (np.ndarray): Coordinates to map.

        Returns:
            np.ndarray: Mapped coordinates.
        """
        x_coords = xr.DataArray(dims=["x"], coords={"x": coords["x"]})
        z_coords = coords["z"]
        z_ref = np.zeros_like(z_coords)
        found = np.zeros_like(z_coords, dtype=bool)

        for layer in self.layers:
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

            zl = upper.z_ref
            zu = lower.z_ref
            z_min = upper.z_phys.get(x_coords).values
            z_max = lower.z_phys.get(x_coords).values

            layer_mask = (z_coords >= z_min) & (z_coords <= z_max) & ~found

            if np.any(layer_mask):
                if layer.frame == "reference":
                    alpha = (z_coords[layer_mask] - z_min) / (z_max - z_min)
                    z_ref[layer_mask] = zl + alpha * (zu - zl)
                else:
                    z_ref[layer_mask] = z_coords[layer_mask]
                found[layer_mask] = True

        if not np.all(found):
            raise ValueError("Some points were not found in any layer")

        return xr.DataArray(dims=["z", "x"], coords={"z": z_ref, "x": coords["x"]})

    @staticmethod
    def _get_unique_name(name: Optional[str], names: Set[str]) -> str:
        orig_name = name
        warn_flag = True
        if name is None:
            warn_flag = False
            name = "no_name"

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
        return self._proj_path / self._rel_path


# TODO: info functions like this:
#     max_elevation()
#     max_layer_thickness()
#     min_elevation()
#     min_layer_thickness()
#     check_intersection()
