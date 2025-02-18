from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.model.model import ModelBase, ModelSubdomain
from frequensolve.model.property import Property
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.util.class_registry import register_class
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
        z_phys: Property,
        xarr: Optional[xr.DataArray] = None,
        scale: float = 1.0,
    ):
        self.name = name
        self.interface = interface
        self.z_ref = z_ref
        self.z_phys = Property(data=z_phys, xarr=xarr, scale=scale)

    @classmethod
    def from_dict(cls, dict: Dict) -> "SimpleSurface":
        data = dict["z_phys"]
        if isinstance(data, float):
            xarr = None
        else:
            xarr = CartesianGrid.from_dict(dict["grid"]).as_xarray()
        return cls(
            name=dict["name"],
            interface=dict.get("interface", True),
            z_ref=dict.get("z_ref"),
            z_phys=data,
            xarr=xarr,
        )

    def __dict__(self):
        if self.z_phys.is_constant:
            type = "ConstantSurface"
            z_phys = self.z_phys.get()
        else:
            type = "GridSurface"
            file = self._path / (self.name + ".bin")
            self.z_phys.write(file)
            z_phys = file.relative_to(self._proj_path)

            grid = self.z_phys.grid

            # TODO: This is again a nasty hack to get around not specifying dims in Grid
            #       update both codes to use named dimensions
            if len(grid.n) == 3:
                grid.n = grid.n[:1]
                grid.dx = grid.dx[:1]
                grid.x0 = grid.x0[:1]
                grid.x1 = grid.x1[:1]

        return {
            "_type": type,
            "name": self.name,
            "z_phys": z_phys,
            **({"z_ref": self.z_ref} if self.z_ref is not None else {}),
            **({"interface": self.interface} if not self.interface else {}),
            **({"grid": grid} if not self.z_phys.is_constant else {}),
        }

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
        xarr: Optional[xr.DataArray] = None,
        L0: Union[float, List[float]] = 1.0,
        nu: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        """Perturb the dataset by a Von Karmanstochastic field.

        std (float):
           Standard deviation of the perturbation
        xarr (xr.DataArray):
           Xarray with final shape of the perturbation
        L0 (float):
           Characteristic length scale of the perturbation
        nu (float):
           Stochastic field smoothness parameter
           (nu -> 0: less smooth, nu -> 1 more smoother)
        seed (int):
           Random seed (for reproducibility)
        """
        if isinstance(L0, float):
            L0 = [L0]
        k0 = []
        for l0 in L0:
            k0.append(1 / l0)
        self.z_phys.stochastic_perturbation(
            std=std,
            method="von_karman",
            xarr=xarr,
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
            xarr = xr.DataArray(dims=dims, coords=limits)
            surf = self.z_phys.get(xarr=xarr)
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

    def perturb(
        self,
        property: Union[str, List[str]],
        std: float,
        xarr: Optional[xr.DataArray] = None,
        L0: Union[float, List[float]] = 1.0,
        nu: float = 0.5,
        anisotropy: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Perturb the dataset by a Von Karmanstochastic field.

        This just wraps the stochastic_perturbation method of Property.

        Args:
           property (str):
              Name of the property to perturb
           std (float):
              Standard deviation of the perturbation
           xarr (xr.DataArray):
              Xarray with final shape of the perturbation
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
                xarr=xarr,
                k0=k0,
                nu=nu,
                anisotropy=anisotropy,
                seed=seed,
                type="additive",
            )


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
    _layer_to_surfs: List[LayerBounds] = field(default_factory=list)
    _last_added: str = "none"
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None
    _names: List[str] = field(default_factory=list)

    def add_layer(
        self,
        name: Optional[str] = None,
        mesh_block_id: int = -1,
        frame: str = "physical",
        properties: Optional[dict] = None,
        xarr: Optional[xr.DataArray] = None,
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
           xarr (xr.DataArray, optional):
              Xarray with final shape of the layer (required for reading file formats
              where grid is not stored with data)
        """
        layer = Layer(
            name=name,
            mesh_block_id=mesh_block_id,
            frame=frame,
            properties=properties,
            xarr=xarr,
        )
        self += layer

    def add_surface(
        self,
        z: Union[float, str, Path, xr.DataArray],
        name: str = "surface",
        z_ref: Optional[float] = None,
        xarr: Optional[xr.DataArray] = None,
        scale: float = 1.0,
    ):
        """Add a surface to the model.

        Args:
           z (float, str, Path, xr.DataArray):
              Coordinates of the surface. Can be defined as a float (constant surface),
              via a file, or as an xr.DataArray (gridded surface).
           name (str):
              Name of the surface.
           z_ref (float):
              Reference z-coordinate of the surface.
           xarr (xr.DataArray, optional):
              Xarray with final shape of the surface.
        """

        interface = len(self.surfaces) == 0
        surface = SimpleSurface(
            name=name,
            interface=interface,
            z_ref=z_ref,
            z_phys=z,
            xarr=xarr,
            scale=scale,
        )
        if z_ref is None:
            if surface.z_phys.is_constant:
                surface.z_ref = surface.z_phys.get()
            else:
                surface.z_ref = surface.z_phys.mean().values
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
        data_copy = data.copy()
        surfs = data_copy.pop("surfaces")
        layers = data_copy.pop("subdomains")

        name = data["name"]
        dimension = data["dimension"]
        x_limits = data["x_limits"]
        y_limits = data.get("y_limits")
        ordering = data.get("ordering", "top_down")
        model = LayeredModel(
            name=name,
            dimension=dimension,
            x_limits=x_limits,
            y_limits=y_limits,
            ordering=ordering,
        )

        model += SimpleSurface.from_dict(surfs[0])

        nsurfs = len(surfs)
        nlayers = len(layers)
        isurf = 1
        ilayer = 0
        while isurf < nsurfs:
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
        return base_dict

    def __iadd__(self, other):
        if isinstance(other, SimpleSurface):

            # TODO: check that surfaces are added in monotone order

            # Add surface
            self.surfaces.append(other)

            # Build layer-to-surface mapping
            if len(self.surfaces) > 1:
                if self.ordering == "top_down":
                    self._layer_to_surfs[-1].lower = other
                else:
                    self._layer_to_surfs[-1].upper = other
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

                self._layer_to_surfs.append(
                    LayerBounds(
                        upper=prev_lower,
                        layer=other,
                    )
                )
            else:
                if len(self.layers) == 1:
                    prev_upper = self.lower_surface()
                else:
                    prev_upper = self.upper_surface(self.layers[-2])

                self._layer_to_surfs.append(
                    LayerBounds(
                        lower=prev_upper,
                        layer=other,
                    )
                )
            self._last_added = "layer"
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

        for layer_bounds in self._layer_to_surfs:
            if isinstance(layer, int):
                if layer_bounds.layer.mesh_block_id == layer:
                    return layer_bounds.upper
            elif isinstance(layer, str):
                if layer_bounds.layer.name == layer:
                    return layer_bounds.upper
            elif isinstance(layer, ModelSubdomain):
                if layer_bounds.layer == layer:
                    return layer_bounds.upper
            else:
                raise ValueError(f"Layer '{layer}' not found")

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

        for layer_bounds in self._layer_to_surfs:
            if isinstance(layer, int):
                if layer_bounds.layer.mesh_block_id == layer:
                    return layer_bounds.lower
            elif isinstance(layer, str):
                if layer_bounds.layer.name == layer:
                    return layer_bounds.lower
            elif isinstance(layer, ModelSubdomain):
                if layer_bounds.layer == layer:
                    return layer_bounds.lower
            else:
                raise ValueError(f"Layer '{layer}' not found")

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
            self._layer_to_surfs[-1].lower = self.lower_surface()
        else:
            self._layer_to_surfs[-1].upper = self.upper_surface()

        save = kwargs.pop("save", None)
        show_surfs = kwargs.pop("surfaces", True)
        figsize = kwargs.pop("figsize", None)
        fontsize = kwargs.pop("fontsize", 12)
        units = kwargs.pop("units", "km/s")
        label = kwargs.pop("label", property)
        origin = kwargs.pop("origin", "upper")
        aspect = kwargs.pop("aspect", None)
        axes_names = kwargs.pop("axes_names", {"x": "X", "z": "Depth"})
        axes_units = kwargs.pop("axes_units", {"x": "km", "z": "km"})
        acq = kwargs.pop("acquisition", None)
        scatter_kwargs = kwargs.pop("scatter_kwargs", {})

        plt.rcParams.update({"font.size": fontsize})

        if "ax" in kwargs:
            ax = kwargs.pop("ax")
            show = False
        else:
            if figsize is not None:
                fig = plt.figure(figsize=figsize)
            else:
                fig = plt.figure()
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

        vrange = vmax - vmin
        vmin -= 0.1 * vrange
        vmax += 0.1 * vrange

        for layer in self.layers:

            if property not in layer.properties:
                continue

            upper = self.upper_surface(layer)
            lower = self.lower_surface(layer)

            prop = layer.properties[property]

            if layer.frame == "reference":
                limits = {}
                if self.ordering == "top_down":
                    limits["z_min"] = upper.z_ref
                    limits["z_max"] = lower.z_ref
                else:
                    limits["z_max"] = upper.z_ref
                    limits["z_min"] = lower.z_ref

                if prop.is_constant:
                    xgrid = samples.coords["x"]
                else:
                    xgrid = prop.get().coords["x"]

                mask = (samples.coords["z"].values > limits["z_min"]) & (
                    samples.coords["z"].values < limits["z_max"]
                )
                zgrid = samples.coords["z"].values[mask]

                # Get surface values at xgrid
                if self.ordering == "top_down":
                    z0 = upper.z_phys.get(xgrid)
                    z1 = lower.z_phys.get(xgrid)
                else:
                    z1 = lower.z_phys.get(xgrid)
                    z0 = upper.z_phys.get(xgrid)

                z0p = limits["z_min"]
                z1p = limits["z_max"]

                tmp = xr.DataArray(
                    dims=["z", "x"],
                    coords={"x": xgrid, "z": [z0p, z1p]},
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
            else:
                data = prop.get(samples)
                xgrid = samples.coords["x"]

                limits = {}
                limits["x_min"] = self.x_limits[0]
                limits["x_max"] = self.x_limits[1]
                if self.y_limits is not None:
                    limits["y_min"] = self.y_limits[0]
                    limits["y_max"] = self.y_limits[1]
                if self.ordering == "top_down":
                    limits["z_min"] = upper.z_phys.get(xgrid)
                    limits["z_max"] = lower.z_phys.get(xgrid)
                else:
                    limits["z_max"] = upper.z_phys.get(xgrid)
                    limits["z_min"] = lower.z_phys.get(xgrid)

                mask = (
                    (data.z < limits["z_max"])
                    & (data.z > limits["z_min"])
                    & (data.x < limits["x_max"])
                    & (data.x > limits["x_min"])
                )
                da = data.where(mask)
                samples.data = np.where(~np.isnan(da), da, samples.data)

        samples.plot.imshow(ax=ax, x="x", vmin=vmin, vmax=vmax, **kwargs)

        # Get surface plotting kwargs
        line_color = kwargs.pop("line_color", "k")
        line_style = kwargs.pop("line_style", "-")
        line_width = kwargs.pop("line_width", 1.5)

        if show_surfs:
            for surf in self.surfaces:
                limits = {}
                limits["x"] = self.x_limits
                if self.y_limits is not None:
                    limits["y"] = self.y_limits

                surf.plot(
                    limits=limits,
                    ax=ax,
                    color=line_color,
                    linestyle=line_style,
                    linewidth=line_width,
                    **kwargs,
                )

        # Plot acquisition if provided
        if acq is not None:
            self._plot_acquisition(acq, ax, **scatter_kwargs)

        if aspect == "equal":
            ax.set_aspect("equal")

        # Set limits with padding after everything is plotted
        xL = self.x_limits[1] - self.x_limits[0]
        zL = self.z_limits[1] - self.z_limits[0]
        L = max(xL, zL)

        ax.set_xlim([self.x_limits[0] - 0.02 * L, self.x_limits[1] + 0.02 * L])

        if self.y_limits is None:
            ax.set_ylim([self.z_limits[0] - 0.02 * L, self.z_limits[1] + 0.02 * L])
        else:
            raise NotImplementedError("2D plotting not implemented")

        if origin == "upper":
            ax.invert_yaxis()

        if save is not None:
            plt.savefig(save, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path / self.name
        for subdomain in self.subdomains:
            subdomain._set_path(proj_path, self._rel_path)
        for surface in self.surfaces:
            surface._set_path(proj_path, self._rel_path)

    def _plot_acquisition(self, acquisition: Acquisition, ax, **kwargs):
        from matplotlib.pyplot import cm

        colors = ["r", "b", "g", "o", "y", "c", "m"]

        # Plot receivers
        for igrp, group in enumerate(acquisition.receiver_groups):
            coords = group.coordinates.get()
            if group.frame == "reference":
                coords = self._map_to_physical(coords)

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
            ax.legend(bbox_to_anchor=(0, 1.02), loc="lower left")

        # Plot sources
        group = acquisition.source_group

        coords = group.get_coordinates()
        color = colors[len(acquisition.receiver_groups)]

        # Map reference coordinates to physical coordinates
        ilist = []
        xlist = []
        for i, source in enumerate(group.sources):
            if source.frame == "reference":
                ilist.append(i)
                xlist.append(coords[i, :])

        xlist = np.array(xlist)
        xlist = self._map_to_physical(xlist)
        for i in ilist:
            coords[i, :] = xlist[i, :]

        # Plot source coordinates as scatter points
        ax.scatter(
            coords[:, 0],
            coords[:, -1],
            marker="x",
            s=80,
            label=f"Sources",
            zorder=7,
            color=color,
            **kwargs,
        )
        ax.legend(bbox_to_anchor=(0, 1.02), loc="lower left")

    def _map_to_physical(self, coords: np.ndarray) -> np.ndarray:
        """Map coordinates from reference to physical coordinates.

        Args:
            coords (np.ndarray): Coordinates to map.

        Returns:
            np.ndarray: Mapped coordinates.
        """

        for layer in self.layers:
            upper = self.upper_surface(layer)
            lower = self.lower_surface(layer)

            if self.ordering == "top_down":
                z_min = upper.z_ref
                z_max = lower.z_ref
            else:
                z_max = upper.z_ref
                z_min = lower.z_ref

            # TODO: for now I'm assuming all are in same layer
            if any(coords[:, -1] < z_min) or any(coords[:, -1] > z_max):
                continue

            # Convert coordinates to xarray
            xcoords = xr.DataArray(dims=["x"], coords={"x": coords[:, 0]})

            # Get surface z values
            if self.ordering == "top_down":
                zl = upper.z_phys.get(xcoords).values
                zu = lower.z_phys.get(xcoords).values
            else:
                zl = lower.z_phys.get(xcoords).values
                zu = upper.z_phys.get(xcoords).values

            # Linear interpolation from reference to physical coordinates
            alpha = (coords[:, -1] - z_min) / (z_max - z_min)
            z_phys = zl + alpha * (zu - zl)

            # Ensure z_phys has same shape as coords[:,-1]
            if len(coords.shape) == 1:
                z_phys = z_phys[0]  # Take single value for single coordinate

            coords[:, -1] = z_phys
            return coords
        raise ValueError(f"No layer found")

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path


# TODO: info functions like this:
#     max_elevation()
#     max_layer_thickness()
#     min_elevation()
#     min_layer_thickness()
#     check_intersection()
