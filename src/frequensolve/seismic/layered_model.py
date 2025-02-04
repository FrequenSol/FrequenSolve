import numpy as np
import xarray as xr

from pathlib      import Path
from abc          import ABC, abstractmethod
from dataclasses  import dataclass, field
from typing       import Optional, List, Literal, Union, Dict, Tuple
from numpy.typing import ArrayLike

from frequensolve.geometry.grids      import *  # noqa
from frequensolve.model.model         import *  # noqa
from frequensolve.model.property      import *  # noqa
from frequensolve.util.class_registry import *  # noqa
from frequensolve.util.named_list     import *  # noqa

__all__ = ['SimpleSurface', 'Layer','LayeredModel']


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
   name:         str = "surface"
   interface:    bool = True
   z_ref:        Optional[float] = None
   z_phys:       Property = field(default_factory=Property)
   _proj_path:   Optional[Path] = None
   _rel_path:    Optional[Path] = None

   def __init__(self, 
                name: str, 
                interface: bool, 
                z_ref: Optional[float], 
                z_phys: Property,
                xarr: Optional[xr.DataArray] = None):
      self.name = name
      self.interface = interface
      self.z_ref = z_ref
      self.z_phys = Property(data=z_phys, xarr=xarr)

   @classmethod
   def from_dict(cls, dict: Dict) -> "SimpleSurface":
      data = dict["z_phys"]
      if isinstance(data, float):
         xarr = None
      else:
         xarr = CartesianGrid.from_dict(dict["grid"]).as_xarray()
      return cls(
         name       = dict["name"],
         interface  = dict.get("interface",True),
         z_ref      = dict.get("z_ref"),
         z_phys     = data,
         xarr       = xarr
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
         **({"grid": grid} if not self.z_phys.is_constant else {})
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

   def perturb(self, 
               std: float, 
               xarr: Optional[xr.DataArray] = None,
               L0: float = 1.0,
               nu: float = 1.0,
               seed: Optional[int] = None) -> None:
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
      self.z_phys.stochastic_perturbation(std=std, 
                                          method="von_karman", 
                                          xarr=xarr, 
                                          k0=1/L0, 
                                          nu=nu, 
                                          seed=seed)
      

   def plot(self,
            limits: Dict[str, ArrayLike],
            **kwargs):
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
            ax = fig.add_subplot(111, projection='3d')

         surf = ax.plot_surface(x, y, z, **kwargs)
         ax.set_xlabel('X')
         ax.set_ylabel('Y') 
         ax.set_zlabel('Z')

      if show:
         plt.show()


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class Layer(ModelSubdomain):

   def perturb(self,
               property: Union[str, List[str]],
               std: float, 
               xarr: Optional[xr.DataArray] = None,
               L0: float = 1.0,
               nu: float = 0.5,
               anisotropy: Optional[List[float]] = None,
               seed: Optional[int] = None) -> None:
      
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

      for property in properties:
         if property not in self.properties:
            raise ValueError(f"Property '{property}' not found in layer")

         if anisotropy is None:
            anisotropy = [1.0]*len(self.properties[property].data.dims)
         self.properties[property].stochastic_perturbation(std=std, 
                                                         method="von_karman", 
                                                         xarr=xarr, 
                                                         k0=1/L0, 
                                                         nu=nu, 
                                                         anisotropy=anisotropy, 
                                                         seed=seed)


# Helper class for LayeredModel
@dataclass(kw_only=True, slots=True)
class LayerBounds:
   """Defines the bounding surfaces of a model layer.

   Attributes:
      upper (Surface): The upper bounding surface.
      lower (Surface): The lower bounding surface.
      layer_id (int):  Unique identifier for the layer.
   """
   upper:      Optional[SimpleSurface] = None
   lower:      Optional[SimpleSurface] = None
   layer:      Layer

   def __str__(self):
      out =  f"Layer {self.layer.name}\n"

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
      interface_flag (bool):              Whether to flag interfaces between layers.
      ordering (Literal["top_down", "bottom_up"]): 
         Ordering of layers. Defaults to "top_down".

      Note: 
         A 'surface' must be added at the top and bottom of the model, and at 
         least one surface must be added between 'layers'.
   """
   x_limits:         List[float]
   y_limits:         Optional[List[float]]           = None
   surfaces:         NamedList                       = field(default_factory=NamedList)
   ordering:         Literal["top_down","bottom_up"] = "top_down"
   interface_flag:   bool                            = True
   _layer_to_surfs:  List[LayerBounds]               = field(default_factory=list)
   _last_added:      str                             = "none"
   _proj_path:       Optional[Path] = None
   _rel_path:        Optional[Path] = None
   _names:           List[str] = field(default_factory=list)

   def add_layer(self,
                 name:          Optional[str] = None,
                 mesh_block_id: int = -1,
                 frame:         str = "physical",
                 properties:    Optional[dict] = None,
                 xarr:          Optional[xr.DataArray] = None) -> None:
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
                  name          = name,
                  mesh_block_id = mesh_block_id,
                  frame         = frame,
                  properties    = properties,
                  xarr          = xarr
               )
      self += layer

   def add_surface(self,
                   z:          Union[float, str, Path, xr.DataArray],
                   name:       str = "surface",
                   z_ref:      Optional[float] = None,
                   xarr:       Optional[xr.DataArray] = None):
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

      surface = SimpleSurface(
                     name       = name,
                     interface  = self.interface_flag,
                     z_ref      = z_ref,
                     z_phys     = z,
                     xarr       = xarr
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
      z0,_ = self.surfaces[0].extrema
      _,z1 = self.surfaces[-1].extrema
      return [z0, z1]
   
   def extreme_values(self, property: str) -> Tuple[float, float]:
      vmin = 1.e8; vmax = -1.e8
      for layer in self.layers:
         if property not in layer.properties:
            continue
         min, max = layer.properties[property].extrema
         vmin = min if min < vmin else vmin
         vmax = max if max > vmax else vmax
      return vmin, vmax
         
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

      name      = data["name"]     
      dimension = data["dimension"]
      x_limits  = data["x_limits"]
      y_limits  = data.get("y_limits")
      ordering  = data.get("ordering", "top_down")
      model = LayeredModel(name      = name, 
                           dimension = dimension, 
                           x_limits  = x_limits, 
                           y_limits  = y_limits, 
                           ordering  = ordering)
      
      model += SimpleSurface.from_dict(surfs[0])
      
      nsurfs = len(surfs)
      nlayers = len(layers)
      isurf = 1; ilayer = 0
      while isurf < nsurfs:
         
         # Add any non-interface surfaces (surfaces not between layers)
         while surfs[isurf].get("interface", True) == False:
            model += SimpleSurface.from_dict(surfs[isurf])
            isurf += 1

         # Add layer
         model += Layer.from_dict(layers[ilayer])
         ilayer += 1

         # Add surface 
         model += SimpleSurface.from_dict(surfs[isurf])
         isurf += 1

      return model
   
   def __dict__(self) -> Dict:
      base_dict = super().__dict__()
      base_dict.update({
         "_type": self.__class__.__name__,
         "x_limits": self.x_limits,
         **({"y_limits": self.y_limits} if self.y_limits is not None else {}),
         "ordering": self.ordering,
         "surfaces": [surface.__dict__() for surface in self.surfaces]
      })
      return base_dict
   
   def __iadd__(self, other):
      if isinstance(other, SimpleSurface):

         # TODO: check that surfaces are added in monotone order

         # Add surface
         self.surfaces.append(other)
         self.interface_flag = False

         # Build layer-to-surface mapping
         if self._last_added == "layer":
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
         self.interface_flag = True

         # Build layer-to-surface mapping
         if self.ordering == "top_down":
            if len(self.layers) == 1:
               prev_lower = self.upper_surface()
            else:
               prev_lower = self.lower_surface(self.layers[-2])

            self._layer_to_surfs.append(LayerBounds(
               upper = prev_lower,
               layer = other,
            ))
         else:
            if len(self.layers) == 1:
               prev_upper = self.lower_surface()
            else:
               prev_upper = self.upper_surface(self.layers[-2])

            self._layer_to_surfs.append(LayerBounds(
               lower = prev_upper,
               layer = other,
            ))
         self._last_added = "layer"
      else:
         raise ValueError(f"Cannot add {type(other)} to LayeredModel")
      return self
   
   def upper_surface(self, layer: Optional[Union[str, int, ModelSubdomain]] = None) -> SimpleSurface:
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

   def lower_surface(self, layer: Optional[Union[str, int, ModelSubdomain]] = None) -> SimpleSurface:
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
         
   def plot(self, property: str, **kwargs):
      """Plot the model."""
      import matplotlib.pyplot as plt

      if self.dimension == 2:
         x = np.linspace(self.x_limits[0], self.x_limits[1], 1000)
         z = np.linspace(self.z_limits[0], self.z_limits[1], 1000)
         samples = xr.DataArray(dims = ["x", "z"], coords={"x": x, "z": z})
      elif self.dimension == 3:
         raise NotImplementedError("3D plotting not implemented")
      
      if self.ordering == "top_down":
         self._layer_to_surfs[-1].lower = self.lower_surface()
      else:
         self._layer_to_surfs[-1].upper = self.upper_surface()

      if "ax" in kwargs:
         ax = kwargs.pop("ax")
         show = False
      else:
         fig = plt.figure()
         ax = fig.gca()

      units = kwargs.pop("units", "km/s")
      label = kwargs.pop("label", property)
      origin = kwargs.pop("origin", "upper")
      axes_names = kwargs.pop("axes_names",  {"x":"X", "z":"Depth"})
      axes_units = kwargs.pop("axes_units",  {"x":"km", "z":"km"})

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
            if self.ordering == "top_down":
               limits["z_min"] = upper.z_ref
               limits["z_max"] = lower.z_ref
            else:
               limits["z_max"] = upper.z_ref
               limits["z_min"] = lower.z_ref

            if prop.is_constant:
               xgrid = samples.coords["x"]
            else:
               xgrid = data.coords["x"]

            mask = (samples.coords["z"].values > limits["z_min"]) & \
                   (samples.coords["z"].values < limits["z_max"])
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
               dims = ["z", "x"],
               coords = {"x": xgrid, "z": [z0p, z1p]}, 
               data = [z0, z1]
            )
            zvals = tmp.interp(coords={"x": xgrid, "z": zgrid})

            samp = xr.Dataset({
               property: prop.get(zvals),
            })
            samp = samp.assign_coords({
               "Xcoord": (("x", "z"), xgrid.broadcast_like(zvals).values.T),
               "Zcoord": (("x", "z"), zvals.values.T)
            })

            plt.pcolormesh(samp.coords["Xcoord"].values,
                           samp.coords["Zcoord"].values,
                           samp[property].values,
                           shading='gouraud',
                           vmin=vmin,
                           vmax=vmax,
                           **kwargs)
            
            # from scipy.interpolate import griddata
            
            # # Interpolate the curvilinear grid to a uniform grid defined by 'samples'
            # Xcurv = samp.coords["Xcoord"].values.ravel()
            # Zcurv = samp.coords["Zcoord"].values.ravel()
            # data_curv = samp[property].values.ravel()

            # # Create a uniform grid using the 'samples' DataArray coordinates
            # x_uniform = samples.coords["x"].values
            # z_uniform = samples.coords["z"].values
            # Xuniform, Zuniform = np.meshgrid(x_uniform, z_uniform, indexing="ij")

            # interp_data = griddata((Xcurv, Zcurv), data_curv, (Xuniform, Zuniform), method='linear')

            # # Plot the interpolated data on a uniform grid
            # plt.pcolormesh(Xuniform, Zuniform, interp_data, shading='gouraud', **kwargs)

         else:
            data = prop.get()

            limits = {}
            limits["x_min"] = self.x_limits[0]
            limits["x_max"] = self.x_limits[1]
            if self.y_limits is not None:
               limits["y_min"] = self.y_limits[0]
               limits["y_max"] = self.y_limits[1]

            if prop.is_constant:
               xgrid = samples.coords["x"]
            else:
               xgrid = data.coords["x"]

            if self.ordering == "top_down":
               limits["z_min"] = upper.z_phys.get(xgrid)
               limits["z_max"] = lower.z_phys.get(xgrid)
            else:
               limits["z_max"] = upper.z_phys.get(xgrid)
               limits["z_min"] = lower.z_phys.get(xgrid)

            if prop.is_constant:
               data = xr.full_like(samples, data)
               # For constant properties, create a mask based on the sample grid coordinates
               mask = (data.z < limits["z_max"]) & \
                     (data.z > limits["z_min"]) & \
                     (data.x < limits["x_max"]) & \
                     (data.x > limits["x_min"])
               
               da = data.where(mask)
               samples.data = np.where(~np.isnan(da), da, samples.data)
            else:
               # Create mask for the samples grid
               mask = (data.z < limits["z_max"]) & \
                     (data.z > limits["z_min"]) & \
                     (data.x < limits["x_max"]) & \
                     (data.x > limits["x_min"])
               
               # Interpolate data onto samples grid
               da = prop.get()
               da = da.where(mask)
               ds = da.interp(coords=samples.coords)
               samples.data = np.where(~np.isnan(ds), ds, samples.data)
      # except Exception as e:
      #    print(f"Error plotting {property} for layer {layer.name}: {e}")
      
      samples.plot.imshow(ax=ax, x="x",
                          vmin=vmin, 
                          vmax=vmax,
                          **kwargs)

      # Get surface plotting kwargs
      line_color = kwargs.pop("line_color", "k")
      line_style = kwargs.pop("line_style", "-")
      line_width = kwargs.pop("line_width", 1.5)

      for surf in self.surfaces:
         limits = {}
         limits["x"] = self.x_limits
         if self.y_limits is not None:
            limits["y"] = self.y_limits
      
         surf.plot(limits=limits, 
                   ax=ax, 
                   color=line_color,
                   linestyle=line_style,
                   linewidth=line_width,
                   **kwargs)
         
            # Set limits
      ax.set_xlim(self.x_limits)
      if self.y_limits is None:
         zlim = self.z_limits
         zL = zlim[1] - zlim[0]
         zlim[0] = zlim[0] - 0.01 * zL
         zlim[1] = zlim[1] + 0.01 * zL
         ax.set_ylim(zlim)   
      else:
         raise NotImplementedError("2D plotting not implemented")
      
      if origin == "upper":
         ax.invert_yaxis()

      ax.set_aspect("equal")
      
      plt.show()
   
   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path/self.name
      for subdomain in self.subdomains:
         subdomain._set_path(proj_path, self._rel_path)
      for surface in self.surfaces:
         surface._set_path(proj_path, self._rel_path)

   @property
   def _path(self) -> Path:
      return self._proj_path/self._rel_path

# TODO: info functions like this:
#     max_elevation()
#     max_layer_thickness()
#     min_elevation()
#     min_layer_thickness()
#     check_intersection()
