import numpy as np
import xarray as xr

from pathlib      import Path
from abc          import ABC, abstractmethod
from dataclasses  import dataclass, field
from typing       import Optional, List, Literal, Union, Dict

from ..geometry.grids      import * # noqa
from ..model.model         import * # noqa
from ..util.class_registry import *  # noqa

__all__ = ['Surface', 'ConstantSurface', 'GridSurface',
            'ConstantLayer','GridLayer','LayeredModel']


# ----------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------
@register_class
@dataclass(kw_only=True)
class Surface(ABC):
   """A base class for defining surfaces in a seismic model.

   Attributes:
      name (str):              Name identifier for the surface. Defaults to "surface".
      z_ref (float, optional): Reference z-coordinate. Defaults to None.
      interface (bool):        Whether this is an interface surface. Defaults to True.
   """
   name:         str = "surface"
   z_ref:        Optional[float] = None
   interface:    bool = True
   _proj_path:   Optional[Path] = None
   _rel_path:    Optional[Path] = None

   def __dict__(self):
      base_dict = { "name": self.name }
      if self.z_ref is not None:
         base_dict["z_ref"] = self.z_ref
      if not self.interface:
         base_dict["interface"] = False
      return base_dict

   @classmethod
   def from_dict(cls, data: Dict) -> "Surface":
      class_name = data["_type"]
      if class_name in class_registry:
         surface_class = class_registry[class_name]
         return surface_class.from_dict(data)
      else:
         raise ValueError(f"Unknown surface class: {class_name}")
      
   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path

   @property
   def _path(self) -> Path:
      return self._proj_path / self._rel_path

@register_class
@dataclass(kw_only=True)
class ConstantSurface(Surface):
   """A constant surface defined by a single z-coordinate.

   Attributes:
      z_phys (float): The physical z-coordinate of the surface.
   """   
   z_phys:   float

   def evaluate(self, x: List[float]):
      return self.z_phys
      
   def get_limits(self):
      """Get the limits of the surface.

      Returns:
         tuple: A tuple containing the minimum and maximum z-coordinates.
      """
      return self.z_phys, self.z_phys
   
   def plot(self, limits: Optional[List[float]] = None, **kwargs):
      """Plot the surface."""
      import matplotlib.pyplot as plt

      dim = self.grid.dimension
      if dim == 1:
         if limits is None:
            ax = kwargs.get("ax", plt.gca())
            limits = ax.get_xlim()
         coords = np.array([x for x in np.linspace(limits[0], limits[1], 100)])
         plt.plot(coords[:,0], self.data)
      elif dim == 2:
         if limits is None:
            ax = kwargs.get("ax", plt.gca())
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
         else:
            limits = np.array(limits).flatten()
            xlim = limits[0:2]
            ylim = limits[2:4]
         coords = np.array([[x, y] for x in np.linspace(xlim[0], xlim[1], 100) 
                                    for y in np.linspace(ylim[0], ylim[1], 100)])
         z = np.full(coords.shape[0], self.z_phys)
         plt.plot_surface(coords[:,0], coords[:,1], z)
   
   def __dict__(self):
      return {
         "_type": self.__class__.__name__,
         "name": self.name,
         "z_phys": self.z_phys,
         **({"z_ref": self.z_ref} if self.z_ref is not None else {}),
         **({"interface": self.interface} if not self.interface else {})
      }
   
   @classmethod
   def from_dict(cls, data: Dict):
      return cls(
         name       = data.get("name", "surface"),
         interface  = data.get("interface", True),
         z_phys     = data["z_phys"],
         z_ref      = data.get("z_ref", data["z_phys"])
      )

# TODO: method to perturb data
# TODO: make flag to note if data needs to be written/rewritten
@register_class
@dataclass(kw_only=True)
class GridSurface(Surface):
   grid:        CartesianGrid
   file:        Optional[Union[str, Path]]       = None
   file_format: Optional[Literal["raw", "HDF5"]] = None
   data:        Optional[np.ndarray]             = None

   def evaluate(self, x: List[float]):
      raise NotImplementedError("TODO: GridSurface.evaluate")

   def get_limits(self):
      """Get the limits of the surface.

      Returns:
         tuple: A tuple containing the minimum and maximum z-coordinates.
      """
      if self.data is None:
         self.data = self.read_data()
      return np.min(self.data), np.max(self.data)
   
   def read_data(self):
      """Read the data from file."""

      if self.file_format is None:
         if self.file.endswith(".h5"):
            self.file_format = "HDF5"
         elif self.file.endswith(".bin"):
            self.file_format = "raw"
         else:
            raise ValueError(f"Unknown file format for {self.file}")
         
      if self.file_format == "raw":
         self.data = np.fromfile(self.file, dtype=np.float32).reshape(self.grid.n)
      elif self.file_format == "HDF5":
         import h5py
         file, dset = self.file.split(":")
         self.data = h5py.File(file, "r")[dset][()]

      return self.data
   
   def plot(self, **kwargs):
      """Plot the surface."""
      import matplotlib.pyplot as plt

      dim = self.grid.dimension
      if dim == 1:
         coords = self.grid.get_coords()
         self.read_data()
         plt.plot(coords[:,0], self.data)
      elif dim == 2:
         coords = self.grid.get_coords()
         x = coords[:,0].reshape(self.grid.n)
         y = coords[:,1].reshape(self.grid.n)
         if self.data is None:
            data = self.read_data()
         
         if "ax" in kwargs:
            ax = kwargs["ax"]
         else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
         surf = ax.plot_surface(x, y, self.data, **kwargs)
         fig.colorbar(surf)
         ax.set_xlabel('X')
         ax.set_ylabel('Y') 
         # ax.set_zlabel('Z')

      if "ax" not in kwargs:
         plt.show()

   @classmethod
   def from_dict(cls, data: Dict):
      return cls(
         name       = data["name"],
         interface  = data["interface"],
         grid       = CartesianGrid.from_dict(data["grid"]),
         file       = data["file"]
      )
   
   def dump_to_file(self):
      """Dump the surface data to file."""

      # Write data to file
      format = self.file_format if self.file_format is not None else "raw"
      if format == "raw":

         # Get file name, make sure directory exists
         if self.file is None:
            self.file = self._path / (self.name + ".bin")
         if not self.file.parent.exists():
            self.file.parent.mkdir(parents=True)

         # write data to file
         self.data.astype(np.float32).tofile(self.file)
      elif format == "HDF5":
         import h5py

         # Get file name, make sure directory exists
         if self.file is None:
            self.file = self._path / ".h5:" + self.name
         if not self.file.parent.exists():
            self.file.parent.mkdir(parents=True)

         file, dset = self.file.split(":")
         with h5py.File(file, "wa") as f:
            f.create_dataset(dset, data=self.data.astype(np.float32))

   def __dict__(self):
      # TODO: check if data needs to be written/rewritten first
      self.dump_to_file()

      return {
         "_type":       self.__class__.__name__,
         "name":        self.name,
         "interface":   self.interface,
         "grid":        self.grid.__dict__(),
         "file":        self.file,
      }
   
   
# TODO: enable xr.Dataset properties
@register_class
@dataclass(kw_only=True)
class GridLayer(ModelSubdomain):
   """A layer sampled on a uniform grid.

   Attributes:
      grid (CartesianGrid): The grid defining the layer.
      file_format (str):    The format of the file containing the grid data.
   """
   grid:          CartesianGrid
   file_format:   Optional[Literal["raw", "HDF5"]] = None
   data:          Optional[Dict[str, Union[np.ndarray, xr.DataArray]]] = None

   def read_data(self):
      """Read the data from file."""
      import h5py

      if self.file_format is None:
         if self.file.endswith(".h5"):
            self.file_format = "HDF5"
         elif self.file.endswith(".bin"):
            self.file_format = "raw"
         else:
            raise ValueError(f"Unknown file format for {self.file}")
         
      if self.data is None:
         self.data = {}
      for prop, val in self.properties.items():
         if isinstance(val, str) or isinstance(val, Path):
            file, dset = val.split(":")
            self.data[prop] = h5py.File(file, "r")[dset][()]
      return self.data

   @classmethod
   def from_dict(cls, data: Dict):
      return cls(
         name          = data["name"],
         mesh_block_id = data["mesh_block_id"],
         frame         = data["frame"],
         properties    = data["properties"],
         grid          = CartesianGrid.from_dict(data["grid"]),
         file_format   = data["file_format"]
      )

   def __dict__(self) -> Dict:

      # If layer defined by array, write to file before writing.
      if isinstance(self.properties, xr.Dataset):
         raise NotImplementedError("GridLayer with xr.Dataset properties not yet supported")
      elif isinstance(self.properties, dict):
         for prop, val in self.properties.items():
            if isinstance(val, np.ndarray):
               self.file_format = "raw"

               file = self._path / (self.name + "__" + prop + ".bin")
               if not file.parent.exists():
                  file.parent.mkdir(parents=True)

               val.astype(np.float32).tofile(file)

               # Update property with file path
               self.properties[prop] = file
            elif isinstance(val, xr.DataArray):
               import h5py
               self.file_format = "HDF5"

               file, dset = self._path / ".h5", self.name + "__" + prop
               if not file.parent.exists():
                  file.parent.mkdir(parents=True)

               with h5py.File(file, "wa") as f:
                  f.create_dataset(dset, data=val.values.astype(np.float32))

               # Update property with file path
               self.properties[prop] = file
            else:
               raise ValueError(f"Invalid property type: {type(val)}")
      else:
         raise ValueError(f"Invalid property type: {type(self.properties)}")

      return {
         "_type":          self.__class__.__name__,
         "name":           self.name,
         "mesh_block_id":  self.mesh_block_id,
         "frame":          self.frame,
         "properties":     self.properties,
         "grid":           self.grid.__dict__(),
         "file_format":    self.file_format,
      }


@register_class
@dataclass(kw_only=True)
class ConstantLayer(ModelSubdomain):

   @classmethod
   def from_dict(cls, data: Dict):
      properties = data["properties"]
      
      # Verify all property values are floats
      for name, value in properties.items():
         if not isinstance(value, (int, float)):
            raise ValueError(f"Property '{name}' has non-numeric value: {value}")
            
      return cls(
         name          = data["name"],
         mesh_block_id = data["mesh_block_id"],
         frame         = data["frame"],
         properties    = properties
      )

   def __dict__(self) -> Dict:
      return {
         "_type": self.__class__.__name__,
         "name": self.name,
         "mesh_block_id": self.mesh_block_id,
         "frame": self.frame,
         "properties": self.properties,
      }




# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
# Helper class for LayeredModel
@dataclass(kw_only=True)
class LayerBounds:
   """Defines the bounding surfaces of a model layer.

   Attributes:
      upper (Surface): The upper bounding surface.
      lower (Surface): The lower bounding surface.
      layer_id (int):  Unique identifier for the layer.
   """
   upper:      Optional[Surface] = None
   lower:      Optional[Surface] = None
   layer:      ModelSubdomain


@register_class
@dataclass(kw_only=True)
class LayeredModel(ModelBase):
   """A class for representing a layered seismic model.  

   Attributes:
      name (str):                         The name of the model.
      dimension (int):                    Dimension of the model.
      subdomains (List[ModelSubdomain]):  Model layers.
      x_limits (List[float]):             X-limits of the model.
      y_limits (Optional[List[float]]):   Y-limits of the model. (3D only)
      surfaces (List[Surface]):           Model (simple) surfaces.
      interface_flag (bool):              Whether to flag interfaces between layers.
      ordering (Literal["top_down", "bottom_up"]): 
         Ordering of layers. Defaults to "top_down".

      Note: 
         A 'surface' must be added at the top and bottom of the model, and at 
         least one surface must be added between 'layers'.
   """
   x_limits:         List[float]
   y_limits:         Optional[List[float]]           = None
   surfaces:         List[Surface]                   = field(default_factory=list)
   ordering:         Literal["top_down","bottom_up"] = "top_down"
   interface_flag:   bool                            = True
   _layer_to_surfs:  List[LayerBounds]               = field(default_factory=list)
   _last_added:      str                             = "none"
   _proj_path:       Optional[Path] = None
   _rel_path:        Optional[Path] = None

   def add_layer(self,
                 name:              Optional[str] = None,
                 mesh_block_id:     int = -1,
                 frame:             str = "physical",
                 properties:        Optional[dict] = None,
                 grid:              Optional[CartesianGrid] = None) -> None:
      """Add a layer to the model.

      Raises:
         ValueError: If no surfaces have been added before adding a layer
         ValueError: If trying to add a layer without adding a surface first
         ValueError: If trying to add consecutive layers without surfaces between them
      """
      if len(self.surfaces) == 0:
         raise ValueError("Must add at least one surface before adding layers")
         
      if self._last_added == "layer":
         raise ValueError("Must add a surface between consecutive layers")

      if grid is None:
         for prop, val in properties.items():
            if isinstance(val, str) or isinstance(val, Path):
               raise ValueError("Must provide grid for grid layers")
         layer = ConstantLayer(
                     name          = name,
                     mesh_block_id = mesh_block_id,
                     frame         = frame,
                     properties    = properties
                  )
      elif isinstance(grid, CartesianGrid):
         layer = GridLayer(
                     name          = name,
                     mesh_block_id = mesh_block_id,
                     frame         = frame,
                     grid          = grid,
                     properties    = properties
                  )
      else:
         raise ValueError(f"Layer 'type' should be 'constant' or 'grid', provided: {type} ")
      self.add_subdomain(layer)
      self.interface_flag = True

      if self.ordering == "top_down":
         self._layer_to_surfs.append(LayerBounds(
            upper = self.surfaces[-1],
            layer = layer,
         ))
      else:
         self._layer_to_surfs.append(LayerBounds(
            lower = self.surfaces[-1],
            layer = layer,
         ))
      self._last_added = "layer"


   def add_surface(self,
                   name:       str = "surface",
                   z:          Optional[float] = None,
                   z_ref:      Optional[float] = None,
                   file:       Optional[Union[str, Path]] = None,
                   grid:       Optional[CartesianGrid] = None) -> None:
      """Add a surface to the model.
      
      Args:
         name (str):           Name of the surface.
         z (float):            Physical z-coordinate of the surface.
         z_ref (float):        Reference z-coordinate of the surface.
         file (str):           File containing the surface data.
         grid (CartesianGrid): Grid of the surface (for grid surfaces only).
      """

      # TODO: check that surfaces are added in monotone order

      if grid is None:
         if z is None:
            raise ValueError("Must provide z for 'constant' surfaces")
         if z_ref is None:
            z_ref = z
         surface = ConstantSurface(
                        name       = name,
                        interface  = self.interface_flag,
                        z_ref      = z_ref,
                        z_phys     = z,
                     )
      elif isinstance(grid, CartesianGrid):
         if file is None:
            raise ValueError("Must provide file for 'gridded' surfaces")
         surface = GridSurface(
                        name       = name,
                        interface  = self.interface_flag,
                        z_ref      = z_ref,
                        grid       = grid,
                        file       = file
                     )
      self.surfaces.append(surface)
      self.interface_flag = False

      if self._last_added == "layer":
         if self.ordering == "top_down":  
            self._layer_to_surfs.append(LayerBounds(
               lower = surface,
               layer = self.subdomains[-1],
            ))
         else:
            self._layer_to_surfs.append(LayerBounds(
               upper = surface,
               layer = self.subdomains[-1],
            ))

      self._last_added = "surface"
      
   
   @property
   def z_limits(self):
      z0,_ = self.surfaces[0].get_limits()
      _,z1 = self.surfaces[-1].get_limits()
      return [z0, z1]
                   

   def add_constant_surfaces(self, z: Union[float, list[float]] = None) -> None:
      """Add constant surfaces located at listed depths."""
      if isinstance(z, float):
         z = [z]
      for z_phys in z:
         surface = ConstantSurface(
                        z = z_phys,
                        interface = self.interface_flag,
                     )
         self.surfaces.append(surface)
         self.interface_flag = False
         self._last_added = "surface"
   

   def upper_surface(self, layer: Optional[Union[str, int, ModelSubdomain]] = None) -> Surface:
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

   def lower_surface(self, layer: Optional[Union[str, int, ModelSubdomain]] = None) -> Surface:
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
      surface_dicts = data_copy.pop("surfaces")
      layer_dicts = data_copy.pop("subdomains")

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
      
      # Add layers
      for layer_dict in layer_dicts:
         class_name = layer_dict["_type"]
         if class_name in class_registry:
            model_class = class_registry[class_name]
            layer = model_class.from_dict(layer_dict)
         else:
            raise ValueError(f"Unknown model class: {class_name}")
         model.add_subdomain(layer)
      
      # Add surfaces
      for surface_dict in surface_dicts:
         class_name = surface_dict["_type"]
         if class_name in class_registry:
            model_class = class_registry[class_name]
            surface = model_class.from_dict(surface_dict)
         else:
            raise ValueError(f"Unknown surface class: {class_name}")
         model.surfaces.append(surface)
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
