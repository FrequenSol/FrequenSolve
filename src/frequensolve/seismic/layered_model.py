import numpy as np

from pathlib      import Path
from abc          import ABC, abstractmethod
from dataclasses  import dataclass, field
from typing       import Optional, List, Literal, Union, Dict

from ..geometry.grids    import * # noqa
from ..model.model       import * # noqa
from ..simulation.config import SimulationConfig

__all__ = ['Surface', 'ConstantSurface', 'GridSurface',
            'ConstantLayer','GridLayer','LayeredModel']


# ----------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------
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

   @abstractmethod
   def to_dict(self):
      raise NotImplementedError("This class must be overwritten by subclasses.")
   
   @abstractmethod
   def from_dict(self):
      raise NotImplementedError("This class must be overwritten by subclasses.")
   
   @abstractmethod
   def plot(self, **kwargs):
      raise NotImplementedError("This class must be overwritten by subclasses.")

   @abstractmethod
   def __str__(self):
      raise NotImplementedError("This class must be overwritten by subclasses.")


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
   

   def to_dict(self):
      base_dict = {
         "name": self.name,
         "type": "constant",
         "z_phys": self.z_phys
      }
      if self.z_ref is not None:
         base_dict["z_ref"] = self.z_ref
      if not self.interface:
         base_dict["interface"] = False
      return base_dict
   

   @classmethod
   def from_dict(cls, data: Dict):
      return cls(
         name       = data.get("name", "surface"),
         interface  = data.get("interface", True),
         z_phys     = data["z_phys"],
         z_ref      = data.get("z_ref", data["z_phys"])
      )


   def __str__(self) -> str:
      """Converts this surface to a formatted string block.

      Returns:
         str: A formatted string block representing the surface.
      """
      out  = f"   [{self.name}]\n"
      out += f"      type   = constant\n"
      out += f"      frame  = {self.frame}\n"
      out += f"      z_phys = {self.z_phys}\n"
      if self.z_ref:
         out += f"      z_ref  = {self.z_ref}\n"
      if not self.interface:
         out += f"      interface = false\n"
      out += "   []\n\n"
      return out


@dataclass(kw_only=True)
class GridSurface(Surface):
   grid:        CartesianGrid
   file:        str
   file_format: Literal["binary", "HDF5"] = "binary"
   data:        Optional[np.ndarray] = None


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
      if self.file_format == "binary":
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

   
   def to_dict(self):
      base_dict = super().to_dict()
      base_dict.update({
         "type": "grid",
         "grid": self.grid.to_dict(),
         "file": str(self.file)
      })
      return base_dict
   
   
   def __str__(self) -> str:
      """Converts this surface to a formatted string block.

      Returns:
         str: A formatted string block representing the surface.
      """
      out  = f"   [{self.name}]\n"
      out += f"      type   = grid\n"
      out += f"      frame  = {self.frame}\n"
      out += f"      file   = {self.file}\n"
      if self.z_ref is not None:
         out += f"      z_ref  = {self.z_ref}\n"
      if not self.interface:
         out += f"      interface = false\n"
      out += str(self.grid)
      out += "   []\n\n"
      return out
   
   
@dataclass(kw_only=True)
class GridLayer(ModelSubdomain):
   """A layer sampled on a uniform grid.

   Attributes:
      grid (CartesianGrid): The grid defining the layer.
      file_format (str):    The format of the file containing the grid data.
   """
   grid:          CartesianGrid
   file_format:   Literal["binary", "HDF5"]
   data:          Optional[Dict[str, np.ndarray]] = None


   def read_data(self):
      """Read the data from file."""
      import h5py
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


   def to_dict(self) -> Dict:
      """Converts the grid layer to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the layer data with keys from parent class and:
            - type:        Type of the subdomain
            - grid:        Grid configuration
            - file_format: Format of the file containing the grid data
      """
      base_dict = super().to_dict()
      base_dict.update({
         "type": "grid",
         "grid": self.grid.to_dict(),
         "file_format": self.file_format
      })
      return base_dict


@dataclass(kw_only=True)
class ConstantLayer(ModelSubdomain):
   """A constant-property layer in a model.
   
   Attributes:
      mesh_block_id (int):    Unique identifier for the mesh block.
      name (Optional[str]):   Optional name for the mesh block.
      frame (str):            Coordinate frame ("physical" or "reference").
      properties (Dict[str, float]]): Dictionary of layer properties.
   """

   @classmethod
   def from_dict(cls, data: Dict):
      """Creates a ConstantLayer from a dictionary.
      
      Args:
         data: Dictionary containing layer configuration with keys:
            - name: Layer name
            - mesh_block_id: Mesh block identifier 
            - frame: Coordinate frame
            - properties: Dictionary of property name/value pairs
            
      Returns:
         ConstantLayer: New layer instance configured from dictionary
            
      Raises:
         ValueError: If any property values are not floats
      """
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

   def to_dict(self) -> Dict:
      """Converts the constant layer to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the layer data with keys from parent class
            plus 'type' = 'constant'
      """
      base_dict = super().to_dict()
      base_dict.update({
         "type": "constant"
      })
      return base_dict





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



@dataclass(kw_only=True)
class LayeredModel(ModelBase):
   """A class for representing a layered seismic model.  

   Attributes:
      name (str):                         The name of the model.
      dimension (int):                    Dimension of the model.
      subdomains (List[ModelSubdomain]):  Model layers.
      x_limits (List[float]):             X-limits of the model.
      y_limits (Optional[List[float]]):   Y-limits of the model. Defaults to None.
      surfaces (List[Surface]):           Model (simple) surfaces.
      ordering (Literal["top_down", "bottom_up"]): Ordering of layers. Defaults to "top_down".
      interface_flag (bool):              Whether to flag interfaces between layers. Defaults to True.

      Note: 
         A 'surface' must be added at the top and bottom of the model, and at 
         least one surface must be added between 'layers'.
   """
   x_limits:         List[float]
   y_limits:         Optional[List[float]] = None
   surfaces:         List[Surface] = field(default_factory=list)
   ordering:         Literal["top_down","bottom_up"] = "top_down"
   interface_flag:   bool = True
   _layer_to_surfs:  List[LayerBounds] = field(default_factory=list)
   _last_added:      str = "none"

   def add_layer(self,
                 type:              str = "constant",
                 name:              Optional[str] = None,
                 mesh_block_id:     int = -1,
                 frame:             str = "physical",
                 properties:        Optional[dict] = None,
                 grid:              Optional[CartesianGrid] = None) -> None:
      """Add a layer to the model.

      Args:
         type (str):           Type of the layer.
         name (str):           Name of the layer.
         mesh_block_id (int):  Mesh block id of the layer.
         frame (str):          Frame of the layer.
         properties (dict):    Properties of the layer.
         grid (CartesianGrid): Grid of the layer (for grid layers only).
      
      Raises:
         ValueError: If no surfaces have been added before adding a layer
         ValueError: If trying to add a layer without adding a surface first
         ValueError: If trying to add consecutive layers without surfaces between them
      """
      if len(self.surfaces) == 0:
         raise ValueError("Must add at least one surface before adding layers")
         
      if self._last_added == "layer":
         raise ValueError("Must add a surface between consecutive layers")

      if type == "constant":
         layer = ConstantLayer(
                     name       = name,
                     mesh_block_id = mesh_block_id,
                     frame       = frame,
                     properties = properties
                  )
      elif type == "grid":
         layer = GridLayer(
                     name   = name,
                     mesh_block_id = mesh_block_id,
                     frame  = frame,
                     grid   = grid,
                     properties = properties
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
                   type:       str = "constant",
                   name:       str = "surface",
                   z:          Optional[float] = None,
                   z_ref:      Optional[float] = None,
                   file:       Optional[Union[str, Path]] = None,
                   grid:       Optional[CartesianGrid] = None) -> None:
      """Add a surface to the model.
      
      Args:
         type (str):           Type of the surface.
         name (str):           Name of the surface.
         z (float):            Physical z-coordinate of the surface.
         z_ref (float):        Reference z-coordinate of the surface.
         file (str):           File containing the surface data.
         grid (CartesianGrid): Grid of the surface (for grid surfaces only).
      """

      # TODO: check that surfaces are added in monotone order

      if type == "constant":
         if z_ref is None:
            z_ref = z
         surface = ConstantSurface(
                        name       = name,
                        interface  = self.interface_flag,
                        z_phys     = z,
                        z_ref      = z_ref
                     )
      elif type == "grid":
         surface = GridSurface(
                        name       = name,
                        interface  = self.interface_flag,
                        grid       = grid,
                        z_ref      = z_ref,
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
   def from_dict(cls, sim: SimulationConfig, data: Dict) -> "LayeredModel":
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
         type = layer_dict["type"]
         if type == "constant":
            layer = ConstantLayer.from_dict(layer_dict)
         elif type == "grid":
            layer = GridLayer.from_dict(layer_dict)
         else:
            raise ValueError(f"Unknown layer type: {type}")
         model.add_subdomain(layer)
      
      # Add surfaces
      for surface_dict in surface_dicts:
         type = surface_dict["type"]
         if type == "constant":
            surface = ConstantSurface.from_dict(surface_dict)
         elif type == "grid":
            surface = GridSurface.from_dict(surface_dict)
         else:
            raise ValueError(f"Unknown surface type: {type}")
         model.surfaces.append(surface)
      return model
   
   def to_dict(self) -> Dict:
      """Converts the model to a dictionary representation.
      
      Returns:
         Dict: Returns parent class dict plus:
            - surfaces: List of surface dictionaries
      """
      base_dict = super().to_dict()
      base_dict.update({
         "x_limits": self.x_limits,
         **({"y_limits": self.y_limits} if self.y_limits is not None else {}),
         "ordering": self.ordering,
         "surfaces": [surface.to_dict() for surface in self.surfaces]
      })
      return base_dict

# TODO: info functions like this:
#     max_elevation()
#     max_layer_thickness()
#     min_elevation()
#     min_layer_thickness()
#     check_intersection()
