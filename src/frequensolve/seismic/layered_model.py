import numpy as np

from dataclasses  import dataclass, field
from typing       import Optional, List, Literal, Union, Dict

from ..geometry.grids import * # noqa
from ..model.model    import * # noqa

__all__ = ['Surface', 'ConstantSurface', 'GridSurface',
            'ConstantLayer','GridLayer','LayeredModel']


# ----------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class Surface:
   """A base class for defining surfaces in a seismic model.

   Attributes:
      name (str):              Name identifier for the surface. Defaults to "surface".
      z_ref (float, optional): Reference z-coordinate. Defaults to None.
      interface (bool):        Whether this is an interface surface. Defaults to True.
   """
   name:         str = "surface"
   z_ref:        Optional[float] = None
   interface:    bool = True
   
   def __str__(self):
      raise NotImplementedError("This class must be overwritten by subclasses.")


@dataclass(kw_only=True)
class ConstantSurface(Surface):
   """A constant surface defined by a single z-coordinate.

   Attributes:
      z_phys (float): The physical z-coordinate of the surface.
   """   
   z_phys:  float

   def evaluate(self, x: List[float]):
      return self.z_phys
      
   def get_limits(self):
      """Get the limits of the surface.

      Returns:
         tuple: A tuple containing the minimum and maximum z-coordinates.
      """
      return self.z_phys, self.z_phys

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
   grid:    CartesianGrid
   file:    str

   def evaluate(self, x: List[float]):
      raise NotImplementedError("TODO: GridSurface.evaluate")
   
   def get_limits(self):
      """Get the limits of the surface.

      Returns:
         tuple: A tuple containing the minimum and maximum z-coordinates.
      """
      data = np.fromfile(self.file, dtype=np.float32)
      return np.min(data), np.max(data)
   
   def __str__(self) -> str:
      """Converts this surface to a formatted string block.

      Returns:
         str: A formatted string block representing the surface.
      """
      out  = f"   [{self.name}]\n"
      out += f"      type   = grid\n"
      out += f"      frame  = {self.frame}\n"
      out += f"      file   = {self.file}\n"
      if self.z_ref:
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
   grid:         CartesianGrid
   file_format:  str

   def to_dict(self) -> Dict:
      """Converts the grid layer to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the layer data with keys from parent class and:
            - type:        The type of the subdomain
            - grid:        The grid configuration
            - file_format: The format of the file containing the grid data
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
      properties (Dict[str, Union[float, str]]): Dictionary of layer properties.
   """
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



# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class LayeredModel(ModelBase):
   """A class for representing a layered seismic model.  

   Attributes:
      dimension (int): The dimension of the model.

      x_limits (List[float]):           X-limits of the model.
      y_limits (Optional[List[float]]): Y-limits of the model. Defaults to None.
      surfaces (List[Surface]):         List of surfaces in the model.
      ordering (Literal["top_down", "bottom_up"]): Ordering of layers. Defaults to "top_down".
      interface_flag (bool):            Whether to flag interfaces between layers. Defaults to True.

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
   _last_added:      str = "none"  # Track last added component

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
                   file:       Optional[str] = None,
                   grid:       Optional[CartesianGrid] = None) -> None:
      """Add a surface to the model."""

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
   
   def to_dict(self) -> Dict:
      """Converts the model to a dictionary representation.
      
      Returns:
         Dict: Returns parent class dict plus:
            - surfaces: List of surface dictionaries
      """
      base_dict = super().to_dict()
      base_dict.update({
         "surfaces": [surface.to_dict() for surface in self.surfaces]
      })
      return base_dict

# TODO: info functions like this:
#     max_elevation()
#     max_layer_thickness()
#     min_elevation()
#     min_layer_thickness()
#     check_intersection()
