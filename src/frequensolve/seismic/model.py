
import numpy as np

from dataclasses  import dataclass, field
from typing       import Optional, List, Literal

__all__ = ['Surface', 'ConstantSurface', 'GridSurface',
            'ModelLayer','ConstantLayer','GridLayer',
            'LayeredModel']


# ----------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class Surface:
   """
   @class Surface
   @brief Defines a simple surface
   """
   name:         str = "surface"
   z_ref:        Optional[float] = None
   interface:    bool = True
   
   def __str__(self):
      raise NotImplementedError("This class must be overwritten by subclasses.")


@dataclass(kw_only=True)
class ConstantSurface(Surface):
   z_phys:  float

   def evaluate(self, x: List[float]):
      return self.z_phys
      
   def get_limits(self):
      return self.z_phys, self.z_phys

   def __str__(self) -> str:
      """
      @brief Converts this surface to a formatted string block.
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
      data = np.fromfile(self.file, dtype=np.float32)
      return np.min(data), np.max(data)
   
   def __str__(self) -> str:
      """
      @brief Converts this surface to a formatted string block.
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


# ----------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class ModelLayer:
   """
   @class ModelLayer
   @brief Represents one material layer in the Model section.
   """
   domain:       int
   name:         str = "layer"
   frame:        str = "physical"
   properties:   dict[str, Union[float, str]] = field(default_factory=dict)
   
   def __str__(self):
      raise NotImplementedError("This class must be overwritten by subclasses.")
   
   
@dataclass(kw_only=True)
class GridLayer(ModelLayer):
   grid:         CartesianGrid
   file_format:  str
   
   def __str__(self) -> str:
      """
      @brief Converts this layer to a formatted string block.
      """
      out  = f"   [{self.name}]\n"
      out += f"      type   = constant\n"
      out += f"      frame  = {self.frame}\n"
      out += f"      domain = {self.domain}\n"
      out += str(self.grid)
      for k, v in self.properties.items():
         out += f"      [{k}]\n"
         if type(v) is str:
            out += f"         file   = {v}\n"
         else:
            out += f"         value  = {v}\n"
         out += f"      []\n"
         out += f"   []\n"
      return out
   
   
@dataclass(kw_only=True)
class ConstantLayer(ModelLayer):
   def add_property(self, key: str, value: Union[float, str]) -> None:
      """
      @brief Add or update a material property.
      """
      self.properties[key] = value

   def __str__(self) -> str:
      """
      @brief Converts this layer to a formatted string block.
      """
      props_str = "\n".join(
         f"         {k} = {v}" for k, v in self.properties.items()
      )
      return (
         f"   [{self.name}]\n"
         f"      type   = constant\n"
         f"      frame  = {self.frame}\n"
         f"      domain = {self.domain}\n"
         f"      [Properties]\n"
         f"{props_str}\n"
         f"      []\n"
         f"   []\n"
      )



# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class LayeredModel:
   """
   @class Model
   @brief Represents the Model section containing one or more material layers.
   """
   x_limits:         List[float]
   y_limits:         Optional[List[float]] = None
   surfaces:         List[Surface]    = field(default_factory=list)
   layers:           List[ModelLayer] = field(default_factory=list)
   ordering:         Literal["top_down","bottom_up"] = "top_down"
   interface_flag:   bool = True

# TODO: check that surface defined between all layers
   def add_layer(self,
                 type:       str = "constant",
                 name:       str = "layer",
                 domain:     int = -1,
                 frame:      str = "physical",
                 properties: Optional[dict] = None,
                 grid:       Optional[CartesianGrid] = None) -> None:
      """
      @brief Add a layer to the model.
      """
      if type == "constant":
         layer = ConstantLayer(
                     name       = name,
                     domain     = domain,
                     frame      = frame,
                     properties = properties
                  )
      elif type == "grid":
         layer = GridLayer(
                     name   = name,
                     domain = domain,
                     frame  = frame,
                     grid   = grid,
                     properties = properties
                  )
      else:
         raise ValueError(f"Layer 'type' should be 'constant' or 'grid', provided: {type} ")
      self.layers.append(layer)
      self.interface_flag = True


   def add_surface(self,
                   type:       str = "constant",
                   name:       str = "surface",
                   z:          Optional[float] = None,
                   z_ref:      Optional[float] = None,
                   file:       Optional[str] = None,
                   grid:       Optional[CartesianGrid] = None) -> None:
      """
      @brief Add a surface to the model.
      """
      if type == "constant":
         if z_ref is None:
            z_ref = z
         surface = ConstantSurface(
                        name       = name,
                        interface  = self.interface_flag,
                        z_phys     = z
                        z_ref      = z_ref
                     )
      elif type == "grid":
         surface = GridSurface(
                        name       = name,
                        interface  = self.interface_flag,
                        grid       = grid,
                        z_ref      = z_ref
                        file       = file
                     )
   
   @property
   def z_limits(self):
      z0,_ = self.surfaces[0].get_limits()
      _,z1 = self.surfaces[-1].get_limits()
      return [z0, z1]
                   
   def add_constant_surfaces(self, z: list[float] = None) -> None:
      """
      @brief Add constant surfaces located at listed depths
      """
      for z in z_list:
         surface = ConstantSurface(
                        z = z,
                        interface = self.interface_flag,
                     )
         self.interface_flag = False
      

   def __str__(self) -> str:
      """
      @brief Converts Model to formatted string
      """
      out  = f"[Model]\n"
      for layer in layers:
         out += str(layer)
      out += f"[]\n\n"
      out += f"[Topography]\n"
      if self.surfaces:
         for surface in self.surfaces:
            out += str(surface)
      out += f"[]\n\n"
      return out

# TODO: query functions like this:
#model.surfaces()                     # returns surfaces
#model.upper_surface()                # returns top surface
#model.lower_surface()                # returns lower surface
#model.upper_surface(layer = "sand")  # returns upper surface of layer

# TODO: info functions like this:
#     max_elevation()
#     max_layer_thickness()
#     min_elevation()
#     min_layer_thickness()
#     check_intersection()
