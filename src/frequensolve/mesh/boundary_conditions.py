
"""
@file   boundary_conditions.py
@brief  Python structures defining boundary conditions
@date   2025-01-04
"""

from typing      import Optional, List
from pydantic    import BaseModel, Field

class BoundaryCondition(BaseModel):
   """
   @class BoundaryCondition
   @brief Defines a single boundary condition type (can be applied to multiple boundaries)
   @param name       BC name
   @param kind       BC type
   @param boundaries List of boundaries where BC should be applied
   @param auto_adjust  (for type == pml) Adjust PML width for frequency
   @param pml_exponent (for type == pml) PML complex stretching exponent
   @param pml_exponent (for type == pml) PML complex stretching constant
   """
   name:        str
   kind:        Literal["dirichlet", "neumann", "pml", "symmetric", "custom"]
   boundaries:  List[str] = Field(default_factory=list)
   
   auto_adjust:      bool  = True
   pml_wavelengths:  float =  2.0
   pml_exponent:     float =  3.0
   pml_constant:     float = 20.0

   def __str__(self) -> str:
      """
      @brief Converts the BC block to a formatted string.
      """
      lines = []
      lines.append(f"   [{self.name}]")
      lines.append(f"      kind     = {self.kind}")
      lines.append(f"      boundary = {self.boundary}")
      if self.kind == "pml":
         lines.append(f"      pml_wavelengths = {n_wavelengths}")
         lines.append(f"      pml_exponent    = {pml_exponent}")
         lines.append(f"      pml_constant    = {pml_constant}")
      lines.append("   []")
      return "\n".join(lines) + "\n"


# TODO: check that boundaries are uniquely listed
class BoundaryConditionManager(BaseModel):
   """
   @class BoundaryConditionManager
   @brief Manages boundary conditions.
   @param boundary_labels     'geometric' or 'labeled'
            geometric: Derives boundaries from geometry (e.g. top, bottom, left, right)
            labeled:   Boundaries specified in mesh
   @param boundary_conditions List of boundary conditions
   """
   label_type:            Literal["geometric", "labeled"]
   boundary_conditions:   List[BoundaryCondition] = Field(default_factory=list)

   def add_boundary_condition(self, bc: BoundaryCondition) -> None:
      self.boundary_conditions.append(bc)

   def __str__(self) -> str:
      """
      @brief Converts this section to a formatted string.
      """
      if not self.boundary_conditions:
         return ""
      bc_str = "[BCs]\n"
      bc_str += f"   labels = {self.label_type}\n\n"
      for bc in self.boundary_conditions:
         bc_str += str(bc) + "\n"
      bc_str += "[]\n"
      return bc_str
