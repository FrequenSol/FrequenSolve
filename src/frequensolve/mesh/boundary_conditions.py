
"""Python structures defining boundary conditions"""

from dataclasses import dataclass, field
from typing      import Optional, List, Literal, Dict

from .mesh import Mesh

@dataclass
class BoundaryCondition:
   """Defines a single boundary condition type (can be applied to multiple boundaries)

   Attributes:
      name (str): BC name
      kind (Literal["dirichlet", "neumann", "pml", "symmetric", "custom"]): BC type
      boundaries (List[str]): List of boundaries where BC should be applied
      pml_wavelengths (float): PML width in wavelengths
      pml_exponent (float): PML complex stretching exponent
      pml_constant (float): PML complex stretching constant

   :warning:
      - When using PML, it is recommended to verify that the PML width and constant are
        sufficient to avoid reflections.
   """
   name:        str
   kind:        Literal["dirichlet", "neumann", "pml", "symmetric", "custom"]
   boundaries:  List[str] = field(default_factory=list)
   
   pml_wavelengths:  float =  2.0
   pml_exponent:     float =  3.0
   pml_constant:     float = 20.0

   def to_dict(self) -> Dict:
      """Converts the boundary condition to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the boundary condition data with keys:
            - name: BC name
            - kind: BC type
            - boundaries: List of boundaries
            - pml_wavelengths: PML width in wavelengths
            - pml_exponent: PML exponent
            - pml_constant: PML constant
      """
      bc_dict = {
         "name": self.name,
         "kind": self.kind,
         "boundaries": self.boundaries,
         **({"pml_wavelengths": self.pml_wavelengths,
            "pml_exponent": self.pml_exponent,
            "pml_constant": self.pml_constant} if self.kind == "pml" else {})
      }
      return bc_dict


@dataclass
class BoundaryConditionManager:
   """Manages boundary conditions.

   Attributes:
      label_type (Literal["geometric", "labeled"]): 'geometric' or 'labeled'
            geometric: Derives boundaries from geometry (e.g. top, bottom, left, right)
            labeled:   Boundaries specified in mesh
      boundary_conditions (List[BoundaryCondition]): List of boundary conditions
   """
   label_type:            Literal["geometric", "labeled"]
   boundary_conditions:   List[BoundaryCondition] = field(default_factory=list)

   def add_BC(self, bc: BoundaryCondition) -> None:
      """Adds a boundary condition to the manager.

      :param bc (BoundaryCondition): The boundary condition to add

      :note:
         - A BC can be applied to multiple boundaries, but each boundary can only be
           assigned one BC.
      """
      # Check for duplicate boundaries
      existing_boundaries = set()
      for existing_bc in self.boundary_conditions:
         existing_boundaries.update(existing_bc.boundaries)
         
      # Check for overlap with new boundaries
      overlap = set(bc.boundaries) & existing_boundaries
      if overlap:
         raise ValueError(f"Boundaries {overlap} already assigned another boundary condition")
         
      self.boundary_conditions.append(bc)

   
   def verify(self, mesh: Mesh) -> None:
      """Verifies that the boundary conditions are valid, and that each boundary has a BC assigned to it"""
      if self.label_type == "geometric":
         if mesh.dimension == 2:
            labels = ["x_min", "x_max", "z_min", "z_max"]
         else:
            labels = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
         for boundary in labels:
            if boundary not in self.boundary_conditions:
               raise ValueError(f"Boundary {boundary} is not assigned a BC")
      else:
         if mesh.boundaries is None:
            raise ValueError("Mesh boundaries are not labeled, and label_type is 'labeled'")
         
         # TODO: check that all boundary elements are assigned a boundary label

         for boundary in mesh.boundaries:
            if boundary not in self.boundary_conditions:
               raise ValueError(f"Boundary {boundary} is not assigned a BC")

   def to_dict(self) -> Dict:
      """Converts the boundary condition manager to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the boundary condition data with keys:
            - label_type: Type of boundary labels
            - boundary_conditions: List of boundary condition dictionaries
      """

      return {
         "label_type": self.label_type,
         "boundary_conditions": [bc.to_dict() for bc in self.boundary_conditions]
      }
