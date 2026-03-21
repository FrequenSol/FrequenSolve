"""Python structures defining boundary conditions"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union

from ..util.printing import print_warn
from .mesh import Mesh

__all__ = ["BoundaryCondition", "BoundaryConditionManager"]


@dataclass
class BoundaryCondition:
    """Defines a single boundary condition type (can be applied to multiple boundaries)

    Attributes:
       name (str): BC name
       kind (Literal["dirichlet", "neumann", "pml", "symmetric"]): BC type
       boundaries (List[str]): List of boundaries where BC should be applied
       pml_wavelengths (float): PML width in wavelengths
       pml_exponent (float): PML complex stretching exponent
       pml_constant (float): PML complex stretching constant

    :warning:
       - When using PML, it is recommended to verify that the PML width and constant are
         sufficient to avoid reflections.
    """

    name: str
    kind: Literal["dirichlet", "neumann", "pml", "symmetric"]
    boundaries: List[str] = field(default_factory=list)

    pml_wavelengths: float = 2.0
    pml_exponent: float = 3.0
    pml_constant: float = 20.0
    stretch_limit: float = 0.25

    @classmethod
    def from_dict(cls, data: Dict) -> "BoundaryCondition":
        return cls(
            name=data["name"],
            kind=data["kind"],
            boundaries=data["boundaries"],
            pml_wavelengths=data.get("pml_wavelengths", 2.0),
            pml_exponent=data.get("pml_exponent", 3.0),
            pml_constant=data.get("pml_constant", 20.0),
            stretch_limit=data.get("stretch_limit", 1.0),
        )

    def __dict__(self) -> Dict:
        bc_dict = {
            "name": self.name,
            "kind": self.kind,
            "boundaries": self.boundaries,
            **(
                {
                    "pml_wavelengths": self.pml_wavelengths,
                    "pml_exponent": self.pml_exponent,
                    "pml_constant": self.pml_constant,
                    "stretch_limit": self.stretch_limit,
                }
                if self.kind == "pml"
                else {}
            ),
        }
        return bc_dict


# TODO: Make class for geometric labels (since right now it accepts multiple values)


@dataclass
class BoundaryConditionManager:
    """Manages boundary conditions.

    Attributes:
       label_type (Literal["geometric", "labeled"]): 'geometric' or 'labeled'
             geometric: Derives boundaries from geometry (e.g. top, bottom, left, right)
             labeled:   Boundaries specified in mesh
       boundary_conditions (List[BoundaryCondition]): List of boundary conditions
    """

    label_type: Literal["geometric", "labeled"] = "geometric"
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    _boundaries: set[Union[str, int]] = field(default_factory=set)

    def add_BC(self, bc: BoundaryCondition) -> None:
        """Adds a boundary condition to the manager.

        Args:
           bc (BoundaryCondition): The boundary condition to add

        Notes:
           - A BC can be applied to multiple boundaries, but each boundary can only be
             assigned one BC.
        """
        overlap = set(bc.boundaries) & self._boundaries
        if overlap:
            for boundary in overlap:
                bc.boundaries.remove(boundary)
                print_warn(
                    f"Boundary {boundary} already assigned a boundary condition; "
                    "duplicate definition has been ignored."
                )
        self._boundaries.update(bc.boundaries)

        self.boundary_conditions.append(bc)

    def __iadd__(self, bc: BoundaryCondition) -> "BoundaryConditionManager":
        """Overrides += operator to invoke add_BC"""
        self.add_BC(bc)
        return self

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
                raise ValueError(
                    "Mesh boundaries are not labeled, and label_type is 'labeled'"
                )

            # TODO: check that all boundary elements are assigned a boundary label

            for boundary in mesh.boundaries:
                if boundary not in self.boundary_conditions:
                    raise ValueError(f"Boundary {boundary} is not assigned a BC")

    @classmethod
    def from_dict(cls, data: Dict) -> "BoundaryConditionManager":
        label_type = data["label_type"]
        boundary_conditions = [
            BoundaryCondition.from_dict(bc_data)
            for bc_data in data["boundary_conditions"]
        ]
        return cls(label_type=label_type, boundary_conditions=boundary_conditions)

    # TODO: change to to_dict
    def __dict__(self) -> Dict:
        return {
            "label_type": self.label_type,
            "boundary_conditions": [bc.__dict__() for bc in self.boundary_conditions],
        }
