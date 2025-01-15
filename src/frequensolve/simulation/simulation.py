import numpy as np

from dataclasses import dataclass
from typing import Optional, Literal, Union

from .config                     import *  # noqa
from .output_manager             import *  # noqa
from .numeric_manager            import *  # noqa
from ..model.model               import *  # noqa
from ..mesh.mesh_manager         import *  # noqa
from ..seismic.acquisition       import *  # noqa
from ..mesh.boundary_conditions  import *  # noqa

__all__ = ['Simulation']

@dataclass 
class Simulation(SimulationConfig):
   """Container for simulation configuration.
   
   Attributes:
      model (ModelBase):               Model configuration.
      mesh (MeshManager):              Mesh configuration.
      bcs (BoundaryConditionManager):  Boundary condition configuration.
      output (OutputManager):          Output configuration.
      acquisition (Acquisition):       Acquisition configuration.
      numerics (NumericsManager):      Numerics configuration.
   """
   model: ModelBase              = field(default_factory=ModelBase)
   mesh: MeshManager             = field(default_factory=MeshManager)
   bcs: BoundaryConditionManager = field(default_factory=BoundaryConditionManager)
   output: OutputManager         = field(default_factory=OutputManager)
   acquisition: Acquisition      = field(default_factory=Acquisition)
   numerics: NumericsManager     = field(default_factory=NumericsManager)

   def to_dict(self) -> Dict:
      """Converts the simulation to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the simulation configuration with keys:
            - name: Simulation name
            - physics: Physics type
            - dimension: Problem dimension
            - directory: Output directory
            - workflow: Workflow type
            - order: Initial mesh order
            - model: Model configuration
            - mesh: Mesh configuration
            - bcs: Boundary condition configuration
            - output: Output configuration
            - acquisition: Acquisition configuration
            - numerics: Numerics configuration
      """
      return {
         "name":        self.name,
         "physics":     self.physics,
         "dimension":   self.dimension,
         "directory":   self.directory,
         "workflow":    self.workflow,
         "order":       self.order,
         "model":       self.model.to_dict(),
         "mesh":        self.mesh.to_dict(),
         "bcs":         self.bcs.to_dict(),
         "output":      self.output.to_dict(),
         "acquisition": self.acquisition.to_dict(),
         "numerics":    self.numerics.to_dict()
      }


   @classmethod
   def from_dict(cls, data: Dict) -> 'Simulation':
      """Creates a Simulation instance from a dictionary.
      
      Args:
         data: Dictionary containing simulation configuration with keys:
            - name:        Simulation name
            - physics:     Physics type
            - dimension:   Problem dimension
            - directory:   Output directory
            - workflow:    Workflow type
            - order:       Initial mesh order (optional)
            - model:       Model configuration
            - mesh:        Mesh configuration
            - bcs:         Boundary condition configuration
            - output:      Output configuration
            - acquisition: Acquisition configuration
            - numerics:    Numerics configuration
            
      Returns:
         Simulation: New simulation instance configured from dictionary
      """
      config = super().from_dict(data)
      sim = cls(
         name=config.name,
         physics=config.physics,
         dimension=config.dimension,
         directory=config.directory,
         workflow=config.workflow,
         order=config.order
      )

      sim.model = ModelBase.from_dict(data["model"])
      sim.mesh = MeshManager.from_dict(sim, data["mesh"])
      sim.bcs = BoundaryConditionManager.from_dict(data["bcs"])
      sim.output = OutputManager.from_dict(data["output"])
      sim.acquisition = Acquisition.from_dict(data["acquisition"])
      sim.numerics = NumericsManager.from_dict(data["numerics"])

      return sim
   

   def __str__(self) -> str:
      """Returns a string representation of the simulation.
      
      Returns:
         str: String describing the simulation configuration
      """
      return (
         f"{super().__str__()}\n"
         f"{str(self.model)}\n"
         f"{str(self.mesh)}\n"
         f"{str(self.bcs)}\n"
         f"{str(self.numerics)}\n"
         f"{str(self.acquisition)}\n"
         f"{str(self.output)}"
      )
