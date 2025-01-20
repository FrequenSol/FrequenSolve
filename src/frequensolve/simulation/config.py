from dataclasses import dataclass, field
from typing import Literal, Dict

from .sampling import Sampling

@dataclass
class SimulationConfig:
   """Container for simulation configuration.
   
   Args:
      name (str):       Name of the simulation.
      physics (str):    Physics type for the simulation.
      dimension (int):  Dimension of the simulation (2D or 3D).
      directory (str):  Subdirectory for simulation outputs.
      workflow (str):   Workflow type for the simulation.
      order (int):      Initial order of the mesh.
      tf_domain (str):  Frequency- or time-domain simulation.

      :Note:
         FrequenSolve only operates in the frequency-domain; time-domain
         simulations are produced by uniformly sampling frequencies and then
         performing a Fourier transform.
   """
   name:      str
   physics:   Literal["acoustic", "elastic", "plasma"]
   dimension: Literal[2, 3]
   directory: str
   workflow:  str
   tf_domain: Literal["time","frequency"]
   sampling:  Sampling = field(default_factory=Sampling)

   order: int = 2
   def to_dict(self) -> Dict:
      """Converts the simulation configuration to a dictionary representation.
      
      Returns:
         Dict: Dictionary containing the simulation configuration with keys:
            - name: Simulation name
            - physics: Physics type 
            - dimension: Problem dimension
            - directory: Output directory
            - workflow: Workflow type
            - order: Initial mesh order
            - tf_domain: Frequency- or time-domain simulation
            - sampling: Sampling configuration
      """
      return {
         "name": self.name,
         "physics": self.physics,
         "dimension": self.dimension,
         "directory": self.directory,
         "workflow": self.workflow,
         "order": self.order,
         "tf_domain": self.tf_domain,
         "sampling": self.sampling.to_dict()
      }

   @classmethod
   def from_dict(cls, data: Dict) -> 'SimulationConfig':
      """Creates a Simulation instance from a dictionary.
      
      Args:
         data: Dictionary containing simulation configuration with keys:
            - name:        Simulation name
            - physics:     Physics type
            - dimension:   Problem dimension
            - directory:   Output directory
            - workflow:    Workflow type
            - order:       Initial mesh order (optional)
            
      Returns:
         Simulation: New simulation instance configured from dictionary
      """
      return cls(
         name=data["name"],
         physics=data["physics"],
         dimension=data["dimension"],
         directory=data["directory"],
         workflow=data["workflow"],
         order=data.get("order", 2),
         tf_domain=data.get("tf_domain", "frequency"),
         sampling=Sampling.from_dict(data["sampling"])
      )

   def __str__(self) -> str:
      return(
         f"[Simulation]\n"
         f"   physics: {self.physics}\n"
         f"   dimension: {self.dimension}D\n" 
         f"   directory: {self.directory}\n"
         f"   workflow: {self.workflow}\n"
         f"   order: {self.order}\n"
         f"   tf_domain: {self.tf_domain}\n"
         f"   {str(self.sampling)}\n"
         f"[]\n\n"
      )


