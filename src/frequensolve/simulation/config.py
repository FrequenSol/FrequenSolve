from dataclasses import dataclass
from typing import Literal, Dict

@dataclass
class SimulationConfig:
   """Container for simulation configuration.
   
   Args:
      name (str):       The name of the simulation.
      physics (str):    The physics type for the simulation.
      dimension (int):  The dimension of the simulation (2D or 3D).
      directory (str):  The subdirectory for simulation outputs.
      workflow (str):   The workflow type for the simulation.
      order (int):      The initial order of the mesh.
   """
   name: str
   physics: Literal["acoustic", "elastic", "plasma"]
   dimension: Literal[2, 3]
   directory: str
   workflow: str
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
      """
      return {
         "name": self.name,
         "physics": self.physics,
         "dimension": self.dimension,
         "directory": self.directory,
         "workflow": self.workflow,
         "order": self.order
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
         order=data.get("order", 2)
      )

   def __str__(self) -> str:
      return(
         f"[Simulation]\n"
         f"   Physics: {self.physics}\n"
         f"   Dimension: {self.dimension}D\n" 
         f"   Directory: {self.directory}\n"
         f"   Workflow: {self.workflow}\n"
         f"   Order: {self.order}\n"
         f"[]\n\n"
      )


