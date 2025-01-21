from dataclasses import dataclass, field
from typing      import Literal, Dict, Union
from pathlib     import Path

from .sampling   import Sampling

__all__ = ['SimulationConfig']

@dataclass(kw_only=True)
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
   directory: Union[str, Path]
   mode:      Literal["forward", "adjoint", "combined", "gradient"]
   tf_domain: Literal["time","frequency"]
   sampling:  Sampling = field(default_factory=Sampling)
   
   def to_dict(self) -> Dict:
      return {
         "name": self.name,
         "physics": self.physics,
         "dimension": self.dimension,
         "directory": self.directory,
         "mode": self.mode,
         "tf_domain": self.tf_domain,
         "sampling": self.sampling.to_dict()
      }

   @classmethod
   def from_dict(cls, data: Dict) -> 'SimulationConfig':
      return cls(
         name=data["name"],
         physics=data["physics"],
         dimension=data["dimension"],
         directory=data["directory"],
         mode=data["mode"],
         tf_domain=data.get("tf_domain", "frequency"),
         sampling=Sampling.from_dict(data["sampling"])
      )

   def __str__(self) -> str:
      return(
         f"[Simulation]\n"
         f"   physics:   {self.physics}\n"
         f"   dimension: {self.dimension}D\n" 
         f"   directory: {self.directory}\n"
         f"   mode:      {self.mode}\n"
         f"   tf_domain: {self.tf_domain}\n"
         f"   {str(self.sampling)}\n"
         f"[]\n\n"
      )


