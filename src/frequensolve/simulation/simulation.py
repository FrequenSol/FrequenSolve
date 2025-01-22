import json
import numpy as np
from json import JSONEncoder

from pathlib     import Path
from dataclasses import dataclass
from typing      import Optional, Union, Dict

from .config                     import *  # noqa
from .output_manager             import *  # noqa
from .numerics_manager           import *  # noqa
from ..model.model               import *  # noqa
from ..mesh.mesh_manager         import *  # noqa
from ..seismic.acquisition       import *  # noqa
from ..mesh.boundary_conditions  import *  # noqa

__all__ = ['Simulation', 'CustomJSONEncoder']

class CustomJSONEncoder(JSONEncoder):
   """Custom JSON encoder for Simulation objects."""
   def default(self, obj):
      if isinstance(obj, np.integer):
         return int(obj)
      if isinstance(obj, np.floating):
         return float(obj)
      if isinstance(obj, np.ndarray):
         return obj.tolist()
      if isinstance(obj, (np.bool_)):
         return bool(obj)
      return super().default(obj)
   

@dataclass(kw_only=True)
class Simulation(SimulationConfig):
   """Container for simulation configuration.
   
   Attributes:
      model (ModelBase):               Model configuration.
      mesh (MeshManager):              Mesh configuration.
      BCs (BoundaryConditionManager):  Boundary condition configuration.
      output (OutputManager):          Output configuration.
      acquisition (Acquisition):       Acquisition configuration.
      numerics (NumericsManager):      Numerics configuration.
   """
   model:       Optional[ModelBase]                = None
   mesh:        Optional[MeshManager]              = None
   BCs:         Optional[BoundaryConditionManager] = None
   output:      Optional[OutputManager]            = None
   acquisition: Optional[Acquisition]              = None
   numerics:    Optional[NumericsManager]          = None

   def to_dict(self) -> Dict:
      return {
         **super().to_dict(),
         **({"Model":       self.model.to_dict()}        if self.model else {}),
         **({"Mesh":        self.mesh.to_dict()}         if self.mesh else {}),
         **({"BCs":         self.BCs.to_dict()}          if self.BCs else {}),
         **({"Output":      self.output.to_dict()}       if self.output else {}),
         **({"Acquisition": self.acquisition.to_dict()}  if self.acquisition else {}),
         **({"Numerics":    self.numerics.to_dict()}     if self.numerics else {})
      }


   def to_json(self, **kwargs) -> str:
      return json.dumps(self.to_dict(), cls=CustomJSONEncoder, **kwargs)
   

   def save(self, path: Union[str, Path], **kwargs) -> str:
      """Save project to JSON file."""
      file = Path(path) / f"{self.name}.json"
      with open(file, "w") as f:
         json.dump(self.to_dict(), f, cls=CustomJSONEncoder, **kwargs)


   @classmethod
   def from_dict(cls, data: Dict) -> 'Simulation':
      config = super().from_dict(data)
      sim = cls(**config)

      sim.model = ModelBase.from_dict(data["Model"])
      sim.mesh = MeshManager.from_dict(sim, data["Mesh"])
      sim.BCs = BoundaryConditionManager.from_dict(data["BCs"])
      sim.output = OutputManager.from_dict(data["Output"])
      sim.acquisition = Acquisition.from_dict(data["Acquisition"])
      sim.numerics = NumericsManager.from_dict(data["Numerics"])

      return sim