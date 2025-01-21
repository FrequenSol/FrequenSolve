import os
import json

from pathlib     import Path
from typing      import List, Union, Literal, Optional, Dict
from dataclasses import dataclass, field

from ..simulation.simulation  import *  # noqa
from ..simulation.sampling    import *  # noqa
from .migration               import *  # noqa

__all__ = ['Project']

@dataclass(kw_only=True)
class Project:
   """Container for storing project information.

   Attributes:
      path (str):                      The path to the project directory.
      simulations (List[Simulation]):  List of simulations in the project.
      version (str):                   FrequenSolve version for this project.
   """
   name:           str
   pretty_name:    Optional[str] = None
   path:           Union[str, Path]
   simulations:    List[Simulation] = field(default_factory = list)
   version:        Version = field(default_factory=lambda: Version(major=0, minor=0, patch=0))
   load_if_exists: bool = True
   auto_migrate:   bool = False


   def __post_init__(self):
      """Load project from file and check version."""
      if self.load_if_exists:
         found = self.load()
         if found:
            self.check_version()


   def load(self) -> bool:
      """Load project from JSON file."""
      file = Path(self.path) / "project.json"
      try:
         if os.exists(file):
            with open(file, "r") as f:
               data = json.load(f)
            self.version = Version.from_string(data["version"])
            for sim in data["simulations"]:
               self.simulations.load(sim)
         return True
      except BaseException as e:
         return False
         

   def save(self, **kwargs) -> str:
      """Save project to JSON file."""
      with open(Path(self.path) / f"{self.name}.json", "w") as f:
         json.dump(self.to_dict(), f, cls=CustomJSONEncoder, **kwargs)


   def to_dict(self) -> Dict:
      return {
         "name": self.name,
         **({"pretty_name": self.pretty_name} if self.pretty_name else {}),
         "version": str(self.version),
         "simulations": [sim.to_dict() for sim in self.simulations]
      }


   def check_version(self):
      """Check project version against current version and migrate if necessary."""
      current_version = Version.current()
      if self.version < current_version:
         self.migrate(current_version)
         self.version = current_version
         self.save()


   def new_TD_simulation(self, 
                         name:      str, 
                         mode:      Literal["forward","adjoint","combined","gradient"], 
                         physics:   Literal["coupled","acoustic","elastic"], 
                         dimension: Literal[2, 3], 
                         f_min:     float, 
                         f_max:     float, 
                         df:        float, 
                         **kwargs) -> Simulation:
      """Create a new time-domain simulation and add it to the project.

      Args:
         name (str):      Simulation name.
         mode (str):      Simulation mode (forward, adjoint, gradient)
         physics (str):   Simulation physics (coupled, acoustic, elastic)
         dimension (int): Simulation dimension (2, 3)
         f_min (float):   Minimum frequency (Hz).
         f_max (float):   Maximum frequency (Hz).
         df (float):      Frequency step (Hz).
         **kwargs: Additional keyword arguments to pass to the Simulation constructor.

      Returns:
         Simulation: The newly created simulation.
      """
      tf_domain = "time"
      sampling = UniformSweepSampling(f_min = f_min, f_max = f_max, df = df)

      sim = Simulation(name      = name, 
                       physics   = physics,
                       dimension = dimension,
                       directory = os.path.join(self.path,name),
                       mode      = mode,
                       tf_domain = tf_domain, 
                       sampling  = sampling)
      self.simulations.append(sim)
      return sim
   

   def new_FD_simulation(self, 
                         name:      str, 
                         mode:      Literal["forward","adjoint","combined","gradient"], 
                         physics:   Literal["coupled","acoustic","elastic"], 
                         dimension: Literal[2, 3], 
                         f_list:    List[float], 
                         **kwargs) -> Simulation:
      """Create a new frequency-domain simulation and add it to the project.

      Args:
         name (str):      Simulation name.
         mode (str):      Simulation mode (forward, adjoint, gradient)
         physics (str):   Simulation physics (coupled, acoustic, elastic)
         dimension (int): Simulation dimension (2, 3)
         f_list (List[float]): List of frequencies (Hz).
         **kwargs: Additional keyword arguments to pass to the Simulation constructor.

      Returns:
         Simulation: The newly created simulation.
      """
      tf_domain = "frequency"
      sampling = DiscreteSampling(freq = f_list)

      sim = Simulation(name      = name, 
                       physics   = physics,
                       dimension = dimension,
                       directory = os.path.join(self.path,name),
                       mode      = mode,
                       tf_domain = tf_domain, 
                       sampling  = sampling)
      self.simulations.append(sim)
      return sim
