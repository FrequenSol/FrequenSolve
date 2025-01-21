import os
import json

from pathlib     import Path
from typing      import List, Union, Literal
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
   path:           Union[str, Path]
   simulations:    List[Simulation] = field(default_factory = list)
   version:        Version = field(default_factory = Version)
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
         


   def save(self):
      """Save project to JSON file."""
      data = {
         "version": str(self.version),
         "simulations": [sim.save() for sim in self.simulations]
      }
      with open(Path(self.path) / "project.json", "w") as f:
         json.dump(data, f, indent=3)


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
