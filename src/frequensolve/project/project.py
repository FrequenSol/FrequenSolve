
from typing      import List, Union, Path
from dataclasses import dataclass, field

from ..simulation.simulation import Simulation

@dataclass
class Project:
   """Container for storing project information.

   Attributes:
      path (str):                      The path to the project directory.
      simulations (List[Simulation]):  List of simulations in the project.
      version (str):                   FrequenSolve version for this project.
   """
   path:           Union[str, Path]
   simulations:    List[Simulation] = field(default_factory = list)
   version:        str

