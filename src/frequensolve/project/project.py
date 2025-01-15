
from typing      import Optional, List
from dataclasses import dataclass, field

@dataclass
class Project:
   """Container for storing project information.

   Attributes:
      path (str): The path to the project directory.
      log_level (int): The logging level.
      version (str): The version of the project.
      auto_migrate (bool): Whether to automatically migrate the project.
   """
   path:             str
   log_level:        int
   #problems:         List[Problem] = field(default_factory = list)
   version:          str
   auto_migrate:     bool

