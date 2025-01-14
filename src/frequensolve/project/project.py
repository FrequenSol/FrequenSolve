
from typing      import Optional, List
from pydantic    import BaseModel, Field
from dataclasses import dataclass, field

@dataclass
class Project:
   """
   @class Project
   @brief
   """
   path:
   log_level:
   problems:         List[Problem] = field(default_factory = list)
   version:
   auto_migrate:
   
   def __init__(self):


