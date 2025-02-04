from abc          import ABC, abstractmethod
from dataclasses  import dataclass, field, asdict
from typing       import List, Optional, Literal, Dict, Tuple
from pathlib      import Path

from ..util.class_registry import *  # noqa

__all__ = ['SourceGroup', 'Source', 'RuptureSource', 'PointSource']


@register_class
@dataclass
class Source(ABC):

   @classmethod
   def from_dict(cls, data: Dict) -> "Source":
      class_name = data.pop("_type")
      if class_name in class_registry:
         source_class = class_registry[class_name]
         return source_class.from_dict(data)
      else:
         raise ValueError(f"Unknown source class: {class_name}")
   
   @abstractmethod
   def __dict__(self) -> Dict:
      pass


@register_class
@dataclass
class RuptureSource(Source):
   """Source defined from Standard Rupture Format file.

   Args:
      srf_file: Path to SRF file containing rupture definition.
      name: Optional name for the source.

   Attributes:
      srf_file (str):   Path to SRF file.
      name (str):       Name of the source.
   """
   srf_file:    str
   name:        str = "rupture"

   @classmethod
   def from_dict(cls, data: Dict) -> "RuptureSource":
      return cls(**data)

   def __dict__(self) -> Dict:
      return {
         "_type": self.__class__.__name__,
         **asdict(self)
      }


@register_class
@dataclass
class PointSource(Source):
   kind:        Literal["scalar", "vector", "moment"]
   frame:       Literal["physical", "reference"] = "physical"
   coordinates: List[float]            = field(default_factory=list)
   direction:   Optional[List[float]]  = None
   name:        str                    = "point"

   @classmethod
   def from_dict(cls, data: Dict):
      return cls(**data)
   
   def __dict__(self) -> Dict:
      return {
         "_type":          self.__class__.__name__,
         "name":           self.name,
         "kind":           self.kind,
         "frame":          self.frame,
         "coordinates":    self.coordinates,
         **({"direction":  self.direction} if self.direction is not None else {})
      }


@dataclass
class SourceGroup:
   """A group of sources (to simulate simultaneously)

   Attributes:
      sources (List[PointSource]):     List of source objects
   """
   sources:  List[Source] = field(default_factory=list)
   _proj_path: Path = None
   _rel_path: Path = None

   @classmethod
   def from_dict(cls, data: Dict):
      sources = [Source.from_dict(source) for source in data.get("sources", [])]
      return cls(sources=sources)

   def __dict__(self) -> Dict:
      return {
         "sources": [source.__dict__() for source in self.sources]
      }
   
   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path
   
   @property
   def _path(self) -> Path:
      return self._proj_path/self._rel_path


