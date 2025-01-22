import numpy as np

from dataclasses  import dataclass, field
from typing       import List, Optional, Literal, Dict

from .signals            import *  # noqa

__all__ = ['SourceGroup','Source']

@dataclass
class Source:
   kind:        Literal["scalar", "vector", "moment"]
   frame:       Literal["physical", "reference"] = "physical"
   coordinates: List[float] = field(default_factory=list)
   direction:   List[float] = field(default_factory=list)
   name:        str         = "source"

   @classmethod
   def from_dict(cls, data: Dict):
      return cls(**data)
   
   
   def to_dict(self) -> Dict:
      return {
         "name": self.name,
         "kind": self.kind,
         "frame": self.frame,
         "coordinates": self.coordinates,
         **({"direction": self.direction} if len(self.direction) > 0 else {})
      }
      
      
@dataclass
class SourceGroup:
   """A group of sources (to simulate simultaneously)

   Attributes:
      signals (Optional[Signal]): Source wavelet
      sources (List[Source]):     List of source objects
   """
   sources:       List[Source] = field(default_factory=list)
   signals:       Optional[Signal] = None   
   

   @classmethod
   def from_dict(cls, data: Dict):
      sources = data.get("sources", [])
      signals = data.get("signals", None)
      return cls(sources=sources, signals=signals)
   

   def to_dict(self) -> Dict:
      data = {
         "sources": [src.to_dict() for src in self.sources]
      }
      if self.signals:
         data["signals"] = self.signals.to_dict()
      return data
   
   
   def signature(self, isrc: int):
      if self.signals:
         return self.signals.get(isrc)
      else:
         return None
