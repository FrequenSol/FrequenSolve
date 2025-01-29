"""Python structures defining seismic acquisition geometry"""

import numpy as np

from dataclasses  import dataclass, field
from typing       import List, Optional, Dict, Tuple
from pathlib      import Path

from .sources    import *  # noqa
from .receivers  import *  # noqa

__all__ = ['Acquisition']


@dataclass
class Acquisition:
   """Defines a seismic source and receiver configuration.

   This class reads the input file to retrieve blocks describing sources, receivers, and 
   wavelet signatures. It then aggregates them into a single cohesive acquisition definition.

   Attributes:
      source_group (SourceGroup): A group of source objects describing all shot points.
      receiver_groups (List[ReceiverGroup]): A list of ReceiverGroup objects (stations, geophones, or fibers).
   """
   source_group:     SourceGroup           = field(default_factory=SourceGroup)
   receiver_groups:  List[ReceiverGroup]   = field(default_factory=list)
   _proj_path:      Optional[Path]        = None
   _rel_path:       Optional[Path]        = None

   @classmethod
   def from_dict(cls, dict: Dict) -> 'Acquisition':
      return cls(
         source_group    = SourceGroup.from_dict(dict["source_group"]),
         receiver_groups = [ReceiverGroup.from_dict(group) for group in dict["receiver_groups"]],
      )
   
   def __dict__(self) -> Dict:
      return {
         "source_group": self.source_group.__dict__(),
         "receiver_groups": [group.__dict__() for group in self.receiver_groups],
      }
      
   def add_source_group(self,
                        kind:        str,
                        coords:      np.ndarray,
                        direction:   Optional[np.ndarray] = None,
                        frame:       str = "phyiscal"):
      """Add a group of recievers with common kind, frame, and direction.
      
      Args:
         kind (str): Kind of the receiver group (e.g., "station", "geophone", "fiber").
         coords (np.ndarray): Coordinates of the receiver group.
         direction (np.ndarray): Direction of the receiver group.
         frame (str): Frame of the receiver group (e.g., "physical", "global").
      """
           
      for row in coords:
         isrc = len(self.source_group.sources)
         self.source_group.sources.append(
            PointSource(
               kind        = kind,
               frame       = frame,
               coordinates = row,
               direction   = direction,
               name        = f"source_{isrc}"
            )
         )
         

   def add_receiver_group(self,
                          name:   str,
                          device: ReceiverDevice,
                          coords: np.ndarray,
                          frame:  str = "phyiscal"):
      """Add a group of recievers with common kind, frame, and direction.

      Args:
         name (str):                Name of the receiver group.
         device (ReceiverDevice):   Device defining receiver type and components.
         coordinates (np.ndarray):  Coordinates of the receiver group.
         frame (str): Frame of the receiver group (e.g., "physical", "global").
      """
                        
      self.receiver_groups.append(
         ReceiverGroup(
            name        = name,
            device      = device,
            frame       = frame,
            coordinates = coords
         )
      )


   def list_fields(self, recv_name: str = "") -> List[str]:
      """List available fields for a specified receiver group or for all groups. """
      field_list = []
      
      if recv_name:
         group = self.receiver_group(recv_name)
         for field in group.components:
            file = f"{group.name}:{field.name}"
            field_list.append(file)
      else:
         for group in self.receiver_groups:
            for field in group.components:
               file = f"{group.name}:{field.name}"
               field_list.append(file)
      return field_list
      
      
   def list_sources(self) -> List[int]:
      """List valid source numbers."""
      return list(range(1, len(self.source_group.sources) + 1))
           
           
   def receiver_group(self, name: str) -> Optional[ReceiverGroup]:
      """Retrieve a named receiver group by its block name."""
      for group in self.receiver_groups:
         if group.name == name:
            return group
      return None
      
      
   def source(self, isrc: int) -> Source:
      """Retrieve a source by index."""
      try:
         return self.source_group.sources[isrc-1]
      except IndexError:
         raise IndexError(f"Source index {isrc} is out of range.")

   def _set_path(self, proj_path: Path, rel_path: Path):
      self._proj_path = proj_path
      self._rel_path = rel_path
      for group in self.receiver_groups:
         group._set_path(proj_path, rel_path)
      self.source_group._set_path(proj_path, rel_path)
   
   @property
   def _path(self) -> Path:
      return self._proj_path/self._rel_path
