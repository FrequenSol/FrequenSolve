import numpy as np

from dataclasses  import dataclass, field
from typing       import List, Optional, Literal, Dict

from ..util.input_parser import *  # noqa
from .signals            import *  # noqa

__all__ = ['SourceGroup','Source']

@dataclass
class Source:
   kind:        Literal["explosive", "viberator"]
   frame:       Literal["physical", "reference"] = "physical"
   coordinates: List[float] = field(default_factory=list)
   direction:   List[float] = field(default_factory=list)
   name:        str         = "source"

   @classmethod
   def from_block(cls, input: InputParser, block: InputBlock):
      """Create a Source object from an input block.

      Args:
         input (InputParser): The input parser.
         block (InputBlock): The input block.

      Returns:
         Source: The created Source object.
      """
      name = block.name
      args = block.args
      
      kind      = args["kind"]
      direction = args.get("direction",[])
      frame     = args.get("frame","physical")
      x         = str_to_array(args["coordinates"])
   
      return cls(
         name        = name,
         kind        = kind,
         frame       = frame,
         coordinates = x,
         direction   = direction
      )
   

   @classmethod
   def from_dict(cls, data: Dict):
      return cls(**data)
   
   
   def to_dict(self) -> Dict:
      """Convert the Source object to a dictionary.

      Returns:
         Dict: The dictionary representation of the Source object.
      """
      return {
         "name": self.name,
         "kind": self.kind,
         "frame": self.frame,
         "coordinates": self.coordinates,
         **({"direction": self.direction} if len(self.direction) > 0 else {})
      }
   
                 
   def __str__(self) -> str:
      """Convert the Source object to a string representation.

      Returns:
         str: The string representation of the Source object.
      """
      coordinates = " ".join(map(str, self.coordinates))
      direction   = " ".join(map(str, self.direction))
      
      string = (
         f"   [{self.name}]\n"
         f"      kind        = {self.kind}\n"
         f"      frame       = {self.frame}\n"
         f"      coordinates = {coordinates}\n"
      )
      if len(self.direction) > 0:
         string += f"      direction   = {direction}\n"
      string += f"   []\n"
      
      return string
      
      
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
   def from_block(cls, input: InputParser, block: InputBlock):
      """Create a SourceGroup object from an input block.

      Args:
         input (InputParser): The input parser.
         block (InputBlock): The input block.

      Returns:
         SourceGroup: The created SourceGroup object.
      """
      name = block.name
      args = block.args
      
      # Identify Signal block
      src_blocks = block.sub_blocks
      wav_block  = block.find_block("Signature")
      if wav_block:
         kind = wav_block.args["kind"]
         if kind == "from_file":
            wav = SignalFromFile.from_block(input, wav_block)
         elif kind in ["Ricker", "Ormsby", "Klauder"]:
            wav = AnalyticalSignal.from_block(input, wav_block)

         # Remove the Signal block
         src_blocks.remove(wav_block)
      else:
         wav = None
      
      # Create source blocks
      srcs = [Source.from_block(input, block) for block in src_blocks]
      
      return cls(signals = wav,
                 sources   = srcs)
   
   
   @classmethod
   def from_dict(cls, data: Dict):
      """Create a SourceGroup object from a dictionary.

      Args:
         data (Dict): The dictionary containing the SourceGroup data.

      Returns:
         SourceGroup: The created SourceGroup object.
      """
      sources = data.get("sources", [])
      signals = data.get("signals", None)
      return cls(sources=sources, signals=signals)
   

   def to_dict(self) -> Dict:
      """Convert the SourceGroup object to a dictionary.
      
      Returns:
         Dict: Dictionary containing:
            - sources:   List of source dictionaries
            - signals: Dictionary of signal data if present
      """
      data = {
         "sources": [src.to_dict() for src in self.sources]
      }
      if self.signals:
         data["signals"] = self.signals.to_dict()
      return data
   
   
   def signature(self, isrc: int):
      """Retrieve a source wavelet by index.

      Args:
         isrc (int): Source number (1-based).

      Returns:
         Signal: Wavelet for isrc-th source
      """
      if self.signals:
         return self.signals.get(isrc)
      else:
         return None


   def __str__(self) -> str:
      """Convert the SourceGroup object to a string representation.

      Returns:
         str: The string representation of the SourceGroup object.
      """
      if not self.sources:
         return ""
      out = "[Source]\n"
      out += str(self.signals)
      for src in self.sources:
         out += str(src)
      out += "[]\n\n"
      return out
