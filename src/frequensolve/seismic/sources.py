import numpy as np

from dataclasses  import dataclass, field
from typing    import List, Optional, Literal

from ..util.input_parser import *  # noqa
from .waveform           import *  # noqa

__all__ = ['SourceGroup','Source']

@dataclass
class Source:
   kind:        Literal["explosive", "viberator"]
   frame:       Literal["physical", "reference"] = "physical"
   coords:      List[float] = field(default_factory=list)
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
      x         = str_to_array(args["x"])
   
      return cls(
         name      = name,
         kind      = kind,
         frame     = frame,
         coords    = x,
         direction = direction
      )
                 
   def __str__(self) -> str:
      """Convert the Source object to a string representation.

      Returns:
         str: The string representation of the Source object.
      """
      coords    = " ".join(map(str, self.coords))
      direction = " ".join(map(str, self.direction))
      
      string = (
         f"   [{self.name}]\n"
         f"      kind        = {self.kind}\n"
         f"      frame       = {self.frame}\n"
         f"      coordinates = {coords}\n"
      )
      if self.direction:
         string += f"      direction   = {direction}\n"
      string += f"   []\n"
      
      return string
      
      
@dataclass
class SourceGroup:
   """A group of sources with a (optional) common waveform.

   Attributes:
      waveforms (Optional[Waveform]): The waveform for the sources.
      sources (List[Source]): The list of sources.
   """
   sources:       List[Source] = field(default_factory=list)
   waveforms:     Optional[Waveform] = None

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
      
      # Identify waveform block
      src_blocks = block.sub_blocks
      wav_block  = block.find_block("Waveform")
      if wav_block:
         kind = wav_block.args["kind"]
         if kind == "from_file":
            wav = WaveformFromFile.from_block(input, wav_block)
         elif kind in ["Ricker", "Ormsby", "Klauder"]:
            wav = AnalyticalWaveform.from_block(input, wav_block)

         # Remove the waveform block
         src_blocks.remove(wav_block)
      else:
         wav = None
      
      # Create source blocks
      srcs = [Source.from_block(input, block) for block in src_blocks]
      
      return cls(waveforms = wav,
                 sources   = srcs)
   

   # def from_dict(cls, data: Dict):




   
   
   # def waveform(self, isrc: int):
   #    """Retrieve a source waveform by index.

   #    Args:
   #       isrc (int): Source number (1-based).

   #    Returns:
   #       Waveform: The waveform for the source.
   #    """
   #    if self.waveforms:
   #       return self.waveforms.get(isrc)
   #    else:
   #       return None


   def __str__(self) -> str:
      """Convert the SourceGroup object to a string representation.

      Returns:
         str: The string representation of the SourceGroup object.
      """
      if not self.sources:
         return ""
      out = "[Source]\n"
      out += str(self.waveforms)
      for src in self.sources:
         out += str(src)
      out += "[]\n\n"
      return out
