import numpy as np

from dataclasses  import dataclass, field
from typing    import List, Optional, Literal

from ..util.input_parser import *  # noqa
from .signature          import *  # noqa
from .wavelet            import *  # noqa

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
   """
   @class Source
   @brief Lists source information
   @param signature   A Signature object describing the source wavelet or shape.
   """
   signatures:    Optional[Signature] = None
   sources:       List[Source] = field(default_factory=list)

   @classmethod
   def from_block(cls, input: InputParser, block: InputBlock):
      
      name = block.name
      args = block.args
      
      # Identify signature block
      src_blocks = block.sub_blocks
      sig_block  = block.find_block("Signature")
      if sig_block:
         kind = sig_block.args["kind"]
         if kind == "from_file":
            sig = SignatureFromFile.from_block(input, sig_block)
         elif kind in ["Ricker", "Ormsby", "Klauder"]:
            sig = GeneratedSignature.from_block(input, sig_block)

         # Remove the signature block
         src_blocks.remove(sig_block)
      else:
         sig = None
      
      # Create source blocks
      srcs = [Source.from_block(input, block) for block in src_blocks]
      
      return cls(signatures = sig,
                 sources    = srcs)
   
   
   def signature(self, isrc: int):
      """
      @brief Retrieve a source signature by index
      @param isrc Source number (1-based).
      @return The Source object at that index.
      """
      if self.signatures:
         return self.signatures.get(isrc)
      else:
         return None


   def __str__(self) -> str:
      if not self.sources:
         return ""
      out = "[Source]\n"
      out += str(self.signature)
      for src in self.sources:
         out += str(src)
      out += "[]\n\n"
      return out
