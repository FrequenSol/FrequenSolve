
import numpy as np

from dataclasses  import dataclass, field
from typing       import Optional, List, Literal

from ..util.input_parser import *  # noqa
from .signature import *           # noqa
from .wavelet   import *           # noqa

__all__ = ['CartesianGrid', 'ReceiverComponent', 'ReceiverGroup']

# Defines a uniform grid (for recievers)
@dataclass
class CartesianGrid:
   """
   @class   CartesianGrid
   @brief   Defines a uniform Cartesian grid
   """
   n:  Optional[List[int]]   = field(default_factory=list)
   x0: List[float]           = field(default_factory=list)
   x1: Optional[List[float]] = field(default_factory=list)
   dx: Optional[List[float]] = field(default_factory=list)

   def __post_init__(self):
      if self.x1 is None:
         self.x1 = [x0 + (n - 1) * dx for x0, n, dx in zip(self.x0, self.n, self.dx)]
      elif self.dx is None:
         self.dx = [(x1 - x0) / (n - 1) for x0, x1, n in zip(self.x0, self.x1, self.n)]
      elif self.n is None:
         self.n = [int((x1 - x0) / dx + 1) for x0, x1, dx in zip(self.x0, self.x1, self.dx)]

   def generate_coords(self) -> List[List[float]]:
      """
      @brief   Lists grid point coordinates
      """
      if len(self.n) == 2:
         return np.array([[x, y] for x in np.linspace(self.x0[0], self.x1[0], self.n[0])
                       for y in np.linspace(self.x0[1], self.x1[1], self.n[1])])
      elif len(self.n) == 3:
         return np.array([[x, y, z] for x in np.linspace(self.x0[0], self.x1[0], self.n[0])
                              for y in np.linspace(self.x0[1], self.x1[1], self.n[1])
                              for z in np.linspace(self.x0[2], self.x1[2], self.n[2])])
      return []
      
   def __str__(self) -> str:
      n  = " ".join(map(str, self.n ))
      x0 = " ".join(map(str, self.x0))
      x1 = " ".join(map(str, self.x1))
      dx = " ".join(map(str, self.dx))
      return (
         f"   [Grid]\n"
         f"      n  = {n}\n"
         f"      x0 = {x0}\n"
         f"      x1 = {x1}\n"
         f"      dx = {dx}\n"
         f"   []\n"
      )


@dataclass
class ReceiverComponent:
   """
   @class   ReceiverComponent
   @brief
   """
    
   name:      str
   field:     str = "pressure"
   direction: Optional[List[float]] = None
   
   def __str__(self) -> str:
      lines = [f"      [{self.name}]",
               f"         type = {self.meas_type}"]
      if self.direction is not None:
         dir_str = " ".join(map(str, self.direction))
         lines.append(f"         direction  = {dir_str}")
      lines.append("      []")
      return "\n".join(lines) + "\n"
   
   
@dataclass
class ReceiverGroup:
   """
   @class   ReceiverGroup
   @brief   Represents a group of multi-component receivers.
   @details All items in a receiver group will be output to the same .h5
            file.
   """
   name:          str
   kind:          Literal["node", "fiber", "grid"]
   frame:         Literal["physical", "reference"] = "physical"
   components:    List[ReceiverComponent] = field(default_factory = list)
   signatures:    Optional[Signature]  = None
   directory:     Optional[str]        = None
   coord_file:    Optional[str]        = None
   coords:        Optional[np.ndarray] = None
   
#      if not directory:
#         self.directory = default_directory
#         
#      if self.file:
#         assert os.path.exists(file)
#         self.file = file
#      elif self.coords:
#         file = os.path.join(directory,f"{name}.csv")
#         with open(file, 'w') as f:
#            for row in coords:
#               f.write(" ".join(map(str,row)) + "\n")
   
   @classmethod
   def from_block(cls, input: InputParser, block: InputBlock):
      
      name = block.name
      args = block.args
      
      dim = input.get_block("Problem").args["dimension"]
         
      # In adjoint mode, get signature at receivers as well
      if block.find_block("Adjoint"):
         sig_block  = block.find_block("Signature")
         kind       = "from_file"
         signatures = SignatureFromFile.from_block(input, sig_block)
      else:
         signatures = None
         
      # Get output directory
      default_dir = input.get_block("Output").args["directory"]
      if "directory" in args:
         output_dir = args["directory"]
      else:
         output_dir = default_dir
      
      # Get measurements and grid subblocks
      components = []
      for blk in block.sub_blocks:
         if (blk.name == "Grid" or
             blk.name == "Signature"):
            continue
         else:
            comp = ReceiverComponent(name      = blk.name,
                                     field     = blk.args["field"],
                                     direction = blk.args.get("direction"))
            components.append(comp)
   
      frame = args.get("frame","physical")
      
      kind  = args["kind"]
      if kind == "grid":
         grid_args = block.find_block("Grid").args
         grid = CartesianGrid(
            n  = str_to_array(grid_args.get("n", [])),
            x0 = str_to_array(grid_args["x0"]),
            x1 = str_to_array(grid_args.get("x1", [])),
            dx = str_to_array(grid_args.get("dx", []))
         )
         coords = grid.generate_coords()
   
      frame = args.get("frame","physical")
      coord_file = args.get("coordinates")
      if coord_file:
         coords    = np.loadtxt(coord_file,
                                delimiter = " ",
                                dtype     = float)

      return cls(
         name       = name,
         kind       = kind,
         frame      = frame,
         components = components,
         directory  = output_dir,
         coord_file = coord_file,
         coords     = coords,
         signatures = signatures
      )

   # TODO: specialize into receiver types
   # TODO: method to define receviers
   # TODO: method to attach signatures
   
   def signature(self, irecv: int) -> Wavelet:
      """
      @brief Retrieve a source signature by index
      @param isrc    Source number (1-based).
      @return The Source object at that index.
      """
      return self.signatures.get(irecv)
      
                    
   def __str__(self) -> str:
      if self.coord_file:
         file = self.coord_file
      else:
         file = os.path.join(problem_dir,self.name + "_coords.csv")
   
      out  = f"   [{self.name}]\n"
      out += f"      kind      = {self.kind}\n"
      out += f"      frame     = {self.frame}\n"
         
      if self.kind == "grid":
         out += str(self.grid)
      else:
         out += f"      coords = {file}\n"
      
      if self.signatures:
         out += str(self.signatures)
         
      for meas in self.measurements:
         out += str(meas)

      out += "   []\n\n"
      return out

   @property
   def size(self):
      return np.shape(self.coords)[0]
