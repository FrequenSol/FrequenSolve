"""Class for managing finite element meshes."""

import numpy as np

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Union

__all__ = ['Mesh']

try:
   import _mesh         # Optional C++ bindings
   HAS_CPP = True
except ImportError:
   HAS_CPP = False

@dataclass
class Vertex:
   """A vertex in 2D or 3D space.
   
   Attributes:
      coords (np.ndarray): Coordinates of the vertex.
      gmp_str (str): GMP string identifier (e.g., "Regular").
   """
   coords: np.ndarray
   
   def to_dict(self) -> Dict:
      return {
         "coords": self.coords.tolist()
      }


@dataclass
class Element:
   """Base class for mesh elements.
   
   Attributes:
      vertices (List[int]): Indices of vertices defining the element (1-based).
      domain (int): Domain ID for the element.
      active (bool): Whether element is active in mesh.
      gmp_str (str): GMP string identifier (e.g., "Seglin", "PlaneTri").
   """
   vertices: List[int]
   kind: int = field(default=0, metadata={"enum": {
      "null": 0,
      "edge": 1,
      "triangle": 2, 
      "quad": 3,
      "tetra": 4,
      "hexa": 5,
      "prism": 6,
      "pyramid": 7,
   }})
   domain: Optional[int] = None
   active: bool = True
   
   @property
   def kind_str(self) -> str:
      """Get element type identifier.
      
      Returns:
         str: The element type as a string (e.g. "edge", "triangle", etc).
      """
      for name, value in self.kind.__class__.metadata["enum"].items():
         if value == self.kind:
            return name
      return "unknown"
   
   @property
   def gmp_str(self) -> str:
      if self.kind == 1:
         return "Seglin"
      elif self.kind == 2:
         return "PlaneTri"
      elif self.kind == 3:
         return "PlaneQuad"
      elif self.kind == 4:
         return "Linear"
      elif self.kind == 5:
         return "Linear"
      elif self.kind == 6:
         return "Linear"
      elif self.kind == 7:
         return "Linear"
      else:
         raise ValueError(f"Invalid element kind: {self.kind}")

   
   def to_dict(self) -> Dict:
      return {
         "vertices": self.vertices,
         "domain": self.domain,
         "active": self.active,
         "kind": self.kind
      }


class Mesh:
   """Container for mesh vertices and elements with support for I/O and manipulation.
   
   This class provides a Python interface with optional C++ mesh implementation.
   """
   
   def __init__(self, dimension: int):
      """Initialize empty mesh.
      
      Args:
         dimension: Spatial dimension (2 or 3).
         initial_order: Initial order of the mesh.
      """
      # Initialize mesh object
      if HAS_CPP:
         self._cpp_mesh = _mesh.Mesh(dimension)  # Use C++ binding if available
      else:
         self._vertices = []
         self._elements = []
      self.dimension = dimension
      
   def add_vertex(self, coords: Union[List[float], np.ndarray]) -> None:
      """Add a vertex to the mesh.
      
      Args:
         coords: Vertex coordinates (length must match mesh dimension).
      """
      coords = np.asarray(coords, dtype=np.float64)
      if len(coords) != self.dimension:
         raise ValueError(f"Coordinate length {len(coords)} != mesh dimension {self.dimension}")
      if HAS_CPP:
         self._cpp_mesh.add_vertex(coords)
      else:
         self._vertices.append(Vertex(coords=coords))
      
   def add_element(self, elem_type: str, vertices: List[int], domain: int = 1) -> None:
      """Add an element to the mesh.
      
      Args:
         elem_type: Type of element ("edge", "triangle", "quad", etc.).
         vertices: List of vertex indices (1-based).
         domain: Domain ID for the element.
      """
      if HAS_CPP:
         self._cpp_mesh.add_element(elem_type, vertices, domain)
      else:
         self._elements.append(Element(kind=elem_type, vertices=vertices, domain=domain))
      
   def write(self, filename: Union[str, Path]) -> None:
      """Write mesh to file.
      
      Args:
         filename: Path to output file.
      """
      if HAS_CPP:
         self._cpp_mesh.write(str(filename))
      else:
         self.write_mesh(filename, format="gmp")
      
   def write_mesh(self, file: Union[str, Path], format: str = "hp3d") -> None:
      """Write mesh to file in hp3d format.
      
      Args:
         file: Path to output file
         format: Output format ("gmp", "gmsh", or "exodus")
      """

      if format == "gmp":
         self._write_gmp(file)
      elif format == "gmsh":
         self._write_gmsh(file)
      elif format == "exodus":
         self._write_exodus(file)
      else:
         raise ValueError(f"Invalid format: {format}")
      
   def _write_exodus(self, file: Union[str, Path]) -> None:
      """Write mesh to file in Exodus format.
      
      Args:
         file: Path to output file.
      """
      try:
         import exodus3 as exodus
      except ImportError:
         raise ImportError("The exodus3 Python package must be installed to use this mesh format (e.g. `pip install exodus3`)")

      # Get vertices and active elements
      if HAS_CPP:
         vertices = self._cpp_mesh.get_vertices()
         active_elems = [e for e in self._cpp_mesh.get_elements() if e.is_active()]
      else:
         vertices = self._vertices
         active_elems = [e for e in self._elements if e.active]
      
      # Create new exodus file
      with exodus.DatabaseFile(str(file), mode='w') as e:
         # Write coordinates
         coords = np.array([v.get_coords() for v in vertices])
         if self.dimension == 2:
            coords = np.pad(coords, ((0,0), (0,1)))
         e.put_coords(coords[:,0], coords[:,1], coords[:,2])
         
         # Write element blocks
         elem_by_kind = {}
         for elem in active_elems:
            kind = elem.kind()
            if kind not in elem_by_kind:
               elem_by_kind[kind] = []
            elem_by_kind[kind].append(elem)
            
         for kind, elems in elem_by_kind.items():
            # Map element types to exodus names
            exodus_name = {
               1: "EDGE2",
               2: "TRI3", 
               3: "QUAD4",
               4: "TETRA4",
               5: "HEXA8",
               6: "PRISM6",
               7: "PYRAMID5"
            }[kind]
            
            # Sort elements by domain and create block
            elems_by_domain = {}
            for elem in elems:
               domain = elem.domain()
               if domain not in elems_by_domain:
                  elems_by_domain[domain] = []
               elems_by_domain[domain].append(elem)
               
            for domain, domain_elems in elems_by_domain.items():
               block_id = domain
               e.put_elem_blk_info(block_id, exodus_name, len(domain_elems))
            
            # Write connectivity
            connect = np.array([elem.indices() for elem in elems]) 
            e.put_elem_connectivity(block_id, connect)

   def _write_gmsh(self, file: Union[str, Path]) -> None:
      """Write mesh to file in Gmsh format.
      
      Args:
         file: Path to output file.
      """
      # Get vertices and active elements
      if HAS_CPP:
         vertices = self._cpp_mesh.get_vertices()
         active_elems = [e for e in self._cpp_mesh.get_elements() if e.is_active()]
      else:
         vertices = self._vertices
         active_elems = [e for e in self._elements if e.active]
      
      with open(file, 'w') as f:
         # Write header
         f.write("$MeshFormat\n")
         f.write("2.2 0 8\n")  # Version 2.2, ASCII format, double precision
         f.write("$EndMeshFormat\n")
         
         # Write nodes
         f.write("$Nodes\n")
         f.write(f"{len(vertices)}\n")
         for i, v in enumerate(vertices, 1):
            coords = v.get_coords()
            # Pad with zeros if 2D
            if len(coords) == 2:
               coords = list(coords) + [0.0]
            f.write(f"{i} {coords[0]} {coords[1]} {coords[2]}\n")
         f.write("$EndNodes\n")
         
         # Write elements
         f.write("$Elements\n")
         f.write(f"{len(active_elems)}\n")
         for i, elem in enumerate(active_elems, 1):
            elem_type = elem.kind()
            indices = elem.indices()
            f.write(f"{i} {elem_type} 2 {elem.domain()} {elem.domain()}")
            for idx in indices:
               f.write(f" {idx}")
            f.write("\n")
         f.write("$EndElements\n")


   def _write_gmp(self, file: Union[str, Path]) -> None:
      """Write mesh to file in GMP format."""
      # Get vertices and active elements
      if HAS_CPP:
         vertices = self._cpp_mesh.get_vertices()
         active_elems = [e for e in self._cpp_mesh.get_elements() if e.is_active()]
      else:
         vertices = self._vertices
         active_elems = [e for e in self._elements if e.active]
      
      # Group elements by kind
      elem_by_kind = {}
      for elem in active_elems:
         kind = elem.kind_str
         if kind not in elem_by_kind:
            elem_by_kind[kind] = []
         elem_by_kind[kind].append(elem)
      
      # Write mesh file
      with open(file, 'w') as f:
         # Write header
         f.write(f"{self.dimension} {self.dimension}\n")
         f.write("\n0 NRSURFS\n")
         f.write("\n1 NRDOMAIN\n")
         
         # Write vertices
         f.write(f"\n{len(vertices)} vertices\n")
         for v in vertices:
            f.write(f"Regular\n")
            coords = v.get_coords()
            f.write(" ".join(str(x) for x in coords) + "\n")
            
         # Write elements by type
         elem_types = ["edge", "triangle", "quad", "prism", "hexa", "tetra", "pyramid"]
         for etype in elem_types:
            elems = elem_by_kind.get(etype, [])
            f.write(f"\n{len(elems)} {etype}s\n")
            for e in elems:
               f.write(f"{e.gmp_str}\n")
               indices = e.indices()
               f.write(f"{e.domain()} " + " ".join(str(i) for i in indices) + "\n")

   def read_mesh(self, file: Union[str, Path], format: str = "hp3d") -> None:
      """Read mesh from file.
      
      Args:
         file: Path to input file
         format: Input format ("gmp", "gmsh", or "exodus")
      """
      if format == "gmp":
         self._read_gmp(file)
      elif format == "gmsh":
         self._read_gmsh(file)
      elif format == "exodus":
         self._read_exodus(file)
      else:
         raise ValueError(f"Invalid format: {format}")

   def _read_exodus(self, file: Union[str, Path]) -> None:
      """Read mesh from Exodus format file.
      
      Args:
         file: Path to input file.
      """
      try:
         import exodus3 as exodus
      except ImportError:
         raise ImportError("exodus3 package required for Exodus format")

      with exodus.DatabaseFile(str(file), mode='r') as e:
         # Read coordinates
         x, y, z = e.get_coords()
         coords = np.column_stack([x, y, z])
         if self.dimension == 2:
            coords = coords[:, :2]
         
         for coord in coords:
            self.add_vertex(coord)

         # Read element blocks
         for block_id in e.get_elem_blk_ids():
            elem_type = e.get_elem_type(block_id)
            connect = e.get_elem_connectivity(block_id)
            
            # Map exodus names to our element types
            kind = {
               "EDGE2": 1,
               "TRI3": 2,
               "QUAD4": 3, 
               "TETRA4": 4,
               "HEXA8": 5,
               "PRISM6": 6,
               "PYRAMID5": 7
            }[elem_type]
            
            # Add elements
            for conn in connect:
               self.add_element(kind, conn.tolist(), domain=block_id)

   def _read_gmsh(self, file: Union[str, Path]) -> None:
      """Read mesh from Gmsh format file.
      
      Args:
         file: Path to input file.
      """
      with open(file) as f:
         lines = f.readlines()
         
      i = 0
      while i < len(lines):
         line = lines[i].strip()
         
         if line == "$Nodes":
            num_nodes = int(lines[i+1])
            i += 2
            for _ in range(num_nodes):
               _, x, y, z = map(float, lines[i].split())
               coords = [x, y] if self.dimension == 2 else [x, y, z]
               self.add_vertex(coords)
               i += 1
               
         elif line == "$Elements":
            num_elems = int(lines[i+1])
            i += 2
            for _ in range(num_elems):
               parts = list(map(int, lines[i].split()))
               elem_type = parts[1]
               domain = parts[3]
               vertices = parts[5:]
               self.add_element(elem_type, vertices, domain)
               i += 1
               
         else:
            i += 1

   def _read_gmp(self, file: Union[str, Path]) -> None:
      """Read mesh from GMP format file.
      
      Args:
         file: Path to input file.
      """
      with open(file) as f:
         lines = [line.lower() for line in f.readlines()]
         
      i = 0
      dim = int(lines[i].split()[0])
      if dim != self.dimension:
         raise ValueError(f"Mesh dimension {dim} != expected {self.dimension}")
         
      # Skip header sections
      while not lines[i].strip().endswith("vertices"):
         i += 1
         
      num_vertices = int(lines[i].split()[0])
      i += 1
      
      # Read vertices
      for _ in range(num_vertices):
         i += 1  # Skip "Regular" line
         coords = list(map(float, lines[i].split()))
         self.add_vertex(coords[:self.dimension])
         i += 1
         
      # Read elements
      while i < len(lines):
         line = lines[i].strip()
         if not line:
            i += 1
            continue
            
         if any(line.endswith(s) for s in ["edges", "triangles", "quads", "tetras", "hexas"]):
            num_elems = int(line.split()[0])
            elem_type = {
               "edges": 1,
               "triangles": 2,
               "quads": 3,
               "tetras": 4,
               "hexas": 5,
               "prisms": 6,
               "pyramids": 7
            }[line.split()[1]]
            
            i += 1
            for _ in range(num_elems):
               i += 1  # Skip GMP string
               parts = lines[i].split()
               domain = int(parts[0])
               vertices = list(map(int, parts[1:]))
               self.add_element(elem_type, vertices, domain)
               i += 1
         else:
            i += 1
