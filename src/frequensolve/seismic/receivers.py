"""Receiver definitions and coordinate systems.

This module defines the various types of receivers and their locations.
"""

import numpy as np
import h5py
import xarray as xr

from pathlib      import Path
from dataclasses  import dataclass, field
from typing       import Optional, List, Literal, Union, Tuple, Dict
from abc          import ABC, abstractmethod

from ..geometry.grids    import *  # noqa
from ..util.input_parser import *  # noqa
from .signals           import *  # noqa
from .wavelet            import *  # noqa

__all__ = ['ReceiverComponent', 'ReceiverGroup', 'ReceiverCoordinates', 'ReceiverDevice',
           'ReceiverNodeArray', 'ReceiverNode', 'ReceiverFiber']

@dataclass
class ReceiverComponent:
   """Defines a single component/measurement type for a receiver.
   
   A receiver component specifies what physical quantity is being measured
   (e.g., pressure, velocity) and in what direction for vector quantities.
   
   Attributes:
      name (str): String identifier for this receiver component.
      field (str): Physical field being measured (e.g., "pressure", "velocity").
      direction (Optional[List[float]]): Measurement direction for vector fields.
   """
   
   name:      str
   field:     str = "pressure"
   direction: Optional[List[float]] = None

   def to_dict(self) -> dict:
      return {
         "name": self.name,
         "field": self.field,
         "direction": self.direction
      }
   
   @classmethod
   def from_dict(cls, data: dict) -> 'ReceiverComponent':
      return cls(name=data["name"], field=data["field"], direction=data["direction"])
   
   def __str__(self) -> str:
      out = f"      [{self.name}]\n" 
      out += f"         type = {self.field}\n"
      if self.direction is not None:
         dir_str = " ".join(map(str, self.direction))
         out += f"         direction  = {dir_str}\n"
      out += "      []\n"
      return out


# ----------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------
@dataclass
class ReceiverDevice:
   """Defines a single receiver.
   
   This class represents a single receiver, which is a point in space where measurements are taken.
   
   Attributes:
      name (str): String identifier for this receiver.
      kind (str): Type of receiver arrangement ("node", "fiber", or "grid").
      frame (str): Coordinate frame for measurements ("physical" or "reference").
      components (List[ReceiverComponent]): List of components defining measurements.
   """
   name: str
   components: List[ReceiverComponent] = field(default_factory=list)
   response:   Optional[Wavelet] = None

   def to_dict(self) -> dict:
      """Convert ReceiverDevice to dictionary representation.
      
      Returns:
         dict: Dictionary containing ReceiverDevice attributes.
      """
      return {
         "name": self.name,
         "components": [c.to_dict() for c in self.components],
         "response": self.response
      }

   @classmethod
   def from_dict(cls, data: dict) -> 'ReceiverDevice':
      """Create ReceiverDevice from dictionary representation.
      
      Args:
         data (dict): Dictionary containing ReceiverDevice attributes.
         
      Returns:
         ReceiverDevice: Created ReceiverDevice object.
      """
      device = cls(
         name=data["name"],
         components=[ReceiverComponent.from_dict(c) for c in data["components"]],
         response=data.get("response")
      )
      return device

   def __str__(self) -> str:
      """Convert ReceiverDevice to string representation.
      
      Returns:
         str: String representation of ReceiverDevice.
      """
      out = f"   [Device]\n"
      out += f"      name = {self.name}\n"
      for comp in self.components:
         out += str(comp)
      out += "   []\n"
      return out


@dataclass(kw_only=True)
class ReceiverNodeArray(ReceiverDevice):
   """Defines a group of nodes on a single channel; defined by list of offsets.
   
   Attributes:
      offsets (List[List[float]]): List of offsets from the 'location' of the array. 
            The fast dimension is over coordinates, the slow dimension is over nodes: 
            e.g., for a 9-node channel in 3D
               dx = 0.005 (5-m spacing)
               dy = 0.010 (10-m spacing)
               offsets = [[-dx, -dy, 0], [0, -dy, 0], [dx,-dy,0], ...
   """
   offsets: List[List[float]] = field(default_factory=list)

   def to_dict(self) -> dict:
      """Convert NodeArray to dictionary representation.
      
      Returns:
         dict: Dictionary containing NodeArray attributes.
      """
      return {
         **super().to_dict(),
         "offsets": self.offsets
      }

   @classmethod 
   def from_dict(cls, data: dict) -> 'ReceiverNodeArray':
      """Create ReceiverNodeArray from dictionary representation.
      
      Args:
         data (dict): Dictionary containing ReceiverNodeArray attributes.
         
      Returns:
         ReceiverNodeArray: Created ReceiverNodeArray object.
      """
      
      node_array = super().from_dict(data)
      node_array.offsets = data["offsets"]
      return node_array

   def __str__(self) -> str:
      """Convert NodeArray to string representation.
      
      Returns:
         str: String representation of NodeArray.
      """
      out = super().__str__()
      
      if self.offsets:
         out = out.replace("   []\n", "")  # Remove closing bracket temporarily
         out += "      [Offsets]\n"
         for offset in self.offsets:
            offset_str = " ".join(map(str, offset))
            out += f"         {offset_str}\n"
         out += "      []\n"
         out += "   []\n"
         
      return out


@dataclass(kw_only=True)
class ReceiverNode(ReceiverDevice):
   """Defines a node receiver."""
   def to_dict(self) -> dict:
      """Convert Node to dictionary representation.
      
      Returns:
         dict: Dictionary containing Node attributes.
      """
      return super().to_dict()


   @classmethod
   def from_dict(cls, data: dict) -> 'ReceiverNode':
      """Create Node from dictionary representation.
      
      Args:
         data (dict): Dictionary containing Node attributes.
         
      Returns:
         ReceiverNode: Created Node object.
      """
      node = super().from_dict(data)
      return node
   

   def __str__(self) -> str:
      """Convert Node to string representation.
      
      Returns:
         str: String representation of Node.
      """
      return super().__str__()


@dataclass(kw_only=True)
class ReceiverFiber(ReceiverDevice):
   """Defines a fiber receiver."""
   L_gauge: float
   n_gauge: int
   radius:  Optional[float] = None
   pitch:   Optional[float] = None

   def to_dict(self) -> dict:
      """Convert ReceiverFiber to dictionary representation.
      
      Returns:
         dict: Dictionary containing ReceiverFiber attributes.
      """
      data = super().to_dict()
      data.update({
         "L_gauge": self.L_gauge,
         "n_gauge": self.n_gauge,
         "radius": self.radius,
         "pitch": self.pitch
      })
      return data


   @classmethod 
   def from_dict(cls, data: dict) -> 'ReceiverFiber':
      """Create ReceiverFiber from dictionary representation.
      
      Args:
         data (dict): Dictionary containing ReceiverFiber attributes.
         
      Returns:
         ReceiverFiber: Created ReceiverFiber object.
      """
      fiber = super().from_dict(data)
      fiber.L_gauge = data["L_gauge"]
      fiber.n_gauge = data["n_gauge"]
      fiber.radius  = data.get("radius")
      fiber.pitch   = data.get("pitch")
      return fiber


   def __str__(self) -> str:
      """Convert ReceiverFiber to string representation.
      
      Returns:
         str: String representation of ReceiverFiber.
      """
      out = super().__str__()
      out = out[:-4]  # Remove closing "[]" to add more attributes
      out += f"         L_gauge = {self.L_gauge}\n"
      out += f"         n_gauge = {self.n_gauge}\n"
      if self.radius is not None:
         out += f"         radius = {self.radius}\n"
      if self.pitch is not None:
         out += f"         pitch = {self.pitch}\n"
      out += "      []\n"
      out += "   []\n"
      return out



# ----------------------------------------------------------------------
# Receiver Coordinates
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class ReceiverCoordinates(ABC):
   """Base class for receiver coordinates.
   
   Provides interface for different ways of specifying receiver locations.
   
   Attributes:
      name (str): Identifier for this set of coordinates.
   """
   name: str

   @abstractmethod
   def size(self) -> int:
      """Get the total number of receivers.
      
      Returns:
         int: Number of receivers.
      """
      pass

   @abstractmethod
   def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
      """Get coordinate bounds without loading full dataset.
      
      Returns:
         Tuple[np.ndarray, np.ndarray]: Min and max coordinates.
      """
      pass

   @abstractmethod
   def get(self, indices) -> np.ndarray:
      """Get coordinates for specified indices.
      
      Args:
         indices: Integer indices or boolean mask.
         
      Returns:
         np.ndarray: Coordinate array for requested receivers.
      """
      pass

   @abstractmethod
   def to_dict(self) -> Dict:
      """Convert coordinates to dictionary representation."""
      pass

   @classmethod
   @abstractmethod
   def from_dict(cls, data: Dict) -> 'ReceiverCoordinates':
      """Create coordinates from dictionary representation."""
      pass


@dataclass(kw_only=True)
class FileCoordinates(ReceiverCoordinates):
   """Receiver coordinates stored in a file.
   
   Attributes:
      path (Union[str, Path]):   Path to coordinate file.
      format (str):              File format ('hdf5', 'asdf', or 'segy').
   """
   path: Union[str, Path]
   format: Literal["hdf5", "asdf", "segy"]


   def size(self) -> int:
      """Get the total number of receivers.

      Returns:
         int: Number of receivers.
      """
      if self.format == "hdf5":
         with h5py.File(self.path, 'r') as f:
            return f['coordinates'].shape[0]
      elif self.format == "segy":
         import segyio
         with segyio.open(self.path, 'r', strict=False) as f:
            return len(f.trace)
      elif self.format == "asdf":
         import pyasdf
         with pyasdf.ASDFDataSet(self.path, mode='r') as ds:
            return len(ds.coordinates)
      else:
         raise NotImplementedError(f"Format {self.format} not implemented")


   def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
      """Get coordinate bounds without loading full dataset.
      
      Returns:
         Tuple[np.ndarray, np.ndarray]: Min and max coordinates.
      """
      if self.format == "hdf5":
         with h5py.File(self.path, 'r') as f:
            coords = f['coordinates']
            return np.min(coords, axis=0), np.max(coords, axis=0)
      elif self.format == "segy":
         import segyio
         with segyio.open(self.path, 'r', strict=False) as f:
            return np.min(f.trace.x, axis=0), np.max(f.trace.x, axis=0)
      elif self.format == "asdf":
         import pyasdf
         with pyasdf.ASDFDataSet(self.path, mode='r') as ds:
            return np.min(ds.coordinates, axis=0), np.max(ds.coordinates, axis=0)
      else:
         raise NotImplementedError(f"Format {self.format} not implemented")


   def get(self, indices) -> np.ndarray:
      """Get coordinates for specified indices.
      
      Args:
         indices: Integer indices or boolean mask.
         
      Returns:
         np.ndarray: Coordinate array for requested receivers.
      """
      if self.format == "hdf5":
         with h5py.File(self.path, 'r') as f:
            return f['coordinates'][indices]
      elif self.format == "segy":
         import segyio
         with segyio.open(self.path, 'r', strict=False) as f:
            # Get coordinates from trace headers
            coords = np.zeros((len(indices), 3))
            for i, idx in enumerate(indices):
               # Get source/receiver coordinates in ft or m
               coords[i,0] = f.header[idx][segyio.TraceField.SourceX]
               coords[i,1] = f.header[idx][segyio.TraceField.SourceY]
               # coords[i,2] = f.header[idx][segyio.TraceField.SourceZ]

# TODO: get receiver depth from SEGY header
# TODO: add unit system and convert to project units

            # Convert to km if in m, or kft if in ft
            if f.header[0][segyio.TraceField.CoordinateUnits] == 1: # m
               coords /= 1000.0  # Convert m to km
            else: # ft
               coords /= 1000.0  # Convert ft to kft
               
            return coords
      elif self.format == "asdf":
         import pyasdf
         with pyasdf.ASDFDataSet(self.path, mode='r') as ds:
            return ds.coordinates[indices]
      else:
         raise NotImplementedError(f"Format {self.format} not implemented")

   def to_dict(self) -> Dict:
      return {
         "kind": "file",
         "name": self.name,
         "path": str(self.path),
         "format": self.format
      }

   @classmethod
   def from_dict(cls, data: Dict) -> 'FileCoordinates':
      return cls(
         name=data["name"],
         path=data["path"],
         format=data["format"]
      )


@dataclass(kw_only=True)
class GridCoordinates(ReceiverCoordinates):
   """Receiver coordinates defined by a Cartesian grid.
   
   Attributes:
      grid (CartesianGrid): Grid defining receiver locations.
   """
   grid: CartesianGrid

   def size(self) -> int:
      return self.grid.nx * self.grid.ny * self.grid.nz

   def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
      return (np.array([self.grid.x0, self.grid.y0, self.grid.z0]),
              np.array([self.grid.x1, self.grid.y1, self.grid.z1]))

   def get(self, indices: Optional[Union[int, slice, List[int], List[slice]]] = None) -> np.ndarray:
      """Get coordinates for specified indices.
      
      Args:
         indices: Can be:
            - None: Return all coordinates
            - int: Single flat index into the coordinate array
            - slice: Slice of flat indices
            - List[int]: List of flat indices
            - List[slice]: Tensor indices directly into the grid dimensions
            
      Returns:
         np.ndarray: Array of coordinates for requested indices
      """
      # Return all coordinates if indices is None
      if indices is None:
         return self.grid.get_coords()
         
      # List[slice] case - pass directly to grid
      elif isinstance(indices, list) and isinstance(indices[0], slice):
         return self.grid.get_coords(indices)
      
      # List[int] case - convert each index to tensor indices
      elif isinstance(indices, list) and isinstance(indices[0], int):
         coords = []
         for idx in indices:
            tensor_indices = []
            remaining = idx
            for n in reversed(self.grid.n):
               tensor_indices.insert(0, slice(remaining % n, (remaining % n) + 1))
               remaining //= n
            coords.append(self.grid.get_coords(tensor_indices)[0])
         return np.array(coords)
      
      # Single int case - convert to tensor indices
      elif isinstance(indices, int):
         tensor_indices = []
         remaining = indices
         for n in reversed(self.grid.n):
            tensor_indices.insert(0, slice(remaining % n, (remaining % n) + 1))
            remaining //= n
         return self.grid.get_coords(tensor_indices)
      
      # For slice, get all coords and then slice
      elif isinstance(indices, list) and isinstance(indices[0], int) or isinstance(indices, slice):
         coords = self.grid.get_coords()
         return coords[indices]
      else:
         raise ValueError("Invalid indices type")

   def to_dict(self) -> Dict:
      return {
         "kind": "grid", 
         "name": self.name,
         "grid": self.grid.to_dict()
      }
   
   @classmethod
   def from_dict(cls, data: Dict) -> 'GridCoordinates':
      return cls(
         name=data["name"],
         grid=CartesianGrid.from_dict(data["grid"])
      )


@dataclass(kw_only=True)
class ArrayCoordinates(ReceiverCoordinates):
   """Receiver coordinates stored as an xarray/numpy array.
   
   Attributes:
      coords (Union[xr.DataArray, np.ndarray]): Coordinate array.
      output_path (Optional[Union[str, Path]]): Path to save coordinates.
   """
   coords: Union[xr.DataArray, np.ndarray]
   output_path: Optional[Union[str, Path]] = None

   def __post_init__(self):
      if isinstance(self.coords, np.ndarray):
         self.coords = xr.DataArray(self.coords, 
                                  dims=['receiver', 'coordinate'],
                                  coords={'coordinate': ['x', 'y', 'z']})

   def size(self) -> int:
      return len(self.coords)

   def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
      return (self.coords.min(dim='receiver').values,
              self.coords.max(dim='receiver').values)

   def slice(self, indices) -> np.ndarray:
      return self.coords[indices].values

   def to_file(self) -> Dict:
      """Write coordinates to HDF5 file and return FileCoordinates dict.
      
      Returns:
         Dict: Dictionary compatible with FileCoordinates.
      """
      if self.output_path is None:
         raise ValueError("output_path must be set to write coordinates")
         
      with h5py.File(self.output_path, 'w') as f:
         f.create_dataset('coordinates', data=self.coords.values)
         
      return {
         "name": self.name,
         "path": self.output_path,
         "format": "hdf5"
      }

   def to_dict(self) -> Dict:
      if self.output_path:
         return self.to_file()
      return {
         "kind": "array",
         "name": self.name, 
         "coords": self.coords.values.tolist()
      }
   
   @classmethod
   def from_dict(cls, data: Dict) -> 'ArrayCoordinates':
      coords = np.array(data["coords"])
      return cls(
         name=data["name"],
         coords=coords,
         output_path=data.get("output_path")
      )



# ----------------------------------------------------------------------
# Receiver Groups
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class ReceiverGroup:
   """A group of multi-component receivers with shared output settings.
   
   This class represents a collection of receivers that measure one or more physical 
   quantities. All receivers in the group share output settings and their data will be 
   written to the same output file.
   
   Attributes:
      name (str):                             String identifier for this receiver group.
      device (ReceiverDevice):                Device defining receiver type and components.
      frame (str):                            Coordinate frame for measurements ("physical" or "reference").
      coordinates (ReceiverCoordinates):      Coordinates defining receiver locations.
      signals (Optional[SignalFromFile]): Optional signals for adjoint calculations.
   """
   name:             str = field(default="")
   device:           ReceiverDevice = field(default_factory=ReceiverDevice)
   frame:            Literal["physical", "reference"] = "physical"
   coordinates:      ReceiverCoordinates = field(default_factory=GridCoordinates)
   signals:          Optional[SignalFromFile] = None
   

   def signal(self, irecv: int) -> Wavelet:
      """Retrieves the signal for a specific receiver.
      
      Used in adjoint calculations where receivers act as sources.
      
      Args:
         irecv (int): 1-based index of the receiver.
         
      Returns:
         Wavelet: The signal associated with the specified receiver.
      """
      return self.signals.get(irecv)
      

   @property
   def size(self):
      return self.coordinates.size()


   # TODO: option to correct signature for device response
   # TODO: method to define receviers
   # TODO: method to attach signals


   def to_dict(self) -> Dict:
      return {
         "name": self.name,
         "device": self.device.to_dict(),
         "frame": self.frame,
         "coordinates": self.coordinates.to_dict(),
         **({"signals": self.signals.to_dict()} if self.signals else {})
      }


   @classmethod
   def from_dict(cls, data: Dict) -> 'ReceiverGroup':
      # Create coordinates based on kind
      coord_data = data["coordinates"]
      if coord_data["kind"] == "file":
         coordinates = FileCoordinates.from_dict(coord_data)
      elif coord_data["kind"] == "grid":
         coordinates = GridCoordinates.from_dict(coord_data)  
      elif coord_data["kind"] == "array":
         coordinates = ArrayCoordinates.from_dict(coord_data)
      else:
         raise ValueError(f"Unknown coordinates kind: {coord_data['kind']}")
         
      return cls(
         name        = data["name"],
         device      = ReceiverDevice.from_dict(data["device"]),
         frame       = data["frame"],
         coordinates = coordinates,
         signals   = SignalFromFile.from_dict(data["signals"]) if "signals" in data else None
      )
   
