"""Receiver definitions and coordinate systems.

This module defines the various types of receivers and their locations.
"""

import numpy as np
import h5py
import xarray as xr

from pathlib      import Path
from dataclasses  import dataclass, field
from typing       import Optional, List, Literal, Union, Tuple, Dict, Any
from abc          import ABC, abstractmethod

from ..geometry.grids   import *  # noqa
from .signals           import *  # noqa
from .wavelet           import *  # noqa

__all__ = ['ReceiverComponent', 'ReceiverGroup', 'ReceiverCoordinates', 'ReceiverDevice',
           'ReceiverNodeArray', 'ReceiverNode', 'ReceiverFiber']

@dataclass
class ReceiverComponent:
   """Defines a single component/measurement type for a receiver.
   
   A receiver component specifies what physical quantity is being measured
   (e.g., pressure, velocity) and in what direction for vector quantities.
   
   Attributes:
      name (str): String identifier for this receiver component.
      field (str): Physical field being measured
      direction (Optional[List[float]]): Measurement direction for vector fields.
   """
   
   name:      str
   field:     Literal["pressure", "velocity", "displacement", "stress", "strain"]
   direction: Optional[List[float]] = None

   def to_dict(self) -> dict:
      return {
         "name": self.name,
         "field": self.field,
         **({"direction": self.direction} if self.direction else {})
      }
   
   @classmethod
   def from_dict(cls, data: dict) -> 'ReceiverComponent':
      return cls(name      = data["name"], 
                 field     = data["field"], 
                 direction = data.get("direction"))


# ----------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------
@dataclass
class ReceiverDevice:
   """Defines a single receiver.
   
   This class represents a single receiver, which is a point in space where measurements are taken.
   
   Attributes:
      name (str): String identifier for this receiver.
      components (List[ReceiverComponent]): List of components defining measurements.
      response (Optional[Wavelet]): Wavelet response of the receiver.
   """
   name:       str
   components: List[ReceiverComponent] = field(default_factory=list)
   response:   Optional[Wavelet] = None

   def add_component(self, name: str, field: str, direction: Optional[List[float]] = None) -> 'ReceiverComponent':
      component = ReceiverComponent(name=name, field=field, direction=direction)
      self.components.append(component)
      return component

   def to_dict(self) -> dict:
      return {
         "name": self.name,
         "components": [c.to_dict() for c in self.components],
         **({"response": self.response.to_dict()} if self.response else {})
      }

   @classmethod
   def from_dict(cls, data: dict) -> 'ReceiverDevice':
      device = cls(
         name=data["name"],
         components=[ReceiverComponent.from_dict(c) for c in data["components"]],
         response=data.get("response")
      )
      return device


@dataclass(kw_only=True)
class ReceiverFiber(ReceiverDevice):
   """Defines a fiber receiver."""
   L_gauge: float
   n_gauge: int
   radius:  Optional[float] = None
   pitch:   Optional[float] = None

   def to_dict(self) -> dict:
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
      fiber = super().from_dict(data)
      fiber.L_gauge = data["L_gauge"]
      fiber.n_gauge = data["n_gauge"]
      fiber.radius  = data.get("radius")
      fiber.pitch   = data.get("pitch")
      return fiber



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
      return {
         **super().to_dict(),
         "offsets": self.offsets
      }

   @classmethod 
   def from_dict(cls, data: dict) -> 'ReceiverNodeArray':
      node_array = super().from_dict(data)
      node_array.offsets = data["offsets"]
      return node_array



@dataclass(kw_only=True)
class ReceiverNode(ReceiverDevice):
   """Defines a node receiver."""

   def to_dict(self) -> dict:
      return super().to_dict()

   @classmethod
   def from_dict(cls, data: dict) -> 'ReceiverNode':
      node = super().from_dict(data)
      return node



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
      format (str):              File format ('HDF5', 'asdf', or 'SEGY').
   """
   path:   Union[str, Path]
   format: Literal["HDF5", "asdf", "SEGY"]

   def size(self) -> int:
      """Get the total number of receivers.

      Returns:
         int: Number of receivers.
      """
      if self.format == "HDF5":
         with h5py.File(self.path, 'r') as f:
            return f['coordinates'].shape[0]
      elif self.format == "SEGY":
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
      if self.format == "HDF5":
         with h5py.File(self.path, 'r') as f:
            coords = f['coordinates']
            return np.min(coords, axis=0), np.max(coords, axis=0)
      elif self.format == "SEGY":
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
      if self.format == "HDF5":
         with h5py.File(self.path, 'r') as f:
            return f['coordinates'][indices]
      elif self.format == "SEGY":
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
         "path": str(self.path),
         "format": self.format
      }

   @classmethod
   def from_dict(cls, data: Dict) -> 'FileCoordinates':
      return cls(
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
         "grid": self.grid.to_dict()
      }
   
   @classmethod
   def from_dict(cls, data: Dict) -> 'GridCoordinates':
      return cls(
         grid=CartesianGrid.from_dict(data["grid"])
      )


@dataclass(kw_only=True)
class ArrayCoordinates(ReceiverCoordinates):
   """Receiver coordinates stored as an xarray/numpy array.
   
   Attributes:
      coords (Union[xr.DataArray, np.ndarray]): Coordinate array.
      output_path (Optional[Union[str, Path]]): Path to save coordinates.
   """
   coordinates: Union[xr.DataArray, np.ndarray]

   def __post_init__(self):
      if isinstance(self.coordinates, np.ndarray):
         if self.coordinates.ndim != 2:
            raise ValueError("Coordinates array must be 2D with shape (n_receivers, n_coordinates)")
         if self.coordinates.shape[1] == 2:
            self.coordinates = xr.DataArray(self.coordinates, 
                                    dims=['receiver', 'coordinate'],
                                    coords={'coordinate': ['x', 'z']})
         elif self.coordinates.shape[1] == 3:
            self.coordinates = xr.DataArray(self.coordinates, 
                                    dims=['receiver', 'coordinate'],
                                    coords={'coordinate': ['x', 'y', 'z']})
         else:
            raise ValueError("Coordinates array must have 2 or 3 columns")

   def size(self) -> int:
      return len(self.coordinates)


   def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
      return (self.coordinates.min(dim='receiver').values,
              self.coordinates.max(dim='receiver').values)


   def slice(self, indices) -> np.ndarray:
      return self.coordinates[indices].values


   def get(self, indices: int) -> np.ndarray:
      return self.coordinates[indices].values


   def to_file(self, 
               path: Union[str, Path], 
               format: Optional[Literal["HDF5", "asdf", "SEGY"]] = None) -> FileCoordinates:
      """Write coordinates to file and return FileCoordinates object.
      
      Returns:
         FileCoordinates: FileCoordinates object.
      """
      if format is None:
         if path.endswith(".h5") or path.endswith(".hdf5"):
            format = "HDF5"
         elif path.endswith(".segy"):
            format = "SEGY"
         elif path.endswith(".asdf"):
            format = "asdf"
         else:
            raise ValueError(f"Unknown coordinates file extension: {path}")

      if format == "HDF5":
         with h5py.File(path, 'w') as f:
            f.create_dataset('coordinates', data=self.coordinates.values)
      elif format == "SEGY":
         raise NotImplementedError("SEGY format not implemented")
      elif format == "asdf":
         raise NotImplementedError("asdf format not implemented")
         
      return FileCoordinates(path=path, format=format)


   def to_dict(self) -> Dict:
      return {
         "kind": "array",
         "coords": self.coordinates.values.tolist()
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
   name:         str = field(default="")
   device:       ReceiverDevice = field(default_factory=ReceiverDevice)
   frame:        Literal["physical", "reference"] = "physical"
   coordinates:  ReceiverCoordinates = field(default_factory=ReceiverCoordinates)
   signals:      Optional[SignalFromFile] = None
   

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

   def __init__(self, 
                name: str, 
                device: ReceiverDevice, 
                coordinates: Union[np.ndarray, xr.DataArray, str, Grid], 
                frame: str = "physical") -> None:
      self.name = name
      self.device = device
      if isinstance(coordinates, list):
         coordinates = np.array(coordinates)
      if isinstance(coordinates, np.ndarray) or \
         isinstance(coordinates, xr.DataArray):
         self.coordinates = ArrayCoordinates(coordinates=coordinates)
      elif isinstance(coordinates, str):
         if coordinates.endswith(".h5") or coordinates.endswith(".hdf5"):
            self.coordinates = FileCoordinates(path=coordinates, format="HDF5")
         elif coordinates.endswith(".segy"):
            self.coordinates = FileCoordinates(path=coordinates, format="SEGY")
         elif coordinates.endswith(".asdf"):
            self.coordinates = FileCoordinates(path=coordinates, format="asdf")
         else:
            raise ValueError(f"Unknown coordinates file extension: {coordinates}")
      elif isinstance(coordinates, Grid):
         self.coordinates = GridCoordinates(grid=coordinates)
      else:
         raise ValueError(f"Unknown coordinates type: {type(coordinates)}")
      self.frame = frame


   def to_dict(self) -> Dict:

      if isinstance(self.coordinates, ArrayCoordinates):
         if self.coordinates.size() > 10:
            self.coordinates = self.coordinates.to_file("./coordinates.h5", "HDF5")

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
   
