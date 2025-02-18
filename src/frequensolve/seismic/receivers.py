"""Receiver definitions and coordinate systems.

This module defines the various types of receivers and their locations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import xarray as xr

from frequensolve.geometry.grids import CartesianGrid, Grid
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.util.class_registry import class_registry, register_class

__all__ = [
    "ReceiverComponent",
    "ReceiverGroup",
    "ReceiverCoords",
    "ReceiverDevice",
    "ReceiverNodeArray",
    "ReceiverNode",
    "ReceiverFiber",
]


@dataclass(kw_only=True)
class ReceiverComponent:
    """Defines a single component/measurement type for a receiver.

    A receiver component specifies what physical quantity is being measured
    (e.g., pressure, velocity) and in what direction for vector quantities.

    Attributes:
       name (str): String identifier for this receiver component.
       field (str): Physical field being measured
       direction (Optional[List[float]]): Measurement direction for vector fields.
    """

    name: str = "name"
    field: Literal["pressure", "velocity", "displacement", "stress", "strain"]
    direction: Optional[List[float]] = None

    def __dict__(self) -> dict:
        return {
            "name": self.name,
            "field": self.field,
            **({"direction": self.direction} if self.direction is not None else {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReceiverComponent":
        return cls(
            name=data["name"], field=data["field"], direction=data.get("direction")
        )


# ----------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------
@register_class
@dataclass(kw_only=True)
class ReceiverDevice:
    """Defines a single receiver.

    This class represents a single receiver, which is a point in space where measurements are taken.

    Attributes:
       name (str): String identifier for this receiver.
       components (List[ReceiverComponent]): List of components defining measurements.
       response (Optional[Wavelet]): Wavelet response of the receiver.
    """

    name: str
    components: List[ReceiverComponent] = field(default_factory=list)
    response: Optional[Wavelet] = None

    def add_component(
        self, name: str, field: str, direction: Optional[List[float]] = None
    ) -> "ReceiverComponent":
        component = ReceiverComponent(name=name, field=field, direction=direction)
        self.components.append(component)
        return component

    def __dict__(self) -> dict:
        return {
            "name": self.name,
            "components": [c.__dict__() for c in self.components],
            **(
                {"response": self.response.__dict__()}
                if self.response is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReceiverDevice":
        class_name = data["_type"]
        if class_name in class_registry:
            device_class = class_registry[class_name]
            return device_class.from_dict(data)
        else:
            raise ValueError(f"Unknown receiver device class: {class_name}")


@register_class
@dataclass(kw_only=True)
class ReceiverFiber(ReceiverDevice):
    """Defines a fiber receiver."""

    L_gauge: float
    n_gauge: int
    radius: Optional[float] = None
    pitch: Optional[float] = None

    def __dict__(self) -> dict:
        return {
            "_type": self.__class__.__name__,
            **super().__dict__(),
            "L_gauge": self.L_gauge,
            "n_gauge": self.n_gauge,
            **(
                {"pitch": self.pitch, "radius": self.radius}
                if self.pitch is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReceiverFiber":
        return cls(
            name=data["name"],
            components=[ReceiverComponent.from_dict(c) for c in data["components"]],
            response=data.get("response"),
            L_gauge=data["L_gauge"],
            n_gauge=data["n_gauge"],
            radius=data.get("radius"),
            pitch=data.get("pitch"),
        )


@register_class
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

    def __dict__(self) -> dict:
        return {
            "_type": self.__class__.__name__,
            **super().__dict__(),
            "offsets": self.offsets,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReceiverNodeArray":
        return cls(
            name=data["name"],
            components=[ReceiverComponent.from_dict(c) for c in data["components"]],
            response=data.get("response"),
            offsets=data["offsets"],
        )


@register_class
@dataclass(kw_only=True)
class ReceiverNode(ReceiverDevice):
    """Defines a node receiver."""

    def __dict__(self) -> dict:
        return {"_type": self.__class__.__name__, **super().__dict__()}

    @classmethod
    def from_dict(cls, data: dict) -> "ReceiverNode":
        return cls(
            name=data["name"],
            components=[ReceiverComponent.from_dict(c) for c in data["components"]],
            response=data.get("response"),
        )


# ----------------------------------------------------------------------
# Receiver Coordinates
# ----------------------------------------------------------------------
@register_class
@dataclass(kw_only=True)
class ReceiverCoords(ABC):
    """Base class for receiver coordinates.

    Enables different ways of specifying receiver locations.

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
    def __dict__(self) -> Dict:
        """Convert coordinates to dictionary representation."""
        pass

    @classmethod
    def from_dict(cls, data: Dict) -> "ReceiverCoords":
        class_name = data["_type"]
        if class_name in class_registry:
            coord_class = class_registry[class_name]
            return coord_class.from_dict(data)
        else:
            raise ValueError(f"Unknown receiver coordinates class: {class_name}")

    def _set_path(self, proj_path: Path, rel_path: Path):
        try:
            self.path = proj_path / self.path.relative_to(self._proj_path)
        except Exception as e:
            pass

        self._proj_path = proj_path
        self._rel_path = rel_path

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path


@register_class
@dataclass(kw_only=True)
class CoordsFromFile(ReceiverCoords):
    """Receiver coordinates stored in a file.

    Attributes:
       path (Union[str, Path]):   Path to coordinate file.
       format (str):              File format ('HDF5').
       dset (str):                Dataset name in file.
    """

    path: Path
    format: Literal["HDF5"]
    dset: Optional[str] = None
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __init__(
        self,
        path: Union[str, Path],
        format: Literal["HDF5"],
        dset: Optional[str] = None,
        **kwargs,
    ):
        self.path = Path(path).resolve()
        self.format = format
        self.dset = dset

    @classmethod
    def from_dict(cls, data: Dict) -> "CoordsFromFile":
        path = data["path"]
        format = data["format"]
        if format == "HDF5":
            if ":" in path:
                path, dset = path.split(":")
            else:
                dset = "coords"
        return cls(path=Path(path), format=format, dset=dset)

    @property
    def size(self) -> int:
        """Get the total number of receivers.

        Returns:
           int: Number of receivers.
        """
        if self.format == "HDF5":
            with h5py.File(self.path, "r") as f:
                return f[self.dset].shape[0]
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinate bounds without loading full dataset.

        Returns:
           Tuple[np.ndarray, np.ndarray]: Min and max coordinates.
        """
        if self.format == "HDF5":
            with h5py.File(self.path, "r") as f:
                coords = f[self.dset]
                return np.min(coords, axis=0), np.max(coords, axis=0)
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    def get(self, indices: Optional[Union[int, slice]] = None) -> np.ndarray:
        """Get coordinates for specified indices.

        Args:
           indices: Integer indices or boolean mask.

        Returns:
           np.ndarray: Coordinate array for requested receivers.
        """
        if self.format == "HDF5":
            with h5py.File(self.path, "r") as f:
                if indices is None:
                    return f[self.dset][:]
                else:
                    return f[self.dset][indices]
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    def __dict__(self) -> Dict:
        rel_path = self.path.relative_to(self._proj_path)

        if self.format == "HDF5":
            if self.dset is None:
                path = str(rel_path.with_suffix(".h5")) + ":coords"
            else:
                path = str(rel_path.with_suffix(".h5")) + ":" + self.dset
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

        return {"_type": self.__class__.__name__, "path": path, "format": self.format}


@register_class
@dataclass(kw_only=True)
class CoordsGrid(ReceiverCoords):
    """Receiver coordinates defined by a Cartesian grid.

    Attributes:
       grid (CartesianGrid): Grid defining receiver locations.
    """

    grid: CartesianGrid

    @property
    def size(self) -> int:
        return self.grid.nx * self.grid.ny * self.grid.nz

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.grid.x0, self.grid.x1

    def get(
        self, indices: Optional[Union[int, slice, List[int], List[slice]]] = None
    ) -> np.ndarray:
        """Get coordinates for specified indices.

        Args:
           indices: Can be:
              - None:        Return all coordinates
              - int:         Single flat index into the coordinate array
              - slice:       Slice of flat indices
              - List[int]:   List of flat indices
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
        elif (
            isinstance(indices, list)
            and isinstance(indices[0], int)
            or isinstance(indices, slice)
        ):
            coords = self.grid.get_coords()
            return coords[indices]
        else:
            raise ValueError("Invalid indices type")

    def __dict__(self) -> Dict:
        return {"_type": self.__class__.__name__, "grid": self.grid.__dict__()}

    @classmethod
    def from_dict(cls, data: Dict) -> "CoordsGrid":
        return cls(grid=CartesianGrid.from_dict(data["grid"]))


@register_class
@dataclass(kw_only=True)
class CoordsArray(ReceiverCoords):
    """Receiver coordinates stored as an xarray/numpy array.

    Attributes:
       coords (Union[xr.DataArray, np.ndarray]): Coordinate array.
       output_path (Optional[Union[str, Path]]): Path to save coordinates.
    """

    coordinates: Union[xr.DataArray, np.ndarray]

    def __post_init__(self):
        if isinstance(self.coordinates, np.ndarray):
            if self.coordinates.ndim != 2:
                raise ValueError(
                    "Coordinates array must be 2D with shape (n_receivers, n_coordinates)"
                )
            if self.coordinates.shape[1] == 2:
                self.coordinates = xr.DataArray(
                    self.coordinates,
                    dims=["receiver", "coordinate"],
                    coords={"coordinate": ["x", "z"]},
                )
            elif self.coordinates.shape[1] == 3:
                self.coordinates = xr.DataArray(
                    self.coordinates,
                    dims=["receiver", "coordinate"],
                    coords={"coordinate": ["x", "y", "z"]},
                )
            else:
                raise ValueError("Coordinates array must have 2 or 3 columns")

    @property
    def size(self) -> int:
        return len(self.coordinates)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.coordinates.min(dim="receiver").values,
            self.coordinates.max(dim="receiver").values,
        )

    def get(self, indices: Optional[Union[int, slice]] = None) -> np.ndarray:
        if indices is None:
            return self.coordinates.values
        else:
            return self.coordinates[indices].values

    def to_file(
        self, file_name: Union[str, Path], format: Optional[Literal["HDF5"]] = None
    ) -> CoordsFromFile:
        """Write coordinates to file and return CoordsFromFile object.

        Returns:
           CoordsFromFile: CoordsFromFile object.
        """
        if format is None:
            if file_name.endswith(".h5") or file_name.endswith(".hdf5"):
                format = "HDF5"
            else:
                raise ValueError(f"Unknown coordinates file extension: {file_name}")

        path = self._path / file_name
        if not path.parent.exists():
            path.parent.mkdir(parents=True)

        if format == "HDF5":
            with h5py.File(path, "w") as f:
                dset = f.create_dataset(
                    "coords", data=(self.coordinates.values).astype(np.float64)
                )
        else:
            raise NotImplementedError(f"Format {format} not implemented")

        return CoordsFromFile(path=path, dset="coords", format=format)

    def __dict__(self) -> Dict:
        return {
            "_type": self.__class__.__name__,
            "coords": self.coordinates.values.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CoordsArray":
        coords = np.array(data["coords"])
        return cls(
            name=data["name"], coords=coords, output_path=data.get("output_path")
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
       name (str):                   String identifier for this receiver group.
       device (ReceiverDevice):      Device defining receiver type and components.
       frame (str):                  Coordinate frame for measurements ("physical" or "reference").
       coordinates (ReceiverCoords): Coordinates defining receiver locations.
    """

    name: str = "group"
    device: ReceiverDevice = field(default_factory=ReceiverDevice)
    frame: Literal["physical", "reference"] = "physical"
    coordinates: ReceiverCoords = field(default_factory=ReceiverCoords)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    @property
    def size(self):
        return self.coordinates.size

    # TODO: option to correct signature for device response
    # TODO: method to define receviers

    def __init__(
        self,
        name: str,
        device: ReceiverDevice,
        coordinates: Union[np.ndarray, xr.DataArray, str, Path, Grid, ReceiverCoords],
        frame: str = "physical",
    ) -> None:
        coords = self._clean_coordinates(coordinates)
        self.name = name
        self.device = device
        self.frame = frame
        self.coordinates = coords

    @staticmethod
    def _clean_coordinates(coords):
        if isinstance(coords, list):
            coords = np.array(coords)

        # Allow coordinates to be defined either as a ReceiverCoords object
        # various other reasonble ways:
        if isinstance(coords, ReceiverCoords):
            return coords
        elif isinstance(coords, np.ndarray) or isinstance(coords, xr.DataArray):
            out = CoordsArray(coordinates=coords)
        elif isinstance(coords, str) or isinstance(coords, Path):
            if coords.endswith(".h5") or coords.endswith(".hdf5"):
                out = CoordsFromFile(path=coords, format="HDF5")
            else:
                raise ValueError(f"Unknown coordinates file extension: {coords}")
        elif isinstance(coords, Grid):
            out = CoordsGrid(grid=coords)
        else:
            raise ValueError(f"Unknown coordinates type: {type(coords)}")
        return out

    def __dict__(self) -> Dict:
        coords = self.coordinates
        if isinstance(coords, CoordsArray):
            if coords.size > 10:
                dump = coords.to_file(file_name="coords.h5", format="HDF5")
                dump._set_path(
                    proj_path=self._proj_path, rel_path=self._rel_path / self.name
                )
                self.coordinates = dump

        return {
            "name": self.name,
            "device": self.device.__dict__(),
            "frame": self.frame,
            "coordinates": self.coordinates.__dict__(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ReceiverGroup":
        coords = ReceiverCoords.from_dict(data["coordinates"])

        return cls(
            name=data["name"],
            device=ReceiverDevice.from_dict(data["device"]),
            frame=data["frame"],
            coordinates=coords,
        )

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path / self.name
        if isinstance(self.coordinates, ReceiverCoords):
            self.coordinates._set_path(
                proj_path=proj_path, rel_path=rel_path / self.name
            )

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path


# class ReceiverPlotter:
#    """Class for plotting receiver groups and devices."""

#    def plot_group(self,
#                   group: ReceiverGroup,
#                   ax: Optional[plt.Axes] = None,
#                   projection: Optional[Literal['2d', '3d']] = None,
#                   **kwargs) -> plt.Axes:
#       """Plot receiver locations for a single group.

#       Args:
#          group: ReceiverGroup to plot.
#          ax: Matplotlib axes to plot on. If None, a new figure will be created.
#          projection: '2d' or '3d'. If None, will be inferred from coordinates.
#          **kwargs: Additional keyword arguments to pass to the scatter function.

#       Returns:
#          Matplotlib axes containing the plot.
#       """
#       if ax is None:
#          fig = plt.figure()
#          if projection is None:
#             projection = '3d' if group.coordinates.coordinates.shape[1] == 3 else '2d'
#          if projection == '2d':
#             ax = fig.add_subplot()
#          else:
#             ax = fig.add_subplot(projection='3d')

#       coords = group.coordinates.coordinates
#       if isinstance(coords, xr.DataArray):
#          coords = coords.values

#       if coords.shape[1] == 2:
#          ax.scatter(coords[:,0], coords[:,1], **kwargs)
#          ax.set_xlabel('X')
#          ax.set_ylabel('Z')
#       else:
#          ax.scatter(coords[:,0], coords[:,1], coords[:,2], **kwargs)
#          ax.set_xlabel('X')
#          ax.set_ylabel('Y')
#          ax.set_zlabel('Z')

#       ax.set_title(f"Receiver Group: {group.name}")

#       return ax

#    def plot_groups(self,
#                    groups: List[ReceiverGroup],
#                    ax: Optional[plt.Axes] = None,
#                    projection: Optional[Literal['2d', '3d']] = None,
#                    **kwargs) -> plt.Axes:
#       """Plot receiver locations for multiple groups.

#       Args:
#          groups: List of ReceiverGroups to plot.
#          ax: Matplotlib axes to plot on. If None, a new figure will be created.
#          projection: '2d' or '3d'. If None, will be inferred from coordinates.
#          **kwargs: Additional keyword arguments to pass to the scatter function.

#       Returns:
#          Matplotlib axes containing the plot.
#       """
#       if ax is None:
#          fig = plt.figure()
#          if projection is None:
#             projection = '3d' if groups[0].coordinates.coordinates.shape[1] == 3 else '2d'
#          if projection == '2d':
#             ax = fig.add_subplot()
#          else:
#             ax = fig.add_subplot(projection='3d')

#       for group in groups:
#          self.plot_group(group, ax=ax, **kwargs)

#       ax.set_title("Receiver Groups")

#       return ax

#    def plot_device(self,
#                    device: ReceiverDevice,
#                    ax: Optional[plt.Axes] = None,
#                    **kwargs) -> plt.Axes:
#       """Plot a schematic of a receiver device.

#       Args:
#          device: ReceiverDevice to plot.
#          ax: Matplotlib axes to plot on. If None, a new figure will be created.
#          **kwargs: Additional keyword arguments to pass to the plotting functions.

#       Returns:
#          Matplotlib axes containing the plot.
#       """
#       if ax is None:
#          fig, ax = plt.subplots()

#       if isinstance(device, ReceiverNode):
#          # Plot a point for a node receiver
#          ax.scatter(0, 0, **kwargs)
#       elif isinstance(device, ReceiverNodeArray):
#          # Plot points for each node in the array
#          for offset in device.offsets:
#             ax.scatter(offset[0], offset[1], **kwargs)
#       elif isinstance(device, ReceiverFiber):
#          # Plot a line for a fiber
#          x = [0, device.L_gauge]
#          y = [0, 0]
#          ax.plot(x, y, **kwargs)

#          # Add points for each gauge
#          dx = device.L_gauge / (device.n_gauge - 1)
#          for i in range(device.n_gauge):
#             ax.scatter(i*dx, 0, color='red', **kwargs)
#       else:
#          raise ValueError(f"Unknown receiver device type: {type(device)}")

#       # Set title and labels
#       ax.set_title(f"Receiver Device: {device.name}")
#       ax.set_xlabel('X')
#       ax.set_ylabel('Y')

#       # Set equal aspect ratio
#       ax.set_aspect('equal')

#       return ax
