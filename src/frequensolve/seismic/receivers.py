"""Receiver definitions and coordinate systems.

This module defines the various types of receivers and their locations.
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import xarray as xr

from frequensolve.geometry.frame import Direction, direction_to_fs
from frequensolve.geometry.grids import CartesianGrid, Grid
from frequensolve.seismic.sparse_survey import ReceiverSampling
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.fields import canonical_field
from frequensolve.util.mixins import merge_extra

__all__ = [
    "ReceiverComponent",
    "ReceiverGroup",
    "ReceiverCoords",
    "ReceiverDevice",
    "ReceiverNodeArray",
    "ReceiverNode",
    "ReceiverFiber",
    "ReceiverFiberOld",
    "ReceiverSampling",
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
    direction: Optional[Union[List[float], Direction]] = None
    units: Optional[str] = None
    weight: Optional[Union[float, List[float], Dict]] = None

    def __post_init__(self) -> None:
        self.field = canonical_field(self.field)

    def to_fs(self, ctx=None) -> dict:
        return {
            "name": self.name,
            "field": canonical_field(self.field),
            **(
                {"direction": direction_to_fs(self.direction)}
                if self.direction is not None
                else {}
            ),
            **({"units": self.units} if self.units is not None else {}),
            **({"weight": self.weight} if self.weight is not None else {}),
        }

    def __dict__(self) -> dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: dict) -> "ReceiverComponent":
        data = copy.deepcopy(data)
        if "direction" in data:
            data["direction"] = Direction.from_fs(data["direction"])
        return cls(
            name=data["name"],
            field=canonical_field(data["field"]),
            direction=data.get("direction"),
            units=data.get("units"),
            weight=data.get("weight"),
        )


# ----------------------------------------------------------------------
# Devices
# ----------------------------------------------------------------------
@register_class
@dataclass(kw_only=True)
class ReceiverDevice(ABC):
    """Defines an abstract base class for a receiver device with multiple measurements (components).

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
        component = ReceiverComponent(
            name=name,
            field=canonical_field(field),
            direction=direction,
        )
        self.components.append(component)
        return component

    def to_fs(self, ctx=None) -> dict:
        return {
            "name": self.name,
            "components": [c.to_fs(ctx) for c in self.components],
            **(
                {
                    "response": (
                        self.response.to_fs(ctx)
                        if hasattr(self.response, "to_fs")
                        else self.response.__dict__()
                    )
                }
                if self.response is not None
                else {}
            ),
        }

    def __dict__(self) -> dict:
        return self.to_fs()

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
    """Defines a fiber receiver device (e.g. a DAS fiber) that integrates
    response over a length.

    The response is integrated over the gauge length by averaging over the
    number of points per gauge (n_gauge).

    Attributes:
       L_gauge (float): Length of the gauge.
       n_gauge (int): Number of points per guage
       radius (Optional[float]): Radius of the fiber.
       pitch (Optional[float]): Pitch of the fiber.
    """

    L_gauge: float
    n_gauge: int
    radius: Optional[float] = None
    pitch: Optional[float] = None

    def to_fs(self, ctx=None) -> dict:
        return {
            "_type": self.__class__.__name__,
            **super().to_fs(ctx),
            "L_gauge": self.L_gauge,
            "n_gauge": self.n_gauge,
            **(
                {"pitch": self.pitch, "radius": self.radius}
                if self.pitch is not None
                else {}
            ),
        }

    def __dict__(self) -> dict:
        return self.to_fs()

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
class ReceiverFiberOld(ReceiverFiber):
    """Same as ReceiverFiber, but uses old API.

    In particular, guages are centered on coords, with specified gauge length."""

    def to_fs(self, ctx=None) -> dict:
        return {
            "_type": self.__class__.__name__,
            **super().to_fs(ctx),
        }

    def __dict__(self) -> dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: dict) -> "ReceiverFiberOld":
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

            - dx = 0.005 (5-m spacing)
            - dy = 0.010 (10-m spacing)
            - offsets = [[-dx, -dy, 0], [0, -dy, 0], [dx,-dy,0], ...
    """

    offsets: List[List[float]] = field(default_factory=list)

    def to_fs(self, ctx=None) -> dict:
        return {
            "_type": self.__class__.__name__,
            **super().to_fs(ctx),
            "offsets": self.offsets,
        }

    def __dict__(self) -> dict:
        return self.to_fs()

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

    def to_fs(self, ctx=None) -> dict:
        return {"_type": self.__class__.__name__, **super().to_fs(ctx)}

    def __dict__(self) -> dict:
        return self.to_fs()

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

    file: Path
    format: Literal["HDF5"]
    dset: Optional[str] = None
    units: Optional[str] = None
    system: Optional[str] = None
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __init__(
        self,
        file: Union[str, Path] = None,
        format: Literal["HDF5"] = "HDF5",
        dset: Optional[str] = None,
        units: Optional[str] = None,
        system: Optional[str] = None,
        **kwargs,
    ):
        if file is None and "path" in kwargs:
            file = kwargs.pop("path")
        self.file = Path(file).resolve()
        self.format = format
        self.dset = dset
        self.units = units
        self.system = system

    @classmethod
    def from_dict(cls, data: Dict) -> "CoordsFromFile":
        file = data["file"]
        format = data["format"]
        if format == "HDF5":
            if ":" in file:
                file, dset = file.split(":", 1)
            else:
                dset = "coords"
        return cls(
            file=Path(file),
            format=format,
            dset=dset,
            units=data.get("units"),
            system=data.get("system"),
        )

    @property
    def size(self) -> int:
        """Get the total number of receivers.

        Returns:
           int: Number of receivers.
        """
        if self.format == "HDF5":
            with h5py.File(self.file, "r") as f:
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
            with h5py.File(self.file, "r") as f:
                coords = f[self.dset]
                return np.min(coords, axis=0), np.max(coords, axis=0)
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    def __getitem__(self, key: Union[tuple, slice]):
        return self.get(key)

    def get(self, indices: Optional[Union[Tuple, slice]] = None) -> np.ndarray:
        """Get coordinates for specified indices.

        Args:
           indices: Integer indices or boolean mask.

        Returns:
           np.ndarray: Coordinate array for requested receivers.
        """
        if self.format == "HDF5":
            with h5py.File(self.file, "r") as f:
                if indices is None:
                    return f[self.dset][:]
                else:
                    return f[self.dset][indices]
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    def to_fs(self, ctx=None) -> Dict:
        try:
            rel_path = self.file.relative_to(self._proj_path)
        except Exception:
            rel_path = self.file

        if self.format == "HDF5":
            if self.dset is None:
                file = str(rel_path.with_suffix(".h5")) + ":coords"
            else:
                file = str(rel_path.with_suffix(".h5")) + ":" + self.dset
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

        return {
            "_type": self.__class__.__name__,
            "file": file,
            "format": self.format,
            **({"units": self.units} if self.units is not None else {}),
            **({"system": self.system} if self.system is not None else {}),
        }

    def __dict__(self) -> Dict:
        return self.to_fs()


@register_class
@dataclass(kw_only=True)
class CoordsGrid(ReceiverCoords):
    """Receiver coordinates defined by a Cartesian grid.

    Attributes:
       grid (CartesianGrid): Grid defining receiver locations.
    """

    grid: CartesianGrid
    units: Optional[str] = None
    system: Optional[str] = None

    @property
    def size(self) -> int:
        return np.prod(self.grid.n)

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

    def to_fs(self, ctx=None) -> Dict:
        payload = {"_type": self.__class__.__name__, "grid": self.grid.to_fs(ctx)}
        if self.units is not None:
            payload["units"] = self.units
        if self.system is not None:
            payload["system"] = self.system
        return payload

    def __dict__(self) -> Dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: Dict) -> "CoordsGrid":
        return cls(
            grid=CartesianGrid.from_dict(data["grid"]),
            units=data.get("units"),
            system=data.get("system"),
        )


@register_class
@dataclass(kw_only=True)
class CoordsArray(ReceiverCoords):
    """Receiver coordinates stored as an xarray/numpy array.

    Attributes:
       coords (Union[xr.DataArray, np.ndarray]): Coordinate array.
    """

    coordinates: Union[xr.DataArray, np.ndarray]
    units: Optional[str] = None
    system: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.coordinates, np.ndarray):
            if self.coordinates.ndim != 2:
                raise ValueError(
                    "Coordinates array must be 2D with shape (n_receivers, <simulation dimension>)"
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

        file = self._path / file_name
        if not file.parent.exists():
            file.parent.mkdir(parents=True)

        if format == "HDF5":
            with h5py.File(file, "w") as f:
                dset = f.create_dataset(
                    "coords", data=(self.coordinates.values).astype(np.float64)
                )
        else:
            raise NotImplementedError(f"Format {format} not implemented")

        return CoordsFromFile(
            file=file,
            dset="coords",
            format=format,
            units=self.units,
            system=self.system,
        )

    def to_fs(self, ctx=None) -> Dict:
        values = self.coordinates.values.tolist()
        payload = {"_type": self.__class__.__name__}
        if self.units is not None or self.system is not None:
            payload["value"] = values
            if self.units is not None:
                payload["units"] = self.units
            if self.system is not None:
                payload["system"] = self.system
        else:
            payload["coords"] = values
        return payload

    def __dict__(self) -> Dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: Dict) -> "CoordsArray":
        coords = np.array(data.get("coords", data.get("value")))
        return cls(
            coordinates=coords, units=data.get("units"), system=data.get("system")
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
    domain: Optional[int] = None
    coordinates: ReceiverCoords = field(default_factory=ReceiverCoords)
    sampling: Optional[ReceiverSampling] = None
    extra: Dict = field(default_factory=dict)
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
        domain: Optional[int] = None,
        sampling: Optional[Union[str, Dict, ReceiverSampling]] = None,
        survey: Optional[Union[str, ReceiverSampling]] = None,
        extra: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        coords = self._clean_coordinates(coordinates)
        sampling_obj = ReceiverSampling.from_value(sampling)
        survey_obj = ReceiverSampling.from_value(survey)
        if sampling_obj is None:
            sampling_obj = survey_obj
        elif survey_obj is not None and sampling_obj.survey is None:
            sampling_obj.survey = survey_obj.survey
        self.name = name
        self.device = device
        self.frame = frame
        self.coordinates = coords
        self.domain = domain
        self.sampling = sampling_obj
        self.extra = copy.deepcopy(dict(extra or {}))
        self.extra.update(copy.deepcopy(kwargs))

    @property
    def survey(self) -> Optional[str]:
        if self.sampling is None:
            return None
        return self.sampling.survey

    @survey.setter
    def survey(self, value: Optional[Union[str, ReceiverSampling]]) -> None:
        self.sampling = ReceiverSampling.from_value(value)

    @property
    def kwargs(self) -> Dict:
        return self.extra

    @kwargs.setter
    def kwargs(self, value: Dict) -> None:
        self.extra = copy.deepcopy(dict(value))

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
            suffix = Path(coords).suffix.lower()
            if suffix in [".h5", ".hdf5"]:
                out = CoordsFromFile(file=coords, format="HDF5")
            else:
                raise ValueError(f"Unknown coordinates file extension: {coords}")
        elif isinstance(coords, Grid):
            out = CoordsGrid(grid=coords)
        else:
            raise ValueError(f"Unknown coordinates type: {type(coords)}")
        return out

    def to_fs(self, ctx=None) -> Dict:
        coords = self.coordinates
        if isinstance(coords, CoordsArray):
            if (
                coords.size > 10
                and ctx is not None
                and getattr(ctx, "store", None) is not None
            ):
                dataset = f"inputs/acquisition/receivers/{self.name}/coordinates"
                attrs = {"fs_kind": "receiver_coordinates"}
                if coords.units is not None:
                    attrs["units"] = coords.units
                if coords.system is not None:
                    attrs["system"] = coords.system
                ref = ctx.store.put_dataarray(dataset, coords.coordinates, attrs=attrs)
                coords_payload = {
                    "_type": "CoordsFromFile",
                    "file": ref.locator(),
                    "format": "HDF5",
                    "hash": f"blake3:{ref.hash}",
                    **({"units": coords.units} if coords.units is not None else {}),
                    **({"system": coords.system} if coords.system is not None else {}),
                }
            elif coords.size > 10:
                dump = coords.to_file(file_name="coords.h5", format="HDF5")
                dump._set_path(
                    proj_path=self._proj_path, rel_path=self._rel_path / self.name
                )
                self.coordinates = dump
                coords_payload = self.coordinates.to_fs(ctx)
            else:
                coords_payload = coords.to_fs(ctx)
        else:
            coords_payload = self.coordinates.to_fs(ctx)

        payload = {
            "name": self.name,
            "device": self.device.to_fs(ctx),
            "frame": self.frame,
            **({"domain": self.domain} if self.domain is not None else {}),
            **(
                {"sampling": self.sampling.to_fs(ctx)}
                if self.sampling is not None
                else {}
            ),
            "coordinates": coords_payload,
        }
        return merge_extra(payload, self.extra, "ReceiverGroup")

    def __dict__(self) -> Dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: Dict) -> "ReceiverGroup":
        data = copy.deepcopy(data)
        coords = ReceiverCoords.from_dict(data.pop("coordinates", None))
        sampling = data.pop("sampling", None)

        return cls(
            name=data.pop("name", None),
            device=ReceiverDevice.from_dict(data.pop("device", None)),
            frame=data.pop("frame", None),
            coordinates=coords,
            domain=data.pop("domain", None),
            sampling=sampling,
            **data,
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
