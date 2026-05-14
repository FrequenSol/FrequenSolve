"""Receiver definitions and coordinate systems.

This module defines the various types of receivers and their locations.
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import xarray as xr

from frequensolve.geometry.frame import CoordinateValue, Direction, direction_to_fs
from frequensolve.geometry.grids import CartesianGrid, Grid
from frequensolve.seismic.sparse_survey import ReceiverSampling
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.units import is_quantity, unit_expression
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.fields import canonical_field
from frequensolve.util.mixins import ExtraFieldsMixin, TypeTaggedMixin, merge_extra
from frequensolve.util.store import hash_dataarray_payload

__all__ = [
    "CoordsArray",
    "CoordsFromFile",
    "CoordsGrid",
    "ReceiverComponent",
    "ReceiverGroup",
    "ReceiverCoords",
    "ReceiverDevice",
    "ReceiverNodeArray",
    "ReceiverNode",
    "ReceiverFiber",
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
    field: str
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

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverComponent":
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
class ReceiverDevice(TypeTaggedMixin, ABC):
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
                {"response": (self.response.to_fs(ctx))}
                if self.response is not None
                else {}
            ),
        }

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverDevice":
        return cls.dispatch_from_fs(data, class_registry)


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

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverFiber":
        return cls(
            name=data["name"],
            components=[ReceiverComponent.from_fs(c) for c in data["components"]],
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

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverNodeArray":
        return cls(
            name=data["name"],
            components=[ReceiverComponent.from_fs(c) for c in data["components"]],
            response=data.get("response"),
            offsets=data["offsets"],
        )


@register_class
@dataclass(kw_only=True)
class ReceiverNode(ReceiverDevice):
    """Defines a node receiver."""

    def to_fs(self, ctx=None) -> dict:
        return {"_type": self.__class__.__name__, **super().to_fs(ctx)}

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverNode":
        return cls(
            name=data["name"],
            components=[ReceiverComponent.from_fs(c) for c in data["components"]],
            response=data.get("response"),
        )


# ----------------------------------------------------------------------
# Receiver Coordinates
# ----------------------------------------------------------------------
@register_class
@dataclass(kw_only=True)
class ReceiverCoords(TypeTaggedMixin, ABC):
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
    def to_fs(self, ctx=None) -> Dict:
        """Convert coordinates to a solver payload."""
        pass

    @classmethod
    def from_fs(cls, data: Dict) -> "ReceiverCoords":
        return cls.dispatch_from_fs(data, class_registry)

    def _set_path(self, proj_path: Path, rel_path: Path):
        try:
            self.path = proj_path / self.path.relative_to(self._proj_path)
        except Exception:
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
    hash: Optional[str] = None
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __init__(
        self,
        file: Union[str, Path] = None,
        format: Literal["HDF5"] = "HDF5",
        dset: Optional[str] = None,
        units: Optional[str] = None,
        system: Optional[str] = None,
        hash: Optional[str] = None,
        **kwargs,
    ):
        if file is None and "path" in kwargs:
            file = kwargs.pop("path")
        self.file = Path(file).expanduser()
        if self.file.is_absolute():
            self.file = self.file.resolve()
        self.format = format
        self.dset = dset
        self.units = units
        self.system = system
        self.hash = hash

    @classmethod
    def from_fs(cls, data: Dict) -> "CoordsFromFile":
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
            hash=data.get("hash"),
        )

    def _set_path(self, proj_path: Path, rel_path: Path):
        proj_path = Path(proj_path).resolve()
        rel_path = Path(rel_path)
        file = Path(self.file).expanduser()

        if not file.is_absolute():
            file = proj_path / file
        else:
            simulation_path = rel_path.parent
            expected = proj_path / simulation_path / f"{simulation_path.name}.h5"
            is_simulation_store = (
                file.name == expected.name
                and file.parent.name == simulation_path.name
                and file.parent.parent.name == "simulations"
            )
            if is_simulation_store and expected.exists():
                file = expected

        self.file = file
        self._proj_path = proj_path
        self._rel_path = rel_path

    def _relative_file(self, ctx=None) -> Path:
        project_path = getattr(ctx, "project_path", None) or self._proj_path
        file = Path(self.file)
        if project_path is None:
            return file
        try:
            return file.resolve().relative_to(Path(project_path).resolve())
        except Exception:
            return file

    def _local_file(self, ctx=None) -> Path:
        file = Path(self.file)
        if file.is_absolute():
            return file
        project_path = getattr(ctx, "project_path", None) or self._proj_path
        if project_path is not None:
            return Path(project_path) / file
        return file

    def _content_hash(self, ctx=None) -> Optional[str]:
        if self.format != "HDF5":
            return None
        file = self._local_file(ctx)
        if not file.exists():
            return None
        dataset = self.dset or "coords"
        try:
            with h5py.File(file, "r") as h5:
                if dataset not in h5:
                    return None
                values = np.asarray(h5[dataset][()])
        except OSError:
            return None
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.ndim != 2:
            return None
        if values.shape[1] == 2:
            coordinate = ["x", "z"]
        elif values.shape[1] == 3:
            coordinate = ["x", "y", "z"]
        else:
            coordinate = list(range(values.shape[1]))
        data = xr.DataArray(
            values,
            dims=["receiver", "coordinate"],
            coords={"coordinate": coordinate},
        )
        return f"blake3:{hash_dataarray_payload(data)}"

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
        rel_path = self._relative_file(ctx)

        if self.format == "HDF5":
            if self.dset is None:
                file = str(rel_path.with_suffix(".h5")) + ":coords"
            else:
                file = str(rel_path.with_suffix(".h5")) + ":" + self.dset
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

        file_hash = self._content_hash(ctx) or self.hash
        return {
            "_type": self.__class__.__name__,
            "file": file,
            "format": self.format,
            **({"hash": file_hash} if file_hash is not None else {}),
            **(
                {"units": unit_expression(self.units)} if self.units is not None else {}
            ),
            **({"system": self.system} if self.system is not None else {}),
        }


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
            payload["units"] = unit_expression(self.units)
        if self.system is not None:
            payload["system"] = self.system
        return payload

    @classmethod
    def from_fs(cls, data: Dict) -> "CoordsGrid":
        return cls(
            grid=CartesianGrid.from_fs(data["grid"]),
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
                payload["units"] = unit_expression(self.units)
            if self.system is not None:
                payload["system"] = self.system
        else:
            payload["coords"] = values
        return payload

    @classmethod
    def from_fs(cls, data: Dict) -> "CoordsArray":
        coords = np.array(data.get("coords", data.get("value")))
        return cls(
            coordinates=coords, units=data.get("units"), system=data.get("system")
        )


def _first_quantity_units(value: Any) -> Optional[Any]:
    if is_quantity(value):
        return value.units
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.flat:
                units = _first_quantity_units(item)
                if units is not None:
                    return units
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            units = _first_quantity_units(item)
            if units is not None:
                return units
    return None


def _strip_coordinate_quantities(value: Any, units: Any) -> Any:
    if is_quantity(value):
        return value.to(units).magnitude
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            values = [_strip_coordinate_quantities(item, units) for item in value.flat]
            return np.asarray(values, dtype=float).reshape(value.shape)
        return value
    if isinstance(value, (list, tuple)):
        return [_strip_coordinate_quantities(item, units) for item in value]
    return value


def coordinate_array_metadata(
    coords: Any,
) -> Tuple[np.ndarray, Optional[Any], Optional[str]]:
    """Return numeric coordinate values plus units/system metadata.

    Pint quantities intentionally become magnitudes here, with their units
    carried on the receiver/source coordinate object instead of being stripped
    implicitly by NumPy.
    """

    system = None
    explicit_units = None
    if isinstance(coords, CoordinateValue):
        explicit_units = coords.units
        system = coords.system
        coords = coords.value

    quantity_units = _first_quantity_units(coords)
    units = explicit_units if explicit_units is not None else quantity_units
    if quantity_units is not None:
        coords = _strip_coordinate_quantities(coords, units)

    return np.asarray(coords, dtype=float), units, system


# ----------------------------------------------------------------------
# Receiver Groups
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class ReceiverGroup(ExtraFieldsMixin):
    """A group of multi-component receivers with shared output settings.

    This class represents a collection of receivers that measure one or more physical
    quantities. All receivers in the group share output settings and their data will be
    written to the same output file.

    Attributes:
       name (str):                   String identifier for this receiver group.
       device (ReceiverDevice):      Device defining receiver type and components.
       coordinates (ReceiverCoords): Coordinates defining receiver locations.
    """

    name: str = "group"
    device: ReceiverDevice = field(default_factory=ReceiverDevice)
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
    # TODO: method to define receivers

    def __init__(
        self,
        name: str,
        device: ReceiverDevice,
        coordinates: Union[np.ndarray, xr.DataArray, str, Path, Grid, ReceiverCoords],
        domain: Optional[int] = None,
        sampling: Optional[Union[str, Dict, ReceiverSampling]] = None,
        survey: Optional[Union[str, ReceiverSampling]] = None,
        extra: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        deprecated_frame_keys = {"frame", "source_frame", "receiver_frame"} & set(
            kwargs
        )
        if deprecated_frame_keys:
            raise TypeError(
                "ReceiverGroup frame is no longer supported; receiver coordinates are physical"
            )
        coords = self._clean_coordinates(coordinates)
        sampling_obj = ReceiverSampling.from_value(sampling)
        survey_obj = ReceiverSampling.from_value(survey)
        if sampling_obj is None:
            sampling_obj = survey_obj
        elif survey_obj is not None and sampling_obj.survey is None:
            sampling_obj.survey = survey_obj.survey
        self.name = name
        self.device = device
        self.coordinates = coords
        self.domain = domain
        self.sampling = sampling_obj
        self._init_extra(extra, **kwargs)
        deprecated_frame_keys = {"frame", "source_frame", "receiver_frame"} & set(
            self.extra
        )
        if deprecated_frame_keys:
            raise TypeError(
                "ReceiverGroup frame is no longer supported; receiver coordinates are physical"
            )

    @property
    def survey(self) -> Optional[str]:
        if self.sampling is None:
            return None
        return self.sampling.survey

    @survey.setter
    def survey(self, value: Optional[Union[str, ReceiverSampling]]) -> None:
        self.sampling = ReceiverSampling.from_value(value)

    @staticmethod
    def _clean_coordinates(coords):
        # Allow coordinates to be defined either as a ReceiverCoords object
        # various other reasonble ways:
        if isinstance(coords, CoordinateValue):
            values, units, system = coordinate_array_metadata(coords)
            out = CoordsArray(coordinates=values, units=units, system=system)
        elif isinstance(coords, ReceiverCoords):
            return coords
        elif isinstance(coords, xr.DataArray):
            out = CoordsArray(coordinates=coords)
        elif isinstance(coords, np.ndarray) or isinstance(coords, list):
            values, units, system = coordinate_array_metadata(coords)
            out = CoordsArray(coordinates=values, units=units, system=system)
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
                coords.size > 200
                and ctx is not None
                and getattr(ctx, "store", None) is not None
            ):
                dataset = f"inputs/acquisition/receivers/{self.name}/coordinates"
                attrs = {"fs_kind": "receiver_coordinates"}
                if coords.units is not None:
                    attrs["units"] = unit_expression(coords.units)
                if coords.system is not None:
                    attrs["system"] = coords.system
                ref = ctx.store.put_dataarray(dataset, coords.coordinates, attrs=attrs)
                coords_payload = {
                    "_type": "CoordsFromFile",
                    "file": ref.locator(),
                    "format": "HDF5",
                    "hash": f"blake3:{ref.hash}",
                    **(
                        {"units": unit_expression(coords.units)}
                        if coords.units is not None
                        else {}
                    ),
                    **({"system": coords.system} if coords.system is not None else {}),
                }
            elif (
                coords.size > 200
                and self._proj_path is not None
                and self._rel_path is not None
            ):
                dump = coords.to_file(file_name="coords.h5", format="HDF5")
                dump._set_path(
                    proj_path=self._proj_path, rel_path=self._rel_path / self.name
                )
                coords_payload = dump.to_fs(ctx)
            else:
                coords_payload = coords.to_fs(ctx)
        else:
            coords_payload = self.coordinates.to_fs(ctx)

        payload = {
            "name": self.name,
            "device": self.device.to_fs(ctx),
            **({"domain": self.domain} if self.domain is not None else {}),
            **(
                {"sampling": self.sampling.to_fs(ctx)}
                if self.sampling is not None
                else {}
            ),
            "coordinates": coords_payload,
        }
        return merge_extra(payload, self.extra, "ReceiverGroup")

    @classmethod
    def from_fs(cls, data: Dict) -> "ReceiverGroup":
        data = copy.deepcopy(data)
        coords = ReceiverCoords.from_fs(data.pop("coordinates", None))
        sampling = data.pop("sampling", None)
        data.pop("frame", None)

        return cls(
            name=data.pop("name", None),
            device=ReceiverDevice.from_fs(data.pop("device", None)),
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
