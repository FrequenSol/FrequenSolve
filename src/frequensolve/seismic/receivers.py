"""Receiver definitions and coordinate systems.

This module defines the various types of receivers and their locations.
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
import numpy as np
import xarray as xr

from frequensolve.geometry.frame import CoordinateValue, Direction, direction_to_fs
from frequensolve.geometry.grids import CartesianGrid, Grid
from frequensolve.seismic.sparse_survey import ReceiverSampling
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.units import is_quantity, unit_expression, value_and_units_to_fs
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.fields import canonical_field
from frequensolve.util.mixins import (
    ExtraFieldsMixin,
    TypeTaggedMixin,
    merge_extra,
)
from frequensolve.util.store import SimulationStore, hash_dataarray_payload

__all__ = [
    "CoordsArray",
    "CoordsFromFile",
    "CoordsGrid",
    "CoordsSurfaceCarpet",
    "ReceiverComponent",
    "ReceiverGroup",
    "ReceiverCoords",
    "ReceiverDevice",
    "ReceiverNodeArray",
    "ReceiverNode",
    "ReceiverFiber",
    "ReceiverSampling",
]


def _is_remote_file_reference(value: Any) -> bool:
    text = str(value)
    return text.startswith("remote:") or "://" in text


@dataclass(kw_only=True)
class ReceiverComponent:
    """Defines a single component/measurement type for a receiver.

    A receiver component specifies what physical quantity is being measured
    (e.g., pressure, velocity) and in what direction for vector quantities.

    Args:
        name: String identifier for this receiver component.
        field: Physical field being measured.
        direction: Optional measurement direction for vector fields.
        units: Optional output units for this component.
        weight: Optional scalar/vector weight applied to the component.
    """

    name: str = "name"
    field: str
    direction: Optional[Union[List[float], Direction]] = None
    units: Optional[str] = None
    weight: Optional[Union[float, List[float], Dict]] = None

    def __post_init__(self) -> None:
        self.field = canonical_field(self.field)

    def to_fs(self, ctx=None) -> dict:
        """Serialize this receiver component for solver input."""

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
        """Deserialize a receiver component payload."""

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
_RECEIVER_FIBER_DEGREE_UNITS = {"deg", "degree", "degrees"}
_RECEIVER_FIBER_RADIAN_UNITS = {"rad", "radian", "radians"}


def _receiver_fiber_angle_degrees(angle: Any) -> float:
    """Validate a fiber winding angle and return its value in degrees."""

    multiplier = 1.0
    if is_quantity(angle):
        try:
            value = angle.to("degree").magnitude
        except Exception as exc:
            raise ValueError(
                "ReceiverFiber angle must be an angular quantity."
            ) from exc
    elif isinstance(angle, Mapping):
        if "value" not in angle:
            raise ValueError("ReceiverFiber angle quantity requires a value.")
        value = angle["value"]
        units = unit_expression(angle.get("units", "deg")).strip()
        if units in _RECEIVER_FIBER_RADIAN_UNITS:
            multiplier = 180.0 / np.pi
        elif units not in _RECEIVER_FIBER_DEGREE_UNITS:
            raise ValueError("ReceiverFiber angle units must be degrees or radians.")
    else:
        value = angle

    if isinstance(value, (str, bytes, bool)):
        raise ValueError("ReceiverFiber angle must be a numeric scalar.")
    try:
        scalar = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ReceiverFiber angle must be a numeric scalar.") from exc
    if scalar.ndim != 0:
        raise ValueError("ReceiverFiber angle must be a numeric scalar.")

    degrees = float(scalar) * multiplier
    if not np.isfinite(degrees) or not 0.0 < degrees < 90.0:
        raise ValueError(
            "ReceiverFiber angle must be strictly between 0 and 90 degrees."
        )
    return degrees


def _receiver_fiber_pitch_from_angle(radius: Any, angle: Any) -> Any:
    """Return the legacy pitch contract equivalent to a winding angle."""

    degrees = _receiver_fiber_angle_degrees(angle)
    factor = 2.0 * np.pi / np.tan(np.deg2rad(degrees))
    radius_payload = value_and_units_to_fs(radius)
    if isinstance(radius_payload, Mapping):
        if "value" not in radius_payload:
            raise ValueError("ReceiverFiber radius quantity requires a value.")
        pitch = copy.deepcopy(radius_payload)
        radius_value = pitch["value"]
    else:
        pitch = None
        radius_value = radius_payload

    if isinstance(radius_value, (str, bytes, bool)):
        raise ValueError("ReceiverFiber radius must be a numeric scalar.")
    try:
        scalar = np.asarray(radius_value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ReceiverFiber radius must be a numeric scalar.") from exc
    if scalar.ndim != 0:
        raise ValueError("ReceiverFiber radius must be a numeric scalar.")

    pitch_value = float(scalar) * factor
    if not np.isfinite(pitch_value) or pitch_value <= 0.0:
        raise ValueError("ReceiverFiber radius must be positive and finite.")
    if pitch is None:
        return pitch_value
    pitch["value"] = pitch_value
    return pitch


@register_class
@dataclass(kw_only=True)
class ReceiverDevice(TypeTaggedMixin, ABC):
    """Abstract base class for a receiver device with measured components.

    Args:
        name: Optional identifier for this receiver device.
        components: Components defining measured quantities.
        response: Optional receiver response wavelet.
    """

    name: Optional[str] = None
    components: List[ReceiverComponent] = field(default_factory=list)
    response: Optional[Wavelet] = None

    def add_component(
        self, name: str, field: str, direction: Optional[List[float]] = None
    ) -> "ReceiverComponent":
        """Add a measured component to this device.

        Args:
            name: Component name used in trace output.
            field: Physical field to measure.
            direction: Optional measurement direction for vector fields.

        Returns:
            Newly added ``ReceiverComponent``.
        """

        component = ReceiverComponent(
            name=name,
            field=canonical_field(field),
            direction=direction,
        )
        self.components.append(component)
        return component

    def to_fs(self, ctx=None) -> dict:
        """Serialize this receiver device for solver input."""

        return {
            **({"name": self.name} if self.name is not None else {}),
            "components": [c.to_fs(ctx) for c in self.components],
            **(
                {"response": (self.response.to_fs(ctx))}
                if self.response is not None
                else {}
            ),
        }

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverDevice":
        """Deserialize a registered receiver-device payload."""

        return cls.dispatch_from_fs(data, class_registry)


@register_class
@dataclass(kw_only=True)
class ReceiverFiber(ReceiverDevice):
    """Defines a fiber receiver device (e.g. a DAS fiber) that integrates
    response over a length.

    The response is integrated over the gauge length using either a reusable
    sample spacing or a requested number of points per gauge.

    Args:
        name: Optional device name.
        components: Receiver components measured by each channel.
        gauge_length: Physical gauge length.
        channel_spacing: Physical spacing between fiber channels. Defaults to
            ``gauge_length``.
        sample_spacing: Physical spacing for integration samples along a gauge.
        points_per_gauge: Number of integration samples when sample spacing is
            not provided.
        radius: Optional helical-fiber radius. Required with ``angle`` or
            ``pitch``.
        pitch: Optional helical-fiber pitch, mutually exclusive with ``angle``.
        angle: Optional helical-fiber winding angle from the cable axis,
            mutually exclusive with ``pitch``. Plain numbers are degrees;
            unit-aware angular quantities are also accepted.
        response: Optional receiver response wavelet.

    Raises:
        ValueError: If ``gauge_length`` is omitted or ``points_per_gauge`` is
            not positive, or if the helical-fiber geometry is invalid.
    """

    gauge_length: Any = None
    channel_spacing: Optional[Any] = None
    sample_spacing: Optional[Any] = None
    points_per_gauge: Optional[int] = None
    radius: Optional[Any] = None
    pitch: Optional[Any] = None
    angle: Optional[Any] = None

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        components: Optional[List[ReceiverComponent]] = None,
        gauge_length: Optional[Any] = None,
        channel_spacing: Optional[Any] = None,
        sample_spacing: Optional[Any] = None,
        points_per_gauge: Optional[int] = None,
        radius: Optional[Any] = None,
        pitch: Optional[Any] = None,
        angle: Optional[Any] = None,
        response: Optional[Wavelet] = None,
    ):
        if gauge_length is None:
            raise ValueError("ReceiverFiber requires gauge_length.")
        if points_per_gauge is not None:
            points_per_gauge = int(points_per_gauge)
            if points_per_gauge < 1:
                raise ValueError("ReceiverFiber points_per_gauge must be positive.")

        self.name = name
        self.components = list(components) if components is not None else []
        self.response = response
        self.gauge_length = gauge_length
        self.channel_spacing = (
            channel_spacing if channel_spacing is not None else gauge_length
        )
        self.sample_spacing = sample_spacing
        self.points_per_gauge = points_per_gauge
        self.radius = radius
        self.pitch = pitch
        self.angle = angle
        self._validate_helical_geometry()

    def to_fs(self, ctx=None) -> dict:
        """Serialize this fiber receiver device for solver input."""

        self._validate_helical_geometry()
        data = {
            "_type": self.__class__.__name__,
            **super().to_fs(ctx),
            "gauge_length": value_and_units_to_fs(self.gauge_length),
        }
        if self.channel_spacing is not None:
            data["channel_spacing"] = value_and_units_to_fs(self.channel_spacing)
        if self.sample_spacing is not None:
            data["sample_spacing"] = value_and_units_to_fs(self.sample_spacing)
        if self.points_per_gauge is not None:
            data["points_per_gauge"] = self.points_per_gauge
        if self.radius is not None:
            data["radius"] = value_and_units_to_fs(self.radius)
        if self.pitch is not None:
            data["pitch"] = value_and_units_to_fs(self.pitch)
        if self.angle is not None:
            data["pitch"] = _receiver_fiber_pitch_from_angle(self.radius, self.angle)
        return data

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverFiber":
        """Deserialize a fiber receiver device payload."""

        return cls(
            name=data.get("name"),
            components=[ReceiverComponent.from_fs(c) for c in data["components"]],
            response=data.get("response"),
            gauge_length=data.get("gauge_length"),
            channel_spacing=data.get("channel_spacing"),
            sample_spacing=data.get("sample_spacing"),
            points_per_gauge=data.get("points_per_gauge"),
            radius=data.get("radius"),
            pitch=data.get("pitch"),
            angle=data.get("angle"),
        )

    def _validate_helical_geometry(self) -> None:
        if self.pitch is not None and self.angle is not None:
            raise ValueError(
                "ReceiverFiber accepts only one of angle or pitch, not both."
            )
        if (self.pitch is not None or self.angle is not None) and self.radius is None:
            raise ValueError(
                "ReceiverFiber radius is required when angle or pitch is specified."
            )
        if self.angle is not None:
            _receiver_fiber_angle_degrees(self.angle)


@register_class
@dataclass(kw_only=True)
class ReceiverNodeArray(ReceiverDevice):
    """Defines a group of nodes on a single channel; defined by list of offsets.

    Args:
        offsets: Node offsets from each receiver location. The fast dimension
            is coordinates and the slow dimension is array node index.
        name: Optional device name.
        components: Receiver components measured by each array.
        response: Optional receiver response wavelet.
    """

    offsets: List[List[float]] = field(default_factory=list)

    def to_fs(self, ctx=None) -> dict:
        """Serialize this node-array device for solver input."""

        return {
            "_type": self.__class__.__name__,
            **super().to_fs(ctx),
            "offsets": self.offsets,
        }

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverNodeArray":
        """Deserialize a node-array receiver device payload."""

        return cls(
            name=data.get("name"),
            components=[ReceiverComponent.from_fs(c) for c in data["components"]],
            response=data.get("response"),
            offsets=data["offsets"],
        )


@register_class
@dataclass(kw_only=True)
class ReceiverNode(ReceiverDevice):
    """Point receiver device evaluated at each receiver coordinate."""

    def to_fs(self, ctx=None) -> dict:
        """Serialize this point receiver device for solver input."""

        return {"_type": self.__class__.__name__, **super().to_fs(ctx)}

    @classmethod
    def from_fs(cls, data: dict) -> "ReceiverNode":
        """Deserialize a point receiver device payload."""

        return cls(
            name=data.get("name"),
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
        """Deserialize a registered receiver-coordinate payload."""

        return cls.dispatch_from_fs(data, class_registry)


@register_class
@dataclass(kw_only=True)
class CoordsFromFile(ReceiverCoords):
    """Receiver coordinates stored in a file.

    Args:
        file: Coordinate file path. Relative paths are resolved from the export
            or project context when serialized/read.
        format: Coordinate file format. Currently ``"HDF5"``.
        dset: HDF5 dataset name. Defaults to ``"coords"``.
        units: Optional coordinate units.
        system: Optional coordinate-system name.
        hash: Optional content hash for freshness checks.
    """

    file: Path
    format: Literal["HDF5"]
    dset: Optional[str] = None
    units: Optional[str] = None
    system: Optional[str] = None
    hash: Optional[str] = None

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
        if kwargs.pop("remote", False) or _is_remote_file_reference(file):
            raise ValueError(
                "CoordsFromFile does not support remote coordinate files yet; "
                "provide a local file or inline/materialized coordinates."
            )
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
        """Deserialize file-backed receiver coordinates."""

        file = data["file"]
        format = data["format"]
        if data.get("remote", False) or _is_remote_file_reference(file):
            raise ValueError(
                "CoordsFromFile does not support remote coordinate files yet; "
                "provide a local file or inline/materialized coordinates."
            )
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

    @staticmethod
    def _simulation_rel_path(rel_path: Optional[Path]) -> Optional[Path]:
        if rel_path is None:
            return None
        rel_path = Path(rel_path)
        parts = rel_path.parts
        if "simulations" not in parts:
            return None
        index = parts.index("simulations")
        if len(parts) <= index + 1:
            return None
        return Path(*parts[: index + 2])

    def _contextual_file(
        self, ctx=None, *, source_project_path: Optional[Path] = None
    ) -> Path:
        file = Path(self.file).expanduser()
        project_path = getattr(ctx, "project_path", None)
        rel_path = getattr(ctx, "rel_path", None)
        if project_path is not None:
            project_path = Path(project_path).expanduser().resolve()

        if not file.is_absolute():
            return project_path / file if project_path is not None else file

        if source_project_path is not None and project_path is not None:
            try:
                project_relative = file.resolve().relative_to(
                    Path(source_project_path).expanduser().resolve()
                )
            except ValueError:
                pass
            else:
                return project_path / project_relative

        simulation_rel = self._simulation_rel_path(rel_path)
        if project_path is None or simulation_rel is None:
            return file
        expected = project_path / simulation_rel / f"{simulation_rel.name}.h5"
        is_simulation_store = (
            file.name == expected.name
            and file.parent.name == expected.parent.name
            and file.parent.parent.name == "simulations"
        )
        if is_simulation_store and expected.exists():
            return expected
        return file

    def _relative_file(self, ctx=None) -> Path:
        project_path = getattr(ctx, "project_path", None)
        file = self._contextual_file(ctx)
        if project_path is None:
            return file
        try:
            return file.resolve().relative_to(Path(project_path).resolve())
        except Exception:
            return file

    def _local_file(self, ctx=None) -> Path:
        file = self._contextual_file(ctx)
        if file.is_absolute():
            return file
        project_path = getattr(ctx, "project_path", None)
        if project_path is not None:
            return Path(project_path) / file
        return file

    def _hdf5_metadata(self, ctx=None) -> Tuple[Optional[str], Optional[str]]:
        if self.format != "HDF5":
            return None, None
        file = self._local_file(ctx)
        dataset = self.dset or "coords"
        if not file.exists():
            return None, None
        try:
            with h5py.File(file, "r") as h5:
                if dataset not in h5:
                    return None, None
                attrs = h5[dataset].attrs
                units = _h5_attr_string(attrs.get("units"))
                system = _h5_attr_string(
                    attrs.get("system", attrs.get("coordinate_system"))
                )
                return units, system
        except OSError:
            return None, None

    def _fill_metadata_from_file(self, ctx=None) -> None:
        if self.units is not None and self.system is not None:
            return
        units, system = self._hdf5_metadata(ctx)
        if self.units is None:
            self.units = units
        if self.system is None:
            self.system = system

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
        attrs = {}
        file_units, file_system = self._hdf5_metadata(ctx)
        units = self.units or file_units or getattr(ctx, "default_length_units", None)
        system = self.system or file_system
        if units is not None:
            attrs["units"] = unit_expression(units)
        if system is not None:
            attrs["system"] = system
        return f"blake3:{hash_dataarray_payload(data, attrs=attrs)}"

    @property
    def size(self) -> int:
        """Get the total number of receivers.

        Returns:
           int: Number of receivers.
        """
        if self.format == "HDF5":
            with h5py.File(self._local_file(), "r") as f:
                return f[self.dset or "coords"].shape[0]
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get coordinate bounds without loading full dataset.

        Returns:
           Tuple[np.ndarray, np.ndarray]: Min and max coordinates.
        """
        if self.format == "HDF5":
            with h5py.File(self._local_file(), "r") as f:
                coords = f[self.dset or "coords"]
                return np.min(coords, axis=0), np.max(coords, axis=0)
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    def __getitem__(self, key: Union[tuple, slice]):
        """Return receiver coordinates selected from the backing file.

        Args:
            key: Slice, integer index, tuple, or boolean mask accepted by the
                HDF5 coordinate dataset.

        Returns:
            Coordinate array for the selected receivers.
        """

        return self.get(key)

    def get(self, indices: Optional[Union[Tuple, slice]] = None) -> np.ndarray:
        """Get coordinates for specified indices.

        Args:
           indices: Integer indices or boolean mask.

        Returns:
           np.ndarray: Coordinate array for requested receivers.
        """
        if self.format == "HDF5":
            file = self._local_file()
            with h5py.File(file, "r") as f:
                dataset = self.dset or "coords"
                if dataset not in f:
                    raise KeyError(
                        f"Receiver coordinate dataset '{dataset}' is missing from "
                        f"'{file}'. Rebuild and save the simulation acquisition "
                        "inputs before copying or running it."
                    )
                if indices is None:
                    return f[dataset][:]
                else:
                    return f[dataset][indices]
        else:
            raise NotImplementedError(f"Format {self.format} not implemented")

    def to_fs(self, ctx=None) -> Dict:
        """Serialize file-backed receiver coordinates for solver input."""

        rel_path = self._relative_file(ctx)
        default_units = getattr(ctx, "default_length_units", None)
        units = self.units if self.units is not None else default_units
        system = self.system

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
            **({"units": unit_expression(units)} if units is not None else {}),
            **({"system": system} if system is not None else {}),
        }


@register_class
@dataclass(kw_only=True)
class CoordsGrid(ReceiverCoords):
    """Receiver coordinates defined by a Cartesian grid.

    Args:
        grid: Cartesian grid defining receiver locations.
        units: Optional coordinate units override.
        system: Optional coordinate-system name override.
    """

    grid: CartesianGrid
    units: Optional[str] = None
    system: Optional[str] = None

    @property
    def size(self) -> int:
        """Return the number of receiver coordinates in the grid."""

        return np.prod(self.grid.n)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper coordinate bounds for the grid."""

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
        """Serialize grid receiver coordinates for solver input."""

        payload = {"_type": self.__class__.__name__, "grid": self.grid.to_fs(ctx)}
        if self.units is not None:
            payload["units"] = unit_expression(self.units)
        if self.system is not None:
            payload["system"] = self.system
        return payload

    @classmethod
    def from_fs(cls, data: Dict) -> "CoordsGrid":
        """Deserialize grid-backed receiver coordinates."""

        return cls(
            grid=CartesianGrid.from_fs(data["grid"]),
            units=data.get("units"),
            system=data.get("system"),
        )


class CoordsSurfaceCarpet(ReceiverCoords):
    """Receiver coordinates on a surface-relative tensor-product carpet.

    The carpet is kept in compact axis form until values are explicitly
    requested or exported to HDF5.
    """

    def __init__(
        self,
        *,
        x: Any,
        y: Optional[Any] = None,
        offset: float = 0.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
    ) -> None:
        self.x = _surface_carpet_axis("x", x, units)
        self.y = None if y is None else _surface_carpet_axis("y", y, units)
        self.offset = float(offset)
        self.units = units
        self.system = system

    @classmethod
    def try_from_surface(
        cls,
        surface: Any,
        *,
        x: Any,
        y: Optional[Any] = None,
        units: Optional[Any] = None,
        above: Optional[Any] = None,
        below: Optional[Any] = None,
    ) -> Optional["CoordsSurfaceCarpet"]:
        """Return a compact carpet when the surface helper exposes metadata."""

        if above is not None and below is not None:
            raise ValueError("Specify only one of above or below")

        simulation = getattr(surface, "_simulation", None)
        if getattr(simulation, "dimension", None) == 3 and y is None:
            raise ValueError("3D surface points_grid requires x and y axes")

        system = getattr(surface, "coordinate_system", surface)
        system_name = getattr(system, "name", None)
        if system_name is None or getattr(system, "type", None) != "surface":
            return None

        carpet_units = units or _first_quantity_units(x)
        if carpet_units is None and y is not None:
            carpet_units = _first_quantity_units(y)

        offset_value = 0.0
        distance = above if above is not None else below
        if carpet_units is None and distance is not None:
            carpet_units = _first_quantity_units(distance)

        if distance is not None:
            normal = str(getattr(system, "normal", "up") or "up").strip().lower()
            if above is not None:
                sign = -1 if normal == "down" else 1
            else:
                sign = 1 if normal == "down" else -1
            try:
                offset_value = sign * _surface_carpet_scalar(
                    "surface offset", distance, carpet_units
                )
            except ValueError as exc:
                if "must be a scalar" in str(exc):
                    return None
                raise

        return cls(
            x=x,
            y=y,
            offset=offset_value,
            units=carpet_units,
            system=system_name,
        )

    @property
    def dimension(self) -> int:
        """Return the number of coordinate columns."""

        return 2 if self.y is None else 3

    @property
    def axes(self) -> List[str]:
        """Return coordinate-axis labels."""

        return ["x", "z"] if self.y is None else ["x", "y", "z"]

    @property
    def shape(self) -> Tuple[int, int]:
        """Return the dense coordinate table shape."""

        return (self.size, self.dimension)

    @property
    def size(self) -> int:
        """Return the number of receiver coordinates."""

        if self.y is None:
            return int(self.x.size)
        return int(self.x.size * self.y.size)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return lower and upper coordinate bounds without materializing rows."""

        if self.y is None:
            lower = np.asarray([np.min(self.x), self.offset], dtype=float)
            upper = np.asarray([np.max(self.x), self.offset], dtype=float)
        else:
            lower = np.asarray(
                [np.min(self.x), np.min(self.y), self.offset], dtype=float
            )
            upper = np.asarray(
                [np.max(self.x), np.max(self.y), self.offset], dtype=float
            )
        return lower, upper

    def get(
        self, indices: Optional[Union[int, slice, Sequence[int], np.ndarray]] = None
    ) -> np.ndarray:
        """Return coordinate values for all or selected receivers."""

        if indices is None:
            values = np.empty(self.shape, dtype=np.float64)
            offset = 0
            for chunk in self.iter_chunks():
                stop = offset + chunk.shape[0]
                values[offset:stop] = chunk
                offset = stop
            return values

        if isinstance(indices, slice):
            flat_indices = np.arange(self.size, dtype=np.int64)[indices]
        elif isinstance(indices, (int, np.integer)):
            index = int(indices)
            if index < 0:
                index += self.size
            return self._rows_for_indices(np.asarray([index], dtype=np.int64))[0]
        else:
            flat_indices = np.asarray(indices)
            if flat_indices.dtype == np.dtype(bool):
                flat_indices = np.nonzero(flat_indices.reshape(-1))[0]
            flat_indices = flat_indices.astype(np.int64, copy=False).reshape(-1)
            flat_indices[flat_indices < 0] += self.size
        return self._rows_for_indices(flat_indices)

    def iter_chunks(self, chunk_size: int = 1 << 20) -> Iterator[np.ndarray]:
        """Yield dense coordinate chunks in receiver-major order."""

        chunk_size = max(1, int(chunk_size))
        if self.y is None:
            for start in range(0, self.size, chunk_size):
                x_values = self.x[start : start + chunk_size]
                chunk = np.empty((x_values.size, 2), dtype=np.float64)
                chunk[:, 0] = x_values
                chunk[:, 1] = self.offset
                yield chunk
            return

        for start in range(0, self.size, chunk_size):
            stop = min(start + chunk_size, self.size)
            yield self._rows_for_indices(np.arange(start, stop, dtype=np.int64))

    def to_hdf5_reference(
        self,
        store: Any,
        dataset: str,
        *,
        attrs: Optional[Dict[str, Any]] = None,
        dtype: Any = np.float64,
    ):
        """Write this carpet to a store-backed HDF5 dataset."""

        return store.put_array_chunks(
            dataset,
            self.shape,
            self.iter_chunks,
            attrs=attrs,
            dims=["receiver", "coordinate"],
            coords={"coordinate": np.asarray(self.axes, dtype=str)},
            dtype=dtype,
        )

    def to_fs(self, ctx=None) -> Dict:
        """Serialize inline when no HDF5 export context is available."""

        units = (
            self.units
            if self.units is not None
            else getattr(ctx, "default_length_units", None)
        )
        return CoordsArray(
            coordinates=self.get(),
            units=units,
            system=self.system,
        ).to_fs(ctx)

    def _rows_for_indices(self, indices: np.ndarray) -> np.ndarray:
        if np.any((indices < 0) | (indices >= self.size)):
            raise IndexError("receiver coordinate index out of range")

        if self.y is None:
            rows = np.empty((indices.size, 2), dtype=np.float64)
            rows[:, 0] = self.x[indices]
            rows[:, 1] = self.offset
            return rows

        nx = self.x.size
        x_index = indices % nx
        y_index = indices // nx
        rows = np.empty((indices.size, 3), dtype=np.float64)
        rows[:, 0] = self.x[x_index]
        rows[:, 1] = self.y[y_index]
        rows[:, 2] = self.offset
        return rows


@register_class
@dataclass(kw_only=True)
class CoordsArray(ReceiverCoords):
    """Receiver coordinates stored as an xarray/numpy array.

    Args:
        coordinates: Coordinate array with shape ``(n_receivers, dimension)``.
            xarray input may carry ``units`` and ``system`` attributes.
        units: Optional coordinate units.
        system: Optional coordinate-system name.

    Raises:
        ValueError: If a NumPy coordinate array is not two-dimensional with two
            or three coordinate columns.
    """

    coordinates: Union[xr.DataArray, np.ndarray]
    units: Optional[str] = None
    system: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.coordinates, xr.DataArray):
            if self.units is None:
                self.units = self.coordinates.attrs.get("units")
            if self.system is None:
                self.system = self.coordinates.attrs.get("system")
            self.coordinates = self.coordinates.astype(np.float64, copy=False)

        if isinstance(self.coordinates, np.ndarray):
            self.coordinates = np.asarray(self.coordinates, dtype=np.float64)
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
        """Return the number of receiver coordinates."""

        return len(self.coordinates)

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return minimum and maximum coordinate values by coordinate axis."""

        return (
            self.coordinates.min(dim="receiver").values,
            self.coordinates.max(dim="receiver").values,
        )

    def get(self, indices: Optional[Union[int, slice]] = None) -> np.ndarray:
        """Return coordinate values for all or selected receivers.

        Args:
            indices: Optional integer, slice, or indexer accepted by xarray.

        Returns:
            ``float64`` coordinate array.
        """

        if indices is None:
            return np.asarray(self.coordinates.values, dtype=np.float64)
        else:
            return np.asarray(self.coordinates[indices].values, dtype=np.float64)

    def to_file(
        self, file_name: Union[str, Path], format: Optional[Literal["HDF5"]] = None
    ) -> CoordsFromFile:
        """Write coordinates to file and return CoordsFromFile object.

        Args:
            file_name: Output coordinate file path.
            format: Optional file format. Inferred from ``file_name`` when
                omitted.

        Returns:
            ``CoordsFromFile`` pointing at the written HDF5 dataset.

        Raises:
            ValueError: If the file extension does not identify a supported
                format.
        """
        file = Path(file_name).expanduser()
        if format is None:
            if str(file).endswith(".h5") or str(file).endswith(".hdf5"):
                format = "HDF5"
            else:
                raise ValueError(f"Unknown coordinates file extension: {file_name}")

        if not file.is_absolute():
            file = file.resolve()
        if not file.parent.exists():
            file.parent.mkdir(parents=True)

        if format == "HDF5":
            with h5py.File(file, "w") as f:
                dset = f.create_dataset(
                    "coords", data=(self.coordinates.values).astype(np.float64)
                )
                if self.units is not None:
                    dset.attrs["units"] = unit_expression(self.units)
                if self.system is not None:
                    dset.attrs["system"] = self.system
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
        """Serialize inline receiver coordinates for solver input."""

        values = np.asarray(self.coordinates.values, dtype=np.float64).tolist()
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
        """Deserialize inline receiver coordinates."""

        coords = np.array(data.get("coords", data.get("value")), dtype=np.float64)
        return cls(
            coordinates=coords, units=data.get("units"), system=data.get("system")
        )


def _surface_carpet_axis(name: str, values: Any, units: Optional[Any]) -> np.ndarray:
    if is_quantity(values):
        values = values.to(units).magnitude if units is not None else values.magnitude
    else:
        quantity_units = _first_quantity_units(values)
        if quantity_units is not None:
            target_units = units or quantity_units
            values = _strip_coordinate_quantities(values, target_units)
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size == 0:
        raise ValueError(f"{name} must contain at least one coordinate")
    if not np.isfinite(axis).all():
        raise ValueError(f"{name} coordinates must be finite")
    return axis


def _surface_carpet_scalar(name: str, value: Any, units: Optional[Any]) -> float:
    if is_quantity(value):
        value = value.to(units).magnitude if units is not None else value.magnitude
    else:
        quantity_units = _first_quantity_units(value)
        if quantity_units is not None:
            target_units = units or quantity_units
            value = _strip_coordinate_quantities(value, target_units)
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar for compact carpet coordinates")
    scalar = float(array.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


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


def _h5_attr_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
        return _h5_attr_string(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _h5_attr_string(value[0])
    return str(value)


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

    Args:
        coords: Raw coordinate array, Pint quantity, or ``CoordinateValue``.

    Returns:
        ``(values, units, system)`` with numeric values in double precision.
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

    return np.asarray(coords, dtype=np.float64), units, system


# ----------------------------------------------------------------------
# Receiver Groups
# ----------------------------------------------------------------------
@dataclass(kw_only=True)
class ReceiverGroup(ExtraFieldsMixin):
    """A group of multi-component receivers with shared output settings.

    This class represents a collection of receivers that measure one or more physical
    quantities. All receivers in the group share output settings and their data will be
    written to the same output file.

    Args:
        name: String identifier for this receiver group.
        device: Device defining receiver type and components.
        coordinates: Receiver coordinates as an array, grid, file path, or
            ``ReceiverCoords`` object.
        domain: Optional domain where the receiver group is evaluated.
        sampling: Optional sparse survey sampling reference.
        survey: Convenience sparse survey name/reference. Merged with
            ``sampling`` when both are supplied.
        extra: Additional solver-facing receiver group fields.
        **kwargs: Additional solver-facing receiver group fields.

    Raises:
        TypeError: If deprecated frame arguments are supplied.
        ValueError: If ``coordinates`` cannot be interpreted.
    """

    name: str = "group"
    device: ReceiverDevice = field(default_factory=ReceiverDevice)
    domain: Optional[int] = None
    coordinates: ReceiverCoords = field(default_factory=ReceiverCoords)
    sampling: Optional[ReceiverSampling] = None
    extra: Dict = field(default_factory=dict)

    @property
    def size(self):
        """Return the number of receiver locations in this group."""

        return self.coordinates.size

    @property
    def grid(self) -> Optional[CartesianGrid]:
        """Return the receiver grid when this group uses ``CoordsGrid``."""

        if isinstance(self.coordinates, CoordsGrid):
            return self.coordinates.grid
        return None

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
        """Return the sparse survey name referenced by this receiver group."""

        if self.sampling is None:
            return None
        return self.sampling.survey

    @survey.setter
    def survey(self, value: Optional[Union[str, ReceiverSampling]]) -> None:
        """Set sparse survey sampling from a name or ``ReceiverSampling``."""

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
        elif (
            is_quantity(coords)
            or isinstance(coords, np.ndarray)
            or isinstance(coords, list)
        ):
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
        """Serialize this receiver group for solver input.

        Large inline coordinate arrays may be written to the simulation store or
        export directory when an export context is available.
        """

        coords = self.coordinates
        if isinstance(coords, CoordsSurfaceCarpet):
            default_units = getattr(ctx, "default_length_units", None)
            coordinate_units = (
                coords.units if coords.units is not None else default_units
            )
            if (
                coords.size > 200
                and ctx is not None
                and getattr(ctx, "store", None) is not None
            ):
                dataset = f"inputs/acquisition/receivers/{self.name}/coordinates"
                attrs = {"fs_kind": "receiver_coordinates"}
                if coordinate_units is not None:
                    attrs["units"] = unit_expression(coordinate_units)
                if coords.system is not None:
                    attrs["system"] = coords.system
                ref = coords.to_hdf5_reference(
                    ctx.store,
                    dataset,
                    attrs=attrs,
                    dtype=np.float64,
                )
                coords_payload = {
                    "_type": "CoordsFromFile",
                    "file": ref.locator(),
                    "format": "HDF5",
                    "hash": f"blake3:{ref.hash}",
                    **(
                        {"units": unit_expression(coordinate_units)}
                        if coordinate_units is not None
                        else {}
                    ),
                    **({"system": coords.system} if coords.system is not None else {}),
                }
            elif coords.size > 200 and ctx is not None and ctx.path is not None:
                file = ctx.path / self.name / "coords.h5"
                store = SimulationStore(file, project_path=ctx.project_path)
                attrs = {"fs_kind": "receiver_coordinates"}
                if coordinate_units is not None:
                    attrs["units"] = unit_expression(coordinate_units)
                if coords.system is not None:
                    attrs["system"] = coords.system
                ref = coords.to_hdf5_reference(
                    store,
                    "coords",
                    attrs=attrs,
                    dtype=np.float64,
                )
                coords_payload = {
                    "_type": "CoordsFromFile",
                    "file": ref.locator(),
                    "format": "HDF5",
                    "hash": f"blake3:{ref.hash}",
                    **(
                        {"units": unit_expression(coordinate_units)}
                        if coordinate_units is not None
                        else {}
                    ),
                    **({"system": coords.system} if coords.system is not None else {}),
                }
            else:
                coords_payload = coords.to_fs(ctx)
                if coordinate_units is not None and "units" not in coords_payload:
                    if "coords" in coords_payload:
                        coords_payload["value"] = coords_payload.pop("coords")
                    coords_payload["units"] = unit_expression(coordinate_units)
        elif isinstance(coords, CoordsArray):
            default_units = getattr(ctx, "default_length_units", None)
            coordinate_units = (
                coords.units if coords.units is not None else default_units
            )
            if (
                coords.size > 200
                and ctx is not None
                and getattr(ctx, "store", None) is not None
            ):
                dataset = f"inputs/acquisition/receivers/{self.name}/coordinates"
                attrs = {"fs_kind": "receiver_coordinates"}
                if coordinate_units is not None:
                    attrs["units"] = unit_expression(coordinate_units)
                if coords.system is not None:
                    attrs["system"] = coords.system
                coordinate_dim = coords.coordinates.dims[1]
                ref = ctx.store.put_dataarray(
                    dataset,
                    coords.coordinates,
                    attrs=attrs,
                    coordinate_dims=(coordinate_dim,),
                    dtype=np.float64,
                )
                coords_payload = {
                    "_type": "CoordsFromFile",
                    "file": ref.locator(),
                    "format": "HDF5",
                    "hash": f"blake3:{ref.hash}",
                    **(
                        {"units": unit_expression(coordinate_units)}
                        if coordinate_units is not None
                        else {}
                    ),
                    **({"system": coords.system} if coords.system is not None else {}),
                }
            elif coords.size > 200 and ctx is not None and ctx.path is not None:
                coords_for_file = CoordsArray(
                    coordinates=coords.coordinates,
                    units=coordinate_units,
                    system=coords.system,
                )
                dump = coords_for_file.to_file(
                    file_name=ctx.path / self.name / "coords.h5",
                    format="HDF5",
                )
                coords_payload = dump.to_fs(ctx)
            else:
                coords_payload = coords.to_fs(ctx)
                if coordinate_units is not None and "units" not in coords_payload:
                    if "coords" in coords_payload:
                        coords_payload["value"] = coords_payload.pop("coords")
                    coords_payload["units"] = unit_expression(coordinate_units)
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
        """Deserialize a receiver group payload."""

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
