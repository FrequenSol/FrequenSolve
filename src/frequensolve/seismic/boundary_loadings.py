"""Distributed boundary loading fields backed by the simulation HDF5 store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from frequensolve.units import ureg
from frequensolve.util.mixins import ExportContext

__all__ = ["SurfacePressureLoading"]


def _coordinate_values(coordinate: xr.DataArray, target_units: str) -> np.ndarray:
    values = np.asarray(coordinate.values, dtype=float)
    units = str(coordinate.attrs.get("units", target_units)).strip() or target_units
    try:
        return np.asarray(
            (values * ureg(units)).to(target_units).magnitude, dtype=float
        )
    except Exception as exc:
        raise ValueError(
            f"Coordinate {coordinate.name!r} units {units!r} are not compatible "
            f"with {target_units}"
        ) from exc


def _source_label(value: Any, fallback: str) -> str:
    label = "" if value is None else str(value).strip()
    if not label:
        label = fallback
    if not label:
        raise ValueError("Boundary loading source names cannot be empty")
    return label


def _single_source_arrays(
    pressure: Any,
) -> List[Tuple[str, xr.DataArray]]:
    if isinstance(pressure, xr.DataArray):
        if "source" not in pressure.dims:
            name = _source_label(pressure.name, "source_000001")
            return [(name, pressure)]
        labels = pressure.coords.get("source")
        if labels is None:
            source_values: Iterable[Any] = range(1, pressure.sizes["source"] + 1)
        else:
            source_values = labels.values
        result = []
        for index, value in enumerate(source_values):
            fallback = f"source_{index + 1:06d}"
            result.append(
                (
                    _source_label(value, fallback),
                    pressure.isel(source=index, drop=True),
                )
            )
        return result

    if isinstance(pressure, xr.Dataset):
        return [
            (_source_label(name, ""), data) for name, data in pressure.data_vars.items()
        ]

    if isinstance(pressure, Mapping):
        result = []
        for name, data in pressure.items():
            if not isinstance(data, xr.DataArray):
                raise TypeError(
                    "Surface-pressure mappings must contain xarray.DataArray values"
                )
            if "source" in data.dims:
                raise ValueError(
                    "Mapped surface-pressure arrays must omit the source dimension; "
                    "the mapping key is the source name"
                )
            result.append((_source_label(name, ""), data))
        return result

    raise TypeError(
        "pressure must be an xarray.DataArray, xarray.Dataset, or mapping of "
        "source names to DataArrays"
    )


def _spatial_dimensions(data: xr.DataArray, selection_dim: str) -> Tuple[str, ...]:
    dims = tuple(str(dim) for dim in data.dims if dim != selection_dim)
    if len(dims) not in {1, 2}:
        raise ValueError(
            "Surface pressure requires one spatial dimension in 2-D or two in 3-D"
        )
    for dim in dims:
        if dim not in data.coords:
            raise ValueError(f"Surface-pressure dimension {dim!r} has no coordinates")
        coordinate = np.asarray(data.coords[dim].values)
        if coordinate.ndim != 1 or coordinate.size != data.sizes[dim]:
            raise ValueError(
                f"Surface-pressure coordinate {dim!r} must be one-dimensional"
            )
        if coordinate.size > 1 and np.any(np.diff(coordinate.astype(float)) <= 0.0):
            raise ValueError(
                f"Surface-pressure coordinate {dim!r} must be strictly increasing"
            )
    return dims


def _reference_coordinate(
    data: xr.DataArray, spatial_dims: Sequence[str]
) -> Dict[str, Any]:
    dimension = len(spatial_dims) + 1
    values = np.zeros(dimension, dtype=float)
    coordinate_units = []
    used_slots = set()
    axis_slots = {"x": 0, "y": 1, "z": 2}
    for dim in spatial_dims:
        units = str(data.coords[dim].attrs.get("units", "")).strip()
        coordinate_units.append(units)
        slot = axis_slots.get(dim.lower())
        if slot is None or slot >= dimension or slot in used_slots:
            slot = next(index for index in range(dimension) if index not in used_slots)
        used_slots.add(slot)
        coordinate = np.asarray(data.coords[dim].values, dtype=float)
        values[slot] = 0.5 * (float(coordinate[0]) + float(coordinate[-1]))

    nonempty_units = {unit for unit in coordinate_units if unit}
    if len(nonempty_units) > 1:
        raise ValueError(
            "Surface-pressure spatial coordinates must use one common length unit"
        )
    system = str(data.attrs.get("system", data.attrs.get("coord_system", "global")))
    payload: Dict[str, Any] = {"value": values.tolist(), "system": system}
    if nonempty_units:
        payload["units"] = nonempty_units.pop()
    return payload


def _pack_frequency_field(
    data: xr.DataArray, spatial_dims: Sequence[str]
) -> Tuple[xr.DataArray, List[float]]:
    frequencies = _coordinate_values(data.coords["frequency"], "Hz")
    if frequencies.size == 0 or np.any(frequencies < 0.0):
        raise ValueError("Surface-pressure frequencies must be non-negative")
    if frequencies.size > 1 and np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("Surface-pressure frequencies must be strictly increasing")
    ordered = data.transpose(*spatial_dims, "frequency")
    values = np.asarray(ordered.values)
    if not np.all(np.isfinite(values)):
        raise ValueError("Surface-pressure frequency data must be finite")
    values = np.asarray(values, dtype=np.complex128)
    packed_values = np.stack((values.real, values.imag), axis=-1).reshape(
        values.shape[:-1] + (2 * values.shape[-1],)
    )
    packed = xr.DataArray(
        packed_values,
        dims=(*spatial_dims, "component"),
        coords={dim: data.coords[dim] for dim in spatial_dims},
    )
    return packed, frequencies.tolist()


def _pack_time_field(
    data: xr.DataArray, spatial_dims: Sequence[str]
) -> Tuple[xr.DataArray, float, float]:
    times = _coordinate_values(data.coords["time"], "s")
    if times.size < 2:
        raise ValueError("Surface-pressure time data requires at least two samples")
    steps = np.diff(times)
    tolerance = max(1.0e-12, 1.0e-9 * abs(float(steps[0])))
    if steps[0] <= 0.0 or not np.allclose(steps, steps[0], rtol=1.0e-9, atol=tolerance):
        raise ValueError("Surface-pressure time coordinates must be uniformly spaced")
    ordered = data.transpose(*spatial_dims, "time")
    values = np.asarray(ordered.values)
    if np.iscomplexobj(values):
        raise ValueError("Surface-pressure time histories must be real-valued")
    if not np.all(np.isfinite(values)):
        raise ValueError("Surface-pressure time data must be finite")
    packed = xr.DataArray(
        np.asarray(values, dtype=np.float64),
        dims=(*spatial_dims, "component"),
        coords={dim: data.coords[dim] for dim in spatial_dims},
    )
    return packed, float(times[0]), float(steps[0])


@dataclass
class SurfacePressureLoading:
    """Pressure applied to a gravity-surface boundary condition.

    ``pressure`` may be a common-grid DataArray with an optional ``source``
    dimension, a Dataset whose variables are sources, or a mapping from source
    names to independently gridded DataArrays. Each array must contain exactly
    one ``frequency`` or ``time`` dimension.
    """

    pressure: Any
    boundary_condition: Optional[str] = None
    window: str = "none"
    detrend: str = "none"

    def __post_init__(self) -> None:
        self.window = str(self.window).strip().lower()
        self.detrend = str(self.detrend).strip().lower()
        if self.window not in {"none", "hann"}:
            raise ValueError("Surface-pressure window must be 'none' or 'hann'")
        if self.detrend not in {"none", "mean"}:
            raise ValueError("Surface-pressure detrend must be 'none' or 'mean'")
        fields = _single_source_arrays(self.pressure)
        names = [name for name, _ in fields]
        if not fields:
            raise ValueError("Surface pressure must contain at least one source")
        if len(names) != len(set(names)):
            raise ValueError("Surface-pressure source names must be unique")

    def source_names(self) -> List[str]:
        """Return the ordered logical RHS/source labels."""

        return [name for name, _ in _single_source_arrays(self.pressure)]

    def with_boundary_condition(self, name: str) -> "SurfacePressureLoading":
        """Return a shallow copy associated with one named boundary condition."""

        return SurfacePressureLoading(
            pressure=self.pressure,
            boundary_condition=name,
            window=self.window,
            detrend=self.detrend,
        )

    @classmethod
    def from_fs(cls, _data: Mapping[str, Any]) -> "SurfacePressureLoading":
        """Reject materialized solver data as an authoring-time pressure field.

        Solver JSON contains references to packed HDF5 slabs rather than the
        xarray objects needed for authoring. ``Acquisition.from_fs`` preserves
        those mappings directly so a loaded simulation can still round-trip.
        """

        raise TypeError(
            "Materialized SurfacePressureLoading JSON is preserved by Acquisition; "
            "construct this class from xarray pressure data when authoring"
        )

    def to_fs(self, ctx: ExportContext, *, loading_index: int) -> Dict[str, Any]:
        """Materialize component-packed fields and serialize the loading."""

        if not self.boundary_condition:
            raise ValueError("SurfacePressureLoading requires a boundary condition")
        if ctx.store is None:
            raise ValueError(
                "Surface-pressure arrays require an export context with an HDF5 store"
            )

        fields_payload = []
        for field_index, (source_name, data) in enumerate(
            _single_source_arrays(self.pressure)
        ):
            has_frequency = "frequency" in data.dims
            has_time = "time" in data.dims
            if has_frequency == has_time:
                raise ValueError(
                    "Each surface-pressure field must contain exactly one of "
                    "'frequency' or 'time'"
                )
            selection_dim = "frequency" if has_frequency else "time"
            spatial_dims = _spatial_dimensions(data, selection_dim)
            units = str(data.attrs.get("units", "Pa")).strip() or "Pa"
            try:
                (1.0 * ureg(units)).to("Pa")
            except Exception as exc:
                raise ValueError(
                    f"Surface-pressure units {units!r} are not pressure units"
                ) from exc
            system = str(
                data.attrs.get("system", data.attrs.get("coord_system", "global"))
            )

            field_payload: Dict[str, Any] = {
                "source": source_name,
                "reference_coordinate": _reference_coordinate(data, spatial_dims),
            }
            if has_frequency:
                packed, frequencies = _pack_frequency_field(data, spatial_dims)
                field_payload.update(
                    {"domain": "frequency", "frequencies": frequencies}
                )
            else:
                packed, t0, dt = _pack_time_field(data, spatial_dims)
                field_payload.update(
                    {
                        "domain": "time",
                        "time_origin": t0,
                        "time_step": dt,
                        "window": self.window,
                        "detrend": self.detrend,
                    }
                )
            rhs_normalization = float(np.max(np.abs(np.asarray(data.values))))
            if not np.isfinite(rhs_normalization) or rhs_normalization <= 0.0:
                raise ValueError(
                    f"Surface-pressure source {source_name!r} must not be identically zero"
                )
            field_payload["rhs_normalization"] = rhs_normalization

            dataset = (
                "inputs/acquisition/boundary_loadings/"
                f"loading_{loading_index:04d}/field_{field_index:06d}"
            )
            ref = ctx.store.put_dataarray(
                dataset,
                packed,
                attrs={
                    "fs_kind": "sampled_boundary_field",
                    "units": units,
                    "system": system,
                },
                coordinate_dims=spatial_dims,
                interpolation_dims=spatial_dims,
                dtype=np.float32,
            )
            field_payload["data"] = {
                **ref.to_fs(format="HDF5"),
                "units": units,
                "system": system,
            }
            fields_payload.append(field_payload)

        return {
            "_type": "SurfacePressure",
            "boundary_condition": self.boundary_condition,
            "fields": fields_payload,
        }
