"""Geometry and coordinate helpers for trace arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

# Register the ``trace.fs`` accessor when available.  The import is intentionally
# direct so this helper does not depend on ``frequensolve.seismic`` package
# exports during initialization.
import frequensolve.seismic.trace_record as _trace_record  # noqa: F401


@dataclass(frozen=True)
class AxisInfo:
    """Display axis extracted from trace coordinates or survey metadata."""

    values: np.ndarray
    label: str
    coordinate_index: int | None = None


def as_trace_array(trace: xr.DataArray, *, caller: str = "trace") -> xr.DataArray:
    """Validate and squeeze a trace ``DataArray``."""

    if not isinstance(trace, xr.DataArray):
        raise TypeError(f"{caller} expects an xarray.DataArray")
    return trace.squeeze(drop=True)


def require_dims(trace: xr.DataArray, *dims: str) -> None:
    """Raise a clear error if ``trace`` is missing required dimensions."""

    missing = [dim for dim in dims if dim not in trace.dims]
    if missing:
        names = ", ".join(repr(dim) for dim in missing)
        raise ValueError(f"trace is missing required dimension(s): {names}")


def to_numpy(value: Any) -> np.ndarray:
    """Convert eager or dask-backed array values to ``numpy``."""

    if hasattr(value, "compute"):
        value = value.compute()
    return np.asarray(value)


def trace_values(trace: xr.DataArray, *, complex_: bool = False) -> np.ndarray:
    """Return trace values as a real or complex ``numpy`` array."""

    values = to_numpy(trace.data)
    if complex_:
        return np.asarray(values, dtype=np.complex128)
    return np.real(values)


def coordinate_values(
    trace: xr.DataArray,
    dim: str,
    *,
    scale: float = 1.0,
    require_numeric: bool = False,
) -> np.ndarray:
    """Return coordinate values for ``dim`` or a one-based ordinal fallback."""

    if dim in trace.coords:
        raw = to_numpy(trace.coords[dim].data)
    else:
        raw = np.arange(1, trace.sizes[dim] + 1)

    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError):
        if require_numeric:
            raise ValueError(f"{dim!r} coordinates must be numeric") from None
        values = np.arange(1, trace.sizes[dim] + 1, dtype=float)
    return values * scale


def coordinate_label(trace: xr.DataArray, dim: str, default: str) -> str:
    """Build a display label from coordinate metadata."""

    if dim not in trace.coords:
        return default
    coord = trace.coords[dim]
    label = coord.attrs.get("long_name") or coord.attrs.get("description") or default
    units = coord.attrs.get("units")
    return f"{label} [{units}]" if units else str(label)


def ensure_monotonic(values: np.ndarray, *, name: str) -> None:
    """Validate a strictly increasing coordinate axis."""

    if values.size < 2:
        raise ValueError(f"{name} needs at least two samples")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} coordinates must be strictly increasing")


def select_time(trace: xr.DataArray, t_max: float | None = None) -> xr.DataArray:
    """Slice a time-domain trace by final time."""

    if t_max is None:
        return trace
    require_dims(trace, "time")
    return trace.sel(time=slice(None, t_max))


def time_limit(T_max: float | None, Tf: float | None) -> float | None:
    """Normalize accepted final-time keyword aliases."""

    return T_max if T_max is not None else Tf


def sampling_rate(trace: xr.DataArray) -> float:
    """Return samples per second from the ``time`` coordinate."""

    times = coordinate_values(trace, "time", require_numeric=True)
    ensure_monotonic(times, name="time")
    return 1.0 / float(np.mean(np.diff(times)))


def receiver_axis(
    trace: xr.DataArray,
    *,
    units: str = "",
    scale: float = 1.0,
) -> AxisInfo:
    """Infer the most useful receiver plotting axis."""

    require_dims(trace, "receiver")
    try:
        coords = np.asarray(trace.fs.receiver_group.coordinates, dtype=float)
    except Exception:
        coords = None

    if (
        coords is not None
        and coords.ndim == 2
        and coords.shape[0] == trace.sizes["receiver"]
    ):
        span = np.nanmax(coords, axis=0) - np.nanmin(coords, axis=0)
        axis = int(np.nanargmax(span))
        if span[axis] > 0:
            names = ["X", "Y", "Z"]
            unit_label = f" [{units}]" if units else ""
            return AxisInfo(
                values=coords[:, axis] * scale,
                label=f"{names[axis]}{unit_label}",
                coordinate_index=axis,
            )

    values = coordinate_values(trace, "receiver", scale=scale)
    return AxisInfo(
        values=values, label=coordinate_label(trace, "receiver", "Receiver")
    )


def source_coordinate(trace: xr.DataArray, axis: int | None) -> float | None:
    """Return the physical source coordinate for ``axis`` when metadata exists."""

    if axis is None:
        return None
    try:
        coords = np.asarray(trace.fs.source_group.source.coordinates, dtype=float)
    except Exception:
        return None
    if coords.ndim == 0 or axis >= coords.size:
        return None
    return float(coords[axis])


def receiver_offsets(
    trace: xr.DataArray,
    *,
    units: str = "",
    scale: float = 1.0,
) -> tuple[np.ndarray, AxisInfo]:
    """Return receiver offsets relative to the source when possible."""

    axis = receiver_axis(trace, units=units, scale=scale)
    source = source_coordinate(trace, axis.coordinate_index)
    offsets = (
        axis.values - source if source is not None else axis.values - axis.values[0]
    )
    return offsets, axis


def receiver_grid_shape(
    trace: xr.DataArray,
    grid_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Infer a 2D receiver grid shape for animation."""

    require_dims(trace, "receiver")
    if grid_shape is not None:
        shape = tuple(int(n) for n in grid_shape)
        if len(shape) != 2 or shape[0] * shape[1] != trace.sizes["receiver"]:
            raise ValueError("grid_shape must be a 2D shape matching receiver count")
        return shape

    try:
        shape = tuple(int(n) for n in trace.fs.receiver_group.grid.n)
    except Exception:
        shape = ()
    if len(shape) >= 2 and int(np.prod(shape[:2])) == trace.sizes["receiver"]:
        return shape[:2]

    n_receiver = trace.sizes["receiver"]
    root = int(round(np.sqrt(n_receiver)))
    if root * root == n_receiver:
        return root, root

    raise ValueError(
        "grid_shape is required unless receiver coordinates describe a 2D grid"
    )
