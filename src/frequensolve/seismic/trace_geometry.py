"""Geometry and coordinate helpers for trace arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
import xarray as xr

# Register the ``trace.fs`` accessor when available.  The import is intentionally
# direct so this helper does not depend on ``frequensolve.seismic`` package
# exports during initialization.
import frequensolve.seismic.trace_record as _trace_record  # noqa: F401
from frequensolve.geometry.grids import CartesianGrid


@dataclass(frozen=True)
class AxisInfo:
    """Display axis extracted from trace coordinates or survey metadata."""

    values: np.ndarray
    label: str
    coordinate_index: int | None = None


def as_trace_array(trace: xr.DataArray, *, caller: str = "trace") -> xr.DataArray:
    """Validate and squeeze a trace ``DataArray``.

    Args:
        trace: Trace data array.
        caller: Name used in error messages.

    Returns:
        Squeezed trace array.
    """

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
    """Convert eager or dask-backed array values to ``numpy``.

    Args:
        value: Array-like object, possibly with ``compute``.

    Returns:
        NumPy array.
    """

    if hasattr(value, "compute"):
        value = value.compute()
    return np.asarray(value)


def trace_values(trace: xr.DataArray, *, complex_: bool = False) -> np.ndarray:
    """Return trace values as a real or complex ``numpy`` array.

    Args:
        trace: Trace data array.
        complex_: Whether to preserve/convert to complex values.

    Returns:
        NumPy array of trace values.
    """

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
    """Build a display label from coordinate metadata.

    Args:
        trace: Trace data array.
        dim: Coordinate dimension name.
        default: Label used when coordinate metadata is unavailable.

    Returns:
        Display label, including units when present.
    """

    if dim not in trace.coords:
        return default
    coord = trace.coords[dim]
    label = coord.attrs.get("long_name") or coord.attrs.get("description") or default
    units = coord.attrs.get("units")
    return f"{label} [{units}]" if units else str(label)


def ensure_monotonic(values: np.ndarray, *, name: str) -> None:
    """Validate a strictly increasing coordinate axis.

    Args:
        values: Coordinate values.
        name: Coordinate name used in error messages.

    Raises:
        ValueError: If fewer than two samples exist or values are not strictly
            increasing.
    """

    if values.size < 2:
        raise ValueError(f"{name} needs at least two samples")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} coordinates must be strictly increasing")


def select_time(trace: xr.DataArray, t_max: float | None = None) -> xr.DataArray:
    """Slice a time-domain trace by final time.

    Args:
        trace: Time-domain trace array.
        t_max: Optional final time.

    Returns:
        Trace sliced through ``t_max``.
    """

    if t_max is None:
        return trace
    require_dims(trace, "time")
    return trace.sel(time=slice(None, t_max))


def select_frequency(
    trace: xr.DataArray,
    *,
    frequency: float | None = None,
    frequency_index: int | None = None,
) -> xr.DataArray:
    """Select one frequency sample from a frequency-domain trace.

    ``frequency`` is matched to the nearest numeric frequency coordinate.
    ``frequency_index`` is a zero-based positional index.  When neither is
    provided, the first positive frequency sample is selected when available;
    otherwise the first sample is selected.
    """

    require_dims(trace, "frequency")
    if frequency is not None and frequency_index is not None:
        raise ValueError("Specify either frequency or frequency_index, not both")
    if frequency_index is not None:
        return trace.isel(frequency=int(frequency_index))
    values = coordinate_values(trace, "frequency", require_numeric=True)
    if frequency is None:
        positive = np.flatnonzero(values > 0.0)
        index = int(positive[0]) if positive.size else 0
        return trace.isel(frequency=index)

    index = int(np.nanargmin(np.abs(values - float(frequency))))
    return trace.isel(frequency=index)


def _selected_frequency_value(trace: xr.DataArray) -> float | None:
    """Return a scalar selected frequency coordinate when present."""

    if "frequency" not in trace.coords:
        return None
    values = np.asarray(trace.coords["frequency"].data, dtype=float).ravel()
    if values.size == 0:
        return None
    return float(values[0])


def sampling_rate(trace: xr.DataArray) -> float:
    """Return samples per second from the ``time`` coordinate.

    Args:
        trace: Time-domain trace array.

    Returns:
        Sampling rate in hertz.
    """

    times = coordinate_values(trace, "time", require_numeric=True)
    ensure_monotonic(times, name="time")
    return 1.0 / float(np.mean(np.diff(times)))


def receiver_axis(
    trace: xr.DataArray,
    *,
    units: str = "",
    scale: float = 1.0,
) -> AxisInfo:
    """Infer the most useful receiver plotting axis.

    Args:
        trace: Trace data array with a receiver dimension.
        units: Optional display units label.
        scale: Multiplicative coordinate scale.

    Returns:
        Axis values, label, and optional physical coordinate index.
    """

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
        coords = np.asarray(trace.fs.source_coordinates, dtype=float)
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
    """Infer the display array shape for a 2D receiver grid.

    Args:
        trace: Trace data array with a receiver dimension.
        grid_shape: Optional explicit ``(ny, nx)`` shape.

    Returns:
        Two-dimensional display shape.
    """

    require_dims(trace, "receiver")
    if grid_shape is not None:
        shape = tuple(int(n) for n in grid_shape)
        if len(shape) != 2 or shape[0] * shape[1] != trace.sizes["receiver"]:
            raise ValueError("grid_shape must be a 2D shape matching receiver count")
        return shape

    try:
        return _receiver_grid_display_shape(trace)
    except ValueError:
        pass

    grid = receiver_grid(trace)
    if grid is not None:
        shape = tuple(int(n) for n in grid.n)
        if len(shape) >= 2 and int(np.prod(shape[:2])) == trace.sizes["receiver"]:
            return shape[1], shape[0]

    n_receiver = trace.sizes["receiver"]
    root = int(round(np.sqrt(n_receiver)))
    if root * root == n_receiver:
        return root, root

    raise ValueError(
        "grid_shape is required unless receiver coordinates describe a 2D grid"
    )


def receiver_grid(trace: xr.DataArray) -> CartesianGrid | None:
    """Return the ``CartesianGrid`` backing a trace receiver group, if present."""

    for key in ("wavefield_grid", "receiver_grid", "grid"):
        payload = trace.attrs.get(key)
        if isinstance(payload, CartesianGrid):
            return payload
        if isinstance(payload, Mapping):
            if "dims" in payload and "coords" in payload:
                try:
                    return _cartesian_grid_from_xarray_payload(payload)
                except Exception:
                    continue
            try:
                return CartesianGrid.from_fs(dict(payload))
            except Exception:
                pass

    try:
        return cast(CartesianGrid | None, trace.fs.receiver_group.grid)
    except Exception:
        return None


def _wavefield_grid_payload(trace: xr.DataArray) -> Mapping[str, Any] | None:
    for key in ("wavefield_grid", "receiver_grid", "grid"):
        payload = trace.attrs.get(key)
        if isinstance(payload, Mapping) and "dims" in payload and "coords" in payload:
            return payload
    return None


def _coord_payload_values(payload: Mapping[str, Any], dim: str) -> np.ndarray:
    coord = payload.get("coords", {}).get(dim)
    if coord is None:
        raise ValueError(f"wavefield grid is missing coordinate {dim!r}")
    if isinstance(coord, Mapping):
        if "data" in coord:
            coord = coord["data"]
        elif "values" in coord:
            coord = coord["values"]
        elif "value" in coord:
            coord = coord["value"]
        else:
            raise ValueError(f"wavefield grid coordinate {dim!r} requires data")
    values = np.asarray(coord, dtype=float).ravel()
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"wavefield grid coordinate {dim!r} must be 1D")
    return values


def _coord_payload_units(payload: Mapping[str, Any], dim: str) -> str:
    coord = payload.get("coords", {}).get(dim)
    if isinstance(coord, Mapping) and coord.get("units"):
        return str(coord["units"])
    if payload.get("units"):
        return str(payload["units"])
    return ""


def _coords_uniform(values: np.ndarray) -> bool:
    if values.size < 3:
        return True
    diffs = np.diff(values)
    return bool(np.allclose(diffs, diffs[0]))


def _cartesian_grid_from_xarray_payload(payload: Mapping[str, Any]) -> CartesianGrid:
    dims = list(payload["dims"])
    coords = {dim: _coord_payload_values(payload, dim) for dim in dims}
    xarr = xr.DataArray(dims=dims, coords=coords)
    grid = CartesianGrid.from_xarray(xarr)
    if payload.get("units"):
        grid.units = str(payload["units"])
    if payload.get("system"):
        grid.system = str(payload["system"])
    return grid


def wavefield_grid_display(
    trace: xr.DataArray,
    *,
    x: str | None = None,
    y: str | None = None,
    L_scale: float = 1.0,
) -> dict[str, Any] | None:
    """Return display metadata for gridded wavefield traces.

    Args:
        trace: Wavefield trace data array.
        x: Grid dimension to display on the horizontal axis.
        y: Grid dimension to display on the vertical axis.
        L_scale: Coordinate scale factor applied to display extents.

    Returns:
        Mapping with extent, labels, coordinates, and uniform-grid flag, or
        ``None`` when no compatible grid metadata is available.
    """

    payload = _wavefield_grid_payload(trace)
    if payload is not None:
        dims = _wavefield_grid_display_dims(payload)
        if dims is None:
            return None
        y_dim, x_dim, transpose = _resolve_wavefield_display_dims(
            (dims[0], dims[1]), x=x, y=y
        )
        x_values = _coord_payload_values(payload, x_dim) * L_scale
        y_values = _coord_payload_values(payload, y_dim) * L_scale
        x_units = _coord_payload_units(payload, x_dim)
        y_units = _coord_payload_units(payload, y_dim)
        return {
            "dimensions": (dims[0], dims[1]),
            "extent": [
                float(x_values[0]),
                float(x_values[-1]),
                float(y_values[-1]),
                float(y_values[0]),
            ],
            "xlabel": f"{x_dim.upper()}{f' [{x_units}]' if x_units else ''}",
            "ylabel": f"{y_dim.upper()}{f' [{y_units}]' if y_units else ''}",
            "x": x_values,
            "y": y_values,
            "uniform": _coords_uniform(x_values) and _coords_uniform(y_values),
            "transpose": transpose,
        }

    grid = receiver_grid(trace)
    if grid is None or len(grid.n) < 2:
        return None
    units = f" [{grid.units}]" if grid.units else ""
    dims = list(grid.dims) if len(grid.dims) >= 2 else ["x", "z"]
    y_dim, x_dim, transpose = _resolve_wavefield_display_dims(
        (dims[1], dims[0]), x=x, y=y
    )
    axis_index = {dims[0]: 0, dims[1]: 1}
    x_index = axis_index[x_dim]
    y_index = axis_index[y_dim]
    x_values = np.linspace(grid.x0[x_index], grid.x1[x_index], grid.n[x_index])
    y_values = np.linspace(grid.x0[y_index], grid.x1[y_index], grid.n[y_index])
    return {
        "dimensions": (dims[0], dims[1]),
        "extent": [
            float(x_values[0]) * L_scale,
            float(x_values[-1]) * L_scale,
            float(y_values[-1]) * L_scale,
            float(y_values[0]) * L_scale,
        ],
        "xlabel": f"{x_dim.upper()}{units}",
        "ylabel": f"{y_dim.upper()}{units}",
        "x": x_values * L_scale,
        "y": y_values * L_scale,
        "uniform": True,
        "transpose": transpose,
    }


def _resolve_wavefield_display_dims(
    native_dims: tuple[str, str],
    *,
    x: str | None,
    y: str | None,
) -> tuple[str, str, bool]:
    """Resolve xarray-style axis selectors against two display dimensions."""

    native_y, native_x = native_dims
    available = (native_y, native_x)
    for axis_name, dim in (("x", x), ("y", y)):
        if dim is not None and dim not in available:
            expected = ", ".join(repr(name) for name in available)
            raise ValueError(
                f"{axis_name}={dim!r} is not a wavefield grid dimension; "
                f"expected one of {expected}"
            )

    if x is None and y is None:
        y, x = native_y, native_x
    elif x is None:
        x = next(dim for dim in available if dim != y)
    elif y is None:
        y = next(dim for dim in available if dim != x)

    assert x is not None and y is not None
    if x == y:
        raise ValueError("x and y must select different wavefield grid dimensions")
    return y, x, (y, x) == (native_x, native_y)


def _receiver_grid_display_shape(
    trace: xr.DataArray,
    *,
    caller: str = "trace",
) -> tuple[int, int]:
    """Return display shape for a grid-backed wavefield array."""

    require_dims(trace, "receiver")
    payload = _wavefield_grid_payload(trace)
    if payload is not None:
        dims = _wavefield_grid_display_dims(payload, caller=caller)
        if dims is None:
            raise ValueError(f"{caller} requires a 2D wavefield grid")
        y_dim, x_dim = dims
        shape = (
            _coord_payload_values(payload, y_dim).size,
            _coord_payload_values(payload, x_dim).size,
        )
        full_shape = tuple(
            _coord_payload_values(payload, dim).size for dim in payload["dims"]
        )
        if int(np.prod(full_shape)) != trace.sizes["receiver"]:
            raise ValueError(
                f"{caller} receiver grid does not match receiver count "
                f"({full_shape} vs {trace.sizes['receiver']})"
            )
        return shape

    grid = receiver_grid(trace)
    if grid is None or len(grid.n) < 2:
        raise ValueError(f"{caller} requires wavefield grid metadata")
    grid_shape = tuple(int(n) for n in grid.n)
    if int(np.prod(grid_shape[:2])) != trace.sizes["receiver"]:
        raise ValueError(
            f"{caller} receiver grid does not match receiver count "
            f"({grid_shape[:2]} vs {trace.sizes['receiver']})"
        )
    return grid_shape[1], grid_shape[0]


def _wavefield_grid_display_dims(
    payload: Mapping[str, Any],
    *,
    caller: str | None = None,
) -> list[str] | None:
    dims = list(payload["dims"])
    if len(dims) == 2:
        return dims
    sizes = {dim: _coord_payload_values(payload, dim).size for dim in dims}
    display_dims = [dim for dim in dims if sizes[dim] > 1]
    if len(display_dims) == 2:
        return display_dims
    if caller is not None:
        raise ValueError(f"{caller} requires a 2D wavefield grid")
    return None
