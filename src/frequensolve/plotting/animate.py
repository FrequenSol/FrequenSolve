"""Animation helpers for trace arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from frequensolve._optional import optional_dependency_error
from frequensolve.seismic.trace_geometry import (
    _receiver_grid_display_shape,
    _selected_frequency_value,
    as_trace_array,
    coordinate_values,
    receiver_grid_shape,
    require_dims,
    select_time,
    to_numpy,
    trace_values,
    wavefield_grid_display,
)

__all__ = ["animate_gather", "animate_wavefield"]


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise optional_dependency_error(
            "Trace animation",
            extra="visual",
            dependencies=("matplotlib",),
            error=exc,
        ) from exc

    return plt


def _animation():
    try:
        import matplotlib.animation as animation
    except ModuleNotFoundError as exc:
        raise optional_dependency_error(
            "Trace animation",
            extra="visual",
            dependencies=("matplotlib",),
            error=exc,
        ) from exc

    return animation


def _grid_display(trace: xr.DataArray):
    return wavefield_grid_display(trace)


def _single_frequency_wavefield(
    trace: xr.DataArray,
    *,
    caller: str,
) -> tuple[xr.DataArray, float | None]:
    if not isinstance(trace, xr.DataArray):
        raise TypeError(f"{caller} expects an xarray.DataArray")
    require_dims(trace, "receiver")
    if "frequency" in trace.dims:
        if trace.sizes["frequency"] != 1:
            raise ValueError(
                f"{caller} expects frequency-domain wavefields to contain "
                "exactly one frequency"
            )
        trace = trace.isel(frequency=0)
    frequency_value = _selected_frequency_value(trace)
    trace = as_trace_array(trace, caller=caller)
    require_dims(trace, "receiver")
    if trace.dims != ("receiver",):
        remaining = ", ".join(repr(dim) for dim in trace.dims if dim != "receiver")
        raise ValueError(
            f"{caller} expects one selected wavefield component/source; "
            f"remaining dimension(s): {remaining}"
        )
    return trace.transpose("receiver"), frequency_value


def _frequency_wavefield_grid(
    trace: xr.DataArray,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    shape = _receiver_grid_display_shape(trace, caller="animate_wavefield")
    values = trace_values(trace, complex_=True).reshape(*shape)
    display = _grid_display(trace)
    return values, display


def _scalar_float(value: Any, *, name: str) -> float:
    values = to_numpy(value.data if isinstance(value, xr.DataArray) else value)
    if values.size != 1:
        raise ValueError(f"{name} must be a scalar")
    return float(values.reshape(-1)[0])


def _amplitude_limit(values: np.ndarray, limit: Any | None) -> float:
    if limit is not None:
        return _scalar_float(limit, name="A")
    finite = values[np.isfinite(values)]
    amplitude = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    return amplitude if amplitude != 0 else 1.0


def _format_time_label(value: float, units: str | None = "s") -> str:
    suffix = f" {units}" if units else ""
    return f"t = {value:.4g}{suffix}"


def _time_units(trace: xr.DataArray) -> str | None:
    if "time" not in trace.coords:
        return "s"
    units = trace.coords["time"].attrs.get("units")
    return str(units) if units else "s"


def _time_box_location(location: str) -> tuple[float, float, str, str]:
    key = location.lower().replace("_", " ").replace("-", " ").strip()
    locations = {
        "southwest": (0.02, 0.02, "left", "bottom"),
        "south west": (0.02, 0.02, "left", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
        "southeast": (0.98, 0.02, "right", "bottom"),
        "south east": (0.98, 0.02, "right", "bottom"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "northwest": (0.02, 0.98, "left", "top"),
        "north west": (0.02, 0.98, "left", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
        "northeast": (0.98, 0.98, "right", "top"),
        "north east": (0.98, 0.98, "right", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
    }
    if key not in locations:
        raise ValueError(
            "time_location must be one of 'southwest', 'southeast', "
            "'northwest', or 'northeast'"
        )
    return locations[key]


def _add_time_box(
    ax,
    label: str,
    *,
    location: str = "southwest",
):
    x, y, ha, va = _time_box_location(location)
    return ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize="small",
        color="black",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.75,
        },
        zorder=10,
        animated=True,
    )


def _animation_artists(image, time_text):
    return (image,) if time_text is None else (image, time_text)


def animate_gather(
    trace: xr.DataArray,
    *,
    grid_shape: tuple[int, int] | None = None,
    ax=None,
    A: float | None = None,
    cmap: str = "Greys",
    interval: int = 50,
    title: str | None = None,
    figsize: tuple[float, float] = (5, 5),
    T_max: float | None = None,
    show_axes: bool | None = None,
    show_time: bool = True,
    time_location: str = "southwest",
    save: str | Path | None = None,
    **imshow_kwargs: Any,
):
    """Animate a time-domain receiver gather that lies on a 2D grid.

    Grid-backed receiver groups created with ``CoordsGrid`` are reshaped and
    labeled automatically, so ``grid_shape`` is only needed for plain arrays.
    """

    trace = as_trace_array(trace, caller="animate_gather")
    require_dims(trace, "time", "receiver")
    trace = select_time(trace, T_max)
    trace = trace.transpose("time", "receiver")

    manual_shape = grid_shape is not None
    shape = receiver_grid_shape(trace, grid_shape)
    values = trace_values(trace).reshape(trace.sizes["time"], *shape)
    time_values = coordinate_values(trace, "time", require_numeric=True)
    time_units = _time_units(trace)
    if manual_shape:
        values = values.transpose(0, 2, 1)
    display = None if grid_shape is not None else _grid_display(trace)
    A = _amplitude_limit(values, A)

    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if display is not None and not display.get("uniform", True):
        mesh_kwargs = {
            key: value
            for key, value in imshow_kwargs.items()
            if key not in {"origin", "interpolation"}
        }
        image = ax.pcolormesh(
            display["x"],
            display["y"],
            values[0],
            cmap=cmap,
            vmin=-A,
            vmax=A,
            shading="auto",
            **mesh_kwargs,
        )
        image.set_animated(True)
        ax.invert_yaxis()
    else:
        image = ax.imshow(
            values[0],
            origin="upper",
            cmap=cmap,
            vmin=-A,
            vmax=A,
            extent=None if display is None else display["extent"],
            animated=True,
            **imshow_kwargs,
        )
    if show_axes is None:
        show_axes = display is not None
    if show_axes and display is not None:
        ax.set_xlabel(display["xlabel"])
        ax.set_ylabel(display["ylabel"])
    elif not show_axes:
        ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    time_text = (
        _add_time_box(
            ax,
            _format_time_label(float(time_values[0]), time_units),
            location=time_location,
        )
        if show_time
        else None
    )

    def update(frame: int):
        if display is not None and not display.get("uniform", True):
            image.set_array(values[frame].ravel())
        else:
            image.set_array(values[frame])
        if time_text is not None:
            time_text.set_text(
                _format_time_label(float(time_values[frame]), time_units)
            )
        return _animation_artists(image, time_text)

    ani = _animation().FuncAnimation(
        fig,
        update,
        frames=values.shape[0],
        interval=interval,
        blit=True,
    )

    if save is not None:
        ani.save(str(save))
    return ani


def _animate_frequency_wavefield(
    trace: xr.DataArray,
    *,
    ax=None,
    A: float | None = None,
    cmap: str = "RdBu_r",
    interval: int = 50,
    figsize: tuple[float, float] = (5, 5),
    frames: int = 30,
    cycles: float = 1.0,
    show_axes: bool | None = None,
    show_time: bool = True,
    time_location: str = "southwest",
    save: str | Path | None = None,
    show: bool = False,
    title: str | None = None,
    **imshow_kwargs: Any,
):
    """Animate one frequency-domain wavefield as a time-harmonic field."""

    if frames < 1:
        raise ValueError("frames must be >= 1")
    if cycles <= 0:
        raise ValueError("cycles must be > 0")

    trace, frequency_value = _single_frequency_wavefield(
        trace,
        caller="animate_wavefield",
    )
    field, display = _frequency_wavefield_grid(trace)
    A = _amplitude_limit(field, A)

    if frequency_value is not None and frequency_value == 0:
        phases = np.zeros(int(frames), dtype=float)
    else:
        phases = np.linspace(0.0, 2.0 * np.pi * cycles, int(frames), endpoint=False)
    if frequency_value is not None and frequency_value > 0:
        time_values = phases / (2.0 * np.pi * frequency_value)
        time_units = "s"
    else:
        time_values = np.zeros_like(phases)
        time_units = "s"

    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    initial = np.real(field * np.exp(1j * phases[0]))
    if display is not None and not display.get("uniform", True):
        mesh_kwargs = {
            key: value
            for key, value in imshow_kwargs.items()
            if key not in {"origin", "interpolation"}
        }
        image = ax.pcolormesh(
            display["x"],
            display["y"],
            initial,
            cmap=cmap,
            vmin=-A,
            vmax=A,
            shading="auto",
            **mesh_kwargs,
        )
        image.set_animated(True)
        ax.invert_yaxis()
    else:
        image = ax.imshow(
            initial,
            origin="upper",
            cmap=cmap,
            vmin=-A,
            vmax=A,
            extent=None if display is None else display["extent"],
            animated=True,
            **imshow_kwargs,
        )
    if show_axes is None:
        show_axes = display is not None
    if show_axes and display is not None:
        ax.set_xlabel(display["xlabel"])
        ax.set_ylabel(display["ylabel"])
    elif not show_axes:
        ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    time_text = (
        _add_time_box(
            ax,
            _format_time_label(float(time_values[0]), time_units),
            location=time_location,
        )
        if show_time
        else None
    )

    def update(frame: int):
        values = np.real(field * np.exp(1j * phases[frame]))
        if display is not None and not display.get("uniform", True):
            image.set_array(values.ravel())
        else:
            image.set_array(values)
        if time_text is not None:
            time_text.set_text(
                _format_time_label(float(time_values[frame]), time_units)
            )
        return _animation_artists(image, time_text)

    ani = _animation().FuncAnimation(
        fig,
        update,
        frames=len(phases),
        interval=interval,
        blit=True,
    )

    if save is not None:
        ani.save(str(save))
    if show:
        plt.show()
    return ani


def animate_wavefield(trace: xr.DataArray, **kwargs: Any):
    """Animate grid-backed wavefield data.

    Time-domain traces are animated frame-by-frame. Frequency-domain traces
    must contain exactly one frequency and are animated as time-harmonic fields
    by plotting ``real(trace * exp(1j * omega * t))``.
    """

    unsupported = {"frequency", "frequency_index", "grid_shape"}.intersection(kwargs)
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(
            f"animate_wavefield does not accept {names}. Select one frequency "
            "before animating and use the receiver grid from the output metadata."
        )
    if not isinstance(trace, xr.DataArray):
        raise TypeError("animate_wavefield expects an xarray.DataArray")
    if "time" in trace.dims:
        return animate_gather(trace, **kwargs)
    if "frequency" in trace.dims or "receiver" in trace.dims:
        return _animate_frequency_wavefield(trace, **kwargs)
    raise ValueError("trace is missing required dimension: 'time' or 'frequency'")
