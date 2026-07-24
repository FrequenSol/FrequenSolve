"""Plotting helpers for trace arrays."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from frequensolve._optional import optional_dependency_error
from frequensolve.plotting.analysis import compute_timelag, phase_velocity_transform
from frequensolve.plotting.animate import animate_gather, animate_wavefield
from frequensolve.seismic.trace_geometry import (
    _receiver_grid_display_shape,
    _selected_frequency_value,
    as_trace_array,
    coordinate_label,
    coordinate_values,
    ensure_monotonic,
    receiver_axis,
    require_dims,
    select_time,
    to_numpy,
    trace_values,
    wavefield_grid_display,
)

__all__ = [
    "plot_gather",
    "animate_gather",
    "animate_wavefield",
    "diff_gathers",
    "plot_xf",
    "plot_wavefield",
    "plot_cf",
    "plot_timelag",
]


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise optional_dependency_error(
            "Seismic plotting",
            extra="visual",
            dependencies=("matplotlib",),
            error=exc,
        ) from exc

    return plt


def _amplitude_limit(values: np.ndarray, limit: Any | None) -> float:
    if limit is not None:
        scalar = to_numpy(limit.data if isinstance(limit, xr.DataArray) else limit)
        if scalar.size != 1:
            raise ValueError("A must be a scalar")
        return float(scalar.reshape(-1)[0])
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    maximum = float(np.nanmax(np.abs(finite)))
    return maximum if maximum > 0 else 1.0


def _finalize_figure(fig, *, save: str | Path | None, show: bool) -> None:
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, bbox_inches="tight")
    if show:
        _pyplot().show()


def _image_extent(x: np.ndarray, y: np.ndarray) -> list[float]:
    return [float(x[0]), float(x[-1]), float(y[-1]), float(y[0])]


def _image_extent_for_origin(
    x: np.ndarray,
    y: np.ndarray,
    *,
    origin: str,
) -> list[float]:
    if origin == "upper":
        return _image_extent(x, y)
    return [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]


def _grid_display(
    trace: xr.DataArray,
    *,
    x: str | None = None,
    y: str | None = None,
):
    return wavefield_grid_display(trace, x=x, y=y)


def _prepare_time_gather(
    trace: xr.DataArray,
    *,
    T_max: float | None = None,
    Tf: float | None = None,
) -> xr.DataArray:
    trace = as_trace_array(trace, caller="plot_gather")
    if "time" not in trace.dims and "frequency" in trace.dims:
        raise ValueError(
            "plot_gather expects time-domain traces. Use traces.td(...) before "
            "plot_gather, or use plot_xf(...) for frequency-domain traces."
        )
    trace = select_time(trace, T_max)
    require_dims(trace, "time", "receiver")
    trace = trace.transpose("time", "receiver")
    time = coordinate_values(trace, "time", require_numeric=True)
    ensure_monotonic(time, name="time")
    return trace


def _prepare_frequency_gather(trace: xr.DataArray) -> xr.DataArray:
    trace = as_trace_array(trace)
    require_dims(trace, "frequency", "receiver")
    trace = trace.transpose("frequency", "receiver")
    frequency = coordinate_values(trace, "frequency", require_numeric=True)
    ensure_monotonic(frequency, name="frequency")
    return trace


def _prepare_frequency_wavefield(
    trace: xr.DataArray,
) -> tuple[xr.DataArray, float | None]:
    if not isinstance(trace, xr.DataArray):
        raise TypeError("plot_wavefield expects an xarray.DataArray")
    require_dims(trace, "receiver")
    if "frequency" in trace.dims:
        if trace.sizes["frequency"] != 1:
            raise ValueError(
                "plot_wavefield expects frequency-domain wavefields to contain "
                "exactly one frequency"
            )
        trace = trace.isel(frequency=0)
    frequency_value = _selected_frequency_value(trace)
    trace = as_trace_array(trace, caller="plot_wavefield")
    require_dims(trace, "receiver")
    if trace.dims != ("receiver",):
        remaining = ", ".join(repr(dim) for dim in trace.dims if dim != "receiver")
        raise ValueError(
            "plot_wavefield expects one selected wavefield component/source; "
            f"remaining dimension(s): {remaining}"
        )
    return trace.transpose("receiver"), frequency_value


def _frequency_wavefield_grid(
    trace: xr.DataArray,
    *,
    x: str | None = None,
    y: str | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    shape = _receiver_grid_display_shape(trace, caller="plot_wavefield")
    values = trace_values(trace, complex_=True).reshape(*shape)
    display = _grid_display(trace, x=x, y=y)
    if display is None and (x is not None or y is not None):
        raise ValueError("plot_wavefield x/y selectors require a 2D named grid")
    if display is not None and display.get("transpose", False):
        values = values.T
    return values, display


def _complex_mode_values(
    values_complex: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, bool]:
    if mode == "real":
        return values_complex.real, True
    if mode == "imag":
        return values_complex.imag, True
    if mode == "abs":
        return np.abs(values_complex), False
    raise ValueError("mode must be 'real', 'imag', or 'abs'")


def _apply_font_size(fontsize: float | None) -> None:
    if fontsize is not None:
        _pyplot().rcParams.update({"font.size": fontsize})


def _figure_axis(
    ax,
    *,
    figsize: tuple[float, float],
    fontsize: float | None,
):
    plt = _pyplot()
    _apply_font_size(fontsize)
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def _image_clim(
    values: np.ndarray,
    limit: Any | None,
    *,
    symmetric: bool,
) -> tuple[float, float]:
    amplitude = _amplitude_limit(values, limit)
    return (-amplitude if symmetric else 0.0, amplitude)


def _imshow(
    ax,
    values: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    origin: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    aspect: str | float,
    interpolation: str,
    **imshow_kwargs: Any,
):
    return ax.imshow(
        values,
        origin=origin,
        cmap=cmap,
        extent=_image_extent_for_origin(x, y, origin=origin),
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        interpolation=interpolation,
        **imshow_kwargs,
    )


def _decorate_image(
    fig,
    ax,
    image,
    *,
    title: str | None = None,
    grid: bool = False,
    colorbar: bool = False,
) -> None:
    if title is not None:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(image, ax=ax)
    if grid:
        ax.grid(True)


def plot_gather(
    trace: xr.DataArray,
    *,
    ax=None,
    A: float | None = None,
    units: str = "",
    cmap: str = "Greys",
    figsize: tuple[float, float] = (5, 5),
    fontsize: float | None = 12,
    Tf: float | None = None,
    T_max: float | None = None,
    T_scale: float = 1.0,
    aspect: str | float = "auto",
    interpolation: str = "bilinear",
    title: str | None = None,
    save: str | Path | None = None,
    show: bool = False,
    grid: bool = False,
    colorbar: bool = False,
    **imshow_kwargs: Any,
):
    """Plot a time-domain trace gather.

    ``trace`` must have ``time`` and ``receiver`` dimensions.  The function
    returns ``(fig, ax)`` and never mutates the input trace.
    """

    trace = _prepare_time_gather(trace, T_max=T_max, Tf=Tf)
    values = trace_values(trace)
    x_axis = receiver_axis(trace, units=units)
    time = coordinate_values(trace, "time", scale=T_scale, require_numeric=True)
    vmin, vmax = _image_clim(values, A, symmetric=True)

    fig, ax = _figure_axis(ax, figsize=figsize, fontsize=fontsize)
    image = _imshow(
        ax,
        values,
        origin="upper",
        cmap=cmap,
        x=x_axis.values,
        y=time,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        interpolation=interpolation,
        **imshow_kwargs,
    )
    ax.set_xlabel(x_axis.label)
    ax.set_ylabel(coordinate_label(trace, "time", "Time"))
    _decorate_image(fig, ax, image, title=title, grid=grid, colorbar=colorbar)

    _finalize_figure(fig, save=save, show=show)
    return fig, ax


def diff_gathers(
    baseline: xr.DataArray,
    monitor: xr.DataArray,
    *,
    ax=None,
    A: float | None = None,
    units: str = "",
    cmap: str = "Greys",
    figsize: tuple[float, float] = (10, 4),
    fontsize: float | None = 12,
    Tf: float | None = None,
    T_max: float | None = None,
    T_scale: float = 1.0,
    aspect: str | float = "auto",
    interpolation: str = "bilinear",
    amplify_diff: float = 1.0,
    titles: Sequence[str] = ("Baseline", "Monitor", "Difference"),
    title: str | None = None,
    stack: str = "horizontal",
    save: str | Path | None = None,
    show: bool = False,
    grid: bool = False,
    colorbar: bool = False,
    **imshow_kwargs: Any,
):
    """Plot baseline, monitor, and difference gathers.

    Args:
        baseline: Baseline time-domain gather.
        monitor: Monitor time-domain gather aligned to ``baseline``.
        ax: Optional sequence of three Matplotlib axes.
        A: Optional symmetric amplitude limit.
        units: Receiver axis display units label.
        cmap: Matplotlib colormap.
        figsize: Figure size when creating axes.
        fontsize: Optional global font size.
        Tf: Deprecated final-time alias.
        T_max: Optional final time to include.
        T_scale: Time-axis scale factor.
        aspect: Image aspect setting.
        interpolation: Image interpolation method.
        amplify_diff: Multiplicative factor applied to the difference panel.
        titles: Three panel titles.
        title: Optional figure title.
        stack: ``"horizontal"`` or ``"vertical"`` panel layout.
        save: Optional output image path.
        show: Whether to show the figure.
        grid: Whether to draw grid lines on each axes.
        colorbar: Whether to add colorbars.
        **imshow_kwargs: Additional image keyword arguments.

    Returns:
        ``(fig, axes)``.
    """

    baseline = _prepare_time_gather(baseline, T_max=T_max, Tf=Tf)
    monitor = _prepare_time_gather(monitor, T_max=T_max, Tf=Tf)
    baseline, monitor = xr.align(baseline, monitor, join="exact")

    data = [trace_values(baseline), trace_values(monitor)]
    data.append(amplify_diff * (data[1] - data[0]))
    vmin, vmax = _image_clim(np.stack(data), A, symmetric=True)
    x_axis = receiver_axis(baseline, units=units)
    time = coordinate_values(baseline, "time", scale=T_scale, require_numeric=True)

    plt = _pyplot()
    _apply_font_size(fontsize)
    if ax is None:
        if stack == "vertical":
            fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        elif stack == "horizontal":
            fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
        else:
            raise ValueError("stack must be 'horizontal' or 'vertical'")
    else:
        axes = np.asarray(ax).ravel()
        if axes.size != 3:
            raise ValueError("ax must contain three axes")
        fig = axes[0].figure

    for axis, values, panel_title in zip(axes, data, titles):
        image = _imshow(
            axis,
            values,
            origin="upper",
            cmap=cmap,
            x=x_axis.values,
            y=time,
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
            interpolation=interpolation,
            **imshow_kwargs,
        )
        _decorate_image(
            fig,
            axis,
            image,
            title=panel_title,
            grid=grid,
            colorbar=colorbar,
        )
        axis.set_xlabel(x_axis.label)
    axes[0].set_ylabel(coordinate_label(baseline, "time", "Time"))
    if title is not None:
        fig.suptitle(title)

    _finalize_figure(fig, save=save, show=show)
    return fig, axes


def plot_xf(
    trace: xr.DataArray,
    *,
    ax=None,
    A: float | None = None,
    units: str = "",
    cmap: str = "Greys",
    figsize: tuple[float, float] = (5, 5),
    fontsize: float | None = 12,
    mode: str = "real",
    aspect: str | float = "auto",
    interpolation: str = "bilinear",
    title: str | None = None,
    save: str | Path | None = None,
    show: bool = False,
    grid: bool = False,
    colorbar: bool = False,
    **imshow_kwargs: Any,
):
    """Plot a frequency-domain receiver gather.

    Args:
        trace: Frequency-domain gather with ``frequency`` and ``receiver``
            dimensions.
        ax: Optional Matplotlib axes.
        A: Optional amplitude limit.
        units: Receiver axis display units label.
        cmap: Matplotlib colormap.
        figsize: Figure size when creating axes.
        fontsize: Optional global font size.
        mode: Complex display mode, such as ``"real"``, ``"imag"``, or
            ``"abs"``.
        aspect: Image aspect setting.
        interpolation: Image interpolation method.
        title: Optional axes title.
        save: Optional output image path.
        show: Whether to show the figure.
        grid: Whether to draw grid lines.
        colorbar: Whether to add a colorbar.
        **imshow_kwargs: Additional image keyword arguments.

    Returns:
        ``(fig, ax)``.
    """

    trace = _prepare_frequency_gather(trace)
    values_complex = trace_values(trace, complex_=True)
    values, symmetric = _complex_mode_values(values_complex, mode=mode)

    x_axis = receiver_axis(trace, units=units)
    frequency = coordinate_values(trace, "frequency", require_numeric=True)
    vmin, vmax = _image_clim(values, A, symmetric=symmetric)

    fig, ax = _figure_axis(ax, figsize=figsize, fontsize=fontsize)
    image = _imshow(
        ax,
        values,
        origin="lower",
        cmap=cmap,
        x=x_axis.values,
        y=frequency,
        vmin=vmin,
        vmax=vmax,
        aspect=aspect,
        interpolation=interpolation,
        **imshow_kwargs,
    )
    ax.set_xlabel(x_axis.label)
    ax.set_ylabel(coordinate_label(trace, "frequency", "Frequency"))
    _decorate_image(fig, ax, image, title=title, grid=grid, colorbar=colorbar)

    _finalize_figure(fig, save=save, show=show)
    return fig, ax


def plot_wavefield(
    trace: xr.DataArray,
    *,
    ax=None,
    x: str | None = None,
    y: str | None = None,
    A: float | None = None,
    cmap: str = "RdBu_r",
    figsize: tuple[float, float] = (5, 5),
    fontsize: float | None = 12,
    mode: str = "real",
    show_axes: bool | None = None,
    aspect: str | float = "auto",
    interpolation: str = "nearest",
    title: str | None = None,
    save: str | Path | None = None,
    show: bool = False,
    colorbar: bool = False,
    **imshow_kwargs: Any,
):
    """Plot one grid-backed frequency-domain wavefield.

    ``trace`` must contain one selected component/source and either exactly one
    ``frequency`` sample or a scalar ``frequency`` coordinate. The receiver
    grid is read from the wavefield output metadata. Use ``x`` and ``y`` to
    select which named grid dimensions appear on each plot axis; specifying one
    infers the other.
    """

    unsupported = {"frequency", "frequency_index", "grid_shape"}.intersection(
        imshow_kwargs
    )
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise TypeError(
            f"plot_wavefield does not accept {names}. Select one frequency "
            "before plotting and use the receiver grid from the output metadata."
        )
    trace, frequency_value = _prepare_frequency_wavefield(trace)
    values_complex, display = _frequency_wavefield_grid(trace, x=x, y=y)
    if display is not None:
        dimensions = set(display.get("dimensions", ()))
        selector_typos = [
            name
            for name, value in imshow_kwargs.items()
            if name in dimensions and value == name
        ]
        if selector_typos:
            name = selector_typos[0]
            if x is not None and y is None:
                correction = f"use y={name!r}, or omit it because y is inferred"
            elif y is not None and x is None:
                correction = f"use x={name!r}, or omit it because x is inferred"
            elif x is None and y is None:
                correction = f"use x={name!r} or y={name!r}"
            else:
                correction = "remove the extra selector"
            raise TypeError(
                f"plot_wavefield received {name}={name!r}; grid dimensions are "
                f"selected with the x= and y= parameters. Please {correction}."
            )
    values, symmetric = _complex_mode_values(values_complex, mode=mode)
    limit = _amplitude_limit(values, A)

    fig, ax = _figure_axis(ax, figsize=figsize, fontsize=fontsize)

    if display is not None and not display.get("uniform", True):
        mesh_kwargs = {
            key: value
            for key, value in imshow_kwargs.items()
            if key not in {"origin", "interpolation"}
        }
        image = ax.pcolormesh(
            display["x"],
            display["y"],
            values,
            cmap=cmap,
            vmin=-limit if symmetric else 0,
            vmax=limit,
            shading="auto",
            **mesh_kwargs,
        )
        ax.invert_yaxis()
        ax.set_aspect(aspect)
    else:
        image = ax.imshow(
            values,
            origin="upper",
            cmap=cmap,
            vmin=-limit if symmetric else 0,
            vmax=limit,
            extent=None if display is None else display["extent"],
            aspect=aspect,
            interpolation=interpolation,
            **imshow_kwargs,
        )
    if show_axes is None:
        show_axes = display is not None
    if show_axes and display is not None:
        ax.set_xlabel(display["xlabel"])
        ax.set_ylabel(display["ylabel"])
    elif not show_axes:
        ax.set_axis_off()
    default_title = (
        None
        if frequency_value is None
        else f"{mode} wavefield at {frequency_value:g} Hz"
    )
    _decorate_image(
        fig,
        ax,
        image,
        title=title if title is not None else default_title,
        colorbar=colorbar,
    )

    _finalize_figure(fig, save=save, show=show)
    return fig, ax


def plot_cf(
    trace: xr.DataArray,
    *,
    ax=None,
    A: float | None = 1.0,
    units: str = "",
    cmap: str = "RdYlBu_r",
    figsize: tuple[float, float] = (5, 5),
    fontsize: float | None = 14,
    c_min: float = 0.5,
    c_max: float = 6.0,
    n_c: int = 500,
    smooth: float | None = None,
    aspect: str | float = "auto",
    interpolation: str = "bilinear",
    title: str | None = None,
    save: str | Path | None = None,
    show: bool = False,
    grid: bool = False,
    colorbar: bool = False,
    **imshow_kwargs: Any,
):
    """Plot a frequency/phase-velocity diagnostic transform.

    Args:
        trace: Frequency-domain receiver gather.
        ax: Optional Matplotlib axes.
        A: Optional amplitude limit.
        units: Receiver/offset display units label.
        cmap: Matplotlib colormap.
        figsize: Figure size when creating axes.
        fontsize: Optional global font size.
        c_min: Minimum phase velocity.
        c_max: Maximum phase velocity.
        n_c: Number of phase-velocity samples.
        smooth: Optional smoothing parameter passed to the transform.
        aspect: Image aspect setting.
        interpolation: Image interpolation method.
        title: Optional axes title.
        save: Optional output image path.
        show: Whether to show the figure.
        grid: Whether to draw grid lines.
        colorbar: Whether to add a colorbar.
        **imshow_kwargs: Additional image keyword arguments.

    Returns:
        ``(fig, ax)``.
    """

    transform = phase_velocity_transform(
        trace,
        c_min=c_min,
        c_max=c_max,
        n_c=n_c,
        units=units,
        smooth=smooth,
    )
    values = trace_values(transform)
    frequency = coordinate_values(transform, "frequency", require_numeric=True)
    velocity = coordinate_values(transform, "phase_velocity", require_numeric=True)
    vmax = None if A is None else float(A) * _amplitude_limit(values, None)

    fig, ax = _figure_axis(ax, figsize=figsize, fontsize=fontsize)
    image = _imshow(
        ax,
        values.T,
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        x=frequency,
        y=velocity,
        aspect=aspect,
        interpolation=interpolation,
        **imshow_kwargs,
    )
    ax.set_xlabel(coordinate_label(transform, "frequency", "Frequency"))
    ax.set_ylabel(coordinate_label(transform, "phase_velocity", "Phase velocity"))
    _decorate_image(fig, ax, image, title=title, grid=grid, colorbar=colorbar)

    _finalize_figure(fig, save=save, show=show)
    return fig, ax


def plot_timelag(
    baseline: xr.DataArray,
    monitor: xr.DataArray,
    *,
    ax=None,
    units: str = "",
    figsize: tuple[float, float] = (5, 3),
    fontsize: float | None = 12,
    save: str | Path | None = None,
    show: bool = False,
    **kwargs: Any,
):
    """Plot receiver-by-receiver lag computed by ``analysis.compute_timelag``."""

    baseline = as_trace_array(baseline)
    lag = compute_timelag(baseline, monitor, **kwargs)
    x_axis = receiver_axis(baseline, units=units)

    plt = _pyplot()
    _apply_font_size(fontsize)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(x_axis.values, trace_values(lag), linewidth=2)
    ax.set_xlabel(x_axis.label)
    ax.set_ylabel("Lag [ms]")
    _finalize_figure(fig, save=save, show=show)
    return fig, ax
