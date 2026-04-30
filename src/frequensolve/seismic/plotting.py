"""Plotting helpers for seismic trace arrays."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from frequensolve.seismic.analysis import compute_timelag, phase_velocity_transform
from frequensolve.seismic.animate import animate_gather
from frequensolve.seismic.trace_geometry import (
    as_trace_array,
    coordinate_label,
    coordinate_values,
    ensure_monotonic,
    receiver_axis,
    require_dims,
    select_time,
    time_limit,
    trace_values,
)

__all__ = [
    "plot_gather",
    "animate_gather",
    "plot_gather_diff",
    "plot_xf",
    "plot_cf",
    "plot_timelag",
]


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _amplitude_limit(values: np.ndarray, limit: float | None) -> float:
    if limit is not None:
        return float(limit)
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


def _prepare_time_gather(
    trace: xr.DataArray,
    *,
    T_max: float | None = None,
    Tf: float | None = None,
) -> xr.DataArray:
    trace = select_time(as_trace_array(trace), time_limit(T_max, Tf))
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


def _apply_font_size(fontsize: float | None) -> None:
    if fontsize is not None:
        _pyplot().rcParams.update({"font.size": fontsize})


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
    L_scale: float = 1.0,
    T_scale: float = 1.0,
    aspect: str | float = "auto",
    interpolation: str = "bilinear",
    title: str | None = None,
    save: str | Path | None = None,
    show: bool = False,
    colorbar: bool = False,
    **imshow_kwargs: Any,
):
    """Plot a time-domain trace gather.

    ``trace`` must have ``time`` and ``receiver`` dimensions.  The function
    returns ``(fig, ax)`` and never mutates the input trace.
    """

    trace = _prepare_time_gather(trace, T_max=T_max, Tf=Tf)
    values = trace_values(trace)
    x_axis = receiver_axis(trace, units=units, scale=L_scale)
    time = coordinate_values(trace, "time", scale=T_scale, require_numeric=True)
    limit = _amplitude_limit(values, A)

    plt = _pyplot()
    _apply_font_size(fontsize)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    image = ax.imshow(
        values,
        origin="upper",
        cmap=cmap,
        extent=_image_extent(x_axis.values, time),
        vmin=-limit,
        vmax=limit,
        aspect=aspect,
        interpolation=interpolation,
        **imshow_kwargs,
    )
    ax.set_xlabel(x_axis.label)
    ax.set_ylabel(coordinate_label(trace, "time", "Time"))
    if title is not None:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(image, ax=ax)

    _finalize_figure(fig, save=save, show=show)
    return fig, ax


def plot_gather_diff(
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
    L_scale: float = 1.0,
    T_scale: float = 1.0,
    aspect: str | float = "auto",
    amplify_diff: float = 1.0,
    titles: Sequence[str] = ("Baseline", "Monitor", "Difference"),
    stack: str = "horizontal",
    save: str | Path | None = None,
    show: bool = False,
    **imshow_kwargs: Any,
):
    """Plot baseline, monitor, and difference gathers."""

    baseline = _prepare_time_gather(baseline, T_max=T_max, Tf=Tf)
    monitor = _prepare_time_gather(monitor, T_max=T_max, Tf=Tf)
    baseline, monitor = xr.align(baseline, monitor, join="exact")

    data = [trace_values(baseline), trace_values(monitor)]
    data.append(amplify_diff * (data[1] - data[0]))
    limit = _amplitude_limit(np.stack(data), A)
    x_axis = receiver_axis(baseline, units=units, scale=L_scale)
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
        axis.imshow(
            values,
            origin="upper",
            cmap=cmap,
            extent=_image_extent(x_axis.values, time),
            vmin=-limit,
            vmax=limit,
            aspect=aspect,
            **imshow_kwargs,
        )
        axis.set_title(panel_title)
        axis.set_xlabel(x_axis.label)
    axes[0].set_ylabel(coordinate_label(baseline, "time", "Time"))

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
    save: str | Path | None = None,
    show: bool = False,
    colorbar: bool = False,
    **imshow_kwargs: Any,
):
    """Plot a frequency-domain receiver gather."""

    trace = _prepare_frequency_gather(trace)
    values_complex = trace_values(trace, complex_=True)
    if mode == "real":
        values = values_complex.real
        symmetric = True
    elif mode == "imag":
        values = values_complex.imag
        symmetric = True
    elif mode == "abs":
        values = np.abs(values_complex)
        symmetric = False
    else:
        raise ValueError("mode must be 'real', 'imag', or 'abs'")

    x_axis = receiver_axis(trace, units=units)
    frequency = coordinate_values(trace, "frequency", require_numeric=True)
    limit = _amplitude_limit(values, A)

    plt = _pyplot()
    _apply_font_size(fontsize)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    image = ax.imshow(
        values,
        origin="lower",
        cmap=cmap,
        extent=[
            float(x_axis.values[0]),
            float(x_axis.values[-1]),
            float(frequency[0]),
            float(frequency[-1]),
        ],
        vmin=-limit if symmetric else 0,
        vmax=limit,
        aspect="auto",
        **imshow_kwargs,
    )
    ax.set_xlabel(x_axis.label)
    ax.set_ylabel(coordinate_label(trace, "frequency", "Frequency"))
    if colorbar:
        fig.colorbar(image, ax=ax)

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
    save: str | Path | None = None,
    show: bool = False,
    colorbar: bool = False,
    **imshow_kwargs: Any,
):
    """Plot a frequency/phase-velocity diagnostic transform."""

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

    plt = _pyplot()
    _apply_font_size(fontsize)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    image = ax.imshow(
        values.T,
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        extent=[
            float(frequency[0]),
            float(frequency[-1]),
            float(velocity[0]),
            float(velocity[-1]),
        ],
        aspect="auto",
        **imshow_kwargs,
    )
    ax.set_xlabel(coordinate_label(transform, "frequency", "Frequency"))
    ax.set_ylabel(coordinate_label(transform, "phase_velocity", "Phase velocity"))
    if colorbar:
        fig.colorbar(image, ax=ax)

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
