"""Animation helpers for trace arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from frequensolve._optional import optional_dependency_error
from frequensolve.seismic.trace_geometry import (
    as_trace_array,
    receiver_grid_shape,
    require_dims,
    select_time,
    time_limit,
    trace_values,
)

__all__ = ["animate_gather"]


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


def animate_gather(
    trace: xr.DataArray,
    *,
    grid_shape: tuple[int, int] | None = None,
    ax=None,
    A: float | None = None,
    cmap: str = "Greys",
    interval: int = 50,
    figsize: tuple[float, float] = (5, 5),
    Tf: float | None = None,
    T_max: float | None = None,
    save: str | Path | None = None,
    show: bool = False,
    **imshow_kwargs: Any,
):
    """Animate a time-domain receiver gather that lies on a 2D grid."""

    trace = as_trace_array(trace, caller="animate_gather")
    require_dims(trace, "time", "receiver")
    trace = select_time(trace, time_limit(T_max, Tf))
    trace = trace.transpose("time", "receiver")

    shape = receiver_grid_shape(trace, grid_shape)
    values = trace_values(trace).reshape(trace.sizes["time"], *shape)
    if A is None:
        finite = values[np.isfinite(values)]
        A = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        if A == 0:
            A = 1.0

    plt = _pyplot()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    image = ax.imshow(
        values[0].T,
        origin="upper",
        cmap=cmap,
        vmin=-A,
        vmax=A,
        animated=True,
        **imshow_kwargs,
    )
    ax.set_axis_off()

    def update(frame: int):
        image.set_array(values[frame].T)
        return (image,)

    ani = _animation().FuncAnimation(
        fig,
        update,
        frames=values.shape[0],
        interval=interval,
        blit=True,
    )

    if save is not None:
        ani.save(str(save))
    if show:
        plt.show()
    return ani
