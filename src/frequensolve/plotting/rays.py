"""Matplotlib plotting for indexed acoustic ray results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from frequensolve._optional import optional_dependency_error
from frequensolve.seismic.rays import RayResults

__all__ = ["plot_rays"]


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ModuleNotFoundError as exc:
        raise optional_dependency_error(
            "Ray plotting",
            extra="visual",
            dependencies=("matplotlib",),
            error=exc,
        ) from exc
    return plt, LineCollection, Line3DCollection


def _color_values(path, color_by: Optional[str]) -> np.ndarray:
    count = max(0, len(path.position) - 1)
    if color_by is None:
        return np.empty(0)
    if color_by == "travel_time":
        tau = path.travel_time
        return 0.5 * (tau[:-1] + tau[1:])
    ray_fields = {
        "ray": "ray_id",
        "source": "source_id",
        "status": "status",
        "energy": "energy_weight",
        "branch_depth": "branch_depth",
    }
    if color_by not in ray_fields:
        choices = ", ".join(["travel_time", *ray_fields])
        raise ValueError(f"color_by must be None or one of {choices}")
    field = ray_fields[color_by]
    if field not in path.ray:
        raise ValueError(f"Ray product does not contain /rays/{field}")
    return np.full(count, float(path.ray[field]))


def _positions(table: dict[str, np.ndarray], dimension: int) -> np.ndarray:
    values = np.asarray(table.get("position", np.empty((0, dimension))))
    if values.size == 0:
        return np.empty((0, dimension))
    return values.reshape(-1, dimension)


def plot_rays(
    results: Union[RayResults, str, Path],
    *,
    ax=None,
    source_id: Optional[int] = None,
    color_by: Optional[str] = "travel_time",
    cmap: str = "viridis",
    color: str = "C0",
    linewidth: float = 1.0,
    alpha: float = 0.8,
    show_sources: bool = True,
    show_receivers: bool = True,
    show_hits: bool = False,
    colorbar: bool = True,
    invert_vertical: bool = False,
    equal_aspect: bool = True,
    title: Optional[str] = None,
    save: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """Plot retained 2D or 3D ray paths.

    Args:
        results: ``RayResults`` handle, manifest, HDF5 file, or result directory.
        ax: Optional Matplotlib 2D or 3D axis.
        source_id: Restrict paths to one stable source id.
        color_by: Segment coloring: ``travel_time``, ``ray``, ``source``,
            ``status``, ``energy``, ``branch_depth``, or ``None``.
        cmap: Matplotlib colormap for scalar coloring.
        color: Fixed line color when ``color_by`` is ``None``.
        linewidth: Ray line width.
        alpha: Ray line opacity.
        show_sources: Draw resolved source positions.
        show_receivers: Draw resolved receiver positions.
        show_hits: Draw captured receiver-hit positions.
        colorbar: Add a scalar colorbar when coloring rays.
        invert_vertical: Reverse the second coordinate axis in 2D.
        equal_aspect: Use equal physical-axis scaling where supported.
        title: Optional plot title.
        save: Optional image output path.
        show: Call ``matplotlib.pyplot.show`` before returning.

    Returns:
        The Matplotlib axis containing the plot.
    """

    if not isinstance(results, RayResults):
        results = RayResults.open(results)
    plt, LineCollection, Line3DCollection = _matplotlib()
    dimension = results.dimension
    if ax is None:
        figure = plt.figure(figsize=(9, 6))
        ax = figure.add_subplot(111, projection="3d" if dimension == 3 else None)
    else:
        figure = ax.figure
        if dimension == 3 and not hasattr(ax, "get_zlim"):
            raise ValueError("Three-dimensional ray results require a 3D axis")

    segments = []
    values = []
    for path in results.iter_paths(source_id=source_id):
        positions = path.position
        if len(positions) < 2:
            continue
        segments.extend(np.stack((positions[:-1], positions[1:]), axis=1))
        if color_by is not None:
            values.extend(_color_values(path, color_by))
    if not segments:
        raise ValueError("Ray product contains no retained path segments to plot")

    collection_class = Line3DCollection if dimension == 3 else LineCollection
    options: dict[str, Any] = {
        "linewidths": linewidth,
        "alpha": alpha,
    }
    if color_by is None:
        options["colors"] = color
    else:
        options["cmap"] = cmap
    collection = collection_class(np.asarray(segments), **options)
    if color_by is not None:
        collection.set_array(np.asarray(values, dtype=float))
    ax.add_collection(collection)

    source_positions = _positions(results.sources, dimension)
    receiver_positions = _positions(results.receivers, dimension)
    hit_positions = _positions(results.receiver_hits, dimension)
    if show_sources and len(source_positions):
        ax.scatter(
            *source_positions.T, marker="*", s=80, color="black", label="sources"
        )
    if show_receivers and len(receiver_positions):
        ax.scatter(
            *receiver_positions.T,
            marker="v",
            s=30,
            color="tab:red",
            label="receivers",
        )
    if show_hits and len(hit_positions):
        ax.scatter(
            *hit_positions.T,
            marker="o",
            s=20,
            color="tab:orange",
            label="receiver hits",
        )

    all_positions = np.concatenate(
        [
            np.asarray(segments).reshape(-1, dimension),
            source_positions,
            receiver_positions,
            hit_positions if show_hits else np.empty((0, dimension)),
        ],
        axis=0,
    )
    lower = np.nanmin(all_positions, axis=0)
    upper = np.nanmax(all_positions, axis=0)
    padding = np.maximum((upper - lower) * 0.03, 1.0e-12)
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    if dimension == 3:
        ax.set_zlim(lower[2] - padding[2], upper[2] + padding[2])
        if equal_aspect and hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect(np.maximum(upper - lower, 1.0e-12))
    elif equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if dimension == 2 and invert_vertical:
        ax.invert_yaxis()

    units = results.metadata.get("geometry_units")
    suffix = f" ({units})" if units else ""
    ax.set_xlabel(f"x{suffix}")
    ax.set_ylabel(f"y{suffix}")
    if dimension == 3:
        ax.set_zlabel(f"z{suffix}")
    if title is not None:
        ax.set_title(title)
    if color_by is not None and colorbar:
        label = "travel time"
        if color_by == "travel_time":
            time_units = results.metadata.get("time_units")
            if time_units:
                label += f" ({time_units})"
        else:
            label = color_by.replace("_", " ")
        figure.colorbar(collection, ax=ax, label=label)
    if (
        (show_sources and len(source_positions))
        or (show_receivers and len(receiver_positions))
        or (show_hits and len(hit_positions))
    ):
        ax.legend()
    figure.tight_layout()
    if save is not None:
        figure.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return ax
