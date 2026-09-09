"""Matplotlib plotting for Eikonal first-arrival fields and characteristics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from frequensolve.seismic.eikonal import EikonalResults

__all__ = ["plot_eikonal"]


def _matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except ImportError as exc:
        raise ImportError(
            "Eikonal plotting requires matplotlib; install frequensolve[visual]"
        ) from exc
    return plt, LineCollection, Line3DCollection


def plot_eikonal(
    results: Union[EikonalResults, str, Path],
    *,
    source: Union[int, str, None] = None,
    ax: Any = None,
    field: bool = True,
    characteristics: bool = True,
    levels: Union[int, np.ndarray] = 30,
    contours: bool = False,
    cmap: str = "viridis",
    characteristic_color: str = "white",
    characteristic_linewidth: float = 1.2,
    field_alpha: float = 0.9,
    show_source: bool = True,
    show_receivers: bool = True,
    colorbar: bool = True,
    invert_vertical: bool = False,
    equal_aspect: bool = True,
    title: Optional[str] = None,
    save: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Any:
    """Plot one 2D/3D first-arrival field and its characteristics.

    Args:
        results: Result handle, manifest, HDF5 file, or output directory.
        source: Source id or name. It may be omitted for a one-source result.
        ax: Optional Matplotlib axis.
        field: Draw the retained native-vertex travel-time field.
        characteristics: Draw retained winning-stencil backtracks.
        levels: Filled-contour levels used for a 2D field.
        contours: Overlay thin travel-time contour lines in 2D.
        cmap: Field colormap.
        characteristic_color: Characteristic line color.
        characteristic_linewidth: Characteristic line width.
        field_alpha: Field opacity.
        show_source: Draw the selected physical source.
        show_receivers: Draw receiver query points.
        colorbar: Add a travel-time colorbar when a field is drawn.
        invert_vertical: Reverse the second axis for depth-positive displays.
        equal_aspect: Use equal physical-axis scaling where supported.
        title: Optional title. The source name is used by default.
        save: Optional output image path.
        show: Call ``matplotlib.pyplot.show`` before returning.

    Returns:
        The Matplotlib axis containing the plot.
    """

    if not isinstance(results, EikonalResults):
        results = EikonalResults.open(results)
    if not field and not characteristics:
        raise ValueError("Enable field, characteristics, or both")
    source_row = results.source_index(source)
    source_id = int(results.sources["id"][source_row])
    source_name = str(results.sources["name"][source_row])
    dimension = results.dimension
    plt, LineCollection, Line3DCollection = _matplotlib()
    if ax is None:
        figure = plt.figure(figsize=(9, 6))
        ax = figure.add_subplot(111, projection="3d" if dimension == 3 else None)
    else:
        figure = ax.figure
        if dimension == 3 and not hasattr(ax, "get_zlim"):
            raise ValueError("Three-dimensional Eikonal results require a 3D axis")

    mappable = None
    all_positions = []
    if field:
        if not results.has_field:
            raise ValueError("Eikonal product does not retain vertex fields")
        selected = results.field(source_id)
        positions = selected.position[selected.reachable]
        travel_time = selected.travel_time[selected.reachable]
        if not len(positions):
            raise ValueError("Selected Eikonal field has no reachable vertices")
        all_positions.append(positions)
        if dimension == 2 and len(positions) >= 3:
            mappable = ax.tricontourf(
                positions[:, 0],
                positions[:, 1],
                travel_time,
                levels=levels,
                cmap=cmap,
                alpha=field_alpha,
            )
            if contours:
                ax.tricontour(
                    positions[:, 0],
                    positions[:, 1],
                    travel_time,
                    levels=levels,
                    colors="black",
                    linewidths=0.35,
                    alpha=0.45,
                )
        else:
            mappable = ax.scatter(
                *positions.T,
                c=travel_time,
                cmap=cmap,
                alpha=field_alpha,
                s=12 if dimension == 3 else 20,
            )

    segments = []
    if characteristics:
        for path in results.iter_characteristics(source_id=source_id):
            if len(path.position) < 2:
                continue
            segments.extend(np.stack((path.position[:-1], path.position[1:]), axis=1))
            all_positions.append(path.position)
        if segments:
            collection_class = Line3DCollection if dimension == 3 else LineCollection
            collection = collection_class(
                np.asarray(segments),
                colors=characteristic_color,
                linewidths=characteristic_linewidth,
            )
            ax.add_collection(collection)

    source_position = np.asarray(results.sources["position"])[
        source_row : source_row + 1
    ]
    receiver_position = np.asarray(
        results.receivers.get("position", np.empty((0, dimension)))
    ).reshape(-1, dimension)
    if show_source:
        ax.scatter(
            *source_position.T,
            marker="*",
            s=90,
            color="black",
            edgecolor="white",
            linewidth=0.5,
            label="source",
        )
    if show_receivers and len(receiver_position):
        ax.scatter(
            *receiver_position.T,
            marker="v",
            s=28,
            color="tab:red",
            label="receivers",
        )
    all_positions.extend([source_position, receiver_position])
    nonempty = [values for values in all_positions if len(values)]
    if not nonempty:
        raise ValueError("Eikonal product contains no plottable positions")
    combined = np.concatenate(nonempty, axis=0)
    lower = np.nanmin(combined, axis=0)
    upper = np.nanmax(combined, axis=0)
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

    coordinate_units = results.metadata.get("coordinate_units")
    suffix = f" ({coordinate_units})" if coordinate_units else ""
    ax.set_xlabel(f"x{suffix}")
    ax.set_ylabel(f"y{suffix}")
    if dimension == 3:
        ax.set_zlabel(f"z{suffix}")
    ax.set_title(title if title is not None else f"First arrivals: {source_name}")
    if mappable is not None and colorbar:
        time_units = results.metadata.get("travel_time_units")
        label = "travel time" + (f" ({time_units})" if time_units else "")
        figure.colorbar(mappable, ax=ax, label=label)
    if show_source or (show_receivers and len(receiver_position)):
        ax.legend()
    figure.tight_layout()
    if save is not None:
        figure.savefig(save, bbox_inches="tight")
    if show:
        plt.show()
    return ax
