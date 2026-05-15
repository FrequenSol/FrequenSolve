"""Plotting helpers for layered seismic models."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import xarray as xr

from frequensolve._optional import optional_dependency_error
from frequensolve.model.property import canonical_property_name

__all__ = ["plot_layered_model"]


def plot_layered_model(model: Any, property: str, resolution: List[int], **kwargs):
    """Plot a layered model property.

    This keeps plotting and optional visualization imports out of the model
    authoring class while preserving ``LayeredModel.plot(...)`` as the user API.
    """

    if model.dimension == 3:
        return _plot_layered_model_3d(model, property, resolution, **kwargs)
    if model.dimension == 2:
        return _plot_layered_model_2d(model, property, resolution, **kwargs)
    raise NotImplementedError(f"Plotting is not implemented for {model.dimension}D")


def _plot_layered_model_3d(model: Any, property: str, resolution: List[int], **kwargs):
    try:
        import pyvista as pv

        if pv.OFF_SCREEN:
            pv.set_jupyter_backend("static")
    except ModuleNotFoundError as exc:
        raise optional_dependency_error(
            "3D layered-model plotting",
            extra="visual",
            dependencies=("pyvista",),
            error=exc,
        ) from exc

    if len(resolution) == 2:
        resolution = [*resolution, resolution[1]]

    property_key = canonical_property_name(property)
    axes_units = kwargs.pop("axes_units", {"x": "km", "y": "km", "z": "km"})
    samples = model.sample_uniform(resolution, axes_units=axes_units)[property_key]
    units = kwargs.pop("units", samples.attrs.get("units"))
    samples = model.convert_property_units(samples, property_key, units)
    x = samples.coords["x"].values
    y = samples.coords["y"].values
    z = samples.coords["z"].values

    vmin, vmax = model.extreme_values(property_key, units=units)
    vmin = kwargs.pop("vmin", vmin)
    vmax = kwargs.pop("vmax", vmax)
    cmap = kwargs.pop("cmap", "viridis")
    label = kwargs.pop("label", property)
    if units:
        label = f"{label} [{units}]"

    mx, my, mz = np.meshgrid(x, y, z, indexing="ij")
    grid = pv.StructuredGrid(mx, my, mz)
    grid[property] = samples.values.flatten(order="F")

    plotter = pv.Plotter()
    scalar_bar_args = dict(
        vertical=True,
        height=0.5,
        width=0.05,
        position_x=0.85,
        position_y=0.25,
        title_font_size=14,
        label_font_size=12,
        fmt="%.2f",
        title=label,
        color="black",
    )

    if kwargs.pop("slices", True):
        plotter.add_mesh_slice_orthogonal(
            grid,
            scalars=property,
            cmap=cmap,
            rng=[vmin, vmax],
            scalar_bar_args=scalar_bar_args,
        )
    else:
        plotter.add_mesh(
            grid,
            scalars=property,
            cmap=cmap,
            opacity=kwargs.pop("opacity", 1.0),
            rng=[vmin, vmax],
            scalar_bar_args=scalar_bar_args,
        )

    plotter.set_scale(zscale=kwargs.pop("z_scale", 1))

    if kwargs.pop("surfaces", True):
        for surface in model.surfaces:
            try:
                sx, sy = np.meshgrid(x, y, indexing="ij")
                surf_coords = xr.DataArray(dims=["x", "y"], coords={"x": x, "y": y})
                sz = surface.depth.get(surf_coords)
                z_units = axes_units.get("z") if axes_units else None
                if z_units is not None:
                    source_units = surface.depth.units or sz.attrs.get("units")
                    if source_units is not None and source_units != z_units:
                        from frequensolve.units import ureg

                        sz = sz.copy(deep=True)
                        sz.data = (sz.data * ureg(source_units)).to(z_units).magnitude
                sz = sz.values
                sgrid = pv.StructuredGrid(sx, sy, sz)
                plotter.add_mesh(
                    sgrid,
                    color="black",
                    opacity=0.3,
                    style="wireframe",
                )
            except Exception:
                continue

    plotter.show_grid(
        font_size=kwargs.pop("fontsize", 12),
        xtitle="X",
        ytitle="Y",
        ztitle="Z",
    )
    plotter.show_axes()
    if kwargs.pop("interactive", False):
        return plotter.show()
    return plotter.show(jupyter_backend="static")


def _plot_layered_model_2d(model: Any, property: str, resolution: List[int], **kwargs):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise optional_dependency_error(
            "2D layered-model plotting",
            extra="visual",
            dependencies=("matplotlib",),
            error=exc,
        ) from exc

    property_key = canonical_property_name(property)
    axes_units = kwargs.pop("axes_units", {"x": "km", "z": "km"})
    samples = model.sample_uniform(resolution, axes_units=axes_units)[property_key]

    units = kwargs.pop("units", samples.attrs.get("units"))
    samples = model.convert_property_units(samples, property_key, units)
    units = samples.attrs.get("units")
    label = kwargs.pop("label", property)
    origin = kwargs.pop("origin", "upper")
    aspect = kwargs.pop("aspect", None)
    axes_names = kwargs.pop("axes_names", {"x": "X", "z": "Depth"})
    add_colorbar = kwargs.pop("add_colorbar", True)

    show_surfaces = kwargs.pop("surfaces", True)
    show_boreholes = kwargs.pop("boreholes", False)
    surface_kwargs = {
        "color": kwargs.pop("linecolor", "k"),
        "linestyle": kwargs.pop("linestyle", "-"),
        "linewidth": kwargs.pop("linewidth", 1),
    }
    borehole_kwargs = kwargs.pop("borehole_kwargs", {})

    acquisition = kwargs.pop("acquisition", None)
    scatter_kwargs = kwargs.pop("scatter_kwargs", {})

    figsize = kwargs.pop("figsize", None)
    fontsize = kwargs.pop("fontsize", 12)
    save = kwargs.pop("save", None)
    dpi = kwargs.pop("dpi", None)

    plt.rcParams.update({"font.size": fontsize})

    show = True
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
        show = False
    else:
        fig = plt.figure(**({"figsize": figsize} if figsize is not None else {}))
        ax = fig.gca()

    _annotate_samples(samples, units, label, axes_names, axes_units)

    vmin, vmax = model.extreme_values(property_key, units=units)
    vmin = kwargs.pop("vmin", vmin)
    vmax = kwargs.pop("vmax", vmax)

    samples = samples.clip(min=vmin, max=vmax)
    image = samples.plot.imshow(
        ax=ax,
        x="x",
        vmin=vmin,
        vmax=vmax,
        extend="neither",
        add_colorbar=add_colorbar,
        **kwargs,
    )

    if show_surfaces:
        limits = {"x": model.x_limits}
        if model.y_limits is not None:
            limits["y"] = model.y_limits
        for surface in model.surfaces:
            surface.plot(
                limits=limits,
                ax=ax,
                units=axes_units.get("z") if axes_units else None,
                **surface_kwargs,
            )

    if show_boreholes and getattr(model, "boreholes", None):
        _plot_boreholes(model, ax, axes_units=axes_units, **borehole_kwargs)

    if acquisition is not None:
        _plot_acquisition(model, acquisition, ax, **scatter_kwargs)
    if aspect == "equal":
        ax.set_aspect("equal")
    if origin == "upper":
        ax.invert_yaxis()
    if save is not None:
        plt.savefig(
            save,
            bbox_inches="tight",
            **({"dpi": dpi} if dpi is not None else {}),
        )
        plt.close()
        return image
    if show:
        plt.show()
    return image


def _length_value(value: Any, units: str | None = None) -> float:
    if isinstance(value, dict):
        raw = value.get("value")
        if raw is None:
            raise ValueError("Length mappings require a value")
        source_units = value.get("units")
        if units is not None and source_units is not None:
            from frequensolve.units import ureg

            return float((float(raw) * ureg(source_units)).to(units).magnitude)
        return float(raw)
    if hasattr(value, "to") and hasattr(value, "magnitude"):
        return (
            float(value.to(units).magnitude)
            if units is not None
            else float(value.magnitude)
        )
    return float(value)


def _surface_by_ref(model: Any, ref: Any) -> Any:
    if hasattr(ref, "depth") and hasattr(ref, "name"):
        return ref
    if isinstance(ref, dict):
        if "surface" in ref:
            ref = ref["surface"]
        elif "index" in ref:
            ref = ref["index"]
    if isinstance(ref, int):
        return model.surfaces[ref - 1]
    if isinstance(ref, str):
        if ref == "top":
            return model.surfaces[0]
        if ref == "bottom":
            return model.surfaces[-1]
        if ref.startswith("surface_"):
            try:
                return model.surfaces[int(ref.split("_", 1)[1]) - 1]
            except (ValueError, IndexError):
                pass
        for surface in model.surfaces:
            if surface.name == ref:
                return surface
    raise ValueError(f"Unknown borehole extent surface reference: {ref!r}")


def _surface_depth_at_x(surface: Any, x: float, units: str | None = None) -> float:
    coords = xr.DataArray(dims=["x"], coords={"x": [x]})
    depth = surface.depth.get(coords)
    value = float(np.asarray(depth.values).reshape(-1)[0])
    source_units = surface.depth.units or depth.attrs.get("units")
    if units is not None and source_units is not None:
        from frequensolve.units import ureg

        value = float((value * ureg(source_units)).to(units).magnitude)
    return value


def _plot_boreholes(
    model: Any,
    ax: Any,
    *,
    axes_units: Dict[str, str] | None = None,
    color: str = "white",
    axis_color: str = "black",
    linewidth: float = 1.2,
    linestyle: str = "--",
    alpha: float = 0.95,
    label: bool = True,
    labels: bool | None = None,
    fill: bool = False,
    fill_alpha: float = 0.12,
    colors: List[str] | None = None,
) -> None:
    if labels is not None:
        label = labels
    x_units = axes_units.get("x") if axes_units else None
    z_units = axes_units.get("z") if axes_units else None
    palette = colors or ["#76b7b2", "#f28e2b", "#59a14f", "#e15759", "#4e79a7"]

    for borehole in model.boreholes:
        try:
            axis_x = borehole.axis_x(x_units)
            top = _surface_by_ref(model, borehole.extent["top"])
            bottom = _surface_by_ref(model, borehole.extent["bottom"])
            z_top = _surface_depth_at_x(top, axis_x, z_units)
            z_bottom = _surface_depth_at_x(bottom, axis_x, z_units)
        except Exception:
            continue

        z0, z1 = sorted([z_top, z_bottom])
        ax.plot(
            [axis_x, axis_x],
            [z0, z1],
            color=axis_color,
            linestyle=":",
            linewidth=linewidth,
            alpha=alpha,
            label=f"{borehole.name} axis" if label else None,
            zorder=5,
        )

        previous = None
        for ipart, part in enumerate(borehole.parts):
            try:
                z_values, radii = borehole.radius_profile(
                    part,
                    units=x_units,
                    depth_units=z_units,
                )
            except Exception:
                continue
            if len(radii) == 1:
                z_plot = np.asarray([z0, z1], dtype=float)
                radii_plot = np.full_like(z_plot, radii[0], dtype=float)
            else:
                inner = z_values[(z_values >= z0) & (z_values <= z1)]
                z_plot = np.unique(np.concatenate(([z0], inner, [z1])))
                radii_plot = np.interp(z_plot, z_values, radii)

            left = axis_x - radii_plot
            right = axis_x + radii_plot
            color_i = palette[ipart % len(palette)]
            for side in (left, right):
                if np.all(side < model.x_limits[0]) or np.all(side > model.x_limits[1]):
                    continue
                ax.plot(
                    side,
                    z_plot,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=5,
                )
            if fill:
                inner_radius = (
                    np.zeros_like(radii_plot)
                    if previous is None
                    else np.interp(z_plot, previous[0], previous[1])
                )
                ax.fill_betweenx(
                    z_plot,
                    axis_x + inner_radius,
                    right,
                    color=color_i,
                    alpha=fill_alpha,
                    zorder=4,
                )
                ax.fill_betweenx(
                    z_plot,
                    left,
                    axis_x - inner_radius,
                    color=color_i,
                    alpha=fill_alpha,
                    zorder=4,
                )
            if label:
                mid = len(z_plot) // 2
                ax.text(
                    axis_x + radii_plot[mid],
                    z_plot[mid],
                    part.name,
                    color=color,
                    fontsize=8,
                    ha="left",
                    va="center",
                    zorder=6,
                )
            previous = (z_plot, radii_plot)


def _annotate_samples(
    samples: xr.DataArray,
    units: str | None,
    label: str | None,
    axes_names: Dict[str, str] | None,
    axes_units: Dict[str, str] | None,
) -> None:
    if units is not None:
        samples.attrs["units"] = units
    if label is not None:
        samples.attrs["long_name"] = label
    if axes_names is not None:
        samples.coords["x"].attrs["long_name"] = axes_names["x"]
        samples.coords["z"].attrs["long_name"] = axes_names["z"]
        if axes_units is not None:
            samples.coords["x"].attrs["units"] = axes_units["x"]
            samples.coords["z"].attrs["units"] = axes_units["z"]


def _plot_acquisition(model: Any, acquisition: Any, ax, **kwargs) -> None:
    colors = ["b", "g", "orange", "y", "c", "m"]

    plot_sources = kwargs.pop("plot_sources", True)
    groups = kwargs.pop("groups", None)
    if groups is None:
        groups = acquisition.receiver_groups

    for igrp, group in enumerate(groups):
        coords = group.coordinates.get()

        ax.scatter(
            coords[:, 0],
            coords[:, -1],
            marker=".",
            s=30,
            label=f"Receivers ({group.name})",
            zorder=6,
            color=colors[igrp % len(colors)],
            **kwargs,
        )

    if not plot_sources:
        return

    for igrp, group in enumerate(acquisition.source_groups):
        coords = group.coordinates()

        ax.scatter(
            coords[:, 0],
            coords[:, -1],
            marker="*",
            s=120,
            label="Sources" if igrp == 0 else None,
            zorder=7,
            facecolors="#fff700",
            edgecolors="r",
            linewidths=0.5,
            **kwargs,
        )
        ax.legend(bbox_to_anchor=(0, 1.02), loc="lower left")
