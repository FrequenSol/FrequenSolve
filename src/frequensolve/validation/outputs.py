"""Output and job-request validation helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np

from frequensolve.model.layered.surfaces import Fracture
from frequensolve.model.property import canonical_property_name
from frequensolve.simulation.outputs import (
    JobOutputs,
    VtkItem,
    VtkOutput,
    WavefieldOutput,
)
from frequensolve.simulation.physics import components_for_physics
from frequensolve.util.fields import FIELD_PASSTHROUGH, canonical_field

from .geometry import (
    _model_property_names,
    _validate_direction,
    _validate_grid_like,
    _validate_projection_reference,
    _validate_system_reference,
    _validate_units,
    _validate_xarray_grid_payload,
    _ValidationContext,
)
from .report import ValidationReport

_ADVANCED_FIELD_PREFIXES = {"acoustic", "elastic", "poroelastic", "em", "EM"}
_FIELD_SELECTOR_SUFFIXES = {
    "x",
    "y",
    "z",
    "r",
    "theta",
    "phi",
    "xx",
    "yy",
    "zz",
    "xy",
    "xz",
    "yz",
    "rr",
    "rz",
    "tt",
    "div",
    "curl",
}


def _validate_outputs(outputs: JobOutputs, job: Any, ctx: _ValidationContext) -> None:
    if outputs.units is not None:
        _validate_output_units(outputs.units, ctx.report)

    if outputs.vtk and len(getattr(job, "f_list", []) or []) != 1:
        ctx.report.error(
            "outputs.vtk.frequency_count",
            "VTK outputs currently require a single-frequency job.",
            path="outputs.vtk",
            hint="Create one FrequencyDomainJob per plotted frequency.",
        )
    acquisition = getattr(ctx.simulation, "acquisition", None)
    source_count = (
        acquisition.known_source_field_count()
        if acquisition is not None and hasattr(acquisition, "known_source_field_count")
        else 0
    )
    for index, output in enumerate(outputs.vtk):
        _validate_paraview_output(output, index, source_count, ctx)
    for index, output in enumerate(outputs.wavefields):
        _validate_wavefield_output(output, index, source_count, ctx)


def _validate_output_units(units: Any, report: ValidationReport) -> None:
    for path, value in _iter_unit_values(units):
        _validate_units(value, path, report, code="outputs.units.invalid")
    for key in getattr(units, "dimensions", {}) or {}:
        if not key:
            report.error(
                "outputs.units.dimension_name.invalid",
                "Output unit dimension names must be non-empty.",
                path="outputs.Units.dimensions",
            )


def _validate_paraview_output(
    output: VtkOutput,
    index: int,
    source_count: Optional[int],
    ctx: _ValidationContext,
) -> None:
    path = f"outputs.vtk[{index}]"
    if not getattr(output, "name", None):
        ctx.report.error(
            "outputs.vtk.name.missing",
            "VTK outputs must have a non-empty name.",
            path=f"{path}.name",
        )
    for source_id in getattr(output, "sources", []) or []:
        _validate_source_id(source_id, source_count, f"{path}.sources", ctx.report)
    for field_index, requested_field in enumerate(getattr(output, "fields", []) or []):
        _validate_field(requested_field, f"{path}.fields[{field_index}]", ctx)
    _validate_requested_properties(
        getattr(output, "properties", None),
        f"{path}.properties",
        ctx,
    )
    for item_index, item in enumerate(getattr(output, "items", []) or []):
        _validate_paraview_item(item, f"{path}.items[{item_index}]", ctx)
    coordinates = getattr(output, "coordinates", None)
    if coordinates is not None:
        _validate_system_reference(coordinates, f"{path}.coordinates", ctx)
    target_coordinates = getattr(output, "target_coordinates", None)
    if target_coordinates is not None:
        _validate_system_reference(
            target_coordinates,
            f"{path}.target_coordinates",
            ctx,
        )
    grid = getattr(output, "grid_spec", None)
    if grid is not None:
        _validate_grid_like(grid, f"{path}.grid", ctx, require_domain=False)
    _validate_model_surface_selections(output, path, ctx)


def _validate_model_surface_selections(
    output: VtkOutput,
    path: str,
    ctx: _ValidationContext,
) -> None:
    selections: list[tuple[Any, str]] = [
        (selection, f"{path}.surfaces[{index}]")
        for index, selection in enumerate(getattr(output, "surfaces", []) or [])
    ]
    target = getattr(output, "target", None)
    if isinstance(target, Mapping):
        for index, selection in enumerate(target.get("selection", []) or []):
            selections.append((selection, f"{path}.target.selection[{index}]"))

    model = getattr(ctx.simulation, "model", None)
    names, surface_count = _paraview_model_surface_references(model)
    for selection, selection_path in selections:
        if isinstance(selection, str):
            _validate_model_surface_name(selection, selection_path, names, ctx)
            continue
        if isinstance(selection, (int, np.integer)):
            _validate_model_surface_index(selection, selection_path, surface_count, ctx)
            continue
        if isinstance(selection, (float, np.floating)):
            _validate_model_surface_index(selection, selection_path, surface_count, ctx)
            continue
        if not isinstance(selection, Mapping):
            continue
        kind = selection.get("kind")
        if kind not in {None, "model_surface"}:
            continue
        if "name" in selection:
            _validate_model_surface_name(
                selection.get("name"), f"{selection_path}.name", names, ctx
            )
        elif "index" in selection:
            _validate_model_surface_index(
                selection.get("index"),
                f"{selection_path}.index",
                surface_count,
                ctx,
            )
        elif kind == "model_surface":
            ctx.report.error(
                "outputs.vtk.model_surface.reference_missing",
                "Model-surface output must specify a surface name or one-based "
                "surface index.",
                path=selection_path,
            )


def _paraview_model_surface_references(model: Any) -> tuple[dict[str, str], int]:
    """Return ParaView-visible horizon names and the expanded horizon count."""

    names: dict[str, str] = {}
    surfaces = list(getattr(model, "surfaces", []) or [])
    expanded_count = 0

    def add_name(value: Any) -> None:
        name = str(value).strip() if value is not None else ""
        if name:
            names.setdefault(name.lower(), name)

    if surfaces:
        add_name("top")

    for surface in surfaces:
        base_name = str(getattr(surface, "name", "") or "").strip()
        if isinstance(surface, Fracture):
            extra = getattr(surface, "extra", {}) or {}
            for side in ("top", "bottom"):
                expanded_count += 1
                add_name(f"surface_{expanded_count}")
                generated_name = str(extra.get(f"{side}_name", "") or "").strip()
                if not generated_name and base_name:
                    generated_name = f"{base_name}_{side}"
                add_name(generated_name or f"surface_{expanded_count}")
            continue

        expanded_count += 1
        add_name(f"surface_{expanded_count}")
        add_name(base_name or f"surface_{expanded_count}")

    return names, expanded_count


def _validate_model_surface_name(
    value: Any,
    path: str,
    names: Mapping[str, str],
    ctx: _ValidationContext,
) -> None:
    name = str(value).strip() if value is not None else ""
    if not name:
        ctx.report.error(
            "outputs.vtk.model_surface.name_missing",
            "Model-surface output requires a non-empty surface name.",
            path=path,
        )
        return
    if name.lower() not in names:
        available_names = sorted(names.values(), key=str.lower)
        hint = (
            f"Available solver surface references are: "
            f"{', '.join(available_names)}."
            if names
            else "Attach a model with named surfaces or remove this selection."
        )
        ctx.report.error(
            "outputs.vtk.model_surface.unknown",
            f"Model-surface output references unknown surface {name!r}.",
            path=path,
            hint=hint,
        )


def _validate_model_surface_index(
    value: Any,
    path: str,
    surface_count: int,
    ctx: _ValidationContext,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        ctx.report.error(
            "outputs.vtk.model_surface.index_invalid",
            "Model-surface index must be an integer.",
            path=path,
        )
        return
    index = int(value)
    if index < 1 or index > surface_count:
        available = (
            f"the available one-based range 1..{surface_count}"
            if surface_count
            else "the model, which has no surfaces"
        )
        ctx.report.error(
            "outputs.vtk.model_surface.index_out_of_range",
            f"Model-surface index {index} is outside {available}.",
            path=path,
        )


def _validate_paraview_item(
    item: Any,
    path: str,
    ctx: _ValidationContext,
) -> None:
    if isinstance(item, VtkItem):
        kind = item.kind
        value = item.value
        units = item.units
        system = item.system
        basis = item.basis
        direction = item.direction
    elif isinstance(item, Mapping):
        kind = str(item.get("kind", "field"))
        value = item.get("field", item.get("property", item.get("info")))
        units = item.get("units")
        system = item.get("system", "global")
        basis = item.get("basis")
        direction = item.get("direction")
    else:
        ctx.report.error(
            "outputs.vtk.item.type.invalid",
            f"Unsupported VTK item type {type(item).__name__}.",
            path=path,
        )
        return

    if units is not None:
        _validate_units(
            units, f"{path}.units", ctx.report, code="outputs.units.invalid"
        )
    if kind == "field":
        _validate_field(value, f"{path}.field", ctx)
    elif kind == "property":
        _validate_requested_properties([value], f"{path}.property", ctx)
    elif kind != "info":
        ctx.report.error(
            "outputs.vtk.item.kind.unsupported",
            f"Unsupported VTK item kind {kind!r}.",
            path=f"{path}.kind",
        )
    if system is not None:
        _validate_system_reference(system, f"{path}.system", ctx)
    _validate_projection_reference(basis, f"{path}.basis", ctx)
    _validate_projection_reference(direction, f"{path}.direction", ctx)


def _validate_wavefield_output(
    output: WavefieldOutput,
    index: int,
    source_count: Optional[int],
    ctx: _ValidationContext,
) -> None:
    path = f"outputs.wavefields[{index}]"
    if not getattr(output, "name", None):
        ctx.report.error(
            "outputs.wavefield.name.missing",
            "Wavefield outputs must have a non-empty name.",
            path=f"{path}.name",
        )
    for source_id in getattr(output, "sources", []) or []:
        _validate_source_id(source_id, source_count, f"{path}.sources", ctx.report)
    for field_index, requested_field in enumerate(getattr(output, "fields", []) or []):
        _validate_field(requested_field, f"{path}.fields[{field_index}]", ctx)
    device = getattr(output, "device", None)
    if device is not None:
        for component_index, component in enumerate(device.components):
            component_path = f"{path}.device.components[{component_index}]"
            _validate_field(getattr(component, "field", None), component_path, ctx)
            if getattr(component, "units", None) is not None:
                _validate_units(
                    component.units,
                    f"{component_path}.units",
                    ctx.report,
                    code="outputs.wavefield.component_units.invalid",
                )
            _validate_direction(component.direction, f"{component_path}.direction", ctx)
    if output.grid is None:
        ctx.report.error(
            "outputs.wavefield.grid.missing",
            "WavefieldOutput requires a grid.",
            path=f"{path}.grid",
        )
        return
    _validate_xarray_grid_payload(output.grid, f"{path}.grid", ctx)


def _validate_field(field: Any, path: str, ctx: _ValidationContext) -> None:
    if field is None:
        ctx.report.error(
            "field.missing",
            "Field selection is missing.",
            path=path,
        )
        return
    value = canonical_field(str(field))
    if value in FIELD_PASSTHROUGH:
        return
    if _is_advanced_field(value):
        return
    try:
        registry = components_for_physics(str(ctx.simulation.physics))
    except Exception:
        return
    allowed = set(registry.allowed_components())
    if value not in allowed and not _is_component_selector(value, allowed):
        ctx.report.error(
            "field.unsupported",
            f"Field {field!r} is not supported by physics "
            f"{ctx.simulation.physics!r}.",
            path=path,
            hint=f"Allowed fields are: {', '.join(sorted(allowed))}.",
        )


def _is_advanced_field(value: str) -> bool:
    if ":" not in value:
        return False
    prefix = value.split(":", 1)[0]
    return prefix in _ADVANCED_FIELD_PREFIXES


def _is_component_selector(value: str, allowed: set[str]) -> bool:
    for field in allowed:
        prefix = f"{field}_"
        if value.startswith(prefix):
            return value.removeprefix(prefix) in _FIELD_SELECTOR_SUFFIXES
    return False


def _validate_requested_properties(
    properties: Optional[Iterable[Any]],
    path: str,
    ctx: _ValidationContext,
) -> None:
    requested = [canonical_property_name(str(item)) for item in properties or []]
    if not requested:
        return
    available = _model_property_names(ctx.simulation)
    if not available:
        ctx.report.warning(
            "outputs.property.model_empty",
            "Property output was requested, but the model has no declared "
            "properties to check against.",
            path=path,
        )
        return
    # unknown = sorted(set(requested).difference(available))
    # if unknown:
    #     ctx.report.error(
    #         "outputs.property.unknown",
    #         f"Requested model properties are not declared: {unknown}.",
    #         path=path,
    #         hint=f"Available properties are: {', '.join(sorted(available))}.",
    #     )


def _validate_source_id(
    source_id: Any,
    source_count: Optional[int],
    path: str,
    report: ValidationReport,
) -> None:
    try:
        value = int(source_id)
    except (TypeError, ValueError):
        report.error(
            "outputs.source_id.invalid",
            f"Source id must be an integer, got {source_id!r}.",
            path=path,
        )
        return
    if value < 1:
        report.error(
            "outputs.source_id.invalid",
            "Source ids are one-based and must be >= 1.",
            path=path,
        )
        return
    if source_count is not None and value > source_count:
        report.error(
            "outputs.source_id.out_of_range",
            f"Source id {value} is outside the available source range 1.."
            f"{source_count}.",
            path=path,
        )


def _validate_frequencies(
    f_list: Optional[Iterable[Any]],
    report: ValidationReport,
) -> None:
    if f_list is None:
        report.error("job.frequencies.missing", "Job requires f_list.")
        return
    values = list(f_list)
    if not values:
        report.error(
            "job.frequencies.empty",
            "Job requires at least one modeled frequency.",
            path="f_list",
        )
        return
    for index, value in enumerate(values):
        try:
            frequency = complex(value)
        except (TypeError, ValueError) as exc:
            report.error(
                "job.frequency.invalid",
                f"Frequency {value!r} is not numeric: {exc}",
                path=f"f_list[{index}]",
            )
            continue
        if not np.isfinite(frequency.real) or not np.isfinite(frequency.imag):
            report.error(
                "job.frequency.nonfinite",
                f"Frequency {value!r} is not finite.",
                path=f"f_list[{index}]",
            )
        if frequency.real <= 0.0:
            report.error(
                "job.frequency.nonpositive",
                f"Frequency real part must be positive, got {frequency.real:g}.",
                path=f"f_list[{index}]",
            )


def _iter_unit_values(units: Any):
    geometry = getattr(units, "geometry", None)
    if geometry is not None:
        yield "outputs.Units.geometry", geometry
    for key, value in getattr(units, "dimensions", {}).items():
        yield f"outputs.Units.dimensions.{key}", value
    for key, value in getattr(units, "fields", {}).items():
        yield f"outputs.Units.fields.{key}", value
    for key, value in getattr(units, "properties", {}).items():
        yield f"outputs.Units.properties.{key}", value
