"""Simulation, model, mesh, and acquisition validation helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from frequensolve.mesh.mesh_generators import BaseMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.property import Property, canonical_property_name
from frequensolve.seismic.receivers import (
    CoordsArray,
    CoordsFromFile,
    CoordsGrid,
    ReceiverGroup,
)
from frequensolve.util.physics import canonical_dimension, canonical_physics

from .geometry import (
    _coordinate_units_and_system,
    _coordinates_to_array,
    _coords_array_axes,
    _coords_file,
    _default_coordinate_axes,
    _default_length_units,
    _validate_bounds,
    _validate_coordinate_systems,
    _validate_coordinate_value_metadata,
    _validate_direction,
    _validate_points,
    _validate_system_reference,
    _validate_unit_config,
    _validate_units,
    _ValidationContext,
)
from .outputs import _validate_field
from .report import ValidationReport

_SOURCE_KINDS = {"scalar", "vector", "tensor", "monopole", "dipole"}


def _validate_simulation(ctx: _ValidationContext) -> None:
    simulation = ctx.simulation
    _validate_simulation_identity(simulation, ctx.report)
    _validate_coordinate_systems(ctx)
    _validate_unit_config(getattr(simulation, "units", None), ctx.report)
    _validate_model(getattr(simulation, "model", None), ctx)
    _validate_mesh(getattr(simulation, "mesh", None), ctx)
    _validate_acquisition(getattr(simulation, "acquisition", None), ctx)


def _validate_simulation_identity(simulation: Any, report: ValidationReport) -> None:
    name = getattr(simulation, "name", None)
    if not name:
        report.error("simulation.name.missing", "Simulation requires a non-empty name.")

    physics = getattr(simulation, "physics", None)
    if physics is None:
        report.error("simulation.physics.missing", "Simulation requires physics.")
    else:
        try:
            canonical_physics(str(physics))
        except ValueError as exc:
            report.error(
                "simulation.physics.unsupported",
                str(exc),
                path="physics",
            )

    dimension = getattr(simulation, "dimension", None)
    if dimension is None:
        report.error("simulation.dimension.missing", "Simulation requires dimension.")
    else:
        try:
            canonical_dimension(dimension)
        except ValueError as exc:
            report.error(
                "simulation.dimension.unsupported",
                str(exc),
                path="dimension",
            )


def _validate_model(model: Any, ctx: _ValidationContext) -> None:
    if model is None:
        ctx.report.warning(
            "model.missing",
            "Simulation has no model attached.",
            path="model",
        )
        return

    model_dim = getattr(model, "dimension", 0)
    if model_dim not in {0, ctx.dimension}:
        ctx.report.error(
            "model.dimension.mismatch",
            f"Model dimension {model_dim!r} does not match simulation model "
            f"dimension {ctx.dimension}.",
            path="model.dimension",
        )

    seen_ids: set[int] = set()
    for index, subdomain in enumerate(getattr(model, "subdomains", []) or []):
        sub_path = f"model.subdomains[{index}]"
        mesh_block_id = getattr(subdomain, "mesh_block_id", None)
        if mesh_block_id is not None and mesh_block_id >= 0:
            if mesh_block_id in seen_ids:
                ctx.report.error(
                    "model.subdomain.mesh_block_id.duplicate",
                    f"Mesh block id {mesh_block_id} is used more than once.",
                    path=f"{sub_path}.mesh_block_id",
                )
            seen_ids.add(mesh_block_id)
        _validate_property_map(getattr(subdomain, "properties", {}), sub_path, ctx)


def _validate_property_map(
    properties: Mapping[str, Property],
    path: str,
    ctx: _ValidationContext,
) -> None:
    for name, prop in properties.items():
        prop_path = f"{path}.properties.{canonical_property_name(name)}"
        units = getattr(prop, "units", None)
        if units:
            _validate_units(
                units,
                prop_path,
                ctx.report,
                code="model.property.units.invalid",
            )
        system = getattr(prop, "system", None)
        if system is not None:
            _validate_system_reference(system, f"{prop_path}.system", ctx)
        data = getattr(prop, "darr", None)
        if data is not None:
            for dim in getattr(data, "dims", ()):
                coord = data.coords.get(dim)
                if coord is not None:
                    coord_units = getattr(coord, "attrs", {}).get("units")
                    if coord_units:
                        _validate_units(
                            coord_units,
                            f"{prop_path}.coords.{dim}.units",
                            ctx.report,
                            code="model.property.coordinate_units.invalid",
                        )


def _validate_mesh(mesh: Any, ctx: _ValidationContext) -> None:
    if mesh is None:
        ctx.report.warning("mesh.missing", "Simulation has no mesh attached.")
        return
    if isinstance(mesh, MeshManager):
        generator = mesh.mesh
        if mesh.file is not None and mesh.format is None:
            ctx.report.error(
                "mesh.file.format_missing",
                "Mesh file input requires a format.",
                path="mesh.format",
            )
    elif isinstance(mesh, BaseMeshGenerator):
        generator = mesh
    else:
        ctx.report.error(
            "mesh.type.invalid",
            f"Expected MeshManager or mesh generator, got {type(mesh).__name__}.",
            path="mesh",
        )
        return

    if generator is None:
        return
    system = getattr(generator, "system", None)
    if system is not None:
        _validate_system_reference(system, "mesh.generator.system", ctx)
    units = getattr(generator, "units", None)
    if units is not None:
        _validate_units(
            units,
            "mesh.generator.units",
            ctx.report,
            code="mesh.generator.units.invalid",
        )
    lower = getattr(generator, "l_bound", None)
    upper = getattr(generator, "u_bound", None)
    if (lower is None) ^ (upper is None):
        ctx.report.error(
            "mesh.generator.bounds.incomplete",
            "Mesh generator requires both l_bound and u_bound.",
            path="mesh.generator",
        )


def _validate_acquisition(acquisition: Any, ctx: _ValidationContext) -> None:
    if acquisition is None:
        ctx.report.warning(
            "acquisition.missing",
            "Simulation has no acquisition attached.",
            path="acquisition",
        )
        return

    source_geometry = getattr(acquisition, "source_geometry", None)
    receiver_groups = list(getattr(acquisition, "receiver_groups", []) or [])
    if source_geometry is None:
        ctx.report.warning(
            "acquisition.sources.missing",
            "Acquisition has no source geometry.",
            path="acquisition.source_geometry",
        )
    else:
        _validate_source_geometry(source_geometry, ctx)
        _validate_source_encoding(
            getattr(acquisition, "source_encoding", None),
            source_geometry,
            ctx,
        )
    for index, group in enumerate(receiver_groups):
        _validate_receiver_group(group, index, ctx)


def _validate_source_geometry(
    geometry: Any,
    ctx: _ValidationContext,
) -> None:
    path = "acquisition.source_geometry"
    kind = getattr(geometry, "kind", None)
    if kind is None:
        ctx.report.error(
            "acquisition.source.kind.missing",
            "Source geometry requires a source kind.",
            path=f"{path}.kind",
        )
    else:
        _validate_source_kind(kind, f"{path}.kind", ctx.report)

    _validate_domain_id(getattr(geometry, "domain", None), f"{path}.domain", ctx)

    if getattr(geometry, "units", None) is not None:
        _validate_units(
            geometry.units,
            f"{path}.units",
            ctx.report,
            code="acquisition.source_geometry.units.invalid",
        )
    if getattr(geometry, "system", None) is not None:
        _validate_system_reference(geometry.system, f"{path}.system", ctx)

    defaults = getattr(geometry, "defaults", {}) or {}
    _validate_direction(defaults.get("direction"), f"{path}.defaults.direction", ctx)

    if getattr(geometry, "geometry_type", None) != "Inline":
        return

    sources = list(getattr(geometry, "sources", []) or [])
    if not sources:
        ctx.report.error(
            "acquisition.sources.empty",
            "Inline source geometry requires at least one source point.",
            path=f"{path}.sources",
        )
        return
    names = (
        geometry.point_names()
        if hasattr(geometry, "point_names")
        else [source.name for source in sources if getattr(source, "name", None)]
    )
    if len(names) != len(set(names)):
        ctx.report.error(
            "acquisition.sources.names.duplicate",
            "Inline source names must be unique.",
            path=f"{path}.sources",
        )
    for index, source in enumerate(sources):
        _validate_source_point(
            source,
            f"{path}.sources[{index}]",
            ctx,
            geometry_kind=kind,
        )


def _validate_source_point(
    source: Any,
    path: str,
    ctx: _ValidationContext,
    *,
    geometry_kind: Any,
) -> None:
    kind = getattr(source, "kind", None)
    if kind is not None:
        _validate_source_kind(kind, f"{path}.kind", ctx.report)
        if str(kind).strip().lower() != str(geometry_kind).strip().lower():
            ctx.report.error(
                "acquisition.source.kind.mismatch",
                f"Point source kind {kind!r} does not match source geometry "
                f"kind {geometry_kind!r}.",
                path=f"{path}.kind",
            )

    coordinates = getattr(source, "coordinates", None)
    if coordinates is None:
        ctx.report.error(
            "acquisition.source.coordinates.missing",
            "Source points require coordinates.",
            path=f"{path}.coordinates",
        )
        return

    units, system = _coordinate_units_and_system(coordinates)
    _validate_coordinate_value_metadata(coordinates, f"{path}.coordinates", ctx)
    values = _coordinates_to_array(coordinates)
    _validate_points(
        values,
        path=f"{path}.coordinates",
        ctx=ctx,
        units=units or _default_length_units(ctx),
        system=system,
    )
    _validate_direction(
        getattr(source, "direction", None),
        f"{path}.direction",
        ctx,
    )


def _validate_source_encoding(
    encoding: Any,
    geometry: Any,
    ctx: _ValidationContext,
) -> None:
    if encoding is None:
        return
    path = "acquisition.source_encoding"
    encoding_type = getattr(encoding, "encoding_type", None)
    if encoding_type not in {"Named", "JsonDense", "HDF5Dense"}:
        ctx.report.error(
            "acquisition.source_encoding.type.unsupported",
            f"Unsupported source encoding type {encoding_type!r}.",
            path=path,
        )
        return

    point_names = (
        set(geometry.point_names()) if hasattr(geometry, "point_names") else set()
    )
    point_count = getattr(geometry, "point_count", None)
    fields = list(getattr(encoding, "fields", []) or [])
    if encoding_type == "HDF5Dense":
        return
    if not fields:
        ctx.report.error(
            "acquisition.source_encoding.fields.empty",
            "Explicit source encoding requires at least one field.",
            path=f"{path}.fields",
        )
        return
    if (
        encoding_type == "Named"
        and getattr(geometry, "geometry_type", None) == "Inline"
    ):
        unnamed = [
            index
            for index, source in enumerate(getattr(geometry, "sources", []) or [])
            if getattr(source, "name", None) is None
        ]
        if unnamed:
            ctx.report.error(
                "acquisition.source_encoding.names.required",
                "Named source encoding requires explicit names for every inline "
                f"physical source point; missing at indices {unnamed}.",
                path="acquisition.source_geometry.sources",
            )
    for index, field_obj in enumerate(fields):
        field_path = f"{path}.fields[{index}]"
        if encoding_type == "Named":
            for source_name in getattr(field_obj, "terms", {}) or {}:
                if point_names and source_name not in point_names:
                    ctx.report.error(
                        "acquisition.source_encoding.source.unknown",
                        f"Source encoding references unknown source {source_name!r}.",
                        path=f"{field_path}.terms",
                    )
        elif point_count is not None:
            coefficients = list(getattr(field_obj, "coefficients", []) or [])
            if len(coefficients) != point_count:
                ctx.report.error(
                    "acquisition.source_encoding.coefficients.length",
                    "Dense source encoding coefficient count must match the "
                    "number of physical source points.",
                    path=f"{field_path}.coefficients",
                )


def _validate_source_kind(kind: Any, path: str, report: ValidationReport) -> None:
    value = str(kind).strip().lower()
    if value in _SOURCE_KINDS:
        return
    hint = None
    if value == "moment":
        hint = "Use 'tensor' for moment-tensor point sources."
    choices = ", ".join(sorted(_SOURCE_KINDS))
    report.error(
        "acquisition.source.kind.unsupported",
        f"Unsupported source kind {kind!r}.",
        path=path,
        hint=hint or f"Use one of: {choices}.",
    )


def _validate_receiver_group(
    group: ReceiverGroup,
    index: int,
    ctx: _ValidationContext,
) -> None:
    path = f"acquisition.receiver_groups[{index}]"
    if not getattr(group, "name", None):
        ctx.report.error(
            "acquisition.receiver_group.name.missing",
            "Receiver groups must have a non-empty name.",
            path=f"{path}.name",
        )
    _validate_domain_id(getattr(group, "domain", None), f"{path}.domain", ctx)

    device = getattr(group, "device", None)
    components = list(getattr(device, "components", []) or [])
    if not components:
        ctx.report.error(
            "acquisition.receiver_group.components.missing",
            "Receiver device must define at least one component.",
            path=f"{path}.device.components",
        )
    for component_index, component in enumerate(components):
        component_path = f"{path}.device.components[{component_index}]"
        _validate_field(
            getattr(component, "field", None),
            component_path,
            ctx,
        )
        units = getattr(component, "units", None)
        if units is not None:
            _validate_units(
                units,
                f"{component_path}.units",
                ctx.report,
                code="acquisition.receiver_component.units.invalid",
            )
        _validate_direction(
            getattr(component, "direction", None),
            f"{component_path}.direction",
            ctx,
        )

    coords = getattr(group, "coordinates", None)
    if coords is None:
        ctx.report.error(
            "acquisition.receiver_group.coordinates.missing",
            "Receiver group has no coordinates.",
            path=f"{path}.coordinates",
        )
        return
    _validate_receiver_coordinates(coords, f"{path}.coordinates", ctx)


def _validate_receiver_coordinates(
    coords: Any,
    path: str,
    ctx: _ValidationContext,
) -> None:
    if isinstance(coords, CoordsArray):
        units = coords.units
        system = coords.system
        if units is not None:
            _validate_units(
                units,
                f"{path}.units",
                ctx.report,
                code="acquisition.receiver_coordinates.units.invalid",
            )
        if system is not None:
            _validate_system_reference(system, f"{path}.system", ctx)
        values = coords.get()
        axes = _coords_array_axes(coords)
        _validate_points(
            values,
            path=path,
            ctx=ctx,
            units=units or _default_length_units(ctx),
            system=system,
            axes=axes,
        )
        return

    if isinstance(coords, CoordsGrid):
        units = coords.units or getattr(coords.grid, "units", None)
        system = coords.system or getattr(coords.grid, "system", None)
        if units is not None:
            _validate_units(
                units,
                f"{path}.units",
                ctx.report,
                code="acquisition.receiver_coordinates.units.invalid",
            )
        if system is not None:
            _validate_system_reference(system, f"{path}.system", ctx)
        lower, upper = coords.bounds
        axes = list(getattr(coords.grid, "dims", []) or [])
        _validate_bounds(
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            path=path,
            ctx=ctx,
            units=units or _default_length_units(ctx),
            system=system,
            axes=axes or None,
        )
        return

    if isinstance(coords, CoordsFromFile):
        if coords.units is not None:
            _validate_units(
                coords.units,
                f"{path}.units",
                ctx.report,
                code="acquisition.receiver_coordinates.units.invalid",
            )
        if coords.system is not None:
            _validate_system_reference(coords.system, f"{path}.system", ctx)
        file = _coords_file(coords)
        if file is not None and not file.exists():
            ctx.report.error(
                "acquisition.receiver_coordinates.file_missing",
                f"Receiver coordinate file does not exist: {file}",
                path=f"{path}.file",
            )
            return
        try:
            lower, upper = coords.bounds
        except Exception as exc:
            ctx.report.error(
                "acquisition.receiver_coordinates.unreadable",
                f"Could not read receiver coordinate bounds: {exc}",
                path=path,
            )
            return
        axes = _default_coordinate_axes(len(lower), ctx)
        _validate_bounds(
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
            path=path,
            ctx=ctx,
            units=coords.units or _default_length_units(ctx),
            system=coords.system,
            axes=axes,
        )
        return

    ctx.report.error(
        "acquisition.receiver_coordinates.type.invalid",
        f"Unsupported receiver coordinate type {type(coords).__name__}.",
        path=path,
    )


def _validate_domain_id(
    domain: Optional[int],
    path: str,
    ctx: _ValidationContext,
) -> None:
    if domain is None:
        return
    try:
        value = int(domain)
    except (TypeError, ValueError):
        ctx.report.error(
            "domain.id.invalid",
            f"Domain id must be an integer, got {domain!r}.",
            path=path,
        )
        return
    subdomains = _model_subdomain_ids(ctx.simulation)
    if subdomains and value not in subdomains:
        ctx.report.error(
            "domain.id.unknown",
            f"Domain id {value} does not match any model subdomain.",
            path=path,
            hint=f"Known subdomain ids are: {sorted(subdomains)}.",
        )


def _model_subdomain_ids(simulation: Any) -> set[int]:
    model = getattr(simulation, "model", None)
    ids = set()
    for subdomain in getattr(model, "subdomains", []) or []:
        value = getattr(subdomain, "mesh_block_id", None)
        if value is not None and value >= 0:
            ids.add(int(value))
    return ids
