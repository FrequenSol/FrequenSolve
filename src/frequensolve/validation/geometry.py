"""Shared coordinate, unit, frame, and domain validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from frequensolve.geometry.frame import Axis, CoordinateSystem, CoordinateValue
from frequensolve.mesh.mesh_generators import BaseMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.property import canonical_property_name
from frequensolve.seismic.receivers import CoordsArray, CoordsFromFile
from frequensolve.units import is_quantity, unit_expression, ureg
from frequensolve.util.physics import model_dimension as simulation_model_dimension

from .report import ValidationReport

_COORDINATE_SYSTEM_TYPES = {
    "cartesian",
    "cylindrical",
    "spherical",
    "geographic",
    "surface",
}
_DIRECTION_TYPES = {"vector", "coordinate_axis", "coordinate_basis"}
_PHYSICAL_DIRECTIONS = {"x", "y", "z"}


@dataclass
class _DomainBounds:
    lower: np.ndarray
    upper: np.ndarray
    axes: list[str]
    units: Optional[str] = None
    system: Optional[str] = None


@dataclass
class _ValidationContext:
    simulation: Any
    report: ValidationReport
    dimension: int
    systems: dict[str, CoordinateSystem]
    domain: Optional[_DomainBounds] = None
    allow_unverified_remote_files: bool = False


def _build_context(
    simulation: Any,
    report: ValidationReport,
    *,
    allow_unverified_remote_files: bool = False,
) -> _ValidationContext:
    dimension = _simulation_model_dimension(simulation, report)
    systems = _known_coordinate_systems(simulation)
    ctx = _ValidationContext(
        simulation=simulation,
        report=report,
        dimension=dimension,
        systems=systems,
        allow_unverified_remote_files=allow_unverified_remote_files,
    )
    ctx.domain = _infer_domain_bounds(ctx)
    return ctx


def _validate_coordinate_systems(ctx: _ValidationContext) -> None:
    simulation = ctx.simulation
    global_system = getattr(simulation, "global_coordinate_system", None)
    if global_system is not None:
        _validate_coordinate_system(global_system, "global_coordinate_system", ctx)

    seen: set[str] = set()
    for index, system in enumerate(getattr(simulation, "coordinate_systems", []) or []):
        path = f"coordinate_systems[{index}]"
        _validate_coordinate_system(system, path, ctx)
        name = getattr(system, "name", None)
        if not name:
            ctx.report.error(
                "coordinate_system.name.missing",
                "Registered coordinate systems must be named.",
                path=path,
            )
            continue
        if name in seen:
            ctx.report.error(
                "coordinate_system.name.duplicate",
                f"Coordinate system {name!r} is registered more than once.",
                path=path,
            )
        seen.add(name)


def _validate_coordinate_system(
    system: Any,
    path: str,
    ctx: _ValidationContext,
) -> None:
    if not isinstance(system, CoordinateSystem):
        ctx.report.error(
            "coordinate_system.type.invalid",
            f"Expected CoordinateSystem, got {type(system).__name__}.",
            path=path,
        )
        return

    system_type = str(getattr(system, "type", ""))
    if system_type not in _COORDINATE_SYSTEM_TYPES:
        choices = ", ".join(sorted(_COORDINATE_SYSTEM_TYPES))
        ctx.report.error(
            "coordinate_system.kind.unsupported",
            f"Unsupported coordinate system type {system_type!r}.",
            path=f"{path}.type",
            hint=f"Use one of: {choices}.",
        )

    ndim = getattr(system, "ndim", None)
    if ndim is not None and int(ndim) not in {2, 3}:
        ctx.report.error(
            "coordinate_system.ndim.unsupported",
            "Coordinate-system ndim must be 2 or 3.",
            path=f"{path}.ndim",
        )

    axes = getattr(system, "axes", None)
    if axes is not None:
        _validate_axes(axes, path, ctx)

    fixed_axis = getattr(system, "fixed_axis", None)
    fixed_value = getattr(system, "fixed_value", None)
    if fixed_axis is not None and fixed_value is None:
        ctx.report.error(
            "coordinate_system.fixed_axis.value_missing",
            "fixed_axis requires fixed_value.",
            path=f"{path}.fixed_axis",
        )
    if fixed_value is not None:
        _validate_quantity_units(
            fixed_value,
            f"{path}.fixed_value",
            ctx.report,
            code="coordinate_system.fixed_value.units.invalid",
        )
    origin = getattr(system, "origin", None)
    if origin is not None:
        _validate_coordinate_value_metadata(origin, f"{path}.origin", ctx)


def _validate_axes(
    axes: Sequence[Any],
    path: str,
    ctx: _ValidationContext,
) -> None:
    names: set[str] = set()
    for index, axis in enumerate(axes):
        axis_path = f"{path}.axes[{index}]"
        if isinstance(axis, Axis):
            name = str(axis.name)
            direction = axis.direction
            if direction not in _PHYSICAL_DIRECTIONS:
                ctx.report.error(
                    "coordinate_system.axis.direction.unsupported",
                    f"Axis {name!r} uses unsupported direction {direction!r}.",
                    path=f"{axis_path}.direction",
                    hint="Axis directions must be x, y, or z.",
                )
            if axis.origin is not None:
                _validate_quantity_units(
                    axis.origin,
                    f"{axis_path}.origin",
                    ctx.report,
                    code="coordinate_system.axis.origin.units.invalid",
                )
        elif isinstance(axis, str):
            name = axis
        else:
            ctx.report.error(
                "coordinate_system.axis.invalid",
                f"Axis definitions must be strings or Axis objects, got "
                f"{type(axis).__name__}.",
                path=axis_path,
            )
            continue

        if not name:
            ctx.report.error(
                "coordinate_system.axis.name.missing",
                "Coordinate-system axes must have non-empty names.",
                path=axis_path,
            )
            continue
        if name in names:
            ctx.report.error(
                "coordinate_system.axis.name.duplicate",
                f"Axis {name!r} appears more than once.",
                path=axis_path,
            )
        names.add(name)


def _validate_xarray_grid_payload(
    grid: Mapping[str, Any],
    path: str,
    ctx: _ValidationContext,
) -> None:
    dims = [str(dim) for dim in grid.get("dims", [])]
    coords = grid.get("coords", {})
    system = grid.get("system")
    grid_units = grid.get("units") or _default_length_units(ctx)
    if system is not None:
        _validate_system_reference(system, f"{path}.system", ctx)
    if grid_units is not None:
        _validate_units(
            grid_units, f"{path}.units", ctx.report, code="grid.units.invalid"
        )
    if not dims:
        ctx.report.error(
            "grid.dims.missing",
            "XArray grid requires dimension names.",
            path=f"{path}.dims",
        )
        return
    if len(set(dims)) != len(dims):
        ctx.report.error(
            "grid.dims.duplicate",
            "XArray grid dimension names must be unique.",
            path=f"{path}.dims",
        )
    expected_dimension = _grid_dimension(ctx, system)
    if expected_dimension and len(dims) != expected_dimension:
        ctx.report.error(
            "grid.dimension.mismatch",
            f"Grid has {len(dims)} dimensions but this coordinate system "
            f"expects {expected_dimension}.",
            path=f"{path}.dims",
        )

    _validate_grid_axes(dims, system, path, ctx)
    lower: list[float] = []
    upper: list[float] = []
    coord_units: list[Optional[str]] = []
    for dim in dims:
        coord_path = f"{path}.coords.{dim}"
        if dim not in coords:
            ctx.report.error(
                "grid.coordinate.missing",
                f"Grid is missing coordinate {dim!r}.",
                path=coord_path,
            )
            return
        try:
            values, units = _coord_values_and_units(coords[dim], grid_units)
        except Exception as exc:
            ctx.report.error(
                "grid.coordinate.invalid",
                f"Coordinate {dim!r} could not be interpreted: {exc}",
                path=coord_path,
            )
            return
        if units is not None:
            _validate_units(
                units,
                f"{coord_path}.units",
                ctx.report,
                code="grid.coordinate_units.invalid",
            )
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 0:
            ctx.report.error(
                "grid.coordinate.empty",
                f"Coordinate {dim!r} is empty.",
                path=coord_path,
            )
            return
        if not np.all(np.isfinite(values)):
            ctx.report.error(
                "grid.coordinate.nonfinite",
                f"Coordinate {dim!r} contains non-finite values.",
                path=coord_path,
            )
            return
        if values.size > 1:
            diffs = np.diff(values)
            if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
                ctx.report.error(
                    "grid.coordinate.not_monotonic",
                    f"Coordinate {dim!r} must be strictly monotonic.",
                    path=coord_path,
                )
                return
        lower.append(float(np.min(values)))
        upper.append(float(np.max(values)))
        coord_units.append(units)

    if _scalar_units(coord_units):
        units = _scalar_units(coord_units)
    else:
        units = grid_units
    _validate_bounds(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
        path=path,
        ctx=ctx,
        units=units,
        system=system,
        axes=dims,
    )


def _validate_grid_like(
    grid: Any,
    path: str,
    ctx: _ValidationContext,
    *,
    require_domain: bool,
) -> None:
    if isinstance(grid, Mapping) and "dims" in grid and "coords" in grid:
        _validate_xarray_grid_payload(grid, path, ctx)
        return
    if hasattr(grid, "dims") and hasattr(grid, "coords"):
        dims = list(grid.dims)
        system = getattr(grid, "attrs", {}).get("system")
        _validate_grid_axes(dims, system, path, ctx)
        return
    if require_domain:
        ctx.report.warning(
            "grid.unchecked",
            f"Grid type {type(grid).__name__} could not be checked.",
            path=path,
        )


def _validate_grid_axes(
    axes: Sequence[str],
    system: Optional[str],
    path: str,
    ctx: _ValidationContext,
) -> None:
    allowed = _active_axes(ctx, system)
    unsupported = [axis for axis in axes if _axis_index(allowed, axis) is None]
    if unsupported:
        ctx.report.error(
            "grid.axis.unsupported",
            f"Grid uses dimension names not exposed by the coordinate system: "
            f"{unsupported}.",
            path=f"{path}.dims",
            hint=f"Available axes are: {', '.join(allowed)}.",
        )


def _grid_dimension(ctx: _ValidationContext, system: Optional[str]) -> int:
    coordinate_system = _coordinate_system_for_axes(ctx, system)
    if coordinate_system is None:
        return ctx.dimension
    axes = _coordinate_system_axes(coordinate_system, ctx.dimension)
    return len(axes) or ctx.dimension


def _validate_points(
    values: Any,
    *,
    path: str,
    ctx: _ValidationContext,
    units: Optional[str] = None,
    system: Optional[str] = None,
    axes: Optional[Sequence[str]] = None,
) -> None:
    try:
        points = np.asarray(values, dtype=float)
    except Exception as exc:
        ctx.report.error(
            "coordinates.invalid",
            f"Coordinates could not be converted to numeric values: {exc}",
            path=path,
        )
        return
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.ndim != 2:
        ctx.report.error(
            "coordinates.dimension.invalid",
            "Coordinates must be a 2D array shaped (n_points, dimension).",
            path=path,
        )
        return
    if ctx.dimension and points.shape[1] != ctx.dimension:
        ctx.report.error(
            "coordinates.dimension.mismatch",
            f"Coordinates have {points.shape[1]} columns but the simulation "
            f"model dimension is {ctx.dimension}.",
            path=path,
        )
        return
    if not np.all(np.isfinite(points)):
        ctx.report.error(
            "coordinates.nonfinite",
            "Coordinates contain non-finite values.",
            path=path,
        )
        return
    _validate_bounds(
        np.min(points, axis=0),
        np.max(points, axis=0),
        path=path,
        ctx=ctx,
        units=units,
        system=system,
        axes=axes,
    )


def _validate_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    path: str,
    ctx: _ValidationContext,
    units: Optional[str] = None,
    system: Optional[str] = None,
    axes: Optional[Sequence[str]] = None,
) -> None:
    units = units or _default_length_units(ctx)
    if lower.shape != upper.shape:
        ctx.report.error(
            "bounds.shape.mismatch",
            "Lower and upper bounds must have the same shape.",
            path=path,
        )
        return
    if axes is not None and lower.size != len(axes):
        ctx.report.error(
            "bounds.dimension.mismatch",
            f"Bounds have {lower.size} dimensions but {len(axes)} axes were "
            "provided.",
            path=path,
        )
        return
    if ctx.dimension and axes is None and lower.size != ctx.dimension:
        ctx.report.error(
            "bounds.dimension.mismatch",
            f"Bounds have {lower.size} dimensions but the simulation model "
            f"dimension is {ctx.dimension}.",
            path=path,
        )
        return
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        ctx.report.error(
            "bounds.nonfinite",
            "Bounds contain non-finite values.",
            path=path,
        )
        return
    domain = ctx.domain
    if domain is None or not _same_system(system, domain.system):
        return
    try:
        lower_in_domain = _convert_units_array(lower, units, domain.units)
        upper_in_domain = _convert_units_array(upper, units, domain.units)
    except Exception as exc:
        ctx.report.error(
            "bounds.units.incompatible",
            f"Could not convert coordinate units {units!r} to domain units "
            f"{domain.units!r}: {exc}",
            path=path,
        )
        return

    for local_index, (low_value, high_value) in enumerate(
        zip(lower_in_domain, upper_in_domain)
    ):
        domain_index = local_index
        axis_name = None
        if axes is not None:
            axis_name = str(axes[local_index])
            mapped = _axis_index(domain.axes, axis_name)
            if mapped is None:
                continue
            domain_index = mapped
        domain_low = min(domain.lower[domain_index], domain.upper[domain_index])
        domain_high = max(domain.lower[domain_index], domain.upper[domain_index])
        coord_low = min(low_value, high_value)
        coord_high = max(low_value, high_value)
        tol = 1e-9 * max(1.0, abs(domain_low), abs(domain_high))
        if coord_low < domain_low - tol or coord_high > domain_high + tol:
            label = axis_name or domain.axes[domain_index]
            ctx.report.error(
                "coordinates.domain.outside",
                f"Coordinate extent along {label!r} is [{coord_low:g}, "
                f"{coord_high:g}], outside model domain [{domain_low:g}, "
                f"{domain_high:g}].",
                path=path,
            )


def _validate_direction(
    direction: Any,
    path: str,
    ctx: _ValidationContext,
) -> None:
    if direction is None:
        return
    if hasattr(direction, "type") and hasattr(direction, "to_fs"):
        direction_type = getattr(direction, "type", "vector")
        if direction_type not in _DIRECTION_TYPES:
            ctx.report.error(
                "direction.type.unsupported",
                f"Unsupported direction type {direction_type!r}.",
                path=f"{path}.type",
            )
        system = getattr(direction, "system", None)
        if system is not None:
            _validate_system_reference(system, f"{path}.system", ctx)
        axis = getattr(direction, "axis", None)
        if (
            axis is not None
            and _axis_index(_active_axes(ctx, system), str(axis)) is None
        ):
            ctx.report.error(
                "direction.axis.unsupported",
                f"Direction axis {axis!r} is not exposed by coordinate system "
                f"{system or 'global'!r}.",
                path=f"{path}.axis",
            )
        if getattr(direction, "units", None) is not None:
            _validate_units(
                direction.units,
                f"{path}.units",
                ctx.report,
                code="direction.units.invalid",
            )
        value = getattr(direction, "value", None)
    else:
        value = direction

    if value is None:
        return
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return
    if not np.all(np.isfinite(array)):
        ctx.report.error(
            "direction.nonfinite",
            "Direction contains non-finite values.",
            path=path,
        )


def _validate_projection_reference(
    value: Any,
    path: str,
    ctx: _ValidationContext,
) -> None:
    if value is None:
        return
    if isinstance(value, Mapping):
        system = value.get("system")
        axis = value.get("axis")
        components = value.get("components")
    else:
        system = "global"
        axis = None
        components = value
    if system is not None:
        _validate_system_reference(system, f"{path}.system", ctx)
    if axis is not None and _axis_index(_active_axes(ctx, system), str(axis)) is None:
        ctx.report.error(
            "projection.axis.unsupported",
            f"Axis {axis!r} is not exposed by coordinate system "
            f"{system or 'global'!r}.",
            path=f"{path}.axis",
        )
    if components is None or isinstance(components, (str, bytes)):
        return
    for index, component in enumerate(components):
        if _axis_index(_active_axes(ctx, system), str(component)) is None:
            ctx.report.error(
                "projection.component.unsupported",
                f"Component {component!r} is not exposed by coordinate system "
                f"{system or 'global'!r}.",
                path=f"{path}.components[{index}]",
            )


def _validate_coordinate_value_metadata(
    value: Any,
    path: str,
    ctx: _ValidationContext,
) -> None:
    if isinstance(value, CoordinateValue):
        if value.units is not None:
            _validate_units(
                value.units,
                f"{path}.units",
                ctx.report,
                code="coordinates.units.invalid",
            )
        if value.system is not None:
            _validate_system_reference(value.system, f"{path}.system", ctx)


def _validate_unit_config(value: Any, report: ValidationReport) -> None:
    if value is None:
        return
    for key, units in getattr(value, "defaults", {}).items():
        _validate_units(
            units,
            f"units.defaults.{key}",
            report,
            code="simulation.units.invalid",
        )
    for key, units in getattr(value, "scales", {}).items():
        _validate_units(
            units,
            f"units.scales.{key}",
            report,
            code="simulation.units.invalid",
        )


def _validate_quantity_units(
    value: Any,
    path: str,
    report: ValidationReport,
    *,
    code: str,
) -> None:
    if is_quantity(value):
        _validate_units(value.units, path, report, code=code)
        return
    if isinstance(value, Mapping) and value.get("units") is not None:
        _validate_units(value["units"], f"{path}.units", report, code=code)


def _validate_units(
    units: Any,
    path: str,
    report: ValidationReport,
    *,
    code: str,
) -> None:
    if units is None:
        return
    values = units if _is_unit_sequence(units) else [units]
    for value in values:
        if value is None or value == "":
            continue
        try:
            ureg.Unit(unit_expression(value))
        except Exception as exc:
            report.error(
                code,
                f"Invalid unit expression {value!r}: {exc}",
                path=path,
            )


def _infer_domain_bounds(ctx: _ValidationContext) -> Optional[_DomainBounds]:
    mesh = getattr(ctx.simulation, "mesh", None)
    generator = None
    if isinstance(mesh, MeshManager):
        generator = mesh.mesh
    elif isinstance(mesh, BaseMeshGenerator):
        generator = mesh
    if generator is not None:
        domain = _domain_from_mesh_generator(generator, ctx)
        if domain is not None:
            return domain

    model = getattr(ctx.simulation, "model", None)
    if model is not None and hasattr(model, "_mesh_bounds"):
        try:
            lower, upper, units = model._mesh_bounds()
        except Exception:
            return None
        axes = _default_coordinate_axes(len(lower), ctx)
        return _DomainBounds(
            lower=np.asarray(lower, dtype=float),
            upper=np.asarray(upper, dtype=float),
            axes=axes,
            units=units or _default_length_units(ctx),
            system=None,
        )
    return None


def _domain_from_mesh_generator(
    generator: BaseMeshGenerator,
    ctx: _ValidationContext,
) -> Optional[_DomainBounds]:
    lower_value = getattr(generator, "l_bound", None)
    upper_value = getattr(generator, "u_bound", None)
    if lower_value is None or upper_value is None:
        return None
    default_units = getattr(generator, "units", None)
    try:
        lower, lower_units = _numeric_vector(lower_value, default_units)
        upper, upper_units = _numeric_vector(upper_value, default_units)
    except Exception as exc:
        ctx.report.error(
            "mesh.generator.bounds.invalid",
            f"Mesh generator bounds could not be interpreted: {exc}",
            path="mesh.generator",
        )
        return None
    if lower.size != upper.size:
        ctx.report.error(
            "mesh.generator.bounds.shape_mismatch",
            "Mesh generator l_bound and u_bound must have the same length.",
            path="mesh.generator",
        )
        return None
    if ctx.dimension and lower.size != ctx.dimension:
        ctx.report.error(
            "mesh.generator.bounds.dimension_mismatch",
            f"Mesh generator bounds have {lower.size} dimensions but the "
            f"simulation model dimension is {ctx.dimension}.",
            path="mesh.generator",
        )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        ctx.report.error(
            "mesh.generator.bounds.nonfinite",
            "Mesh generator bounds contain non-finite values.",
            path="mesh.generator",
        )
        return None
    units = (
        lower_units
        or upper_units
        or (unit_expression(default_units) if default_units is not None else None)
        or _default_length_units(ctx)
    )
    axes = _active_axes(ctx, getattr(generator, "system", None))
    if len(axes) != lower.size:
        axes = _default_coordinate_axes(lower.size, ctx)
    return _DomainBounds(
        lower=lower,
        upper=upper,
        axes=axes,
        units=units,
        system=getattr(generator, "system", None),
    )


def _simulation_model_dimension(simulation: Any, report: ValidationReport) -> int:
    try:
        return simulation_model_dimension(getattr(simulation, "dimension"))
    except Exception as exc:
        report.error(
            "simulation.dimension.unsupported",
            f"Could not determine model dimension: {exc}",
            path="dimension",
        )
        return 0


def _known_coordinate_systems(simulation: Any) -> dict[str, CoordinateSystem]:
    systems: dict[str, CoordinateSystem] = {}
    global_system = getattr(simulation, "global_coordinate_system", None)
    if isinstance(global_system, CoordinateSystem):
        systems["global"] = global_system
        if global_system.name:
            systems[str(global_system.name)] = global_system
    for system in getattr(simulation, "coordinate_systems", []) or []:
        if isinstance(system, CoordinateSystem) and system.name:
            systems[str(system.name)] = system
    return systems


def _validate_system_reference(
    system: Any,
    path: str,
    ctx: _ValidationContext,
) -> None:
    name = str(system)
    if name in {"", "global", "None"}:
        return
    if name not in ctx.systems:
        ctx.report.error(
            "coordinate_system.reference.unknown",
            f"Unknown coordinate system {name!r}.",
            path=path,
            hint="Register the coordinate system on simulation.coordinate_systems.",
        )


def _active_axes(ctx: _ValidationContext, system: Optional[str]) -> list[str]:
    coordinate_system = _coordinate_system_for_axes(ctx, system)
    if coordinate_system is None:
        return _default_coordinate_axes(ctx.dimension, ctx)
    axes = _coordinate_system_axes(coordinate_system, ctx.dimension)
    return axes or _default_coordinate_axes(ctx.dimension, ctx)


def _coordinate_system_for_axes(
    ctx: _ValidationContext,
    system: Optional[str],
) -> Optional[CoordinateSystem]:
    if system not in {None, "", "global"}:
        return ctx.systems.get(str(system))
    return ctx.systems.get("global")


def _coordinate_system_axes(
    system: CoordinateSystem,
    dimension: int,
) -> list[str]:
    axes = getattr(system, "axes", None)
    if axes:
        names = [axis.name if isinstance(axis, Axis) else str(axis) for axis in axes]
        return names

    system_type = str(getattr(system, "type", "cartesian"))
    if system_type == "cylindrical":
        names = ["r", "theta", "z"]
    elif system_type == "spherical":
        names = ["r", "theta", "phi"]
    elif system_type == "geographic":
        names = ["longitude", "latitude", "depth"]
    elif system_type == "surface":
        names = ["x", "y", "z"]
    else:
        names = ["x", "y", "z"]

    fixed_axis = getattr(system, "fixed_axis", None)
    if fixed_axis in names:
        names = [name for name in names if name != fixed_axis]
    elif dimension == 2 and system_type in {"cartesian", "surface"}:
        names = ["x", "z"]
    elif dimension == 2 and system_type == "cylindrical":
        names = ["r", "z"]
    return names[:dimension]


def _default_coordinate_axes(dimension: int, ctx: _ValidationContext) -> list[str]:
    physics = str(getattr(ctx.simulation, "physics", ""))
    if dimension == 2:
        if "axisym" in physics:
            return ["r", "z"]
        return ["x", "z"]
    if dimension == 3:
        return ["x", "y", "z"]
    return [f"axis_{index}" for index in range(dimension)]


def _axis_index(axes: Sequence[str], axis: str) -> Optional[int]:
    axis = str(axis)
    if axis in axes:
        return list(axes).index(axis)
    aliases = {"r": "x", "x": "r", "depth": "z"}
    alias = aliases.get(axis)
    if alias in axes:
        return list(axes).index(alias)
    return None


def _same_system(left: Optional[str], right: Optional[str]) -> bool:
    left_name = "global" if left in {None, "", "None"} else str(left)
    right_name = "global" if right in {None, "", "None"} else str(right)
    return left_name == right_name


def _numeric_vector(
    value: Any, default_units: Optional[Any]
) -> tuple[np.ndarray, Optional[str]]:
    units = unit_expression(default_units) if default_units is not None else None
    if isinstance(value, Mapping):
        units = (
            unit_expression(value.get("units", units))
            if value.get("units", units)
            else None
        )
        value = value.get("value", value.get("data", value.get("coords")))
    if is_quantity(value):
        payload = value.to_base_units()
        units = unit_expression(payload.units)
        value = payload.magnitude
    if isinstance(value, (list, tuple, np.ndarray)):
        quantity_units = _first_quantity_units(value)
        if quantity_units is not None:
            units = unit_expression(quantity_units)
            value = _strip_quantities(value, quantity_units)
    values = np.asarray(value, dtype=float).reshape(-1)
    return values, units


def _coord_values_and_units(
    value: Any, default_units: Optional[str]
) -> tuple[np.ndarray, Optional[str]]:
    units = default_units
    if isinstance(value, Mapping):
        units = value.get("units", units)
        value = value.get("data", value.get("value", value.get("values")))
    if hasattr(value, "attrs"):
        units = value.attrs.get("units", units)
        value = value.values
    if is_quantity(value):
        if units is None:
            units = unit_expression(value.units)
        value = value.magnitude
    return np.asarray(value, dtype=float).reshape(-1), units


def _coordinate_units_and_system(value: Any) -> tuple[Optional[str], Optional[str]]:
    if isinstance(value, CoordinateValue):
        units = unit_expression(value.units) if value.units is not None else None
        return units, value.system
    return None, None


def _coordinates_to_array(value: Any) -> np.ndarray:
    if isinstance(value, CoordinateValue):
        value = value.value
    if is_quantity(value):
        value = value.magnitude
    if isinstance(value, (list, tuple, np.ndarray)):
        quantity_units = _first_quantity_units(value)
        if quantity_units is not None:
            value = _strip_quantities(value, quantity_units)
    return np.asarray(value, dtype=float)


def _first_quantity_units(value: Any) -> Optional[Any]:
    if is_quantity(value):
        return value.units
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.flat:
                units = _first_quantity_units(item)
                if units is not None:
                    return units
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            units = _first_quantity_units(item)
            if units is not None:
                return units
    return None


def _strip_quantities(value: Any, units: Any) -> Any:
    if is_quantity(value):
        return value.to(units).magnitude
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            values = [_strip_quantities(item, units) for item in value.flat]
            return np.asarray(values, dtype=float).reshape(value.shape)
        return value
    if isinstance(value, (list, tuple)):
        return [_strip_quantities(item, units) for item in value]
    return value


def _convert_units_array(
    values: np.ndarray,
    from_units: Optional[Any],
    to_units: Optional[Any],
) -> np.ndarray:
    from_expr = unit_expression(from_units) if from_units is not None else None
    to_expr = unit_expression(to_units) if to_units is not None else None
    if from_expr is None or to_expr is None or from_expr == to_expr:
        return np.asarray(values, dtype=float)
    return np.asarray((values * ureg(from_expr)).to(to_expr).magnitude, dtype=float)


def _coords_array_axes(coords: CoordsArray) -> Optional[list[str]]:
    coordinate = coords.coordinates.coords.get("coordinate")
    if coordinate is None:
        return None
    try:
        return [str(value) for value in coordinate.values.tolist()]
    except Exception:
        return None


def _coords_file(coords: CoordsFromFile) -> Optional[Path]:
    try:
        return coords._local_file()
    except Exception:
        file = getattr(coords, "file", None)
        return Path(file).expanduser() if file is not None else None


def _model_property_names(simulation: Any) -> set[str]:
    model = getattr(simulation, "model", None)
    names = set()
    for subdomain in getattr(model, "subdomains", []) or []:
        for name in getattr(subdomain, "properties", {}) or {}:
            names.add(canonical_property_name(name))
    return names


def _default_length_units(ctx: _ValidationContext) -> Optional[str]:
    units = getattr(ctx.simulation, "units", None)
    defaults = getattr(units, "defaults", {}) or {}
    value = defaults.get("length")
    if value is None:
        return None
    try:
        return unit_expression(value)
    except Exception:
        return None


def _is_unit_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _scalar_units(values: Sequence[Optional[str]]) -> Optional[str]:
    present = [value for value in values if value]
    if present and all(value == present[0] for value in present):
        return present[0]
    return None
