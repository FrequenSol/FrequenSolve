"""Material-property containers, expressions, file references, and unit metadata."""

from __future__ import annotations

import copy
import shlex
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from frequensolve._property_names import (
    canonical_property_name as _canonical_property_name,
)
from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.dispersion import DispersionScaling
from frequensolve.units import is_quantity, quantity_to_fs, unit_expression, ureg
from frequensolve.util.mixins import ExportContext, merge_extra
from frequensolve.util.stochastic_fields import von_karman_stochastic_field

__all__ = [
    "Property",
    "PropertyExpression",
    "PropertyMap",
    "canonical_property_name",
    "coord",
    "prop",
    "ref",
    "remap",
    "var",
    "_ensure_minimum_coordinates",
]


def canonical_property_name(name: str) -> str:
    """Normalize a user-facing material-property name."""

    return _canonical_property_name(name)


def _is_hdf5_locator(value: Any) -> bool:
    text = _strip_remote_prefix(value)
    if ":" not in text:
        return False
    file_part = text.split(":", 1)[0].lower()
    return file_part.endswith(".h5") or file_part.endswith(".hdf5")


def _strip_remote_prefix(value: Any) -> str:
    text = str(value)
    return text.replace("remote:", "", 1) if text.startswith("remote:") else text


def _infer_file_format(value: Any) -> Optional[str]:
    if _is_hdf5_locator(value):
        return "hdf5"
    text = _strip_remote_prefix(value)
    if ":" in text:
        file_part, _ = text.split(":", 1)
        if Path(file_part).suffix:
            text = file_part
    if Path(text).suffix.lower() == ".rsf":
        return "rsf"
    return None


def _read_rsf_header(file: Path) -> Dict[str, str]:
    try:
        tokens = shlex.split(file.read_text(), comments=True, posix=True)
    except ValueError as exc:
        raise ValueError(f"Could not parse RSF header {file}: {exc}") from exc
    header: Dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        header[key] = value
    return header


def rsf_binary_path(file: Union[str, Path]) -> Optional[Path]:
    """Return the binary sidecar referenced by an RSF header, if present."""

    path = Path(file)
    header = _read_rsf_header(path)
    data_ref = header.get("in")
    if data_ref is None or data_ref in {"", "stdin"}:
        return None
    data_path = Path(data_ref)
    if not data_path.is_absolute():
        data_path = path.parent / data_path
    return data_path


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


def _normalize_fill_invalid(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in {"nearest", "none"}:
        raise ValueError("Property fill_invalid must be one of: nearest, none")
    return normalized


def _range_bound_units(value: Any) -> Optional[str]:
    if value is None:
        return None
    if is_quantity(value):
        return unit_expression(value.units)
    if isinstance(value, Mapping):
        units = value.get("units")
        return unit_expression(units) if units is not None else None
    return None


def _range_bound_value(value: Any, units: Optional[str]) -> Any:
    if is_quantity(value):
        quantity = value.to(units) if units is not None else value
        return _plain_value(quantity.magnitude)
    if isinstance(value, Mapping) and "value" in value:
        source_units = value.get("units")
        raw_value = value["value"]
        if source_units is not None and units is not None:
            converted = (raw_value * ureg(unit_expression(source_units))).to(units)
            return _plain_value(converted.magnitude)
        return _plain_value(raw_value)
    return _plain_value(value)


def _range_bound_payload(value: Any, units: Optional[str]) -> Any:
    if is_quantity(value):
        return quantity_to_fs(value)
    if isinstance(value, Mapping) and "value" in value:
        payload = {"value": _plain_value(value["value"])}
        if value.get("units") is not None:
            payload["units"] = unit_expression(value["units"])
            return payload
        if units is not None:
            payload["units"] = units
            return payload
        return payload["value"]
    raw_value = _plain_value(value)
    if units is None:
        return raw_value
    return {"value": raw_value, "units": units}


def _range_alias_value(
    payload: Mapping[str, Any],
    primary: str,
    alias: str,
) -> Any:
    if primary in payload and alias in payload:
        raise ValueError(f"Specify only one of valid_range.{primary} or {alias}")
    if primary in payload:
        return payload[primary]
    return payload.get(alias)


def _normalize_valid_range(
    valid_range: Optional[Any] = None,
    *,
    valid_min: Optional[Any] = None,
    valid_max: Optional[Any] = None,
    units: Optional[Any] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    lower = upper = None
    default_units = unit_expression(units) if units is not None else None

    if valid_range is not None:
        if isinstance(valid_range, Mapping):
            payload = dict(valid_range)
            top_units = payload.pop("units", None)
            if top_units is not None:
                default_units = unit_expression(top_units)
            lower = _range_alias_value(payload, "lower", "min")
            upper = _range_alias_value(payload, "upper", "max")
            unexpected = set(payload) - {"lower", "upper", "min", "max"}
            if unexpected:
                names = ", ".join(sorted(unexpected))
                raise ValueError(f"Unexpected valid_range field(s): {names}")
        elif isinstance(valid_range, Sequence) and not isinstance(
            valid_range, (str, bytes)
        ):
            if len(valid_range) != 2:
                raise ValueError(
                    "Property valid_range sequences must be (lower, upper)"
                )
            lower, upper = valid_range
        else:
            raise TypeError("Property valid_range must be a mapping or (lower, upper)")

    if valid_min is not None:
        if lower is not None:
            raise ValueError("Pass valid_min or valid_range lower, not both")
        lower = valid_min
    if valid_max is not None:
        if upper is not None:
            raise ValueError("Pass valid_max or valid_range upper, not both")
        upper = valid_max

    if lower is None and upper is None:
        return None, default_units

    for value in (lower, upper):
        value_units = _range_bound_units(value)
        if default_units is None and value_units is not None:
            default_units = value_units

    out: Dict[str, Any] = {}
    if lower is not None:
        out["lower"] = _range_bound_payload(lower, default_units)
    if upper is not None:
        out["upper"] = _range_bound_payload(upper, default_units)
    return out, default_units


def _range_pair(value: Any, *, name: str) -> tuple[Any, Any]:
    if isinstance(value, Mapping):
        payload = dict(value)
        lower = _range_alias_value(payload, "lower", "min")
        upper = _range_alias_value(payload, "upper", "max")
        if lower is None or upper is None:
            raise ValueError(f"{name} must define both lower/min and upper/max")
        unexpected = set(payload) - {"lower", "upper", "min", "max"}
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"Unexpected {name} field(s): {names}")
        return lower, upper
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2:
            return value[0], value[1]
    raise ValueError(f"{name} must be a two-item range")


def _infer_range_units(units: Optional[Any], *values: Any) -> Optional[str]:
    if units is not None:
        return unit_expression(units)
    for value in values:
        value_units = _range_bound_units(value)
        if value_units is not None:
            return value_units
    return None


def _remap_bound_node(value: Any, units: Optional[str]) -> Dict[str, Any]:
    payload = {"value": _range_bound_value(value, units)}
    if units is not None:
        payload["units"] = units
    elif isinstance(value, Mapping) and value.get("units") is not None:
        payload["units"] = unit_expression(value["units"])
    elif is_quantity(value):
        payload["units"] = unit_expression(value.units)
    return payload


def _remap_bound_magnitude(value: Any, units: Optional[str]) -> float:
    return float(_range_bound_value(value, units))


def _normalize_remap_outside(
    outside: Optional[Any],
    clamp: Optional[Any],
) -> str:
    if isinstance(clamp, str):
        if outside is not None:
            raise ValueError("Specify either remap clamp or outside, not both")
        outside = clamp
        clamp = None
    elif clamp is not None and not isinstance(clamp, bool):
        raise TypeError("remap clamp must be a boolean when provided")

    if outside is None:
        return "clamp" if clamp is not False else "extrapolate"

    normalized = str(outside).strip().lower().replace("-", "_")
    aliases = {
        "clamp": "clamp",
        "saturate": "clamp",
        "to_range": "clamp",
        "preserve": "preserve",
        "unchanged": "preserve",
        "identity": "preserve",
        "keep": "preserve",
        "extrapolate": "extrapolate",
        "none": "extrapolate",
        "linear": "extrapolate",
    }
    mode = aliases.get(normalized)
    if mode is None:
        raise ValueError("remap outside must be one of: clamp, preserve, extrapolate")

    if clamp is not None:
        clamp_mode = "clamp" if clamp else "extrapolate"
        if mode != clamp_mode:
            raise ValueError("Specify either remap clamp or outside, not both")
    return mode


def _coord_units(coord: xr.DataArray) -> Optional[str]:
    units = coord.attrs.get("units")
    return unit_expression(units) if units is not None else None


def _convert_coord_units(
    coord: xr.DataArray,
    target_units: Optional[str],
) -> xr.DataArray:
    source_units = _coord_units(coord)
    if source_units is None or target_units is None or source_units == target_units:
        return coord
    converted = (
        (np.asarray(coord.values) * ureg(source_units)).to(target_units).magnitude
    )
    out = xr.DataArray(
        converted,
        dims=coord.dims,
        coords=coord.coords,
        attrs=dict(coord.attrs),
        name=coord.name,
    )
    out.attrs["units"] = target_units
    return out


def _coords_in_data_units(
    coords: Mapping[str, xr.DataArray],
    data_coords: Mapping[str, xr.DataArray],
    dims: Optional[Sequence[str]] = None,
) -> Dict[str, xr.DataArray]:
    dims = list(dims if dims is not None else data_coords.keys())
    out = {}
    for dim in dims:
        coord = coords[dim]
        target_units = _coord_units(data_coords[dim]) if dim in data_coords else None
        out[dim] = _convert_coord_units(coord, target_units)
    return out


def _normalize_system_alias(payload: Dict[str, Any]) -> None:
    coordinate_system = payload.pop("coordinate_system", None)
    if coordinate_system is None:
        return
    if "system" in payload and payload["system"] != coordinate_system:
        raise ValueError("Specify only one of system or coordinate_system")
    payload["system"] = coordinate_system


def _resolve_system_alias(
    system: Optional[str], coordinate_system: Optional[str]
) -> Optional[str]:
    if coordinate_system is not None:
        if system is not None and system != coordinate_system:
            raise ValueError("Specify only one of system or coordinate_system")
        system = coordinate_system
    return system


class PropertyExpression:
    """Algebraic expression used to define a property from other properties.

    Expressions are stored as small solver-friendly dictionaries. They support
    arithmetic operator overloads, references to other property names, scalar
    literals, and unit conversion nodes.

    Args:
        node: Serialized expression node to copy into this expression.
    """

    def __init__(self, node: Mapping[str, Any]):
        """Create an expression from an already-normalized expression node."""

        self.node = copy.deepcopy(dict(node))

    @classmethod
    def ref(cls, name: str) -> "PropertyExpression":
        """Reference another property by canonical name.

        Args:
            name: Property name or alias to reference.

        Returns:
            A property expression containing a ``ref`` node.
        """

        return cls({"ref": canonical_property_name(name)})

    @classmethod
    def field(cls, name: str) -> "PropertyExpression":
        """Reference independent named data on the current subdomain.

        Args:
            name: Field name stored in ``ModelSubdomain.fields``.

        Returns:
            A property expression containing a solver ``field`` node.
        """

        name = str(name).strip()
        if not name:
            raise ValueError("Expression field name cannot be empty")
        return cls({"field": name})

    @classmethod
    def var(cls, name: str) -> "PropertyExpression":
        """Reference a named expression variable.

        Variables are for non-material expression contexts, such as mesh
        adaptivity branches based on ``epw``.

        Args:
            name: Variable name to reference.

        Returns:
            A property expression containing a ``var`` node.
        """

        name = str(name).strip()
        if not name:
            raise ValueError("Expression variable name cannot be empty")
        return cls({"var": name})

    @classmethod
    def value(cls, value: Any) -> "PropertyExpression":
        """Create a scalar literal expression.

        Args:
            value: Scalar Python value, Pint quantity, or scalar
                ``xarray.DataArray`` to embed in the expression.

        Returns:
            A property expression containing a literal ``value`` node.

        Raises:
            ValueError: If ``value`` is a non-scalar ``xarray.DataArray``.
        """

        if is_quantity(value):
            return cls(quantity_to_fs(value))
        if isinstance(value, xr.DataArray):
            if value.size != 1:
                raise ValueError("Expression constants must be scalar values")
            units = value.attrs.get("units")
            payload = {"value": _plain_value(value.values.item())}
            if units is not None:
                payload["units"] = unit_expression(units)
            return cls(payload)
        return cls({"value": _plain_value(value)})

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        symbols: Optional[Mapping[str, Any]] = None,
        default_symbol: str = "ref",
    ) -> "PropertyExpression":
        """Coerce user input into a ``PropertyExpression``.

        Args:
            value: Existing expression, serialized expression payload, scalar
                value, or mapping containing an ``expr`` field.
            symbols: Optional expression symbol bindings. Bound symbols become
                ``var`` nodes.
            default_symbol: How unbound SymPy symbols are lowered. Supported
                values are ``"ref"`` and ``"var"``.

        Returns:
            A normalized property expression with canonicalized references.
        """

        if isinstance(value, PropertyExpression):
            return value
        if _is_sympy_expression(value):
            return cls(
                _sympy_to_expression_node(
                    value,
                    symbols=set((symbols or {}).keys()),
                    default_symbol=default_symbol,
                )
            )
        if isinstance(value, Mapping):
            payload = copy.deepcopy(dict(value))
            if "expr" in payload:
                return cls.from_value(
                    payload["expr"],
                    symbols=symbols,
                    default_symbol=default_symbol,
                )
            return cls(_canonicalize_expression_refs(payload))
        return cls.value(value)

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize the expression.

        Args:
            ctx: Optional export context. It is accepted for API consistency;
                expression nodes are self-contained and do not currently use it.

        Returns:
            A deep copy of the solver expression payload.
        """

        return copy.deepcopy(self.node)

    def depends_on(self) -> List[str]:
        """List property references used by the expression.

        Returns:
            Canonical property names in first-seen traversal order.
        """

        out: List[str] = []

        def visit(node: Any) -> None:
            if isinstance(node, PropertyExpression):
                node = node.node
            if isinstance(node, Mapping):
                if "ref" in node:
                    name = canonical_property_name(node["ref"])
                    if name not in out:
                        out.append(name)
                for key, child in node.items():
                    if key != "ref":
                        visit(child)
            elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
                for child in node:
                    visit(child)

        visit(self.node)
        return out

    def field_names(self) -> List[str]:
        """List auxiliary subdomain fields referenced by the expression."""

        out: List[str] = []

        def visit(node: Any) -> None:
            if isinstance(node, PropertyExpression):
                node = node.node
            if isinstance(node, Mapping):
                if "field" in node:
                    name = str(node["field"])
                    if name not in out:
                        out.append(name)
                for key, child in node.items():
                    if key != "field":
                        visit(child)
            elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
                for child in node:
                    visit(child)

        visit(self.node)
        return out

    def evaluate(
        self,
        references: Mapping[str, Any],
        variables: Optional[Mapping[str, Any]] = None,
        fields: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Evaluate this expression against materialized array values.

        Args:
            references: Property values keyed by property name or alias. Values
                may be scalars, NumPy arrays, Pint quantities, or xarray data
                arrays with optional ``units`` metadata.
            variables: Optional values for ``var`` nodes, keyed by the exact
                expression symbol name.
            fields: Optional independent subdomain data keyed by the exact
                name used in ``field`` nodes.

        Returns:
            A scalar, NumPy array, or Pint quantity produced by evaluating the
            solver expression AST elementwise.

        Raises:
            ValueError: If a reference, variable, operation, or node is missing
                or unsupported.
        """

        normalized_references = {
            canonical_property_name(name): _expression_operand(value)
            for name, value in references.items()
        }
        normalized_variables = {
            str(name): _expression_operand(value)
            for name, value in (variables or {}).items()
        }
        normalized_fields = {
            str(name): _expression_operand(value)
            for name, value in (fields or {}).items()
        }
        return _evaluate_expression_node(
            self.node,
            references=normalized_references,
            variables=normalized_variables,
            fields=normalized_fields,
        )

    def __add__(self, other: Any) -> "PropertyExpression":
        return self._binary("add", other)

    def __radd__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("add", other)

    def __sub__(self, other: Any) -> "PropertyExpression":
        return self._binary("sub", other)

    def __rsub__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("sub", other)

    def __mul__(self, other: Any) -> "PropertyExpression":
        return self._binary("mul", other)

    def __rmul__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("mul", other)

    def __truediv__(self, other: Any) -> "PropertyExpression":
        return self._binary("div", other)

    def __rtruediv__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("div", other)

    def __pow__(self, other: Any) -> "PropertyExpression":
        return self._binary("pow", other)

    def __rpow__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("pow", other)

    def __lt__(self, other: Any) -> "PropertyExpression":
        return self._binary("<", other)

    def __le__(self, other: Any) -> "PropertyExpression":
        return self._binary("<=", other)

    def __gt__(self, other: Any) -> "PropertyExpression":
        return self._binary(">", other)

    def __ge__(self, other: Any) -> "PropertyExpression":
        return self._binary(">=", other)

    def __eq__(self, other: Any) -> "PropertyExpression":  # type: ignore[override]
        return self._binary("==", other)

    def __ne__(self, other: Any) -> "PropertyExpression":  # type: ignore[override]
        return self._binary("!=", other)

    __hash__ = object.__hash__

    def __and__(self, other: Any) -> "PropertyExpression":
        return self._binary("and", other)

    def __rand__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("and", other)

    def __or__(self, other: Any) -> "PropertyExpression":
        return self._binary("or", other)

    def __ror__(self, other: Any) -> "PropertyExpression":
        return self._rbinary("or", other)

    def __invert__(self) -> "PropertyExpression":
        return PropertyExpression({"op": "not", "arg": self.to_fs()})

    def __bool__(self) -> bool:
        raise TypeError(
            "Expression conditions cannot be evaluated as Python booleans; "
            "use '&' for and, '|' for or, and '~' for not."
        )

    def __neg__(self) -> "PropertyExpression":
        return PropertyExpression({"op": "neg", "arg": self.to_fs()})

    def magnitude(self, units: Any) -> "PropertyExpression":
        """Convert a quantity expression to a scalar magnitude.

        Args:
            units: Target units for the magnitude operation.

        Returns:
            A new expression node representing the magnitude conversion.
        """

        return PropertyExpression(
            {
                "op": "magnitude",
                "arg": self.to_fs(),
                "units": unit_expression(units),
            }
        )

    def to(self, units: Any) -> "PropertyExpression":
        """Convert this expression to target units.

        Args:
            units: Target units accepted by ``unit_expression``.

        Returns:
            A new expression node representing the unit conversion.
        """

        return PropertyExpression(
            {
                "op": "convert",
                "arg": self.to_fs(),
                "units": unit_expression(units),
            }
        )

    def __repr__(self) -> str:
        return f"PropertyExpression({self.node!r})"

    def _binary(self, op: str, other: Any) -> "PropertyExpression":
        return PropertyExpression(
            {
                "op": op,
                "args": [self.to_fs(), PropertyExpression.from_value(other).to_fs()],
            }
        )

    def _rbinary(self, op: str, other: Any) -> "PropertyExpression":
        return PropertyExpression(
            {
                "op": op,
                "args": [PropertyExpression.from_value(other).to_fs(), self.to_fs()],
            }
        )


def _expression_operand(value: Any) -> Any:
    if not isinstance(value, xr.DataArray):
        return value
    values = np.asarray(value.values)
    units = value.attrs.get("units")
    if units is None:
        return values
    return ureg.Quantity(values, unit_expression(units))


def _inline_dataarray_coordinates_to_fs(data: xr.DataArray) -> Dict[str, Any]:
    """Serialize exact dimension-coordinate metadata for an inline array value."""

    coords: Dict[str, Any] = {}
    for dim in data.dims:
        coord = data.coords.get(dim)
        if coord is None:
            continue
        coord_payload = {"data": np.asarray(coord.values).tolist()}
        if coord.attrs.get("units"):
            coord_payload["units"] = unit_expression(coord.attrs["units"])
        coords[dim] = coord_payload
    return {"dims": list(data.dims), "coords": coords}


def _dataarray_from_inline_value(
    value: Any,
    *,
    dims: Any,
    coords: Any,
) -> xr.DataArray:
    """Reconstruct an inline array value with its serialized dimensions."""

    if not isinstance(dims, Sequence) or isinstance(dims, (str, bytes)):
        raise ValueError("Inline property dims must be a sequence")
    dimension_names = [str(dim) for dim in dims]
    if not isinstance(coords, Mapping):
        raise ValueError("Inline property coords must be a mapping")

    restored_coords: Dict[str, Any] = {}
    for dim in dimension_names:
        coord_payload = coords.get(dim)
        if coord_payload is None:
            continue
        if isinstance(coord_payload, Mapping):
            coord_values = coord_payload.get(
                "data",
                coord_payload.get("values"),
            )
            if coord_values is None:
                raise ValueError(f"Inline property coord {dim!r} requires data")
            coord = xr.DataArray(coord_values, dims=[dim])
            if coord_payload.get("units"):
                coord.attrs["units"] = unit_expression(coord_payload["units"])
            restored_coords[dim] = coord
        else:
            restored_coords[dim] = coord_payload

    return xr.DataArray(
        data=np.asarray(value),
        dims=dimension_names,
        coords=restored_coords,
    )


def _expression_args(
    node: Mapping[str, Any],
    *,
    references: Mapping[str, Any],
    variables: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> List[Any]:
    args = node.get("args")
    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        raise ValueError(f"Expression operation {node.get('op')!r} requires args")
    return [
        _evaluate_expression_node(
            arg,
            references=references,
            variables=variables,
            fields=fields,
        )
        for arg in args
    ]


def _expression_arg(
    node: Mapping[str, Any],
    *,
    references: Mapping[str, Any],
    variables: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> Any:
    if "arg" not in node:
        raise ValueError(f"Expression operation {node.get('op')!r} requires an arg")
    return _evaluate_expression_node(
        node["arg"],
        references=references,
        variables=variables,
        fields=fields,
    )


def _expression_boolean(value: Any) -> np.ndarray:
    if is_quantity(value):
        value = value.to(ureg.dimensionless).magnitude
    return np.asarray(value, dtype=bool)


def _expression_reduce(op: Callable[[Any, Any], Any], values: List[Any]) -> Any:
    if not values:
        raise ValueError("Variadic expression operation requires at least one arg")
    result = values[0]
    for value in values[1:]:
        result = op(result, value)
    return result


def _evaluate_expression_node(
    node: Any,
    *,
    references: Mapping[str, Any],
    variables: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> Any:
    if isinstance(node, PropertyExpression):
        node = node.node
    if not isinstance(node, Mapping):
        raise ValueError(f"Expression nodes must be mappings; got {node!r}")

    if "ref" in node:
        name = canonical_property_name(node["ref"])
        if name not in references:
            raise ValueError(f"Expression references unavailable property {name!r}")
        return references[name]
    if "field" in node:
        name = str(node["field"])
        if name not in fields:
            raise ValueError(f"Expression references unavailable field {name!r}")
        return fields[name]
    if "var" in node:
        name = str(node["var"])
        if name not in variables:
            raise ValueError(f"Expression references unavailable variable {name!r}")
        return variables[name]
    if "value" in node:
        value = node["value"]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            value = np.asarray(value)
        units = node.get("units")
        if units is not None:
            return ureg.Quantity(value, unit_expression(units))
        return value

    op = node.get("op")
    if not isinstance(op, str):
        raise ValueError(f"Expression node has no operation: {node!r}")

    if op == "case":
        if "else" not in node:
            raise ValueError("Expression case operation requires an else value")
        result = _evaluate_expression_node(
            node["else"],
            references=references,
            variables=variables,
            fields=fields,
        )
        branches = node.get("branches")
        if not isinstance(branches, Sequence) or isinstance(branches, (str, bytes)):
            raise ValueError("Expression case operation requires branches")
        for branch in reversed(branches):
            if not isinstance(branch, Mapping) or not {"if", "then"}.issubset(branch):
                raise ValueError("Expression case branches require if and then nodes")
            condition = _evaluate_expression_node(
                branch["if"],
                references=references,
                variables=variables,
                fields=fields,
            )
            value = _evaluate_expression_node(
                branch["then"],
                references=references,
                variables=variables,
                fields=fields,
            )
            result = np.where(_expression_boolean(condition), value, result)
        return result

    if op in {"not", "neg", "magnitude", "convert"}:
        value = _expression_arg(
            node,
            references=references,
            variables=variables,
            fields=fields,
        )
        if op == "not":
            return np.logical_not(_expression_boolean(value))
        if op == "neg":
            return -value
        units = node.get("units")
        if units is None:
            raise ValueError(f"Expression operation {op!r} requires units")
        converted = (
            value.to(unit_expression(units))
            if is_quantity(value)
            else ureg.Quantity(value).to(unit_expression(units))
        )
        return converted.magnitude if op == "magnitude" else converted

    unary_ops: Dict[str, Callable[[Any], Any]] = {
        "abs": np.abs,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
        "floor": np.floor,
        "ceil": np.ceil,
        "sign": np.sign,
    }
    if op in unary_ops:
        return unary_ops[op](
            _expression_arg(
                node,
                references=references,
                variables=variables,
                fields=fields,
            )
        )

    values = _expression_args(
        node,
        references=references,
        variables=variables,
        fields=fields,
    )
    if op in {"add", "sub", "mul", "div", "pow", ">", ">=", "<", "<=", "==", "!="}:
        if len(values) != 2:
            raise ValueError(f"Expression operation {op!r} requires two args")
        left, right = values
        binary_ops: Dict[str, Callable[[Any, Any], Any]] = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b,
            "pow": lambda a, b: a**b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        return binary_ops[op](left, right)
    if op == "and":
        return _expression_reduce(
            np.logical_and,
            [_expression_boolean(value) for value in values],
        )
    if op == "or":
        return _expression_reduce(
            np.logical_or,
            [_expression_boolean(value) for value in values],
        )
    if op == "min":
        return _expression_reduce(np.minimum, values)
    if op == "max":
        return _expression_reduce(np.maximum, values)
    if op == "atan2":
        if len(values) != 2:
            raise ValueError("Expression operation 'atan2' requires two args")
        return np.arctan2(values[0], values[1])
    if op == "clamp":
        if len(values) != 3:
            raise ValueError("Expression operation 'clamp' requires three args")
        return np.clip(values[0], values[1], values[2])
    raise ValueError(f"Unsupported expression operation {op!r}")


def prop(name: str) -> PropertyExpression:
    """Create a property-reference expression.

    Args:
        name: Property name or alias to reference.

    Returns:
        A ``PropertyExpression`` equivalent to ``PropertyExpression.ref(name)``.
    """
    return PropertyExpression.ref(name)


def ref(name: str) -> PropertyExpression:
    """Create a property-reference expression.

    This is an alias for :func:`prop` for branch-expression DSL symmetry with
    :func:`var`.
    """

    return prop(name)


def var(name: str) -> PropertyExpression:
    """Create a named variable-reference expression."""

    return PropertyExpression.var(name)


def remap(
    value: Any,
    *,
    from_range: Any,
    to_range: Any,
    units: Optional[Any] = None,
    clamp: Optional[bool] = None,
    outside: Optional[str] = None,
) -> PropertyExpression:
    """Affine-remap a property expression between numeric ranges.

    The helper lowers entirely to existing expression operations:
    ``add``, ``sub``, ``mul``, and optionally ``clamp`` or ``case``. Numeric
    range endpoints become unit-bearing literals when ``units`` is supplied or
    when endpoints carry Pint/unit metadata.

    Args:
        value: Expression or scalar to remap. This can be a composed expression,
            such as ``0.5 * prop("Vp")``.
        from_range: Source ``(lower, upper)`` range.
        to_range: Destination ``(lower, upper)`` range.
        units: Optional units for all range endpoint literals.
        clamp: Backward-compatible shortcut for ``outside``. ``True`` means
            ``outside="clamp"``; ``False`` means ``outside="extrapolate"``.
        outside: Behavior outside ``from_range``. ``"clamp"`` saturates to the
            destination range, ``"extrapolate"`` applies the affine map
            everywhere, and ``"preserve"`` leaves values outside ``from_range``
            unchanged.

    Returns:
        A ``PropertyExpression`` representing the remap.
    """

    from_lower, from_upper = _range_pair(from_range, name="from_range")
    to_lower, to_upper = _range_pair(to_range, name="to_range")
    range_units = _infer_range_units(
        units,
        from_lower,
        from_upper,
        to_lower,
        to_upper,
    )
    from_span = _remap_bound_magnitude(
        from_upper, range_units
    ) - _remap_bound_magnitude(from_lower, range_units)
    if from_span == 0.0:
        raise ValueError("from_range endpoints must be distinct")
    to_span = _remap_bound_magnitude(to_upper, range_units) - _remap_bound_magnitude(
        to_lower, range_units
    )
    scale = to_span / from_span

    from_lower_node = _remap_bound_node(from_lower, range_units)
    from_upper_node = _remap_bound_node(from_upper, range_units)
    to_lower_node = _remap_bound_node(to_lower, range_units)
    to_upper_node = _remap_bound_node(to_upper, range_units)
    value_expr = PropertyExpression.from_value(value)

    expression = (
        PropertyExpression.from_value(to_lower_node)
        + (value_expr - from_lower_node) * scale
    )
    outside_mode = _normalize_remap_outside(outside, clamp)
    if outside_mode == "extrapolate":
        return expression

    from_lower_mag = _remap_bound_magnitude(from_lower, range_units)
    from_upper_mag = _remap_bound_magnitude(from_upper, range_units)
    if from_lower_mag <= from_upper_mag:
        interval_lower, interval_upper = from_lower_node, from_upper_node
    else:
        interval_lower, interval_upper = from_upper_node, from_lower_node

    if outside_mode == "preserve":
        condition = (value_expr >= interval_lower) & (value_expr <= interval_upper)
        return PropertyExpression(
            {
                "op": "case",
                "branches": [
                    {
                        "if": condition.to_fs(),
                        "then": expression.to_fs(),
                    }
                ],
                "else": value_expr.to_fs(),
            }
        )

    lower_mag = _remap_bound_magnitude(to_lower, range_units)
    upper_mag = _remap_bound_magnitude(to_upper, range_units)
    clamp_lower, clamp_upper = (
        (to_lower_node, to_upper_node)
        if lower_mag <= upper_mag
        else (to_upper_node, to_lower_node)
    )
    return PropertyExpression(
        {
            "op": "clamp",
            "args": [expression.to_fs(), clamp_lower, clamp_upper],
        }
    )


def coord(system: str, axis: str, units: Optional[Any] = None) -> Dict[str, Any]:
    """Bind an expression symbol to a coordinate-system axis.

    Args:
        system: Coordinate-system name that gives the axis its context.
        axis: Axis name within ``system``.
        units: Optional units into which coordinates are converted before their
            numeric magnitudes are supplied to the expression.

    Returns:
        A solver-facing symbol binding payload.
    """

    system = str(system).strip()
    axis = str(axis).strip()
    if not system:
        raise ValueError("Coordinate symbol binding requires a system")
    if not axis:
        raise ValueError("Coordinate symbol binding requires an axis")
    payload = {"kind": "coordinate", "system": system, "axis": axis}
    if units is not None:
        payload["units"] = unit_expression(units)
    return payload


def _canonicalize_expression_refs(node: Any) -> Any:
    if isinstance(node, Mapping):
        payload = {}
        for key, value in node.items():
            if key == "ref":
                payload[key] = canonical_property_name(value)
            else:
                payload[key] = _canonicalize_expression_refs(value)
        return payload
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        return [_canonicalize_expression_refs(item) for item in node]
    return node


def _is_sympy_expression(value: Any) -> bool:
    return type(value).__module__.startswith("sympy.")


def _sympy_module():
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError(
            "SymPy expressions require sympy. Install `sympy` directly or "
            "reinstall frequensolve with dependencies."
        ) from exc
    return sp


def _sympy_to_expression_node(
    value: Any,
    *,
    symbols: set[str],
    default_symbol: str,
) -> Dict[str, Any]:
    sp = _sympy_module()
    if default_symbol not in {"ref", "var"}:
        raise ValueError("default_symbol must be 'ref' or 'var'")

    if value == sp.S.true:
        return {"value": True}
    if value == sp.S.false:
        return {"value": False}

    if getattr(value, "is_Number", False):
        if not getattr(value, "is_real", True):
            raise ValueError(f"Complex SymPy numbers are not supported: {value!r}")
        if getattr(value, "is_Integer", False):
            return {"value": int(value)}
        return {"value": float(value)}
    if getattr(value, "is_number", False):
        if not getattr(value, "is_real", True):
            raise ValueError(f"Complex SymPy constants are not supported: {value!r}")
        return {"value": float(value.evalf())}

    if getattr(value, "is_Symbol", False):
        name = str(value)
        if name in symbols or default_symbol == "var":
            return {"var": name}
        return {"ref": canonical_property_name(name)}

    if isinstance(value, sp.Piecewise):
        branches = []
        fallback = None
        for expr, condition in value.args:
            expr_node = _sympy_to_expression_node(
                expr,
                symbols=symbols,
                default_symbol=default_symbol,
            )
            if condition == sp.S.true:
                fallback = expr_node
                break
            branches.append(
                {
                    "if": _sympy_to_expression_node(
                        condition,
                        symbols=symbols,
                        default_symbol=default_symbol,
                    ),
                    "then": expr_node,
                }
            )
        if fallback is None:
            raise ValueError("SymPy Piecewise expressions require a True fallback")
        return {"op": "case", "branches": branches, "else": fallback}

    if isinstance(value, sp.core.relational.Relational):
        rel_ops = {
            ">": ">",
            ">=": ">=",
            "<": "<",
            "<=": "<=",
            "==": "==",
            "!=": "!=",
        }
        op = rel_ops.get(value.rel_op)
        if op is None:
            raise ValueError(f"Unsupported SymPy relation: {value.rel_op}")
        lhs, rhs = value.args
        return {
            "op": op,
            "args": [
                _sympy_to_expression_node(
                    lhs,
                    symbols=symbols,
                    default_symbol=default_symbol,
                ),
                _sympy_to_expression_node(
                    rhs,
                    symbols=symbols,
                    default_symbol=default_symbol,
                ),
            ],
        }

    if value.func == sp.Add:
        return _fold_expression_args(
            "add",
            value.args,
            symbols=symbols,
            default_symbol=default_symbol,
        )
    if value.func == sp.Mul:
        return _fold_expression_args(
            "mul",
            value.args,
            symbols=symbols,
            default_symbol=default_symbol,
        )
    if value.func == sp.Pow:
        base, exponent = value.args
        return {
            "op": "pow",
            "args": [
                _sympy_to_expression_node(
                    base,
                    symbols=symbols,
                    default_symbol=default_symbol,
                ),
                _sympy_to_expression_node(
                    exponent,
                    symbols=symbols,
                    default_symbol=default_symbol,
                ),
            ],
        }
    if value.func == sp.And:
        return _fold_expression_args(
            "and",
            value.args,
            symbols=symbols,
            default_symbol=default_symbol,
        )
    if value.func == sp.Or:
        return _fold_expression_args(
            "or",
            value.args,
            symbols=symbols,
            default_symbol=default_symbol,
        )
    if value.func == sp.Not:
        (arg,) = value.args
        return {
            "op": "not",
            "arg": _sympy_to_expression_node(
                arg,
                symbols=symbols,
                default_symbol=default_symbol,
            ),
        }
    unary_function_ops = _sympy_unary_function_ops(sp)
    if value.func in unary_function_ops:
        (arg,) = value.args
        return {
            "op": unary_function_ops[value.func],
            "arg": _sympy_to_expression_node(
                arg,
                symbols=symbols,
                default_symbol=default_symbol,
            ),
        }
    variadic_function_ops = _sympy_variadic_function_ops(sp)
    if value.func in variadic_function_ops:
        return _function_expression_args(
            variadic_function_ops[value.func],
            value.args,
            symbols=symbols,
            default_symbol=default_symbol,
        )
    if isinstance(value, sp.Function):
        raise ValueError(f"Unsupported SymPy function: {value.func}")

    raise ValueError(f"Unsupported SymPy expression: {value!r}")


def _sympy_unary_function_ops(sp) -> Dict[Any, str]:
    return {
        sp.Abs: "abs",
        sp.exp: "exp",
        sp.log: "log",
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",
        sp.asin: "asin",
        sp.acos: "acos",
        sp.atan: "atan",
        sp.sinh: "sinh",
        sp.cosh: "cosh",
        sp.tanh: "tanh",
        sp.floor: "floor",
        sp.ceiling: "ceil",
        sp.sign: "sign",
    }


def _sympy_variadic_function_ops(sp) -> Dict[Any, str]:
    return {
        sp.Min: "min",
        sp.Max: "max",
        sp.atan2: "atan2",
    }


def _function_expression_args(
    op: str,
    args: Sequence[Any],
    *,
    symbols: set[str],
    default_symbol: str,
) -> Dict[str, Any]:
    nodes = [
        _sympy_to_expression_node(
            arg,
            symbols=symbols,
            default_symbol=default_symbol,
        )
        for arg in args
    ]
    if not nodes:
        raise ValueError(f"SymPy {op} function requires at least one argument")
    return {"op": op, "args": nodes}


def _fold_expression_args(
    op: str,
    args: Sequence[Any],
    *,
    symbols: set[str],
    default_symbol: str,
) -> Dict[str, Any]:
    nodes = [
        _sympy_to_expression_node(
            arg,
            symbols=symbols,
            default_symbol=default_symbol,
        )
        for arg in args
    ]
    if not nodes:
        raise ValueError(f"SymPy {op} expression requires at least one argument")
    if len(nodes) == 1:
        return nodes[0]
    out = {"op": op, "args": [nodes[0], nodes[1]]}
    for node in nodes[2:]:
        out = {"op": op, "args": [out, node]}
    return out


def _normalize_expression_symbols(
    symbols: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if symbols is None:
        return {}
    out = {}
    for raw_name, raw_binding in symbols.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("Expression symbol names cannot be empty")
        out[name] = _normalize_expression_symbol_binding(name, raw_binding)
    return out


def _normalize_expression_symbol_binding(name: str, binding: Any) -> Dict[str, Any]:
    if hasattr(binding, "to_fs"):
        binding = binding.to_fs()
    if isinstance(binding, str):
        return {"kind": "coordinate", "system": binding, "axis": name}
    if not isinstance(binding, Mapping):
        raise TypeError(
            "Expression symbol bindings must be mappings, strings, or objects "
            f"with to_fs(); got {type(binding).__name__}"
        )

    payload = copy.deepcopy(dict(binding))
    if "coordinate_system" in payload:
        coordinate_system = payload.pop("coordinate_system")
        if "system" in payload and payload["system"] != coordinate_system:
            raise ValueError("Specify only one of system or coordinate_system")
        payload["system"] = coordinate_system
    if payload.get("kind") == "coordinate":
        if not payload.get("system"):
            raise ValueError("Coordinate symbol binding requires system")
        if not payload.get("axis"):
            raise ValueError("Coordinate symbol binding requires axis")
        if payload.get("units") is not None:
            payload["units"] = unit_expression(payload["units"])
    return payload


class PropertyMap(MutableMapping):
    """Mutable mapping of names to ``Property`` objects.

    Args:
        values: Optional initial mapping of property names to property-like
            values.
        grid: Default grid used when coercing array or file-backed values.
        units: Default units applied to properties that do not specify units.
        system: Default coordinate-system name applied to properties that do
            not specify a system.
        canonicalize_keys: Whether to canonicalize material-property aliases.
            Disable this for mappings whose names belong to another namespace.
    """

    def __init__(
        self,
        values: Optional[Mapping[str, Any]] = None,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        canonicalize_keys: bool = True,
    ):
        """Create a mapping from property names to :class:`Property` objects.

        Args:
            values: Initial property mapping. Values are normalized with
                :meth:`Property.from_value`.
            grid: Optional grid metadata inherited by array-like values.
            units: Default units applied to values that do not specify units.
            system: Default coordinate-system name applied to values that do
                not specify one.
            canonicalize_keys: Whether keys should use material-property alias
                normalization.
        """

        self._store: Dict[str, Property] = {}
        self.grid = grid
        self.units = unit_expression(units) if units is not None else None
        self.system = system
        self.canonicalize_keys = bool(canonicalize_keys)
        if values:
            self.update(values)

    def __getitem__(self, key: str) -> "Property":
        """Return a property by its configured key normalization.

        Args:
            key: Stored name or supported material-property alias.

        Returns:
            Stored :class:`Property` instance.
        """

        return self._store[self._normalize_key(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        prop = (
            copy.deepcopy(value)
            if isinstance(value, Property)
            else Property.from_value(value, grid=self.grid)
        )
        if self.units is not None and prop.units is None:
            prop.units = self.units
        if self.system is not None and prop.system is None:
            prop.system = self.system
        self._store[self._normalize_key(key)] = prop

    def __delitem__(self, key: str) -> None:
        del self._store[self._normalize_key(key)]

    def __iter__(self) -> Iterator[str]:
        """Iterate over stored names in insertion order."""

        return iter(self._store)

    def __len__(self) -> int:
        """Return the number of stored properties."""

        return len(self._store)

    def __contains__(self, key: object) -> bool:
        try:
            return self._normalize_key(key) in self._store
        except Exception:
            return False

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        file_factory: Optional[Callable[[str, "Property"], Path]] = None,
        dataset_factory: Optional[Callable[[str, "Property"], str]] = None,
        preserve_inline_coordinates: bool = False,
    ) -> Dict[str, Any]:
        """Serialize every stored property.

        Args:
            ctx: Optional export context used for project-relative paths and
                HDF5-backed storage.
            file_factory: Optional callback that returns the raw binary file
                path for a non-constant in-memory property.
            dataset_factory: Optional callback that returns the HDF5 dataset
                path for a non-constant in-memory property.
            preserve_inline_coordinates: Include exact xarray dimension and
                coordinate metadata when an array is serialized as an inline
                value.

        Returns:
            Mapping from canonical property name to serialized property payload.
        """

        payload = {}
        for key, prop in self._store.items():
            file = None
            dataset = None
            use_store = ctx is not None and getattr(ctx, "store", None) is not None
            if (
                file_factory is not None
                and not use_store
                and not prop.is_constant
                and prop.darr is not None
            ):
                file = file_factory(key, prop)
            if (
                dataset_factory is not None
                and not prop.is_constant
                and prop.darr is not None
            ):
                dataset = dataset_factory(key, prop)
            payload[key] = prop.to_fs(
                ctx=ctx,
                file=file,
                dataset=dataset,
                preserve_inline_coordinates=preserve_inline_coordinates,
            )
        return payload

    def __repr__(self) -> str:
        keys = ", ".join(self._store)
        return f"PropertyMap({{{keys}}})"

    def _normalize_key(self, key: Any) -> str:
        if self.canonicalize_keys:
            return canonical_property_name(key)
        normalized = str(key)
        if not normalized.strip():
            raise ValueError("Property map keys cannot be empty")
        return normalized


class Property:
    """Material property value, expression, or file reference.

    A property can be a scalar constant, Pint quantity, in-memory
    ``xarray.DataArray``, local file loaded into memory, lazy file reference,
    HDF5 dataset locator, remote solver-visible path, or derived
    ``PropertyExpression``.

    Args:
        data: Property value or reference to wrap.
        grid: Grid metadata used for file-backed or ungridded array values.
        scale: Multiplicative scale applied to loaded or in-memory numeric data.
        units: Optional units for the property values.
        system: Optional coordinate-system name for array dimensions.
        format: Optional explicit file format hint for lazy file references.
        absolute: Whether a file path should be serialized as absolute or
            solver-visible.
        read: Whether a local file path should be read immediately.
        **kwargs: Extra payload fields preserved during serialization.

    Raises:
        ValueError: If ``data`` has an unsupported type or conflicting
            coordinate-system aliases are supplied.
    """

    def __init__(
        self,
        data: Union[
            int, float, str, Path, np.ndarray, xr.DataArray, DispersionScaling
        ] = 0.0,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        format: Optional[str] = None,
        absolute: bool = False,
        read: bool = True,
        fill_invalid: Optional[str] = None,
        valid_range: Optional[Any] = None,
        valid_min: Optional[Any] = None,
        valid_max: Optional[Any] = None,
        **kwargs,
    ):
        """Normalize user property input into one solver-facing representation.

        ``data`` may be a scalar, Pint quantity, numpy array, xarray
        ``DataArray``, local/remote file path, HDF5 locator, expression, or
        dispersion-scaled value. Set ``read=False`` through :meth:`file` when a
        file should be referenced without loading it locally.
        """

        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")
        system = _resolve_system_alias(system, kwargs.pop("coordinate_system", None))

        self.dispersion = None
        if isinstance(data, DispersionScaling):
            self.dispersion = data.dispersion
            data = data.property

        data_is_sympy = _is_sympy_expression(data)
        if is_quantity(data) and units is None:
            units = data.units
        self.scale = float(scale)
        self.units = unit_expression(units) if units is not None else None
        self.valid_range, _range_units = _normalize_valid_range(
            valid_range,
            valid_min=valid_min,
            valid_max=valid_max,
            units=self.units,
        )
        self.fill_invalid = _normalize_fill_invalid(fill_invalid)
        self.system = system
        self.format = format if data_is_sympy else format or _infer_file_format(data)
        self.absolute = bool(absolute)
        self.extra = dict(kwargs)
        self.file_path: Optional[Union[Path, str]] = None
        self.file_grid = grid
        self.darr: Optional[xr.DataArray] = None
        self.expression: Optional[PropertyExpression] = None
        self.expression_symbols: Dict[str, Any] = {}
        self.is_remote = False
        self.remote_path = None
        self.remote_scale = self.scale

        if isinstance(data, PropertyExpression):
            self.expression = data
            return
        if data_is_sympy:
            other = self.expr(
                data,
                units=self.units,
                system=self.system,
                fill_invalid=self.fill_invalid,
                valid_range=self.valid_range,
                **self.extra,
            )
            self.__dict__.update(other.__dict__)
            return

        if is_quantity(data):
            q = quantity_to_fs(data)
            data = q["value"]
            self.units = q["units"]

        if isinstance(data, Mapping):
            payload = dict(data)
            if self.units is not None and "units" not in payload:
                payload["units"] = self.units
            if self.valid_range is not None and "valid_range" not in payload:
                payload["valid_range"] = copy.deepcopy(self.valid_range)
            if self.fill_invalid is not None and "fill_invalid" not in payload:
                payload["fill_invalid"] = self.fill_invalid
            if (
                self.system is not None
                and "system" not in payload
                and "coordinate_system" not in payload
            ):
                payload["system"] = self.system
            other = self.from_value(payload, grid=grid)
            self.__dict__.update(other.__dict__)
            return

        if isinstance(data, str) and _is_hdf5_locator(data):
            self.file_path = data
            self.format = self.format or _infer_file_format(data)
            self.darr = None
            return

        if isinstance(data, str):
            parsed = self._parse_legacy_string(data)
            if parsed is not None:
                data, parsed_scale, parsed_grid = parsed
                self.scale = parsed_scale
                self.remote_scale = parsed_scale
                if parsed_grid is not None:
                    self.extra.setdefault("dims", parsed_grid)
            data = Path(data) if not str(data).startswith("remote:") else data

        if isinstance(data, Path) and str(data).startswith("remote:"):
            self.is_remote = True
            self.remote_path = str(data).replace("remote:", "", 1)
            self.file_path = Path(self.remote_path)
            self.format = self.format or _infer_file_format(data)
            self.absolute = True
            self.darr = None
        elif isinstance(data, str) and data.startswith("remote:"):
            self.is_remote = True
            self.remote_path = data.replace("remote:", "", 1)
            self.file_path = Path(self.remote_path)
            self.format = self.format or _infer_file_format(data)
            self.absolute = True
            self.darr = None
        elif isinstance(data, (int, float, np.integer, np.floating)):
            self.darr = xr.DataArray(data=float(data))
        elif isinstance(data, np.ndarray):
            self.darr = xr.DataArray(data=data)
        elif isinstance(data, list):
            self.darr = xr.DataArray(data=np.asarray(data))
        elif isinstance(data, Path):
            self.file_path = data
            self.format = self.format or _infer_file_format(data)
            if read:
                self.darr = Property.read(data.resolve(), grid=grid)
                if self.units is None and "units" in self.darr.attrs:
                    self.units = self.darr.attrs["units"]
                if self.system is None:
                    self.system = self.darr.attrs.get(
                        "system",
                        self.darr.attrs.get("coordinate_system"),
                    )
                if self.scale != 1.0:
                    self.darr = self.darr.copy(deep=True)
                    self.darr.values = self.darr.values * self.scale
            else:
                self.darr = None
        elif isinstance(data, xr.DataArray):
            self.darr = data.copy(deep=True)
            if self.units is None and "units" in data.attrs:
                self.units = data.attrs["units"]
            if self.system is None:
                self.system = data.attrs.get(
                    "system",
                    data.attrs.get("coordinate_system"),
                )
        else:
            raise ValueError(f"Unknown property type: {type(data)}")

        if self.scale != 1.0 and self.darr is not None and self.file_path is None:
            self.darr = self.darr.copy(deep=True)
            self.darr.values = self.darr.values * self.scale

    @classmethod
    def file(
        cls,
        path: Union[str, Path],
        *,
        scale: float = 1.0,
        units: Optional[Any] = None,
        grid: Optional[Union[CartesianGrid, xr.DataArray, Mapping[str, Any]]] = None,
        format: Optional[str] = None,
        absolute: bool = False,
        remote: bool = False,
        system: Optional[str] = None,
        coordinate_system: Optional[str] = None,
        fill_invalid: Optional[str] = None,
        valid_range: Optional[Any] = None,
        valid_min: Optional[Any] = None,
        valid_max: Optional[Any] = None,
        **extra,
    ) -> "Property":
        """Create a lazy file-backed property.

        Args:
            path: Local path, remote path prefixed by ``remote:``, or HDF5
                locator of the form ``file.h5:dataset``.
            scale: Multiplicative scale recorded for the file data.
            units: Optional property value units.
            grid: Optional grid metadata required by raw binary files and some
                external formats.
            format: Optional explicit file format hint.
            absolute: Serialize the path as solver-visible instead of
                project-relative.
            remote: Treat the path as solver-visible and never try to read it
                from the host. Serialized payloads use ``absolute: true`` for
                solver compatibility.
            system: Optional coordinate-system name for the property grid.
            coordinate_system: Alias accepted in serialized payloads.
            fill_invalid: Optional invalid-sample repair mode: ``"nearest"`` or
                ``"none"``.
            valid_range: Optional valid sample range mapping or ``(lower, upper)``
                tuple.
            valid_min: Optional lower valid sample bound.
            valid_max: Optional upper valid sample bound.
            **extra: Additional payload fields to preserve.

        Returns:
            A ``Property`` that references the file without reading it.
        """

        system = _resolve_system_alias(system, coordinate_system)
        path_text = str(path)
        remote = bool(remote) or path_text.startswith("remote:")
        data_arg = str(path) if _is_hdf5_locator(path) else Path(path)
        prop = cls(
            data_arg,
            grid=grid,
            scale=scale,
            units=units,
            system=system,
            format=format or _infer_file_format(path),
            absolute=bool(absolute) or remote,
            read=False,
            fill_invalid=fill_invalid,
            valid_range=valid_range,
            valid_min=valid_min,
            valid_max=valid_max,
            **extra,
        )
        prop.file_path = str(path) if _is_hdf5_locator(path) else Path(path)
        prop.absolute = bool(prop.absolute) or remote
        prop.is_remote = bool(prop.absolute)
        prop.remote_path = _strip_remote_prefix(path_text) if prop.is_remote else None
        return prop

    @classmethod
    def expr(
        cls,
        expression: Any,
        *,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        coordinate_system: Optional[str] = None,
        symbols: Optional[Mapping[str, Any]] = None,
        fill_invalid: Optional[str] = None,
        valid_range: Optional[Any] = None,
        valid_min: Optional[Any] = None,
        valid_max: Optional[Any] = None,
        **extra,
    ) -> "Property":
        """Create a property derived from a solver expression.

        Args:
            expression: Expression object, scalar, or serialized expression
                payload accepted by ``PropertyExpression.from_value``.
            units: Optional units associated with the derived value.
            system: Optional coordinate-system name for the derived value.
            coordinate_system: Alias accepted in serialized payloads.
            symbols: Optional expression-symbol bindings keyed by symbol name.
            fill_invalid: Optional invalid-sample repair mode: ``"nearest"`` or
                ``"none"``.
            valid_range: Optional valid sample range mapping or ``(lower, upper)``
                tuple.
            valid_min: Optional lower valid sample bound.
            valid_max: Optional upper valid sample bound.
            **extra: Additional payload fields to preserve.

        Returns:
            A ``Property`` containing a ``PropertyExpression`` instead of
            concrete data.
        """

        system = _resolve_system_alias(system, coordinate_system)
        prop = cls(
            0.0,
            units=units,
            system=system,
            fill_invalid=fill_invalid,
            valid_range=valid_range,
            valid_min=valid_min,
            valid_max=valid_max,
            **extra,
        )
        prop.darr = None
        prop.file_path = None
        prop.file_grid = None
        prop.is_remote = False
        prop.remote_path = None
        prop.expression_symbols = _normalize_expression_symbols(symbols)
        prop.expression = PropertyExpression.from_value(
            expression,
            symbols=prop.expression_symbols,
            default_symbol="ref",
        )
        return prop

    @classmethod
    def from_value(cls, value: Any, grid: Optional[xr.DataArray] = None) -> "Property":
        """Coerce a user value or serialized payload into a ``Property``.

        Args:
            value: Existing property, expression, scalar, array, file payload,
                expression payload, or other property-like value.
            grid: Optional default grid metadata used when ``value`` does not
                include its own grid.

        Returns:
            A normalized ``Property`` instance.
        """

        if isinstance(value, Property):
            return value
        if isinstance(value, PropertyExpression):
            return cls.expr(value)
        if _is_sympy_expression(value):
            return cls.expr(value)
        if isinstance(value, Mapping):
            payload = dict(value)
            _normalize_system_alias(payload)
            if "expr" in payload:
                payload.pop("depends_on", None)
                return cls.expr(
                    payload.pop("expr"),
                    units=payload.pop("units", None),
                    system=payload.pop("system", None),
                    symbols=payload.pop("symbols", None),
                    **payload,
                )
            if "value" in payload and "file" not in payload:
                prop_grid = payload.pop("grid", grid)
                inline_dims = payload.pop("dims", None)
                inline_coords = (
                    payload.pop("coords", {}) if inline_dims is not None else None
                )
                prop_units = payload.pop("units", None)
                prop_system = payload.pop("system", None)
                prop_value = payload.pop("value")
                if inline_dims is not None:
                    prop_value = _dataarray_from_inline_value(
                        prop_value,
                        dims=inline_dims,
                        coords=inline_coords,
                    )
                    if prop_units is not None:
                        prop_value.attrs["units"] = unit_expression(prop_units)
                    if prop_system is not None:
                        prop_value.attrs["system"] = prop_system
                return cls(
                    prop_value,
                    grid=prop_grid,
                    units=prop_units,
                    system=prop_system,
                    **payload,
                )
            if "file" in payload:
                prop_grid = payload.pop("grid", grid)
                return cls.file(
                    payload.pop("file"),
                    scale=payload.pop("scale", 1.0),
                    units=payload.pop("units", None),
                    grid=prop_grid,
                    format=payload.pop("format", None),
                    absolute=payload.pop("absolute", False),
                    remote=payload.pop("remote", False),
                    system=payload.pop("system", None),
                    **payload,
                )
        return cls(value, grid=grid)

    @staticmethod
    def _parse_legacy_string(value: str):
        if "|" not in value:
            return None
        parts = value.split("|")
        path = parts[0]
        scale = 1.0
        dims = None
        for part in parts[1:]:
            try:
                scale = float(part)
            except ValueError:
                dims = [d for d in part]
        return path, scale, dims

    @property
    def data(self):
        """Return the in-memory data array.

        Returns:
            The underlying ``xarray.DataArray`` when the property is materialized,
            otherwise ``None`` for expressions, remote references, and lazy files.
        """

        return self.darr

    @property
    def is_constant(self) -> bool:
        """Return whether the property is a materialized scalar constant.

        Returns:
            ``True`` when the property contains an in-memory scalar data array;
            ``False`` for arrays, expressions, and file references.
        """

        if self.expression is not None or self.is_remote or self.darr is None:
            return False
        return len(self.darr.coords) == 0

    @property
    def extrema(self):
        """Return the minimum and maximum materialized property values.

        Returns:
            Tuple of ``xarray.DataArray`` scalar reductions ``(min, max)``.

        Raises:
            ValueError: If the property is an expression or lazy file reference.
        """

        if self.expression is not None:
            raise ValueError("Cannot get extrema of a derived property expression")
        if self.is_remote or self.darr is None:
            raise ValueError(
                "Cannot get extrema of a file reference without loading data"
            )
        return (
            self.darr.min(skipna=True).compute(),
            self.darr.max(skipna=True).compute(),
        )

    @property
    def grid(self) -> CartesianGrid:
        """Return Cartesian grid metadata for this property.

        Returns:
            A ``CartesianGrid`` inferred from explicit file grid metadata or the
            in-memory data array.

        Raises:
            ValueError: If a file-backed property has no explicit grid metadata.
        """

        if isinstance(self.file_grid, CartesianGrid):
            return self.file_grid
        if isinstance(self.file_grid, Mapping):
            return CartesianGrid.from_fs(dict(self.file_grid))
        if isinstance(self.file_grid, xr.DataArray):
            return CartesianGrid.from_xarray(self.file_grid)
        if self.darr is None:
            raise ValueError("File-backed property requires explicit grid metadata")
        return CartesianGrid.from_xarray(self.darr)

    def get(self, grid: Optional[xr.DataArray] = None):
        """Return materialized property values.

        Args:
            grid: Optional target xarray grid. Constants are broadcast to this
                grid; gridded data are linearly interpolated to matching
                coordinate names with nearest extrapolation used for edge NaNs.

        Returns:
            A scalar value or ``xarray.DataArray`` on the requested grid.

        Raises:
            ValueError: If the property is an expression, lazy file reference,
                or its dimensions are incompatible with the requested grid.
        """

        if self.expression is not None:
            raise ValueError("Cannot access data from a derived property expression")
        if self.is_remote or self.darr is None:
            raise ValueError(f"Cannot access data from file property: {self.file_path}")

        if grid is None:
            if self.is_constant:
                return self.darr.values
            coords = self.darr.coords
        else:
            coords = grid.coords

        if _coords_compatible(coords, self.darr.coords):
            result = self.darr
        else:
            if self.is_constant:
                dims = grid.dims
                shape = tuple(len(coords[dim]) for dim in dims)
                result = xr.DataArray(
                    data=np.full(shape, self.darr.values), dims=dims, coords=coords
                )
            elif _dims_compatible(self.darr.dims, coords):
                coords2 = _coords_in_data_units(
                    coords,
                    self.darr.coords,
                    self.darr.dims,
                )
                out = self.darr.interp(coords=coords2, method="linear")
                if np.isnan(out.values).any():
                    nearest_interp = self.darr.interp(
                        coords=coords2,
                        method="nearest",
                        kwargs={"fill_value": "extrapolate"},
                    )
                    nan_mask = np.isnan(out.values)
                    out.values[nan_mask] = nearest_interp.values[nan_mask]
                if len(coords.keys()) != len(out.coords.keys()):
                    out = out.broadcast_like(grid)
                result = out
            else:
                raise self._dimension_error(coords)
        return result

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        file: Optional[Union[str, Path]] = None,
        dataset: Optional[str] = None,
        preserve_inline_coordinates: bool = False,
    ) -> Dict[str, Any]:
        """Serialize the property for a FrequenSolve input payload.

        Args:
            ctx: Optional export context used for project-relative paths and
                HDF5-backed storage.
            file: Optional binary file path used when writing non-constant
                in-memory arrays outside an HDF5 store.
            dataset: Optional HDF5 dataset path used when ``ctx`` provides a
                store.
            preserve_inline_coordinates: Include exact xarray dimension and
                coordinate metadata when this property is emitted as an inline
                array value.

        Returns:
            Serialized property payload containing one of ``value``, ``file``,
            or ``expr`` plus optional units, system, grid, and extra fields.
        """

        if self.expression is not None:
            payload = {"expr": self.expression.to_fs(ctx)}
            depends_on = self.expression.depends_on()
            if depends_on:
                payload["depends_on"] = depends_on
            if self.expression_symbols:
                payload["symbols"] = copy.deepcopy(self.expression_symbols)
        elif self.file_path is not None and self.darr is None:
            payload = self._file_payload(ctx)
        elif self.is_constant:
            value = self.get()
            payload = {"value": value.tolist() if hasattr(value, "tolist") else value}
        else:
            if (
                ctx is not None
                and getattr(ctx, "store", None) is not None
                and dataset is not None
            ):
                attrs = {"fs_kind": "property"}
                if self.units is not None:
                    attrs["units"] = self.units
                if self.system is not None:
                    attrs["system"] = self.system
                ref = ctx.store.put_dataarray(dataset, self.darr, attrs=attrs)
                payload = ref.to_fs()
            elif file is None:
                payload = {"value": self.darr.values.tolist()}
                if preserve_inline_coordinates:
                    payload.update(_inline_dataarray_coordinates_to_fs(self.darr))
            else:
                path = Path(file)
                path.parent.mkdir(parents=True, exist_ok=True)
                self.write(path)
                rel = ctx.relative_to_project(path) if ctx else path
                payload = {"file": rel}
                payload["grid"] = (
                    self.grid.to_fs()
                    if hasattr(self.grid, "to_fs")
                    else self.grid.to_fs()
                )

        if self.units is not None:
            payload["units"] = self.units
        if self.system is not None:
            payload["system"] = self.system
        if self.fill_invalid is not None:
            payload["fill_invalid"] = self.fill_invalid
        if self.valid_range is not None:
            payload["valid_range"] = copy.deepcopy(self.valid_range)
        return merge_extra(payload, self.extra, "Property")

    def _file_payload(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        path = self.file_path
        if self.is_remote:
            file_value = self.remote_path if self.remote_path is not None else str(path)
        elif isinstance(path, str) and _is_hdf5_locator(path):
            file_value = path
        elif ctx is not None:
            file_value = ctx.relative_to_project(path)
        else:
            file_value = path

        payload: Dict[str, Any] = {"file": file_value}
        if self.format is not None:
            payload["format"] = self.format
        if self.absolute:
            payload["absolute"] = True
        if self.scale != 1.0:
            payload["scale"] = self.scale
        try:
            grid = self.grid
            payload["grid"] = grid.to_fs()
        except ValueError:
            pass
        return payload

    def _dimension_error(self, coords: Mapping[str, Any]) -> ValueError:
        return ValueError(
            _property_dimension_error(
                self.darr.dims if self.darr is not None else (),
                coords,
                system=self.system,
            )
        )

    def __iadd__(self, other: Union[float, xr.DataArray]) -> "Property":
        if self.is_remote or self.darr is None:
            raise ValueError(
                f"Cannot perform addition on file property: {self.file_path}"
            )

        if isinstance(other, float):
            self.darr = self.darr + other
        elif isinstance(other, xr.DataArray):
            if self.is_constant or _coords_compatible(self.darr.coords, other.coords):
                self.darr = self.darr + other
            else:
                self.darr = self.darr.interp(other.coords) + other
        else:
            raise ValueError(f"Unknown type for addition: {type(other)}")
        return self

    def __add__(self, other: Union[float, xr.DataArray]) -> "Property":
        prop = copy.deepcopy(self)
        prop += other
        return prop

    def write(self, file: Path):
        """Write materialized property values to a raw binary file.

        Args:
            file: Output path for single-precision floating-point values.

        Returns:
            The output path that was written.

        Raises:
            ValueError: If the property is not materialized in memory.
        """

        if self.is_remote or self.darr is None:
            raise ValueError(f"Cannot write file property: {self.file_path}")
        if not file.parent.exists():
            file.parent.mkdir(parents=True)
        self.darr.values.astype(np.single).tofile(file)
        return file

    def __repr__(self) -> str:
        if self.expression is not None:
            kind = "expression"
        elif self.file_path is not None and self.darr is None:
            kind = f"file={self.file_path!s}"
        elif self.is_constant:
            kind = f"value={self.get()!r}"
        elif self.darr is not None:
            kind = f"array shape={self.darr.shape}"
        else:
            kind = "empty"
        suffix = []
        if self.units is not None:
            suffix.append(f"units={self.units!r}")
        if self.system is not None:
            suffix.append(f"system={self.system!r}")
        detail = ", ".join([kind, *suffix])
        return f"Property({detail})"

    @staticmethod
    def read(file: Path, grid: Optional[xr.DataArray] = None) -> xr.DataArray:
        """Read a property file into an ``xarray.DataArray``.

        Args:
            file: File path or HDF5 locator to read.
            grid: Optional grid metadata required by raw binary files and used
                for interpolation by some readers.

        Returns:
            Data array containing the property values.

        Raises:
            FileNotFoundError: If the referenced file does not exist.
            ValueError: If the file format is unsupported or required grid
                metadata are missing.
        """

        reader = Property._get_reader(file)
        return reader(file, grid=grid)

    @staticmethod
    def _get_reader(file: Path) -> Callable:
        path = (
            Path(str(file).split(":", 1)[0]) if _is_hdf5_locator(file) else Path(file)
        )
        if not path.exists():
            raise FileNotFoundError(f"File {path} not found")
        suffix = path.suffix.lower()
        if suffix == ".bin" or suffix == "":
            return Property._bin_reader
        if suffix == ".rsf":
            return Property._rsf_reader
        if suffix in [".sgy", ".segy"]:
            return Property._segy_reader
        if suffix in [".h5", ".hdf5"]:
            return Property._h5_reader
        if suffix == ".zarr":
            return Property._zarr_reader
        if suffix == ".nc":
            return Property._netcdf_reader
        raise ValueError(f"Unknown file format for {file}")

    @staticmethod
    def _bin_reader(file: Path, grid: xr.DataArray) -> xr.DataArray:
        data = np.fromfile(file, dtype=np.float32).reshape(grid.shape)
        return xr.DataArray(data, coords=grid.coords, dims=grid.dims)

    @staticmethod
    def _rsf_reader(file: Path, **kwargs) -> xr.DataArray:
        header = _read_rsf_header(file)
        axes = []
        axis = 1
        while f"n{axis}" in header:
            axes.append(axis)
            axis += 1
        if not axes:
            raise ValueError(f"RSF header {file} does not define any n# axes")

        n = [int(header[f"n{axis}"]) for axis in axes]
        d = [float(header.get(f"d{axis}", 1.0)) for axis in axes]
        o = [float(header.get(f"o{axis}", 0.0)) for axis in axes]
        dims = [
            Property._rsf_axis_name(header.get(f"label{axis}"), index)
            for index, axis in enumerate(axes)
        ]
        if len(set(dims)) != len(dims):
            dims = Property._default_rsf_dims(len(dims))

        data_path = rsf_binary_path(file)
        if data_path is None:
            raise ValueError(
                f"RSF header {file} does not reference a binary 'in=' file"
            )

        dtype = Property._rsf_dtype(header)
        expected = int(np.prod(n, dtype=np.int64))
        data = np.fromfile(data_path, dtype=dtype, count=expected)
        if data.size != expected:
            raise ValueError(
                f"RSF binary {data_path} contains {data.size} values; "
                f"expected {expected} from header axes {n}"
            )
        data = data.reshape(tuple(n), order="F")
        coords = {
            dim: o[index] + d[index] * np.arange(count)
            for index, (dim, count) in enumerate(zip(dims, n))
        }
        da = xr.DataArray(data, dims=dims, coords=coords, name=header.get("label"))
        if "unit" in header:
            da.attrs["units"] = header["unit"]
        if "label" in header:
            da.attrs["label"] = header["label"]
        for index, axis in enumerate(axes):
            dim = dims[index]
            if f"unit{axis}" in header:
                da.coords[dim].attrs["units"] = header[f"unit{axis}"]
            if f"label{axis}" in header:
                da.coords[dim].attrs["label"] = header[f"label{axis}"]
        return da

    @staticmethod
    def _default_rsf_dims(ndim: int) -> List[str]:
        if ndim == 1:
            return ["x"]
        if ndim == 2:
            return ["x", "z"]
        if ndim == 3:
            return ["x", "y", "z"]
        return [f"dim{axis}" for axis in range(1, ndim + 1)]

    @staticmethod
    def _rsf_axis_name(label: Optional[str], index: int) -> str:
        defaults = Property._default_rsf_dims(index + 1)
        fallback = defaults[index]
        if label is None:
            return fallback
        normalized = "".join(
            char.lower() if char.isalnum() else "_" for char in str(label).strip()
        ).strip("_")
        normalized = "_".join(part for part in normalized.split("_") if part)
        return {"depth": "z"}.get(normalized, normalized or fallback)

    @staticmethod
    def _rsf_dtype(header: Mapping[str, str]) -> np.dtype:
        data_format = header.get("data_format", "native_float").lower()
        dtype_map = {
            "float": np.dtype(np.float32),
            "native_float": np.dtype(np.float32),
            "xdr_float": np.dtype(">f4"),
            "double": np.dtype(np.float64),
            "native_double": np.dtype(np.float64),
            "xdr_double": np.dtype(">f8"),
            "int": np.dtype(np.int32),
            "native_int": np.dtype(np.int32),
            "xdr_int": np.dtype(">i4"),
        }
        if data_format not in dtype_map:
            raise ValueError(f"Unsupported RSF data_format: {data_format!r}")
        dtype = dtype_map[data_format]
        esize = header.get("esize")
        if esize is not None and int(esize) != dtype.itemsize:
            raise ValueError(
                f"RSF esize={esize} does not match {data_format} item size "
                f"{dtype.itemsize}"
            )
        return dtype

    @staticmethod
    def _h5_reader(file: Path, **kwargs) -> xr.DataArray:
        import h5py

        fname, dset = str(file).split(":") if ":" in str(file) else (file, "data")
        with h5py.File(fname, "r") as f:
            dset_obj = f[dset]
            if "dims" in dset_obj.attrs:
                dims = [
                    (dim.decode("utf-8") if isinstance(dim, bytes) else str(dim))
                    for dim in dset_obj.attrs["dims"]
                ]
                coords = {}
                for dim in dims:
                    coordinate = dset_obj.attrs[dim]
                    values = np.asarray(coordinate)
                    if values.ndim == 0:
                        reference = values.item()
                        if isinstance(reference, bytes):
                            reference = reference.decode("utf-8")
                        if (
                            isinstance(reference, str)
                            and reference.startswith("/")
                            and reference.strip("/") in f
                            and isinstance(f[reference.strip("/")], h5py.Dataset)
                        ):
                            coordinate = f[reference.strip("/")][()]
                    coords[dim] = coordinate
            elif "coords" not in f:
                if "grid" not in kwargs or kwargs["grid"] is None:
                    raise ValueError(
                        "Coords not found in h5 file, must be provided via 'grid'"
                    )
                grid = kwargs["grid"]
                coords = grid.coords
                dims = grid.dims
            else:
                dims = f["coords"].attrs["dims"]
                coords = {dim: f["coords"][dim][()] for dim in dims}
            return xr.DataArray(dset_obj[()], coords=coords, dims=dims)

    @staticmethod
    def _netcdf_reader(file: Path, **kwargs) -> xr.DataArray:
        grid = kwargs.pop("grid", None)
        ds = xr.open_dataset(file, **kwargs)
        da = ds[list(ds.data_vars)[0]]
        return da.interp(coords=grid.coords) if grid is not None else da

    @staticmethod
    def _segy_reader(file: Path, **kwargs) -> xr.DataArray:
        try:
            import segyio
        except ModuleNotFoundError as exc:
            from frequensolve._optional import optional_dependency_error

            raise optional_dependency_error(
                "SEG-Y property reader",
                extra="seismic-io",
                dependencies=("segyio",),
                error=exc,
            ) from exc

        with segyio.open(file, mode="r", strict=False) as sgy:
            scale = kwargs.get("scale", 1.0)
            L = kwargs.get("L", 4.5)
            dims = ["x", "z"]
            coords = {"z": sgy.samples / 1000.0, "x": np.linspace(0, L, sgy.tracecount)}
            coords["z"] -= coords["z"][0]
            da = xr.DataArray(
                dims=dims,
                coords=coords,
                data=np.zeros((sgy.tracecount, len(sgy.samples))),
            )
            for i in range(sgy.tracecount):
                da.values[i, :] = (
                    np.array(sgy.trace[i].data[:], dtype=np.float32) * scale
                )
        return da.transpose(*sorted(da.dims))

    @staticmethod
    def _zarr_reader(file: Path, **kwargs) -> xr.DataArray:
        kwargs.pop("grid", None)
        return xr.open_zarr(file, **kwargs)

    def _mask(self, mask: xr.DataArray) -> None:
        if self.is_remote or self.darr is None:
            raise ValueError(f"Cannot mask file property: {self.file_path}")
        if not self.is_constant:
            self.darr = self.darr.where(mask)

    def _like(self, da: xr.DataArray) -> xr.DataArray:
        if self.is_remote or self.darr is None:
            raise ValueError(f"Cannot interpolate file property: {self.file_path}")
        dims = set(self.darr.dims).intersection(set(da.dims))
        coords = {dim: da.coords[dim] for dim in dims}
        return self.darr.interp(
            coords=coords, method="nearest", kwargs={"fill_value": "extrapolate"}
        ).broadcast_like(da)

    def stochastic_perturbation(
        self,
        std: float,
        method: str,
        type: str,
        grid: Optional[xr.DataArray] = None,
        **kwargs,
    ) -> None:
        """Apply a stochastic perturbation to materialized values in place.

        Args:
            std: Standard deviation of the generated perturbation field.
            method: Perturbation generator name. Currently only
                ``"von_karman"`` is supported.
            type: Perturbation application mode, either ``"additive"`` or
                ``"multiplicative"``.
            grid: Optional grid used to generate the perturbation field.
            **kwargs: Generator options such as ``k0``, ``nu``, ``anisotropy``,
                and ``seed``.

        Raises:
            ValueError: If the property is file-backed or ``method`` is not
                supported.
        """

        if self.is_remote or self.darr is None:
            raise ValueError(
                f"Cannot perform stochastic perturbation on file property: {self.file_path}"
            )
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")
        if method != "von_karman":
            raise ValueError(f"Unknown perturbation method: {method}")

        k0 = kwargs.get("k0", [1.0])
        nu = kwargs.get("nu", 0.5)
        anisotropy = kwargs.get("anisotropy", [1.0] * len(self.darr.dims))
        seed = kwargs.get("seed", None)
        grid = grid if grid is not None else self.darr
        da = von_karman_stochastic_field(grid, 0.0, std, k0, nu, anisotropy, seed)

        base = (
            self.darr
            if _coords_compatible(self.darr.coords, grid.coords)
            else self._like(grid)
        )
        if type == "additive":
            self.darr = base + da
        elif type == "multiplicative":
            self.darr = base * (1 + da)


def _ensure_minimum_coordinates(da: xr.DataArray) -> xr.DataArray:
    """Ensure every dimension has at least two coordinates for interpolation."""
    if not da.dims:
        return da
    out = da
    for dim in list(out.dims):
        if len(out.coords[dim]) == 1:
            coord0 = out.coords[dim].values[0]
            coord1 = coord0 + 1
            out = xr.concat([out, out.copy(deep=True)], dim=dim)
            out = out.assign_coords({dim: [coord0, coord1]})
    return out


def _dims_compatible(dims1: Sequence[str], dims2: Sequence[str]) -> bool:
    return set(dims1).issubset(set(dims2))


def _format_names(values: Sequence[str]) -> str:
    return ", ".join(repr(str(value)) for value in values) or "none"


def _property_dimension_error(
    data_dims: Sequence[str],
    coords: Mapping[str, Any],
    *,
    system: Optional[str],
) -> str:
    data_dims = [str(dim) for dim in data_dims]
    coord_names = [str(name) for name in coords.keys()]
    missing = [dim for dim in data_dims if dim not in coord_names]
    if system is not None:
        return (
            f"Property dimensions [{_format_names(missing or data_dims)}] are not "
            f"available on the requested grid for coordinate system {system!r}. "
            f"Grid coordinates are [{_format_names(coord_names)}]. Define "
            f"coordinate-system axes with these property dimension names, bind that "
            "coordinate system to the simulation/model before sampling, or rename "
            "the property dimensions to match available grid coordinates."
        )
    return (
        f"Property dimensions [{_format_names(missing or data_dims)}] are not "
        f"available on the requested grid. Grid coordinates are "
        f"[{_format_names(coord_names)}]. Property array dimensions may be physical "
        "grid coordinates such as 'x', 'y', and 'z', or coordinate-system axes. "
        "For coordinate-system axes, set the property `system`/`coordinate_system` "
        "and define matching axes on the simulation."
    )


def _coords_compatible(
    coords1: Dict[str, ArrayLike],
    coords2: Dict[str, ArrayLike],
    rtol: float = 1e-06,
    atol: float = 1e-08,
) -> bool:
    dims1 = set(coords1.keys())
    dims2 = set(coords2.keys())
    if not _dims_compatible(dims1, dims2):
        return False
    for dim in coords1.keys():
        if dim not in coords2:
            return False
        if len(coords1[dim]) != len(coords2[dim]):
            return False
        if not np.allclose(
            coords1[dim].values, coords2[dim].values, rtol=rtol, atol=atol
        ):
            return False
    return True
