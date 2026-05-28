"""Material-property containers, expressions, file references, and unit metadata."""

from __future__ import annotations

import copy
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

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
    "prop",
    "_ensure_minimum_coordinates",
]


PROPERTY_ALIASES = {
    "vp": "vp",
    "vs": "vs",
    "rho": "rho",
    "density": "rho",
    "qp": "qp",
    "qs": "qs",
    "vadapt": "vadapt",
    "epsilon": "epsilon",
    "gamma": "gamma",
    "delta": "delta",
    "phi": "phi",
    "theta": "theta",
}


def canonical_property_name(name: str) -> str:
    """Normalize a user-facing material property name.

    Args:
        name: Property name or alias, such as ``"Vp"``, ``"density"``, or
            ``"rho"``.

    Returns:
        The canonical lowercase name used in serialized model payloads.
    """

    key = str(name).strip()
    normalized = key.lower()
    return PROPERTY_ALIASES.get(normalized, normalized)


def _is_hdf5_locator(value: Any) -> bool:
    text = str(value)
    if ":" not in text:
        return False
    file_part = text.split(":", 1)[0].lower()
    return file_part.endswith(".h5") or file_part.endswith(".hdf5")


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
    def from_value(cls, value: Any) -> "PropertyExpression":
        """Coerce user input into a ``PropertyExpression``.

        Args:
            value: Existing expression, serialized expression payload, scalar
                value, or mapping containing an ``expr`` field.

        Returns:
            A normalized property expression with canonicalized references.
        """

        if isinstance(value, PropertyExpression):
            return value
        if isinstance(value, Mapping):
            payload = copy.deepcopy(dict(value))
            if "expr" in payload:
                return cls.from_value(payload["expr"])
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
                if "arg" in node:
                    visit(node["arg"])
                for child in node.get("args", []):
                    visit(child)
            elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
                for child in node:
                    visit(child)

        visit(self.node)
        return out

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


def prop(name: str) -> PropertyExpression:
    """Create a property-reference expression.

    Args:
        name: Property name or alias to reference.

    Returns:
        A ``PropertyExpression`` equivalent to ``PropertyExpression.ref(name)``.
    """
    return PropertyExpression.ref(name)


def _canonicalize_expression_refs(node: Any) -> Any:
    if isinstance(node, Mapping):
        payload = {}
        for key, value in node.items():
            if key == "ref":
                payload[key] = canonical_property_name(value)
            elif key in {"arg", "args"}:
                payload[key] = _canonicalize_expression_refs(value)
            else:
                payload[key] = value
        return payload
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        return [_canonicalize_expression_refs(item) for item in node]
    return node


class PropertyMap(MutableMapping):
    """Mutable mapping of canonical property names to ``Property`` objects.

    Args:
        values: Optional initial mapping of property names to property-like
            values.
        grid: Default grid used when coercing array or file-backed values.
        units: Default units applied to properties that do not specify units.
        system: Default coordinate-system name applied to properties that do
            not specify a system.
    """

    def __init__(
        self,
        values: Optional[Mapping[str, Any]] = None,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
    ):
        """Create a mapping from property names to :class:`Property` objects.

        Args:
            values: Initial property mapping. Values are normalized with
                :meth:`Property.from_value`.
            grid: Optional grid metadata inherited by array-like values.
            units: Default units applied to values that do not specify units.
            system: Default coordinate-system name applied to values that do
                not specify one.
        """

        self._store: Dict[str, Property] = {}
        self.grid = grid
        self.units = unit_expression(units) if units is not None else None
        self.system = system
        if values:
            self.update(values)

    def __getitem__(self, key: str) -> "Property":
        """Return a property by canonical or alias name.

        Args:
            key: Property name or supported alias.

        Returns:
            Stored :class:`Property` instance.
        """

        return self._store[canonical_property_name(key)]

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
        self._store[canonical_property_name(key)] = prop

    def __delitem__(self, key: str) -> None:
        del self._store[canonical_property_name(key)]

    def __iter__(self) -> Iterator[str]:
        """Iterate over canonical property names in insertion order."""

        return iter(self._store)

    def __len__(self) -> int:
        """Return the number of stored properties."""

        return len(self._store)

    def __contains__(self, key: object) -> bool:
        try:
            return canonical_property_name(str(key)) in self._store
        except Exception:
            return False

    def to_fs(
        self,
        ctx: Optional[ExportContext] = None,
        file_factory: Optional[Callable[[str, "Property"], Path]] = None,
        dataset_factory: Optional[Callable[[str, "Property"], str]] = None,
    ) -> Dict[str, Any]:
        """Serialize every stored property.

        Args:
            ctx: Optional export context used for project-relative paths and
                HDF5-backed storage.
            file_factory: Optional callback that returns the raw binary file
                path for a non-constant in-memory property.
            dataset_factory: Optional callback that returns the HDF5 dataset
                path for a non-constant in-memory property.

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
            payload[key] = prop.to_fs(ctx=ctx, file=file, dataset=dataset)
        return payload

    def __repr__(self) -> str:
        keys = ", ".join(self._store)
        return f"PropertyMap({{{keys}}})"


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

        self.scale = float(scale)
        self.units = unit_expression(units) if units is not None else None
        self.system = system
        self.format = format
        self.absolute = bool(absolute)
        self.extra = dict(kwargs)
        self.file_path: Optional[Union[Path, str]] = None
        self.file_grid = grid
        self.darr: Optional[xr.DataArray] = None
        self.expression: Optional[PropertyExpression] = None
        self.is_remote = False
        self.remote_path = None
        self.remote_scale = self.scale

        if isinstance(data, PropertyExpression):
            self.expression = data
            return

        if is_quantity(data):
            q = quantity_to_fs(data)
            data = q["value"]
            self.units = q["units"]

        if isinstance(data, Mapping):
            payload = dict(data)
            if self.units is not None and "units" not in payload:
                payload["units"] = self.units
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
            self.format = self.format or "hdf5"
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
            self.absolute = True
            self.darr = None
        elif isinstance(data, str) and data.startswith("remote:"):
            self.is_remote = True
            self.remote_path = data.replace("remote:", "", 1)
            self.file_path = Path(self.remote_path)
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
            if read:
                self.darr = Property.read(data.resolve(), grid=grid)
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
        system: Optional[str] = None,
        coordinate_system: Optional[str] = None,
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
            system: Optional coordinate-system name for the property grid.
            coordinate_system: Alias accepted in serialized payloads.
            **extra: Additional payload fields to preserve.

        Returns:
            A ``Property`` that references the file without reading it.
        """

        system = _resolve_system_alias(system, coordinate_system)
        data_arg = str(path) if _is_hdf5_locator(path) else Path(path)
        prop = cls(
            data_arg,
            grid=grid,
            scale=scale,
            units=units,
            system=system,
            format=format,
            absolute=absolute,
            read=False,
            **extra,
        )
        prop.file_path = str(path) if _is_hdf5_locator(path) else Path(path)
        prop.is_remote = bool(absolute) or str(path).startswith("remote:")
        prop.remote_path = (
            str(path).replace("remote:", "", 1) if prop.is_remote else None
        )
        return prop

    @classmethod
    def expr(
        cls,
        expression: Any,
        *,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        coordinate_system: Optional[str] = None,
        **extra,
    ) -> "Property":
        """Create a property derived from a solver expression.

        Args:
            expression: Expression object, scalar, or serialized expression
                payload accepted by ``PropertyExpression.from_value``.
            units: Optional units associated with the derived value.
            system: Optional coordinate-system name for the derived value.
            coordinate_system: Alias accepted in serialized payloads.
            **extra: Additional payload fields to preserve.

        Returns:
            A ``Property`` containing a ``PropertyExpression`` instead of
            concrete data.
        """

        system = _resolve_system_alias(system, coordinate_system)
        prop = cls(0.0, units=units, system=system, **extra)
        prop.darr = None
        prop.file_path = None
        prop.file_grid = None
        prop.is_remote = False
        prop.remote_path = None
        prop.expression = PropertyExpression.from_value(expression)
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
        if isinstance(value, Mapping):
            payload = dict(value)
            _normalize_system_alias(payload)
            if "expr" in payload:
                payload.pop("depends_on", None)
                return cls.expr(
                    payload.pop("expr"),
                    units=payload.pop("units", None),
                    system=payload.pop("system", None),
                    **payload,
                )
            if "value" in payload and "file" not in payload:
                prop_grid = payload.pop("grid", grid)
                return cls(
                    payload.pop("value"),
                    grid=prop_grid,
                    units=payload.pop("units", None),
                    system=payload.pop("system", None),
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
    ) -> Dict[str, Any]:
        """Serialize the property for a FrequenSolve input payload.

        Args:
            ctx: Optional export context used for project-relative paths and
                HDF5-backed storage.
            file: Optional binary file path used when writing non-constant
                in-memory arrays outside an HDF5 store.
            dataset: Optional HDF5 dataset path used when ``ctx`` provides a
                store.

        Returns:
            Serialized property payload containing one of ``value``, ``file``,
            or ``expr`` plus optional units, system, grid, and extra fields.
        """

        if self.expression is not None:
            payload = {"expr": self.expression.to_fs(ctx)}
            depends_on = self.expression.depends_on()
            if depends_on:
                payload["depends_on"] = depends_on
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
        suffix = path.suffix
        if suffix == ".bin" or suffix == "":
            return Property._bin_reader
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
    def _h5_reader(file: Path, **kwargs) -> xr.DataArray:
        import h5py

        fname, dset = str(file).split(":") if ":" in str(file) else (file, "data")
        with h5py.File(fname, "r") as f:
            dset_obj = f[dset]
            if "dims" in dset_obj.attrs:
                dims = list(dset_obj.attrs["dims"])
                coords = {dim: dset_obj.attrs[dim] for dim in dims}
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
