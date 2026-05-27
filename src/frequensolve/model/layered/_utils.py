from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from frequensolve.model.property import Property
from frequensolve.units import is_quantity, unit_expression, ureg


def _property_units(prop: Property) -> Optional[str]:
    if prop.units:
        return unit_expression(prop.units)
    if prop.data is not None and hasattr(prop.data, "attrs"):
        units = prop.data.attrs.get("units")
        if units:
            return unit_expression(units)
    return None


def _axis_units(units: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    units = dict(units or {})
    out = {"x": None, "y": None, "z": None}
    for key, value in units.items():
        out[key] = unit_expression(value) if value is not None else None
    return out


def _convert_units(
    value: float, source_units: Optional[str], target_units: Optional[str]
):
    if not source_units or not target_units or source_units == target_units:
        return value
    return (value * ureg(source_units)).to(target_units).magnitude


def _coerce_domain_limits(
    limits: Any,
    name: str,
) -> Tuple[List[float], Optional[str]]:
    """Return numeric [min, max] limits and optional length units."""

    units = None
    value = limits
    if isinstance(value, Mapping):
        if "value" not in value:
            raise ValueError(f"LayeredModel {name} mappings require a value")
        units = value.get("units")
        units = unit_expression(units) if units is not None else None
        value = value["value"]

    if is_quantity(value):
        units = unit_expression(value.units)
        value = value.magnitude
    elif isinstance(value, (list, tuple)) and any(is_quantity(item) for item in value):
        quantity_units = units
        if quantity_units is None:
            for item in value:
                if is_quantity(item):
                    quantity_units = unit_expression(item.units)
                    break
        converted = []
        for item in value:
            if is_quantity(item):
                converted.append(float(item.to(quantity_units).magnitude))
            else:
                converted.append(float(item))
        units = quantity_units
        value = converted

    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 2:
        raise ValueError(f"LayeredModel {name} must contain [min, max]")
    out = [float(values[0]), float(values[1])]
    if out[1] <= out[0]:
        raise ValueError(f"LayeredModel {name} must be increasing")
    return out, units


def _convert_surface_value(
    value: float,
    surface: Any,
    target_units: Optional[str],
) -> float:
    return _convert_units(value, _property_units(surface.depth), target_units)


def _convert_dataarray_units(
    data: xr.DataArray,
    source_units: Optional[str],
    target_units: Optional[str],
) -> xr.DataArray:
    if target_units is None:
        if source_units is not None:
            data = data.copy(deep=True)
            data.attrs["units"] = source_units
        return data
    if source_units is None or source_units == target_units:
        data = data.copy(deep=True)
        data.attrs["units"] = target_units
        return data
    converted = data.copy(deep=True)
    converted.data = (converted.data * ureg(source_units)).to(target_units).magnitude
    converted.attrs["units"] = target_units
    return converted


def _convert_surface_depth(
    depth: xr.DataArray,
    surface: Any,
    target_units: Optional[str],
) -> xr.DataArray:
    return _convert_dataarray_units(depth, _property_units(surface.depth), target_units)


def _inline_dataarray_to_fs(data: xr.DataArray) -> Dict[str, Any]:
    """Serialize a small curve/table DataArray directly into the JSON contract."""

    payload: Dict[str, Any] = {
        "dims": list(data.dims),
        "coords": {},
        "data": np.asarray(data.values).tolist(),
    }
    for dim in data.dims:
        coord = data.coords.get(dim)
        if coord is None:
            coord_payload: Dict[str, Any] = {
                "data": np.arange(data.sizes[dim], dtype=float).tolist()
            }
        else:
            coord_payload = {"data": np.asarray(coord.values).tolist()}
            if coord.attrs.get("units"):
                coord_payload["units"] = unit_expression(coord.attrs["units"])
        payload["coords"][dim] = coord_payload
    if data.attrs.get("units"):
        payload["units"] = unit_expression(data.attrs["units"])
    if data.attrs.get("system"):
        payload["system"] = data.attrs["system"]
    elif data.attrs.get("coordinate_system"):
        payload["system"] = data.attrs["coordinate_system"]
    return payload


def _dataarray_with_property_metadata(prop: Property) -> xr.DataArray:
    data = prop.darr.copy(deep=True)
    if prop.units is not None and not data.attrs.get("units"):
        data.attrs["units"] = prop.units
    if prop.system is not None and not (
        data.attrs.get("system") or data.attrs.get("coordinate_system")
    ):
        data.attrs["system"] = prop.system
    return data


def _inline_dataarray_from_fs(data: Mapping[str, Any]) -> xr.DataArray:
    dims = list(data.get("dims", []))
    if not dims:
        raise ValueError("Inline DataArray payload requires dims")
    coords = {}
    raw_coords = data.get("coords", {})
    if not isinstance(raw_coords, Mapping):
        raise ValueError("Inline DataArray coords must be a mapping")
    for dim in dims:
        coord_payload = raw_coords.get(dim)
        if isinstance(coord_payload, Mapping):
            coord_values = coord_payload.get("data", coord_payload.get("values"))
            if coord_values is None:
                raise ValueError(f"Inline DataArray coord {dim!r} requires data")
            coord = xr.DataArray(coord_values, dims=[dim])
            if coord_payload.get("units"):
                coord.attrs["units"] = unit_expression(coord_payload["units"])
            coords[dim] = coord
        elif coord_payload is not None:
            coords[dim] = coord_payload
    out = xr.DataArray(data.get("data"), dims=dims, coords=coords)
    if data.get("units"):
        out.attrs["units"] = unit_expression(data["units"])
    if data.get("system"):
        out.attrs["system"] = data["system"]
    elif data.get("coordinate_system"):
        out.attrs["system"] = data["coordinate_system"]
    return out
