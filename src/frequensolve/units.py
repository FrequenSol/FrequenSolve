"""Unit helpers for FrequenSolve authoring objects.

Pint quantities are accepted at API boundaries and serialized into the solver's
contract form: {"value": ..., "units": "..."}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import numpy as np

from frequensolve.util.mixins import merge_extra

__all__ = [
    "Q_",
    "UnitConfig",
    "is_quantity",
    "quantity_to_fs",
    "unit_expression",
    "ureg",
    "value_and_units_to_fs",
]


def _load_pint():
    try:
        import pint
    except ModuleNotFoundError as exc:
        raise ImportError(
            "FrequenSolve unit support requires `pint`. Reinstall FrequenSolve "
            "with its core dependencies or install `pint`."
        ) from exc
    return pint


class UnitRegistryProxy:
    """Lazy proxy for the shared Pint unit registry.

    The proxy keeps importing FrequenSolve cheap while still exposing the same
    attribute and call surface as ``pint.UnitRegistry`` once unit functionality
    is used.
    """

    _registry = None

    def _load(self):
        if self._registry is None:
            self._registry = _load_pint().UnitRegistry()
        return self._registry

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __call__(self, *args, **kwargs) -> Any:
        """Forward calls to the lazily created Pint unit registry.

        Args:
            *args: Positional arguments forwarded to ``pint.UnitRegistry``.
            **kwargs: Keyword arguments forwarded to ``pint.UnitRegistry``.

        Returns:
            Pint unit or quantity object returned by the registry.
        """

        return self._load()(*args, **kwargs)

    def __repr__(self) -> str:
        if self._registry is None:
            return "<lazy Pint UnitRegistry>"
        return repr(self._registry)


ureg = UnitRegistryProxy()


def Q_(*args, **kwargs):
    """Construct a Pint quantity using FrequenSolve's shared unit registry.

    Args:
        *args: Positional arguments forwarded to ``ureg.Quantity``.
        **kwargs: Keyword arguments forwarded to ``ureg.Quantity``.

    Returns:
        Pint quantity.
    """

    return ureg.Quantity(*args, **kwargs)


def unit_expression(units: Any) -> str:
    """Return a compact solver-friendly unit expression.

    Args:
        units: Unit string, Pint unit, or ``None``.

    Returns:
        Unit expression string. ``None`` becomes an empty string.
    """
    if units is None:
        return ""
    if isinstance(units, str):
        return units
    expr = f"{units:~}"
    expr = expr.replace(" ** ", "^").replace("**", "^")
    expr = expr.replace(" / ", "/").replace(" * ", "*")
    expr = expr.replace(" ", "")
    return expr


def is_quantity(value: Any) -> bool:
    """Return whether ``value`` is a Pint quantity.

    Args:
        value: Object to inspect.

    Returns:
        ``True`` for Pint quantities, otherwise ``False``.
    """

    if value is None:
        return False
    pint = _load_pint()
    return isinstance(value, pint.Quantity)


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    return value


def quantity_to_fs(value: Any) -> Dict[str, Any]:
    """Serialize a Pint quantity as ``{"value": ..., "units": ...}``.

    Args:
        value: Pint quantity to serialize.

    Returns:
        JSON-compatible value/unit mapping.

    Raises:
        TypeError: If ``value`` is not a Pint quantity.
    """

    if not is_quantity(value):
        raise TypeError(f"Expected Pint quantity, got {type(value)}")
    return {
        "value": _plain_value(value.magnitude),
        "units": unit_expression(value.units),
    }


def value_and_units_to_fs(value: Any, units: Optional[Any] = None) -> Any:
    """Serialize a value with optional unit metadata.

    Args:
        value: Scalar, array-like, mapping, xarray data array, or Pint
            quantity.
        units: Optional units to attach when ``value`` does not already carry
            units.

    Returns:
        Raw JSON-compatible value or ``{"value": ..., "units": ...}`` mapping.
    """
    if is_quantity(value):
        return quantity_to_fs(value)

    if isinstance(value, Mapping):
        payload = _plain_value(value)
        if units is not None and "units" not in payload:
            payload["units"] = unit_expression(units)
        return payload

    detected_units = units
    if detected_units is None and hasattr(value, "attrs"):
        detected_units = value.attrs.get("units")

    plain = value.values if _has_dataarray_values(value) else value
    plain = _plain_value(plain)
    if detected_units:
        return {"value": plain, "units": unit_expression(detected_units)}
    return plain


def _has_dataarray_values(value: Any) -> bool:
    try:
        import xarray as xr
    except ImportError:
        return False
    return isinstance(value, xr.DataArray)


@dataclass
class UnitConfig:
    """Simulation-level unit scaling and output-unit defaults.

    Args:
        disable_scaling: Optional flag that disables solver unit scaling.
        f0: Optional reference frequency scale.
        length_scale: Optional length scale.
        time_scale: Optional time scale.
        mass_scale: Optional mass scale.
        defaults: Default units by physical quantity or output name.
        scales: Unit scales by solver scale name.
        units_extra: Additional fields nested inside the serialized ``Units``
            block.
        extra: Additional top-level serialized fields.
    """

    disable_scaling: Optional[bool] = None
    f0: Optional[Any] = None
    length_scale: Optional[Any] = None
    time_scale: Optional[Any] = None
    mass_scale: Optional[Any] = None
    defaults: Dict[str, Any] = field(default_factory=dict)
    scales: Dict[str, Any] = field(default_factory=dict)
    units_extra: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "UnitConfig":
        """Deserialize unit configuration from a solver payload.

        Args:
            data: Serialized simulation or unit configuration mapping.

        Returns:
            ``UnitConfig`` with unit defaults, scales, and extra fields
            restored.
        """

        payload = dict(data)
        units = dict(payload.pop("Units", payload.pop("units", {})) or {})
        defaults = dict(units.pop("defaults", {}))
        scales = dict(units.pop("scales", {}))
        known = {
            key: payload.pop(key, None)
            for key in [
                "disable_scaling",
                "f0",
                "length_scale",
                "time_scale",
                "mass_scale",
            ]
        }
        return cls(
            defaults=defaults, scales=scales, units_extra=units, extra=payload, **known
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize unit configuration for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible unit configuration payload.
        """

        payload: Dict[str, Any] = {}
        units_payload = dict(self.units_extra)
        if self.scales:
            units_payload["scales"] = {
                key: unit_expression(value) for key, value in self.scales.items()
            }
        for key in [
            "disable_scaling",
            "f0",
            "length_scale",
            "time_scale",
            "mass_scale",
        ]:
            value = getattr(self, key)
            if value is not None:
                if is_quantity(value):
                    q = quantity_to_fs(value)
                    payload[key] = q["value"]
                    units_payload.setdefault("scales", {})[key] = q["units"]
                else:
                    payload[key] = _plain_value(value)
        if self.defaults:
            units_payload["defaults"] = {
                key: unit_expression(value) for key, value in self.defaults.items()
            }
        if units_payload:
            payload["Units"] = units_payload
        return merge_extra(payload, self.extra, "UnitConfig")
