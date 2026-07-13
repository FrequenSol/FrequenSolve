"""Model-wide material attenuation configuration."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Union

import numpy as np

from frequensolve.units import (
    is_quantity,
    quantity_to_fs,
    unit_expression,
    ureg,
    value_and_units_to_fs,
)
from frequensolve.util.mixins import ExtraFieldsMixin, merge_extra

AttenuationModel = Literal["kjartansson", "none"]

__all__ = ["AttenuationConfig", "AttenuationModel"]


_SUPPORTED_MODELS = frozenset({"kjartansson", "none"})
_REFERENCE_FREQUENCY_KEYS = ("reference_frequency", "f0", "f_ref")


@dataclass(init=False)
class AttenuationConfig(ExtraFieldsMixin):
    """Model-wide Q-attenuation law and reference frequency.

    Args:
        model: Attenuation model name. Names are case-insensitive. Supported
            values are ``"kjartansson"`` (the solver default) and ``"none"``.
        reference_frequency: Positive scalar reference frequency. Bare values
            are interpreted as hertz. Pint quantities and ``{"value": ...,
            "units": ...}`` mappings must have frequency-compatible units.
            When omitted, the solver default is 10 Hz.
        f0: Alias for ``reference_frequency``.
        f_ref: Alias for ``reference_frequency``.
        extra: Additional solver fields preserved on round trip.
        **kwargs: Additional solver fields preserved on round trip.

    Notes:
        ``reference_frequency``, ``f0``, and ``f_ref`` are mutually exclusive.
        FrequenSolve accepts every solver alias but always exports the canonical
        ``reference_frequency`` key.

        The ``"none"`` model makes the solver ignore solid-frame quality
        factors such as ``Qp``, ``Qs``, ``Qk``, and ``Qmu``. It does not disable
        JKD hydraulic dispersion.
    """

    model: AttenuationModel = "kjartansson"
    reference_frequency: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        model: str = "kjartansson",
        reference_frequency: Optional[Any] = None,
        *,
        f0: Optional[Any] = None,
        f_ref: Optional[Any] = None,
        extra: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        provided = [
            ("reference_frequency", reference_frequency),
            ("f0", f0),
            ("f_ref", f_ref),
        ]
        provided = [(name, value) for name, value in provided if value is not None]
        if len(provided) > 1:
            names = ", ".join(name for name, _ in provided)
            raise ValueError(
                "Specify only one of reference_frequency, f0, or f_ref; "
                f"received {names}"
            )

        self.model = _normalize_model(model)
        self.reference_frequency = (
            _normalize_reference_frequency(provided[0][1]) if provided else None
        )
        self._init_extra(extra, **kwargs)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "AttenuationConfig":
        """Deserialize a solver attenuation mapping.

        Args:
            data: Solver attenuation payload.

        Returns:
            Normalized attenuation configuration.

        Raises:
            ValueError: If multiple reference-frequency aliases are present.
        """
        payload = copy.deepcopy(dict(data))
        reference_keys = [key for key in _REFERENCE_FREQUENCY_KEYS if key in payload]
        if len(reference_keys) > 1:
            names = ", ".join(reference_keys)
            raise ValueError(
                "Attenuation accepts only one of reference_frequency, f0, or "
                f"f_ref; received {names}"
            )

        model = payload.pop("model", "kjartansson")
        reference_frequency = payload.pop(reference_keys[0]) if reference_keys else None
        if reference_keys and reference_frequency is None:
            raise TypeError("attenuation reference_frequency must be a positive scalar")
        return cls(
            model=model,
            reference_frequency=reference_frequency,
            extra=payload,
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize the canonical solver attenuation block.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            Solver-ready attenuation mapping.
        """

        model = _normalize_model(self.model)
        payload: Dict[str, Any] = {"model": model}
        if self.reference_frequency is not None:
            reference_frequency = _normalize_reference_frequency(
                self.reference_frequency
            )
            payload["reference_frequency"] = (
                quantity_to_fs(reference_frequency)
                if is_quantity(reference_frequency)
                else copy.deepcopy(reference_frequency)
            )

        reserved_extra = set(self.extra).intersection(_REFERENCE_FREQUENCY_KEYS)
        if reserved_extra:
            names = ", ".join(sorted(reserved_extra))
            raise ValueError(
                "Attenuation reference-frequency aliases must use the typed "
                f"fields, not extra: {names}"
            )
        return merge_extra(payload, self.extra, "AttenuationConfig")


def _normalize_model(model: Any) -> AttenuationModel:
    if not isinstance(model, str):
        raise TypeError(
            f"attenuation model must be a string, got {type(model).__name__}"
        )
    normalized = model.strip().casefold()
    if normalized not in _SUPPORTED_MODELS:
        supported = ", ".join(sorted(_SUPPORTED_MODELS))
        raise ValueError(
            f"Unsupported attenuation model {model!r}; supported models: {supported}"
        )
    return normalized  # type: ignore[return-value]


def _normalize_reference_frequency(value: Any) -> Any:
    if is_quantity(value):
        try:
            converted = value.to("Hz")
        except Exception as exc:
            raise ValueError(
                "attenuation reference_frequency units must be compatible "
                "with frequency"
            ) from exc
        _normalize_positive_scalar(converted.magnitude)
        return value

    if isinstance(value, Mapping):
        payload = copy.deepcopy(dict(value))
        unknown = set(payload) - {"value", "units"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                "attenuation reference_frequency mapping accepts only value "
                f"and units; received {names}"
            )
        if "value" not in payload:
            raise ValueError(
                "attenuation reference_frequency mapping must contain value"
            )
        magnitude = _normalize_positive_scalar(payload["value"])
        if "units" not in payload:
            return {"value": magnitude}
        units = payload["units"]
        try:
            ureg.Quantity(1.0, units).to("Hz")
        except Exception as exc:
            raise ValueError(
                "attenuation reference_frequency units must be compatible "
                "with frequency"
            ) from exc
        return value_and_units_to_fs(magnitude, unit_expression(units))

    return _normalize_positive_scalar(value)


def _normalize_positive_scalar(value: Any) -> Union[int, float]:
    if isinstance(value, (bool, str, bytes)) or np.ndim(value) != 0:
        raise TypeError("attenuation reference_frequency must be a positive scalar")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "attenuation reference_frequency must be a positive scalar"
        ) from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(
            "attenuation reference_frequency must be a finite positive scalar"
        )
    if isinstance(value, np.generic):
        value = value.item()
    return value if isinstance(value, (int, float)) else numeric
