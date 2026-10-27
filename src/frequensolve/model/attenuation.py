"""Model-wide material attenuation configuration."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, Literal, Mapping, Optional, Union, cast

import numpy as np

from frequensolve.units import (
    is_quantity,
    quantity_to_fs,
    unit_expression,
    ureg,
    value_and_units_to_fs,
)
from frequensolve.util.mixins import ExtraFieldsMixin, merge_extra

AttenuationModel = Literal["kjartansson", "gsls", "none"]

__all__ = ["AttenuationConfig", "AttenuationModel"]


_SUPPORTED_MODELS = frozenset({"kjartansson", "gsls", "none"})
_REFERENCE_FREQUENCY_KEYS = ("reference_frequency", "f0", "f_ref")
_GSLS_KEYS = ("mechanisms", "frequency_min", "frequency_max", "fit_tolerance")
_GSLS_MAX_MECHANISMS = 16


@dataclass(init=False)
class AttenuationConfig(ExtraFieldsMixin):
    """Model-wide Q-attenuation law, reference frequency and GSLS fit controls.

    Args:
        model: Attenuation model name. Names are case-insensitive. Supported
            values are ``"kjartansson"`` (the solver default, frequency-domain
            constant-Q), ``"gsls"`` (transient generalized standard linear
            solid relaxation, ``TransientJob`` only) and ``"none"``.
        reference_frequency: Positive scalar reference frequency at which the
            real-valued material parameters (phase velocities) are defined.
            Bare values are interpreted as hertz. Pint quantities and
            ``{"value": ..., "units": ...}`` mappings must have
            frequency-compatible units. When omitted, the solver default is
            10 Hz for ``"kjartansson"`` and the geometric band centre
            ``sqrt(frequency_min * frequency_max)`` for ``"gsls"``.
        mechanisms: GSLS only. Number of relaxation mechanisms, an integer in
            ``1..16``. When omitted, the solver default (3) is used.
        frequency_min: GSLS only, required. Lower edge of the physical fitting
            band, accepted in the same forms as ``reference_frequency``.
        frequency_max: GSLS only, required. Upper edge of the physical fitting
            band; must exceed ``frequency_min``.
        fit_tolerance: GSLS only. Bound on the sampled relative inverse-Q
            fitting error, in ``(0, 1)``. When omitted, the solver default
            (0.15) is used.
        f0: Alias for ``reference_frequency``.
        f_ref: Alias for ``reference_frequency``.
        extra: Additional solver fields preserved on round trip.
        **kwargs: Additional solver fields preserved on round trip.

    Raises:
        ValueError: If a supplied field is out of range, or, for ``"gsls"``,
            if the band is inverted or the reference frequency lies outside
            it. A missing GSLS band is reported by :meth:`from_fs`,
            :meth:`to_fs` and :meth:`validate` rather than the constructor so
            that a model can select ``"gsls"`` before its band is attached.
        TypeError: If a supplied field has the wrong type.

    Notes:
        ``reference_frequency``, ``f0``, and ``f_ref`` are mutually exclusive.
        FrequenSolve accepts every solver alias but always exports the canonical
        ``reference_frequency`` key.

        GSLS band and fit fields are exported only when ``model == "gsls"``;
        they are never written for ``"kjartansson"`` or ``"none"``. For
        ``"gsls"``, GSLS keys found in ``extra`` or ``**kwargs`` are promoted
        to the typed fields. Layer properties keep supplying ``Qp``/``Qs``/
        ``Q``; GSLS reads exactly one of them per material as shared
        ``Qp = Qs``.

        The ``"none"`` model makes the solver ignore solid-frame quality
        factors such as ``Qp``, ``Qs``, ``Qk``, and ``Qmu``. It does not disable
        JKD hydraulic dispersion.
    """

    model: AttenuationModel = "kjartansson"
    reference_frequency: Optional[Any] = None
    mechanisms: Optional[int] = None
    frequency_min: Optional[Any] = None
    frequency_max: Optional[Any] = None
    fit_tolerance: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        model: str = "kjartansson",
        reference_frequency: Optional[Any] = None,
        *,
        mechanisms: Optional[int] = None,
        frequency_min: Optional[Any] = None,
        frequency_max: Optional[Any] = None,
        fit_tolerance: Optional[float] = None,
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

        typed = {
            "mechanisms": mechanisms,
            "frequency_min": frequency_min,
            "frequency_max": frequency_max,
            "fit_tolerance": fit_tolerance,
        }
        if self.model == "gsls":
            for key in _GSLS_KEYS:
                if key not in self.extra:
                    continue
                value = self.extra.pop(key)
                if typed[key] is not None:
                    raise ValueError(
                        f"attenuation {key} was given both as a typed field "
                        "and in extra"
                    )
                if value is None:
                    raise TypeError(f"attenuation {key} must not be null")
                typed[key] = value

        self.mechanisms = (
            _normalize_mechanisms(typed["mechanisms"])
            if typed["mechanisms"] is not None
            else None
        )
        self.frequency_min = (
            _normalize_reference_frequency(typed["frequency_min"], "frequency_min")
            if typed["frequency_min"] is not None
            else None
        )
        self.frequency_max = (
            _normalize_reference_frequency(typed["frequency_max"], "frequency_max")
            if typed["frequency_max"] is not None
            else None
        )
        self.fit_tolerance = (
            _normalize_fit_tolerance(typed["fit_tolerance"])
            if typed["fit_tolerance"] is not None
            else None
        )
        if self.model == "gsls":
            self._check_gsls(require_band=False)

    def validate(self) -> None:
        """Validate the configuration as the solver will read it.

        Raises:
            ValueError: If a ``"gsls"`` configuration lacks its band, has an
                inverted band, or places the reference frequency outside it,
                or if any field is out of range.
            TypeError: If a field has the wrong type.
        """

        model = _normalize_model(self.model)
        if self.reference_frequency is not None:
            _normalize_reference_frequency(self.reference_frequency)
        if model != "gsls":
            return
        if self.mechanisms is not None:
            _normalize_mechanisms(self.mechanisms)
        if self.fit_tolerance is not None:
            _normalize_fit_tolerance(self.fit_tolerance)
        self._check_gsls(require_band=True)

    def _check_gsls(self, *, require_band: bool) -> None:
        missing = [
            name
            for name in ("frequency_min", "frequency_max")
            if getattr(self, name) is None
        ]
        if missing:
            if require_band:
                raise ValueError(
                    "gsls attenuation requires frequency_min and frequency_max "
                    "in Hz; missing " + ", ".join(missing)
                )
            return
        low = _frequency_hz(self.frequency_min, "frequency_min")
        high = _frequency_hz(self.frequency_max, "frequency_max")
        if not low < high:
            raise ValueError(
                "gsls attenuation requires frequency_min < frequency_max; "
                f"received {low:g} Hz and {high:g} Hz"
            )
        if self.reference_frequency is not None:
            reference = _frequency_hz(self.reference_frequency)
            if not low <= reference <= high:
                raise ValueError(
                    "gsls attenuation reference_frequency must lie in the band "
                    f"[{low:g}, {high:g}] Hz; received {reference:g} Hz"
                )

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "AttenuationConfig":
        """Deserialize a solver attenuation mapping.

        Args:
            data: Solver attenuation payload.

        Returns:
            Normalized attenuation configuration.

        Raises:
            ValueError: If multiple reference-frequency aliases are present, or
                a ``"gsls"`` payload is incomplete or inconsistent.
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
        config = cls(
            model=model,
            reference_frequency=reference_frequency,
            extra=payload,
        )
        config.validate()
        return config

    def to_fs(self, ctx: Any = None) -> Dict[str, Any]:
        """Serialize the canonical solver attenuation block.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            Solver-ready attenuation mapping.

        Raises:
            ValueError: If the configuration is invalid (see :meth:`validate`).
        """

        self.validate()
        model = _normalize_model(self.model)
        payload: Dict[str, Any] = {"model": model}
        if self.reference_frequency is not None:
            payload["reference_frequency"] = _frequency_to_fs(
                self.reference_frequency
            )
        if model == "gsls":
            if self.mechanisms is not None:
                payload["mechanisms"] = _normalize_mechanisms(self.mechanisms)
            payload["frequency_min"] = _frequency_to_fs(
                self.frequency_min, "frequency_min"
            )
            payload["frequency_max"] = _frequency_to_fs(
                self.frequency_max, "frequency_max"
            )
            if self.fit_tolerance is not None:
                payload["fit_tolerance"] = _normalize_fit_tolerance(
                    self.fit_tolerance
                )

        reserved_extra = set(self.extra).intersection(_REFERENCE_FREQUENCY_KEYS)
        if reserved_extra:
            names = ", ".join(sorted(reserved_extra))
            raise ValueError(
                "Attenuation reference-frequency aliases must use the typed "
                f"fields, not extra: {names}"
            )
        return merge_extra(payload, self.extra, "AttenuationConfig")

    def fs_extra(self) -> Dict[str, Any]:
        """Return ``extra`` plus the exported GSLS fields as plain solver keys.

        Model containers that store only ``model``/``reference_frequency`` and
        a pass-through mapping can keep this mapping and hand it back as
        ``extra``; the constructor promotes the GSLS keys again for
        ``model == "gsls"``.

        Returns:
            Deep copy of ``extra`` merged with the GSLS band and fit fields.
        """

        payload = copy.deepcopy(dict(self.extra))
        if _normalize_model(self.model) != "gsls":
            return payload
        for key in _GSLS_KEYS:
            value = getattr(self, key)
            if value is None:
                continue
            payload[key] = (
                _frequency_to_fs(value, key)
                if key.startswith("frequency_")
                else copy.deepcopy(value)
            )
        return payload


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
    return cast(AttenuationModel, normalized)


def _normalize_reference_frequency(
    value: Any, name: str = "reference_frequency"
) -> Any:
    if is_quantity(value):
        try:
            converted = value.to("Hz")
        except Exception as exc:
            raise ValueError(
                f"attenuation {name} units must be compatible with frequency"
            ) from exc
        _normalize_positive_scalar(converted.magnitude, name)
        return value

    if isinstance(value, Mapping):
        payload = copy.deepcopy(dict(value))
        unknown = set(payload) - {"value", "units"}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"attenuation {name} mapping accepts only value "
                f"and units; received {names}"
            )
        if "value" not in payload:
            raise ValueError(f"attenuation {name} mapping must contain value")
        magnitude = _normalize_positive_scalar(payload["value"], name)
        if "units" not in payload:
            return {"value": magnitude}
        units = payload["units"]
        try:
            ureg.Quantity(1.0, units).to("Hz")
        except Exception as exc:
            raise ValueError(
                f"attenuation {name} units must be compatible with frequency"
            ) from exc
        return value_and_units_to_fs(magnitude, unit_expression(units))

    return _normalize_positive_scalar(value, name)


def _frequency_to_fs(value: Any, name: str = "reference_frequency") -> Any:
    normalized = _normalize_reference_frequency(value, name)
    return (
        quantity_to_fs(normalized)
        if is_quantity(normalized)
        else copy.deepcopy(normalized)
    )


def _frequency_hz(value: Any, name: str = "reference_frequency") -> float:
    normalized = _normalize_reference_frequency(value, name)
    if is_quantity(normalized):
        return float(normalized.to("Hz").magnitude)
    if isinstance(normalized, Mapping):
        magnitude = float(normalized["value"])
        units = normalized.get("units")
        if units is None:
            return magnitude
        return float(ureg.Quantity(magnitude, units).to("Hz").magnitude)
    return float(normalized)


def _normalize_mechanisms(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            "attenuation mechanisms must be an integer in "
            f"1..{_GSLS_MAX_MECHANISMS}"
        )
    count = int(value)
    if not 1 <= count <= _GSLS_MAX_MECHANISMS:
        raise ValueError(
            f"attenuation mechanisms must lie in 1..{_GSLS_MAX_MECHANISMS}; "
            f"received {count}"
        )
    return count


def _normalize_fit_tolerance(value: Any) -> float:
    if isinstance(value, (bool, str, bytes)) or np.ndim(value) != 0:
        raise TypeError("attenuation fit_tolerance must be a real scalar in (0, 1)")
    try:
        tolerance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            "attenuation fit_tolerance must be a real scalar in (0, 1)"
        ) from exc
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError(
            f"attenuation fit_tolerance must lie in (0, 1); received {value!r}"
        )
    return tolerance


def _normalize_positive_scalar(
    value: Any, name: str = "reference_frequency"
) -> Union[int, float]:
    if isinstance(value, (bool, str, bytes)) or np.ndim(value) != 0:
        raise TypeError(f"attenuation {name} must be a positive scalar")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"attenuation {name} must be a positive scalar") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"attenuation {name} must be a finite positive scalar")
    if isinstance(value, np.generic):
        value = value.item()
    return value if isinstance(value, (int, float)) else numeric
