"""Misfit definition: losses, comparisons, normalization, projection, hooks.

Every class here maps onto one part of the ``Imaging.misfit`` block of the
``fs-imaging-1`` contract:

* :class:`Loss` -> ``objective_terms[].objective``
* :class:`Comparison` -> ``objective_terms[].comparison``
* :class:`Normalization` -> ``objective_terms[].normalization``
* :class:`Preprocess` -> one ``fs-preprocess-hook-1`` object
* :class:`ReceiverProjection` -> ``receiver_groups[].projection``
* :class:`ObjectiveTerm` -> one ``objective_terms[]`` entry
* :class:`Misfit` -> the whole ``misfit`` payload, given the observed
  receiver groups from :mod:`frequensolve.imaging.data`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import xarray as xr

from frequensolve.imaging.data import ObservedGroup
from frequensolve.units import is_quantity, value_and_units_to_fs
from frequensolve.util.mixins import ExportContext

__all__ = [
    "Comparison",
    "Loss",
    "Misfit",
    "Normalization",
    "ObjectiveTerm",
    "Preprocess",
    "ReceiverProjection",
]

HookStage = Literal[
    "trace_pair",
    "observed",
    "simulated",
    "residual",
    "forward_wavefield",
    "adjoint_wavefield",
]
_HOOK_STAGES = {
    "trace_pair",
    "observed",
    "simulated",
    "residual",
    "forward_wavefield",
    "adjoint_wavefield",
}
_HOOK_SCHEMA = "fs-preprocess-hook-1"


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------


def _finite_float(value: Any, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _positive_float(value: Any, name: str) -> float:
    scalar = _finite_float(value, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _nonnegative_float(value: Any, name: str) -> float:
    scalar = _finite_float(value, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return scalar


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _positive_physical_scalar(value: Any, name: str) -> Any:
    """Validate and serialize one positive scalar with optional units."""

    payload = value_and_units_to_fs(value)
    magnitude = payload.get("value") if isinstance(payload, Mapping) else payload
    values = np.asarray(magnitude)
    if values.ndim != 0 or np.iscomplexobj(values):
        raise ValueError(f"{name} must be a positive real scalar")
    scalar = _finite_float(values, name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return payload


def _nonnegative_physical_scalar(value: Any, name: str) -> Any:
    payload = value_and_units_to_fs(value)
    magnitude = payload.get("value") if isinstance(payload, Mapping) else payload
    values = np.asarray(magnitude)
    if values.ndim != 0 or np.iscomplexobj(values):
        raise ValueError(f"{name} must be a nonnegative real scalar")
    scalar = _finite_float(values, name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return payload


def _physical_magnitude(payload: Any) -> Tuple[float, Optional[str]]:
    if isinstance(payload, Mapping):
        return float(payload["value"]), payload.get("units")
    return float(payload), None


def _finite_fraction(value: Any, name: str, *, allow_zero: bool = False) -> float:
    scalar = _finite_float(value, name)
    lower_ok = scalar >= 0.0 if allow_zero else scalar > 0.0
    if not lower_ok:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return scalar


def _complex_pairs(values: Any, name: str) -> List[List[float]]:
    """Serialize complex values as Sauce's ``[real, imaginary]`` pairs."""

    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 2 and array.shape[-1] == 2 and not np.iscomplexobj(array):
        array = array[:, 0] + 1j * array[:, 1]
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    array = np.asarray(array, dtype=complex)
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} values must be finite")
    return [[float(v.real), float(v.imag)] for v in array]


def _int_ids(values: Any, name: str) -> List[int]:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional list of ids")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if np.any(array < 1):
        raise ValueError(f"{name} ids are one-based and must be positive")
    return [int(v) for v in array]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

_LOSS_ALIASES = {
    "l2": "l2",
    "none": "l2",
    "huber": "huber",
    "hybrid": "huber",
    "student_t": "student_t",
    "studentst": "student_t",
    "student-t": "student_t",
}


@dataclass(frozen=True)
class Loss:
    """Radial loss applied to each metric-weighted residual sample.

    Args:
        kind: ``"l2"``, ``"huber"`` (positive ``delta``), or ``"student_t"``
            (positive ``nu`` and squared scale ``c2``).
        delta: Huber transition amplitude in data-metric coordinates.
        c2: Student-t squared scale in data-metric coordinates.
        nu: Student-t degrees of freedom.
    """

    kind: str = "l2"
    delta: Optional[float] = None
    c2: Optional[float] = None
    nu: Optional[float] = None

    def __post_init__(self) -> None:
        kind = _LOSS_ALIASES.get(str(self.kind).strip().lower())
        if kind is None:
            raise ValueError(f"unsupported loss kind {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        delta = None if self.delta is None else _positive_float(self.delta, "delta")
        c2 = None if self.c2 is None else _positive_float(self.c2, "c2")
        nu = None if self.nu is None else _positive_float(self.nu, "nu")
        if kind == "l2" and any(v is not None for v in (delta, c2, nu)):
            raise ValueError("l2 loss takes no parameters")
        if kind == "huber":
            if c2 is not None or nu is not None:
                raise ValueError("huber loss takes only delta")
            delta = 1.5 if delta is None else delta
        if kind == "student_t":
            if delta is not None:
                raise ValueError("student_t loss takes only nu and c2")
            c2 = 1.0 if c2 is None else c2
            nu = 2.0 if nu is None else nu
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "c2", c2)
        object.__setattr__(self, "nu", nu)

    @classmethod
    def l2(cls) -> "Loss":
        """Return the least-squares loss."""

        return cls(kind="l2")

    @classmethod
    def huber(cls, delta: float = 1.5) -> "Loss":
        """Return a Huber loss with transition amplitude ``delta``."""

        return cls(kind="huber", delta=delta)

    @classmethod
    def student_t(cls, nu: float = 2.0, c2: float = 1.0) -> "Loss":
        """Return a Student-t loss with ``nu`` degrees of freedom and scale ``c2``."""

        return cls(kind="student_t", nu=nu, c2=c2)

    @classmethod
    def from_value(cls, value: Any) -> "Loss":
        """Normalize a loss name, mapping, or :class:`Loss`."""

        if value is None:
            return cls.l2()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(kind=value)
        if isinstance(value, Mapping):
            return cls.from_fs(value)
        raise TypeError("loss must be a Loss, a kind name, or a mapping")

    def to_fs(self) -> Dict[str, Any]:
        """Serialize as ``objective`` in the imaging contract."""

        payload: Dict[str, Any] = {"kind": self.kind}
        if self.kind == "huber":
            payload["delta"] = self.delta
        elif self.kind == "student_t":
            payload["c2"] = self.c2
            payload["nu"] = self.nu
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Loss":
        """Deserialize an ``objective`` mapping."""

        allowed = {"kind", "delta", "c2", "nu"}
        unknown = sorted(set(data).difference(allowed))
        if unknown:
            raise ValueError(f"unsupported objective option(s): {', '.join(unknown)}")
        kind = _LOSS_ALIASES.get(str(data.get("kind", "l2")).strip().lower())
        if kind is None:
            raise ValueError(f"unsupported loss kind {data.get('kind')!r}")
        params: Dict[str, Any] = {}
        if kind == "huber" and "delta" in data:
            params["delta"] = data["delta"]
        if kind == "student_t":
            if "c2" in data:
                params["c2"] = data["c2"]
            if "nu" in data:
                params["nu"] = data["nu"]
        return cls(kind=kind, **params)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """Receiver attribute compared before the loss is applied.

    ``phase_derivative`` compares the stabilized phase slope with respect to
    physical frequency (instantaneous travel time) and requires observed
    ``df`` derivative traces for every receiver group.
    """

    kind: Literal["waveform", "phase_derivative"] = "waveform"
    source_derivative: Literal["frozen", "total"] = "frozen"
    relative_amplitude_floor: float = 0.01

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {"waveform", "phase_derivative"}:
            raise ValueError("comparison kind must be 'waveform' or 'phase_derivative'")
        source_derivative = str(self.source_derivative).strip().lower()
        if source_derivative not in {"frozen", "total"}:
            raise ValueError("source_derivative must be 'frozen' or 'total'")
        floor = _positive_float(
            self.relative_amplitude_floor, "relative_amplitude_floor"
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "source_derivative", source_derivative)
        object.__setattr__(self, "relative_amplitude_floor", floor)

    @classmethod
    def waveform(cls) -> "Comparison":
        """Return the plain waveform comparison."""

        return cls(kind="waveform")

    @classmethod
    def phase_derivative(
        cls,
        *,
        source_derivative: Literal["frozen", "total"] = "frozen",
        relative_amplitude_floor: float = 0.01,
    ) -> "Comparison":
        """Return the instantaneous travel-time (phase-slope) comparison."""

        return cls(
            kind="phase_derivative",
            source_derivative=source_derivative,
            relative_amplitude_floor=relative_amplitude_floor,
        )

    @property
    def requires_derivatives(self) -> Tuple[str, ...]:
        """Return the observed derivative axes this comparison needs."""

        return ("df",) if self.kind == "phase_derivative" else ()

    @classmethod
    def from_value(cls, value: Any) -> "Comparison":
        """Normalize a comparison name, mapping, or :class:`Comparison`."""

        if value is None:
            return cls.waveform()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(kind=value)
        if isinstance(value, Mapping):
            return cls.from_fs(value)
        raise TypeError("comparison must be a Comparison, a kind name, or a mapping")

    def to_fs(self) -> Dict[str, Any]:
        """Serialize as ``comparison`` in the imaging contract."""

        if self.kind == "waveform":
            return {"kind": "waveform"}
        return {
            "kind": self.kind,
            "derivative_axis": "frequency",
            "source_derivative": self.source_derivative,
            "relative_amplitude_floor": self.relative_amplitude_floor,
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Comparison":
        """Deserialize a ``comparison`` mapping."""

        allowed = {
            "kind",
            "derivative_axis",
            "source_derivative",
            "relative_amplitude_floor",
        }
        unknown = sorted(set(data).difference(allowed))
        if unknown:
            raise ValueError(f"unsupported comparison option(s): {', '.join(unknown)}")
        axis = str(data.get("derivative_axis", "frequency")).strip().lower()
        if axis != "frequency":
            raise ValueError(
                "phase-derivative comparison requires derivative_axis='frequency'"
            )
        return cls(
            kind=data.get("kind", "waveform"),
            source_derivative=data.get("source_derivative", "frozen"),
            relative_amplitude_floor=data.get("relative_amplitude_floor", 0.01),
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _scale_value(value: Any, name: str) -> Any:
    """Serialize one positive objective scale (plain or unit-bearing)."""

    return _positive_physical_scalar(value, name)


@dataclass(frozen=True)
class Normalization:
    """Objective-term data scale and reduction.

    Args:
        kind: ``"explicit"`` (one ``value`` or per-component ``components``),
            ``"observed_rms"`` (optional ``minimum`` floor), or
            ``"balance_artifact"`` (``file`` written by ``action=calibrate``).
        value: Explicit scale, plain or with units.
        components: Explicit per-component scales.
        minimum: Floor for the observed RMS scale.
        file: Balance artifact path.
        reduction: ``"weighted_mean"`` (default) or ``"sum"``.
    """

    kind: str = "observed_rms"
    value: Any = None
    components: Optional[Mapping[str, Any]] = None
    minimum: Any = None
    file: Optional[Path] = None
    reduction: Literal["weighted_mean", "sum"] = "weighted_mean"

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {"explicit", "observed_rms", "balance_artifact"}:
            raise ValueError(f"unsupported normalization kind {self.kind!r}")
        reduction = str(self.reduction).strip().lower()
        if reduction not in {"weighted_mean", "sum"}:
            raise ValueError("reduction must be 'weighted_mean' or 'sum'")
        value = None
        components = None
        minimum = None
        file = None
        if kind == "explicit":
            if (self.value is None) == (self.components is None):
                raise ValueError("explicit normalization needs value or components")
            if self.value is not None:
                value = _scale_value(self.value, "normalization value")
            else:
                components = {
                    str(name): _scale_value(scale, f"normalization component {name!r}")
                    for name, scale in dict(self.components or {}).items()
                }
                if not components:
                    raise ValueError("explicit component scales must be non-empty")
        elif kind == "observed_rms":
            if self.value is not None or self.components is not None or self.file:
                raise ValueError("observed_rms normalization takes only minimum")
            if self.minimum is not None:
                minimum = _scale_value(self.minimum, "normalization minimum")
        else:
            if self.file is None or not str(self.file).strip():
                raise ValueError("balance_artifact normalization requires a file")
            if self.value is not None or self.components is not None or self.minimum:
                raise ValueError("balance_artifact normalization takes only file")
            file = Path(str(self.file).strip())
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "file", file)

    @classmethod
    def explicit(
        cls,
        value: Any = None,
        *,
        components: Optional[Mapping[str, Any]] = None,
        reduction: Literal["weighted_mean", "sum"] = "weighted_mean",
    ) -> "Normalization":
        """Return a fixed scale (one value or per-component values)."""

        return cls(
            kind="explicit", value=value, components=components, reduction=reduction
        )

    @classmethod
    def observed_rms(
        cls,
        minimum: Any = None,
        *,
        reduction: Literal["weighted_mean", "sum"] = "weighted_mean",
    ) -> "Normalization":
        """Return the observed-RMS scale with an optional floor."""

        return cls(kind="observed_rms", minimum=minimum, reduction=reduction)

    @classmethod
    def balance_artifact(
        cls,
        file: Union[str, Path],
        *,
        reduction: Literal["weighted_mean", "sum"] = "weighted_mean",
    ) -> "Normalization":
        """Return the scale stored in a calibration balance artifact."""

        return cls(kind="balance_artifact", file=file, reduction=reduction)

    @classmethod
    def from_value(cls, value: Any) -> "Normalization":
        """Normalize a kind name, number, mapping, or :class:`Normalization`."""

        if value is None:
            return cls.observed_rms()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            kind = value.strip().lower()
            if kind == "explicit":
                return cls.explicit(1.0)
            return cls(kind=kind)
        if isinstance(value, Mapping):
            return cls.from_fs(value)
        if is_quantity(value) or isinstance(value, (int, float, np.number)):
            return cls.explicit(value)
        raise TypeError(
            "normalization must be a Normalization, a kind name, a scale, or a mapping"
        )

    def to_fs(self) -> Dict[str, Any]:
        """Serialize as ``normalization`` in the imaging contract."""

        scale: Dict[str, Any] = {"kind": self.kind}
        if self.kind == "explicit":
            if self.value is not None:
                scale["value"] = self.value
            else:
                scale["components"] = dict(self.components or {})
        elif self.kind == "observed_rms":
            if self.minimum is not None:
                scale["minimum"] = self.minimum
        else:
            scale["file"] = str(self.file)
        return {"scale": scale, "reduction": self.reduction}

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Normalization":
        """Deserialize a ``normalization`` mapping."""

        scale = dict(data["scale"])
        kind = scale.pop("kind")
        return cls(
            kind=kind,
            value=scale.get("value"),
            components=scale.get("components"),
            minimum=scale.get("minimum"),
            file=scale.get("file"),
            reduction=data.get("reduction", "weighted_mean"),
        )


# ---------------------------------------------------------------------------
# Receiver projection
# ---------------------------------------------------------------------------


def _impedance(value: Any, name: str) -> Any:
    if value is None:
        return None
    if is_quantity(value):
        raise TypeError(
            f"{name} must be a plain number, one value per receiver, or an "
            "HDF5Dense mapping in aligned traction/velocity units"
        )
    if isinstance(value, Mapping):
        if (
            value.get("_type") != "HDF5Dense"
            or not value.get("file")
            or not value.get("dataset")
        ):
            raise ValueError(
                f"{name} mapping must be an HDF5Dense file/dataset reference"
            )
        return {
            "_type": "HDF5Dense",
            "file": str(value["file"]),
            "dataset": str(value["dataset"]),
        }
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return _positive_float(array, name)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a scalar or one value per receiver")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"{name} values must be finite and positive")
    return [float(v) for v in array]


def _component_names(value: Any, name: str) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    names = [str(v).strip() for v in value]
    if not names or len(names) > 2 or any(not v for v in names):
        raise ValueError(f"{name} must list one or two component names")
    return names


@dataclass(frozen=True)
class ReceiverProjection:
    """Receiver-space projection applied before comparison.

    ``identity`` compares raw channels. ``up_down`` reconstructs the upgoing
    characteristic from pressure and normal velocity (acoustic) or from
    velocity and vertical traction (elastic, ``shear_impedance`` required)
    and pulls the exact transpose back into adjoint receiver loads.
    """

    kind: Literal["identity", "up_down", "acoustic_upgoing"] = "identity"
    impedance: Any = None
    shear_impedance: Any = None
    physics: Optional[Literal["acoustic", "elastic"]] = None
    pressure_component: Optional[str] = None
    normal_velocity_component: Optional[str] = None
    normal_traction_component: Optional[str] = None
    tangential_velocity_components: Optional[Sequence[str]] = None
    tangential_traction_components: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if kind not in {"identity", "up_down", "acoustic_upgoing"}:
            raise ValueError(f"unsupported receiver projection {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        physics = None if self.physics is None else str(self.physics).strip().lower()
        if physics is not None and physics not in {"acoustic", "elastic"}:
            raise ValueError("projection physics must be 'acoustic' or 'elastic'")
        if kind == "identity":
            if any(
                v is not None
                for v in (
                    self.impedance,
                    self.shear_impedance,
                    physics,
                    self.pressure_component,
                    self.normal_velocity_component,
                    self.normal_traction_component,
                    self.tangential_velocity_components,
                    self.tangential_traction_components,
                )
            ):
                raise ValueError("identity projection takes no parameters")
            return
        if self.impedance is None:
            raise ValueError(f"{kind} projection requires impedance")
        if kind == "acoustic_upgoing" and physics == "elastic":
            raise ValueError("acoustic_upgoing is an acoustic-only alias")
        if physics == "elastic" and self.shear_impedance is None:
            raise ValueError("elastic up_down projection requires shear_impedance")
        object.__setattr__(self, "physics", physics)
        object.__setattr__(self, "impedance", _impedance(self.impedance, "impedance"))
        object.__setattr__(
            self, "shear_impedance", _impedance(self.shear_impedance, "shear_impedance")
        )
        for attr in (
            "pressure_component",
            "normal_velocity_component",
            "normal_traction_component",
        ):
            value = getattr(self, attr)
            if value is not None:
                text = str(value).strip()
                if not text:
                    raise ValueError(f"{attr} must be non-empty")
                object.__setattr__(self, attr, text)
        object.__setattr__(
            self,
            "tangential_velocity_components",
            _component_names(
                self.tangential_velocity_components, "tangential_velocity_components"
            ),
        )
        object.__setattr__(
            self,
            "tangential_traction_components",
            _component_names(
                self.tangential_traction_components, "tangential_traction_components"
            ),
        )

    @classmethod
    def identity(cls) -> "ReceiverProjection":
        """Return the identity projection."""

        return cls(kind="identity")

    @classmethod
    def up_down(cls, impedance: Any, **options: Any) -> "ReceiverProjection":
        """Return the up/down characteristic projection.

        Args:
            impedance: Acoustic or elastic P impedance (scalar, one value per
                receiver, or an ``HDF5Dense`` mapping).
            **options: ``shear_impedance``, ``physics``, and component names.
        """

        return cls(kind="up_down", impedance=impedance, **options)

    @classmethod
    def acoustic_upgoing(cls, impedance: Any, **options: Any) -> "ReceiverProjection":
        """Return the deprecated acoustic-only alias of :meth:`up_down`."""

        return cls(kind="acoustic_upgoing", impedance=impedance, **options)

    @classmethod
    def from_value(cls, value: Any) -> "ReceiverProjection":
        """Normalize a kind name, mapping, or :class:`ReceiverProjection`."""

        if value is None:
            return cls.identity()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(kind=value)
        if isinstance(value, Mapping):
            return cls.from_fs(value)
        raise TypeError("projection must be a ReceiverProjection, a kind, or a mapping")

    def to_fs(self) -> Dict[str, Any]:
        """Serialize as ``projection`` in the imaging contract."""

        if self.kind == "identity":
            return {"kind": "identity"}
        payload: Dict[str, Any] = {"kind": self.kind}
        for attr in (
            "physics",
            "pressure_component",
            "normal_velocity_component",
            "normal_traction_component",
            "tangential_velocity_components",
            "tangential_traction_components",
        ):
            value = getattr(self, attr)
            if value is not None:
                payload[attr] = list(value) if isinstance(value, list) else value
        payload["impedance"] = self.impedance
        if self.shear_impedance is not None:
            payload["shear_impedance"] = self.shear_impedance
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ReceiverProjection":
        """Deserialize a ``projection`` mapping."""

        payload = dict(data)
        if "vertical_velocity_component" in payload:
            payload.setdefault(
                "normal_velocity_component", payload.pop("vertical_velocity_component")
            )
        allowed = {
            "kind",
            "impedance",
            "shear_impedance",
            "physics",
            "pressure_component",
            "normal_velocity_component",
            "normal_traction_component",
            "tangential_velocity_components",
            "tangential_traction_components",
        }
        unknown = sorted(set(payload).difference(allowed))
        if unknown:
            raise ValueError(f"unsupported projection option(s): {', '.join(unknown)}")
        payload.setdefault("kind", "identity")
        return cls(**payload)


# ---------------------------------------------------------------------------
# Preprocessing hooks
# ---------------------------------------------------------------------------


def _taper_distance(value: Any, name: str, scale: str, units: Any) -> Any:
    if scale == "domain_fraction":
        if is_quantity(value) or isinstance(value, Mapping):
            raise ValueError("domain_fraction taper distances do not accept units")
        return _nonnegative_float(value, name)
    payload = _nonnegative_physical_scalar(value, name)
    if units is not None and not isinstance(payload, Mapping):
        payload = value_and_units_to_fs(payload, units)
    return payload


def _stage(stage: str) -> str:
    text = str(stage).strip().lower()
    if text not in _HOOK_STAGES:
        raise ValueError(f"unsupported preprocessing stage {stage!r}")
    return text


@dataclass(frozen=True)
class Preprocess:
    """One ``fs-preprocess-hook-1`` object.

    Use the classmethods, one per Sauce built-in hook kind, to author hooks
    with validated parameters. Stage defaults follow the roles documented in
    the imaging contract: objective weights (``offset_power``,
    ``offset_taper``, ``component_scale``, ``frequency_weight``,
    ``trace_mask``, ``trace_weight``) default to ``residual``; fixed linear
    receiver operators and ``source_scalar_fit`` default to ``trace_pair``;
    ``source_spectrum_correction`` is a ``simulated`` hook; data transforms
    (``trace_normalize``, ``amplitude_clip``) default to ``observed``.
    """

    kind: str
    stage: str
    params: Mapping[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    schema: str = _HOOK_SCHEMA
    _weight_values: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )

    _MAX_INLINE_TRACE_WEIGHTS: ClassVar[int] = 256

    def __post_init__(self) -> None:
        kind = str(self.kind).strip()
        if not kind:
            raise ValueError("preprocessing hook kind must be non-empty")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "stage", _stage(self.stage))
        object.__setattr__(self, "params", dict(self.params or {}))
        if self.name is not None:
            name = str(self.name).strip()
            if not name:
                raise ValueError("preprocessing hook name must be non-empty")
            object.__setattr__(self, "name", name)

    # -- objective weights ---------------------------------------------------

    @classmethod
    def offset_power(
        cls,
        power: Optional[float] = None,
        *,
        normalize: bool = False,
        stage: HookStage = "residual",
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Weight traces by source-receiver offset raised to ``power``.

        ``power`` defaults to Sauce's dimension-dependent value (0.5 in 2D,
        1 in 3D) when omitted.
        """

        params: Dict[str, Any] = {"normalize": bool(normalize)}
        if power is not None:
            params["power"] = _finite_float(power, "power")
        return cls(kind="offset_power", stage=stage, name=name, params=params)

    @classmethod
    def offset_taper(
        cls,
        d0: Any,
        d1: Any,
        *,
        mode: Literal["near", "far"] = "near",
        scale: Literal["absolute", "domain_fraction"] = "absolute",
        units: Any = None,
        max_domain_fraction: Optional[float] = None,
        stage: HookStage = "residual",
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Raised-cosine offset taper between ``d0`` and ``d1``.

        With ``scale="absolute"`` the distances may carry units (Pint
        quantities or ``units=``); with ``scale="domain_fraction"`` they are
        fractions of the largest domain extent.
        """

        mode_text = str(mode).strip().lower()
        if mode_text not in {"near", "far"}:
            raise ValueError("offset_taper mode must be 'near' or 'far'")
        scale_text = str(scale).strip().lower()
        if scale_text not in {"absolute", "domain_fraction"}:
            raise ValueError(
                "offset_taper scale must be 'absolute' or 'domain_fraction'"
            )
        params: Dict[str, Any] = {
            "scale": scale_text,
            "mode": mode_text,
            "d0": _taper_distance(d0, "d0", scale_text, units),
            "d1": _taper_distance(d1, "d1", scale_text, units),
        }
        d0_value, d0_units = _physical_magnitude(params["d0"])
        d1_value, d1_units = _physical_magnitude(params["d1"])
        if d0_units == d1_units and d1_value < d0_value:
            raise ValueError("offset_taper d1 must not be smaller than d0")
        if max_domain_fraction is not None:
            params["max_domain_fraction"] = _nonnegative_float(
                max_domain_fraction, "max_domain_fraction"
            )
        return cls(kind="offset_taper", stage=stage, name=name, params=params)

    @classmethod
    def component_scale(
        cls,
        scale: Any,
        *,
        stage: HookStage = "residual",
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Scale each receiver component by one (possibly complex) factor."""

        return cls(
            kind="component_scale",
            stage=stage,
            name=name,
            params={"scale": _complex_pairs(scale, "component scale")},
        )

    @classmethod
    def frequency_weight(
        cls,
        power: float = 0.0,
        *,
        f0: float = 1.0,
        amplitude: float = 1.0,
        epsilon: Optional[float] = None,
        stage: HookStage = "residual",
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Deterministic frequency weight ``amplitude * (f / f0) ** power``."""

        params: Dict[str, Any] = {
            "power": _finite_float(power, "power"),
            "f0": _finite_float(f0, "f0"),
            "amplitude": _finite_float(amplitude, "amplitude"),
        }
        if epsilon is not None:
            params["epsilon"] = _positive_float(epsilon, "epsilon")
        return cls(kind="frequency_weight", stage=stage, name=name, params=params)

    @classmethod
    def trace_mask(
        cls,
        *,
        source_ids: Optional[Sequence[int]] = None,
        receiver_ids: Optional[Sequence[int]] = None,
        trace_ids: Optional[Sequence[int]] = None,
        components: Optional[Sequence[int]] = None,
        invalid: bool = False,
        stage: HookStage = "residual",
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Zero traces by one-based source, receiver, trace, or component id.

        ``invalid=True`` additionally masks non-finite observed samples and is
        restricted to the ``observed`` stage by Sauce.
        """

        params: Dict[str, Any] = {}
        for key, value in (
            ("source_ids", source_ids),
            ("receiver_ids", receiver_ids),
            ("trace_ids", trace_ids),
            ("components", components),
        ):
            if value is not None:
                params[key] = _int_ids(value, key)
        if invalid:
            params["invalid"] = True
        if not params:
            raise ValueError("trace_mask requires at least one id list or invalid=True")
        if invalid and _stage(stage) != "observed":
            raise ValueError("trace_mask invalid=True requires the observed stage")
        return cls(kind="trace_mask", stage=stage, name=name, params=params)

    @classmethod
    def trace_weight(
        cls,
        weights: Any,
        *,
        layout: Optional[
            Literal[
                "receiver",
                "component_receiver",
                "source_component_receiver",
                "sparse_trace",
            ]
        ] = None,
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Residual-stage per-trace objective weights.

        Multidimensional arrays infer layouts ``receiver`` (1-D),
        ``component_receiver`` (2-D), or ``source_component_receiver`` (3-D)
        with receiver varying fastest. Sparse trace-catalog weights require
        ``layout="sparse_trace"`` explicitly. Weights above 256 entries are
        materialized into the export context's HDF5 store.
        """

        values = np.asarray(weights)
        if values.ndim < 1 or values.ndim > 3 or values.size == 0:
            raise ValueError("trace weights must be a nonempty 1-D, 2-D, or 3-D array")
        if np.iscomplexobj(values) and np.any(np.imag(values) != 0):
            raise ValueError("residual trace weights must be real")
        try:
            real_values = np.asarray(np.real(values), dtype=float)
        except (TypeError, ValueError) as exc:
            raise TypeError("residual trace weights must be numeric") from exc
        if not np.all(np.isfinite(real_values)) or np.any(real_values < 0):
            raise ValueError("residual trace weights must be finite and nonnegative")

        inferred = {
            1: "receiver",
            2: "component_receiver",
            3: "source_component_receiver",
        }
        selected = inferred[real_values.ndim] if layout is None else layout
        expected_ndim = {
            "receiver": 1,
            "component_receiver": 2,
            "source_component_receiver": 3,
            "sparse_trace": 1,
        }
        if selected not in expected_ndim:
            raise ValueError(f"unsupported trace-weight layout {selected!r}")
        if real_values.ndim != expected_ndim[selected]:
            raise ValueError(
                f"trace-weight layout {selected!r} requires a "
                f"{expected_ndim[selected]}-D array"
            )
        return cls(
            kind="trace_weight",
            stage="residual",
            name=name,
            params={"layout": selected},
            _weight_values=np.array(real_values, dtype=np.float64, copy=True),
        )

    # -- data transforms -----------------------------------------------------

    @classmethod
    def trace_normalize(
        cls,
        scope: Literal["source", "component", "receiver_group"] = "source",
        *,
        epsilon: float = 1.0e-12,
        stage: HookStage = "observed",
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Normalize trace amplitudes per source, component, or receiver group."""

        scope_text = str(scope).strip().lower()
        if scope_text in {"group", "all"}:
            scope_text = "receiver_group"
        if scope_text not in {"source", "component", "receiver_group"}:
            raise ValueError(f"unsupported trace_normalize scope {scope!r}")
        return cls(
            kind="trace_normalize",
            stage=stage,
            name=name,
            params={
                "scope": scope_text,
                "epsilon": _positive_float(epsilon, "epsilon"),
            },
        )

    @classmethod
    def amplitude_clip(
        cls,
        threshold: float,
        *,
        clip: Literal["hard", "soft"] = "hard",
        stage: HookStage = "observed",
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Clip trace amplitudes above ``threshold`` (hard or soft)."""

        mode = str(clip).strip().lower()
        if mode not in {"hard", "soft"}:
            raise ValueError("amplitude_clip clip must be 'hard' or 'soft'")
        return cls(
            kind="amplitude_clip",
            stage=stage,
            name=name,
            params={"threshold": _positive_float(threshold, "threshold"), "clip": mode},
        )

    # -- source hooks --------------------------------------------------------

    @classmethod
    def source_scalar_fit(
        cls,
        *,
        norm: str = "inherit",
        max_iterations: int = 20,
        relative_tolerance: float = 1.0e-8,
        delta: Optional[float] = None,
        c2: Optional[float] = None,
        nu: Optional[float] = None,
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Variable-projection complex source scalar per RHS (``trace_pair``).

        ``norm="inherit"`` reuses the enclosing objective's loss; an explicit
        norm must be equivalent to it.
        """

        norm_text = str(norm).strip().lower()
        if norm_text != "inherit":
            norm_text = Loss(kind=norm_text).kind
        params: Dict[str, Any] = {
            "norm": norm_text,
            "max_iterations": _positive_int(max_iterations, "max_iterations"),
            "relative_tolerance": _positive_float(
                relative_tolerance, "relative_tolerance"
            ),
        }
        for key, value in (("delta", delta), ("c2", c2), ("nu", nu)):
            if value is not None:
                params[key] = _positive_float(value, key)
        return cls(
            kind="source_scalar_fit", stage="trace_pair", name=name, params=params
        )

    @classmethod
    def source_spectrum_correction(
        cls,
        scale: Any,
        frequency_derivative: Any,
        *,
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Externally estimated source correction ``q`` and its Hz derivative.

        ``scale`` and ``frequency_derivative`` hold one broadcast value or one
        value per encoded RHS; complex values become ``[real, imaginary]``.
        """

        scale_pairs = _complex_pairs(scale, "scale")
        derivative_pairs = _complex_pairs(frequency_derivative, "frequency_derivative")
        if len(scale_pairs) != len(derivative_pairs):
            raise ValueError(
                "source_spectrum_correction frequency_derivative must match scale"
            )
        return cls(
            kind="source_spectrum_correction",
            stage="simulated",
            name=name,
            params={"scale": scale_pairs, "frequency_derivative": derivative_pairs},
        )

    # -- fixed linear receiver operators ------------------------------------

    @classmethod
    def receiver_ar1_whiten(
        cls, correlation: float, *, name: Optional[str] = None
    ) -> "Preprocess":
        """Exact stationary AR(1) whitening along a dense receiver axis."""

        rho = _finite_float(correlation, "receiver AR(1) correlation")
        if abs(rho) >= 1.0:
            raise ValueError(
                "receiver AR(1) correlation must be finite with absolute value below one"
            )
        return cls(
            kind="receiver_ar1_whiten",
            stage="trace_pair",
            name=name,
            params={"correlation": rho},
        )

    @classmethod
    def scholte_notch(
        cls,
        phase_velocity: Optional[Any] = None,
        *,
        wavenumber: Optional[Any] = None,
        relative_half_width: float = 0.04,
        relative_taper_width: float = 0.04,
        spacing_tolerance: float = 1.0e-3,
        spectral_derivative: Literal["total", "frozen"] = "total",
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Smooth receiver-wavenumber notch around a Scholte ridge.

        Supply exactly one of ``phase_velocity`` (ridge at ``2*pi*f/v``) or
        an angular ``wavenumber``; both may carry units.
        """

        if (phase_velocity is None) == (wavenumber is None):
            raise ValueError(
                "scholte_notch requires exactly one of phase_velocity or wavenumber"
            )
        params: Dict[str, Any] = {
            "relative_half_width": _finite_fraction(
                relative_half_width, "relative_half_width", allow_zero=True
            ),
            "relative_taper_width": _finite_fraction(
                relative_taper_width, "relative_taper_width"
            ),
            "spacing_tolerance": _finite_fraction(
                spacing_tolerance, "spacing_tolerance"
            ),
            "spectral_derivative": _spectral_derivative(spectral_derivative),
        }
        if params["spacing_tolerance"] >= 1.0:
            raise ValueError("spacing_tolerance must be less than one")
        if phase_velocity is not None:
            params["phase_velocity"] = _positive_physical_scalar(
                phase_velocity, "phase_velocity"
            )
        else:
            params["wavenumber"] = _positive_physical_scalar(wavenumber, "wavenumber")
        return cls(kind="scholte_notch", stage="trace_pair", name=name, params=params)

    @classmethod
    def slow_velocity_mute(
        cls,
        stop_velocity: Any,
        pass_velocity: Any,
        *,
        mode: Literal["reject_slow", "keep_slow"] = "reject_slow",
        spacing_tolerance: float = 1.0e-3,
        spectral_derivative: Literal["total", "frozen"] = "total",
        name: Optional[str] = None,
    ) -> "Preprocess":
        """Smooth apparent-velocity fan mute along a dense cable."""

        if mode not in {"reject_slow", "keep_slow"}:
            raise ValueError(f"unsupported slow-velocity mute mode {mode!r}")
        stop = _positive_physical_scalar(stop_velocity, "stop_velocity")
        passed = _positive_physical_scalar(pass_velocity, "pass_velocity")
        stop_value, stop_units = _physical_magnitude(stop)
        pass_value, pass_units = _physical_magnitude(passed)
        if stop_units == pass_units and pass_value <= stop_value:
            raise ValueError("pass_velocity must exceed stop_velocity")
        tolerance = _finite_fraction(spacing_tolerance, "spacing_tolerance")
        if tolerance >= 1.0:
            raise ValueError("spacing_tolerance must be less than one")
        return cls(
            kind="slow_velocity_mute",
            stage="trace_pair",
            name=name,
            params={
                "stop_velocity": stop,
                "pass_velocity": passed,
                "mode": mode,
                "spacing_tolerance": tolerance,
                "spectral_derivative": _spectral_derivative(spectral_derivative),
            },
        )

    # -- serialization -------------------------------------------------------

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Serialize this hook, materializing authored trace weights when possible.

        Args:
            ctx: Export context; its HDF5 store receives large trace weights.
            dataset: Store dataset path for materialized trace weights.
        """

        params = dict(self.params)
        if self.kind == "trace_weight" and self._weight_values is not None:
            store = getattr(ctx, "store", None) if ctx is not None else None
            if store is None:
                if self._weight_values.size > self._MAX_INLINE_TRACE_WEIGHTS:
                    raise ValueError(
                        "large trace weights require an export context with an HDF5 store"
                    )
                params["weights"] = self._weight_values.reshape(-1, order="C").tolist()
            else:
                if dataset is None:
                    raise ValueError(
                        "trace-weight HDF5 serialization requires a dataset path"
                    )
                dims = {
                    1: ("receiver",),
                    2: ("component", "receiver"),
                    3: ("source", "component", "receiver"),
                }[self._weight_values.ndim]
                if params.get("layout") == "sparse_trace":
                    dims = ("trace",)
                ref = store.put_dataarray(
                    dataset,
                    xr.DataArray(self._weight_values, dims=dims),
                    attrs={"fs_kind": "residual_trace_weights"},
                    coordinate_dims=(),
                    dtype=np.float64,
                )
                params["weights"] = {"_type": "HDF5Dense", **ref.to_fs(format="HDF5")}
        return {
            "schema": self.schema,
            **({"name": self.name} if self.name is not None else {}),
            "kind": self.kind,
            "stage": self.stage,
            "params": params,
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Preprocess":
        """Deserialize a preprocessing hook payload."""

        return cls(
            schema=data.get("schema", _HOOK_SCHEMA),
            name=data.get("name"),
            kind=data["kind"],
            stage=data["stage"],
            params=dict(data.get("params", {})),
        )

    @classmethod
    def from_value(cls, value: Any) -> "Preprocess":
        """Normalize a :class:`Preprocess` or hook mapping."""

        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls.from_fs(value)
        raise TypeError("preprocess hooks must be Preprocess objects or hook mappings")


def _spectral_derivative(value: str) -> str:
    text = str(value).strip().lower()
    if text not in {"total", "frozen"}:
        raise ValueError("spectral_derivative must be 'total' or 'frozen'")
    return text


def _hooks(value: Any) -> Tuple[Preprocess, ...]:
    if value is None:
        return ()
    if isinstance(value, (Preprocess, Mapping)):
        return (Preprocess.from_value(value),)
    return tuple(Preprocess.from_value(item) for item in value)


def _hooks_to_fs(
    hooks: Sequence[Preprocess], ctx: Optional[ExportContext], *, scope: str
) -> List[Dict[str, Any]]:
    return [
        hook.to_fs(ctx, dataset=f"inputs/imaging/trace_weights/{scope}/{index}")
        for index, hook in enumerate(hooks)
    ]


def _hooks_from_fs(value: Any) -> Tuple[Tuple[Preprocess, ...], bool]:
    """Return ``(hooks, include_defaults)`` from a ``preprocess`` payload."""

    if value is None:
        return (), False
    if isinstance(value, Mapping):
        include = bool(value.get("include_defaults", value.get("default", False)))
        if "kind" in value:
            return (Preprocess.from_fs(value),), include
        return tuple(Preprocess.from_fs(h) for h in value.get("hooks", [])), include
    if isinstance(value, str):
        if value.strip().lower() != "default":
            raise ValueError(f"unsupported preprocess option {value!r}")
        return (), True
    return tuple(Preprocess.from_fs(h) for h in value), False


# ---------------------------------------------------------------------------
# Objective terms and misfit
# ---------------------------------------------------------------------------

_TERM_ID_PATTERN = "^[A-Za-z][A-Za-z0-9_.-]*$"


def _term_id(value: str) -> str:
    text = str(value).strip()
    if not re.match(_TERM_ID_PATTERN, text):
        raise ValueError(
            f"objective term id {value!r} must start with a letter and contain only "
            "letters, digits, '_', '.', or '-'"
        )
    return text


@dataclass(frozen=True)
class ObjectiveTerm:
    """One objective term over a receiver group.

    Args:
        receiver_group: Receiver group compared by this term.
        loss: :class:`Loss` or kind name.
        comparison: :class:`Comparison` or kind name.
        weight: Positive dimensionless tradeoff weight.
        normalization: :class:`Normalization`, kind name, scale, or mapping.
        preprocess: Term-local preprocessing hooks.
        id: Unique term id; defaults to the receiver group name.
    """

    receiver_group: str
    loss: Any = "l2"
    comparison: Any = "waveform"
    weight: float = 1.0
    normalization: Any = None
    preprocess: Sequence[Any] = ()
    id: Optional[str] = None

    def __post_init__(self) -> None:
        group = str(self.receiver_group).strip()
        if not group:
            raise ValueError("objective term receiver_group must be non-empty")
        object.__setattr__(self, "receiver_group", group)
        object.__setattr__(self, "loss", Loss.from_value(self.loss))
        object.__setattr__(self, "comparison", Comparison.from_value(self.comparison))
        object.__setattr__(self, "weight", _positive_float(self.weight, "weight"))
        object.__setattr__(
            self, "normalization", Normalization.from_value(self.normalization)
        )
        object.__setattr__(self, "preprocess", _hooks(self.preprocess))
        object.__setattr__(self, "id", _term_id(group if self.id is None else self.id))

    def to_fs(
        self, ctx: Optional[ExportContext] = None, *, scope: Optional[str] = None
    ) -> Dict[str, Any]:
        """Serialize as one ``objective_terms[]`` entry."""

        payload: Dict[str, Any] = {
            "id": self.id,
            "receiver_group": self.receiver_group,
            "objective": self.loss.to_fs(),
            "comparison": self.comparison.to_fs(),
            "weight": self.weight,
            "normalization": self.normalization.to_fs(),
        }
        if self.preprocess:
            payload["preprocess"] = _hooks_to_fs(
                self.preprocess, ctx, scope=scope or f"objective_terms/{self.id}"
            )
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ObjectiveTerm":
        """Deserialize one ``objective_terms[]`` entry."""

        hooks, include_defaults = _hooks_from_fs(data.get("preprocess"))
        if include_defaults:
            raise ValueError("objective term preprocess cannot include legacy defaults")
        return cls(
            receiver_group=data["receiver_group"],
            loss=data.get("objective"),
            comparison=data.get("comparison"),
            weight=data.get("weight", 1.0),
            normalization=data.get("normalization"),
            preprocess=hooks,
            id=data["id"],
        )


class Misfit:
    """Data misfit: objective terms plus receiver-group and hook settings.

    Build one with the constructors :meth:`l2`, :meth:`huber`,
    :meth:`student_t`, or :meth:`terms`, or directly:

    >>> Misfit(loss="huber", comparison="phase_derivative", normalization="observed_rms")

    Args:
        loss: Loss shared by every generated term.
        comparison: Comparison shared by every generated term.
        normalization: Normalization shared by every generated term.
        weights: Term weight, either one positive number or a mapping from
            receiver group to weight.
        preprocess: Misfit-level preprocessing hooks for every receiver group.
        projection: Receiver projection for every receiver group, or a
            mapping from receiver group to projection.
        terms: Explicit :class:`ObjectiveTerm` list; when given, ``loss``,
            ``comparison``, ``normalization`` and ``weights`` must be omitted.
        group_preprocess: Optional receiver-group-local hooks keyed by group.
        include_default_preprocess: Keep Sauce's legacy default hooks in
            addition to ``preprocess``.
    """

    def __init__(
        self,
        loss: Any = None,
        *,
        comparison: Any = None,
        normalization: Any = None,
        weights: Any = None,
        preprocess: Any = None,
        projection: Any = None,
        terms: Optional[Sequence[ObjectiveTerm]] = None,
        group_preprocess: Optional[Mapping[str, Any]] = None,
        include_default_preprocess: bool = False,
    ) -> None:
        if terms is not None:
            if any(v is not None for v in (loss, comparison, normalization, weights)):
                raise ValueError(
                    "Misfit.terms cannot be combined with loss, comparison, "
                    "normalization, or weights"
                )
            explicit = tuple(terms)
            if not explicit:
                raise ValueError("Misfit requires at least one objective term")
            if not all(isinstance(term, ObjectiveTerm) for term in explicit):
                raise TypeError("terms must be ObjectiveTerm objects")
            ids = [term.id for term in explicit]
            if len(set(ids)) != len(ids):
                raise ValueError("objective term ids must be unique")
            self._terms: Optional[Tuple[ObjectiveTerm, ...]] = explicit
            self.loss = None
            self.comparison = None
            self.normalization = None
            self.weights = None
        else:
            self._terms = None
            self.loss = Loss.from_value(loss)
            self.comparison = Comparison.from_value(comparison)
            self.normalization = Normalization.from_value(normalization)
            self.weights = self._normalize_weights(weights)
        self.preprocess = _hooks(preprocess)
        self.include_default_preprocess = bool(include_default_preprocess)
        if isinstance(projection, Mapping) and "kind" not in projection:
            self.projection: Any = {
                str(name): ReceiverProjection.from_value(value)
                for name, value in projection.items()
            }
        else:
            self.projection = ReceiverProjection.from_value(projection)
        self.group_preprocess = {
            str(name): _hooks(hooks)
            for name, hooks in dict(group_preprocess or {}).items()
        }

    @staticmethod
    def _normalize_weights(weights: Any) -> Any:
        if weights is None:
            return None
        if isinstance(weights, Mapping):
            return {
                str(name): _positive_float(value, f"weight for {name!r}")
                for name, value in weights.items()
            }
        return _positive_float(weights, "weight")

    # -- constructors ---------------------------------------------------------

    @classmethod
    def l2(cls, **options: Any) -> "Misfit":
        """Return a least-squares misfit."""

        return cls(loss=Loss.l2(), **options)

    @classmethod
    def huber(cls, delta: float = 1.5, **options: Any) -> "Misfit":
        """Return a Huber misfit."""

        return cls(loss=Loss.huber(delta), **options)

    @classmethod
    def student_t(cls, nu: float = 2.0, c2: float = 1.0, **options: Any) -> "Misfit":
        """Return a Student-t misfit."""

        return cls(loss=Loss.student_t(nu=nu, c2=c2), **options)

    @classmethod
    def terms(cls, *terms: ObjectiveTerm, **options: Any) -> "Misfit":
        """Return a misfit from explicit objective terms."""

        return cls(terms=terms, **options)

    # -- queries ----------------------------------------------------------------

    @property
    def explicit_terms(self) -> Optional[Tuple[ObjectiveTerm, ...]]:
        """Return the explicit terms, or ``None`` when terms follow receiver groups."""

        return self._terms

    def objective_terms(self, receiver_groups: Sequence[str]) -> List[ObjectiveTerm]:
        """Return the objective terms for the named receiver groups, in order."""

        groups = [str(name) for name in receiver_groups]
        if self._terms is not None:
            unknown = sorted({t.receiver_group for t in self._terms}.difference(groups))
            if unknown:
                raise ValueError(
                    "objective terms reference unknown receiver group(s): "
                    + ", ".join(unknown)
                )
            return list(self._terms)
        terms = []
        for name in groups:
            if isinstance(self.weights, Mapping):
                if name not in self.weights:
                    raise KeyError(f"no misfit weight for receiver group {name!r}")
                weight = self.weights[name]
            else:
                weight = 1.0 if self.weights is None else self.weights
            terms.append(
                ObjectiveTerm(
                    receiver_group=name,
                    loss=self.loss,
                    comparison=self.comparison,
                    weight=weight,
                    normalization=self.normalization,
                )
            )
        return terms

    def projection_for(self, group: str) -> ReceiverProjection:
        """Return the receiver projection applied to ``group``."""

        if isinstance(self.projection, Mapping):
            return self.projection.get(group, ReceiverProjection.identity())
        return self.projection

    def required_derivatives(self, group: str) -> Tuple[str, ...]:
        """Return observed derivative axes every term over ``group`` needs."""

        axes: List[str] = []
        terms = (
            self._terms if self._terms is not None else self.objective_terms([group])
        )
        for term in terms:
            if term.receiver_group != group:
                continue
            for axis in term.comparison.requires_derivatives:
                if axis not in axes:
                    axes.append(axis)
        return tuple(axes)

    # -- serialization ----------------------------------------------------------

    def to_fs(
        self,
        receiver_groups: Union[Mapping[str, ObservedGroup], Sequence[ObservedGroup]],
        ctx: Optional[ExportContext] = None,
    ) -> Dict[str, Any]:
        """Serialize the ``Imaging.misfit`` payload.

        Args:
            receiver_groups: Ordered observed groups (mapping by name or
                sequence) from :meth:`ObservedData.resolve`.
            ctx: Optional export context for HDF5-materialized trace weights.

        Raises:
            ValueError: If a phase-derivative term lacks observed ``df`` data,
                or explicit terms name receiver groups that are absent.
        """

        if isinstance(receiver_groups, Mapping):
            groups = [
                group if isinstance(group, ObservedGroup) else ObservedGroup(name=name)
                for name, group in receiver_groups.items()
            ]
            for name, group in zip(receiver_groups, groups):
                if group.name != str(name):
                    raise ValueError(
                        f"receiver group mapping key {name!r} differs from group "
                        f"name {group.name!r}"
                    )
        else:
            groups = list(receiver_groups)
        if not groups:
            raise ValueError("Misfit.to_fs requires at least one receiver group")
        names = [group.name for group in groups]
        if len(set(names)) != len(names):
            raise ValueError("receiver group names must be unique")
        terms = self.objective_terms(names)
        by_name = {group.name: group for group in groups}
        for term in terms:
            group = by_name[term.receiver_group]
            for axis in term.comparison.requires_derivatives:
                if axis not in group.derivatives:
                    raise ValueError(
                        f"objective term {term.id!r} uses a {term.comparison.kind} "
                        f"comparison but receiver group {group.name!r} has no "
                        f"observed {axis!r} derivatives"
                    )
        unknown = sorted(set(self.group_preprocess).difference(names))
        if unknown:
            raise ValueError(
                "group_preprocess names unknown receiver group(s): "
                + ", ".join(unknown)
            )

        receiver_payload = []
        for index, group in enumerate(groups):
            entry = group.to_fs()
            entry["projection"] = self.projection_for(group.name).to_fs()
            hooks = self.group_preprocess.get(group.name, ())
            if hooks:
                entry["preprocess"] = _hooks_to_fs(
                    hooks, ctx, scope=f"receiver_groups/{index}"
                )
            receiver_payload.append(entry)

        return {
            "objective_terms": [
                term.to_fs(ctx, scope=f"objective_terms/{index}")
                for index, term in enumerate(terms)
            ],
            "receiver_groups": receiver_payload,
            "preprocess": {
                "include_defaults": self.include_default_preprocess,
                "hooks": _hooks_to_fs(self.preprocess, ctx, scope="misfit"),
            },
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "Misfit":
        """Deserialize a ``misfit`` payload (terms or legacy objective form).

        Observed data references are not part of the misfit; recover them
        with :meth:`receiver_groups_from_fs`.
        """

        hooks, include_defaults = _hooks_from_fs(data.get("preprocess"))
        projections: Dict[str, ReceiverProjection] = {}
        group_hooks: Dict[str, Tuple[Preprocess, ...]] = {}
        for entry in data.get("receiver_groups", []):
            if "projection" in entry:
                projections[entry["name"]] = ReceiverProjection.from_fs(
                    entry["projection"]
                )
            entry_hooks, entry_defaults = _hooks_from_fs(entry.get("preprocess"))
            if entry_defaults:
                raise ValueError(
                    "receiver-group preprocess cannot include legacy defaults"
                )
            if entry_hooks:
                group_hooks[entry["name"]] = entry_hooks
        distinct = {json.dumps(p.to_fs(), sort_keys=True) for p in projections.values()}
        projection: Any
        if not projections:
            projection = None
        elif len(distinct) == 1 and len(projections) == len(
            data.get("receiver_groups", [])
        ):
            projection = next(iter(projections.values()))
        else:
            projection = projections
        common = {
            "preprocess": hooks,
            "projection": projection,
            "group_preprocess": group_hooks or None,
            "include_default_preprocess": include_defaults,
        }
        if "objective_terms" in data:
            return cls(
                terms=[ObjectiveTerm.from_fs(term) for term in data["objective_terms"]],
                **common,
            )
        return cls(
            loss=data.get("objective"),
            comparison=data.get("comparison"),
            normalization=Normalization.observed_rms(),
            **common,
        )

    @staticmethod
    def receiver_groups_from_fs(data: Mapping[str, Any]) -> Dict[str, ObservedGroup]:
        """Return the observed groups of a ``misfit`` payload, in order."""

        return {
            entry["name"]: ObservedGroup.from_fs(entry)
            for entry in data.get("receiver_groups", [])
        }

    def __repr__(self) -> str:
        if self._terms is not None:
            return f"Misfit.terms({', '.join(term.id for term in self._terms)})"
        return (
            f"Misfit(loss={self.loss.kind!r}, comparison={self.comparison.kind!r}, "
            f"normalization={self.normalization.kind!r})"
        )
