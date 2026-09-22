"""Auxiliary model extension (FWIME) on top of an :class:`ImagingProblem`.

An :class:`Extension` attaches a tap space to material control blocks: one
time-lag axis (:class:`Lags`) or one spatial half-offset axis
(:class:`HalfOffsets`) per field.  :meth:`ImagingProblem.extend` returns an
:class:`ExtendedProblem` that shares the state, cache, site and simulation of
the problem and exposes Sauce's extension actions:

``solve``
    the regularized inner solve for the taps (``fs-extension-vector-1``
    solution and ``fs-extension-solve-1`` report per frequency task);
``value`` / ``gradient``
    the reduced objective and its background gradient (``solve`` with
    ``model_gradient``);
``normal``
    the reduced Gauss-Newton Schur action ``G*WG - G*WB (B*WB + D)^-1 B*WG``
    (``solve`` with ``reduced_normal``);
``linearize(v).jacobian`` / ``.tap_normal``
    the tap-space Jacobian (extension ``jvp`` / ``vjp``) and the unregularized
    ``B*WB`` (extension ``normal``).

Every extension action runs one single-frequency job per saved task of the
extension linearization, like the derivative actions of
:class:`~frequensolve.imaging.problem.Linearization`; each frequency has its
own inner solve (Sauce composes no cross-frequency bands) and reduced
covectors and objective values are summed with the view's frequency weights.

Deviation from the design sketch: ``linearize(v).normal`` is the
:class:`~frequensolve.imaging.operators.ReducedNormal` on the control space
(what ``xp.normal(v)`` returns and what :class:`~frequensolve.imaging.workflows.FWI`
applies as the Hessian action); the unregularized tap-space ``B*WB`` is
``linearize(v).tap_normal``.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from frequensolve.imaging._artifacts import (
    ControlVectorFile,
    ExtensionSolveReport,
    ExtensionVectorField,
    ExtensionVectorFile,
    ObjectiveReport,
    unqualified_block_name,
)
from frequensolve.imaging._backend import (
    LinearizationEntry,
    fingerprint,
    read_report,
    read_task_objective_vectors,
    reduce_covectors,
    total_value,
)
from frequensolve.imaging.controls import (
    ControlSpace,
    ControlState,
    ControlVector,
    ResolvedBlock,
)
from frequensolve.imaging.data import DataSpace, DataVector
from frequensolve.imaging.jobs import FWIOperatorJob
from frequensolve.imaging.misfit import Loss
from frequensolve.imaging.operators import (
    ExtensionJacobian,
    ExtensionNormal,
    ReducedNormal,
)
from frequensolve.inversion.validation import gradient_taylor_test, real_adjoint_test
from frequensolve.units import is_quantity, unit_expression, ureg

if TYPE_CHECKING:  # pragma: no cover - typing only
    from frequensolve.imaging.problem import ImagingProblem

__all__ = [
    "ExtendedProblem",
    "Extension",
    "ExtensionLinearization",
    "ExtensionManifest",
    "ExtensionSpace",
    "ExtensionVector",
    "HalfOffsets",
    "Lags",
]

EXTENSION_MANIFEST_SCHEMA = "fs-model-extension-1"
MATERIAL_KINDS = frozenset({"profile", "grid", "mesh"})
REDUCED_LOSSES = frozenset({"l2", "huber", "student_t"})
_REDUCED_NORMAL_KEYS = ("relative_tolerance", "absolute_tolerance", "max_iterations")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _magnitude(value: Any, units: str, label: str) -> float:
    """Return ``value`` as a float in ``units`` (quantities are converted)."""

    if is_quantity(value):
        try:
            number = float(value.to(units).magnitude)
        except Exception as exc:
            raise ValueError(f"{label} {value!r} is not a {units} quantity") from exc
    else:
        number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _convert(values: np.ndarray, units: str, target: str) -> np.ndarray:
    """Convert magnitudes from ``units`` to ``target`` with pint."""

    if units == target:
        return np.asarray(values, dtype=np.float64)
    try:
        return np.asarray(
            ureg.Quantity(np.asarray(values, dtype=np.float64), units)
            .to(target)
            .magnitude,
            dtype=np.float64,
        )
    except Exception as exc:
        raise ValueError(f"units {units!r} are not convertible to {target!r}") from exc


def _scale_payload(value: Any, units: str, label: str) -> Optional[Dict[str, Any]]:
    """Return ``{value, units}`` for a scale given as a quantity or a number."""

    if value is None:
        return None
    if is_quantity(value):
        text = unit_expression(value.units)
        return {"value": _magnitude(value, text, label), "units": text}
    if isinstance(value, Mapping):
        if set(value) != {"value", "units"}:
            raise ValueError(f"{label} mapping needs exactly value and units")
        number = float(value["value"])
        text = str(value["units"]).strip()
        if not text:
            raise ValueError(f"{label} units must be non-empty")
        return {"value": number, "units": text}
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return {"value": float(value[0]), "units": str(value[1]).strip()}
    return {"value": float(value), "units": units}


def _digest(values: Any) -> str:
    import hashlib

    array = np.ascontiguousarray(np.asarray(values))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _task_file(path: Path, task: int) -> Path:
    """Return Sauce's ``<stem>_<task><ext>`` sibling of an exact output path."""

    return path.with_name(f"{path.stem}_{int(task)}{path.suffix}")


def _weights(view: Any, count: int) -> np.ndarray:
    table = getattr(view, "weights", None)
    if table is None:
        return np.ones(count, dtype=np.float64)
    array = np.asarray(table, dtype=np.float64).reshape(-1)
    if array.size != count:
        raise ValueError(f"expected {count} frequency weights, received {array.size}")
    return array


def _control_vector_from_file(
    file: ControlVectorFile, space: ControlSpace
) -> ControlVector:
    from frequensolve.imaging.problem import _vector_from_file

    return _vector_from_file(file, space)


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Lags:
    """Uniform time-lag axis on one material control.

    Args:
        control: User block key of the problem's control space or the
            unqualified material control id.
        count: Number of lags (``>= 1``).
        origin: First lag (number in ``units`` or a time quantity).
        spacing: Lag spacing (``> 0``; number in ``units`` or a quantity).
        units: Time unit of ``origin`` and ``spacing`` (``"ms"`` by default).
    """

    control: str
    count: int
    origin: Any
    spacing: Any
    units: str = "ms"

    axis: str = dataclasses.field(default="lag", init=False, repr=False)

    def __post_init__(self) -> None:
        control = str(self.control).strip()
        if not control:
            raise ValueError("Lags requires a control")
        units = str(self.units).strip()
        if not units:
            raise ValueError("Lags requires non-empty time units")
        count = int(self.count)
        if count < 1:
            raise ValueError("Lags count must be >= 1")
        origin = _magnitude(self.origin, units, "Lags origin")
        spacing = _magnitude(self.spacing, units, "Lags spacing")
        if spacing <= 0.0:
            raise ValueError("Lags spacing must be positive")
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)

    @property
    def n_axis(self) -> int:
        return self.count

    def coordinates(self) -> np.ndarray:
        """Return the lags in seconds."""

        lags = self.origin + self.spacing * np.arange(self.count, dtype=np.float64)
        return _convert(lags, self.units, "s")

    def to_fs(self, control_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the ``fwi_operator.extension.fields[]`` entry."""

        return {
            "control": self.control if control_id is None else control_id,
            "lags": {
                "count": self.count,
                "origin": self.origin,
                "spacing": self.spacing,
                "units": self.units,
            },
        }


@dataclasses.dataclass(frozen=True)
class HalfOffsets:
    """Spatial half-offset axis on one material control.

    Args:
        control: User block key or unqualified material control id.
        half_offsets: ``(n, dim)`` half-offset vectors (``dim`` 2 or 3), as
            numbers in ``units`` or one length quantity array.
        units: Length unit (``"m"`` by default).
        packet_mb: Optional native donor-exchange packet bound (``> 0``).
        artifact: Optional mesh-control halo artifact path override.
    """

    control: str
    half_offsets: Any
    units: str = "m"
    packet_mb: Optional[float] = None
    artifact: Optional[Union[str, Path]] = None

    axis: str = dataclasses.field(default="offset", init=False, repr=False)

    def __post_init__(self) -> None:
        control = str(self.control).strip()
        if not control:
            raise ValueError("HalfOffsets requires a control")
        units = str(self.units).strip()
        if not units:
            raise ValueError("HalfOffsets requires non-empty length units")
        raw = self.half_offsets
        if is_quantity(raw):
            try:
                raw = raw.to(units).magnitude
            except Exception as exc:
                raise ValueError(
                    f"half_offsets {raw!r} are not a {units} quantity"
                ) from exc
        array = np.asarray(raw, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] not in (2, 3):
            raise ValueError("half_offsets must be an (n, 2) or (n, 3) array")
        if not np.all(np.isfinite(array)):
            raise ValueError("half_offsets must be finite")
        packet = self.packet_mb
        if packet is not None:
            packet = float(packet)
            if not math.isfinite(packet) or packet <= 0.0:
                raise ValueError("packet_mb must be positive")
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "half_offsets", array)
        object.__setattr__(self, "packet_mb", packet)
        object.__setattr__(
            self, "artifact", None if self.artifact is None else Path(self.artifact)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HalfOffsets):
            return NotImplemented
        return (
            self.control == other.control
            and self.units == other.units
            and self.packet_mb == other.packet_mb
            and self.artifact == other.artifact
            and np.array_equal(self.half_offsets, other.half_offsets)
        )

    def __hash__(self) -> int:
        return hash((self.control, self.units, self.half_offsets.tobytes()))

    @property
    def n_axis(self) -> int:
        return int(self.half_offsets.shape[0])

    def coordinates(self) -> np.ndarray:
        """Return the half-offset vectors in meters, shape ``(n, dim)``."""

        return _convert(self.half_offsets, self.units, "m")

    def to_fs(self, control_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the ``fwi_operator.extension.fields[]`` entry."""

        offsets: Dict[str, Any] = {
            "half_offsets": self.half_offsets.tolist(),
            "units": self.units,
        }
        if self.packet_mb is not None:
            offsets["packet_mb"] = self.packet_mb
        if self.artifact is not None:
            offsets["artifact"] = str(self.artifact)
        return {
            "control": self.control if control_id is None else control_id,
            "offsets": offsets,
        }


ExtensionField = Union[Lags, HalfOffsets]


# ---------------------------------------------------------------------------
# extension
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Extension:
    """Auxiliary model extension: fields plus the inner-solve settings.

    Args:
        fields: One :class:`Lags` or :class:`HalfOffsets` per material control.
        damping: Positive damping amplitude (``damping**2`` enters the normal).
        lag_penalty, lag_scale: Squared lag penalty ``lag_penalty * tau /
            lag_scale``; a nonzero penalty needs a positive time scale
            (quantity, ``(value, units)`` or seconds).
        offset_penalty, offset_scale: The same for spatial half-offsets
            (``offset_penalty * |h| / offset_scale``; length scale).
        field_scales: Optional physical amplitude per field (Krylov
            coordinates ``z`` with ``taps = scale * z``).
        tolerance, absolute_tolerance, max_iterations: Inner CG controls.
        require_convergence: Fail an unconverged inner solve.
        cache_mb, workspace_mb: Per-rank memory budgets.
        max_outer_iterations, max_line_search, gradient_tolerance
        (relative), gradient_absolute_tolerance: Robust (Huber, Student-t)
            observed-data inner iteration controls.
        reduced_normal: Optional overrides ``{relative_tolerance,
            absolute_tolerance, max_iterations}`` for the Schur response
            solve (defaults inherit the inner solver settings).
    """

    fields: Sequence[ExtensionField]
    damping: float
    lag_penalty: float = 0.0
    lag_scale: Any = None
    offset_penalty: float = 0.0
    offset_scale: Any = None
    field_scales: Optional[Sequence[float]] = None
    tolerance: float = 1.0e-6
    absolute_tolerance: float = 0.0
    max_iterations: int = 100
    require_convergence: bool = True
    cache_mb: Optional[float] = None
    workspace_mb: Optional[float] = None
    max_outer_iterations: Optional[int] = None
    max_line_search: Optional[int] = None
    gradient_tolerance: Optional[float] = None
    gradient_absolute_tolerance: Optional[float] = None
    reduced_normal: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        fields = (
            [self.fields]
            if isinstance(self.fields, (Lags, HalfOffsets))
            else list(self.fields)
        )
        if not fields:
            raise ValueError("Extension requires at least one field")
        seen = set()
        for field_ in fields:
            if not isinstance(field_, (Lags, HalfOffsets)):
                raise TypeError("Extension fields must be Lags or HalfOffsets")
            if field_.control in seen:
                raise ValueError(f"duplicate extension control {field_.control!r}")
            seen.add(field_.control)
        object.__setattr__(self, "fields", tuple(fields))

        damping = float(self.damping)
        if not math.isfinite(damping) or damping <= 0.0:
            raise ValueError("damping must be finite and positive")
        object.__setattr__(self, "damping", damping)

        for key in ("lag_penalty", "offset_penalty"):
            penalty = float(getattr(self, key))
            if not math.isfinite(penalty) or penalty < 0.0:
                raise ValueError(f"{key} must be finite and nonnegative")
            object.__setattr__(self, key, penalty)
        lag_scale = _scale_payload(self.lag_scale, "s", "lag_scale")
        offset_scale = _scale_payload(self.offset_scale, "m", "offset_scale")
        for label, scale in (("lag_scale", lag_scale), ("offset_scale", offset_scale)):
            if scale is not None and (
                not math.isfinite(scale["value"]) or scale["value"] <= 0.0
            ):
                raise ValueError(f"{label} must be positive")
        if self.lag_penalty > 0.0 and lag_scale is None:
            raise ValueError("lag_penalty > 0 requires lag_scale")
        if self.offset_penalty > 0.0 and offset_scale is None:
            raise ValueError("offset_penalty > 0 requires offset_scale")
        if lag_scale is not None:
            _convert(np.ones(1), lag_scale["units"], "s")
        if offset_scale is not None:
            _convert(np.ones(1), offset_scale["units"], "m")
        object.__setattr__(self, "lag_scale", lag_scale)
        object.__setattr__(self, "offset_scale", offset_scale)

        if self.field_scales is not None:
            scales = tuple(float(v) for v in np.asarray(self.field_scales).reshape(-1))
            if len(scales) != len(fields):
                raise ValueError(
                    f"field_scales needs one entry per field ({len(fields)})"
                )
            if any(not math.isfinite(s) or s <= 0.0 for s in scales):
                raise ValueError("field_scales must be positive")
            object.__setattr__(self, "field_scales", scales)

        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        object.__setattr__(self, "tolerance", tolerance)
        absolute = float(self.absolute_tolerance)
        if not math.isfinite(absolute) or absolute < 0.0:
            raise ValueError("absolute_tolerance must be nonnegative")
        object.__setattr__(self, "absolute_tolerance", absolute)
        iterations = int(self.max_iterations)
        if iterations < 0:
            raise ValueError("max_iterations must be nonnegative")
        object.__setattr__(self, "max_iterations", iterations)
        object.__setattr__(self, "require_convergence", bool(self.require_convergence))
        for key in (
            "cache_mb",
            "workspace_mb",
            "gradient_tolerance",
            "gradient_absolute_tolerance",
        ):
            value = getattr(self, key)
            if value is not None:
                number = float(value)
                if not math.isfinite(number) or number < 0.0:
                    raise ValueError(f"{key} must be finite and nonnegative")
                object.__setattr__(self, key, number)
        for key in ("max_outer_iterations", "max_line_search"):
            value = getattr(self, key)
            if value is not None:
                count = int(value)
                if count < 1:
                    raise ValueError(f"{key} must be positive")
                object.__setattr__(self, key, count)
        if self.reduced_normal is not None:
            if not isinstance(self.reduced_normal, Mapping):
                raise TypeError("reduced_normal must be a mapping")
            unknown = sorted(set(self.reduced_normal).difference(_REDUCED_NORMAL_KEYS))
            if unknown:
                raise ValueError(
                    f"unsupported reduced_normal option(s): {', '.join(unknown)}"
                )
            options: Dict[str, Any] = {}
            for key in ("relative_tolerance", "absolute_tolerance"):
                if key in self.reduced_normal:
                    number = float(self.reduced_normal[key])
                    if not math.isfinite(number) or number < 0.0:
                        raise ValueError(f"reduced_normal {key} must be nonnegative")
                    options[key] = number
            if "max_iterations" in self.reduced_normal:
                count = int(self.reduced_normal["max_iterations"])
                if count < 0:
                    raise ValueError(
                        "reduced_normal max_iterations must be nonnegative"
                    )
                options["max_iterations"] = count
            object.__setattr__(self, "reduced_normal", options)

    @property
    def controls(self) -> Tuple[str, ...]:
        """Return the field controls in order."""

        return tuple(field_.control for field_ in self.fields)

    def solver_fs(self) -> Dict[str, Any]:
        """Return ``extension.solver`` without the ``solution``/``report`` paths."""

        solver: Dict[str, Any] = {
            "damping": self.damping,
            "relative_tolerance": self.tolerance,
            "absolute_tolerance": self.absolute_tolerance,
            "max_iterations": self.max_iterations,
            "require_convergence": self.require_convergence,
        }
        if self.lag_penalty > 0.0 or self.lag_scale is not None:
            solver["lag_penalty"] = self.lag_penalty
        if self.lag_scale is not None:
            solver["lag_scale"] = dict(self.lag_scale)
        if self.offset_penalty > 0.0 or self.offset_scale is not None:
            solver["offset_penalty"] = self.offset_penalty
        if self.offset_scale is not None:
            solver["offset_scale"] = dict(self.offset_scale)
        if self.field_scales is not None:
            solver["field_scales"] = list(self.field_scales)
        for key, name in (
            ("cache_mb", "cache_mb"),
            ("workspace_mb", "workspace_mb"),
            ("max_outer_iterations", "max_outer_iterations"),
            ("max_line_search", "max_line_search"),
            ("gradient_tolerance", "gradient_relative_tolerance"),
            ("gradient_absolute_tolerance", "gradient_absolute_tolerance"),
        ):
            value = getattr(self, key)
            if value is not None:
                solver[name] = value
        return solver

    def fields_fs(
        self, ids: Optional[Mapping[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Return ``extension.fields`` (``ids`` maps controls to Sauce ids)."""

        table = dict(ids or {})
        return [field_.to_fs(table.get(field_.control)) for field_ in self.fields]

    def to_fs(self, ids: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        """Return the ``fields`` and ``solver`` parts of ``fwi_operator.extension``."""

        payload: Dict[str, Any] = {
            "fields": self.fields_fs(ids),
            "solver": self.solver_fs(),
        }
        if self.reduced_normal is not None:
            payload["reduced_normal"] = dict(self.reduced_normal)
        return payload

    def lag_scale_seconds(self) -> Optional[float]:
        if self.lag_scale is None:
            return None
        return float(
            _convert(np.array([self.lag_scale["value"]]), self.lag_scale["units"], "s")[
                0
            ]
        )

    def offset_scale_meters(self) -> Optional[float]:
        if self.offset_scale is None:
            return None
        return float(
            _convert(
                np.array([self.offset_scale["value"]]), self.offset_scale["units"], "m"
            )[0]
        )

    def axis_penalties(self, space: "ExtensionSpace") -> np.ndarray:
        """Return the squared axis penalty per packed tap of ``space``.

        ``(lag_penalty * tau / lag_scale)**2`` for lag fields and
        ``(offset_penalty * |h| / offset_scale)**2`` for offset fields, zero
        when the penalty is off; ``damping**2`` is not included.
        """

        out = np.zeros(space.size, dtype=np.float64)
        for control, sl in space.slices.items():
            field_ = space.field(control)
            count = space.spatial_counts[control]
            coordinates = space.axis_coordinates(control)
            if isinstance(field_, Lags):
                scale = self.lag_scale_seconds()
                weights = (
                    np.zeros(field_.n_axis)
                    if self.lag_penalty == 0.0 or scale is None
                    else (self.lag_penalty * np.abs(coordinates) / scale) ** 2
                )
            else:
                scale = self.offset_scale_meters()
                weights = (
                    np.zeros(field_.n_axis)
                    if self.offset_penalty == 0.0 or scale is None
                    else (
                        self.offset_penalty
                        * np.linalg.norm(coordinates, axis=1)
                        / scale
                    )
                    ** 2
                )
            out[sl] = np.repeat(weights, count)
        return out


# ---------------------------------------------------------------------------
# tap space and vectors
# ---------------------------------------------------------------------------


class ExtensionSpace:
    """Tap space of an :class:`Extension` over the material blocks of a space.

    Each field contributes ``spatial_count x n_axis`` real taps in Sauce's
    ``fs-extension-vector-1`` order: spatial index fastest, then axis, then
    field.  The spatial count is the full Sauce-layout size of the borrowed
    material block (taps are never masked by support).

    Args:
        space: Control space whose material blocks the fields borrow.
        fields: :class:`Extension` or the field sequence.
    """

    def __init__(self, space: ControlSpace, fields: Any) -> None:
        if isinstance(fields, Extension):
            fields = fields.fields
        elif isinstance(fields, (Lags, HalfOffsets)):
            fields = [fields]
        self.space = space
        self.fields: Tuple[ExtensionField, ...] = tuple(fields)
        if not self.fields:
            raise ValueError("ExtensionSpace requires at least one field")
        self.blocks: Dict[str, ResolvedBlock] = {}
        self.control_ids: Dict[str, str] = {}
        self.spatial_counts: Dict[str, int] = {}
        self._slices: Dict[str, slice] = {}
        offset = 0
        for field_ in self.fields:
            try:
                matches = space._select(field_.control)
            except KeyError as exc:
                raise ValueError(
                    f"extension control {field_.control!r} is not a block of the "
                    f"control space {list(space.blocks)}"
                ) from exc
            if len(matches) != 1:
                raise ValueError(
                    f"extension control {field_.control!r} addresses "
                    f"{len(matches)} blocks; extension fields borrow one material "
                    "control map each"
                )
            block = matches[0]
            if block.kind not in MATERIAL_KINDS:
                raise ValueError(
                    f"extension control {field_.control!r} is a {block.kind} block; "
                    "extension fields borrow material control maps only"
                )
            if field_.control in self.blocks:
                raise ValueError(f"duplicate extension control {field_.control!r}")
            self.blocks[field_.control] = block
            self.control_ids[field_.control] = unqualified_block_name(block.name)
            count = int(block.size)
            self.spatial_counts[field_.control] = count
            self._slices[field_.control] = slice(offset, offset + count * field_.n_axis)
            offset += count * field_.n_axis
        self._size = offset

    # -- descriptors ----------------------------------------------------------

    @property
    def controls(self) -> Tuple[str, ...]:
        return tuple(self.blocks)

    def field(self, control: str) -> ExtensionField:
        for field_ in self.fields:
            if field_.control == control:
                return field_
        raise KeyError(f"extension space has no field {control!r}")

    @property
    def size(self) -> int:
        return self._size

    @property
    def shape(self) -> Tuple[int]:
        return (self._size,)

    @property
    def slices(self) -> Dict[str, slice]:
        """Return ``control -> slice`` into the packed tap vector."""

        return dict(self._slices)

    def shapes(self) -> Dict[str, Tuple[int, int]]:
        """Return ``control -> (spatial_count, n_axis)``."""

        return {
            field_.control: (self.spatial_counts[field_.control], field_.n_axis)
            for field_ in self.fields
        }

    def axis_coordinates(self, control: str) -> np.ndarray:
        """Return the lags in seconds ``(n,)`` or half-offsets in meters ``(n, dim)``."""

        return self.field(control).coordinates()

    def descriptor(self) -> List[Dict[str, Any]]:
        """Return a JSON-compatible description of the fields (fingerprinting)."""

        out = []
        for field_ in self.fields:
            entry = field_.to_fs(self.control_ids[field_.control])
            entry["spatial_count"] = self.spatial_counts[field_.control]
            out.append(entry)
        return out

    def fields_fs(self) -> List[Dict[str, Any]]:
        """Return ``fwi_operator.extension.fields`` with Sauce control ids."""

        return [
            field_.to_fs(self.control_ids[field_.control]) for field_ in self.fields
        ]

    def equivalent(self, other: "ExtensionSpace") -> bool:
        return self is other or (
            self.size == other.size and self.descriptor() == other.descriptor()
        )

    def __repr__(self) -> str:
        parts = ", ".join(f"{c}[{n}x{k}]" for c, (n, k) in self.shapes().items())
        return f"ExtensionSpace({parts})"

    # -- vectors --------------------------------------------------------------

    def zeros(self) -> "ExtensionVector":
        return ExtensionVector(np.zeros(self.size), self)

    def ones(self) -> "ExtensionVector":
        return ExtensionVector(np.ones(self.size), self)

    def random(self, seed: Optional[int] = None) -> "ExtensionVector":
        rng = np.random.default_rng(seed)
        return ExtensionVector(rng.standard_normal(self.size), self)

    def vector(self, values: Any) -> "ExtensionVector":
        return ExtensionVector(values, self)

    def pack(self, fields: Mapping[str, Any]) -> "ExtensionVector":
        """Pack ``control -> (spatial_count, n_axis)`` arrays into a vector."""

        values = np.zeros(self.size, dtype=np.float64)
        unknown = sorted(set(fields).difference(self.blocks))
        if unknown:
            raise KeyError(f"unknown extension field(s): {', '.join(unknown)}")
        for control, shape in self.shapes().items():
            if control not in fields:
                continue
            array = np.asarray(fields[control], dtype=np.float64)
            if array.ndim == 1:
                array = array.reshape(-1, 1)
            if array.shape != shape:
                raise ValueError(
                    f"field {control!r} has shape {array.shape}; expected {shape}"
                )
            values[self._slices[control]] = array.reshape(-1, order="F")
        return ExtensionVector(values, self)

    def unpack(self, vector: Any) -> Dict[str, np.ndarray]:
        """Return ``control -> (spatial_count, n_axis)`` arrays of a vector."""

        values = (
            vector.values if isinstance(vector, ExtensionVector) else np.asarray(vector)
        )
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != self.size:
            raise ValueError(
                f"vector has {values.size} entries; the space has {self.size}"
            )
        return {
            control: np.array(values[self._slices[control]].reshape(shape, order="F"))
            for control, shape in self.shapes().items()
        }


class ExtensionVector:
    """Real tap vector on an :class:`ExtensionSpace` (tangent or covector)."""

    __array_priority__ = 20.0

    def __init__(self, values: Any, space: ExtensionSpace) -> None:
        if not isinstance(space, ExtensionSpace):
            raise TypeError("ExtensionVector requires an ExtensionSpace")
        array = np.asarray(values)
        if np.iscomplexobj(array):
            raise ValueError("extension vectors are real")
        array = np.array(array, dtype=np.float64).reshape(-1)
        if array.size != space.size:
            raise ValueError(
                f"vector has {array.size} entries; the extension space has {space.size}"
            )
        self._values = array
        self.space = space

    @property
    def values(self) -> np.ndarray:
        return self._values

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)

    @property
    def shape(self) -> Tuple[int]:
        return (int(self._values.size),)

    @property
    def size(self) -> int:
        return int(self._values.size)

    def __len__(self) -> int:
        return int(self._values.size)

    def __iter__(self) -> Iterator[float]:
        return iter(self._values.tolist())

    def copy(self) -> "ExtensionVector":
        return ExtensionVector(np.array(self._values, copy=True), self.space)

    def _other(self, other: Any) -> Any:
        if isinstance(other, ExtensionVector):
            if other.space is not self.space and not other.space.equivalent(self.space):
                raise ValueError("extension vectors belong to different spaces")
            return other._values
        array = np.asarray(other)
        if array.ndim == 0:
            return array
        if array.shape != self._values.shape:
            raise ValueError(f"operand shape {array.shape} does not match {self.shape}")
        return array

    def __add__(self, other: Any) -> "ExtensionVector":
        return ExtensionVector(self._values + self._other(other), self.space)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "ExtensionVector":
        return ExtensionVector(self._values - self._other(other), self.space)

    def __rsub__(self, other: Any) -> "ExtensionVector":
        return ExtensionVector(self._other(other) - self._values, self.space)

    def __mul__(self, other: Any) -> "ExtensionVector":
        return ExtensionVector(self._values * self._other(other), self.space)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "ExtensionVector":
        return ExtensionVector(self._values / self._other(other), self.space)

    def __neg__(self) -> "ExtensionVector":
        return ExtensionVector(-self._values, self.space)

    def __pos__(self) -> "ExtensionVector":
        return self.copy()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtensionVector):
            return NotImplemented
        return self.space.equivalent(other.space) and np.array_equal(
            self._values, other._values
        )

    __hash__ = None  # type: ignore[assignment]  # noqa: PLW1641

    def dot(self, other: Any) -> float:
        return float(np.dot(self._values, self._other(other)))

    def norm(self, order: Any = None) -> float:
        return float(np.linalg.norm(self._values, ord=order))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self.space.unpack(self._values)[key]
        return self._values[key]

    def fields(self) -> Dict[str, np.ndarray]:
        """Return ``control -> (spatial_count, n_axis)`` arrays."""

        return self.space.unpack(self._values)

    def __repr__(self) -> str:
        return f"ExtensionVector(size={self.size}, norm={self.norm():.6g})"

    # -- conversions ----------------------------------------------------------

    def to_xarray(self) -> Any:
        """Return an ``xarray.Dataset`` with one array per control.

        Dimensions are the block's coordinate dimensions (or ``coefficient``)
        times ``lag`` (seconds) or ``offset`` (index, with the half-offset
        vectors in meters as the ``half_offset`` coordinate).
        """

        import xarray as xr

        arrays: Dict[str, Any] = {}
        for control, values in self.fields().items():
            block = self.space.blocks[control]
            field_ = self.space.field(control)
            count = values.shape[0]
            dims: List[str] = []
            coords: Dict[str, Any] = {}
            shape: List[int] = []
            block_coords = block.coords or {}
            if block.dims and all(d in block_coords for d in block.dims):
                sizes = [int(np.asarray(block_coords[d]).size) for d in block.dims]
                if int(np.prod(sizes)) == count:
                    dims = list(block.dims)
                    shape = sizes
                    coords = {d: np.asarray(block_coords[d]) for d in block.dims}
            if not dims:
                dims, shape = ["coefficient"], [count]
                coords = {"coefficient": np.arange(count)}
            axis = field_.axis
            if isinstance(field_, Lags):
                coords[axis] = field_.coordinates()
            else:
                offsets = field_.coordinates()
                coords[axis] = np.arange(field_.n_axis)
                coords["half_offset"] = ((axis, "component"), offsets)
            data = values.reshape(shape + [field_.n_axis], order="F")
            arrays[self.space.control_ids[control]] = xr.DataArray(
                data,
                dims=dims + [axis],
                coords=coords,
                name=self.space.control_ids[control],
            )
        return xr.Dataset(arrays)

    def to_file(
        self, *, fingerprint: str, baseline: str, role: str = "tangent"
    ) -> ExtensionVectorFile:
        """Return the ``fs-extension-vector-1`` representation."""

        fields = [
            ExtensionVectorField(
                values,
                axis=self.space.field(control).axis,
                control=self.space.control_ids[control],
            )
            for control, values in self.fields().items()
        ]
        return ExtensionVectorFile(
            fields, fingerprint=fingerprint, baseline=baseline, role=role
        )

    @classmethod
    def from_file(
        cls, file: Union[ExtensionVectorFile, str, Path], space: ExtensionSpace
    ) -> "ExtensionVector":
        """Read a vector file onto ``space`` (fields matched by position)."""

        if not isinstance(file, ExtensionVectorFile):
            file = ExtensionVectorFile.read(file)
        if len(file.fields) != len(space.fields):
            raise ValueError(
                f"extension vector has {len(file.fields)} fields; the space has "
                f"{len(space.fields)}"
            )
        packed: Dict[str, np.ndarray] = {}
        for stored, field_ in zip(file.fields, space.fields):
            expected = (space.spatial_counts[field_.control], field_.n_axis)
            if stored.axis != field_.axis:
                raise ValueError(
                    f"field {field_.control!r} is a {field_.axis} field; the file "
                    f"carries a {stored.axis} axis"
                )
            if stored.values.shape != expected:
                raise ValueError(
                    f"field {field_.control!r} has shape {stored.values.shape} in the "
                    f"file; the space expects {expected}"
                )
            if (
                stored.control is not None
                and stored.control != space.control_ids[field_.control]
            ):
                raise ValueError(
                    f"field {field_.control!r} is bound to control "
                    f"{stored.control!r} in the file"
                )
            packed[field_.control] = stored.values
        return space.pack(packed)


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExtensionManifest:
    """Reader for the ``fs-model-extension-1`` descriptor of one task."""

    fingerprint: str
    baseline: str
    fields: Tuple[Dict[str, Any], ...]
    raw: Dict[str, Any] = dataclasses.field(
        default_factory=dict, repr=False, compare=False
    )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ExtensionManifest":
        data = json.loads(Path(path).read_text())
        schema = data.get("schema")
        if schema != EXTENSION_MANIFEST_SCHEMA:
            raise ValueError(
                f"{path} has schema {schema!r}; expected {EXTENSION_MANIFEST_SCHEMA!r}"
            )
        return cls(
            fingerprint=str(data["fingerprint"]),
            baseline=str(data["baseline"]),
            fields=tuple(dict(f) for f in data.get("fields", [])),
            raw=dict(data),
        )

    def check(self, space: ExtensionSpace, path: Path) -> None:
        """Fail when the descriptor's spatial counts differ from ``space``."""

        if not self.fields:
            return
        if len(self.fields) != len(space.fields):
            raise ValueError(
                f"{path} describes {len(self.fields)} fields; the extension has "
                f"{len(space.fields)}"
            )
        for entry, field_ in zip(self.fields, space.fields):
            count = entry.get("spatial_count")
            expected = space.spatial_counts[field_.control]
            if count is not None and int(count) != expected:
                raise ValueError(
                    f"{path}: field {field_.control!r} has {count} spatial taps in "
                    f"Sauce; the control block has {expected} DOFs"
                )


# ---------------------------------------------------------------------------
# linearization
# ---------------------------------------------------------------------------


def _reduced_value(report: ExtensionSolveReport) -> float:
    """Return the reduced objective of one task from its solve report."""

    if report.reduced_objective is not None:
        return float(report.reduced_objective)
    if report.objective is not None:
        return float(report.objective)
    if report.data_objective is not None and report.regularization is not None:
        return float(report.data_objective) + float(report.regularization)
    if report.quadratic_objective is not None:
        return float(report.quadratic_objective)
    raise ValueError("extension solve report carries no objective value")


class ExtensionLinearization:
    """One saved extension ``linearize`` state (one task per frequency).

    Attributes:
        problem: The :class:`ExtendedProblem` that produced it.
        space: Control space of the active material blocks (support applied).
        extension_space: The tap space.
        point / state: The linearization point (active slice / full state).
        job: The extension ``linearize`` job.
        reports: Per-task objective reports of the fixed background.
        baseline_value: Weighted misfit at zero taps.
        covector: Extension covector of the residual (``Re B^H W r`` at
            zero taps), summed with the frequency weights.
        manifests: Per-task ``fs-model-extension-1`` descriptors.
    """

    def __init__(
        self,
        problem: "ExtendedProblem",
        *,
        space: ControlSpace,
        state: ControlState,
        entry: LinearizationEntry,
        reports: Sequence[ObjectiveReport],
        manifests: Sequence[ExtensionManifest],
        covector: ExtensionVector,
    ) -> None:
        self.problem = problem
        self.space = space
        self.extension_space = problem.extension_space
        self.extension = problem.extension
        self.state = state
        self.point = state.vector(space)
        self.entry = entry
        self.job: FWIOperatorJob = entry.job
        self.frequencies: List[Any] = list(self.job.f_list)
        self.fingerprint = entry.fingerprint
        assert entry.control_registry_fingerprint is not None
        self.registry_fingerprint: str = entry.control_registry_fingerprint
        self.reports = list(reports)
        self.manifests = list(manifests)
        self.frequency_weights: np.ndarray = _weights(problem, len(self.reports))
        self.baseline_value: float = total_value(
            self.reports, self.frequency_weights.tolist()
        )
        self.task_fingerprints: List[Tuple[str, str]] = [
            (
                report.state_fingerprint
                or entry.state_fingerprint
                or manifest.baseline,
                self.registry_fingerprint,
            )
            for report, manifest in zip(self.reports, self.manifests)
        ]
        self.extension_fingerprints: List[Tuple[str, str]] = [
            (manifest.fingerprint, manifest.baseline) for manifest in self.manifests
        ]
        self.covector = covector
        self._data_space: Optional[DataSpace] = None
        self._solutions: Optional[
            List[Tuple[ExtensionVector, ExtensionSolveReport]]
        ] = None
        self._gradient: Optional[ControlVector] = None
        self._jacobian: Optional[ExtensionJacobian] = None
        self._tap_normal: Optional[ExtensionNormal] = None
        self._normal: Optional[ReducedNormal] = None
        self._ops: Dict[Tuple[str, str], Any] = {}
        self._counter = itertools.count(1)

    def __repr__(self) -> str:
        return (
            f"ExtensionLinearization(blocks={list(self.space.blocks)}, "
            f"taps={self.extension_space.size}, frequencies={self.frequencies})"
        )

    # -- descriptors ----------------------------------------------------------

    @property
    def view(self) -> "ImagingProblem":
        return self.problem.problem

    @property
    def data_space(self) -> DataSpace:
        if self._data_space is None:
            self._data_space = DataSpace.from_simulation(
                self.view.simulation, frequencies=self.frequencies
            )
        return self._data_space

    @property
    def state_fingerprint(self) -> str:
        return self.task_fingerprints[0][0]

    # -- inner solve ----------------------------------------------------------

    def _solve_family(
        self,
        *,
        model_gradient: bool = False,
        per_task: Optional[Callable[[int], Dict[str, Any]]] = None,
    ) -> List[FWIOperatorJob]:
        def options(task: int) -> Dict[str, Any]:
            extra = {} if per_task is None else per_task(task)
            extension = {
                "fields": self.extension_space.fields_fs(),
                "solver": {
                    **self.extension.solver_fs(),
                    "solution": "taps.h5",
                    "report": "inner_solve.json",
                },
            }
            return {"extension": extension, "model_gradient": model_gradient, **extra}

        jobs = self._family("solve", options)
        self.problem.problem._run_jobs(jobs)
        return jobs

    def _read_solutions(
        self, jobs: Sequence[FWIOperatorJob]
    ) -> List[Tuple[ExtensionVector, ExtensionSolveReport]]:
        out = []
        for task, job in enumerate(jobs, start=1):
            report = ExtensionSolveReport.load(job.extension_report_file(1))
            taps = ExtensionVector.from_file(
                job.extension_solution_file(1), self.extension_space
            )
            if self.extension.require_convergence and not report.converged:
                raise RuntimeError(
                    f"extension inner solve of task {task} "
                    f"({self.frequencies[task - 1]} Hz) did not converge after "
                    f"{report.iterations} iterations"
                )
            out.append((taps, report))
        return out

    def _ensure_solved(self, *, gradient: bool) -> None:
        if gradient:
            if self._gradient is not None:
                return
            jobs = self._solve_family(
                model_gradient=True, per_task=lambda task: {"covector": "gradient.h5"}
            )
            self._solutions = self._read_solutions(jobs)
            self._gradient = self._reduce(jobs, weighted=True)
            return
        if self._solutions is None:
            jobs = self._solve_family()
            self._solutions = self._read_solutions(jobs)

    @property
    def solutions(self) -> List[Tuple[ExtensionVector, ExtensionSolveReport]]:
        """Return ``(taps, report)`` per frequency task (runs ``solve`` once)."""

        self._ensure_solved(gradient=False)
        assert self._solutions is not None
        return list(self._solutions)

    @property
    def taps(self) -> List[ExtensionVector]:
        return [taps for taps, _ in self.solutions]

    @property
    def solve_reports(self) -> List[ExtensionSolveReport]:
        return [report for _, report in self.solutions]

    @property
    def converged(self) -> bool:
        return all(report.converged for report in self.solve_reports)

    @property
    def value(self) -> float:
        """Return the reduced objective (weighted sum of the task reports)."""

        return float(
            np.dot(
                self.frequency_weights,
                [_reduced_value(report) for report in self.solve_reports],
            )
        )

    @property
    def gradient(self) -> ControlVector:
        """Return the reduced background gradient on ``space``."""

        self._ensure_solved(gradient=True)
        assert self._gradient is not None
        return self._gradient

    @property
    def report(self) -> Dict[str, float]:
        """Return ``label -> weighted value`` of the reduced objective parts."""

        merged: Dict[str, float] = {}
        for weight, report in zip(self.frequency_weights, self.solve_reports):
            for label in ("data_objective", "regularization", "reduced_objective"):
                value = getattr(report, label)
                if value is not None:
                    merged[label] = merged.get(label, 0.0) + float(weight) * float(
                        value
                    )
        return merged

    # -- operators ------------------------------------------------------------

    @property
    def jacobian(self) -> ExtensionJacobian:
        """Return the tap-space Jacobian ``B`` (extension ``jvp`` / ``vjp``)."""

        if self._jacobian is None:
            self._jacobian = ExtensionJacobian(self)
        return self._jacobian

    @property
    def tap_normal(self) -> ExtensionNormal:
        """Return the unregularized ``Re(B^H W B)`` on taps (extension ``normal``)."""

        if self._tap_normal is None:
            self._tap_normal = ExtensionNormal(self)
        return self._tap_normal

    @property
    def normal(self) -> ReducedNormal:
        """Return the reduced GN Schur normal on the control space."""

        if self._normal is None:
            self._normal = ReducedNormal(self.problem, self.point, linearization=self)
        return self._normal

    # -- actions --------------------------------------------------------------

    def _tap_vector(self, t: Any) -> ExtensionVector:
        if isinstance(t, ExtensionVector):
            if t.space is not self.extension_space and not t.space.equivalent(
                self.extension_space
            ):
                raise ValueError("tap vector belongs to a different extension space")
            return t
        return ExtensionVector(t, self.extension_space)

    def _control_vector(self, dv: Any) -> ControlVector:
        if isinstance(dv, ControlVector):
            if dv.space is not self.space and not dv.space.equivalent(self.space):
                raise ValueError("direction belongs to a different control space")
            return dv
        return ControlVector(dv, self.space)

    def _data_vector(self, r: Any) -> DataVector:
        if isinstance(r, DataVector):
            if r.space != self.data_space:
                raise ValueError("objective vector belongs to a different data space")
            return r
        return DataVector(np.asarray(r, dtype=self.data_space.dtype), self.data_space)

    def _ops_dir(self) -> Path:
        assert self.entry.directory is not None
        path = Path(self.entry.directory) / "xops" / f"{next(self._counter):04d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _memo(self, action: str, digest: str, compute: Callable[[], Any]) -> Any:
        key = (action, digest)
        if key not in self._ops:
            self._ops[key] = compute()
        return self._ops[key]

    def _family(
        self, action: str, per_task: Callable[[int], Dict[str, Any]]
    ) -> List[FWIOperatorJob]:
        """Build one single-frequency ``action`` job per saved task."""

        view = self.view
        view._sync_simulation(self.state)
        return [
            view._operator_job(
                self.space,
                action,
                frequencies=[frequency],
                state=self.job.state_file(task),
                **per_task(task),
            )
            for task, frequency in enumerate(self.frequencies, start=1)
        ]

    def _tap_file(self, taps: ExtensionVector, directory: Path, task: int) -> Path:
        fp, baseline = self.extension_fingerprints[task - 1]
        return taps.to_file(fingerprint=fp, baseline=baseline, role="tangent").write(
            directory / f"taps_{task}.h5"
        )

    def _direction_file(self, dv: ControlVector, directory: Path, task: int) -> Path:
        state_fp, registry_fp = self.task_fingerprints[task - 1]
        return dv.to_file(
            state_fingerprint=state_fp, registry_fingerprint=registry_fp
        ).write(directory / f"direction_{task}.h5")

    def _reduce(
        self, jobs: Sequence[FWIOperatorJob], *, weighted: bool
    ) -> ControlVector:
        total = self.space.zeros()
        for task, job in enumerate(jobs, start=1):
            part = _control_vector_from_file(reduce_covectors(job), self.space)
            if weighted:
                part = part * float(self.frequency_weights[task - 1])
            total = total + part
        return total

    def _reduce_taps(
        self, jobs: Sequence[FWIOperatorJob], *, weighted: bool
    ) -> ExtensionVector:
        total = self.extension_space.zeros()
        for task, job in enumerate(jobs, start=1):
            part = ExtensionVector.from_file(
                job.extension_covector_file(1), self.extension_space
            )
            if weighted:
                part = part * float(self.frequency_weights[task - 1])
            total = total + part
        return total

    def weight_data(self, r: Any) -> DataVector:
        """Return ``W r`` (the per-frequency objective weights applied)."""

        dual = self._data_vector(r)
        values = np.array(dual.values, copy=True)
        for weight, frequency in zip(self.frequency_weights, self.frequencies):
            if weight == 1.0:
                continue
            for layout in self.data_space.term_layouts(frequency=frequency):
                values[layout.indices] *= float(weight)
        return DataVector(values, self.data_space)

    def jvp(self, t: Any) -> DataVector:
        """Return ``B @ t`` (extension ``jvp``, memoized per tap vector)."""

        taps = self._tap_vector(t)

        def compute() -> DataVector:
            directory = self._ops_dir()
            jobs = self._family(
                "jvp",
                lambda task: {
                    "extension": {
                        "fields": self.extension_space.fields_fs(),
                        "direction": self._tap_file(taps, directory, task),
                    },
                    "objective_vector": "jvp.json",
                },
            )
            self.view._run_jobs(jobs)
            values = np.zeros(self.data_space.size, dtype=self.data_space.dtype)
            for task, job in enumerate(jobs, start=1):
                values += read_task_objective_vectors(
                    job,
                    self.data_space,
                    state_fingerprint=self.task_fingerprints[task - 1][0],
                ).values
            return DataVector(values, self.data_space)

        return self._memo("jvp", _digest(taps.values), compute)

    def vjp(self, r: Any) -> ExtensionVector:
        """Return ``Re(B^H r)`` on taps (extension ``vjp``, memoized)."""

        dual = self._data_vector(r)

        def compute() -> ExtensionVector:
            directory = self._ops_dir()
            jobs = self._family(
                "vjp",
                lambda task: {
                    "extension": {
                        "fields": self.extension_space.fields_fs(),
                        "covector": "vjp.h5",
                    },
                    "objective_vector": directory / f"task_{task}" / "dual.json",
                },
            )
            for task, job in enumerate(jobs, start=1):
                assert job.objective_vector is not None
                dual.write_objective_vector(
                    job.objective_vector,
                    state_fingerprint=self.task_fingerprints[task - 1][0],
                    term_layout=self.data_space.term_layouts(frequency=job.f_list[0]),
                    n_ranks=1,
                )
            self.view._run_jobs(jobs)
            return self._reduce_taps(jobs, weighted=False)

        return self._memo("vjp", _digest(dual.values), compute)

    def apply_tap_normal(self, t: Any) -> ExtensionVector:
        """Return ``Re(B^H W B) t`` (extension ``normal``, memoized)."""

        taps = self._tap_vector(t)

        def compute() -> ExtensionVector:
            directory = self._ops_dir()
            jobs = self._family(
                "normal",
                lambda task: {
                    "extension": {
                        "fields": self.extension_space.fields_fs(),
                        "direction": self._tap_file(taps, directory, task),
                        "covector": "normal.h5",
                    }
                },
            )
            self.view._run_jobs(jobs)
            return self._reduce_taps(jobs, weighted=True)

        return self._memo("normal", _digest(taps.values), compute)

    def apply_reduced_normal(self, dv: Any) -> ControlVector:
        """Return the reduced GN Schur action on ``dv`` (memoized per direction)."""

        direction = self._control_vector(dv)

        def compute() -> ControlVector:
            directory = self._ops_dir()
            options = dict(self.extension.reduced_normal or {})
            jobs = self._solve_family(
                per_task=lambda task: {
                    "direction": self._direction_file(direction, directory, task),
                    "covector": "reduced_normal.h5",
                    "reduced_normal": options,
                }
            )
            for task, job in enumerate(jobs, start=1):
                report = ExtensionSolveReport.load(job.extension_report_file(1))
                response = report.reduced_normal
                if not report.converged or response is None or not response.converged:
                    raise RuntimeError(
                        f"reduced normal action of task {task} "
                        f"({self.frequencies[task - 1]} Hz) did not converge"
                    )
            if self._solutions is None:
                self._solutions = self._read_solutions(jobs)
            return self._reduce(jobs, weighted=True)

        return self._memo("reduced_normal", _digest(direction.values), compute)


# ---------------------------------------------------------------------------
# extended problem
# ---------------------------------------------------------------------------


class ExtendedProblem:
    """An :class:`ImagingProblem` view with an auxiliary model extension.

    Built by :meth:`ImagingProblem.extend`; shares the problem's state,
    cache, backend and simulation and satisfies the objective protocol of
    :class:`~frequensolve.imaging.workflows.FWI` (``restrict``, ``linearize``,
    ``value``, ``gradient``, ``normal``, ``state``, ``space``, ``vector``).
    """

    def __init__(
        self,
        problem: "ImagingProblem",
        extension: Extension,
        *,
        _cache: Optional[Dict[str, ExtensionLinearization]] = None,
    ) -> None:
        if not isinstance(extension, Extension):
            raise TypeError("extension must be an Extension")
        self._problem = problem
        self.extension = extension
        self._extension_space: Optional[ExtensionSpace] = None
        self._cache: Dict[str, ExtensionLinearization] = (
            {} if _cache is None else _cache
        )
        report = self.capabilities()
        if report["errors"]:
            raise ValueError(
                "extended imaging problem is not supported by Sauce: "
                + "; ".join(report["errors"])
            )

    # -- identity -------------------------------------------------------------

    @property
    def problem(self) -> "ImagingProblem":
        """Return the underlying problem view."""

        return self._problem

    @property
    def name(self) -> str:
        return self._problem.name

    @property
    def simulation(self) -> Any:
        return self._problem.simulation

    @property
    def site(self) -> Any:
        return self._problem.site

    @property
    def backend(self) -> Any:
        return self._problem.backend

    @property
    def workdir(self) -> Path:
        return self._problem.workdir

    @property
    def cache(self) -> Any:
        return self._problem.cache

    @property
    def misfit(self) -> Any:
        return self._problem.misfit

    @property
    def smoothing(self) -> None:
        """Extension actions carry no Sauce smoothing; always ``None``."""

        return None

    @property
    def min_support(self) -> Optional[float]:
        return self._problem.min_support

    @property
    def weights(self) -> Optional[Tuple[float, ...]]:
        return self._problem.weights

    @property
    def frequencies(self) -> List[Any]:
        return self._problem.frequencies

    @property
    def observed_data(self) -> Any:
        return self._problem.observed_data

    @property
    def observed_groups(self) -> List[Any]:
        return self._problem.observed_groups

    @property
    def full_space(self) -> Any:
        return self._problem.full_space

    @property
    def space(self) -> ControlSpace:
        """Return the active control space (material blocks, adopted support)."""

        return self._problem.space

    @property
    def data_space(self) -> DataSpace:
        return self._problem.data_space

    @property
    def extension_space(self) -> ExtensionSpace:
        """Return the tap space (fields resolved on the full control space)."""

        if self._extension_space is None:
            self._extension_space = ExtensionSpace(
                self._problem.full_space, self.extension
            )
        return self._extension_space

    @property
    def state(self) -> Optional[ControlState]:
        return self._problem.state

    @state.setter
    def state(self, value: ControlState) -> None:
        self._problem.state = value

    def identity(self) -> Dict[str, Any]:
        """Return the state-independent identity (problem view plus extension)."""

        return {**self._problem.identity(), "extension": self.extension.to_fs()}

    def __repr__(self) -> str:
        return (
            f"ExtendedProblem(name={self.name!r}, blocks={list(self.space.blocks)}, "
            f"fields={list(self.extension.controls)}, frequencies={self.frequencies})"
        )

    # -- views ----------------------------------------------------------------

    def restrict(self, *args: Any, **kwargs: Any) -> "ExtendedProblem":
        """Return a stage view (see :meth:`ImagingProblem.restrict`) with the extension."""

        return ExtendedProblem(
            self._problem.restrict(*args, **kwargs), self.extension, _cache=self._cache
        )

    def with_misfit(self, misfit: Any = None, *, loss: Any = None) -> "ExtendedProblem":
        return self.restrict(misfit=misfit, loss=loss)

    # -- states and vectors ---------------------------------------------------

    def vector(self, state: Optional[ControlState] = None) -> ControlVector:
        return self._problem.vector(state)

    def state_from(self, vector: Any) -> ControlState:
        return self._problem.state_from(vector)

    def is_authored(self, state: Optional[ControlState] = None) -> bool:
        return self._problem.is_authored(state)

    # -- capabilities ---------------------------------------------------------

    def capabilities(self) -> Dict[str, Any]:
        """Statically validate the extension against Sauce's restrictions.

        Errors: complex (Laplace) frequencies, non-waveform comparisons,
        losses other than L2/Huber/Student-t, source/geometry/reflectivity
        blocks in the active space, reflectivity anywhere in the registry,
        fields that do not borrow a material block, and
        ``Solver/relaxed_assembly`` left on.  Warnings: gradient smoothing
        (not carried by extension jobs) and frozen non-material blocks.
        """

        problem = self._problem
        errors: List[str] = []
        warnings: List[str] = []
        base = problem.capabilities()
        errors.extend(base["errors"])
        warnings.extend(base["warnings"])

        if any(abs(complex(f).imag) > 0.0 for f in problem.frequencies):
            errors.append("the extension requires real frequencies")
        comparisons = set(base["comparisons"])
        if comparisons - {"waveform"}:
            errors.append(
                "the extension requires waveform comparisons "
                f"(misfit uses {sorted(comparisons)})"
            )
        losses = {
            Loss.from_value(term.loss).kind
            for term in problem.misfit.objective_terms(
                [group.name for group in problem.observed_groups]
            )
        }
        if losses - REDUCED_LOSSES:
            errors.append(
                "reduced background actions require L2, Huber or Student-t losses "
                f"(misfit uses {sorted(losses)})"
            )
        active = list(problem.space.resolved_blocks)
        bad = [b.name for b in active if b.kind not in MATERIAL_KINDS]
        if bad:
            errors.append(
                "the extension requires material control blocks; active space has "
                + ", ".join(bad)
            )
        full = list(problem.full_space.resolved_blocks)
        if any(b.kind == "reflectivity" for b in full):
            errors.append("extension and reflectivity are mutually exclusive")
        frozen = [
            b.name
            for b in full
            if b.kind not in MATERIAL_KINDS and b.kind != "reflectivity"
        ]
        if frozen and not bad:
            warnings.append(
                "non-material blocks stay frozen under the extension: "
                + ", ".join(frozen)
            )
        try:
            space = self.extension_space
        except ValueError as exc:
            errors.append(str(exc))
            space = None
        if problem.smoothing is not None:
            warnings.append("gradient smoothing is not applied by extension jobs")
        solver = getattr(problem.simulation, "solver", None)
        extra = getattr(solver, "extra", None)
        relaxed = (
            None if not isinstance(extra, Mapping) else extra.get("relaxed_assembly")
        )
        if relaxed is None:
            warnings.append(
                "Sauce requires Solver/relaxed_assembly=false for extension actions; "
                "set SolverConfig(relaxed_assembly=False) on the simulation"
            )
        elif bool(relaxed):
            errors.append("the extension requires Solver/relaxed_assembly=false")
        return {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
            "blocks": list(problem.space.blocks),
            "fields": list(self.extension.controls),
            "taps": None if space is None else space.size,
            "comparisons": sorted(comparisons),
            "losses": sorted(losses),
            "frequencies": problem.frequencies,
        }

    # -- linearization --------------------------------------------------------

    def _fingerprint(self, state: Optional[ControlState]) -> str:
        return fingerprint(
            **self.identity(), state=None if state is None else state.values
        )

    def _linearize_job(self, control_state: Optional[Path]) -> FWIOperatorJob:
        problem = self._problem
        shared = problem._shared
        return FWIOperatorJob(
            shared.backend.job_name("xlinearize"),
            shared.simulation,
            list(problem.frequencies),
            action="linearize",
            active=list(problem.space.blocks),
            state="state.json",
            objective="report.json",
            control_state=control_state,
            min_support=problem.min_support,
            misfit=problem._misfit_payload,
            extension={
                "fields": self.extension_space.fields_fs(),
                "manifest": "extension.json",
                "covector": "residual.h5",
            },
        )

    def _linearize_state(self, state: Optional[ControlState]) -> ExtensionLinearization:
        problem = self._problem
        shared = problem._shared
        key = self._fingerprint(state)
        cached = self._cache.get(key)
        if cached is not None:
            shared.cache.get(key)
            return cached
        if shared.baseline is None:
            problem._discover_registry()
            if state is None:
                state = shared.state
                key = self._fingerprint(state)
                cached = self._cache.get(key)
                if cached is not None:
                    return cached
        assert state is not None
        problem._sync_simulation(state)
        stage, control_state = problem._stage_state(key, state)
        job = self._linearize_job(control_state)
        problem._run_job(job)

        masks = problem._read_masks(job, problem.space)
        if masks is None:
            masks = problem._fallback_masks(problem.space)
        problem._adopt_masks(masks)
        space = problem.space

        reports = read_report(job)
        manifest_stem = job.extension_manifest_file()
        covector_stem = job.extension_covector_file()
        manifests: List[ExtensionManifest] = []
        weights = _weights(self, len(reports))
        covector = self.extension_space.zeros()
        for task in range(1, len(reports) + 1):
            path = _task_file(manifest_stem, task)
            if not path.is_file():
                raise FileNotFoundError(
                    f"task {task} extension manifest is missing: {path}"
                )
            manifest = ExtensionManifest.load(path)
            manifest.check(self.extension_space, path)
            manifests.append(manifest)
            part_path = _task_file(covector_stem, task)
            if part_path.is_file():
                part = ExtensionVector.from_file(part_path, self.extension_space)
                covector = covector + part * float(weights[task - 1])
        assert shared.manifest is not None
        state_fp = next(
            (r.state_fingerprint for r in reports if r.state_fingerprint), None
        )
        if state_fp is None:
            state_fp = manifests[0].baseline
        entry = LinearizationEntry(
            fingerprint=key,
            job=job,
            directory=stage,
            state_fingerprint=state_fp,
            control_registry_fingerprint=shared.manifest.fingerprint,
            extra={
                "active": list(space.blocks),
                "frequencies": list(job.f_list),
                "extension": self.extension_space.descriptor(),
            },
        )
        linearization = ExtensionLinearization(
            self,
            space=space,
            state=state,
            entry=entry,
            reports=reports,
            manifests=manifests,
            covector=covector,
        )
        evicted = shared.cache.put(entry)
        shared.forget(evicted)
        for old in evicted:
            self._cache.pop(old.fingerprint, None)
        self._cache[key] = linearization
        return linearization

    def linearize(
        self, v: Any = None, *, gradient: bool = False
    ) -> ExtensionLinearization:
        """Save (or reuse) the extension state at ``v``.

        Args:
            v: Point on the active space, a full state, or ``None``.
            gradient: Also run the inner solve with ``model_gradient`` so
                ``.value`` and ``.gradient`` are available without a further
                job (what :class:`~frequensolve.imaging.workflows.FWI` asks).
        """

        problem = self._problem
        state = None if (v is None and problem.state is None) else problem._state_at(v)
        linearization = self._linearize_state(state)
        if gradient:
            linearization._ensure_solved(gradient=True)
        return linearization

    # -- actions --------------------------------------------------------------

    def solve_all(
        self, v: Any = None
    ) -> List[Tuple[ExtensionVector, ExtensionSolveReport]]:
        """Return ``(taps, report)`` of the inner solve per frequency task."""

        return self.linearize(v).solutions

    def solve(self, v: Any = None) -> Tuple[ExtensionVector, ExtensionSolveReport]:
        """Return the solved taps and report of a single-frequency view.

        Each frequency task has its own inner solve; restrict the view to one
        frequency or use :meth:`solve_all` for several.
        """

        solutions = self.solve_all(v)
        if len(solutions) != 1:
            raise ValueError(
                f"solve() needs a single-frequency view ({len(solutions)} tasks); "
                "use solve_all() or restrict(frequencies=[...])"
            )
        return solutions[0]

    def value(self, v: Any = None) -> float:
        """Return the reduced objective at ``v``."""

        return self.linearize(v).value

    def gradient(self, v: Any = None) -> ControlVector:
        """Return the reduced background gradient at ``v``."""

        return self.linearize(v).gradient

    def normal(self, v: Any = None) -> ReducedNormal:
        """Return the reduced Gauss-Newton Schur operator at ``v``."""

        return self.linearize(v).normal

    def jacobian(self, v: Any = None) -> ExtensionJacobian:
        """Return the tap-space Jacobian at ``v``."""

        return self.linearize(v).jacobian

    # -- diagnostics ----------------------------------------------------------

    def dry_run(self, v: Any = None) -> Dict[str, Any]:
        """Describe the extension linearize job at ``v`` without submitting it."""

        problem = self._problem
        state = None if (v is None and problem.state is None) else problem._state_at(v)
        key = self._fingerprint(state)
        _stage, control_state = problem._stage_state(key, state)
        payload = self.backend.dry_run(self._linearize_job(control_state))
        payload["fingerprint"] = key
        payload["registry_discovery"] = problem._shared.baseline is None
        payload["extension"] = self.extension.to_fs(self.extension_space.control_ids)
        payload["taps"] = self.extension_space.size
        return payload

    def clear_cache(self) -> None:
        self._cache.clear()
        self._problem.clear_cache()

    def check(
        self,
        v: Any = None,
        *,
        seed: int = 0,
        steps: Sequence[float] = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3),
        tolerance: float = 1.0e-8,
        taylor: bool = True,
    ) -> Dict[str, Any]:
        """Run adjoint, normal and reduced-normal symmetry tests at ``v``.

        Returns ``adjoint`` (``<B t, r>_Re`` versus ``<t, B^H r>``), ``normal``
        (``tap_normal @ t`` versus ``B^H W B t``), ``reduced_normal``
        (symmetry ``<H a, b>`` versus ``<a, H b>``), optionally ``taylor``
        on ``value`` / ``gradient``, and ``passed``.
        """

        lin = self.linearize(v, gradient=taylor)
        B = lin.jacobian
        t = lin.extension_space.random(seed)
        r = lin.data_space.random(seed + 1)
        adjoint = real_adjoint_test(
            lambda p: np.asarray(B @ p),
            lambda y: np.asarray(B.H @ y),
            t.values,
            r.values,
            relative_tolerance=tolerance,
        )
        n_t = np.asarray(lin.tap_normal @ t)
        bhb_t = np.asarray(B.H @ lin.weight_data(B @ t))
        scale = max(float(np.linalg.norm(bhb_t)), float(np.finfo(np.float64).tiny))
        normal_error = float(np.linalg.norm(n_t - bhb_t) / scale)
        normal = {"relative_error": normal_error, "passed": normal_error <= tolerance}
        a = lin.space.random(seed + 2)
        b = lin.space.random(seed + 3)
        H = lin.normal
        left = float(np.dot(np.asarray(H @ a), b.values))
        right = float(np.dot(a.values, np.asarray(H @ b)))
        denominator = max(abs(left), abs(right), float(np.finfo(np.float64).tiny))
        symmetry_error = abs(left - right) / denominator
        reduced = {
            "relative_error": symmetry_error,
            "passed": symmetry_error <= max(tolerance, 1.0e-6),
        }
        report: Dict[str, Any] = {
            "fingerprint": lin.fingerprint,
            "adjoint": adjoint,
            "normal": normal,
            "reduced_normal": reduced,
        }
        passed = bool(adjoint["passed"] and normal["passed"] and reduced["passed"])
        if taylor:
            base = lin.state
            space = lin.space

            def objective(m: np.ndarray) -> float:
                state = self._problem._state_at(ControlVector(m, space), base=base)
                return self._linearize_state(state).value

            def gradient(m: np.ndarray) -> np.ndarray:
                state = self._problem._state_at(ControlVector(m, space), base=base)
                return self._linearize_state(state).gradient.values

            report["taylor"] = gradient_taylor_test(
                objective, gradient, lin.point.values, a.values, steps=steps
            )
            passed = passed and bool(report["taylor"]["passed"])
        report["passed"] = passed
        return report
