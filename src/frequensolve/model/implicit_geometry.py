"""Authoring objects for Sauce implicit-geometry surfaces.

These classes author, serialize and round-trip the ``fs-implicit-geometry-1``
surface registry entries that a material model lists under ``surfaces``
alongside its ordered graph surfaces. They do not evaluate geometry; Sauce
owns the signed-distance evaluators.

Radial-basis surfaces (``rbf`` and ``rbf_level_set``) are the only kinds whose
coefficients can be optimized. Their optional :class:`ImplicitSurfaceControl`
publishes the authored-order coefficient vector under one stable block ``id``
and configures the representation-owned step limiter.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from frequensolve.util.mixins import ExportContext, merge_extra

__all__ = [
    "IMPLICIT_SURFACE_TYPES",
    "RBF_SURFACE_TYPES",
    "ImplicitSurface",
    "ImplicitSurfaceControl",
    "RBFSurface",
    "implicit_surface_from_fs",
    "is_implicit_surface_payload",
    "split_surface_payloads",
]

#: Surface ``_type`` values accepted by the ``fs-implicit-geometry-1`` schema.
IMPLICIT_SURFACE_TYPES: FrozenSet[str] = frozenset(
    {
        "simple",
        "elevation",
        "plane",
        "sphere",
        "ellipsoid",
        "cone",
        "cylinder",
        "box",
        "rbf",
        "rbf_level_set",
        "transform",
        "abs",
        "offset",
        "capsule",
        "repeat",
        "angular_repeat",
        "union",
        "intersection",
        "difference",
        "smooth_union",
        "smooth_intersection",
        "smooth_difference",
        "nunion",
        "nintersection",
        "ndifference",
        "nsmooth_union",
        "nsmooth_intersection",
        "nsmooth_difference",
    }
)

#: Compact Wendland-C2 expansion kinds that may carry a coefficient control.
RBF_SURFACE_TYPES: FrozenSet[str] = frozenset({"rbf", "rbf_level_set"})

_RBF_KERNEL = "wendland_c2"


def _surface_type(data: Mapping[str, Any]) -> Optional[str]:
    type_name = data.get("_type", data.get("type"))
    if type_name is None:
        return None
    return str(type_name).strip().lower()


def is_implicit_surface_payload(data: Mapping[str, Any]) -> bool:
    """Return whether a ``surfaces`` entry is an implicit-geometry surface.

    Graph surfaces of a layered model carry a ``depth`` and either no
    ``_type`` or the ``Fracture`` marker; every other entry whose ``_type``
    is an implicit-geometry kind is an implicit surface.
    """

    if not isinstance(data, Mapping) or "depth" in data:
        return False
    return _surface_type(data) in IMPLICIT_SURFACE_TYPES


def split_surface_payloads(
    payloads: Optional[Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a ``surfaces`` list into ``(implicit, graph)`` payload lists.

    Relative order is preserved within each list.
    """

    implicit: List[Dict[str, Any]] = []
    graph: List[Dict[str, Any]] = []
    for entry in list(payloads or []):
        if is_implicit_surface_payload(entry):
            implicit.append(entry)
        else:
            graph.append(entry)
    return implicit, graph


def _positive_or_none(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"implicit surface control {name} must be positive")
    return number


@dataclass
class ImplicitSurfaceControl:
    """Optimizable coefficient block of a radial-basis implicit surface.

    Args:
        id: Globally unique block name used for vector ordering and HDF5
            datasets. Must not contain ``/``.
        maximum_displacement: Maximum sampled normal interface displacement
            per accepted optimizer step. Sauce defaults to one quarter of the
            surface ``support_radius``.
        feasibility_band: Distance band around the current zero set used by
            the geometric step limiter. Sauce defaults to ``support_radius``.
    """

    id: str
    maximum_displacement: Optional[float] = None
    feasibility_band: Optional[float] = None

    def __post_init__(self) -> None:
        block_id = str(self.id).strip()
        if not block_id:
            raise ValueError("implicit surface control requires a non-empty id")
        if "/" in block_id or block_id in {".", ".."}:
            raise ValueError("implicit surface control id is not HDF5-safe")
        self.id = block_id
        self.maximum_displacement = _positive_or_none(
            self.maximum_displacement, "maximum_displacement"
        )
        self.feasibility_band = _positive_or_none(
            self.feasibility_band, "feasibility_band"
        )

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the control block using the implicit-geometry contract."""

        payload: Dict[str, Any] = {"id": self.id}
        if self.maximum_displacement is not None:
            payload["maximum_displacement"] = self.maximum_displacement
        if self.feasibility_band is not None:
            payload["feasibility_band"] = self.feasibility_band
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ImplicitSurfaceControl":
        """Deserialize a control block."""

        return cls(
            id=data["id"],
            maximum_displacement=data.get("maximum_displacement"),
            feasibility_band=data.get("feasibility_band"),
        )


@dataclass
class RBFSurface:
    """Compact Wendland-C2 radial-basis implicit surface.

    The field is ``bias + sum_i coefficients[i] psi(|x - centers[i]| /
    support_radius)``; negative values select the inside. ``level_set=True``
    serializes as ``rbf_level_set`` (the kind used by material blends and
    interface controls) and ``False`` as the plain ``rbf`` registry kind.

    Args:
        name: Stable surface name referenced by blends, regions and controls.
        support_radius: Positive compact support radius of the kernel.
        centers: ``(n, d)`` array of kernel centers with ``d`` in ``{2, 3}``.
        coefficients: ``n`` real kernel weights in authored order.
        bias: Constant field offset.
        control: Optional :class:`ImplicitSurfaceControl` (or its mapping)
            that exposes the coefficients to the optimizer.
        level_set: Whether to emit ``rbf_level_set`` rather than ``rbf``.
        extra: Additional serialized fields preserved on round trip.
    """

    name: str
    support_radius: float
    centers: ArrayLike
    coefficients: ArrayLike
    bias: float = 0.0
    control: Optional[Union[ImplicitSurfaceControl, Mapping[str, Any]]] = None
    level_set: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)
    kernel: str = field(default=_RBF_KERNEL, init=False)

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        if not self.name:
            raise ValueError("implicit surface requires a non-empty name")
        self.support_radius = float(self.support_radius)
        if not np.isfinite(self.support_radius) or self.support_radius <= 0.0:
            raise ValueError("rbf surface support_radius must be positive")
        self.bias = float(self.bias)
        if not np.isfinite(self.bias):
            raise ValueError("rbf surface bias must be finite")
        centers = np.asarray(self.centers, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[0] < 1 or centers.shape[1] not in (2, 3):
            raise ValueError("rbf surface centers must be an (n, 2) or (n, 3) array")
        if not np.all(np.isfinite(centers)):
            raise ValueError("rbf surface centers must be finite")
        self.centers = np.array(centers, copy=True)
        self.coefficients = self._coefficient_vector(self.coefficients)
        if isinstance(self.control, Mapping):
            self.control = ImplicitSurfaceControl.from_fs(self.control)
        if self.control is not None and not isinstance(
            self.control, ImplicitSurfaceControl
        ):
            raise TypeError("rbf surface control must be an ImplicitSurfaceControl")
        self.level_set = bool(self.level_set)
        self.extra = dict(self.extra or {})

    def _coefficient_vector(self, values: ArrayLike) -> np.ndarray:
        array = np.asarray(values)
        if np.iscomplexobj(array):
            raise ValueError("rbf surface coefficients must be real-valued")
        array = np.asarray(array, dtype=np.float64)
        if array.ndim != 1 or array.size != self.centers.shape[0]:
            raise ValueError("rbf surface coefficients must have one value per center")
        if not np.all(np.isfinite(array)):
            raise ValueError("rbf surface coefficients must be finite")
        return np.array(array, copy=True)

    @property
    def type(self) -> str:
        """Return the serialized ``_type`` of this surface."""

        return "rbf_level_set" if self.level_set else "rbf"

    @property
    def ndim(self) -> int:
        """Return the spatial dimension of the kernel centers."""

        return int(np.asarray(self.centers).shape[1])

    @property
    def size(self) -> int:
        """Return the number of authored-order coefficients."""

        return int(np.asarray(self.coefficients).size)

    @property
    def coordinates(self) -> np.ndarray:
        """Return the ``(size, ndim)`` kernel centers, one row per coefficient."""

        return np.array(self.centers, copy=True)

    def with_coefficients(self, coefficients: ArrayLike) -> "RBFSurface":
        """Return a copy with a replacement coefficient vector."""

        return replace(
            self,
            coefficients=coefficients,
            centers=np.array(self.centers, copy=True),
            control=copy.deepcopy(self.control),
            extra=copy.deepcopy(self.extra),
        )

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize this surface using the implicit-geometry contract."""

        payload: Dict[str, Any] = {
            "_type": self.type,
            "name": self.name,
            "kernel": self.kernel,
            "support_radius": self.support_radius,
            "bias": self.bias,
            "centers": np.asarray(self.centers).tolist(),
            "coefficients": np.asarray(self.coefficients).tolist(),
        }
        if self.control is not None:
            payload["control"] = self.control.to_fs()
        return merge_extra(payload, self.extra, "RBFSurface")

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "RBFSurface":
        """Deserialize an ``rbf`` or ``rbf_level_set`` surface."""

        payload = copy.deepcopy(dict(data))
        type_name = _surface_type(payload)
        payload.pop("_type", None)
        payload.pop("type", None)
        if type_name not in RBF_SURFACE_TYPES:
            raise ValueError(f"expected an rbf surface payload, got {type_name!r}")
        kernel = payload.pop("kernel", _RBF_KERNEL)
        if str(kernel) != _RBF_KERNEL:
            raise ValueError(f"unsupported rbf kernel {kernel!r}")
        return cls(
            name=payload.pop("name"),
            support_radius=payload.pop("support_radius"),
            centers=payload.pop("centers"),
            coefficients=payload.pop("coefficients"),
            bias=payload.pop("bias", 0.0),
            control=payload.pop("control", None),
            level_set=type_name == "rbf_level_set",
            extra=payload,
        )


@dataclass
class ImplicitSurface:
    """Generic implicit-geometry surface of any registry kind.

    This is a schema-level container for primitives, transforms and Boolean
    operators (``plane``, ``sphere``, ``union`` and so on). Kind-specific
    fields are kept verbatim in ``fields`` and validated by the Sauce schema,
    not by FrequenSolve. Use :class:`RBFSurface` for controllable surfaces.

    Args:
        name: Stable surface name referenced by other surfaces and regions.
        type: Registry ``_type`` of the surface.
        fields: Kind-specific serialized fields.
        **kwargs: Additional kind-specific fields merged into ``fields``.
    """

    name: str
    type: str
    fields: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        type: str,
        fields: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.name = str(name).strip()
        if not self.name:
            raise ValueError("implicit surface requires a non-empty name")
        self.type = str(type).strip().lower()
        if self.type not in IMPLICIT_SURFACE_TYPES:
            raise ValueError(f"unsupported implicit surface type {self.type!r}")
        merged = copy.deepcopy(dict(fields or {}))
        merged.update(copy.deepcopy(kwargs))
        for key in ("_type", "type", "name"):
            merged.pop(key, None)
        self.fields = merged

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize this surface using the implicit-geometry contract."""

        payload: Dict[str, Any] = {"_type": self.type, "name": self.name}
        return merge_extra(payload, self.fields, "ImplicitSurface")

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ImplicitSurface":
        """Deserialize a generic implicit surface."""

        payload = copy.deepcopy(dict(data))
        type_name = _surface_type(payload)
        if type_name is None:
            raise ValueError("implicit surface payload requires _type")
        payload.pop("_type", None)
        payload.pop("type", None)
        return cls(name=payload.pop("name"), type=type_name, fields=payload)


ImplicitSurfaceLike = Union[RBFSurface, ImplicitSurface]


def implicit_surface_from_fs(data: Mapping[str, Any]) -> ImplicitSurfaceLike:
    """Deserialize one implicit ``surfaces`` entry into its authoring class."""

    if _surface_type(data) in RBF_SURFACE_TYPES:
        return RBFSurface.from_fs(data)
    return ImplicitSurface.from_fs(data)
