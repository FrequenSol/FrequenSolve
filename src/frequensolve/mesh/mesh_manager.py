"""Mesh manager and mesh adaptivity configuration objects."""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from frequensolve.units import value_and_units_to_fs
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    merge_extra,
    warn_deprecated_path_api,
)

from .mesh_generators import BaseMeshGenerator

__all__ = [
    "MeshParallelism",
    "DistanceGrading",
    "SurfaceGrading",
    "MeshAdaptor",
    "MeshManager",
]


@dataclass
class MeshParallelism:
    """Parallel mesh distribution settings.

    Args:
        distribute: Whether the solver should distribute mesh parts across MPI
            ranks.
        ranks_per_part: Optional number of MPI ranks assigned to each mesh
            partition.
        partitioner: Optional solver partitioner name.
    """

    distribute: Optional[bool] = True
    ranks_per_part: Optional[int] = None
    partitioner: Optional[str] = None

    def to_fs(self, ctx=None) -> Dict:
        """Serialize mesh parallelism settings.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible parallelism payload.
        """

        return {
            "distribute": self.distribute,
            **({"ranks_per_part": self.ranks_per_part} if self.ranks_per_part else {}),
            **({"partitioner": self.partitioner} if self.partitioner else {}),
        }

    @classmethod
    def from_fs(cls, data: Dict) -> "MeshParallelism":
        """Deserialize mesh parallelism settings.

        Args:
            data: Serialized parallelism mapping.

        Returns:
            ``MeshParallelism`` instance.
        """

        return cls(
            distribute=data["distribute"],
            ranks_per_part=data.get("ranks_per_part"),
            partitioner=data.get("partitioner"),
        )


_GRADE_MODES = {"none", "inside", "outside", "band", "abs_band"}
GradingValue = Union[float, Mapping[str, float]]


def _pop_alias(
    payload: Dict[str, Any],
    primary: str,
    alias: str,
    default: Any = None,
) -> Any:
    if primary in payload:
        return payload.pop(primary)
    return payload.pop(alias, default)


def _resolve_elems_per_wave(
    elems_per_wave: Any = None,
    *,
    epw: Any = None,
    min_epw: Any = None,
) -> Any:
    values = [
        (name, value)
        for name, value in (
            ("elems_per_wave", elems_per_wave),
            ("epw", epw),
            ("min_epw", min_epw),
        )
        if value is not None
    ]
    if not values:
        raise TypeError("elems_per_wave is required")
    if len(values) > 1:
        names = ", ".join(name for name, _ in values)
        raise ValueError(f"Specify only one of {names}")
    return values[0][1]


def _pop_elems_per_wave(payload: Dict[str, Any]) -> Any:
    return _resolve_elems_per_wave(
        payload.pop("elems_per_wave", None),
        epw=payload.pop("epw", None),
        min_epw=payload.pop("min_epw", None),
    )


def _resolve_f_low(
    f_low: Optional[float] = None,
    *,
    f_adapt: Optional[float] = None,
) -> Optional[float]:
    values = [
        (name, value)
        for name, value in (
            ("f_low", f_low),
            ("f_adapt", f_adapt),
        )
        if value is not None
    ]
    if not values:
        return None
    if len(values) > 1:
        names = ", ".join(name for name, _ in values)
        raise ValueError(f"Specify only one of {names}")
    return values[0][1]


def _pop_f_low(payload: Dict[str, Any]) -> Optional[float]:
    return _resolve_f_low(
        payload.pop("f_low", None),
        f_adapt=payload.pop("f_adapt", None),
    )


def _normalize_grade_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in _GRADE_MODES:
        choices = ", ".join(sorted(_GRADE_MODES))
        raise ValueError(
            f"Unknown surface grading mode '{mode}'. Expected one of: {choices}"
        )
    return mode


def _validate_grading_power(power: GradingValue) -> GradingValue:
    if isinstance(power, Mapping):
        out = {}
        for axis, value in power.items():
            value = float(value)
            if value <= 0.0:
                raise ValueError("Grading power must be positive")
            out[str(axis)] = value
        return out

    power = float(power)
    if power <= 0.0:
        raise ValueError("Grading power must be positive")
    return power


def _normalize_optional_grading_value(
    value: Optional[GradingValue],
) -> Optional[GradingValue]:
    if isinstance(value, Mapping):
        return {str(axis): float(axis_value) for axis, axis_value in value.items()}
    return value


def _is_default_grading_power(power: GradingValue) -> bool:
    if isinstance(power, Mapping):
        return all(float(value) == 1.0 for value in power.values())
    return float(power) == 1.0


@dataclass
class DistanceGrading(ExtraFieldsMixin):
    """Distance-based source/receiver mesh grading.

    This maps to the fast solver's ``kd_grade_t`` JSON contract. ``d0`` is the
    distance where the full factor is applied and ``d1`` is the distance where
    the factor returns to one. ``power`` curves the transition between those
    distances; ``1`` is linear. ``factor`` and ``power`` may be scalars or
    dictionaries keyed by the global coordinate-system axis names.

    Args:
        d1: Outer distance where the grading factor returns to one.
        factor: Optional grading factor, either scalar or keyed by axis.
        power: Transition power, either scalar or keyed by axis. Values must be
            positive.
        d0: Inner distance where the full grading factor is applied.
        extra: Additional solver-facing grading fields.

    Raises:
        ValueError: If ``power`` is non-positive.
    """

    d1: float
    factor: Optional[GradingValue] = None
    power: GradingValue = 1.0
    d0: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.factor = _normalize_optional_grading_value(self.factor)
        self.power = _validate_grading_power(self.power)

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize the source/receiver grading rule.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible distance grading payload.
        """

        payload = {
            "d0": value_and_units_to_fs(self.d0),
            "d1": value_and_units_to_fs(self.d1),
            **({"factor": self.factor} if self.factor is not None else {}),
            **(
                {"power": self.power}
                if not _is_default_grading_power(self.power)
                else {}
            ),
        }
        return merge_extra(payload, self.extra, "DistanceGrading")

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "DistanceGrading":
        """Deserialize a distance grading rule.

        Args:
            data: Serialized distance grading mapping.

        Returns:
            ``DistanceGrading`` instance.
        """

        payload = copy.deepcopy(dict(data))
        return cls(
            d0=payload.pop("d0", 0.0),
            d1=payload.pop("d1"),
            factor=payload.pop("factor", None),
            power=payload.pop("power", 1.0),
            extra=payload,
        )


def _coerce_distance_grading(
    value: Optional[Union[DistanceGrading, Mapping[str, Any]]],
) -> Optional[DistanceGrading]:
    if value is None:
        return None
    if isinstance(value, DistanceGrading):
        return value
    if isinstance(value, Mapping):
        return DistanceGrading.from_fs(value)
    raise TypeError(f"Expected DistanceGrading or mapping, got {type(value).__name__}")


@dataclass
class SurfaceGrading(ExtraFieldsMixin):
    """Geometric mesh grading around a named implicit surface.

    The exported JSON follows the fast solver's ``grading_fields_m`` contract.
    ``d0`` is the inner distance where the strongest factor is applied and
    ``d1`` is the outer distance where the factor returns to ``factor_min``.
    ``power`` curves the transition between those distances; ``1`` is linear.
    ``factor``, ``factor_max``, ``factor_min``, and ``power`` may be scalars or
    dictionaries keyed by the global coordinate-system axis names.

    Args:
        surface: Name of the implicit model surface to grade around.
        d1: Outer distance where the grading returns to ``factor_min``.
        factor: Optional legacy/simple grading factor.
        power: Transition power, either scalar or keyed by axis. Values must be
            positive.
        d0: Inner distance where strongest grading is applied.
        mode: Grading mode. Supported values are ``"none"``, ``"inside"``,
            ``"outside"``, ``"band"``, and ``"abs_band"``.
        factor_max: Optional maximum grading factor.
        factor_min: Optional minimum grading factor.
        phi_scale: Optional scaling applied to the implicit surface field.
        extra: Additional solver-facing grading fields.

    Raises:
        ValueError: If ``surface`` is empty, ``mode`` is unsupported, or
            ``power`` is non-positive.
    """

    surface: str
    d1: float
    factor: Optional[GradingValue] = None
    power: GradingValue = 1.0
    d0: float = 0.0
    mode: str = "abs_band"
    factor_max: Optional[GradingValue] = None
    factor_min: Optional[GradingValue] = None
    phi_scale: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.mode = _normalize_grade_mode(self.mode)
        self.factor = _normalize_optional_grading_value(self.factor)
        self.factor_max = _normalize_optional_grading_value(self.factor_max)
        self.factor_min = _normalize_optional_grading_value(self.factor_min)
        self.power = _validate_grading_power(self.power)
        if not self.surface:
            raise ValueError("SurfaceGrading requires a non-empty surface name")

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        """Serialize the surface grading rule.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible surface grading payload.
        """

        payload = {
            "surface": self.surface,
            "mode": self.mode,
            "d0": value_and_units_to_fs(self.d0),
            "d1": value_and_units_to_fs(self.d1),
            **({"factor": self.factor} if self.factor is not None else {}),
            **({"factor_max": self.factor_max} if self.factor_max is not None else {}),
            **({"factor_min": self.factor_min} if self.factor_min is not None else {}),
            **(
                {"power": self.power}
                if not _is_default_grading_power(self.power)
                else {}
            ),
            **({"phi_scale": self.phi_scale} if self.phi_scale is not None else {}),
        }
        return merge_extra(payload, self.extra, "SurfaceGrading")

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SurfaceGrading":
        """Deserialize a surface grading rule.

        Args:
            data: Serialized surface grading mapping.

        Returns:
            ``SurfaceGrading`` instance.
        """

        payload = copy.deepcopy(dict(data))
        return cls(
            surface=payload.pop("surface"),
            mode=payload.pop("mode", "abs_band"),
            d0=payload.pop("d0", 0.0),
            d1=payload.pop("d1"),
            factor=payload.pop("factor", None),
            power=payload.pop("power", 1.0),
            factor_max=payload.pop("factor_max", None),
            factor_min=payload.pop("factor_min", None),
            phi_scale=payload.pop("phi_scale", None),
            extra=payload,
        )


def _coerce_surface_grading(
    value: Union[SurfaceGrading, Mapping[str, Any]],
) -> SurfaceGrading:
    if isinstance(value, SurfaceGrading):
        return value
    if isinstance(value, Mapping):
        return SurfaceGrading.from_fs(value)
    raise TypeError(f"Expected SurfaceGrading or mapping, got {type(value).__name__}")


def _coerce_surface_gradings(
    gradings: Optional[
        Union[
            Mapping[str, Union[SurfaceGrading, Mapping[str, Any]]],
            Iterable[Union[SurfaceGrading, Mapping[str, Any]]],
        ]
    ],
) -> List[SurfaceGrading]:
    if gradings is None:
        return []
    if isinstance(gradings, Mapping):
        out = []
        for surface, grading in gradings.items():
            if isinstance(grading, SurfaceGrading):
                item = copy.deepcopy(grading)
                if item.surface != surface:
                    raise ValueError(
                        "Surface grading mapping key does not match grading.surface "
                        f"({surface!r} != {item.surface!r})"
                    )
            else:
                payload = copy.deepcopy(dict(grading))
                payload.setdefault("surface", surface)
                item = SurfaceGrading.from_fs(payload)
            out.append(item)
        return out
    return [_coerce_surface_grading(item) for item in gradings]


@dataclass(init=False)
class MeshAdaptor(ExtraFieldsMixin):
    """Mesh adaptivity and grading options.

    Args:
        elems_per_wave: Target minimum elements per wavelength. May be a scalar
            or axis-keyed mapping.
        epw: Alias for ``elems_per_wave``.
        min_epw: Alias for ``elems_per_wave``.
        order: Element order used during mesh adaptivity.
        jump_tolerance: Relative material-property jump threshold used for
            interface refinement.
        jump_factor: Multiplicative adaptation factor on jump elements.
        smooth_refs: Whether to add smoothing refinements around constrained
            degrees of freedom.
        f_low: Low frequency used for adaptation.
        f_high: High frequency used for adaptation.
        f_adapt: Alias for ``f_low``.
        adapt_order: Whether to adapt element order.
        source_grading: Optional distance grading around sources.
        receiver_grading: Optional distance grading around receivers.
        surface_gradings: Optional surface grading rules.
        extra: Additional solver-facing adaptivity fields.
        **kwargs: Additional solver-facing adaptivity fields.

    Raises:
        TypeError: If no elements-per-wave argument is supplied.
        ValueError: If multiple elements-per-wave aliases or low-frequency
            aliases are supplied.
    """

    elems_per_wave: Union[float, Dict[str, float]]
    order: Union[int, Dict[str, int]] = 3
    jump_tolerance: Optional[float] = None  # 0.2
    jump_factor: Optional[float] = None  # 1.0
    smooth_refs: Optional[bool] = None  # False
    f_low: Optional[float] = None
    f_high: Optional[float] = None
    adapt_order: bool = False
    source_grading: Optional[DistanceGrading] = None
    receiver_grading: Optional[DistanceGrading] = None
    surface_gradings: List[SurfaceGrading] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)

    def __init__(
        self,
        elems_per_wave: Optional[Union[float, Dict[str, float]]] = None,
        *,
        epw: Optional[Union[float, Dict[str, float]]] = None,
        min_epw: Optional[Union[float, Dict[str, float]]] = None,
        order: Union[int, Dict[str, int]] = 3,
        jump_tolerance: Optional[float] = None,
        jump_factor: Optional[float] = None,
        smooth_refs: Optional[bool] = None,
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        f_adapt: Optional[float] = None,
        adapt_order: bool = False,
        source_grading: Optional[Union[DistanceGrading, Mapping[str, Any]]] = None,
        receiver_grading: Optional[Union[DistanceGrading, Mapping[str, Any]]] = None,
        surface_gradings: Optional[
            Union[
                Mapping[str, Union[SurfaceGrading, Mapping[str, Any]]],
                Iterable[Union[SurfaceGrading, Mapping[str, Any]]],
            ]
        ] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.elems_per_wave = _resolve_elems_per_wave(
            elems_per_wave,
            epw=epw,
            min_epw=min_epw,
        )
        self.order = order
        self.jump_tolerance = jump_tolerance
        self.jump_factor = jump_factor
        self.smooth_refs = smooth_refs
        self.f_low = _resolve_f_low(f_low, f_adapt=f_adapt)
        self.f_high = f_high
        self.adapt_order = adapt_order
        self.source_grading = source_grading
        self.receiver_grading = receiver_grading
        self.surface_gradings = surface_gradings or []
        self.extra = dict(extra or {})
        self.extra.update(kwargs)
        self.__post_init__()

    def __post_init__(self):
        self.source_grading = _coerce_distance_grading(self.source_grading)
        self.receiver_grading = _coerce_distance_grading(self.receiver_grading)
        self.surface_gradings = _coerce_surface_gradings(self.surface_gradings)

    @property
    def epw(self) -> Union[float, Dict[str, float]]:
        """Alias for ``elems_per_wave``."""

        return self.elems_per_wave

    @epw.setter
    def epw(self, value: Union[float, Dict[str, float]]) -> None:
        """Set ``elems_per_wave`` through the ``epw`` alias."""

        self.elems_per_wave = value

    @property
    def min_epw(self) -> Union[float, Dict[str, float]]:
        """Alias for ``elems_per_wave`` used by older examples."""

        return self.elems_per_wave

    @min_epw.setter
    def min_epw(self, value: Union[float, Dict[str, float]]) -> None:
        """Set ``elems_per_wave`` through the ``min_epw`` alias."""

        self.elems_per_wave = value

    @property
    def f_adapt(self) -> Optional[float]:
        """Alias for the low adaptation frequency ``f_low``."""

        return self.f_low

    @f_adapt.setter
    def f_adapt(self, value: Optional[float]) -> None:
        """Set ``f_low`` through the ``f_adapt`` alias."""

        self.f_low = value

    def to_fs(self, ctx=None) -> Dict:
        """Serialize mesh adaptivity settings for solver input.

        Args:
            ctx: Optional export context forwarded to nested grading objects.

        Returns:
            JSON-compatible adaptivity payload.
        """

        payload = {
            "elems_per_wave": self.elems_per_wave,
            "order": self.order,
            **({"jump_tolerance": self.jump_tolerance} if self.jump_tolerance else {}),
            **({"jump_factor": self.jump_factor} if self.jump_factor else {}),
            **({"smooth_refs": self.smooth_refs} if self.smooth_refs else {}),
            **({"f_low": self.f_low} if self.f_low is not None else {}),
            **({"f_high": self.f_high} if self.f_high is not None else {}),
            **({"adapt_order": self.adapt_order} if self.adapt_order else {}),
            **(
                {"src_grading": self.source_grading.to_fs(ctx)}
                if self.source_grading is not None
                else {}
            ),
            **(
                {"rcv_grading": self.receiver_grading.to_fs(ctx)}
                if self.receiver_grading is not None
                else {}
            ),
            **(
                {
                    "surface_gradings": [
                        grading.to_fs(ctx) for grading in self.surface_gradings
                    ]
                }
                if self.surface_gradings
                else {}
            ),
        }
        return merge_extra(payload, self.extra, "MeshAdaptor")

    @classmethod
    def from_fs(cls, data: Dict) -> "MeshAdaptor":
        """Deserialize mesh adaptivity settings.

        Args:
            data: Serialized adaptivity mapping.

        Returns:
            ``MeshAdaptor`` instance.
        """

        data = copy.deepcopy(data)
        return cls(
            elems_per_wave=_pop_elems_per_wave(data),
            order=data.pop("order", 3),
            jump_tolerance=data.pop("jump_tolerance", None),
            jump_factor=data.pop("jump_factor", None),
            smooth_refs=data.pop("smooth_refs", None),
            f_low=_pop_f_low(data),
            f_high=data.pop("f_high", None),
            adapt_order=data.pop("adapt_order", False),
            source_grading=_pop_alias(data, "source_grading", "src_grading"),
            receiver_grading=_pop_alias(data, "receiver_grading", "rcv_grading"),
            surface_gradings=data.pop("surface_gradings", None),
            extra=data,
        )

    def set_source_grading(
        self,
        d1: float,
        factor: Optional[GradingValue] = None,
        power: GradingValue = 1.0,
        d0: float = 0.0,
        **kwargs,
    ) -> DistanceGrading:
        """Configure distance grading around sources.

        Args:
            d1: Outer grading distance.
            factor: Optional grading factor.
            power: Transition power.
            d0: Inner grading distance.
            **kwargs: Additional solver-facing grading fields.

        Returns:
            The stored ``DistanceGrading`` instance.
        """

        self.source_grading = DistanceGrading(
            d1=d1,
            factor=factor,
            power=power,
            d0=d0,
            **kwargs,
        )
        return self.source_grading

    def set_receiver_grading(
        self,
        d1: float,
        factor: Optional[GradingValue] = None,
        power: GradingValue = 1.0,
        d0: float = 0.0,
        **kwargs,
    ) -> DistanceGrading:
        """Configure distance grading around receivers.

        Args:
            d1: Outer grading distance.
            factor: Optional grading factor.
            power: Transition power.
            d0: Inner grading distance.
            **kwargs: Additional solver-facing grading fields.

        Returns:
            The stored ``DistanceGrading`` instance.
        """

        self.receiver_grading = DistanceGrading(
            d1=d1,
            factor=factor,
            power=power,
            d0=d0,
            **kwargs,
        )
        return self.receiver_grading

    def add_surface_grading(
        self,
        surface: str,
        d1: float,
        factor: Optional[GradingValue] = None,
        power: GradingValue = 1.0,
        d0: float = 0.0,
        mode: str = "abs_band",
        **kwargs,
    ) -> SurfaceGrading:
        """Add a surface-based grading rule.

        Args:
            surface: Name of the implicit surface.
            d1: Outer grading distance.
            factor: Optional grading factor.
            power: Transition power.
            d0: Inner grading distance.
            mode: Surface grading mode.
            **kwargs: Additional solver-facing grading fields.

        Returns:
            Newly added ``SurfaceGrading`` instance.
        """

        grading = SurfaceGrading(
            surface=surface,
            d1=d1,
            factor=factor,
            power=power,
            d0=d0,
            mode=mode,
            **kwargs,
        )
        self.surface_gradings.append(grading)
        return grading


@dataclass
class MeshManager:
    """Mesh source, parallelism, and adaptivity for a simulation.

    Args:
        mesh: Mesh generator configuration. If omitted, ``file`` and ``format``
            must identify an existing mesh file.
        file: Mesh file path, relative to the project when possible.
        format: Mesh file format understood by the solver.
        parallel: Optional mesh distribution settings.
        adapt: Optional mesh adaptivity settings.
    """

    mesh: Optional[BaseMeshGenerator] = None
    file: Optional[str] = None
    format: Optional[str] = None
    parallel: Optional[MeshParallelism] = None
    adapt: Optional[MeshAdaptor] = None
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def set_adapt(
        self,
        elems_per_wave: Optional[Union[float, Dict[str, float]]] = None,
        *,
        epw: Optional[Union[float, Dict[str, float]]] = None,
        min_epw: Optional[Union[float, Dict[str, float]]] = None,
        order: Union[int, Dict[str, int]] = 3,
        jump_tolerance: Optional[float] = None,
        jump_factor: Optional[float] = None,
        smooth_refs: Optional[bool] = None,
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        f_adapt: Optional[float] = None,
        adapt_order: Optional[bool] = False,
        source_grading: Optional[Union[DistanceGrading, Mapping[str, Any]]] = None,
        receiver_grading: Optional[Union[DistanceGrading, Mapping[str, Any]]] = None,
        surface_gradings: Optional[
            Union[
                Mapping[str, Union[SurfaceGrading, Mapping[str, Any]]],
                Iterable[Union[SurfaceGrading, Mapping[str, Any]]],
            ]
        ] = None,
        **kwargs,
    ) -> None:
        """Replace mesh adaptivity settings.

        Args:
            elems_per_wave: Target minimum elements per wavelength.
            epw: Alias for ``elems_per_wave``.
            min_epw: Alias for ``elems_per_wave``.
            order: Element order used during mesh adaptivity.
            jump_tolerance: Relative material-property jump threshold used for
                interface refinement.
            jump_factor: Multiplicative adaptation factor on jump elements.
            smooth_refs: Whether to add smoothing refinements around constrained
                degrees of freedom.
            f_low: Low frequency used for adaptation.
            f_high: High frequency used for adaptation.
            f_adapt: Alias for ``f_low``.
            adapt_order: Whether to adapt element order.
            source_grading: Optional distance grading around sources.
            receiver_grading: Optional distance grading around receivers.
            surface_gradings: Optional surface grading rules.
            **kwargs: Additional solver-facing adaptivity fields.

        Raises:
            TypeError: If no elements-per-wave value or alias is supplied.
            ValueError: If conflicting aliases are supplied.
        """
        self.adapt = MeshAdaptor(
            elems_per_wave=elems_per_wave,
            epw=epw,
            min_epw=min_epw,
            order=order,
            jump_tolerance=jump_tolerance,
            jump_factor=jump_factor,
            smooth_refs=smooth_refs,
            f_low=f_low,
            f_high=f_high,
            f_adapt=f_adapt,
            adapt_order=adapt_order,
            source_grading=source_grading,
            receiver_grading=receiver_grading,
            surface_gradings=surface_gradings,
            extra=kwargs,
        )

    def set_source_grading(
        self,
        d1: float,
        factor: Optional[GradingValue] = None,
        power: GradingValue = 1.0,
        d0: float = 0.0,
        **kwargs,
    ) -> DistanceGrading:
        """Configure distance grading around sources.

        Args:
            d1: Outer grading distance.
            factor: Optional grading factor.
            power: Transition power.
            d0: Inner grading distance.
            **kwargs: Additional solver-facing grading fields.

        Returns:
            The stored ``DistanceGrading`` instance.
        """

        if self.adapt is None:
            self.set_adapt(elems_per_wave=2.0, adapt_sources=1)
        return self.adapt.set_source_grading(
            d1=d1,
            factor=factor,
            power=power,
            d0=d0,
            **kwargs,
        )

    def set_receiver_grading(
        self,
        d1: float,
        factor: Optional[GradingValue] = None,
        power: GradingValue = 1.0,
        d0: float = 0.0,
        **kwargs,
    ) -> DistanceGrading:
        """Configure distance grading around receivers.

        Args:
            d1: Outer grading distance.
            factor: Optional grading factor.
            power: Transition power.
            d0: Inner grading distance.
            **kwargs: Additional solver-facing grading fields.

        Returns:
            The stored ``DistanceGrading`` instance.
        """

        if self.adapt is None:
            self.set_adapt(elems_per_wave=2.0, adapt_sources=1)
        return self.adapt.set_receiver_grading(
            d1=d1,
            factor=factor,
            power=power,
            d0=d0,
            **kwargs,
        )

    def add_surface_grading(
        self,
        surface: str,
        d1: float,
        factor: Optional[GradingValue] = None,
        power: GradingValue = 1.0,
        d0: float = 0.0,
        mode: str = "abs_band",
        **kwargs,
    ) -> SurfaceGrading:
        """Add a surface-based grading rule to the mesh adaptor.

        Args:
            surface: Name of the implicit surface.
            d1: Outer grading distance.
            factor: Optional grading factor.
            power: Transition power.
            d0: Inner grading distance.
            mode: Surface grading mode.
            **kwargs: Additional solver-facing grading fields.

        Returns:
            Newly added ``SurfaceGrading`` instance.
        """

        if self.adapt is None:
            self.set_adapt(elems_per_wave=2.0, adapt_sources=1)
        return self.adapt.add_surface_grading(
            surface=surface,
            d1=d1,
            factor=factor,
            power=power,
            d0=d0,
            mode=mode,
            **kwargs,
        )

    def set_parallel(
        self,
        distribute: bool,
        ranks_per_part: Optional[int] = None,
        partitioner: Optional[str] = None,
    ) -> None:
        """Set mesh parallel distribution options.

        Args:
            distribute: Whether the solver should distribute mesh parts.
            ranks_per_part: Optional MPI rank count per mesh part.
            partitioner: Optional partitioner name.
        """
        self.parallel = MeshParallelism(
            distribute=distribute,
            ranks_per_part=ranks_per_part,
            partitioner=partitioner,
        )

    @classmethod
    def from_fs(cls, data: Dict) -> "MeshManager":
        """Deserialize mesh configuration from solver JSON.

        Args:
            data: Serialized mesh block.

        Returns:
            ``MeshManager`` instance.
        """

        data = copy.deepcopy(data)
        manager = cls()

        # From file
        file = data.get("file")
        format = data.get("format")
        if file is not None and format is not None:
            manager.file = file
            manager.format = format

        # From generator
        if "generator" in data:
            manager.mesh = BaseMeshGenerator.from_fs(data["generator"])

        # Parallel
        if "parallel" in data:
            p = data["parallel"]
            manager.set_parallel(
                distribute=p["distribute"],
                ranks_per_part=p.get("ranks_per_part"),
                partitioner=p.get("partitioner"),
            )

        if "adapt" in data:
            a = copy.deepcopy(data["adapt"])
            manager.set_adapt(
                elems_per_wave=_pop_elems_per_wave(a),
                order=a.pop("order", 3),
                adapt_sources=a.pop("adapt_sources", 0),
                adapt_receivers=a.pop("adapt_receivers", 0),
                jump_tolerance=a.pop("jump_tolerance", None),
                jump_factor=a.pop("jump_factor", None),
                smooth_refs=a.pop("smooth_refs", False),
                f_low=_pop_f_low(a),
                f_high=a.pop("f_high", None),
                adapt_order=a.pop("adapt_order", False),
                source_grading=_pop_alias(a, "source_grading", "src_grading"),
                receiver_grading=_pop_alias(a, "receiver_grading", "rcv_grading"),
                surface_gradings=a.pop("surface_gradings", None),
                **a,
            )

        return manager

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Serialize mesh configuration for solver input.

        Existing mesh files are copied into the export path when necessary so
        the solver JSON can reference a project-relative path.

        Args:
            ctx: Optional export context containing project and output paths.

        Returns:
            JSON-compatible mesh block.

        Raises:
            AssertionError: If neither a mesh generator nor a mesh file/format
                pair has been configured.
        """

        ctx = ctx or ExportContext(self._proj_path, self._rel_path)
        project_path = ctx.project_path or self._proj_path
        export_path = ctx.path
        if project_path is not None:
            project_path = Path(project_path).expanduser().resolve()
        if self.adapt is None:
            self.set_adapt(elems_per_wave=2.0, adapt_sources=1)

        mesh_dict = {
            "adapt": self.adapt.to_fs(ctx),
        }

        if self.parallel:
            mesh_dict["parallel"] = self.parallel.to_fs(ctx)

        # Mesh determined by file
        if self.mesh is None:
            assert self.file is not None and self.format is not None, (
                "if a mesh or mesh generator has not been provided, "
                "'file' and 'format' must be provided"
            )
            mesh_file = Path(self.file)
            if mesh_file.is_absolute():
                source = mesh_file
            else:
                project_source = (
                    project_path / mesh_file if project_path is not None else None
                )
                source = (
                    project_source
                    if project_source is not None and project_source.exists()
                    else mesh_file
                )

            if (
                project_path is not None
                and source.is_absolute()
                and source.is_relative_to(project_path)
            ):
                stored_file = source.relative_to(project_path)
            elif export_path is not None and project_path is not None:
                dest = (export_path / mesh_file.name).resolve()
                dest.parent.mkdir(parents=True, exist_ok=True)
                copy2(source, dest)
                stored_file = dest.relative_to(project_path)
            else:
                stored_file = mesh_file

            self.file = stored_file
            mesh_dict["file"] = stored_file

            mesh_dict["format"] = self.format

        # Mesh determined by generator (defined in backend)
        elif isinstance(self.mesh, BaseMeshGenerator):
            mesh_dict["generator"] = self.mesh.to_fs(ctx)

        return mesh_dict

    def _set_path(self, proj_path: Path, rel_path: Path):
        warn_deprecated_path_api(f"{self.__class__.__name__}._set_path")
        self._proj_path = Path(proj_path).expanduser().resolve()
        self._rel_path = Path(rel_path)

    @property
    def _path(self) -> Path:
        warn_deprecated_path_api(f"{self.__class__.__name__}._path")
        return self._proj_path / self._rel_path
