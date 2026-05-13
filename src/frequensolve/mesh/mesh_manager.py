"""Python structures defining mesh API"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from frequensolve.util.mixins import ExportContext, ExtraFieldsMixin, merge_extra

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
    distribute: Optional[bool] = True
    ranks_per_part: Optional[int] = None
    partitioner: Optional[str] = None

    def to_fs(self, ctx=None) -> Dict:
        return {
            "distribute": self.distribute,
            **({"ranks_per_part": self.ranks_per_part} if self.ranks_per_part else {}),
            **({"partitioner": self.partitioner} if self.partitioner else {}),
        }

    @classmethod
    def from_fs(cls, data: Dict) -> "MeshParallelism":
        return cls(
            distribute=data["distribute"],
            ranks_per_part=data.get("ranks_per_part"),
            partitioner=data.get("partitioner"),
        )


_GRADE_MODES = {"none", "inside", "outside", "band", "abs_band"}


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


@dataclass
class DistanceGrading(ExtraFieldsMixin):
    """Distance-based source/receiver mesh grading.

    This maps to the fast solver's ``kd_grade_t`` JSON contract. ``d0`` is the distance
    where the full multiplier is applied and ``d1`` is the distance where the
    multiplier returns to one.
    """

    d1: float
    mult: Optional[float] = None
    d0: float = 0.0
    mult_max: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mult is not None and self.mult_max is not None:
            raise ValueError(
                "Use either 'mult' or 'mult_max' for DistanceGrading, not both"
            )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        mult = self.mult if self.mult is not None else self.mult_max
        payload = {
            "d0": self.d0,
            "d1": self.d1,
            **({"mult": mult} if mult is not None else {}),
        }
        return merge_extra(payload, self.extra, "DistanceGrading")

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "DistanceGrading":
        payload = copy.deepcopy(dict(data))
        return cls(
            d0=payload.pop("d0", 0.0),
            d1=payload.pop("d1"),
            mult=payload.pop("mult", None),
            mult_max=payload.pop("mult_max", None),
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

    The exported JSON follows the fast solver's ``grading_fields_m`` contract. ``d0`` is
    the inner distance where the strongest multiplier is applied and ``d1`` is
    the outer distance where the multiplier returns to ``mult_min``.
    """

    surface: str
    d1: float
    mult: Optional[float] = None
    d0: float = 0.0
    mode: str = "abs_band"
    mult_max: Optional[float] = None
    mult_min: Optional[float] = None
    phi_scale: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.mode = _normalize_grade_mode(self.mode)
        if not self.surface:
            raise ValueError("SurfaceGrading requires a non-empty surface name")
        if self.mult is not None and self.mult_max is not None:
            raise ValueError(
                "Use either 'mult' or 'mult_max' for SurfaceGrading, not both"
            )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        payload = {
            "surface": self.surface,
            "mode": self.mode,
            "d0": self.d0,
            "d1": self.d1,
            **({"mult": self.mult} if self.mult is not None else {}),
            **({"mult_max": self.mult_max} if self.mult_max is not None else {}),
            **({"mult_min": self.mult_min} if self.mult_min is not None else {}),
            **({"phi_scale": self.phi_scale} if self.phi_scale is not None else {}),
        }
        return merge_extra(payload, self.extra, "SurfaceGrading")

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "SurfaceGrading":
        payload = copy.deepcopy(dict(data))
        return cls(
            surface=payload.pop("surface"),
            mode=payload.pop("mode", "abs_band"),
            d0=payload.pop("d0", 0.0),
            d1=payload.pop("d1"),
            mult=payload.pop("mult", None),
            mult_max=payload.pop("mult_max", None),
            mult_min=payload.pop("mult_min", None),
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
    """Sets mesh adaptivity options

    Attributes:
       elems_per_wave (float | Dict[str, float]): Elements per wavelength.
       order (int | Dict[str, int]): Element order used by mesh adaptivity.
       adapt_sources (Optional[int]): Number of additional refinements near sources
       adapt_receivers (Optional[int]): Number of additional refinements near receivers
       jump_tolerance (Optional[float]): Maximum relative change in wavespeed that consitutes
                                         a "jump" in material properties
       jump_factor (Optional[float]): Multiplicative factor for elems_per_wave on "jump" elements
       smooth_refs (Optional[bool]): Do additional refinements to unconstrain element DOFs
       f_low (Optional[float]): Frequency used for low-frequency mesh adaptation
       f_high (Optional[float]): Maximum frequency used for mesh adaptation
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
        return self.elems_per_wave

    @epw.setter
    def epw(self, value: Union[float, Dict[str, float]]) -> None:
        self.elems_per_wave = value

    @property
    def min_epw(self) -> Union[float, Dict[str, float]]:
        return self.elems_per_wave

    @min_epw.setter
    def min_epw(self, value: Union[float, Dict[str, float]]) -> None:
        self.elems_per_wave = value

    @property
    def f_adapt(self) -> Optional[float]:
        return self.f_low

    @f_adapt.setter
    def f_adapt(self, value: Optional[float]) -> None:
        self.f_low = value

    def to_fs(self, ctx=None) -> Dict:
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
        mult: Optional[float] = None,
        d0: float = 0.0,
        **kwargs,
    ) -> DistanceGrading:
        self.source_grading = DistanceGrading(d1=d1, mult=mult, d0=d0, **kwargs)
        return self.source_grading

    def set_receiver_grading(
        self,
        d1: float,
        mult: Optional[float] = None,
        d0: float = 0.0,
        **kwargs,
    ) -> DistanceGrading:
        self.receiver_grading = DistanceGrading(d1=d1, mult=mult, d0=d0, **kwargs)
        return self.receiver_grading

    def add_surface_grading(
        self,
        surface: str,
        d1: float,
        mult: Optional[float] = None,
        d0: float = 0.0,
        mode: str = "abs_band",
        **kwargs,
    ) -> SurfaceGrading:
        grading = SurfaceGrading(
            surface=surface,
            d1=d1,
            mult=mult,
            d0=d0,
            mode=mode,
            **kwargs,
        )
        self.surface_gradings.append(grading)
        return grading


@dataclass
class MeshManager:
    """Defines mesh type, dimension, refinement, etc.

    Attributes:
       mesh_generator (Optional[BaseMeshGenerator]): The mesh generator object
       parallel (Optional[MeshParallelism]): Mesh parallelism options
       adapt (Optional[MeshAdaptor]): Mesh adaptivity options
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
        """Sets mesh adaptivity options

        Attributes:
           elems_per_wave (float):           Elements per wavelength.
           order (int | Dict[str, int]):      Element order used by mesh adaptivity.
           adapt_sources (Optional[int]):    Number of additional refinements near sources
           adapt_receivers (Optional[int]):  Number of additional refinements near receivers
           jump_tolerance (Optional[float]): Maximum relative change in wavespeed that consitutes
                                             a "jump" in material properties
           jump_factor (Optional[float]):    Multiplicative factor for elems_per_wave on "jump" elements
           smooth_refs (Optional[bool]):     Do additional refinements to unconstrain element DOFs
           f_low (Optional[float]):           Frequency used for low-frequency mesh adaptation
           f_high (Optional[float]):          Maximum frequency used for mesh adaptation
           source_grading:                   Distance grading around sources
           receiver_grading:                 Distance grading around receivers
           surface_gradings:                 Geometry-based grading rules keyed by implicit surface
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
        mult: Optional[float] = None,
        d0: float = 0.0,
        **kwargs,
    ) -> DistanceGrading:
        if self.adapt is None:
            self.set_adapt(elems_per_wave=2.0, adapt_sources=1)
        return self.adapt.set_source_grading(d1=d1, mult=mult, d0=d0, **kwargs)

    def set_receiver_grading(
        self,
        d1: float,
        mult: Optional[float] = None,
        d0: float = 0.0,
        **kwargs,
    ) -> DistanceGrading:
        if self.adapt is None:
            self.set_adapt(elems_per_wave=2.0, adapt_sources=1)
        return self.adapt.set_receiver_grading(d1=d1, mult=mult, d0=d0, **kwargs)

    def add_surface_grading(
        self,
        surface: str,
        d1: float,
        mult: Optional[float] = None,
        d0: float = 0.0,
        mode: str = "abs_band",
        **kwargs,
    ) -> SurfaceGrading:
        if self.adapt is None:
            self.set_adapt(elems_per_wave=2.0, adapt_sources=1)
        return self.adapt.add_surface_grading(
            surface=surface,
            d1=d1,
            mult=mult,
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
        """Sets mesh parallel options

        Attributes:
           distribute (bool):               Distribute mesh
           ranks_per_part (Optional[int]):  Number of ranks per mesh part
           partitioner (Optional[str]):     Partitioner type
        """
        self.parallel = MeshParallelism(
            distribute=distribute,
            ranks_per_part=ranks_per_part,
            partitioner=partitioner,
        )

    @classmethod
    def from_fs(cls, data: Dict) -> "MeshManager":
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
        ctx = ctx or ExportContext(self._proj_path, self._rel_path)
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
            # Copy mesh file to project directory if not already there
            mesh_file = Path(self.file)
            if not mesh_file.is_relative_to(self._proj_path):
                dest = (self._path / mesh_file.name).resolve()
                dest.parent.mkdir(parents=True, exist_ok=True)
                copy2(mesh_file, dest)
                self.file = dest
                mesh_dict["file"] = self.file.relative_to(self._proj_path)
            else:
                mesh_dict["file"] = self.file

            mesh_dict["format"] = self.format

        # Mesh determined by generator (defined in backend)
        elif isinstance(self.mesh, BaseMeshGenerator):
            mesh_dict["generator"] = self.mesh.to_fs(ctx)

        return mesh_dict

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path
        if isinstance(self.mesh, BaseMeshGenerator):
            self.mesh._set_path(proj_path, rel_path)

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path
