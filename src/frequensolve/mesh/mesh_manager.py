"""Python structures defining mesh API"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from frequensolve.util.mixins import ExportContext, merge_extra

from .mesh import *  # noqa
from .mesh_generators import *  # noqa

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

    def __dict__(self) -> Dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: Dict) -> "MeshParallelism":
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


def _normalize_grade_mode(mode: str) -> str:
    mode = str(mode).strip().lower()
    if mode not in _GRADE_MODES:
        choices = ", ".join(sorted(_GRADE_MODES))
        raise ValueError(
            f"Unknown surface grading mode '{mode}'. Expected one of: {choices}"
        )
    return mode


@dataclass
class DistanceGrading:
    """Distance-based source/receiver mesh grading.

    This maps to Sauce's ``kd_grade_t`` JSON contract. ``d0`` is the distance
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

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self.extra

    @kwargs.setter
    def kwargs(self, value: Mapping[str, Any]) -> None:
        self.extra = copy.deepcopy(dict(value))

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        mult = self.mult if self.mult is not None else self.mult_max
        payload = {
            "d0": self.d0,
            "d1": self.d1,
            **({"mult": mult} if mult is not None else {}),
        }
        return merge_extra(payload, self.extra, "DistanceGrading")

    def __dict__(self) -> Dict[str, Any]:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DistanceGrading":
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
        return DistanceGrading.from_dict(value)
    raise TypeError(f"Expected DistanceGrading or mapping, got {type(value).__name__}")


@dataclass
class SurfaceGrading:
    """Geometric mesh grading around a named implicit surface.

    The exported JSON follows Sauce's ``grading_fields_m`` contract. ``d0`` is
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

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self.extra

    @kwargs.setter
    def kwargs(self, value: Mapping[str, Any]) -> None:
        self.extra = copy.deepcopy(dict(value))

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

    def __dict__(self) -> Dict[str, Any]:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SurfaceGrading":
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
        return SurfaceGrading.from_dict(value)
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
                item = SurfaceGrading.from_dict(payload)
            out.append(item)
        return out
    return [_coerce_surface_grading(item) for item in gradings]


@dataclass
class MeshAdaptor:
    """Sets mesh adaptivity options

    Attributes:
       min_epw (float | Dict[str, float]): Minimum # of elements per wavelength.
       adapt_sources (Optional[int]): Number of additional refinements near sources
       adapt_receivers (Optional[int]): Number of additional refinements near receivers
       jump_tolerance (Optional[float]): Maximum relative change in wavespeed that consitutes
                                         a "jump" in material properties
       jump_factor (Optional[float]): Multiplicative factor for min_epw on "jump" elements
       smooth_refs (Optional[bool]): Do additional refinements to unconstrain element DOFs
    """

    min_epw: Union[float, Dict[str, float]]
    jump_tolerance: Optional[float] = None  # 0.2
    jump_factor: Optional[float] = None  # 1.0
    smooth_refs: Optional[bool] = None  # False
    f_adapt: Optional[float] = None
    adapt_order: bool = False
    source_grading: Optional[DistanceGrading] = None
    receiver_grading: Optional[DistanceGrading] = None
    surface_gradings: List[SurfaceGrading] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.source_grading = _coerce_distance_grading(self.source_grading)
        self.receiver_grading = _coerce_distance_grading(self.receiver_grading)
        self.surface_gradings = _coerce_surface_gradings(self.surface_gradings)

    @property
    def kwargs(self) -> Dict:
        return self.extra

    @kwargs.setter
    def kwargs(self, value: Dict) -> None:
        self.extra = copy.deepcopy(dict(value))

    def to_fs(self, ctx=None) -> Dict:
        payload = {
            "min_epw": self.min_epw,
            **({"jump_tolerance": self.jump_tolerance} if self.jump_tolerance else {}),
            **({"jump_factor": self.jump_factor} if self.jump_factor else {}),
            **({"smooth_refs": self.smooth_refs} if self.smooth_refs else {}),
            **({"f_adapt": self.f_adapt} if self.f_adapt else {}),
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

    def __dict__(self) -> Dict:
        return self.to_fs()

    @classmethod
    def from_dict(cls, data: Dict) -> "MeshAdaptor":
        data = copy.deepcopy(data)
        return cls(
            min_epw=data.pop("min_epw"),
            jump_tolerance=data.pop("jump_tolerance", None),
            jump_factor=data.pop("jump_factor", None),
            smooth_refs=data.pop("smooth_refs", None),
            f_adapt=data.pop("f_adapt", None),
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
       mesh (Optional[Mesh]): The mesh object
       mesh_generator (Optional[BaseMeshGenerator]): The mesh generator object
       parallel (Optional[MeshParallelism]): Mesh parallelism options
       adapt (Optional[MeshAdaptor]): Mesh adaptivity options
    """

    mesh: Optional[Union[Mesh, BaseMeshGenerator]] = None
    file: Optional[str] = None
    format: Optional[str] = None
    parallel: Optional[MeshParallelism] = None
    adapt: Optional[MeshAdaptor] = None
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def set_adapt(
        self,
        min_epw: Union[float, Dict[str, float]],
        jump_tolerance: Optional[float] = None,
        jump_factor: Optional[float] = None,
        smooth_refs: Optional[bool] = None,
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
           min_epw (float):                  Minimum # of elements per wavelength.
           adapt_sources (Optional[int]):    Number of additional refinements near sources
           adapt_receivers (Optional[int]):  Number of additional refinements near receivers
           jump_tolerance (Optional[float]): Maximum relative change in wavespeed that consitutes
                                             a "jump" in material properties
           jump_factor (Optional[float]):    Multiplicative factor for min_epw on "jump" elements
           smooth_refs (Optional[bool]):     Do additional refinements to unconstrain element DOFs
           source_grading:                   Distance grading around sources
           receiver_grading:                 Distance grading around receivers
           surface_gradings:                 Geometry-based grading rules keyed by implicit surface
        """
        self.adapt = MeshAdaptor(
            min_epw=min_epw,
            jump_tolerance=jump_tolerance,
            jump_factor=jump_factor,
            smooth_refs=smooth_refs,
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
            self.set_adapt(min_epw=2.0, adapt_sources=1)
        return self.adapt.set_source_grading(d1=d1, mult=mult, d0=d0, **kwargs)

    def set_receiver_grading(
        self,
        d1: float,
        mult: Optional[float] = None,
        d0: float = 0.0,
        **kwargs,
    ) -> DistanceGrading:
        if self.adapt is None:
            self.set_adapt(min_epw=2.0, adapt_sources=1)
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
            self.set_adapt(min_epw=2.0, adapt_sources=1)
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
    def from_dict(cls, data: Dict) -> "MeshManager":
        data = copy.deepcopy(data)
        manager = cls()

        # From file
        file = data.get("file")
        format = data.get("format")
        if file is not None and format is not None:
            manager.file = file
            manager.format = format
            manager.mesh = Mesh.read_mesh(file, format)

        # From generator
        if "generator" in data:
            manager.mesh = BaseMeshGenerator.from_dict(data["generator"])

        # Parallel
        if "parallel" in data:
            p = data["parallel"]
            manager.set_parallel(
                distribute=p["distribute"],
                ranks_per_part=p.get("ranks_per_part"),
                partitioner=p.get("partitioner"),
            )

        if "adapt" in data:
            a = data["adapt"]
            manager.set_adapt(
                min_epw=a.pop("min_epw"),
                adapt_sources=a.pop("adapt_sources", 0),
                adapt_receivers=a.pop("adapt_receivers", 0),
                jump_tolerance=a.pop("jump_tolerance", None),
                jump_factor=a.pop("jump_factor", None),
                smooth_refs=a.pop("smooth_refs", False),
                f_adapt=a.pop("f_adapt", None),
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
            self.set_adapt(min_epw=2.0, adapt_sources=1)

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
            mesh_dict["generator"] = (
                self.mesh.to_fs(ctx)
                if hasattr(self.mesh, "to_fs")
                else self.mesh.__dict__()
            )

        # Write mesh to file (if mesh is a Mesh object)
        elif isinstance(self.mesh, Mesh):
            path = self._path / "mesh"
            self.mesh.write_mesh(path, "gmp")
            mesh_dict["file"] = path.relative_to(self._proj_path)
            mesh_dict["format"] = "gmp"

        return mesh_dict

    def __dict__(self) -> Dict:
        return self.to_fs()

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path
        if isinstance(self.mesh, BaseMeshGenerator):
            self.mesh._set_path(proj_path, rel_path)

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path
