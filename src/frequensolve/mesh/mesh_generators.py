"""Python structures defining mesh API"""

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from frequensolve.units import value_and_units_to_fs

from ..util.class_registry import class_registry, register_class
from ..util.mixins import TypeTaggedMixin, merge_extra

__all__ = [
    "BaseMeshGenerator",
    "HorizontalSpacing",
    "HorizontalSpacingControl",
    "HexMeshGenerator",
    "LayeredMeshGenerator",
    "TetMeshGenerator",
]


@register_class
@dataclass
class BaseMeshGenerator(TypeTaggedMixin, ABC):
    """Base class for mesh generators"""

    _proj_path: Path = Path()
    _rel_path: Path = Path()

    @classmethod
    def from_fs(cls, data: Dict) -> "BaseMeshGenerator":
        return cls.dispatch_from_fs(data, class_registry)

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path


@dataclass
class HorizontalSpacingControl:
    """Local horizontal mesh spacing control."""

    around_borehole: str
    padding: Optional[Any] = None
    max_size: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "HorizontalSpacingControl":
        payload = dict(data)
        return cls(
            around_borehole=payload.pop("around_borehole"),
            padding=payload.pop("padding", None),
            max_size=payload.pop("max_size", None),
            extra=payload,
        )

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        payload = {
            "around_borehole": self.around_borehole,
            **(
                {"padding": value_and_units_to_fs(self.padding)}
                if self.padding is not None
                else {}
            ),
            **(
                {"max_size": value_and_units_to_fs(self.max_size)}
                if self.max_size is not None
                else {}
            ),
        }
        return merge_extra(payload, self.extra, "HorizontalSpacingControl")


@dataclass
class HorizontalSpacing:
    """Horizontal mesh-spacing policy for layered meshes."""

    include_borehole_edges: Optional[bool] = None
    max_growth: Optional[float] = None
    controls: List[HorizontalSpacingControl] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "HorizontalSpacing":
        payload = dict(data)
        controls = [
            (
                item
                if isinstance(item, HorizontalSpacingControl)
                else HorizontalSpacingControl.from_fs(item)
            )
            for item in payload.pop("controls", [])
        ]
        return cls(
            include_borehole_edges=payload.pop("include_borehole_edges", None),
            max_growth=payload.pop("max_growth", None),
            controls=controls,
            extra=payload,
        )

    def add_around_borehole(
        self,
        name: str,
        *,
        padding: Optional[Any] = None,
        max_size: Optional[Any] = None,
        **kwargs,
    ) -> "HorizontalSpacing":
        self.controls.append(
            HorizontalSpacingControl(
                around_borehole=name,
                padding=padding,
                max_size=max_size,
                extra=kwargs,
            )
        )
        return self

    def to_fs(self, ctx=None) -> Dict[str, Any]:
        payload = {
            **(
                {"include_borehole_edges": self.include_borehole_edges}
                if self.include_borehole_edges is not None
                else {}
            ),
            **({"max_growth": self.max_growth} if self.max_growth is not None else {}),
            **(
                {"controls": [control.to_fs(ctx) for control in self.controls]}
                if self.controls
                else {}
            ),
        }
        return merge_extra(payload, self.extra, "HorizontalSpacing")


@register_class
@dataclass
class HexMeshGenerator(BaseMeshGenerator):
    """Generates a hexahedral mesh

    Attributes:
       l_bound (List[float]):
          The lower bounds of a 'box' domain
       u_bound (List[float]):
          The upper bounds of a 'box' domain
       n (List[int]):
          Number of elements in each direction
    """

    l_bound: Optional[List[float]] = None
    u_bound: Optional[List[float]] = None
    n: Optional[List[int]] = None
    units: Optional[str] = None
    system: Optional[str] = None
    clip_to_envelope: Optional[bool] = None
    triangulate_strips: Optional[bool] = None
    horizontal_spacing: Optional[Union[HorizontalSpacing, Dict[str, Any]]] = None

    def to_fs(self, ctx=None) -> Dict:
        if self.l_bound is not None:
            assert self.u_bound is not None
            l_bound = self.l_bound
            u_bound = self.u_bound

        if self.n is None:
            self.n = [8] * len(self.l_bound)

        horizontal_spacing = self.horizontal_spacing
        if isinstance(horizontal_spacing, Mapping):
            horizontal_spacing = HorizontalSpacing.from_fs(horizontal_spacing)

        return {
            "_type": self.__class__.__name__,
            "path": self._rel_path,
            "n": self.n,
            "l_bound": l_bound,
            "u_bound": u_bound,
            **({"units": self.units} if self.units is not None else {}),
            **({"system": self.system} if self.system is not None else {}),
            **(
                {"clip_to_envelope": self.clip_to_envelope}
                if self.clip_to_envelope is not None
                else {}
            ),
            **(
                {"triangulate_strips": self.triangulate_strips}
                if self.triangulate_strips is not None
                else {}
            ),
            **(
                {"horizontal_spacing": horizontal_spacing.to_fs(ctx)}
                if horizontal_spacing is not None
                else {}
            ),
        }

    def refine_around_borehole(
        self,
        name: str,
        *,
        padding: Optional[Any] = None,
        max_size: Optional[Any] = None,
        include_edges: Optional[bool] = True,
        max_growth: Optional[float] = None,
        **kwargs,
    ) -> "HexMeshGenerator":
        if self.horizontal_spacing is None:
            self.horizontal_spacing = HorizontalSpacing()
        elif isinstance(self.horizontal_spacing, Mapping):
            self.horizontal_spacing = HorizontalSpacing.from_fs(self.horizontal_spacing)
        if include_edges is not None:
            self.horizontal_spacing.include_borehole_edges = include_edges
        if max_growth is not None:
            self.horizontal_spacing.max_growth = max_growth
        self.horizontal_spacing.add_around_borehole(
            name,
            padding=padding,
            max_size=max_size,
            **kwargs,
        )
        return self

    @classmethod
    def from_fs(cls, data: Dict) -> "HexMeshGenerator":
        return cls(
            n=data["n"],
            l_bound=data["l_bound"],
            u_bound=data["u_bound"],
            units=data.get("units"),
            system=data.get("system"),
            clip_to_envelope=data.get("clip_to_envelope"),
            triangulate_strips=data.get("triangulate_strips"),
            horizontal_spacing=(
                HorizontalSpacing.from_fs(data["horizontal_spacing"])
                if "horizontal_spacing" in data
                else None
            ),
        )


@register_class
@dataclass
class LayeredMeshGenerator(HexMeshGenerator):
    """Generates the current layered structured mesh contract.

    This is the contract-facing name for the layered box mesh generator. The
    legacy ``HexMeshGenerator`` remains available for existing examples.
    """

    @classmethod
    def from_fs(cls, data: Dict) -> "LayeredMeshGenerator":
        return cls(
            n=data["n"],
            l_bound=data["l_bound"],
            u_bound=data["u_bound"],
            units=data.get("units"),
            system=data.get("system"),
            clip_to_envelope=data.get("clip_to_envelope"),
            triangulate_strips=data.get("triangulate_strips"),
            horizontal_spacing=(
                HorizontalSpacing.from_fs(data["horizontal_spacing"])
                if "horizontal_spacing" in data
                else None
            ),
        )


@register_class
@dataclass
class TetMeshGenerator(HexMeshGenerator):
    """Generates a tetrahedral mesh

    Attributes:
       l_bound (List[float]):
          The lower bounds of a 'box' domain
       u_bound (List[float]):
          The upper bounds of a 'box' domain
       n (List[int]):
          Number of elements in each direction
    """

    @classmethod
    def from_fs(cls, data: Dict) -> "TetMeshGenerator":
        return cls(
            n=data["n"],
            l_bound=data["l_bound"],
            u_bound=data["u_bound"],
            units=data.get("units"),
            system=data.get("system"),
            clip_to_envelope=data.get("clip_to_envelope"),
            triangulate_strips=data.get("triangulate_strips"),
            horizontal_spacing=(
                HorizontalSpacing.from_fs(data["horizontal_spacing"])
                if "horizontal_spacing" in data
                else None
            ),
        )
