"""Mesh generator configuration objects for solver input."""

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from frequensolve.units import is_quantity, unit_expression, value_and_units_to_fs

from ..util.class_registry import class_registry, register_class
from ..util.mixins import ExportContext, TypeTaggedMixin, merge_extra

__all__ = [
    "BaseMeshGenerator",
    "HorizontalSpacing",
    "HorizontalSpacingControl",
    "HexMeshGenerator",
    "LayeredMeshGenerator",
    "TetMeshGenerator",
]


def _first_quantity_units(value: Any) -> Optional[Any]:
    if is_quantity(value):
        return value.units
    if isinstance(value, Mapping):
        return None
    if isinstance(value, (str, bytes)):
        return None
    try:
        iterator = iter(value)
    except TypeError:
        return None
    for item in iterator:
        units = _first_quantity_units(item)
        if units is not None:
            return units
    return None


def _strip_bound_quantities(value: Any, units: Any) -> Any:
    if is_quantity(value):
        return value.to(units).magnitude
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, bytes)):
        return value
    try:
        iterator = iter(value)
    except TypeError:
        return value
    return [_strip_bound_quantities(item, units) for item in iterator]


def _mesh_bounds_and_units(
    l_bound: Any,
    u_bound: Any,
    units: Optional[Any],
) -> tuple[Any, Any, Optional[str]]:
    target_units = (
        units or _first_quantity_units(l_bound) or _first_quantity_units(u_bound)
    )
    if target_units is None:
        return l_bound, u_bound, None
    units_expr = unit_expression(target_units)
    return (
        _strip_bound_quantities(l_bound, target_units),
        _strip_bound_quantities(u_bound, target_units),
        units_expr,
    )


@register_class
@dataclass
class BaseMeshGenerator(TypeTaggedMixin, ABC):
    """Base class for type-tagged mesh generator configurations."""

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "BaseMeshGenerator":
        """Deserialize a registered mesh generator payload.

        Args:
            data: Serialized mesh generator mapping containing ``_type``.

        Returns:
            Concrete mesh generator instance.
        """

        return cls.dispatch_from_fs(data, class_registry)


@dataclass
class HorizontalSpacingControl:
    """Local horizontal mesh spacing control around a named borehole.

    Args:
        around_borehole: Borehole name the control applies to.
        padding: Optional physical padding around the borehole before the
            maximum element size is relaxed.
        max_size: Optional maximum horizontal element size near the borehole.
        extra: Additional solver-facing control fields.
    """

    around_borehole: str
    padding: Optional[Any] = None
    max_size: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "HorizontalSpacingControl":
        """Deserialize a local spacing-control payload.

        Args:
            data: Serialized horizontal spacing control mapping.

        Returns:
            ``HorizontalSpacingControl`` instance.
        """

        payload = dict(data)
        return cls(
            around_borehole=payload.pop("around_borehole"),
            padding=payload.pop("padding", None),
            max_size=payload.pop("max_size", None),
            extra=payload,
        )

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize this local spacing control for solver input.

        Args:
            ctx: Optional export context accepted for API consistency.

        Returns:
            JSON-compatible spacing-control payload.
        """

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
    """Horizontal mesh-spacing policy for layered meshes.

    Args:
        include_borehole_edges: Whether borehole edges should be included in the
            horizontal spacing constraints.
        max_growth: Optional maximum growth factor between adjacent horizontal
            element sizes.
        controls: Local spacing controls, usually around boreholes.
        extra: Additional solver-facing spacing policy fields.
    """

    include_borehole_edges: Optional[bool] = None
    max_growth: Optional[float] = None
    controls: List[HorizontalSpacingControl] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "HorizontalSpacing":
        """Deserialize a horizontal spacing policy.

        Args:
            data: Serialized horizontal spacing mapping.

        Returns:
            ``HorizontalSpacing`` instance.
        """

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
        **kwargs: Any,
    ) -> "HorizontalSpacing":
        """Add a local spacing control around a borehole.

        Args:
            name: Borehole name.
            padding: Optional physical padding around the borehole.
            max_size: Optional maximum horizontal element size near the
                borehole.
            **kwargs: Additional solver-facing control fields.

        Returns:
            This spacing policy, enabling fluent configuration.
        """

        self.controls.append(
            HorizontalSpacingControl(
                around_borehole=name,
                padding=padding,
                max_size=max_size,
                extra=kwargs,
            )
        )
        return self

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize the horizontal spacing policy for solver input.

        Args:
            ctx: Optional export context forwarded to local controls.

        Returns:
            JSON-compatible horizontal spacing payload.
        """

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
    """Structured hexahedral mesh generator.

    Args:
        l_bound: Lower coordinate bounds of the mesh box.
        u_bound: Upper coordinate bounds of the mesh box.
        n: Element counts in each mesh direction. When omitted, a small default
            is inferred from the number of bounds.
        units: Coordinate units for the bounds.
        system: Coordinate-system name for the bounds.
        clip_to_envelope: Whether to clip the mesh to the model envelope.
        triangulate_strips: Whether to triangulate strip-like cells.
        horizontal_spacing: Optional layered-mesh horizontal spacing policy.
    """

    l_bound: Optional[List[float]] = None
    u_bound: Optional[List[float]] = None
    n: Optional[List[int]] = None
    units: Optional[str] = None
    system: Optional[str] = None
    clip_to_envelope: Optional[bool] = None
    triangulate_strips: Optional[bool] = None
    horizontal_spacing: Optional[Union[HorizontalSpacing, Dict[str, Any]]] = None

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize the mesh generator for solver input.

        Args:
            ctx: Optional export context. ``ctx.rel_path`` is used as the mesh
                output path when available.

        Returns:
            JSON-compatible mesh generator payload.
        """

        rel_path = ctx.rel_path if ctx is not None else Path()
        if self.l_bound is None or self.u_bound is None:
            raise ValueError("Mesh generator requires l_bound and u_bound")
        l_bound, u_bound, units = _mesh_bounds_and_units(
            self.l_bound,
            self.u_bound,
            self.units,
        )

        if self.n is None:
            self.n = [8] * len(self.l_bound)

        horizontal_spacing = self.horizontal_spacing
        if isinstance(horizontal_spacing, Mapping):
            horizontal_spacing = HorizontalSpacing.from_fs(horizontal_spacing)

        return {
            "_type": self.__class__.__name__,
            "path": rel_path,
            "n": self.n,
            "l_bound": l_bound,
            "u_bound": u_bound,
            **({"units": units} if units is not None else {}),
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
        **kwargs: Any,
    ) -> "HexMeshGenerator":
        """Request local horizontal refinement around a borehole.

        Args:
            name: Borehole name referenced by the layered model.
            padding: Optional physical padding around the borehole.
            max_size: Optional maximum horizontal element size near the
                borehole.
            include_edges: Whether borehole edges are included in spacing
                constraints.
            max_growth: Optional maximum horizontal element-size growth.
            **kwargs: Additional solver-facing spacing-control fields.

        Returns:
            This mesh generator, enabling fluent configuration.
        """

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
    def from_fs(cls, data: Mapping[str, Any]) -> "HexMeshGenerator":
        """Deserialize a structured hexahedral mesh generator.

        Args:
            data: Serialized mesh generator mapping.

        Returns:
            ``HexMeshGenerator`` instance.
        """

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
    def from_fs(cls, data: Mapping[str, Any]) -> "LayeredMeshGenerator":
        """Deserialize a layered structured mesh generator.

        Args:
            data: Serialized mesh generator mapping.

        Returns:
            ``LayeredMeshGenerator`` instance.
        """

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
    """Tetrahedral mesh generator using the structured box parameters.

    Args:
        l_bound: Lower coordinate bounds of the mesh box.
        u_bound: Upper coordinate bounds of the mesh box.
        n: Element counts in each direction before tetrahedralization.
        units: Coordinate units for the bounds.
        system: Coordinate-system name for the bounds.
        clip_to_envelope: Whether to clip the mesh to the model envelope.
        triangulate_strips: Whether to triangulate strip-like cells.
        horizontal_spacing: Optional layered-mesh horizontal spacing policy.
    """

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "TetMeshGenerator":
        """Deserialize a tetrahedral mesh generator.

        Args:
            data: Serialized mesh generator mapping.

        Returns:
            ``TetMeshGenerator`` instance.
        """

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
