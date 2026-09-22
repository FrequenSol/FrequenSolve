"""Control spaces, states and vectors for the imaging API.

A :class:`ControlSpace` is an ordered collection of *blocks*.  Every block
names one Sauce registry block (or one family of blocks) and is defined by
where it lives (a subdomain or surface), its basis (spacing, count or explicit
nodes) and, optionally, how its values are constrained (``transform`` and
``limits``).  The space fixes the vector ordering (``controls.active``), the
optimizer-coordinate bounds, the frozen-DOF support mask and how a vector is
rendered.

Two vector layouts exist side by side:

- the **Sauce layout** covers every DOF of every block, in block order, with
  complex blocks interleaved as ``[Re, Im, Re, Im, ...]``;
- the **optimizer layout** drops DOFs that the support mask marks frozen.

Blocks are contiguous in both, so :attr:`ControlSpace.slices` and
:attr:`ControlSpace.full_slices` are plain :class:`slice` objects.

Binding a space to a simulation (:meth:`ControlSpace.bind`) resolves every
block against a deep copy of that simulation: material blocks become
:class:`~frequensolve.model.parameterization.ParameterizedProperty`
decorators with zero coefficients, interface blocks install an rbf
``control`` and mesh blocks register ``Model/property_spaces`` entries.  The
caller's simulation is never mutated.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
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
import xarray as xr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from frequensolve.imaging._artifacts import (
    ControlRegistryManifest,
    ControlStateFile,
    ControlVectorFile,
    qualified_block_name,
    unqualified_block_name,
)
from frequensolve.model.implicit_geometry import ImplicitSurfaceControl, RBFSurface
from frequensolve.model.parameterization import (
    CONTROL_TRANSFORMS,
    BSplineControl,
    HatControl,
    MeshControl,
    MeshPropertySpace,
    ParameterizedProperty,
    TensorHatControl,
)
from frequensolve.model.property import Property, canonical_property_name
from frequensolve.model.representation import ControlRepresentation, EvaluationContext
from frequensolve.units import is_quantity, unit_expression

__all__ = [
    "BoundControlSpace",
    "ControlSpace",
    "ControlState",
    "ControlVector",
    "DepthProfile",
    "GridParameters",
    "InterfaceParameters",
    "MeshParameters",
    "ReflectivityField",
    "ReflectivityParameters",
    "ResolvedBlock",
    "SourceParameters",
    "SupportMask",
    "UnresolvedControlError",
]

_SOURCE_QUANTITIES: Tuple[str, ...] = (
    "position",
    "mechanism",
    "signature",
    "signature_df",
)
_REFLECTIVITY_PARAMETERIZATIONS = ("vp_ip", "vp_vs_ip", "ip_is_rho")
_DEFAULT_LENGTH_UNITS = "km"  # Sauce's default model coordinate unit


class UnresolvedControlError(RuntimeError):
    """Raised when a space needs a simulation or manifest to know its layout."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _hdf5_safe_id(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text or "/" in text or "." in text or text in {".", ".."}:
        raise ValueError(f"{label} must be a non-empty HDF5-safe name without '.'")
    return text


def _validate_transform(transform: str) -> str:
    text = str(transform).strip().lower()
    if text not in CONTROL_TRANSFORMS:
        raise ValueError(
            f"transform must be one of {', '.join(CONTROL_TRANSFORMS)}; "
            f"got {transform!r}"
        )
    return text


def _limit_value(value: Any, units: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if is_quantity(value):
        if units is not None:
            try:
                value = value.to(units).magnitude
            except Exception as exc:
                raise ValueError(
                    f"limit {value!r} is not compatible with property units {units!r}"
                ) from exc
        else:
            value = value.magnitude
    number = float(value)
    if math.isnan(number):
        raise ValueError("limits cannot be NaN")
    return number


def _validate_limits(limits: Any) -> Optional[Tuple[Any, Any]]:
    if limits is None:
        return None
    if isinstance(limits, (str, bytes)) or not hasattr(limits, "__len__"):
        raise ValueError("limits must be a (lower, upper) pair")
    if len(limits) != 2:
        raise ValueError("limits must be a (lower, upper) pair")
    lower, upper = limits
    low = _limit_value(lower, None)
    high = _limit_value(upper, None)
    if low is not None and high is not None and not low < high:
        raise ValueError("limits must satisfy lower < upper")
    return (lower, upper)


def _optimizer_bounds(
    transform: str,
    limits: Optional[Tuple[Any, Any]],
    reference: Optional[float],
    units: Optional[str],
) -> Tuple[float, float]:
    """Map physical value limits to bounds on the control coefficient.

    ``identity`` evaluates ``r + c``, ``log`` evaluates ``r exp(c)``,
    ``inverse`` evaluates ``1 / (1/r + c)`` and ``logit`` evaluates
    ``sigmoid(logit(r) + c)``; the bounds are the images of the limits under
    the inverse of each map.
    """

    if limits is None:
        return -math.inf, math.inf
    if reference is None:
        raise ValueError(
            "limits require a constant reference property to express bounds in "
            "optimizer coordinates"
        )
    low = _limit_value(limits[0], units)
    high = _limit_value(limits[1], units)
    r = float(reference)
    if transform == "identity":
        return (
            -math.inf if low is None else low - r,
            math.inf if high is None else high - r,
        )
    if transform == "log":
        if r <= 0.0:
            raise ValueError("log transform bounds require a positive reference")
        if low is not None and low <= 0.0:
            raise ValueError("log transform limits must be positive")
        return (
            -math.inf if low is None else math.log(low / r),
            math.inf if high is None else math.log(high / r),
        )
    if transform == "inverse":
        if r <= 0.0:
            raise ValueError("inverse transform bounds require a positive reference")
        if low is not None and low <= 0.0:
            raise ValueError("inverse transform limits must be positive")
        return (
            -math.inf if high is None else 1.0 / high - 1.0 / r,
            math.inf if low is None else 1.0 / low - 1.0 / r,
        )
    if transform == "logit":
        if not 0.0 < r < 1.0:
            raise ValueError("logit transform bounds require a reference in (0, 1)")

        def logit(p: float) -> float:
            if not 0.0 < p < 1.0:
                raise ValueError("logit transform limits must lie in (0, 1)")
            return math.log(p / (1.0 - p))

        return (
            -math.inf if low is None else logit(low) - logit(r),
            math.inf if high is None else logit(high) - logit(r),
        )
    raise ValueError(f"unsupported transform {transform!r}")


def _constant_value(prop: Property) -> Optional[float]:
    try:
        if not prop.is_constant:
            return None
        low, high = prop.extrema
        low = float(np.asarray(low))
        high = float(np.asarray(high))
    except Exception:
        return None
    return low if low == high else None


def _surface_extrema(surface: Any) -> Tuple[float, float]:
    low, high = surface.extrema
    return float(np.asarray(low)), float(np.asarray(high))


def _fit_uniform(extent: Tuple[float, float], spacing: float) -> Tuple[int, float]:
    """Return ``(count, spacing)`` for uniform nodes ending exactly at the extent.

    ``spacing`` is treated as a maximum: the node count is the smallest one
    whose uniform spacing does not exceed it.
    """

    length = extent[1] - extent[0]
    if spacing <= 0.0 or not math.isfinite(spacing):
        raise ValueError("spacing must be finite and positive")
    count = max(2, int(math.ceil(length / spacing - 1e-9)) + 1)
    return count, length / (count - 1)


def _interleave(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = array.reshape(-1)
        out = np.empty(2 * array.size, dtype=np.float64)
        out[0::2] = array.real
        out[1::2] = array.imag
        return out
    return np.asarray(array, dtype=np.float64).reshape(-1)


def _deinterleave(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[0::2] + 1j * array[1::2]


def _open_uniform_knots(
    extent: Tuple[float, float], count: int, degree: int
) -> np.ndarray:
    if count < degree + 1:
        raise ValueError("B-spline profiles need at least degree + 1 coefficients")
    interior = np.linspace(extent[0], extent[1], count - degree + 1)
    return np.concatenate(
        ([extent[0]] * degree, interior, [extent[1]] * degree)
    ).astype(np.float64)


def _mechanism_components(kind: str, dimension: int) -> int:
    kind = str(kind).strip().lower()
    if kind in {"scalar", "monopole"}:
        return 1
    if kind in {"vector", "dipole"}:
        return int(dimension)
    if kind == "tensor":
        return int(dimension) * (int(dimension) + 1) // 2
    raise ValueError(f"unsupported source kind {kind!r} for mechanism controls")


def _axis_names(dimension: int) -> Tuple[str, ...]:
    return ("x", "z") if int(dimension) == 2 else ("x", "y", "z")


# ---------------------------------------------------------------------------
# resolved blocks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedBlock:
    """One Sauce registry block with a known layout.

    Attributes:
        name: Qualified block name (``model.<id>``, ``source.<i>.<q>``, ...).
        key: User key of the owning block in the :class:`ControlSpace`.
        address: User address of this block (``key`` or ``key.<sub>``).
        size: Real DOF count (complex blocks count two per coefficient).
        complex: Whether the block interleaves complex coefficients.
        kind: ``profile``, ``grid``, ``mesh``, ``interface``, ``source``,
            ``reflectivity`` or ``registry``.
        control: Authoring control object behind the block, when any.
        transform: Sauce transform applied to the coefficients.
        lower: Scalar lower bound in optimizer coordinates.
        upper: Scalar upper bound in optimizer coordinates.
        dims: xarray dimension names for lattice-like blocks.
        coords: xarray coordinates per dimension.
        units: Coordinate units of ``coords`` when known.
        coordinate_system: Coordinate system of ``coords``.
        baseline: Authored coefficient values in the Sauce layout, when known.
        source_id: One-based physical source id for source blocks.
        quantity: Source quantity for source blocks.
        components: Component labels for source blocks.
        prop: Material property name for material blocks.
        subdomain: Subdomain name for material blocks.
    """

    name: str
    key: str
    address: str
    size: int
    complex: bool = False
    kind: str = "registry"
    control: Any = None
    transform: str = "identity"
    lower: float = -math.inf
    upper: float = math.inf
    dims: Tuple[str, ...] = ()
    coords: Optional[Dict[str, np.ndarray]] = None
    units: Optional[str] = None
    coordinate_system: Optional[str] = None
    baseline: Optional[np.ndarray] = None
    source_id: Optional[int] = None
    quantity: Optional[str] = None
    components: Tuple[str, ...] = ()
    prop: Optional[str] = None
    subdomain: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", qualified_block_name(self.name))
        if int(self.size) < 1:
            raise ValueError(f"block {self.name!r} must have at least one DOF")
        object.__setattr__(self, "size", int(self.size))
        if self.complex and self.size % 2:
            raise ValueError(f"complex block {self.name!r} needs an even DOF count")
        if self.baseline is not None:
            baseline = np.asarray(self.baseline, dtype=np.float64).reshape(-1)
            if baseline.size != self.size:
                raise ValueError(f"baseline of {self.name!r} has the wrong size")
            object.__setattr__(self, "baseline", baseline)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the lattice shape for xarray rendering."""

        if self.coords is None or not self.dims:
            return (self.size // (2 if self.complex else 1),)
        return tuple(int(np.asarray(self.coords[dim]).size) for dim in self.dims)

    @property
    def coefficient_count(self) -> int:
        """Return the number of (possibly complex) coefficients."""

        return self.size // 2 if self.complex else self.size


# ---------------------------------------------------------------------------
# bind context
# ---------------------------------------------------------------------------


class _BindContext:
    """Resolution helpers around a (copied) simulation."""

    def __init__(self, simulation: Any):
        self.simulation = simulation
        self.model = simulation.model
        self.dimension = int(simulation.dimension)
        self.property_spaces: Dict[str, MeshPropertySpace] = {}
        self.reflectivity: List[Dict[str, Any]] = []
        self.source_kinds: set = set()

    # -- model geometry -----------------------------------------------------

    @property
    def is_layered(self) -> bool:
        from frequensolve.model.layered.model import LayeredModel

        return isinstance(self.model, LayeredModel)

    @property
    def length_units(self) -> Optional[str]:
        """Return the declared model length units when any."""

        units = getattr(self.model, "_x_units", None)
        if units:
            return unit_expression(units)
        for surface in getattr(self.model, "surfaces", []) or []:
            depth_units = getattr(getattr(surface, "depth", None), "units", None)
            if depth_units:
                return unit_expression(depth_units)
        defaults = getattr(getattr(self.simulation, "units", None), "defaults", {})
        if isinstance(defaults, Mapping) and defaults.get("length"):
            return unit_expression(defaults["length"])
        return None

    def length(self, value: Any) -> Tuple[float, Optional[str]]:
        """Return ``(magnitude, units)`` of a length in model coordinate units."""

        if is_quantity(value):
            units = self.length_units or _DEFAULT_LENGTH_UNITS
            try:
                return float(value.to(units).magnitude), units
            except Exception as exc:
                raise ValueError(f"{value!r} is not a length") from exc
        return float(value), None

    def subdomain(self, name: str) -> Any:
        for subdomain in self.model.subdomains:
            if subdomain.name == name:
                return subdomain
        names = [s.name for s in self.model.subdomains]
        raise KeyError(f"model has no subdomain {name!r}; known: {names}")

    def layer(self, name: str) -> Any:
        if not self.is_layered:
            raise UnresolvedControlError(
                "subdomain extents require a LayeredModel; pass explicit nodes"
            )
        for layer in self.model.layers:
            if layer.name == name:
                if layer.upper is None or layer.lower is None:
                    raise ValueError(f"layer {name!r} is not bounded by two surfaces")
                return layer
        raise KeyError(f"layered model has no layer {name!r}")

    def lateral_extent(self, axis: str) -> Tuple[float, float]:
        if axis == "x":
            limits = self.model.x_limits
            return float(limits[0]), float(limits[1])
        if axis == "y":
            if getattr(self.model, "y_limits", None) is None:
                raise ValueError("2D model has no y extent")
            limits = self.model.y_limits
            return float(limits[0]), float(limits[1])
        raise ValueError(f"unknown lateral axis {axis!r}")

    def depth_extent(self, subdomain: Optional[str]) -> Tuple[float, float]:
        """Return the global-z span of a layer (or of the model)."""

        if not self.is_layered:
            raise UnresolvedControlError("depth extents require a LayeredModel")
        if subdomain is None:
            limits = self.model.z_limits
            return float(limits[0]), float(limits[1])
        layer = self.layer(subdomain)
        top = _surface_extrema(layer.upper)[0]
        bottom = _surface_extrema(layer.lower)[1]
        return top, bottom

    def is_global(self, name: str) -> bool:
        """Return whether ``name`` is the global Cartesian system."""

        if name == "global":
            return True
        system = getattr(self.simulation, "global_coordinate_system", None)
        return system is not None and getattr(system, "name", None) == name

    def coordinate_system(self, name: str) -> Any:
        for system in self.simulation.coordinate_systems:
            if system.name == name:
                return system
        raise KeyError(f"simulation has no coordinate system {name!r}")

    def ensure_below_system(self, layer: Any) -> str:
        """Return a surface-relative system with a ``below`` axis on ``layer``'s top."""

        from frequensolve.geometry.frame import Axis, SurfaceCoordinateSystem

        name = f"{layer.upper.name}_below"
        for system in self.simulation.coordinate_systems:
            if system.name == name:
                return name
        self.simulation.coordinate_systems.append(
            SurfaceCoordinateSystem(
                name,
                layer.upper.name,
                axes=[Axis("below", direction="z", positive="down")],
                normal="down",
            )
        )
        return name

    def profile_extent(
        self, subdomain: str, axis: str, coordinate_system: Optional[str]
    ) -> Tuple[Tuple[float, float], str, str]:
        """Return ``(extent, axis, coordinate_system)`` for a depth profile."""

        layer = self.layer(subdomain)
        upper = _surface_extrema(layer.upper)
        lower = _surface_extrema(layer.lower)
        if coordinate_system is not None and self.is_global(coordinate_system):
            if axis == "below":
                raise ValueError("a 'below' profile needs a surface-relative system")
            coordinate_system = None
        if coordinate_system is None:
            if axis == "below":
                system = self.ensure_below_system(layer)
                return (0.0, lower[1] - upper[0]), "below", system
            if axis == "z":
                return (upper[0], lower[1]), "z", "global"
            if axis in {"x", "y"}:
                return self.lateral_extent(axis), axis, "global"
            raise ValueError(
                f"axis {axis!r} needs an explicit coordinate_system declaring it"
            )
        system = self.coordinate_system(coordinate_system)
        if getattr(system, "type", "") == "surface":
            axes = {a.name: a for a in (system.axes or [])}
            if axis not in axes:
                raise ValueError(
                    f"coordinate system {coordinate_system!r} declares no axis {axis!r}"
                )
            positive = str(axes[axis].positive or system.normal or "up").lower()
            surface = self._surface(system.surface_ref)
            ref = _surface_extrema(surface)
            if positive == "down":
                return (upper[0] - ref[1], lower[1] - ref[0]), axis, coordinate_system
            return (ref[0] - lower[1], ref[1] - upper[0]), axis, coordinate_system
        if axis == "z":
            return (upper[0], lower[1]), axis, coordinate_system
        if axis in {"x", "y"}:
            return self.lateral_extent(axis), axis, coordinate_system
        raise ValueError(f"cannot derive the extent of axis {axis!r}")

    def _surface(self, ref: Any) -> Any:
        surfaces = list(getattr(self.model, "surfaces", []) or [])
        if isinstance(ref, int) and not isinstance(ref, bool):
            return surfaces[ref - 1]
        for surface in surfaces:
            if surface.name == ref:
                return surface
        raise KeyError(f"model has no surface {ref!r}")

    def bounding_box(self, subdomain: Optional[str]) -> Dict[str, Tuple[float, float]]:
        box = {"x": self.lateral_extent("x")}
        if self.dimension == 3:
            box["y"] = self.lateral_extent("y")
        box["z"] = self.depth_extent(subdomain)
        return box

    # -- installation -------------------------------------------------------

    def install_property(
        self, subdomain: str, prop: str, block_id: str, control: Any, transform: str
    ) -> Property:
        target = self.subdomain(subdomain)
        key = canonical_property_name(prop)
        if key not in target.properties:
            raise KeyError(f"subdomain {subdomain!r} has no property {prop!r}")
        existing = target.properties[key]
        if isinstance(existing, ParameterizedProperty):
            if existing.id != block_id:
                raise ValueError(
                    f"property {prop!r} of {subdomain!r} is already parameterized "
                    f"as {existing.id!r}"
                )
            existing = existing.reference
        for other in self.model.subdomains:
            for name, candidate in other.properties.items():
                if isinstance(candidate, ParameterizedProperty) and (
                    candidate.id == block_id and (other is not target or name != key)
                ):
                    raise ValueError(f"block id {block_id!r} is already used")
        parameterized = ParameterizedProperty(
            copy.deepcopy(existing), id=block_id, control=control, transform=transform
        )
        target.properties[key] = parameterized
        return existing

    def install_surface_control(
        self,
        surface: str,
        block_id: Optional[str],
        maximum_displacement: Optional[float],
        feasibility_band: Optional[float],
    ) -> RBFSurface:
        for candidate in self.model.implicit_surfaces:
            if candidate.name == surface:
                if not isinstance(candidate, RBFSurface):
                    raise TypeError(
                        f"surface {surface!r} is not an rbf surface; only rbf "
                        "surfaces carry coefficient controls"
                    )
                existing = candidate.control
                control_id = block_id or (existing.id if existing else surface)
                candidate.control = ImplicitSurfaceControl(
                    control_id,
                    maximum_displacement=maximum_displacement,
                    feasibility_band=feasibility_band,
                )
                return candidate
        names = [s.name for s in self.model.implicit_surfaces]
        raise KeyError(f"model has no implicit surface {surface!r}; known: {names}")

    def register_property_space(self, name: str, space: MeshPropertySpace) -> None:
        if name in self.model.property_spaces or name in self.property_spaces:
            raise ValueError(f"property space {name!r} is already declared")
        self.model.property_spaces[name] = space
        self.property_spaces[name] = space

    # -- acquisition --------------------------------------------------------

    def source_ids(self) -> List[int]:
        acquisition = self.simulation.acquisition
        count = acquisition.known_source_point_count()
        if count is None:
            raise UnresolvedControlError(
                "the acquisition's physical source count is not known locally"
            )
        if count == 0:
            raise ValueError("the acquisition declares no physical sources")
        return list(range(1, int(count) + 1))

    def source_kind(self) -> str:
        geometry = self.simulation.acquisition.source_geometry
        return str(geometry.kind)

    def source_coordinates(self) -> Optional[np.ndarray]:
        try:
            coords = self.simulation.acquisition.source_point_coords()
        except Exception:
            return None
        coords = np.asarray(coords, dtype=np.float64)
        return coords if coords.ndim == 2 and coords.size else None

    def material_control(self, basis: str) -> Any:
        for subdomain in self.model.subdomains:
            for prop in subdomain.properties.values():
                if isinstance(prop, ParameterizedProperty) and prop.id == basis:
                    return prop.control
        raise KeyError(f"no parameterized property has block id {basis!r}")


# ---------------------------------------------------------------------------
# block specifications
# ---------------------------------------------------------------------------


class _BlockSpec:
    """Common behaviour of the frozen block dataclasses."""

    id: Optional[str]

    def default_key(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def block_id(self, key: str) -> str:
        return self.id or key

    def resolve(self, key: str, ctx: Optional[_BindContext]) -> List[ResolvedBlock]:
        raise NotImplementedError


@dataclass(frozen=True)
class DepthProfile(_BlockSpec):
    """One-dimensional hat or B-spline profile of a property in a subdomain.

    Args:
        prop: Material property name (``vp``, ``rho`` ...).
        subdomain: Layer or subdomain the profile lives in.
        axis: Coordinate axis; ``below`` is depth below the layer's upper
            surface, ``z`` the global depth axis.
        spacing: Maximum node spacing; nodes end exactly at the extent.
        count: Number of coefficients.
        nodes: Explicit uniform node coordinates (hat) or breakpoints
            (B-spline).  Exactly one of ``spacing``, ``count`` and ``nodes``.
        coordinate_system: Coordinate system of ``axis``; ``None`` selects
            ``global`` (or an automatic surface-relative system for ``below``).
        transform: Sauce control transform.
        limits: Optional ``(lower, upper)`` physical value limits.
        degree: B-spline degree; ``None`` selects the compact ``hat`` map.
        id: Block id (``model.<id>``); defaults to the space key.
    """

    prop: str
    subdomain: str
    axis: str = "below"
    spacing: Any = None
    count: Optional[int] = None
    nodes: Any = None
    coordinate_system: Optional[str] = None
    transform: str = "identity"
    limits: Any = None
    degree: Optional[int] = None
    id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prop", str(self.prop).strip())
        object.__setattr__(self, "subdomain", str(self.subdomain).strip())
        object.__setattr__(self, "axis", str(self.axis).strip())
        if not self.prop or not self.subdomain or not self.axis:
            raise ValueError("DepthProfile requires prop, subdomain and axis")
        given = [
            name
            for name, value in (
                ("spacing", self.spacing),
                ("count", self.count),
                ("nodes", self.nodes),
            )
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError("DepthProfile needs exactly one of spacing, count, nodes")
        if self.count is not None:
            count = int(self.count)
            if count < 2:
                raise ValueError("DepthProfile count must be at least 2")
            object.__setattr__(self, "count", count)
        if self.spacing is not None and not is_quantity(self.spacing):
            spacing = float(self.spacing)
            if not math.isfinite(spacing) or spacing <= 0.0:
                raise ValueError("DepthProfile spacing must be finite and positive")
            object.__setattr__(self, "spacing", spacing)
        if self.nodes is not None:
            nodes = self.nodes
            if is_quantity(nodes):
                pass
            else:
                nodes = np.asarray(nodes, dtype=np.float64)
                if nodes.ndim != 1 or nodes.size < 2:
                    raise ValueError("DepthProfile nodes must be a 1-D array of >= 2")
                if not np.all(np.isfinite(nodes)) or np.any(np.diff(nodes) <= 0.0):
                    raise ValueError("DepthProfile nodes must be finite and increasing")
                object.__setattr__(self, "nodes", tuple(float(v) for v in nodes))
        if self.coordinate_system is not None:
            system = str(self.coordinate_system).strip()
            if not system:
                raise ValueError("coordinate_system cannot be empty")
            object.__setattr__(self, "coordinate_system", system)
        object.__setattr__(self, "transform", _validate_transform(self.transform))
        object.__setattr__(self, "limits", _validate_limits(self.limits))
        if self.degree is not None:
            degree = int(self.degree)
            if degree < 1:
                raise ValueError("B-spline degree must be at least 1")
            object.__setattr__(self, "degree", degree)
        if self.id is not None:
            object.__setattr__(self, "id", _hdf5_safe_id(self.id, "block id"))

    @classmethod
    def bspline(cls, prop: str, subdomain: str, *, degree: int = 3, **kwargs: Any):
        """Return a B-spline profile with open-uniform knots over the extent."""

        return cls(prop, subdomain, degree=degree, **kwargs)

    def default_key(self) -> str:
        return self.id or self.prop

    def _node_values(
        self, ctx: Optional[_BindContext]
    ) -> Tuple[np.ndarray, Optional[str]]:
        nodes = self.nodes
        if is_quantity(nodes):
            if ctx is None:
                raise UnresolvedControlError("unit-bearing nodes need a simulation")
            magnitudes = [ctx.length(v)[0] for v in nodes]
            return (
                np.asarray(magnitudes, dtype=np.float64),
                ctx.length_units or _DEFAULT_LENGTH_UNITS,
            )
        return np.asarray(nodes, dtype=np.float64), None

    def build_control(
        self, ctx: Optional[_BindContext]
    ) -> Tuple[Union[HatControl, BSplineControl], str, Tuple[float, float]]:
        """Return ``(control, coordinate_system, extent)`` with zero coefficients."""

        units: Optional[str] = None
        if self.nodes is not None:
            nodes, units = self._node_values(ctx)
            system = self.coordinate_system
            if system is None:
                if self.axis == "below":
                    if ctx is None:
                        raise UnresolvedControlError(
                            "a 'below' profile needs a simulation to anchor its axis"
                        )
                    system = ctx.ensure_below_system(ctx.layer(self.subdomain))
                else:
                    system = "global"
            extent = (float(nodes[0]), float(nodes[-1]))
            if self.degree is None:
                spacing = np.diff(nodes)
                if not np.allclose(spacing, spacing[0], rtol=1e-9, atol=0.0):
                    raise ValueError("hat profiles require uniformly spaced nodes")
                control: Union[HatControl, BSplineControl] = HatControl(
                    axis=self.axis,
                    spacing=float(spacing[0]),
                    origin=float(nodes[0]),
                    coefficients=np.zeros(nodes.size),
                    coordinate_system=system,
                    units=units,
                )
            else:
                degree = self.degree
                knots = np.concatenate(
                    ([nodes[0]] * degree, nodes, [nodes[-1]] * degree)
                )
                control = BSplineControl(
                    axis=self.axis,
                    knots=knots,
                    degree=degree,
                    coefficients=np.zeros(knots.size - degree - 1),
                    coordinate_system=system,
                    units=units,
                )
            return control, system, extent
        if ctx is None:
            raise UnresolvedControlError(
                f"DepthProfile({self.prop!r}, {self.subdomain!r}) needs a simulation "
                "to derive its extent; bind the space first"
            )
        extent, axis, system = ctx.profile_extent(
            self.subdomain, self.axis, self.coordinate_system
        )
        length = extent[1] - extent[0]
        if length <= 0.0:
            raise ValueError(
                f"subdomain {self.subdomain!r} has no extent along {axis!r}"
            )
        if self.spacing is not None:
            spacing, units = ctx.length(self.spacing)
            if self.degree is None:
                count, spacing = _fit_uniform(extent, spacing)
            else:
                spans = max(1, int(math.ceil(length / spacing - 1e-9)))
                count = spans + self.degree
        else:
            count = int(self.count)
            spacing = length / (count - 1)
        if self.degree is None:
            control = HatControl(
                axis=axis,
                spacing=spacing,
                origin=extent[0],
                coefficients=np.zeros(count),
                coordinate_system=system,
                units=units,
            )
        else:
            control = BSplineControl(
                axis=axis,
                knots=_open_uniform_knots(extent, count, self.degree),
                degree=self.degree,
                coefficients=np.zeros(count),
                coordinate_system=system,
                units=units,
            )
        return control, system, extent

    def resolve(self, key: str, ctx: Optional[_BindContext]) -> List[ResolvedBlock]:
        control, system, _extent = self.build_control(ctx)
        block_id = self.block_id(key)
        reference: Optional[float] = None
        units: Optional[str] = None
        if ctx is None and self.limits is not None:
            raise UnresolvedControlError(
                "limits need the reference property; bind the space to a simulation"
            )
        if ctx is not None:
            existing = ctx.install_property(
                self.subdomain, self.prop, block_id, control, self.transform
            )
            reference = _constant_value(existing)
            units = existing.units
        lower, upper = (-math.inf, math.inf)
        if self.limits is not None and ctx is not None:
            lower, upper = _optimizer_bounds(
                self.transform, self.limits, reference, units
            )
        coordinates = np.asarray(control.coordinates, dtype=np.float64)
        return [
            ResolvedBlock(
                name=f"model.{block_id}",
                key=key,
                address=key,
                size=control.size,
                kind="profile",
                control=control,
                transform=self.transform,
                lower=lower,
                upper=upper,
                dims=(control.axis,),
                coords={control.axis: coordinates},
                units=control.units,
                coordinate_system=system,
                baseline=np.zeros(control.size),
                prop=self.prop,
                subdomain=self.subdomain,
            )
        ]


@dataclass(frozen=True)
class GridParameters(_BlockSpec):
    """Tensor-product hat lattice over a subdomain's bounding box.

    Args:
        props: One property name or a list of names; one block per property.
        subdomain: Subdomain whose bounding box the lattice covers; ``None``
            covers the whole model.
        spacing: Maximum spacing per axis (``x, [y,] z``) or one scalar.
        shape: Node count per axis.
        grid: Explicit :class:`~frequensolve.geometry.grids.CartesianGrid`.
        transform: Sauce control transform shared by every property.
        limits: Optional ``(lower, upper)`` physical limits (all properties).
        id: Block id prefix; blocks are ``model.<id>`` for one property and
            ``model.<id>_<prop>`` for several.
    """

    props: Any
    subdomain: Optional[str] = None
    spacing: Any = None
    shape: Any = None
    grid: Any = None
    transform: str = "identity"
    limits: Any = None
    id: Optional[str] = None

    def __post_init__(self) -> None:
        props = self.props
        if isinstance(props, str):
            props = (props,)
        props = tuple(str(p).strip() for p in props)
        if not props or any(not p for p in props) or len(set(props)) != len(props):
            raise ValueError(
                "GridParameters requires distinct non-empty property names"
            )
        object.__setattr__(self, "props", props)
        if self.subdomain is not None:
            object.__setattr__(self, "subdomain", str(self.subdomain).strip() or None)
        given = [
            name
            for name, value in (
                ("spacing", self.spacing),
                ("shape", self.shape),
                ("grid", self.grid),
            )
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError("GridParameters needs exactly one of spacing, shape, grid")
        if self.shape is not None:
            shape = tuple(int(n) for n in np.asarray(self.shape).reshape(-1))
            if len(shape) not in (2, 3) or any(n < 2 for n in shape):
                raise ValueError("GridParameters shape needs 2 or 3 counts of >= 2")
            object.__setattr__(self, "shape", shape)
        if self.spacing is not None:
            spacing = self.spacing
            if is_quantity(spacing) or not hasattr(spacing, "__len__"):
                spacing = (spacing,)
            spacing = tuple(spacing)
            for value in spacing:
                magnitude = value.magnitude if is_quantity(value) else float(value)
                if not np.all(np.isfinite(magnitude)) or np.any(
                    np.asarray(magnitude) <= 0.0
                ):
                    raise ValueError("GridParameters spacing must be positive")
            object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "transform", _validate_transform(self.transform))
        object.__setattr__(self, "limits", _validate_limits(self.limits))
        if self.id is not None:
            object.__setattr__(self, "id", _hdf5_safe_id(self.id, "block id"))

    def default_key(self) -> str:
        if self.id:
            return self.id
        return self.props[0] if len(self.props) == 1 else "grid"

    def block_ids(self, key: str) -> Dict[str, str]:
        base = self.block_id(key)
        if len(self.props) == 1:
            return {self.props[0]: base}
        return {prop: f"{base}_{prop}" for prop in self.props}

    def build_control(self, ctx: Optional[_BindContext]) -> TensorHatControl:
        """Return the shared lattice with zero coefficients."""

        if self.grid is not None:
            grid = self.grid
            axes = tuple(str(d) for d in grid.dims)
            return TensorHatControl(
                axes=axes,
                shape=[int(n) for n in grid.n],
                origin=[float(v) for v in grid.x0],
                spacing=[float(v) for v in grid.dx],
                coefficients=np.zeros(int(np.prod(grid.n))),
                coordinate_system=grid.system or "global",
                units=unit_expression(grid.units) if grid.units else None,
            )
        if ctx is None:
            raise UnresolvedControlError(
                "GridParameters needs a simulation to derive its bounding box"
            )
        box = ctx.bounding_box(self.subdomain)
        axes = _axis_names(ctx.dimension)
        units: Optional[str] = None
        shape: List[int] = []
        spacing: List[float] = []
        origin: List[float] = []
        if self.shape is not None:
            if len(self.shape) != len(axes):
                raise ValueError(f"GridParameters shape needs {len(axes)} counts")
            for axis, count in zip(axes, self.shape):
                lo, hi = box[axis]
                shape.append(count)
                spacing.append((hi - lo) / (count - 1))
                origin.append(lo)
        else:
            values = self.spacing
            if len(values) == 1:
                values = values * len(axes)
            if len(values) != len(axes):
                raise ValueError(f"GridParameters spacing needs {len(axes)} values")
            for axis, value in zip(axes, values):
                magnitude, unit = ctx.length(value)
                units = units or unit
                lo, hi = box[axis]
                count, fitted = _fit_uniform((lo, hi), magnitude)
                shape.append(count)
                spacing.append(fitted)
                origin.append(lo)
        return TensorHatControl(
            axes=axes,
            shape=shape,
            origin=origin,
            spacing=spacing,
            coefficients=np.zeros(int(np.prod(shape))),
            coordinate_system="global",
            units=units,
        )

    def resolve(self, key: str, ctx: Optional[_BindContext]) -> List[ResolvedBlock]:
        control = self.build_control(ctx)
        if ctx is None and self.limits is not None:
            raise UnresolvedControlError(
                "limits need the reference property; bind the space to a simulation"
            )
        blocks: List[ResolvedBlock] = []
        ids = self.block_ids(key)
        coords = {
            axis: np.asarray(values, dtype=np.float64)
            for axis, values in zip(control.axes, control.axis_coordinates)
        }
        for prop, block_id in ids.items():
            lower, upper = (-math.inf, math.inf)
            if ctx is not None:
                targets = (
                    [ctx.subdomain(self.subdomain)]
                    if self.subdomain is not None
                    else list(ctx.model.subdomains)
                )
                if not targets:
                    raise ValueError("model has no subdomains")
                reference: Optional[float] = None
                units: Optional[str] = None
                if self.subdomain is None and len(targets) > 1:
                    # Whole-model lattice: one block per property shared across
                    # subdomains is not expressible; require one subdomain
                    # owning the property.
                    owners = [
                        s
                        for s in targets
                        if canonical_property_name(prop) in s.properties
                    ]
                    if len(owners) != 1:
                        raise ValueError(
                            f"GridParameters over the whole model needs exactly one "
                            f"subdomain owning {prop!r}; found {len(owners)}"
                        )
                    targets = owners
                existing = ctx.install_property(
                    targets[0].name,
                    prop,
                    block_id,
                    copy.deepcopy(control),
                    self.transform,
                )
                reference = _constant_value(existing)
                units = existing.units
                if self.limits is not None:
                    lower, upper = _optimizer_bounds(
                        self.transform, self.limits, reference, units
                    )
            blocks.append(
                ResolvedBlock(
                    name=f"model.{block_id}",
                    key=key,
                    address=key if len(ids) == 1 else f"{key}.{prop}",
                    size=control.size,
                    kind="grid",
                    control=control,
                    transform=self.transform,
                    lower=lower,
                    upper=upper,
                    dims=tuple(control.axes),
                    coords=coords,
                    units=control.units,
                    coordinate_system=control.coordinate_system,
                    baseline=np.zeros(control.size),
                    prop=prop,
                    subdomain=self.subdomain,
                )
            )
        return blocks


@dataclass(frozen=True)
class MeshParameters(_BlockSpec):
    """First-order nodal controls on a frozen mesh property space.

    The coefficient count is fixed by Sauce when it writes the property-space
    artifact; until a :class:`ControlRegistryManifest` is supplied the block
    has no size.

    Args:
        prop: Material property name.
        subdomain: Subdomain owning the property.
        frequency: Sizing frequency (hertz or a Pint frequency).
        epw: Elements per wavelength (scalar or per dimension).
        artifact: ``.h5`` artifact path; defaults to ``<id>.h5``.
        transform: Sauce control transform.
        limits: Optional physical value limits.
        id: Block id; defaults to the space key.
    """

    prop: str
    subdomain: str
    frequency: Any
    epw: Any
    artifact: Optional[str] = None
    transform: str = "identity"
    limits: Any = None
    id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prop", str(self.prop).strip())
        object.__setattr__(self, "subdomain", str(self.subdomain).strip())
        if not self.prop or not self.subdomain:
            raise ValueError("MeshParameters requires prop and subdomain")
        # Validate frequency and epw eagerly through the contract class.
        MeshPropertySpace("probe.h5", self.frequency, self.epw)
        if self.artifact is not None:
            artifact = str(self.artifact).strip()
            if not artifact.endswith(".h5"):
                raise ValueError("MeshParameters artifact must be an .h5 path")
            object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "transform", _validate_transform(self.transform))
        object.__setattr__(self, "limits", _validate_limits(self.limits))
        if self.id is not None:
            object.__setattr__(self, "id", _hdf5_safe_id(self.id, "block id"))

    def default_key(self) -> str:
        return self.id or self.prop

    def property_space(self, key: str) -> MeshPropertySpace:
        """Return the ``Model/property_spaces`` declaration for this block."""

        block_id = self.block_id(key)
        return MeshPropertySpace(
            self.artifact or f"{block_id}.h5", self.frequency, self.epw
        )

    def resolve(self, key: str, ctx: Optional[_BindContext]) -> List[ResolvedBlock]:
        if ctx is None:
            raise UnresolvedControlError("MeshParameters needs a simulation")
        block_id = self.block_id(key)
        ctx.register_property_space(block_id, self.property_space(key))
        control = MeshControl(space=block_id)
        existing = ctx.install_property(
            self.subdomain, self.prop, block_id, control, self.transform
        )
        lower, upper = (-math.inf, math.inf)
        if self.limits is not None:
            lower, upper = _optimizer_bounds(
                self.transform, self.limits, _constant_value(existing), existing.units
            )
        # Size unknown until Sauce writes the artifact; ``ControlSpace`` keeps
        # the block pending and ``with_manifest`` fills it in.
        return [
            ResolvedBlock(
                name=f"model.{block_id}",
                key=key,
                address=key,
                size=1,
                kind="mesh",
                control=control,
                transform=self.transform,
                lower=lower,
                upper=upper,
                prop=self.prop,
                subdomain=self.subdomain,
            )
        ]


@dataclass(frozen=True)
class InterfaceParameters(_BlockSpec):
    """Coefficient control of a radial-basis implicit surface.

    Args:
        surface: Name of the rbf surface on the model.
        maximum_displacement: Step limiter, in model length units (Sauce
            defaults to 0.25 x support radius).
        feasibility_band: Feasibility band (Sauce defaults to the support
            radius).
        id: Control id (``model.<id>``); defaults to the surface's existing
            control id, else the surface name.
    """

    surface: str
    maximum_displacement: Any = None
    feasibility_band: Any = None
    id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface", str(self.surface).strip())
        if not self.surface:
            raise ValueError("InterfaceParameters requires a surface name")
        for label in ("maximum_displacement", "feasibility_band"):
            value = getattr(self, label)
            if value is not None and not is_quantity(value):
                number = float(value)
                if not math.isfinite(number) or number <= 0.0:
                    raise ValueError(f"InterfaceParameters {label} must be positive")
                object.__setattr__(self, label, number)
        if self.id is not None:
            object.__setattr__(self, "id", _hdf5_safe_id(self.id, "control id"))

    def default_key(self) -> str:
        return self.id or self.surface

    def block_id(self, key: str) -> str:
        return self.id or self.surface

    def resolve(self, key: str, ctx: Optional[_BindContext]) -> List[ResolvedBlock]:
        if ctx is None:
            raise UnresolvedControlError("InterfaceParameters needs a simulation")
        displacement = (
            None
            if self.maximum_displacement is None
            else ctx.length(self.maximum_displacement)[0]
        )
        band = (
            None
            if self.feasibility_band is None
            else ctx.length(self.feasibility_band)[0]
        )
        surface = ctx.install_surface_control(self.surface, self.id, displacement, band)
        control_id = surface.control.id
        centers = np.asarray(surface.centers, dtype=np.float64)
        coords = {"center": np.arange(centers.shape[0])}
        for k, axis in enumerate(_axis_names(centers.shape[1])):
            coords[axis] = centers[:, k]
        return [
            ResolvedBlock(
                name=f"model.{control_id}",
                key=key,
                address=key,
                size=surface.size,
                kind="interface",
                control=surface,
                dims=("center",),
                coords=coords,
                units=ctx.length_units,
                coordinate_system="global",
                baseline=np.asarray(surface.coefficients, dtype=np.float64),
            )
        ]


@dataclass(frozen=True)
class SourceParameters(_BlockSpec):
    """Per-source position, mechanism, signature and signature-derivative blocks.

    Blocks are ordered quantity-major (``position`` of every source, then
    ``mechanism`` ...).  Position blocks are real with one entry per spatial
    dimension; the others are complex and interleaved.

    Args:
        sources: ``"all"`` or an iterable of one-based physical source ids.
        position: Whether to control source positions.
        mechanism: Whether to control the mechanism amplitudes.
        signature: Whether to control the complex source signature.
        signature_df: Whether to control the signature frequency derivative.
    """

    sources: Any = "all"
    position: bool = False
    mechanism: bool = False
    signature: bool = True
    signature_df: bool = False
    id: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        sources = self.sources
        if isinstance(sources, str):
            if sources.strip().lower() != "all":
                raise ValueError("sources must be 'all' or a list of source ids")
            object.__setattr__(self, "sources", "all")
        else:
            ids = tuple(int(v) for v in sources)
            if not ids or any(v < 1 for v in ids) or len(set(ids)) != len(ids):
                raise ValueError("source ids must be distinct positive integers")
            object.__setattr__(self, "sources", ids)
        for label in _SOURCE_QUANTITIES:
            object.__setattr__(self, label, bool(getattr(self, label)))
        if not any(getattr(self, label) for label in _SOURCE_QUANTITIES):
            raise ValueError("SourceParameters must enable at least one quantity")

    @property
    def quantities(self) -> Tuple[str, ...]:
        """Return the enabled quantities in block order."""

        return tuple(q for q in _SOURCE_QUANTITIES if getattr(self, q))

    def default_key(self) -> str:
        return "source"

    def resolve(self, key: str, ctx: Optional[_BindContext]) -> List[ResolvedBlock]:
        if ctx is None:
            raise UnresolvedControlError("SourceParameters needs a simulation")
        available = ctx.source_ids()
        ids = available if self.sources == "all" else list(self.sources)
        unknown = sorted(set(ids) - set(available))
        if unknown:
            raise ValueError(f"acquisition has no source id(s) {unknown}")
        kind = ctx.source_kind()
        ctx.source_kinds.add(kind)
        dimension = ctx.dimension
        axes = _axis_names(dimension)
        coordinates = ctx.source_coordinates()
        blocks: List[ResolvedBlock] = []
        for quantity in self.quantities:
            for source_id in ids:
                if quantity == "position":
                    baseline = None
                    if coordinates is not None and source_id <= coordinates.shape[0]:
                        baseline = coordinates[source_id - 1, :dimension]
                    blocks.append(
                        ResolvedBlock(
                            name=f"source.{source_id}.position",
                            key=key,
                            address=f"{key}.position",
                            size=dimension,
                            kind="source",
                            components=axes,
                            baseline=baseline,
                            source_id=source_id,
                            quantity=quantity,
                        )
                    )
                    continue
                if quantity == "mechanism":
                    components = _mechanism_components(kind, dimension)
                    baseline = np.zeros(2 * components)
                elif quantity == "signature":
                    components = 1
                    baseline = np.array([1.0, 0.0])
                else:
                    components = 1
                    baseline = np.zeros(2)
                blocks.append(
                    ResolvedBlock(
                        name=f"source.{source_id}.{quantity}",
                        key=key,
                        address=f"{key}.{quantity}",
                        size=2 * components,
                        complex=True,
                        kind="source",
                        components=tuple(f"c{i + 1}" for i in range(components)),
                        baseline=baseline,
                        source_id=source_id,
                        quantity=quantity,
                    )
                )
        return blocks


@dataclass(frozen=True)
class ReflectivityField:
    """One ``fwi_operator.reflectivity.fields[]`` entry.

    Args:
        name: Field name, registered as ``reflectivity.<name>``.
        layer: One-based material layer index.
        axis: One-based axis of the parameterization chart.
        basis: Borrow the spatial basis of this material block id.
        control: Independent hat or B-spline control (or its mapping).
    """

    name: str
    layer: int
    axis: int
    basis: Optional[str] = None
    control: Any = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name or not (name[0].isalpha() and name.replace("_", "").isalnum()):
            raise ValueError("reflectivity field name must match [A-Za-z][A-Za-z0-9_]*")
        object.__setattr__(self, "name", name)
        layer = int(self.layer)
        axis = int(self.axis)
        if layer < 1:
            raise ValueError("reflectivity layer is one-based and positive")
        if not 1 <= axis <= 3:
            raise ValueError("reflectivity axis must be between 1 and 3")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "axis", axis)
        if (self.basis is None) == (self.control is None):
            raise ValueError("reflectivity fields need exactly one of basis or control")
        if self.basis is not None:
            object.__setattr__(
                self,
                "basis",
                _hdf5_safe_id(unqualified_block_name(self.basis), "basis id"),
            )
        if self.control is not None:
            control = self.control
            if isinstance(control, Mapping):
                from frequensolve.model.parameterization import control_from_fs

                control = control_from_fs(control)
            if not isinstance(control, (HatControl, BSplineControl)):
                raise TypeError(
                    "reflectivity control must be a hat or B-spline control"
                )
            object.__setattr__(self, "control", control)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this field for ``fwi_operator.reflectivity.fields``."""

        payload: Dict[str, Any] = {
            "name": self.name,
            "layer": self.layer,
            "axis": self.axis,
        }
        if self.basis is not None:
            payload["basis"] = self.basis
        else:
            payload["control"] = self.control.to_fs()
        return payload


@dataclass(frozen=True)
class ReflectivityParameters(_BlockSpec):
    """Joint reflectivity fields (``fwi_operator.reflectivity``).

    Args:
        parameterization: ``vp_ip``, ``vp_vs_ip`` or ``ip_is_rho``.
        fields: Non-empty sequence of :class:`ReflectivityField`.
        workspace_mb: Optional per-rank workspace budget.
    """

    parameterization: str = "vp_ip"
    fields: Any = ()
    workspace_mb: Optional[float] = None
    id: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        parameterization = str(self.parameterization).strip().lower()
        if parameterization not in _REFLECTIVITY_PARAMETERIZATIONS:
            raise ValueError(
                "reflectivity parameterization must be one of "
                + ", ".join(_REFLECTIVITY_PARAMETERIZATIONS)
            )
        object.__setattr__(self, "parameterization", parameterization)
        fields = tuple(
            f if isinstance(f, ReflectivityField) else ReflectivityField(**f)
            for f in (self.fields or ())
        )
        if not fields:
            raise ValueError("ReflectivityParameters requires at least one field")
        names = [f.name for f in fields]
        if len(set(names)) != len(names):
            raise ValueError("reflectivity field names must be unique")
        max_axis = 2 if parameterization == "vp_ip" else 3
        for f in fields:
            if f.axis > max_axis:
                raise ValueError(
                    f"reflectivity axis must be <= {max_axis} for {parameterization}"
                )
        object.__setattr__(self, "fields", fields)
        if self.workspace_mb is not None:
            workspace = float(self.workspace_mb)
            if not math.isfinite(workspace) or workspace <= 0.0:
                raise ValueError("reflectivity workspace_mb must be positive")
            object.__setattr__(self, "workspace_mb", workspace)

    def default_key(self) -> str:
        return "reflectivity"

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the ``fwi_operator.reflectivity`` mapping."""

        payload: Dict[str, Any] = {
            "parameterization": self.parameterization,
            "fields": [f.to_fs() for f in self.fields],
        }
        if self.workspace_mb is not None:
            payload["workspace_mb"] = self.workspace_mb
        return payload

    def resolve(self, key: str, ctx: Optional[_BindContext]) -> List[ResolvedBlock]:
        blocks: List[ResolvedBlock] = []
        for f in self.fields:
            control = f.control
            if control is None:
                if ctx is None:
                    raise UnresolvedControlError(
                        f"reflectivity field {f.name!r} borrows basis {f.basis!r}; "
                        "bind the space to size it"
                    )
                control = ctx.material_control(f.basis)
            size = control.size
            if size is None:
                raise UnresolvedControlError(
                    f"reflectivity basis {f.basis!r} is a mesh control whose size is "
                    "unknown until the manifest is read"
                )
            dims: Tuple[str, ...] = ()
            coords: Optional[Dict[str, np.ndarray]] = None
            if isinstance(control, (HatControl, BSplineControl)):
                dims = (control.axis,)
                coords = {
                    control.axis: np.asarray(control.coordinates, dtype=np.float64)
                }
            elif isinstance(control, TensorHatControl):
                dims = tuple(control.axes)
                coords = {
                    axis: np.asarray(v, dtype=np.float64)
                    for axis, v in zip(control.axes, control.axis_coordinates)
                }
            blocks.append(
                ResolvedBlock(
                    name=f"reflectivity.{f.name}",
                    key=key,
                    address=f"{key}.{f.name}",
                    size=size,
                    kind="reflectivity",
                    control=control,
                    dims=dims,
                    coords=coords,
                    units=getattr(control, "units", None),
                    coordinate_system=getattr(control, "coordinate_system", None),
                    baseline=np.zeros(size),
                )
            )
        if ctx is not None:
            ctx.reflectivity.append(self.to_fs())
        return blocks


@dataclass(frozen=True)
class _ManifestBlocks(_BlockSpec):
    """Blocks read from a ``fs-control-registry-1`` manifest."""

    blocks: Tuple[ResolvedBlock, ...]
    id: Optional[str] = field(default=None, init=False, repr=False)

    def default_key(self) -> str:
        return self.blocks[0].key

    def resolve(self, key: str, ctx: Optional[_BindContext]) -> List[ResolvedBlock]:
        return list(self.blocks)


BlockSpec = Union[
    DepthProfile,
    GridParameters,
    MeshParameters,
    InterfaceParameters,
    SourceParameters,
    ReflectivityParameters,
]
_MATERIAL_KINDS = {"profile", "grid", "mesh"}


# ---------------------------------------------------------------------------
# support mask
# ---------------------------------------------------------------------------


class SupportMask(Mapping[str, np.ndarray]):
    """Per-DOF support flags of a space, addressable by block.

    ``mask[key]`` returns the boolean mask of one block (user key, address or
    qualified name) in the Sauce layout; ``np.asarray(mask)`` returns the
    concatenated mask over the whole Sauce layout.
    """

    def __init__(self, space: "ControlSpace"):
        self._space = space

    def __getitem__(self, key: str) -> np.ndarray:
        blocks = self._space._select(key)
        if len(blocks) == 1:
            return np.array(self._space._mask_of(blocks[0]), copy=True)
        return np.concatenate([self._space._mask_of(b) for b in blocks])

    def __iter__(self) -> Iterator[str]:
        return iter(self._space.blocks)

    def __len__(self) -> int:
        return len(self._space.blocks)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.asarray(self.vector, dtype=dtype)

    @property
    def vector(self) -> np.ndarray:
        """Return the concatenated mask over the Sauce layout."""

        return np.concatenate(
            [self._space._mask_of(b) for b in self._space._resolved()]
        )

    @property
    def frozen_count(self) -> int:
        """Return the number of frozen DOFs."""

        return int(np.count_nonzero(~self.vector))

    def all(self) -> bool:
        """Return whether every DOF is supported."""

        return bool(np.all(self.vector))

    def __repr__(self) -> str:
        return f"SupportMask(frozen={self.frozen_count}, size={self._space.full_size})"


# ---------------------------------------------------------------------------
# control space
# ---------------------------------------------------------------------------


class ControlSpace:
    """Ordered collection of control blocks defining an optimizer vector space.

    Construct from keyword blocks (``ControlSpace(vp=DepthProfile(...))``),
    a single positional block, or :meth:`from_manifest`.  Layout-dependent
    members (``size``, ``slices``, ``zeros`` ...) require every block to be
    resolved; blocks that depend on a simulation resolve through
    :meth:`bind`.
    """

    def __init__(self, *blocks: Any, **named: Any):
        specs: Dict[str, Any] = {}
        for block in blocks:
            if isinstance(block, ControlSpace):
                for key, spec in block._specs.items():
                    if key in specs:
                        raise ValueError(f"duplicate control key {key!r}")
                    specs[key] = spec
                continue
            if not isinstance(block, _BlockSpec):
                raise TypeError(f"{type(block).__name__} is not a control block")
            key = block.default_key()
            if key in specs:
                raise ValueError(f"duplicate control key {key!r}")
            specs[key] = block
        for key, block in named.items():
            if not isinstance(block, _BlockSpec):
                raise TypeError(f"{key}: {type(block).__name__} is not a control block")
            if key in specs:
                raise ValueError(f"duplicate control key {key!r}")
            if not key.isidentifier():
                raise ValueError(f"control key {key!r} must be an identifier")
            specs[key] = block
        if not specs:
            raise ValueError("ControlSpace requires at least one block")
        self._specs: Dict[str, Any] = specs
        self._blocks: Optional[Tuple[ResolvedBlock, ...]] = None
        self._support: Dict[str, np.ndarray] = {}
        self._min_support: Optional[float] = None
        self._pending_mesh: bool = False
        self._try_resolve()

    # -- construction helpers ---------------------------------------------

    def _try_resolve(self) -> None:
        blocks: List[ResolvedBlock] = []
        try:
            for key, spec in self._specs.items():
                blocks.extend(spec.resolve(key, None))
        except UnresolvedControlError:
            self._blocks = None
            return
        self._set_blocks(blocks)

    def _set_blocks(self, blocks: Sequence[ResolvedBlock]) -> None:
        names = [b.name for b in blocks]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate qualified block name(s): {duplicates}")
        self._blocks = tuple(blocks)
        # Mesh blocks are sized by Sauce; until ``with_manifest`` supplies the
        # size they carry no baseline and the layout stays pending.
        self._pending_mesh = any(
            b.kind == "mesh" and b.baseline is None for b in blocks
        )

    def _clone(
        self, blocks: Sequence[ResolvedBlock], support: Mapping[str, np.ndarray]
    ) -> "ControlSpace":
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone._specs = {
            key: spec
            for key, spec in self._specs.items()
            if any(b.key == key for b in blocks)
        }
        clone._blocks = tuple(blocks)
        names = {b.name for b in blocks}
        clone._support = {
            n: np.array(m, copy=True) for n, m in support.items() if n in names
        }
        return clone

    @classmethod
    def from_manifest(cls, manifest: ControlRegistryManifest) -> "ControlSpace":
        """Build a fully resolved space from a registry manifest.

        Keys are the qualified block names; the active selection of the
        manifest is ignored (use :meth:`restrict`).  Baselines come from the
        manifest ``values`` when present.
        """

        values = manifest.unpack_state() if manifest.values.size else {}
        blocks: List[ResolvedBlock] = []
        for block in manifest.blocks:
            kind = "registry"
            source_id = None
            quantity = None
            if block.name.startswith("source."):
                kind = "source"
                parts = block.name.split(".")
                source_id = int(parts[1])
                quantity = parts[2]
            elif block.name.startswith("reflectivity."):
                kind = "reflectivity"
            blocks.append(
                ResolvedBlock(
                    name=block.name,
                    key=block.name,
                    address=block.name,
                    size=block.size,
                    complex=block.complex,
                    kind=kind,
                    units=block.units or None,
                    baseline=values.get(block.name),
                    source_id=source_id,
                    quantity=quantity,
                    components=tuple(block.components),
                )
            )
        space = object.__new__(cls)
        space._specs = {b.name: _ManifestBlocks((b,)) for b in blocks}
        space._support = {}
        space._min_support = None
        space._pending_mesh = False
        space._set_blocks(blocks)
        return space

    def with_manifest(self, manifest: ControlRegistryManifest) -> "ControlSpace":
        """Return a copy whose block sizes agree with ``manifest``.

        Mesh blocks (sized by Sauce) take their size from the manifest; every
        other block must already match.  Blocks missing from the manifest
        raise.
        """

        layout = {block.name: block for block in manifest.blocks}
        blocks: List[ResolvedBlock] = []
        for block in self._resolved(allow_pending=True):
            registered = layout.get(block.name)
            if registered is None:
                raise ValueError(f"manifest does not register block {block.name!r}")
            if block.kind == "mesh":
                blocks.append(
                    replace(
                        block, size=registered.size, baseline=np.zeros(registered.size)
                    )
                )
                continue
            if registered.size != block.size:
                raise ValueError(
                    f"block {block.name!r} has {block.size} DOFs locally but "
                    f"{registered.size} in the manifest"
                )
            blocks.append(block)
        clone = self._clone(blocks, self._support)
        clone._pending_mesh = False
        return clone

    # -- resolution ---------------------------------------------------------

    @property
    def resolved(self) -> bool:
        """Return whether every block has a known layout."""

        return self._blocks is not None and not self._pending_mesh

    def _resolved(self, allow_pending: bool = False) -> Tuple[ResolvedBlock, ...]:
        if self._blocks is None:
            raise UnresolvedControlError(
                "control space layout is unknown; bind it to a simulation first"
            )
        if self._pending_mesh and not allow_pending:
            raise UnresolvedControlError(
                "mesh control sizes are unknown until a control registry manifest "
                "is supplied (ControlSpace.with_manifest)"
            )
        return self._blocks

    def bind(self, simulation: Any) -> "BoundControlSpace":
        """Resolve every block against a deep copy of ``simulation``."""

        return BoundControlSpace(self, simulation)

    # -- introspection ------------------------------------------------------

    @property
    def keys(self) -> Tuple[str, ...]:
        """Return the user keys in order."""

        return tuple(self._specs)

    @property
    def specs(self) -> Dict[str, Any]:
        """Return ``key -> block specification``."""

        return dict(self._specs)

    def __getitem__(self, key: str) -> Any:
        return self._specs[key]

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        try:
            self._select(key)
        except KeyError:
            return False
        return True

    @property
    def blocks(self) -> Tuple[str, ...]:
        """Return the ordered qualified block names (``controls.active``)."""

        return tuple(b.name for b in self._resolved(allow_pending=True))

    @property
    def resolved_blocks(self) -> Tuple[ResolvedBlock, ...]:
        """Return the ordered resolved blocks."""

        return self._resolved(allow_pending=True)

    def block(self, name: str) -> ResolvedBlock:
        """Return the single resolved block addressed by ``name``."""

        blocks = self._select(name)
        if len(blocks) != 1:
            raise KeyError(f"{name!r} addresses {len(blocks)} blocks; name one")
        return blocks[0]

    def _select(self, name: str) -> List[ResolvedBlock]:
        text = str(name).strip()
        blocks = self._resolved(allow_pending=True)
        matches = [b for b in blocks if b.name == text]
        if matches:
            return matches
        matches = [b for b in blocks if b.key == text]
        if matches:
            return matches
        matches = [b for b in blocks if b.address == text]
        if matches:
            return matches
        parts = text.split(".")
        if len(parts) == 3:
            key, source_id, quantity = parts
            matches = [
                b
                for b in blocks
                if b.key == key
                and str(b.source_id) == source_id
                and b.quantity == quantity
            ]
            if matches:
                return matches
        try:
            qualified = qualified_block_name(text)
        except ValueError:
            qualified = None
        if qualified is not None:
            matches = [b for b in blocks if b.name == qualified]
            if matches:
                return matches
        raise KeyError(f"control space has no block {name!r}")

    # -- layout -------------------------------------------------------------

    def _mask_of(self, block: ResolvedBlock) -> np.ndarray:
        mask = self._support.get(block.name)
        if mask is None:
            return np.ones(block.size, dtype=bool)
        return mask

    @property
    def support(self) -> SupportMask:
        """Return the per-DOF support mask (frozen DOFs are ``False``)."""

        return SupportMask(self)

    @property
    def min_support(self) -> Optional[float]:
        """Return the relative support threshold recorded by :meth:`with_support`."""

        return self._min_support

    @property
    def full_size(self) -> int:
        """Return the Sauce-layout size (every DOF of every block)."""

        return int(sum(b.size for b in self._resolved()))

    @property
    def size(self) -> int:
        """Return the optimizer-layout size (frozen DOFs excluded)."""

        return int(
            sum(int(np.count_nonzero(self._mask_of(b))) for b in self._resolved())
        )

    @property
    def shape(self) -> Tuple[int]:
        """Return the optimizer vector shape."""

        return (self.size,)

    @property
    def full_slices(self) -> Dict[str, slice]:
        """Return ``qualified name -> slice`` into the Sauce layout."""

        slices: Dict[str, slice] = {}
        offset = 0
        for block in self._resolved():
            slices[block.name] = slice(offset, offset + block.size)
            offset += block.size
        return slices

    @property
    def slices(self) -> Dict[str, slice]:
        """Return ``qualified name -> slice`` into the optimizer layout."""

        slices: Dict[str, slice] = {}
        offset = 0
        for block in self._resolved():
            count = int(np.count_nonzero(self._mask_of(block)))
            slices[block.name] = slice(offset, offset + count)
            offset += count
        return slices

    @property
    def active_indices(self) -> np.ndarray:
        """Return the Sauce-layout indices kept in the optimizer vector."""

        return np.flatnonzero(self.support.vector)

    @property
    def frozen_indices(self) -> np.ndarray:
        """Return the Sauce-layout indices excluded from the optimizer vector."""

        return np.flatnonzero(~self.support.vector)

    @property
    def sizes(self) -> Dict[str, int]:
        """Return ``qualified name -> Sauce-layout size``."""

        return {b.name: b.size for b in self._resolved()}

    # -- vectors --------------------------------------------------------------

    def zeros(self) -> "ControlVector":
        """Return the zero optimizer vector."""

        return ControlVector(np.zeros(self.size), self)

    def ones(self) -> "ControlVector":
        """Return the all-ones optimizer vector."""

        return ControlVector(np.ones(self.size), self)

    def random(self, seed: Optional[int] = None) -> "ControlVector":
        """Return a standard-normal optimizer vector."""

        rng = np.random.default_rng(seed)
        return ControlVector(rng.standard_normal(self.size), self)

    def to_sauce_vector(self, vector: Any) -> np.ndarray:
        """Expand an optimizer vector to the Sauce layout with zeros at frozen DOFs."""

        values = self._optimizer_values(vector)
        full = np.zeros(self.full_size, dtype=np.float64)
        full[self.active_indices] = values
        return full

    def from_sauce_vector(self, values: Any) -> "ControlVector":
        """Drop frozen DOFs from a Sauce-layout vector."""

        array = np.asarray(values)
        if np.iscomplexobj(array):
            raise ValueError("Sauce vectors are real (complex blocks interleave)")
        array = np.asarray(array, dtype=np.float64).reshape(-1)
        if array.size != self.full_size:
            raise ValueError(
                f"vector has {array.size} entries; the Sauce layout has "
                f"{self.full_size}"
            )
        return ControlVector(array[self.active_indices], self)

    def _optimizer_values(self, vector: Any) -> np.ndarray:
        if isinstance(vector, ControlVector):
            if vector.space is not self and not vector.space.equivalent(self):
                raise ValueError("vector belongs to a different control space")
            array = vector.values
        else:
            array = np.asarray(vector)
            if np.iscomplexobj(array):
                raise ValueError("optimizer vectors are real")
            array = np.asarray(array, dtype=np.float64).reshape(-1)
        if array.size != self.size:
            raise ValueError(f"vector has {array.size} entries; expected {self.size}")
        return array

    def pack(self, values: Mapping[str, Any]) -> "ControlVector":
        """Pack per-block arrays (Sauce layout, complex allowed) into a vector.

        Keys may be user keys, addresses or qualified names; every block must
        be given exactly once.  Missing blocks default to zero when ``values``
        covers a strict subset only if they are entirely frozen.
        """

        if not isinstance(values, Mapping):
            raise TypeError("pack expects a mapping of block name -> values")
        full = np.zeros(self.full_size, dtype=np.float64)
        seen: set = set()
        slices = self.full_slices
        for name, block_values in values.items():
            blocks = self._select(name)
            if len(blocks) > 1:
                if not isinstance(block_values, Mapping):
                    raise ValueError(
                        f"{name!r} addresses {len(blocks)} blocks; pass a mapping"
                    )
                for sub_name, sub_values in block_values.items():
                    sub_blocks = [
                        b
                        for b in blocks
                        if sub_name in {b.name, b.address, b.address.split(".", 1)[-1]}
                    ]
                    if len(sub_blocks) != 1:
                        raise KeyError(f"{name}.{sub_name} does not address one block")
                    self._assign_block(full, slices, sub_blocks[0], sub_values, seen)
                continue
            self._assign_block(full, slices, blocks[0], block_values, seen)
        missing = [b.name for b in self._resolved() if b.name not in seen]
        if missing:
            raise ValueError(f"pack is missing block(s) {missing}")
        return self.from_sauce_vector(full)

    def _assign_block(
        self,
        full: np.ndarray,
        slices: Mapping[str, slice],
        block: ResolvedBlock,
        values: Any,
        seen: set,
    ) -> None:
        if block.name in seen:
            raise ValueError(f"block {block.name!r} given twice")
        seen.add(block.name)
        array = np.asarray(values)
        if np.iscomplexobj(array):
            if not block.complex:
                raise ValueError(f"block {block.name!r} is real; got complex values")
            flat = _interleave(array)
        else:
            flat = np.asarray(array, dtype=np.float64).reshape(-1)
            if block.complex and flat.size == block.size // 2:
                flat = _interleave(flat.astype(np.complex128))
        if flat.size != block.size:
            raise ValueError(
                f"block {block.name!r} expects {block.coefficient_count} "
                f"{'complex ' if block.complex else ''}coefficients; got {array.shape}"
            )
        if not np.all(np.isfinite(flat)):
            raise ValueError(f"block {block.name!r} values must be finite")
        full[slices[block.name]] = flat

    def unpack(self, vector: Any) -> Dict[str, np.ndarray]:
        """Split a vector into ``qualified name -> Sauce-layout block`` arrays.

        Complex blocks come back as complex arrays; frozen DOFs are zero.
        """

        full = self.to_sauce_vector(vector)
        out: Dict[str, np.ndarray] = {}
        for block, sl in zip(self._resolved(), self.full_slices.values()):
            values = np.array(full[sl], copy=True)
            out[block.name] = _deinterleave(values) if block.complex else values
        return out

    # -- bounds ---------------------------------------------------------------

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` optimizer-layout bound arrays."""

        lower = np.empty(self.size, dtype=np.float64)
        upper = np.empty(self.size, dtype=np.float64)
        for block, sl in zip(self._resolved(), self.slices.values()):
            lower[sl] = block.lower
            upper[sl] = block.upper
        return lower, upper

    # -- restriction and support --------------------------------------------

    def restrict(self, names: Union[str, Sequence[str]]) -> "ControlSpace":
        """Return the ordered subspace addressed by ``names``.

        Each entry is a user key (all of its blocks), an address such as
        ``src.signature`` or ``grid.vp``, or a qualified block name.  The
        resulting block order follows ``names``.
        """

        if isinstance(names, str):
            names = [names]
        blocks: List[ResolvedBlock] = []
        for name in names:
            for block in self._select(name):
                if block in blocks:
                    raise ValueError(f"block {block.name!r} selected twice")
                blocks.append(block)
        if not blocks:
            raise ValueError("restrict requires at least one block")
        return self._clone(blocks, self._support)

    def with_support(
        self,
        masks: Any,
        min_support: Optional[float] = None,
    ) -> "ControlSpace":
        """Return a copy carrying support masks (``False`` freezes a DOF).

        ``masks`` is a mapping ``block -> bool array`` (user keys, addresses
        or qualified names) or a :class:`ControlStateFile` /
        :class:`ControlVectorFile` whose ``support`` is used.  Blocks not
        mentioned keep their current mask.
        """

        if isinstance(masks, (ControlStateFile, ControlVectorFile)):
            masks = masks.support
        if not isinstance(masks, Mapping):
            raise TypeError("with_support expects a mapping or an artifact file")
        support = dict(self._support)
        for name, mask in masks.items():
            block = self.block(name)
            flags = np.asarray(mask).reshape(-1).astype(bool)
            if flags.size != block.size:
                raise ValueError(
                    f"support mask for {block.name!r} needs {block.size} flags; "
                    f"got {flags.size}"
                )
            if block.complex:
                pairs = flags.reshape(-1, 2)
                if np.any(pairs[:, 0] != pairs[:, 1]):
                    raise ValueError(
                        f"complex block {block.name!r} must freeze whole coefficients"
                    )
            if np.all(flags):
                support.pop(block.name, None)
            else:
                support[block.name] = flags
        clone = self._clone(list(self._resolved(allow_pending=True)), support)
        if min_support is not None:
            threshold = float(min_support)
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError("min_support must be finite and non-negative")
            clone._min_support = threshold
        return clone

    def without_support(self) -> "ControlSpace":
        """Return a copy with every DOF supported."""

        clone = self._clone(list(self._resolved(allow_pending=True)), {})
        clone._min_support = None
        return clone

    def support_masks(self) -> Dict[str, np.ndarray]:
        """Return ``qualified name -> mask`` for blocks with frozen DOFs."""

        return {name: np.array(mask, copy=True) for name, mask in self._support.items()}

    def equivalent(self, other: "ControlSpace") -> bool:
        """Return whether two spaces share layout, order and support."""

        if not isinstance(other, ControlSpace):
            return False
        try:
            mine = self._resolved()
            theirs = other._resolved()
        except UnresolvedControlError:
            return False
        if [(b.name, b.size, b.complex) for b in mine] != [
            (b.name, b.size, b.complex) for b in theirs
        ]:
            return False
        return all(
            np.array_equal(self._mask_of(a), other._mask_of(b))
            for a, b in zip(mine, theirs)
        )

    # -- transfer -------------------------------------------------------------

    def transfer_to(self, other: "ControlSpace", vector: Any) -> "ControlVector":
        """Transfer a vector to another resolution of the same blocks.

        Material profile and lattice blocks are transferred by evaluating the
        source field and least-squares projecting into the target basis
        (:mod:`frequensolve.model.representation`).  Blocks matched by key
        with identical layout are copied; anything else raises.
        """

        full = self.to_sauce_vector(vector)
        out = np.zeros(other.full_size, dtype=np.float64)
        my_slices = self.full_slices
        their_slices = other.full_slices
        mine_by_key: Dict[Tuple[str, str], ResolvedBlock] = {}
        for block in self._resolved():
            mine_by_key[(block.key, block.address)] = block
        for target in other._resolved():
            source = mine_by_key.get((target.key, target.address))
            if source is None:
                candidates = [b for b in self._resolved() if b.name == target.name]
                if not candidates:
                    raise ValueError(
                        f"source space has no block matching {target.name!r}"
                    )
                source = candidates[0]
            values = full[my_slices[source.name]]
            if (
                source.size == target.size
                and source.complex == target.complex
                and _same_basis(source, target)
            ):
                out[their_slices[target.name]] = values
                continue
            if source.kind in {"profile", "reflectivity"} and target.kind in {
                "profile",
                "reflectivity",
            }:
                out[their_slices[target.name]] = _transfer_profile(
                    source, target, values
                )
                continue
            if source.kind == "grid" and target.kind == "grid":
                out[their_slices[target.name]] = _transfer_lattice(
                    source, target, values
                )
                continue
            raise ValueError(
                f"cannot transfer block {source.name!r} ({source.kind}) to "
                f"{target.name!r} ({target.kind}) with different layouts"
            )
        return other.from_sauce_vector(out)

    def __len__(self) -> int:
        return len(self._specs)

    def __repr__(self) -> str:
        if self._blocks is None:
            return f"ControlSpace(keys={list(self._specs)}, unresolved)"
        size = "?" if self._pending_mesh else self.size
        return f"ControlSpace(blocks={list(self.blocks)}, size={size})"


def _same_basis(a: ResolvedBlock, b: ResolvedBlock) -> bool:
    if a.control is None or b.control is None:
        return True
    try:
        return a.control.to_fs() == b.control.to_fs()
    except Exception:
        return False


def _transfer_profile(
    source: ResolvedBlock, target: ResolvedBlock, values: np.ndarray
) -> np.ndarray:
    if not isinstance(source.control, (HatControl, BSplineControl)) or not isinstance(
        target.control, (HatControl, BSplineControl)
    ):
        raise ValueError("profile transfer requires hat or B-spline controls")
    if source.control.axis != target.control.axis or (
        source.control.coordinate_system != target.control.coordinate_system
    ):
        raise ValueError(
            "profile transfer requires the same axis and coordinate system"
        )
    src = ControlRepresentation(source.control)
    dst = ControlRepresentation(target.control)
    lo = max(src.knots[src.degree], dst.knots[dst.degree])
    hi = min(src.knots[src.size], dst.knots[dst.size])
    if not hi > lo:
        raise ValueError("profile transfer requires overlapping extents")
    samples = np.linspace(lo, hi, 40 * max(src.size, dst.size) + 1)
    context = EvaluationContext(
        {target.control.axis: samples},
        coordinate_system=target.control.coordinate_system,
    )
    return src.transfer_to(dst, values, context, damping=0.0, tolerance=1e-12)


class _LatticeRepresentation:
    """Separable multilinear evaluation of a tensor-hat lattice."""

    def __init__(self, control: TensorHatControl):
        self.control = control
        self.axes_1d = [
            ControlRepresentation(
                HatControl(
                    axis=axis,
                    spacing=float(control.spacing[k]),
                    origin=float(control.origin[k]),
                    coefficients=np.zeros(control.shape[k]),
                    coordinate_system=control.coordinate_system,
                )
            )
            for k, axis in enumerate(control.axes)
        ]

    @property
    def size(self) -> int:
        return self.control.size

    def sampling_operator(self, points: Mapping[str, np.ndarray]) -> csr_matrix:
        """Return the point-sampling operator (row-wise Khatri-Rao of 1-D hats).

        Coefficients follow the Sauce ordering with the first axis fastest,
        so the flat column index is ``i_0 + n_0 * (i_1 + n_1 * i_2)``.
        """

        n_points = int(np.asarray(next(iter(points.values()))).size)
        result: Optional[csr_matrix] = None
        stride = 1
        for k, rep in enumerate(self.axes_1d):
            axis = self.control.axes[k]
            context = EvaluationContext(
                {axis: points[axis]}, coordinate_system=self.control.coordinate_system
            )
            op = rep.sampling_operator(context).tocsr()
            if result is None:
                result = op
            else:
                prev = result.tocsr()
                prev_rows = np.repeat(np.arange(n_points), np.diff(prev.indptr))
                repeats = np.diff(op.indptr)[prev_rows]
                prev_index = np.repeat(np.arange(prev.nnz), repeats)
                starts = np.repeat(op.indptr[prev_rows], repeats)
                offsets = np.arange(prev_index.size) - np.repeat(
                    np.cumsum(repeats) - repeats, repeats
                )
                op_index = starts + offsets
                result = csr_matrix(
                    (
                        prev.data[prev_index] * op.data[op_index],
                        (
                            prev_rows[prev_index],
                            prev.indices[prev_index] + stride * op.indices[op_index],
                        ),
                    ),
                    shape=(n_points, stride * rep.size),
                )
            stride *= rep.size
        assert result is not None
        return result


def _transfer_lattice(
    source: ResolvedBlock, target: ResolvedBlock, values: np.ndarray
) -> np.ndarray:
    if not isinstance(source.control, TensorHatControl) or not isinstance(
        target.control, TensorHatControl
    ):
        raise ValueError("lattice transfer requires tensor-hat controls")
    if tuple(source.control.axes) != tuple(target.control.axes):
        raise ValueError("lattice transfer requires the same axes")
    src = _LatticeRepresentation(source.control)
    dst = _LatticeRepresentation(target.control)
    axes_points: List[np.ndarray] = []
    for k, axis in enumerate(target.control.axes):
        s_coords = source.control.axis_coordinates[k]
        t_coords = target.control.axis_coordinates[k]
        lo = max(s_coords[0], t_coords[0])
        hi = min(s_coords[-1], t_coords[-1])
        if not hi > lo:
            raise ValueError("lattice transfer requires overlapping extents")
        axes_points.append(
            np.linspace(lo, hi, 4 * max(s_coords.size, t_coords.size) + 1)
        )
    mesh = np.meshgrid(*axes_points, indexing="ij")
    points = {axis: grid.reshape(-1) for axis, grid in zip(target.control.axes, mesh)}
    samples = src.sampling_operator(points) @ values
    operator = dst.sampling_operator(points)
    return lsqr(operator, samples, atol=1e-10, btol=1e-10)[0]


# ---------------------------------------------------------------------------
# bound control space
# ---------------------------------------------------------------------------


class BoundControlSpace(ControlSpace):
    """A :class:`ControlSpace` resolved against a copy of a simulation.

    Attributes:
        simulation: Deep copy of the caller's simulation with the control
            decorators, rbf controls, property spaces and coordinate systems
            installed.  The original is untouched.
    """

    def __init__(self, space: ControlSpace, simulation: Any):
        simulation = copy.deepcopy(simulation)
        ctx = _BindContext(simulation)
        blocks: List[ResolvedBlock] = []
        for key, spec in space._specs.items():
            blocks.extend(spec.resolve(key, ctx))
        self._specs = dict(space._specs)
        self._support = {}
        self._min_support = space._min_support
        self._pending_mesh = False
        self._set_blocks(blocks)
        self.simulation = simulation
        self._reflectivity = list(ctx.reflectivity)
        self._property_spaces = dict(ctx.property_spaces)
        self._source_kinds = set(ctx.source_kinds)
        self._support = {
            name: mask for name, mask in space._support.items() if name in self.blocks
        }

    @property
    def qualified_names(self) -> Tuple[str, ...]:
        """Return the ordered qualified block names."""

        return self.blocks

    @property
    def mesh_property_spaces(self) -> Dict[str, MeshPropertySpace]:
        """Return the ``Model/property_spaces`` entries installed by binding."""

        return dict(self._property_spaces)

    def reflectivity_payload(self) -> Optional[Dict[str, Any]]:
        """Return the merged ``fwi_operator.reflectivity`` mapping, if any."""

        active = {b.name for b in self._resolved(allow_pending=True)}
        payloads = [
            p
            for p in self._reflectivity
            if any(f"reflectivity.{f['name']}" in active for f in p["fields"])
        ]
        if not payloads:
            return None
        parameterizations = {p["parameterization"] for p in payloads}
        if len(parameterizations) != 1:
            raise ValueError("reflectivity blocks must share one parameterization")
        fields = [
            f
            for p in payloads
            for f in p["fields"]
            if f"reflectivity.{f['name']}" in active
        ]
        payload: Dict[str, Any] = {
            "parameterization": parameterizations.pop(),
            "fields": copy.deepcopy(fields),
        }
        workspaces = [p["workspace_mb"] for p in payloads if "workspace_mb" in p]
        if workspaces:
            payload["workspace_mb"] = max(workspaces)
        return payload

    def source_controls_payload(self) -> Optional[Dict[str, Any]]:
        """Return ``fwi_operator.source_controls`` when source blocks are active."""

        if not any(b.kind == "source" for b in self._resolved(allow_pending=True)):
            return None
        return {"location_method": "analytic"}

    def controls_payload(self) -> Dict[str, Any]:
        """Return the ``fwi_operator.controls`` selection for the job."""

        payload: Dict[str, Any] = {"active": list(self.blocks)}
        if self._min_support is not None:
            payload["min_support"] = self._min_support
        return payload

    def geometric_support(self) -> Dict[str, np.ndarray]:
        """Return geometric fallback masks for layered-model material blocks.

        A profile coefficient is supported when the open support of its basis
        function meets the open extent of its subdomain along the profile
        axis; lattice nodes are supported when their hat support meets the
        subdomain's bounding box.  Only blocks with at least one frozen DOF
        are returned.
        """

        ctx = _BindContext(self.simulation)
        masks: Dict[str, np.ndarray] = {}
        for block in self._resolved(allow_pending=True):
            spec = self._specs.get(block.key)
            if block.kind == "profile" and isinstance(spec, DepthProfile):
                try:
                    extent, _axis, _system = ctx.profile_extent(
                        spec.subdomain,
                        block.control.axis,
                        block.control.coordinate_system,
                    )
                except (UnresolvedControlError, KeyError, ValueError):
                    continue
                mask = _profile_support(block.control, extent)
            elif block.kind == "grid" and isinstance(spec, GridParameters):
                try:
                    box = ctx.bounding_box(spec.subdomain)
                except (UnresolvedControlError, KeyError, ValueError):
                    continue
                mask = _lattice_support(block.control, box)
            else:
                continue
            if not np.all(mask):
                masks[block.name] = mask
        return masks

    def with_geometric_support(
        self, min_support: Optional[float] = None
    ) -> "BoundControlSpace":
        """Return a copy frozen by :meth:`geometric_support`."""

        frozen = self.with_support(self.geometric_support(), min_support=min_support)
        return frozen  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"BoundControlSpace(blocks={list(self.blocks)})"


def _profile_support(
    control: Union[HatControl, BSplineControl], extent: Tuple[float, float]
) -> np.ndarray:
    lo, hi = extent
    if isinstance(control, HatControl):
        nodes = np.asarray(control.coordinates)
        starts = np.concatenate(([nodes[0]], nodes[:-1]))
        ends = np.concatenate((nodes[1:], [nodes[-1]]))
        # A node whose only support is the closed endpoint still counts when it
        # sits on the extent boundary.
        starts = np.where(starts == nodes, nodes - 0.5 * control.spacing, starts)
        ends = np.where(ends == nodes, nodes + 0.5 * control.spacing, ends)
    else:
        knots = np.asarray(control.knots)
        starts = knots[: control.size]
        ends = knots[control.degree + 1 : control.degree + 1 + control.size]
    return (np.minimum(ends, hi) - np.maximum(starts, lo)) > 0.0


def _lattice_support(
    control: TensorHatControl, box: Mapping[str, Tuple[float, float]]
) -> np.ndarray:
    mask = np.ones(control.size, dtype=bool)
    coordinates = control.coordinates
    for k, axis in enumerate(control.axes):
        lo, hi = box[axis]
        spacing = float(control.spacing[k])
        values = coordinates[:, k]
        mask &= (
            np.minimum(values + spacing, hi) - np.maximum(values - spacing, lo)
        ) > 0.0
    return mask


# ---------------------------------------------------------------------------
# control vector
# ---------------------------------------------------------------------------


class ControlVector:
    """Real optimizer-layout vector bound to a :class:`ControlSpace`.

    Arithmetic with other vectors, ndarrays and scalars returns a new
    :class:`ControlVector`; ``np.asarray(v)`` exposes the values.
    """

    __array_priority__ = 20.0

    def __init__(self, values: Any, space: ControlSpace):
        if not isinstance(space, ControlSpace):
            raise TypeError("ControlVector requires a ControlSpace")
        array = np.asarray(values)
        if np.iscomplexobj(array):
            raise ValueError("control vectors are real; complex blocks interleave")
        array = np.array(array, dtype=np.float64).reshape(-1)
        if array.size != space.size:
            raise ValueError(
                f"vector has {array.size} entries; the space has {space.size}"
            )
        self._values = array
        self.space = space

    # -- ndarray protocol -----------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """Return the optimizer-layout values (a view)."""

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

    def copy(self) -> "ControlVector":
        """Return a deep copy of the values on the same space."""

        return ControlVector(np.array(self._values, copy=True), self.space)

    def _other(self, other: Any) -> Any:
        if isinstance(other, ControlVector):
            if other.space is not self.space and not other.space.equivalent(self.space):
                raise ValueError("control vectors belong to different spaces")
            return other._values
        array = np.asarray(other)
        if array.ndim == 0:
            return array
        if array.shape != self._values.shape:
            raise ValueError(f"operand shape {array.shape} does not match {self.shape}")
        return array

    def __add__(self, other: Any) -> "ControlVector":
        return ControlVector(self._values + self._other(other), self.space)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "ControlVector":
        return ControlVector(self._values - self._other(other), self.space)

    def __rsub__(self, other: Any) -> "ControlVector":
        return ControlVector(self._other(other) - self._values, self.space)

    def __mul__(self, other: Any) -> "ControlVector":
        return ControlVector(self._values * self._other(other), self.space)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "ControlVector":
        return ControlVector(self._values / self._other(other), self.space)

    def __neg__(self) -> "ControlVector":
        return ControlVector(-self._values, self.space)

    def __pos__(self) -> "ControlVector":
        return self.copy()

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, ControlVector):
            return NotImplemented
        return self.space.equivalent(other.space) and np.array_equal(
            self._values, other._values
        )

    __hash__ = None  # type: ignore[assignment]

    def dot(self, other: Any) -> float:
        """Return the Euclidean inner product with another vector."""

        return float(np.dot(self._values, self._other(other)))

    def norm(self, order: Any = None) -> float:
        """Return the vector norm (Euclidean by default)."""

        return float(np.linalg.norm(self._values, ord=order))

    def clip(self) -> "ControlVector":
        """Return a copy clipped to the space bounds."""

        lower, upper = self.space.bounds
        return ControlVector(np.clip(self._values, lower, upper), self.space)

    # -- block access -----------------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, (int, slice, np.integer, np.ndarray, list)):
            return self._values[key]
        blocks = self.space._select(key)
        unpacked = self.space.unpack(self)
        if len(blocks) == 1:
            return unpacked[blocks[0].name]
        return {b.name: unpacked[b.name] for b in blocks}

    def blocks(self) -> Dict[str, np.ndarray]:
        """Return ``qualified name -> block array`` (complex for complex blocks)."""

        return self.space.unpack(self)

    def per_source(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Return ``source id -> quantity -> values`` for source blocks."""

        table: Dict[int, Dict[str, np.ndarray]] = {}
        unpacked = self.space.unpack(self)
        for block in self.space.resolved_blocks:
            if block.kind != "source" or block.source_id is None:
                continue
            table.setdefault(block.source_id, {})[block.quantity or block.name] = (
                unpacked[block.name]
            )
        if not table:
            raise ValueError("the space has no source blocks")
        return table

    def to_xarray(self, key: Optional[str] = None, *, frozen: Any = np.nan) -> Any:
        """Render material blocks as xarray objects.

        A single-block address returns a :class:`xarray.DataArray` with the
        block's coordinates; ``None`` returns a :class:`xarray.Dataset` of
        every representable block.  Frozen DOFs are filled with ``frozen``.
        Mesh blocks are not representable and raise.
        """

        if key is not None:
            block = self.space.block(key)
            return self._block_to_xarray(block, frozen)
        representable = [
            block
            for block in self.space.resolved_blocks
            if block.kind in {"profile", "grid", "reflectivity", "interface"}
            and block.coords
        ]
        addresses = [block.address for block in representable]
        unique = len(set(addresses)) == len(addresses)
        arrays = {
            (block.address if unique else block.name): self._block_to_xarray(
                block, frozen
            )
            for block in representable
        }
        if not arrays:
            raise ValueError("the space has no xarray-representable blocks")
        return xr.Dataset(arrays)

    def _block_to_xarray(self, block: ResolvedBlock, frozen: Any) -> xr.DataArray:
        if block.kind == "mesh":
            raise NotImplementedError(
                f"mesh block {block.name!r} has no lattice; use ControlVector.to_mesh"
            )
        if block.kind == "source":
            raise ValueError(f"source block {block.name!r}: use per_source()")
        if not block.coords:
            raise ValueError(f"block {block.name!r} carries no coordinates")
        full = self.space.to_sauce_vector(self)
        values = np.array(full[self.space.full_slices[block.name]], copy=True)
        mask = self.space._mask_of(block)
        values[~mask] = frozen
        if block.complex:
            values = _deinterleave(values)
        data = values.reshape(block.shape, order="F")
        coords = {dim: np.asarray(block.coords[dim]) for dim in block.dims}
        extra_coords = {
            name: (block.dims[0], np.asarray(v))
            for name, v in block.coords.items()
            if name not in block.dims and len(block.dims) == 1
        }
        attrs: Dict[str, Any] = {"block": block.name, "transform": block.transform}
        if block.units:
            attrs["units"] = block.units
        if block.coordinate_system:
            attrs["coordinate_system"] = block.coordinate_system
        return xr.DataArray(
            data,
            dims=block.dims,
            coords={**coords, **extra_coords},
            name=block.address,
            attrs=attrs,
        )

    def to_mesh(self, *args: Any, **kwargs: Any) -> Any:
        """Render mesh blocks on the property-space mesh (Phase 2)."""

        raise NotImplementedError("ControlVector.to_mesh arrives in Phase 2")

    # -- files --------------------------------------------------------------------

    def to_file(
        self,
        *,
        state_fingerprint: Optional[str] = None,
        registry_fingerprint: Optional[str] = None,
        native: bool = False,
    ) -> ControlVectorFile:
        """Return the ``fs-control-vector-1`` representation.

        Frozen DOFs are written as zeros and the space's support masks travel
        along under ``/support/<block>``.
        """

        full = self.space.to_sauce_vector(self)
        slices = self.space.full_slices
        blocks = {name: full[sl] for name, sl in slices.items()}
        return ControlVectorFile(
            blocks,
            state_fingerprint=state_fingerprint,
            control_registry_fingerprint=registry_fingerprint,
            native=native,
            support=self.space.support_masks(),
            support_min_support=self.space.min_support,
        )

    @classmethod
    def from_file(
        cls, file: Union[ControlVectorFile, str, Path], space: ControlSpace
    ) -> "ControlVector":
        """Read a vector file onto ``space``.

        Blocks are matched by qualified name and must all be present with
        the space's sizes.  When the file carries support masks and ``space``
        freezes nothing, the returned vector lives on
        ``space.with_support(file.support)``.
        """

        if not isinstance(file, ControlVectorFile):
            file = ControlVectorFile.read(file)
        if file.support and not space.support_masks():
            space = space.with_support(
                {n: m for n, m in file.support.items() if n in space.blocks},
                min_support=file.support_min_support,
            )
        full = np.zeros(space.full_size, dtype=np.float64)
        for name, sl in space.full_slices.items():
            try:
                values = file[name]
            except KeyError as exc:
                raise ValueError(f"vector file has no block {name!r}") from exc
            if values.size != sl.stop - sl.start:
                raise ValueError(
                    f"block {name!r} has {values.size} entries in the file; "
                    f"the space has {sl.stop - sl.start}"
                )
            full[sl] = values
        return space.from_sauce_vector(full)

    def save(
        self,
        path: Union[str, Path],
        *,
        state_fingerprint: Optional[str] = None,
        registry_fingerprint: Optional[str] = None,
        native: bool = False,
    ) -> Path:
        """Write the vector as an ``fs-control-vector-1`` file."""

        return self.to_file(
            state_fingerprint=state_fingerprint,
            registry_fingerprint=registry_fingerprint,
            native=native,
        ).write(path)

    @classmethod
    def load(cls, path: Union[str, Path], space: ControlSpace) -> "ControlVector":
        """Read an ``fs-control-vector-1`` file onto ``space``."""

        return cls.from_file(ControlVectorFile.read(path), space)

    def __repr__(self) -> str:
        return f"ControlVector(size={self.size}, blocks={list(self.space.blocks)})"


# ---------------------------------------------------------------------------
# control state
# ---------------------------------------------------------------------------


class ControlState:
    """Complete baseline over every block of a space (``fs-control-state-1``).

    Values live in the Sauce layout (frozen DOFs included, complex blocks
    interleaved).  Use :meth:`vector` to take the active slice on a
    restricted space and :meth:`with_update` to write one back.
    """

    def __init__(self, space: ControlSpace, values: Any):
        if not isinstance(space, ControlSpace):
            raise TypeError("ControlState requires a ControlSpace")
        array = np.asarray(values)
        if np.iscomplexobj(array):
            raise ValueError("control states are real; complex blocks interleave")
        array = np.array(array, dtype=np.float64).reshape(-1)
        if array.size != space.full_size:
            raise ValueError(
                f"state has {array.size} entries; the space layout has "
                f"{space.full_size}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("control state values must be finite")
        self.space = space
        self._values = array

    @property
    def values(self) -> np.ndarray:
        """Return the Sauce-layout values (a view)."""

        return self._values

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)

    @property
    def size(self) -> int:
        return int(self._values.size)

    @classmethod
    def from_simulation(cls, space: ControlSpace) -> "ControlState":
        """Return the authored baseline of a bound space.

        Material blocks start at zero (the bind installs zero coefficients);
        interface blocks carry the rbf coefficients; source positions carry
        the acquisition coordinates.  Mechanism and signature-derivative
        blocks are zero and signatures are ``1 + 0j``; Sauce's
        ``controls.state_output`` (see :meth:`from_file`) is authoritative for
        those.
        """

        values = np.zeros(space.full_size, dtype=np.float64)
        for block, sl in zip(space.resolved_blocks, space.full_slices.values()):
            if block.baseline is not None:
                values[sl] = block.baseline
        return cls(space, values)

    @classmethod
    def from_blocks(
        cls, space: ControlSpace, blocks: Mapping[str, Any]
    ) -> "ControlState":
        """Build a state from per-block arrays (complex allowed)."""

        full = np.zeros(space.full_size, dtype=np.float64)
        slices = space.full_slices
        seen: set = set()
        for name, values in blocks.items():
            block = space.block(name)
            space._assign_block(full, slices, block, values, seen)
        missing = [b.name for b in space.resolved_blocks if b.name not in seen]
        if missing:
            raise ValueError(f"state is missing block(s) {missing}")
        return cls(space, full)

    @classmethod
    def from_file(
        cls, file: Union[ControlStateFile, str, Path], space: ControlSpace
    ) -> "ControlState":
        """Read an ``fs-control-state-1`` file onto ``space``.

        Blocks are matched by qualified name.  Support masks in the file are
        adopted when ``space`` freezes nothing.
        """

        if not isinstance(file, ControlStateFile):
            file = ControlStateFile.read(file)
        if file.support and not space.support_masks():
            space = space.with_support(
                {n: m for n, m in file.support.items() if n in space.blocks},
                min_support=file.support_min_support,
            )
        full = np.zeros(space.full_size, dtype=np.float64)
        for name, sl in space.full_slices.items():
            if name not in file.blocks:
                raise ValueError(f"state file has no block {name!r}")
            values = file[name]
            if values.size != sl.stop - sl.start:
                raise ValueError(
                    f"block {name!r} has {values.size} entries in the file; "
                    f"the space has {sl.stop - sl.start}"
                )
            full[sl] = values
        return cls(space, full)

    @classmethod
    def from_manifest(
        cls, manifest: ControlRegistryManifest, space: Optional[ControlSpace] = None
    ) -> "ControlState":
        """Return the manifest baseline on ``space`` (default: the manifest space)."""

        if space is None:
            space = ControlSpace.from_manifest(manifest)
        blocks = manifest.unpack_state()
        return cls.from_blocks(space, {name: blocks[name] for name in space.blocks})

    def to_file(self) -> ControlStateFile:
        """Return the ``fs-control-state-1`` representation."""

        blocks = {name: self._values[sl] for name, sl in self.space.full_slices.items()}
        return ControlStateFile(
            blocks,
            support=self.space.support_masks(),
            support_min_support=self.space.min_support,
        )

    def save(self, path: Union[str, Path]) -> Path:
        """Write the state as an ``fs-control-state-1`` file."""

        return self.to_file().write(path)

    @classmethod
    def load(cls, path: Union[str, Path], space: ControlSpace) -> "ControlState":
        """Read an ``fs-control-state-1`` file onto ``space``."""

        return cls.from_file(ControlStateFile.read(path), space)

    def __getitem__(self, key: str) -> Any:
        blocks = self.space._select(key)
        out = {}
        slices = self.space.full_slices
        for block in blocks:
            values = np.array(self._values[slices[block.name]], copy=True)
            out[block.name] = _deinterleave(values) if block.complex else values
        if len(blocks) == 1:
            return out[blocks[0].name]
        return out

    def blocks(self) -> Dict[str, np.ndarray]:
        """Return ``qualified name -> Sauce-layout block`` arrays."""

        return {
            name: np.array(self._values[sl], copy=True)
            for name, sl in self.space.full_slices.items()
        }

    def vector(self, subspace: Optional[ControlSpace] = None) -> ControlVector:
        """Return the active slice of this state on ``subspace``."""

        subspace = self.space if subspace is None else subspace
        full = np.zeros(subspace.full_size, dtype=np.float64)
        mine = self.space.full_slices
        for name, sl in subspace.full_slices.items():
            if name not in mine:
                raise ValueError(f"state has no block {name!r}")
            if mine[name].stop - mine[name].start != sl.stop - sl.start:
                raise ValueError(f"block {name!r} differs in size between spaces")
            full[sl] = self._values[mine[name]]
        return subspace.from_sauce_vector(full)

    def with_update(self, *args: Any) -> "ControlState":
        """Return a new state with the vector's active DOFs written back.

        Accepts ``with_update(vector)`` or ``with_update(space, values)``.
        Frozen DOFs of the vector's space keep their state values.
        """

        if len(args) == 1:
            vector = args[0]
            if not isinstance(vector, ControlVector):
                raise TypeError("with_update(vector) requires a ControlVector")
        elif len(args) == 2:
            space, values = args
            vector = (
                values
                if isinstance(values, ControlVector)
                else ControlVector(values, space)
            )
            if not vector.space.equivalent(space):
                raise ValueError("vector does not live on the given space")
        else:
            raise TypeError("with_update takes a vector or (space, values)")
        values = np.array(self._values, copy=True)
        mine = self.space.full_slices
        sub = vector.space
        full = sub.to_sauce_vector(vector)
        for block, sl in zip(sub.resolved_blocks, sub.full_slices.values()):
            if block.name not in mine:
                raise ValueError(f"state has no block {block.name!r}")
            target = mine[block.name]
            if target.stop - target.start != block.size:
                raise ValueError(f"block {block.name!r} differs in size between spaces")
            mask = sub._mask_of(block)
            segment = values[target]
            segment[mask] = full[sl][mask]
            values[target] = segment
        return ControlState(self.space, values)

    def to_xarray(self, key: Optional[str] = None) -> Any:
        """Render material blocks as xarray (see :meth:`ControlVector.to_xarray`)."""

        return self.vector(self.space.without_support()).to_xarray(key)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, ControlState):
            return NotImplemented
        return self.space.blocks == other.space.blocks and np.array_equal(
            self._values, other._values
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"ControlState(size={self.size}, blocks={list(self.space.blocks)})"
