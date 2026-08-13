"""Layered seismic model container and serialization logic.

``LayeredModel`` is the main user-facing model builder for stratigraphic
velocity/property models. It owns ordered surfaces, layers, fractures,
boreholes, solver-contract export, and reconstruction
from saved FrequenSolve payloads.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import xarray as xr

from frequensolve.mesh.mesh_generators import HexMeshGenerator, TetMeshGenerator
from frequensolve.model.attenuation import AttenuationConfig
from frequensolve.model.model import ModelBase, ModelSubdomain
from frequensolve.model.property import (
    canonical_property_name,
)
from frequensolve.units import unit_expression, value_and_units_to_fs
from frequensolve.util.class_registry import register_class
from frequensolve.util.mixins import (
    ExportContext,
    merge_extra,
)
from frequensolve.util.named_list import NamedList
from frequensolve.util.physics import model_dimension

from ._utils import (
    _coerce_domain_limits,
    _convert_dataarray_units,
    _convert_surface_value,
    _convert_units,
    _property_units,
)
from .authoring import LayeredAuthoringMixin
from .borehole import (
    Borehole,
)
from .sampling import LayeredSamplingMixin
from .surfaces import (
    Fracture,
    Layer,
    SimpleSurface,
    _is_fracture_surface_payload,
    _model_surface_from_fs,
)

__all__ = ["LayeredModel"]


@register_class
@dataclass(kw_only=True)
class LayeredModel(LayeredAuthoringMixin, LayeredSamplingMixin, ModelBase):
    """Layered seismic model with ordered surfaces and material intervals.

    Args:
        name: Model name used in project paths and serialized payloads.
        dimension: Model dimension. Layered models support 2D and 3D.
        x_limits: Physical x-domain limits. Values may be bare floats or
            unit-bearing payloads.
        y_limits: Physical y-domain limits for 3D models. Must be omitted for
            2D models.
        surfaces: Optional initial surface collection.
        boreholes: Optional initial borehole collection.
        ordering: Whether layers are authored from top to bottom or bottom to
            top.
        attenuation_model: Optional model-wide attenuation model name.
        reference_frequency: Optional positive attenuation reference frequency.
            Bare values are interpreted as hertz; Pint quantities and
            unit-bearing mappings may use compatible frequency units.
        extra: Additional serialized fields preserved on round trip.

    Raises:
        ValueError: If the dimension is unsupported, required 3D limits are
            missing, or 2D models are given ``y_limits``.

    Notes:
        A complete layered model has at least two surfaces and at least one
        layer. Authoring normally alternates ``add_surface(...)`` and
        ``add_layer(...)`` so each layer receives an upper and lower surface.
    """

    x_limits: Any
    y_limits: Optional[Any] = None
    surfaces: NamedList = field(default_factory=NamedList)
    boreholes: NamedList = field(default_factory=NamedList)
    ordering: Literal["top_down", "bottom_up"] = "top_down"
    extra: Dict[str, Any] = field(default_factory=dict)

    _last_added: str = "none"
    _surface_names: Set[str] = field(default_factory=set)
    _layer_names: Set[str] = field(default_factory=set)
    _borehole_names: Set[str] = field(default_factory=set)
    _fracture_names: Set[str] = field(default_factory=set)
    _x_units: Optional[str] = field(default=None, init=False, repr=False)
    _y_units: Optional[str] = field(default=None, init=False, repr=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "x_limits" and "_x_units" in self.__dict__:
            object.__setattr__(self, "_x_units", None)
        elif name == "y_limits" and "_y_units" in self.__dict__:
            object.__setattr__(self, "_y_units", None)
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dimension = model_dimension(self.dimension)
        if self.dimension not in {2, 3}:
            raise ValueError("LayeredModel dimension must be 2 or 3")
        self._normalize_domain_limits()
        if self.dimension == 3:
            if self.y_limits is None:
                raise ValueError("3D LayeredModel requires y_limits=[min, max]")
        elif self.y_limits is not None:
            raise ValueError("2D LayeredModel does not accept y_limits")

    def _normalize_domain_limits(self) -> None:
        x_units = self._x_units
        x_limits, detected_x_units = _coerce_domain_limits(
            self.x_limits,
            "x_limits",
        )
        object.__setattr__(self, "x_limits", x_limits)
        object.__setattr__(self, "_x_units", detected_x_units or x_units)
        if self.y_limits is None:
            object.__setattr__(self, "_y_units", None)
        else:
            y_units = self._y_units
            y_limits, detected_y_units = _coerce_domain_limits(
                self.y_limits,
                "y_limits",
            )
            object.__setattr__(self, "y_limits", y_limits)
            object.__setattr__(self, "_y_units", detected_y_units or y_units)

    def x_limits_in(self, units: Optional[Any] = None) -> Tuple[float, float]:
        """Return x-domain limits converted to requested units.

        Args:
            units: Optional target units. When omitted, limits are returned in
                their stored units.

        Returns:
            Tuple ``(xmin, xmax)`` as floats.
        """

        self._normalize_domain_limits()
        target_units = unit_expression(units) if units is not None else None
        return (
            float(_convert_units(self.x_limits[0], self._x_units, target_units)),
            float(_convert_units(self.x_limits[1], self._x_units, target_units)),
        )

    def y_limits_in(self, units: Optional[Any] = None) -> Tuple[float, float]:
        """Return y-domain limits converted to requested units.

        Args:
            units: Optional target units. When omitted, limits are returned in
                their stored units.

        Returns:
            Tuple ``(ymin, ymax)`` as floats.

        Raises:
            ValueError: If called on a 2D model without ``y_limits``.
        """

        self._normalize_domain_limits()
        if self.y_limits is None:
            raise ValueError("LayeredModel has no y_limits")
        target_units = unit_expression(units) if units is not None else None
        return (
            float(_convert_units(self.y_limits[0], self._y_units, target_units)),
            float(_convert_units(self.y_limits[1], self._y_units, target_units)),
        )

    @property
    def surface_names(self) -> List[str]:
        """Return model surface names in their current ordering.

        Returns:
            List of surface names.
        """

        return [surf.name for surf in self.surfaces]

    @property
    def layer_names(self) -> List[str]:
        """Return stratigraphic layer names in their current ordering.

        Returns:
            List of layer names.
        """

        return [layer.name for layer in self.layers]

    @property
    def borehole_names(self) -> List[str]:
        """Return names of boreholes attached to this model.

        Returns:
            List of borehole names.
        """

        return [borehole.name for borehole in self.boreholes]

    @property
    def layers(self) -> NamedList:
        """Return formation layer subdomains.

        Returns:
            ``NamedList`` containing only stratigraphic ``Layer`` objects,
            excluding fracture and borehole material subdomains.
        """

        return NamedList(
            [subdomain for subdomain in self.subdomains if isinstance(subdomain, Layer)]
        )

    @property
    def z_limits(self) -> Tuple[float, float]:
        """Return model depth limits implied by the first and last surfaces.

        Returns:
            Tuple ``(zmin, zmax)`` as floats.

        Raises:
            ValueError: If fewer than two surfaces are available.
        """

        if len(self.surfaces) < 2:
            raise ValueError("LayeredModel requires at least two surfaces for z_limits")
        z0, _ = self.surfaces[0].extrema  # extrema returns values already
        _, z1 = self.surfaces[-1].extrema
        return float(z0), float(z1)

    def z_limits_in(self, units: Optional[Any] = None) -> Tuple[float, float]:
        """Return model depth limits converted to requested units.

        Args:
            units: Optional target units. When omitted, surface units are used.

        Returns:
            Tuple ``(zmin, zmax)`` as floats.

        Raises:
            ValueError: If fewer than two surfaces are available.
        """

        if len(self.surfaces) < 2:
            raise ValueError("LayeredModel requires at least two surfaces for z_limits")
        target_units = unit_expression(units) if units is not None else None
        z0, _ = self.surfaces[0].extrema
        _, z1 = self.surfaces[-1].extrema
        return (
            float(_convert_surface_value(float(z0), self.surfaces[0], target_units)),
            float(_convert_surface_value(float(z1), self.surfaces[-1], target_units)),
        )

    def property_units(self, property: str) -> Optional[str]:
        """Return the first declared units for a model property.

        Args:
            property: Property name or alias.

        Returns:
            Unit expression string, or ``None`` if no subdomain declares units
            for the property.
        """

        property = canonical_property_name(property)
        for subdomain in self.subdomains:
            if property not in subdomain.properties:
                continue
            units = _property_units(subdomain.properties[property])
            if units:
                return units
        return None

    def convert_property_units(
        self,
        data: xr.DataArray,
        property: str,
        units: Optional[Any],
    ) -> xr.DataArray:
        """Convert sampled property data to requested display units.

        Args:
            data: Sampled property data.
            property: Property name or alias used to infer source units when
                ``data`` has no units attribute.
            units: Optional target units. When omitted, the property's stored
                units are used.

        Returns:
            Data array with values converted and unit metadata updated.
        """

        target_units = unit_expression(units) if units is not None else None
        source_units = data.attrs.get("units") or self.property_units(property)
        if target_units is None:
            target_units = source_units
        return _convert_dataarray_units(data, source_units, target_units)

    def extreme_values(
        self,
        property: str,
        units: Optional[Any] = None,
    ) -> Tuple[float, float]:
        """Return global minimum and maximum for a property across subdomains.

        Args:
            property: Property name or alias.
            units: Optional units for returned extrema.

        Returns:
            Tuple ``(minimum, maximum)`` as floats.

        Raises:
            ValueError: If no subdomain declares the requested property.
        """

        property = canonical_property_name(property)
        target_units = (
            unit_expression(units)
            if units is not None
            else self.property_units(property)
        )
        vmin = 1.0e8
        vmax = -1.0e8
        for subdomain in self.subdomains:
            if property not in subdomain.properties:
                continue
            prop = subdomain.properties[property]
            prop_min, prop_max = prop.extrema
            source_units = _property_units(prop)
            prop_min = float(prop_min)
            prop_max = float(prop_max)
            prop_min = float(_convert_units(prop_min, source_units, target_units))
            prop_max = float(_convert_units(prop_max, source_units, target_units))
            vmin = min(prop_min, vmin)
            vmax = max(prop_max, vmax)
        if vmin == 1.0e8:
            raise ValueError(f"Property '{property}' not found in any layer")
        return vmin, vmax

    def hex_mesh_generator(self, n: Optional[List[int]] = None) -> HexMeshGenerator:
        """Create a hexahedral mesh generator covering the model bounds.

        Args:
            n: Optional element counts passed to ``HexMeshGenerator``.

        Returns:
            Configured hexahedral mesh generator.

        Raises:
            ValueError: If the model does not yet have enough surfaces to
                determine bounds.
        """

        self._validate_bounds_for_meshing()
        l_bound, u_bound, units = self._mesh_bounds()
        return HexMeshGenerator(l_bound=l_bound, u_bound=u_bound, n=n, units=units)

    def tet_mesh_generator(self, n: Optional[List[int]] = None) -> TetMeshGenerator:
        """Create a tetrahedral mesh generator covering the model bounds.

        Args:
            n: Optional element counts passed to ``TetMeshGenerator``.

        Returns:
            Configured tetrahedral mesh generator.

        Raises:
            ValueError: If the model does not yet have enough surfaces to
                determine bounds.
        """

        self._validate_bounds_for_meshing()
        l_bound, u_bound, units = self._mesh_bounds()
        return TetMeshGenerator(l_bound=l_bound, u_bound=u_bound, n=n, units=units)

    def _mesh_bounds(self) -> Tuple[List[float], List[float], Optional[str]]:
        self._normalize_domain_limits()
        units = self._x_units or self._y_units
        x_limits = self.x_limits_in(units)
        z_limits = self.z_limits_in(units)
        if self.dimension == 2:
            l_bound = [x_limits[0], z_limits[0]]
            u_bound = [x_limits[1], z_limits[1]]
        else:
            y_limits = self.y_limits_in(units)
            l_bound = [x_limits[0], y_limits[0], z_limits[0]]
            u_bound = [x_limits[1], y_limits[1], z_limits[1]]
        return l_bound, u_bound, units

    @classmethod
    def from_fs(cls, data: Dict) -> "LayeredModel":
        """Deserialize a layered model from a solver payload.

        Args:
            data: Serialized layered model payload containing surfaces,
                subdomains, bounds, ordering, and optional boreholes.

        Returns:
            New ``LayeredModel`` instance with surfaces, layers, fractures, and
            boreholes restored.

        Raises:
            ValueError: If the payload is missing required surfaces or layers,
                has too few layer definitions for the surface intervals, or has
                unused layer definitions after reconstruction.
        """
        data = copy.deepcopy(data)
        # Create copy and remove surfaces to pass rest to parent
        surfs = data.pop("surfaces")
        subdomains = data.pop("subdomains")
        boreholes = data.pop("boreholes", [])
        data.pop("fractures", None)
        if len(surfs) < 2:
            raise ValueError("LayeredModel requires at least two surfaces")
        if len(subdomains) == 0:
            raise ValueError("LayeredModel requires at least one layer")

        fracture_mesh_block_ids = {
            surface.get("mesh_block_id")
            for surface in surfs
            if isinstance(surface, Mapping)
            and _is_fracture_surface_payload(surface)
            and surface.get("mesh_block_id") is not None
        }
        formation_subdomains = [
            subdomain
            for subdomain in subdomains
            if subdomain.get("mesh_block_id") not in fracture_mesh_block_ids
        ]
        reserved_subdomains = [
            subdomain
            for subdomain in subdomains
            if subdomain.get("mesh_block_id") in fracture_mesh_block_ids
        ]
        fracture_subdomains = {
            subdomain.get("mesh_block_id"): subdomain
            for subdomain in reserved_subdomains
        }
        emitted_fracture_subdomains: Set[int] = set()
        layer_count = sum(1 for surface in surfs if surface.get("interface", True)) - 1
        if len(formation_subdomains) < layer_count:
            raise ValueError("LayeredModel has fewer layer definitions than intervals")
        layers = formation_subdomains[:layer_count]
        extra_subdomains = formation_subdomains[layer_count:]

        name = data.pop("name", None)
        data.pop("_type", None)
        data.pop("schema", None)
        dimension = data.pop("dimension", None)
        x_limits = data.pop("x_limits", data.pop("xlimits", None))
        y_limits = data.pop("y_limits", data.pop("ylimits", None))
        ordering = data.pop("ordering", "top_down")
        attenuation_payload = data.pop("attenuation", None)
        attenuation = (
            AttenuationConfig.from_fs(attenuation_payload)
            if attenuation_payload is not None
            else None
        )
        model = LayeredModel(
            name=name,
            dimension=dimension,
            x_limits=x_limits,
            y_limits=y_limits,
            ordering=ordering,
            attenuation_model=attenuation.model if attenuation else None,
            reference_frequency=(
                attenuation.reference_frequency if attenuation else None
            ),
        )
        if attenuation is not None:
            model._attenuation_extra = attenuation.extra
        model.extra = data

        def add_model_surface(surface_payload: Mapping[str, Any]) -> None:
            nonlocal model
            surface = _model_surface_from_fs(surface_payload)
            model += surface
            if not isinstance(surface, Fracture) or surface.mesh_block_id is None:
                return
            subdomain_payload = fracture_subdomains.get(surface.mesh_block_id)
            if subdomain_payload is None:
                return
            model += ModelSubdomain.from_fs(subdomain_payload)
            emitted_fracture_subdomains.add(surface.mesh_block_id)

        add_model_surface(surfs[0])

        n_surf = len(surfs)
        isurf = 1
        ilayer = 0
        while isurf < n_surf:
            if ilayer >= len(layers):
                raise ValueError("LayeredModel has more surfaces than layer intervals")
            # Add layer
            model += Layer.from_fs(layers[ilayer])
            ilayer += 1

            # Add any non-interface surfaces (surfaces not between layers)
            while surfs[isurf].get("interface", True) is False:
                add_model_surface(surfs[isurf])
                isurf += 1
                if isurf >= n_surf:
                    raise ValueError("LayeredModel ended with a non-interface surface")

            # Add surface
            add_model_surface(surfs[isurf])
            isurf += 1

        if ilayer != len(layers):
            raise ValueError("LayeredModel has unused layer definitions")

        for subdomain in extra_subdomains:
            model += ModelSubdomain.from_fs(subdomain)
        for mesh_block_id, subdomain in fracture_subdomains.items():
            if mesh_block_id not in emitted_fracture_subdomains:
                model += ModelSubdomain.from_fs(subdomain)
        for borehole in boreholes:
            model += Borehole.from_fs(borehole)

        return model

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict[str, Any]:
        """Serialize the layered model to the solver model contract.

        Args:
            ctx: Optional export context used for project-relative paths and
                HDF5-backed property storage.

        Returns:
            Solver-ready model payload with bounds, surfaces, subdomains,
            ordering, and boreholes.

        Raises:
            ValueError: If the model is incomplete.
        """

        ctx = ctx or ExportContext()
        self._normalize_domain_limits()
        self._validate_complete()

        base_dict = super().to_fs(ctx)
        surfaces = []
        for i, surface in enumerate(self.surfaces):
            payload = surface.to_fs(ctx)
            if i == len(self.surfaces) - 1:
                payload["interface"] = True
            surfaces.append(payload)
        base_dict.update(
            {
                "_type": self.__class__.__name__,
                "x_limits": value_and_units_to_fs(self.x_limits, self._x_units),
                **(
                    {"y_limits": value_and_units_to_fs(self.y_limits, self._y_units)}
                    if self.y_limits is not None
                    else {}
                ),
                "ordering": self.ordering,
                "surfaces": surfaces,
                **(
                    {"boreholes": [borehole.to_fs(ctx) for borehole in self.boreholes]}
                    if self.boreholes
                    else {}
                ),
            }
        )
        return merge_extra(base_dict, self.extra, "LayeredModel")

    def upper_surface(
        self, layer: Optional[Union[str, int, ModelSubdomain]] = None
    ) -> SimpleSurface:
        """Return the upper surface of the model or one layer.

        Args:
            layer: Optional layer selector. May be a layer name, zero-based
                layer index, ``Layer`` object, or subdomain with a matching
                mesh-block id. When omitted, the model's top surface is
                returned according to ``ordering``.

        Returns:
            Matching upper ``SimpleSurface``.

        Raises:
            ValueError: If the model has no surfaces or the selected layer is
                not found.
        """
        if layer is None:
            if not self.surfaces:
                raise ValueError("LayeredModel has no surfaces")
            if self.ordering == "top_down":
                return self.surfaces[0]
            else:
                return self.surfaces[-1]

        if isinstance(layer, int):
            return self._require_layer_surface(self.layers[layer], "upper")
        elif isinstance(layer, str):
            return self._require_layer_surface(self.layers[layer], "upper")
        elif isinstance(layer, Layer):
            return self._require_layer_surface(layer, "upper")
        elif isinstance(layer, ModelSubdomain):
            mesh_block_id = layer.mesh_block_id
            for candidate in self.layers:
                if candidate.mesh_block_id == mesh_block_id:
                    return self._require_layer_surface(candidate, "upper")
        raise ValueError(f"Layer not found: {layer}")

    def lower_surface(
        self, layer: Optional[Union[str, int, ModelSubdomain]] = None
    ) -> SimpleSurface:
        """Return the lower surface of the model or one layer.

        Args:
            layer: Optional layer selector. May be a layer name, zero-based
                layer index, ``Layer`` object, or subdomain with a matching
                mesh-block id. When omitted, the model's bottom surface is
                returned according to ``ordering``.

        Returns:
            Matching lower ``SimpleSurface``.

        Raises:
            ValueError: If the model has no surfaces or the selected layer is
                not found.
        """
        if layer is None:
            if not self.surfaces:
                raise ValueError("LayeredModel has no surfaces")
            if self.ordering == "top_down":
                return self.surfaces[-1]
            else:
                return self.surfaces[0]

        if isinstance(layer, int):
            return self._require_layer_surface(self.layers[layer], "lower")
        elif isinstance(layer, str):
            return self._require_layer_surface(self.layers[layer], "lower")
        elif isinstance(layer, Layer):
            return self._require_layer_surface(layer, "lower")
        elif isinstance(layer, ModelSubdomain):
            mesh_block_id = layer.mesh_block_id
            for candidate in self.layers:
                if candidate.mesh_block_id == mesh_block_id:
                    return self._require_layer_surface(candidate, "lower")
        raise ValueError(f"Layer not found: {layer}")

    @staticmethod
    def _require_layer_surface(
        layer: Layer, bound: Literal["upper", "lower"]
    ) -> SimpleSurface:
        surface = layer.upper if bound == "upper" else layer.lower
        if surface is None:
            raise ValueError(f"Layer '{layer.name}' has no {bound} surface")
        return surface

    def _validate_bounds_for_meshing(self) -> None:
        if len(self.surfaces) < 2:
            raise ValueError("At least two surfaces are required before meshing")

    def _validate_complete(self) -> None:
        if len(self.surfaces) < 2:
            raise ValueError("LayeredModel requires at least two surfaces")
        if not self.layers:
            raise ValueError("LayeredModel requires at least one layer")
        for layer in self.layers:
            if layer.upper is None or layer.lower is None:
                raise ValueError(
                    f"Layer '{layer.name}' must be bounded by upper and lower surfaces"
                )
