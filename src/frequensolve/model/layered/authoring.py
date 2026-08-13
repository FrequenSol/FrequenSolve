"""High-level builder methods mixed into :class:`LayeredModel`.

The public ``LayeredModel`` class inherits these methods so users can build a
model incrementally with surfaces, layers, fractures, boreholes, and explicit
subdomains while the implementation keeps names, ordering, and legacy keyword
aliases consistent.
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, TypeVar, Union

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from frequensolve.model.model import ModelSubdomain
from frequensolve.util.named_list import NamedList

from ._utils import _convert_units, _property_units
from .borehole import (
    Borehole,
    BoreholeAnnularPadding,
    BoreholeLayer,
    BoreholePart,
    BoreholePlug,
    BoreholeSurface,
)
from .surfaces import Fracture, Layer, SimpleSurface

_LayeredModelT = TypeVar("_LayeredModelT", bound="LayeredAuthoringMixin")


class LayeredAuthoringMixin:
    """Builder API for adding layered-model geometry and material domains.

    The mixin owns the high-level authoring workflow used by ``LayeredModel``:
    users alternate surfaces and layers, then optionally add boreholes,
    fractures, or explicit subdomains. Names are normalized to stay unique
    within their model collections.
    """

    subdomains: NamedList

    def add_layer(
        self,
        name: Optional[str] = None,
        mesh_block_id: int = -1,
        physics: Optional[str] = None,
        properties: Optional[dict] = None,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        fields: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """Add one stratigraphic material layer to the model.

        Args:
            name: Optional layer name. If omitted or duplicated, the model
                assigns a unique name.
            mesh_block_id: Solver mesh-block id for this layer. ``-1`` leaves
                the id unlabeled until model export.
            physics: Optional material physics family or solver variant.
            properties: Mapping of property names to scalar, array, file, or
                expression values accepted by ``Property``.
            fields: Mapping of independent named data fields referenced by
                property-expression ``field`` nodes.
            grid: Default xarray grid metadata used when coercing file-backed
                or ungridded property values.
            units: Default units applied to properties that do not declare
                units themselves.
            system: Optional coordinate-system name for layer property grids.
                The legacy ``coordinate_system`` keyword is accepted as an
                alias.
            **kwargs: Legacy aliases and extra fields. ``xarr`` is accepted as
                a grid alias; ``frame`` is rejected because material
                coordinates are physical.

        Raises:
            TypeError: If unsupported legacy ``frame`` metadata is supplied.
            ValueError: If ``system`` and ``coordinate_system`` disagree, or if
                the layer order would violate the surface/layer authoring
                rules.
        """

        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")
        if "frame" in kwargs:
            raise TypeError(
                "LayeredModel.add_layer frame is no longer supported; layer coordinates are physical"
            )
        coordinate_system = kwargs.pop("coordinate_system", None)
        if coordinate_system is not None:
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system

        name = self._get_unique_name(name, self._layer_names)
        self._layer_names.add(name)

        layer = Layer(
            name=name,
            mesh_block_id=mesh_block_id,
            physics=physics,
            properties=properties,
            fields=fields,
            grid=grid,
            units=units,
            system=system,
        )
        self += layer

    def add_surface(
        self,
        depth: Optional[Union[float, str, Path, xr.DataArray, Dict[str, Any]]] = None,
        name: Optional[str] = None,
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        cutting: bool = False,
        **kwargs,
    ):
        """Add a depth surface to the model.

        Args:
            depth: Surface depth as a scalar, Pint quantity, path, serialized
                property payload, or ``xarray.DataArray``.
            name: Optional surface name. If omitted or duplicated, the model
                assigns a unique name.
            grid: Optional grid metadata for file-backed or ungridded depth
                values.
            scale: Multiplicative scale applied to loaded depth values.
            units: Optional depth units. Pint quantities may also carry units
                directly.
            system: Optional coordinate-system name for the surface depth
                coordinates. The legacy ``coordinate_system`` keyword is
                accepted as an alias.
            cutting: Whether to label the surface as a model-cutting surface
                for the mesher. Use :meth:`truncate` to also discard geometry
                below the surface and make it the lower model boundary.
            **kwargs: Legacy aliases and extra fields. ``xarr`` is accepted as
                a grid alias and ``z`` as a depth alias.

        Raises:
            TypeError: If unexpected keyword arguments are supplied.
            ValueError: If no depth is provided or coordinate-system aliases
                conflict.
        """

        interface = len(self.surfaces) == 0

        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")
        if depth is None and "z" in kwargs:
            depth = kwargs.pop("z")
        if "frame" in kwargs:
            raise TypeError(
                "LayeredModel.add_surface frame is no longer supported; surface coordinates are physical"
            )
        coordinate_system = kwargs.pop("coordinate_system", None)
        if coordinate_system is not None:
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(
                f"Unexpected LayeredModel.add_surface arguments: {unexpected}"
            )
        if depth is None:
            raise ValueError("LayeredModel.add_surface requires depth")

        name = self._get_unique_name(name, self._surface_names)
        self._surface_names.add(name)

        surface = SimpleSurface(
            name=name,
            interface=interface,
            depth=depth,
            grid=grid,
            scale=scale,
            units=units,
            system=system,
            cutting=cutting,
        )
        self += surface

    def truncate(
        self: _LayeredModelT,
        depth: Optional[Union[float, str, Path, xr.DataArray, Dict[str, Any]]] = None,
        *,
        name: Optional[str] = "cutting_surface",
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> _LayeredModelT:
        """Truncate the model below a new cutting surface.

        Surfaces proven to be nowhere shallower than the cut are discarded,
        including an existing surface coincident with the cut. Surfaces that
        cross the cut are retained so a layered mesher can clip them against
        the serialized ``cutting`` surface. The cut becomes the lower model
        boundary, and formation layers that are wholly below it are removed.

        This method returns an independent deep copy and leaves the source
        model unchanged. Depth increases downward, following the layered-model
        coordinate convention.

        Args:
            depth: Cutting-surface depth in the forms accepted by
                :meth:`add_surface`.
            name: Surface name. Defaults to ``"cutting_surface"``.
            grid: Optional grid metadata for file-backed or ungridded depth
                values.
            scale: Multiplicative scale applied to loaded depth values.
            units: Optional depth units.
            system: Optional coordinate-system name for surface coordinates.
            **kwargs: Legacy aliases accepted by :meth:`add_surface`, including
                ``xarr``, ``z``, and ``coordinate_system``.

        Returns:
            A copied layered model truncated at a new ``SimpleSurface`` with
            ``cutting=True``.

        Raises:
            ValueError: If the model is incomplete, the cut would extend the
                model downward, a retained cutting surface already exists, or
                a surface depth cannot be inspected.
        """

        truncated = copy.deepcopy(self)
        truncated._truncate_inplace(
            depth=depth,
            name=name,
            grid=grid,
            scale=scale,
            units=units,
            system=system,
            **kwargs,
        )
        return truncated

    def add_borehole(
        self,
        name: Optional[str] = None,
        *,
        axis: Optional[Mapping[str, Any]] = None,
        x: Optional[Any] = None,
        y: Optional[Any] = None,
        top: Optional[Any] = None,
        bottom: Optional[Any] = None,
        layers: Optional[List[Union[BoreholePart, Mapping[str, Any]]]] = None,
        surfaces: Optional[List[Union[BoreholeSurface, Mapping[str, Any]]]] = None,
        plugs: Optional[List[Union[BoreholePlug, Mapping[str, Any]]]] = None,
        annular_padding: Optional[
            Union[BoreholeAnnularPadding, Mapping[str, Any]]
        ] = None,
        **kwargs,
    ) -> Borehole:
        """Add a vertical borehole and any declared material subdomains.

        ``layers`` may be ``BoreholeLayer`` objects or dictionaries. A layer
        value can also include ``physics`` and ``properties``; those fields
        create the corresponding ``ModelSubdomain`` while the borehole layer
        keeps only the geometry and mesh-block fields required by the solver.
        When a layer does not include ``properties``, its ``mesh_block_id`` must
        already reference an existing model subdomain. ``plugs`` follow the same
        material rule and describe local axial obstructions inside 2D
        boreholes. The current 3D solver path supports vertical boreholes only
        and does not support plugs/tool-body intervals.

        Args:
            name: Optional borehole name. If omitted or duplicated, the model
                assigns a unique name.
            axis: Mapping that describes the borehole axis, such as
                ``{"x": value}`` in 2D or ``{"x": value, "y": value}`` in 3D.
            x: Convenience axis x-coordinate.
            y: Convenience axis y-coordinate required for 3D boreholes.
            top: Top extent as a surface reference, one-based surface index,
                ``SimpleSurface``, mapping, or depth-like value.
            bottom: Bottom extent in the same forms as ``top``.
            layers: Radial material layers or already-closed borehole parts.
            surfaces: Optional named cumulative-radius surfaces.
            plugs: Optional plug/tool-body intervals inside the borehole.
            annular_padding: Optional 3D annular padding block with ``n``,
                ``outer_radius``, and optional ``power``.
            **kwargs: Legacy aliases. ``parts`` aliases ``layers`` and
                ``extent`` may provide ``top``/``bottom``.

        Returns:
            The created ``Borehole`` attached to this model.

        Raises:
            TypeError: If a layer, surface, or plug specification has an
                unsupported type.
            ValueError: If the axis is ambiguous or missing, unsupported 3D
                plug geometry is requested, or borehole material domains cannot
                be resolved.
        """

        if "parts" in kwargs:
            if layers is not None:
                raise ValueError("Specify only one of borehole layers or parts")
            layers = kwargs.pop("parts")
        if "extent" in kwargs:
            if top is not None or bottom is not None:
                raise ValueError("Specify either extent or top/bottom, not both")
            extent = dict(kwargs.pop("extent"))
            top = extent.get("top")
            bottom = extent.get("bottom")
        p_enrich = kwargs.pop("p_enrich", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            warnings.warn(
                f"Possibly unexpected LayeredModel.add_borehole arguments: {unexpected}"
            )
        if axis is not None:
            axis_payload = copy.deepcopy(dict(axis))
            if x is not None or y is not None:
                raise ValueError("Specify either axis or x/y coordinates, not both")
        else:
            axis_payload = {}
            if x is not None:
                axis_payload["x"] = x
            if y is not None:
                axis_payload["y"] = y
        if not axis_payload:
            raise ValueError("LayeredModel.add_borehole requires axis or x/y")
        if self.dimension == 2 and "y" in axis_payload:
            raise ValueError("2D boreholes accept x, not y")
        if "x" not in axis_payload:
            raise ValueError("Boreholes require axis/x")
        if self.dimension == 3 and "y" not in axis_payload:
            raise ValueError("3D boreholes require axis/y")
        if self.dimension == 3 and plugs:
            raise ValueError("3D borehole meshing does not support plugs")
        if self.dimension != 3 and annular_padding is not None:
            raise ValueError(
                "Borehole annular_padding is supported for 3D boreholes only"
            )
        annular_padding_obj = (
            annular_padding
            if isinstance(annular_padding, BoreholeAnnularPadding)
            else (
                BoreholeAnnularPadding.from_fs(annular_padding)
                if annular_padding is not None
                else None
            )
        )

        if top is None:
            top = self.upper_surface()
        if bottom is None:
            bottom = self.lower_surface()

        name = self._get_unique_name(name or "borehole", self._borehole_names)
        self._borehole_names.add(name)

        next_block = self._next_mesh_block_id()
        borehole_parts = []
        borehole_plugs = []
        material_subdomains = []
        surface_specs = [
            (
                surface
                if isinstance(surface, BoreholeSurface)
                else BoreholeSurface(**surface)
            )
            for surface in surfaces or []
        ]
        surface_by_name = {
            (surface.name or f"{name}_surface_{index + 1}"): surface
            for index, surface in enumerate(surface_specs)
        }

        def default_outer_surface(index: int) -> BoreholeSurface:
            surface_index = (
                index + 1 if surface_specs and surface_specs[0].is_axis() else index
            )
            if surface_index >= len(surface_specs):
                raise ValueError("Borehole layers without r require matching surfaces")
            return surface_specs[surface_index]

        layer_specs = list(layers if layers is not None else [])
        if layers is not None:
            for index, spec in enumerate(layer_specs):
                if isinstance(spec, BoreholePart):
                    continue
                if isinstance(spec, BoreholeLayer):
                    layer_payload = {
                        "name": spec.name,
                        "mesh_block_id": spec.mesh_block_id,
                        **spec.extra,
                    }
                    if spec.physics is not None:
                        layer_payload["physics"] = spec.physics
                    if spec.properties is not None:
                        layer_payload["properties"] = spec.properties
                    if spec.grid is not None:
                        layer_payload["grid"] = spec.grid
                    if spec.units is not None:
                        layer_payload["units"] = spec.units
                    if spec.system is not None:
                        layer_payload["system"] = spec.system
                    if spec.subdomain_name is not None:
                        layer_payload["subdomain_name"] = spec.subdomain_name
                    if spec.inner_surface is not None:
                        layer_payload["inner_surface"] = spec.inner_surface
                    if spec.outer_surface is not None:
                        layer_payload["outer_surface"] = spec.outer_surface
                    spec = layer_payload
                if not isinstance(spec, Mapping):
                    raise TypeError(
                        "Borehole layers must be BoreholeLayer, BoreholePart, "
                        "or mapping objects"
                    )
                if "r" not in spec:
                    outer_name = spec.get("outer_surface")
                    if outer_name is not None:
                        if outer_name not in surface_by_name:
                            raise ValueError(
                                "Borehole layer references unknown outer_surface "
                                f"'{outer_name}'"
                            )
                        surface = surface_by_name[outer_name]
                    else:
                        surface = default_outer_surface(index)
                    layer_specs[index] = {
                        **dict(spec),
                        "r": surface.r,
                    }
        for spec in layer_specs:
            part, subdomain, next_block = self._coerce_borehole_part(
                name, spec, next_block
            )
            borehole_parts.append(part)
            if subdomain is not None:
                material_subdomains.append(subdomain)
        if not surface_specs:
            surface_specs = Borehole._surfaces_from_parts(borehole_parts)
        for spec in plugs or []:
            plug, subdomain, next_block = self._coerce_borehole_plug(
                name,
                spec,
                next_block,
            )
            borehole_plugs.append(plug)
            if subdomain is not None:
                material_subdomains.append(subdomain)

        for subdomain in material_subdomains:
            self._add_unique_subdomain(subdomain)
        missing_domains = [
            part.mesh_block_id
            for part in borehole_parts
            if not any(
                subdomain.mesh_block_id == part.mesh_block_id
                for subdomain in self.subdomains
            )
        ]
        missing_domains.extend(
            plug.mesh_block_id
            for plug in borehole_plugs
            if not any(
                subdomain.mesh_block_id == plug.mesh_block_id
                for subdomain in self.subdomains
            )
        )
        if missing_domains:
            missing = ", ".join(str(value) for value in missing_domains)
            raise ValueError(
                "Borehole part/plug mesh_block_id must reference a model subdomain; "
                f"missing: {missing}"
            )

        borehole = Borehole(
            name=name,
            axis=axis_payload,
            extent={"top": top, "bottom": bottom},
            layers=borehole_parts,
            surfaces=surface_specs,
            plugs=borehole_plugs,
            model=self,
            annular_padding=annular_padding_obj,
            **({"p_enrich": p_enrich} if p_enrich is not None else {}),
            **kwargs,
        )
        self.boreholes.append(borehole)
        return borehole

    def add_fracture(
        self,
        name: Optional[str] = None,
        *,
        depth: Any,
        gap: Union[xr.DataArray, Mapping[str, Any], ArrayLike],
        mesh_block_id: Optional[int] = None,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        interface: Optional[bool] = None,
        physics: Optional[str] = None,
        properties: Optional[dict] = None,
        property_grid: Optional[xr.DataArray] = None,
        property_units: Optional[Any] = None,
        property_system: Optional[str] = None,
        subdomain_name: Optional[str] = None,
        **kwargs,
    ) -> Fracture:
        """Add a curve-based fracture opened by the layered mesh generator.

        ``gap`` is the fracture aperture as a one-dimensional curve, usually an
        ``xarray.DataArray`` over the horizontal coordinate. ``depth`` locates
        the fracture centerline. Fractures are ordered model surfaces, so adding
        one after a layer can close that layer just like ``add_surface(...)``.
        When ``properties`` are supplied, a matching material ``ModelSubdomain``
        is created for the opened fracture interval.

        Args:
            name: Optional fracture name. If omitted or duplicated, the model
                assigns a unique name.
            depth: Fracture center depth as a scalar, Pint quantity, property
                payload, file reference, or ``xarray.DataArray``.
            gap: One-dimensional aperture curve as an array, property payload,
                or array-like value.
            mesh_block_id: Optional solver mesh-block id for the opened
                fracture material interval.
            grid: Grid metadata for depth and gap values when needed.
            units: Units for depth and gap values.
            system: Coordinate-system name for the fracture geometry.
            interface: Whether this fracture should be treated as an ordering
                interface between layers.
            physics: Optional physics/material family for the optional
                fracture subdomain.
            properties: Optional material properties for the opened fracture
                subdomain.
            property_grid: Default grid metadata for fracture material
                properties.
            property_units: Default units for fracture material properties.
            property_system: Coordinate-system name for fracture material
                property grids.
            subdomain_name: Optional name for the generated material subdomain.
            **kwargs: Extra serialized fracture fields preserved on export.

        Returns:
            The created ``Fracture`` attached to this model.

        Raises:
            ValueError: If generated or supplied mesh-block ids collide, or if
                the fracture gap is not one-dimensional.
        """

        name = self._get_unique_name(
            name or "fracture", self._surface_names | self._fracture_names
        )
        self._surface_names.add(name)
        self._fracture_names.add(name)
        if mesh_block_id is None and properties is not None:
            mesh_block_id = self._next_mesh_block_id()
        if interface is None:
            interface = len(self.surfaces) == 0
        fracture = Fracture(
            name=name,
            mesh_block_id=mesh_block_id,
            depth=depth,
            gap=gap,
            grid=grid,
            units=units,
            system=system,
            interface=interface,
            physics=physics,
            properties=properties,
            property_grid=property_grid,
            property_units=property_units,
            property_system=property_system,
            subdomain_name=subdomain_name,
            **kwargs,
        )
        self += fracture
        return fracture

    def __iadd__(self, other):
        if isinstance(other, Fracture):
            if any(
                other.mesh_block_id is not None
                and surface.mesh_block_id == other.mesh_block_id
                for surface in self.surfaces
                if isinstance(surface, Fracture)
            ):
                raise ValueError(
                    f"Fracture mesh block id {other.mesh_block_id} is already used"
                )
            self._add_fracture_subdomain(other)
            self.surfaces.append(other)
            if len(self.surfaces) > 1:
                if self.ordering == "top_down":
                    self.layers[-1].lower = other
                else:
                    self.layers[-1].upper = other
            self._fracture_names.add(other.name)
            self._last_added = "surface"

        elif isinstance(other, SimpleSurface):
            self.surfaces.append(other)

            # Build layer-to-surface mapping
            if len(self.surfaces) > 1:
                if self.ordering == "top_down":
                    self.layers[-1].lower = other
                else:
                    self.layers[-1].upper = other
            self._last_added = "surface"

        elif isinstance(other, Layer):
            # Check that surfaces sandwiching layers
            if len(self.surfaces) == 0:
                raise ValueError("Must add at least one surface before adding layers")
            if self._last_added == "layer":
                raise ValueError("Must add a surface between consecutive layers")

            # Add layer
            self.add_subdomain(other)
            if len(self.surfaces) > 1:
                self.surfaces[-1].interface = True

            # Build layer-to-surface mapping
            if self.ordering == "top_down":
                if len(self.layers) == 1:
                    prev_lower = self.upper_surface()
                else:
                    prev_lower = self.lower_surface(self.layers[-2])

                self.layers[-1].upper = prev_lower
            else:
                if len(self.layers) == 1:
                    prev_upper = self.lower_surface()
                else:
                    prev_upper = self.upper_surface(self.layers[-2])

                self.layers[-1].lower = prev_upper
            self._last_added = "layer"
        elif isinstance(other, ModelSubdomain):
            self.add_subdomain(other)
        elif isinstance(other, Borehole):
            other._model = self
            self.boreholes.append(other)
            self._borehole_names.add(other.name)
        else:
            raise ValueError(f"Cannot add {type(other)} to LayeredModel")
        return self

    def _truncate_inplace(
        self,
        depth: Optional[Union[float, str, Path, xr.DataArray, Dict[str, Any]]] = None,
        *,
        name: Optional[str] = "cutting_surface",
        grid: Optional[xr.DataArray] = None,
        scale: float = 1.0,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> SimpleSurface:
        self._validate_complete()

        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")
        if depth is None and "z" in kwargs:
            depth = kwargs.pop("z")
        if "frame" in kwargs:
            raise TypeError(
                "LayeredModel.truncate frame is no longer supported; "
                "surface coordinates are physical"
            )
        coordinate_system = kwargs.pop("coordinate_system", None)
        if coordinate_system is not None:
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected LayeredModel.truncate arguments: {unexpected}")
        if depth is None:
            raise ValueError("LayeredModel.truncate requires depth")

        cutting_surface = SimpleSurface(
            name=name or "cutting_surface",
            interface=True,
            cutting=True,
            depth=depth,
            grid=grid,
            scale=scale,
            units=units,
            system=system,
        )
        lower_boundary = self.lower_surface()
        if self._surface_is_fully_below(cutting_surface, lower_boundary):
            raise ValueError(
                "Cutting surface is fully below the current model boundary; "
                "truncate cannot extend the model"
            )

        discarded = [
            surface
            for surface in self.surfaces
            if self._surface_is_fully_below(
                surface,
                cutting_surface,
                include_coincident=True,
            )
        ]
        discarded_ids = {id(surface) for surface in discarded}
        retained = [
            surface for surface in self.surfaces if id(surface) not in discarded_ids
        ]
        if any(getattr(surface, "cutting", False) for surface in retained):
            raise ValueError(
                "LayeredModel already has a retained cutting surface; "
                "a new cut must fully replace it"
            )

        cutting_surface.name = self._get_unique_name(
            cutting_surface.name,
            {surface.name for surface in retained},
        )
        lower_boundary_retained = any(surface is lower_boundary for surface in retained)
        new_surfaces = (
            retained + [cutting_surface]
            if self.ordering == "top_down"
            else [cutting_surface] + retained
        )
        boundaries = [
            surface
            for index, surface in enumerate(new_surfaces)
            if index in {0, len(new_surfaces) - 1}
            or (
                surface.interface
                and not (lower_boundary_retained and surface is lower_boundary)
            )
        ]
        required_layer_count = len(boundaries) - 1
        existing_layers = list(self.layers)
        if required_layer_count < 1:
            raise ValueError(
                "Cutting surface removes the entire model; at least one layer "
                "must remain above the cut"
            )
        if required_layer_count > len(existing_layers):
            raise ValueError(
                "Cutting surface does not truncate the current lower model boundary"
            )

        retained_layers = (
            existing_layers[:required_layer_count]
            if self.ordering == "top_down"
            else existing_layers[-required_layer_count:]
        )
        retained_layer_ids = {id(layer) for layer in retained_layers}
        discarded_fracture_block_ids = {
            surface.mesh_block_id
            for surface in discarded
            if isinstance(surface, Fracture) and surface.mesh_block_id is not None
        }

        boreholes_to_rebind = []
        for borehole in self.boreholes:
            try:
                top_surface = self._surface_from_reference(borehole.extent["top"])
            except (KeyError, TypeError, ValueError, IndexError):
                top_surface = None
            if top_surface is not None and id(top_surface) in discarded_ids:
                raise ValueError(
                    f"Cannot truncate model: borehole '{borehole.name}' starts "
                    f"on discarded surface '{top_surface.name}'"
                )
            try:
                bottom_surface = self._surface_from_reference(borehole.extent["bottom"])
            except (KeyError, TypeError, ValueError, IndexError):
                bottom_surface = None
            if bottom_surface is lower_boundary or (
                bottom_surface is not None and id(bottom_surface) in discarded_ids
            ):
                boreholes_to_rebind.append(borehole)

        if lower_boundary_retained:
            lower_boundary.interface = False
        self.surfaces = NamedList(new_surfaces)
        self.subdomains = NamedList(
            [
                subdomain
                for subdomain in self.subdomains
                if (
                    not isinstance(subdomain, Layer)
                    or id(subdomain) in retained_layer_ids
                )
                and subdomain.mesh_block_id not in discarded_fracture_block_ids
            ]
        )

        boundaries = [
            surface
            for index, surface in enumerate(self.surfaces)
            if index in {0, len(self.surfaces) - 1} or surface.interface
        ]
        for index, layer in enumerate(retained_layers):
            if self.ordering == "top_down":
                layer.upper = boundaries[index]
                layer.lower = boundaries[index + 1]
            else:
                layer.lower = boundaries[index]
                layer.upper = boundaries[index + 1]

        for borehole in boreholes_to_rebind:
            borehole.extent["bottom"] = cutting_surface

        self._surface_names = {surface.name for surface in self.surfaces}
        self._fracture_names = {
            surface.name for surface in self.surfaces if isinstance(surface, Fracture)
        }
        self._layer_names = {layer.name for layer in retained_layers}
        self._last_added = "surface"
        return cutting_surface

    @staticmethod
    def _surface_is_fully_below(
        surface: SimpleSurface,
        cutting_surface: SimpleSurface,
        *,
        include_coincident: bool = False,
    ) -> bool:
        surface_data = surface.depth.data
        cutting_data = cutting_surface.depth.data
        if surface_data is None or cutting_data is None:
            missing = surface.name if surface_data is None else cutting_surface.name
            raise ValueError(
                f"Cannot truncate using surface '{missing}': its depth is not "
                "materialized"
            )

        target_units = _property_units(cutting_surface.depth) or _property_units(
            surface.depth
        )
        surface_values = _convert_units(
            np.asarray(surface_data.values),
            _property_units(surface.depth),
            target_units,
        )
        cutting_values = _convert_units(
            np.asarray(cutting_data.values),
            _property_units(cutting_surface.depth),
            target_units,
        )

        same_grid = surface_data.dims == cutting_data.dims and all(
            surface_data.coords[dim].attrs.get("units")
            == cutting_data.coords[dim].attrs.get("units")
            and np.array_equal(
                surface_data.coords[dim].values,
                cutting_data.coords[dim].values,
            )
            for dim in surface_data.dims
        )
        if surface.depth.is_constant or cutting_surface.depth.is_constant or same_grid:
            comparison = (
                surface_values >= cutting_values
                if include_coincident
                else surface_values > cutting_values
            )
            return bool(np.all(comparison))

        surface_min, _ = surface.extrema
        _, cutting_max = cutting_surface.extrema
        surface_min = _convert_units(
            float(surface_min),
            _property_units(surface.depth),
            target_units,
        )
        cutting_max = _convert_units(
            float(cutting_max),
            _property_units(cutting_surface.depth),
            target_units,
        )
        return bool(
            surface_min >= cutting_max
            if include_coincident
            else surface_min > cutting_max
        )

    def _fracture_subdomain_name(self, fracture: Fracture) -> str:
        return fracture.subdomain_name or fracture.name

    def _add_fracture_subdomain(self, fracture: Fracture) -> None:
        if fracture.properties is None:
            return
        if fracture.mesh_block_id is None:
            raise ValueError("Fracture properties require mesh_block_id")
        self._add_unique_subdomain(
            ModelSubdomain(
                name=self._fracture_subdomain_name(fracture),
                mesh_block_id=fracture.mesh_block_id,
                physics=fracture.physics,
                properties=fracture.properties,
                grid=fracture.property_grid,
                units=fracture.property_units,
                system=fracture.property_system,
            )
        )

    def _add_unique_subdomain(self, subdomain: ModelSubdomain) -> None:
        if any(
            existing.mesh_block_id == subdomain.mesh_block_id
            for existing in self.subdomains
        ):
            raise ValueError(f"Mesh block id {subdomain.mesh_block_id} is already used")
        self.add_subdomain(subdomain)

    def _next_mesh_block_id(self) -> int:
        used = [subdomain.mesh_block_id for subdomain in self.subdomains]
        used.extend(
            surface.mesh_block_id
            for surface in self.surfaces
            if isinstance(surface, Fracture) and surface.mesh_block_id is not None
        )
        return max([value for value in used if value >= 0], default=0) + 1

    def _coerce_borehole_part(
        self,
        borehole_name: str,
        spec: Union[BoreholePart, Mapping[str, Any]],
        next_block: int,
    ) -> Tuple[BoreholePart, Optional[ModelSubdomain], int]:
        if isinstance(spec, BoreholePart):
            return spec, None, max(next_block, spec.mesh_block_id + 1)
        if not isinstance(spec, Mapping):
            raise TypeError("Borehole layers must be BoreholePart objects or mappings")

        payload = copy.deepcopy(dict(spec))
        physics = payload.pop("physics", None)
        properties = payload.pop("properties", None)
        grid = payload.pop("grid", None)
        units = payload.pop("units", None)
        system = payload.pop("system", None)
        subdomain_name = payload.pop("subdomain_name", None)
        if payload.get("mesh_block_id") is None:
            if properties is None:
                raise ValueError(
                    "Borehole layers without properties must provide mesh_block_id"
                )
            payload["mesh_block_id"] = next_block
        part = BoreholePart.from_fs(payload)
        next_block = max(next_block, part.mesh_block_id + 1)

        subdomain = None
        if properties is not None:
            subdomain = ModelSubdomain(
                mesh_block_id=part.mesh_block_id,
                name=subdomain_name or f"{borehole_name}_{part.name}",
                physics=physics,
                properties=properties,
                grid=grid,
                units=units,
                system=system,
            )
        return part, subdomain, next_block

    def _coerce_borehole_plug(
        self,
        borehole_name: str,
        spec: Union[BoreholePlug, Mapping[str, Any]],
        next_block: int,
    ) -> Tuple[BoreholePlug, Optional[ModelSubdomain], int]:
        if isinstance(spec, BoreholePlug):
            return spec, None, max(next_block, spec.mesh_block_id + 1)
        if not isinstance(spec, Mapping):
            raise TypeError("Borehole plugs must be BoreholePlug objects or mappings")

        payload = copy.deepcopy(dict(spec))
        physics = payload.pop("physics", None)
        properties = payload.pop("properties", None)
        grid = payload.pop("grid", None)
        units = payload.pop("units", None)
        system = payload.pop("system", None)
        subdomain_name = payload.pop("subdomain_name", None)
        if payload.get("mesh_block_id") is None:
            if properties is None:
                raise ValueError(
                    "Borehole plugs without properties must provide mesh_block_id"
                )
            payload["mesh_block_id"] = next_block
        plug = BoreholePlug.from_fs(payload)
        next_block = max(next_block, plug.mesh_block_id + 1)

        subdomain = None
        if properties is not None:
            subdomain = ModelSubdomain(
                mesh_block_id=plug.mesh_block_id,
                name=subdomain_name or f"{borehole_name}_{plug.name}",
                physics=physics,
                properties=properties,
                grid=grid,
                units=units,
                system=system,
            )
        return plug, subdomain, next_block

    @staticmethod
    def _get_unique_name(name: Optional[str], names: Set[str]) -> str:
        orig_name = name
        warn_flag = True
        if name is None:
            warn_flag = False
            name = "unlabeled"

        if name in names:
            i = 1
            while f"{name}_{i}" in names and i < 1000:
                i += 1
            name = f"{name}_{i}"

        if warn_flag and orig_name != name:
            warnings.warn(
                f"\nSurface name '{orig_name}' was not unique; name was changed to '{name}'\n\n"
            )
        return name
