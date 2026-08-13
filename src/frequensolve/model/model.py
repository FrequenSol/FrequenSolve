"""Model base classes for managing simulation models."""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import xarray as xr

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.attenuation import AttenuationConfig
from frequensolve.model.property import Property, PropertyMap
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    merge_extra,
)
from frequensolve.util.named_list import NamedList
from frequensolve.util.physics import model_dimension

__all__ = ["ModelSubdomain", "ModelBase"]


@dataclass(kw_only=True)
class ModelSubdomain(ExtraFieldsMixin):
    """Material subdomain associated with one mesh block.

    Args:
        mesh_block_id: Solver mesh-block identifier. ``-1`` may be used for
            unlabeled subdomains that will be assigned during model export.
        name: Optional user-facing subdomain name.
        physics: Optional physics/material family for this subdomain.
        properties: Mapping of property names to property-like values.
        fields: Mapping of independent named data used by property-expression
            ``field`` nodes.
        grid: Default grid metadata used when coercing subdomain properties.
        units: Default units applied to properties that omit units.
        system: Default coordinate-system name applied to properties that omit
            a system.
        extra: Additional serialized fields preserved on round trip.
        **kwargs: Additional serialized fields preserved on round trip.

    Raises:
        TypeError: If deprecated ``frame`` metadata is provided.
        ValueError: If both ``system`` and ``coordinate_system`` are provided
            with different values.
    """

    mesh_block_id: int = -1
    name: Optional[str] = None
    physics: Optional[str] = None
    properties: PropertyMap = field(default_factory=PropertyMap)
    fields: PropertyMap = field(default_factory=PropertyMap)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        mesh_block_id: int,
        name: Optional[str] = None,
        physics: Optional[str] = None,
        properties: Optional[Dict[str, Union[float, str, Path, xr.DataArray]]] = None,
        grid: Optional[xr.DataArray] = None,
        units: Optional[Any] = None,
        system: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        fields: Optional[Dict[str, Union[float, str, Path, xr.DataArray]]] = None,
        **kwargs,
    ):
        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")
        if "frame" in kwargs:
            raise TypeError(
                "ModelSubdomain frame is no longer supported; material coordinates are physical"
            )
        coordinate_system = kwargs.pop("coordinate_system", None)
        if coordinate_system is not None:
            if system is not None and system != coordinate_system:
                raise ValueError("Specify only one of system or coordinate_system")
            system = coordinate_system

        self.mesh_block_id = mesh_block_id
        self.name = name
        self.physics = physics
        self.properties = PropertyMap(
            properties or {},
            grid=grid,
            units=units,
            system=system,
        )
        extra = copy.deepcopy(dict(extra or {}))
        serialized_fields = extra.pop("fields", None)
        if fields is not None and serialized_fields is not None:
            raise ValueError("Specify subdomain fields directly or in extra, not both")
        self.fields = PropertyMap(
            fields if fields is not None else serialized_fields or {},
            grid=grid,
            system=system,
            canonicalize_keys=False,
        )
        if "frame" in extra:
            raise TypeError(
                "ModelSubdomain frame is no longer supported; material coordinates are physical"
            )
        self._init_extra(extra, **kwargs)

    def set_property(self, key: str, value: Union[float, xr.DataArray]):
        """Set or replace a material property.

        Args:
            key: Property name or alias.
            value: Property-like value accepted by ``PropertyMap``.
        """

        self.properties[key] = value

    def set_field(self, key: str, value: Union[float, xr.DataArray]) -> None:
        """Set or replace named data used by property expressions.

        Args:
            key: Independent field name referenced by a ``field`` expression node.
            value: Property-like scalar, array, or file value.
        """

        self.fields[key] = value

    def __getitem__(self, key: str):
        """Return the materialized value for one subdomain property.

        Args:
            key: Property name or alias.

        Returns:
            The value returned by the underlying :class:`Property`.
        """

        return self.properties[key].get()

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Serialize the subdomain for solver input.

        Args:
            ctx: Optional export context used for project-relative paths and
                property storage.

        Returns:
            Payload containing the mesh-block id, optional name/physics, and
            serialized property mapping.
        """

        ctx = ctx or ExportContext()

        def property_file(key: str, prop: Property) -> Path:
            return ctx.path / f"layer_{self.mesh_block_id}_{key}.bin"

        def property_dataset(key: str, prop: Property) -> str:
            return f"inputs/model/subdomains/{self.mesh_block_id}/properties/{key}"

        def field_file(key: str, prop: Property) -> Path:
            return ctx.path / f"layer_{self.mesh_block_id}_field_{key}.bin"

        def field_dataset(key: str, prop: Property) -> str:
            return f"inputs/model/subdomains/{self.mesh_block_id}/fields/{key}"

        payload = {
            "mesh_block_id": self.mesh_block_id,
            "name": self.name,
            **({"physics": self.physics} if self.physics is not None else {}),
            "properties": self.properties.to_fs(
                ctx=ctx,
                file_factory=property_file if ctx.path is not None else None,
                dataset_factory=property_dataset,
            ),
            **(
                {
                    "fields": self.fields.to_fs(
                        ctx=ctx,
                        file_factory=field_file if ctx.path is not None else None,
                        dataset_factory=field_dataset,
                        preserve_inline_coordinates=True,
                    )
                }
                if self.fields
                else {}
            ),
        }
        return merge_extra(payload, self.extra, "Subdomain")

    @classmethod
    def from_fs(cls, data: Dict) -> "ModelSubdomain":
        """Deserialize a model subdomain payload.

        Args:
            data: Serialized subdomain mapping from a model payload.

        Returns:
            A ``ModelSubdomain`` with normalized properties and preserved extra
            fields.
        """

        data = copy.deepcopy(data)
        props = {}
        fields = data.pop("fields", {})
        grid = None
        for prop, value in data.pop("properties").items():
            if isinstance(value, dict) and "file" in value and "grid" in value:
                try:
                    grid = CartesianGrid.from_fs(value["grid"]).as_xarray()
                except Exception:
                    grid = None
            props[prop] = value

        data.pop("frame", None)
        return cls(
            mesh_block_id=data.pop("mesh_block_id"),
            name=data.pop("name", None),
            physics=data.pop("physics", None),
            properties=props,
            fields=fields,
            extra=data,
            grid=grid,
        )

    def like(self, grid: xr.DataArray, **kwargs) -> None:
        """Interpolate all compatible properties onto another grid in place.

        Args:
            grid: Target xarray grid whose dimensions and coordinates define
                the new property layout.
            **kwargs: Legacy keyword aliases. ``grid=...`` is accepted for
                older call sites.

        Raises:
            ValueError: If any property dimensions are incompatible with
                ``grid``.
        """

        # Legacy argument naming convention
        if "grid" in kwargs:
            grid = kwargs.pop("grid")

        for collection_name, collection in (
            ("Property", self.properties),
            ("Field", self.fields),
        ):
            for key, prop in collection.items():
                if prop.data.dims == grid.dims:
                    collection[key]._like(grid)
                else:
                    raise ValueError(
                        f"{collection_name} {key} does not match dimensions of grid"
                    )


@register_class
@dataclass(kw_only=True)
class ModelBase(ExtraFieldsMixin):
    """Base class for serializable simulation models.

    Args:
        name: Model name used in project paths and serialized payloads.
        dimension: Model dimension. ``0`` is allowed for an uninitialized base
            model; concrete models normalize to 2D or 3D.
        attenuation_model: Optional model-wide attenuation model name.
        reference_frequency: Optional positive attenuation reference frequency.
            Bare values are interpreted as hertz; Pint quantities and
            unit-bearing mappings may use compatible frequency units.
        subdomains: Material subdomains belonging to this model.
        extra: Additional serialized fields preserved on round trip.

    Notes:
        Concrete model types are registered in ``class_registry`` and are
        dispatched by ``from_fs`` when the payload ``_type`` names a subclass.
    """

    name: str = "model"
    dimension: Union[Literal[0], int, float, str] = 0  # 0 is used as an invalid value.
    attenuation_model: Optional[str] = None
    reference_frequency: Optional[Any] = None
    subdomains: NamedList = field(default_factory=NamedList)
    extra: Dict[str, Any] = field(default_factory=dict)
    _attenuation_extra: Dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.dimension != 0:
            self.dimension = model_dimension(self.dimension)
        attenuation = self._attenuation_config()
        if attenuation is not None:
            self.attenuation_model = attenuation.model
            self.reference_frequency = attenuation.reference_frequency

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Serialize the model and its material subdomains.

        Args:
            ctx: Optional export context passed to subdomain/property export.

        Returns:
            Solver model payload with model type, name, dimension, and
            serialized subdomains. Unlabeled subdomains receive unique positive
            mesh-block identifiers during export.

        Raises:
            ValueError: If two explicit subdomains use the same mesh-block id.
        """

        assert self.dimension in [0, 2, 3], "Dimension must be 0, 2, or 3"

        # Label any unlabeled subdomains
        labels = {}
        for i, subdomain in enumerate(self.subdomains):
            id = subdomain.mesh_block_id
            if id >= 0:
                if id in labels:
                    raise ValueError(
                        f"Mesh block ids must be unique: {id} is was repeated"
                    )
            labels[id] = i

        j = 1
        for i, subdomain in enumerate(self.subdomains):
            if subdomain.mesh_block_id == -1:
                while j in labels:
                    j += 1
                labels[j] = i
                subdomain.mesh_block_id = j

        ctx = ctx or ExportContext()
        attenuation = self._attenuation_config()
        payload = {
            "_type": self.__class__.__name__,
            "name": self.name,
            "dimension": self.dimension,
            **(
                {"attenuation": attenuation.to_fs(ctx)}
                if attenuation is not None
                else {}
            ),
            "subdomains": [subdomain.to_fs(ctx) for subdomain in self.subdomains],
        }
        return merge_extra(payload, self.extra, "Model")

    @classmethod
    def from_fs(cls, data: Dict) -> "ModelBase":
        """Deserialize a model payload.

        Args:
            data: Serialized model mapping.

        Returns:
            A ``ModelBase`` instance or a registered concrete model subclass
            named by the payload ``_type`` field.

        Raises:
            ValueError: If the payload names an unknown model class.
        """

        data = copy.deepcopy(data)
        data.pop("schema", None)
        class_name = data.pop("_type", cls.__name__)
        if class_name == cls.__name__:
            subdomains = data.pop("subdomains", [])
            attenuation_payload = data.pop("attenuation", None)
            attenuation = (
                AttenuationConfig.from_fs(attenuation_payload)
                if attenuation_payload is not None
                else None
            )
            model = cls(
                name=data.pop("name", "model"),
                dimension=data.pop("dimension", 0),
                attenuation_model=attenuation.model if attenuation else None,
                reference_frequency=(
                    attenuation.reference_frequency if attenuation else None
                ),
                subdomains=NamedList(
                    [ModelSubdomain.from_fs(item) for item in subdomains]
                ),
            )
            if attenuation is not None:
                model._attenuation_extra = attenuation.extra
            model.extra = data
            return model
        if class_name in class_registry:
            model_class = class_registry[class_name]
            return model_class.from_fs(data)
        raise ValueError(f"Unknown model class: {class_name}")

    def add_subdomain(self, subdomain: ModelSubdomain) -> None:
        """Add a material subdomain to the model.

        Args:
            subdomain: Subdomain to append. If it has no name, a stable
                ``unlabeled_N`` name is assigned before insertion.
        """
        if subdomain.name is None:
            subdomain.name = f"unlabeled_{len(self.subdomains)}"
        self.subdomains.append(subdomain)

    def __iadd__(self, other):
        self.add_subdomain(other)
        return self

    def _attenuation_config(self) -> Optional[AttenuationConfig]:
        if self.attenuation_model is None and self.reference_frequency is None:
            return None
        return AttenuationConfig(
            model=self.attenuation_model or "kjartansson",
            reference_frequency=self.reference_frequency,
            extra=self._attenuation_extra,
        )
