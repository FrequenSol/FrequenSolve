"""Model base classes for managing simulation models."""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import xarray as xr

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.property import Property, PropertyMap
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.mixins import ExportContext, ExtraFieldsMixin, merge_extra
from frequensolve.util.named_list import NamedList

__all__ = ["ModelSubdomain", "ModelBase"]


@dataclass(kw_only=True)
class ModelSubdomain(ExtraFieldsMixin):
    """A subdomain within a model with associated properties.

    Attributes:
       mesh_block_id (int):  Unique identifier for the mesh block.
       name (Optional[str]): Optional name for the mesh block.
       physics (Optional[str]): Optional physics model name for the subdomain.
       properties (Dict[str, Union[float, str, xarray.DataArray]]): Dictionary of subdomain properties.
          Keys are property names, values can be numeric constants, file paths, or xarray DataArrays.
    """

    mesh_block_id: int = -1
    name: Optional[str] = None
    physics: Optional[str] = None
    properties: PropertyMap = field(default_factory=PropertyMap)
    extra: Dict[str, Any] = field(default_factory=dict)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

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
        if extra and "frame" in extra:
            raise TypeError(
                "ModelSubdomain frame is no longer supported; material coordinates are physical"
            )
        self._init_extra(extra, **kwargs)

    def set_property(self, key: str, value: Union[float, xr.DataArray]):
        self.properties[key] = value

    def __getitem__(self, key: str):
        return self.properties[key].get()

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        ctx = ctx or ExportContext(self._proj_path, self._rel_path)

        def property_file(key: str, prop: Property) -> Path:
            return self._path / f"layer_{self.mesh_block_id}_{key}.bin"

        def property_dataset(key: str, prop: Property) -> str:
            return f"inputs/model/subdomains/{self.mesh_block_id}/properties/{key}"

        payload = {
            "mesh_block_id": self.mesh_block_id,
            "name": self.name,
            **({"physics": self.physics} if self.physics is not None else {}),
            "properties": self.properties.to_fs(
                ctx=ctx,
                file_factory=property_file,
                dataset_factory=property_dataset,
            ),
        }
        return merge_extra(payload, self.extra, "Subdomain")

    @classmethod
    def from_fs(cls, data: Dict) -> "ModelSubdomain":
        data = copy.deepcopy(data)
        props = {}
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
            extra=data,
            **({"grid": grid} if grid is not None else {}),
        )

    def like(self, grid: xr.DataArray, **kwargs) -> None:
        # Legacy argument naming convention
        if "grid" in kwargs:
            grid = kwargs.pop("grid")

        for key, prop in self.properties.items():
            if prop.data.dims == grid.dims:
                self.properties[key]._like(grid)
            else:
                raise ValueError(f"Property {key} does not match dimensions of grid")

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path


@register_class
@dataclass(kw_only=True)
class ModelBase(ExtraFieldsMixin):
    """Base class for simulation models.

    Provides common attributes and functionality shared by different model types.

    Attributes:
       name (str):                Name identifier for the model.
       dimension (Literal[2, 3]): Model dimension (2D or 3D).
       x_limits (List[float]):    Model extent in x-direction [xmin, xmax].
       y_limits (List[float]):    Model extent in y-direction [ymin, ymax].
       z_limits (List[float]):    Model extent in z-direction [zmin, zmax].
       properties (Dict[str, Union[float, str]]): Dictionary of model properties.
          Keys are property names, values can be numeric constants or file paths.
    """

    name: str = "model"
    dimension: Literal[0, 2, 3] = 0  # 0 is used as an invalid value.
    subdomains: NamedList = field(default_factory=NamedList)
    extra: Dict[str, Any] = field(default_factory=dict)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:

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

        ctx = ctx or ExportContext(self._proj_path, self._rel_path)
        payload = {
            "_type": self.__class__.__name__,
            "name": self.name,
            "dimension": self.dimension,
            "subdomains": [subdomain.to_fs(ctx) for subdomain in self.subdomains],
        }
        return merge_extra(payload, self.extra, "Model")

    @classmethod
    def from_fs(cls, data: Dict) -> "ModelBase":
        data = copy.deepcopy(data)
        data.pop("schema", None)
        class_name = data.pop("_type", cls.__name__)
        if class_name == cls.__name__:
            subdomains = data.pop("subdomains", [])
            model = cls(
                name=data.pop("name", "model"),
                dimension=data.pop("dimension", 0),
                subdomains=NamedList(
                    [ModelSubdomain.from_fs(item) for item in subdomains]
                ),
            )
            model.extra = data
            return model
        if class_name in class_registry:
            model_class = class_registry[class_name]
            return model_class.from_fs(data)
        else:
            raise ValueError(f"Unknown model class: {class_name}")

    def add_subdomain(self, subdomain: ModelSubdomain) -> None:
        """Adds a subdomain to the model.

        Args:
           id (int): Unique mesh block identifier
           **kwargs: Additional subdomain parameters.
        """
        if subdomain.name is None:
            subdomain.name = f"unlabeled_{len(self.subdomains)}"
        self.subdomains.append(subdomain)

    def __iadd__(self, other):
        self.add_subdomain(other)
        return self

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path / self.name
        for subdomain in self.subdomains:
            subdomain._set_path(proj_path, self._rel_path)

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path
