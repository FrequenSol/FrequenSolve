"""Model base classes for managing simulation models."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import xarray as xr
from numpy.typing import ArrayLike

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.model.property import Property
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.data_file import save_data_if_new
from frequensolve.util.named_list import NamedList

__all__ = ["ModelSubdomain", "ModelBase"]


@dataclass(kw_only=True)
class ModelSubdomain:
    """A subdomain within a model with associated properties.

    Attributes:
       mesh_block_id (int):  Unique identifier for the mesh block.
       name (Optional[str]): Optional name for the mesh block.
       frame (str):          Coordinate frame for mapping subdomain materials ('physical' or 'reference').
       properties (Dict[str, Union[float, str, xarray.DataArray]]): Dictionary of subdomain properties.
          Keys are property names, values can be numeric constants, file paths, or xarray DataArrays.
    """

    mesh_block_id: int = -1
    name: Optional[str] = None
    frame: str = "physical"
    properties: Dict[str, Property] = field(default_factory=dict)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __init__(
        self,
        mesh_block_id: int,
        name: Optional[str] = None,
        frame: str = "physical",
        properties: Dict[str, Union[float, str, Path, xr.DataArray]] = {},
        grid: Optional[xr.DataArray] = None,
        **kwargs,
    ):
        # Legacy argument naming convention
        if "xarr" in kwargs:
            grid = kwargs.pop("xarr")

        self.mesh_block_id = mesh_block_id
        self.name = name
        self.frame = frame
        self._properties = {}
        for key, val in properties.items():
            if isinstance(val, str) or isinstance(val, Path):
                dims = None
                scale = 1.0
                split = str(val).split("|")
                if len(split) >= 2:
                    path = split[0]
                    if "x" in split[1]:
                        dims = split[1]
                        if len(split) >= 3:
                            scale = split[2]
                    else:
                        scale = split[1]
                        if len(split) >= 3:
                            dims = split[2]
                    if dims is not None:
                        dims = [d for d in dims]
                    self._properties[key] = Property(
                        data=path,
                        grid=grid,
                        scale=float(scale),
                        **({"dims": dims} if dims else {}),
                    )
                else:
                    self._properties[key] = Property(data=val, grid=grid)
            else:
                self._properties[key] = Property(data=val)

    def set_property(self, key: str, value: Union[float, xr.DataArray]):
        self._properties[key] = Property(data=value)

    @property
    def properties(self) -> Dict[str, Property]:
        if self._properties is None:
            raise ValueError("Properties not set for subdomain")
        return self._properties

    @properties.setter
    def properties(self, dict: Dict[str, Union[float, str, Path, xr.DataArray]]):
        # TODO: Will need a way to specify the grid for the properties
        self._properties = {key: Property(data=val) for key, val in dict.items()}

    def __getitem__(self, key: str):
        return self.properties[key].get()

    def __dict__(self) -> Dict:
        props = {}
        for key, prop in self._properties.items():
            if prop.is_constant:
                props[key] = {"value": self._properties[key].get()}
            else:
                if prop.is_remote:
                    props[key] = {
                        "absolute": True,
                        "file": f"{prop.remote_path}",
                        **(
                            {"scale": prop.remote_scale}
                            if prop.remote_scale != 1.0
                            else {}
                        ),
                    }
                else:
                    file = self._path / (f"layer_{self.mesh_block_id}_{key}.bin")
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file = save_data_if_new(self.properties[key].darr, file)
                    props[key] = {
                        "file": file.relative_to(self._proj_path),
                    }
                grid = self.properties[key].grid
                props[key]["grid"] = grid.__dict__()
        return {
            "mesh_block_id": self.mesh_block_id,
            "name": self.name,
            "frame": self.frame,
            "properties": props,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelSubdomain":
        props = {}
        grid = None
        for prop, value in data["properties"].items():
            if "file" in value:
                grid = CartesianGrid.from_dict(value["grid"]).as_xarray()
                props[prop] = value["file"]
                if "absolute" in value:
                    props[prop] = f"remote:{props[prop]}"
                if "scale" in value:
                    scale = value["scale"]
                    props[prop] = f"{props[prop]}|{scale}"
                if "order" in value:
                    order = value["order"]
                    props[prop] = f"{props[prop]}|{order}"
            else:
                props[prop] = value["value"]

        return cls(
            mesh_block_id=data["mesh_block_id"],
            name=data["name"],
            frame=data["frame"],
            properties=props,
            **({"grid": grid} if grid is not None else {}),
        )

    def like(self, grid: xr.DataArray, **kwargs) -> None:
        # Legacy argument naming convention
        if "grid" in kwargs:
            grid = kwargs.pop("grid")

        for key, prop in self._properties.items():
            if prop.data.dims == grid.dims:
                self._properties[key]._like(grid)
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
class ModelBase:
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
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None

    def __dict__(self) -> Dict:

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

        return {
            "_type": self.__class__.__name__,
            "name": self.name,
            "dimension": self.dimension,
            "subdomains": [subdomain.__dict__() for subdomain in self.subdomains],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelBase":
        class_name = data["_type"]
        if class_name in class_registry:
            model_class = class_registry[class_name]
            return model_class.from_dict(data)
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
