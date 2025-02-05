"""Python structures defining mesh API"""

from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..util.class_registry import class_registry, register_class

__all__ = ["BaseMeshGenerator", "HexMeshGenerator"]


@register_class
@dataclass
class BaseMeshGenerator(ABC):
    """Base class for mesh generators"""

    _proj_path: Path = Path()
    _rel_path: Path = Path()

    @classmethod
    def from_dict(cls, data: Dict) -> "BaseMeshGenerator":
        class_name = data["_type"]
        if class_name in class_registry:
            mesh_class = class_registry[class_name]
            return mesh_class.from_dict(data)
        else:
            raise ValueError(f"Unknown mesh generator class: {class_name}")

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path


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

    def __dict__(self) -> Dict:
        if self.l_bound is not None:
            assert self.u_bound is not None
            l_bound = self.l_bound
            u_bound = self.u_bound

        if self.n is None:
            self.n = [8] * len(self.l_bound)

        return {
            "_type": self.__class__.__name__,
            "path": self._rel_path,
            "n": self.n,
            "l_bound": l_bound,
            "u_bound": u_bound,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "HexMeshGenerator":
        return cls(
            n=data["n"],
            l_bound=data["l_bound"],
            u_bound=data["u_bound"],
        )
