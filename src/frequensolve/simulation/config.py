from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

__all__ = ["SimulationConfig"]


@dataclass(kw_only=True)
class SimulationConfig:
    """Container for simulator configuration.

    Args:
       name (str):       Name of the simulator.
       physics (str):    Physics type for the simulator.
       dimension (int):  Dimension of the simulator (2D or 3D).
    """

    name: str
    physics: Literal["acoustic", "elastic", "plasma"]
    dimension: Literal[2, 3]
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None
    _file: Optional[Path] = None

    def to_fs(self, ctx=None) -> Dict:
        return {
            "name": self.name,
            "physics": self.physics,
            "dimension": self.dimension,
            "project_path": str(self._proj_path),
        }

    @classmethod
    def from_fs(cls, data: Dict) -> "SimulationConfig":
        return cls(
            name=data.get("name"),
            physics=data.get("physics"),
            dimension=data.get("dimension"),
        )

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path / self.name

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path
