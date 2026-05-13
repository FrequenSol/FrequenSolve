from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from frequensolve.util.physics import canonical_dimension, normalize_simulation_physics

__all__ = ["SimulationConfig"]


@dataclass(kw_only=True)
class SimulationConfig:
    """Container for simulator configuration.

    Args:
       name (str):       Name of the simulator.
       physics (str):    Physics type for the simulator.
       dimension (int | float | str): Dimension of the simulator (2D, 2.5D, or 3D).
       axisymmetric (bool): Whether a 2D/2.5D simulation uses axisymmetric geometry.
    """

    name: str
    physics: str
    dimension: int | float | str
    axisymmetric: bool = False
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None
    _file: Optional[Path] = None

    def __post_init__(self) -> None:
        self.dimension = canonical_dimension(self.dimension)
        self.physics, self.axisymmetric = normalize_simulation_physics(
            self.physics,
            axisymmetric=self.axisymmetric,
            dimension=self.dimension,
        )

    def to_fs(self, ctx=None) -> Dict:
        project_path = self._proj_path
        if project_path is None:
            project_path = getattr(self, "project_path", None)
        return {
            "name": self.name,
            "physics": self.physics,
            "dimension": self.dimension,
            "project_path": str(project_path),
            **({"axisymmetric": True} if self.axisymmetric else {}),
        }

    @classmethod
    def from_fs(cls, data: Dict[str, Any]) -> "SimulationConfig":
        return cls(
            name=data.get("name"),
            physics=data.get("physics"),
            dimension=data.get("dimension"),
            axisymmetric=data.get("axisymmetric", False),
        )

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path / self.name

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path
