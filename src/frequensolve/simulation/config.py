"""Simulation-level metadata shared by jobs, signals, and solver contracts."""

from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from frequensolve.util.physics import (
    canonical_dimension,
    canonical_physics,
    normalize_simulation_physics,
)

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
    axisymmetric: InitVar[bool] = False
    _axisymmetric: bool = field(default=False, init=False, repr=False)
    _proj_path: Optional[Path] = None
    _rel_path: Optional[Path] = None
    _file: Optional[Path] = None

    _AXISYMMETRIC_BASE_PHYSICS = {
        "acoustic_axisym": "acoustic",
        "elastic_axisym": "elastic",
        "coupled_axisym": "coupled",
    }

    def __post_init__(self, axisymmetric: bool) -> None:
        """Normalize physics and dimension aliases after dataclass initialization."""

        self.dimension = canonical_dimension(self.dimension)
        self.physics, self._axisymmetric = normalize_simulation_physics(
            self.physics,
            axisymmetric=axisymmetric,
            dimension=self.dimension,
        )

    def _set_axisymmetric(self, axisymmetric: bool) -> None:
        axisymmetric = bool(axisymmetric)
        if axisymmetric:
            self.physics, self._axisymmetric = normalize_simulation_physics(
                self.physics,
                axisymmetric=True,
                dimension=self.dimension,
            )
            return

        canonical = canonical_physics(self.physics)
        if canonical in self._AXISYMMETRIC_BASE_PHYSICS:
            self.physics = self._AXISYMMETRIC_BASE_PHYSICS[canonical]
            self._axisymmetric = False
            return
        if canonical.endswith("_axisym_torsion"):
            raise ValueError(
                f"Cannot disable axisymmetric mode for {canonical!r}; "
                "choose a non-axisymmetric physics formulation explicitly."
            )
        self.physics = canonical
        self._axisymmetric = False

    def to_fs(self, ctx=None) -> Dict:
        """Serialize this simulation configuration to the solver payload."""

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
        """Deserialize simulation configuration from a solver payload."""

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


def _get_axisymmetric(self: SimulationConfig) -> bool:
    return self._axisymmetric


def _set_axisymmetric(self: SimulationConfig, value: bool) -> None:
    self._set_axisymmetric(value)


SimulationConfig.axisymmetric = property(_get_axisymmetric, _set_axisymmetric)
