import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import toml
import yaml

from frequensolve.mesh.boundary_conditions import BoundaryConditionManager
from frequensolve.mesh.mesh import Mesh
from frequensolve.mesh.mesh_generators import BaseMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.model import ModelBase
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.simulation.config import SimulationConfig
from frequensolve.simulation.numerics_manager import Discretization, SolverConfig
from frequensolve.simulation.output_manager import OutputManager
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.encoders import CustomJSONEncoder, CustomTOMLEncoder
from frequensolve.util.memoization import memoized_func, quantize

__all__ = [
    "CustomJSONEncoder",
    "SeismicSimulation",
]


@register_class
class BaseSimulation(SimulationConfig):
    """Container for simulation configuration.

    Attributes:
       model (ModelBase):               Model configuration.
       mesh (MeshManager):              Mesh configuration.
       BCs (BoundaryConditionManager):  Boundary condition configuration.
       numerics (NumericsManager):      Numerics configuration.
       output (OutputManager):          Output configuration.
    """

    model: Optional[ModelBase] = None
    mesh: Optional[MeshManager] = None
    BCs: BoundaryConditionManager = field(default_factory=BoundaryConditionManager)
    solver: SolverConfig = field(default_factory=SolverConfig)
    discretization: Discretization = field(default_factory=Discretization)
    outputs: OutputManager = field(default_factory=OutputManager)

    def __post_init__(self):
        from frequensolve.util.printing import print_note

        if isinstance(self.mesh, Mesh) or isinstance(self.mesh, BaseMeshGenerator):
            print_note(
                f"Simulation was initialized with a Mesh or MeshGenerator; specifying the mesh via a\n"
                "the 'MeshManager' class is recommended as it specifies mesh parallelism and adaptivity parameters.\n"
                "We've specified a default MeshManager for you but for fine grained control you'll want to specify\n"
                f"your own."
            )
            self.mesh = MeshManager(self.mesh)

    @classmethod
    def from_dict(cls, data: Dict) -> "BaseSimulation":
        class_name = data["_type"]
        if class_name in class_registry:
            simulation_class = class_registry[class_name]
            return simulation_class.from_dict(data)
        else:
            raise ValueError(f"Unknown simulation class: {class_name}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "SeismicSimulation":
        """Load seismic simulation from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
            return cls.from_dict(data)

    def __dict__(self) -> Dict:
        from frequensolve.util.printing import print_note

        if isinstance(self.mesh, Mesh) or isinstance(self.mesh, BaseMeshGenerator):
            print_note(
                f"Simulation was initialized with a Mesh or MeshGenerator; specifying the mesh via a\n"
                "the 'MeshManager' class is recommended as it specifies mesh parallelism and adaptivity parameters.\n"
                "We've specified a default MeshManager for you but for fine grained control you'll want to specify\n"
                f"your own."
            )
            self.mesh = MeshManager(self.mesh)
        return {
            "_type": self.__class__.__name__,
            **super().__dict__(),
            **({"Model": self.model.__dict__()} if self.model else {}),
            **({"Mesh": self.mesh.__dict__()} if self.mesh else {}),
            **({"BCs": self.BCs.__dict__()} if self.BCs else {}),
            **({"Solver": self.solver.__dict__()} if self.solver else {}),
            **(
                {"Discretization": self.discretization.__dict__()}
                if self.discretization
                else {}
            ),
            **({"Outputs": self.outputs.__dict__()} if self.outputs else {}),
        }


@register_class
@dataclass(kw_only=True)
class SeismicSimulation(BaseSimulation):
    """Container for seismic simulation configuration.

    Attributes:
       name (str):                      Name of the simulation.
       model (ModelBase):               Model configuration.
       mesh (MeshManager):              Mesh configuration.
       BCs (BoundaryConditionManager):  Boundary condition configuration.
       solver (SolverConfig):           Solver configuration.
       discretization (Discretization): Discretization configuration.
       output (OutputManager):          Output configuration.
       acquisition (Acquisition):       Acquisition configuration.
    """

    name: str
    physics: Literal["acoustic", "elastic", "plasma"]
    dimension: Literal[2, 3]
    model: ModelBase = field(default_factory=ModelBase)
    mesh: MeshManager = field(default_factory=MeshManager)
    BCs: BoundaryConditionManager = field(default_factory=BoundaryConditionManager)
    solver: SolverConfig = field(default_factory=SolverConfig)
    discretization: Discretization = field(default_factory=Discretization)
    outputs: OutputManager = field(default_factory=OutputManager)
    acquisition: Acquisition = field(default_factory=Acquisition)

    def __post_init__(self):
        if self.model.dimension == 0:
            self.model.dimension = self.dimension

    @classmethod
    def from_dict(
        cls, data: Dict, project_dir: Optional[Path] = None
    ) -> "SeismicSimulation":
        name = data["name"]
        physics = data["physics"]
        dimension = data["dimension"]
        sim = cls(name=name, physics=physics, dimension=dimension)

        # TODO: this path stuff is sloppy; need to make better way to manage all of this.
        if project_dir is None:
            project_dir = Path(data["project_path"])
        else:
            project_dir = project_dir

        # Load simulation from project directory
        if os.getcwd() != project_dir:
            cwd = os.getcwd()
            os.chdir(project_dir)

            if "Model" in data:
                sim.model = ModelBase.from_dict(data["Model"])
            if "Mesh" in data:
                sim.mesh = MeshManager.from_dict(data["Mesh"])
            if "BCs" in data:
                sim.BCs = BoundaryConditionManager.from_dict(data["BCs"])
            if "Solver" in data:
                sim.solver = SolverConfig.from_dict(data["Solver"])
            if "Discretization" in data:
                sim.discretization = Discretization.from_dict(data["Discretization"])
            if "Outputs" in data:
                sim.outputs = OutputManager.from_dict(data["Outputs"])
            if "Acquisition" in data:
                sim.acquisition = Acquisition.from_dict(data["Acquisition"])

            os.chdir(cwd)

        sim._set_path(project_dir, Path("simulations"))
        return sim

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "SeismicSimulation":
        """Load seismic simulation from JSON file."""

        with open(path, "r") as f:
            data = json.load(f)
        project_dir = Path(path).parent.parent
        sim = cls.from_dict(data, project_dir=project_dir)
        sim._file = path
        return sim

    def __dict__(self) -> Dict:
        dict = super().__dict__()
        dict.update(
            {
                "_type": self.__class__.__name__,
                **(
                    {"Acquisition": self.acquisition.__dict__()}
                    if self.acquisition
                    else {}
                ),
            }
        )
        return dict

    def __iadd__(self, other):
        if isinstance(other, ModelBase):
            self.model = other
        elif isinstance(other, MeshManager):
            self.mesh = other
        elif isinstance(other, BaseMeshGenerator) or isinstance(other, Mesh):
            self.mesh = MeshManager(other)
        elif isinstance(other, BoundaryConditionManager):
            self.BCs = other
        elif isinstance(other, SolverConfig):
            self.solver = other
        elif isinstance(other, Discretization):
            self.discretization = other
        elif isinstance(other, OutputManager):
            self.outputs = other
        elif isinstance(other, Acquisition):
            self.acquisition = other
        else:
            raise ValueError(f"Cannot add {type(other)} to simulation")
        return self

    def as_json(self, **kwargs) -> str:
        """Convert simulation to JSON string."""
        indent = kwargs.get("indent", 3)
        return json.dumps(
            self.__dict__(), cls=CustomJSONEncoder, indent=indent, **kwargs
        )

    def as_toml(self, **kwargs) -> str:
        """Convert simulation to TOML string."""
        indent = kwargs.get("indent", 3)
        return toml.dumps(
            self.__dict__(), encoder=CustomTOMLEncoder(), indent=indent, **kwargs
        )

    def as_yaml(self, **kwargs) -> str:
        """Convert simulation to YAML string."""

        def numpy_representer(dumper, data):
            """Convert numpy values to native Python types."""
            return dumper.represent_float(float(data))

        indent = kwargs.get("indent", 3)
        try:
            import numpy.typing as npt

            yaml.add_representer(npt.Float64, numpy_representer)
            yaml.add_representer(npt.Float32, numpy_representer)
            yaml.add_representer(
                npt.Int64, lambda dumper, data: dumper.represent_int(int(data))
            )
            yaml.add_representer(
                npt.Int32, lambda dumper, data: dumper.represent_int(int(data))
            )

            return yaml.dump(
                self.__dict__(),
                indent=indent,
                default_flow_style=False,
                sort_keys=False,
                **kwargs,
            )
        except Exception as e:
            print(f"Failed to convert to YAML: {e}")
            return self.__repr__()

    def save(self, path: Union[str, Path], **kwargs) -> str:
        """Save seismic simulation to JSON file."""
        file = (Path(path) / f"{self.name}").with_suffix(".json").resolve()

        if not file.parent.exists():
            file.parent.mkdir(parents=True, exist_ok=True)

        self._file = file
        indent = kwargs.get("indent", 3)
        with open(file, "w") as f:
            json.dump(
                self.__dict__(), f, cls=CustomJSONEncoder, indent=indent, **kwargs
            )
        return file.relative_to(self._proj_path)

    def check(self) -> bool:
        """Check the simulation is defined correctly."""
        return True

    @quantize
    @memoized_func
    def _estimate_memory(self, f) -> int:
        """Estimate memory required for the simulation."""
        pass

    def _set_path(self, proj_path: Path, rel_path: Path):
        self._proj_path = proj_path
        self._rel_path = rel_path / self.name
        if self.acquisition:
            self.acquisition._set_path(self._proj_path, self._rel_path)
        if self.model:
            self.model._set_path(self._proj_path, self._rel_path)
        if self.mesh:
            self.mesh._set_path(self._proj_path, self._rel_path)
        if self.outputs:
            path = self._proj_path / Path("outputs") / self.name
            self.outputs._set_path(self._proj_path, path)

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path


# TODO: loop over source groups
# TODO: also generate jobs for particular source groups and frequencies
# def _generate_jobs(self,
#                    f_list: List[float]) -> List[BaseJob]:
#    jobs = []
#    for f in f_list:
#       mem = self._estimate_memory(f)
#       jobs.append(BaseJob(self, f, required_memory = mem))
#    return jobs

# def get_material_mesh(self, f_max: float) -> MaterialMesh:
#    """Get the mesh the material model is defined on given the maximum frequency."""
#    pass
