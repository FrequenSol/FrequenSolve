import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from frequensolve.geometry.frame import CoordinateSystem
from frequensolve.mesh.boundary_conditions import BoundaryCondition, BoundaryConditions
from frequensolve.mesh.mesh_generators import BaseMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.model import ModelBase
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import CoordsArray
from frequensolve.simulation.config import SimulationConfig
from frequensolve.simulation.numerics_manager import Discretization, SolverConfig
from frequensolve.units import UnitConfig
from frequensolve.util.class_registry import class_registry, register_class
from frequensolve.util.encoders import CustomJSONEncoder
from frequensolve.util.memoization import memoized_func, quantize
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    merge_extra,
)
from frequensolve.util.named_list import NamedList
from frequensolve.util.physics import (
    canonical_dimension,
    model_dimension,
    normalize_simulation_physics,
)
from frequensolve.util.store import SimulationStore

__all__ = [
    "BaseSimulation",
    "SeismicSimulation",
]


class _SimulationSurface:
    """Simulation-bound model surface coordinate helper."""

    def __init__(
        self, simulation: "SeismicSimulation", system: CoordinateSystem
    ) -> None:
        self._simulation = simulation
        self._system = system

    @property
    def coordinate_system(self) -> CoordinateSystem:
        return self._system

    def points(
        self,
        values: Any,
        *,
        units: Optional[Any] = None,
        offset: Optional[Any] = None,
    ):
        return self._system.points(values, units=units, offset=offset)

    on = points

    def above(
        self,
        values: Any,
        distance: Optional[Any] = None,
        *,
        units: Optional[Any] = None,
    ):
        return self._system.above(values, distance=distance, units=units)

    def below(
        self,
        values: Any,
        distance: Optional[Any] = None,
        *,
        units: Optional[Any] = None,
    ):
        return self._system.below(values, distance=distance, units=units)


def _model_surface_name(surface: Union[str, int]) -> str:
    if isinstance(surface, int):
        return f"surface_{surface}"
    return str(surface)


@register_class
class BaseSimulation(SimulationConfig):
    """Container for simulation configuration.

    Attributes:
       model (ModelBase):               Model configuration.
       mesh (MeshManager):              Mesh configuration.
       BCs (BoundaryConditions):        Boundary condition configuration.
       numerics (NumericsManager):      Numerics configuration.
    """

    model: Optional[ModelBase] = None
    mesh: Optional[MeshManager] = None
    BCs: BoundaryConditions = field(default_factory=BoundaryConditions)
    solver: SolverConfig = field(default_factory=SolverConfig)
    discretization: Discretization = field(default_factory=Discretization)

    def __post_init__(self):
        from frequensolve.util.printing import print_note

        if isinstance(self.mesh, BaseMeshGenerator):
            print_note(
                "Simulation was initialized with a MeshGenerator; specifying the mesh via a\n"
                "the 'MeshManager' class is recommended as it specifies mesh parallelism and adaptivity parameters.\n"
                "We've specified a default MeshManager for you but for fine grained control you'll want to specify\n"
                "your own."
            )
            self.mesh = MeshManager(self.mesh)

    @classmethod
    def from_fs(cls, data: Dict) -> "BaseSimulation":
        class_name = data["_type"]
        if class_name in class_registry:
            simulation_class = class_registry[class_name]
            return simulation_class.from_fs(data)
        raise ValueError(f"Unknown simulation class: {class_name}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "SeismicSimulation":
        """Load seismic simulation from JSON file."""
        path = Path(path).resolve()
        with open(path, "r") as f:
            data = json.load(f)
        class_name = data["_type"]
        if class_name in class_registry:
            sim_class = class_registry[class_name]
            sim = sim_class.from_fs(data)
            sim._file = path
            return sim
        else:
            raise Exception(f"Unknown simulation class: {class_name}")

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        from frequensolve.util.printing import print_note

        if isinstance(self.mesh, BaseMeshGenerator):
            print_note(
                "Simulation was initialized with a MeshGenerator; specifying the mesh via a\n"
                "the 'MeshManager' class is recommended as it specifies mesh parallelism and adaptivity parameters.\n"
                "We've specified a default MeshManager for you but for fine grained control you'll want to specify\n"
                "your own."
            )
            self.mesh = MeshManager(self.mesh)
        ctx = ctx or ExportContext(self._proj_path, self._rel_path)
        return {
            "_type": self.__class__.__name__,
            **super().to_fs(ctx),
            **(
                {"Model": self.model.to_fs(ctx.child(self.model.name))}
                if self.model
                else {}
            ),
            **({"Mesh": self.mesh.to_fs(ctx.child("mesh"))} if self.mesh else {}),
            **({"BCs": self.BCs.to_fs(ctx)} if self.BCs else {}),
            **({"Solver": self.solver.to_fs(ctx)} if self.solver else {}),
            **(
                {"Discretization": self.discretization.to_fs(ctx)}
                if self.discretization
                else {}
            ),
        }


@register_class
@dataclass(kw_only=True)
class SeismicSimulation(ExtraFieldsMixin, BaseSimulation):
    """Container for seismic simulation configuration.

    Attributes:
       name (str):                      Name of the simulation.
       model (ModelBase):               Model configuration.
       mesh (MeshManager):              Mesh configuration.
       BCs (BoundaryConditions):        Boundary condition configuration.
       solver (SolverConfig):           Solver configuration.
       discretization (Discretization): Discretization configuration.
       acquisition (Acquisition):       Acquisition configuration.
    """

    name: str
    physics: str
    dimension: int | float | str
    axisymmetric: bool = False
    project_path: Union[str, Path] = None
    model: ModelBase = field(default_factory=ModelBase)
    mesh: MeshManager = field(default_factory=MeshManager)
    BCs: BoundaryConditions = field(default_factory=BoundaryConditions)
    solver: SolverConfig = field(default_factory=SolverConfig)
    discretization: Discretization = field(default_factory=Discretization)
    acquisition: Acquisition = field(default_factory=Acquisition)
    units: UnitConfig = field(default_factory=UnitConfig)
    global_coordinate_system: Optional[CoordinateSystem] = None
    coordinate_systems: NamedList = field(default_factory=NamedList)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.coordinate_systems, NamedList):
            self.coordinate_systems = NamedList(self.coordinate_systems)
        self.dimension = canonical_dimension(self.dimension)
        self.physics, self.axisymmetric = normalize_simulation_physics(
            self.physics,
            axisymmetric=self.axisymmetric,
            dimension=self.dimension,
        )
        if self.project_path is None:
            raise ValueError(
                "When initializing a simulation, you now must either call project.new_simulation() "
                "(recommended), or specify the project_path as an argument to the constructor."
            )
        else:
            self.project_path = Path(self.project_path)
        if self.model.dimension == 0:
            self.model.dimension = model_dimension(self.dimension)

    @classmethod
    def from_fs(cls, data: Dict) -> "SeismicSimulation":
        data = copy.deepcopy(data)
        data.pop("schema", None)
        data.pop("_type", None)
        name = data.pop("name")
        physics = data.pop("physics")
        dimension = data.pop("dimension")
        axisymmetric = data.pop("axisymmetric", False)
        project_path = Path(data.pop("project_path"))
        sim = cls(
            name=name,
            physics=physics,
            dimension=dimension,
            axisymmetric=axisymmetric,
            project_path=project_path,
        )

        unit_payload = {}
        for key in [
            "disable_scaling",
            "f0",
            "length_scale",
            "time_scale",
            "mass_scale",
            "Units",
            "units",
        ]:
            if key in data:
                unit_payload[key] = data.pop(key)
        sim.units = UnitConfig.from_fs(unit_payload)

        if "global_coordinate_system" in data:
            sim.global_coordinate_system = CoordinateSystem.from_fs(
                data.pop("global_coordinate_system")
            )
        if "coordinate_systems" in data:
            sim.coordinate_systems = NamedList(
                CoordinateSystem.from_fs(item)
                for item in data.pop("coordinate_systems")
            )

        if "Model" in data:
            sim.model = ModelBase.from_fs(data.pop("Model"))
            sim._bind_model_coordinate_systems()
        if "Mesh" in data:
            sim.mesh = MeshManager.from_fs(data.pop("Mesh"))
        if "BCs" in data:
            sim.BCs = BoundaryConditions.from_fs(data.pop("BCs"))
        if "Solver" in data:
            sim.solver = SolverConfig.from_fs(data.pop("Solver"))
        if "Discretization" in data:
            sim.discretization = Discretization.from_fs(data.pop("Discretization"))
        data.pop("Outputs", None)
        if "Acquisition" in data:
            sim.acquisition = Acquisition.from_fs(data.pop("Acquisition"))

        sim.extra = data
        sim._set_path(project_path, Path("simulations"))
        return sim

    def _bind_model_coordinate_systems(self) -> None:
        if self.model is not None:
            self.model._coordinate_systems = self.coordinate_systems

    @property
    def coordinate_system(self) -> NamedList:
        """Alias for the named coordinate-system collection."""

        return self.coordinate_systems

    def copy(self, name, **kwargs) -> "SeismicSimulation":
        file = self.save()
        sim_copy = self.__class__.load(file)
        sim_copy.name = name

        for key, value in kwargs.items():
            setattr(sim_copy, key, value)

        # Load coords as array so they can be saved where they need to be
        for i, grp in enumerate(sim_copy.acquisition.receiver_groups):
            if grp.coordinates.__class__.__name__ == "CoordsFromFile":
                coords = grp.coordinates.get()
                grp.coordinates = CoordsArray(coordinates=coords)

        # Change file path
        sim_copy._file = file.parent.parent / name / name
        sim_copy._set_path(self.project_path, Path("simulations") / name)
        return sim_copy

    def fwi(
        self,
        observed,
        frequencies=None,
        parameters=None,
        grid=None,
        site=None,
        **kwargs,
    ):
        """Create an elastic FWI problem bound to this simulation.

        The returned object exposes PyLops-compatible Jacobian and adjoint
        operators. The adjoint is the inverse-problem transpose, not a true
        inverse solve.
        """

        from frequensolve.simulation.fwi import FWIProblem

        return FWIProblem(
            simulation=self,
            observed=observed,
            frequencies=frequencies,
            parameters=parameters,
            grid=grid,
            site=site,
            **kwargs,
        )

    def imaging(
        self,
        observed,
        frequencies=None,
        parameters=None,
        grid=None,
        fields=None,
        condition=None,
        images=None,
        **kwargs,
    ):
        """Create an imaging job using natural parameter/field specifications."""

        from frequensolve.simulation.fwi import build_imaging_job

        return build_imaging_job(
            self,
            observed=observed,
            frequencies=frequencies,
            parameters=parameters,
            grid=grid,
            fields=fields,
            condition=condition,
            images=images,
            **kwargs,
        )

    def add_coordinate_system(
        self, system: Union[CoordinateSystem, Mapping[str, Any]]
    ) -> CoordinateSystem:
        """Add or replace a named coordinate system on this simulation."""

        if isinstance(system, Mapping):
            system = CoordinateSystem.from_fs(system)
        if not isinstance(system, CoordinateSystem):
            raise TypeError(
                f"Expected CoordinateSystem or mapping, got {type(system).__name__}"
            )
        if not system.name:
            raise ValueError("Coordinate systems added to a simulation must be named")
        for index, existing in enumerate(self.coordinate_systems):
            if existing.name == system.name:
                self.coordinate_systems[index] = system
                self._bind_model_coordinate_systems()
                return system
        self.coordinate_systems.append(system)
        self._bind_model_coordinate_systems()
        return system

    def add_surface_coordinate_system(
        self,
        name: str,
        surface: Union[str, int],
        *,
        normal: str = "up",
        offset: Optional[Any] = None,
        offset_units: Optional[Any] = None,
        **kwargs,
    ) -> CoordinateSystem:
        """Create and register a coordinate system tied to a model surface."""

        system = CoordinateSystem.surface(
            name,
            surface,
            normal=normal,
            offset=offset,
            offset_units=offset_units,
            **kwargs,
        )
        return self.add_coordinate_system(system)

    def model_surface(
        self,
        surface: Union[str, int],
        *,
        name: Optional[str] = None,
        normal: str = "up",
        **kwargs,
    ) -> _SimulationSurface:
        """Return a registered model-surface helper for surface-relative points."""

        system = self.add_surface_coordinate_system(
            name or _model_surface_name(surface),
            surface,
            normal=normal,
            **kwargs,
        )
        return _SimulationSurface(self, system)

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        ctx = ctx or self.export_context()
        payload = super().to_fs(ctx)
        payload["schema"] = "fs-simulation-1"
        payload["_type"] = self.__class__.__name__
        payload.update(self.units.to_fs(ctx))
        if self.global_coordinate_system is not None:
            payload["global_coordinate_system"] = self.global_coordinate_system.to_fs()
        if self.coordinate_systems:
            payload["coordinate_systems"] = [
                cs.to_fs() for cs in self.coordinate_systems
            ]
        payload["Acquisition"] = self.acquisition.to_fs(ctx)
        return merge_extra(payload, self.extra, "Simulation")

    def __iadd__(self, other):
        if isinstance(other, ModelBase):
            self.model = other
            self._bind_model_coordinate_systems()
        elif isinstance(other, MeshManager):
            self.mesh = other
        elif isinstance(other, BaseMeshGenerator):
            self.mesh = MeshManager(other)
        elif isinstance(other, CoordinateSystem):
            self.add_coordinate_system(other)
        elif isinstance(other, BoundaryCondition):
            self.BCs.append(other)
        elif isinstance(other, BoundaryConditions):
            self.BCs = other
        elif isinstance(other, SolverConfig):
            self.solver = other
        elif isinstance(other, Discretization):
            self.discretization = other
        elif isinstance(other, Acquisition):
            self.acquisition = other
        else:
            raise ValueError(f"Cannot add {type(other)} to simulation")
        return self

    def export_context(self) -> ExportContext:
        proj_path = self._proj_path or Path(self.project_path)
        rel_path = self._rel_path or Path("simulations") / self.name
        store = SimulationStore(
            proj_path / rel_path / f"{self.name}.h5",
            project_path=proj_path,
        )
        return ExportContext(proj_path, rel_path, store=store)

    def as_json(self, **kwargs) -> str:
        """Convert simulation to JSON string."""
        indent = kwargs.get("indent", 3)
        return json.dumps(self.to_fs(), cls=CustomJSONEncoder, indent=indent, **kwargs)

    def save(self, **json_kwargs) -> Path:
        """Save seismic simulation to JSON file."""
        self._set_path(self.project_path, Path("simulations"))

        file = self.project_path / "simulations" / f"{self.name}" / f"{self.name}"
        file = file.with_suffix(".json").resolve()
        if not file.parent.exists():
            file.parent.mkdir(parents=True, exist_ok=True)

        self._file = file
        indent = json_kwargs.pop("indent", 3)
        ctx = self.export_context()
        with open(file, "w") as f:
            json.dump(
                self.to_fs(ctx), f, cls=CustomJSONEncoder, indent=indent, **json_kwargs
            )
        return file

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

    @property
    def _path(self) -> Path:
        return self._proj_path / self._rel_path

    @property
    def _remote_path(self) -> Path:
        return self._proj_path.name / self._rel_path


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
