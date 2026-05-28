"""Simulation authoring containers and solver-contract serialization."""

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
from frequensolve.util.mixins import (
    ExportContext,
    ExtraFieldsMixin,
    merge_extra,
    warn_deprecated_path_api,
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


@register_class
class BaseSimulation(SimulationConfig):
    """Shared simulation container behavior.

    ``BaseSimulation`` provides type-dispatched loading and the common
    solver-facing payload for model, mesh, boundary conditions, solver settings,
    and discretization. Concrete user-facing simulations should normally use
    :class:`SeismicSimulation`.
    """

    model: Optional[ModelBase] = None
    mesh: Optional[MeshManager] = None
    BCs: BoundaryConditions = field(default_factory=BoundaryConditions)
    solver: SolverConfig = field(default_factory=SolverConfig)
    discretization: Discretization = field(default_factory=Discretization)

    def __post_init__(self):
        """Normalize member objects supplied as mappings or generators."""

        from frequensolve.util.printing import print_note

        if isinstance(self.mesh, BaseMeshGenerator):
            print_note(
                "Simulation was initialized with a MeshGenerator; specifying the mesh via a\n"
                "the 'MeshManager' class is recommended as it specifies mesh parallelism and adaptivity parameters.\n"
                "We've specified a default MeshManager for you but for fine grained control you'll want to specify\n"
                "your own."
            )
        self.model = _coerce_model(self.model)
        self.mesh = _coerce_mesh(self.mesh)
        self.BCs = _coerce_boundary_conditions(self.BCs)
        if isinstance(self.solver, Mapping):
            self.solver = SolverConfig.from_fs(self.solver)
        if isinstance(self.discretization, Mapping):
            self.discretization = Discretization.from_fs(self.discretization)

    @classmethod
    def from_fs(cls, data: Dict) -> "BaseSimulation":
        """Dispatch a solver simulation payload to its registered Python class.

        Args:
            data: Serialized simulation mapping containing ``_type``.

        Returns:
            Concrete simulation instance.
        """

        class_name = data["_type"]
        if class_name in class_registry:
            simulation_class = class_registry[class_name]
            return simulation_class.from_fs(data)
        raise ValueError(f"Unknown simulation class: {class_name}")

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "SeismicSimulation":
        """Load a simulation JSON file and dispatch by its ``_type`` field.

        Args:
            path: Simulation JSON path.
            **kwargs: Optional load controls, including ``project_path``.

        Returns:
            Loaded ``SeismicSimulation`` or other registered subclass.
        """

        path = Path(path).resolve()
        project_path = kwargs.pop("project_path", None)
        with open(path, "r") as f:
            data = json.load(f)
        class_name = data["_type"]
        if class_name in class_registry:
            sim_class = class_registry[class_name]
            sim = sim_class.from_fs(data)
            sim._file = path
            project = (
                Path(project_path).expanduser().resolve()
                if project_path is not None
                else _project_path_from_simulation_file(path)
            )
            if project is not None:
                sim.project_path = project
                sim._attach_project_path(project, Path("simulations"))
            return sim
        else:
            raise Exception(f"Unknown simulation class: {class_name}")

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Serialize common simulation blocks to the FrequenSolve JSON contract.

        Args:
            ctx: Optional export context used by nested model, mesh, and
                acquisition serializers.

        Returns:
            JSON-compatible base simulation payload.
        """

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
            **({"Mesh": self.mesh.to_fs(ctx)} if self.mesh else {}),
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
    """Authoring container for a seismic simulation.

    A seismic simulation owns the model, mesh, flat named boundary-condition
    list, solver/discretization settings, acquisition geometry, units, and
    coordinate systems that are exported to the solver JSON contract. Member
    fields may be supplied either as Python objects or as solver-style mappings;
    construction normalizes them to the current Python classes.

    Boundary conditions are stored in ``BCs`` as a flat
    :class:`BoundaryConditions` list. Named boundary conditions can be accessed
    directly, e.g. ``simulation.BCs["free_surface"]``.
    """

    name: str
    physics: str
    dimension: int | float | str
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

    def __post_init__(self, axisymmetric: bool):
        """Normalize inputs and derive canonical physics/dimension metadata."""

        self.model = _coerce_model(self.model)
        self.mesh = _coerce_mesh(self.mesh)
        self.BCs = _coerce_boundary_conditions(self.BCs)
        if isinstance(self.solver, Mapping):
            self.solver = SolverConfig.from_fs(self.solver)
        if isinstance(self.discretization, Mapping):
            self.discretization = Discretization.from_fs(self.discretization)
        if isinstance(self.acquisition, Mapping):
            self.acquisition = Acquisition.from_fs(self.acquisition)
        if isinstance(self.units, Mapping):
            self.units = UnitConfig.from_fs(self.units)
        if isinstance(self.global_coordinate_system, Mapping):
            self.global_coordinate_system = CoordinateSystem.from_fs(
                self.global_coordinate_system
            )
        self.coordinate_systems = _coerce_named_coordinate_systems(
            self.coordinate_systems
        )
        self.dimension = canonical_dimension(self.dimension)
        self.physics, self._axisymmetric = normalize_simulation_physics(
            self.physics,
            axisymmetric=axisymmetric,
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
        """Build a seismic simulation from an ``fs-simulation-1`` payload.

        Args:
            data: Serialized seismic simulation mapping.

        Returns:
            ``SeismicSimulation`` instance with nested objects deserialized.
        """

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
        sim._attach_project_path(project_path, Path("simulations"))
        return sim

    def _bind_model_coordinate_systems(self) -> None:
        """Expose this simulation's coordinate systems to coordinate-aware models."""

        if self.model is not None:
            self.model._coordinate_systems = self.coordinate_systems

    @property
    def coordinate_system(self) -> NamedList:
        """Alias for ``coordinate_systems`` for concise notebook use."""

        return self.coordinate_systems

    def copy(self, name, **kwargs) -> "SeismicSimulation":
        """Persist, reload, and rename this simulation with optional overrides.

        Args:
            name: Name for the copied simulation.
            **kwargs: Attribute overrides applied to the copied simulation.

        Returns:
            Copied simulation instance.
        """

        file = self.save()
        sim_copy = self.__class__.load(file)
        sim_copy.name = name

        for key, value in kwargs.items():
            setattr(sim_copy, key, value)

        # Load coords as array so they can be saved where they need to be
        for grp in sim_copy.acquisition.receiver_groups:
            if grp.coordinates.__class__.__name__ == "CoordsFromFile":
                coord_ref = grp.coordinates
                coords = coord_ref.get()
                units = coord_ref.units or sim_copy.units.defaults.get("length")
                grp.coordinates = CoordsArray(
                    coordinates=coords,
                    units=units,
                    system=coord_ref.system,
                )

        # Change file path
        sim_copy._file = file.parent.parent / name / name
        sim_copy._attach_project_path(self.project_path, Path("simulations"))
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
        """Create an FWI problem bound to this simulation.

        The returned object exposes PyLops-compatible Jacobian and adjoint
        operators. The adjoint is the inverse-problem transpose, not a true
        inverse solve.

        Args:
            observed: Observed data used by the inverse problem.
            frequencies: Optional modeled frequencies.
            parameters: Optional model parameters to invert.
            grid: Optional inversion grid.
            site: Optional execution site.
            **kwargs: Additional FWI problem options.
        """

        from frequensolve.simulation.jobs.fwi import FWIProblem

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
        """Create an imaging job using natural parameter/field specifications.

        Args:
            observed: Observed data used by the imaging condition.
            frequencies: Optional modeled frequencies.
            parameters: Optional image parameters.
            grid: Optional imaging grid.
            fields: Optional field selections.
            condition: Optional imaging condition.
            images: Optional explicit image outputs.
            **kwargs: Additional imaging job options.
        """

        from frequensolve.simulation.jobs.fwi import build_imaging_job

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
        """Add a coordinate system, replacing an existing one with the same name.

        Args:
            system: Coordinate system instance or serialized mapping.

        Returns:
            Stored coordinate system.
        """

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
        """Create and register a coordinate system tied to a model surface.

        Args:
            name: Coordinate-system name.
            surface: Model surface name or index.
            normal: Surface-normal direction convention.
            offset: Optional surface offset.
            offset_units: Units for ``offset`` when it is not a quantity.
            **kwargs: Additional coordinate-system fields.
        """

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
    ) -> "_SimulationSurface":
        """Return a registered model-surface helper for surface-relative points.

        Args:
            surface: Model surface name or index.
            name: Optional coordinate-system name.
            normal: Surface-normal direction convention.
            **kwargs: Additional surface coordinate-system fields.
        """

        system = self.add_surface_coordinate_system(
            name or _model_surface_name(surface),
            surface,
            normal=normal,
            **kwargs,
        )
        return _SimulationSurface(self, system)

    def to_fs(self, ctx: Optional[ExportContext] = None) -> Dict:
        """Serialize the complete seismic simulation to solver JSON.

        Args:
            ctx: Optional export context.

        Returns:
            JSON-compatible ``fs-simulation-1`` payload.
        """

        ctx = ctx or self.export_context()
        if getattr(ctx, "default_length_units", None) is None:
            ctx.default_length_units = self.units.defaults.get("length")
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
        """Add or replace simulation member objects with ``simulation += obj``.

        Args:
            other: Model, mesh, coordinate system, boundary condition, solver,
                discretization, or acquisition object.

        Returns:
            This simulation instance.
        """

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
        """Return the project-relative export context and backing HDF5 store."""

        proj_path = self._proj_path or Path(self.project_path)
        rel_path = self._rel_path or Path("simulations") / self.name
        store = SimulationStore(
            proj_path / rel_path / f"{self.name}.h5",
            project_path=proj_path,
        )
        return ExportContext(proj_path, rel_path, store=store)

    def as_json(self, **kwargs) -> str:
        """Return the solver JSON payload as a formatted string.

        Args:
            **kwargs: Keyword arguments forwarded to ``json.dumps``.
        """

        indent = kwargs.get("indent", 3)
        return json.dumps(self.to_fs(), cls=CustomJSONEncoder, indent=indent, **kwargs)

    def save(self, **json_kwargs) -> Path:
        """Write the simulation JSON file under this simulation's project path.

        Args:
            **json_kwargs: Keyword arguments forwarded to ``json.dump``.

        Returns:
            Path to the written simulation JSON file.
        """

        self._attach_project_path(self.project_path, Path("simulations"))

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
        """Return whether the simulation passes SDK validation."""

        return self.validate().ok

    def validate(self, *, raise_errors: bool = False):
        """Validate this simulation for common authoring mistakes.

        Args:
            raise_errors: If ``True``, raise ``ValidationError`` when blocking
                issues are found.

        Returns:
            ``ValidationReport`` with errors and warnings.
        """

        from frequensolve.validation import validate_simulation

        return validate_simulation(self, raise_errors=raise_errors)

    def _attach_project_path(self, proj_path: Path, rel_path: Path) -> None:
        """Attach this simulation to a project-relative export root."""

        previous_project_path = self._proj_path
        self._proj_path = Path(proj_path).expanduser().resolve()
        self._rel_path = Path(rel_path) / self.name
        self._resolve_project_references(source_project_path=previous_project_path)

    def _set_path(self, proj_path: Path, rel_path: Path) -> None:
        """Backward-compatible alias for attaching the simulation path."""

        warn_deprecated_path_api(f"{self.__class__.__name__}._set_path")
        self._attach_project_path(proj_path, rel_path)

    def _resolve_project_references(
        self, *, source_project_path: Optional[Path] = None
    ) -> None:
        """Normalize loaded file references that need project context at runtime."""

        from frequensolve.seismic.receivers import CoordsFromFile

        if not self.acquisition:
            return
        ctx = self.export_context()
        for group in self.acquisition.receiver_groups:
            coords = getattr(group, "coordinates", None)
            if isinstance(coords, CoordsFromFile):
                coords.file = coords._contextual_file(
                    ctx, source_project_path=source_project_path
                )
                coords._fill_metadata_from_file(ctx)

    @property
    def _path(self) -> Path:
        warn_deprecated_path_api(f"{self.__class__.__name__}._path")
        return self._proj_path / self._rel_path

    @property
    def _remote_path(self) -> Path:
        warn_deprecated_path_api(f"{self.__class__.__name__}._remote_path")
        return Path(self._proj_path.name) / self._rel_path


class _SimulationSurface:
    """Model-surface coordinate helper bound to one simulation."""

    def __init__(
        self, simulation: "SeismicSimulation", system: CoordinateSystem
    ) -> None:
        self._simulation = simulation
        self._system = system

    @property
    def coordinate_system(self) -> CoordinateSystem:
        """Registered coordinate system used by this helper."""

        return self._system

    def points(
        self,
        values: Any,
        *,
        units: Optional[Any] = None,
        offset: Optional[Any] = None,
    ):
        """Return points on this model surface."""

        return self._system.points(values, units=units, offset=offset)

    on = points

    def above(
        self,
        values: Any,
        distance: Optional[Any] = None,
        *,
        units: Optional[Any] = None,
    ):
        """Return points offset above this model surface."""

        return self._system.above(values, distance=distance, units=units)

    def below(
        self,
        values: Any,
        distance: Optional[Any] = None,
        *,
        units: Optional[Any] = None,
    ):
        """Return points offset below this model surface."""

        return self._system.below(values, distance=distance, units=units)


def _model_surface_name(surface: Union[str, int]) -> str:
    if isinstance(surface, int):
        return f"surface_{surface}"
    return str(surface)


def _coerce_model(model: Optional[Union[ModelBase, Mapping[str, Any]]]) -> ModelBase:
    if model is None:
        return ModelBase()
    if isinstance(model, Mapping):
        return ModelBase.from_fs(model)
    if not isinstance(model, ModelBase):
        raise TypeError(f"Expected ModelBase or mapping, got {type(model).__name__}")
    return model


def _coerce_mesh(
    mesh: Optional[Union[MeshManager, BaseMeshGenerator, Mapping[str, Any]]],
) -> MeshManager:
    if mesh is None:
        return MeshManager()
    if isinstance(mesh, MeshManager):
        return mesh
    if isinstance(mesh, BaseMeshGenerator):
        return MeshManager(mesh)
    if isinstance(mesh, Mapping):
        return MeshManager.from_fs(mesh)
    raise TypeError(
        f"Expected MeshManager, mesh generator, or mapping, got {type(mesh).__name__}"
    )


def _coerce_boundary_conditions(bcs: Any) -> BoundaryConditions:
    if bcs is None:
        return BoundaryConditions()
    if isinstance(bcs, BoundaryConditions):
        return bcs
    if isinstance(bcs, BoundaryCondition):
        return BoundaryConditions([bcs])
    if isinstance(bcs, Mapping):
        raise TypeError("BCs must be a list of boundary condition payloads")
    return BoundaryConditions(bcs)


def _coerce_named_coordinate_systems(systems: Any) -> NamedList:
    if systems is None:
        return NamedList()
    if isinstance(systems, Mapping):
        systems = [systems]
    return NamedList(
        CoordinateSystem.from_fs(system) if isinstance(system, Mapping) else system
        for system in systems
    )


def _project_path_from_simulation_file(path: Path) -> Optional[Path]:
    path = Path(path).expanduser().resolve()
    if path.parent.parent.name == "simulations":
        return path.parent.parent.parent.resolve()
    return None
