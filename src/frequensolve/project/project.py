import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import toml
import yaml

from frequensolve.project.migrate_version import *  # noqa
from frequensolve.project.workflows import *  # noqa
from frequensolve.simulation.simulation import *  # noqa
from frequensolve.util.named_list import NamedList

__all__ = ["Project", "BaseProjectComponent"]


class BaseProjectComponent(ABC):
    """Base class for additional project components."""

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass


@dataclass(kw_only=True)
class Project:
    """Container for storing project information.

    Attributes:
       name (str):                      The name of the project.
       pretty_name (str):               The pretty name of the project.
       path (str):                      The path to the project directory.
       problems (List[Problem]):        List of problems in the project.
       workflows (List[Workflow]):      List of workflows in the project.
       version (str):                   FrequenSolve version for this project.
    """

    name: str
    pretty_name: Optional[str] = None
    path: Union[str, Path]
    version: Version = field(default_factory=Version)
    load_if_exists: bool = False
    auto_migrate: bool = False
    site: Optional[str] = None
    simulations: NamedList[BaseSimulation] = field(default_factory=NamedList)
    workflows: Dict[str, BaseWorkflow] = field(default_factory=dict)
    extras: Dict[str, BaseProjectComponent] = field(default_factory=dict)

    def __post_init__(self):
        """Load project from file and check version."""
        self.path = Path(self.path)
        if self.load_if_exists:
            if self.path.exists() and self.path.suffix == ".json":
                self = Project.load(self.path)

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

        self._set_path_deep()

    def check_version(self):
        """Check project version against current version and migrate if necessary."""
        current_version = Version.current()
        if self.version < current_version:
            if self.auto_migrate:
                self.migrate(current_version)
                self.version = current_version
                self.save()
            else:
                # TODO: show changes to user
                pass

    @classmethod
    def copy(cls, src: Union[str, Path], dest: Union[str, Path], **kwargs):
        load_if_exists = kwargs.get("load_if_exists", False)

        if load_if_exists:
            if Path(dest).exists():
                json_files = list(Path(dest).glob("*.json"))
                if len(json_files) == 1:
                    return cls.load(json_files[0])

        dest = Path(dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        old = Project.load(src)

        name = kwargs.get("name", old.name)
        pretty_name = kwargs.get("pretty_name", old.pretty_name)
        version = kwargs.get("version", old.version)

        new = Project(
            name=name,
            pretty_name=pretty_name,
            path=dest,
            version=version,
            simulations=old.simulations,
            workflows=old.workflows,
            extras=old.extras,
        )
        new.save()
        del new, old

        return Project.load(dest / f"{name}.json")

    @classmethod
    def load(cls, file: Union[str, Path], auto_migrate: bool = False) -> "Project":
        """Load project from JSON file."""
        try:
            file_in = Path(file).resolve()
        except Exception as e:
            raise ValueError(f"Failed to load project: {e}")

        path = file_in.parent

        try:
            if file_in.exists():
                with open(file_in, "r") as f:
                    data = json.load(f)

                name = data.get("name")
                version = Version.from_string(data.get("version"))
                pretty_name = data.get("pretty_name")
                sim_files = data.get("simulations", [])
                wf_files = data.get("workflows", [])
                extra_files = data.get("extras", [])

                if name is None or version is None:
                    raise ValueError(
                        "Project JSON must include 'name' and 'version' fields."
                    )

                project = cls(
                    name=name,
                    pretty_name=pretty_name,
                    path=path,
                    version=version,
                    load_if_exists=False,
                )
                # project.check_version()

                # Change directory to project path for loading files
                current_dir = os.getcwd()
                os.chdir(path)

                # Load simulations
                for f in sim_files:
                    f = Path(path) / f
                    sim = SeismicSimulation.load(f)
                    project.simulations.append(sim)

                # TODO: Load workflows
                # for file in data["workflows"]:
                #    wf = Workflow.load(file)
                #    project.workflows[wf.name] = wf

                # TODO: Load extra components (survey, etc.)
                # for file in extra_files:
                #    extra = BaseProjectComponent.load(file)
                #    project.extras[extra.name] = extra

                os.chdir(current_dir)
                project._set_path_deep()
                return project
            else:
                raise FileNotFoundError(f"Project file not found: {file}")
        except Exception as e:
            raise ValueError(f"Failed to load project: {e}")

    def save(self, file: Optional[Union[str, Path]] = None, **kwargs) -> str:
        """Save project to JSON file."""

        self._set_path_deep()

        # TODO: file does nothing right now
        if file is None:
            file = Path(self.path) / f"{self.name}.json"
        else:
            file = Path(file)

        dict = {
            "name": self.name,
            "path": str(self.path),
            **({"pretty_name": self.pretty_name} if self.pretty_name else {}),
            "version": str(self.version),
        }
        sims = []
        for sim in self.simulations:
            path = Path(self.path) / "simulations"
            sim_file = sim.save(path, **kwargs)
            sims.append(sim_file)
        dict["simulations"] = sims

        indent = kwargs.get("indent", 3)
        with open(file, "w") as f:
            json.dump(dict, f, cls=CustomJSONEncoder, indent=indent, **kwargs)

        return str(file)

    def as_json(self, **kwargs) -> str:
        indent = kwargs.get("indent", 3)
        return json.dumps(self.__dict__(), cls=CustomJSONEncoder, **kwargs)

    def as_toml(self, **kwargs) -> str:
        """Convert project to TOML string."""
        indent = kwargs.get("indent", 3)
        try:
            return toml.dumps(self.__dict__(), encoder=CustomTOMLEncoder(), **kwargs)
        except Exception as e:
            print(f"Failed to convert to TOML: {e}")
            return self.__repr__()

    def as_yaml(self, **kwargs) -> str:
        def numpy_representer(dumper, data):
            """Convert numpy values to native Python types."""
            return dumper.represent_float(float(data))

        indent = kwargs.get("indent", 3)
        try:
            import numpy as np

            yaml.add_representer(np.float64, numpy_representer)
            yaml.add_representer(np.float32, numpy_representer)
            yaml.add_representer(
                np.int64, lambda dumper, data: dumper.represent_int(int(data))
            )
            yaml.add_representer(
                np.int32, lambda dumper, data: dumper.represent_int(int(data))
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

    def __dict__(self) -> Dict:
        return {
            "name": self.name,
            **({"pretty_name": self.pretty_name} if self.pretty_name else {}),
            "version": str(self.version),
            "simulations": [sim.__dict__() for sim in self.simulations],
            "workflows": [wf.__dict__() for wf in self.workflows.values()],
            "extras": [extra.__dict__() for extra in self.extras.values()],
        }

    def __iadd__(
        self, base: Union[BaseSimulation, BaseWorkflow, BaseProjectComponent]
    ) -> "Project":
        """Overrides += operator"""
        if isinstance(base, BaseSimulation):
            self.simulations.append(base)
        elif isinstance(base, BaseWorkflow):
            self.workflows[base.name] = base
        elif isinstance(base, BaseProjectComponent):
            self.extras[base.name] = base
        self._set_path_deep()
        return self

    def _set_path_deep(self):
        proj_path = self.path.resolve()
        rel_path = Path("./simulations")
        for sim in self.simulations:
            sim._set_path(proj_path, rel_path)

    def __repr__(self) -> str:
        return f"Project(name='{self.name}', path='{self.path}')"

    # def new_TD_simulation(self,
    #                       name:      str,
    #                       physics:   Literal["coupled","acoustic","elastic"],
    #                       dimension: Literal[2, 3],
    #                       **kwargs) -> BaseSimulation:
    #    """Create a new time-domain simulation and add it to the project.

    #    Args:
    #       name (str):      Simulation name.
    #       mode (str):      Simulation mode (forward, adjoint, gradient)
    #       physics (str):   Simulation physics (coupled, acoustic, elastic)
    #       dimension (int): Simulation dimension (2, 3)
    #       f_min (float):   Minimum frequency (Hz).
    #       f_max (float):   Maximum frequency (Hz).
    #       df (float):      Frequency step (Hz).
    #       **kwargs: Additional keyword arguments to pass to the Simulation constructor.

    #    Returns:
    #       Simulation: The newly created simulation.
    #    """
    #    tf_domain = "time"
    #    sampling = UniformSweepSampling(f_min = f_min, f_max = f_max, df = df)

    #    sim = SeismicSimulation(name      = name,
    #                            physics   = physics,
    #                            dimension = dimension,
    #                            directory = os.path.join(self.path,name),
    #                            tf_domain = tf_domain,
    #                            sampling  = sampling)
    #    self.simulations.append(sim)
    #    return sim

    # def new_FD_simulation(self,
    #                       name:      str,
    #                       mode:      Literal["forward","adjoint","combined","gradient"],
    #                       physics:   Literal["coupled","acoustic","elastic"],
    #                       dimension: Literal[2, 3],
    #                       f_list:    List[float],
    #                       **kwargs) -> Simulation:
    #    """Create a new frequency-domain simulation and add it to the project.

    #    Args:
    #       name (str):      Simulation name.
    #       mode (str):      Simulation mode (forward, adjoint, gradient)
    #       physics (str):   Simulation physics (coupled, acoustic, elastic)
    #       dimension (int): Simulation dimension (2, 3)
    #       f_list (List[float]): List of frequencies (Hz).
    #       **kwargs: Additional keyword arguments to pass to the Simulation constructor.

    #    Returns:
    #       Simulation: The newly created simulation.
    #    """
    #    tf_domain = "frequency"
    #    sampling = DiscreteSampling(freq = f_list)

    #    sim = Simulation(name      = name,
    #                     physics   = physics,
    #                     dimension = dimension,
    #                     directory = os.path.join(self.path,name),
    #                     mode      = mode,
    #                     tf_domain = tf_domain,
    #                     sampling  = sampling)
    #    self.simulations.append(sim)
    #    return sim
