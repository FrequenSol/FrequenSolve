import asyncio
import json
import logging
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
from warnings import warn

from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.project.migrate_version import Version
from frequensolve.project.workflows import BaseWorkflow
from frequensolve.seismic.record_database import RecordDatabase
from frequensolve.simulation.simulation import BaseSimulation, SeismicSimulation
from frequensolve.util.encoders import CustomJSONEncoder
from frequensolve.util.named_list import NamedList
from frequensolve.util.setup_logger import disable_jupyter_logging, set_log_level

# Import LocalSite conditionally (requires dask dependency)
try:
    from frequensolve.orchestrator.sites.local import LocalSite

    HAS_LOCAL_SITE = True
except ImportError:
    LocalSite = None
    HAS_LOCAL_SITE = False

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
    log_level: int = logging.DEBUG
    jupyter_logging: bool = True
    load_if_exists: bool = False
    auto_migrate: bool = False
    simulations: NamedList[BaseSimulation] = field(default_factory=NamedList)
    workflows: Dict[str, BaseWorkflow] = field(default_factory=dict)
    extras: Dict[str, BaseProjectComponent] = field(default_factory=dict)
    _active_jobs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Load project from file and check version."""
        self.path = Path(self.path).resolve()
        if self.load_if_exists:
            if self.path.exists() and self.path.suffix == ".json":
                self = Project.load(self.path)

        set_log_level(self.log_level)

        if not self.jupyter_logging:
            disable_jupyter_logging()

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
        """Copy a project from an existing project."""
        from shutil import copytree

        load_if_exists = kwargs.get("load_if_exists", False)

        if load_if_exists:
            if Path(dest).exists():
                json_files = list(Path(dest).glob("*.json"))
                if len(json_files) == 1:
                    return cls.load(json_files[0])

        src = Path(src).resolve()
        old = Project.load(src)

        dest = Path(dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        # TODO: optionally simlink instead.
        if (src.parent / "simulations").exists():
            copytree(
                src.parent / "simulations", dest / "simulations", dirs_exist_ok=True
            )
        if (src.parent / "jobs").exists():
            copytree(src.parent / "jobs", dest / "jobs", dirs_exist_ok=True)

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

    def _transfer(self, site: BaseSite):
        """Transfer project files to remote site with path substitution."""

        if HAS_LOCAL_SITE and isinstance(site, LocalSite):
            return

        # Use project directory name (e.g., "ex_01") as the base remote path
        # This matches how jobs determine the project name in AWSSite.submit()
        project_dir_name = Path(self.path).name
        remote = site.work_dir / project_dir_name
        proj_file = (Path(self.path) / f"{self.name}").with_suffix(".json")
        sim_dir = Path(self.path) / "simulations"

        # Create temporary directory for all files to transfer
        temp_dir = Path(self.path) / ".temp_transfer"
        temp_dir.mkdir(exist_ok=True)

        if proj_file.exists():
            site.put(proj_file, (remote / f"{self.name}").with_suffix(".json"))

        if sim_dir.exists():
            shutil.copytree(sim_dir, temp_dir / "simulations", dirs_exist_ok=True)

            # Create temporary modified simulation files
            temp_files = []
            for sim in self.simulations:
                if hasattr(sim, "_file") and sim._file:
                    # Load simulation file
                    with open(sim._file, "r") as f:
                        sim_data = json.load(f)

                    # Create temporary file with modified project_path
                    if "project_path" in sim_data:
                        rel_path = Path(sim._file).relative_to(self.path)
                        temp_file = temp_dir / rel_path
                        sim_data["project_path"] = str(remote)
                        with open(temp_file, "w") as f:
                            json.dump(sim_data, f, cls=CustomJSONEncoder, indent=3)

            # Copy mesh files to temp directory
            for sim in self.simulations:
                if sim.mesh.file is not None:
                    mesh_file = self.path / sim.mesh.file
                    dest = temp_dir / mesh_file.relative_to(self.path)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(mesh_file, dest)

            # # Recursively list all files in temp_dir for debugging
            # print(f"Files to be transferred to {remote}:")
            # for root, dirs, files in os.walk(temp_dir):
            #     for file in files:
            #         rel_path = Path(root).relative_to(temp_dir)
            #         if rel_path == Path("."):
            #             print(f"  {file}")
            #         else:
            #             print(f"  {rel_path / file}")

            site.put(temp_dir, remote)
            shutil.rmtree(temp_dir)

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
                for sim_file in sim_files:
                    sim_file = Path(path) / sim_file
                    sim = SeismicSimulation.load(sim_file)
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

    def save(self, file: Optional[Union[str, Path]] = None, **json_kwargs) -> str:
        """Save project to JSON file."""

        self._set_path_deep()

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
            sim_file = sim.save(**json_kwargs)
            sims.append(sim_file)
        dict["simulations"] = sims

        indent = json_kwargs.pop("indent", 3)
        with open(file, "w") as f:
            json.dump(dict, f, cls=CustomJSONEncoder, indent=indent, **json_kwargs)
        return file

    def as_json(self, **kwargs) -> str:
        return json.dumps(self.__dict__(), cls=CustomJSONEncoder, **kwargs)

    def __dict__(self) -> Dict:
        return {
            "name": self.name,
            **({"pretty_name": self.pretty_name} if self.pretty_name else {}),
            "version": str(self.version),
            "simulations": [sim.__dict__() for sim in self.simulations],
            "workflows": [wf.__dict__() for wf in self.workflows.values()],
            "extras": [extra.__dict__() for extra in self.extras.values()],
        }

    def new_simulation(
        self, name: str, physics: str, dimension: int, **kwargs
    ) -> BaseSimulation:
        """Create a new simulation in the project.

        Args:
            name (str): The name of the simulation.
            physics (str): The physics of the simulation.
            dimension (int): The dimension of the simulation.
        """
        sim = SeismicSimulation(
            name=name,
            physics=physics,
            dimension=dimension,
            project_path=self.path,
        )
        sim.kwargs = kwargs
        self.simulations.append(sim)
        return sim

    def __iadd__(
        self, base: Union[BaseSimulation, BaseWorkflow, BaseProjectComponent]
    ) -> "Project":
        """Overrides += operator to add simulations, workflows, and extras to the project."""
        if isinstance(base, BaseSimulation):
            base.project_path = self.path
            self.simulations.append(base)
        elif isinstance(base, BaseWorkflow):
            base.project_path = self.path
            self.workflows[base.name] = base
        elif isinstance(base, BaseProjectComponent):
            base.project_path = self.path
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

    def terminate_jobs(self):
        """Terminate all running jobs associated with this project."""
        # TODO: on job submission, point to site

        # TODO: keep list of active sites;
        # TODO: get_site should check if the site is active otherwise it should connect and try to cancel the job
        for job, future in list(self._active_jobs.items()):
            try:
                if hasattr(future, "cancel"):  # Dask future
                    future.cancel()
                else:  # Job ID
                    NotImplementedError("Job cancelation needs work")
                    # job.get_site().cancel_job(future)
                logging.info(f"Terminated job {job}")
            except Exception as e:
                logging.error(f"Failed to terminate job {job}: {e}")
            self._active_jobs.pop(job)

    def __del__(self):
        """Cleanup method called when project object is destroyed."""
        self.terminate_jobs()
