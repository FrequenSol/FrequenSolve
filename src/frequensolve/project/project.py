import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

from frequensolve.orchestrator.sites.frontera import FronteraSite
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project.migrate_version import Version
from frequensolve.project.workflows import BaseWorkflow
from frequensolve.seismic.record_database import RecordDatabase
from frequensolve.simulation.simulation import BaseSimulation, SeismicSimulation
from frequensolve.util.encoders import CustomJSONEncoder, CustomTOMLEncoder
from frequensolve.util.named_list import NamedList
from frequensolve.util.setup_logger import disable_jupyter_logging, set_log_level

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
    site: Optional[str] = None
    simulations: NamedList[BaseSimulation] = field(default_factory=NamedList)
    workflows: Dict[str, BaseWorkflow] = field(default_factory=dict)
    extras: Dict[str, BaseProjectComponent] = field(default_factory=dict)
    _active_jobs: Dict[str, Any] = field(
        default_factory=dict
    )  # job_name -> future/job_id

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

    def transfer(self):
        """Transfer project files to remote site with path substitution."""

        if isinstance(self.site, LocalSite):
            return

        remote = self.site.work_dir

        proj_file = (Path(self.path) / f"{self.name}").with_suffix(".json")
        sim_dir = Path(self.path) / "simulations"

        if proj_file.exists():
            self.site.put(proj_file, (remote / f"{self.name}").with_suffix(".json"))

        if sim_dir.exists():
            # Create temporary modified simulation files
            temp_files = []
            for sim in self.simulations:
                if hasattr(sim, "_file") and sim._file:
                    # Load simulation file
                    with open(sim._file, "r") as f:
                        sim_data = json.load(f)

                    # Create temporary file with modified project_path
                    if "project_path" in sim_data:
                        temp_file = Path(sim._file).with_suffix(".temp.json")
                        sim_data["project_path"] = str(remote)
                        with open(temp_file, "w") as f:
                            json.dump(sim_data, f, cls=CustomJSONEncoder, indent=3)
                        temp_files.append(
                            (temp_file, Path(sim._file).relative_to(self.path))
                        )

            # Transfer simulation directory
            self.site.put(sim_dir, remote / "simulations")

            # Transfer modified simulation files
            for temp_file, rel_path in temp_files:
                self.site.put(temp_file, remote / rel_path)
                temp_file.unlink()  # Clean up temporary file

    def get_records(self, results: dict) -> RecordDatabase:
        """Get records from Frontera.

        Args:
            results: A dictionary of results from a Frontera job.
        """

        if self.site is None:
            raise ValueError("No site specified for project")
        elif isinstance(self.site, LocalSite):
            pass
        elif isinstance(self.site, FronteraSite):
            self.site.download_records(results, self.path.resolve())
        else:
            raise NotImplementedError(
                f"Site type {type(self.site)} not supported (yet)"
            )
        return RecordDatabase.from_results(results, self.path.resolve())

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
        import toml

        indent = kwargs.get("indent", 3)
        try:
            return toml.dumps(self.__dict__(), encoder=CustomTOMLEncoder(), **kwargs)
        except Exception as e:
            print(f"Failed to convert to TOML: {e}")
            return self.__repr__()

    def as_yaml(self, **kwargs) -> str:
        import yaml

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

    def terminate_jobs(self):
        """Terminate all running jobs associated with this project."""
        if not self.site:
            return

        for job_name, future in list(self._active_jobs.items()):
            try:
                if hasattr(future, "cancel"):  # Dask future
                    future.cancel()
                else:  # Job ID
                    self.site.cancel_job(future)
                logging.info(f"Terminated job {job_name}")
            except Exception as e:
                logging.error(f"Failed to terminate job {job_name}: {e}")

            self._active_jobs.pop(job_name)

    def __del__(self):
        """Cleanup method called when project object is destroyed."""
        self.terminate_jobs()

    def submit_job(self, job, **kwargs) -> list:
        """Submit job and block until completion."""
        return self.site.submit(job, **kwargs)

    def submit_job_async(self, job, **kwargs) -> asyncio.Future:
        """Submit job asynchronously and return a future."""
        future = self.site.submit_async(job, **kwargs)
        self._active_jobs[job.name] = future
        return future
