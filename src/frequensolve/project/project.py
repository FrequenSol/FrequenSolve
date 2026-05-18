import json
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.project.migrate_version import Version
from frequensolve.simulation.simulation import BaseSimulation, SeismicSimulation
from frequensolve.units import UnitConfig
from frequensolve.util.encoders import CustomJSONEncoder
from frequensolve.util.named_list import NamedList
from frequensolve.util.setup_logger import (
    configure_logging,
    disable_jupyter_logging,
    normalize_log_level,
)

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
    """Project container for simulations, persisted inputs, and run state."""

    name: str
    pretty_name: Optional[str] = None
    path: Union[str, Path]
    version: Version = field(default_factory=Version)
    log_level: Union[int, str] = logging.INFO
    log_file: Optional[Union[str, Path]] = None
    log_to_console: bool = False
    dependency_log_level: Optional[Union[int, str]] = logging.WARNING
    jupyter_logging: bool = True
    load_if_exists: bool = False
    auto_migrate: bool = False
    simulations: NamedList[BaseSimulation] = field(default_factory=NamedList)
    extras: Dict[str, BaseProjectComponent] = field(default_factory=dict)
    _active_jobs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Load project from file and check version."""
        self.path = Path(self.path).resolve()
        log_level = self.log_level
        log_file = self.log_file
        log_to_console = self.log_to_console
        dependency_log_level = self.dependency_log_level
        jupyter_logging = self.jupyter_logging
        if self.load_if_exists:
            project_file = self._project_file(self.path, self.name)
            if project_file.exists():
                loaded = Project.load(project_file, auto_migrate=self.auto_migrate)
                self.__dict__.update(loaded.__dict__)
                self.log_level = log_level
                self.log_file = log_file
                self.log_to_console = log_to_console
                self.dependency_log_level = dependency_log_level
                self.jupyter_logging = jupyter_logging
                self._configure_logging()
                return

        self._configure_logging()

        if not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

        self._set_path_deep()

    def _configure_logging(self) -> None:
        """Apply project-level logging preferences."""

        configure_logging(
            level=self.log_level,
            log_file=self.log_file,
            console=self.log_to_console,
            dependency_level=self.dependency_log_level,
        )
        if not self.jupyter_logging:
            disable_jupyter_logging()

    def check_version(self):
        """Check project version against current version and migrate if necessary."""
        try:
            current_version = Version.current()
        except Exception:
            logging.debug("Skipping project version check for non-release SDK version")
            return
        if self.version < current_version:
            if self.auto_migrate:
                raise NotImplementedError(
                    "Project migrations are not implemented for this SDK release; "
                    f"project version is {self.version}, current version is {current_version}."
                )
            else:
                logging.warning(
                    "Project %s was created with version %s; current SDK version is %s.",
                    self.name,
                    self.version,
                    current_version,
                )

    @staticmethod
    def _project_file(path: Union[str, Path], name: Optional[str] = None) -> Path:
        path = Path(path).expanduser().resolve()
        if path.suffix == ".json":
            return path
        if name is not None:
            return path / f"{name}.json"
        json_files = sorted(path.glob("*.json"))
        if len(json_files) == 1:
            return json_files[0]
        if not json_files:
            raise FileNotFoundError(f"No project JSON file found in {path}")
        names = ", ".join(file.name for file in json_files)
        raise ValueError(
            f"Multiple project JSON files found in {path}; specify one explicitly: {names}"
        )

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

        src_file = cls._project_file(src)
        old = Project.load(src_file)

        dest = Path(dest).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        src_root = src_file.parent
        if (src_root / "simulations").exists():
            copytree(src_root / "simulations", dest / "simulations", dirs_exist_ok=True)
        if (src_root / "jobs").exists():
            copytree(src_root / "jobs", dest / "jobs", dirs_exist_ok=True)

        name = kwargs.get("name", old.name)
        pretty_name = kwargs.get("pretty_name", old.pretty_name)
        version = kwargs.get("version", old.version)

        new = Project(
            name=name,
            pretty_name=pretty_name,
            path=dest,
            version=version,
            log_level=kwargs.get("log_level", old.log_level),
            log_file=kwargs.get("log_file", old.log_file),
            log_to_console=kwargs.get("log_to_console", old.log_to_console),
            dependency_log_level=kwargs.get(
                "dependency_log_level", old.dependency_log_level
            ),
            jupyter_logging=kwargs.get("jupyter_logging", old.jupyter_logging),
            simulations=old.simulations,
            extras=old.extras,
        )
        # Update project_path on simulations so they save to the new location
        for sim in new.simulations:
            if hasattr(sim, "project_path"):
                sim.project_path = dest
        new.save()
        del new, old

        return Project.load(dest / f"{name}.json")

    def _transfer(self, site: BaseSite):
        """Transfer project files to remote site with path substitution."""
        self.save()
        site_class = site.__class__

        if site_class.__name__ == "LocalSite" and site_class.__module__.endswith(
            ".local"
        ):
            return

        if site_class.__name__ == "AWSSite" and ".aws" in site_class.__module__:
            project_dir_name = Path(self.path).name
            remote = site.work_dir / project_dir_name
        else:
            remote = site.work_dir
        proj_file = (Path(self.path) / f"{self.name}").with_suffix(".json")
        sim_dir = Path(self.path) / "simulations"

        if proj_file.exists():
            site.put(proj_file, (remote / f"{self.name}").with_suffix(".json"))

        if sim_dir.exists():
            with tempfile.TemporaryDirectory(
                prefix=".fs_transfer_", dir=self.path
            ) as temp:
                temp_dir = Path(temp)
                shutil.copytree(sim_dir, temp_dir / "simulations", dirs_exist_ok=True)

                for sim in self.simulations:
                    if hasattr(sim, "_file") and sim._file:
                        with open(sim._file, "r") as f:
                            sim_data = json.load(f)

                        if "project_path" in sim_data:
                            # sim._file can point outside self.path after project copies.
                            rel_path = (
                                Path("simulations") / sim.name / f"{sim.name}.json"
                            )
                            temp_file = temp_dir / rel_path
                            temp_file.parent.mkdir(parents=True, exist_ok=True)
                            sim_data["project_path"] = str(remote)
                            with open(temp_file, "w") as f:
                                json.dump(
                                    sim_data,
                                    f,
                                    cls=CustomJSONEncoder,
                                    indent=3,
                                )

                for sim in self.simulations:
                    if sim.mesh.file is not None:
                        mesh_file = self.path / sim.mesh.file
                        dest = temp_dir / mesh_file.relative_to(self.path)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(mesh_file, dest)

                site.put(temp_dir, remote)

    @classmethod
    def load(cls, file: Union[str, Path], auto_migrate: bool = False) -> "Project":
        """Load project from JSON file."""
        try:
            file_in = cls._project_file(file)
        except Exception as e:
            raise ValueError(f"Failed to load project: {e}")

        path = file_in.parent

        if not file_in.exists():
            raise ValueError(f"Failed to load project: Project file not found: {file}")

        try:
            with open(file_in, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load project JSON {file_in}: {e}") from e

        try:
            name = data.get("name")
            version = Version.from_string(data.get("version"))
            pretty_name = data.get("pretty_name")
            sim_files = data.get("simulations", [])
            logging_config = data.get("logging", {})
            if name is None or version is None:
                raise ValueError("Project JSON must include 'name' and 'version'.")

            project = cls(
                name=name,
                pretty_name=pretty_name,
                path=path,
                version=version,
                log_level=logging_config.get("level", logging.INFO),
                log_file=logging_config.get("file"),
                log_to_console=logging_config.get("console", False),
                dependency_log_level=logging_config.get(
                    "dependency_level", logging.WARNING
                ),
                jupyter_logging=logging_config.get("jupyter", True),
                load_if_exists=False,
                auto_migrate=auto_migrate,
            )

            for sim_file in sim_files:
                sim_file = Path(sim_file)
                if not sim_file.is_absolute():
                    sim_file = path / sim_file
                sim = SeismicSimulation.load(sim_file)
                project.simulations.append(sim)

            project._set_path_deep()
            project.check_version()
            return project
        except Exception as e:
            raise ValueError(f"Failed to load project {file_in}: {e}") from e

    def save(self, file: Optional[Union[str, Path]] = None, **json_kwargs) -> Path:
        """Save project to JSON file."""

        self._set_path_deep()

        file = Path(self.path) / f"{self.name}.json" if file is None else Path(file)
        file = file.expanduser().resolve()
        file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "name": self.name,
            "path": str(self.path),
            **({"pretty_name": self.pretty_name} if self.pretty_name else {}),
            "version": str(self.version),
        }
        logging_payload = self._logging_payload()
        if logging_payload:
            payload["logging"] = logging_payload
        sims = []
        for sim in self.simulations:
            sim_file = sim.save(**json_kwargs)
            try:
                sims.append(str(Path(sim_file).resolve().relative_to(self.path)))
            except ValueError:
                sims.append(str(Path(sim_file).resolve()))
        payload["simulations"] = sims

        indent = json_kwargs.pop("indent", 3)
        tmp_file = file.with_name(f".{file.name}.tmp")
        with open(tmp_file, "w") as f:
            json.dump(payload, f, cls=CustomJSONEncoder, indent=indent, **json_kwargs)
        tmp_file.replace(file)
        return file

    def _logging_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if normalize_log_level(self.log_level) != logging.INFO:
            payload["level"] = self.log_level
        if self.log_file is not None:
            payload["file"] = str(self.log_file)
        if self.log_to_console:
            payload["console"] = True
        if self.dependency_log_level is None:
            payload["dependency_level"] = None
        elif normalize_log_level(self.dependency_log_level) != logging.WARNING:
            payload["dependency_level"] = self.dependency_log_level
        if not self.jupyter_logging:
            payload["jupyter"] = False
        return payload

    def as_json(self, **kwargs) -> str:
        return json.dumps(self.to_fs(), cls=CustomJSONEncoder, **kwargs)

    def to_fs(self) -> Dict:
        return {
            "name": self.name,
            **({"pretty_name": self.pretty_name} if self.pretty_name else {}),
            "version": str(self.version),
            "simulations": [sim.to_fs() for sim in self.simulations],
            "extras": [
                extra.to_fs() if hasattr(extra, "to_fs") else dict(extra)
                for extra in self.extras.values()
            ],
        }

    def new_simulation(
        self, name: str, physics: str, dimension: int | float | str, **kwargs
    ) -> BaseSimulation:
        """Create a new simulation in the project.

        Args:
            name (str): The name of the simulation.
            physics (str): The physics of the simulation.
            dimension (int | float | str): The simulation dimension, accepting
                2D, 2.5D, and 3D forms.
            units/default_units (Mapping[str, Any]): Optional simulation-level
                default output units. Values may be Pint units or unit strings.
        """
        extra = dict(kwargs.pop("extra", {}) or {})
        axisymmetric = kwargs.pop("axisymmetric", False)
        default_units = kwargs.pop("default_units", None)
        units = kwargs.pop("units", None)
        unit_config = kwargs.pop("unit_config", None)
        if units is not None:
            if isinstance(units, UnitConfig):
                if unit_config is not None:
                    raise ValueError("Pass only one of units and unit_config")
                unit_config = units
            else:
                if default_units is not None:
                    raise ValueError("Pass only one of units and default_units")
                default_units = units
        extra.update(kwargs)
        sim = SeismicSimulation(
            name=name,
            physics=physics,
            dimension=dimension,
            axisymmetric=axisymmetric,
            project_path=self.path,
            extra=extra,
        )
        if unit_config is not None:
            if not isinstance(unit_config, UnitConfig):
                raise TypeError(
                    "unit_config must be a frequensolve.units.UnitConfig instance"
                )
            sim.units = unit_config
        if default_units is not None:
            if not isinstance(default_units, Mapping):
                raise TypeError(
                    "default_units must be a mapping of quantity names to units"
                )
            sim.units.defaults.update(default_units)
        sim._project = self
        self.simulations.append(sim)
        return sim

    def __iadd__(self, base: Union[BaseSimulation, BaseProjectComponent]) -> "Project":
        """Add simulations and project components to the project."""
        if isinstance(base, BaseSimulation):
            base.project_path = self.path
            base._project = self
            self.simulations.append(base)
        elif isinstance(base, BaseProjectComponent):
            base.project_path = self.path
            self.extras[base.name] = base
        else:
            raise TypeError(f"Cannot add {type(base).__name__} to Project")
        self._set_path_deep()
        return self

    def _set_path_deep(self):
        proj_path = self.path.resolve()
        rel_path = Path("./simulations")
        for sim in self.simulations:
            sim._project = self
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
        try:
            self.terminate_jobs()
        except Exception:
            pass
