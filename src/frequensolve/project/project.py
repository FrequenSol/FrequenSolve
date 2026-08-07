"""Project container APIs for saving simulations, jobs, and run artifacts.

``Project`` is the top-level filesystem container for FrequenSolve authoring
work. It creates a project directory, writes project and simulation JSON,
rebinds loaded simulations to the project root, and configures package logging
for local or notebook workflows.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from frequensolve._version import get_versions
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.orchestrator.sites.config_file import _host_tmp_path_for_config
from frequensolve.simulation.simulation import BaseSimulation, SeismicSimulation
from frequensolve.units import UnitConfig
from frequensolve.util.encoders import CustomJSONEncoder
from frequensolve.util.named_list import NamedList
from frequensolve.util.setup_logger import (
    configure_logging,
    disable_jupyter_logging,
    normalize_log_level,
)
from frequensolve.util.store import compact_hdf5_file

__all__ = ["Project", "BaseProjectComponent"]


def _current_sdk_version() -> str:
    return str(get_versions().get("version", "0+unknown"))


class BaseProjectComponent(ABC):
    """Base class for extension objects persisted with a project."""

    @abstractmethod
    def load(self):
        """Load component state from project storage."""

        pass

    @abstractmethod
    def save(self):
        """Persist component state into project storage."""

        pass


@dataclass(kw_only=True)
class Project:
    """Project container for simulations, persisted inputs, and run state.

    Args:
        name: Project name and default JSON file stem.
        pretty_name: Optional display name for user interfaces.
        path: Project root directory. ``~`` is expanded and the result is
            resolved during initialization.
        version: FrequenSolve SDK version that saved the project.
        log_level: FrequenSolve logger level.
        log_file: Optional file path for FrequenSolve logs.
        log_to_console: Whether to emit FrequenSolve logs to the console.
        dependency_log_level: Log level applied to noisy dependencies, or
            ``None`` to leave them unchanged.
        jupyter_logging: Whether to keep notebook display logging enabled.
        load_if_exists: Load an existing project JSON from ``path`` instead of
            creating a new project when possible.
        simulations: Initial simulations to attach to this project.
        extras: Additional project components keyed by name.
    """

    name: str
    pretty_name: Optional[str] = None
    path: Union[str, Path]
    version: str = field(default_factory=_current_sdk_version)
    log_level: Union[int, str] = logging.INFO
    log_file: Optional[Union[str, Path]] = None
    log_to_console: bool = False
    dependency_log_level: Optional[Union[int, str]] = logging.WARNING
    jupyter_logging: bool = True
    load_if_exists: bool = False
    simulations: NamedList[BaseSimulation] = field(default_factory=NamedList)
    extras: Dict[str, BaseProjectComponent] = field(default_factory=dict)
    _active_jobs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.path = Path(self.path).expanduser().resolve()
        self.version = str(self.version)
        logging_options = self._logging_options()

        if self.load_if_exists:
            project_file = self._project_file(self.path, self.name)
            if project_file.exists():
                loaded = type(self).load(project_file)
                self.__dict__.update(loaded.__dict__)
                self._restore_logging_options(logging_options)
                self._configure_logging()
                return

        self._configure_logging()
        self.path.mkdir(parents=True, exist_ok=True)
        self._bind_simulations_to_project()

    @classmethod
    def load(cls, file: Union[str, Path]) -> "Project":
        """Load a project from a JSON file or project directory.

        Args:
            file: Project JSON file, or a project directory containing exactly
                one project JSON file.

        Returns:
            Loaded ``Project`` instance with simulations rebound to the project
            root.

        Raises:
            ValueError: If the project file cannot be located, parsed, or
                deserialized.
        """

        try:
            project_file = cls._project_file(file)
        except Exception as exc:
            raise ValueError(f"Failed to load project: {exc}") from exc

        if not project_file.exists():
            raise ValueError(f"Failed to load project: Project file not found: {file}")

        try:
            data = cls._read_json_file(project_file)
            name = data.get("name")
            version = data.get("version")
            if name is None or version is None:
                raise ValueError("Project JSON must include 'name' and 'version'.")

            logging_config = data.get("logging", {})
            project = cls(
                name=name,
                pretty_name=data.get("pretty_name"),
                path=project_file.parent,
                version=str(version),
                log_level=logging_config.get("level", logging.INFO),
                log_file=logging_config.get("file"),
                log_to_console=logging_config.get("console", False),
                dependency_log_level=logging_config.get(
                    "dependency_level", logging.WARNING
                ),
                jupyter_logging=logging_config.get("jupyter", True),
                load_if_exists=False,
            )
            for sim_file in data.get("simulations", []):
                project.simulations.append(
                    SeismicSimulation.load(
                        project._project_local_path(Path(sim_file)),
                    )
                )
            project._bind_simulations_to_project()
            project.check_version()
            return project
        except Exception as exc:
            raise ValueError(f"Failed to load project {project_file}: {exc}") from exc

    @classmethod
    def copy(cls, src: Union[str, Path], dest: Union[str, Path], **kwargs) -> "Project":
        """Copy a project tree and rewrite project-local paths.

        Args:
            src: Source project JSON file or directory.
            dest: Destination project directory.
            **kwargs: Optional constructor overrides for the copied project,
                such as ``name`` or logging options.

        Returns:
            Reloaded ``Project`` instance rooted at ``dest``.
        """

        dest = Path(dest).expanduser().resolve()
        if kwargs.get("load_if_exists", False):
            existing = cls._load_existing_project(dest)
            if existing is not None:
                return existing

        src_file = cls._project_file(src)
        source = cls.load(src_file)
        source_root = src_file.parent
        dest.mkdir(parents=True, exist_ok=True)
        cls._copy_project_inputs(source_root, dest)

        copied = cls(
            name=kwargs.get("name", source.name),
            pretty_name=kwargs.get("pretty_name", source.pretty_name),
            path=dest,
            version=kwargs.get("version", source.version),
            log_level=kwargs.get("log_level", source.log_level),
            log_file=kwargs.get("log_file", source.log_file),
            log_to_console=kwargs.get("log_to_console", source.log_to_console),
            dependency_log_level=kwargs.get(
                "dependency_log_level", source.dependency_log_level
            ),
            jupyter_logging=kwargs.get("jupyter_logging", source.jupyter_logging),
            simulations=source.simulations,
            extras=source.extras,
        )
        copied.save()
        cls._rewrite_copied_input_files(
            dest,
            source_project=source_root,
            target_project=dest,
        )
        return cls.load(dest / f"{copied.name}.json")

    def save(self, file: Optional[Union[str, Path]] = None, **json_kwargs) -> Path:
        """Save the project and all attached simulations.

        Args:
            file: Optional project JSON path. Relative paths are resolved under
                the project root.
            **json_kwargs: Additional keyword arguments forwarded to
                ``json.dumps`` when writing project and simulation files.

        Returns:
            Path to the written project JSON file.
        """

        self._bind_simulations_to_project()
        project_file = self._project_local_path(file or f"{self.name}.json")
        project_file.parent.mkdir(parents=True, exist_ok=True)

        payload: Dict[str, Any] = {
            "name": self.name,
            "path": str(self.path),
            **({"pretty_name": self.pretty_name} if self.pretty_name else {}),
            "version": str(self.version),
        }
        logging_payload = self._logging_payload()
        if logging_payload:
            payload["logging"] = logging_payload
        payload["simulations"] = [
            self._saved_simulation_path(sim, **json_kwargs) for sim in self.simulations
        ]

        indent = json_kwargs.pop("indent", 3)
        self._write_json_file(project_file, payload, indent=indent, **json_kwargs)
        return project_file

    def as_json(self, **kwargs) -> str:
        """Return the project payload as formatted JSON.

        Args:
            **kwargs: Keyword arguments forwarded to ``json.dumps``.

        Returns:
            Serialized project JSON string.
        """

        return json.dumps(self.to_fs(), cls=CustomJSONEncoder, **kwargs)

    def to_fs(self) -> Dict:
        """Serialize the project to the FrequenSolve project payload.

        Returns:
            JSON-compatible project mapping including attached simulations and
            extension components.
        """

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

    def check_version(self) -> None:
        """Warn when a project was saved by a different SDK version."""

        current_version = _current_sdk_version()
        if self.version == current_version:
            return
        logging.warning(
            "Project %s was saved with FrequenSolve version %s; current SDK "
            "version is %s. Automatic project migrations are not supported.",
            self.name,
            self.version,
            current_version,
        )

    def new_simulation(
        self, name: str, physics: str, dimension: int | float | str, **kwargs
    ) -> BaseSimulation:
        """Create a new simulation in the project.

        Args:
            name (str): The name of the simulation.
            physics (str): The physics of the simulation.
            dimension (int | float | str): The simulation dimension, accepting
                2D, 2.5D, and 3D forms.
            **kwargs: Simulation options. Recognized keys include
                ``axisymmetric``, ``units``, ``unit_config``, and
                ``default_units``; remaining keys are passed through as
                simulation ``extra`` fields.

        Returns:
            Newly created and project-bound ``SeismicSimulation``.
        """

        return self._create_simulation(
            name=name,
            physics=physics,
            dimension=dimension,
            attach=True,
            **kwargs,
        )

    def study(
        self,
        name: str,
        *,
        name_template: Optional[str] = None,
        max_cases: Optional[int] = 1000,
        **parameters: Mapping[str, Any],
    ):
        """Define related simulations from named parameter choices.

        Args:
            name: Study name used by the default simulation name template.
            name_template: Optional Python format string containing ``study``,
                ``index``, or parameter-name fields.
            max_cases: Maximum combinations allowed in one preview or
                materialization, or ``None`` for no limit.
            **parameters: Parameter names mapped to non-empty mappings of
                choice labels to authoring values.

        Returns:
            A :class:`frequensolve.simulation.SimulationStudy` bound to this
            project.
        """

        from frequensolve.simulation.study import SimulationStudy

        return SimulationStudy(
            self,
            name,
            parameters,
            name_template=name_template,
            max_cases=max_cases,
        )

    def _create_simulation(
        self,
        *,
        name: str,
        physics: str,
        dimension: int | float | str,
        attach: bool,
        **kwargs: Any,
    ) -> SeismicSimulation:
        """Construct a project-bound simulation with optional list attachment."""

        options = self._simulation_options(kwargs)
        sim = SeismicSimulation(
            name=name,
            physics=physics,
            dimension=dimension,
            axisymmetric=options["axisymmetric"],
            project_path=self.path,
            extra=options["extra"],
        )
        self._apply_simulation_units(
            sim,
            unit_config=options["unit_config"],
            default_units=options["default_units"],
        )
        sim._project = self
        if attach:
            self.simulations.append(sim)
        return sim

    def list_jobs(
        self,
        *,
        simulation: Optional[Union[str, BaseSimulation]] = None,
    ) -> list[Dict[str, Any]]:
        """List saved project jobs and their persisted result status.

        The returned rows are dictionaries with the saved job path, linked
        simulation, workflow, result path, and two high-level checks:
        ``results_exist`` reports whether the result directory contains
        outputs or run metadata, and ``results_current`` reports whether the
        saved result still matches the current job/simulation definitions.

        Args:
            simulation: Optional simulation name or object used to filter the
                job tree.

        Returns:
            One dictionary per saved job JSON file.
        """

        return [
            self._job_status_row(job_file)
            for job_file in self._job_files(simulation=simulation)
        ]

    def job_file(
        self,
        name: str,
        *,
        simulation: Optional[Union[str, BaseSimulation]] = None,
    ) -> Path:
        """Return the saved job JSON path for a project job.

        Jobs are stored under ``jobs/<simulation>/<job>/<job>.json``. When
        ``simulation`` is omitted, the job name must be unique across the
        project jobs tree.

        Args:
            name: Saved job name.
            simulation: Optional simulation name or object used to disambiguate
                jobs with the same name.

        Returns:
            Absolute path to the saved job JSON file.

        Raises:
            FileNotFoundError: If no matching job JSON exists.
            ValueError: If the job name is ambiguous without ``simulation``.
        """

        job_name = str(name)
        if simulation is not None:
            path = (
                self.path
                / "jobs"
                / self._simulation_name(simulation)
                / job_name
                / f"{job_name}.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"Job JSON file not found: {path}")
            return path.resolve()

        root = self.path / "jobs"
        candidates = sorted(root.glob(f"*/{job_name}/{job_name}.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No job named {job_name!r} found under {root}; pass simulation=..."
            )
        if len(candidates) > 1:
            names = ", ".join(str(path.relative_to(self.path)) for path in candidates)
            raise ValueError(
                f"Multiple jobs named {job_name!r} found; pass simulation=... "
                f"to choose one: {names}"
            )
        return candidates[0].resolve()

    def load_job(
        self,
        name: Union[str, Path],
        *,
        simulation: Optional[Union[str, BaseSimulation]] = None,
    ):
        """Load a saved job from this project without submitting it.

        Args:
            name: Job name or explicit job JSON path.
            simulation: Optional simulation name or object used when ``name`` is
                not a path.

        Returns:
            Loaded ``BaseJob`` subclass instance.
        """

        from frequensolve.simulation.jobs import BaseJob

        path = Path(name).expanduser()
        if path.suffix == ".json" or path.exists():
            job = BaseJob.load(path, project_path=self.path)
        else:
            job = BaseJob.load(
                self.job_file(str(name), simulation=simulation),
                project_path=self.path,
            )

        # Keep the job and project on one simulation object. Remote submission
        # saves the job and then saves/synchronizes the whole project; leaving
        # two same-named objects here could let the project save overwrite
        # edits made through ``job.simulation`` with stale inputs.
        try:
            project_simulation = self.simulations[job.simulation.name]
        except ValueError:
            project_simulation = job.simulation
            self.simulations.append(project_simulation)
        job.simulation = project_simulation
        project_simulation._project = self
        return job

    def terminate_jobs(self) -> None:
        """Terminate all active job futures tracked by this project."""

        for job, future in list(self._active_jobs.items()):
            try:
                if hasattr(future, "cancel"):
                    future.cancel()
                else:
                    NotImplementedError("Job cancelation needs work")
                logging.info("Terminated job %s", job)
            except Exception as exc:
                logging.error("Failed to terminate job %s: %s", job, exc)
            self._active_jobs.pop(job)

    def __iadd__(self, base: Union[BaseSimulation, BaseProjectComponent]) -> "Project":
        """Attach a simulation or extension component to this project.

        Args:
            base: Simulation or project component to attach.

        Returns:
            This project instance.

        Raises:
            TypeError: If ``base`` is not a supported project member.
        """

        if isinstance(base, BaseSimulation):
            base._project = self
            self.simulations.append(base)
        elif isinstance(base, BaseProjectComponent):
            base.project_path = self.path
            self.extras[base.name] = base
        else:
            raise TypeError(f"Cannot add {type(base).__name__} to Project")
        self._bind_simulations_to_project()
        return self

    def __repr__(self) -> str:
        return f"Project(name='{self.name}', path='{self.path}')"

    def __del__(self):
        """Cleanup method called when project object is destroyed."""

        try:
            self.terminate_jobs()
        except Exception:
            pass

    def _transfer(self, site: BaseSite) -> None:
        """Transfer project files to a remote site with path substitution."""

        self.save()
        if self._site_is_local(site):
            return

        remote = self._remote_project_root(site)
        project_file = self.path / f"{self.name}.json"
        if project_file.exists():
            site.put(project_file, remote / f"{self.name}.json")

        sim_dir = self.path / "simulations"
        if not sim_dir.exists():
            return

        with tempfile.TemporaryDirectory(
            prefix="fs_transfer_",
            dir=self._transfer_host_tmp_dir(site),
        ) as tmp:
            temp_dir = Path(tmp)
            shutil.copytree(sim_dir, temp_dir / "simulations", dirs_exist_ok=True)
            self._rewrite_transfer_simulations(temp_dir, remote)
            self._copy_transfer_mesh_files(temp_dir)
            self._compact_transfer_hdf5_files(temp_dir)
            site.put(temp_dir, remote)

    @staticmethod
    def _transfer_host_tmp_dir(site: BaseSite) -> Path:
        path = _host_tmp_path_for_config(getattr(site, "_site_config_path", None))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _configure_logging(self) -> None:
        configure_logging(
            level=self.log_level,
            log_file=self.log_file,
            console=self.log_to_console,
            dependency_level=self.dependency_log_level,
        )
        if not self.jupyter_logging:
            disable_jupyter_logging()

    def _logging_options(self) -> Dict[str, Any]:
        return {
            "log_level": self.log_level,
            "log_file": self.log_file,
            "log_to_console": self.log_to_console,
            "dependency_log_level": self.dependency_log_level,
            "jupyter_logging": self.jupyter_logging,
        }

    def _restore_logging_options(self, options: Mapping[str, Any]) -> None:
        for key, value in options.items():
            setattr(self, key, value)

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

    def _bind_simulations_to_project(self) -> None:
        proj_path = self.path.resolve()
        for sim in self.simulations:
            sim._project = self
            sim.relocate(proj_path)

    def _project_local_path(self, path: Union[str, Path]) -> Path:
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.path / path

    def _saved_simulation_path(self, sim: BaseSimulation, **json_kwargs) -> str:
        sim_file = Path(sim.save(**json_kwargs)).resolve()
        try:
            return str(sim_file.relative_to(self.path))
        except ValueError:
            return str(sim_file)

    @classmethod
    def _load_existing_project(cls, path: Path) -> Optional["Project"]:
        if not path.exists():
            return None
        json_files = sorted(path.glob("*.json"))
        return cls.load(json_files[0]) if len(json_files) == 1 else None

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

    @staticmethod
    def _copy_project_inputs(source_root: Path, target_root: Path) -> None:
        for dirname in ("simulations", "jobs"):
            source = source_root / dirname
            if source.exists():
                shutil.copytree(source, target_root / dirname, dirs_exist_ok=True)

    @classmethod
    def _rewrite_copied_input_files(
        cls,
        project_dir: Path,
        *,
        source_project: Path,
        target_project: Path,
    ) -> None:
        """Rewrite copied simulation/job JSON files to the new project root."""

        for input_file in cls._copied_input_files(project_dir):
            payload = cls._read_json_file(input_file)
            if not isinstance(payload, dict):
                continue
            payload = cls._rewrite_copied_payload(
                payload,
                input_file=input_file,
                source_project=source_project,
                target_project=target_project,
            )
            cls._write_json_file(input_file, payload)

    @staticmethod
    def _copied_input_files(project_dir: Path) -> list[Path]:
        patterns = ("simulations/*/*.json", "jobs/*/*/*.json")
        return [
            path for pattern in patterns for path in sorted(project_dir.glob(pattern))
        ]

    @classmethod
    def _rewrite_copied_payload(
        cls,
        payload: Dict[str, Any],
        *,
        input_file: Path,
        source_project: Path,
        target_project: Path,
    ) -> Dict[str, Any]:
        from frequensolve.simulation.jobs import JobLayout

        path_roots = cls._payload_project_roots(payload, fallback=source_project)
        for root in path_roots:
            payload = cls._map_project_paths(
                payload,
                source_project=root,
                target_project=target_project,
            )
        payload["project_path"] = str(target_project)

        if "simulation" not in payload or "name" not in payload:
            return payload

        layout_payload = {**payload, "project_path": str(path_roots[0])}
        layout = JobLayout.from_payload(layout_payload, job_file=input_file)
        target_layout = layout.with_project(target_project)
        payload["simulation"] = cls._project_relative_or_absolute(
            target_layout.simulation_file,
            target_project,
        )
        payload["result_path"] = cls._project_relative_or_absolute(
            target_layout.result_dir,
            target_project,
        )
        return payload

    @staticmethod
    def _payload_project_roots(
        payload: Mapping[str, Any],
        *,
        fallback: Path,
    ) -> list[Path]:
        roots = []
        for value in (payload.get("project_path"), fallback):
            try:
                root = Path(value).expanduser().resolve()
            except TypeError:
                continue
            if root not in roots:
                roots.append(root)
        return roots or [fallback]

    @staticmethod
    def _map_project_paths(
        value: Any,
        *,
        source_project: Path,
        target_project: Path,
    ) -> Any:
        """Replace absolute local project paths in a JSON-like payload."""

        if isinstance(value, Mapping):
            return {
                key: Project._map_project_paths(
                    item,
                    source_project=source_project,
                    target_project=target_project,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                Project._map_project_paths(
                    item,
                    source_project=source_project,
                    target_project=target_project,
                )
                for item in value
            ]
        if isinstance(value, Path):
            return Project._map_project_paths(
                str(value),
                source_project=source_project,
                target_project=target_project,
            )
        if isinstance(value, str):
            source = str(source_project)
            if source and source in value:
                return value.replace(source, str(target_project))
        return value

    def _rewrite_transfer_simulations(self, temp_dir: Path, remote: Path) -> None:
        for sim in self.simulations:
            sim_file = getattr(sim, "_file", None)
            if not sim_file:
                continue
            rel_path = Path(sim_file).relative_to(self.path)
            temp_file = temp_dir / rel_path
            payload = self._read_json_file(sim_file)
            payload = self._map_project_paths(
                payload,
                source_project=self.path,
                target_project=remote,
            )
            payload["project_path"] = str(remote)
            self._write_json_file(temp_file, payload)

    def _copy_transfer_mesh_files(self, temp_dir: Path) -> None:
        for sim in self.simulations:
            if sim.mesh.file is None:
                continue
            mesh_file = self.path / sim.mesh.file
            dest = temp_dir / mesh_file.relative_to(self.path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mesh_file, dest)

    @staticmethod
    def _compact_transfer_hdf5_files(temp_dir: Path) -> None:
        """Repack bloated HDF5 files in a disposable transfer staging tree."""

        candidates = {
            path
            for pattern in ("*.h5", "*.hdf5")
            for path in temp_dir.rglob(pattern)
            if path.is_file()
        }
        for path in sorted(candidates):
            try:
                reclaimed = compact_hdf5_file(path)
            except (OSError, RuntimeError, ValueError) as exc:
                logging.warning(
                    "Could not compact staged HDF5 file %s: %s",
                    path.relative_to(temp_dir),
                    exc,
                )
                continue
            if reclaimed:
                logging.info(
                    "Compacted staged HDF5 file %s; removed %.2f GiB of dead space",
                    path.relative_to(temp_dir),
                    reclaimed / (1024**3),
                )

    @staticmethod
    def _site_is_local(site: BaseSite) -> bool:
        site_class = site.__class__
        return site_class.__name__ == "LocalSite" and site_class.__module__.endswith(
            ".local"
        )

    def _remote_project_root(self, site: BaseSite) -> Path:
        site_class = site.__class__
        if site_class.__name__ == "AWSSite" and ".aws" in site_class.__module__:
            return site.work_dir / self.path.name
        return site.work_dir

    def _job_files(
        self,
        *,
        simulation: Optional[Union[str, BaseSimulation]] = None,
    ) -> list[Path]:
        root = self.path / "jobs"
        if not root.exists():
            return []

        if simulation is None:
            search_root = root
            pattern = "*/*/*.json"
        else:
            search_root = root / self._simulation_name(simulation)
            pattern = "*/*.json"
        if not search_root.exists():
            return []
        return sorted(
            path.resolve() for path in search_root.glob(pattern) if path.is_file()
        )

    def _job_status_row(self, job_file: Path) -> Dict[str, Any]:
        """Load a saved job and return a compact project listing row."""

        from frequensolve.simulation.jobs import BaseJob

        base = self._base_job_status_row(job_file)
        try:
            job = BaseJob.load(job_file, project_path=self.path)
        except Exception as exc:
            return {**base, "load_error": str(exc)}

        metadata = job.run_metadata
        state = job.run_state()
        try:
            traces_exist = job.trace_outputs_exist()
        except Exception:
            traces_exist = False
        try:
            results_current = job.is_run_current()
        except Exception:
            results_current = False

        return {
            **base,
            "name": job.name,
            "simulation": job.simulation.name,
            "job_type": job.__class__.__name__,
            "workflow": job.workflow,
            "n_tasks": job.n_tasks,
            "result_path": str(job._result_path),
            "loaded": True,
            "results_exist": self._job_results_exist(job),
            "trace_outputs_exist": traces_exist,
            "results_current": results_current,
            "run_status": self._job_run_status(state, metadata, results_current),
            "successful": metadata.successful,
            "task_summary": self._job_task_summary(state, metadata),
        }

    def _base_job_status_row(self, job_file: Path) -> Dict[str, Any]:
        rel_file = self._relative_to_project(job_file)
        parts = rel_file.parts
        return {
            "name": job_file.stem,
            "simulation": parts[1] if len(parts) >= 4 and parts[0] == "jobs" else None,
            "job_file": str(job_file),
            "relative_job_file": str(rel_file),
            "loaded": False,
            "results_exist": False,
            "results_current": False,
        }

    @staticmethod
    def _job_results_exist(job) -> bool:
        result_path = Path(job._result_path)
        if not result_path.exists():
            return False

        try:
            if job.trace_outputs_exist():
                return True
        except Exception:
            pass

        metadata = job.run_metadata
        if any(
            (
                metadata.manifest,
                metadata.outputs,
                metadata.timings,
                metadata.error,
                metadata.state,
            )
        ):
            return True

        try:
            return any(path.is_file() for path in result_path.rglob("*"))
        except OSError:
            return False

    @staticmethod
    def _job_run_status(state: Any, metadata: Any, current: bool) -> Any:
        if isinstance(state, Mapping) and state.get("status") is not None:
            return state["status"]
        if isinstance(metadata.manifest, Mapping):
            exit_status = metadata.manifest.get("exit_status")
            if isinstance(exit_status, Mapping):
                return exit_status.get("status")
            if exit_status is not None:
                return exit_status
        return "current" if current else "not_run"

    @staticmethod
    def _job_task_summary(state: Any, metadata: Any) -> Dict[str, Any]:
        for source in (state, metadata.manifest):
            if isinstance(source, Mapping) and isinstance(
                source.get("task_summary"), Mapping
            ):
                return dict(source["task_summary"])
        return {}

    @staticmethod
    def _simulation_name(simulation: Union[str, BaseSimulation]) -> str:
        return (
            simulation.name
            if isinstance(simulation, BaseSimulation)
            else str(simulation)
        )

    def _relative_to_project(self, path: Path) -> Path:
        try:
            return path.relative_to(self.path)
        except ValueError:
            return path

    @staticmethod
    def _project_relative_or_absolute(path: Path, project: Path) -> str:
        try:
            return str(path.relative_to(project))
        except ValueError:
            return str(path)

    @staticmethod
    def _read_json_file(path: Union[str, Path]) -> Any:
        try:
            return json.loads(Path(path).read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to load JSON {path}: {exc}") from exc

    @staticmethod
    def _write_json_file(
        path: Union[str, Path],
        payload: Mapping[str, Any],
        **json_kwargs,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        indent = json_kwargs.pop("indent", 3)
        tmp_file = path.with_name(f".{path.name}.tmp")
        tmp_file.write_text(
            json.dumps(
                payload,
                cls=CustomJSONEncoder,
                indent=indent,
                **json_kwargs,
            )
        )
        tmp_file.replace(path)

    @staticmethod
    def _simulation_options(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        options = dict(kwargs)
        extra = dict(options.pop("extra", {}) or {})
        axisymmetric = options.pop("axisymmetric", False)
        default_units = options.pop("default_units", None)
        units = options.pop("units", None)
        unit_config = options.pop("unit_config", None)

        if units is not None:
            if isinstance(units, UnitConfig):
                if unit_config is not None:
                    raise ValueError("Pass only one of units and unit_config")
                unit_config = units
            else:
                if default_units is not None:
                    raise ValueError("Pass only one of units and default_units")
                default_units = units

        extra.update(options)
        return {
            "axisymmetric": axisymmetric,
            "default_units": default_units,
            "unit_config": unit_config,
            "extra": extra,
        }

    @staticmethod
    def _apply_simulation_units(
        sim: BaseSimulation,
        *,
        unit_config: Optional[UnitConfig],
        default_units: Optional[Mapping[str, Any]],
    ) -> None:
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
