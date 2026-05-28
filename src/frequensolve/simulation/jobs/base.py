"""Base job classes shared by FrequenSolve simulation workflows.

``BaseJob`` and the accompanying layout/record dataclasses provide the common
project paths, frequency metadata, output validation, serialization hooks, and
run-state mixins used by concrete forward and imaging jobs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np

from frequensolve.simulation.jobs.artifacts import JobArtifactMixin
from frequensolve.simulation.jobs.records import JobRecordMixin
from frequensolve.simulation.jobs.remote import JobRemoteMixin
from frequensolve.simulation.jobs.run_state import JobRunStateMixin
from frequensolve.simulation.jobs.serialization import JobSerializationMixin
from frequensolve.simulation.jobs.timings import JobTimingMixin
from frequensolve.simulation.outputs import (
    JobOutputs,
    Output,
    OutputUnits,
    TraceOutput,
)
from frequensolve.simulation.simulation import BaseSimulation

__all__ = [
    "JobLayout",
    "JobRecord",
    "BaseJob",
]


@dataclass(frozen=True)
class JobLayout:
    """Canonical project paths described by a saved job payload.

    Args:
        project: Project root that owns simulations, jobs, logs, and results.
        simulation_name: Name of the serialized simulation.
        job_name: Name of the job.
        simulation_relpath: Optional project-relative path to the simulation
            JSON. Defaults to the standard project layout when omitted.
        result_relpath: Optional project-relative result directory. Defaults to
            the standard project layout when omitted.
        job_file_name: Optional serialized job filename.
    """

    project: Path
    simulation_name: str
    job_name: str
    simulation_relpath: Optional[Path] = None
    result_relpath: Optional[Path] = None
    job_file_name: Optional[str] = None

    @classmethod
    def from_job(cls, job: "BaseJob", project: Union[str, Path]) -> "JobLayout":
        """Create a default project layout for an in-memory job.

        Args:
            job: Job whose simulation and job names define the layout.
            project: Project root for the layout.

        Returns:
            Layout using the standard project-relative directories.
        """

        return cls(
            project=Path(project),
            simulation_name=job.simulation.name,
            job_name=job.name,
        )

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        job_file: Optional[Union[str, Path]] = None,
    ) -> "JobLayout":
        """Build a layout from paths explicitly stored in a job JSON.

        Args:
            payload: Serialized job payload containing ``project_path``,
                ``simulation``, ``name``, and optional ``result_path``.
            job_file: Optional path to the job JSON being read.

        Returns:
            Layout preserving the payload's relative path choices.
        """

        project = Path(payload["project_path"])
        job_name = str(payload["name"])
        simulation_relpath = cls._project_relative(payload["simulation"], project)
        simulation_name = Path(payload["simulation"]).stem
        result_path = payload.get("result_path")
        result_relpath = (
            cls._project_relative(result_path, project)
            if result_path is not None
            else None
        )
        return cls(
            project=project,
            simulation_name=simulation_name,
            job_name=job_name,
            simulation_relpath=simulation_relpath,
            result_relpath=result_relpath,
            job_file_name=Path(job_file).name if job_file is not None else None,
        )

    def with_project(self, project: Union[str, Path]) -> "JobLayout":
        """Return the same relative layout rooted at another project path.

        Args:
            project: New project root.

        Returns:
            Copy of this layout with all relative paths resolved under
            ``project``.
        """

        return JobLayout(
            project=Path(project),
            simulation_name=self.simulation_name,
            job_name=self.job_name,
            simulation_relpath=self.simulation_relpath,
            result_relpath=self.result_relpath,
            job_file_name=self.job_file_name,
        )

    @property
    def simulation_dir(self) -> Path:
        """Return the directory containing the serialized simulation JSON.

        Returns:
            Absolute simulation directory path.
        """

        return self.simulation_file.parent

    @property
    def simulation_file(self) -> Path:
        """Return the absolute path to the simulation JSON used by the job.

        Returns:
            Simulation JSON path, using either the stored relative path or the
            standard project layout.
        """

        relpath = self.simulation_relpath
        if relpath is None:
            relpath = (
                Path("simulations")
                / self.simulation_name
                / (f"{self.simulation_name}.json")
            )
        return self.project / relpath

    @property
    def job_dir(self) -> Path:
        """Return the directory containing the job JSON, logs, and results.

        Returns:
            Absolute job directory path.
        """

        return self.result_dir.parent

    @property
    def job_file(self) -> Path:
        """Return the absolute path to the serialized job JSON.

        Returns:
            Job JSON path, using the stored filename or ``<job_name>.json``.
        """

        return self.job_dir / (self.job_file_name or f"{self.job_name}.json")

    @property
    def result_dir(self) -> Path:
        """Return the directory where solver outputs for this job are stored.

        Returns:
            Absolute result directory path.
        """

        relpath = self.result_relpath
        if relpath is None:
            relpath = Path("jobs") / self.simulation_name / self.job_name / "results"
        return self.project / relpath

    @property
    def logs_dir(self) -> Path:
        """Return the directory where scheduler and solver logs are stored.

        Returns:
            Absolute logs directory path.
        """

        return self.job_dir / "logs"

    @staticmethod
    def _project_relative(value: Union[str, Path], project: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            try:
                return path.relative_to(project)
            except ValueError:
                return path
        return path


@dataclass(frozen=True)
class JobRecord:
    """Recorded local or remote execution location for a job.

    Args:
        site: Site name that staged or ran the job.
        work_dir: Site work directory used as the remote project root.
        project_path: Project root visible to the site.
        job_dir: Directory containing the staged job JSON.
        job_file: Full path to the staged job JSON.
        result_dir: Directory where the site writes solver results.
        logs_dir: Directory where the site writes scheduler or solver logs.
        scheduler_id: Optional scheduler allocation/job id.
        status: Submission or execution status.
        submitted_at: Optional ISO timestamp for submission time.
        updated_at: Optional ISO timestamp for the last record update.
        fingerprint: Optional job fingerprint at submission time.
        fingerprint_payload: Payload used to compute ``fingerprint``.
        site_module: Optional import module used to recreate the site object.
        site_class: Optional class name used to recreate the site object.
        rel_path: Optional site configuration path.
        metadata: Additional site-specific fields.
    """

    site: str
    work_dir: Path
    project_path: Path
    job_dir: Path
    job_file: Path
    result_dir: Path
    logs_dir: Path
    scheduler_id: Optional[str] = None
    status: str = "submitted"
    submitted_at: Optional[str] = None
    updated_at: Optional[str] = None
    fingerprint: Optional[str] = None
    fingerprint_payload: Dict[str, Any] = field(default_factory=dict)
    site_module: Optional[str] = None
    site_class: Optional[str] = None
    rel_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the run record to the job-record schema.

        Returns:
            JSON-compatible mapping describing the recorded run location.
        """

        payload = {
            "site": self.site,
            "work_dir": str(self.work_dir),
            "project_path": str(self.project_path),
            "job_dir": str(self.job_dir),
            "job_file": str(self.job_file),
            "result_dir": str(self.result_dir),
            "logs_dir": str(self.logs_dir),
            "status": self.status,
            "fingerprint_payload": self.fingerprint_payload,
            "metadata": self.metadata,
        }
        for key in (
            "scheduler_id",
            "submitted_at",
            "updated_at",
            "fingerprint",
            "site_module",
            "site_class",
            "rel_path",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "JobRecord":
        """Deserialize a run record from a serialized payload.

        Args:
            data: Mapping produced by :meth:`to_fs`.

        Returns:
            ``JobRecord`` with path fields restored as ``Path`` objects.
        """

        return cls(
            site=str(data["site"]),
            work_dir=Path(data["work_dir"]),
            project_path=Path(data["project_path"]),
            job_dir=Path(data["job_dir"]),
            job_file=Path(data["job_file"]),
            result_dir=Path(data["result_dir"]),
            logs_dir=Path(data["logs_dir"]),
            scheduler_id=data.get("scheduler_id"),
            status=str(data.get("status", "submitted")),
            submitted_at=data.get("submitted_at"),
            updated_at=data.get("updated_at"),
            fingerprint=data.get("fingerprint"),
            fingerprint_payload=dict(data.get("fingerprint_payload") or {}),
            site_module=data.get("site_module"),
            site_class=data.get("site_class"),
            rel_path=data.get("rel_path"),
            metadata=dict(data.get("metadata") or {}),
        )

    def with_updates(self, **updates: Any) -> "JobRecord":
        """Return a copy of this record with selected fields replaced.

        Args:
            **updates: Field values to replace in the serialized record.

        Returns:
            Updated ``JobRecord`` instance.
        """

        data = self.to_fs()
        data.update(updates)
        return self.from_fs(data)


@dataclass
class BaseJob(
    JobSerializationMixin,
    JobRecordMixin,
    JobArtifactMixin,
    JobRunStateMixin,
    JobTimingMixin,
    JobRemoteMixin,
):
    """Common base class for saved solver jobs.

    ``BaseJob`` owns the job name, simulation reference, frequency list,
    output configuration, serialization, run-state helpers, and artifact
    accessors shared by forward, time-domain, imaging, and FWI workflows.

    Args:
        name: Job name used in project paths and serialized payloads.
        simulation: Simulation object this job runs.
        workflow: Solver workflow identifier.
        f_list: One frequency per solver task. Complex values encode Laplace
            damping in their imaginary component.
        outputs: Output configuration or output requests for this job.

    Raises:
        ValueError: If the name, simulation, or frequency list is invalid.
    """

    name: str
    simulation: BaseSimulation
    workflow: str
    f_list: List[Union[float, complex]]
    outputs: JobOutputs = field(default_factory=JobOutputs)
    _file: Optional[Path] = None
    _job_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BaseJob requires a non-empty name")
        if self.simulation is None:
            raise ValueError("BaseJob requires a simulation")
        frequencies = np.asarray(self.f_list)
        if frequencies.size == 0:
            raise ValueError("BaseJob requires at least one frequency")
        self.f_list = frequencies.tolist()
        if not isinstance(self.outputs, JobOutputs):
            self.outputs = JobOutputs(self.outputs)

    def __iadd__(self, output: Union[Output, Iterable[Output]]) -> "BaseJob":
        self.outputs += output
        return self

    def add_output(self, output: Union[Output, Iterable[Output]]) -> "BaseJob":
        """Add one or more output requests.

        Args:
            output: Single output request or iterable of output requests.

        Returns:
            This job, allowing fluent configuration.
        """

        self += output
        return self

    def output_units(self, **units) -> "BaseJob":
        """Set default units for solver outputs.

        Args:
            **units: Unit overrides accepted by ``OutputUnits``.

        Returns:
            This job, allowing fluent configuration.
        """

        self.outputs.units = OutputUnits(**units)
        return self

    def output_traces(self, path: Union[str, Path] = "traces", **kwargs) -> "BaseJob":
        """Configure receiver trace output for this job.

        Args:
            path: Output directory name or path relative to the job result
                directory.
            **kwargs: Additional ``TraceOutput`` options such as groups,
                components, or frequency selection.

        Returns:
            This job, allowing fluent configuration.
        """

        self.outputs.traces = TraceOutput(path=path, **kwargs)
        return self

    def paraview(self, *args, **kwargs) -> "BaseJob":
        """Add a ParaView field output request.

        Args:
            *args: Positional arguments forwarded to ``outputs.paraview``.
            **kwargs: Keyword arguments forwarded to ``outputs.paraview``.

        Returns:
            This job, allowing fluent configuration.
        """

        from frequensolve.simulation.outputs import paraview

        self += paraview(*args, **kwargs)
        return self

    def wavefield(self, *args, **kwargs) -> "BaseJob":
        """Add a grid-backed wavefield output request.

        Args:
            *args: Positional arguments forwarded to ``outputs.wavefield``.
            **kwargs: Keyword arguments forwarded to ``outputs.wavefield``.

        Returns:
            This job, allowing fluent configuration.
        """

        from frequensolve.simulation.outputs import wavefield

        self += wavefield(*args, **kwargs)
        return self

    def validate(self, *, raise_errors: bool = False):
        """Validate this job for common pre-run authoring mistakes.

        Args:
            raise_errors: If ``True``, raise ``ValidationError`` when blocking
                issues are found.

        Returns:
            ``ValidationReport`` with errors and warnings.
        """

        from frequensolve.validation import validate_job

        return validate_job(self, raise_errors=raise_errors)

    def validate_outputs(self) -> None:
        """Validate output requests before saving or executing the job.

        Raises:
            ValueError: If an output configuration is incompatible with the job
                frequency layout.
        """

        if self.outputs.paraview and len(self.f_list) != 1:
            raise ValueError(
                "ParaView outputs currently require a single-frequency job. "
                "Create one FrequencyDomainJob per plotted frequency."
            )

    @property
    def job_file(self) -> Path:
        """Return the saved or default job JSON path.

        Returns:
            Absolute path to the job JSON. Unsaved jobs return the path they
            will use under the local project layout.
        """

        if self._file is not None:
            return Path(self._file).resolve()
        return (self._local_path / f"{self.name}.json").resolve()

    @property
    def project_path(self):
        """Return the absolute project path that owns this job.

        Raises:
            ValueError: If the simulation is not attached to a project.
        """

        return self._project_path()

    @property
    def n_tasks(self):
        """Return the number of one-frequency solver tasks in this job.

        Returns:
            Length of ``f_list``.
        """

        return len(self.f_list)

    def _project_path(self) -> Path:
        project_path = getattr(self.simulation, "_proj_path", None)
        if project_path is None:
            project_path = getattr(self.simulation, "project_path", None)
        if project_path is None:
            raise ValueError("Job simulation is not attached to a project path")
        return Path(project_path).resolve()

    @property
    def _local_path(self):
        project_path = Path(self.project_path)
        return project_path / "jobs" / self.simulation.name / self.name

    @property
    def _stdout_path(self):
        """Path where solver stdout is stored."""
        if self._file is None:
            return self._local_path / "logs"
        return self._file.parent / "logs"

    @property
    def _result_path(self):
        """Path where solver results will be stored."""
        if self._file is None:
            return self._local_path / "results"
        return self._file.parent / "results"

    def _remote_path(self, work_dir: Union[Path, str]):
        """Get remote job path."""
        work_dir = Path(work_dir)
        return work_dir / "jobs" / self.simulation.name / self.name
