"""Simulation job containers, saved-job layouts, and run bookkeeping."""

import copy
import hashlib
import importlib
import json
import os
import shutil
import warnings
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import blake3
import numpy as np

from frequensolve.simulation.artifacts import (
    RunMetadata,
    TraceManifest,
    TraceOutputHandle,
    TraceOutputSpec,
    WavefieldOutputHandle,
)
from frequensolve.simulation.outputs import (
    JobOutputs,
    Output,
    OutputUnits,
    TraceOutput,
)
from frequensolve.simulation.simulation import BaseSimulation, CustomJSONEncoder
from frequensolve.util.class_registry import class_registry, register_class

__all__ = [
    "JobLayout",
    "JobRunRecord",
    "SimulationJob",
    "FrequencyDomainJob",
    "TimeDomainJob",
]

SOLVER_RESIDUAL_FAILURE_THRESHOLD = 1.0e-3


@dataclass(frozen=True)
class JobLayout:
    """Canonical paths described by a saved job payload."""

    project: Path
    simulation_name: str
    job_name: str
    simulation_relpath: Optional[Path] = None
    result_relpath: Optional[Path] = None
    job_file_name: Optional[str] = None

    @classmethod
    def from_job(cls, job: "SimulationJob", project: Union[str, Path]) -> "JobLayout":
        """Construct default saved paths for a job under ``project``."""

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
        """Build a layout from the paths explicitly stored in a job JSON."""

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

    @staticmethod
    def _project_relative(value: Union[str, Path], project: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            try:
                return path.relative_to(project)
            except ValueError:
                return path
        return path

    def with_project(self, project: Union[str, Path]) -> "JobLayout":
        """Return this layout remapped to a different project root."""

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
        """Directory containing the saved simulation JSON."""

        return self.simulation_file.parent

    @property
    def simulation_file(self) -> Path:
        """Absolute path to the saved simulation JSON."""

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
        """Directory containing the saved job JSON and logs."""

        return self.result_dir.parent

    @property
    def job_file(self) -> Path:
        """Absolute path to the saved job JSON."""

        return self.job_dir / (self.job_file_name or f"{self.job_name}.json")

    @property
    def result_dir(self) -> Path:
        """Directory where local or fetched solver results are stored."""

        relpath = self.result_relpath
        if relpath is None:
            relpath = Path("jobs") / self.simulation_name / self.job_name / "results"
        return self.project / relpath

    @property
    def logs_dir(self) -> Path:
        """Directory where job logs are stored."""

        return self.job_dir / "logs"


@dataclass(frozen=True)
class JobRunRecord:
    """Where a job was staged or run on a remote site."""

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
        """Serialize this run record to a JSON-compatible mapping."""

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
    def from_fs(cls, data: Mapping[str, Any]) -> "JobRunRecord":
        """Deserialize a run record from saved job metadata."""

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

    def with_updates(self, **updates: Any) -> "JobRunRecord":
        """Return a copy with selected fields replaced."""

        data = self.to_fs()
        data.update(updates)
        return self.from_fs(data)


@register_class
@dataclass
class SimulationJob(ABC):
    """Base class for saved, runnable FrequenSolve simulation jobs.

    Jobs bind a saved simulation to a workflow, frequency list, and requested
    outputs. They can be serialized to the solver contract, fingerprinted for
    rerun checks, submitted to a site, and reopened later from JSON.
    """

    name: str
    simulation: BaseSimulation
    workflow: str
    f_list: List[Union[float, complex]]
    outputs: JobOutputs = field(default_factory=JobOutputs)
    _file: Optional[Path] = None
    _job_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the job and normalize frequency/output containers."""

        if not self.name:
            raise ValueError("SimulationJob requires a non-empty name")
        if self.simulation is None:
            raise ValueError("SimulationJob requires a simulation")
        frequencies = np.asarray(self.f_list)
        if frequencies.size == 0:
            raise ValueError("SimulationJob requires at least one frequency")
        self.f_list = frequencies.tolist()
        if not isinstance(self.outputs, JobOutputs):
            self.outputs = JobOutputs(self.outputs)

    def __iadd__(self, output: Union[Output, Iterable[Output]]) -> "SimulationJob":
        self.outputs += output
        return self

    def add_output(self, output: Union[Output, Iterable[Output]]) -> "SimulationJob":
        """Add one or more output requests and return ``self``."""

        self += output
        return self

    def output_units(self, **units) -> "SimulationJob":
        """Set output-unit defaults and return ``self``."""

        self.outputs.units = OutputUnits(**units)
        return self

    def output_traces(
        self, path: Union[str, Path] = "traces", **kwargs
    ) -> "SimulationJob":
        """Set the receiver trace output path and return ``self``."""

        self.outputs.traces = TraceOutput(path=path, **kwargs)
        return self

    def paraview(self, *args, **kwargs) -> "SimulationJob":
        """Add a ParaView output request and return ``self``."""

        from frequensolve.simulation.outputs import paraview

        self += paraview(*args, **kwargs)
        return self

    def wavefield(self, *args, **kwargs) -> "SimulationJob":
        """Add a wavefield trace output request and return ``self``."""

        from frequensolve.simulation.outputs import wavefield

        self += wavefield(*args, **kwargs)
        return self

    def validate_outputs(self) -> None:
        """Validate output requests against the current job frequency list."""

        if self.outputs.paraview and len(self.f_list) != 1:
            raise ValueError(
                "ParaView outputs currently require a single-frequency job. "
                "Create one FrequencyDomainJob per plotted frequency."
            )

    @classmethod
    def from_fs(
        cls,
        d: dict,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "SimulationJob":
        """Deserialize a concrete job subclass from a solver payload."""

        data = dict(d)
        class_name = data.get("_type")
        if class_name not in class_registry:
            raise ValueError(f"Unknown job class: {class_name}")
        job_class = class_registry[class_name]
        return job_class.from_fs(
            data,
            base_path=base_path,
            project_path=project_path,
        )

    @staticmethod
    def _job_file_from_path(path: Union[Path, str]) -> Path:
        path = Path(path).expanduser().resolve()
        if path.is_file():
            return path
        if path.is_dir():
            named = path / f"{path.name}.json"
            if named.exists():
                return named
            json_files = sorted(path.glob("*.json"))
            if len(json_files) == 1:
                return json_files[0].resolve()
            if not json_files:
                raise FileNotFoundError(f"No job JSON file found in {path}")
            names = ", ".join(file.name for file in json_files)
            raise ValueError(
                f"Multiple job JSON files found in {path}; specify one explicitly: {names}"
            )
        return path

    @classmethod
    def load(
        cls,
        path: Union[Path, str, "SimulationJob"],
        *,
        project_path: Optional[Union[str, Path]] = None,
    ):
        """Load a saved job JSON file or directory."""

        if isinstance(path, SimulationJob):
            path = path.job_file
        path = cls._job_file_from_path(path)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load job JSON {path}: {e}") from e

        job = cls.from_fs(data, base_path=path.parent, project_path=project_path)
        job._file = path
        job._job_id = data.get("job_id")
        return job

    @property
    def job_file(self) -> Path:
        """Return the saved job JSON path, whether or not this object was saved."""

        if self._file is not None:
            return Path(self._file).resolve()
        return (self._local_path / f"{self.name}.json").resolve()

    def load_saved(self) -> "SimulationJob":
        """Load the saved job JSON corresponding to this job object."""

        return self.__class__.load(self.job_file)

    @staticmethod
    def _resolve_simulation_file(
        path: Union[str, Path],
        *,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
        source_project: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Resolve a job's simulation path from explicit project/base context."""

        path = Path(path)
        if path.is_absolute():
            if source_project is not None and project_path is not None:
                try:
                    relative = path.relative_to(Path(source_project).expanduser())
                except ValueError:
                    pass
                else:
                    candidate = Path(project_path).expanduser().resolve() / relative
                    if candidate.exists():
                        return candidate
            return path

        candidates = []
        if project_path is not None:
            candidates.append(Path(project_path).expanduser().resolve() / path)
        if base_path is not None:
            candidates.append(Path(base_path).expanduser().resolve() / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return path

    @staticmethod
    def _load_simulation_for_job(
        path: Union[str, Path],
        *,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
        source_project: Optional[Union[str, Path]] = None,
    ) -> BaseSimulation:
        sim_file = SimulationJob._resolve_simulation_file(
            path,
            base_path=base_path,
            project_path=project_path,
            source_project=source_project,
        )
        sim = BaseSimulation.load(sim_file)
        if project_path is not None:
            project = Path(project_path).expanduser().resolve()
            sim.project_path = project
            sim._file = Path(sim_file).expanduser().resolve()
            sim._set_path(project, Path("simulations"))
        return sim

    @staticmethod
    def _decode_frequencies(value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.size == 0:
            raise ValueError("Job f_list must contain at least one frequency")
        if array.ndim == 1:
            if np.iscomplexobj(array):
                return np.asarray([f.real - 1j * abs(f.imag) for f in array])
            return array.astype(float)
        if array.ndim == 2 and array.shape[1] == 2:
            return np.asarray([f[0] - 1j * abs(f[1]) for f in array])
        raise ValueError("Job f_list must be a 1D real list or Nx2 complex list")

    def _encoded_frequencies(self) -> np.ndarray:
        f_list = np.asarray(self.f_list)
        if np.iscomplexobj(f_list):
            return np.asarray([[f.real, -abs(f.imag)] for f in f_list])
        return np.asarray(f_list)

    def _simulation_path(self, *, project_relative: bool = False) -> str:
        if self.simulation._file is None:
            raise ValueError("Simulation has not been saved.")
        path = Path(self.simulation._file).resolve()
        if project_relative:
            try:
                return str(path.relative_to(self._project_path()))
            except ValueError:
                return str(path)
        return str(path)

    def _project_path(self) -> Path:
        project_path = getattr(self.simulation, "_proj_path", None)
        if project_path is None:
            project_path = getattr(self.simulation, "project_path", None)
        if project_path is None:
            raise ValueError("Job simulation is not attached to a project path")
        return Path(project_path).resolve()

    def to_fs(self, *, project_relative: bool = False) -> Dict[str, Any]:
        """Serialize this job to the solver job contract."""

        self.validate_outputs()
        f_list = self._encoded_frequencies()
        payload = {
            "schema": "fs-job-1",
            "_type": self.__class__.__name__,
            "name": self.name,
            "project_path": str(self._project_path()),
            "simulation": self._simulation_path(project_relative=project_relative),
            "workflow": self.workflow,
            "f_list": f_list,
            "Outputs": self.outputs.to_fs(),
        }
        if self._job_id is not None:
            payload["job_id"] = self._job_id
        return payload

    @staticmethod
    def _hash_payload(payload: Any) -> str:
        encoded = json.dumps(
            payload,
            cls=CustomJSONEncoder,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"blake3:{blake3.blake3(encoded).hexdigest()}"

    @staticmethod
    def _hash_json_file(path: Union[str, Path]) -> str:
        with open(path, "r") as f:
            payload = json.load(f)
        return SimulationJob._hash_payload(payload)

    @staticmethod
    def _sha256_file(path: Union[str, Path]) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    def fingerprint_payload(self) -> Dict[str, Any]:
        """Return the deterministic payload used for whole-job rerun checks."""

        job_data = self.to_fs()
        if self.simulation._file is None:
            raise ValueError("Simulation must be saved before fingerprinting a job")
        simulation_hash = self._hash_json_file(self.simulation._file)
        return {
            "schema": "frequensolve-job-fingerprint-1",
            "job": {
                "_type": job_data["_type"],
                "workflow": job_data["workflow"],
                "f_list": job_data["f_list"],
                "Outputs": job_data["Outputs"],
            },
            "simulation": {"hash": simulation_hash},
        }

    def fingerprint(self) -> str:
        """Return a stable hash of the whole-job fingerprint payload."""

        return self._hash_payload(self.fingerprint_payload())

    @staticmethod
    def _canonical_frequency_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, complex):
            return {
                "real": float(f"{float(value.real):.12g}"),
                "imag": float(f"{float(value.imag):.12g}"),
            }
        return float(f"{float(value):.12g}")

    @staticmethod
    def _real_frequency_value(value: Any) -> float:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, complex):
            return float(value.real)
        return float(value)

    def task_fingerprint_payload(self, task: int) -> Dict[str, Any]:
        """Return the rerun fingerprint payload for one frequency task.

        ``task`` is one-based to match the solver task IDs and trace filenames.
        Unlike the whole-job fingerprint, this intentionally excludes the full
        frequency list.  Changing ``f_max`` or ``df`` should only invalidate
        frequencies whose own value changed.
        """

        if task < 1 or task > self.n_tasks:
            raise IndexError(f"Task {task} is outside 1..{self.n_tasks}")
        job_data = self.to_fs()
        if self.simulation._file is None:
            raise ValueError("Simulation must be saved before fingerprinting a task")
        simulation_hash = self._hash_json_file(self.simulation._file)
        return {
            "schema": "frequensolve-job-task-fingerprint-1",
            "job": {
                "_type": job_data["_type"],
                "workflow": job_data["workflow"],
                "Outputs": job_data["Outputs"],
            },
            "simulation": {"hash": simulation_hash},
            "frequency": self._canonical_frequency_value(self.f_list[task - 1]),
        }

    def task_fingerprint(self, task: int) -> str:
        """Return a stable hash for one one-based frequency task."""

        return self._hash_payload(self.task_fingerprint_payload(task))

    @property
    def run_state_file(self) -> Path:
        """Path to Python-side run-state metadata for this job."""

        return self._result_path / "_fs_python_run.json"

    @property
    def run_records_file(self) -> Path:
        """Path to saved remote run records for this job."""

        return self._result_path / "_fs_run" / "runs.json"

    @property
    def run_metadata(self) -> RunMetadata:
        """Metadata read from the current result directory."""

        return RunMetadata.read(self._result_path)

    def run_records(self) -> List[JobRunRecord]:
        """Return saved locations where this job has been staged or run."""

        path = self.run_records_file
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        raw_records = payload.get("runs") if isinstance(payload, Mapping) else None
        if raw_records is None and isinstance(payload, list):
            raw_records = payload
        if not isinstance(raw_records, list):
            return []
        records = []
        for record in raw_records:
            if not isinstance(record, Mapping):
                continue
            try:
                records.append(JobRunRecord.from_fs(record))
            except (KeyError, TypeError, ValueError):
                continue
        return records

    def latest_run(self, site: Optional[str] = None) -> Optional[JobRunRecord]:
        """Return the most recently recorded run location for this job."""

        records = self.run_records()
        if site is not None:
            records = [record for record in records if record.site == site]
        if not records:
            return None
        return sorted(
            records,
            key=lambda record: record.updated_at or record.submitted_at or "",
        )[-1]

    def write_run_record(self, record: JobRunRecord) -> JobRunRecord:
        """Persist or update a remote run location record."""

        def key(item: JobRunRecord) -> tuple:
            return (item.site, str(item.job_dir), item.scheduler_id or "")

        records = [item for item in self.run_records() if key(item) != key(record)]
        records.append(record)
        payload = {
            "schema": "fs-job-run-records-1",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runs": [item.to_fs() for item in records],
        }
        self._write_json_file(self.run_records_file, payload)
        return record

    def record_site_run(
        self,
        *,
        site: str,
        work_dir: Union[str, Path],
        scheduler_id: Optional[str] = None,
        status: str = "submitted",
        site_module: Optional[str] = None,
        site_class: Optional[str] = None,
        rel_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> JobRunRecord:
        """Record the remote layout used by a site submission."""

        if self._file is None or self.simulation._file is None:
            self.save()
        now = datetime.now(timezone.utc).isoformat()
        layout = JobLayout.from_job(self, work_dir)
        record = JobRunRecord(
            site=site,
            work_dir=Path(work_dir),
            project_path=layout.project,
            job_dir=layout.job_dir,
            job_file=layout.job_file,
            result_dir=layout.result_dir,
            logs_dir=layout.logs_dir,
            scheduler_id=str(scheduler_id) if scheduler_id is not None else None,
            status=status,
            submitted_at=now,
            updated_at=now,
            fingerprint=self.fingerprint(),
            fingerprint_payload=self.fingerprint_payload(),
            site_module=site_module,
            site_class=site_class,
            rel_path=str(rel_path) if rel_path is not None else None,
            metadata=dict(metadata or {}),
        )
        return self.write_run_record(record)

    def _site_from_run_record(self, record: JobRunRecord):
        if not record.site_module or not record.site_class or not record.rel_path:
            raise ValueError(
                f"Run record for {record.site} cannot recreate a site; "
                "fetch through an initialized site instead."
            )
        module = importlib.import_module(record.site_module)
        site_class = getattr(module, record.site_class)
        return site_class(record.rel_path)

    def _resolve_fetch_site(self, site=None):
        if site is not None:
            return site
        record = self.latest_run()
        if record is None:
            raise ValueError(
                "This job has no recorded remote run. Submit it once or pass a site."
            )
        return self._site_from_run_record(record)

    def fetch_traces(self, site=None, upscale: int = 1):
        """Fetch receiver traces from the last recorded remote run location."""

        return self._resolve_fetch_site(site).fetch_traces(self, upscale=upscale)

    def fetch_wavefields(self, site=None, upscale: int = 1):
        """Fetch wavefield outputs from the last recorded remote run location."""

        return self._resolve_fetch_site(site).fetch_wavefields(self, upscale=upscale)

    def fetch_outputs(self, site=None):
        """Fetch common outputs from the last recorded remote run location."""

        return self._resolve_fetch_site(site).fetch_outputs(self)

    def fetch_run_metadata(self, site=None):
        """Fetch run metadata from the last recorded remote run location."""

        return self._resolve_fetch_site(site).fetch_run_metadata(self)

    def fetch_logs(self, site=None, **kwargs):
        """Fetch logs from the last recorded remote run location."""

        return self._resolve_fetch_site(site).fetch_logs(self, **kwargs)

    def expected_trace_files(self) -> List[Path]:
        """Trace files expected for this job's receiver trace output."""

        return list(self.trace_manifest.files)

    @staticmethod
    def _legacy_trace_file(path: Path) -> Path:
        return path.with_name(path.name.replace("traces_", "receivers_", 1))

    @classmethod
    def _trace_file_exists(cls, path: Path) -> bool:
        return path.exists() or cls._legacy_trace_file(path).exists()

    def trace_outputs_exist(self) -> bool:
        """Whether all expected receiver trace outputs are present."""

        manifest = self.trace_manifest
        if manifest.packed_file is not None and not manifest.packed_complete:
            warnings.warn(
                manifest.packed_incomplete_message(),
                RuntimeWarning,
                stacklevel=2,
            )
        if manifest.complete:
            return True
        return all(
            self._trace_output_path_for_task(task, manifest=manifest)[1]
            for task in range(1, self.n_tasks + 1)
        )

    def _packed_trace_has_current_task(
        self,
        task: int,
        *,
        manifest: Optional[TraceManifest] = None,
    ) -> bool:
        manifest = self.trace_manifest if manifest is None else manifest
        if manifest.packed_file is None:
            return False
        if task < 1 or task > self.n_tasks:
            return False
        packed_frequencies = manifest.packed_frequencies
        if not packed_frequencies:
            return True
        expected_frequency = self._real_frequency_value(self.f_list[task - 1])
        return np.isclose(
            list(packed_frequencies.values()),
            expected_frequency,
            rtol=0.0,
            atol=1.0e-9,
        ).any()

    def _candidate_frequency_trace_files(
        self,
        task: int,
        *,
        manifest: Optional[TraceManifest] = None,
    ) -> List[Path]:
        manifest = self.trace_manifest if manifest is None else manifest
        files = self.expected_trace_files()
        candidates: List[Path] = []
        if 1 <= task <= len(files):
            path = Path(files[task - 1])
            candidates.extend([path, self._legacy_trace_file(path)])
        shard_dir = manifest.output_path / "shards"
        if shard_dir.exists():
            candidates.extend(sorted(shard_dir.glob("*.h5")))
        candidates.extend(sorted(manifest.output_path.glob("f_*_hz.h5")))

        out: List[Path] = []
        seen = set()
        for path in candidates:
            key = str(Path(path).resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            out.append(Path(path))
        return out

    def _matching_frequency_trace_file(
        self,
        task: int,
        *,
        manifest: Optional[TraceManifest] = None,
    ) -> Optional[Path]:
        if task < 1 or task > self.n_tasks:
            return None
        expected_frequency = self._real_frequency_value(self.f_list[task - 1])
        from frequensolve.seismic.trace_store import TraceStore

        for path in self._candidate_frequency_trace_files(task, manifest=manifest):
            if not path.exists():
                continue
            try:
                if TraceStore._is_packed_trace_file(path):
                    continue
                frequencies = TraceStore._read_trace_frequencies(path)
            except (OSError, KeyError, ValueError):
                continue
            if any(
                np.isclose(
                    float(frequency),
                    expected_frequency,
                    rtol=0.0,
                    atol=1.0e-9,
                )
                for frequency in frequencies
            ):
                return path
        return None

    def _trace_output_path_for_task(
        self,
        task: int,
        *,
        manifest: Optional[TraceManifest] = None,
    ) -> tuple[Path, bool]:
        manifest = self.trace_manifest if manifest is None else manifest
        files = self.expected_trace_files()
        path = Path(files[task - 1])
        existing = path if path.exists() else self._legacy_trace_file(path)
        if existing.exists():
            return existing, True
        shard = self._matching_frequency_trace_file(task, manifest=manifest)
        if shard is not None:
            return shard, True
        packed_file = manifest.packed_file
        if manifest.packed_complete and self._packed_trace_has_current_task(
            task,
            manifest=manifest,
        ):
            if packed_file is not None:
                return packed_file, True
        return path, False

    def invalidate_trace_cache(self) -> None:
        """Remove derived trace VDS files so the next read reflects current HDF5."""

        candidates = [
            self._result_path / "_fs_run" / "cache",
            self.trace_outputs.path,
        ]
        for directory in candidates:
            if not directory.exists():
                continue
            for path in directory.glob("*_vds.h5"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def remove_packed_trace_products(self) -> bool:
        """Remove packed trace products so solver packing rebuilds from shards."""

        removed = False
        manifests = [self.trace_manifest]
        try:
            wavefield_manifest = self.wavefield_manifest
        except Exception:
            wavefield_manifest = None
        if wavefield_manifest is not None and wavefield_manifest.groups:
            manifests.append(wavefield_manifest)

        for manifest in manifests:
            candidates = [
                manifest.packed_file,
                manifest.output_path / "traces.h5",
                manifest.output_path / "manifest.json",
            ]
            for path in candidates:
                if path is None:
                    continue
                try:
                    Path(path).unlink()
                    removed = True
                except FileNotFoundError:
                    pass

        if removed:
            self.invalidate_trace_cache()
        return removed

    def _stored_trace_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_path))
        except Exception:
            return str(path)

    def _resolve_stored_trace_path(self, value: Any) -> Optional[Path]:
        if value is None:
            return None
        path = Path(str(value))
        if not path.is_absolute():
            path = self.project_path / path
        path = self._legacy_trace_file(path) if not path.exists() else path
        return path

    def _task_record_by_task(
        self, records: Iterable[Mapping[str, Any]]
    ) -> Dict[int, Mapping[str, Any]]:
        out = {}
        for record in records:
            task = self._task_number_from_record(record)
            if task is not None:
                out[task] = record
        return out

    def _state_task_records(
        self, state: Optional[Mapping[str, Any]] = None
    ) -> List[Mapping[str, Any]]:
        state = self.run_state() if state is None else state
        records = []
        if isinstance(state, Mapping):
            records.extend(self._as_records(state.get("tasks")))
            records.extend(self._as_records(state.get("task_results")))
        return records

    @staticmethod
    def _as_records(value: Any) -> List[Mapping[str, Any]]:
        if value is None:
            return []
        if isinstance(value, Mapping):
            records = []
            for key, item in value.items():
                if isinstance(item, Mapping):
                    record = dict(item)
                    record.setdefault("task", key)
                    records.append(record)
                else:
                    records.append({"task": key, "value": item})
            return records
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
        return []

    @staticmethod
    def _task_number_from_value(value: Any, *, zero_based: bool) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            digits = "".join(char for char in value if char.isdigit())
            if not digits:
                return None
            value = digits
        try:
            task = int(value)
        except (TypeError, ValueError):
            return None
        if task < 0:
            return None
        return task + 1 if zero_based else task

    @classmethod
    def _task_number_from_record(cls, record: Mapping[str, Any]) -> Optional[int]:
        zero_based_keys = ("task_id", "task_index", "index")
        one_based_keys = ("frequency_index", "ifreq", "frequency_task", "task")
        for key in zero_based_keys:
            if key in record:
                return cls._task_number_from_value(record.get(key), zero_based=True)
        for key in one_based_keys:
            if key in record:
                return cls._task_number_from_value(record.get(key), zero_based=False)
        return None

    @staticmethod
    def _normalized_task_status(value: Any) -> Optional[str]:
        if value is None:
            return None
        status = str(value).strip().lower().replace(" ", "_")
        if status in {
            "success",
            "succeeded",
            "complete",
            "completed",
            "done",
            "current",
            "reused",
            "skipped",
        }:
            return "succeeded"
        if status in {"failed", "failure", "error", "timeout", "cancelled", "killed"}:
            return "failed"
        if status in {"pending", "queued", "submitted", "running", "not_run"}:
            return "not_run"
        return None

    @staticmethod
    def _rounded_solver_float(value: Any, *, digits: int = 4) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return float(f"{numeric:.{digits}g}")

    @classmethod
    def solver_convergence_summary(
        cls,
        data: Mapping[str, Any],
        *,
        residual_failure_threshold: float = SOLVER_RESIDUAL_FAILURE_THRESHOLD,
    ) -> Optional[Dict[str, Any]]:
        """Extract concise convergence information from solver run metadata."""

        if not isinstance(data, Mapping):
            return None
        solver = data.get("solver", data)
        if not isinstance(solver, Mapping):
            return None
        convergence = solver.get("convergence")
        if not isinstance(convergence, Mapping):
            return None

        solves = []
        for solve in cls._as_records(convergence.get("solves")):
            residual = cls._rounded_solver_float(solve.get("residual"))
            row = {}
            for key in ("context", "solver", "status", "code", "grid", "tolerance"):
                if key in solve:
                    row[key] = solve[key]
            if "converged" in solve:
                row["converged"] = bool(solve.get("converged"))
            if "iterations" in solve:
                try:
                    row["iterations"] = int(solve.get("iterations"))
                except (TypeError, ValueError):
                    pass
            if residual is not None:
                row["residual"] = residual
            solves.append(row)

        residuals = [
            solve["residual"] for solve in solves if solve.get("residual") is not None
        ]
        iterations = [
            solve["iterations"]
            for solve in solves
            if solve.get("iterations") is not None
        ]
        converged = convergence.get("converged")
        if converged is None and solves:
            converged = all(solve.get("converged", True) for solve in solves)
        if converged is not None:
            converged = bool(converged)
        residual = max(residuals) if residuals else None
        residual_failed = residual is not None and residual > float(
            residual_failure_threshold
        )
        convergence_failed = converged is False
        failed = bool(convergence_failed or residual_failed)
        raw_status = str(convergence.get("status", "unknown"))
        if not failed and solves and raw_status == "not_run":
            raw_status = "converged"

        summary: Dict[str, Any] = {
            "status": "failed" if failed else raw_status,
            "converged": converged,
            "failed": failed,
            "residual_failure_threshold": residual_failure_threshold,
        }
        for key in ("solve_count", "failure_count", "worst_code"):
            if key in convergence:
                summary[key] = convergence[key]
        try:
            solve_count = int(summary.get("solve_count", 0))
        except (TypeError, ValueError):
            solve_count = 0
        if solves and solve_count <= 0:
            summary["solve_count"] = len(solves)
        if iterations:
            summary["iterations"] = max(iterations)
            summary["total_iterations"] = int(sum(iterations))
        if residual is not None:
            summary["residual"] = residual
        if solves:
            summary["solves"] = solves
        return summary

    @classmethod
    def _record_solver_convergence(
        cls, record: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if str(record.get("status", "")).strip().lower().replace(" ", "_") in {
            "skipped",
            "current",
            "reused",
        }:
            return None
        solver = record.get("solver")
        if isinstance(solver, Mapping):
            summary = cls.solver_convergence_summary({"solver": solver})
            if summary is not None:
                return summary
        manifest = record.get("run_manifest")
        if isinstance(manifest, Mapping):
            return cls.solver_convergence_summary(manifest)
        if isinstance(manifest, (str, os.PathLike)):
            try:
                manifest_data = json.loads(Path(manifest).read_text())
            except (OSError, json.JSONDecodeError):
                return None
            return cls.solver_convergence_summary(manifest_data)
        return None

    @classmethod
    def _record_solver_failed(cls, record: Mapping[str, Any]) -> bool:
        summary = cls._record_solver_convergence(record)
        return bool(summary and summary.get("failed"))

    @staticmethod
    def _failure_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, Mapping):
            for key in ("reason", "message", "error", "status"):
                text = SimulationJob._failure_text(value.get(key))
                if text:
                    return text
            try:
                return json.dumps(value, cls=CustomJSONEncoder)
            except TypeError:
                return str(value)
        if isinstance(value, (list, tuple)):
            parts = [
                text for item in value if (text := SimulationJob._failure_text(item))
            ]
            return "; ".join(parts) if parts else None
        text = str(value).strip()
        return text or None

    @classmethod
    def _task_failure_reason(cls, record: Mapping[str, Any]) -> str:
        for key in ("reason", "error", "exception", "message", "stderr"):
            text = cls._failure_text(record.get(key))
            if text:
                return text

        convergence = cls._record_solver_convergence(record)
        if convergence and convergence.get("failed"):
            residual = convergence.get("residual", convergence.get("final_residual"))
            threshold = convergence.get(
                "residual_failure_threshold", SOLVER_RESIDUAL_FAILURE_THRESHOLD
            )
            iterations = convergence.get("iterations")
            try:
                residual_value = None if residual is None else float(residual)
                threshold_value = float(threshold)
            except (TypeError, ValueError):
                residual_value = None
                threshold_value = SOLVER_RESIDUAL_FAILURE_THRESHOLD
            if residual_value is not None and residual_value > threshold_value:
                suffix = (
                    f" after {iterations} iterations" if iterations is not None else ""
                )
                return (
                    f"Solver residual {residual_value:.4g} exceeded failure "
                    f"threshold {threshold_value:.4g}{suffix}."
                )
            if convergence.get("converged") is False:
                details = []
                if iterations is not None:
                    details.append(f"{iterations} iterations")
                if residual is not None:
                    try:
                        details.append(f"residual {float(residual):.4g}")
                    except (TypeError, ValueError):
                        details.append(f"residual {residual}")
                detail_text = f" ({', '.join(details)})" if details else ""
                return f"Solver did not converge{detail_text}."
            try:
                failure_count = int(convergence.get("failure_count", 0))
            except (TypeError, ValueError):
                failure_count = 0
            if failure_count:
                return f"Solver reported {failure_count} failed solve(s)."
            status = cls._failure_text(convergence.get("status"))
            if status:
                return f"Solver convergence status: {status}."
            return "Solver convergence failed."

        exit_status = record.get("exit_status")
        if isinstance(exit_status, Mapping):
            status = cls._failure_text(exit_status.get("status"))
            code = exit_status.get("code")
            if code is not None and status:
                return f"Task exited with code {code} ({status})."
            if code is not None:
                return f"Task exited with code {code}."
            if status:
                return f"Task exit status: {status}."
        else:
            status = cls._failure_text(exit_status)
            if status:
                return f"Task exit status: {status}."

        for key in ("returncode", "return_code", "code"):
            if key in record:
                return f"Task exited with code {record[key]}."

        status = cls._failure_text(record.get("status"))
        if status:
            return f"Task status: {status}."
        return "Task failed; no failure metadata was available."

    @classmethod
    def _aggregate_solver_convergence(
        cls, task_convergences: Sequence[Mapping[str, Any]], failed_tasks: int
    ) -> Dict[str, Any]:
        """Aggregate task-level solver convergence for the Python run manifest."""

        solve_count = 0
        failure_count = 0
        worst_code = 0
        total_iterations = 0
        residuals = []
        any_converged_false = False
        any_converged_value = False

        for convergence in task_convergences:
            solves = cls._as_records(convergence.get("solves"))
            try:
                count = int(convergence.get("solve_count", 0))
            except (TypeError, ValueError):
                count = 0
            solve_count += count if count > 0 else len(solves)

            try:
                failure_count += int(convergence.get("failure_count", 0))
            except (TypeError, ValueError):
                pass

            try:
                worst_code = max(worst_code, int(convergence.get("worst_code", 0)))
            except (TypeError, ValueError):
                pass

            iterations = convergence.get("total_iterations")
            if iterations is None:
                iterations = convergence.get("iterations")
            try:
                total_iterations += int(iterations)
            except (TypeError, ValueError):
                pass

            residual = convergence.get("residual", convergence.get("final_residual"))
            if residual is not None:
                try:
                    residuals.append(float(residual))
                except (TypeError, ValueError):
                    pass

            if "converged" in convergence:
                any_converged_value = True
                if convergence.get("converged") is False:
                    any_converged_false = True

        failed = bool(failed_tasks or failure_count or any_converged_false)
        if solve_count == 0 and not task_convergences:
            status = "not_run"
        else:
            status = "failed" if failed else "converged"
        residual = max(residuals) if residuals else None
        residual_failed = (
            residual is not None and residual > SOLVER_RESIDUAL_FAILURE_THRESHOLD
        )
        if residual_failed:
            failed = True
            status = "failed"
        return {
            "status": status,
            "converged": (not failed) if any_converged_value or solve_count else None,
            "failure_count": failure_count + int(residual_failed),
            "solve_count": solve_count,
            "worst_code": worst_code,
            **({"iterations": total_iterations} if total_iterations else {}),
            **(
                {"residual": cls._rounded_solver_float(residual)}
                if residual is not None
                else {}
            ),
            "residual_failure_threshold": SOLVER_RESIDUAL_FAILURE_THRESHOLD,
        }

    @classmethod
    def _task_is_complete(
        cls, record: Mapping[str, Any], normalized_status: Optional[str]
    ) -> bool:
        if "complete" in record:
            return bool(record.get("complete"))
        if normalized_status in {"succeeded", "current", "reused", "skipped"}:
            return True
        return cls._normalized_task_status(record.get("status")) == "succeeded"

    @staticmethod
    def _record_duration_seconds(record: Mapping[str, Any]) -> Optional[float]:
        duration_keys = (
            "duration_seconds",
            "elapsed_seconds",
            "runtime_seconds",
            "wall_time_seconds",
            "time_seconds",
            "seconds",
            "duration",
            "elapsed",
            "runtime",
            "wall_time",
            "total_seconds",
            "total",
        )
        for key in duration_keys:
            if key not in record:
                continue
            value = record[key]
            if isinstance(value, Mapping):
                value = value.get("seconds")
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _numeric_record_value(
        record: Mapping[str, Any], keys: Iterable[str]
    ) -> Optional[float]:
        for key in keys:
            if key not in record:
                continue
            value = record[key]
            if isinstance(value, Mapping):
                value = value.get("value") or value.get("count")
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _record_core_count(cls, record: Mapping[str, Any]) -> Optional[float]:
        cores = cls._numeric_record_value(
            record,
            (
                "core_count",
                "cores",
                "n_cores",
                "num_cores",
                "cpu_count",
                "cpus",
                "ncpus",
            ),
        )
        if cores is not None and cores > 0:
            return cores

        ranks = cls._numeric_record_value(
            record,
            (
                "n_ranks",
                "ranks",
                "mpi_ranks",
                "num_ranks",
                "nprocs",
                "procs",
                "processes",
            ),
        )
        threads = cls._numeric_record_value(
            record,
            (
                "threads_per_rank",
                "omp_threads",
                "threads",
                "n_threads",
                "num_threads",
            ),
        )
        if ranks is not None and threads is not None:
            return max(1.0, ranks * threads)
        if threads is not None:
            return max(1.0, threads)
        return None

    def _default_core_count(self) -> Optional[float]:
        metadata = self.run_metadata
        for source in (metadata.manifest, metadata.timings):
            if isinstance(source, Mapping):
                cores = self._record_core_count(source)
                if cores is not None:
                    return cores
        return None

    def _task_records(self) -> List[Mapping[str, Any]]:
        metadata = self.run_metadata
        state = self.run_state()
        records: List[Mapping[str, Any]] = []
        for source in (state or metadata.state, metadata.timings, metadata.manifest):
            if not isinstance(source, Mapping):
                continue
            records.extend(self._as_records(source.get("tasks")))
            records.extend(self._as_records(source.get("task_results")))
            records.extend(self._as_records(source.get("task_timings")))
            records.extend(self._as_records(source.get("frequencies")))
            records.extend(self._as_records(source.get("errors")))
        if metadata.error:
            records.extend(self._as_records(metadata.error.get("tasks")))
            records.extend(self._as_records(metadata.error.get("errors")))
            if not records:
                records.append(metadata.error)
        return records

    def frequency_status(self) -> List[Dict[str, Any]]:
        """Return per-frequency run status rows for this job.

        Status is inferred from expected trace files plus any Python/solver run
        metadata available beside the results. Task numbers are one-based to
        match solver task IDs and trace filenames.
        """

        manifest = self.trace_manifest
        state = self.run_state()
        rows: Dict[int, Dict[str, Any]] = {}
        ordered_files = list(manifest.files)
        for task, file in enumerate(ordered_files, start=1):
            trace_file, trace_exists = self._trace_output_path_for_task(
                task,
                manifest=manifest,
            )
            current = self.is_task_current(task, state=state)
            rows[task] = {
                "task": task,
                "frequency": manifest.frequencies.get(task),
                "status": "succeeded" if current else "not_run",
                "trace_file": trace_file if trace_exists else file,
                "trace_exists": trace_exists,
                "current": current,
                "duration_seconds": None,
                "metadata": {},
            }

        for record in self._task_records():
            task = self._task_number_from_record(record)
            if task is None or task not in rows:
                continue
            row = rows[task]
            row["metadata"] = {**row["metadata"], **dict(record)}
            duration = self._record_duration_seconds(record)
            if duration is not None:
                row["duration_seconds"] = duration
            status = self._normalized_task_status(record.get("status"))
            if status == "failed" or self._record_solver_failed(record):
                row["status"] = "failed"
            elif status == "succeeded" and row["current"] and row["status"] != "failed":
                row["status"] = "succeeded"

        return [rows[task] for task in sorted(rows)]

    def failed_tasks(self, *, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """Return failed task rows with the best available failure reason.

        The failure classification matches :meth:`frequency_status`, including
        solver convergence failures such as residuals above the failure
        threshold.  Task numbers are one-based.
        """

        failures: List[Dict[str, Any]] = []
        assigned_tasks = set()
        for row in self.frequency_status():
            assigned_tasks.add(row["task"])
            if row["status"] != "failed":
                continue
            metadata = dict(row.get("metadata") or {})
            failure = {
                "task": row["task"],
                "frequency": row["frequency"],
                "status": "failed",
                "reason": self._task_failure_reason(metadata),
                "trace_file": row["trace_file"],
            }
            if row.get("duration_seconds") is not None:
                failure["duration_seconds"] = row["duration_seconds"]
            run_manifest = metadata.get("run_manifest")
            if run_manifest is not None:
                failure["run_manifest"] = run_manifest
            convergence = self._record_solver_convergence(metadata)
            if convergence is not None:
                failure["solver"] = {"convergence": convergence}
            if include_metadata:
                failure["metadata"] = metadata
            failures.append(failure)

        for record in self._task_records():
            task = self._task_number_from_record(record)
            if task in assigned_tasks:
                continue
            status = self._normalized_task_status(record.get("status"))
            if status != "failed" and not self._record_solver_failed(record):
                continue
            failure = {
                "task": task,
                "frequency": (
                    self._canonical_frequency_value(self.f_list[task - 1])
                    if task is not None and 1 <= task <= self.n_tasks
                    else None
                ),
                "status": "failed",
                "reason": self._task_failure_reason(record),
            }
            run_manifest = record.get("run_manifest")
            if run_manifest is not None:
                failure["run_manifest"] = run_manifest
            convergence = self._record_solver_convergence(record)
            if convergence is not None:
                failure["solver"] = {"convergence": convergence}
            if include_metadata:
                failure["metadata"] = dict(record)
            failures.append(failure)

        return failures

    def list_failed_tasks(
        self, *, include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """Alias for :meth:`failed_tasks`."""

        return self.failed_tasks(include_metadata=include_metadata)

    def frequency_summary(self) -> Dict[str, int]:
        """Count succeeded, failed, and not-yet-run frequencies."""

        rows = self.frequency_status()
        summary = {"total": len(rows), "succeeded": 0, "failed": 0, "not_run": 0}
        for row in rows:
            status = row["status"]
            summary[status] = summary.get(status, 0) + 1
        assigned_tasks = {row["task"] for row in rows}
        unassigned_failures = 0
        for record in self._task_records():
            status = self._normalized_task_status(record.get("status"))
            task = self._task_number_from_record(record)
            if status == "failed" and task not in assigned_tasks:
                unassigned_failures += 1
        if unassigned_failures:
            summary["unassigned_failures"] = unassigned_failures
        return summary

    def print_frequency_summary(self, file: Optional[Any] = None) -> Dict[str, int]:
        """Print and return a concise frequency status summary."""

        summary = self.frequency_summary()
        message = (
            f"Job {self.name}: {summary['succeeded']}/{summary['total']} frequencies "
            f"succeeded; {summary['failed']} failed; "
            f"{summary['not_run']} not run."
        )
        print(message, file=file)
        if summary.get("unassigned_failures"):
            print(
                f"  {summary['unassigned_failures']} failure records were not tied "
                "to a frequency task.",
                file=file,
            )
        return summary

    def task_timings(self) -> List[Dict[str, Any]]:
        """Return frequency rows that include per-task runtime in seconds."""

        rows = self.frequency_status()
        default_cores = self._default_core_count()
        timings = []
        for row in rows:
            duration = row.get("duration_seconds")
            if duration is None:
                continue
            core_count = self._record_core_count(row.get("metadata", {}))
            if core_count is None:
                core_count = default_cores
            duration = float(duration)
            core_hours = (
                duration * float(core_count) / 3600.0
                if core_count is not None
                else None
            )
            timings.append(
                {
                    "task": row["task"],
                    "frequency": row["frequency"],
                    "duration_seconds": duration,
                    "core_count": core_count,
                    "core_hours": core_hours,
                    "status": row["status"],
                    "trace_file": row["trace_file"],
                }
            )
        return timings

    def plot_task_timings(
        self,
        ax: Optional[Any] = None,
        *,
        x: str = "frequency",
        style: str = "auto",
        unit: str = "seconds",
        cores: Optional[float] = None,
        max_xticks: int = 8,
        show: bool = False,
        title: Optional[str] = None,
        **bar_kwargs: Any,
    ) -> Any:
        """Plot per-frequency task runtimes and return the matplotlib axes.

        ``unit`` may be ``"seconds"``, ``"hours"``, or ``"core-hours"``. For
        large frequency sweeps the default style switches from bars to lines so
        the axis stays readable.
        """

        timings = self.task_timings()
        if not timings:
            raise ValueError(
                "No per-task timings were found in run metadata for this job"
            )

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            from frequensolve._optional import optional_dependency_error

            raise optional_dependency_error(
                "Job timing plotting",
                extra="visual",
                dependencies=("matplotlib",),
                error=exc,
            ) from exc

        if ax is None:
            _, ax = plt.subplots()

        from matplotlib.ticker import MaxNLocator

        def frequency_x(value: Any) -> Optional[float]:
            if isinstance(value, Mapping) and "real" in value:
                return float(value["real"])
            try:
                return float(np.real(value))
            except (TypeError, ValueError):
                return None

        status_colors = {
            "succeeded": "#2e7d32",
            "failed": "#c62828",
            "not_run": "#757575",
        }

        if x not in {"frequency", "task"}:
            raise ValueError("x must be 'frequency' or 'task'")
        x_values = [frequency_x(row["frequency"]) for row in timings]
        use_frequency_axis = x == "frequency" and all(
            value is not None for value in x_values
        )
        if use_frequency_axis:
            plot_rows = sorted(
                zip(x_values, timings),
                key=lambda item: float(item[0]),
            )
            xs = np.asarray([float(value) for value, _row in plot_rows])
            rows = [row for _value, row in plot_rows]
            ax.set_xlabel("Frequency (Hz)")
            integer_axis = False
        else:
            rows = list(timings)
            xs = np.asarray([row["task"] for row in rows], dtype=float)
            ax.set_xlabel("Frequency task")
            integer_axis = True

        normalized_unit = unit.strip().lower().replace("_", "-")
        if normalized_unit in {"s", "sec", "secs", "second", "seconds"}:
            values = np.asarray([float(row["duration_seconds"]) for row in rows])
            ylabel = "Runtime (s)"
        elif normalized_unit in {"h", "hr", "hrs", "hour", "hours"}:
            values = np.asarray(
                [float(row["duration_seconds"]) / 3600.0 for row in rows]
            )
            ylabel = "Runtime (hours)"
        elif normalized_unit in {"core-hour", "core-hours", "core hour", "core hours"}:
            plot_values = []
            for row in rows:
                core_count = cores if cores is not None else row.get("core_count")
                if core_count is None:
                    raise ValueError(
                        "Core-hour plotting requires per-task core metadata or "
                        "a `cores=` override."
                    )
                plot_values.append(
                    float(row["duration_seconds"]) * float(core_count) / 3600.0
                )
            values = np.asarray(plot_values)
            ylabel = "Runtime (core-hours)"
        else:
            raise ValueError("unit must be 'seconds', 'hours', or 'core-hours'")

        if style == "auto":
            plot_style = "bar" if len(rows) <= 80 else "line"
        elif style in {"bar", "line"}:
            plot_style = style
        else:
            raise ValueError("style must be 'auto', 'bar', or 'line'")

        if plot_style == "bar":
            width = 0.8
            if use_frequency_axis and len(xs) > 1:
                diffs = np.diff(np.unique(xs))
                diffs = diffs[diffs > 0.0]
                if len(diffs):
                    width = 0.8 * float(np.min(diffs))

            bar_kwargs.setdefault(
                "color",
                [status_colors.get(row["status"], "#546e7a") for row in rows],
            )
            ax.bar(xs, values, width=width, **bar_kwargs)
        else:
            marker = "o" if len(rows) <= 50 else None
            ax.plot(xs, values, marker=marker, linewidth=1.8, color="#2e7d32")

        ax.set_ylabel(ylabel)
        ax.set_title(title or f"{self.name} task timings")
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=max(2, int(max_xticks)), integer=integer_axis)
        )
        ax.grid(axis="y", alpha=0.25)
        if show:
            plt.show()
        return ax

    def phase_timings(
        self,
        phases: Optional[Sequence[str]] = None,
        *,
        include_zero: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return per-frequency solver phase timings from run metadata.

        The fast solver writes phase timings in ``_fs_run/timings.json`` when
        they are available. Rows returned by this method are keyed by task and
        frequency, with one numeric column per phase. Missing phases are filled
        with zero so the result is easy to tabulate or plot.
        """

        requested = [str(phase) for phase in phases] if phases is not None else None
        rows_by_task = {row["task"]: row for row in self.frequency_status()}
        timing_records: Dict[int, Mapping[str, Any]] = {}
        phase_names: List[str] = []

        for record in self._task_records():
            phase_map = record.get("phases") or record.get("phase_timings")
            if not isinstance(phase_map, Mapping):
                continue
            task = self._task_number_from_record(record)
            if task is None:
                continue
            timing_records[task] = phase_map
            for phase in phase_map:
                phase = str(phase)
                if requested is not None and phase not in requested:
                    continue
                if phase not in phase_names:
                    phase_names.append(phase)

        if requested is not None:
            phase_names = requested

        rows: List[Dict[str, Any]] = []
        for task in sorted(timing_records):
            phase_map = timing_records[task]
            values: Dict[str, float] = {}
            total = 0.0
            for phase in phase_names:
                try:
                    seconds = float(phase_map.get(phase, 0.0))
                except (TypeError, ValueError):
                    seconds = 0.0
                if seconds != 0.0 or include_zero:
                    values[phase] = seconds
                total += seconds
            if not values and not include_zero:
                continue
            status_row = rows_by_task.get(task, {})
            rows.append(
                {
                    "task": task,
                    "frequency": status_row.get("frequency"),
                    "status": status_row.get("status"),
                    "total_seconds": total,
                    **values,
                }
            )
        return rows

    def plot_phase_timings(
        self,
        ax: Optional[Any] = None,
        *,
        phases: Optional[Sequence[str]] = None,
        x: str = "frequency",
        style: str = "auto",
        unit: str = "seconds",
        include_zero: bool = False,
        max_xticks: int = 8,
        show: bool = False,
        title: Optional[str] = None,
        colors: Optional[Mapping[str, str]] = None,
        **plot_kwargs: Any,
    ) -> Any:
        """Plot stacked per-frequency solver phase timings.

        ``unit`` may be ``"seconds"`` or ``"hours"``. For long frequency
        sweeps the default style switches from stacked bars to one line per
        phase so the plot remains readable.
        """

        rows = self.phase_timings(phases=phases, include_zero=include_zero)
        if not rows:
            raise ValueError(
                "No solver phase timings were found in run metadata for this job"
            )

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            from frequensolve._optional import optional_dependency_error

            raise optional_dependency_error(
                "Job phase timing plotting",
                extra="visual",
                dependencies=("matplotlib",),
                error=exc,
            ) from exc

        if ax is None:
            _, ax = plt.subplots()

        from matplotlib.ticker import MaxNLocator

        def frequency_x(value: Any) -> Optional[float]:
            if isinstance(value, Mapping) and "real" in value:
                return float(value["real"])
            try:
                return float(np.real(value))
            except (TypeError, ValueError):
                return None

        if x not in {"frequency", "task"}:
            raise ValueError("x must be 'frequency' or 'task'")

        x_values = [frequency_x(row["frequency"]) for row in rows]
        use_frequency_axis = x == "frequency" and all(
            value is not None for value in x_values
        )
        if use_frequency_axis:
            plot_rows = sorted(zip(x_values, rows), key=lambda item: float(item[0]))
            xs = np.asarray([float(value) for value, _row in plot_rows])
            rows = [row for _value, row in plot_rows]
            ax.set_xlabel("Frequency (Hz)")
            integer_axis = False
        else:
            xs = np.asarray([row["task"] for row in rows], dtype=float)
            ax.set_xlabel("Frequency task")
            integer_axis = True

        normalized_unit = unit.strip().lower().replace("_", "-")
        if normalized_unit in {"s", "sec", "secs", "second", "seconds"}:
            scale = 1.0
            ylabel = "Runtime (s)"
        elif normalized_unit in {"h", "hr", "hrs", "hour", "hours"}:
            scale = 1.0 / 3600.0
            ylabel = "Runtime (hours)"
        else:
            raise ValueError("unit must be 'seconds' or 'hours'")

        phase_names = list(phases) if phases is not None else []
        if not phase_names:
            for row in rows:
                for key, value in row.items():
                    if key in {"task", "frequency", "status", "total_seconds"}:
                        continue
                    if not include_zero and float(value) == 0.0:
                        continue
                    if key not in phase_names:
                        phase_names.append(key)
        if not phase_names:
            raise ValueError("No nonzero solver phases were found for plotting")

        if style == "auto":
            plot_style = "bar" if len(rows) <= 80 else "line"
        elif style in {"bar", "line"}:
            plot_style = style
        else:
            raise ValueError("style must be 'auto', 'bar', or 'line'")

        default_colors = {
            "setup": "#546e7a",
            "mesh": "#00897b",
            "assembly": "#3949ab",
            "solve_forward": "#ef6c00",
            "solve_adjoint": "#8e24aa",
            "imaging": "#c62828",
        }
        palette = {**default_colors, **dict(colors or {})}

        values_by_phase = {
            phase: np.asarray([float(row.get(phase, 0.0)) * scale for row in rows])
            for phase in phase_names
        }
        if not include_zero:
            values_by_phase = {
                phase: values
                for phase, values in values_by_phase.items()
                if np.any(values != 0.0)
            }
            phase_names = [phase for phase in phase_names if phase in values_by_phase]
            if not phase_names:
                raise ValueError("No nonzero solver phases were found for plotting")

        if plot_style == "bar":
            width = 0.8
            if use_frequency_axis and len(xs) > 1:
                diffs = np.diff(np.unique(xs))
                diffs = diffs[diffs > 0.0]
                if len(diffs):
                    width = 0.8 * float(np.min(diffs))
            bottoms = np.zeros(len(rows), dtype=float)
            for phase in phase_names:
                values = values_by_phase[phase]
                ax.bar(
                    xs,
                    values,
                    bottom=bottoms,
                    width=width,
                    label=phase.replace("_", " "),
                    color=palette.get(phase),
                    **plot_kwargs,
                )
                bottoms += values
        else:
            marker = "o" if len(rows) <= 50 else None
            for phase in phase_names:
                ax.plot(
                    xs,
                    values_by_phase[phase],
                    marker=marker,
                    linewidth=1.8,
                    label=phase.replace("_", " "),
                    color=palette.get(phase),
                    **plot_kwargs,
                )

        ax.set_ylabel(ylabel)
        ax.set_title(title or f"{self.name} solver phase timings")
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=max(2, int(max_xticks)), integer=integer_axis)
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        if show:
            plt.show()
        return ax

    def run_state(self) -> Dict[str, Any]:
        if not self.run_state_file.exists():
            return {}
        try:
            return json.loads(self.run_state_file.read_text())
        except json.JSONDecodeError:
            return {}

    def is_task_current(
        self,
        task: int,
        *,
        state: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        files = self.expected_trace_files()
        if task < 1 or task > len(files):
            return False
        state = self.run_state() if state is None else state
        expected_fingerprint = self.task_fingerprint(task)
        full_run_matches = (
            state.get("fingerprint") == self.fingerprint()
            and state.get("status") in {"completed", "skipped"}
            and self._task_summary_successful(
                state.get("task_summary"),
                expected_total=len(files),
            )
        )
        manifest = self.trace_manifest
        trace_file, expected_exists = self._trace_output_path_for_task(
            task,
            manifest=manifest,
        )
        packed_task_exists = self._packed_trace_has_current_task(
            task,
            manifest=manifest,
        )
        packed_task_reusable = packed_task_exists and manifest.packed_complete
        if not expected_exists:
            if not packed_task_reusable:
                return False
        records = self._state_task_records(state)
        for record in records:
            if self._task_number_from_record(record) != task:
                continue
            record_fingerprint = record.get("fingerprint")
            if (
                record_fingerprint is not None
                and record_fingerprint != expected_fingerprint
            ):
                continue
            if record_fingerprint is None and not full_run_matches:
                continue
            status = self._normalized_task_status(record.get("status"))
            if status != "succeeded":
                continue
            stored_path = self._resolve_stored_trace_path(
                record.get("path") or record.get("trace_file")
            )
            if stored_path is None:
                return expected_exists or packed_task_reusable
            stored_matches_trace = (
                Path(stored_path).resolve(strict=False)
                == trace_file.resolve(strict=False)
                and trace_file.exists()
            )
            return (
                self._trace_file_exists(stored_path)
                or stored_matches_trace
                or packed_task_reusable
            )
        return packed_task_reusable and full_run_matches

    def current_tasks(self) -> List[int]:
        """Return one-based frequency tasks that are current for this job."""

        state = self.run_state()
        return [
            task
            for task in range(1, self.n_tasks + 1)
            if self.is_task_current(task, state=state)
        ]

    def _reuse_task_outputs_from_state(
        self, state: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        records = self._state_task_records(state)
        source_by_fingerprint: Dict[str, Path] = {}
        for record in records:
            if self._normalized_task_status(record.get("status")) != "succeeded":
                continue
            fingerprint = record.get("fingerprint")
            source = self._resolve_stored_trace_path(
                record.get("path") or record.get("trace_file")
            )
            if source is not None and source == self.trace_manifest.packed_file:
                continue
            if fingerprint and source is not None and self._trace_file_exists(source):
                source_by_fingerprint.setdefault(fingerprint, source)

        copies = []
        for task, target in enumerate(self.expected_trace_files(), start=1):
            if self.is_task_current(task, state=state):
                continue
            source = source_by_fingerprint.get(self.task_fingerprint(task))
            if source is None:
                continue
            target = Path(target)
            if source.resolve() == target.resolve():
                continue
            copies.append((task, source, target))

        if not copies:
            return []

        stage_dir = self._result_path / "_fs_run" / "reuse"
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged = []
        try:
            for task, source, _target in copies:
                stage = stage_dir / f"task_{task}{source.suffix}"
                shutil.copy2(source, stage)
                staged.append((task, stage, _target))

            reused = []
            for task, stage, target in staged:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(stage, target)
                reused.append(
                    {
                        "task": task,
                        "status": "reused",
                        "duration_seconds": 0.0,
                        "fingerprint": self.task_fingerprint(task),
                        "path": self._stored_trace_path(target),
                    }
                )
            return reused
        finally:
            for _task, stage, _target in staged:
                try:
                    stage.unlink()
                except FileNotFoundError:
                    pass

    def _remove_trace_outputs_for_tasks(self, tasks: Iterable[int]) -> bool:
        removed = False
        files = self.expected_trace_files()
        for task in tasks:
            if task < 1 or task > len(files):
                continue
            shard = self._matching_frequency_trace_file(task)
            paths = {files[task - 1], self._legacy_trace_file(files[task - 1])}
            if shard is not None:
                paths.add(shard)
            for path in paths:
                try:
                    path.unlink()
                    removed = True
                except FileNotFoundError:
                    pass
        return removed

    def task_run_plan(
        self, *, reuse: bool = False, force: bool = False
    ) -> Dict[str, Any]:
        """Plan which zero-based solver task indices still need to run.

        When ``reuse`` is true, matching trace files from an earlier frequency
        layout are copied into their current task-numbered locations and the
        run state is updated to record those reused tasks.
        """

        if force:
            pending = list(range(self.n_tasks))
            removed_stale_outputs = self._remove_trace_outputs_for_tasks(
                range(1, self.n_tasks + 1)
            )
            removed_stale_outputs = (
                self.remove_packed_trace_products() or removed_stale_outputs
            )
            if removed_stale_outputs:
                self.invalidate_trace_cache()
            return {
                "pending_indices": pending,
                "current_tasks": [],
                "reused_tasks": [],
            }

        state = self.run_state()
        current_records = []
        manifest = self.trace_manifest
        for task, path in enumerate(self.expected_trace_files(), start=1):
            if not self.is_task_current(task, state=state):
                continue
            file_path, _exists = self._trace_output_path_for_task(
                task,
                manifest=manifest,
            )
            current_records.append(
                {
                    "task": task,
                    "status": "current",
                    "duration_seconds": 0.0,
                    "fingerprint": self.task_fingerprint(task),
                    "path": self._stored_trace_path(file_path),
                }
            )
        reused = self._reuse_task_outputs_from_state(state) if reuse and state else []
        if reused:
            self.write_run_state(status="partial", tasks=[*current_records, *reused])
            state = self.run_state()
        reused_tasks = {self._task_number_from_record(record) for record in reused}
        pending = []
        current = []
        for task in range(1, self.n_tasks + 1):
            if self.is_task_current(task, state=state) or task in reused_tasks:
                current.append(task)
            else:
                pending.append(task - 1)
        removed_stale_outputs = self._remove_trace_outputs_for_tasks(
            index + 1 for index in pending
        )
        if reused or removed_stale_outputs:
            self.invalidate_trace_cache()
        return {
            "pending_indices": pending,
            "current_tasks": current,
            "reused_tasks": reused,
        }

    def is_run_current(self) -> bool:
        if not self.trace_outputs_exist():
            return False

        metadata = self.run_metadata
        if metadata.manifest:
            if not metadata.successful:
                return False
            task_summary = metadata.manifest.get("task_summary")
            if task_summary is not None:
                if not self._task_summary_successful(
                    task_summary,
                    expected_total=self.n_tasks,
                ):
                    return False
            if metadata.job_file_hash and self._file is not None:
                if metadata.job_file_hash != self._sha256_file(self._file):
                    return False
            if metadata.simulation_file_hash and self.simulation._file is not None:
                if metadata.simulation_file_hash != self._sha256_file(
                    self.simulation._file
                ):
                    return False
            return True

        state = self.run_state()
        if not state:
            return False
        if state.get("fingerprint") != self.fingerprint():
            return False
        if state.get("status") not in {"completed", "skipped"}:
            return False
        if not self._task_summary_successful(
            state.get("task_summary"),
            expected_total=self.n_tasks,
        ):
            return False
        return len(self.current_tasks()) == self.n_tasks

    @staticmethod
    def _task_summary_successful(
        summary: Any,
        *,
        expected_total: int,
    ) -> bool:
        if not isinstance(summary, Mapping):
            return False
        try:
            total = int(summary.get("total") or 0)
            complete = int(summary.get("complete") or 0)
            succeeded = int(summary.get("succeeded") or 0)
            failed = int(summary.get("failed") or 0)
            not_run = int(summary.get("not_run") or 0)
        except (TypeError, ValueError):
            return False
        return (
            total == int(expected_total)
            and complete == total
            and succeeded == total
            and failed == 0
            and not_run == 0
        )

    def write_run_state(self, status: str = "completed", **extra) -> Path:
        self._result_path.mkdir(parents=True, exist_ok=True)
        extra = dict(extra)
        task_results = self._as_records(extra.pop("tasks", None))
        result_by_task = self._task_record_by_task(task_results)
        previous_state = self.run_state()
        previous_by_task = self._task_record_by_task(
            self._state_task_records(previous_state)
        )
        bootstrap_existing_outputs = not task_results and status in {
            "completed",
            "skipped",
        }

        files = []
        task_rows = []
        manifest = self.trace_manifest
        for task, path in enumerate(self.expected_trace_files(), start=1):
            file_path, exists = self._trace_output_path_for_task(
                task,
                manifest=manifest,
            )
            stored_path = self._stored_trace_path(file_path)
            files.append({"path": stored_path, "exists": exists})

            result = dict(result_by_task.get(task, {}))
            task_status = self._normalized_task_status(result.get("status"))
            previously_current = self.is_task_current(task, state=previous_state)
            if not result and previously_current:
                result = dict(previous_by_task.get(task, {}))
            if task_status is None and previously_current:
                task_status = "current"
            elif task_status is None and exists and bootstrap_existing_outputs:
                task_status = "succeeded"
            if task_status is None:
                task_status = "not_run"
            solver_convergence = self._record_solver_convergence(result)
            if solver_convergence is not None and solver_convergence.get("failed"):
                task_status = "failed"

            duration = self._record_duration_seconds(result)
            row = {
                "task": task,
                "frequency": self._canonical_frequency_value(self.f_list[task - 1]),
                "status": task_status,
                "complete": self._task_is_complete(result, task_status),
                "fingerprint": self.task_fingerprint(task),
                "path": stored_path,
                "exists": exists,
            }
            if solver_convergence is not None:
                row["solver"] = {"convergence": solver_convergence}
            if duration is not None:
                row["duration_seconds"] = duration
            core_count = self._record_core_count(result)
            if core_count is not None:
                row["core_count"] = core_count
            for key in (
                "returncode",
                "n_ranks",
                "ranks",
                "threads_per_rank",
                "n_threads",
                "threads",
            ):
                if key in result:
                    row[key] = result[key]
            task_rows.append(row)

        task_summary = {
            "total": len(task_rows),
            "complete": 0,
            "succeeded": 0,
            "failed": 0,
            "not_run": 0,
        }
        solver_convergences = []
        solver_task_summaries = []
        for row in task_rows:
            if row.get("complete"):
                task_summary["complete"] += 1
            status_key = row.get("status")
            normalized_status = self._normalized_task_status(status_key)
            if normalized_status == "succeeded":
                task_summary["succeeded"] += 1
            elif normalized_status == "failed":
                task_summary["failed"] += 1
            elif normalized_status == "not_run":
                task_summary["not_run"] += 1
            convergence = row.get("solver", {}).get("convergence")
            if isinstance(convergence, Mapping):
                solver_convergences.append(convergence)
                solver_task_summaries.append(
                    {
                        "task": row["task"],
                        "frequency": row["frequency"],
                        "converged": convergence.get("converged"),
                        "iterations": convergence.get("iterations"),
                        "residual": convergence.get(
                            "residual", convergence.get("final_residual")
                        ),
                        "status": convergence.get("status"),
                    }
                )

        payload = {
            "schema": "frequensolve-python-run-1",
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": self.fingerprint(),
            "fingerprint_payload": self.fingerprint_payload(),
            "tasks": task_rows,
            "task_summary": task_summary,
            "outputs": {"traces": files},
        }
        if solver_task_summaries:
            aggregate_convergence = self._aggregate_solver_convergence(
                solver_convergences, task_summary["failed"]
            )
            payload["solver"] = {
                "convergence": {
                    **aggregate_convergence,
                    "tasks": solver_task_summaries,
                }
            }
        if task_results:
            payload["task_results"] = task_results
        payload.update(extra)
        self._write_json_file(self.run_state_file, payload)
        self._write_solver_run_manifest_summary(payload)
        return self.run_state_file

    def _write_solver_run_manifest_summary(self, payload: Mapping[str, Any]) -> None:
        """Mirror Python job-level task/convergence summaries into solver metadata."""

        manifest_path = self._result_path / "_fs_run" / "run_manifest.json"
        if not manifest_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(manifest, dict):
            return

        task_summary = payload.get("task_summary")
        if isinstance(task_summary, Mapping):
            manifest["task_summary"] = dict(task_summary)

        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            manifest["tasks"] = [
                dict(task) for task in tasks if isinstance(task, Mapping)
            ]

        solver_payload = payload.get("solver")
        convergence = (
            solver_payload.get("convergence")
            if isinstance(solver_payload, Mapping)
            else None
        )
        if isinstance(convergence, Mapping):
            solver = manifest.get("solver")
            if not isinstance(solver, dict):
                solver = {}
            solver["convergence"] = dict(convergence)
            manifest["solver"] = solver

        self._write_json_file(manifest_path, manifest)

    def task_run_manifest_path(self, task: int) -> Path:
        """Return the local solver run manifest path for a one-based task number."""

        if task < 1:
            raise ValueError("Task numbers are one-based and must be >= 1")
        return (
            self._result_path
            / "_fs_run"
            / "tasks"
            / f"task_{task:06d}"
            / "run_manifest.json"
        )

    def collect_task_run_manifests(
        self,
        *,
        status: str = "completed",
    ) -> Optional[Path]:
        """Aggregate fetched task run manifests into the job run manifests.

        Remote sites can fetch ``results/_fs_run`` and then call this method on
        the host.  The method scans task-level solver manifests, converts their
        solver convergence blocks into the same task records used by local runs,
        writes ``_fs_python_run.json``, and mirrors the task/convergence summary
        into ``results/_fs_run/run_manifest.json``.
        """

        task_records = []
        for task in range(1, self.n_tasks + 1):
            manifest_path = self.task_run_manifest_path(task)
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            convergence = self.solver_convergence_summary(manifest)
            solver_failed = bool(convergence and convergence.get("failed"))
            exit_status = manifest.get("exit_status")
            returncode = None
            exit_failed = False
            raw_exit_status = ""
            if isinstance(exit_status, Mapping):
                try:
                    returncode = int(exit_status.get("code", 0))
                except (TypeError, ValueError):
                    returncode = None
                raw_exit_status = str(exit_status.get("status", "")).lower()
                exit_failed = bool(
                    (returncode is not None and returncode != 0)
                    or raw_exit_status in {"failed", "failure", "error"}
                )
            elif exit_status is not None:
                raw_exit_status = str(exit_status).lower()
                exit_failed = raw_exit_status in {"failed", "failure", "error"}
            execution = manifest.get("execution")
            skipped = bool(raw_exit_status == "skipped")
            if isinstance(execution, Mapping):
                skipped = skipped or bool(execution.get("skipped"))
            if skipped:
                solver_failed = False
                exit_failed = False
                convergence = None
            record: Dict[str, Any] = {
                "task_id": task - 1,
                "status": (
                    "skipped"
                    if skipped
                    else ("error" if solver_failed or exit_failed else "success")
                ),
                "complete": True,
                "run_manifest": str(manifest_path),
            }
            if returncode is not None:
                record["returncode"] = returncode
            if convergence is not None:
                record["solver"] = {"convergence": convergence}
            if isinstance(execution, Mapping):
                mpi = execution.get("mpi")
                if isinstance(mpi, Mapping) and "ranks" in mpi:
                    record["n_ranks"] = mpi["ranks"]
                openmp = execution.get("openmp")
                if isinstance(openmp, Mapping) and "threads" in openmp:
                    record["threads_per_rank"] = openmp["threads"]
            task_records.append(record)

        if not task_records:
            return None
        return self.write_run_state(status=status, tasks=task_records)

    @staticmethod
    def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(payload, cls=CustomJSONEncoder, indent=3))
        tmp.replace(path)

    @staticmethod
    def _map_payload_paths(
        value: Any,
        *,
        source_project: Path,
        target_project: Path,
    ) -> Any:
        """Map absolute project paths in a JSON-like payload to another project."""

        if isinstance(value, Mapping):
            return {
                key: SimulationJob._map_payload_paths(
                    item,
                    source_project=source_project,
                    target_project=target_project,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                SimulationJob._map_payload_paths(
                    item,
                    source_project=source_project,
                    target_project=target_project,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            return [
                SimulationJob._map_payload_paths(
                    item,
                    source_project=source_project,
                    target_project=target_project,
                )
                for item in value
            ]
        if isinstance(value, Path):
            return SimulationJob._map_payload_paths(
                str(value),
                source_project=source_project,
                target_project=target_project,
            )
        if isinstance(value, str):
            source = str(source_project)
            if source and source in value:
                return value.replace(source, str(target_project))
        return value

    @staticmethod
    def _unique_paths(paths: Iterable[Union[str, Path]]) -> List[Path]:
        unique: List[Path] = []
        seen: set[str] = set()
        for value in paths:
            if value is None:
                continue
            try:
                path = Path(value).expanduser()
            except TypeError:
                continue
            candidates = [path]
            if path.is_absolute():
                try:
                    resolved = path.resolve()
                except OSError:
                    resolved = path
                candidates.append(resolved)
            for candidate in candidates:
                text = str(candidate)
                if text in {"", "."} or candidate.anchor == text:
                    continue
                if text in seen:
                    continue
                seen.add(text)
                unique.append(candidate)
        return unique

    @staticmethod
    def _project_root_from_artifact_path(value: Union[str, Path]) -> Optional[Path]:
        text = SimulationJob._strip_file_locator(value)
        try:
            path = Path(text).expanduser()
        except TypeError:
            return None
        if not path.is_absolute():
            return None
        parts = path.parts
        for marker in ("simulations", "jobs"):
            if marker not in parts:
                continue
            index = parts.index(marker)
            if index <= 0:
                continue
            return Path(*parts[:index])
        return None

    @staticmethod
    def _payload_project_roots(value: Any) -> List[Path]:
        roots: List[Path] = []
        if isinstance(value, Mapping):
            project_path = value.get("project_path")
            if isinstance(project_path, (str, Path)):
                roots.append(Path(project_path).expanduser())
            for item in value.values():
                roots.extend(SimulationJob._payload_project_roots(item))
        elif isinstance(value, (list, tuple)):
            for item in value:
                roots.extend(SimulationJob._payload_project_roots(item))
        elif isinstance(value, (str, Path)):
            root = SimulationJob._project_root_from_artifact_path(value)
            if root is not None:
                roots.append(root)
        return roots

    def _remote_source_projects(
        self,
        local_layout: JobLayout,
        *payloads: Any,
    ) -> List[Path]:
        roots: List[Union[str, Path]] = [local_layout.project]
        for payload in payloads:
            roots.extend(self._payload_project_roots(payload))
        roots.append(self.project_path)
        simulation = getattr(self, "simulation", None)
        if simulation is not None:
            sim_project = getattr(simulation, "project_path", None)
            if sim_project is not None:
                roots.append(sim_project)
            sim_proj_path = getattr(simulation, "_proj_path", None)
            if sim_proj_path is not None:
                roots.append(sim_proj_path)
        return self._unique_paths(roots)

    @staticmethod
    def _map_payload_project_roots(
        payload: Mapping[str, Any],
        *,
        source_projects: Iterable[Path],
        target_project: Path,
    ) -> Dict[str, Any]:
        data: Any = dict(payload)
        source_projects = sorted(
            SimulationJob._unique_paths(source_projects),
            key=lambda path: len(str(path)),
            reverse=True,
        )
        for source_project in source_projects:
            data = SimulationJob._map_payload_paths(
                data,
                source_project=source_project,
                target_project=target_project,
            )
        return data

    @staticmethod
    def _assert_remote_payload_has_no_local_roots(
        payload: Mapping[str, Any],
        *,
        source_projects: Iterable[Path],
        target_project: Path,
        payload_name: str,
    ) -> None:
        text = json.dumps(payload, cls=CustomJSONEncoder)
        target_text = str(target_project)
        remaining = [
            str(path)
            for path in SimulationJob._unique_paths(source_projects)
            if str(path) != target_text and str(path) in text
        ]
        if not remaining:
            return
        compact = ", ".join(remaining[:3])
        if len(remaining) > 3:
            compact = f"{compact}, ..."
        raise ValueError(
            f"Remote-staged {payload_name} still contains local project path(s): "
            f"{compact}. The job was not submitted because the solver would try "
            "to open local files on the remote site."
        )

    def save(self):
        self.simulation.save()
        file = self._local_path / f"{self.name}.json"
        self._file = file
        data = self.to_fs(project_relative=True)
        data["result_path"] = str(self._result_path.relative_to(self.project_path))
        self._write_json_file(file, data)
        return file

    @staticmethod
    def _payload_for_layout(
        payload: Mapping[str, Any],
        *,
        source: JobLayout,
        target: JobLayout,
        source_projects: Iterable[Path] = (),
    ) -> Dict[str, Any]:
        data = SimulationJob._map_payload_project_roots(
            payload,
            source_projects=[source.project, *source_projects],
            target_project=target.project,
        )
        data["project_path"] = str(target.project)
        data["simulation"] = str(target.simulation_file)
        data["result_path"] = str(target.result_dir)
        return data

    def _saved_layout(self) -> JobLayout:
        if self._file is None:
            self.save()
        with open(self._file, "r") as f:
            payload = json.load(f)
        return JobLayout.from_payload(payload, job_file=self._file)

    def save_for_remote(self, site: str, remote_project: Union[Path, str]):
        """Stage a remote job JSON without replacing the local job definition."""

        local_file = self.save()
        with open(local_file, "r") as f:
            payload = json.load(f)
        local_layout = JobLayout.from_payload(payload, job_file=local_file)
        remote_layout = local_layout.with_project(remote_project)
        source_projects = self._remote_source_projects(local_layout, payload)
        data = self._payload_for_layout(
            payload,
            source=local_layout,
            target=remote_layout,
            source_projects=source_projects,
        )
        self._assert_remote_payload_has_no_local_roots(
            data,
            source_projects=source_projects,
            target_project=remote_layout.project,
            payload_name="job JSON",
        )

        stage_dir = self._result_path / "_fs_run" / "remote" / site
        staged_file = stage_dir / Path(local_file).name
        self._write_json_file(staged_file, data)
        return staged_file, remote_layout.job_file

    def save_simulation_for_remote(self, site: str, remote_project: Union[Path, str]):
        """Stage this job's simulation JSON for a remote project layout."""

        self.save()
        local_layout = self._saved_layout()
        remote_layout = local_layout.with_project(remote_project)
        with open(local_layout.simulation_file, "r") as f:
            data = json.load(f)
        source_projects = self._remote_source_projects(local_layout, data)
        data = self._map_payload_project_roots(
            data,
            source_projects=source_projects,
            target_project=remote_layout.project,
        )
        data["project_path"] = str(remote_layout.project)
        self._assert_remote_payload_has_no_local_roots(
            data,
            source_projects=source_projects,
            target_project=remote_layout.project,
            payload_name="simulation JSON",
        )
        try:
            staged_relpath = remote_layout.simulation_file.relative_to(
                remote_layout.project
            )
        except ValueError:
            staged_relpath = Path(remote_layout.simulation_file.name)

        staged_file = self._result_path / "_fs_run" / "remote" / site / staged_relpath
        self._write_json_file(staged_file, data)
        return staged_file, remote_layout.simulation_file

    def remote_input_files(self, remote_project: Union[Path, str]) -> List[tuple]:
        """Return local input files that must accompany the remote simulation JSON."""

        remote_project = Path(remote_project)
        files = []
        seen = set()
        local_layout = self._saved_layout()
        payloads = []

        def add_pair(pair: Optional[tuple]) -> None:
            if pair is None:
                return
            local, remote = pair
            key = (Path(local), Path(remote))
            if key in seen:
                return
            seen.add(key)
            files.append(pair)

        mesh = getattr(self.simulation, "mesh", None)
        mesh_file = getattr(mesh, "file", None)
        if mesh_file is not None:
            payloads.append(mesh_file)

        for payload_file in [local_layout.job_file, local_layout.simulation_file]:
            if not payload_file.exists():
                continue
            with open(payload_file, "r") as f:
                payloads.append(json.load(f))

        source_projects = self._remote_source_projects(local_layout, *payloads)

        if mesh_file is not None:
            add_pair(
                self._remote_project_file_pair(
                    mesh_file,
                    source_project=local_layout.project,
                    remote_project=remote_project,
                    source_projects=source_projects,
                )
            )

        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            for file_ref in self._iter_file_references(payload):
                add_pair(
                    self._remote_project_file_pair(
                        file_ref,
                        source_project=local_layout.project,
                        remote_project=remote_project,
                        source_projects=source_projects,
                    )
                )
        return files

    @staticmethod
    def _iter_file_references(value: Any) -> Iterable[str]:
        if isinstance(value, Mapping):
            file_ref = value.get("file")
            if isinstance(file_ref, (str, Path)):
                yield SimulationJob._strip_file_locator(file_ref)
            for item in value.values():
                yield from SimulationJob._iter_file_references(item)
        elif isinstance(value, list):
            for item in value:
                yield from SimulationJob._iter_file_references(item)

    @staticmethod
    def _strip_file_locator(value: Union[str, Path]) -> str:
        text = str(value)
        if ":" not in text:
            return text
        file_part, _ = text.split(":", 1)
        if Path(file_part).suffix:
            return file_part
        return text

    def _remote_project_file_pair(
        self,
        value: Union[str, Path],
        *,
        source_project: Path,
        remote_project: Path,
        source_projects: Iterable[Path] = (),
    ) -> Optional[tuple]:
        path = Path(value)
        source_roots = self._unique_paths([source_project, *source_projects])
        relative: Optional[Path] = None
        if path.is_absolute():
            for root in source_roots:
                try:
                    relative = path.relative_to(root)
                    break
                except ValueError:
                    continue
            if relative is None:
                return None
            local_candidates = [root / relative for root in source_roots]
            if path not in local_candidates:
                local_candidates.append(path)
        else:
            relative = path
            local_candidates = [root / relative for root in source_roots]
        for local in local_candidates:
            if local.exists():
                return local, remote_project / relative
        return None

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

    # TODO: note that for now these will always be local; will be troublesome for remote simulation
    @property
    def _result_path(self):
        """Path where solver results will be stored."""
        if self._file is None:
            return self._local_path / "results"
        return self._file.parent / "results"

    @property
    def project_path(self):
        return self._project_path()

    @property
    def n_tasks(self):
        return len(self.f_list)

    @property
    def trace_manifest(self) -> TraceManifest:
        """Typed description of trace files that should be produced by this job."""

        return TraceManifest.from_job(self)

    @property
    def wavefield_manifest(self) -> TraceManifest:
        """Typed description of grid-backed wavefield files for this job."""

        return TraceManifest.from_job(self, output=self.wavefield_trace_outputs)

    @property
    def paraview_outputs(self) -> dict:
        """Lists ParaView outputs.

        Returns:
            dict: Dictionary containing:
                - ParaView: ParaView outputs
        """
        return {out.name: out.path for out in self.outputs.paraview}

    @property
    def trace_path(self) -> Path:
        """Lists receiver trace groups.

        Returns:
            - traces: Receiver traces
        """
        return self.trace_outputs.path

    @property
    def trace_outputs(self) -> TraceOutputSpec:
        """Lists receiver trace groups.

        Returns:
            - traces: Receiver traces
        """
        sim = self.simulation
        groups = []
        components = []
        sources = []

        for group in sim.acquisition.receiver_groups:
            groups.append(group.name)
            for component in group.device.components:
                components.append(f"{group.name}:{component.name}")

        for isrc, _sgroup in enumerate(sim.acquisition.source_groups):
            sources.append(f"{isrc + 1}")

        return TraceOutputSpec(
            path=self._result_path / self.outputs.traces.path,
            frequencies=self.f_list,
            groups=groups,
            components=components,
            sources=sources,
        )

    @property
    def traces(self) -> TraceOutputHandle:
        """Trace output handle for opening or inspecting this job's traces."""
        return TraceOutputHandle(self)

    @property
    def wavefield_trace_outputs(self) -> TraceOutputSpec:
        """Lists first-class wavefield output groups."""

        groups = []
        components = []
        sources = set()
        wavefields = {}
        output_paths = set()

        for out in self.outputs.wavefields:
            if out.grid is None:
                raise ValueError("WavefieldOutput requires a grid")
            output_paths.add(str(out.path))
            fields = out.fields if out.fields is not None else ["primary"]
            component_names = out.component_names
            component_specs = out.component_payloads()
            components_for_group = [
                f"{out.name}:{component_name}" for component_name in component_names
            ]
            source_ids = (
                [str(source) for source in out.sources]
                if out.sources is not None
                else [
                    f"{isrc + 1}"
                    for isrc, _sgroup in enumerate(
                        self.simulation.acquisition.source_groups
                    )
                ]
            )
            groups.append(out.name)
            components.extend(components_for_group)
            sources.update(source_ids)
            wavefield = {
                "name": out.name,
                "path": str(self._result_path / out.path),
                "fields": list(fields),
                "components": components_for_group,
                "component_names": component_names,
                "component_specs": component_specs,
                "sources": source_ids,
                "grid": copy.deepcopy(out.grid),
            }
            if out.device is not None:
                wavefield["device"] = out.device.to_fs()
            wavefields[out.name] = wavefield

        if len(output_paths) > 1:
            raise ValueError(
                "job.wavefields.open() requires all wavefield outputs to share "
                "one path"
            )
        output_path = (
            Path(next(iter(output_paths)))
            if output_paths
            else (
                Path(self.outputs.wavefields[0].path)
                if self.outputs.wavefields
                else Path("wavefields")
            )
        )

        return TraceOutputSpec(
            path=self._result_path / output_path,
            frequencies=self.f_list,
            groups=groups,
            components=components,
            sources=sorted(sources, key=lambda value: int(value)),
            wavefields=wavefields,
        )

    @property
    def wavefields(self) -> WavefieldOutputHandle:
        """Wavefield output handle for opening grid-backed wavefield outputs."""

        return WavefieldOutputHandle(self)

    @property
    def wavefield_outputs(self) -> dict:
        """Lists wavefield outputs.

        Returns:
            - wavefields: Wavefield outputs
        """
        wave_out = {}
        for out in self.outputs.wavefields:
            if out.grid is None:
                raise ValueError("WavefieldOutput requires a grid")
            fields = out.fields if out.fields is not None else ["primary"]
            component_names = out.component_names
            component_specs = out.component_payloads()
            components = [
                f"{out.name}:{component_name}" for component_name in component_names
            ]
            sources = (
                [str(source) for source in out.sources]
                if out.sources is not None
                else [
                    f"{isrc + 1}"
                    for isrc, _source_group in enumerate(
                        self.simulation.acquisition.source_groups
                    )
                ]
            )
            wave_out[out.name] = {
                "domain": (self.__class__.__name__,),
                "path": str(self._result_path / out.path),
                "frequencies": self.f_list,
                "grid": copy.deepcopy(out.grid),
                "fields": list(fields),
                "components": components,
                "component_names": component_names,
                "component_specs": component_specs,
                "sources": sources,
            }
            if out.device is not None:
                wave_out[out.name]["device"] = out.device.to_fs()
        return wave_out

    def _remote_path(self, work_dir: Union[Path, str]):
        """Get remote job path."""
        work_dir = Path(work_dir)
        return work_dir / "jobs" / self.simulation.name / self.name


@register_class
class FrequencyDomainJob(SimulationJob):
    """Forward job at explicitly requested real or complex frequencies."""

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: List[Union[float, complex]],
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
    ):
        """Create a frequency-domain forward job."""

        workflow = "forward"
        frequencies = np.asarray(f_list)
        if np.iscomplexobj(frequencies):
            frequencies = np.asarray([f.real - 1j * abs(f.imag) for f in frequencies])
        super().__init__(
            name,
            simulation,
            workflow,
            frequencies.tolist(),
            JobOutputs(outputs),
        )

    @classmethod
    def from_fs(
        cls,
        d: dict,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ):
        """Deserialize a frequency-domain job from a solver payload."""

        sim = SimulationJob._load_simulation_for_job(
            d["simulation"],
            base_path=base_path,
            project_path=project_path or d.get("project_path"),
            source_project=d.get("project_path"),
        )
        f_list = cls._decode_frequencies(d["f_list"])
        job = cls(
            name=d["name"],
            simulation=sim,
            f_list=f_list,
            outputs=JobOutputs.from_fs(d.get("Outputs")),
        )
        job._job_id = d.get("job_id")
        return job


@register_class
class TimeDomainJob(SimulationJob):
    """Forward job represented by a uniform frequency sweep for time synthesis."""

    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_max: float,
        f_min: float = 0.0,
        damping_factor: Optional[float] = None,
        laplace: Optional[float] = None,
        df: Optional[float] = None,
        T_max: Optional[float] = None,
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
    ):
        """Create a time-domain job from ``f_min``/``f_max`` and ``df`` or ``T_max``."""

        if damping_factor is not None and laplace is not None:
            raise ValueError("Specify only one of damping_factor or laplace")
        if df is None and T_max is None:
            raise ValueError("TimeDomainJob requires either df or T_max")
        if T_max is not None:
            if df is not None:
                raise ValueError("Specify only one of df or T_max")
            if T_max <= 0:
                raise ValueError("T_max must be positive")
            df = 1.0 / T_max
        if df <= 0:
            raise ValueError("df must be positive")
        if f_max <= f_min:
            raise ValueError("f_max must be greater than f_min")

        period = 1.0 / df
        if damping_factor is not None:
            if damping_factor < 1.0:
                raise ValueError("damping_factor must be greater than or equal to 1")
            laplace = -np.log(float(damping_factor)) / (2.0 * np.pi * period)

        if f_min == 0.0:
            f_min = f_min + df
        f_list = np.arange(f_min, f_max + df / 2, df)

        laplace = -abs(float(laplace or 0.0))
        if laplace != 0.0:
            f_list = f_list + 1j * laplace

        workflow = "forward"
        super().__init__(name, simulation, workflow, f_list, JobOutputs(outputs))

    @classmethod
    def from_fs(
        cls,
        d: dict,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ):
        """Deserialize a time-domain job from a solver payload."""

        f_list = cls._decode_frequencies(d["f_list"])
        if f_list.size < 2:
            raise ValueError("TimeDomainJob requires at least two frequencies")

        f_min = float(np.real(f_list[0]))
        f_max = float(np.real(f_list[-1]))
        df = float(np.real(f_list[1] - f_list[0]))
        laplace = float(np.imag(f_list[0]))
        expected = np.arange(f_min, f_max + df / 2, df)
        if laplace != 0.0:
            expected = expected + 1j * laplace
        if not np.allclose(f_list, expected):
            raise ValueError("Frequency list does not appear to be uniform")

        sim = SimulationJob._load_simulation_for_job(
            d["simulation"],
            base_path=base_path,
            project_path=project_path or d.get("project_path"),
            source_project=d.get("project_path"),
        )
        job = cls(
            name=d["name"],
            simulation=sim,
            f_min=f_min,
            f_max=f_max,
            df=df,
            laplace=laplace,
            outputs=JobOutputs.from_fs(d.get("Outputs")),
        )
        job._job_id = d.get("job_id")
        return job
