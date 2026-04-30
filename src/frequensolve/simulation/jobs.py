import hashlib
import json
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import blake3
import numpy as np

from frequensolve.simulation.artifacts import (
    RunMetadata,
    TraceManifest,
    TraceOutputHandle,
    TraceOutputSpec,
)
from frequensolve.simulation.simulation import BaseSimulation, CustomJSONEncoder
from frequensolve.util.class_registry import class_registry, register_class

__all__ = ["SimulationJob", "FrequencyDomainJob", "TimeDomainJob"]


@register_class
@dataclass
class SimulationJob(ABC):
    name: str
    simulation: BaseSimulation
    workflow: str
    f_list: List[Union[float, complex]]
    _file: Optional[Path] = None
    _job_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SimulationJob requires a non-empty name")
        if self.simulation is None:
            raise ValueError("SimulationJob requires a simulation")
        frequencies = np.asarray(self.f_list)
        if frequencies.size == 0:
            raise ValueError("SimulationJob requires at least one frequency")
        self.f_list = frequencies.tolist()

    @classmethod
    def from_fs(
        cls, d: dict, base_path: Optional[Union[str, Path]] = None
    ) -> "SimulationJob":
        data = dict(d)
        class_name = data.get("_type")
        if class_name not in class_registry:
            raise ValueError(f"Unknown job class: {class_name}")
        job_class = class_registry[class_name]
        return job_class.from_fs(data, base_path=base_path)

    @classmethod
    def load(cls, path: Union[Path, str]):
        path = Path(path).resolve()
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load job JSON {path}: {e}") from e

        job = cls.from_fs(data, base_path=path.parent)
        job._file = path
        job._job_id = data.get("job_id")
        return job

    @staticmethod
    def _project_root_from_job_path(path: Path) -> Optional[Path]:
        parts = path.resolve().parts
        if "jobs" not in parts:
            return None
        index = parts.index("jobs")
        if index == 0:
            return None
        return Path(*parts[:index])

    @staticmethod
    def _resolve_project_relative_path(
        path: Union[str, Path],
        *,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        path = Path(path)
        if path.is_absolute():
            return path
        if project_path is not None:
            candidate = Path(project_path) / path
            if candidate.exists():
                return candidate
        if base_path is not None:
            base = Path(base_path).resolve()
            project_root = SimulationJob._project_root_from_job_path(base)
            if project_root is not None:
                candidate = project_root / path
                if candidate.exists():
                    return candidate
            candidate = base / path
            if candidate.exists():
                return candidate
        return path

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
        f_list = self._encoded_frequencies()
        payload = {
            "schema": "frequensolve-job-1",
            "_type": self.__class__.__name__,
            "name": self.name,
            "project_path": str(self._project_path()),
            "simulation": self._simulation_path(project_relative=project_relative),
            "workflow": self.workflow,
            "f_list": f_list,
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
            },
            "simulation": {
                "path": str(Path(self.simulation._file).resolve()),
                "hash": simulation_hash,
            },
        }

    def fingerprint(self) -> str:
        return self._hash_payload(self.fingerprint_payload())

    @property
    def run_state_file(self) -> Path:
        return self._result_path / "_fs_python_run.json"

    @property
    def run_metadata(self) -> RunMetadata:
        return RunMetadata.read(self._result_path)

    def expected_trace_files(self) -> List[Path]:
        return list(self.trace_manifest.files)

    @staticmethod
    def _legacy_trace_file(path: Path) -> Path:
        return path.with_name(path.name.replace("traces_", "receivers_", 1))

    @classmethod
    def _trace_file_exists(cls, path: Path) -> bool:
        return path.exists() or cls._legacy_trace_file(path).exists()

    def trace_outputs_exist(self) -> bool:
        files = self.expected_trace_files()
        return bool(files) and all(self._trace_file_exists(path) for path in files)

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
        if status in {"success", "succeeded", "complete", "completed", "done"}:
            return "succeeded"
        if status in {"failed", "failure", "error", "timeout", "cancelled", "killed"}:
            return "failed"
        if status in {"pending", "queued", "submitted", "running"}:
            return "not_run"
        return None

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

    def _task_records(self) -> List[Mapping[str, Any]]:
        metadata = self.run_metadata
        state = self.run_state()
        records: List[Mapping[str, Any]] = []
        for source in (state or metadata.state, metadata.timings, metadata.manifest):
            if not isinstance(source, Mapping):
                continue
            records.extend(self._as_records(source.get("tasks")))
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

        Status is inferred from expected trace files plus any Python/Sauce run
        metadata available beside the results. Task numbers are one-based to
        match solver task IDs and trace filenames.
        """

        manifest = self.trace_manifest
        rows: Dict[int, Dict[str, Any]] = {}
        ordered_files = list(manifest.files)
        for task, file in enumerate(ordered_files, start=1):
            trace_file = file if file.exists() else self._legacy_trace_file(file)
            trace_exists = trace_file.exists()
            rows[task] = {
                "task": task,
                "frequency": manifest.frequencies.get(task),
                "status": "succeeded" if trace_exists else "not_run",
                "trace_file": trace_file if trace_exists else file,
                "trace_exists": trace_exists,
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
            if status == "failed":
                row["status"] = "failed"
            elif status == "succeeded" and row["status"] != "failed":
                row["status"] = "succeeded"

        return [rows[task] for task in sorted(rows)]

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

        timings = []
        for row in self.frequency_status():
            duration = row.get("duration_seconds")
            if duration is None:
                continue
            timings.append(
                {
                    "task": row["task"],
                    "frequency": row["frequency"],
                    "duration_seconds": duration,
                    "status": row["status"],
                    "trace_file": row["trace_file"],
                }
            )
        return timings

    def plot_task_timings(
        self,
        ax: Optional[Any] = None,
        *,
        show: bool = False,
        title: Optional[str] = None,
        **bar_kwargs: Any,
    ) -> Any:
        """Plot per-frequency task runtimes and return the matplotlib axes."""

        timings = self.task_timings()
        if not timings:
            raise ValueError(
                "No per-task timings were found in run metadata for this job"
            )

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        colors = {"succeeded": "#2e7d32", "failed": "#c62828", "not_run": "#757575"}
        tasks = [row["task"] for row in timings]
        values = [row["duration_seconds"] for row in timings]
        bar_kwargs.setdefault(
            "color", [colors.get(row["status"], "#546e7a") for row in timings]
        )
        ax.bar(tasks, values, **bar_kwargs)
        ax.set_xlabel("Frequency task")
        ax.set_ylabel("Runtime (s)")
        ax.set_title(title or f"{self.name} task timings")
        ax.set_xticks(tasks)
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

    def is_run_current(self) -> bool:
        if not self.trace_outputs_exist():
            return False

        metadata = self.run_metadata
        if metadata.manifest:
            if metadata.manifest.get("exit_status") != "success":
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
        return True

    def write_run_state(self, status: str = "completed", **extra) -> Path:
        self._result_path.mkdir(parents=True, exist_ok=True)
        files = []
        for path in self.expected_trace_files():
            existing = path if path.exists() else self._legacy_trace_file(path)
            file_path = existing if existing.exists() else path
            try:
                stored_path = str(file_path.resolve().relative_to(self.project_path))
            except Exception:
                stored_path = str(file_path)
            files.append({"path": stored_path, "exists": file_path.exists()})

        payload = {
            "schema": "frequensolve-python-run-1",
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": self.fingerprint(),
            "fingerprint_payload": self.fingerprint_payload(),
            "outputs": {"traces": files},
        }
        payload.update(extra)
        self._write_json_file(self.run_state_file, payload)
        return self.run_state_file

    @staticmethod
    def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(payload, cls=CustomJSONEncoder, indent=3))
        tmp.replace(path)

    def save(self):
        self.simulation.save()
        file = self._local_path / f"{self.name}.json"
        self._file = file
        data = self.to_fs(project_relative=True)
        data["result_path"] = str(self._result_path.relative_to(self.project_path))
        self._write_json_file(file, data)
        return file

    def save_for_remote(self, site: str, remote_project: Union[Path, str]):
        """Save the job for remote simulation.

        Args:
            site (str): The site to save the job for.
            remote_project (Union[Path, str]): The remote project to save the job to.
        """
        self.simulation.save()
        remote_project = Path(remote_project)
        remote_path = remote_project / "jobs" / self.simulation.name / self.name
        local_path = self._local_path
        local_path.mkdir(parents=True, exist_ok=True)
        data = self.to_fs(project_relative=False)
        local_project = f"{self.simulation._proj_path}"

        # Recursively process the data dictionary
        def replace_path(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    replace_path(value)
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            replace_path(item)
                if isinstance(value, Path):
                    if local_project in str(value):
                        d[key] = str(value).replace(local_project, str(remote_project))
                if isinstance(value, str):
                    if local_project in value:
                        d[key] = value.replace(local_project, str(remote_project))
            return d

        local_file = local_path / f"{self.name}.json"
        remote_file = remote_path / f"{self.name}.json"

        data = replace_path(data)
        self._file = local_file
        data["result_path"] = str(self._result_path.relative_to(self.project_path))
        self._write_json_file(local_file, data)
        return local_file, remote_file

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
    def paraview_outputs(self) -> dict:
        """Lists ParaView outputs.

        Returns:
            dict: Dictionary containing:
                - ParaView: ParaView outputs
        """
        sim = self.simulation

        sim_file = sim._file
        with open(sim_file, "r") as f:
            sim_data = json.load(f)

        pv_out = {}
        for out in sim_data["Outputs"]["ParaView"]:
            pv_out[out["name"]] = out["path"]

        return pv_out

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
        receivers = sim.acquisition.receiver_groups

        sim_file = sim._file
        with open(sim_file, "r") as f:
            sim_data = json.load(f)

        outputs = sim_data.get("Outputs", {})
        out = outputs.get("traces") or outputs.get("receivers")
        if out is None:
            raise ValueError("Simulation does not request trace output")

        groups = []
        components = []
        sources = []

        for group in receivers:
            groups.append(group.name)
            for component in group.device.components:
                components.append(f"{group.name}:{component.name}")

        for isrc, _sgroup in enumerate(sim.acquisition.source_groups):
            sources.append(f"{isrc + 1}")

        return TraceOutputSpec(
            path=self._result_path / out["path"],
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
    def wavefield_outputs(self) -> dict:
        """Lists wavefield outputs.

        Returns:
            - wavefields: Wavefield outputs
        """
        sim = self.simulation
        receivers = sim.acquisition.receiver_groups

        sim_file = sim._file
        with open(sim_file, "r") as f:
            sim_data = json.load(f)

        outputs = sim_data["Outputs"]["wavefields"]

        wave_out = {}
        for out in outputs:
            wave_out["domain"] = (self.__class__.__name__,)
            wave_out["path"] = out["path"]
            wave_out["frequencies"] = self.f_list
            wave_out["grid"] = out["grid"]
            wave_out["components"] = []
            wave_out["sources"] = []

            for group in receivers:
                for component in group.device.components:
                    wave_out["components"].append(f"{group.name}:{component.name}")

            for isrc, _source_group in enumerate(sim.acquisition.source_groups):
                wave_out["sources"].append(f"{isrc + 1}")
        return wave_out

    def _remote_path(self, work_dir: Union[Path, str]):
        """Get remote job path."""
        work_dir = Path(work_dir)
        return work_dir / "jobs" / self.simulation.name / self.name


@register_class
class FrequencyDomainJob(SimulationJob):
    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_list: List[Union[float, complex]],
    ):
        workflow = "forward"
        frequencies = np.asarray(f_list)
        if np.iscomplexobj(frequencies):
            frequencies = np.asarray([f.real - 1j * abs(f.imag) for f in frequencies])
        super().__init__(
            name,
            simulation,
            workflow,
            frequencies.tolist(),
        )

    @classmethod
    def from_fs(cls, d: dict, base_path: Optional[Union[str, Path]] = None):
        sim_file = SimulationJob._resolve_project_relative_path(
            d["simulation"],
            base_path=base_path,
            project_path=d.get("project_path"),
        )
        sim = BaseSimulation.load(sim_file)
        f_list = cls._decode_frequencies(d["f_list"])
        job = cls(
            name=d["name"],
            simulation=sim,
            f_list=f_list,
        )
        job._job_id = d.get("job_id")
        return job


@register_class
class TimeDomainJob(SimulationJob):
    def __init__(
        self,
        name: str,
        simulation: BaseSimulation,
        f_max: float,
        f_min: float = 0.0,
        s_laplace: float = 0.0,
        df: Optional[float] = None,
        T_max: Optional[float] = None,
    ):
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

        if f_min == 0.0:
            f_min = f_min + df
        f_list = np.arange(f_min, f_max + df / 2, df)

        s_laplace = -abs(s_laplace)
        f_list = f_list + 1j * s_laplace

        workflow = "forward"
        super().__init__(name, simulation, workflow, f_list)

    @classmethod
    def from_fs(cls, d: dict, base_path: Optional[Union[str, Path]] = None):
        f_list = cls._decode_frequencies(d["f_list"])
        if f_list.size < 2:
            raise ValueError("TimeDomainJob requires at least two frequencies")

        f_min = float(np.real(f_list[0]))
        f_max = float(np.real(f_list[-1]))
        df = float(np.real(f_list[1] - f_list[0]))
        s_laplace = float(np.imag(f_list[0]))
        expected = np.arange(f_min, f_max + df / 2, df) + 1j * s_laplace
        if not np.allclose(f_list, expected):
            raise ValueError("Frequency list does not appear to be uniform")

        sim_file = SimulationJob._resolve_project_relative_path(
            d["simulation"],
            base_path=base_path,
            project_path=d.get("project_path"),
        )
        sim = BaseSimulation.load(sim_file)
        job = cls(
            name=d["name"],
            simulation=sim,
            f_min=f_min,
            f_max=f_max,
            df=df,
            s_laplace=s_laplace,
        )
        job._job_id = d.get("job_id")
        return job
