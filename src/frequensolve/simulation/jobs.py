import hashlib
import json
import shutil
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import blake3
import numpy as np

from frequensolve.simulation.artifacts import (
    RunMetadata,
    TraceManifest,
    TraceOutputHandle,
    TraceOutputSpec,
)
from frequensolve.simulation.outputs import (
    JobOutputs,
    Output,
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
    outputs: JobOutputs = field(default_factory=JobOutputs)
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
        if not isinstance(self.outputs, JobOutputs):
            self.outputs = JobOutputs(self.outputs)

    def __iadd__(self, output: Union[Output, Iterable[Output]]) -> "SimulationJob":
        self.outputs += output
        return self

    def add_output(self, output: Union[Output, Iterable[Output]]) -> "SimulationJob":
        self += output
        return self

    def validate_outputs(self) -> None:
        if self.outputs.paraview and len(self.f_list) != 1:
            raise ValueError(
                "ParaView outputs currently require a single-frequency job. "
                "Create one FrequencyDomainJob per plotted frequency."
            )

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
            "simulation": {
                "path": str(Path(self.simulation._file).resolve()),
                "hash": simulation_hash,
            },
        }

    def fingerprint(self) -> str:
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
            "simulation": {
                "path": str(Path(self.simulation._file).resolve()),
                "hash": simulation_hash,
            },
            "frequency": self._canonical_frequency_value(self.f_list[task - 1]),
        }

    def task_fingerprint(self, task: int) -> str:
        return self._hash_payload(self.task_fingerprint_payload(task))

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
        return self.trace_manifest.complete

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
        if task not in packed_frequencies:
            return False
        return np.isclose(
            packed_frequencies[task],
            self._real_frequency_value(self.f_list[task - 1]),
            rtol=0.0,
            atol=1.0e-9,
        )

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
        packed_file = manifest.packed_file
        ordered_files = list(manifest.files)
        for task, file in enumerate(ordered_files, start=1):
            trace_file = file if file.exists() else self._legacy_trace_file(file)
            packed_task_exists = self._packed_trace_has_current_task(
                task,
                manifest=manifest,
            )
            trace_exists = trace_file.exists() or packed_task_exists
            current = self.is_task_current(task, state=state)
            rows[task] = {
                "task": task,
                "frequency": manifest.frequencies.get(task),
                "status": "succeeded" if current else "not_run",
                "trace_file": (
                    trace_file
                    if trace_file.exists()
                    else (
                        packed_file
                        if packed_task_exists and packed_file is not None
                        else file
                    )
                ),
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
            if status == "failed":
                row["status"] = "failed"
            elif status == "succeeded" and row["current"] and row["status"] != "failed":
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
        expected = files[task - 1]
        state = self.run_state() if state is None else state
        expected_fingerprint = self.task_fingerprint(task)
        full_run_matches = state.get("fingerprint") == self.fingerprint() and state.get(
            "status"
        ) in {"completed", "skipped"}
        expected_exists = self._trace_file_exists(expected)
        packed_task_exists = self._packed_trace_has_current_task(task)
        if not expected_exists:
            if not packed_task_exists or not full_run_matches:
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
                return True
            return self._trace_file_exists(stored_path) or (
                packed_task_exists and full_run_matches
            )
        return packed_task_exists and full_run_matches

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
            for path in {files[task - 1], self._legacy_trace_file(files[task - 1])}:
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
            manifest = self.trace_manifest
            for path in [manifest.packed_file, manifest.output_path / "manifest.json"]:
                if path is None:
                    continue
                try:
                    path.unlink()
                    removed_stale_outputs = True
                except FileNotFoundError:
                    pass
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
        packed_file = manifest.packed_file
        for task, path in enumerate(self.expected_trace_files(), start=1):
            if not self.is_task_current(task, state=state):
                continue
            existing = path if path.exists() else self._legacy_trace_file(path)
            if existing.exists():
                file_path = existing
            elif (
                self._packed_trace_has_current_task(task, manifest=manifest)
                and packed_file is not None
            ):
                file_path = packed_file
            else:
                file_path = path
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
        if len(self.current_tasks()) == self.n_tasks:
            return True
        if state.get("fingerprint") != self.fingerprint():
            return False
        if state.get("status") not in {"completed", "skipped"}:
            return False
        return True

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
        packed_file = manifest.packed_file
        for task, path in enumerate(self.expected_trace_files(), start=1):
            existing = path if path.exists() else self._legacy_trace_file(path)
            packed_task_exists = self._packed_trace_has_current_task(
                task,
                manifest=manifest,
            )
            if existing.exists():
                file_path = existing
            elif packed_task_exists and packed_file is not None:
                file_path = packed_file
            else:
                file_path = path
            stored_path = self._stored_trace_path(file_path)
            exists = file_path.exists()
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

            duration = self._record_duration_seconds(result)
            row = {
                "task": task,
                "frequency": self._canonical_frequency_value(self.f_list[task - 1]),
                "status": task_status,
                "fingerprint": self.task_fingerprint(task),
                "path": stored_path,
                "exists": exists,
            }
            if duration is not None:
                row["duration_seconds"] = duration
            core_count = self._record_core_count(result)
            if core_count is not None:
                row["core_count"] = core_count
            for key in ("n_ranks", "ranks", "threads_per_rank", "n_threads", "threads"):
                if key in result:
                    row[key] = result[key]
            task_rows.append(row)

        payload = {
            "schema": "frequensolve-python-run-1",
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": self.fingerprint(),
            "fingerprint_payload": self.fingerprint_payload(),
            "tasks": task_rows,
            "outputs": {"traces": files},
        }
        if task_results:
            payload["task_results"] = task_results
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
        receivers = sim.acquisition.receiver_groups

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
    def wavefield_outputs(self) -> dict:
        """Lists wavefield outputs.

        Returns:
            - wavefields: Wavefield outputs
        """
        receivers = self.simulation.acquisition.receiver_groups
        wave_out = {}
        for out in self.outputs.wavefields:
            wave_out[out.name] = {
                "domain": (self.__class__.__name__,),
                "path": out.path,
                "frequencies": self.f_list,
                "grid": out.grid.to_fs() if out.grid is not None else None,
                "components": [],
                "sources": [],
            }

            for group in receivers:
                for component in group.device.components:
                    wave_out[out.name]["components"].append(
                        f"{group.name}:{component.name}"
                    )

            for isrc, _source_group in enumerate(
                self.simulation.acquisition.source_groups
            ):
                wave_out[out.name]["sources"].append(f"{isrc + 1}")
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
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
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
            JobOutputs(outputs),
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
            outputs=JobOutputs.from_fs(d.get("Outputs")),
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
        outputs: Optional[Union[Output, Iterable[Output], JobOutputs]] = None,
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
        super().__init__(name, simulation, workflow, f_list, JobOutputs(outputs))

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
            outputs=JobOutputs.from_fs(d.get("Outputs")),
        )
        job._job_id = d.get("job_id")
        return job
