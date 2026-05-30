"""Job serialization and fingerprinting helpers.

The mixin in this module loads concrete job classes from saved JSON payloads,
emits solver-contract job files, and computes stable whole-job and per-task
fingerprints used to decide whether outputs are current.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import blake3
import numpy as np

from frequensolve.simulation.simulation import BaseSimulation, CustomJSONEncoder
from frequensolve.util.class_registry import class_registry

if TYPE_CHECKING:
    from frequensolve.simulation.jobs.base import BaseJob


class JobSerializationMixin:
    """Serialization and fingerprinting helpers shared by job classes.

    The mixin expects the concrete job to provide ``from_fs`` constructors,
    ``to_fs`` fields such as ``name`` and ``f_list``, and project path helpers
    from ``BaseJob``.
    """

    @classmethod
    def load(
        cls,
        path: Union[Path, str, "BaseJob"],
        *,
        project_path: Optional[Union[str, Path]] = None,
    ):
        """Load a job from a JSON file, job directory, or existing job object.

        Args:
            path: Job JSON path, directory containing one job JSON, or job-like
                object with a ``job_file`` attribute.
            project_path: Optional project root used to resolve or remap
                relative simulation paths.

        Returns:
            Concrete ``BaseJob`` subclass reconstructed from the JSON payload.

        Raises:
            FileNotFoundError: If a directory contains no job JSON file.
            ValueError: If the JSON cannot be loaded or the directory contains
                multiple ambiguous job JSON files.
        """

        if not isinstance(path, (str, Path)) and hasattr(path, "job_file"):
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

    def load_saved(self) -> "BaseJob":
        """Load the saved job JSON corresponding to this job object.

        Returns:
            Fresh job instance loaded from ``self.job_file``.
        """

        return self.__class__.load(self.job_file)

    @classmethod
    def from_fs(
        cls,
        d: dict,
        base_path: Optional[Union[str, Path]] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "BaseJob":
        """Deserialize a job payload using the registered concrete job class.

        Args:
            d: Serialized job payload containing ``_type``.
            base_path: Optional directory used to resolve relative paths.
            project_path: Optional project root used to resolve project-relative
                paths.

        Returns:
            Concrete ``BaseJob`` subclass named by ``_type``.

        Raises:
            ValueError: If the payload names an unknown job class.
        """

        data = dict(d)
        class_name = data.get("_type")
        if class_name not in class_registry:
            import frequensolve.simulation.jobs.forward  # noqa: F401
            import frequensolve.simulation.jobs.imaging  # noqa: F401

        if class_name not in class_registry:
            raise ValueError(f"Unknown job class: {class_name}")
        job_class = class_registry[class_name]
        return job_class.from_fs(
            data,
            base_path=base_path,
            project_path=project_path,
        )

    def to_fs(self, *, project_relative: bool = False) -> Dict[str, Any]:
        """Serialize this job to the job JSON contract.

        Args:
            project_relative: When true, emit simulation and result paths
                relative to the project root where possible.

        Returns:
            JSON-compatible job payload.

        Raises:
            ValueError: If output validation fails or the simulation is not
                attached to a project.
        """

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
        payload.update(self._wavenumber_payload())
        if self._job_id is not None:
            payload["job_id"] = self._job_id
        return payload

    def fingerprint_payload(self) -> Dict[str, Any]:
        """Return the stable payload used to determine whole-job freshness.

        Returns:
            JSON-compatible payload hashed by :meth:`fingerprint`.

        Raises:
            ValueError: If the simulation has not been saved.
        """

        job_data = self.to_fs()
        if self.simulation._file is None:
            raise ValueError("Simulation must be saved before fingerprinting a job")
        simulation_hash = self._hash_json_file(self.simulation._file)
        job_fingerprint = {
            "_type": job_data["_type"],
            "workflow": job_data["workflow"],
            "f_list": job_data["f_list"],
            "Outputs": job_data["Outputs"],
        }
        job_fingerprint.update(self._wavenumber_payload())
        return {
            "schema": "frequensolve-job-fingerprint-1",
            "job": job_fingerprint,
            "simulation": {"hash": simulation_hash},
        }

    def fingerprint(self) -> str:
        """Return the stable hash identifying this job definition.

        Returns:
            ``blake3:...`` hash string for the whole-job fingerprint payload.
        """

        return self._hash_payload(self.fingerprint_payload())

    def task_fingerprint_payload(self, task: int) -> Dict[str, Any]:
        """Return the rerun fingerprint payload for one frequency task.

        ``task`` is one-based to match the solver task IDs and trace filenames.
        Unlike the whole-job fingerprint, this intentionally excludes the full
        frequency list.  Changing ``f_max`` or ``df`` should only invalidate
        frequencies whose own value changed.

        Args:
            task: One-based solver task number.

        Returns:
            JSON-compatible payload hashed by :meth:`task_fingerprint`.

        Raises:
            IndexError: If ``task`` is outside the job's task range.
            ValueError: If the simulation has not been saved.
        """

        if task < 1 or task > self.n_tasks:
            raise IndexError(f"Task {task} is outside 1..{self.n_tasks}")
        job_data = self.to_fs()
        if self.simulation._file is None:
            raise ValueError("Simulation must be saved before fingerprinting a task")
        simulation_hash = self._hash_json_file(self.simulation._file)
        job_fingerprint = {
            "_type": job_data["_type"],
            "workflow": job_data["workflow"],
            "Outputs": job_data["Outputs"],
        }
        job_fingerprint.update(self._wavenumber_payload())
        return {
            "schema": "frequensolve-job-task-fingerprint-1",
            "job": job_fingerprint,
            "simulation": {"hash": simulation_hash},
            "frequency": self._canonical_frequency_value(self.f_list[task - 1]),
        }

    def task_fingerprint(self, task: int) -> str:
        """Return the stable hash for one one-based frequency task.

        Args:
            task: One-based solver task number.

        Returns:
            ``blake3:...`` hash string for the task fingerprint payload.
        """

        return self._hash_payload(self.task_fingerprint_payload(task))

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
        sim_file = JobSerializationMixin._resolve_simulation_file(
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
            sim._attach_project_path(project, Path("simulations"))
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

    def _wavenumber_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.k_list is not None:
            payload["k_list"] = list(self.k_list)
        if self.k_weights is not None:
            payload["k_weights"] = list(self.k_weights)
        if self.k_units is not None:
            payload["k_units"] = self.k_units
        return payload

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
        return JobSerializationMixin._hash_payload(payload)

    @staticmethod
    def _sha256_file(path: Union[str, Path]) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

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

    @staticmethod
    def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(payload, cls=CustomJSONEncoder, indent=3))
        tmp.replace(path)
