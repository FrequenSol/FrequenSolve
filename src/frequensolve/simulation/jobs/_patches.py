# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Native geometry preparation through the existing job scheduler."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.util.mixins import ExportContext

if TYPE_CHECKING:
    from frequensolve.mesh.patches import PreparedPatchSet
    from frequensolve.orchestrator.sites.base import RunResult

import numpy as np

from frequensolve.simulation.artifact_contract import (
    ArtifactContractError,
    MaterialArtifactRecord,
    OperationResult,
    TaskResult,
    load_operation_result,
)
from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.util.class_registry import register_class


@dataclass(frozen=True)
class PatchRunResult:
    """Composite forward results in stable prepared-patch order."""

    jobs: tuple[BaseJob, ...]
    runs: tuple[RunResult, ...]
    prepared: PreparedPatchSet

    @property
    def successful(self) -> bool:
        return all(run.successful for run in self.runs)

    @property
    def job(self) -> BaseJob:
        if len(self.jobs) != 1:
            raise ValueError("Patch execution used multiple jobs; inspect .jobs")
        return self.jobs[0]

    def raise_for_status(self) -> None:
        for run in self.runs:
            run.raise_for_status()


def _validate_request(
    request: Mapping[str, Any], dimension: int | float | str
) -> dict[str, Any]:
    data = deepcopy(dict(request))
    if set(data) - {"units", "patches", "edge_samples"} or data.get("units") != "m":
        raise ValueError("PatchPreparation requires metre coordinates (units='m')")
    if "edge_samples" in data and not isinstance(data["edge_samples"], bool):
        raise ValueError("PatchPreparation edge_samples must be a boolean")
    names = set()
    for patch in data.get("patches", []):
        if set(patch) - {"name", "roots", "lower", "upper", "padding", "points"}:
            raise ValueError("Unknown patch preparation field")
        name = patch.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Patch names must be nonempty and unique")
        names.add(name)
        padding = patch.get("padding")
        if (
            isinstance(padding, bool)
            or not isinstance(padding, Real)
            or not math.isfinite(padding)
            or padding < 0
        ):
            raise ValueError(
                "Every patch requires finite nonnegative padding in metres"
            )
        patch["padding"] = float(padding)
        if not isinstance(patch.get("points", []), (list, tuple)):
            raise ValueError("Patch points must be a sequence of acquisition records")
        identities = set()
        for point in patch.get("points", []):
            if not isinstance(point, dict):
                raise ValueError("Patch points must contain acquisition records")
            if set(point) - {"kind", "id", "group", "coordinates"}:
                raise ValueError("Unknown patch acquisition point field")
            kind, identity = point.get("kind"), point.get("id")
            if kind not in {"source", "receiver"} or (
                isinstance(identity, bool)
                or not isinstance(identity, Integral)
                or identity <= 0
            ):
                raise ValueError(
                    "Patch points require a source/receiver kind and positive ID"
                )
            group = point.get("group")
            if (kind == "source" and group is not None) or (
                kind == "receiver" and (not isinstance(group, str) or not group.strip())
            ):
                raise ValueError("Only receiver points require a nonempty group")
            key = (kind, group, int(identity))
            if key in identities:
                raise ValueError("Patch acquisition point identities must be unique")
            identities.add(key)
            coordinates = np.asarray(point.get("coordinates"), dtype=float)
            if coordinates.shape != (dimension,) or not np.all(
                np.isfinite(coordinates)
            ):
                raise ValueError(
                    f"Patch points require {dimension} finite metre coordinates"
                )
            point["id"] = int(identity)
            point["coordinates"] = coordinates.tolist()
        if "roots" in patch:
            if "lower" in patch or "upper" in patch:
                raise ValueError("Specify roots or envelope bounds, not both")
            roots = tuple(patch["roots"])
            if not roots or any(
                isinstance(root, bool) or not isinstance(root, Integral) or root <= 0
                for root in roots
            ):
                raise ValueError("Patch roots must be positive integer parent IDs")
            if len(set(roots)) != len(roots):
                raise ValueError("Patch roots must be unique")
            patch["roots"] = sorted(int(root) for root in roots)
        else:
            for bound in ("lower", "upper"):
                values = np.asarray(patch.get(bound), dtype=float)
                if values.shape != (dimension,) or not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Patch {bound} must have {dimension} finite coordinates"
                    )
                patch[bound] = values.tolist()
            if np.any(np.array(patch["lower"]) > patch["upper"]):
                raise ValueError("Patch lower bounds must not exceed upper bounds")
    return data


_TASK_RESULTS = "jobs/*/*/results/_fs_run/tasks/task_*/result.json"
_OPERATION_RESULTS = "jobs/*/*/results/_fs_run/operations/*/result.json"


def _material_results(project: Path) -> list[OperationResult | TaskResult]:
    # Newest first; a result that cannot be read is not a committed record.
    files = [*project.glob(_TASK_RESULTS), *project.glob(_OPERATION_RESULTS)]
    stamped = []
    for file in files:
        try:
            stamped.append((file.stat().st_mtime_ns, file))
        except OSError:
            continue
    results: list[OperationResult | TaskResult] = []
    for _, file in sorted(stamped, reverse=True):
        result_path = file.parents[3]
        try:
            if file.parents[1].name == "operations":
                result: OperationResult | TaskResult = OperationResult.read(
                    file, result_path=result_path, workflow=file.parent.name
                )
            else:
                result = TaskResult.read(file, result_path=result_path)
        except (ArtifactContractError, OSError):
            continue
        if result.successful:
            results.append(result)
    return results


def mesh_sizing_frequency(
    f_list: Sequence[complex] | np.ndarray, simulation_payload: Mapping[str, Any]
) -> float:
    """Return the sizing frequency Sauce keys a job-sized generated mesh by.

    This is ``max(max |Re f_list|, Mesh/adapt/f_low)``; Sauce records it with
    each mesh-keyed material artifact.
    """

    mesh = simulation_payload.get("Mesh")
    adapt = mesh.get("adapt") if isinstance(mesh, Mapping) else None
    f_low = float(adapt.get("f_low", 0.0)) if isinstance(adapt, Mapping) else 0.0
    real = np.abs(np.real(np.asarray(f_list, dtype=complex)))
    return max(float(np.max(real)), f_low)


def recorded_material_artifact(
    project_path: str | Path,
    space: str,
    artifact: str | Path,
    *,
    simulation_fingerprint: str,
    sizing_frequency: float,
) -> tuple[Path, MaterialArtifactRecord]:
    """Return the material artifact Sauce will read for a space, from run records.

    Sauce records every mesh-space artifact it resolves (built or reused) in the
    run result. An existing declared path wins, as in Sauce, and must have been
    recorded. Otherwise the artifact is the mesh-keyed ``<stem>.<key>.h5`` that
    a run of the same simulation file (``fingerprints.simulation``) recorded at
    the same sizing frequency. Zero or several distinct candidates raise instead
    of guessing; among records of one path the newest supplies the identity.
    """

    project = Path(project_path).resolve()
    declared = Path(artifact)
    if not declared.is_absolute():
        declared = project / declared
    declared = declared.resolve()
    literal = declared.is_file()
    matches: dict[Path, MaterialArtifactRecord] = {}
    for result in _material_results(project):
        try:
            records = result.material_artifacts
        except ArtifactContractError:
            continue
        for record in records:
            path = record.resolve(project).resolve()
            if record.space != space or path.parent != declared.parent:
                continue
            if literal:
                selected = path == declared
            else:
                selected = (
                    record.mesh_keyed
                    and path.name == f"{declared.stem}.{record.mesh_key}.h5"
                    and result.fingerprints.get("simulation") == simulation_fingerprint
                    and math.isclose(
                        record.sizing_frequency or 0.0, sizing_frequency, rel_tol=1e-12
                    )
                )
            if selected:
                matches.setdefault(path, record)  # Results are newest first.
    if not matches:
        if literal:
            detail = f"no run has recorded {declared}"
        else:
            detail = (
                f"{declared} does not exist and no run of this simulation file "
                f"recorded a mesh-keyed artifact at sizing frequency "
                f"{sizing_frequency:g}"
            )
        raise FileNotFoundError(
            f"Cannot locate the material artifact for property space {space!r}: "
            f"{detail}. Run a job that builds the parent material space with the "
            "same simulation and frequency band first."
        )
    if len(matches) > 1:
        paths = ", ".join(str(path) for path in sorted(matches))
        raise RuntimeError(
            f"Runs recorded several material artifacts for property space {space!r} "
            f"at sizing frequency {sizing_frequency:g}: {paths}"
        )
    ((path, record),) = matches.items()
    if not path.is_file():
        raise FileNotFoundError(f"Recorded material artifact is missing: {path}")
    return path, record


@register_class
class PatchPreparationJob(BaseJob):
    """One geometry-only operation for a complete stage frequency band."""

    simulation: SeismicSimulation
    supports_trace_packing = False

    def __init__(
        self,
        name: str,
        simulation: SeismicSimulation,
        f_list: Sequence[complex] | np.ndarray,
        request: Mapping[str, Any],
    ) -> None:
        if simulation.dimension not in (2, 3):
            raise ValueError("Root patches require full-dimensional 2D or 3D geometry")
        if simulation.mesh.root_patch is not None:
            raise ValueError("Prepare patches from the parent simulation")
        frequencies = np.asarray(f_list, dtype=complex)
        if frequencies.ndim != 1 or not frequencies.size:
            raise ValueError("Patch preparation requires stage frequencies")
        if not np.all(np.isfinite(frequencies)) or np.any(frequencies.real <= 0):
            raise ValueError(
                "Patch preparation requires finite positive real frequencies"
            )
        self.request = _validate_request(request, simulation.dimension)
        super().__init__(name, simulation, "patch_prepare", frequencies.tolist())

    def _material_selection(self) -> dict[str, tuple[Path, MaterialArtifactRecord]]:
        """Return each space's recorded artifact that Sauce will read."""

        if not self.request.get("patches"):
            return {}
        spaces = getattr(self.simulation.model, "property_spaces", {})
        if not spaces:
            return {}
        if self.simulation._file is None:
            raise FileNotFoundError(
                "Save the parent simulation and run a job that builds its material "
                "spaces before selecting patches"
            )
        simulation_file = self._simulation_path()
        with open(simulation_file, encoding="utf-8") as stream:
            frequency = mesh_sizing_frequency(self.f_list, json.load(stream))
        fingerprint = self._sha256_file(simulation_file)
        return {
            name: recorded_material_artifact(
                self.simulation.project_path,
                name,
                space.artifact,
                simulation_fingerprint=fingerprint,
                sizing_frequency=frequency,
            )
            for name, space in sorted(spaces.items())
        }

    def _input_fingerprint_payload(self) -> dict[str, Any]:
        # Sauce records the artifact each run resolved; never re-derive its key.
        return {
            name: {
                **self._path_content_fingerprint(path),
                "basis_identity": record.basis_identity,
            }
            for name, (path, record) in self._material_selection().items()
        }

    def _verify_recorded_materials(self, operation: OperationResult) -> None:
        """Fail loudly unless Sauce read exactly the artifacts that were fingerprinted."""

        project = Path(self.simulation.project_path).resolve()
        read = {
            record.space: (record.resolve(project).resolve(), record.basis_identity)
            for record in operation.material_artifacts
        }
        expected = {
            name: (path, record.basis_identity)
            for name, (path, record) in self._material_selection().items()
        }
        if read != expected:
            raise RuntimeError(
                "Patch preparation read different material artifacts than were "
                f"fingerprinted: solver read {read}, expected {expected}"
            )

    def _external_input_fingerprint(self) -> dict[str, str] | None:
        # A saved preparation must still detect replaced parent artifacts.
        return self._refresh_external_input_fingerprint(require_inputs=True)

    @property
    def n_tasks(self) -> int:
        return 1

    def validate_outputs(self) -> None:
        _validate_request(self.request, self.simulation.dimension)

    def to_fs(
        self, ctx: ExportContext | None = None, *, project_relative: bool = False
    ) -> dict[str, Any]:
        payload = super().to_fs(ctx, project_relative=project_relative)
        payload.pop("Outputs", None)
        payload["PatchPreparation"] = deepcopy(self.request)
        return payload

    @classmethod
    def from_fs(
        cls,
        data: dict[str, Any],
        base_path: str | Path | None = None,
        project_path: str | Path | None = None,
    ) -> PatchPreparationJob:
        simulation = cls._load_simulation_for_job(
            data["simulation"],
            base_path=base_path,
            project_path=project_path or data.get("project_path"),
            source_project=data.get("project_path"),
        )
        if not isinstance(simulation, SeismicSimulation):
            raise TypeError("Patch preparation requires a SeismicSimulation")
        job = cls(
            data["name"],
            simulation,
            cls._decode_frequencies(data["f_list"]),
            data["PatchPreparation"],
        )
        job._job_id = data.get("job_id")
        return job

    @staticmethod
    def _fingerprint_job_payload(
        job_data: dict[str, Any], *, include_frequencies: bool = False
    ) -> dict[str, Any]:
        # All frequencies affect the one operation's scaling and parent generation.
        payload = BaseJob._fingerprint_job_payload(job_data, include_frequencies=True)
        payload["PatchPreparation"] = job_data["PatchPreparation"]
        return payload

    @staticmethod
    def effective_output_request_payload_from_fs(
        job_payload: Mapping[str, Any],
        *,
        simulation_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = BaseJob.effective_output_request_payload_from_fs(
            job_payload, simulation_payload=simulation_payload
        )
        payload["PatchPreparation"] = job_payload["PatchPreparation"]
        return payload

    def _current_task_artifact(
        self,
        task: int,
        *,
        fingerprints: Mapping[str, Any] | None = None,
        result: Any = None,
        catalog: Any = None,
    ) -> Any:
        if task != 1:
            return None
        fingerprints = (
            self._artifact_contract_fingerprints()
            if fingerprints is None
            else fingerprints
        )
        if fingerprints is None:
            return None
        try:
            saved_inputs = (
                json.loads(self.job_file.read_text())
                .get("artifact_contract", {})
                .get("external_inputs")
            )
            if saved_inputs != self._compute_external_input_fingerprint():
                return None
            operation = load_operation_result(self._result_path, "patch_prepare")
            if not operation.successful or any(
                operation.fingerprints.get(key) != value
                for key, value in fingerprints.items()
            ):
                return None
            records = {record.id: record for record in operation.artifacts}
            for key in ("patch_parent", "patch_geometry"):
                record = records[key]
                if (
                    not record.path.is_file()
                    or record.path.stat().st_size != record.bytes
                ):
                    return None
            return records["patch_geometry"]
        except (ArtifactContractError, OSError, KeyError, json.JSONDecodeError):
            return None

    def results_exist(self) -> bool:
        return self._current_task_artifact(1) is not None

    @property
    def geometry_report(self) -> dict[str, Any]:
        artifact = self._current_task_artifact(1)
        if artifact is None:
            raise FileNotFoundError("No current, committed patch preparation result")
        self._verify_recorded_materials(
            load_operation_result(self._result_path, "patch_prepare")
        )
        with artifact.path.open() as stream:
            report = json.load(stream)
        if report.get("schema") != "fs-patch-geometry-1":
            raise ValueError("Unsupported patch geometry report")
        return report
