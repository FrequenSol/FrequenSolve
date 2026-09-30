# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Native geometry preparation through the existing job scheduler."""

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from frequensolve.simulation.artifact_contract import (
    ArtifactContractError,
    load_operation_result,
)
from frequensolve.simulation.jobs.base import BaseJob
from frequensolve.util.class_registry import register_class


@dataclass(frozen=True)
class PatchRunResult:
    """Composite forward results in stable prepared-patch order."""

    jobs: tuple
    runs: tuple
    prepared: object

    @property
    def successful(self):
        return all(run.successful for run in self.runs)

    @property
    def job(self):
        if len(self.jobs) != 1:
            raise ValueError("Patch execution used multiple jobs; inspect .jobs")
        return self.jobs[0]

    def raise_for_status(self):
        for run in self.runs:
            run.raise_for_status()


def _validate_request(request, dimension):
    data = deepcopy(dict(request))
    if set(data) - {"units", "patches"} or data.get("units") != "m":
        raise ValueError("PatchPreparation requires metre coordinates (units='m')")
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
            for key in ("lower", "upper"):
                values = np.asarray(patch.get(key), dtype=float)
                if values.shape != (dimension,) or not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Patch {key} must have {dimension} finite coordinates"
                    )
                patch[key] = values.tolist()
            if np.any(np.array(patch["lower"]) > patch["upper"]):
                raise ValueError("Patch lower bounds must not exceed upper bounds")
    return data


@register_class
class PatchPreparationJob(BaseJob):
    """One geometry-only operation for a complete stage frequency band."""

    supports_trace_packing = False

    def __init__(self, name, simulation, f_list, request):
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

    def _input_fingerprint_payload(self):
        if not self.request.get("patches"):
            return {}
        spaces = getattr(self.simulation.model, "property_spaces", {})
        return {
            name: self._path_content_fingerprint(
                Path(self.simulation.project_path) / space.artifact
            )
            for name, space in sorted(spaces.items())
        }

    def _external_input_fingerprint(self):
        # A saved preparation must still detect replaced parent artifacts.
        return self._refresh_external_input_fingerprint(require_inputs=True)

    @property
    def n_tasks(self):
        return 1

    def validate_outputs(self):
        _validate_request(self.request, self.simulation.dimension)

    def to_fs(self, ctx=None, *, project_relative=False):
        payload = super().to_fs(ctx, project_relative=project_relative)
        payload.pop("Outputs", None)
        payload["PatchPreparation"] = deepcopy(self.request)
        return payload

    @classmethod
    def from_fs(cls, data, base_path=None, project_path=None):
        simulation = cls._load_simulation_for_job(
            data["simulation"],
            base_path=base_path,
            project_path=project_path or data.get("project_path"),
            source_project=data.get("project_path"),
        )
        job = cls(
            data["name"],
            simulation,
            cls._decode_frequencies(data["f_list"]),
            data["PatchPreparation"],
        )
        job._job_id = data.get("job_id")
        return job

    @staticmethod
    def _fingerprint_job_payload(job_data, *, include_frequencies=False):
        # All frequencies affect the one operation's scaling and parent generation.
        payload = BaseJob._fingerprint_job_payload(job_data, include_frequencies=True)
        payload["PatchPreparation"] = job_data["PatchPreparation"]
        return payload

    @staticmethod
    def effective_output_request_payload_from_fs(
        job_payload, *, simulation_payload=None
    ):
        payload = BaseJob.effective_output_request_payload_from_fs(
            job_payload, simulation_payload=simulation_payload
        )
        payload["PatchPreparation"] = job_payload["PatchPreparation"]
        return payload

    def _current_task_artifact(
        self, task, *, fingerprints=None, result=None, catalog=None
    ):
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

    def results_exist(self):
        return self._current_task_artifact(1) is not None

    @property
    def geometry_report(self):
        artifact = self._current_task_artifact(1)
        if artifact is None:
            raise FileNotFoundError("No current, committed patch preparation result")
        with artifact.path.open() as stream:
            report = json.load(stream)
        if report.get("schema") != "fs-patch-geometry-1":
            raise ValueError("Unsupported patch geometry report")
        return report
