"""Run-state inspection and rerun helpers mixed into jobs.

This module reads solver task metadata, compares trace files with stable job
fingerprints, summarizes convergence failures, and writes the Python-side
``_fs_python_run.json`` state used by status displays and incremental reruns.
"""

import json
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from frequensolve.simulation.artifact_contract import (
    ArtifactCatalog,
    ArtifactContractError,
    ArtifactRecord,
    ArtifactRequest,
    TaskResult,
    task_result_path,
)
from frequensolve.simulation.simulation import CustomJSONEncoder
from frequensolve.simulation.task_index import TaskCatalog, TaskIndex, load_task_catalog

SOLVER_RESIDUAL_FAILURE_THRESHOLD = 1.0e-3
_RECEIVER_TRACE_REQUEST = ArtifactRequest(
    id="traces",
    role="simulated_traces",
    representations=("hdf5_shard", "packed_trace"),
    retention="durable",
)


@dataclass(frozen=True)
class SkipPolicy:
    """Policy controlling which existing task outputs may be skipped.

    Args:
        mode: Human-readable policy name.
        reuse: Whether matching outputs from other task slots may be reused.
        ignore_solver_options: Ignore solver-only simulation JSON keys when
            comparing task compatibility.
        accept_failed: Accept failed/max-iter task records when quality checks
            pass.
        residual: Maximum residual accepted for failed/max-iter records.
        force: Ignore all existing outputs and rerun every task.
        ignored_simulation_keys: Top-level simulation JSON keys ignored by the
            compatibility fingerprint when ``ignore_solver_options`` is true.
    """

    mode: str = "strict"
    reuse: bool = True
    ignore_solver_options: bool = False
    accept_failed: bool = False
    residual: Optional[float] = None
    force: bool = False
    ignored_simulation_keys: tuple[str, ...] = ("Solver",)

    @classmethod
    def strict(cls) -> "SkipPolicy":
        """Require exact task fingerprints and successful/current outputs."""

        return cls(mode="strict")

    @classmethod
    def compatible(cls) -> "SkipPolicy":
        """Reuse compatible successful outputs while ignoring solver options."""

        return cls(mode="compatible", ignore_solver_options=True)

    @classmethod
    def tolerant(
        cls,
        *,
        residual: float = SOLVER_RESIDUAL_FAILURE_THRESHOLD,
        ignore_solver_options: bool = True,
    ) -> "SkipPolicy":
        """Reuse compatible outputs and accept failed records below residual."""

        return cls(
            mode="tolerant",
            ignore_solver_options=ignore_solver_options,
            accept_failed=True,
            residual=float(residual),
        )

    @classmethod
    def none(cls) -> "SkipPolicy":
        """Rerun all tasks."""

        return cls(mode="none", reuse=False, force=True)

    @classmethod
    def from_value(
        cls,
        value: Optional[Any] = None,
        *,
        residual: Optional[float] = None,
        ignore_solver_options: Optional[bool] = None,
        reuse: Optional[bool] = None,
    ) -> "SkipPolicy":
        """Normalize strings, booleans, or policies into ``SkipPolicy``."""

        if isinstance(value, SkipPolicy):
            policy = value
        elif value is None or value is True:
            policy = cls.strict()
        elif value is False:
            policy = cls.none()
        else:
            name = str(value).strip().lower().replace("-", "_")
            if name in {"strict", "current"}:
                policy = cls.strict()
            elif name in {"compatible", "compat"}:
                policy = cls.compatible()
            elif name in {"tolerant", "tolerance", "residual"}:
                policy = cls.tolerant(
                    residual=(
                        SOLVER_RESIDUAL_FAILURE_THRESHOLD
                        if residual is None
                        else float(residual)
                    ),
                    ignore_solver_options=(
                        True
                        if ignore_solver_options is None
                        else bool(ignore_solver_options)
                    ),
                )
            elif name in {"none", "false", "force", "rerun", "all"}:
                policy = cls.none()
            else:
                raise ValueError(
                    "skip policy must be 'strict', 'compatible', 'tolerant', "
                    "'none', or 'false'"
                )

        if residual is not None:
            policy = replace(
                policy,
                residual=float(residual),
                accept_failed=policy.accept_failed or not policy.force,
            )
            if policy.mode in {"strict", "compatible"}:
                policy = replace(
                    policy,
                    mode="tolerant",
                    ignore_solver_options=(
                        True
                        if ignore_solver_options is None
                        else bool(ignore_solver_options)
                    ),
                )
        if ignore_solver_options is not None:
            policy = replace(
                policy,
                ignore_solver_options=bool(ignore_solver_options),
            )
        if reuse is not None:
            policy = replace(policy, reuse=bool(reuse))
        return policy


class TaskRunPlan(dict):
    """Dictionary task plan with backward-compatible equality."""

    _legacy_keys = {"pending_indices", "current_tasks", "reused_tasks"}

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            other_keys = set(other.keys())
            if other_keys <= self._legacy_keys:
                return {key: self.get(key) for key in other_keys} == dict(other)
        return super().__eq__(other)


class JobRunStateMixin:
    """Inspect, write, and reuse per-task solver run state.

    The mixin tracks one-based solver tasks, validates task fingerprints
    against trace outputs, aggregates solver manifests, and writes the
    Python-side run-state summary used by local status widgets and rerun logic.
    """

    @classmethod
    def solver_convergence_summary(
        cls,
        data: Mapping[str, Any],
        *,
        residual_failure_threshold: float = SOLVER_RESIDUAL_FAILURE_THRESHOLD,
    ) -> Optional[Dict[str, Any]]:
        """Extract concise convergence information from solver metadata.

        Args:
            data: Solver metadata mapping or nested solver section.
            residual_failure_threshold: Residual above which a solve is marked
                failed even if the raw solver status is ambiguous.

        Returns:
            Summary mapping with convergence status, residuals, iterations, and
            per-solve details, or ``None`` when metadata has no convergence
            section.
        """

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
        if not residuals:
            top_level_residual = cls._rounded_solver_float(
                convergence.get("residual", convergence.get("final_residual"))
            )
            if top_level_residual is not None:
                residuals.append(top_level_residual)
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

    def frequency_status(self) -> List[Dict[str, Any]]:
        """Return per-frequency run status rows for this job.

        Status is inferred from expected trace files plus any Python/solver run
        metadata available beside the results. Task numbers are one-based to
        match solver task IDs and trace filenames.

        Returns:
            List of rows containing task number, frequency, status, trace path,
            current-output flag, duration, and raw metadata.
        """

        fingerprints = self._artifact_contract_fingerprints()
        rows: Dict[int, Dict[str, Any]] = {}
        for task in range(1, self.n_tasks + 1):
            result = self._committed_task_result(task)
            artifact = self._current_task_artifact(
                task,
                fingerprints=fingerprints,
                result=result,
            )
            current = artifact is not None
            trace_file = artifact.path if artifact is not None else None
            status = "succeeded" if current else "not_run"
            if result is not None and not result.successful:
                status = "failed"
            rows[task] = {
                "task": task,
                "frequency": self._canonical_frequency_value(self.f_list[task - 1]),
                "status": status,
                "trace_file": trace_file,
                "trace_exists": current,
                "current": current,
                "duration_seconds": None,
                "metadata": (
                    {}
                    if result is None
                    else {
                        "task_result": str(result.path),
                        "status": result.state,
                        "returncode": result.code,
                        "solver": dict(result.solver),
                        "timings": dict(result.timings),
                        "resources": dict(result.resources),
                        "timestamps": dict(result.timestamps),
                        "duration_seconds": sum(result.timings.values()),
                    }
                ),
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
                if not row["current"]:
                    row["status"] = "failed"
            elif status == "succeeded" and row["current"] and row["status"] != "failed":
                row["status"] = "succeeded"

        return [rows[task] for task in sorted(rows)]

    def failed_tasks(self, *, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """Return failed task rows with the best available failure reason.

        The failure classification matches :meth:`frequency_status`, including
        solver convergence failures such as residuals above the failure
        threshold.  Task numbers are one-based.

        Args:
            include_metadata: Include raw task metadata in each returned row.

        Returns:
            List of failed task summaries.
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
        """Return failed tasks; compatibility alias for ``failed_tasks``.

        Args:
            include_metadata: Include raw task metadata in each returned row.

        Returns:
            List of failed task summaries.
        """

        return self.failed_tasks(include_metadata=include_metadata)

    def frequency_summary(self) -> Dict[str, int]:
        """Count succeeded, failed, and not-yet-run frequencies.

        Returns:
            Mapping with ``total``, ``succeeded``, ``failed``, ``not_run``, and
            optionally ``unassigned_failures``.
        """

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
        """Print and return a concise frequency status summary.

        Args:
            file: Optional text stream passed to ``print``.

        Returns:
            Same summary mapping returned by :meth:`frequency_summary`.
        """

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

    def run_state(self) -> Dict[str, Any]:
        """Return the persisted Python-side run state.

        Returns:
            Parsed run-state mapping, or an empty mapping if the state file is
            missing or invalid JSON.
        """

        if not self.run_state_file.exists():
            return {}
        try:
            return json.loads(self.run_state_file.read_text())
        except json.JSONDecodeError:
            return {}

    def _artifact_contract_fingerprints(self) -> Optional[Dict[str, str]]:
        """Return the three Sauce input digests used by task-result v2.

        The job and simulation digests intentionally cover the exact staged
        JSON documents. The output digest matches Sauce's effective per-task
        output-request digest. Callers that inspect multiple tasks compute
        this mapping once and pass it through the batch.
        """

        job_file = getattr(self, "_file", None)
        simulation_file = getattr(self.simulation, "_file", None)
        if job_file is None or simulation_file is None:
            return None
        job_file = Path(job_file)
        simulation_file = Path(simulation_file)
        if not job_file.is_file() or not simulation_file.is_file():
            return None
        output_fingerprint = self.effective_output_request_fingerprint()
        try:
            artifact_metadata = json.loads(job_file.read_text(encoding="utf-8")).get(
                "artifact_contract",
                {},
            )
        except (OSError, json.JSONDecodeError):
            return None
        saved_output_fingerprint = artifact_metadata.get("output_request", {}).get(
            "digest"
        )
        if (
            saved_output_fingerprint is not None
            and saved_output_fingerprint != output_fingerprint
        ):
            return None
        job_fingerprint = self._sha256_file(job_file)
        return {
            "job": job_fingerprint,
            "simulation": self._sha256_file(simulation_file),
            # Directly authored solver jobs predate the persisted request
            # digest. Sauce conservatively records the exact job-file digest
            # for those inputs, so the consumer must make the same fallback.
            "outputs": (
                output_fingerprint
                if saved_output_fingerprint is not None
                else job_fingerprint
            ),
        }

    def _committed_task_result(self, task: int) -> Optional[TaskResult]:
        """Read one known task-result path without discovering payload files."""

        if task < 1 or task > self.n_tasks:
            return None
        path = task_result_path(self._result_path, task)
        if not path.is_file():
            return None
        try:
            result = TaskResult.read(path, result_path=self._result_path)
        except (ArtifactContractError, OSError):
            return None
        if result.partition.task != task:
            return None
        return result

    def _load_task_catalog(self, tasks: Iterable[int]) -> TaskCatalog:
        """Load the consolidated index, conservatively rejecting bad metadata."""

        try:
            return load_task_catalog(self._result_path, tasks=tasks)
        except (ArtifactContractError, OSError):
            return ArtifactCatalog(
                result_path=Path(self._result_path).resolve(strict=False),
                results={},
            )

    def _retry_compatibility_hash(self):
        """Match Sauce's retry identity while retaining full producer hashes."""

        try:
            job = json.loads(self.job_file.read_text())
            simulation = json.loads(self.simulation._file.read_text())
        except (OSError, AttributeError, TypeError):
            return None
        return self._compatibility_hash_payloads(job, simulation)

    @staticmethod
    def _compatibility_hash_payloads(job, simulation):
        """Hash solver-independent inputs using the native canonical JSON form."""
        import copy
        import hashlib

        job, simulation = copy.deepcopy(job), copy.deepcopy(simulation)
        job["simulation"] = None
        job["preserve_task_outputs"] = None
        job.setdefault("artifact_contract", {})["output_request"] = None
        identity = {"job": job, "simulation": simulation}
        protected = (
            "grids",
            "refinements",
            "refinement_flags",
            "hp",
            "galerkin_multigrid",
            "relaxed_assembly",
            "mode",
        )
        structure = {
            key: value
            for key, value in simulation.get("Solver", {}).items()
            if key in protected
        }
        if structure:
            identity["solver_structure"] = structure
        simulation["Solver"] = None
        encoded = json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def _task_reuse_fingerprints(self):
        """Select the explicitly requested retry policy for task consumers."""
        if self.preserve_task_outputs:
            digest = self._retry_compatibility_hash()
            if digest is not None:
                return {"compatibility": digest}
        return self._artifact_contract_fingerprints()

    def _current_task_artifact(
        self,
        task: int,
        *,
        fingerprints: Optional[Mapping[str, str]] = None,
        result: Optional[TaskResult] = None,
        catalog: Optional[TaskCatalog] = None,
    ) -> Optional[ArtifactRecord]:
        """Return a verified durable receiver artifact for one current task.

        Validation is control-plane only: the fixed JSON commit record is
        checked against exact task identity and input fingerprints, then the
        declared payload is checked with one stat call. HDF5 payloads are never
        opened and no filename or directory scan participates in currentness.
        """

        if self.preserve_task_outputs:
            compatibility = self._retry_compatibility_hash()
            if compatibility is not None:
                fingerprints = {"compatibility": compatibility}
                if result is None:
                    result = self._committed_task_result(task)
                catalog = None
        if task < 1 or task > self.n_tasks:
            return None
        fingerprints = (
            self._artifact_contract_fingerprints()
            if fingerprints is None
            else fingerprints
        )
        if fingerprints is None:
            return None
        if catalog is None and result is None:
            catalog = self._load_task_catalog((task,))
        if catalog is None:
            if result is None or not result.successful:
                return None
            if result.partition.task != task:
                return None
            if result.partition.task_count is not None and (
                result.partition.task_count != self.n_tasks
            ):
                return None
            if result.partition.frequency != complex(self.f_list[task - 1]):
                return None
            if any(
                result.fingerprints.get(name) != digest
                for name, digest in fingerprints.items()
            ):
                return None
            catalog = ArtifactCatalog(
                result_path=Path(self._result_path).resolve(strict=False),
                results={task: result},
            )
        elif isinstance(catalog, TaskIndex):
            if not catalog.is_task_current(
                task,
                frequency=complex(self.f_list[task - 1]),
                fingerprints=fingerprints,
            ):
                return None
        else:
            result = catalog.results.get(task)
            if result is None or not result.successful:
                return None
            if result.partition.task_count is not None and (
                result.partition.task_count != self.n_tasks
            ):
                return None
            if result.partition.frequency != complex(self.f_list[task - 1]):
                return None
            if any(
                result.fingerprints.get(name) != digest
                for name, digest in fingerprints.items()
            ):
                return None
        try:
            artifact = catalog.require_one(_RECEIVER_TRACE_REQUEST, task=task)
            stat = artifact.path.stat()
        except (ArtifactContractError, OSError):
            return None
        if not artifact.path.is_file() or stat.st_size != artifact.bytes:
            return None
        return artifact

    def is_task_current(
        self,
        task: int,
        *,
        state: Optional[Mapping[str, Any]] = None,
        manifest: Optional[Any] = None,
    ) -> bool:
        """Return whether a one-based task has current reusable outputs.

        Args:
            task: One-based solver task number.
            state: Optional preloaded run-state mapping.
            manifest: Optional preloaded trace manifest for batch checks.

        Returns:
            ``True`` when the task output exists and matches the current task
            fingerprint; otherwise ``False``.
        """

        # ``state`` and ``manifest`` remain accepted for API compatibility but
        # are deliberately not authoritative in task-result v2.
        del state, manifest
        return self._current_task_artifact(task) is not None

    def current_tasks(self) -> List[int]:
        """Return one-based frequency tasks that are current for this job.

        Returns:
            Sorted task numbers whose outputs can be reused.
        """

        fingerprints = self._artifact_contract_fingerprints()
        catalog = self._load_task_catalog(range(1, self.n_tasks + 1))
        return [
            task
            for task in range(1, self.n_tasks + 1)
            if self._current_task_artifact(
                task,
                fingerprints=fingerprints,
                catalog=catalog,
            )
            is not None
        ]

    def _simulation_hash_for_policy(self, policy: SkipPolicy) -> str:
        if self.simulation._file is None:
            raise ValueError("Simulation must be saved before fingerprinting a task")
        if not policy.ignore_solver_options:
            return self._hash_json_file(self.simulation._file)
        with open(self.simulation._file, "r") as f:
            payload = json.load(f)
        if isinstance(payload, Mapping):
            payload = dict(payload)
            for key in policy.ignored_simulation_keys:
                payload.pop(key, None)
        return self._hash_payload(payload)

    def task_policy_fingerprint_payload(
        self,
        task: int,
        skip_policy: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return the task compatibility payload for a skip policy."""

        policy = SkipPolicy.from_value(skip_policy)
        if not policy.ignore_solver_options:
            return self.task_fingerprint_payload(task)
        if task < 1 or task > self.n_tasks:
            raise IndexError(f"Task {task} is outside 1..{self.n_tasks}")
        job_data = self.to_fs(self.export_context())
        job_payload = self._fingerprint_job_payload(job_data)
        return {
            "schema": "frequensolve-job-task-compatibility-fingerprint-1",
            "job": job_payload,
            "simulation": {
                "hash": self._simulation_hash_for_policy(policy),
                "ignored_keys": list(policy.ignored_simulation_keys),
            },
            **(
                {"frequency": self._canonical_frequency_value(self.f_list[task - 1])}
                if "f_list" in job_data
                else {}
            ),
        }

    def task_policy_fingerprint(
        self,
        task: int,
        skip_policy: Optional[Any] = None,
    ) -> str:
        """Return the task fingerprint used by a skip policy."""

        return self._hash_payload(
            self.task_policy_fingerprint_payload(task, skip_policy)
        )

    def _task_policy_fingerprint_keys(
        self,
        task: int,
        policy: SkipPolicy,
    ) -> tuple[str, ...]:
        keys = [self.task_policy_fingerprint(task, policy)]
        if policy.ignore_solver_options:
            keys.append(self.task_fingerprint(task))
        return tuple(dict.fromkeys(keys))

    @staticmethod
    def _record_policy_fingerprint_keys(
        record: Mapping[str, Any],
        policy: SkipPolicy,
    ) -> tuple[str, ...]:
        keys = []
        if policy.ignore_solver_options:
            key = record.get("compatibility_fingerprint")
            if key:
                keys.append(str(key))
        key = record.get("fingerprint")
        if key:
            keys.append(str(key))
        return tuple(dict.fromkeys(keys))

    def _record_trace_source(self, record: Mapping[str, Any]) -> Optional[Path]:
        source = self._resolve_stored_trace_path(
            record.get("path") or record.get("trace_file")
        )
        if source is None or source in self.trace_manifest.packed_files:
            return None
        return source if self._trace_file_exists(source) else None

    def _record_policy_acceptance(
        self,
        record: Mapping[str, Any],
        policy: SkipPolicy,
    ) -> tuple[bool, bool]:
        status = self._normalized_task_status(record.get("status"))
        if status == "succeeded" and not self._record_solver_not_run(record):
            return True, False

        failed = status == "failed" or self._record_solver_failed(record)
        if not failed or not policy.accept_failed:
            return False, False

        convergence = self._record_solver_convergence(record)
        if not isinstance(convergence, Mapping):
            return False, False
        residual = convergence.get("residual", convergence.get("final_residual"))
        if residual is None:
            return False, False
        threshold = (
            SOLVER_RESIDUAL_FAILURE_THRESHOLD
            if policy.residual is None
            else float(policy.residual)
        )
        try:
            return float(residual) <= threshold, True
        except (TypeError, ValueError):
            return False, False

    def _current_task_record(
        self,
        task: int,
        path: Path,
    ) -> Dict[str, Any]:
        return {
            "task": task,
            "status": "current",
            "duration_seconds": 0.0,
            "fingerprint": self.task_fingerprint(task),
            "compatibility_fingerprint": self.task_policy_fingerprint(
                task,
                SkipPolicy.compatible(),
            ),
            "path": self._stored_trace_path(path),
        }

    def plan_tasks(
        self,
        *,
        skip_policy: Optional[Any] = None,
        reuse: bool = False,
        force: bool = False,
        apply: bool = False,
        residual: Optional[float] = None,
        ignore_solver_options: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Plan frequency tasks under a skip policy.

        Args:
            skip_policy: ``SkipPolicy`` or policy name.
            reuse: Allow matching outputs from other task slots.
            force: Rerun every task.
            apply: Copy reusable outputs, update run state, and remove stale
                pending outputs. ``False`` is a read-only preview.
            residual: Optional residual threshold for tolerant policies.
            ignore_solver_options: Override whether solver JSON keys are
                included in compatibility fingerprints.

        Returns:
            Mapping with pending indices and skipped/reused/accepted records.
        """

        policy = SkipPolicy.from_value(
            skip_policy,
            residual=residual,
            ignore_solver_options=ignore_solver_options,
            reuse=reuse,
        )
        force = bool(force or policy.force)
        if force:
            pending = list(range(self.n_tasks))
            return TaskRunPlan(
                {
                    "pending_indices": pending,
                    "strict_current_tasks": [],
                    "current_tasks": [],
                    "reused_tasks": [],
                    "accepted_tasks": [],
                    "accepted_failed_tasks": [],
                    "skipped_task_records": [],
                    "skip_policy": policy,
                    "removed_stale_outputs": False,
                }
            )

        fingerprints = self._artifact_contract_fingerprints()
        catalog = self._load_task_catalog(range(1, self.n_tasks + 1))
        current_records = []
        for task in range(1, self.n_tasks + 1):
            artifact = self._current_task_artifact(
                task,
                fingerprints=fingerprints,
                catalog=catalog,
            )
            if artifact is not None:
                current_records.append(self._current_task_record(task, artifact.path))

        current_task_numbers = {
            int(record["task"])
            for record in current_records
            if record.get("task") is not None
        }
        # Generation-scoped v2 task results are authoritative for their exact
        # partition. Relabelling or copying an opaque payload into another
        # task slot would sever that commit record from its producer metadata,
        # so cross-slot filename reuse is intentionally not part of planning.
        applied_records: List[Dict[str, Any]] = []
        skipped_tasks = set(current_task_numbers)
        pending = [
            task - 1 for task in range(1, self.n_tasks + 1) if task not in skipped_tasks
        ]

        if apply and current_records:
            self.write_run_state(
                status="partial",
                tasks=current_records,
            )
        removed_stale_outputs = False

        reused_records = [
            record for record in applied_records if record.get("status") == "reused"
        ]
        accepted_failed = [
            int(record["task"])
            for record in applied_records
            if record.get("accepted_failed")
        ]
        accepted_tasks = [
            int(record["task"])
            for record in applied_records
            if record.get("status") == "accepted"
        ]
        skipped_records = list(current_records)
        return TaskRunPlan(
            {
                "pending_indices": pending,
                "strict_current_tasks": sorted(current_task_numbers),
                "current_tasks": sorted(skipped_tasks),
                "reused_tasks": reused_records,
                "accepted_tasks": accepted_tasks,
                "accepted_failed_tasks": accepted_failed,
                "skipped_task_records": skipped_records,
                "skip_policy": policy,
                "removed_stale_outputs": removed_stale_outputs,
            }
        )

    def task_run_plan(
        self,
        *,
        reuse: bool = False,
        force: bool = False,
        skip_policy: Optional[Any] = None,
        residual: Optional[float] = None,
        ignore_solver_options: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Plan which zero-based solver task indices still need to run.

        ``reuse`` is retained as a policy input, but task reuse is limited to
        exact authoritative task results. Opaque generation payloads are never
        copied or relabelled across task partitions.

        Args:
            reuse: Permit exact committed-task reuse under the selected policy.
            force: Ignore reusable outputs and rerun all tasks.

        Returns:
            Mapping with zero-based ``pending_indices``, one-based
            ``current_tasks``, and records for any ``reused_tasks``.
        """

        return self.plan_tasks(
            skip_policy=skip_policy,
            reuse=reuse,
            force=force,
            apply=True,
            residual=residual,
            ignore_solver_options=ignore_solver_options,
        )

    def is_run_current(self) -> bool:
        """Return whether all expected outputs match the saved job inputs.

        Returns:
            ``True`` when result metadata, fingerprints, and trace outputs show
            that every task is current.
        """

        return len(self.current_tasks()) == self.n_tasks

    def write_run_state(self, status: str = "completed", **extra) -> Path:
        """Write the Python-side run state summary.

        Args:
            status: Overall run status to store.
            **extra: Additional fields for the run-state payload. ``tasks`` may
                contain per-task result records.

        Returns:
            Path to the written run-state JSON file.
        """

        self._result_path.mkdir(parents=True, exist_ok=True)
        extra = dict(extra)
        task_results = self._as_records(extra.pop("tasks", None))
        result_by_task = self._task_record_by_task(task_results)
        previous_state = self.run_state()
        previous_by_task = self._task_record_by_task(
            self._state_task_records(previous_state)
        )
        files = []
        task_rows = []
        fingerprints = self._artifact_contract_fingerprints()
        for task in range(1, self.n_tasks + 1):
            committed = self._committed_task_result(task)
            current_artifact = self._current_task_artifact(
                task,
                fingerprints=fingerprints,
                result=committed,
            )
            reported_artifact = None
            if committed is not None:
                catalog = ArtifactCatalog(
                    result_path=Path(self._result_path).resolve(strict=False),
                    results={task: committed},
                )
                try:
                    reported_artifact = catalog.require_one(
                        _RECEIVER_TRACE_REQUEST,
                        task=task,
                    )
                except ArtifactContractError:
                    pass
            file_path = (
                current_artifact.path
                if current_artifact is not None
                else (reported_artifact.path if reported_artifact is not None else None)
            )
            exists = bool(file_path is not None and file_path.is_file())
            stored_path = (
                None if file_path is None else self._stored_trace_path(file_path)
            )
            files.append({"path": stored_path, "exists": exists})

            result = dict(result_by_task.get(task, {}))
            raw_status = str(result.get("status", "")).strip().lower().replace(" ", "_")
            accepted_by_policy = raw_status in {"accepted", "accepted_failed"}
            task_status = self._normalized_task_status(result.get("status"))
            previously_current = current_artifact is not None
            if not result and previously_current:
                result = dict(previous_by_task.get(task, {}))
                raw_status = (
                    str(result.get("status", "")).strip().lower().replace(" ", "_")
                )
                accepted_by_policy = raw_status in {"accepted", "accepted_failed"}
                task_status = self._normalized_task_status(result.get("status"))
            if task_status is None and previously_current:
                task_status = "current"
            elif task_status is None and committed is not None:
                task_status = "succeeded" if committed.successful else "failed"
            if task_status is None:
                task_status = "not_run"
            solver_convergence = self._record_solver_convergence(result)
            if (
                solver_convergence is not None
                and solver_convergence.get("failed")
                and not accepted_by_policy
            ):
                task_status = "failed"
            elif accepted_by_policy:
                task_status = raw_status

            duration = self._record_duration_seconds(result)
            row = {
                "task": task,
                "frequency": self._canonical_frequency_value(self.f_list[task - 1]),
                "status": task_status,
                "complete": self._task_is_complete(result, task_status),
                "fingerprint": self.task_fingerprint(task),
                "compatibility_fingerprint": self.task_policy_fingerprint(
                    task,
                    SkipPolicy.compatible(),
                ),
                "path": stored_path,
                "exists": exists,
            }
            if solver_convergence is not None:
                row["solver"] = {"convergence": solver_convergence}
            if accepted_by_policy:
                row["accepted"] = True
            if raw_status == "accepted_failed":
                row["accepted_failed"] = True
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
        return self.run_state_file

    def collect_task_results(
        self,
        *,
        status: str = "completed",
    ) -> Optional[Path]:
        """Aggregate committed task results into Python diagnostic state.

        Args:
            status: Overall status to use when writing the aggregated
                Python-side run state.

        Returns:
            Path to the written run-state file, or ``None`` if no task results
            were found.
        """

        task_records = []
        for task in range(1, self.n_tasks + 1):
            result = self._committed_task_result(task)
            if result is None:
                continue
            record: Dict[str, Any] = {
                "task_id": task - 1,
                "status": result.state if result.successful else "error",
                "complete": True,
                "fingerprint": self.task_fingerprint(task),
                "compatibility_fingerprint": self.task_policy_fingerprint(
                    task,
                    SkipPolicy.compatible(),
                ),
                "task_result": str(result.path),
                "returncode": result.code,
                "artifacts": [artifact.to_fs() for artifact in result.artifacts],
            }
            if result.solver:
                record["solver"] = dict(result.solver)
            if result.resources:
                record["resources"] = dict(result.resources)
            if result.timings:
                record["timings"] = dict(result.timings)
            task_records.append(record)

        if not task_records:
            return None
        return self.write_run_state(status=status, tasks=task_records)

    def _task_outputs_hash(self, task: int) -> str:
        """Compatibility alias for the v2 effective output-request digest."""

        if task < 1 or task > self.n_tasks:
            raise IndexError(f"Task {task} is outside 1..{self.n_tasks}")
        return self.effective_output_request_fingerprint()

    @staticmethod
    def _frequency_parts(value: Any) -> Optional[tuple[float, float]]:
        if isinstance(value, Mapping):
            real = value.get("real", value.get("value"))
            imag = value.get("imag", 0.0)
        elif isinstance(value, complex):
            real = value.real
            imag = value.imag
        else:
            real = value
            imag = 0.0
        try:
            return float(real), float(imag)
        except (TypeError, ValueError):
            return None

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
            "accepted",
            "accepted_failed",
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
    def _record_solver_convergence(
        cls, record: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if str(record.get("status", "")).strip().lower().replace(" ", "_") in {
            "skipped",
            "current",
            "reused",
            "accepted",
            "accepted_failed",
        }:
            return None
        solver = record.get("solver")
        if isinstance(solver, Mapping):
            summary = cls.solver_convergence_summary({"solver": solver})
            if summary is not None:
                return summary
        return None

    @classmethod
    def _record_solver_failed(cls, record: Mapping[str, Any]) -> bool:
        summary = cls._record_solver_convergence(record)
        return bool(summary and summary.get("failed"))

    @classmethod
    def _record_solver_not_run(cls, record: Mapping[str, Any]) -> bool:
        summary = cls._record_solver_convergence(record)
        if not isinstance(summary, Mapping):
            return False
        try:
            solve_count = int(summary.get("solve_count", 0))
        except (TypeError, ValueError):
            solve_count = 0
        has_solves = bool(cls._as_records(summary.get("solves")))
        return (
            str(summary.get("status", "")).strip().lower() == "not_run"
            and solve_count == 0
            and not has_solves
        )

    @staticmethod
    def _failure_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, Mapping):
            for key in ("reason", "message", "error", "status"):
                text = JobRunStateMixin._failure_text(value.get(key))
                if text:
                    return text
            try:
                return json.dumps(value, cls=CustomJSONEncoder)
            except TypeError:
                return str(value)
        if isinstance(value, (list, tuple)):
            parts = [
                text for item in value if (text := JobRunStateMixin._failure_text(item))
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
            "elapsed_s",
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
        for record in self._task_records():
            cores = self._record_core_count(record)
            if cores is not None:
                return cores
        return None

    def _task_records(self) -> List[Mapping[str, Any]]:
        state = self.run_state()
        records: List[Mapping[str, Any]] = []
        for source in (state,):
            if not isinstance(source, Mapping):
                continue
            records.extend(self._as_records(source.get("tasks")))
            records.extend(self._as_records(source.get("task_results")))
            records.extend(self._as_records(source.get("task_timings")))
            records.extend(self._as_records(source.get("frequencies")))
            records.extend(self._as_records(source.get("errors")))
        for task in range(1, self.n_tasks + 1):
            result = self._committed_task_result(task)
            if result is None:
                continue
            records.append(
                {
                    "task": task,
                    "frequency": self._canonical_frequency_value(
                        result.partition.frequency
                    ),
                    "status": result.state,
                    "returncode": result.code,
                    "solver": dict(result.solver),
                    "timings": dict(result.timings),
                    "phases": dict(result.timings),
                    "resources": dict(result.resources),
                    "timestamps": dict(result.timestamps),
                    "duration_seconds": sum(result.timings.values()),
                }
            )
        return records

    def _reuse_task_outputs_from_state(
        self, state: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        reusable = self._reusable_task_outputs_from_state(state)
        if not reusable:
            return []

        stage_dir = self._result_path / "_fs_run" / "reuse"
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged = []
        try:
            for record in reusable:
                task = int(record["task"])
                source = Path(record["source_path"])
                target = Path(record.get("target_path", record["path"]))
                stage = stage_dir / f"task_{task}{source.suffix}"
                shutil.copy2(source, stage)
                staged.append((task, stage, target, record))

            reused = []
            for task, stage, target, record in staged:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(stage, target)
                reused_record = {
                    key: value for key, value in record.items() if key != "source_path"
                }
                reused.append(reused_record)
            return reused
        finally:
            for _task, stage, _target, _record in staged:
                try:
                    stage.unlink()
                except FileNotFoundError:
                    pass

    def _reusable_task_outputs_from_state(
        self, state: Mapping[str, Any]
    ) -> List[Dict[str, Any]]:
        manifest = self.trace_manifest
        records = self._state_task_records(state)
        source_by_fingerprint: Dict[str, Path] = {}
        for record in records:
            if self._normalized_task_status(record.get("status")) != "succeeded":
                continue
            fingerprint = record.get("fingerprint")
            source = self._resolve_stored_trace_path(
                record.get("path") or record.get("trace_file")
            )
            if source is not None and source in manifest.packed_files:
                continue
            if fingerprint and source is not None and self._trace_file_exists(source):
                source_by_fingerprint.setdefault(fingerprint, source)

        copies = []
        for task, target in enumerate(manifest.files, start=1):
            if self.is_task_current(task, state=state, manifest=manifest):
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

        return [
            {
                "task": task,
                "status": "reused",
                "duration_seconds": 0.0,
                "fingerprint": self.task_fingerprint(task),
                "path": self._stored_trace_path(target),
                "source_path": str(source),
                "target_path": str(target),
            }
            for task, source, target in copies
        ]

    def reusable_task_outputs(self) -> List[Dict[str, Any]]:
        """Return read-only records for prior outputs reusable by this job.

        Unlike :meth:`task_run_plan`, this method does not copy files, write run
        state, or remove stale outputs. It is intended for submission previews.
        """

        state = self.run_state()
        return self._reusable_task_outputs_from_state(state) if state else []

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
