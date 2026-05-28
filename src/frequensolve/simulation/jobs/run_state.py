"""Run-state inspection and rerun helpers mixed into jobs.

This module reads solver task metadata, compares trace files with stable job
fingerprints, summarizes convergence failures, and writes the Python-side
``_fs_python_run.json`` state used by status displays and incremental reruns.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from frequensolve.simulation.simulation import CustomJSONEncoder

SOLVER_RESIDUAL_FAILURE_THRESHOLD = 1.0e-3


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

    def is_task_current(
        self,
        task: int,
        *,
        state: Optional[Mapping[str, Any]] = None,
    ) -> bool:
        """Return whether a one-based task has current reusable outputs.

        Args:
            task: One-based solver task number.
            state: Optional preloaded run-state mapping.

        Returns:
            ``True`` when the task output exists and matches the current task
            fingerprint; otherwise ``False``.
        """

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
        """Return one-based frequency tasks that are current for this job.

        Returns:
            Sorted task numbers whose outputs can be reused.
        """

        state = self.run_state()
        return [
            task
            for task in range(1, self.n_tasks + 1)
            if self.is_task_current(task, state=state)
        ]

    def task_run_plan(
        self, *, reuse: bool = False, force: bool = False
    ) -> Dict[str, Any]:
        """Plan which zero-based solver task indices still need to run.

        When ``reuse`` is true, matching trace files from an earlier frequency
        layout are copied into their current task-numbered locations and the
        run state is updated to record those reused tasks.

        Args:
            reuse: Attempt to reuse matching trace outputs from a previous
                frequency layout.
            force: Ignore reusable outputs and rerun all tasks.

        Returns:
            Mapping with zero-based ``pending_indices``, one-based
            ``current_tasks``, and records for any ``reused_tasks``.
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
        """Return whether all expected outputs match the saved job inputs.

        Returns:
            ``True`` when result metadata, fingerprints, and trace outputs show
            that every task is current.
        """

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

    def task_run_manifest_path(self, task: int) -> Path:
        """Return the local solver run manifest path for one task.

        Args:
            task: One-based solver task number.

        Returns:
            Path where the task-level solver ``run_manifest.json`` is expected.

        Raises:
            ValueError: If ``task`` is less than one.
        """

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

        Args:
            status: Overall status to use when writing the aggregated
                Python-side run state.

        Returns:
            Path to the written run-state file, or ``None`` if no task
            manifests were found.
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
            if source is not None and source in self.trace_manifest.packed_files:
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
