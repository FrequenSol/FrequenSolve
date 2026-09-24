#!/usr/bin/env python3
"""Standalone adaptive task scheduler transferred with an HPC sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import time
from collections import deque
from pathlib import Path

_FAILURE_REASON_LIMIT = 300
_LOG_TAIL_BYTES = 64 * 1024
_LOG_ERROR_LINE = re.compile(r"^\s*(?:Error|Fortran runtime error)\s*:\s*(\S.*)$")
# error.json older than the step (beyond clock skew) belongs to an earlier run.
_STALE_ERROR_SLACK_SECONDS = 5.0
_MAX_RECORDED_REASONS = 10
_STEP_LABELS = {
    "mpi_startup": "MPI startup check",
    "init": "Mesh preparation",
    "tasks": "Adaptive scheduler",
    "smooth": "Solver postprocess",
    "pack": "Packing",
}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-sizing", nargs=2, metavar=("PATH", "TASKS"))
    parser.add_argument("--record-failure", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--job")
    parser.add_argument("--output")
    parser.add_argument("--status")
    parser.add_argument("--tasks", type=int, default=0)
    parser.add_argument("--step", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--return-code", type=int, default=1)
    parser.add_argument("--not-before", type=float, default=0.0)
    args = parser.parse_args()
    if args.record_failure:
        if args.status is None:
            parser.error("--record-failure requires --status")
    elif args.validate_sizing is None:
        missing = [
            name
            for name in ("config", "job", "output", "status")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error(
                "the following arguments are required: "
                + ", ".join(f"--{name}" for name in missing)
            )
    return args


def _parse_mem_to_gib(value: str) -> float:
    match = re.match(r"([0-9]*\.?[0-9]+)\s*(KB|MB|GB|TB)", value.strip(), re.I)
    if not match:
        raise ValueError(f"Unrecognized memory string: {value!r}")
    amount = float(match.group(1))
    unit = match.group(2).upper()
    return {
        "KB": amount / (1024**2),
        "MB": amount / 1024,
        "GB": amount,
        "TB": amount * 1024,
    }[unit]


def _load_task_memory_file(path: str, *, expected_tasks=None):
    with open(path) as file:
        tasks = json.load(file).get("task")
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit(f"Could not find non-empty JSON['task'] list in {path}")
    if expected_tasks is not None and len(tasks) < expected_tasks:
        raise SystemExit(
            f"missing task estimates in {path}: "
            f"found {len(tasks)}, expected {expected_tasks}"
        )
    memory = []
    for index, task in enumerate(tasks, start=1):
        try:
            memory.append(_parse_mem_to_gib(task["memory"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"task {index} missing valid memory estimate in {path}"
            ) from exc
    return memory


def validate_sizing_checkpoint(path: str, expected_tasks: int):
    with open(path) as file:
        payload = json.load(file)
    if payload.get("schema") != "fs-sizing-2":
        raise SystemExit(f"invalid sizing schema in {path}")
    if payload.get("sweep_status", "complete") not in {
        "forward_sweep_checkpoint",
        "complete",
    }:
        raise SystemExit(f"invalid sizing status in {path}")
    return _load_task_memory_file(path, expected_tasks=expected_tasks)


def _job_result_path(job_file):
    try:
        job_file = Path(job_file)
        with open(job_file) as file:
            data = json.load(file)
    except Exception:
        return None
    result_path = data.get("result_path") if isinstance(data, dict) else None
    if not result_path:
        return None
    path = Path(str(result_path))
    if path.is_absolute():
        return path
    project_path = data.get("project_path")
    if project_path:
        return Path(str(project_path)) / path
    return job_file.parent / path


def _step_run_dir(job_file, step):
    """Return the solver run directory where one step writes ``error.json``."""

    result_path = _job_result_path(job_file)
    if result_path is None:
        return None
    if isinstance(step, int):
        return result_path / "_fs_run" / "tasks" / f"task_{step:06d}"
    if step in {"init", "smooth", "pack"}:
        return result_path / "_fs_run" / "operations" / step
    return None


def _read_error_json(run_dir, not_before: float):
    if run_dir is None:
        return None
    path = run_dir / "error.json"
    try:
        if path.stat().st_mtime < not_before - _STALE_ERROR_SLACK_SECONDS:
            return None
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return str(data.get("message") or "").strip() or None


def _read_log_error_line(log_file):
    """Return the last solver ``Error:`` line from the tail of a step log."""

    if not log_file:
        return None
    try:
        with open(log_file, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - _LOG_TAIL_BYTES))
            tail = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        match = _LOG_ERROR_LINE.match(line)
        if match:
            return match.group(1)
    return None


def _compact_reason(text: str) -> str:
    text = " ".join(text.split())
    if len(text) > _FAILURE_REASON_LIMIT:
        text = text[: _FAILURE_REASON_LIMIT - 3].rstrip() + "..."
    return text


def _exit_description(return_code: int) -> str:
    # Shells report a signal death as 128+N; subprocess reports it as -N.
    signum = -return_code if return_code < 0 else return_code - 128
    if return_code < 0 or return_code > 128:
        try:
            return f"solver terminated by {signal.Signals(signum).name}"
        except ValueError:
            pass
    return f"solver exited with status {return_code}"


def step_failure_reason(job_file, step, log_file, return_code: int, *, not_before):
    """Return a compact reason for one failed solver step.

    Mirrors the local site: the step's ``error.json`` message, else the last
    ``Error:`` line in its log, after the exit description.
    """

    detail = _read_error_json(
        _step_run_dir(job_file, step), not_before
    ) or _read_log_error_line(log_file)
    reason = _exit_description(return_code)
    if detail:
        reason = f"{reason}: {detail}"
    return _compact_reason(reason)


def _write_json_atomic(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as file:
        json.dump(payload, file, separators=(",", ":"))
        file.write("\n")
    os.replace(temporary, path)


def record_step_failure(
    status_file,
    *,
    job_file,
    n_tasks: int,
    step: str,
    log_file: str,
    return_code: int,
    not_before: float,
):
    """Mark the sweep failed and name the failing step and its reason.

    A reason already recorded by the scheduler or MPI check is kept.
    """

    status_file = Path(status_file)
    try:
        with open(status_file) as file:
            payload = json.load(file)
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    for key, value in (
        ("total", n_tasks),
        ("successful", 0),
        ("failed", 0),
        ("running", 0),
        ("pending", n_tasks),
        ("complete", 0),
    ):
        payload.setdefault(key, value)
    payload["state"] = "failed"
    payload["return_code"] = return_code
    if step:
        payload.setdefault("phase", step)
    if not payload.get("abort_reason") and step:
        label = _STEP_LABELS.get(step, step)
        if step == "tasks":
            reason = f"exited with status {return_code}"
        elif job_file:
            reason = step_failure_reason(
                job_file, step, log_file, return_code, not_before=not_before
            )
        else:
            reason = _exit_description(return_code)
        payload["abort_reason"] = f"{label} failed: {reason}" + (
            f"; see {Path(log_file).name}" if log_file else ""
        )
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json_atomic(status_file, payload)


def _round_up(value: int, base: int) -> int:
    return value if base <= 1 else int(math.ceil(value / base) * base)


def _total_free_ranks(intervals) -> int:
    return sum(length for _, length in intervals)


def _allocate_interval(intervals, ranks: int):
    for index, (start, length) in enumerate(intervals):
        if length < ranks:
            continue
        updated = list(intervals)
        if length == ranks:
            del updated[index]
        else:
            updated[index] = (start + ranks, length - ranks)
        return start, updated
    return None, intervals


def _free_interval(intervals, offset: int, ranks: int):
    merged = []
    for start, length in sorted(intervals + [(offset, ranks)]):
        end = start + length
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end - start) for start, end in merged]


class AdaptiveScheduler:
    def __init__(self, config, *, job_file: str, output: str, status: str):
        self.config = config
        self.job_file = job_file
        self.output = Path(output)
        self.status_file = Path(status)
        self.executable = str(config["executable"])
        self.mpi = str(config.get("mpi", "ibrun"))
        self.mpi_args = [str(value) for value in config.get("mpi_args", [])]
        self.mpi_launcher = Path(self.mpi).name
        self.fresh_flag = ["--fresh"] if config.get("fresh") else []
        self.total_ranks = int(config["total_ranks"])
        self.omp_threads = int(config["omp_threads"])
        self.mem_per_rank = float(config["mem_per_rank_gib"])
        self.job_task_count = int(config.get("job_task_count", 0))
        self.task_indices = [int(value) for value in config.get("task_indices", [])]
        self.skip_sizing = bool(config.get("skip_sizing", False))
        self.min_ranks = int(config.get("min_ranks", 1))
        self.round_to = int(config.get("round_to", 1))
        self.cap_fraction = float(config.get("cap_fraction", 1.0))
        self.mem_cushion = float(config.get("mem_cushion", 1.5))
        self.boost_max_factor = float(config.get("boost_max_factor", 8.0))
        self.failure_tolerance = config.get("failure_tolerance")
        if self.failure_tolerance is not None:
            self.failure_tolerance = int(self.failure_tolerance)
        self.sizing_json = str(config.get("sizing_json", "FS_sizing.json"))
        # A frequency postprocess consumes every task, so any failure is fatal.
        self.require_all_tasks = bool(config.get("require_all_tasks", False))
        self.launch_delay = float(config.get("launch_delay_seconds", 0.25))
        self.max_ranks_per_task = (
            self.total_ranks
            if self.skip_sizing
            else max(1, int(self.total_ranks * self.cap_fraction))
        )
        rank_limit = int(config.get("max_ranks_per_task", self.total_ranks))
        if rank_limit < 1 or self.min_ranks > rank_limit:
            raise ValueError("max_ranks_per_task must be positive and >= min_ranks")
        self.max_ranks_per_task = min(self.max_ranks_per_task, rank_limit)
        if self.mpi_launcher in {"mpiexec", "mpirun"}:
            # These launchers cannot isolate concurrent steps within an allocation.
            self.max_ranks_per_task = min(self.total_ranks, rank_limit)
        self.running = []
        self.successful_tasks = []
        self.failed_tasks = []
        self.failed_reasons = {}
        self.launched_at = {}
        self.free_intervals = [(0, self.total_ranks)]
        if self.mpi_launcher in {"mpiexec", "mpirun"}:
            self.free_intervals = [(0, self.max_ranks_per_task)]

    def _load_task_memory(self):
        if self.skip_sizing:
            return [0.0] * max(self.job_task_count, max(self.task_indices, default=1))
        return _load_task_memory_file(self.sizing_json)

    def _choose_base_ranks(self, task_memory: float) -> int:
        if self.mpi_launcher in {"mpiexec", "mpirun"}:
            return self.max_ranks_per_task
        ranks = int(math.ceil(task_memory * self.mem_cushion / self.mem_per_rank))
        ranks = max(ranks, self.min_ranks)
        ranks = min(ranks, self.max_ranks_per_task, self.total_ranks)
        ranks = _round_up(ranks, self.round_to)
        return min(
            max(ranks, self.min_ranks), self.total_ranks, self.max_ranks_per_task
        )

    def _write_status(self, state: str, *, aborted=None, reason=None):
        running = [entry[3] for entry in self.running]
        complete = len(self.successful_tasks) + len(self.failed_tasks)
        payload = {
            "state": state,
            "total": len(self.task_indices),
            "successful": len(self.successful_tasks),
            "failed": len(self.failed_tasks),
            "running": len(running),
            "pending": max(0, len(self.task_indices) - complete - len(running)),
            "complete": complete,
            "running_tasks": running,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "aborted_tasks": list(aborted or []),
            "tolerate_failures": self.failure_tolerance,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if self.failed_reasons:
            payload["failed_reasons"] = {
                str(task): reason for task, reason in self.failed_reasons.items()
            }
        if reason:
            payload["abort_reason"] = reason
        _write_json_atomic(self.status_file, payload)

    def _launch_command(self, task_id: int, offset: int, ranks: int) -> list[str]:
        if self.mpi_launcher == "ibrun":
            prefix = [self.mpi, "-n", str(ranks), "-o", str(offset), "task_affinity"]
        elif self.mpi_launcher == "srun":
            prefix = [
                self.mpi,
                "--exclusive",
                "--ntasks",
                str(ranks),
                "--cpus-per-task",
                str(self.omp_threads),
            ]
        elif self.mpi_launcher in {"mpiexec", "mpirun"}:
            if offset != 0 or ranks != self.max_ranks_per_task:
                raise SystemExit(
                    f"{self.mpi_launcher} task launch requires exclusive use of the allocation"
                )
            prefix = [self.mpi, "-n", str(ranks)]
        else:
            raise SystemExit(f"unsupported MPI launcher: {self.mpi_launcher}")
        return [
            prefix[0],
            *self.mpi_args,
            *prefix[1:],
            self.executable,
            "-nthreads",
            str(self.omp_threads),
            "--job",
            self.job_file,
            *self.fresh_flag,
            "--task",
            str(task_id),
        ]

    def _record_task_failure(self, task: int, return_code: int) -> None:
        reason = step_failure_reason(
            self.job_file,
            task,
            self.output / f"task_{task}.log",
            return_code,
            not_before=self.launched_at.get(task, 0.0),
        )
        print(f"[scheduler] task={task} failed: {reason}", flush=True)
        if len(self.failed_reasons) < _MAX_RECORDED_REASONS:
            self.failed_reasons[task] = reason

    def _launch(self, task_id: int, offset: int, ranks: int, memory: float):
        self.launched_at[task_id] = time.time()
        output = open(self.output / f"task_{task_id}.log", "ab", buffering=0)
        command = self._launch_command(task_id, offset, ranks)
        try:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        except Exception:
            output.close()
            raise
        self.running.append((process, offset, ranks, task_id, memory, output))
        print(
            f"[scheduler] launch task={task_id} mem≈{memory:.2f}GiB "
            f"offset={offset} ranks={ranks}",
            flush=True,
        )
        time.sleep(self.launch_delay)
        self._write_status("running")

    def _abort(self, reason: str):
        aborted = []
        for process, _, _, task_id, _, output in self.running:
            aborted.append(task_id)
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            finally:
                try:
                    output.close()
                except Exception:
                    pass
        self.running = []
        self._write_status("failed", aborted=aborted, reason=reason)
        raise SystemExit(reason)

    def run(self):
        memory = self._load_task_memory()
        if self.job_task_count <= 0:
            self.job_task_count = len(memory)
        if len(memory) < self.job_task_count:
            raise SystemExit(
                f"missing task estimates in {self.sizing_json}: "
                f"found {len(memory)}, expected {self.job_task_count}"
            )
        if not self.task_indices:
            self.task_indices = list(range(1, self.job_task_count + 1))
        bad = [
            task for task in self.task_indices if task < 1 or task > self.job_task_count
        ]
        if bad:
            raise SystemExit(f"invalid task indices: {bad}")
        if self.skip_sizing and len(self.task_indices) != 1:
            raise SystemExit(
                "--init-no-size scheduling requires exactly one submitted task"
            )

        queue = deque(
            sorted(self.task_indices, key=lambda task: memory[task - 1], reverse=True)
        )
        self._write_status("running")
        while queue or self.running:
            launched = False
            free_ranks = _total_free_ranks(self.free_intervals)
            remaining_memory = sum(memory[task - 1] for task in queue) + sum(
                entry[4] for entry in self.running
            )
            candidates = []
            remaining = len(queue) + len(self.running)
            for task in list(queue):
                base = self._choose_base_ranks(memory[task - 1])
                maximum = base
                if len(queue) == 1:
                    maximum = free_ranks
                elif remaining < self.total_ranks and remaining_memory > 0:
                    share = int(
                        math.ceil(
                            self.total_ranks * memory[task - 1] / remaining_memory
                        )
                    )
                    maximum = min(
                        max(base, share),
                        int(math.ceil(max(2.0, self.boost_max_factor) * base)),
                    )
                minimum = min(
                    max(base, self.min_ranks), self.total_ranks, self.max_ranks_per_task
                )
                maximum = min(
                    max(maximum, minimum), self.max_ranks_per_task, self.total_ranks
                )
                candidates.append((task, minimum, maximum, memory[task - 1]))

            simulated = list(self.free_intervals)
            plan = []
            chosen = set()

            boost_pool = sorted(
                candidates, key=lambda item: (item[2], item[3]), reverse=True
            )
            while True:
                best = None
                best_metric = None
                for task, _, maximum, task_memory in boost_pool:
                    if task in chosen:
                        continue
                    offset, updated = _allocate_interval(simulated, maximum)
                    if offset is None:
                        continue
                    metric = (
                        _total_free_ranks(updated),
                        max((length for _, length in updated), default=0),
                    )
                    if best is None or metric < best_metric:
                        best = (task, offset, maximum, task_memory, updated)
                        best_metric = metric
                if best is None:
                    break
                task, offset, maximum, task_memory, simulated = best
                plan.append((task, offset, maximum, task_memory))
                chosen.add(task)

            for task, minimum, _, task_memory in sorted(
                (item for item in candidates if item[0] not in chosen),
                key=lambda item: item[1],
                reverse=True,
            ):
                offset, updated = _allocate_interval(simulated, minimum)
                if offset is not None:
                    plan.append((task, offset, minimum, task_memory))
                    simulated = updated

            for task, _, ranks, task_memory in plan:
                queue.remove(task)
                offset, self.free_intervals = _allocate_interval(
                    self.free_intervals, ranks
                )
                if offset is None:
                    raise RuntimeError("Adaptive scheduler allocation mismatch")
                self._launch(task, offset, ranks, task_memory)
                launched = True

            if self.running:
                time.sleep(1)
                active = []
                for process, offset, ranks, task, task_memory, output in self.running:
                    return_code = process.poll()
                    if return_code is None:
                        active.append(
                            (process, offset, ranks, task, task_memory, output)
                        )
                        continue
                    output.close()
                    target = (
                        self.successful_tasks if return_code == 0 else self.failed_tasks
                    )
                    target.append(task)
                    if return_code != 0:
                        self._record_task_failure(task, return_code)
                    self.free_intervals = _free_interval(
                        self.free_intervals, offset, ranks
                    )
                self.running = active
                if (
                    self.failure_tolerance is not None
                    and len(self.failed_tasks) > self.failure_tolerance
                ):
                    self._abort(
                        f"failure tolerance exceeded: {len(self.failed_tasks)} "
                        f"failed tasks (tolerate_failures={self.failure_tolerance})"
                    )
                self._write_status(
                    "running"
                    if queue or self.running
                    else ("failed" if self.failed_tasks else "complete")
                )
            elif not launched:
                self._write_status("running")
                time.sleep(1)

        if self.failed_tasks and self.require_all_tasks:
            failed = ", ".join(str(task) for task in sorted(self.failed_tasks))
            reason = (
                f"{len(self.failed_tasks)} of {len(self.task_indices)} tasks failed "
                f"(tasks {failed}); solver postprocess not started"
            )
            self._write_status("failed", reason=reason)
            raise SystemExit(reason)
        self._write_status("failed" if self.failed_tasks else "complete")
        print("[scheduler] all tasks done", flush=True)


def main():
    args = _parse_args()
    if args.record_failure:
        record_step_failure(
            args.status,
            job_file=args.job,
            n_tasks=args.tasks,
            step=args.step,
            log_file=args.log,
            return_code=args.return_code,
            not_before=args.not_before,
        )
        return
    if args.validate_sizing is not None:
        path, expected_tasks = args.validate_sizing
        validate_sizing_checkpoint(path, int(expected_tasks))
        return
    with open(args.config) as file:
        config = json.load(file)
    AdaptiveScheduler(
        config,
        job_file=args.job,
        output=args.output,
        status=args.status,
    ).run()


if __name__ == "__main__":
    main()
