#!/usr/bin/env python3
"""Standalone adaptive task scheduler transferred with an HPC sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from collections import deque
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status", required=True)
    return parser.parse_args()


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
        self.launch_delay = float(config.get("launch_delay_seconds", 0.25))
        self.max_ranks_per_task = (
            self.total_ranks
            if self.skip_sizing
            else max(1, int(self.total_ranks * self.cap_fraction))
        )
        self.running = []
        self.successful_tasks = []
        self.failed_tasks = []
        self.free_intervals = [(0, self.total_ranks)]

    def _load_task_memory(self):
        if self.skip_sizing:
            return [0.0] * max(self.job_task_count, max(self.task_indices, default=1))
        with open(self.sizing_json) as file:
            tasks = json.load(file).get("task")
        if not isinstance(tasks, list) or not tasks:
            raise SystemExit(
                f"Could not find non-empty JSON['task'] list in {self.sizing_json}"
            )
        return [_parse_mem_to_gib(task["memory"]) for task in tasks]

    def _choose_base_ranks(self, task_memory: float) -> int:
        ranks = int(math.ceil(task_memory * self.mem_cushion / self.mem_per_rank))
        ranks = max(ranks, self.min_ranks)
        ranks = min(ranks, self.max_ranks_per_task, self.total_ranks)
        ranks = _round_up(ranks, self.round_to)
        return min(max(ranks, self.min_ranks), self.total_ranks)

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
        if reason:
            payload["abort_reason"] = reason
        temporary = self.status_file.with_suffix(self.status_file.suffix + ".tmp")
        with open(temporary, "w") as file:
            json.dump(payload, file, separators=(",", ":"))
            file.write("\n")
        os.replace(temporary, self.status_file)

    def _launch(self, task_id: int, offset: int, ranks: int, memory: float):
        output = open(self.output / f"task_{task_id}.log", "ab", buffering=0)
        command = [
            self.mpi,
            "-n",
            str(ranks),
            "-o",
            str(offset),
            "task_affinity",
            self.executable,
            "-nthreads",
            str(self.omp_threads),
            "--job",
            self.job_file,
            *self.fresh_flag,
            "--task",
            str(task_id),
        ]
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
                minimum = min(max(base, self.min_ranks), self.total_ranks)
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

        self._write_status("failed" if self.failed_tasks else "complete")
        print("[scheduler] all tasks done", flush=True)


def main():
    args = _parse_args()
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
