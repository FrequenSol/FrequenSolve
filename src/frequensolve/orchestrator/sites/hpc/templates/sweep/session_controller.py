#!/usr/bin/env python3
"""Allocation-resident controller; requires only Python's standard library.

Requests and commands are immutable JSON files, published by atomic rename.
One controller owns the entire allocation. Solver processes are disposable;
completed requests and their status survive client disconnections.
"""

from __future__ import annotations

import argparse
import faulthandler
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from adaptive_scheduler import (
    AdaptiveScheduler,
    _allocate_interval,
    _free_interval,
    _write_json_atomic,
    step_failure_reason,
    validate_sizing_checkpoint,
)

TERMINAL = {"complete", "failed", "cancelled", "timeout"}


class SessionController:
    def __init__(self, root, config):
        self.root = Path(root)
        self.config = config
        self.generation = config["generation"]
        self.total = int(config["total_ranks"])
        self.free = [(0, self.total)]
        self.runs = {}
        self.running = []
        self.seen_commands = set()
        self.closing = False
        self.stop_reason = None
        self.started = time.monotonic()
        self.interval = float(config.get("poll_interval", 0.2))
        self.progress_at = time.monotonic()
        for name in ("requests", "commands", "leases", "runs"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def _status(self, run, state=None, reason=None):
        if state is not None:
            run["state"] = state
        if reason is not None:
            run["reason"] = reason
        rows = run["rows"]
        tasks = [row for row in rows if isinstance(row["step"], int)]
        payload = {
            "generation": self.generation,
            "allocation_id": os.environ.get("SLURM_JOB_ID"),
            "run_id": run["id"],
            "state": run["state"],
            "reason": run.get("reason", ""),
            "updated_at": time.time(),
            "tasks": tasks,
            "steps": rows,
            "worker_seconds": sum(row["duration_seconds"] for row in rows),
            "total": len(run.get("task_indices", [])),
            "successful": sum(row["return_code"] == 0 for row in tasks),
            "failed": sum(row["return_code"] != 0 for row in tasks),
            "running": sum(entry["run"] is run for entry in self.running),
            "task_ranks": {str(row["step"]): row["ranks"] for row in tasks},
        }
        payload["complete"] = payload["successful"] + payload["failed"]
        payload["pending"] = max(
            0, payload["total"] - payload["complete"] - payload["running"]
        )
        _write_json_atomic(self.root / "runs" / run["id"] / "status.json", payload)
        if run.get("logs"):
            _write_json_atomic(Path(run["logs"]) / "scheduler_status.json", payload)

    def _controller_status(self, state):
        _write_json_atomic(
            self.root / "status.json",
            {
                "schema": "fs-allocation-session-1",
                "generation": self.generation,
                "allocation_id": os.environ.get("SLURM_JOB_ID"),
                "state": state,
                "updated_at": time.time(),
                "reason": self.stop_reason or "",
                "active_runs": sum(
                    run["state"] not in TERMINAL for run in self.runs.values()
                ),
            },
        )

    def _admit(self, path):
        request = json.loads(path.read_text())
        run_id = path.stem
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            return
        if run_id in self.runs:
            return
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        run = {
            "id": run_id,
            "digest": digest,
            "state": "pending",
            "rows": [],
            "phase": "init",
        }
        self.runs[run_id] = run
        (self.root / "runs" / run_id).mkdir(exist_ok=True)
        try:
            if (
                request.get("schema") != "fs-session-request-1"
                or request.get("generation") != self.generation
            ):
                raise ValueError("request schema or allocation generation mismatch")
            if self.closing:
                raise ValueError("allocation session is draining")
            allowed = {
                "schema",
                "generation",
                "kind",
                "job_file",
                "logs",
                "cwd",
                "scheduler",
                "postprocess",
                "postprocess_ranks",
                "pack",
                "ranks",
                "request_file",
            }
            if set(request) - allowed:
                raise ValueError("unsupported request fields")
            run.update(request)
            run["id"] = run_id
            run["logs"] = str(request["logs"])
            if any(
                other is not run
                and other["state"] not in TERMINAL
                and other.get("logs") == run["logs"]
                for other in self.runs.values()
            ):
                # Do not overwrite another run's output or status directory.
                run.pop("logs")
                raise ValueError("another active run owns this job/output directory")
            Path(run["logs"]).mkdir(parents=True, exist_ok=True)
            if request["kind"] == "job":
                if (
                    not 1
                    <= int(request.get("postprocess_ranks", self.total))
                    <= self.total
                ):
                    raise ValueError("postprocess ranks exceed the allocation")
                options = dict(self.config["scheduler"], **request["scheduler"])
                # Clients cannot expand or change the allocation/launcher.
                for key in (
                    "total_ranks",
                    "omp_threads",
                    "mem_per_rank_gib",
                    "mpi",
                    "mpi_args",
                    "executable",
                ):
                    options[key] = self.config["scheduler"][key]
                run["scheduler"] = options
                run["adaptive"] = AdaptiveScheduler(
                    options,
                    job_file=request["job_file"],
                    output=run["logs"],
                    status=str(self.root / "runs" / run_id / "adaptive.json"),
                    rank_binding=self.config.get("rank_binding", "explicit"),
                )
                run["task_indices"] = list(run["adaptive"].task_indices)
                if len(set(run["task_indices"])) != len(run["task_indices"]) or any(
                    task < 1 or task > int(options["job_task_count"])
                    for task in run["task_indices"]
                ):
                    raise ValueError("invalid frequency task indices")
                if (
                    options["skip_sizing"]
                    and len(run["task_indices"]) > 1
                    and not set(run["task_indices"]) <= set(run["adaptive"].task_ranks)
                ):
                    raise ValueError(
                        "skipping sizing requires a single task or pinned ranks"
                    )
                if not run["task_indices"]:
                    run["phase"] = "smooth" if request.get("postprocess") else "pack"
            elif request["kind"] == "curvature":
                run["phase"] = "curvature"
                if not 1 <= int(request["ranks"]) <= self.total:
                    raise ValueError("curvature ranks exceed the allocation")
            else:
                raise ValueError("unsupported session request kind")
            self._status(run)
        except (ValueError, TypeError, KeyError, SystemExit) as error:
            self._status(run, "failed", str(error))

    def _receive(self):
        for path in sorted((self.root / "requests").glob("*.json")):
            if path.stem not in self.runs:
                try:
                    self._admit(path)
                except (ValueError, OSError):
                    # Malformed JSON is published as a failed request, not a
                    # fatal controller error. Atomic publication avoids partials.
                    run = {"id": path.stem, "state": "failed", "rows": []}
                    self.runs[path.stem] = run
                    (self.root / "runs" / path.stem).mkdir(exist_ok=True)
                    self._status(run, reason="invalid request JSON")
        for path in sorted((self.root / "commands").glob("*.json")):
            if path.name in self.seen_commands:
                continue
            self.seen_commands.add(path.name)
            try:
                command = json.loads(path.read_text())
                if command.get("generation") != self.generation:
                    continue
                if command["kind"] == "cancel":
                    run = self.runs.get(command["run_id"])
                    if run is not None:
                        self._cancel(run)
                elif command["kind"] == "close":
                    self.closing = True
                    if not command.get("drain", True):
                        for run in self.runs.values():
                            self._cancel(run)
            except (ValueError, KeyError, OSError):
                continue

    def _prefix(self, ranks, offset):
        options = self.config["scheduler"]
        mpi = options["mpi"]
        name = Path(mpi).name
        args = [mpi, *options.get("mpi_args", [])]
        if name == "ibrun":
            return [*args, "-n", str(ranks), "-o", str(offset), "task_affinity"]
        if name == "srun":
            # Slurm chooses non-overlapping step CPUs. Node count is explicit
            # so a small step does not inherit the entire allocation's nodes.
            nodes = min(
                int(self.config["nodes"]),
                math.ceil(ranks / int(self.config["ranks_per_node"])),
            )
            return [
                *args,
                "--exclusive",
                "--exact",
                "--nodes",
                str(nodes),
                "--ntasks",
                str(ranks),
                "--cpus-per-task",
                str(options["omp_threads"]),
            ]
        args += ["-n", str(ranks)]
        if self.config.get("rank_binding") == "cores":
            args += [
                "--map-by",
                f"slot:PE={options['omp_threads']}",
                "--bind-to",
                "core",
            ]
        return args

    def _launch(self, run, step, ranks):
        # Generic MPI launchers lack an offset/step isolation mechanism.
        reserve = (
            self.total
            if Path(self.config["scheduler"]["mpi"]).name in {"mpirun", "mpiexec"}
            else ranks
        )
        # Auxiliary operations do not provide reliable per-node memory
        # estimates yet. Give them exclusive use of the allocation, while
        # allowing the solver to use fewer ranks when its contract requires it.
        if not isinstance(step, int):
            reserve = self.total
        offset, free = _allocate_interval(self.free, reserve)
        if offset is None:
            return False
        options = self.config["scheduler"]
        command = [
            *self._prefix(ranks, offset),
            options["executable"],
            "-nthreads",
            str(options["omp_threads"]),
        ]
        if step == "curvature":
            command += ["--curvature", run["request_file"]]
            log = Path(run["logs"]) / "solver.log"
        else:
            command += ["--job", run["job_file"]]
            fresh = ["--fresh"] if run["scheduler"].get("fresh") else []
            if isinstance(step, int):
                command += [*fresh, "--task", str(step)]
                log = Path(run["logs"]) / f"task_{step}.log"
            else:
                log = Path(run["logs"]) / f"{step}.log"
                if step == "init":
                    if run["scheduler"]["skip_sizing"]:
                        command += [*fresh, "--init-no-size"]
                    else:
                        command += [
                            "--fresh",
                            "--init",
                            "--sizing",
                            run["scheduler"]["sizing_json"],
                        ]
                elif step == "smooth":
                    command += [*fresh, "--smooth"]
                elif step == "pack":
                    command += ["--fresh", "--pack"]
        if step == "init" and not run["scheduler"]["skip_sizing"]:
            Path(run["scheduler"]["sizing_json"]).unlink(missing_ok=True)
        output = log.open("wb")
        started = time.time()
        environment = dict(os.environ, OMP_NUM_THREADS=str(options["omp_threads"]))
        try:
            process = subprocess.Popen(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                cwd=run["cwd"],
                env=environment,
                start_new_session=True,
            )
        except Exception as error:
            output.close()
            self._cancel(run, "failed", str(error))
            return True
        self.free = free
        self.running.append(
            {
                "process": process,
                "run": run,
                "step": step,
                "offset": offset,
                "reserve": reserve,
                "ranks": ranks,
                "started": started,
                "log": log,
                "output": output,
            }
        )
        if step == "init":
            run["phase"] = "initializing"
        elif not isinstance(step, int):
            run["phase"] = "finishing"
        self._status(run, "running")
        return True

    def _stop_process(self, entry):
        process = entry["process"]
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except PermissionError:
            # Some local sandboxes allow signaling a child but not a process
            # group. Remote MPI launchers normally propagate termination.
            process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except PermissionError:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=2)

    def _cancel(self, run, state="cancelled", reason="Cancelled by client"):
        if run["state"] in TERMINAL:
            return
        for entry in list(self.running):
            if entry["run"] is run:
                self._stop_process(entry)
                entry["output"].close()
                self.free = _free_interval(self.free, entry["offset"], entry["reserve"])
                self.running.remove(entry)
        self._status(run, state, reason)

    def _finished(self, entry, code):
        run, step = entry["run"], entry["step"]
        run["rows"].append(
            {
                "step": step,
                "ranks": entry["ranks"],
                "offset": entry["offset"],
                "return_code": code,
                "stdout": str(entry["log"]),
                "duration_seconds": time.time() - entry["started"],
            }
        )
        if step == "init":
            try:
                adaptive = run["adaptive"]
                memory = (
                    [0.0] * int(run["scheduler"]["job_task_count"])
                    if adaptive.skip_sizing
                    else validate_sizing_checkpoint(
                        adaptive.sizing_json, int(run["scheduler"]["job_task_count"])
                    )
                )
                if code and adaptive.skip_sizing:
                    raise ValueError("initialization failed")
                pending = []
                for task in run["task_indices"]:
                    required = math.ceil(
                        memory[task - 1] * adaptive.mem_cushion / adaptive.mem_per_rank
                    )
                    ranks = adaptive._choose_base_ranks(memory[task - 1], task)
                    if required > ranks or required > adaptive.max_ranks_per_task:
                        raise ValueError(
                            f"task {task} does not fit its rank/memory limit; increase allocation or worker rank limit"
                        )
                    pending.append((task, ranks))
                run["pending"] = pending
                run["phase"] = "tasks"
                return
            except (ValueError, OSError, KeyError, SystemExit) as error:
                reason = str(error)
                if code:
                    failure = step_failure_reason(
                        job_file=run["job_file"],
                        step=step,
                        log_file=str(entry["log"]),
                        return_code=code,
                        not_before=entry["started"],
                    )
                    reason = f"{failure}; {reason}"
                self._cancel(run, "failed", reason)
                return
        if code:
            reason = step_failure_reason(
                job_file=run.get("job_file", ""),
                step=step,
                log_file=str(entry["log"]),
                return_code=code,
                not_before=entry["started"],
            )
            self._cancel(run, "failed", reason)
            return
        if step == "curvature" or step == "pack":
            self._status(run, "complete")
        elif step == "smooth":
            run["phase"] = "pack"

    def tick(self):
        self._receive()
        for entry in list(self.running):
            if entry not in self.running:
                continue
            code = entry["process"].poll()
            if code is None:
                continue
            entry["output"].close()
            self.running.remove(entry)
            self.free = _free_interval(self.free, entry["offset"], entry["reserve"])
            self._finished(entry, code)
        # Initialize a submitted family before sharing ranks across its
        # frequency tasks; unknown initialization memory stays exclusive.
        initializing = any(
            run["state"] not in TERMINAL and run["phase"] in {"init", "initializing"}
            for run in self.runs.values()
        )
        remaining = sum(
            len(run.get("pending", []))
            for run in self.runs.values()
            if run["state"] not in TERMINAL
        ) + sum(isinstance(entry["step"], int) for entry in self.running)
        for run in self.runs.values():
            if run["state"] in TERMINAL:
                continue
            phase = run["phase"]
            if phase == "init":
                self._launch(run, "init", run["adaptive"].max_ranks_per_task)
            elif phase == "curvature":
                self._launch(run, "curvature", int(run["ranks"]))
            elif phase == "tasks":
                if initializing:
                    continue
                for task, base in list(run["pending"]):
                    adaptive = run["adaptive"]
                    ranks = base
                    if task not in adaptive.task_ranks:
                        share = (
                            self.total
                            if adaptive.mpi_launcher in {"mpirun", "mpiexec"}
                            else max(1, self.total // max(1, remaining))
                        )
                        desired = min(adaptive.max_ranks_per_task, max(base, share))
                        if _allocate_interval(self.free, desired)[0] is not None:
                            ranks = desired
                    if self._launch(run, task, ranks):
                        run["pending"].remove((task, base))
                    if run["state"] in TERMINAL:
                        break
                if not run["pending"] and not any(
                    entry["run"] is run for entry in self.running
                ):
                    run["phase"] = "smooth" if run.get("postprocess") else "pack"
            elif phase == "smooth":
                self._launch(
                    run, "smooth", int(run.get("postprocess_ranks", self.total))
                )
            elif phase == "pack":
                if run.get("pack"):
                    self._launch(run, "pack", 1)
                else:
                    self._status(run, "complete")
            if run["state"] not in TERMINAL:
                self._status(run)
        self._controller_status("draining" if self.closing else "ready")

    def _lease_alive(self):
        for path in (self.root / "leases").glob("*.json"):
            try:
                lease = json.loads(path.read_text())
                if (
                    lease.get("generation") == self.generation
                    and float(lease["expires_at"]) > time.time()
                ):
                    return True
            except (ValueError, KeyError, OSError):
                continue
        return False

    def run(self):
        # A second controller cannot accidentally execute the same requests.
        with (self.root / "controller.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                while True:
                    self.tick()
                    self.progress_at = time.monotonic()
                    if self.stop_reason:
                        for run in self.runs.values():
                            self._cancel(
                                run,
                                (
                                    "timeout"
                                    if self.stop_reason == "walltime"
                                    else "cancelled"
                                ),
                                self.stop_reason,
                            )
                        break
                    if self.closing and all(
                        run["state"] in TERMINAL for run in self.runs.values()
                    ):
                        break
                    if (
                        time.monotonic() - self.started
                        > float(self.config["lease_timeout"])
                        and not self._lease_alive()
                    ):
                        self.stop_reason = "client lease expired"
                    time.sleep(self.interval)
            except BaseException:
                self.stop_reason = "controller failed"
                for run in self.runs.values():
                    self._cancel(run, "failed", self.stop_reason)
                self._controller_status("failed")
                raise
            self._controller_status("closed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    config = json.loads((root / "config.json").read_text())
    if Path(config["scheduler"]["mpi"]).name in {"mpirun", "mpiexec"}:
        try:
            version = subprocess.check_output(
                [config["scheduler"]["mpi"], "--version"],
                timeout=10,
                stderr=subprocess.STDOUT,
            ).decode()
        except (OSError, subprocess.SubprocessError):
            version = ""
        if os.uname().sysname == "Linux" and "Open MPI" in version:
            config["rank_binding"] = "cores"
        else:
            os.environ.setdefault("OMPI_MCA_hwloc_base_binding_policy", "none")
            os.environ.setdefault("PRTE_MCA_hwloc_default_binding_policy", "none")
    controller = SessionController(root, config)

    def diagnose_stall():
        reported = False
        while True:
            time.sleep(5)
            stalled = time.monotonic() - controller.progress_at > 60
            if stalled and not reported:
                print(
                    "Controller progress stalled; Python traceback follows", flush=True
                )
                faulthandler.dump_traceback()
            reported = stalled

    threading.Thread(target=diagnose_stall, daemon=True).start()

    def stop(signum, _frame):
        controller.stop_reason = (
            "walltime" if signum == signal.SIGUSR1 else "allocation cancelled"
        )
        controller.closing = True

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    controller.run()


if __name__ == "__main__":
    main()
