"""Opt-in placement check inside an existing, otherwise idle Slurm allocation.

No batch submission or solver is involved. FS_SLURM_PROBE_DIR must point to a
shared directory visible at the same absolute path from every allocated node.
"""

import importlib.util
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import pytest

from frequensolve.orchestrator.sites.hpc import site

pytestmark = pytest.mark.hpc


@pytest.mark.skipif(
    os.environ.get("FS_SLURM_PLACEMENT_PROBE") != "1",
    reason="requires explicit approval and an idle Slurm allocation",
)
def test_live_concurrent_slurm_placement():
    shared = os.environ["FS_SLURM_PROBE_DIR"]
    path = Path(site.__file__).parent / "templates/sweep/adaptive_scheduler.py"
    spec = importlib.util.spec_from_file_location("placement_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ranks = int(os.environ["SLURM_NTASKS"])
    assert ranks >= 3, "probe needs at least three allocated rank slots"
    with tempfile.TemporaryDirectory(prefix="fs-placement-", dir=shared) as directory:
        root = Path(directory)
        probe = root / "probe.sh"
        probe.write_text('#!/bin/sh\nprintf "%s\\n" "$SLURMD_NODENAME"\nsleep 2\n')
        probe.chmod(0o700)
        scheduler = module.AdaptiveScheduler(
            {
                "executable": str(probe),
                "mpi": "srun",
                "total_ranks": ranks,
                "omp_threads": int(os.environ["SLURM_CPUS_PER_TASK"]),
                "mem_per_rank_gib": 1,
                "task_indices": [1, 2],
                "mpi_args": ["--cpu-bind=cores", "--kill-on-bad-exit=1", "--wait=10"],
            },
            job_file="unused.json",
            output=directory,
            status=str(root / "status.json"),
        )
        try:
            scheduler._launch(1, 0, 1, 0)
            scheduler._launch(2, 1, 2, 0)
            deadline = time.monotonic() + 45
            while any(entry[0].poll() is None for entry in scheduler.running):
                assert time.monotonic() < deadline, "Slurm placement probe timed out"
                scheduler._write_status("running")
                time.sleep(0.2)
            scheduler._write_status("running")
            for process, _, _, task, _, output in scheduler.running:
                output.close()
                log = (root / f"task_{task}.log").read_text()
                assert process.returncode == 0, log
                expected = Counter((root / f"task_{task}.hosts").read_text().split())
                observed = Counter(
                    line for line in log.splitlines() if line in expected
                )
                assert observed == expected, log
                assert task in scheduler._started_steps
        finally:
            for process, _, _, _, _, output in scheduler.running:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except module.subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                output.close()
