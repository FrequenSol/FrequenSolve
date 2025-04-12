import asyncio
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from frequensolve.orchestrator.config.local import LocalSiteConfig
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    SiteStatus,
    _wait_for_path,
)
from frequensolve.simulation.jobs import SimulationJob

__all__ = ["LocalSite"]


@dataclass
class LocalSite(BaseSite):
    """Local site configuration."""

    status: SiteStatus = field(default_factory=SiteStatus)
    config: LocalSiteConfig = field(init=False)
    executable: str = field(init=False)
    env: dict = field(default_factory=dict)
    _futures: list = field(default_factory=list)

    def __post_init__(self):
        self.status = SiteStatus(status="running")
        self.config = LocalSiteConfig()
        self.executable = self._get_solver_path()
        self.env = os.environ.copy()
        self.env["FREQUENSOLVE_DIR"] = os.getenv("FS_SOLVER_PATH")
        self._is_notebook = self._check_if_notebook()

    def __del__(self):
        """Cleanup when object is destroyed."""
        for future in self._futures:
            if not future.done():
                future.cancel()

    def _get_solver_path(self) -> str:
        """Get the solver path."""
        load_dotenv()
        executable = os.getenv("LOCAL_SOLVER_EXECUTABLE")
        if not executable:
            raise RuntimeError("LOCAL_SOLVER_EXECUTABLE not set in environment")
        if not Path(executable).exists():
            raise FileNotFoundError(f"Solver executable not found at {executable}")
        return executable

    def submit(self, job: SimulationJob, **kwargs) -> list:
        """Submit job and block until completion."""
        if self._is_notebook:
            import nest_asyncio

            nest_asyncio.apply()

        # Create and use a new loop for consistency
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(self._run_job(job, **kwargs))
        finally:
            loop.close()

    async def _run_job(self, job: SimulationJob, **kwargs) -> list:
        """Run all tasks for a job and return results."""
        results = []
        for i in range(job.n_tasks):
            result = await self._run_single_task(job, i, **kwargs)
            if result is not None:
                results.append(result)
        return job.records

    def sync(self, project):
        """Dummy method for consistency."""
        pass

    def _sync_project(self, project):
        """Dummy method for consistency."""
        pass

    def connect_to_existing_job(self):
        """Dummy method for consistency."""
        pass

    def submit_async(self, job: SimulationJob, **kwargs) -> asyncio.Future:
        """Submit job asynchronously and return a future."""
        if self._is_notebook:
            import nest_asyncio

            nest_asyncio.apply()

        loop = asyncio.get_event_loop()
        future = loop.create_task(self._run_job(job, **kwargs))
        self._futures.append(future)
        return future

    async def _run_single_task(
        self, job: SimulationJob, task_id: int, **kwargs
    ) -> dict:
        """Run a single task and return its results."""
        job_file = job.save()
        _wait_for_path(job_file)

        nranks = kwargs.get("nranks", 1)
        assert (
            nranks <= self.config.cores
        ), "Number of ranks must be less than or equal to the number of cores"
        nthreads = kwargs.get("nthreads", self.config.cores // nranks)

        if nranks > 1:
            args = [
                "mpirun",
                "-np",
                f"{nranks}",
            ]
        else:
            args = []

        args += [
            self.executable,
            "-nthreads",
            f"{nthreads}",
            "-j",
            f"{job_file}",
            "-i",
            f"{task_id + 1}",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
            )

            # Collect output for result
            output_lines = []

            # Read stdout and stderr concurrently
            async def read_stream(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line = line.decode().rstrip()
                    print(line, flush=True)
                    output_lines.append(line)

            # Create tasks for reading both streams
            stdout_task = asyncio.create_task(read_stream(proc.stdout))

            # Wait for process to complete and streams to be fully read
            await asyncio.gather(stdout_task)
            await proc.wait()

            if proc.returncode != 0:
                # Create task for reading stderr stream
                stderr_task = asyncio.create_task(read_stream(proc.stderr))
                await stderr_task

                raise subprocess.CalledProcessError(
                    proc.returncode, args, "\n".join(output_lines), None
                )

            return
            # return job.records

        except Exception as e:
            print(f"Task {task_id+1} failed: {str(e)}", file=sys.stderr)
            raise

        finally:
            sys.stdout.flush()
            sys.stderr.flush()

    def transfer():
        pass

    def cancel_job(self, job_id: str):
        """Cancel a running job.

        Args:
            job_id: The ID of the job to cancel
        """
        try:
            os.kill(int(job_id), signal.SIGTERM)
        except ProcessLookupError:
            pass  # Process already terminated
        except ValueError:
            raise ValueError(f"Invalid process ID: {job_id}")

    async def wait(self) -> list:
        """Wait for all submitted tasks to complete.

        Returns:
            list: Results from all completed tasks
        """
        try:
            # Wait for all futures to complete
            results = []
            for future in self._futures:
                try:
                    result = await asyncio.wrap_future(future)
                    results.extend(result)
                except Exception as e:
                    print(f"Task failed: {str(e)}", file=sys.stderr)
                    raise
            return results
        finally:
            # Clear the futures list
            self._futures.clear()

    def wait_sync(self) -> list:
        """Synchronous version of wait()."""
        if self._is_notebook:
            import nest_asyncio

            nest_asyncio.apply()

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.wait())
