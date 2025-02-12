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

    def submit(self, job: SimulationJob) -> list:
        """Submit job and block until completion."""
        if self._is_notebook:
            import nest_asyncio

            nest_asyncio.apply()

        # Use existing loop if available, otherwise create new one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            future = self.submit_async(job)
            return loop.run_until_complete(future)
        finally:
            # Don't close the loop if we didn't create it
            if not loop.is_running():
                loop.close()

    def submit_async(self, job: SimulationJob) -> asyncio.Future:
        """Submit job asynchronously and return a future."""
        # Use the current event loop
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        async def run_tasks():
            try:
                results = []
                for i in range(job.n_tasks):
                    print(f"Running task {i+1}")
                    result = await self._run_single_task(job, i)

                    # TODO: once HPC side matches, update this
                    if result is not None:
                        results.append(result)
                future.set_result(job.records)
            except Exception as e:
                future.set_exception(e)

        if self._is_notebook:
            import nest_asyncio

            nest_asyncio.apply()

        loop.create_task(run_tasks())
        self._futures.append(future)
        return future

    async def _run_single_task(self, job: SimulationJob, task_id: int) -> dict:
        """Run a single task and return its results."""
        job_file = job.save()
        _wait_for_path(job_file)

        args = [
            "mpirun",
            "-np",
            "2",
            self.executable,
            "-nthreads",
            str(self.config.cores // 2 - 1),
            "-j",
            str(job_file),
            "-i",
            str(task_id + 1),
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
            if self._is_notebook:
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
