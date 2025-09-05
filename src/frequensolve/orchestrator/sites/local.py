# Set up logging (send Dask logging to files)
import logging
import os
import signal
import subprocess
import warnings
from dataclasses import dataclass, field
from logging import ERROR, INFO, FileHandler, Formatter, getLogger
from pathlib import Path
from typing import Dict, List, Optional, Union

from dask import config
from dask.distributed import Client, Future, LocalCluster, get_task_stream, wait
from dotenv import load_dotenv
from numpy.typing import ArrayLike

from frequensolve.orchestrator.config.local import LocalSiteConfig
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    SiteStatus,
    _wait_for_path,
)
from frequensolve.seismic.record_database import RecordDatabase
from frequensolve.simulation.imaging import RawImage, RTMImagingJob
from frequensolve.simulation.jobs import SimulationJob
from frequensolve.util.setup_logger import init_logger

logging.basicConfig(level=ERROR)

logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/local.log")
logger.setLevel(ERROR)

for logger_name in ["distributed", "bokeh", "tornado"]:
    log = getLogger(logger_name)
    log.setLevel(INFO)
    log.handlers = []
    handler = FileHandler("/tmp/log/frequensolve/local.log")
    handler.setFormatter(
        Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    log.addHandler(handler)
    log.propagate = False

__all__ = ["LocalSite"]


def run_task(
    job_file: str,
    task_id: int,
    executable: str,
    env: dict,
    n_ranks: int = 1,
    n_threads: int = 1,
    stdout_dir: str = None,
) -> dict:
    """Run a single task and return its results.

    Args:
        job_file: Path to the job file
        task_id: Task ID to run
        executable: Path to the solver executable
        env: Environment variables
        n_ranks: Number of MPI ranks
        n_threads: Number of threads per rank
        output_dir: Directory to store completed output files
        active_dir: Directory to store active output files

    Returns:
        Dict containing task results
    """
    _wait_for_path(job_file)

    threads_per_rank = n_threads // n_ranks

    if n_ranks > 1:
        args = [
            "mpirun",
            "-np",
            f"{n_ranks}",
        ]
    else:
        args = []

    args += [
        executable,
        "-nthreads",
        f"{threads_per_rank}",
        "-j",
        f"{job_file}",
        "-i",
        f"{task_id + 1}",
    ]

    logger.info(f"Executing: {' '.join(args)}")

    if stdout_dir:
        stdout_file = os.path.join(stdout_dir, f"task_{task_id+1}.out")
    else:
        stdout_file = None

    try:
        with (
            open(stdout_file, "w") if stdout_file else open(os.devnull, "w") as stdout,
        ):
            proc = subprocess.Popen(
                args, stdout=stdout, stderr=stdout, env=env, text=True
            )
            return_code = proc.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, args)

        return {
            "task_id": task_id,
            "status": "success",
            "stdout": (
                os.path.join(stdout_dir, f"task_{task_id+1}.out")
                if stdout_dir
                else None
            ),
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e),
            "stdout": (
                os.path.join(stdout_dir, f"task_{task_id+1}.out")
                if stdout_dir
                else None
            ),
        }


@dataclass(kw_only=True)
class LocalSite(BaseSite):
    """Site for local execution."""

    status: SiteStatus = field(default_factory=SiteStatus)
    config: LocalSiteConfig = field(init=False)
    executable: str = field(init=False)
    env: dict = field(default_factory=dict)
    n_workers: Optional[int] = None
    threads_per_worker: Optional[int] = None
    memory_per_worker: Optional[int] = None
    _dask_client: Optional[Client] = field(default=None)
    _dask_cluster: Optional[LocalCluster] = field(default=None)
    _futures: List[Future] = field(default_factory=list)
    _worker_status: Dict[str, str] = field(default_factory=dict)
    _status_display: Optional[object] = field(default=None)

    def __post_init__(self):
        self.status = SiteStatus(status="running")
        self.config = LocalSiteConfig()
        self.executable = self._get_solver_path()
        self.env = os.environ.copy()
        self.env["FS_SOLVER_PATH"] = os.getenv("FS_SOLVER_PATH")

    def _initialize_dask(self, n_workers: Optional[int] = None):
        """Initialize Dask client and cluster."""

        if n_workers is None:
            self.n_workers = self.config.cores
        else:
            self.n_workers = n_workers

        if self.threads_per_worker is None:
            self.threads_per_worker = self.config.cores // self.n_workers
        if self.memory_per_worker is None:
            if self.config.memory:
                self.memory_per_worker = int(
                    (0.9 * self.config.memory) / self.n_workers
                )
            else:
                self.memory_per_worker = 4096

        total_threads = self.n_workers * self.threads_per_worker
        total_memory = self.n_workers * self.memory_per_worker
        if total_threads > self.config.cores:
            raise ValueError(
                f"Total threads ({total_threads}) exceed available cores ({self.config.cores})"
            )
        if self.config.memory:
            if total_memory > self.config.memory:
                raise ValueError(
                    f"Total memory ({total_memory}MB) exceed available memory ({self.config.memory}MB)"
                )

        print(
            f"Dask initialized with {self.n_workers} workers, "
            f"{self.threads_per_worker} threads per worker, "
            f"and {self.memory_per_worker}MB memory per worker"
        )
        # logger.info(
        #     f"Dask initialized with {self.n_workers} workers, "
        #     f"{self.threads_per_worker} threads per worker, "
        #     f"and {self.memory_per_worker}MB memory per worker"
        # )

        if self._dask_client is None:
            try:
                config.set(
                    {
                        "distributed.worker.memory.target": 0.6,
                        "distributed.worker.memory.pause": 0.8,
                        "distributed.worker.threads": self.threads_per_worker,
                        "distributed.scheduler.work-stealing": True,
                        "distributed.scheduler.work-stealing-interval": "1s",
                        "distributed.scheduler.bandwidth": 1,
                    }
                )

                self._dask_cluster = LocalCluster(
                    n_workers=self.n_workers,
                    threads_per_worker=self.threads_per_worker,
                    memory_limit=f"{self.memory_per_worker}MB",
                    dashboard_address=":0",  # Let Dask choose an available port
                    local_directory="/tmp/dask-worker-space",
                    scheduler_port=0,
                    silence_logs=ERROR,
                    processes=True,
                    resources={"CPU": self.threads_per_worker},
                )
                self._dask_client = Client(self._dask_cluster)
                self._dashboard_port = self._dask_cluster.dashboard_link.split(":")[-1]
                print(
                    f"Dask Dashboard available at: http://localhost:{self._dashboard_port}"
                )

                try:
                    self._dask_client.get_worker_logs()
                    logger.info("Dask dashboard initialized successfully")
                except Exception as e:
                    logger.warning(f"Dashboard may not be fully accessible: {str(e)}")

                try:
                    self._task_stream = get_task_stream(self._dask_client, plot=False)
                    logger.info("Task stream initialized successfully")
                except Exception as e:
                    logger.warning(f"Task stream not available: {str(e)}")

            except Exception as e:
                logger.error(f"Failed to initialize Dask cluster: {str(e)}")
                raise

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, "_task_stream"):
            try:
                self._task_stream.stop()
            except:
                pass
        if self._dask_client is not None:
            self._dask_client.close()
        if self._dask_cluster is not None:
            self._dask_cluster.close()

    def _get_solver_path(self) -> str:
        """Get the solver path."""
        load_dotenv()
        executable = os.getenv("LOCAL_SOLVER_EXECUTABLE")
        if not executable:
            warnings.warn("LOCAL_SOLVER_EXECUTABLE not set in environment")
            return None
        if not Path(executable).exists():
            warnings.warn(f"Solver executable not found at {executable}")
            return None
        return executable

    def submit(self, job: SimulationJob, **kwargs) -> List[dict]:
        """Submit job and block until completion with progress tracking.

        Args:
            job: The simulation job to run
            **kwargs: Additional arguments for task configuration

        Returns:
            List of results from completed tasks
        """

        if self._dask_client is None:
            if job.n_tasks < self.config.cores:
                nw = self.config.cores
                while nw > job.n_tasks:
                    nw = nw // 2
                self._initialize_dask(n_workers=nw)
            else:
                self._initialize_dask(self.n_workers)

        futures = self.submit_async(job, **kwargs)

        if self._is_notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        pbar = tqdm(
            total=len(futures),
            desc=f"Running: {job.name}",
            bar_format="{desc} {n_fmt}/{total_fmt} |{bar}| Elapsed time: {elapsed}s",
            colour="#4ec9b0",
        )

        def update_progress(future):
            pbar.update(1)

        for future in futures:
            future.add_done_callback(update_progress)

        results = wait(futures)
        pbar.close()
        return results

    def submit_async(self, job: SimulationJob, **kwargs) -> List[Future]:
        """Submit job asynchronously and return Dask futures.

        Args:
            job: The simulation job to run
            **kwargs: Additional arguments for task configuration

        Returns:
            List of Dask futures for the submitted tasks
        """
        if not self.executable:
            raise RuntimeError("Solver executable not found, cannot submit job")

        job_file = job.save()

        if self._dask_client is None:
            self._initialize_dask()

        n_ranks = kwargs.get("procs_per_job", 1)

        stdout_dir = str(job._stdout_path)
        if os.path.exists(stdout_dir):
            for file in os.listdir(stdout_dir):
                os.remove(os.path.join(stdout_dir, file))
        os.makedirs(stdout_dir, exist_ok=True)

        futures = []

        # Mesh and size first
        future = self._dask_client.submit(
            run_task,
            job_file,
            -1,
            self.executable,
            self.env,
            n_ranks=1,
            n_threads=1,
            resources={"CPU": 1},
        )
        future.result()

        # TODO: Verify success of mesh and size before submitting other tasks

        # Loop tasks in reverse order for improved load balancing
        for i in range(job.n_tasks - 1, -1, -1):
            try:
                future = self._dask_client.submit(
                    run_task,
                    job_file,
                    i,
                    self.executable,
                    self.env,
                    n_ranks=n_ranks,
                    n_threads=self.threads_per_worker,
                    stdout_dir=stdout_dir,
                    retries=0,
                    priority=i,
                    actor=False,
                    pure=True,
                    resources={"CPU": self.threads_per_worker},
                )

                # # Set up callback to update status display periodically
                # def make_callback():
                #     def callback(fut):
                #         self._print_worker_status()
                #     return callback

                # future.add_done_callback(make_callback())
                futures.append(future)
            except Exception as e:
                logger.error(f"Failed to submit task {i}: {str(e)}")
                raise

        self._futures.extend(futures)
        return futures

    # TODO: use path (will need changes elsewhere to support)
    def fetch_traces(
        self,
        job: Union[SimulationJob, List[SimulationJob]],
        upscale: int = 1,
        path: Optional[Union[str, Path]] = None,
    ) -> Union[RecordDatabase, Dict[str, RecordDatabase]]:
        if isinstance(job, SimulationJob):
            db = RecordDatabase.from_job(job, upscale)
            return db
        else:
            db_map = {}
            for j in job:
                db = RecordDatabase.from_job(j, upscale)
                db_map[j.name] = db
            return db_map

    def fetch_image(
        self,
        job: Union[RTMImagingJob, List[RTMImagingJob]],
    ) -> ArrayLike:
        """Gets and accumulates images."""

        if isinstance(job, RTMImagingJob):
            jobs = [job]
        else:
            jobs = job

        images = {}

        for job in jobs:
            local = job._local_image_path
            image = RawImage(
                path=local,
                shape=job.grid.shape,
                parts=job.n_tasks,
            )
            images[job.name] = image

        if len(images) == 1:
            return images[jobs[0].name]
        else:
            return images

    def fetch_paraview(
        self, job: SimulationJob, path: Optional[Union[str, Path]] = None
    ):
        """Dummy method for consistency."""
        for name, pv_path in job.paraview_outputs.items():
            print(f"Fetching ParaView output '{name}' from {pv_path}")
        pass

    @property
    def provisioned(self) -> bool:
        """Dummy method for consistency."""
        return True

    def sync(self, project):
        """Dummy method for consistency."""
        pass

    def _sync_project(self, project):
        """Dummy method for consistency."""
        pass

    def connect_to_existing_job(self):
        """Dummy method for consistency."""
        pass

    def wait_completion(self, jobs: List[SimulationJob]):
        """Wait for all jobs to complete."""
        # TODO: Wait specific jobs
        # return wait(self._dask_client.futures())
        pass

    def get(
        self,
        remote_path: Union[str, Path],
        local_path: Union[str, Path],
        overwrite: bool = False,
    ):
        """Download a file or directory from the site."""
        pass

    def put(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        """Send a file or directory to the site."""
        pass

    @property
    def dashboard_url(self) -> Optional[str]:
        """Get the URL for the Dask dashboard.

        Returns:
            URL string if dashboard is available, None otherwise
        """
        if self._dashboard_port:
            return f"http://localhost:{self._dashboard_port}"
        return None

    def cancel_job(self, job_id: str):
        """Cancel a running job.

        Args:
            job_id: The ID of the job to cancel
        """
        try:
            os.kill(int(job_id), signal.SIGTERM)
        except ProcessLookupError:
            pass
        except ValueError:
            raise ValueError(f"Invalid process ID: {job_id}")
