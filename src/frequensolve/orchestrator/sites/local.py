# Set up logging (send Dask logging to files)
import logging
import os
import signal
import subprocess
from dataclasses import dataclass, field
from logging import ERROR, INFO, FileHandler, Formatter, getLogger
from pathlib import Path
from typing import Dict, List, Optional, Union

from dask import config
from dask.distributed import Client, Future, LocalCluster, get_task_stream, wait
from dotenv import load_dotenv

from frequensolve.orchestrator.config.local import LocalSiteConfig
from frequensolve.orchestrator.sites.base import (
    BaseSite,
    SiteStatus,
    _wait_for_path,
)
from frequensolve.seismic.record_database import RecordDatabase
from frequensolve.simulation.imaging import RTMImagingJob
from frequensolve.simulation.jobs import SimulationJob
from frequensolve.util.setup_logger import init_logger

logging.basicConfig(level=ERROR)

logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/local.log")
logger.setLevel(INFO)

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
    output_dir: str = None,
    active_dir: str = None,
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

    if active_dir:
        stdout_file = os.path.join(active_dir, f"task_{task_id+1}.out")
        stderr_file = os.path.join(active_dir, f"task_{task_id+1}.err")
    else:
        stdout_file = None
        stderr_file = None

    try:
        with (
            open(stdout_file, "w") if stdout_file else open(os.devnull, "w") as stdout,
            open(stderr_file, "w") if stderr_file else open(os.devnull, "w") as stderr,
        ):

            proc = subprocess.Popen(
                args, stdout=stdout, stderr=stderr, env=env, text=True
            )
            return_code = proc.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, args)

            # Move files from active to output directory if successful
            if output_dir and active_dir:
                os.makedirs(output_dir, exist_ok=True)
                for file in [stdout_file, stderr_file]:
                    if os.path.exists(file):
                        dest_file = os.path.join(output_dir, os.path.basename(file))
                        os.rename(file, dest_file)
                        file = dest_file

        return {
            "task_id": task_id,
            "status": "success",
            "output_file": (
                os.path.join(output_dir, f"task_{task_id+1}.out")
                if output_dir
                else None
            ),
            "error_file": (
                os.path.join(output_dir, f"task_{task_id+1}.err")
                if output_dir
                else None
            ),
        }
    except Exception as e:
        # Move files from active to output directory even if failed
        if output_dir and active_dir:
            os.makedirs(output_dir, exist_ok=True)
            for file in [stdout_file, stderr_file]:
                if os.path.exists(file):
                    dest_file = os.path.join(output_dir, os.path.basename(file))
                    os.rename(file, dest_file)
                    file = dest_file

        return {
            "task_id": task_id,
            "status": "error",
            "error": str(e),
            "output_file": (
                os.path.join(output_dir, f"task_{task_id+1}.out")
                if output_dir
                else None
            ),
            "error_file": (
                os.path.join(output_dir, f"task_{task_id+1}.err")
                if output_dir
                else None
            ),
        }


def is_notebook() -> bool:
    """Check if running in a Jupyter notebook."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


@dataclass
class LocalSite(BaseSite):
    """Site for local execution."""

    status: SiteStatus = field(default_factory=SiteStatus)
    config: LocalSiteConfig = field(init=False)
    executable: str = field(init=False)
    env: dict = field(default_factory=dict)
    n_workers: Optional[int] = 1
    threads_per_worker: Optional[int] = None
    memory_per_worker: Optional[int] = None
    _dask_client: Optional[Client] = field(default=None)
    _dask_cluster: Optional[LocalCluster] = field(default=None)
    _futures: List[Future] = field(default_factory=list)
    _worker_status: Dict[str, str] = field(default_factory=dict)
    _status_display: Optional[object] = field(default=None)
    _is_notebook: bool = field(default_factory=is_notebook)

    def __post_init__(self):
        self.status = SiteStatus(status="running")
        self.config = LocalSiteConfig()
        self.executable = self._get_solver_path()
        self.env = os.environ.copy()
        self.env["FREQUENSOLVE_DIR"] = os.getenv("FS_SOLVER_PATH")

    def _initialize_dask(self):
        """Initialize Dask client and cluster."""

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

        logger.info(
            f"Initializing Dask with {self.n_workers} workers, "
            f"{self.threads_per_worker} threads per worker, "
            f"and {self.memory_per_worker}MB memory per worker"
        )

        if self._dask_client is None:
            try:
                dashboard_port = 8787

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
                    dashboard_address=f"localhost:{dashboard_port}",
                    local_directory="/tmp/dask-worker-space",
                    scheduler_port=0,
                    silence_logs=ERROR,
                    processes=True,
                    resources={"CPU": self.threads_per_worker},
                )
                self._dask_client = Client(self._dask_cluster)
                self._dashboard_port = dashboard_port
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
            raise RuntimeError("LOCAL_SOLVER_EXECUTABLE not set in environment")
        if not Path(executable).exists():
            raise FileNotFoundError(f"Solver executable not found at {executable}")
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
            self._initialize_dask()

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

        # Wait for all futures to complete
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
        if self._dask_client is None:
            self._initialize_dask()

        job_file = job.save()
        n_ranks = kwargs.get("n_ranks", 1)

        output_dir = os.path.join(job.project_path, "jobs", "out", job.name)
        active_dir = os.path.join(job.project_path, "jobs", "active")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(active_dir, exist_ok=True)

        if not self._is_notebook:
            print("\n" * (self.n_workers + 2))
        # self._print_worker_status()

        futures = []

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
                    output_dir=output_dir,
                    active_dir=active_dir,
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
        """Gets traces and consolidates them into a single HDF5 file."""
        if isinstance(job, SimulationJob):
            db = RecordDatabase.from_results(
                job.records, job.project_path.resolve(), upscale
            )
            db.consolidate_h5()
            return db
        else:
            db_map = {}
            for j in job:
                db = RecordDatabase.from_results(
                    j.records, j.project_path.resolve(), upscale
                )
                db.consolidate_h5()
                db_map[j.name] = db
            return db_map

    def fetch_image(
        self,
        job: Union[RTMImagingJob, List[RTMImagingJob]],
        path: Optional[Union[str, Path]] = None,
    ) -> Union[RecordDatabase, Dict[str, RecordDatabase]]:
        """Gets traces and consolidates them into a single HDF5 file."""
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

    def fetch_paraview(
        self, job: SimulationJob, path: Optional[Union[str, Path]] = None
    ):
        """Dummy method for consistency."""
        for name, pv_path in job.paraview_outputs.items():
            print(f"Fetching ParaView output '{name}' from {pv_path}")
        pass

    def wait_completion(self, jobs: List[SimulationJob]):
        """Wait for all jobs to complete."""
        # TODO: Wait specific jobs
        # return wait(self._dask_client.futures())
        pass

    def transfer():
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

    # def get_job_status(self, futures: Optional[List[Future]] = None) -> Dict:
    #     """Get status of submitted jobs.

    #     Args:
    #         futures: Optional list of futures to check. If None, checks all futures.

    #     Returns:
    #         Dict containing job status information including:
    #         - total_jobs: Total number of jobs
    #         - completed: Number of completed jobs
    #         - running: Number of running jobs
    #         - failed: Number of failed jobs
    #         - pending: Number of pending jobs
    #     """
    #     if futures is None:
    #         futures = self._futures

    #     if not futures:
    #         return {"total_jobs": 0, "status": "no_jobs"}

    #     status = {
    #         "total_jobs": len(futures),
    #         "completed": 0,
    #         "running": 0,
    #         "failed": 0,
    #         "pending": 0
    #     }

    #     for future in futures:
    #         if future.done():
    #             if future.status == "finished":
    #                 status["completed"] += 1
    #             else:
    #                 status["failed"] += 1
    #         elif future.status == "pending":
    #             status["pending"] += 1
    #         else:
    #             status["running"] += 1

    #     return status

    # def _print_worker_status(self):
    #     """Update worker status display based on environment."""
    #     if self._is_notebook:
    #         self._print_worker_status_notebook()
    #     else:
    #         self._print_worker_status_terminal()

    # def _get_theme_colors(self) -> dict:
    #     """Detect the current theme and return appropriate colors.

    #     Returns:
    #         Dict containing color scheme for the current theme
    #     """
    #     if self._is_notebook:
    #         try:
    #             from IPython.display import display
    #             import ipywidgets as widgets
    #             from IPython.core.display import HTML

    #             # Try to detect theme from Jupyter
    #             try:
    #                 from jupyterthemes import jtplot
    #                 theme = jtplot.style().get('theme', 'default')
    #             except:
    #                 # Default to dark theme if we can't detect
    #                 theme = 'dark'

    #             if theme == 'dark':
    #                 return {
    #                     'bg': '#1e1e1e',
    #                     'text': '#d4d4d4',
    #                     'border': '#3c3c3c',
    #                     'idle': '#6b6b6b',
    #                     'active': '#4ec9b0',
    #                     'header': '#9cdcfe'
    #                 }
    #             else:
    #                 return {
    #                     'bg': '#ffffff',
    #                     'text': '#000000',
    #                     'border': '#e0e0e0',
    #                     'idle': '#666666',
    #                     'active': '#007acc',
    #                     'header': '#0000ff'
    #                 }
    #         except ImportError:
    #             pass

    #     return {
    #         'bg': '#1e1e1e',
    #         'text': '#d4d4d4',
    #         'border': '#3c3c3c',
    #         'idle': '#6b6b6b',
    #         'active': '#4ec9b0',
    #         'header': '#9cdcfe'
    #     }

    # def _print_worker_status_notebook(self):
    #     """Update worker status display in Jupyter notebook."""
    #     try:
    #         from IPython.display import display
    #         import ipywidgets as widgets

    #         if self._status_display is None:
    #             self._status_display = widgets.HTML()
    #             display(self._status_display)

    #         colors = self._get_theme_colors()

    #         # Create a container for both worker and task status
    #         status_html = f"""
    #         <div style='
    #             font-family: "JetBrains Mono", "Fira Code", monospace;
    #             background-color: {colors['bg']};
    #             color: {colors['text']};
    #             padding: 10px;
    #             border: 1px solid {colors['border']};
    #             border-radius: 4px;
    #             margin: 5px 0;
    #             display: flex;
    #             flex-direction: column;
    #             gap: 10px;
    #         '>
    #             <div style='
    #                 color: {colors['header']};
    #                 font-weight: bold;
    #                 border-bottom: 1px solid {colors['border']};
    #                 padding-bottom: 5px;
    #                 margin-bottom: 5px;
    #             '>
    #                 Dask Dashboard: <a href="http://localhost:{self._dashboard_port}" target="_blank" style="color: {colors['active']};">http://localhost:{self._dashboard_port}</a>
    #             </div>
    #             <div style='display: flex; gap: 20px;'>
    #                 <div style='width: 300px;'>
    #                     <div style='
    #                         color: {colors['header']};
    #                         font-weight: bold;
    #                         margin-bottom: 3px;
    #                         border-bottom: 1px solid {colors['border']};
    #                         padding-bottom: 3px;
    #                     '>Workers:</div>
    #         """

    #         worker_states = self._get_worker_states()

    #         for i in range(self.n_workers):
    #             worker_key = f"worker_{i}"
    #             worker_info = worker_states.get(worker_key, {'status': 'idle', 'cores': None, 'memory': 0})
    #             status_color = colors['active'] if 'running' in worker_info['status'] else colors['idle']
    #             status_html += f"""
    #             <div style='
    #                 margin: 1px 0;
    #                 padding: 1px 5px;
    #                 line-height: 1.2;
    #             '>
    #                 <span style='color: {colors['text']};'>Worker {i} ({worker_info['cores']} cores):</span>
    #                 <span style='color: {status_color};'>{worker_info['status']}</span>
    #             </div>
    #             """

    #         status_html += "</div>"

    #         # Add task status section
    #         status_html += f"""
    #                 <div style='width: 300px;'>
    #                     <div style='
    #                         color: {colors['header']};
    #                         font-weight: bold;
    #                         margin-bottom: 3px;
    #                         border-bottom: 1px solid {colors['border']};
    #                         padding-bottom: 3px;
    #                     '>Tasks:</div>
    #         """

    #         # Get task states
    #         task_states = self._get_task_states()

    #         # Display task status
    #         for task_id, task_info in task_states.items():
    #             status_color = colors['active'] if 'Running' in task_info['status'] else colors['idle']
    #             status_html += f"""
    #             <div style='
    #                 margin: 1px 0;
    #                 padding: 1px 5px;
    #                 line-height: 1.2;
    #             '>
    #                 <span style='color: {colors['text']};'>Task {task_id} ({task_info['frequency']} Hz):</span>
    #                 <span style='color: {status_color};'>{task_info['status']}</span>
    #             </div>
    #             """

    #         status_html += "</div></div></div>"
    #         self._status_display.value = status_html
    #     except ImportError:
    #         # Fall back to terminal display if IPython widgets not available
    #         self._is_notebook = False
    #         self._print_worker_status_terminal()

    # def _print_worker_status_terminal(self):
    #     """Print current worker status in terminal-friendly format."""
    #     colors = self._get_theme_colors()
    #     # Move cursor up to overwrite previous status
    #     print("\033[F" * (self.n_workers + 2), end="")
    #     print(f"\033[38;2;156;220;254mWorkers:\033[0m", end="")  # Header in light blue

    #     # Get current worker states from Dask
    #     worker_states = self._get_worker_states()

    #     for i in range(self.n_workers):
    #         worker_key = f"worker_{i}"
    #         status = worker_states.get(worker_key, "idle")
    #         status_color = "\033[38;2;78;201;176m" if status != "idle" else "\033[38;2;107;107;107m"  # Active in teal, idle in gray
    #         print(f"\n\r  {i}: {status_color}{status}\033[0m", end="")
    #     print("\n", end="", flush=True)

    # def _get_worker_states(self) -> Dict[str, str]:
    #     """Get current state of all workers by polling Dask scheduler.

    #     Returns:
    #         Dict mapping worker keys to their current task status
    #     """
    #     if not self._dask_client:
    #         return {f"worker_{i}": "idle" for i in range(self.n_workers)}

    #     try:
    #         scheduler_info = self._dask_client.scheduler_info()
    #         workers = scheduler_info.get('workers', {})

    #         # Initialize all workers as idle
    #         worker_states = {}

    #         for worker_addr, worker_info in workers.items():
    #             try:
    #                 worker_num = worker_info["name"]
    #                 worker_key = f"worker_{worker_num}"
    #                 task_counts = worker_info.get('metrics', {}).get('task_counts', {})
    #                 executing = task_counts.get('executing', 0)
    #                 memory = task_counts.get('memory', 0)
    #                 constrained = task_counts.get('constrained', 0)
    #                 nthreads = worker_info.get('nthreads', 8)
    #                 if executing > 0:
    #                     status = f"running - Task {executing}"
    #                 else:
    #                     status = "idle"

    #                 worker_states[worker_key] = {
    #                     'status': status,
    #                     'cores': nthreads,
    #                     'memory': worker_info.get('metrics', {}).get('memory', 0) / (1024*1024)  # Convert to MB
    #                 }
    #             except (ValueError, IndexError, KeyError):
    #                 continue

    #         return worker_states
    #     except Exception as e:
    #         logger.error(f"Error getting worker states: {str(e)}")
    #         return {f"worker_{i}": "idle" for i in range(self.n_workers)}

    # def _get_task_states(self) -> Dict[int, Dict]:
    #     """Get current state of all tasks using task stream data.

    #     Returns:
    #         Dict mapping task IDs to their current status and info
    #     """
    #     if not self._dask_client or not hasattr(self, '_task_stream'):
    #         return {}

    #     try:
    #         task_data = self._task_stream.data
    #         print(task_data)
    #         if task_data is None or len(task_data) == 0:
    #             logger.debug("No task data available yet")
    #             return {i: {'status': 'Pending', 'frequency': None, 'worker': '', 'key': future.key}
    #                    for i, future in enumerate(self._futures)}

    #         tasks = {}
    #         for i, future in enumerate(self._futures):
    #             task_key = future.key
    #             task_info = [t for t in task_data if t['key'] == task_key]

    #             if not task_info:
    #                 status = "Pending"
    #             else:
    #                 latest = task_info[-1]
    #                 if latest['action'] == 'compute':
    #                     if 'stop' not in latest or latest['stop'] is None:
    #                         worker = latest['worker']
    #                         status = f"Running on Worker {worker}"
    #                     else:
    #                         duration = (latest['stop'] - latest['start']).total_seconds()
    #                         status = f"Completed in {duration:.1f}s"
    #                 else:
    #                     status = f"Pending ({latest['action']})"

    #             tasks[i] = {
    #                 'status': status,
    #                 'frequency': '5',  # TODO: Get actual frequency from job info
    #                 'worker': task_info[-1]['worker'] if task_info else '',
    #                 'key': task_key
    #             }

    #         return tasks
    #     except Exception as e:
    #         logger.error(f"Error getting task states: {str(e)}")
    #         return {}
