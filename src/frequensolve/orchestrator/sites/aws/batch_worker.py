#!/usr/bin/env python3
"""
AWS Batch Worker for FrequenSolve.

This module provides a CLI interface for running on AWS Batch instances.
It handles downloading simulation and job data from S3, running preliminary analysis,
and submitting additional batch jobs for each task.
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from frequensolve._optional import optional_dependency_error

try:
    import boto3
    from boto3.exceptions import S3UploadFailedError  # type: ignore[import-untyped]
    from botocore.exceptions import ClientError
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "AWS Batch worker",
        extra="cloud",
        dependencies=("boto3", "botocore"),
        error=exc,
    ) from exc

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
MAX_TASKS = 10_000


def _client_error_code(error: BaseException) -> str:
    """Return an AWS error code without serializing private request details."""

    response = getattr(error, "response", {})
    if not isinstance(response, dict):
        return "unknown"
    details = response.get("Error", {})
    if not isinstance(details, dict):
        return "unknown"
    code = details.get("Code")
    return str(code) if code else "unknown"


def _new_upload_id() -> str:
    """Return a collision-resistant identifier for one result upload."""

    return uuid.uuid4().hex


class BatchWorker:
    """Worker class for running on AWS Batch instances."""

    def __init__(
        self,
        s3_bucket: str,
        region: str = "us-east-1",
        *,
        s3_client: Any = None,
        batch_client: Any = None,
        local_base: Optional[Path] = None,
        process_runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        clock: Callable[[], float] = time.time,
        upload_id_factory: Callable[[], str] = _new_upload_id,
        connect_aws: bool = True,
        owns_local_base: Optional[bool] = None,
    ):
        """Initialize the batch worker.

        Args:
            s3_bucket: S3 bucket containing simulation and job data.
            region: AWS region for S3 operations.
        """
        self.s3_bucket = s3_bucket
        self.region = region
        self._run_process = process_runner
        self._clock = clock
        self._upload_id_factory = upload_id_factory
        self.s3_client = s3_client
        self.batch_client = batch_client
        if connect_aws:
            if self.s3_client is None:
                self.s3_client = boto3.client("s3", region_name=region)
            if self.batch_client is None:
                self.batch_client = boto3.client("batch", region_name=region)

        # Local paths for downloaded data
        if local_base is None:
            self.local_base = Path(tempfile.mkdtemp(prefix="frequensolve-batch-"))
            self._owns_local_base = True
        else:
            self.local_base = Path(local_base)
            self._owns_local_base = bool(owns_local_base)
        self.simulation_dir = self.local_base / "simulation"
        self.job_dir = self.local_base / "job"

        # Create local directories
        self.local_base.mkdir(parents=True, exist_ok=True)
        self.simulation_dir.mkdir(exist_ok=True)
        self.job_dir.mkdir(exist_ok=True)

    def download_from_s3(self, s3_key: str, local_path: Path) -> None:
        """Download data from S3 to local path using aws s3 sync.

        Args:
            s3_key: S3 key to download.
            local_path: Local path to download to.

        Raises:
            RuntimeError: If download fails.
        """
        try:
            logger.info("Downloading an S3 input prefix")

            # Use aws s3 sync for efficient downloading
            sync_command = [
                "aws",
                "s3",
                "sync",
                f"s3://{self.s3_bucket}/{s3_key}",
                str(local_path),
                "--region",
                self.region,
            ]

            result = self._run_process(
                sync_command, capture_output=True, text=True, check=True
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"AWS CLI S3 sync exited with status {result.returncode}"
                )

            logger.info("S3 input download completed")

        except FileNotFoundError as exc:
            raise RuntimeError(
                "AWS CLI is unavailable; install and configure it before running "
                "the batch worker"
            ) from None
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"S3 input download failed with AWS CLI status {exc.returncode}"
            ) from None
        except ClientError as exc:
            raise RuntimeError(
                "S3 input download failed with AWS error code "
                f"{_client_error_code(exc)}"
            ) from None

    def run_preliminary_analysis(self) -> Dict[str, Any]:
        """Run preliminary analysis on the simulation data.

        Returns:
            Dictionary containing analysis results.
        """
        logger.info("Running preliminary analysis...")

        # This is a placeholder - implement your actual analysis logic here
        analysis_results = {
            "simulation_type": "acoustic",
            "grid_dimensions": [100, 100, 50],
            "time_steps": 1000,
            "tasks_required": 10,
            "estimated_runtime": 3600,  # seconds
            "memory_requirements": 8192,  # MB
            "cpu_requirements": 4,
        }

        # Example: analyze simulation files to determine requirements
        try:
            # Look for configuration files
            config_files = list(self.simulation_dir.glob("*.json")) + list(
                self.simulation_dir.glob("*.yaml")
            )
            if config_files:
                logger.info(
                    "Found %d simulation configuration file(s)", len(config_files)
                )

            # Look for mesh files
            mesh_files = list(self.simulation_dir.glob("*.h5")) + list(
                self.simulation_dir.glob("*.nc")
            )
            if mesh_files:
                logger.info("Found %d simulation mesh file(s)", len(mesh_files))

        except OSError as exc:
            logger.warning("Simulation file analysis failed (%s)", type(exc).__name__)

        logger.info("Preliminary analysis completed")
        return analysis_results

    def submit_task_jobs(
        self, analysis_results: Dict[str, Any], job_queue: str, job_definition: str
    ) -> List[str]:
        """Submit individual task jobs to AWS Batch.

        Args:
            analysis_results: Results from preliminary analysis.
            job_queue: AWS Batch job queue name.
            job_definition: AWS Batch job definition name.

        Returns:
            List of submitted job IDs.
        """
        logger.info("Submitting task jobs to AWS Batch...")

        n_tasks = analysis_results.get("tasks_required", 1)
        if (
            not isinstance(n_tasks, int)
            or isinstance(n_tasks, bool)
            or not 0 <= n_tasks <= MAX_TASKS
        ):
            raise ValueError(f"tasks_required must be an integer from 0 to {MAX_TASKS}")
        if not isinstance(job_queue, str) or not job_queue.strip():
            raise ValueError("job_queue must be a non-empty string")
        if not isinstance(job_definition, str) or not job_definition.strip():
            raise ValueError("job_definition must be a non-empty string")
        if self.batch_client is None:
            raise RuntimeError("AWS Batch client is unavailable")
        job_ids = []
        submitted_at = int(self._clock())

        for task_id in range(n_tasks):
            try:
                # Prepare task-specific parameters
                task_params = {
                    "task_id": str(task_id),
                    "simulation_dir": str(self.simulation_dir),
                    "job_dir": str(self.job_dir),
                    "analysis_results": json.dumps(analysis_results),
                }

                # Submit task job
                response = self.batch_client.submit_job(
                    jobName=f"frequensolve-task-{task_id}-{submitted_at}",
                    jobQueue=job_queue,
                    jobDefinition=job_definition,
                    parameters=task_params,
                    containerOverrides={
                        "environment": [
                            {"name": "TASK_ID", "value": str(task_id)},
                            {
                                "name": "SIMULATION_DIR",
                                "value": str(self.simulation_dir),
                            },
                            {"name": "JOB_DIR", "value": str(self.job_dir)},
                            {
                                "name": "ANALYSIS_RESULTS",
                                "value": json.dumps(analysis_results),
                            },
                        ]
                    },
                )

                job_id = response.get("jobId") if isinstance(response, dict) else None
                if not isinstance(job_id, str) or not job_id:
                    raise RuntimeError(
                        f"AWS Batch response for task {task_id} did not contain a job ID"
                    )
                job_ids.append(job_id)
                logger.info("Submitted task %d", task_id)

            except ClientError as exc:
                logger.error(
                    "Failed to submit task %d (AWS error code %s)",
                    task_id,
                    _client_error_code(exc),
                )
                continue

        logger.info(f"Submitted {len(job_ids)} task jobs")
        return job_ids

    def run_simulation_task(
        self, task_id: int, analysis_results: Dict[str, Any]
    ) -> bool:
        """Run a single simulation task.

        Args:
            task_id: Task ID to run.
            analysis_results: Analysis results containing requirements.

        Returns:
            True if task completed successfully, False otherwise.
        """
        logger.info(f"Running simulation task {task_id}")

        if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
            raise ValueError("task_id must be a non-negative integer")
        if not isinstance(analysis_results, dict):
            raise ValueError("analysis_results must be an object")
        if not self.simulation_dir.is_dir() or not self.job_dir.is_dir():
            raise ValueError(
                "simulation_dir and job_dir must identify existing directories"
            )
        timeout = analysis_results.get("estimated_runtime", 3600)
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or timeout <= 0
        ):
            raise ValueError("estimated_runtime must be a positive number")

        try:
            # This is a placeholder - implement your actual simulation logic here
            # You would typically:
            # 1. Set up the simulation environment
            # 2. Run the simulation executable
            # 3. Collect results

            # Example simulation command (replace with your actual executable)
            cmd = [
                sys.executable,
                "-c",
                f"print('Running task {task_id} simulation...'); "
                f"import time; time.sleep(5); "
                f"print('Task {task_id} completed')",
            ]

            # Run simulation
            result = self._run_process(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.simulation_dir,
                timeout=timeout,
            )

            if result.returncode == 0:
                logger.info(f"Task {task_id} completed successfully")
                return True
            else:
                logger.error(
                    "Task %d failed with process status %d",
                    task_id,
                    result.returncode,
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Task {task_id} timed out")
            return False
        except OSError as exc:
            logger.error("Task %d could not start (%s)", task_id, type(exc).__name__)
            return False

    def upload_results_to_s3(self, results_dir: Path, s3_prefix: str) -> str:
        """Upload results to S3.

        Args:
            results_dir: Local directory containing results.
            s3_prefix: S3 prefix for results.

        Returns:
            S3 key where results were uploaded.
        """
        try:
            if not results_dir.is_dir():
                raise ValueError("results_dir must identify an existing directory")
            if (
                not isinstance(s3_prefix, str)
                or not s3_prefix.strip("/")
                or Path(s3_prefix).is_absolute()
                or ".." in Path(s3_prefix).parts
                or "\\" in s3_prefix
            ):
                raise ValueError("s3_prefix must be a non-empty relative prefix")
            if self.s3_client is None:
                raise RuntimeError("S3 client is unavailable")

            upload_id = self._upload_id_factory()
            if (
                not isinstance(upload_id, str)
                or len(upload_id) != 32
                or any(character not in "0123456789abcdef" for character in upload_id)
            ):
                raise RuntimeError("result upload identifier was malformed")
            s3_key = f"{s3_prefix}/results/{upload_id}"

            logger.info("Uploading batch-worker results to S3")

            result_files = sorted(
                file_path for file_path in results_dir.rglob("*") if file_path.is_file()
            )
            if not result_files:
                raise ValueError("results_dir contains no result files")
            if any(file_path.is_symlink() for file_path in result_files):
                raise ValueError("results_dir must not contain symbolic links")

            # Upload all regular files in results directory
            uploaded_keys: list[str] = []
            for file_path in result_files:
                relative_path = file_path.relative_to(results_dir)
                file_s3_key = f"{s3_key}/{relative_path}"

                self.s3_client.upload_file(str(file_path), self.s3_bucket, file_s3_key)
                uploaded_keys.append(file_s3_key)
                metadata = self.s3_client.head_object(
                    Bucket=self.s3_bucket,
                    Key=file_s3_key,
                )
                remote_size = metadata.get("ContentLength")
                if remote_size is not None and remote_size != file_path.stat().st_size:
                    raise RuntimeError("uploaded result size verification failed")
                logger.debug("Uploaded one batch-worker result file")

            logger.info("Batch-worker result upload completed")
            return s3_key

        except (ClientError, S3UploadFailedError, OSError, RuntimeError) as exc:
            client = self.s3_client
            for uploaded_key in locals().get("uploaded_keys", []):
                try:
                    client.delete_object(Bucket=self.s3_bucket, Key=uploaded_key)
                except Exception as cleanup_exc:  # pragma: no cover - best effort
                    logger.warning(
                        "Failed to remove a partial S3 result (%s)",
                        _client_error_code(cleanup_exc),
                    )
            code = _client_error_code(exc)
            detail = f" (AWS error code {code})" if code != "unknown" else ""
            raise RuntimeError(f"Batch-worker result upload failed{detail}") from None

    def cleanup(self) -> None:
        """Clean up local files."""
        if not self._owns_local_base:
            logger.debug("Skipping cleanup of an externally managed local directory")
            return
        try:
            import shutil

            shutil.rmtree(self.local_base, ignore_errors=False)
            logger.info("Cleaned up local files")
        except FileNotFoundError:
            logger.debug("Local batch-worker files were already clean")
        except OSError as exc:
            logger.warning("Failed to clean local files (%s)", type(exc).__name__)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    worker_factory: Callable[..., BatchWorker] = BatchWorker,
) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AWS Batch Worker for FrequenSolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Run as main batch job (downloads data, runs analysis, submits tasks)
        python batch_worker.py --mode main --simulation-key sim/ --job-key job/ --bucket my-bucket

        # Run as individual task
        python batch_worker.py --mode task --task-id 0 --simulation-dir /tmp/sim --job-dir /tmp/job
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["main", "task"],
        required=True,
        help="Execution mode: main (orchestrator) or task (individual simulation)",
    )

    # S3 parameters
    parser.add_argument("--bucket", help="S3 bucket name (required for main mode)")

    parser.add_argument(
        "--region", default="us-east-1", help="AWS region (default: us-east-1)"
    )

    parser.add_argument(
        "--simulation-key", help="S3 key for simulation data (required for main mode)"
    )

    parser.add_argument(
        "--job-key", help="S3 key for job data (required for main mode)"
    )

    # Task parameters
    parser.add_argument(
        "--task-id", type=int, help="Task ID to run (required for task mode)"
    )

    parser.add_argument(
        "--simulation-dir", help="Local simulation directory (required for task mode)"
    )

    parser.add_argument(
        "--job-dir", help="Local job directory (required for task mode)"
    )

    # AWS Batch parameters
    parser.add_argument("--job-queue", help="AWS Batch job queue for task submission")

    parser.add_argument(
        "--job-definition", help="AWS Batch job definition for task submission"
    )

    # Output parameters
    parser.add_argument(
        "--results-prefix",
        default="frequensolve",
        help="S3 prefix for results (default: frequensolve)",
    )

    parser.add_argument(
        "--upload-results",
        action="store_true",
        help="Upload results to S3 after completion",
    )

    args = parser.parse_args(argv)

    if args.mode == "main" and not args.bucket:
        parser.error("--bucket is required for main mode")
    if args.mode == "main" and (not args.simulation_key or not args.job_key):
        parser.error("--simulation-key and --job-key are required for main mode")
    if args.mode == "task" and (
        args.task_id is None or not args.simulation_dir or not args.job_dir
    ):
        parser.error(
            "--task-id, --simulation-dir, and --job-dir are required for task mode"
        )

    worker: Optional[BatchWorker] = None
    try:
        worker = worker_factory(
            args.bucket or "",
            args.region,
            connect_aws=args.mode == "main",
        )

        if args.mode == "main":
            # Main orchestrator mode
            # Download data from S3
            worker.download_from_s3(args.simulation_key, worker.simulation_dir)
            worker.download_from_s3(args.job_key, worker.job_dir)

            # Run preliminary analysis
            analysis_results = worker.run_preliminary_analysis()

            # Submit task jobs if queue and definition provided
            if args.job_queue and args.job_definition:
                job_ids = worker.submit_task_jobs(
                    analysis_results, args.job_queue, args.job_definition
                )
                logger.info("Submitted %d task jobs", len(job_ids))
            else:
                logger.info(
                    "No job queue/definition provided, skipping task submission"
                )

            # Upload results if requested
            if args.upload_results:
                results_dir = worker.local_base / "results"
                results_dir.mkdir(exist_ok=True)

                # Save analysis results
                with open(results_dir / "analysis.json", "w") as f:
                    json.dump(analysis_results, f, indent=2)

                worker.upload_results_to_s3(results_dir, args.results_prefix)
                logger.info("Results upload completed")

        elif args.mode == "task":
            # Individual task mode
            # Set local directories
            worker.simulation_dir = Path(args.simulation_dir)
            worker.job_dir = Path(args.job_dir)

            # Run the simulation task
            success = worker.run_simulation_task(args.task_id, {})

            if success:
                logger.info(f"Task {args.task_id} completed successfully")
                return 0
            else:
                logger.error(f"Task {args.task_id} failed")
                return 1

        return 0
    except Exception as exc:
        logger.error("Batch worker failed (%s)", type(exc).__name__)
        return 1
    finally:
        if worker is not None:
            worker.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
