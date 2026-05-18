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
import time
from pathlib import Path
from typing import Any, Dict, List

from frequensolve._optional import optional_dependency_error

try:
    import boto3
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


class BatchWorker:
    """Worker class for running on AWS Batch instances."""

    def __init__(self, s3_bucket: str, region: str = "us-east-1"):
        """Initialize the batch worker.

        Args:
            s3_bucket: S3 bucket containing simulation and job data.
            region: AWS region for S3 operations.
        """
        self.s3_bucket = s3_bucket
        self.s3_client = boto3.client("s3", region_name=region)
        self.batch_client = boto3.client("batch", region_name=region)

        # Local paths for downloaded data
        self.local_base = Path("/tmp/frequensolve")
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
            logger.info(f"Downloading {s3_key} to {local_path}")

            # Use aws s3 sync for efficient downloading
            sync_command = [
                "aws",
                "s3",
                "sync",
                f"s3://{self.s3_bucket}/{s3_key}",
                str(local_path),
                "--region",
                "us-east-1",  # TODO: Make this configurable
            ]

            result = subprocess.run(
                sync_command, capture_output=True, text=True, check=True
            )

            if result.returncode != 0:
                raise RuntimeError(f"aws s3 sync failed: {result.stderr}")

            logger.info(f"Successfully downloaded {s3_key}")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to download {s3_key} from S3: {e}")
        except ClientError as e:
            raise RuntimeError(f"Failed to download {s3_key} from S3: {e}")

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
                    f"Found configuration files: {[f.name for f in config_files]}"
                )

            # Look for mesh files
            mesh_files = list(self.simulation_dir.glob("*.h5")) + list(
                self.simulation_dir.glob("*.nc")
            )
            if mesh_files:
                logger.info(f"Found mesh files: {[f.name for f in mesh_files]}")

        except Exception as e:
            logger.warning(f"Error during file analysis: {e}")

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
        job_ids = []

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
                    jobName=f"frequensolve-task-{task_id}-{int(time.time())}",
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

                job_id = response["jobId"]
                job_ids.append(job_id)
                logger.info(f"Submitted task {task_id} job: {job_id}")

            except ClientError as e:
                logger.error(f"Failed to submit task {task_id} job: {e}")
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

        try:
            # This is a placeholder - implement your actual simulation logic here
            # You would typically:
            # 1. Set up the simulation environment
            # 2. Run the simulation executable
            # 3. Collect results

            # Example simulation command (replace with your actual executable)
            cmd = [
                "python",
                "-c",
                f"print('Running task {task_id} simulation...'); "
                f"import time; time.sleep(5); "
                f"print('Task {task_id} completed')",
            ]

            # Run simulation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.simulation_dir,
                timeout=analysis_results.get("estimated_runtime", 3600),
            )

            if result.returncode == 0:
                logger.info(f"Task {task_id} completed successfully")
                return True
            else:
                logger.error(f"Task {task_id} failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Task {task_id} timed out")
            return False
        except Exception as e:
            logger.error(f"Task {task_id} failed with error: {e}")
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
            timestamp = int(time.time())
            s3_key = f"{s3_prefix}/results/{timestamp}"

            logger.info(f"Uploading results to S3: {s3_key}")

            # Upload all files in results directory
            for file_path in results_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(results_dir)
                    file_s3_key = f"{s3_key}/{relative_path}"

                    self.s3_client.upload_file(
                        str(file_path), self.s3_bucket, file_s3_key
                    )
                    logger.debug(f"Uploaded {file_path} to {file_s3_key}")

            logger.info(f"Results uploaded successfully to {s3_key}")
            return s3_key

        except ClientError as e:
            raise RuntimeError(f"Failed to upload results to S3: {e}")

    def cleanup(self) -> None:
        """Clean up local files."""
        try:
            import shutil

            shutil.rmtree(self.local_base)
            logger.info("Cleaned up local files")
        except Exception as e:
            logger.warning(f"Failed to cleanup local files: {e}")


def main():
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
    parser.add_argument("--bucket", required=True, help="S3 bucket name")

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

    args = parser.parse_args()

    try:
        worker = BatchWorker(args.bucket, args.region)

        if args.mode == "main":
            # Main orchestrator mode
            if not args.simulation_key or not args.job_key:
                parser.error(
                    "--simulation-key and --job-key are required for main mode"
                )

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
                logger.info(f"Submitted {len(job_ids)} task jobs: {job_ids}")
            else:
                logger.info(
                    "No job queue/definition provided, skipping task submission"
                )

            # Upload results if requested
            if args.upload_results:
                results_dir = Path("/tmp/results")
                results_dir.mkdir(exist_ok=True)

                # Save analysis results
                with open(results_dir / "analysis.json", "w") as f:
                    json.dump(analysis_results, f, indent=2)

                s3_key = worker.upload_results_to_s3(results_dir, args.results_prefix)
                logger.info(f"Results uploaded to: {s3_key}")

        elif args.mode == "task":
            # Individual task mode
            if args.task_id is None or not args.simulation_dir or not args.job_dir:
                parser.error(
                    "--task-id, --simulation-dir, and --job-dir are required for task mode"
                )

            # Set local directories
            worker.simulation_dir = Path(args.simulation_dir)
            worker.job_dir = Path(args.job_dir)

            # Run the simulation task
            success = worker.run_simulation_task(args.task_id, {})

            if success:
                logger.info(f"Task {args.task_id} completed successfully")
                sys.exit(0)
            else:
                logger.error(f"Task {args.task_id} failed")
                sys.exit(1)

        # Cleanup
        worker.cleanup()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
