import json
import os
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import boto3
from botocore.exceptions import ClientError, WaiterError

from frequensolve.orchestrator.config.base import BaseSiteConfig
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.orchestrator.tasks.base import BaseTask
from frequensolve.simulation.jobs import SimulationJob

__all__ = ["AWSSiteConfig", "AWSSite"]


@dataclass
class AWSSiteConfig(BaseSiteConfig):
    """AWS Batch configuration for job execution.

    This class defines resource requirements and constraints for job execution
    on AWS Batch using storage optimized instances with local disks.

    Args:
        job_queue: AWS Batch job queue name.
        job_definition: AWS Batch job definition name.
        region: AWS region for Batch operations.
        s3_bucket: S3 bucket for storing simulation and job data.
        s3_prefix: S3 prefix for organizing data (default: 'frequensolve').
        max_duration: Maximum time resources can be requested (HH:MM:SS).
        config_file: Path to configuration file.

    Attributes:
        job_queue: AWS Batch job queue name.
        job_definition: AWS Batch job definition name.
        region: AWS region for Batch operations.
        s3_bucket: S3 bucket for storing simulation and job data.
        s3_prefix: S3 prefix for organizing data.
        max_duration: Maximum time resources can be requested (HH:MM:SS).
    """

    job_queue: str
    job_definition: str
    region: Optional[str] = None
    s3_bucket: Optional[str] = None
    s3_prefix: str = "frequensolve"
    max_duration: Optional[str] = None
    config_file: Optional[Union[str, Path]] = None

    def __post_init__(self):
        """Initialize configuration from file if provided."""
        if self.config_file is not None:
            with open(self.config_file, "r") as f:
                config = json.load(f)
            self.region = config.get("region", self.region)
            self.s3_bucket = config.get("s3_bucket", self.s3_bucket)
            self.job_queue = config.get("job_queue", self.job_queue)
            self.job_definition = config.get("job_definition", self.job_definition)

    @classmethod
    def from_dict(cls, data: dict) -> "AWSSiteConfig":
        """Create configuration from dictionary."""
        return cls(**data)

    def __dict__(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "job_queue": self.job_queue,
            "job_definition": self.job_definition,
            "region": self.region,
            "s3_bucket": self.s3_bucket,
            "s3_prefix": self.s3_prefix,
            "max_duration": self.max_duration,
        }

    @classmethod
    def load(cls, name: str) -> "AWSSiteConfig":
        """Load configuration from JSON file."""
        name += ".json" if not name.endswith(".json") else ""
        try:
            with open(name, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {name}: {e}")
        return cls.from_dict(data)

    def save(self, name: str) -> None:
        """Save configuration to JSON file."""
        name += ".json" if not name.endswith(".json") else ""
        try:
            with open(name, "w") as f:
                json.dump(self.__dict__(), f, indent=3)
        except Exception as e:
            warnings.warn(f"Failed to save configuration to {name}: {e}")


class AWSSite(BaseSite):
    """AWS Batch site for running FrequenSolve simulations."""

    def __init__(self, session: boto3.Session, config: AWSSiteConfig):
        """Initialize AWS Batch site.

        Args:
            session: boto3 Session to use for AWS operations.
            config: AWS Batch configuration.
        """
        self.session = session
        self.config = config
        self.batch_client = session.client("batch", region_name=self.config.region)
        self.s3_client = session.client("s3", region_name=self.config.region)
        self._validate_config()

    def _validate_config(self):
        """Validate AWS Batch configuration."""
        try:
            print("=== AWS Credentials and Access Test ===")
            self._check_credentials()
            self._check_profile()
            self._test_aws_access()
            self._list_job_definitions()

            print("\n=== Configuration Validation ===")

            queues = self.batch_client.describe_job_queues(
                jobQueues=[self.config.job_queue]
            )
            if not queues["jobQueues"]:
                raise ValueError(f"Job queue '{self.config.job_queue}' not found")

            definitions = self.batch_client.describe_job_definitions(
                jobDefinitionName=self.config.job_definition
            )
            if not definitions["jobDefinitions"]:
                raise ValueError(
                    f"Job definition '{self.config.job_definition}' not found"
                )

            active_definitions = [
                jd for jd in definitions["jobDefinitions"] if jd["status"] == "ACTIVE"
            ]
            if not active_definitions:
                raise ValueError(
                    f"Job definition '{self.config.job_definition}' exists but is not ACTIVE. Available statuses: {[jd['status'] for jd in definitions['jobDefinitions']]}"
                )

            print(f"✓ Found job queue: {queues['jobQueues'][0]['jobQueueArn']}")
            print(
                f"✓ Found ACTIVE job definition: {active_definitions[0]['jobDefinitionArn']}"
            )
            print(f"✓ Configuration validation successful!")

        except ClientError as e:
            raise RuntimeError(f"Failed to validate AWS Batch configuration: {e}")

    def _check_credentials(self):
        """Check what AWS credentials are being used."""
        try:
            # Use the session's STS client to get caller identity
            sts_client = self.session.client("sts")
            identity = sts_client.get_caller_identity()
            print(f"Account ID: {identity['Account']}")
            print(f"User ID: {identity['UserId']}")
            print(f"ARN: {identity['Arn']}")

            # Also check the region being used
            print(f"Region: {self.config.region}")

        except Exception as e:
            print(f"Error checking credentials: {e}")

    def _check_profile(self):
        """Check what AWS profile is being used and list available profiles."""
        try:
            # Check current profile
            current_profile = self.session.profile_name
            print(f"Current profile: {current_profile}")

            # List available profiles
            import subprocess

            result = subprocess.run(
                ["aws", "configure", "list-profiles"],
                capture_output=True,
                text=True,
                check=True,
            )
            profiles = result.stdout.strip().split("\n")
            print(f"Available profiles: {profiles}")

        except Exception as e:
            print(f"Error checking profile: {e}")

    def _test_aws_access(self):
        """Test basic AWS access to see if credentials are working."""
        try:
            # Test STS access
            sts_client = self.session.client("sts")
            identity = sts_client.get_caller_identity()
            print(f"✓ STS access successful - Account: {identity['Account']}")

            # Test Batch access
            response = self.batch_client.describe_job_queues()
            print(
                f"✓ Batch access successful - Found {len(response['jobQueues'])} job queues"
            )

            # Test S3 access
            response = self.s3_client.list_buckets()
            print(f"✓ S3 access successful - Found {len(response['Buckets'])} buckets")

            # Test specific S3 bucket access
            try:
                response = self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
                print(f"✓ S3 bucket '{self.config.s3_bucket}' access successful")
            except Exception as e:
                print(f"✗ S3 bucket '{self.config.s3_bucket}' access failed: {e}")

        except Exception as e:
            print(f"✗ AWS access test failed: {e}")

    def _list_job_definitions(self):
        """List all available job definitions to debug access issues."""
        try:
            response = self.batch_client.describe_job_definitions()
            print(f"Found {len(response['jobDefinitions'])} job definitions:")

            # Group by name and show statuses
            definitions_by_name = {}
            for jd in response["jobDefinitions"]:
                name = jd["jobDefinitionName"]
                if name not in definitions_by_name:
                    definitions_by_name[name] = []
                definitions_by_name[name].append(jd["status"])

            for name, statuses in definitions_by_name.items():
                status_str = ", ".join(statuses)
                print(f"  - {name} (Statuses: {status_str})")

        except Exception as e:
            print(f"Error listing job definitions: {e}")

    def sync_s3(self, local_path: Union[str, Path], s3_key: str) -> str:
        """Sync files/directories with S3 using boto3.

        Args:
            local_path: Local path to sync.
            s3_key: S3 key where data should be synced.

        Returns:
            S3 key where data was synced.

        Raises:
            RuntimeError: If sync fails.
        """
        local_path = Path(local_path)

        if not local_path.exists():
            raise FileNotFoundError(f"Path {local_path} does not exist")

        try:
            if local_path.is_file():
                # Upload single file
                self.s3_client.upload_file(
                    str(local_path), self.config.s3_bucket, str(s3_key)
                )
            else:
                # TODO: Would be nice to use s3 sync here, but then have to use CLI
                #       and it doesn't have user temporary credentials

                # Upload directory recursively using boto3
                for file_path in local_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        file_s3_key = f"{s3_key}/{relative_path}"
                        file_s3_key = file_s3_key.replace("\\", "/")

                        self.s3_client.upload_file(
                            str(file_path),
                            self.config.s3_bucket,
                            str(file_s3_key),
                        )
            return s3_key

        except ClientError as e:
            raise RuntimeError(f"Failed to sync {local_path} to S3: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error syncing {local_path} to S3: {e}")

    # Note: We don't need to sync the whole project with S3, leaving this unimplemented for now
    def sync(self, project):
        pass

    def submit(self, job: SimulationJob, **kwargs) -> str:
        """Submit a job to AWS Batch.

        Args:
            job: The task to submit.
            simulation_dir: Path to simulation directory.
            job_dir: Path to job directory.
            **kwargs: Additional job parameters.

        Returns:
            AWS Batch job ID.

        Raises:
            RuntimeError: If job submission fails.
        """
        try:
            local_sim = job.simulation._path
            remote_sim = job.simulation._remote_path
            s3_sim_key = self.sync_s3(local_sim, remote_sim)

            project = remote_sim.parts[0]
            local_job, remote_job = job.save_for_remote(
                self.__class__.__name__, project
            )
            s3_job_key = self.sync_s3(local_job, remote_job)

            # Prepare job parameters
            job_params = {
                "simulation_s3_key": str(s3_sim_key),
                "job_s3_key": str(s3_job_key),
                "s3_bucket": self.config.s3_bucket,
                **kwargs,
            }

            # # Submit job to AWS Batch
            # response = self.batch_client.submit_job(
            #     jobName=f"frequensolve-{uuid.uuid4().hex[:8]}",
            #     jobQueue=self.config.job_queue,
            #     jobDefinition=self.config.job_definition,
            #     parameters=job_params,
            #     containerOverrides={
            #         "environment": [
            #             {
            #                 "name": "SIMULATION_S3_KEY",
            #                 "value": f"{s3_sim_key}"
            #             },
            #             {
            #                 "name": "JOB_S3_KEY",
            #                 "value": f"{s3_job_key}"
            #             },
            #             {
            #                 "name": "S3_BUCKET",
            #                 "value": self.config.s3_bucket
            #             },
            #         ]
            #     }
            # )

            # return response["jobId"]

        except Exception as e:
            raise RuntimeError(f"Failed to submit job to AWS Batch: {e}")

    def check_status(self, job_id: str) -> str:
        """Check the status of a submitted job.

        Args:
            job_id: AWS Batch job ID.

        Returns:
            Job status string.
        """
        try:
            response = self.batch_client.describe_jobs(jobs=[job_id])
            if response["jobs"]:
                return response["jobs"][0]["status"]
            return "unknown"
        except ClientError:
            return "unknown"

    def wait_for_completion(self, job_id: str, timeout: int = 3600) -> bool:
        """Wait for job completion.

        Args:
            job_id: AWS Batch job ID.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if job completed successfully, False otherwise.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.check_status(job_id)

            if status == "SUCCEEDED":
                return True
            elif status in ["FAILED", "CANCELLED"]:
                return False
            elif status == "RUNNING":
                time.sleep(30)  # Check every 30 seconds for running jobs
            else:
                time.sleep(10)  # Check every 10 seconds for queued jobs

        return False

    def cancel_job(self, job_id: str) -> None:
        """Cancel a running job.

        Args:
            job_id: AWS Batch job ID.
        """
        try:
            self.batch_client.cancel_job(jobId=job_id, reason="Cancelled by user")
        except ClientError as e:
            warnings.warn(f"Failed to cancel job {job_id}: {e}")

    def list_jobs(self, job_status: Optional[str] = None) -> list:
        """List jobs with optional status filter.

        Args:
            job_status: Optional job status to filter by.

        Returns:
            List of job information dictionaries.
        """
        try:
            params = {"jobQueue": self.config.job_queue}
            if job_status:
                params["jobStatus"] = job_status

            response = self.batch_client.list_jobs(**params)
            return response["jobSummaryList"]
        except ClientError as e:
            warnings.warn(f"Failed to list jobs: {e}")
            return []

    def get_job_logs(self, job_id: str) -> Dict[str, Any]:
        """Get logs and details for a completed job.

        Args:
            job_id: AWS Batch job ID.

        Returns:
            Dictionary containing job logs and details.
        """
        try:
            response = self.batch_client.describe_jobs(jobs=[job_id])
            if response["jobs"]:
                job = response["jobs"][0]
                return {
                    "status": job["status"],
                    "started_at": job.get("startedAt"),
                    "stopped_at": job.get("stoppedAt"),
                    "exit_code": job.get("attempts", [{}])[-1].get("exitCode"),
                    "reason": job.get("attempts", [{}])[-1].get("reason"),
                    "log_stream_name": job.get("attempts", [{}])[-1].get(
                        "logStreamName"
                    ),
                }
            return {}
        except ClientError as e:
            warnings.warn(f"Failed to get job logs for {job_id}: {e}")
            return {}


# if __name__ == "__main__":
#     # Example usage
#     config = AWSSiteConfig(
#         job_queue="frequensolve-queue",
#         job_definition="frequensolve-job-definition",
#         instance_type="i3.xlarge",
#         region="us-east-1",
#         s3_bucket="my-frequensolve-bucket"
#     )

#     site = AWSSite(config)

#     # Example job submission
#     # job_id = site.submit(job, "/path/to/simulation", "/path/to/job")
#     # print(f"Submitted job: {job_id}")
#     #
#     # # Wait for completion
#     # success = site.wait_for_completion(job_id)
#     # print(f"Job completed successfully: {success}")
