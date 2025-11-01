import getpass
import json
import os
import subprocess
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
import requests
from botocore.exceptions import ClientError, WaiterError

from frequensolve.orchestrator.config.base import BaseSiteConfig
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.orchestrator.tasks.base import BaseTask
from frequensolve.seismic.record_database import RecordDatabase
from frequensolve.simulation.jobs import SimulationJob
from frequensolve.util.setup_logger import init_logger

__all__ = ["AWSSiteConfig", "AWSSite"]

# Initialize the logger
logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/aws.log")


@dataclass
class AWSSiteConfig(BaseSiteConfig):
    """Unified configuration for AWS Site with domain-based setup.

    This class provides configuration for both authentication and AWS Batch execution.
    All configuration is automatically fetched from the domain's public API endpoint.

    Usage:
        # Fetch config from domain
        config = AWSSiteConfig.from_domain('frequensolve.app')

        # Or use FREQUENSOL_DOMAIN environment variable
        export FREQUENSOL_DOMAIN='frequensolve.app'
        config = AWSSiteConfig.from_domain()

    Attributes:
        # Cognito/API settings (auto-populated from domain)
        user_pool_id: Cognito User Pool ID
        client_id: Cognito App Client ID
        identity_pool_id: Cognito Identity Pool ID
        api_url: GraphQL API endpoint URL
        domain: Frontend domain

        # AWS Batch settings (auto-populated from stack info after auth)
        job_queue: AWS Batch job queue name
        job_definition: AWS Batch job definition name
        s3_bucket: S3 bucket for simulation data

        # Shared settings
        region: AWS region
        s3_prefix: S3 prefix for organizing data
        max_duration: Maximum time resources can be requested
    """

    # Cognito/API configuration (from domain or env vars)
    user_pool_id: Optional[str] = None
    client_id: Optional[str] = None
    identity_pool_id: Optional[str] = None
    api_url: Optional[str] = None
    domain: Optional[str] = None

    # AWS Batch configuration (auto-populated from stack info)
    job_queue: str = ""
    job_definition: str = ""
    s3_bucket: Optional[str] = None

    # Shared configuration
    region: str = "us-east-1"
    s3_prefix: str = "frequensolve"
    max_duration: Optional[str] = None

    @classmethod
    def from_domain(cls, domain: Optional[str] = None) -> "AWSSiteConfig":
        """Create configuration by fetching from a domain.

        Args:
            domain: Frontend domain (e.g., 'frequensolve.app', 'localhost:5173')
                   If not provided, will try FREQUENSOL_DOMAIN environment variable.

        Returns:
            AWSSiteConfig instance with Cognito settings populated

        Raises:
            ValueError: If domain is not provided and cannot be inferred
            requests.RequestException: If configuration cannot be fetched
        """
        domain = domain or os.getenv("FREQUENSOL_DOMAIN")
        if not domain:
            raise ValueError(
                "Domain is required. Pass domain parameter or set FREQUENSOL_DOMAIN environment variable."
            )

        # Fetch configuration from domain
        config_data = cls._fetch_config_from_domain(domain)

        # Cache configuration locally
        cls._cache_config(domain, config_data)

        # Create instance from fetched data
        return cls(
            user_pool_id=config_data["auth"]["userPoolId"],
            client_id=config_data["auth"]["clientId"],
            identity_pool_id=config_data["auth"]["identityPoolId"],
            api_url=config_data["api"]["graphqlUrl"],
            region=config_data["region"],
            domain=domain,
        )

    @staticmethod
    def _get_config_cache_path(domain: str) -> Path:
        """Get path to cached configuration file for a domain."""
        config_dir = Path.home() / ".frequensolve"
        config_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize domain for filename
        safe_domain = domain.replace(":", "_").replace("/", "_")
        return config_dir / f"config_{safe_domain}.json"

    @staticmethod
    def _fetch_config_from_domain(domain: str, force_refresh: bool = False) -> dict:
        """Fetch configuration from a domain's public API endpoint.

        Args:
            domain: Frontend domain
            force_refresh: If True, bypass cache and fetch fresh config

        Returns:
            Configuration dictionary

        Raises:
            requests.RequestException: If fetch fails
        """
        # First check cache (unless force_refresh is True)
        cache_path = AWSSiteConfig._get_config_cache_path(domain)
        if not force_refresh and cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cached_config = json.load(f)
                    logger.info(f"Using cached configuration for {domain}")
                    logger.debug(f"Cache path: {cache_path}")
                    return cached_config
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Failed to read cached config, fetching fresh: {e}")

        # Try both HTTPS and HTTP (for local development)
        logger.info(f"Fetching configuration from {domain}...")
        for protocol in ["https", "http"]:
            url = f"{protocol}://{domain}/api/config.json"
            try:
                logger.debug(f"Trying {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                config_data = response.json()
                logger.info(f"✓ Configuration loaded from {domain}")
                return config_data
            except requests.RequestException as e:
                if protocol == "http":  # Last attempt failed
                    raise ValueError(
                        f"Failed to fetch configuration from {domain}. "
                        f"Make sure the domain is correct and accessible. Error: {e}"
                    )
                # Try next protocol
                continue

    @staticmethod
    def _cache_config(domain: str, config_data: dict):
        """Cache configuration locally for faster subsequent access."""
        cache_path = AWSSiteConfig._get_config_cache_path(domain)
        try:
            with open(cache_path, "w") as f:
                json.dump(config_data, f, indent=2)
            logger.debug(f"Cached configuration to {cache_path}")
        except IOError as e:
            logger.debug(f"Failed to cache configuration: {e}")

    def __post_init__(self):
        """Initialize configuration from domain if provided."""
        # If domain is provided but Cognito fields aren't, fetch from domain
        if self.domain and not self.user_pool_id and not self.client_id:
            config_data = self._fetch_config_from_domain(self.domain)
            self.user_pool_id = config_data["auth"]["userPoolId"]
            self.client_id = config_data["auth"]["clientId"]
            self.identity_pool_id = config_data["auth"]["identityPoolId"]
            self.api_url = config_data["api"]["graphqlUrl"]
            self.region = config_data["region"]


class AWSSite(BaseSite):
    """AWS Batch site for running FrequenSolve simulations.

    Automatically authenticates users and configures AWS resources by simply providing
    your FrequenSol domain. All configuration is discovered automatically.

    Usage:
        # Just provide your domain - everything else is automatic
        site = AWSSite(domain='frequensolve.app')

        # Or set FREQUENSOL_DOMAIN environment variable once
        export FREQUENSOL_DOMAIN='frequensolve.app'
        site = AWSSite()

        # Provide credentials to skip interactive prompt
        site = AWSSite(domain='frequensolve.app', email='user@example.com', password='...')

    What happens automatically:
        1. Fetches configuration from domain (User Pool ID, API URL, etc.)
        2. Authenticates with cached credentials (or prompts for login)
        3. Fetches your infrastructure details (S3 bucket, Batch queue, etc.)
        4. Ready to submit jobs and upload files!
    """

    def __init__(
        self,
        domain: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize AWS Batch site with domain-based authentication.

        Args:
            domain: Frontend domain (e.g., 'frequensolve.app', 'localhost:5173').
                   If not provided, will try FREQUENSOL_DOMAIN environment variable.
            email: User email (optional - will prompt if not provided and not cached).
            password: User password (optional - will prompt if not provided and not cached).

        Raises:
            ValueError: If domain cannot be determined or authentication fails.
            RuntimeError: If infrastructure is not deployed or stack info cannot be fetched.
        """
        from frequensolve.orchestrator.sites.aws.cognito import CognitoAuth
        from frequensolve.orchestrator.sites.aws.graphql_client import GraphQLClient

        # Load configuration from domain
        config = AWSSiteConfig.from_domain(domain)

        # Store domain for potential config refresh
        if domain is None:
            domain = os.getenv("FREQUENSOL_DOMAIN") or config.domain

        # Initialize authentication
        auth = CognitoAuth(
            user_pool_id=config.user_pool_id,
            client_id=config.client_id,
            identity_pool_id=config.identity_pool_id,
            region=config.region,
        )

        # Try to use cached tokens first
        auth_successful = False
        max_retries = 1  # Allow one retry with fresh config

        for attempt in range(max_retries + 1):
            try:
                auth.get_cached_tokens()
                logger.info("Using cached credentials")
                auth_successful = True
                break
            except ValueError:
                # No cached tokens - need to login
                if not email:
                    email = input("FrequenSol Email: ")
                if not password:
                    password = getpass.getpass("Password: ")

                logger.info(f"Authenticating as {email}...")
                auth.login(email, password)
                logger.info("✓ Authentication successful")
                auth_successful = True
                break
            except ClientError as e:
                # Check if this is a ResourceNotFoundException (invalid cached config)
                if (
                    e.response["Error"]["Code"] == "ResourceNotFoundException"
                    and attempt == 0
                ):
                    logger.warning(
                        "⚠️  Cached configuration is outdated (AWS resources not found)"
                    )
                    logger.info("🔄 Refetching configuration from domain...")

                    # Refetch configuration (force refresh to bypass cache)
                    config_data = AWSSiteConfig._fetch_config_from_domain(
                        domain, force_refresh=True
                    )

                    # Cache the new configuration
                    AWSSiteConfig._cache_config(domain, config_data)

                    # Create new config instance
                    config = AWSSiteConfig(
                        user_pool_id=config_data["auth"]["userPoolId"],
                        client_id=config_data["auth"]["clientId"],
                        identity_pool_id=config_data["auth"]["identityPoolId"],
                        api_url=config_data["api"]["graphqlUrl"],
                        region=config_data["region"],
                        domain=domain,
                    )

                    # Reinitialize auth with new config
                    auth = CognitoAuth(
                        user_pool_id=config.user_pool_id,
                        client_id=config.client_id,
                        identity_pool_id=config.identity_pool_id,
                        region=config.region,
                    )

                    # Clear old credentials since they're for wrong User Pool
                    if auth.credentials_path.exists():
                        auth.credentials_path.unlink()
                        logger.info("🗑️  Cleared outdated cached credentials")

                    # Continue to next iteration to try again with new config
                    continue
                else:
                    # Either not a ResourceNotFoundException or we already retried
                    raise

        if not auth_successful:
            raise RuntimeError(
                "Failed to authenticate after retrying with fresh configuration"
            )

        # Store auth for later use
        self.cognito_auth = auth

        # Get AWS credentials from Identity Pool
        credentials = auth.get_aws_credentials()
        self.session = boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretKey"],
            aws_session_token=credentials["SessionToken"],
            region_name=config.region,
        )

        # Initialize GraphQL client
        self.graphql_client = GraphQLClient(config.api_url, auth)

        # Fetch stack info from API to populate config
        try:
            stack_info = self.graphql_client.get_my_stack()

            # Update config with stack information
            config.job_queue = stack_info["jobQueue"]
            config.job_definition = stack_info["jobDefinition"]
            config.s3_bucket = stack_info["bucketName"]

            logger.info(
                f"Stack info loaded: bucket={config.s3_bucket}, queue={config.job_queue}"
            )

        except Exception as e:
            logger.error(f"Failed to fetch stack info: {e}")
            raise RuntimeError(
                f"Failed to fetch stack information: {e}\n"
                f"Please ensure you have deployed both storage and compute infrastructure at https://{config.domain}"
            ) from e

        self.config = config
        self.batch_client = self.session.client("batch", region_name=self.config.region)
        self.s3_client = self.session.client("s3", region_name=self.config.region)

    def _validate_config(self):
        """Validate AWS Batch configuration."""
        try:
            print("=== AWS Credentials and Access Test ===")
            self._check_credentials()
            self._check_profile()
            # self._test_aws_access()
            # self._list_job_definitions()

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

        If using Cognito authentication, submits via GraphQL API.
        Otherwise, uses the traditional REST API method.

        Args:
            job: The task to submit.
            **kwargs: Additional job parameters (vcpu, memory, name, description).

        Returns:
            AWS Batch job ID.

        Raises:
            RuntimeError: If job submission fails.
        """
        try:
            # Sync simulation and job data to S3
            local_sim = job.simulation._path
            remote_sim = job.simulation._remote_path
            s3_sim_key = self.sync_s3(local_sim, remote_sim)

            project = remote_sim.parts[0]
            local_job, remote_job = job.save_for_remote(
                self.__class__.__name__, project
            )
            s3_job_key = self.sync_s3(local_job, remote_job)

            # Check if using Cognito/GraphQL authentication
            if self.graphql_client is not None:
                # New path: Submit via GraphQL API
                logger.info(f"Submitting job via GraphQL API: {s3_job_key}")

                result = self.graphql_client.submit_job(
                    job_file_s3_key=str(s3_job_key),
                    vcpu=kwargs.get("vcpu"),
                    memory=kwargs.get("memory"),
                    job_name=kwargs.get("name", f"frequensolve-{uuid.uuid4().hex[:8]}"),
                )

                batch_job_id = result["batchJobId"]
                logger.info(f"✓ Job submitted: {batch_job_id}")
                logger.info(f"  Simulation ID: {result['simulationId']}")
                logger.info(f"  Status: {result['status']}")

                return batch_job_id

            else:
                # Old path: Submit via REST API (backwards compatibility)
                logger.info(f"Submitting job via REST API: {s3_job_key}")

                # Prepare the API request data
                api_data = {
                    "name": kwargs.get("name", f"frequensolve-{uuid.uuid4().hex[:8]}"),
                    "description": kwargs.get("description", ""),
                    "job_s3_key": str(s3_job_key),
                }

                if "vcpu" in kwargs:
                    api_data["vcpu"] = kwargs["vcpu"]
                if "memory" in kwargs:
                    api_data["memory"] = kwargs["memory"]

                # Make API request to submit the job
                headers = {}
                if self.config.api_token:
                    headers["Authorization"] = f"Bearer {self.config.api_token}"

                response = requests.post(
                    f"{self.config.api_base_url}/api/simulation/run/",
                    json=api_data,
                    headers=headers,
                    timeout=1800,
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"API request failed with status {response.status_code}: {response.text}"
                    )

                response_data = response.json()
                if response_data.get("status") != "success":
                    raise RuntimeError(
                        f"Job submission failed: {response_data.get('message', 'Unknown error')}"
                    )

                batch_job_id = response_data.get("batch_job_id")
                if not batch_job_id:
                    raise RuntimeError("No batch job ID returned from API")

                return batch_job_id

        except Exception as e:
            raise RuntimeError(f"Failed to submit job: {e}")

    def fetch_traces(
        self,
        job: Union[SimulationJob, List[SimulationJob]],
        path: Optional[Union[str, Path]] = None,
        upscale: int = 1,
    ) -> Union[RecordDatabase, Dict[str, RecordDatabase]]:
        """Get results from Stampede3.

        Args:
            job: A SimulationJob object.
            path: The path to save the results to.
        """

        if isinstance(job, SimulationJob):
            jobs = [job]
        else:
            jobs = job

        if path is None:
            path = jobs[0].project_path
        else:
            path = Path(path)

        db_map = {}

        for job in jobs:
            try:
                # files = job.records["datasets"].keys()

                # # Create temporary directory name for the payload
                # payload_name = f"records_{int(time.time())}"
                # remote_payload = self.work_dir / f"{payload_name}.tar.gz"
                # local_payload = path / f"{payload_name}.tar.gz"

                # print(payload_name)
                # print(remote_payload)
                # print(local_payload)
                # # print(files)

                # # Create payload on remote
                # tar_cmd = f"cd {self.work_dir} && tar czf {remote_payload.name} "
                # tar_cmd += " ".join(files)
                # _, _, stderr = self.run_login_cmd(tar_cmd)
                # print(stderr.read().decode().strip())
                # # err = stderr.read().decode().strip()
                # # if err:
                # #     raise RuntimeError(f"Failed to create payload on remote: {err}")

                # Build the path for the s3 results directory and the local results directory.
                # The job results are stored in the job's result_path, not the simulation path
                # Format: ex_01/jobs/simulation_name/job_name/results/receivers/
                project_name = job.simulation._remote_path.parts[0]  # e.g., "ex_01"
                simulation_name = job.simulation.name  # e.g., "simple_acoustic"
                job_name = job.name  # e.g., "time"

                results_receivers_path = (
                    f"jobs/{simulation_name}/{job_name}/results/receivers"
                )
                s3_results_path = f"s3://{self.config.s3_bucket}/{project_name}/{results_receivers_path}"
                local_results_path = path / results_receivers_path
                self.get(s3_results_path, local_results_path)

                # cwd = os.getcwd()
                # os.chdir(local_payload.parent)
                # with tarfile.open(local_payload, "r:gz") as tar:
                #     logger.debug("Extracting files from payload:")
                #     tar.extractall()
                # os.chdir(cwd)

                # local_payload.unlink()
                # self.run_login(f"rm {remote_payload}")

                # TODO: Copy job, simulation file to database so that it can be read independently.

                db = RecordDatabase.from_results(job.records, path.resolve(), upscale)
                db_map[job.name] = db

            except Exception as e:
                logger.exception("Error downloading records: %s", str(e))
                raise

        if len(db_map) == 1:
            return db_map[jobs[0].name]
        else:
            return db_map

    def test_api_connectivity(self) -> bool:
        """Test connectivity to the FrequenSol API endpoint.

        Returns:
            True if API is accessible, False otherwise.
        """
        try:
            headers = {}
            if self.config.api_token:
                headers["Authorization"] = f"Token {self.config.api_token}"

            response = requests.get(
                f"{self.config.api_base_url}/api/simulation/",
                headers=headers,
                timeout=10,
            )

            return response.status_code in [
                200,
                401,
                403,
            ]  # 401/403 means endpoint exists but auth failed
        except Exception as e:
            warnings.warn(f"API connectivity test failed: {e}")
            return False

    def get_job_status_from_api(self, job_id: str) -> Dict[str, Any]:
        """Get job status from the FrequenSol API.

        Args:
            job_id: The job ID to check.

        Returns:
            Dictionary containing job status information.
        """
        try:
            headers = {}
            if self.config.api_token:
                headers["Authorization"] = f"Token {self.config.api_token}"

            response = requests.get(
                f"{self.config.api_base_url}/api/simulation/jobs/{job_id}/",
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()
            else:
                warnings.warn(
                    f"Failed to get job status from API: {response.status_code}"
                )
                return {}
        except Exception as e:
            warnings.warn(f"API request failed: {e}")
            return {}

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

    def get(
        self,
        s3_path: Union[str, Path],
        local_path: Union[str, Path],
        overwrite: bool = False,
    ):
        """Transfer files from S3 path to local path using aws s3 sync.

        Args:
            s3_path: S3 path to transfer from (e.g., 's3://bucket/key' or 's3://bucket/key/')
            local_path: Local path to transfer to
            overwrite: Overwrite existing files (not used with aws s3 sync as it always overwrites)
        """
        logger.debug("Attempting to transfer from %s to %s", s3_path, local_path)

        local_path = Path(local_path)
        s3_path = str(s3_path)

        try:
            # Create parent directory on local if it doesn't exist
            parent_path = str(local_path.parent)
            os.makedirs(parent_path, exist_ok=True)

            # Use aws s3 sync command
            # The sync command will:
            # - Download all files from the S3 path to the local path
            # - Automatically handle directories vs files
            # - Skip files that are already up-to-date
            # - Overwrite files that have changed
            sync_cmd = ["aws", "s3", "sync", s3_path, str(local_path)]

            logger.debug("aws s3 sync command: %s", " ".join(sync_cmd))

            result = subprocess.run(
                sync_cmd,
                capture_output=True,
                text=True,
                check=True,  # This will raise CalledProcessError if the command fails
            )

            if result.stdout:
                logger.debug("aws s3 sync output: %s", result.stdout.strip())

            logger.debug("Transfer completed successfully")

        except subprocess.CalledProcessError as e:
            logger.error(
                "aws s3 sync failed with return code %d: %s", e.returncode, e.stderr
            )
            raise RuntimeError(f"aws s3 sync failed: {e.stderr}")
        except Exception as e:
            logger.exception("Error during S3 file transfer: %s", str(e))
            raise


# if __name__ == "__main__":
#     # Example usage
#     config = AWSSiteConfig(
#         job_queue="frequensolve-queue",
#         job_definition="frequensolve-job-definition",
#         region="us-east-1",
#         s3_bucket="my-frequensolve-bucket",
#         api_base_url="https://api.frequensol.com",
#         api_token="your-api-token-here"
#     )

#     site = AWSSite(config)

#     # Example job submission via API
#     # job_id = site.submit(job)
#     # print(f"Submitted job: {job_id}")
#     #
#     # # Wait for completion
#     # success = site.wait_for_completion(job_id)
#     # print(f"Job completed successfully: {success}")
