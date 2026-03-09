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
from botocore.exceptions import ClientError

from frequensolve.orchestrator.config.base import BaseSiteConfig
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.seismic.record_database import RecordDatabase
from frequensolve.simulation.jobs import SimulationJob
from frequensolve.util.setup_logger import init_logger

__all__ = ["AWSSiteConfig", "AWSSite"]

# Initialize the logger
logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/aws.log")


@dataclass
class AWSSiteConfig(BaseSiteConfig):
    """Unified configuration for AWS Site with domain-based setup.

    This class provides configuration for authentication and AWS resource access.
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

        # AWS settings (auto-populated from stack info after auth)
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

    # AWS configuration (auto-populated from stack info)
    s3_bucket: Optional[str] = None

    # Shared configuration
    region: str = "us-east-1"
    s3_prefix: str = ""
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
    """AWS site for running FrequenSolve simulations.

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
        3. Fetches your infrastructure details (S3 bucket, etc.)
        4. Ready to submit simulations and upload files!
    """

    def __init__(
        self,
        domain: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize AWS site with domain-based authentication.

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

        # Fetch storage stack info from API to populate config
        # Storage stack is required for S3 operations; compute stack will be created on-demand
        try:
            # Try to get storage stack info
            storage_info = self.graphql_client.get_storage_stack_info()
            config.s3_bucket = storage_info["bucketName"]
            logger.info(f"Stack info loaded: bucket={config.s3_bucket}")
        except RuntimeError as e:
            error_msg = str(e).lower()
            # If storage stack doesn't exist, we'll create it on first sync
            if "no active storage stack" in error_msg:
                logger.info(
                    "Storage stack not found. It will be created automatically on first sync."
                )
                # Don't set bucket yet - it will be set after stack creation in sync()
                config.s3_bucket = None
            else:
                logger.error(f"Failed to fetch storage stack info: {e}")
                raise RuntimeError(
                    f"Failed to fetch storage stack information: {e}\n"
                    f"Storage stack will be created automatically on first sync."
                ) from e
        except Exception as e:
            logger.error(f"Failed to fetch stack info: {e}")
            raise RuntimeError(f"Failed to fetch stack information: {e}") from e

        self.config = config
        self.s3_client = self.session.client("s3", region_name=self.config.region)

    def _refresh_s3_credentials(self) -> None:
        """Refresh session and S3 client with fresh Identity Pool credentials.

        Call when credentials may have expired (e.g. after a long simulation).
        Only applies when using Cognito authentication.
        """
        if not hasattr(self, "cognito_auth") or self.cognito_auth is None:
            return
        credentials = self.cognito_auth.get_aws_credentials()
        self.session = boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretKey"],
            aws_session_token=credentials["SessionToken"],
            region_name=self.config.region,
        )
        self.s3_client = self.session.client("s3", region_name=self.config.region)
        logger.debug("Refreshed S3 credentials from Identity Pool")

    @property
    def work_dir(self) -> Path:
        """Get the S3 work directory as a Path-like object.

        Returns:
            Path object representing the S3 prefix.
        """
        # Return a Path object that represents the S3 prefix
        # This is used by project._transfer() to construct remote paths
        return Path(self.config.s3_prefix)

    def _validate_config(self):
        """Validate AWS configuration."""
        try:
            print("=== AWS Credentials and Access Test ===")
            self._check_credentials()
            self._check_profile()

            print("\n=== Configuration Validation ===")
            print(f"✓ S3 bucket: {self.config.s3_bucket}")
            print(f"✓ Configuration validation successful!")

        except Exception as e:
            raise RuntimeError(f"Failed to validate AWS configuration: {e}")

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

    def sync(self, project):
        """Sync the project to S3.

        Automatically creates storage stack if it doesn't exist.

        Args:
            project: The Project to sync.

        Raises:
            RuntimeError: If sync fails or stack creation fails.
        """
        # Ensure we have a bucket name (create storage stack if needed)
        if self.graphql_client is not None:
            # Check if we need to get/create storage stack
            if (
                not self.config.s3_bucket
                or not self.graphql_client._check_storage_stack_exists()
            ):
                # Try to get existing storage stack first
                try:
                    storage_info = self.graphql_client.get_storage_stack_info()
                    self.config.s3_bucket = storage_info["bucketName"]
                    logger.info(
                        f"Using existing storage stack: bucket={self.config.s3_bucket}"
                    )
                except RuntimeError:
                    # Storage stack doesn't exist, create it
                    logger.info(
                        "Storage stack not found. Creating storage infrastructure..."
                    )
                    try:
                        environment = getattr(self.config, "environment", "dev")

                        # Deploy storage stack (userId extracted automatically from auth context)
                        deploy_result = self.graphql_client.deploy_storage_stack(
                            environment
                        )

                        # Wait for stack to be ready (pass stackId so we wait for the one we just created)
                        logger.info("Waiting for storage stack to be ready...")
                        expected_stack_id = deploy_result.get("stackId")
                        self.graphql_client.wait_for_stack_ready(
                            "storage", expected_stack_id=expected_stack_id
                        )

                        # Get storage stack info to update bucket name
                        storage_info = self.graphql_client.get_storage_stack_info()
                        self.config.s3_bucket = storage_info["bucketName"]
                        logger.info(
                            f"✓ Storage stack ready: bucket={storage_info['bucketName']}"
                        )
                    except Exception as create_error:
                        raise RuntimeError(
                            f"Failed to create storage stack: {create_error}"
                        ) from create_error

        logger.info(f"Syncing project '{project.name}' to S3...")
        project._transfer(self)
        logger.info(f"✓ Project '{project.name}' synced to S3")

    def submit(self, job: SimulationJob, **kwargs) -> str:
        """Submit a simulation job.

        Automatically creates compute stack if it doesn't exist.

        If using Cognito authentication, submits via GraphQL API.
        Otherwise, uses the traditional REST API method.

        Args:
            job: The task to submit.
            **kwargs: Additional job parameters (vcpu, memory, name, description).

        Returns:
            Simulation ID (for GraphQL path) or job ID (for REST API path).

        Raises:
            RuntimeError: If job submission fails or stack creation fails.
        """
        # Check if compute stack exists, create if missing
        if self.graphql_client is not None:
            if not self.graphql_client._check_compute_stack_exists():
                # Compute stack doesn't exist, create it
                logger.info(
                    "Compute stack not found. Creating compute infrastructure..."
                )
                try:
                    environment = getattr(self.config, "environment", "dev")

                    # Deploy compute stack - backend will automatically fetch and use user's compute settings
                    # (userId extracted automatically from auth context)
                    deploy_result = self.graphql_client.deploy_compute_stack(
                        environment
                    )

                    # Wait for stack to be ready, passing the stackId from deployment for accurate matching
                    logger.info("Waiting for compute stack to be ready...")
                    expected_stack_id = deploy_result.get("stackId")
                    stack_info = self.graphql_client.wait_for_stack_ready(
                        "compute", expected_stack_id=expected_stack_id
                    )

                    logger.info(
                        f"✓ Compute stack ready: {stack_info.get('stackId', 'unknown')}"
                    )
                except Exception as create_error:
                    raise RuntimeError(
                        f"Failed to create compute stack: {create_error}"
                    ) from create_error

        try:
            # Sync job file to S3
            project = job.simulation._remote_path.parts[0]
            local_job, remote_job = job.save_for_remote(
                self.__class__.__name__, project
            )
            s3_job_key = self.sync_s3(local_job, remote_job)
            logger.info(f"✓ Synced job file to S3: {s3_job_key}")

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

                simulation_id = result["simulationId"]
                logger.info(f"✓ Simulation submitted")
                logger.info(f"  Simulation ID: {simulation_id}")
                logger.info(f"  Status: {result['status']}")

                return simulation_id

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

                # Check if simulation_id is available in REST API response
                simulation_id = response_data.get("simulation_id")
                if simulation_id:
                    logger.info(f"✓ Simulation submitted via REST API")
                    logger.info(f"  Simulation ID: {simulation_id}")
                    return simulation_id

                # Fallback to job_id for backwards compatibility
                job_id = response_data.get("batch_job_id") or response_data.get(
                    "job_id"
                )
                if not job_id:
                    raise RuntimeError("No job ID or simulation ID returned from API")

                logger.warning(
                    "REST API did not return simulation_id, returning job_id. "
                    "Consider using GraphQL API for full functionality."
                )
                return job_id

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

    def wait_for_completion(
        self, simulation_id: str, poll_interval: float = 10, timeout: int = 3600
    ) -> str:
        """Wait for simulation completion by polling status.

        Polls the simulation status at the specified interval until the
        simulation reaches a terminal state (SUCCEEDED, FAILED, or CANCELED) or timeout.

        Args:
            simulation_id: The simulation ID to poll.
            poll_interval: Interval in seconds between status checks.
            timeout: Maximum time to wait in seconds (default: 3600).

        Returns:
            Final simulation status string ('SUCCEEDED', 'FAILED', or 'CANCELED').

        Raises:
            RuntimeError: If simulation not found, polling fails, or timeout exceeded.
        """
        if self.graphql_client is None:
            raise RuntimeError(
                "GraphQL client not available. Cannot poll simulation status. "
                "This method requires Cognito authentication."
            )

        logger.info(
            f"Polling simulation {simulation_id} every {poll_interval} seconds "
            f"(timeout: {timeout}s)..."
        )

        start_time = time.time()

        while True:
            # Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                raise RuntimeError(
                    f"Timeout waiting for simulation {simulation_id} to complete "
                    f"(waited {elapsed_time:.0f}s, timeout: {timeout}s)"
                )

            try:
                status = self.graphql_client.get_simulation_status(simulation_id)
                logger.debug(f"Simulation {simulation_id} status: {status}")

                # Check if simulation is in a terminal state
                if status in ["SUCCEEDED", "FAILED", "CANCELED"]:
                    logger.info(
                        f"Simulation {simulation_id} completed with status: {status} "
                        f"(elapsed: {elapsed_time:.0f}s)"
                    )
                    return status

                # Continue polling for PENDING or RUNNING status
                if status in ["PENDING", "RUNNING"]:
                    time.sleep(poll_interval)
                    continue

                # Unknown status - log warning but continue polling
                logger.warning(
                    f"Unknown simulation status '{status}' for {simulation_id}. "
                    "Continuing to poll..."
                )
                time.sleep(poll_interval)

            except RuntimeError as e:
                # Re-raise if it's a clear error (simulation not found, etc.)
                error_msg = str(e).lower()
                if "not found" in error_msg or "access denied" in error_msg:
                    raise RuntimeError(
                        f"Failed to poll simulation {simulation_id}: {e}"
                    ) from e
                # For other errors, log and retry after interval
                logger.warning(
                    f"Error polling simulation {simulation_id}: {e}. "
                    f"Retrying in {poll_interval} seconds..."
                )
                time.sleep(poll_interval)

    def cancel_job(self, job_id: str) -> None:
        """Cancel a running simulation.

        Args:
            job_id: The simulation ID to cancel.

        Raises:
            RuntimeError: If cancellation fails or simulation cannot be cancelled.
        """
        if self.graphql_client is None:
            raise RuntimeError(
                "GraphQL client not available. Cannot cancel simulation. "
                "This method requires Cognito authentication."
            )

        # Check simulation status
        try:
            status = self.graphql_client.get_simulation_status(job_id)
        except RuntimeError as e:
            raise RuntimeError(f"Failed to get simulation status: {e}") from e

        # Check if simulation is in a cancellable state
        if status in ["SUCCEEDED", "FAILED", "CANCELED"]:
            logger.warning(
                f"Simulation {job_id} is already in terminal state: {status}. "
                "Nothing to cancel."
            )
            return

        # For now, raise an error indicating cancellation is not yet implemented
        # This will be implemented when the cancelSimulation GraphQL mutation is available
        raise NotImplementedError(
            f"Simulation cancellation is not yet implemented. "
            f"Simulation {job_id} is currently in state: {status}. "
            f"Please cancel it manually through the web interface."
        )

    def put(
        self,
        local_path: Union[str, Path],
        remote_path: Union[str, Path],
    ):
        """Transfer files from local path to S3.

        Args:
            local_path: Local path to transfer from
            remote_path: S3 key path (relative to work_dir) to transfer to
        """
        local_path = Path(local_path)
        remote_path = Path(remote_path)

        # Construct full S3 key by combining work_dir (s3_prefix) with remote_path
        s3_key = str(self.work_dir / remote_path).replace("\\", "/")

        logger.debug(f"Uploading {local_path} to s3://{self.config.s3_bucket}/{s3_key}")

        try:
            if local_path.is_file():
                # Upload single file
                self.s3_client.upload_file(
                    str(local_path), self.config.s3_bucket, s3_key
                )
            else:
                # Upload directory recursively
                for file_path in local_path.rglob("*"):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(local_path)
                        file_s3_key = f"{s3_key}/{relative_path}".replace("\\", "/")
                        self.s3_client.upload_file(
                            str(file_path), self.config.s3_bucket, file_s3_key
                        )
        except ClientError as e:
            raise RuntimeError(f"Failed to upload {local_path} to S3: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error uploading {local_path} to S3: {e}")

    def get(
        self,
        s3_path: Union[str, Path],
        local_path: Union[str, Path],
        overwrite: bool = False,
    ):
        """Transfer files from S3 path to local path using boto3.

        Uses the site's Cognito-backed S3 client (not AWS CLI) so that
        temporary credentials from the Identity Pool are used. The AWS CLI
        subprocess would use default credentials (e.g. IAM user) which may
        not have access to the Amplify-deployed storage bucket.

        Refreshes credentials automatically if they expired (e.g. after a
        long simulation wait). Identity Pool credentials typically expire
        in 1 hour.

        Args:
            s3_path: S3 path to transfer from (e.g., 's3://bucket/key' or 's3://bucket/key/')
            local_path: Local path to transfer to
            overwrite: Overwrite existing files (not used; always overwrites)
        """
        logger.debug("Attempting to transfer from %s to %s", s3_path, local_path)

        local_path = Path(local_path)
        s3_path_str = str(s3_path)

        def _do_get() -> None:
            # Parse s3://bucket/key/ format
            if not s3_path_str.startswith("s3://"):
                raise ValueError(f"Invalid S3 path: {s3_path_str}")
            path_parts = s3_path_str[5:].split("/", 1)  # Remove 's3://'
            bucket = path_parts[0]
            prefix = path_parts[1].rstrip("/") + "/" if len(path_parts) > 1 else ""

            # Create local directory
            local_path.mkdir(parents=True, exist_ok=True)

            # List and download objects using Cognito-backed s3_client
            paginator = self.s3_client.get_paginator("list_objects_v2")
            downloaded = 0
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith("/"):
                        continue
                    # Preserve relative path under prefix
                    rel_key = key[len(prefix) :] if prefix else key
                    local_file = local_path / rel_key
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    self.s3_client.download_file(bucket, key, str(local_file))
                    downloaded += 1

            logger.debug("Transfer completed successfully (%d files)", downloaded)

        try:
            _do_get()
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in ("ExpiredToken", "InvalidToken") and hasattr(
                self, "cognito_auth"
            ):
                logger.info("Credentials expired, refreshing from Identity Pool...")
                self._refresh_s3_credentials()
                _do_get()
            else:
                logger.error("S3 transfer failed: %s", e)
                raise RuntimeError(f"S3 transfer failed: {e}") from e
        except Exception as e:
            logger.exception("Error during S3 file transfer: %s", str(e))
            raise


# if __name__ == "__main__":
#     # Example usage
#     config = AWSSiteConfig.from_domain('frequensolve.app')
#     site = AWSSite(domain='frequensolve.app')

#     # Example simulation submission via API
#     # simulation_id = site.submit(job)
#     # print(f"Submitted simulation: {simulation_id}")
#     #
#     # # Wait for completion
#     # status = site.wait_for_completion(simulation_id, poll_interval=30, timeout=3600)
#     # print(f"Simulation completed with status: {status}")
