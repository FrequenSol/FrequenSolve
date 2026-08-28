"""FrequenSol cloud execution site backed by Cognito, AppSync, S3, and Batch."""

import getpass
import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Union

from frequensolve._optional import optional_dependency_error
from frequensolve.orchestrator.sites.aws.cache_paths import (
    cloud_config_cache_path,
    legacy_config_cache_path,
)

try:
    import boto3
    import requests
    from botocore.exceptions import ClientError
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "AWSSite",
        extra="cloud",
        dependencies=("boto3", "botocore", "requests"),
        error=exc,
    ) from exc

from frequensolve.orchestrator.sites.base import BaseSite, JobStatus, RunHandle
from frequensolve.orchestrator.sites.config import BaseSiteConfig
from frequensolve.orchestrator.utils.environment import build_subprocess_environment
from frequensolve.seismic.traces import TraceDataset
from frequensolve.simulation.jobs import BaseJob, ImagingJob, SkipPolicy
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
        config = AWSSiteConfig.from_domain('app.frequensol.com')

        # Or use FREQUENSOL_DOMAIN environment variable
        export FREQUENSOL_DOMAIN='app.frequensol.com'
        config = AWSSiteConfig.from_domain()

    Attributes:
        user_pool_id: Cognito user pool ID populated from the cloud domain.
        client_id: Cognito app client ID populated from the cloud domain.
        identity_pool_id: Cognito identity pool ID populated from the cloud domain.
        api_url: GraphQL API endpoint URL.
        domain: Frontend domain used to discover public configuration.
        s3_bucket: S3 bucket for simulation data, populated after authentication.
        region: AWS region used for Cognito, S3, and Batch resources.
        s3_prefix: Prefix for organizing simulation data inside the S3 bucket.
        max_duration: Maximum duration users may request for cloud resources.
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
            domain: Frontend domain (e.g., 'app.frequensol.com', 'localhost:5173')
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
        return cloud_config_cache_path(domain)

    @staticmethod
    def _read_cached_config(domain: str) -> Optional[dict]:
        """Read cached domain config, migrating the legacy path when present."""
        cache_path = AWSSiteConfig._get_config_cache_path(domain)
        candidates = [cache_path]
        legacy_path = legacy_config_cache_path(domain)
        if legacy_path != cache_path:
            candidates.append(legacy_path)

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                with open(candidate, "r") as f:
                    cached_config = json.load(f)
                logger.debug(f"Using cached configuration for {domain}")
                logger.debug(f"Cache path: {candidate}")
                if candidate == legacy_path:
                    AWSSiteConfig._cache_config(domain, cached_config)
                return cached_config
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Failed to read cached config, fetching fresh: {e}")
        return None

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
        if not force_refresh:
            cached_config = AWSSiteConfig._read_cached_config(domain)
            if cached_config is not None:
                return cached_config

        # Try both HTTPS and HTTP (for local development)
        logger.debug(f"Fetching configuration from {domain}...")
        for protocol in ["https", "http"]:
            url = f"{protocol}://{domain}/api/config.json"
            try:
                logger.debug(f"Trying {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                config_data = response.json()
                logger.debug(f"✓ Configuration loaded from {domain}")
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
        site = AWSSite(domain='app.frequensol.com')

        # Or set FREQUENSOL_DOMAIN environment variable once
        export FREQUENSOL_DOMAIN='app.frequensol.com'
        site = AWSSite()

        # Provide credentials to skip interactive prompt
        site = AWSSite(domain='app.frequensol.com', email='user@example.com', password='...')

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
        interactive: bool = False,
        verbose: bool = False,
        force_login: bool = False,
        _credential_profile: Optional[str] = None,
    ):
        """Initialize AWS site with domain-based authentication.

        Args:
            domain: Frontend domain (e.g., 'app.frequensol.com', 'localhost:5173').
                   If not provided, will try FREQUENSOL_DOMAIN environment variable.
            email: User email. Required when cached tokens are unavailable unless
                interactive is True.
            password: User password. Required when cached tokens are unavailable unless
                interactive is True.
            interactive: If True, prompt for missing credentials. Defaults to False so
                library code never blocks unexpectedly.
            verbose: If True, print user-facing status messages in addition to logs.
            force_login: If True, skip a valid cached login and authenticate again.
                The existing cache is replaced only after authentication succeeds.
            _credential_profile: Internal profile name supplied by ``fs.Site()``
                so cached credentials stay isolated across configured sites.

        Raises:
            ValueError: If domain cannot be determined or authentication fails.
            RuntimeError: If infrastructure is not deployed or stack info cannot be fetched.
        """
        from frequensolve.orchestrator.sites.aws.cognito import CognitoAuth
        from frequensolve.orchestrator.sites.aws.graphql_client import GraphQLClient

        self.verbose = verbose
        self._site_profile = _credential_profile

        # Load configuration from domain
        self._emit(
            f"Loading AWS configuration for {domain or os.getenv('FREQUENSOL_DOMAIN')}"
        )
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
            profile_name=_credential_profile,
            domain=domain or config.domain,
            expected_email=email,
        )

        # Try to use cached tokens first
        auth_successful = False
        max_retries = 1  # Allow one retry with fresh config

        for attempt in range(max_retries + 1):
            try:
                if force_login:
                    raise ValueError("A fresh Cloud login was explicitly requested.")
                cached_tokens = auth.get_cached_tokens()
                identity_reader = getattr(auth, "cached_identity", None)
                identity = (
                    identity_reader(cached_tokens) if callable(identity_reader) else {}
                )
                identity_details = ", ".join(
                    f"{name}={value}" for name, value in identity.items()
                )
                profile_details = (
                    f"profile={_credential_profile}"
                    if _credential_profile is not None
                    else ""
                )
                details = ", ".join(
                    value for value in (profile_details, identity_details) if value
                )
                self._emit(
                    "Using cached AWS credentials"
                    + (f" ({details})" if details else "")
                )
                auth_successful = True
                break
            except ValueError:
                # No cached tokens - need to login
                if not email:
                    if not interactive:
                        raise RuntimeError(
                            "No cached AWS/Cognito credentials are available. "
                            "Pass email and password, or use interactive=True."
                        )
                    email = input("FrequenSol Email: ")
                if not password:
                    if not interactive:
                        raise RuntimeError(
                            "No cached AWS/Cognito credentials are available. "
                            "Pass email and password, or use interactive=True."
                        )
                    password = getpass.getpass("Password: ")

                self._emit(f"Authenticating as {email}...")
                auth.login(email, password)
                self._emit("AWS authentication successful")
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
                    self._emit("Refetching AWS configuration from domain")

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
                        profile_name=_credential_profile,
                        domain=domain or config.domain,
                        expected_email=email,
                    )

                    # Leave the previous cache untouched until a successful login
                    # atomically replaces it. Binding validation rejects it below.
                    self._emit("Cloud credential binding changed; login is required")

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
        credentials = self._get_aws_credentials_with_relogin(
            auth,
            email=email,
            password=password,
            interactive=interactive,
        )
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
            self._emit(f"AWS storage stack loaded: bucket={config.s3_bucket}")
        except RuntimeError as e:
            error_msg = str(e).lower()
            # If storage stack doesn't exist, we'll create it on first sync
            if "no active storage stack" in error_msg:
                self._emit(
                    "AWS storage stack not found; it will be created on first sync."
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

    def _get_aws_credentials_with_relogin(
        self,
        auth,
        *,
        email: Optional[str],
        password: Optional[str],
        interactive: bool,
    ) -> Dict[str, str]:
        try:
            return auth.get_aws_credentials()
        except ValueError as exc:
            if not self._requires_cloud_relogin(exc):
                raise

            email, password = self._relogin_to_cloud(
                auth,
                email=email,
                password=password,
                interactive=interactive,
                reason=str(exc),
            )
            try:
                return auth.get_aws_credentials()
            except ValueError as retry_exc:
                credentials_path = getattr(auth, "credentials_path", "unknown")
                raise RuntimeError(
                    "FrequenSol cloud authentication failed after re-login. "
                    f"Credentials cache remains at: {credentials_path}. "
                    f"Original error: {retry_exc}"
                ) from retry_exc

    @staticmethod
    def _requires_cloud_relogin(error: ValueError) -> bool:
        message = str(error)
        return any(
            phrase in message
            for phrase in (
                "Please login again",
                "No refresh token found",
                "Refresh token expired",
            )
        )

    def _relogin_to_cloud(
        self,
        auth,
        *,
        email: Optional[str],
        password: Optional[str],
        interactive: bool,
        reason: str,
    ) -> tuple[str, str]:
        credentials_path = getattr(auth, "credentials_path", "unknown")
        if not email:
            if not interactive:
                raise RuntimeError(
                    self._cloud_relogin_required_message(
                        credentials_path=credentials_path,
                        reason=reason,
                    )
                )
            email = input("FrequenSol Email: ")
        if not password:
            if not interactive:
                raise RuntimeError(
                    self._cloud_relogin_required_message(
                        credentials_path=credentials_path,
                        reason=reason,
                    )
                )
            password = getpass.getpass("Password: ")

        self._emit("Cached FrequenSol cloud login expired; authenticating again")
        self._emit(f"Authenticating as {email}...")
        try:
            auth.login(email, password)
        except ValueError as exc:
            raise RuntimeError(
                "Cached FrequenSol cloud credentials could not be refreshed, "
                "and re-login did not complete. "
                f"Credentials cache remains at: {credentials_path}. "
                f"Login error: {exc}"
            ) from exc
        self._emit("AWS authentication successful")
        return email, password

    @staticmethod
    def _cloud_relogin_required_message(
        *, credentials_path: Union[str, Path], reason: str
    ) -> str:
        return (
            "Your cached FrequenSol cloud login expired. "
            f"Reason: {reason}\n"
            f"Credentials cache: {credentials_path}\n"
            "The credentials file was left unchanged and will be overwritten "
            "after a successful login.\n\n"
            "Re-run with interactive login enabled:\n"
            '  fs.Site(profile="cloud", interactive=True)\n\n'
            "Or pass credentials explicitly:\n"
            '  fs.Site(profile="cloud", email="you@example.com", password="...")'
        )

    def _refresh_s3_credentials(self) -> None:
        """Reload boto3 session and S3 client from the Identity Pool.

        Call when credentials may have expired (e.g. after a long simulation).
        No-op when not using Cognito authentication. Refreshed keys match what
        :meth:`_aws_cli_env` would pass to the AWS CLI.
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
        logger.debug("Refreshed AWS session and S3 client from Identity Pool")

    def _aws_cli_env(self) -> Dict[str, str]:
        """Build a process environment so the AWS CLI uses Cognito Identity Pool credentials.

        Starts from a sanitized process environment, sets temporary AWS
        credentials and the region from this site's session, and clears AWS
        profile selection so the CLI does not fall back to shared credentials.
        """
        creds = self.session.get_credentials()
        if creds is None:
            raise RuntimeError(
                "No AWS credentials available for the AWS CLI. "
                "Re-authenticate with AWSSite (Cognito login)."
            )
        frozen = creds.get_frozen_credentials()
        env = build_subprocess_environment()
        env["AWS_ACCESS_KEY_ID"] = frozen.access_key
        env["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
        if frozen.token:
            env["AWS_SESSION_TOKEN"] = frozen.token
        else:
            env.pop("AWS_SESSION_TOKEN", None)
        env["AWS_DEFAULT_REGION"] = self.config.region
        env.pop("AWS_PROFILE", None)
        env.pop("AWS_DEFAULT_PROFILE", None)
        return env

    def _run_aws_cli(self, argv: List[str], **kwargs) -> subprocess.CompletedProcess:
        """Run an ``aws`` subprocess using Identity Pool credentials (not ~/.aws)."""
        run_kw = dict(kwargs)
        run_kw["env"] = self._aws_cli_env()
        return subprocess.run(argv, **run_kw)

    @property
    def work_dir(self) -> Path:
        """Get the S3 work directory as a Path-like object.

        Returns:
            Path object representing the S3 prefix.
        """
        # Return a Path object that represents the S3 prefix
        # This is used by project._transfer() to construct remote paths
        return Path(self.config.s3_prefix)

    @property
    def provisioned(self) -> bool:
        """Check if the site is provisioned.

        AWS provisions compute resources automatically on demand when jobs are
        submitted. This property always returns True to maintain interface
        compatibility with other sites (e.g. HPC sites that require explicit
        provisioning before job submission).
        """
        return True

    def fetch_vtk(self, job: BaseJob, path: Optional[Union[str, Path]] = None) -> None:
        """Get VTK/visualization files from S3.

        Downloads the results/ParaView/ directory for the job from the
        project's S3 bucket to the local project path.

        Args:
            job: A BaseJob object.
            path: Optional local path to save results. If None, uses
                job.project_path.
        """
        if path is None:
            path = job.project_path
        else:
            path = Path(path)

        project_name = Path(job.simulation.project_path).name
        simulation_name = job.simulation.name
        job_name = job.name
        vtk_paths: List[str] = []
        outputs = getattr(getattr(job, "outputs", None), "paraview", None) or []
        for output in outputs:
            output_path = getattr(output, "path", None)
            if output_path is None:
                continue
            normalized = PurePosixPath(str(output_path)).as_posix().strip("/")
            if normalized and normalized != "." and normalized not in vtk_paths:
                vtk_paths.append(normalized)
        if not vtk_paths:
            vtk_paths.append("ParaView")

        for vtk_path in vtk_paths:
            results_vtk_path = f"jobs/{simulation_name}/{job_name}/results/{vtk_path}"
            s3_results_path = (
                f"s3://{self.config.s3_bucket}/{project_name}/{results_vtk_path}"
            )
            local_results_path = path / results_vtk_path

            try:
                logger.info(
                    "Fetching VTK outputs from %s to %s",
                    s3_results_path,
                    local_results_path,
                )
                self.get(s3_results_path, local_results_path)
            except Exception as e:
                logger.exception("Error downloading VTK outputs: %s", str(e))
                raise

    def fetch_paraview(
        self, job: BaseJob, path: Optional[Union[str, Path]] = None
    ) -> None:
        """Fetch visualization files using the historical method name."""

        return self.fetch_vtk(job, path=path)

    def fetch_output_files(
        self,
        job: BaseJob,
        *,
        kind: Optional[str] = None,
        suffix: Optional[Union[str, tuple[str, ...]]] = None,
    ) -> Path:
        """Fetch supported filesystem-backed AWS outputs used by discovery."""

        paraview_kinds = {"vtk", "vtu", "vtr", "vtp", "vts", "xmf", "xdmf"}
        normalized_kind = str(kind).strip().lower() if kind is not None else None
        suffixes = (suffix,) if isinstance(suffix, str) else (suffix or ())
        paraview_suffixes = (".vtk", ".vtu", ".vtr", ".vtp", ".vts", ".xmf")
        suffix_can_match_paraview = not suffixes or any(
            not str(requested_suffix)
            or str(requested_suffix).lower().endswith(paraview_suffixes)
            for requested_suffix in suffixes
        )
        can_fetch_paraview = (
            normalized_kind in {None, *paraview_kinds} and suffix_can_match_paraview
        )

        if can_fetch_paraview and getattr(
            getattr(job, "outputs", None), "paraview", None
        ):
            self.fetch_paraview(job)
        return job._result_path

    def _validate_config(self):
        """Validate AWS configuration."""
        try:
            credentials = self._check_credentials()
            profile = self._check_profile()

            return {
                "credentials": credentials,
                "profile": profile,
                "s3_bucket": self.config.s3_bucket,
            }

        except Exception as e:
            raise RuntimeError(f"Failed to validate AWS configuration: {e}")

    def _check_credentials(self):
        """Check what AWS credentials are being used."""
        sts_client = self.session.client("sts")
        identity = sts_client.get_caller_identity()
        return {
            "account": identity["Account"],
            "user_id": identity["UserId"],
            "arn": identity["Arn"],
            "region": self.config.region,
        }

    def _check_profile(self):
        """Check what AWS profile is being used and list available profiles."""
        result = self._run_aws_cli(
            ["aws", "configure", "list-profiles"],
            capture_output=True,
            text=True,
            check=True,
        )
        return {
            "current_profile": self.session.profile_name,
            "available_profiles": result.stdout.strip().splitlines(),
        }

    def _test_aws_access(self):
        """Test basic AWS access to see if credentials are working."""
        identity = self.session.client("sts").get_caller_identity()
        buckets = self.s3_client.list_buckets()
        bucket_accessible = True
        bucket_error = None
        try:
            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
        except Exception as e:
            bucket_accessible = False
            bucket_error = str(e)
        return {
            "account": identity["Account"],
            "bucket_count": len(buckets["Buckets"]),
            "s3_bucket": self.config.s3_bucket,
            "bucket_accessible": bucket_accessible,
            "bucket_error": bucket_error,
        }

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

    def _ensure_storage_bucket(self) -> None:
        """Load or create the authenticated user's Cloud storage bucket."""

        if self.graphql_client is not None:
            if (
                not self.config.s3_bucket
                or not self.graphql_client._check_storage_stack_exists()
            ):
                # Try to get existing storage stack first
                try:
                    storage_info = self.graphql_client.get_storage_stack_info()
                    self.config.s3_bucket = storage_info["bucketName"]
                    logger.debug(
                        f"Using existing storage stack: bucket={self.config.s3_bucket}"
                    )
                except RuntimeError:
                    # Storage stack doesn't exist, create it
                    logger.debug(
                        "Storage stack not found. Creating storage infrastructure..."
                    )
                    try:
                        # Deploy storage stack (userId extracted automatically from auth context)
                        deploy_result = self.graphql_client.deploy_storage_stack()

                        # Wait for stack to be ready (pass stackId so we wait for the one we just created)
                        logger.debug("Waiting for storage stack to be ready...")
                        expected_stack_id = deploy_result.get("stackId")
                        self.graphql_client.wait_for_stack_ready(
                            "storage", expected_stack_id=expected_stack_id
                        )

                        # Get storage stack info to update bucket name
                        storage_info = self.graphql_client.get_storage_stack_info()
                        self.config.s3_bucket = storage_info["bucketName"]
                        self._emit(
                            f"AWS storage stack ready: bucket={storage_info['bucketName']}"
                        )
                    except Exception as create_error:
                        raise RuntimeError(
                            f"Failed to create storage stack: {create_error}"
                        ) from create_error

    def sync(self, project):
        """Sync the project to S3.

        Automatically creates storage stack if it doesn't exist.

        Args:
            project: The Project to sync.

        Raises:
            RuntimeError: If sync fails or stack creation fails.
        """

        self._ensure_storage_bucket()

        self._emit(f"Syncing project '{project.name}' to S3")
        project._transfer(self)
        self._emit(f"Project '{project.name}' synced to S3")

    def _sync_loaded_job_inputs(self, job: BaseJob, remote_project: str) -> None:
        """Stage simulation inputs when a loaded job has no live Project owner."""

        self._ensure_storage_bucket()
        local_simulation, remote_simulation = job.save_simulation_for_remote(
            self.__class__.__name__, remote_project
        )
        self.sync_s3(local_simulation, remote_simulation)
        self._emit(f"Synced simulation file to S3: {remote_simulation}")

        for local_file, remote_file in job.remote_input_files(remote_project):
            self.sync_s3(local_file, remote_file)
            self._emit(f"Synced simulation input to S3: {remote_file}")

    @staticmethod
    def _authored_project_metadata(
        job: BaseJob,
    ) -> tuple[Optional[str], Optional[str]]:
        """Recover authored project identity without treating a path as a name.

        Live jobs retain their owning ``Project``. Deserialized jobs intentionally
        do not, so recover metadata only from an unambiguous saved project file.
        Cloud can still derive its legacy storage key when no authored identity is
        available.
        """

        project_owner = getattr(job.simulation, "_project", None)
        project_name = getattr(project_owner, "name", None)
        if isinstance(project_name, str) and project_name.strip():
            project_display_name = getattr(project_owner, "pretty_name", None)
            if (
                not isinstance(project_display_name, str)
                or not project_display_name.strip()
            ):
                project_display_name = None
            return project_name.strip(), (
                project_display_name.strip() if project_display_name else None
            )

        project_path = getattr(job, "project_path", None)
        if project_path is None:
            return None, None

        try:
            project_files = sorted(Path(project_path).expanduser().glob("*.json"))
        except (OSError, TypeError, ValueError):
            return None, None
        if len(project_files) != 1:
            return None, None

        try:
            project_payload = json.loads(project_files[0].read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None, None
        if not isinstance(project_payload, dict):
            return None, None
        project_name = project_payload.get("name")
        if (
            not isinstance(project_name, str)
            or not project_name.strip()
            or "version" not in project_payload
            or not isinstance(project_payload.get("simulations"), list)
        ):
            return None, None

        project_display_name = project_payload.get("pretty_name")
        if (
            not isinstance(project_display_name, str)
            or not project_display_name.strip()
        ):
            project_display_name = None
        return project_name.strip(), (
            project_display_name.strip() if project_display_name else None
        )

    def submit(self, job: BaseJob, **kwargs) -> RunHandle:
        """Submit a simulation job.

        Uses platform-managed shared compute when advertised by the Cloud API,
        while retaining legacy per-user compute-stack provisioning.

        Submits through the authenticated GraphQL API.

        Args:
            job: The task to submit.
            **kwargs: Additional job parameters (vcpu, memory, name,
                description). Pass ``check=True`` to make ``wait()`` raise by
                default for failed runs, or ``validate=False`` to skip SDK
                pre-run validation.

        Returns:
            Awaitable run handle.

        Raises:
            RuntimeError: If job submission fails or stack creation fails.
        """
        fresh_run = bool(kwargs.pop("force", False) or kwargs.pop("rerun", False))
        skip_policy = SkipPolicy.from_value(
            kwargs.pop("skip", kwargs.pop("skip_policy", None))
        )
        fresh_run = bool(fresh_run or skip_policy.force)
        check = bool(kwargs.pop("check", False))
        validate = kwargs.pop("validate", True)
        fetch = kwargs.pop("fetch", False)
        poll_interval = kwargs.pop("poll_interval", 10)
        vcpu = kwargs.pop("vcpu", None)
        memory = kwargs.pop("memory", None)
        if self.graphql_client is None:
            raise RuntimeError(
                "FrequenSolve Cloud requires Cognito authentication and the "
                "GraphQL API. Recreate this Site with a current Cloud profile."
            )
        self.prepare_job(job, validate=validate)
        if not fresh_run and job.is_run_current():
            job.write_run_state(status="skipped")
            self._emit(f"Skipping {job.name}; run is current")
            handle = RunHandle.skipped(self, job)
            if fetch:
                handle._fetch_fn = lambda run: self.fetch_outputs(run.job)
                handle._fetch_on_wait = True
            return handle

        self.prepare_job(job, sync_project=True, validate=False)

        project = job.project_path.name
        if getattr(job.simulation, "_project", None) is None:
            self._sync_loaded_job_inputs(job, project)

        # Current Cloud deployments use platform-managed shared compute. Older
        # deployments require a per-user compute stack. Discover the contract
        # before invoking any deployment mutation so a removed legacy field is
        # never called speculatively.
        if self.graphql_client is not None:
            try:
                compute_mode = self.graphql_client.get_compute_provisioning_mode()
            except Exception as capability_error:
                raise RuntimeError(
                    "Failed to determine whether this FrequenSolve Cloud "
                    "environment uses shared or per-user compute. No compute "
                    "infrastructure was changed. Update FrequenSolve or contact "
                    f"the Cloud environment owner. Details: {capability_error}"
                ) from capability_error

            if (
                compute_mode == "per-user"
                and not self.graphql_client._check_compute_stack_exists()
            ):
                # Compute stack doesn't exist, create it
                self._emit(
                    "AWS compute stack not found; creating compute infrastructure."
                )
                try:
                    # Deploy compute stack - backend will automatically fetch and use user's compute settings
                    # (userId extracted automatically from auth context)
                    deploy_result = self.graphql_client.deploy_compute_stack()

                    # Wait for stack to be ready, passing the stackId from deployment for accurate matching
                    self._emit("Waiting for AWS compute stack to be ready")
                    expected_stack_id = deploy_result.get("stackId")
                    stack_info = self.graphql_client.wait_for_stack_ready(
                        "compute", expected_stack_id=expected_stack_id
                    )

                    self._emit(
                        f"AWS compute stack ready: {stack_info.get('stackId', 'unknown')}"
                    )
                except Exception as create_error:
                    raise RuntimeError(
                        f"Failed to create compute stack: {create_error}"
                    ) from create_error

        try:
            # Sync job file to S3
            local_job, remote_job = job.save_for_remote(
                self.__class__.__name__, project
            )
            s3_job_key = self.sync_s3(local_job, remote_job)
            self._emit(f"Synced job file to S3: {s3_job_key}")

            # Check if using Cognito/GraphQL authentication
            self._emit(f"Submitting {job.name} via AWS GraphQL API")
            (
                authored_project_name,
                authored_project_display_name,
            ) = self._authored_project_metadata(job)

            result = self.graphql_client.submit_job(
                job_file_s3_key=str(s3_job_key),
                vcpu=vcpu,
                memory=memory,
                job_name=kwargs.get("name", f"frequensolve-{uuid.uuid4().hex[:8]}"),
                project_name=authored_project_name,
                project_display_name=authored_project_display_name,
                simulation_name=job.simulation.name,
                simulation_job_name=job.name,
                send_simulation_status_email=kwargs.get("send_simulation_status_email"),
                fresh=fresh_run,
            )

            simulation_id = result["simulationId"]
            self._emit(
                f"AWS simulation submitted: id={simulation_id}, status={result['status']}"
            )

            job._job_id = simulation_id
            return self._make_run_handle(
                job,
                simulation_id,
                poll_interval=poll_interval,
                fetch=fetch,
                check=check,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to submit job: {e}")

    def _make_run_handle(
        self,
        job: BaseJob,
        job_id: str,
        poll_interval: float = 10,
        fetch: bool = False,
        check: bool = False,
    ) -> RunHandle:
        return RunHandle(
            site=self,
            job=job,
            id=str(job_id),
            mode="aws",
            poll_interval=poll_interval,
            check=check,
            _status_fn=self._poll_run,
            _cancel_fn=lambda run: self.cancel_job(str(run.id)),
            _fetch_fn=(lambda run: self.fetch_outputs(run.job)) if fetch else None,
            _fetch_on_wait=fetch,
        )

    def _poll_run(self, run: RunHandle) -> JobStatus:
        if self.graphql_client is None:
            raise RuntimeError("Authenticated GraphQL client is unavailable")
        details_getter = getattr(
            self.graphql_client, "get_simulation_status_details", None
        )
        if callable(details_getter):
            status_details = details_getter(str(run.id))
            raw_status = status_details["status"]
            raw = {**status_details, "source": "graphql"}
            message = str(status_details.get("failureMessage") or "")
        else:
            raw_status = self.graphql_client.get_simulation_status(str(run.id))
            raw = {"status": raw_status, "source": "graphql"}
            message = ""
        state = {
            "PENDING": "pending",
            "SUBMITTED": "pending",
            "RUNNING": "running",
            "SUCCEEDED": "completed",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "CANCELED": "cancelled",
            "CANCELLED": "cancelled",
        }.get(str(raw_status).upper(), "unknown")
        return_code = (
            0
            if state == "completed"
            else (1 if state in {"failed", "cancelled"} else -1)
        )
        return JobStatus(
            state=state,
            return_code=return_code,
            job_id=str(run.id),
            message=message,
            raw=raw,
        )

    def fetch_traces(
        self,
        job: Union[BaseJob, List[BaseJob]],
        path: Optional[Union[str, Path]] = None,
        upscale: int = 1,
    ) -> Union[TraceDataset, Dict[str, TraceDataset]]:
        """Get results from Stampede3.

        Args:
            job: A BaseJob object.
            path: The path to save the results to.
        """

        if isinstance(job, BaseJob):
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
                # Build the path for the s3 results directory and the local results directory.
                # The job results are stored in the job's result_path, not the simulation path
                # Format: ex_01/jobs/simulation_name/job_name/results/traces/
                project_name = job.project_path.name  # e.g., "ex_01"
                simulation_name = job.simulation.name  # e.g., "simple_acoustic"
                job_name = job.name  # e.g., "time"
                trace_dir_name = Path(job.trace_outputs.path).name

                results_traces_path = (
                    f"jobs/{simulation_name}/{job_name}/results/{trace_dir_name}"
                )
                s3_results_path = (
                    f"s3://{self.config.s3_bucket}/{project_name}/{results_traces_path}"
                )
                local_results_path = path / results_traces_path
                self.get(s3_results_path, local_results_path)
                self._emit(f"Fetched AWS traces from {s3_results_path}")

                # TODO: Copy job, simulation file to database so that it can be read independently.

                db = TraceDataset.from_job(job, upscale, project_path=path.resolve())
                db_map[job.name] = db

            except Exception as e:
                logger.exception("Error downloading traces: %s", str(e))
                raise

        if len(db_map) == 1:
            return db_map[jobs[0].name]
        else:
            return db_map

    def fetch_wavefields(
        self,
        job: Union[BaseJob, List[BaseJob]],
        path: Optional[Union[str, Path]] = None,
        upscale: int = 1,
    ) -> Union[TraceDataset, Dict[str, TraceDataset]]:
        """Get wavefield results from AWS storage.

        Args:
            job: Single job or list of jobs to fetch.
            path: Optional local project root for downloaded artifacts.
            upscale: Optional upscaling factor for wavefield trace reads.

        Returns:
            Wavefield dataset for one job, or a mapping keyed by job name.
        """

        if isinstance(job, BaseJob):
            jobs = [job]
        else:
            jobs = job

        if path is None:
            path = jobs[0].project_path
        else:
            path = Path(path)

        db_map = {}

        for item in jobs:
            try:
                wavefield_outputs = item.wavefield_trace_outputs
                if not wavefield_outputs.groups:
                    raise ValueError("Job has no wavefield outputs")

                project_name = item.project_path.name
                simulation_name = item.simulation.name
                job_name = item.name
                wavefield_dir_name = Path(wavefield_outputs.path).name

                results_wavefields_path = (
                    f"jobs/{simulation_name}/{job_name}/results/{wavefield_dir_name}"
                )
                s3_results_path = (
                    f"s3://{self.config.s3_bucket}/{project_name}/"
                    f"{results_wavefields_path}"
                )
                local_results_path = path / results_wavefields_path
                self.get(s3_results_path, local_results_path)
                self._emit(f"Fetched AWS wavefields from {s3_results_path}")

                db_map[item.name] = item.wavefields.open(
                    upscale=upscale,
                    project_path=path.resolve(),
                )

            except Exception as e:
                logger.exception("Error downloading wavefields: %s", str(e))
                raise

        if len(db_map) == 1:
            return db_map[jobs[0].name]
        return db_map

    def fetch_run_metadata(self, job: BaseJob) -> Optional[Path]:
        """Fetch ``_fs_run`` metadata and aggregate task manifests locally."""

        project_name = Path(job.project_path).name
        simulation_name = job.simulation.name
        job_name = job.name
        results_run_path = f"jobs/{simulation_name}/{job_name}/results/_fs_run"
        s3_results_path = (
            f"s3://{self.config.s3_bucket}/{project_name}/{results_run_path}"
        )
        local_run_path = job._result_path / "_fs_run"
        self.get(s3_results_path, local_run_path)
        self._emit(f"Fetched AWS run metadata from {s3_results_path}")
        return job.collect_task_run_manifests()

    def fetch_image(self, job: ImagingJob) -> Any:
        """Download and open the aggregate image for one imaging job."""

        if not isinstance(job, ImagingJob):
            raise TypeError("fetch_image expects an ImagingJob")

        project_path = Path(job.project_path).resolve()
        local_image_file = Path(job.image_file()).resolve()
        try:
            relative_image_file = local_image_file.relative_to(project_path)
        except ValueError as exc:
            raise ValueError(
                f"Imaging output path {local_image_file} is outside project "
                f"root {project_path}"
            ) from exc

        bucket = self.config.s3_bucket
        key = f"{project_path.name}/{relative_image_file.as_posix()}"
        local_image_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.s3_client.download_file(bucket, key, str(local_image_file))
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("ExpiredToken", "InvalidToken") and hasattr(
                self, "cognito_auth"
            ):
                self._refresh_s3_credentials()
                self.s3_client.download_file(bucket, key, str(local_image_file))
            elif error_code in ("404", "NoSuchKey", "NotFound"):
                raise FileNotFoundError(
                    f"AWS imaging output s3://{bucket}/{key} is missing"
                ) from exc
            else:
                raise RuntimeError(f"S3 image download failed: {exc}") from exc

        self._emit(f"Fetched AWS image from s3://{bucket}/{key}")
        return job.load_images()

    def fetch_outputs(self, job: BaseJob):
        """Fetch common AWS result artifacts for a completed job.

        Args:
            job: Completed job whose S3 artifacts should be downloaded.

        Returns:
            Trace dataset, or a mapping containing traces and wavefields when
            wavefield outputs exist. Run metadata and configured ParaView
            outputs are downloaded as side effects.
        """

        self.fetch_run_metadata(job)
        traces = self.fetch_traces(job)
        wavefields = None
        if getattr(job.outputs, "wavefields", None):
            wavefields = self.fetch_wavefields(job)
        if getattr(job.outputs, "paraview", None):
            self.fetch_paraview(job)
        if isinstance(job, ImagingJob):
            self.fetch_image(job)
        if wavefields is None:
            return traces
        return {"traces": traces, "wavefields": wavefields}

    def fetch_logs(
        self,
        job: Union[BaseJob, List[BaseJob]],
        *,
        local_dir: Optional[Union[str, Path]] = None,
        task: Optional[int] = None,
        frequency: Optional[Union[float, complex]] = None,
        show: bool = False,
    ) -> Union[Path, Dict[str, Path]]:
        """Fetch AWS task logs and optionally return one task log file.

        ``task`` is one-based. ``frequency`` selects the matching frequency in
        ``job.f_list``. Without either selector, the local log directory is
        returned.

        Args:
            job: Single job or list of jobs.
            local_dir: Optional local destination for downloaded logs.
            task: Optional one-based task number to select.
            frequency: Optional frequency used to select a task log.
            show: Whether to print the selected log contents.

        Returns:
            Log path for one job, or a mapping keyed by job name.
        """

        jobs, single = self._as_jobs(job)
        requested_local_dir = Path(local_dir) if local_dir is not None else None
        result: Dict[str, Path] = {}

        for item in jobs:
            project_name = item.project_path.name
            remote_logs_path = (
                f"s3://{self.config.s3_bucket}/"
                f"{project_name}/jobs/{item.simulation.name}/{item.name}/logs"
            )
            if requested_local_dir is None:
                log_dir = item._stdout_path
            elif single:
                log_dir = requested_local_dir
            else:
                log_dir = requested_local_dir / item.name

            self.get(remote_logs_path, log_dir)
            selected = self._select_log_path(
                item,
                log_dir,
                task=task,
                frequency=frequency,
            )
            if show:
                self._show_logs(selected, job_name=item.name)
            result[item.name] = selected

        if single:
            return result[jobs[0].name]
        return result

    def test_api_connectivity(self) -> bool:
        """Verify the authenticated GraphQL submission contract is reachable.

        Returns:
            True if API is accessible, False otherwise.
        """
        if self.graphql_client is None:
            return False
        try:
            return self.graphql_client.get_compute_provisioning_mode() in {
                "shared",
                "per-user",
            }
        except Exception as exc:
            logger.warning("GraphQL connectivity probe failed: %s", exc)
            return False

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

        Uses the site's Cognito-backed S3 client so Identity Pool temporary
        credentials apply (same principal as :meth:`_run_aws_cli`). Refreshes
        credentials on expiry (e.g. after a long simulation wait; typical
        lifetime ~1 hour).

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
            key = path_parts[1] if len(path_parts) > 1 else ""

            def _download_object(object_key: str) -> int:
                if local_path.exists() and local_path.is_dir():
                    local_file = local_path / Path(object_key).name
                elif local_path.exists():
                    local_file = local_path
                elif local_path.suffix:
                    local_file = local_path
                else:
                    local_file = local_path / Path(object_key).name
                local_file.parent.mkdir(parents=True, exist_ok=True)
                self.s3_client.download_file(bucket, object_key, str(local_file))
                return 1

            def _missing_object(error: ClientError) -> bool:
                error_code = error.response.get("Error", {}).get("Code", "")
                return error_code in ("404", "NoSuchKey", "NotFound")

            if key and not s3_path_str.endswith("/"):
                try:
                    self.s3_client.head_object(Bucket=bucket, Key=key)
                except ClientError as e:
                    if not _missing_object(e):
                        raise
                else:
                    downloaded = _download_object(key)
                    logger.debug(
                        "Transfer completed successfully (%d files)", downloaded
                    )
                    return

            prefix = key.rstrip("/") + "/" if key else ""

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
#     config = AWSSiteConfig.from_domain('app.frequensol.com')
#     site = AWSSite(domain='app.frequensol.com')
