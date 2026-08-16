"""
Cognito authentication for FrequenSol Cloud.

This module handles:
- User authentication with Cognito User Pool
- Token caching and refresh
- AWS credential exchange via Cognito Identity Pool
"""

import base64
import json
import logging
import os
import random
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from frequensolve._cloud_credentials import (
    CREDENTIAL_CACHE_BINDING_KEY,
    credential_cache_binding,
    credential_cache_binding_matches,
)
from frequensolve._optional import optional_dependency_error
from frequensolve.orchestrator.sites.aws.cache_paths import (
    cloud_credentials_path,
    legacy_credentials_path,
)

try:
    import boto3
    from botocore.exceptions import ClientError
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "CognitoAuth",
        extra="cloud",
        dependencies=("boto3", "botocore"),
        error=exc,
    ) from exc

logger = logging.getLogger(__name__)

_TOKEN_CACHE_FIELDS = (
    "email",
    "id_token",
    "access_token",
    "refresh_token",
    "expires_at",
)

_IDENTITY_THROTTLE_CODES = frozenset(
    {
        "TooManyRequestsException",
        "Throttling",
        "ThrottlingException",
    }
)
_IDENTITY_MAX_ATTEMPTS = 5
_IDENTITY_RETRY_BASE_DELAY_SECONDS = 0.25
_IDENTITY_RETRY_MAX_DELAY_SECONDS = 2.0
_IDENTITY_BUSY_MESSAGE = (
    "FrequenSol Cloud authentication is temporarily busy because several "
    "sessions started at once. Wait a moment, then retry."
)


def _call_identity_with_retry(
    operation_name: str,
    operation: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    """Run a Cognito Identity operation with bounded, jittered throttling retries."""

    for attempt in range(_IDENTITY_MAX_ATTEMPTS):
        try:
            return operation()
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code not in _IDENTITY_THROTTLE_CODES:
                raise

            if attempt == _IDENTITY_MAX_ATTEMPTS - 1:
                logger.warning(
                    "Cognito Identity %s remained throttled after %d attempts",
                    operation_name,
                    _IDENTITY_MAX_ATTEMPTS,
                )
                raise RuntimeError(_IDENTITY_BUSY_MESSAGE) from None

            delay_ceiling = min(
                _IDENTITY_RETRY_BASE_DELAY_SECONDS * (2**attempt),
                _IDENTITY_RETRY_MAX_DELAY_SECONDS,
            )
            delay = random.uniform(delay_ceiling / 2, delay_ceiling)
            logger.debug(
                "Cognito Identity %s was throttled; retrying (attempt %d/%d)",
                operation_name,
                attempt + 2,
                _IDENTITY_MAX_ATTEMPTS,
            )
            time.sleep(delay)

    raise AssertionError("Cognito Identity retry loop exited unexpectedly")


def _cognito_issuer(*, region: str, user_pool_id: str) -> str:
    """Return the partition-correct Cognito issuer for a user pool."""

    dns_suffix = "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"
    return f"https://cognito-idp.{region}.{dns_suffix}/{user_pool_id}"


def _decode_jwt_payload(token: str) -> dict:
    """Decode JWT payload without verification (for reading our own token claims).

    Args:
        token: JWT string (header.payload.signature)

    Returns:
        Decoded payload dict

    Raises:
        ValueError: If token is invalid
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")
        payload_b64 = parts[1]
        # Add padding if needed for base64
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload_bytes = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to decode JWT payload: {e}") from e


class CognitoAuth:
    """Handle Cognito authentication and AWS credential management.

    This class manages the complete authentication flow:
    1. Authenticate with Cognito User Pool (email/password)
    2. Store tokens in a profile-scoped local cache for configured sites
    3. Automatically refresh expired tokens
    4. Exchange ID token for AWS credentials via Identity Pool

    Args:
        user_pool_id: Cognito User Pool ID
        client_id: Cognito App Client ID
        identity_pool_id: Cognito Identity Pool ID
        region: AWS region (default: us-east-1)
        profile_name: Selected site profile for an isolated credential cache.
        domain: Cloud domain included in the profile cache binding.
        expected_email: Optional configured user that cached tokens must match.
    """

    def __init__(
        self,
        user_pool_id: str,
        client_id: str,
        identity_pool_id: str,
        region: str = "us-east-1",
        profile_name: Optional[str] = None,
        domain: Optional[str] = None,
        expected_email: Optional[str] = None,
    ):
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.identity_pool_id = identity_pool_id
        self.region = region
        self.profile_name = profile_name
        self.expected_email = expected_email

        # Initialize AWS clients
        self.cognito_client = boto3.client("cognito-idp", region_name=region)
        self.identity_client = boto3.client("cognito-identity", region_name=region)

        # Path to credentials file
        self.credentials_path = cloud_credentials_path(profile_name)
        self.legacy_credentials_path = legacy_credentials_path()
        self.cache_binding = (
            credential_cache_binding(
                profile_name=profile_name,
                domain=domain or "",
                region=region,
                user_pool_id=user_pool_id,
                client_id=client_id,
                identity_pool_id=identity_pool_id,
            )
            if profile_name is not None
            else None
        )

    def login(self, email: str, password: str) -> Dict[str, str]:
        """Authenticate user with Cognito User Pool.

        Args:
            email: User's email address
            password: User's password

        Returns:
            Dict containing id_token, access_token, refresh_token, and expires_at

        Raises:
            ClientError: If authentication fails
        """
        logger.debug(f"Authenticating with Cognito as {email}...")
        try:
            response = self.cognito_client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters={"USERNAME": email, "PASSWORD": password},
            )

            auth_result = response["AuthenticationResult"]

            # Calculate expiration time (tokens typically expire in 1 hour)
            expires_at = datetime.now() + timedelta(
                seconds=auth_result.get("ExpiresIn", 3600)
            )

            tokens = {
                "email": email,
                "id_token": auth_result["IdToken"],
                "access_token": auth_result["AccessToken"],
                "refresh_token": auth_result["RefreshToken"],
                "expires_at": expires_at.isoformat(),
            }

            # Save tokens to file
            self.save_tokens(tokens)

            logger.debug("✓ Authentication successful")

            return tokens

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NotAuthorizedException":
                raise ValueError("Invalid email or password") from e
            elif error_code == "UserNotFoundException":
                raise ValueError("User not found") from e
            else:
                raise

    def refresh_tokens(self) -> Dict[str, str]:
        """Refresh expired tokens using refresh token.

        Returns:
            Dict containing refreshed id_token, access_token, and new expires_at

        Raises:
            ValueError: If no cached tokens found or refresh token expired
            ClientError: If token refresh fails
        """
        cached = self.get_cached_tokens()

        if "refresh_token" not in cached:
            raise ValueError("No refresh token found. Please login again.")

        try:
            response = self.cognito_client.initiate_auth(
                ClientId=self.client_id,
                AuthFlow="REFRESH_TOKEN_AUTH",
                AuthParameters={"REFRESH_TOKEN": cached["refresh_token"]},
            )

            auth_result = response["AuthenticationResult"]

            # Calculate new expiration time
            expires_at = datetime.now() + timedelta(
                seconds=auth_result.get("ExpiresIn", 3600)
            )

            # Update tokens (keep refresh_token from cached)
            tokens = {
                "email": cached.get("email", ""),
                "id_token": auth_result["IdToken"],
                "access_token": auth_result["AccessToken"],
                "refresh_token": cached[
                    "refresh_token"
                ],  # Refresh token doesn't change
                "expires_at": expires_at.isoformat(),
            }

            # Save updated tokens
            self.save_tokens(tokens)

            return tokens

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NotAuthorizedException":
                raise ValueError("Refresh token expired. Please login again.") from e
            else:
                raise

    def get_id_token(self) -> str:
        """Get current valid ID token (auto-refresh if expired).

        Returns:
            Valid ID token string

        Raises:
            ValueError: If no tokens cached or refresh fails
        """
        tokens = self.get_cached_tokens()

        # Check if token is expired
        if self._is_token_expired(tokens):
            tokens = self.refresh_tokens()

        return tokens["id_token"]

    def get_aws_credentials(self) -> Dict[str, str]:
        """Exchange Cognito ID token for AWS credentials via Identity Pool.

        Returns:
            Dict containing AWS credentials:
                - AccessKeyId
                - SecretKey (SecretAccessKey)
                - SessionToken
                - Expiration

        Raises:
            ValueError: If authentication fails or no tokens cached
            ClientError: If AWS API calls fail
        """
        # Get valid ID token (auto-refreshes if needed)
        id_token = self.get_id_token()

        # Construct the login provider key
        provider_name = f"cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}"

        logger.debug("Fetching AWS credentials from Identity Pool...")
        logger.debug("Getting Identity ID...")
        logger.debug(f"  Identity Pool ID: {self.identity_pool_id}")
        logger.debug(f"  Provider: {provider_name}")

        try:
            # Get Identity ID
            identity_response = _call_identity_with_retry(
                "GetId",
                lambda: self.identity_client.get_id(
                    IdentityPoolId=self.identity_pool_id,
                    Logins={provider_name: id_token},
                ),
            )

            identity_id = identity_response["IdentityId"]
            logger.debug("Identity ID obtained successfully")

            # Get AWS credentials for this identity
            logger.debug("Getting credentials for identity...")
            credentials_response = _call_identity_with_retry(
                "GetCredentialsForIdentity",
                lambda: self.identity_client.get_credentials_for_identity(
                    IdentityId=identity_id,
                    Logins={provider_name: id_token},
                ),
            )

            credentials = credentials_response["Credentials"]

            logger.debug(f"Extracted credentials keys: {list(credentials.keys())}")

            result = {
                "AccessKeyId": credentials["AccessKeyId"],
                "SecretKey": credentials[
                    "SecretKey"
                ],  # AWS returns 'SecretKey', not 'SecretAccessKey'
                "SessionToken": credentials["SessionToken"],
                "Expiration": credentials["Expiration"].isoformat(),
                "IdentityId": identity_id,
            }

            logger.debug(f"Final credential dict keys: {list(result.keys())}")

            logger.debug("✓ AWS credentials obtained successfully")

            return result

        except ClientError as e:
            logger.debug("ClientError occurred:")
            logger.debug(f"  Error Code: {e.response['Error']['Code']}")
            logger.debug(f"  Error Message: {e.response['Error']['Message']}")
            error_code = e.response["Error"]["Code"]
            if error_code == "NotAuthorizedException":
                raise ValueError("Authentication failed. Please login again.") from e
            else:
                raise
        except KeyError as e:
            logger.debug(f"KeyError - missing key: {e}")
            raise

    def get_cached_tokens(self) -> Dict[str, str]:
        """Load tokens from local cache file.

        Returns:
            Dict containing cached tokens

        Raises:
            ValueError: If credentials are missing, invalid, or belong to a
                different Cloud profile
        """
        if not self.credentials_path.exists():
            if self.cache_binding is None and self.legacy_credentials_path.exists():
                try:
                    with open(self.legacy_credentials_path, "r") as f:
                        tokens = json.load(f)
                    tokens = self._validate_cached_tokens(tokens)
                    self.save_tokens(tokens)
                    logger.info("Migrated cached credentials to cloud cache directory")
                    return tokens
                except (json.JSONDecodeError, IOError) as e:
                    raise ValueError(f"Failed to read credentials file: {e}") from e
            raise ValueError(
                "No cached credentials found. Please login first.\n"
                "Run fs.Site() for an interactive configured login, or pass "
                "email and password to AWSSite."
            )

        try:
            with open(self.credentials_path, "r") as f:
                document = json.load(f)
            tokens = self._validate_cached_tokens(document)
            logger.debug("Using cached credentials")
            return tokens
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Failed to read credentials file: {e}") from e

    def _validate_cached_tokens(self, document: object) -> Dict[str, str]:
        """Validate cache binding and token scope, returning only token fields."""

        if not isinstance(document, dict):
            raise ValueError("Cached credentials are invalid. Please login again.")
        if self.cache_binding is not None and not credential_cache_binding_matches(
            document, self.cache_binding
        ):
            raise ValueError(
                "Cached credentials do not match the selected FrequenSol Cloud "
                "profile. Please login again."
            )
        allowed_fields = {*_TOKEN_CACHE_FIELDS, CREDENTIAL_CACHE_BINDING_KEY}
        if any(
            not isinstance(key, str) or key not in allowed_fields for key in document
        ):
            raise ValueError("Cached credentials are invalid. Please login again.")

        tokens = dict(document)
        tokens.pop(CREDENTIAL_CACHE_BINDING_KEY, None)
        self._validate_cached_token_scope(tokens)
        cached_email = tokens.get("email")
        if self.expected_email is not None and (
            not isinstance(cached_email, str)
            or cached_email.casefold() != self.expected_email.casefold()
        ):
            raise ValueError(
                "Cached credentials do not match the configured FrequenSol Cloud "
                "user. Please login again."
            )
        return tokens

    def _validate_cached_token_scope(self, tokens: object) -> None:
        """Reject cached tokens issued for another configured Cloud profile."""

        if not isinstance(tokens, dict):
            raise ValueError("Cached credentials are invalid. Please login again.")

        expected_issuer = _cognito_issuer(
            region=self.region,
            user_pool_id=self.user_pool_id,
        )
        token_contracts = (
            ("id_token", "id", "aud"),
            ("access_token", "access", "client_id"),
        )
        for token_name, token_use, client_claim in token_contracts:
            token = tokens.get(token_name)
            if not isinstance(token, str) or not token:
                raise ValueError(
                    "Cached credentials are incomplete. Please login again."
                )
            try:
                # This is only a local scope check. AWS services (and the MCP's
                # Cloud adapter) still perform the authenticity verification.
                claims = _decode_jwt_payload(token)
            except ValueError as exc:
                raise ValueError(
                    "Cached credentials are invalid. Please login again."
                ) from exc
            if (
                not isinstance(claims, dict)
                or claims.get("iss") != expected_issuer
                or claims.get("token_use") != token_use
                or claims.get(client_claim) != self.client_id
            ):
                raise ValueError(
                    "Cached credentials do not match the selected FrequenSol "
                    "Cloud profile. Please login again."
                )

    def save_tokens(self, tokens: Dict[str, str]) -> None:
        """Save tokens to local cache file.

        Args:
            tokens: Dict containing tokens to save
        """
        document = {
            field: tokens[field] for field in _TOKEN_CACHE_FIELDS if field in tokens
        }
        if self.cache_binding is not None:
            document[CREDENTIAL_CACHE_BINDING_KEY] = self.cache_binding

        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=self.credentials_path.parent,
                prefix=f".{self.credentials_path.name}.",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                os.chmod(temporary_path, 0o600)
                json.dump(document, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.credentials_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def cached_identity(self, tokens: Dict[str, str]) -> Dict[str, str]:
        """Return non-secret identity details suitable for verbose status output."""

        identity: Dict[str, str] = {}
        email = tokens.get("email")
        if isinstance(email, str) and email:
            identity["email"] = email
        try:
            claims = _decode_jwt_payload(tokens["id_token"])
        except (KeyError, ValueError, TypeError):
            return identity
        account_id = claims.get("custom:accountId")
        if isinstance(account_id, str) and account_id:
            identity["account"] = account_id
        return identity

    def _is_token_expired(self, tokens: Dict[str, str]) -> bool:
        """Check if ID token is expired.

        Args:
            tokens: Dict containing tokens with expires_at field

        Returns:
            True if token is expired or will expire in next 5 minutes
        """
        if "expires_at" not in tokens:
            return True

        try:
            expires_at = datetime.fromisoformat(tokens["expires_at"])
            # Add 5-minute buffer to refresh before actual expiration
            return datetime.now() >= expires_at - timedelta(minutes=5)
        except (ValueError, TypeError):
            return True

    def clear_cached_tokens(self) -> None:
        """Remove cached credentials file."""
        paths = [self.credentials_path]
        if self.cache_binding is None:
            paths.append(self.legacy_credentials_path)
        for path in paths:
            if path.exists():
                path.unlink()

    def get_account_id(self) -> Optional[str]:
        """Get account ID from the current ID token's custom claims.

        Required for multi-tenant operations; the backend filters stacks by
        userId + accountId. Without accountId, the Python client may see stacks
        that submitJob cannot use.

        Returns:
            The custom:accountId claim value, or None if not present
        """
        try:
            tokens = self.get_cached_tokens()
        except (ValueError, OSError):
            return None
        try:
            id_token = tokens.get("id_token")
            if not id_token:
                return None
            payload = _decode_jwt_payload(id_token)
            return payload.get("custom:accountId")
        except (ValueError, KeyError):
            return None
