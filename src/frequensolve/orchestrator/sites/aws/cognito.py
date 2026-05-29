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
from datetime import datetime, timedelta
from typing import Dict, Optional

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
    2. Store tokens locally (~/.frequensolve/cloud/credentials)
    3. Automatically refresh expired tokens
    4. Exchange ID token for AWS credentials via Identity Pool

    Args:
        user_pool_id: Cognito User Pool ID
        client_id: Cognito App Client ID
        identity_pool_id: Cognito Identity Pool ID
        region: AWS region (default: us-east-1)
    """

    def __init__(
        self,
        user_pool_id: str,
        client_id: str,
        identity_pool_id: str,
        region: str = "us-east-1",
        credential_cache_name: Optional[str] = None,
    ):
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.identity_pool_id = identity_pool_id
        self.region = region
        self.credential_cache_name = credential_cache_name

        # Initialize AWS clients
        self.cognito_client = boto3.client("cognito-idp", region_name=region)
        self.identity_client = boto3.client("cognito-identity", region_name=region)

        # Path to credentials file
        self.credentials_path = cloud_credentials_path(credential_cache_name)
        self.legacy_credentials_path = legacy_credentials_path()

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
        logger.debug(f"  ID Token (first 50 chars): {id_token[:50]}...")

        try:
            # Get Identity ID
            identity_response = self.identity_client.get_id(
                IdentityPoolId=self.identity_pool_id, Logins={provider_name: id_token}
            )

            logger.debug("Identity Response:")
            logger.debug(f"  {json.dumps(identity_response, indent=2, default=str)}")

            identity_id = identity_response["IdentityId"]
            logger.debug(f"Identity ID: {identity_id}")

            # Get AWS credentials for this identity
            logger.debug("Getting credentials for identity...")
            credentials_response = self.identity_client.get_credentials_for_identity(
                IdentityId=identity_id, Logins={provider_name: id_token}
            )

            logger.debug("Credentials Response:")
            logger.debug(f"  {json.dumps(credentials_response, indent=2, default=str)}")

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
            logger.debug(
                f"  Full Response: {json.dumps(e.response, indent=2, default=str)}"
            )

            error_code = e.response["Error"]["Code"]
            if error_code == "NotAuthorizedException":
                raise ValueError("Authentication failed. Please login again.") from e
            else:
                raise
        except KeyError as e:
            logger.debug(f"KeyError - missing key: {e}")
            logger.debug(f"  Available keys in credentials: {list(credentials.keys())}")
            raise

    def get_cached_tokens(self) -> Dict[str, str]:
        """Load tokens from local cache file.

        Returns:
            Dict containing cached tokens

        Raises:
            ValueError: If credentials file not found or invalid
        """
        if not self.credentials_path.exists():
            if self.legacy_credentials_path.exists():
                try:
                    with open(self.legacy_credentials_path, "r") as f:
                        tokens = json.load(f)
                    self.save_tokens(tokens)
                    logger.info("Migrated cached credentials to cloud cache directory")
                    return tokens
                except (json.JSONDecodeError, IOError) as e:
                    raise ValueError(f"Failed to read credentials file: {e}") from e
            raise ValueError(
                "No cached credentials found. Please login first.\n"
                'Run: site = AWSSite.from_cognito(email="your@email.com", password="...")'
            )

        try:
            with open(self.credentials_path, "r") as f:
                tokens = json.load(f)
            logger.debug("Using cached credentials")
            return tokens
        except (json.JSONDecodeError, IOError) as e:
            raise ValueError(f"Failed to read credentials file: {e}") from e

    def save_tokens(self, tokens: Dict[str, str]) -> None:
        """Save tokens to local cache file.

        Args:
            tokens: Dict containing tokens to save
        """
        with open(self.credentials_path, "w") as f:
            json.dump(tokens, f, indent=2)

        # Make file readable only by owner (chmod 600)
        os.chmod(self.credentials_path, 0o600)

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
        for path in (self.credentials_path, self.legacy_credentials_path):
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
