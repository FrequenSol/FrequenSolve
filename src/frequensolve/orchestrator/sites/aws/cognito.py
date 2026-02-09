"""
Cognito authentication for FrequenSol Cloud.

This module handles:
- User authentication with Cognito User Pool
- Token caching and refresh
- AWS credential exchange via Cognito Identity Pool
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class CognitoAuth:
    """Handle Cognito authentication and AWS credential management.

    This class manages the complete authentication flow:
    1. Authenticate with Cognito User Pool (email/password)
    2. Store tokens locally (~/.frequensolve/credentials)
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
    ):
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.identity_pool_id = identity_pool_id
        self.region = region

        # Initialize AWS clients
        self.cognito_client = boto3.client("cognito-idp", region_name=region)
        self.identity_client = boto3.client("cognito-identity", region_name=region)

        # Path to credentials file
        self.credentials_path = Path.home() / ".frequensolve" / "credentials"
        self.credentials_path.parent.mkdir(parents=True, exist_ok=True)

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
        logger.info(f"Authenticating with Cognito as {email}...")
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

            logger.info("✓ Authentication successful")

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

        logger.info("Fetching AWS credentials from Identity Pool...")
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

            logger.info("✓ AWS credentials obtained successfully")

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
            raise ValueError(
                "No cached credentials found. Please login first.\n"
                'Run: site = AWSSite.from_cognito(email="your@email.com", password="...")'
            )

        try:
            with open(self.credentials_path, "r") as f:
                tokens = json.load(f)
            logger.info("Using cached credentials")
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
        if self.credentials_path.exists():
            self.credentials_path.unlink()
