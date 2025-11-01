"""
Test script for Cognito authentication.

This script tests Phase 3 of the implementation:
- Cognito User Pool authentication
- Token caching
- Identity Pool credential exchange

Usage:
    python tests/test_auth.py
"""

import getpass
import sys
from pathlib import Path

import pytest

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frequensolve.orchestrator.sites.aws import AWSSiteConfig
from frequensolve.orchestrator.sites.aws.cognito import CognitoAuth


@pytest.mark.interactive
def test_login():
    """Test Cognito authentication flow."""
    print("=" * 60)
    print("PHASE 3: Testing Cognito Authentication")
    print("=" * 60)

    try:
        # Load configuration from environment variables
        print("\n[1/5] Loading configuration...")
        config = AWSSiteConfig()
        print(f"  ✓ User Pool ID: {config.user_pool_id}")
        print(f"  ✓ Client ID: {config.client_id}")
        print(f"  ✓ Identity Pool ID: {config.identity_pool_id}")
        print(f"  ✓ Region: {config.region}")

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("\nPlease provide a domain or set FREQUENSOL_DOMAIN:")
        print("  export FREQUENSOL_DOMAIN='frequensolve.app'")
        print("\nOr simply use:")
        print("  site = AWSSite(domain='frequensolve.app')")
        return False

    # Initialize authentication
    print("\n[2/5] Initializing Cognito authentication...")
    auth = CognitoAuth(
        user_pool_id=config.user_pool_id,
        client_id=config.client_id,
        identity_pool_id=config.identity_pool_id,
        region=config.region,
    )
    print("  ✓ Cognito clients initialized")

    # Check for cached tokens
    print("\n[3/5] Checking for cached tokens...")
    try:
        cached = auth.get_cached_tokens()
        print(f"  ✓ Found cached tokens for: {cached.get('email', 'unknown')}")
        print(f"  ✓ Tokens expire at: {cached.get('expires_at', 'unknown')}")
        use_cached = input("\n  Use cached tokens? [Y/n]: ").strip().lower()
        if use_cached == "n":
            raise ValueError("User chose to re-authenticate")
    except (ValueError, FileNotFoundError) as e:
        print("  ℹ No cached tokens found - authentication required")

        # Prompt for credentials
        email = input("\nFrequenSol Email: ")
        password = getpass.getpass("Password: ")

        print("\n  Authenticating with Cognito User Pool...")
        try:
            tokens = auth.login(email, password)
            print("  ✓ Login successful!")
            print(f"  ✓ ID Token: {tokens['id_token'][:50]}...")
            print(f"  ✓ Refresh Token: {tokens['refresh_token'][:50]}...")
            print(f"  ✓ Expires at: {tokens['expires_at']}")
        except ValueError as e:
            print(f"\n❌ Authentication failed: {e}")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error during login: {e}")
            return False

    # Test AWS credential exchange
    print("\n[4/5] Exchanging tokens for AWS credentials...")
    try:
        creds = auth.get_aws_credentials()
        print("  ✓ AWS credentials obtained!")
        print(f"  ✓ Access Key ID: {creds['AccessKeyId'][:20]}...")
        print(f"  ✓ Identity ID: {creds['IdentityId']}")
        print(f"  ✓ Expires: {creds['Expiration']}")
    except Exception as e:
        print(f"\n❌ Failed to get AWS credentials: {e}")
        return False

    # Verify cached tokens can be read
    print("\n[5/5] Verifying token cache...")
    try:
        cached = auth.get_cached_tokens()
        print("  ✓ Tokens successfully cached!")
        print(f"  ✓ Cache location: {auth.credentials_path}")
    except Exception as e:
        print(f"\n❌ Failed to read cached tokens: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ CHECKPOINT 3 PASSED: Authentication works!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify ~/.frequensolve/credentials file exists")
    print("2. Run this script again (should use cached tokens)")
    print("3. Proceed to Phase 4: test_s3_upload.py")

    return True


if __name__ == "__main__":
    success = test_login()
    sys.exit(0 if success else 1)
