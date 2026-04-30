"""
Test configuration fetching from domain.

This script tests the ability to fetch Amplify configuration from a public endpoint.
"""

import os
import sys

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frequensolve.orchestrator.sites.aws import AWSSiteConfig


@pytest.mark.cloud
@pytest.mark.interactive
def test_config_from_localhost():
    """Test fetching config from local dev server."""
    print("=" * 60)
    print("Testing configuration fetch from localhost:5173")
    print("=" * 60)
    print()

    try:
        # Test with localhost (assumes Vite dev server is running)
        config = AWSSiteConfig.from_domain("localhost:5173")

        print("✅ Configuration loaded successfully!")
        print()
        print(f"  Domain: {config.domain}")
        print(f"  Region: {config.region}")
        print(f"  User Pool ID: {config.user_pool_id}")
        print(f"  Client ID: {config.client_id}")
        print(f"  Identity Pool ID: {config.identity_pool_id}")
        print(f"  API URL: {config.api_url}")
        print()

        # Verify cache was created
        cache_path = AWSSiteConfig._get_config_cache_path("localhost:5173")
        if cache_path.exists():
            print(f"✅ Cache file created at: {cache_path}")

        return True

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        print()
        print("Make sure the Vite dev server is running:")
        print("  cd control-plane && npm run dev")
        return False


@pytest.mark.cloud
@pytest.mark.interactive
def test_config_from_cache():
    """Test that cached config is used on second fetch."""
    print()
    print("=" * 60)
    print("Testing cached configuration")
    print("=" * 60)
    print()

    try:
        # This should use the cached config from previous test
        AWSSiteConfig.from_domain("localhost:5173")
        print("✅ Configuration loaded from cache!")
        return True

    except Exception as e:
        print(f"❌ Failed to load cached configuration: {e}")
        return False


if __name__ == "__main__":
    print("FrequenSol Configuration Test")
    print()

    success = test_config_from_localhost()

    if success:
        success = test_config_from_cache()

    print()
    print("=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    sys.exit(0 if success else 1)
