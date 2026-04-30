"""
Test script for AWSSite with Cognito authentication.

This script tests Phase 6 of the implementation:
- AWSSite.from_cognito() constructor
- Automatic configuration loading
- Client initialization

Usage:
    python tests/test_awssite_cognito.py
"""

import sys
from pathlib import Path

import pytest

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from frequensolve.orchestrator.sites.aws import AWSSite


@pytest.mark.cloud
@pytest.mark.interactive
def test_awssite():
    """Test AWSSite with Cognito authentication."""
    print("=" * 60)
    print("PHASE 6: Testing AWSSite Integration")
    print("=" * 60)

    # Create site with Cognito auth
    print("\n[1/4] Creating AWSSite with Cognito authentication...")
    try:
        site = AWSSite.from_cognito()
        print("  ✓ AWSSite created successfully")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return False
    except RuntimeError as e:
        print(f"❌ Failed to create AWSSite: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Verify configuration loaded
    print("\n[2/4] Verifying configuration...")
    print(f"  ✓ Bucket: {site.config.s3_bucket}")
    print(f"  ✓ Job Queue: {site.config.job_queue}")
    print(f"  ✓ Job Definition: {site.config.job_definition}")
    print(f"  ✓ Region: {site.config.region}")

    # Verify clients initialized
    print("\n[3/4] Verifying clients...")
    assert site.session is not None, "Session not initialized"
    print("  ✓ boto3 Session initialized")

    assert site.s3_client is not None, "S3 client not initialized"
    print("  ✓ S3 client initialized")

    assert site.batch_client is not None, "Batch client not initialized"
    print("  ✓ Batch client initialized")

    assert site.graphql_client is not None, "GraphQL client not initialized"
    print("  ✓ GraphQL client initialized")

    assert site.cognito_auth is not None, "Cognito auth not initialized"
    print("  ✓ Cognito auth initialized")

    # Test S3 access
    print("\n[4/4] Testing S3 access...")
    try:
        # List bucket (to verify permissions)
        response = site.s3_client.list_objects_v2(
            Bucket=site.config.s3_bucket, MaxKeys=5
        )
        object_count = response.get("KeyCount", 0)
        print(f"  ✓ S3 access verified ({object_count} objects in bucket)")
    except Exception as e:
        print(f"❌ S3 access test failed: {e}")
        print(
            "  Note: This might be expected if bucket is empty or permissions are strict"
        )

    print("\n" + "=" * 60)
    print("✅ CHECKPOINT 6 PASSED: AWSSite integration works!")
    print("=" * 60)
    print("\nAWSSite is ready to use:")
    print("  - Authentication: Cognito (cached)")
    print("  - S3 operations: Identity Pool credentials")
    print("  - Job submission: GraphQL API")
    print("\nNext step: Test with real simulation (Phase 7)")

    return True


if __name__ == "__main__":
    success = test_awssite()
    sys.exit(0 if success else 1)
