"""
Test AWSSite creation using domain-based configuration.
"""

import os
import sys

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frequensolve.orchestrator.sites.aws import AWSSite


@pytest.mark.cloud
@pytest.mark.interactive
def test_awssite_from_domain():
    """Test creating AWSSite using only a domain."""
    print("=" * 60)
    print("Testing AWSSite() with domain")
    print("=" * 60)
    print()

    try:
        # Create site with domain (uses cached credentials from previous tests)
        print("Creating AWSSite with domain='localhost:5173'...")
        site = AWSSite(domain="localhost:5173")

        print("✅ AWSSite created successfully!")
        print()
        print(f"  Bucket: {site.config.s3_bucket}")
        print(f"  Job Queue: {site.config.job_queue}")
        print(f"  Job Definition: {site.config.job_definition}")
        print(f"  Region: {site.config.region}")
        print()

        # Verify clients are initialized
        assert site.session is not None, "Session should be initialized"
        assert site.s3_client is not None, "S3 client should be initialized"
        assert site.batch_client is not None, "Batch client should be initialized"
        assert site.graphql_client is not None, "GraphQL client should be initialized"

        print("✅ All clients initialized correctly")

        return True

    except Exception as e:
        print(f"❌ Failed to create AWSSite: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("FrequenSol AWSSite Domain Test")
    print()

    success = test_awssite_from_domain()

    print()
    print("=" * 60)
    if success:
        print("✅ Test passed!")
        print()
        print("Users can now authenticate with just:")
        print("  site = AWSSite(domain='frequensolve.app')")
    else:
        print("❌ Test failed")
    print("=" * 60)

    sys.exit(0 if success else 1)
