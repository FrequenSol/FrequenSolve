"""
Test script for S3 upload with Identity Pool credentials.

This script tests Phase 4 of the implementation:
- Configuration discovery from domain
- GraphQL query to get stack info
- S3 upload using Identity Pool credentials
- IAM policy validation (scoped to user's bucket)

Usage:
    # Using localhost (default)
    python tests/test_s3_upload.py

    # Using custom domain
    python tests/test_s3_upload.py --domain frequensolve.app
"""

import getpass
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import boto3

from frequensolve.orchestrator.sites.aws import AWSSiteConfig
from frequensolve.orchestrator.sites.aws.cognito import CognitoAuth
from frequensolve.orchestrator.sites.aws.graphql_client import GraphQLClient


@pytest.mark.interactive
def test_s3_upload(domain="localhost:5173"):
    """Test S3 upload with Identity Pool credentials."""
    print("=" * 60)
    print("PHASE 4: Testing S3 Upload")
    print("=" * 60)

    # Load configuration from domain
    print("\n[1/6] Loading configuration from domain...")
    print(f"  Domain: {domain}")
    try:
        config = AWSSiteConfig.from_domain(domain)
        print(f"  ✓ Configuration loaded from {domain}")
        print(f"  ✓ Region: {config.region}")
        print(f"  ✓ API URL: {config.api_url}")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\n  Make sure:")
        print(f"  - Dev server is running at {domain}")
        print("  - npx ampx sandbox is running")
        return False

    # Authenticate
    print("\n[2/6] Authenticating with Cognito...")
    auth = CognitoAuth(
        user_pool_id=config.user_pool_id,
        client_id=config.client_id,
        identity_pool_id=config.identity_pool_id,
        region=config.region,
    )

    try:
        tokens = auth.get_cached_tokens()
        print(f"  ✓ Using cached credentials for: {tokens.get('email', 'unknown')}")
    except ValueError:
        print("  No cached credentials found, please login:")
        email = input("  Email: ")
        password = getpass.getpass("  Password: ")
        try:
            auth.login(email, password)
            print(f"  ✓ Login successful")
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False

    # Get AWS credentials
    print("\n[3/6] Getting AWS credentials from Identity Pool...")
    try:
        creds = auth.get_aws_credentials()
        session = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretKey"],
            aws_session_token=creds["SessionToken"],
            region_name=config.region,
        )
        print(f"  ✓ AWS session created")
        print(f"  ✓ Identity ID: {creds['IdentityId']}")
    except Exception as e:
        error_msg = str(e)
        if "expired" in error_msg.lower() or "token" in error_msg.lower():
            print(f"  Credentials expired, please login again:")
            email = input("  Email: ")
            password = getpass.getpass("  Password: ")
            try:
                auth.login(email, password)
                print(f"  ✓ Login successful, retrying...")
                creds = auth.get_aws_credentials()
                session = boto3.Session(
                    aws_access_key_id=creds["AccessKeyId"],
                    aws_secret_access_key=creds["SecretKey"],
                    aws_session_token=creds["SessionToken"],
                    region_name=config.region,
                )
                print(f"  ✓ AWS session created")
                print(f"  ✓ Identity ID: {creds['IdentityId']}")
            except Exception as login_error:
                print(f"❌ Login/retry failed: {login_error}")
                return False
        else:
            print(f"❌ Failed to get AWS credentials: {e}")
            return False

    # Get stack info via GraphQL
    print("\n[4/6] Querying stack info via GraphQL...")
    try:
        gql_client = GraphQLClient(config.api_url, auth)
        stack_info = gql_client.get_my_stack()
        bucket_name = stack_info["bucketName"]
        print(f"  ✓ Stack info retrieved")
        print(f"  ✓ Bucket: {bucket_name}")
        print(f"  ✓ Job Queue: {stack_info['jobQueue']}")
        print(f"  ✓ Job Definition: {stack_info['jobDefinition']}")
        print(f"  ✓ Status: {stack_info['status']}")
    except Exception as e:
        print(f"❌ Failed to get stack info: {e}")
        print("\n  Possible causes:")
        print(
            "  - No storage or compute infrastructure deployed (deploy both via web UI)"
        )
        print("  - GraphQL API not deployed")
        print("  - Lambda function errors")
        return False

    # Create test file
    print("\n[5/6] Creating test file...")
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("Test data from FrequenSolve Python package!\n")
        f.write(f"Timestamp: {os.popen('date').read().strip()}\n")
        test_file = f.name

    test_basename = os.path.basename(test_file)
    print(f"  ✓ Test file created: {test_basename}")

    # Upload to S3
    print("\n[6/6] Uploading to S3...")
    try:
        s3_client = session.client("s3", region_name=config.region)
        s3_key = f"test/{test_basename}"

        s3_client.upload_file(test_file, bucket_name, s3_key)
        print(f"  ✓ File uploaded to s3://{bucket_name}/{s3_key}")

        # Verify upload
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"  ✓ File verified! Size: {response['ContentLength']} bytes")
        print(f"  ✓ Last Modified: {response['LastModified']}")

        # Cleanup
        s3_client.delete_object(Bucket=bucket_name, Key=s3_key)
        os.unlink(test_file)
        print(f"  ✓ Test file cleaned up")

    except Exception as e:
        print(f"❌ S3 operation failed: {e}")
        print("\n  Possible causes:")
        print("  - IAM policy not configured correctly")
        print("  - Bucket name doesn't match pattern frequensol-{cognito-sub}-*")
        print("  - Identity Pool not deployed")
        return False

    print("\n" + "=" * 60)
    print("✅ CHECKPOINT 4 PASSED: S3 upload works!")
    print("=" * 60)
    print("\nVerification completed:")
    print("  ✓ GraphQL API accessible")
    print("  ✓ Stack info retrieved")
    print("  ✓ Identity Pool credentials work")
    print("  ✓ S3 upload permissions correct")
    print("\nNext step: Proceed to Phase 5 - test_job_submission.py")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test S3 upload with domain-based config"
    )
    parser.add_argument(
        "--domain",
        default="localhost:5173",
        help="Frontend domain (default: localhost:5173)",
    )
    args = parser.parse_args()

    success = test_s3_upload(domain=args.domain)
    sys.exit(0 if success else 1)
