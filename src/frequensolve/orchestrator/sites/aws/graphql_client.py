"""
GraphQL client for FrequenSol Cloud API.

This module provides a simple GraphQL client for interacting with
the FrequenSol AppSync API using Cognito authentication.
"""

import json
import logging
from typing import Any, Dict, Optional

import requests

from .cognito import CognitoAuth

logger = logging.getLogger(__name__)


class GraphQLClient:
    """GraphQL client for FrequenSol Cloud API.

    This client handles:
    - GraphQL query and mutation execution
    - Automatic ID token injection
    - Helper methods for common operations

    Args:
        api_url: GraphQL API endpoint URL
        auth: CognitoAuth instance for authentication
    """

    def __init__(self, api_url: str, auth: CognitoAuth):
        self.api_url = api_url
        self.auth = auth

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with ID token for AppSync authentication.

        Returns:
            Dict of HTTP headers including Authorization
        """
        id_token = self.auth.get_id_token()
        return {
            "Authorization": id_token,  # AppSync expects just the token
            "Content-Type": "application/json",
        }

    def execute(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a GraphQL query or mutation.

        Args:
            query: GraphQL query or mutation string
            variables: Optional variables for the query

        Returns:
            Response data from GraphQL API

        Raises:
            RuntimeError: If API request fails
        """
        payload = {
            "query": query,
        }

        if variables:
            payload["variables"] = variables

        try:
            response = requests.post(
                self.api_url, headers=self._get_headers(), json=payload, timeout=30
            )
            response.raise_for_status()

            result = response.json()

            # Check for GraphQL errors
            if "errors" in result:
                error_messages = [
                    err.get("message", str(err)) for err in result["errors"]
                ]
                raise RuntimeError(f"GraphQL errors: {'; '.join(error_messages)}")

            return result.get("data", {})

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}") from e

    def get_my_stack(self) -> Dict[str, str]:
        """Query user's stack information.

        Queries the Stack model for both storage and compute stacks separately.
        Owner-based authorization ensures only the user's stacks are returned.

        Returns:
            Dict containing:
                - stackId: CloudFormation compute stack ID
                - bucketName: S3 bucket name for simulations (from storage stack)
                - status: Combined stack status

        Raises:
            RuntimeError: If query fails or required stacks not found
        """
        import json

        # Query storage stack
        storage_query = """
            query ListStorageStacks {
                listStacks(filter: {
                    stackType: { eq: "storage" }
                    status: { eq: "CREATE_COMPLETE" }
                }) {
                    items {
                        stackId
                        outputs
                        status
                        createdAt
                    }
                }
            }
        """

        # Query compute stack
        compute_query = """
            query ListComputeStacks {
                listStacks(filter: {
                    stackType: { eq: "compute" }
                    status: { eq: "CREATE_COMPLETE" }
                }) {
                    items {
                        stackId
                        outputs
                        status
                        createdAt
                    }
                }
            }
        """

        logger.info("Fetching stack information from API...")
        logger.debug("Executing GraphQL query: listStacks (storage)")
        storage_result = self.execute(storage_query)

        logger.debug("Storage stack GraphQL result:")
        logger.debug(f"  {json.dumps(storage_result, indent=2)}")

        logger.debug("Executing GraphQL query: listStacks (compute)")
        compute_result = self.execute(compute_query)

        logger.debug("Compute stack GraphQL result:")
        logger.debug(f"  {json.dumps(compute_result, indent=2)}")

        # Check storage stack
        if (
            "listStacks" not in storage_result
            or not storage_result["listStacks"]["items"]
        ):
            logger.debug("No storage stacks found in result")
            raise RuntimeError(
                "No active storage stack found. Please deploy storage infrastructure first at "
                "https://app.frequensol.com"
            )

        # Check compute stack
        if (
            "listStacks" not in compute_result
            or not compute_result["listStacks"]["items"]
        ):
            logger.debug("No compute stacks found in result")
            raise RuntimeError(
                "No active compute stack found. Please deploy compute infrastructure first at "
                "https://app.frequensol.com"
            )

        # Get most recent storage stack (in case there are multiple)
        storage_stacks = storage_result["listStacks"]["items"]
        logger.debug(f"Found {len(storage_stacks)} storage stack(s)")

        storage_stack = sorted(
            storage_stacks, key=lambda s: s["createdAt"], reverse=True
        )[0]
        logger.debug("Using most recent storage stack:")
        logger.debug(f"  Stack ID: {storage_stack.get('stackId')}")
        logger.debug(f"  Status: {storage_stack.get('status')}")
        logger.debug(f"  Outputs (raw): {storage_stack.get('outputs', 'NULL')}")

        # Get most recent compute stack (in case there are multiple)
        compute_stacks = compute_result["listStacks"]["items"]
        logger.debug(f"Found {len(compute_stacks)} compute stack(s)")

        compute_stack = sorted(
            compute_stacks, key=lambda s: s["createdAt"], reverse=True
        )[0]
        logger.debug("Using most recent compute stack:")
        logger.debug(f"  Stack ID: {compute_stack.get('stackId')}")
        logger.debug(f"  Status: {compute_stack.get('status')}")
        logger.debug(f"  Outputs (raw): {compute_stack.get('outputs', 'NULL')}")

        # Parse outputs JSON from both stacks
        storage_outputs = (
            json.loads(storage_stack["outputs"]) if storage_stack.get("outputs") else {}
        )
        logger.debug("Parsed storage outputs:")
        logger.debug(f"  {json.dumps(storage_outputs, indent=2)}")

        compute_outputs = (
            json.loads(compute_stack["outputs"]) if compute_stack.get("outputs") else {}
        )
        logger.debug("Parsed compute outputs:")
        logger.debug(f"  {json.dumps(compute_outputs, indent=2)}")

        # Merge outputs from both stacks
        bucket_name = storage_outputs.get("StorageBucketName", "")

        # Verify all required outputs are present
        if not bucket_name:
            raise RuntimeError(
                "Storage stack outputs are incomplete. StorageBucketName not found. "
                "Please check your storage infrastructure deployment at https://app.frequensol.com"
            )

        result_dict = {
            "stackId": compute_stack["stackId"],
            "bucketName": bucket_name,
            "status": f"storage:{storage_stack['status']}, compute:{compute_stack['status']}",
        }

        logger.debug("Final merged result:")
        logger.debug(f"  {json.dumps(result_dict, indent=2)}")

        logger.info(f"✓ Stack info loaded: bucket={bucket_name}")

        return result_dict

    def submit_job(
        self,
        job_file_s3_key: str,
        vcpu: Optional[int] = None,
        memory: Optional[int] = None,
        job_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """Submit a simulation job.

        Args:
            job_file_s3_key: S3 key of the job configuration file
            vcpu: Number of vCPUs for the job (optional)
            memory: Memory in MB for the job (optional)
            job_name: Custom name for the job (optional)

        Returns:
            Dict containing:
                - simulationId: Database simulation record ID
                - batchJobId: Internal job ID (for backwards compatibility)
                - status: Initial job status (typically 'PENDING')

        Raises:
            RuntimeError: If job submission fails
        """
        mutation = """
            mutation SubmitJob(
                $jobFileS3Key: String!
                $vcpu: Int
                $memory: Int
                $jobName: String
            ) {
                submitJob(
                    jobFileS3Key: $jobFileS3Key
                    vcpu: $vcpu
                    memory: $memory
                    jobName: $jobName
                ) {
                    simulationId
                    batchJobId
                    status
                }
            }
        """

        variables = {
            "jobFileS3Key": job_file_s3_key,
        }

        if vcpu is not None:
            variables["vcpu"] = vcpu
        if memory is not None:
            variables["memory"] = memory
        if job_name is not None:
            variables["jobName"] = job_name

        result = self.execute(mutation, variables)

        if "submitJob" not in result or not result["submitJob"]:
            raise RuntimeError("Job submission failed: No response from API")

        return result["submitJob"]

    def get_simulation_status(self, simulation_id: str) -> str:
        """Get simulation status by ID.

        Args:
            simulation_id: The simulation ID to query.

        Returns:
            Simulation status string (e.g., 'PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED').

        Raises:
            RuntimeError: If query fails or simulation not found.
        """
        query = """
            query GetSimulation($id: ID!) {
                getSimulation(id: $id) {
                    id
                    status
                }
            }
        """

        variables = {"id": simulation_id}

        result = self.execute(query, variables)

        if "getSimulation" not in result or not result["getSimulation"]:
            raise RuntimeError(
                f"Simulation not found or access denied: {simulation_id}"
            )

        status = result["getSimulation"].get("status")
        if not status:
            raise RuntimeError(
                f"Simulation status not found in response: {simulation_id}"
            )

        return status
