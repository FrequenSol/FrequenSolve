"""
GraphQL client for FrequenSol Cloud API.

This module provides a simple GraphQL client for interacting with
the FrequenSol AppSync API using Cognito authentication.
"""

import json
import logging
import time
from typing import Any, Dict, Optional

from frequensolve._optional import optional_dependency_error

try:
    import requests
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "GraphQLClient",
        extra="cloud",
        dependencies=("requests",),
        error=exc,
    ) from exc

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

    def get_compute_provisioning_mode(self) -> str:
        """Discover whether the Cloud API uses shared or per-user compute.

        The current Cloud API exposes ``fetchAvailableComputeRuntimes`` and
        submits work to platform-managed shared queues. Older deployments
        expose ``deployComputeInfrastructure`` for a per-user compute stack.
        Inspecting both root operation types lets one client remain compatible
        without blindly invoking a mutation that may not exist.

        Returns:
            ``"shared"`` for the runtime-catalog API or ``"per-user"`` for
            the legacy compute-deployment API.

        Raises:
            RuntimeError: If the API exposes neither recognizable contract or
                its capability response is incomplete.
        """

        capability_query = """
            query FrequenSolveCloudComputeCapabilities {
                queryType: __type(name: "Query") {
                    fields {
                        name
                    }
                }
                mutationType: __type(name: "Mutation") {
                    fields {
                        name
                    }
                }
            }
        """
        result = self.execute(capability_query)

        def field_names(type_name: str) -> set[str]:
            type_result = result.get(type_name)
            if not isinstance(type_result, dict):
                return set()
            fields = type_result.get("fields")
            if not isinstance(fields, list):
                return set()
            return {
                field["name"]
                for field in fields
                if isinstance(field, dict) and isinstance(field.get("name"), str)
            }

        query_fields = field_names("queryType")
        mutation_fields = field_names("mutationType")
        if "fetchAvailableComputeRuntimes" in query_fields:
            return "shared"
        if "deployComputeInfrastructure" in mutation_fields:
            return "per-user"
        raise RuntimeError(
            "The FrequenSolve Cloud API does not expose a supported compute "
            "submission contract. Update FrequenSolve or contact the Cloud "
            "environment owner before submitting a job."
        )

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

    def _build_storage_stack_filter(self, account_id: Optional[str] = None) -> dict:
        """Build filter for storage stack queries.

        Uses same criteria as submitJob backend: userId (via owner auth),
        accountId (explicit), stackType, and status. This ensures the Python
        client only considers stacks that submitJob will accept.
        """
        # ROLLBACK_COMPLETE = failed initial creation, no usable resources.
        # Only CREATE_COMPLETE, UPDATE_COMPLETE, UPDATE_ROLLBACK_COMPLETE are usable.
        base_filter = {
            "stackType": {"eq": "storage"},
            "or": [
                {"status": {"eq": "CREATE_COMPLETE"}},
                {"status": {"eq": "UPDATE_COMPLETE"}},
                {"status": {"eq": "UPDATE_ROLLBACK_COMPLETE"}},
            ],
        }
        if account_id:
            base_filter["accountId"] = {"eq": account_id}
        return base_filter

    def _build_compute_stack_filter(self, account_id: Optional[str] = None) -> dict:
        """Build filter for compute stack queries.

        Uses same criteria as submitJob backend: userId (via owner auth),
        accountId (explicit), stackType. Ensures the Python client only considers
        stacks that submitJob will accept.
        """
        base_filter = {"stackType": {"eq": "compute"}}
        if account_id:
            base_filter["accountId"] = {"eq": account_id}
        return base_filter

    def _check_storage_stack_exists(self) -> bool:
        """Check if storage stack exists.

        Uses same filter as submitJob backend (accountId when available) so
        we don't report a stack that submit won't accept.
        """
        account_id = (
            self.auth.get_account_id() if hasattr(self.auth, "get_account_id") else None
        )
        if account_id:
            logger.debug(
                "Filtering storage stacks by accountId to match submitJob backend"
            )
        try:
            filter_obj = self._build_storage_stack_filter(account_id)
            # Build query with dynamic filter
            variables = {"filter": filter_obj}
            storage_query = """
                query ListStorageStacks($filter: ModelStackFilterInput) {
                    listStacks(filter: $filter) {
                        items {
                            stackId
                            outputs
                            status
                            createdAt
                        }
                    }
                }
            """
            result = self.execute(storage_query, variables)
            return (
                "listStacks" in result
                and result["listStacks"]["items"]
                and len(result["listStacks"]["items"]) > 0
            )
        except Exception:
            return False

    def get_storage_stack_info(self) -> Dict[str, str]:
        """Get storage stack information (bucket name).

        Uses same filter as submitJob backend (accountId when available) so
        we only return stacks that submit will accept.

        Returns:
            Dict containing:
                - bucketName: S3 bucket name for simulations
                - stackId: CloudFormation storage stack ID
                - status: Stack status

        Raises:
            RuntimeError: If storage stack not found
        """
        account_id = (
            self.auth.get_account_id() if hasattr(self.auth, "get_account_id") else None
        )
        filter_obj = self._build_storage_stack_filter(account_id)
        variables = {"filter": filter_obj}
        storage_query = """
            query ListStorageStacks($filter: ModelStackFilterInput) {
                listStacks(filter: $filter) {
                    items {
                        stackId
                        outputs
                        status
                        createdAt
                    }
                }
            }
        """

        logger.debug("Executing GraphQL query: listStacks (storage)")
        storage_result = self.execute(storage_query, variables)

        if (
            "listStacks" not in storage_result
            or not storage_result["listStacks"]["items"]
        ):
            raise RuntimeError(
                "No active storage stack found. Storage stack will be created automatically on first sync."
            )

        # Get most recent storage stack
        storage_stacks = storage_result["listStacks"]["items"]
        storage_stack = sorted(
            storage_stacks, key=lambda s: s.get("createdAt", ""), reverse=True
        )[0]

        # Parse outputs
        storage_outputs = (
            json.loads(storage_stack["outputs"]) if storage_stack.get("outputs") else {}
        )
        bucket_name = storage_outputs.get("StorageBucketName", "")

        if not bucket_name:
            raise RuntimeError(
                "Storage stack outputs are incomplete. StorageBucketName not found."
            )

        return {
            "bucketName": bucket_name,
            "stackId": storage_stack.get("stackId", ""),
            "status": storage_stack.get("status", ""),
        }

    def _check_compute_stack_exists(self) -> bool:
        """Check if compute stack exists.

        Returns:
            True if compute stack exists, False otherwise
        """
        account_id = (
            self.auth.get_account_id() if hasattr(self.auth, "get_account_id") else None
        )
        filter_obj = self._build_compute_stack_filter(account_id)
        filter_obj["or"] = [
            {"status": {"eq": "CREATE_COMPLETE"}},
            {"status": {"eq": "UPDATE_COMPLETE"}},
            {"status": {"eq": "UPDATE_ROLLBACK_COMPLETE"}},
        ]
        variables = {"filter": filter_obj}
        compute_query = """
            query ListComputeStacks($filter: ModelStackFilterInput) {
                listStacks(filter: $filter) {
                    items {
                        stackId
                        outputs
                        status
                        createdAt
                    }
                }
            }
        """
        try:
            result = self.execute(compute_query, variables)
            return (
                "listStacks" in result
                and result["listStacks"]["items"]
                and len(result["listStacks"]["items"]) > 0
            )
        except Exception:
            return False

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
        # Only usable states: CREATE_COMPLETE, UPDATE_COMPLETE, UPDATE_ROLLBACK_COMPLETE.
        # ROLLBACK_COMPLETE = failed initial creation, no usable resources.
        storage_query = """
            query ListStorageStacks {
                listStacks(filter: {
                    stackType: { eq: "storage" }
                    or: [
                        { status: { eq: "CREATE_COMPLETE" } },
                        { status: { eq: "UPDATE_COMPLETE" } },
                        { status: { eq: "UPDATE_ROLLBACK_COMPLETE" } }
                    ]
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
        # Only usable states: CREATE_COMPLETE, UPDATE_COMPLETE, UPDATE_ROLLBACK_COMPLETE.
        # ROLLBACK_COMPLETE = failed initial creation, no usable resources.
        compute_query = """
            query ListComputeStacks {
                listStacks(filter: {
                    stackType: { eq: "compute" }
                    or: [
                        { status: { eq: "CREATE_COMPLETE" } },
                        { status: { eq: "UPDATE_COMPLETE" } },
                        { status: { eq: "UPDATE_ROLLBACK_COMPLETE" } }
                    ]
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

        logger.debug("Fetching stack information from API...")
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

        logger.debug(f"✓ Stack info loaded: bucket={bucket_name}")

        return result_dict

    def submit_job(
        self,
        job_file_s3_key: str,
        vcpu: Optional[int] = None,
        memory: Optional[int] = None,
        job_name: Optional[str] = None,
        send_simulation_status_email: Optional[bool] = None,
        fresh: bool = False,
        *,
        project_name: Optional[str] = None,
        project_display_name: Optional[str] = None,
        simulation_name: Optional[str] = None,
        simulation_job_name: Optional[str] = None,
    ) -> Dict[str, str]:
        """Submit a simulation job.

        Args:
            job_file_s3_key: S3 key of the job configuration file
            vcpu: Number of vCPUs for the job (optional)
            memory: Memory in MB for the job (optional)
            job_name: Custom name for the job (optional)
            send_simulation_status_email: If True/False, overrides cloud communication preferences for this run only
            fresh: Force a fresh solver run when the backend supports it
            project_name: Authored project name for Cloud display metadata
            project_display_name: Optional authored project display name
            simulation_name: Authored simulation name for Cloud display metadata
            simulation_job_name: Authored FrequenSolve job name for Cloud display metadata

        Returns:
            Dict containing:
                - simulationId: Database simulation record ID
                - batchJobId: Internal job ID (for backwards compatibility)
                - status: Initial job status (typically 'PENDING')

        Raises:
            RuntimeError: If job submission fails
        """
        metadata = {
            "projectName": project_name,
            "projectDisplayName": project_display_name,
            "simulationName": simulation_name,
            "simulationJobName": simulation_job_name,
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}

        def build_mutation(include_metadata: bool) -> str:
            force_var = "$forceRun: Boolean" if fresh else ""
            force_arg = "forceRun: $forceRun" if fresh else ""
            metadata_variables = ""
            metadata_arguments = ""
            if include_metadata:
                metadata_variables = """
                $projectName: String
                $projectDisplayName: String
                $simulationName: String
                $simulationJobName: String
                """
                metadata_arguments = """
                    projectName: $projectName
                    projectDisplayName: $projectDisplayName
                    simulationName: $simulationName
                    simulationJobName: $simulationJobName
                """
            return f"""
            mutation SubmitJob(
                $jobFileS3Key: String!
                $vcpu: Int
                $memory: Int
                $jobName: String
                $sendSimulationStatusEmail: Boolean
                {metadata_variables}
                {force_var}
            ) {{
                submitJob(
                    jobFileS3Key: $jobFileS3Key
                    vcpu: $vcpu
                    memory: $memory
                    jobName: $jobName
                    sendSimulationStatusEmail: $sendSimulationStatusEmail
                    {metadata_arguments}
                    {force_arg}
                ) {{
                    simulationId
                    batchJobId
                    status
                }}
            }}
        """

        variables: Dict[str, Any] = {
            "jobFileS3Key": job_file_s3_key,
            "sendSimulationStatusEmail": send_simulation_status_email,
        }

        if vcpu is not None:
            variables["vcpu"] = vcpu
        if memory is not None:
            variables["memory"] = memory
        if job_name is not None:
            variables["jobName"] = job_name
        variables.update(metadata)
        if fresh:
            variables["forceRun"] = True

        def execute_submission(include_metadata: bool) -> Dict[str, Any]:
            request_variables = dict(variables)
            if not include_metadata:
                for key in metadata:
                    request_variables.pop(key, None)
            try:
                return self.execute(
                    build_mutation(include_metadata),
                    request_variables,
                )
            except RuntimeError as exc:
                error_message = str(exc)
                unsupported_argument = (
                    "Unknown argument" in error_message
                    or "UnknownArgument" in error_message
                )
                unsupported_metadata = (
                    include_metadata
                    and unsupported_argument
                    and any(key in error_message for key in metadata)
                )
                if unsupported_metadata:
                    logger.info(
                        "Cloud run metadata is not supported by this environment; "
                        "retrying the submission with the legacy mutation contract."
                    )
                    return execute_submission(False)
                unsupported_force_run = (
                    fresh and "forceRun" in error_message and unsupported_argument
                )
                if unsupported_force_run:
                    raise RuntimeError(
                        "Fresh Cloud reruns are not supported by this environment. "
                        "Submit without force=True, rerun=True, or skip=False, or "
                        "update the Cloud backend before retrying."
                    ) from exc
                raise

        result = execute_submission(bool(metadata))

        if "submitJob" not in result or not result["submitJob"]:
            raise RuntimeError("Job submission failed: No response from API")

        return result["submitJob"]

    def get_simulation_status_details(self, simulation_id: str) -> Dict[str, Any]:
        """Get simulation status and customer-safe failure details by ID.

        Args:
            simulation_id: The simulation ID to query.

        Returns:
            Mapping containing ``status`` and any customer-safe failure fields
            exposed by the Cloud environment.

        Raises:
            RuntimeError: If query fails or simulation not found.
        """
        query = """
            query GetSimulation($id: ID!) {
                getSimulation(id: $id) {
                    id
                    status
                    failureCode
                    failureMessage
                }
            }
        """

        variables = {"id": simulation_id}
        try:
            result = self.execute(query, variables)
        except RuntimeError as exc:
            error_message = str(exc)
            unsupported_failure_fields = (
                "failureCode" in error_message or "failureMessage" in error_message
            ) and (
                "Cannot query field" in error_message or "is undefined" in error_message
            )
            if not unsupported_failure_fields:
                raise
            legacy_query = """
                query GetSimulation($id: ID!) {
                    getSimulation(id: $id) {
                        id
                        status
                    }
                }
            """
            result = self.execute(legacy_query, variables)

        if "getSimulation" not in result or not result["getSimulation"]:
            raise RuntimeError(
                f"Simulation not found or access denied: {simulation_id}"
            )

        details = result["getSimulation"]
        status = details.get("status")
        if not status:
            raise RuntimeError(
                f"Simulation status not found in response: {simulation_id}"
            )

        return {
            "id": details.get("id"),
            "status": status,
            "failureCode": details.get("failureCode"),
            "failureMessage": details.get("failureMessage"),
        }

    def get_simulation_status(self, simulation_id: str) -> str:
        """Get a simulation status string by ID."""

        return str(self.get_simulation_status_details(simulation_id)["status"])

    def deploy_storage_stack(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Deploy storage infrastructure stack.

        User ID is automatically extracted from the authentication context by AppSync.
        ``environment`` is accepted for backward compatibility and ignored; the
        backend derives the deployment environment from its own runtime context.

        Returns:
            Dict containing stackId, stackName, status, outputs, error

        Raises:
            RuntimeError: If deployment fails
        """
        mutation = """
            mutation DeployStorage {
                deployStorage {
                    stackId
                    stackName
                    status
                    outputs
                    error
                }
            }
        """

        logger.debug("Deploying storage stack...")
        result = self.execute(mutation)

        if "deployStorage" not in result:
            raise RuntimeError("Storage stack deployment failed: No response from API")

        deploy_result = result["deployStorage"]

        if deploy_result.get("error"):
            raise RuntimeError(
                f"Storage stack deployment failed: {deploy_result['error']}"
            )

        logger.debug(
            f"✓ Storage stack deployment initiated: {deploy_result.get('stackName', 'unknown')}"
        )
        logger.debug(f"  Stack ID: {deploy_result.get('stackId')}")
        logger.debug(f"  Status: {deploy_result.get('status')}")

        return deploy_result

    def deploy_compute_stack(
        self,
        environment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Deploy compute infrastructure stack.

        User ID is automatically extracted from the authentication context by AppSync.
        Stack parameters are automatically fetched from user's compute settings in the database.
        ``environment`` is accepted for backward compatibility and ignored; the
        backend derives the deployment environment from its own runtime context.

        Returns:
            Dict containing stackId, stackName, status, outputs, error

        Raises:
            RuntimeError: If deployment fails
        """
        mutation = """
            mutation DeployComputeInfrastructure {
                deployComputeInfrastructure {
                    stackId
                    stackName
                    status
                    outputs
                    error
                }
            }
        """

        logger.debug("Deploying compute stack...")
        result = self.execute(mutation)

        if "deployComputeInfrastructure" not in result:
            raise RuntimeError("Compute stack deployment failed: No response from API")

        deploy_result = result["deployComputeInfrastructure"]

        if deploy_result.get("error"):
            raise RuntimeError(
                f"Compute stack deployment failed: {deploy_result['error']}"
            )

        logger.debug(
            f"✓ Compute stack deployment initiated: {deploy_result.get('stackName', 'unknown')}"
        )
        logger.debug(f"  Stack ID: {deploy_result.get('stackId')}")
        logger.debug(f"  Status: {deploy_result.get('status')}")

        return deploy_result

    def wait_for_stack_ready(
        self,
        stack_type: str,
        timeout: int = 1800,
        poll_interval: int = 15,
        expected_stack_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Wait for stack creation to complete by polling stack status.

        After triggering stack creation, waits 30 seconds before acting on status
        responses to avoid race conditions where AWS might return stale status
        (e.g., DELETE_COMPLETE from a previously deleted stack).

        Args:
            stack_type: Type of stack to wait for ("storage" or "compute")
            timeout: Maximum time to wait in seconds (default: 1800 = 30 minutes)
            poll_interval: Interval between status checks in seconds (default: 15)
            expected_stack_id: Optional stack ID to match (helps avoid stale status)

        Returns:
            Dict containing stack info (stackId, bucketName, status)

        Raises:
            RuntimeError: If stack creation fails, times out, or enters failed state
        """
        logger.debug(
            f"Waiting for {stack_type} stack to be ready (timeout: {timeout}s, poll interval: {poll_interval}s)..."
        )

        start_time = time.time()
        last_status = None
        # Wait 30 seconds before acting on status responses to avoid stale status
        grace_period = 30

        while True:
            # Check timeout
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout:
                raise RuntimeError(
                    f"Timeout waiting for {stack_type} stack to be ready "
                    f"(waited {elapsed_time:.0f}s, timeout: {timeout}s)"
                )

            try:
                # Use same filter as get_storage_stack_info / submitJob (accountId when available)
                # so we only consider stacks that will be accepted downstream
                account_id = (
                    self.auth.get_account_id()
                    if hasattr(self.auth, "get_account_id")
                    else None
                )
                if stack_type == "storage":
                    filter_obj = self._build_storage_stack_filter(account_id)
                    # Relax status for polling - we want to see IN_PROGRESS too
                    filter_obj.pop("or", None)  # Remove status filter to see any status
                    variables = {"filter": filter_obj}
                    query = """
                        query ListStorageStacks($filter: ModelStackFilterInput) {
                            listStacks(filter: $filter) {
                                items {
                                    stackId
                                    outputs
                                    status
                                    createdAt
                                }
                            }
                        }
                    """
                    result = self.execute(query, variables)
                elif stack_type == "compute":
                    filter_obj = self._build_compute_stack_filter(account_id)
                    variables = {"filter": filter_obj}
                    query = """
                        query ListComputeStacks($filter: ModelStackFilterInput) {
                            listStacks(filter: $filter) {
                                items {
                                    stackId
                                    outputs
                                    status
                                    createdAt
                                }
                            }
                        }
                    """
                    result = self.execute(query, variables)
                else:
                    raise ValueError(
                        f"Invalid stack_type: {stack_type}. Must be 'storage' or 'compute'"
                    )

                stacks = result.get("listStacks", {}).get("items", [])

                if not stacks:
                    # Stack not found yet, continue polling
                    if last_status != "NOT_FOUND":
                        logger.debug(
                            f"{stack_type} stack not found yet, continuing to poll..."
                        )
                        last_status = "NOT_FOUND"
                    time.sleep(poll_interval)
                    continue

                # If we have an expected stackId, try to match it first
                stack = None
                if expected_stack_id:
                    matching_stack = next(
                        (s for s in stacks if s.get("stackId") == expected_stack_id),
                        None,
                    )
                    if matching_stack:
                        stack = matching_stack
                        status = stack.get("status", "")
                        logger.debug(
                            f"Matched {stack_type} stack by stackId: {expected_stack_id}, status: {status}"
                        )
                    else:
                        logger.debug(
                            f"Expected stackId {expected_stack_id} not found yet, will use most recent"
                        )
                        # Fall through to most recent logic
                        expected_stack_id = None  # Clear it so we don't keep trying

                # If we didn't match by stackId, use filtering and sorting logic
                if stack is None:
                    # Filter out stacks with UNKNOWN status (they're likely stale or incorrectly updated)
                    # Prefer stacks with actual status values
                    known_status_stacks = [
                        s for s in stacks if s.get("status", "") != "UNKNOWN"
                    ]

                    if known_status_stacks:
                        # Use stacks with known status, sorted by most recent
                        stacks_to_use = known_status_stacks
                    else:
                        # Fall back to all stacks if all are UNKNOWN (shouldn't happen, but be safe)
                        stacks_to_use = stacks
                        logger.warning(
                            f"All {stack_type} stacks have UNKNOWN status, using most recent one"
                        )

                    # Get most recent stack (by createdAt)
                    stack = sorted(
                        stacks_to_use,
                        key=lambda s: s.get("createdAt", ""),
                        reverse=True,
                    )[0]
                    status = stack.get("status", "")

                    # Log which stack we're using for debugging
                    logger.debug(
                        f"Using most recent {stack_type} stack: {stack.get('stackId', 'unknown')} with status: {status}"
                    )

                # Log status changes
                if status != last_status:
                    logger.debug(f"{stack_type} stack status: {status}")
                    last_status = status

                # Before grace period expires, ignore terminal failure states that might be stale
                # (especially DELETE_COMPLETE from a previously deleted stack)
                if elapsed_time < grace_period:
                    # Log that we're in grace period and ignoring terminal states
                    if status in [
                        "CREATE_FAILED",
                        "ROLLBACK_COMPLETE",
                        "ROLLBACK_FAILED",
                        "DELETE_COMPLETE",
                        "DELETE_FAILED",
                        "UPDATE_ROLLBACK_COMPLETE",
                        "UPDATE_ROLLBACK_FAILED",
                    ]:
                        logger.debug(
                            f"Ignoring {status} status during grace period "
                            f"({elapsed_time:.0f}s < {grace_period}s) - may be stale status from previous stack"
                        )
                        time.sleep(poll_interval)
                        continue

                # Check for terminal success states
                if status in ["CREATE_COMPLETE", "UPDATE_COMPLETE"]:
                    logger.debug(f"✓ {stack_type} stack is ready: {status}")
                    # Return stack info - for storage we need bucketName, for compute we need stackId
                    if stack_type == "storage":
                        outputs = (
                            json.loads(stack["outputs"]) if stack.get("outputs") else {}
                        )
                        bucket_name = outputs.get("StorageBucketName", "")
                        return {
                            "stackId": stack.get("stackId", ""),
                            "bucketName": bucket_name,
                            "status": status,
                        }
                    else:  # compute
                        return {
                            "stackId": stack.get("stackId", ""),
                            "status": status,
                        }

                # Check for terminal failure states (only after grace period)
                if status in [
                    "CREATE_FAILED",
                    "ROLLBACK_COMPLETE",
                    "ROLLBACK_FAILED",
                    "DELETE_COMPLETE",
                    "DELETE_FAILED",
                    "UPDATE_ROLLBACK_COMPLETE",
                    "UPDATE_ROLLBACK_FAILED",
                ]:
                    error_msg = (
                        f"{stack_type} stack creation failed with status: {status}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Continue polling for in-progress states
                if status in [
                    "CREATE_IN_PROGRESS",
                    "UPDATE_IN_PROGRESS",
                    "ROLLBACK_IN_PROGRESS",
                    "DELETE_IN_PROGRESS",
                ]:
                    time.sleep(poll_interval)
                    continue

                # Unknown status - log warning but continue polling
                logger.warning(
                    f"Unknown {stack_type} stack status '{status}'. Continuing to poll..."
                )
                time.sleep(poll_interval)

            except RuntimeError as e:
                # If it's a "not found" error, continue polling
                error_msg = str(e).lower()
                if "no active" in error_msg or "not found" in error_msg:
                    if last_status != "NOT_FOUND":
                        logger.debug(
                            f"{stack_type} stack not found yet, continuing to poll..."
                        )
                        last_status = "NOT_FOUND"
                    time.sleep(poll_interval)
                    continue
                # Re-raise other RuntimeErrors
                raise
            except Exception as e:
                # For other errors, log and retry after interval
                logger.warning(
                    f"Error checking {stack_type} stack status: {e}. "
                    f"Retrying in {poll_interval} seconds..."
                )
                time.sleep(poll_interval)
