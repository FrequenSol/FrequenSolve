"""
GraphQL client for FrequenSol Cloud API.

This module provides a simple GraphQL client for interacting with
the FrequenSol AppSync API using Cognito authentication.
"""

import json
import logging
import time
from collections.abc import Mapping
from typing import Any, Dict, Optional

from frequensolve._optional import optional_dependency_error
from frequensolve.orchestrator.sites.execution import normalize_execution_state
from frequensolve.orchestrator.utils.status_errors import TransientStatusReadError

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


def _string_values(value: object) -> set[str]:
    """Collect request strings that must be redacted from provider errors."""

    if isinstance(value, str):
        return {value} if value else set()
    if isinstance(value, Mapping):
        result: set[str] = set()
        for item in value.values():
            result.update(_string_values(item))
        return result
    if isinstance(value, (list, tuple)):
        result = set()
        for item in value:
            result.update(_string_values(item))
        return result
    return set()


def _redact_provider_message(message: object, secrets: set[str]) -> str:
    """Return a bounded provider diagnostic without request-supplied values."""

    safe = str(message).replace("\r", " ").replace("\n", " ")
    for secret in sorted(secrets, key=len, reverse=True):
        if len(secret) >= 3:
            safe = safe.replace(secret, "<redacted>")
    return safe[:500]


class CloudTransportError(RuntimeError):
    """The Cloud request could not obtain a response."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class CloudAPIError(RuntimeError):
    """Cloud rejected the request at the HTTP, GraphQL, or resolver level."""


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

        headers = self._get_headers()
        try:
            response = requests.post(
                self.api_url, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise CloudTransportError(
                "Cloud API request timed out after 30 seconds", retryable=True
            ) from None
        except requests.exceptions.HTTPError:
            raise CloudAPIError("Cloud API rejected the HTTP request") from None
        except requests.exceptions.RequestException as exc:
            raise CloudTransportError(
                f"Cloud API request failed ({type(exc).__name__})",
                retryable=isinstance(exc, requests.exceptions.ConnectionError)
                and not isinstance(exc, requests.exceptions.SSLError),
            ) from None

        try:
            result = response.json()
        except (TypeError, ValueError):
            raise RuntimeError("Cloud API returned malformed JSON") from None
        if not isinstance(result, Mapping):
            raise RuntimeError("Cloud API returned a non-object response")

        # Check for GraphQL errors without echoing tokens or request identifiers.
        errors = result.get("errors")
        if errors:
            if not isinstance(errors, list):
                raise RuntimeError("GraphQL errors: malformed error envelope")
            secrets = _string_values(variables)
            secrets.add(str(headers.get("Authorization", "")))
            account_getter = getattr(self.auth, "get_account_id", None)
            if callable(account_getter):
                try:
                    account_id = account_getter()
                except Exception:
                    account_id = None
                if isinstance(account_id, str) and account_id:
                    secrets.add(account_id)
            error_messages = [
                _redact_provider_message(
                    err.get("message", str(err)) if isinstance(err, Mapping) else err,
                    secrets,
                )
                for err in errors[:10]
            ]
            raise CloudAPIError(f"GraphQL errors: {'; '.join(error_messages)}")

        data = result.get("data")
        if not isinstance(data, Mapping):
            raise RuntimeError(
                "Cloud API response did not contain an object data field"
            )

        return dict(data)

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

    def submit_job(
        self,
        job_file_s3_key: str,
        *,
        job_name: Optional[str] = None,
        send_simulation_status_email: Optional[bool] = None,
        fresh: bool = False,
        retry: bool = False,
        project_name: Optional[str] = None,
        project_display_name: Optional[str] = None,
        simulation_name: Optional[str] = None,
        simulation_job_name: Optional[str] = None,
        execution_site_id: str = "managed-slurm",
        execution_resources: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit through the current managed execution-site contract."""
        if execution_site_id != "managed-slurm":
            raise ValueError("execution_site_id must be managed-slurm")
        mutation = """
            mutation SubmitJob(
                $jobFileS3Key: String!, $jobName: String, $sendSimulationStatusEmail: Boolean,
                $projectName: String, $projectDisplayName: String, $simulationName: String,
                $simulationJobName: String, $executionSiteId: String, $executionResources: AWSJSON,
                $forceRun: Boolean
            ) {
                submitJob(jobFileS3Key: $jobFileS3Key, jobName: $jobName,
                    sendSimulationStatusEmail: $sendSimulationStatusEmail,
                    projectName: $projectName, projectDisplayName: $projectDisplayName,
                    simulationName: $simulationName, simulationJobName: $simulationJobName,
                    executionSiteId: $executionSiteId, executionResources: $executionResources,
                    forceRun: $forceRun) {
                    simulationId status executionSiteId logicalAttemptId providerJobId executionState
                }
            }
        """
        if retry:
            # Ordinary submissions remain compatible with backends predating retries.
            mutation = mutation.replace(
                "$forceRun: Boolean", "$forceRun: Boolean, $retry: Boolean"
            ).replace(
                "forceRun: $forceRun)",
                "forceRun: $forceRun, retry: $retry)",
            )
        variables = {
            "jobFileS3Key": job_file_s3_key,
            "jobName": job_name,
            "sendSimulationStatusEmail": send_simulation_status_email,
            "projectName": project_name,
            "projectDisplayName": project_display_name,
            "simulationName": simulation_name,
            "simulationJobName": simulation_job_name,
            "executionSiteId": execution_site_id,
            "executionResources": (
                json.dumps(execution_resources)
                if execution_resources is not None
                else None
            ),
            "forceRun": fresh,
            "retry": retry if retry else None,
        }
        result = self.execute(
            mutation,
            {key: value for key, value in variables.items() if value is not None},
        )
        if not result.get("submitJob"):
            raise RuntimeError("Job submission failed: No response from API")
        return result["submitJob"]

    def cancel_simulation(self, simulation_id: str) -> None:
        """Request cancellation; terminal state remains authoritative in Cloud."""
        query = """
            mutation CancelSimulation($simulationId: String!) {
                cancelSimulation(simulationId: $simulationId) {
                    success
                    error
                }
            }
        """
        try:
            result = self.execute(query, {"simulationId": simulation_id})
        except CloudTransportError:
            raise CloudTransportError(
                "Cloud cancellation request failed; check connectivity and refresh "
                "the simulation status before retrying"
            ) from None
        except CloudAPIError:
            raise CloudAPIError(
                "Cloud rejected cancellation; check API compatibility, access, "
                "and the simulation status"
            ) from None
        except RuntimeError:
            raise RuntimeError(
                "Cloud cancellation request failed; refresh the simulation status "
                "and check the Cloud configuration"
            ) from None
        response = result.get("cancelSimulation") if isinstance(result, dict) else None
        if not isinstance(response, dict) or response.get("success") is not True:
            raise CloudAPIError(
                "Cloud did not accept cancellation; refresh the simulation status "
                "and retry once an execution attempt is assigned"
            )
        if response.get("error"):
            raise RuntimeError("Cloud returned an inconsistent cancellation response")

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
                    outputIdentity
                    failureCode
                    failureMessage
                    creditSettlementMode
                    creditSettlementStatus
                    creditSettlementOperationId
                    creditSettlementAmount
                    requestedResources
                    allocatedResources
                    executionSiteId
                    logicalAttemptId
                    providerJobId
                    executionState
                    failureReason
                }
            }
        """

        variables = {"id": simulation_id}

        try:
            result = self.execute(query, variables)
        except CloudTransportError as exc:
            if not exc.retryable:
                raise
            raise TransientStatusReadError(
                f"Could not read status for submitted simulation {simulation_id}. "
                "The simulation may still be running. Observe this existing run "
                "in Cloud or call wait() again on the retained run handle; "
                "do not resubmit the job."
            ) from None

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

        def resources(name: str) -> Optional[Dict[str, Any]]:
            value = details.get(name)
            if isinstance(value, str):
                value = json.loads(value)
            return value if isinstance(value, dict) else None

        normalized = {
            "requestedResources": resources("requestedResources"),
            "allocatedResources": resources("allocatedResources"),
            "executionSiteId": details.get("executionSiteId"),
            "logicalAttemptId": details.get("logicalAttemptId"),
            "providerJobId": details.get("providerJobId"),
            "executionState": normalize_execution_state(
                details.get("executionState") or status
            ),
            "failureReason": details.get("failureReason") or details.get("failureCode"),
        }
        return {
            **normalized,
            "id": details.get("id"),
            "status": status,
            "outputIdentity": details.get("outputIdentity"),
            "failureCode": details.get("failureCode"),
            "failureMessage": details.get("failureMessage"),
            **{
                key: details[key]
                for key in (
                    "creditSettlementMode",
                    "creditSettlementStatus",
                    "creditSettlementOperationId",
                    "creditSettlementAmount",
                )
                if details.get(key) is not None
            },
        }

    def get_simulation_status(self, simulation_id: str) -> str:
        """Get a simulation status string by ID."""

        return str(self.get_simulation_status_details(simulation_id)["status"])

    def deploy_storage_stack(self) -> Dict[str, Any]:
        """Deploy storage infrastructure stack.

        User ID is automatically extracted from the authentication context by AppSync.
        The backend derives the deployment environment from its runtime context.

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

    def wait_for_storage_ready(
        self,
        timeout: int = 1800,
        poll_interval: int = 15,
        expected_stack_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Wait for stack creation to complete by polling stack status.

        After triggering stack creation, waits 30 seconds before acting on status
        responses to avoid race conditions where AWS might return stale status
        (e.g., DELETE_COMPLETE from a previously deleted stack).

        Args:
            timeout: Maximum time to wait in seconds (default: 1800 = 30 minutes)
            poll_interval: Interval between status checks in seconds (default: 15)
            expected_stack_id: Optional stack ID to match (helps avoid stale status)

        Returns:
            Dict containing stack info (stackId, bucketName, status)

        Raises:
            RuntimeError: If stack creation fails, times out, or enters failed state
        """
        logger.debug(
            f"Waiting for storage stack to be ready (timeout: {timeout}s, poll interval: {poll_interval}s)..."
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
                    f"Timeout waiting for storage stack to be ready "
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

                stacks = result.get("listStacks", {}).get("items", [])

                if not stacks:
                    # Stack not found yet, continue polling
                    if last_status != "NOT_FOUND":
                        logger.debug(
                            "storage stack not found yet, continuing to poll..."
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
                            f"Matched storage stack by stackId: {expected_stack_id}, status: {status}"
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
                            "All storage stacks have UNKNOWN status, using most recent one"
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
                        f"Using most recent storage stack: {stack.get('stackId', 'unknown')} with status: {status}"
                    )

                # Log status changes
                if status != last_status:
                    logger.debug(f"storage stack status: {status}")
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
                    logger.debug(f"✓ storage stack is ready: {status}")
                    outputs = (
                        json.loads(stack["outputs"]) if stack.get("outputs") else {}
                    )
                    bucket_name = outputs.get("StorageBucketName", "")
                    return {
                        "stackId": stack.get("stackId", ""),
                        "bucketName": bucket_name,
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
                    error_msg = f"storage stack creation failed with status: {status}"
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
                    f"Unknown storage stack status '{status}'. Continuing to poll..."
                )
                time.sleep(poll_interval)

            except RuntimeError as e:
                # If it's a "not found" error, continue polling
                error_msg = str(e).lower()
                if "no active" in error_msg or "not found" in error_msg:
                    if last_status != "NOT_FOUND":
                        logger.debug(
                            "storage stack not found yet, continuing to poll..."
                        )
                        last_status = "NOT_FOUND"
                    time.sleep(poll_interval)
                    continue
                # Re-raise other RuntimeErrors
                raise
            except Exception as e:
                # For other errors, log and retry after interval
                logger.warning(
                    f"Error checking storage stack status: {e}. "
                    f"Retrying in {poll_interval} seconds..."
                )
                time.sleep(poll_interval)
