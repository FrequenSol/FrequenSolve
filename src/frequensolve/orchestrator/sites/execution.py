"""Small provider-neutral view shared by run handles and completed results."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ExecutionDetails:
    execution_site_id: str | None = None
    logical_attempt_id: str | None = None
    provider_job_id: str | None = None
    state: str | None = None
    failure_reason: str | None = None
    requested_resources: Mapping[str, Any] | None = None
    allocated_resources: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionDetails":
        return cls(
            value.get("executionSiteId"),
            value.get("logicalAttemptId"),
            value.get("providerJobId"),
            value.get("executionState"),
            value.get("failureReason"),
            value.get("requestedResources"),
            value.get("allocatedResources"),
        )


def normalize_execution_state(value: str | None) -> str:
    """Use the same execution vocabulary for submission and status reads."""
    return {
        "PENDING": "queued",
        "SUBMITTED": "queued",
        "QUEUED": "queued",
        "RUNNABLE": "queued",
        "STARTING": "queued",
        "RUNNING": "running",
        "SUCCEEDED": "succeeded",
        "COMPLETED": "succeeded",
        "FAILED": "failed",
        "CANCELED": "canceled",
        "CANCELLED": "canceled",
        "ABORTED": "canceled",
    }.get((value or "").upper(), "queued")
