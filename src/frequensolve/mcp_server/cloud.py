"""Fixed, user-self-scoped reads from FrequenSol Cloud.

This module intentionally imports only the Python standard library and
dependency-light FrequenSolve configuration helpers at import time. Cloud
dependencies are loaded only when an authenticated operation is executed.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import ipaddress
import json
import math
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from frequensolve.mcp_server._contracts import (
    CLOUD_READ_CONTRACT_ID,
    CLOUD_READ_CONTRACT_VERSION,
)
from frequensolve.orchestrator.sites.config_file import site_config_path
from frequensolve.storage import frequensolve_home

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import toml as tomllib

__all__ = [
    "CLOUD_READ_CONTRACT_ID",
    "CLOUD_READ_CONTRACT_VERSION",
    "CLOUD_READ_OPERATIONS",
    "CONTRACT_ID",
    "CONTRACT_VERSION",
    "CloudReadClient",
    "CloudReadError",
    "execute_cloud_operation",
    "validate_cloud_operation_output",
]

CONTRACT_ID = CLOUD_READ_CONTRACT_ID
CONTRACT_VERSION = CLOUD_READ_CONTRACT_VERSION
CLOUD_READ_CATALOG_SHA256 = (
    "d4290a58887aae079e13f008579e9c06d727f5a6167897e68cba615dc4e166f2"
)
CLOUD_READ_OPERATIONS = (
    "getCloudReadiness",
    "listMySimulations",
    "getMySimulation",
    "listMySimulationDiagnostics",
    "listMySimulationResultArtifacts",
)

_CONTRACT_ERRORS = {
    "AUTHENTICATION_REQUIRED": (
        "Sign in to FrequenSol Cloud to use this operation.",
        False,
    ),
    "INVALID_INPUT": (
        "The request did not match the supported Cloud read contract.",
        False,
    ),
    "NOT_FOUND_OR_ACCESS_DENIED": (
        "The requested simulation was not found or is not available to this user.",
        False,
    ),
    "RESULTS_NOT_AVAILABLE": (
        "Safe result metadata is not available for this simulation.",
        False,
    ),
    "RESPONSE_LIMIT_EXCEEDED": (
        "The bounded response limit was reached. Request a smaller page.",
        True,
    ),
    "UPSTREAM_UNAVAILABLE": (
        "FrequenSol Cloud could not complete the read request.",
        True,
    ),
}
_LOCAL_ERRORS = {
    "CLOUD_SUPPORT_REQUIRED": (
        "Cloud support is not installed.",
        False,
    ),
    "CLOUD_CONFIGURATION_REQUIRED": (
        "The selected FrequenSolve Cloud profile is not configured.",
        False,
    ),
}
_ERRORS = {**_CONTRACT_ERRORS, **_LOCAL_ERRORS}

_REMEDIATIONS = {
    "cloud-extra-missing": (
        "Install `frequensolve[cloud]`, then sign in through the selected "
        "FrequenSolve Cloud site profile."
    ),
    "profile-unavailable": (
        "Select an existing AWS Cloud profile in the FrequenSolve site config."
    ),
    "configuration-unavailable": (
        "Initialize the selected Cloud profile once so its public configuration "
        "is cached locally."
    ),
    "login-required": (
        "Sign in through the selected FrequenSolve Cloud site profile, then retry."
    ),
    "identity-invalid": (
        "Sign in again through the selected FrequenSolve Cloud site profile."
    ),
    "input-invalid": "Use only the published fields and bounds for this operation.",
    "response-too-large": "Request a smaller page and retry.",
    "upstream-unavailable": "Retry later without changing the requested identifier.",
}
_DEFAULT_REASONS = {
    "CLOUD_SUPPORT_REQUIRED": "cloud-extra-missing",
    "CLOUD_CONFIGURATION_REQUIRED": "profile-unavailable",
    "AUTHENTICATION_REQUIRED": "login-required",
    "INVALID_INPUT": "input-invalid",
    "NOT_FOUND_OR_ACCESS_DENIED": "upstream-unavailable",
    "RESULTS_NOT_AVAILABLE": "upstream-unavailable",
    "RESPONSE_LIMIT_EXCEEDED": "response-too-large",
    "UPSTREAM_UNAVAILABLE": "upstream-unavailable",
}

_PROFILE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_DOMAIN_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_REGION_PATTERN = re.compile(r"^[a-z]{2}(?:-gov)?-[a-z]+-\d$")
_USER_POOL_SUFFIX_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SUB_PATTERN = re.compile(r"^[A-Za-z0-9:_-]{1,128}$")
_RELATIVE_RESULT_PATTERN = re.compile(
    r"^(?![/\\])(?![A-Za-z][A-Za-z0-9+.-]*:)"
    r"(?!.*(?:^|/)\.\.?(?:/|$))[^\x00-\x1f\x7f\\]+$"
)
_CONTROL_PATTERN = re.compile(r"[\x00-\x1f\x7f]")
_DATE_TIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}" r"(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})$"
)
_COGNITO_TIMEOUT_SECONDS = 10.0
_COGNITO_MAX_RESPONSE_BYTES = 32_768
_SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    re.compile(r"(?i)\barn:(?:aws|aws-us-gov|aws-cn):[^\s]+"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"),
    re.compile(r"(?i)\bbearer(?:%20|\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)(?:^|[^a-z0-9])(?:access[_-]?token|refresh[_-]?token|"
        r"id[_-]?token|token|auth(?:orization)?|api[_-]?key|secret|"
        r"credential|password)s?\s*(?:=|:|/)\s*[^\s&/]{4,}"
    ),
    re.compile(
        r"(?i)\b(?:github_pat_[a-z0-9_]{20,}|gh[pousr]_[a-z0-9]{20,}|"
        r"xox[baprs]-[a-z0-9-]{10,}|npm_[a-z0-9]{30,}|"
        r"AIza[a-z0-9_-]{30,}|sk_(?:live|test)_[a-z0-9]{16,}|"
        r"sk-(?:proj|svcacct)-[a-z0-9_-]{8,})\b"
    ),
    re.compile(r"(?i)-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?i)\b(?:https?|s3|file|ssh)://[^\s]+"),
    re.compile(
        r"(?i)\b(?:[a-z0-9][a-z0-9.-]{0,62}\.)?"
        r"s3[a-z0-9.-]{0,80}\.amazonaws\.com(?:\.cn)?\b"
    ),
    re.compile(
        r"(?i)\bfrequensol-[a-z0-9][a-z0-9.-]{0,49}-" r"(?:dev|staging|prod)-storage\b"
    ),
    re.compile(r"(?:^|\s)/(?:[^\s/]+/)+[^\s]*"),
    re.compile(r"\b[A-Za-z]:\\[^\s]+"),
)
_OBJECT_KEY_TEXT_PATTERN = re.compile(r"(?:^|\s)[^\s/]+(?:/[^\s/]+)+")
_OPAQUE_OUTPUT_KEYS = {"cursor"}
_HIGH_ENTROPY_EXEMPT_KEYS = {"simulationId"}


class CloudReadError(RuntimeError):
    """Stable, allowlisted Cloud read failure.

    Underlying provider messages, URLs, identifiers, and credentials are never
    retained on this exception.
    """

    def __init__(
        self,
        code: str,
        message_or_reason: str | None = None,
        retryable: bool | None = None,
    ):
        if code not in _ERRORS:
            code = "UPSTREAM_UNAVAILABLE"
        reason = (
            message_or_reason
            if message_or_reason in _REMEDIATIONS
            else _DEFAULT_REASONS[code]
        )
        self.code = code
        self.safe_message, self.retryable = _ERRORS[code]
        self.reason = reason
        self.remediation = _REMEDIATIONS[reason]
        super().__init__(code, reason)

    def __str__(self) -> str:
        return f"{self.code}: {self.safe_message}"

    def __repr__(self) -> str:
        return f"CloudReadError(code={self.code!r}, reason={self.reason!r})"

    def __reduce__(self) -> tuple[type[CloudReadError], tuple[str, str]]:
        return type(self), (self.code, self.reason)


class _Transport(Protocol):
    def __call__(
        self,
        document: str,
        variables: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _SiteProfile:
    name: str
    domain: str


@dataclass(frozen=True, repr=False)
class _CloudConfig:
    region: str
    user_pool_id: str
    client_id: str
    graphql_url: str


class CloudReadClient:
    """Execute only the five fixed customer-self Cloud read operations."""

    def __init__(
        self,
        profile: str | None = None,
        *,
        _transport_factory: Callable[[str | None], _Transport] | None = None,
    ):
        if profile is not None:
            _validate_profile_name(profile)
        self.profile = profile
        self._transport_factory = _transport_factory

    def check_readiness(self) -> dict[str, Any]:
        """Return the signed-in user's coarse Cloud readiness."""

        return self._execute("getCloudReadiness", {})

    def list_simulations(
        self,
        *,
        status: str | None = None,
        project_name: str | None = None,
        limit: int = 10,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Return one bounded page of simulations owned by the signed-in user."""

        arguments: dict[str, Any] = {"limit": limit}
        if status is not None:
            arguments["status"] = status
        if project_name is not None:
            arguments["projectName"] = project_name
        if cursor is not None:
            arguments["cursor"] = cursor
        return self._execute("listMySimulations", arguments)

    def get_simulation(self, *, simulation_id: str) -> dict[str, Any]:
        """Return safe status and progress for one owned simulation."""

        return self._execute(
            "getMySimulation",
            {"simulationId": simulation_id},
        )

    def list_diagnostics(
        self,
        *,
        simulation_id: str,
        limit: int = 25,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Return bounded stored diagnostics beneath one owned simulation."""

        arguments: dict[str, Any] = {
            "simulationId": simulation_id,
            "limit": limit,
        }
        if cursor is not None:
            arguments["cursor"] = cursor
        return self._execute("listMySimulationDiagnostics", arguments)

    def list_result_artifacts(
        self,
        *,
        simulation_id: str,
        limit: int = 25,
        after: str | None = None,
    ) -> dict[str, Any]:
        """Return safe relative result metadata without reading object contents."""

        arguments: dict[str, Any] = {
            "simulationId": simulation_id,
            "limit": limit,
        }
        if after is not None:
            arguments["after"] = after
        return self._execute("listMySimulationResultArtifacts", arguments)

    def _execute(
        self,
        operation: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        contract = _load_contract()
        operation_contract = contract["operationsByName"].get(operation)
        if operation_contract is None:
            raise CloudReadError("INVALID_INPUT", "input-invalid")

        try:
            normalized_arguments = _prune_optional_nulls(
                arguments,
                operation_contract["inputSchema"],
            )
            validated_arguments = _validate_value(
                normalized_arguments,
                operation_contract["inputSchema"],
                path="input",
            )
            variables = _build_variables(operation_contract, validated_arguments)
        except CloudReadError:
            raise
        except Exception:
            raise CloudReadError("INVALID_INPUT", "input-invalid") from None

        try:
            if self._transport_factory is None:
                transport = _build_authenticated_transport(self.profile)
            else:
                transport = self._transport_factory(self.profile)
            response = transport(
                operation_contract["source"]["document"],
                variables,
                operation_contract["limits"]["timeoutMs"] / 1000.0,
                operation_contract["limits"]["maxResponseBytes"],
            )
        except CloudReadError:
            raise
        except Exception:
            raise CloudReadError(
                "UPSTREAM_UNAVAILABLE",
                "upstream-unavailable",
            ) from None

        return _project_and_validate_response(operation_contract, response)


def execute_cloud_operation(
    profile: str | None,
    operation: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Picklable finite dispatch entry point for a bounded worker process."""

    if operation not in CLOUD_READ_OPERATIONS or not isinstance(arguments, dict):
        raise CloudReadError("INVALID_INPUT", "input-invalid")
    return CloudReadClient(profile)._execute(operation, arguments)


def validate_cloud_operation_output(
    operation: str,
    arguments: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one host-executed result against the packaged fixed contract."""

    try:
        if operation not in CLOUD_READ_OPERATIONS or not isinstance(arguments, dict):
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        selected = _load_contract()["operationsByName"][operation]
        normalized_arguments = dict(arguments)
        _validate_value(
            normalized_arguments,
            selected["inputSchema"],
            path="input",
        )
    except CloudReadError as error:
        if error.code == "INVALID_INPUT":
            raise
        raise CloudReadError(
            "UPSTREAM_UNAVAILABLE",
            "upstream-unavailable",
        ) from None
    except Exception:
        raise CloudReadError("INVALID_INPUT", "input-invalid") from None

    try:
        pruned = _prune_optional_nulls(dict(output), selected["outputSchema"])
        validated = _validate_value(
            pruned,
            selected["outputSchema"],
            path="output",
        )
        if not isinstance(validated, dict):
            raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
        _reject_forbidden_output(selected, validated)
        encoded = json.dumps(
            validated,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > selected["limits"]["maxResponseBytes"]:
            raise CloudReadError("RESPONSE_LIMIT_EXCEEDED", "response-too-large")
        return validated
    except CloudReadError as error:
        if error.code == "RESPONSE_LIMIT_EXCEEDED":
            raise
        raise CloudReadError(
            "UPSTREAM_UNAVAILABLE",
            "upstream-unavailable",
        ) from None
    except Exception:
        raise CloudReadError(
            "UPSTREAM_UNAVAILABLE",
            "upstream-unavailable",
        ) from None


@lru_cache(maxsize=1)
def _load_contract() -> dict[str, Any]:
    try:
        resource = files("frequensolve.mcp_server").joinpath(
            "contracts/customer_cloud_read_v1.json"
        )
        raw = resource.read_bytes()
        if hashlib.sha256(raw).hexdigest() != CLOUD_READ_CATALOG_SHA256:
            raise ValueError
        catalog = _strict_json_loads(raw)
    except Exception:
        raise CloudReadError(
            "UPSTREAM_UNAVAILABLE",
            "upstream-unavailable",
        ) from None
    if not isinstance(catalog, dict):
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
    if (
        catalog.get("contractId") != CLOUD_READ_CONTRACT_ID
        or catalog.get("contractVersion") != CLOUD_READ_CONTRACT_VERSION
        or catalog.get("classification") != "read-only"
        or catalog.get("audience") != "customer-self"
    ):
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")

    catalog_errors = {
        item.get("code"): (item.get("safeMessage"), item.get("retryable"))
        for item in catalog.get("errors", [])
        if isinstance(item, dict)
    }
    if catalog_errors != _CONTRACT_ERRORS:
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")

    operations = catalog.get("operations")
    if not isinstance(operations, list):
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
    operations_by_name: dict[str, dict[str, Any]] = {}
    for operation in operations:
        if not isinstance(operation, dict):
            raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
        name = operation.get("name")
        source = operation.get("source")
        document = source.get("document") if isinstance(source, dict) else None
        if (
            name not in CLOUD_READ_OPERATIONS
            or name in operations_by_name
            or operation.get("classification") != "read-only"
            or not isinstance(source, dict)
            or source.get("kind") != "fixed-graphql-query"
            or not isinstance(document, str)
            or not document.lstrip().startswith("query ")
            or re.search(r"\b(?:mutation|subscription)\b", document, re.IGNORECASE)
        ):
            raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
        operations_by_name[name] = operation
    if tuple(operations_by_name) != CLOUD_READ_OPERATIONS:
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")

    return {
        "catalog": catalog,
        "operationsByName": operations_by_name,
    }


def _strict_json_loads(raw: bytes | str) -> Any:
    def reject_constant(_: str) -> None:
        raise ValueError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = raw
    return json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _prune_optional_nulls(
    value: Any,
    schema: Mapping[str, Any],
) -> Any:
    expected = schema.get("type")
    if expected == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return value
        return [_prune_optional_nulls(item, item_schema) for item in value]
    if expected != "object" or not isinstance(value, dict):
        return value
    properties = schema.get("properties")
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return value
    required_names = set(required)
    result: dict[str, Any] = {}
    for key, item in value.items():
        if item is None and key not in required_names and key in properties:
            continue
        property_schema = properties.get(key)
        result[key] = (
            _prune_optional_nulls(item, property_schema)
            if isinstance(property_schema, dict)
            else item
        )
    return result


def _validate_value(value: Any, schema: Mapping[str, Any], *, path: str) -> Any:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
        if any(key not in value for key in required):
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        if schema.get("additionalProperties") is False:
            if any(key not in properties for key in value):
                raise CloudReadError("INVALID_INPUT", "input-invalid")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key not in properties:
                raise CloudReadError("INVALID_INPUT", "input-invalid")
            result[key] = _validate_value(
                item,
                properties[key],
                path=f"{path}.{key}",
            )
        return result

    if expected == "array":
        if not isinstance(value, list):
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        maximum = schema.get("maxItems")
        if isinstance(maximum, int) and len(value) > maximum:
            raise CloudReadError("RESPONSE_LIMIT_EXCEEDED", "response-too-large")
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
        return [
            _validate_value(item, item_schema, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]

    if expected == "string":
        if not isinstance(value, str):
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        if path.startswith("output") and _CONTROL_PATTERN.search(value):
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        if isinstance(maximum, int) and len(value) > maximum:
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        allowed = schema.get("enum")
        if isinstance(allowed, list) and value not in allowed:
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        if schema.get("format") == "date-time" and not _is_date_time(value):
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        if path.rsplit(".", 1)[-1] in {"relativePath", "nextAfter", "after"}:
            if _RELATIVE_RESULT_PATTERN.fullmatch(value) is None:
                raise CloudReadError("INVALID_INPUT", "input-invalid")
        return value

    if expected == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        _validate_numeric_bounds(value, schema)
        return value

    if expected == "number":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        _validate_numeric_bounds(value, schema)
        return value

    if expected == "boolean":
        if not isinstance(value, bool):
            raise CloudReadError("INVALID_INPUT", "input-invalid")
        return value

    raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")


def _validate_numeric_bounds(value: int | float, schema: Mapping[str, Any]) -> None:
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if isinstance(minimum, (int, float)) and value < minimum:
        raise CloudReadError("INVALID_INPUT", "input-invalid")
    if isinstance(maximum, (int, float)) and value > maximum:
        raise CloudReadError("INVALID_INPUT", "input-invalid")
    allowed = schema.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        raise CloudReadError("INVALID_INPUT", "input-invalid")


def _is_date_time(value: str) -> bool:
    if _DATE_TIME_PATTERN.fullmatch(value) is None:
        return False
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        return parsed.tzinfo is not None
    except (TypeError, ValueError):
        return False


def _build_variables(
    operation: Mapping[str, Any],
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    source = operation.get("source")
    variable_contracts = source.get("variables") if isinstance(source, dict) else None
    if not isinstance(variable_contracts, dict):
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
    variables: dict[str, Any] = {}
    for name, variable_contract in variable_contracts.items():
        if (
            not isinstance(name, str)
            or not isinstance(variable_contract, dict)
            or not isinstance(variable_contract.get("source"), str)
            or not variable_contract["source"].startswith("input.")
        ):
            raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
        input_name = variable_contract["source"][len("input.") :]
        if input_name in arguments:
            variables[name] = arguments[input_name]
        elif "default" in variable_contract:
            variables[name] = variable_contract["default"]
    return variables


def _project_and_validate_response(
    operation: Mapping[str, Any],
    response: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
    errors = response.get("errors")
    if errors:
        raise _normalized_graphql_error(operation, errors)
    if "data" not in response or not isinstance(response["data"], dict):
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")

    source = operation["source"]
    projection = source.get("responseProjection")
    path = projection.get("path") if isinstance(projection, dict) else None
    if (
        not isinstance(path, list)
        or any(not isinstance(item, str) for item in path)
        or projection.get("source") != "graphql-data"
    ):
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
    value: Any = response["data"]
    for item in path:
        if not isinstance(value, dict) or item not in value:
            raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
        value = value[item]
    if value is None:
        name = operation.get("name")
        if name in {"getMySimulation", "listMySimulationDiagnostics"}:
            raise CloudReadError(
                "NOT_FOUND_OR_ACCESS_DENIED",
                "upstream-unavailable",
            )
        if name == "listMySimulationResultArtifacts":
            raise CloudReadError("RESULTS_NOT_AVAILABLE", "upstream-unavailable")
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")

    try:
        pruned_value = _prune_optional_nulls(value, operation["outputSchema"])
        validated = _validate_value(
            pruned_value,
            operation["outputSchema"],
            path="output",
        )
    except CloudReadError as error:
        if error.code == "RESPONSE_LIMIT_EXCEEDED":
            raise
        raise CloudReadError(
            "UPSTREAM_UNAVAILABLE",
            "upstream-unavailable",
        ) from None
    if not isinstance(validated, dict):
        raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
    _reject_forbidden_output(operation, validated)
    encoded = json.dumps(
        validated,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > operation["limits"]["maxResponseBytes"]:
        raise CloudReadError("RESPONSE_LIMIT_EXCEEDED", "response-too-large")
    return validated


def _normalized_graphql_error(
    operation: Mapping[str, Any],
    errors: Any,
) -> CloudReadError:
    allowed = set(operation.get("errors", []))
    if isinstance(errors, list):
        for item in errors[:10]:
            if not isinstance(item, dict):
                continue
            candidates = [item.get("message"), item.get("errorType")]
            extensions = item.get("extensions")
            if isinstance(extensions, dict):
                candidates.extend([extensions.get("code"), extensions.get("errorType")])
            for candidate in candidates:
                if not isinstance(candidate, str) or len(candidate) > 4096:
                    continue
                for code in _ERRORS:
                    if code in allowed and re.search(
                        rf"(?<![A-Z0-9_]){re.escape(code)}(?![A-Z0-9_])",
                        candidate,
                    ):
                        reason = (
                            "input-invalid"
                            if code == "INVALID_INPUT"
                            else (
                                "response-too-large"
                                if code == "RESPONSE_LIMIT_EXCEEDED"
                                else (
                                    "login-required"
                                    if code == "AUTHENTICATION_REQUIRED"
                                    else "upstream-unavailable"
                                )
                            )
                        )
                        return CloudReadError(code, reason)
                if candidate in {
                    "Unauthorized",
                    "UnauthorizedException",
                    "Not Authorized",
                }:
                    return CloudReadError(
                        "AUTHENTICATION_REQUIRED",
                        "login-required",
                    )
    return CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")


def _reject_forbidden_output(
    operation: Mapping[str, Any],
    value: Mapping[str, Any],
) -> None:
    contract = _load_contract()["catalog"]
    forbidden = {
        _normalized_key(item) for item in contract["globalRedaction"]["neverReturn"]
    }
    forbidden.update(
        _normalized_key(item)
        for item in operation.get("redaction", {}).get("neverReturn", [])
    )

    def visit(item: Any, *, key: str = "") -> None:
        if isinstance(item, dict):
            for nested_key, nested_value in item.items():
                if _normalized_key(nested_key) in forbidden:
                    raise CloudReadError(
                        "UPSTREAM_UNAVAILABLE",
                        "upstream-unavailable",
                    )
                visit(nested_value, key=nested_key)
        elif isinstance(item, list):
            for nested_value in item:
                visit(nested_value, key=key)
        elif isinstance(item, str):
            if _contains_sensitive_output_text(item, key=key):
                raise CloudReadError(
                    "UPSTREAM_UNAVAILABLE",
                    "upstream-unavailable",
                )

    visit(value)


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _contains_sensitive_output_text(value: str, *, key: str) -> bool:
    variants = _decoded_text_variants(value)
    for candidate in variants:
        if any(pattern.search(candidate) for pattern in _SENSITIVE_TEXT_PATTERNS):
            return True
        if key not in {
            "relativePath",
            "nextAfter",
            *_OPAQUE_OUTPUT_KEYS,
            *_HIGH_ENTROPY_EXEMPT_KEYS,
        }:
            if _OBJECT_KEY_TEXT_PATTERN.search(candidate):
                return True
        if key not in {
            *_OPAQUE_OUTPUT_KEYS,
            *_HIGH_ENTROPY_EXEMPT_KEYS,
        } and _has_high_entropy_shape(candidate):
            return True
    return False


def _decoded_text_variants(value: str) -> tuple[str, ...]:
    variants = [value]
    current = value
    for _ in range(3):
        try:
            decoded = re.sub(
                r"(?i)&#(?:64|x40);|&commat;",
                "@",
                unquote(current),
            )
        except Exception:
            break
        if decoded == current:
            break
        variants.append(decoded)
        current = decoded
    return tuple(variants)


def _has_high_entropy_shape(value: str) -> bool:
    return any(
        len(candidate) >= 32
        and re.search(r"[a-z]", candidate) is not None
        and re.search(r"[A-Z]", candidate) is not None
        and re.search(r"\d", candidate) is not None
        for candidate in re.split(r"[^A-Za-z0-9_+=.-]+", value)
    )


def _validate_profile_name(profile: str) -> str:
    if not isinstance(profile, str) or _PROFILE_PATTERN.fullmatch(profile) is None:
        raise CloudReadError("INVALID_INPUT", "input-invalid")
    return profile


def _load_site_profile(profile: str | None) -> _SiteProfile:
    path = site_config_path()
    try:
        raw = _read_bounded_regular_file(
            path,
            max_bytes=131_072,
            integrity_protected=True,
        )
        document = tomllib.loads(raw.decode("utf-8"))
    except Exception:
        raise CloudReadError(
            "CLOUD_CONFIGURATION_REQUIRED",
            "profile-unavailable",
        ) from None
    if not isinstance(document, dict):
        raise CloudReadError("CLOUD_CONFIGURATION_REQUIRED", "profile-unavailable")

    selected = profile if profile is not None else document.get("default")
    if not isinstance(selected, str):
        raise CloudReadError("CLOUD_CONFIGURATION_REQUIRED", "profile-unavailable")
    try:
        selected = _validate_profile_name(selected)
    except CloudReadError:
        raise CloudReadError(
            "CLOUD_CONFIGURATION_REQUIRED",
            "profile-unavailable",
        ) from None

    sites = document.get("sites")
    site = sites.get(selected) if isinstance(sites, dict) else None
    if not isinstance(site, dict):
        raise CloudReadError("CLOUD_CONFIGURATION_REQUIRED", "profile-unavailable")
    site_type = site.get("type")
    if not isinstance(site_type, str) or site_type.casefold() not in {
        "aws",
        "awssite",
        "cloud",
    }:
        raise CloudReadError("CLOUD_CONFIGURATION_REQUIRED", "profile-unavailable")
    domain = site.get("domain")
    if not isinstance(domain, str):
        raise CloudReadError("CLOUD_CONFIGURATION_REQUIRED", "profile-unavailable")
    try:
        normalized_domain = _validate_domain(domain)
    except ValueError:
        raise CloudReadError(
            "CLOUD_CONFIGURATION_REQUIRED",
            "profile-unavailable",
        ) from None
    return _SiteProfile(name=selected, domain=normalized_domain)


def _validate_domain(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 253
        or value != value.strip()
        or any(character in value for character in "/\\@?#")
    ):
        raise ValueError("unsafe domain")
    hostname, separator, port_text = value.casefold().partition(":")
    if separator:
        if not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
            raise ValueError("unsafe port")
    if hostname == "localhost":
        if not separator:
            raise ValueError("localhost requires an explicit port")
    else:
        labels = hostname.split(".")
        if len(labels) < 2 or any(
            _DOMAIN_LABEL_PATTERN.fullmatch(label) is None for label in labels
        ):
            raise ValueError("unsafe hostname")
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            pass
        else:
            raise ValueError("IP literals are not Cloud domains")
    return hostname + (f":{port_text}" if separator else "")


def _load_cached_cloud_config(profile: _SiteProfile) -> _CloudConfig:
    cache_name = profile.domain.replace(":", "_")
    path = frequensolve_home() / "cloud" / f"config_{cache_name}.json"
    try:
        document = _strict_json_loads(
            _read_bounded_regular_file(
                path,
                max_bytes=65_536,
                integrity_protected=True,
            )
        )
    except Exception:
        raise CloudReadError(
            "CLOUD_CONFIGURATION_REQUIRED",
            "configuration-unavailable",
        ) from None
    try:
        region = document["region"]
        auth = document["auth"]
        api = document["api"]
        user_pool_id = auth["userPoolId"]
        client_id = auth["clientId"]
        graphql_url = api["graphqlUrl"]
    except (KeyError, TypeError):
        raise CloudReadError(
            "CLOUD_CONFIGURATION_REQUIRED",
            "configuration-unavailable",
        ) from None
    if (
        not isinstance(region, str)
        or _REGION_PATTERN.fullmatch(region) is None
        or not isinstance(user_pool_id, str)
        or not user_pool_id.startswith(f"{region}_")
        or _USER_POOL_SUFFIX_PATTERN.fullmatch(user_pool_id[len(region) + 1 :]) is None
        or not isinstance(client_id, str)
        or _CLIENT_ID_PATTERN.fullmatch(client_id) is None
        or not isinstance(graphql_url, str)
        or not _is_safe_appsync_url(graphql_url, region)
    ):
        raise CloudReadError(
            "CLOUD_CONFIGURATION_REQUIRED",
            "configuration-unavailable",
        )
    return _CloudConfig(
        region=region,
        user_pool_id=user_pool_id,
        client_id=client_id,
        graphql_url=graphql_url,
    )


def _is_safe_appsync_url(value: str, region: str) -> bool:
    if len(value) > 512:
        return False
    try:
        parsed = urlsplit(value)
        if parsed.port is not None:
            return False
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/graphql"
        or parsed.query
        or parsed.fragment
    ):
        return False
    suffixes = (
        f".appsync-api.{region}.amazonaws.com",
        f".appsync-api.{region}.amazonaws.com.cn",
    )
    return any(parsed.hostname.endswith(suffix) for suffix in suffixes)


def _aws_dns_suffix(region: str) -> str:
    return "amazonaws.com.cn" if region.startswith("cn-") else "amazonaws.com"


def _cognito_endpoint(region: str) -> str:
    return f"https://cognito-idp.{region}.{_aws_dns_suffix(region)}"


def _cognito_issuer(config: _CloudConfig) -> str:
    return f"{_cognito_endpoint(config.region)}/{config.user_pool_id}"


def _build_authenticated_transport(profile_name: str | None) -> _Transport:
    profile = _load_site_profile(profile_name)
    config = _load_cached_cloud_config(profile)
    try:
        requests = importlib.import_module("requests")
    except (ImportError, ModuleNotFoundError, AttributeError):
        raise CloudReadError(
            "CLOUD_SUPPORT_REQUIRED",
            "cloud-extra-missing",
        ) from None

    try:
        tokens, id_claims = _select_cached_identity(profile.name, config=config)
        user = _get_verified_cognito_user(
            requests,
            config=config,
            access_token=tokens["access_token"],
        )
        verified_sub = _get_user_sub(user)
        if verified_sub != id_claims["sub"]:
            raise ValueError
    except CloudReadError:
        raise
    except Exception:
        raise CloudReadError(
            "AUTHENTICATION_REQUIRED",
            "identity-invalid",
        ) from None

    session = _new_requests_session(requests)
    return _RequestsTransport(
        session=session,
        graphql_url=config.graphql_url,
        id_token=tokens["id_token"],
    )


def _new_requests_session(requests: Any) -> Any:
    try:
        session = requests.Session()
        session.trust_env = False
    except Exception:
        raise CloudReadError(
            "CLOUD_SUPPORT_REQUIRED",
            "cloud-extra-missing",
        ) from None
    return session


def _get_verified_cognito_user(
    requests: Any,
    *,
    config: _CloudConfig,
    access_token: str,
) -> Mapping[str, Any]:
    session = _new_requests_session(requests)
    response = None
    try:
        response = session.post(
            _cognito_endpoint(config.region),
            headers={
                "Content-Type": "application/x-amz-json-1.1",
                "X-Amz-Target": "AWSCognitoIdentityProviderService.GetUser",
            },
            json={"AccessToken": access_token},
            timeout=_COGNITO_TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=False,
        )
        status_code = getattr(response, "status_code", None)
        if status_code in {400, 401, 403}:
            raise CloudReadError("AUTHENTICATION_REQUIRED", "identity-invalid")
        if (
            isinstance(status_code, bool)
            or not isinstance(status_code, int)
            or not 200 <= status_code < 300
        ):
            raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
        document = _read_bounded_http_json(
            response,
            max_bytes=_COGNITO_MAX_RESPONSE_BYTES,
            oversized_code="UPSTREAM_UNAVAILABLE",
        )
        if not isinstance(document, dict):
            raise CloudReadError("UPSTREAM_UNAVAILABLE", "upstream-unavailable")
        return document
    except CloudReadError:
        raise
    except Exception:
        raise CloudReadError(
            "UPSTREAM_UNAVAILABLE",
            "upstream-unavailable",
        ) from None
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        try:
            session.close()
        except Exception:
            pass


def _credential_paths(profile_name: str) -> tuple[Path, ...]:
    root = frequensolve_home()
    return (
        root / "cloud" / "credentials",
        # Earlier development builds could write these two cache layouts.
        # They remain read-only compatibility fallbacks when the canonical
        # package-owned cache above does not exist.
        root / "cloud" / f"credentials_{profile_name}",
        root / "credentials",
    )


def _select_cached_identity(
    profile_name: str,
    *,
    config: _CloudConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    for path in _credential_paths(profile_name):
        tokens = _read_token_cache(path, missing_ok=True)
        if tokens is None:
            continue
        try:
            id_claims = _decode_and_validate_token(
                tokens.get("id_token"),
                token_use="id",
                config=config,
            )
            access_claims = _decode_and_validate_token(
                tokens.get("access_token"),
                token_use="access",
                config=config,
            )
            if id_claims["sub"] != access_claims["sub"]:
                raise ValueError
        except ValueError:
            raise CloudReadError(
                "AUTHENTICATION_REQUIRED",
                "identity-invalid",
            ) from None
        if _token_needs_refresh(id_claims) or _token_needs_refresh(access_claims):
            raise CloudReadError("AUTHENTICATION_REQUIRED", "login-required")
        return tokens, id_claims
    raise CloudReadError("AUTHENTICATION_REQUIRED", "login-required")


def _read_token_cache(
    path: Path,
    *,
    missing_ok: bool = False,
) -> dict[str, Any] | None:
    try:
        raw = _read_bounded_regular_file(
            path,
            max_bytes=65_536,
            private=True,
        )
    except FileNotFoundError:
        if missing_ok:
            return None
        raise CloudReadError("AUTHENTICATION_REQUIRED", "identity-invalid") from None
    except Exception:
        raise CloudReadError("AUTHENTICATION_REQUIRED", "identity-invalid") from None
    try:
        document = _strict_json_loads(raw)
    except Exception:
        raise CloudReadError("AUTHENTICATION_REQUIRED", "identity-invalid") from None
    if not isinstance(document, dict):
        raise CloudReadError("AUTHENTICATION_REQUIRED", "identity-invalid")
    allowed = {
        "email",
        "id_token",
        "access_token",
        "refresh_token",
        "expires_at",
    }
    if any(not isinstance(key, str) or key not in allowed for key in document):
        raise CloudReadError("AUTHENTICATION_REQUIRED", "identity-invalid")
    for required in ("id_token", "access_token"):
        value = document.get(required)
        if not isinstance(value, str) or not 1 <= len(value) <= 16_384:
            raise CloudReadError("AUTHENTICATION_REQUIRED", "identity-invalid")
    refresh = document.get("refresh_token")
    if refresh is not None and (
        not isinstance(refresh, str) or not 1 <= len(refresh) <= 16_384
    ):
        raise CloudReadError("AUTHENTICATION_REQUIRED", "identity-invalid")
    return {
        "id_token": document["id_token"],
        "access_token": document["access_token"],
    }


def _read_bounded_regular_file(
    path: Path,
    *,
    max_bytes: int,
    private: bool = False,
    integrity_protected: bool = False,
) -> bytes:
    if not hasattr(os, "O_NOFOLLOW"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OSError("symlinked files are not accepted")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > max_bytes
        ):
            raise OSError("file is outside the bounded regular-file policy")
        if os.name == "posix" and (private or integrity_protected):
            if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
                raise OSError("trusted cache has a different owner")
            if private and metadata.st_mode & 0o077:
                raise OSError("private cache permissions are too broad")
            if integrity_protected and metadata.st_mode & 0o022:
                raise OSError("trusted cache permissions allow external writes")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if not raw or len(raw) > max_bytes:
            raise OSError("file exceeds the bounded read policy")
        return raw
    finally:
        os.close(descriptor)


def _decode_and_validate_token(
    token: Any,
    *,
    token_use: str,
    config: _CloudConfig,
) -> dict[str, Any]:
    if not isinstance(token, str) or len(token) > 16_384:
        raise ValueError
    parts = token.split(".")
    if len(parts) != 3 or any(not part for part in parts):
        raise ValueError
    header = _decode_jwt_part(parts[0])
    claims = _decode_jwt_part(parts[1])
    if (
        header.get("alg") != "RS256"
        or not isinstance(header.get("kid"), str)
        or not 1 <= len(header["kid"]) <= 256
        or claims.get("token_use") != token_use
        or claims.get("iss") != _cognito_issuer(config)
        or not isinstance(claims.get("sub"), str)
        or _SUB_PATTERN.fullmatch(claims["sub"]) is None
        or isinstance(claims.get("exp"), bool)
        or not isinstance(claims.get("exp"), (int, float))
        or not math.isfinite(claims["exp"])
    ):
        raise ValueError
    if token_use == "id":
        if claims.get("aud") != config.client_id:
            raise ValueError
    elif claims.get("client_id") != config.client_id:
        raise ValueError
    return claims


def _decode_jwt_part(value: str) -> dict[str, Any]:
    if len(value) > 32_768 or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError
    padding = "=" * (-len(value) % 4)
    decoded = base64.urlsafe_b64decode(value + padding)
    if len(decoded) > 16_384:
        raise ValueError
    document = _strict_json_loads(decoded)
    if not isinstance(document, dict):
        raise ValueError
    return document


def _token_needs_refresh(claims: Mapping[str, Any]) -> bool:
    return float(claims["exp"]) <= time.time() + 60.0


def _get_user_sub(response: Any) -> str:
    if not isinstance(response, dict):
        raise ValueError
    attributes = response.get("UserAttributes")
    if not isinstance(attributes, list):
        raise ValueError
    values = [
        item.get("Value")
        for item in attributes
        if isinstance(item, dict) and item.get("Name") == "sub"
    ]
    if (
        len(values) != 1
        or not isinstance(values[0], str)
        or _SUB_PATTERN.fullmatch(values[0]) is None
    ):
        raise ValueError
    return values[0]


@dataclass(frozen=True)
class _RequestsTransport:
    session: Any
    graphql_url: str = field(repr=False)
    id_token: str = field(repr=False)

    def __call__(
        self,
        document: str,
        variables: Mapping[str, Any],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> Mapping[str, Any]:
        response = None
        try:
            response = self.session.post(
                self.graphql_url,
                headers={
                    "Authorization": self.id_token,
                    "Content-Type": "application/json",
                },
                json={"query": document, "variables": dict(variables)},
                timeout=timeout_seconds,
                stream=True,
                allow_redirects=False,
            )
            status_code = getattr(response, "status_code", None)
            if status_code in {401, 403}:
                raise CloudReadError(
                    "AUTHENTICATION_REQUIRED",
                    "login-required",
                )
            if not isinstance(status_code, int) or not 200 <= status_code < 300:
                raise CloudReadError(
                    "UPSTREAM_UNAVAILABLE",
                    "upstream-unavailable",
                )
            raw_limit = min(139_264, max_response_bytes + 8_192)
            document_value = _read_bounded_http_json(
                response,
                max_bytes=raw_limit,
                oversized_code="RESPONSE_LIMIT_EXCEEDED",
            )
            if not isinstance(document_value, dict):
                raise CloudReadError(
                    "UPSTREAM_UNAVAILABLE",
                    "upstream-unavailable",
                )
            return document_value
        except CloudReadError:
            raise
        except Exception:
            raise CloudReadError(
                "UPSTREAM_UNAVAILABLE",
                "upstream-unavailable",
            ) from None
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            try:
                self.session.close()
            except Exception:
                pass


def _read_bounded_http_json(
    response: Any,
    *,
    max_bytes: int,
    oversized_code: str,
) -> Any:
    content_length = getattr(response, "headers", {}).get("Content-Length")
    if content_length is not None:
        if not str(content_length).isdigit() or int(content_length) > max_bytes:
            raise CloudReadError(
                oversized_code,
                (
                    "response-too-large"
                    if oversized_code == "RESPONSE_LIMIT_EXCEEDED"
                    else "upstream-unavailable"
                ),
            )
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=8192):
        if not isinstance(chunk, bytes):
            raise CloudReadError(
                "UPSTREAM_UNAVAILABLE",
                "upstream-unavailable",
            )
        size += len(chunk)
        if size > max_bytes:
            raise CloudReadError(
                oversized_code,
                (
                    "response-too-large"
                    if oversized_code == "RESPONSE_LIMIT_EXCEEDED"
                    else "upstream-unavailable"
                ),
            )
        chunks.append(chunk)
    try:
        return _strict_json_loads(b"".join(chunks))
    except Exception:
        raise CloudReadError(
            "UPSTREAM_UNAVAILABLE",
            "upstream-unavailable",
        ) from None
