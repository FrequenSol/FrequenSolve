from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from frequensolve.mcp_server import cloud


def _resource_bytes(name: str) -> bytes:
    return files("frequensolve.mcp_server").joinpath(f"contracts/{name}").read_bytes()


def _fixtures() -> list[dict[str, Any]]:
    return json.loads(
        _resource_bytes("customer_cloud_read_v2_fixtures.json").decode("utf-8")
    )


def _graphql_response(path: list[str], output: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = output
    for item in reversed(path):
        value = {item: value}
    return {"data": value}


def _fixture_transport_factory(
    calls: list[dict[str, Any]],
    operation: dict[str, Any],
    output: dict[str, Any],
):
    def factory(_: str | None):
        def transport(document, variables, timeout_seconds, max_response_bytes):
            calls.append(
                {
                    "document": document,
                    "variables": variables,
                    "timeout_seconds": timeout_seconds,
                    "max_response_bytes": max_response_bytes,
                }
            )
            return _graphql_response(
                operation["source"]["responseProjection"]["path"],
                output,
            )

        return transport

    return factory


def _call_public_method(
    client: cloud.CloudReadClient,
    operation: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if operation == "getCloudReadiness":
        return client.check_readiness()
    if operation == "listMySimulations":
        return client.list_simulations(
            status=arguments.get("status"),
            project_name=arguments.get("projectName"),
            limit=arguments.get("limit", 10),
            cursor=arguments.get("cursor"),
        )
    if operation == "getMySimulation":
        return client.get_simulation(simulation_id=arguments["simulationId"])
    if operation == "listMySimulationDiagnostics":
        return client.list_diagnostics(
            simulation_id=arguments["simulationId"],
            limit=arguments.get("limit", 25),
            cursor=arguments.get("cursor"),
        )
    if operation == "listMySimulationResultArtifacts":
        return client.list_result_artifacts(
            simulation_id=arguments["simulationId"],
            limit=arguments.get("limit", 25),
            after=arguments.get("after"),
        )
    raise AssertionError("unexpected fixture operation")


def _jwt(claims: dict[str, Any]) -> str:
    def encode(value: dict[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return ".".join(
        (
            encode({"alg": "RS256", "kid": "test-key"}),
            encode(claims),
            "test-signature",
        )
    )


def _cloud_config() -> dict[str, Any]:
    return {
        "auth": {
            "userPoolId": "us-east-1_TestPool",
            "clientId": "testclient123",
        },
        "api": {
            "graphqlUrl": (
                "https://example123.appsync-api.us-east-1.amazonaws.com/graphql"
            )
        },
        "region": "us-east-1",
    }


def _prepare_profile(
    root: Path,
    *,
    profile: str = "staging",
    domain: str = "app.staging.frequensol.com",
    site_type: str = "aws",
    create_config: bool = True,
) -> Path:
    root.mkdir()
    (root / "site.toml").write_text(
        "\n".join(
            (
                f'default = "{profile}"',
                "",
                f"[sites.{profile}]",
                f'type = "{site_type}"',
                f'domain = "{domain}"',
                'email = "ignored@example.test"',
                'password = "ignored-secret"',
                "interactive = true",
            )
        )
    )
    cache = root / "cloud"
    cache.mkdir()
    if create_config:
        config_path = cache / f"config_{domain.replace(':', '_')}.json"
        config_path.write_text(json.dumps(_cloud_config()))
    return cache


def _fresh_tokens(
    *,
    sub: str = "self-user",
    expires_in: float = 3600,
    issuer: str = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_TestPool",
) -> dict[str, str]:
    expiration = time.time() + expires_in
    return {
        "id_token": _jwt(
            {
                "iss": issuer,
                "token_use": "id",
                "aud": "testclient123",
                "sub": sub,
                "exp": expiration,
            }
        ),
        "access_token": _jwt(
            {
                "iss": issuer,
                "token_use": "access",
                "client_id": "testclient123",
                "sub": sub,
                "exp": expiration,
            }
        ),
        "refresh_token": "test-refresh-token",
    }


class _FakeResponse:
    def __init__(
        self,
        body: dict[str, Any] | bytes,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ):
        self.body = (
            json.dumps(body, separators=(",", ":")).encode()
            if isinstance(body, dict)
            else body
        )
        self.status_code = status_code
        self.headers = (
            {"Content-Length": str(len(self.body))} if headers is None else headers
        )
        self.closed = False
        self.iterated = False

    def iter_content(self, *, chunk_size):
        assert chunk_size == 8192
        self.iterated = True
        yield self.body

    def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, *responses: _FakeResponse | Exception):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.trust_env = True
        self.closed = False

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            pytest.fail("unexpected HTTP request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


def _fake_requests_module(*sessions: _FakeSession) -> SimpleNamespace:
    pending = list(sessions)

    def create_session():
        if not pending:
            pytest.fail("unexpected requests.Session construction")
        return pending.pop(0)

    return SimpleNamespace(Session=create_session)


def test_exact_cloud_contract_and_fixture_snapshots_are_vendored():
    catalog = _resource_bytes("customer_cloud_read_v2.json")
    fixtures = _resource_bytes("customer_cloud_read_v2_fixtures.json")

    assert hashlib.sha256(catalog).hexdigest() == (
        "520846921445e62128dbdfbd18a8b8b61b90c56d3e3fe3b8ac29d78c0a7a1091"
    )
    assert hashlib.sha256(fixtures).hexdigest() == (
        "ef406b2ad10780ebf8a60d2c7efde978d8662d3258f5f9fb953bd60b11c950c3"
    )


def test_contract_contains_only_fixed_bounded_queries():
    catalog = cloud._load_contract()["catalog"]

    assert catalog["contractId"] == cloud.CLOUD_READ_CONTRACT_ID
    assert catalog["contractVersion"] == cloud.CLOUD_READ_CONTRACT_VERSION
    assert tuple(item["name"] for item in catalog["operations"]) == (
        cloud.CLOUD_READ_OPERATIONS
    )
    for operation in catalog["operations"]:
        source = operation["source"]
        document = source["document"]
        assert operation["classification"] == "read-only"
        assert source["kind"] == "fixed-graphql-query"
        assert document.lstrip().startswith("query ")
        assert not re.search(
            r"\b(?:mutation|subscription)\b",
            document,
            re.IGNORECASE,
        )
        assert operation["inputSchema"]["additionalProperties"] is False
        assert operation["outputSchema"]["additionalProperties"] is False
        assert operation["limits"]["timeoutMs"] <= 15_000
        assert operation["limits"]["maxResponseBytes"] <= 65_536
        forbidden = {
            "accountId",
            "userId",
            "token",
            "graphql",
            "query",
            "mutation",
        }
        assert forbidden.isdisjoint(operation["inputSchema"]["properties"])


@pytest.mark.parametrize("fixture", _fixtures(), ids=lambda item: item["operation"])
def test_all_five_public_methods_project_exact_v2_fixture_outputs(fixture):
    operation = cloud._load_contract()["operationsByName"][fixture["operation"]]
    calls: list[dict[str, Any]] = []
    client = cloud.CloudReadClient(
        "staging",
        _transport_factory=_fixture_transport_factory(
            calls,
            operation,
            fixture["output"],
        ),
    )

    output = _call_public_method(
        client,
        fixture["operation"],
        fixture["input"],
    )

    assert output == fixture["output"]
    assert len(calls) == 1
    assert calls[0]["document"] == operation["source"]["document"]
    assert calls[0]["timeout_seconds"] == operation["limits"]["timeoutMs"] / 1000
    assert calls[0]["max_response_bytes"] == operation["limits"]["maxResponseBytes"]
    expected_variables = {}
    for name, variable in operation["source"]["variables"].items():
        input_name = variable["source"].removeprefix("input.")
        if input_name in fixture["input"]:
            expected_variables[name] = fixture["input"][input_name]
        elif "default" in variable:
            expected_variables[name] = variable["default"]
    assert calls[0]["variables"] == expected_variables


@pytest.mark.parametrize(
    "arguments",
    [
        {"accountId": "other-tenant"},
        {"userId": "other-user"},
        {"token": "secret"},
        {"query": "{ raw }"},
        {"mutation": "deleteEverything"},
        {"graphql": "query Raw"},
    ],
)
def test_dispatch_rejects_identity_and_raw_api_arguments(arguments):
    client = cloud.CloudReadClient(
        _transport_factory=lambda _: pytest.fail("transport must not run")
    )

    with pytest.raises(cloud.CloudReadError) as exc_info:
        client._execute("getCloudReadiness", arguments)

    assert exc_info.value.code == "INVALID_INPUT"


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        ("listMySimulations", {"limit": True}),
        ("listMySimulations", {"limit": 26}),
        ("listMySimulations", {"status": "DELETED"}),
        ("listMySimulations", {"cursor": "x" * 4097}),
        ("getMySimulation", {"simulationId": ""}),
        (
            "listMySimulationResultArtifacts",
            {"simulationId": "sim-1", "after": "../other-user/result.json"},
        ),
        (
            "listMySimulationResultArtifacts",
            {"simulationId": "sim-1", "after": "s3://bucket/key"},
        ),
    ],
)
def test_dispatch_enforces_strict_input_types_and_bounds(operation, arguments):
    client = cloud.CloudReadClient(
        _transport_factory=lambda _: pytest.fail("transport must not run")
    )

    with pytest.raises(cloud.CloudReadError) as exc_info:
        client._execute(operation, arguments)

    assert exc_info.value.code == "INVALID_INPUT"


def test_dispatch_accepts_catalog_valid_opaque_and_relative_inputs():
    contract = cloud._load_contract()["operationsByName"]
    cases = (
        (
            "getMySimulation",
            {"simulationId": "sim/with+contract(valid)"},
            next(
                item["output"]
                for item in _fixtures()
                if item["operation"] == "getMySimulation"
            ),
        ),
        (
            "listMySimulations",
            {"cursor": "opaque\ncatalog-cursor", "limit": 10},
            {"items": []},
        ),
        (
            "listMySimulationResultArtifacts",
            {
                "simulationId": "sim/with+contract(valid)",
                "after": "résult+/data (1).h5",
                "limit": 25,
            },
            {"items": []},
        ),
    )
    for operation_name, arguments, output in cases:
        calls: list[dict[str, Any]] = []
        client = cloud.CloudReadClient(
            _transport_factory=_fixture_transport_factory(
                calls,
                contract[operation_name],
                output,
            )
        )

        assert client._execute(operation_name, arguments) == output
        assert len(calls) == 1


def test_top_level_dispatch_is_finite_and_error_is_process_picklable():
    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud.execute_cloud_operation(None, "rawGraphQL", {})

    restored = pickle.loads(pickle.dumps(exc_info.value))
    assert restored.code == "INVALID_INPUT"
    assert restored.reason == "input-invalid"


def test_error_compatibility_constructor_discards_arbitrary_message_text():
    secret = "token=private-value at https://private.example/graphql"

    error = cloud.CloudReadError(
        "NOT_FOUND_OR_ACCESS_DENIED",
        secret,
        False,
    )

    assert error.code == "NOT_FOUND_OR_ACCESS_DENIED"
    assert error.retryable is False
    assert secret not in str(error)
    assert secret not in repr(error)
    assert secret not in error.__dict__.values()


def test_optional_null_arguments_are_pruned_before_strict_validation():
    operation = cloud._load_contract()["operationsByName"]["listMySimulations"]
    calls: list[dict[str, Any]] = []
    client = cloud.CloudReadClient(
        _transport_factory=_fixture_transport_factory(
            calls,
            operation,
            {"items": []},
        )
    )

    output = client._execute(
        "listMySimulations",
        {
            "status": None,
            "projectName": None,
            "limit": 10,
            "cursor": None,
        },
    )

    assert output == {"items": []}
    assert calls[0]["variables"] == {"limit": 10}


def test_optional_output_nulls_are_pruned_recursively_in_list_items():
    fixture = next(
        item for item in _fixtures() if item["operation"] == "listMySimulations"
    )
    output = json.loads(json.dumps(fixture["output"]))
    output["cursor"] = None
    output["items"][0]["durationSeconds"] = None
    output["items"][0]["requestedComputeMode"] = None
    operation = cloud._load_contract()["operationsByName"]["listMySimulations"]
    response = _graphql_response(
        operation["source"]["responseProjection"]["path"],
        output,
    )
    client = cloud.CloudReadClient(_transport_factory=lambda _: lambda *args: response)

    result = client.list_simulations()

    assert "cursor" not in result
    assert "durationSeconds" not in result["items"][0]
    assert "requestedComputeMode" not in result["items"][0]


def test_optional_output_nulls_are_pruned_in_nested_objects():
    fixture = next(
        item for item in _fixtures() if item["operation"] == "getMySimulation"
    )
    output = json.loads(json.dumps(fixture["output"]))
    output["durationSeconds"] = None
    output["guardrail"] = {
        "maxRuntimeMinutes": 60,
        "message": None,
    }
    operation = cloud._load_contract()["operationsByName"]["getMySimulation"]
    response = _graphql_response(
        operation["source"]["responseProjection"]["path"],
        output,
    )
    client = cloud.CloudReadClient(_transport_factory=lambda _: lambda *args: response)

    result = client.get_simulation(simulation_id="sim-example-001")

    assert "durationSeconds" not in result
    assert result["guardrail"] == {"maxRuntimeMinutes": 60}


@pytest.mark.parametrize(
    "required_null_change",
    [
        lambda output: output.update(status=None),
        lambda output: output["progress"].update(total=None),
    ],
)
def test_required_output_nulls_fail_closed(required_null_change):
    fixture = next(
        item for item in _fixtures() if item["operation"] == "getMySimulation"
    )
    output = json.loads(json.dumps(fixture["output"]))
    required_null_change(output)
    operation = cloud._load_contract()["operationsByName"]["getMySimulation"]
    response = _graphql_response(
        operation["source"]["responseProjection"]["path"],
        output,
    )
    client = cloud.CloudReadClient(_transport_factory=lambda _: lambda *args: response)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        client.get_simulation(simulation_id="sim-example-001")

    assert exc_info.value.code == "UPSTREAM_UNAVAILABLE"


def test_response_projection_rejects_missing_and_malformed_data():
    operation = cloud._load_contract()["operationsByName"]["getMySimulation"]

    for response in ({}, {"data": []}, {"data": {}}, {"data": {"x": {}}}):
        client = cloud.CloudReadClient(
            _transport_factory=lambda _, response=response: (lambda *args: response)
        )
        with pytest.raises(cloud.CloudReadError) as exc_info:
            client.get_simulation(simulation_id="sim-1")
        assert exc_info.value.code == "UPSTREAM_UNAVAILABLE"

    assert operation["source"]["responseProjection"]["path"] == [
        "getMySimulationRead",
        "simulation",
    ]


def test_missing_owned_simulations_have_one_non_probeable_result():
    responses = [
        {"errors": [{"message": "NOT_FOUND_OR_ACCESS_DENIED", "path": ["private-id"]}]},
        {"data": {"getMySimulationRead": {"simulation": None}}},
    ]
    failures = []
    for response in responses:
        client = cloud.CloudReadClient(
            _transport_factory=lambda _, response=response: (lambda *args: response)
        )
        with pytest.raises(cloud.CloudReadError) as exc_info:
            client.get_simulation(simulation_id="unavailable-id")
        failures.append(
            (
                exc_info.value.code,
                exc_info.value.safe_message,
                str(exc_info.value),
            )
        )

    assert failures[0] == failures[1]
    assert "private-id" not in repr(failures)


def test_provider_error_text_and_secrets_are_never_retained():
    secret = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJvdGhlciJ9.signature123"

    def factory(_):
        def transport(*args):
            raise RuntimeError(f"token={secret} at https://private.example/graphql")

        return transport

    client = cloud.CloudReadClient(_transport_factory=factory)
    with pytest.raises(cloud.CloudReadError) as exc_info:
        client.check_readiness()

    assert exc_info.value.code == "UPSTREAM_UNAVAILABLE"
    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)
    assert "private.example" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("operation", "code"),
    [
        ("getCloudReadiness", "AUTHENTICATION_REQUIRED"),
        ("listMySimulations", "INVALID_INPUT"),
        ("getMySimulation", "NOT_FOUND_OR_ACCESS_DENIED"),
        ("listMySimulationResultArtifacts", "RESULTS_NOT_AVAILABLE"),
        ("listMySimulations", "RESPONSE_LIMIT_EXCEEDED"),
        ("getCloudReadiness", "UPSTREAM_UNAVAILABLE"),
    ],
)
def test_exact_upstream_contract_error_codes_are_preserved(operation, code):
    client = cloud.CloudReadClient(
        _transport_factory=lambda _: lambda *args: {
            "errors": [{"message": f"Error: {code}"}]
        }
    )

    with pytest.raises(cloud.CloudReadError) as exc_info:
        client._execute(
            operation,
            (
                {"simulationId": "sim-1"}
                if operation
                in {
                    "getMySimulation",
                    "listMySimulationResultArtifacts",
                }
                else {}
            ),
        )

    assert exc_info.value.code == code
    assert exc_info.value.safe_message == cloud._CONTRACT_ERRORS[code][0]
    assert exc_info.value.retryable is cloud._CONTRACT_ERRORS[code][1]


def test_response_string_enums_fail_closed_as_upstream_unavailable():
    fixture = next(
        item for item in _fixtures() if item["operation"] == "getCloudReadiness"
    )
    output = json.loads(json.dumps(fixture["output"]))
    output["infrastructure"]["compute"] = "INTERNAL_ONLY"
    client = cloud.CloudReadClient(
        _transport_factory=lambda _: lambda *args: {"data": output}
    )

    with pytest.raises(cloud.CloudReadError) as exc_info:
        client.check_readiness()

    assert exc_info.value.code == "UPSTREAM_UNAVAILABLE"


@pytest.mark.parametrize(
    "unsafe_change",
    [
        lambda output: output.update(accountId="tenant-secret"),
        lambda output: output.update(status="https://private.example/log"),
        lambda output: output.update(status="user@example.test"),
        lambda output: output.update(status="x" * 65),
        lambda output: output.update(batchJobArn="arn:aws:batch:secret"),
    ],
)
def test_output_schema_and_redaction_fail_closed(unsafe_change):
    fixture = next(
        item for item in _fixtures() if item["operation"] == "getMySimulation"
    )
    output = json.loads(json.dumps(fixture["output"]))
    unsafe_change(output)
    operation = cloud._load_contract()["operationsByName"]["getMySimulation"]
    response = _graphql_response(
        operation["source"]["responseProjection"]["path"],
        output,
    )
    client = cloud.CloudReadClient(_transport_factory=lambda _: lambda *args: response)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        client.get_simulation(simulation_id="sim-example-001")

    assert exc_info.value.code == "UPSTREAM_UNAVAILABLE"


def test_result_paths_are_relative_and_secret_free():
    fixture = next(
        item
        for item in _fixtures()
        if item["operation"] == "listMySimulationResultArtifacts"
    )
    operation = cloud._load_contract()["operationsByName"][fixture["operation"]]
    for relative_path in (
        "/account/user/result.json",
        "../other-user/result.json",
        "bucket\\full\\key",
        "AKIA1234567890ABCDEF",
        "user@example.test",
    ):
        output = json.loads(json.dumps(fixture["output"]))
        output["items"][0]["relativePath"] = relative_path
        response = _graphql_response(
            operation["source"]["responseProjection"]["path"],
            output,
        )
        client = cloud.CloudReadClient(
            _transport_factory=lambda _, response=response: (lambda *args: response)
        )
        with pytest.raises(cloud.CloudReadError) as exc_info:
            client.list_result_artifacts(simulation_id="sim-example-001")
        assert exc_info.value.code == "UPSTREAM_UNAVAILABLE"


def test_valid_opaque_cursors_and_relative_result_paths_are_preserved():
    list_operation = cloud._load_contract()["operationsByName"]["listMySimulations"]
    opaque_cursor = "QWxwaGFCZXRhR2FtbWExMjM0NTY3ODkwU2VjcmV0UGFnZTI"
    list_response = _graphql_response(
        list_operation["source"]["responseProjection"]["path"],
        {"items": [], "cursor": opaque_cursor},
    )
    list_client = cloud.CloudReadClient(
        _transport_factory=lambda _: lambda *args: list_response
    )

    assert list_client.list_simulations()["cursor"] == opaque_cursor

    result_operation = cloud._load_contract()["operationsByName"][
        "listMySimulationResultArtifacts"
    ]
    relative_path = "résult+/frequency (10 Hz)/receiver-data.h5"
    result_response = _graphql_response(
        result_operation["source"]["responseProjection"]["path"],
        {
            "items": [
                {
                    "relativePath": relative_path,
                    "sizeBytes": 12,
                    "storageClass": "STANDARD",
                }
            ],
            "nextAfter": relative_path,
        },
    )
    result_client = cloud.CloudReadClient(
        _transport_factory=lambda _: lambda *args: result_response
    )

    assert result_client.list_result_artifacts(simulation_id="sim-example-001") == {
        "items": [
            {
                "relativePath": relative_path,
                "sizeBytes": 12,
                "storageClass": "STANDARD",
            }
        ],
        "nextAfter": relative_path,
    }


@pytest.mark.parametrize(
    "cursor",
    [
        "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.signature123",
        "xo" + "xb-123456789012-abcdefghijklmnop",
        "https://private.example/pagination",
    ],
)
def test_known_secret_shapes_in_opaque_cursors_fail_closed(cursor):
    operation = cloud._load_contract()["operationsByName"]["listMySimulations"]
    response = _graphql_response(
        operation["source"]["responseProjection"]["path"],
        {"items": [], "cursor": cursor},
    )
    client = cloud.CloudReadClient(_transport_factory=lambda _: lambda *args: response)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        client.list_simulations()

    assert exc_info.value.code == "UPSTREAM_UNAVAILABLE"


@pytest.mark.parametrize(
    "secret",
    [
        "xo" + "xb-123456789012-abcdefghijklmnop",
        "sk_live_abcdefghijklmnop",
        "sk-proj-abcdefghijklmnop",
        "frequensol-private-user-dev-storage/full/key.dat",
        "example-bucket.s3.us-east-1.amazonaws.com",
        "example-bucket.s3.cn-north-1.amazonaws.com.cn",
        "access_token/opaquevalue1234567890",
        "QWxwaGFCZXRhR2FtbWExMjM0NTY3ODkwU2VjcmV0",
        "alice%2540example.com",
        "alice&#X40;example.com",
    ],
)
def test_output_text_secret_shapes_fail_closed(secret):
    operation = cloud._load_contract()["operationsByName"][
        "listMySimulationDiagnostics"
    ]
    response = _graphql_response(
        operation["source"]["responseProjection"]["path"],
        {
            "simulationId": "sim-example-001",
            "simulationStatus": "FAILED",
            "items": [
                {
                    "frequencyIndex": 0,
                    "status": "FAILED",
                    "message": secret,
                }
            ],
        },
    )
    client = cloud.CloudReadClient(_transport_factory=lambda _: lambda *args: response)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        client.list_diagnostics(simulation_id="sim-example-001")

    assert exc_info.value.code == "UPSTREAM_UNAVAILABLE"
    assert secret not in str(exc_info.value)


def test_s3_endpoint_detection_handles_long_repeated_hyphens_without_backtracking():
    candidate = f"example.s3{'--' * 10_000}.not-amazonaws.invalid"

    assert not cloud._contains_sensitive_output_text(candidate, key="message")


def test_response_page_and_byte_limits_are_enforced():
    operation = cloud._load_contract()["operationsByName"]["listMySimulations"]
    fixture = next(
        item for item in _fixtures() if item["operation"] == "listMySimulations"
    )
    output = {"items": [fixture["output"]["items"][0]] * 26}
    response = _graphql_response(
        operation["source"]["responseProjection"]["path"],
        output,
    )
    client = cloud.CloudReadClient(_transport_factory=lambda _: lambda *args: response)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        client.list_simulations()

    assert exc_info.value.code == "RESPONSE_LIMIT_EXCEEDED"


def test_profile_loader_uses_existing_default_aws_profile_and_ignores_secrets(
    monkeypatch, tmp_path
):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    (root / "site.toml").chmod(0o600)
    (cache / "config_app.staging.frequensol.com.json").chmod(0o644)
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))

    profile = cloud._load_site_profile(None)
    config = cloud._load_cached_cloud_config(profile)

    assert profile == cloud._SiteProfile(
        name="staging",
        domain="app.staging.frequensol.com",
    )
    assert config.region == "us-east-1"
    assert config.graphql_url.endswith(".amazonaws.com/graphql")
    assert "ignored-secret" not in repr(profile)
    assert "ignored@example.test" not in repr(profile)


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership and mode policy")
@pytest.mark.parametrize(
    ("target", "mode"),
    [
        ("site", 0o666),
        ("config", 0o664),
    ],
)
def test_profile_loader_rejects_group_or_world_writable_trust_files(
    monkeypatch,
    tmp_path,
    target,
    mode,
):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    site_path = root / "site.toml"
    config_path = cache / "config_app.staging.frequensol.com.json"
    site_path.chmod(0o600)
    config_path.chmod(0o644)
    (site_path if target == "site" else config_path).chmod(mode)
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))

    with pytest.raises(cloud.CloudReadError) as exc_info:
        profile = cloud._load_site_profile(None)
        if target == "config":
            cloud._load_cached_cloud_config(profile)

    assert exc_info.value.code == "CLOUD_CONFIGURATION_REQUIRED"


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "getuid"),
    reason="POSIX ownership policy",
)
def test_profile_loader_rejects_trust_files_not_owned_by_current_user(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / ".frequensolve"
    _prepare_profile(root)
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    actual_uid = os.getuid()
    monkeypatch.setattr(cloud.os, "getuid", lambda: actual_uid + 1)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._load_site_profile(None)

    assert exc_info.value.code == "CLOUD_CONFIGURATION_REQUIRED"


def test_profile_loader_never_creates_starter_configuration(monkeypatch, tmp_path):
    root = tmp_path / ".frequensolve"
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._load_site_profile(None)

    assert exc_info.value.code == "CLOUD_CONFIGURATION_REQUIRED"
    assert exc_info.value.reason == "profile-unavailable"
    assert not root.exists()


@pytest.mark.parametrize(
    ("site_type", "domain"),
    [
        ("local", "app.staging.frequensol.com"),
        ("aws", "https://app.staging.frequensol.com"),
        ("aws", "169.254.169.254"),
        ("aws", "../staging"),
        ("aws", "localhost"),
    ],
)
def test_profile_loader_rejects_noncloud_and_unsafe_domains(
    monkeypatch, tmp_path, site_type, domain
):
    root = tmp_path / ".frequensolve"
    _prepare_profile(
        root,
        site_type=site_type,
        domain=domain,
        create_config=False,
    )
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._load_site_profile(None)

    assert exc_info.value.code == "CLOUD_CONFIGURATION_REQUIRED"


def test_cached_config_rejects_non_appsync_token_destination(monkeypatch, tmp_path):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    config_path = cache / "config_app.staging.frequensol.com.json"
    config = _cloud_config()
    config["api"]["graphqlUrl"] = "https://attacker.example/graphql"
    config_path.write_text(json.dumps(config))
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))

    profile = cloud._load_site_profile(None)
    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._load_cached_cloud_config(profile)

    assert exc_info.value.code == "CLOUD_CONFIGURATION_REQUIRED"
    assert "attacker.example" not in str(exc_info.value)


def test_authenticated_transport_uses_profile_cache_and_verified_same_sub(
    monkeypatch, tmp_path
):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    tokens = _fresh_tokens()
    credential_path = cache / "credentials_staging"
    credential_path.write_text(json.dumps(tokens))
    credential_path.chmod(0o600)
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    original_bytes = credential_path.read_bytes()
    original_mtime = credential_path.stat().st_mtime_ns
    assert cloud._read_token_cache(credential_path) == {
        "id_token": tokens["id_token"],
        "access_token": tokens["access_token"],
    }
    cognito_response = _FakeResponse(
        {
            "Username": "opaque-cognito-username",
            "UserAttributes": [{"Name": "sub", "Value": "self-user"}],
        }
    )
    cognito_session = _FakeSession(cognito_response)
    appsync_session = _FakeSession(_FakeResponse({"data": {"safe": True}}))
    fake_requests = _fake_requests_module(cognito_session, appsync_session)
    original_import = cloud.importlib.import_module

    def fake_import(name):
        if name == "requests":
            return fake_requests
        if name.endswith(".cognito") or name in {"boto3", "botocore"}:
            pytest.fail("the read adapter must not import AWS credential clients")
        return original_import(name)

    monkeypatch.setenv(
        "AWS_ENDPOINT_URL_COGNITO_IDENTITY_PROVIDER",
        "https://attacker.example",
    )
    monkeypatch.setenv("HTTPS_PROXY", "https://attacker.example:8443")
    monkeypatch.setattr(cloud.importlib, "import_module", fake_import)

    transport = cloud._build_authenticated_transport("staging")

    assert isinstance(transport, cloud._RequestsTransport)
    assert transport.id_token == tokens["id_token"]
    assert credential_path.read_bytes() == original_bytes
    assert credential_path.stat().st_mtime_ns == original_mtime
    assert cognito_session.trust_env is False
    assert appsync_session.trust_env is False
    assert cognito_session.closed is True
    assert len(cognito_session.calls) == 1
    call = cognito_session.calls[0]
    assert call["url"] == "https://cognito-idp.us-east-1.amazonaws.com"
    assert call["json"] == {"AccessToken": tokens["access_token"]}
    assert call["allow_redirects"] is False
    assert call["stream"] is True
    assert call["headers"]["X-Amz-Target"].endswith(".GetUser")
    assert "attacker.example" not in json.dumps(call)
    assert transport("query Fixed { safe }", {}, 10.0, 16_384) == {
        "data": {"safe": True}
    }
    assert appsync_session.calls[0]["url"] == (
        "https://example123.appsync-api.us-east-1.amazonaws.com/graphql"
    )
    assert appsync_session.calls[0]["allow_redirects"] is False
    assert "attacker.example" not in json.dumps(appsync_session.calls[0])


def test_authenticated_transport_prefers_fresh_canonical_login_over_stale_profile_cache(
    monkeypatch, tmp_path
):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    canonical_tokens = _fresh_tokens(sub="fresh-login-user")
    stale_profile_tokens = _fresh_tokens(
        sub="old-profile-user",
        expires_in=-60,
    )
    canonical_path = cache / "credentials"
    profile_path = cache / "credentials_staging"
    canonical_path.write_text(json.dumps(canonical_tokens))
    profile_path.write_text(json.dumps(stale_profile_tokens))
    canonical_path.chmod(0o600)
    profile_path.chmod(0o600)
    original_files = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (canonical_path, profile_path)
    }
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    cognito_session = _FakeSession(
        _FakeResponse(
            {
                "UserAttributes": [
                    {"Name": "sub", "Value": "fresh-login-user"},
                ]
            }
        )
    )
    appsync_session = _FakeSession(_FakeResponse({"data": {"safe": True}}))
    fake_requests = _fake_requests_module(cognito_session, appsync_session)
    original_import = cloud.importlib.import_module

    def fake_import(name):
        if name == "requests":
            return fake_requests
        return original_import(name)

    monkeypatch.setattr(cloud.importlib, "import_module", fake_import)

    transport = cloud._build_authenticated_transport("staging")

    assert transport.id_token == canonical_tokens["id_token"]
    assert cognito_session.calls[0]["json"] == {
        "AccessToken": canonical_tokens["access_token"]
    }
    for path, (original_bytes, original_mtime) in original_files.items():
        assert path.read_bytes() == original_bytes
        assert path.stat().st_mtime_ns == original_mtime


@pytest.mark.parametrize(
    "canonical_contents",
    [
        b"{not-json",
        b"",
        b"x" * 65_537,
        json.dumps(
            _fresh_tokens(
                issuer=(
                    "https://cognito-idp.us-east-1.amazonaws.com/" "us-east-1_OtherPool"
                )
            )
        ).encode(),
    ],
    ids=("malformed", "empty", "oversized", "wrong-profile"),
)
def test_present_invalid_canonical_cache_never_falls_through_to_profile_identity(
    monkeypatch,
    tmp_path,
    canonical_contents,
):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    canonical_path = cache / "credentials"
    profile_path = cache / "credentials_staging"
    canonical_path.write_bytes(canonical_contents)
    profile_path.write_text(json.dumps(_fresh_tokens(sub="fallback-user")))
    canonical_path.chmod(0o600)
    profile_path.chmod(0o600)
    original_files = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (canonical_path, profile_path)
    }
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    original_import = cloud.importlib.import_module

    def fake_import(name):
        if name == "requests":
            return SimpleNamespace(
                Session=lambda: pytest.fail(
                    "an invalid canonical cache must not use the profile identity"
                )
            )
        return original_import(name)

    monkeypatch.setattr(cloud.importlib, "import_module", fake_import)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._build_authenticated_transport("staging")

    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.reason == "identity-invalid"
    for path, (original_bytes, original_mtime) in original_files.items():
        assert path.read_bytes() == original_bytes
        assert path.stat().st_mtime_ns == original_mtime


def test_present_expired_canonical_cache_requires_login_without_profile_fallback(
    monkeypatch, tmp_path
):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    canonical_path = cache / "credentials"
    profile_path = cache / "credentials_staging"
    canonical_path.write_text(json.dumps(_fresh_tokens(expires_in=-60)))
    profile_path.write_text(json.dumps(_fresh_tokens(sub="fallback-user")))
    canonical_path.chmod(0o600)
    profile_path.chmod(0o600)
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    original_import = cloud.importlib.import_module

    def fake_import(name):
        if name == "requests":
            return SimpleNamespace(
                Session=lambda: pytest.fail(
                    "an expired canonical cache must require a new login"
                )
            )
        return original_import(name)

    monkeypatch.setattr(cloud.importlib, "import_module", fake_import)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._build_authenticated_transport("staging")

    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.reason == "login-required"


def test_legacy_root_cache_is_used_only_when_newer_cache_paths_are_absent(
    monkeypatch, tmp_path
):
    root = tmp_path / ".frequensolve"
    _prepare_profile(root)
    legacy_tokens = _fresh_tokens(sub="legacy-user")
    legacy_path = root / "credentials"
    legacy_path.write_text(json.dumps(legacy_tokens))
    legacy_path.chmod(0o600)
    original_bytes = legacy_path.read_bytes()
    original_mtime = legacy_path.stat().st_mtime_ns
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    profile = cloud._load_site_profile("staging")
    config = cloud._load_cached_cloud_config(profile)

    tokens, id_claims = cloud._select_cached_identity(
        "staging",
        config=config,
    )

    assert tokens["id_token"] == legacy_tokens["id_token"]
    assert id_claims["sub"] == "legacy-user"
    assert legacy_path.read_bytes() == original_bytes
    assert legacy_path.stat().st_mtime_ns == original_mtime


def test_present_invalid_profile_cache_blocks_legacy_identity_fallback(
    monkeypatch, tmp_path
):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    profile_path = cache / "credentials_staging"
    legacy_path = root / "credentials"
    profile_path.write_text("{not-json")
    legacy_path.write_text(json.dumps(_fresh_tokens(sub="legacy-user")))
    profile_path.chmod(0o600)
    legacy_path.chmod(0o600)
    original_files = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (profile_path, legacy_path)
    }
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    profile = cloud._load_site_profile("staging")
    config = cloud._load_cached_cloud_config(profile)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._select_cached_identity("staging", config=config)

    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.reason == "identity-invalid"
    for path, (original_bytes, original_mtime) in original_files.items():
        assert path.read_bytes() == original_bytes
        assert path.stat().st_mtime_ns == original_mtime


def test_missing_all_credential_caches_requires_login_without_creating_files(
    monkeypatch, tmp_path
):
    root = tmp_path / ".frequensolve"
    _prepare_profile(root)
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    profile = cloud._load_site_profile("staging")
    config = cloud._load_cached_cloud_config(profile)
    expected_paths = cloud._credential_paths("staging")

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._select_cached_identity("staging", config=config)

    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.reason == "login-required"
    assert all(not path.exists() for path in expected_paths)


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-file policy")
def test_insecure_canonical_cache_blocks_profile_identity_fallback(
    monkeypatch, tmp_path
):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    canonical_path = cache / "credentials"
    profile_path = cache / "credentials_staging"
    canonical_path.write_text(json.dumps(_fresh_tokens()))
    profile_path.write_text(json.dumps(_fresh_tokens(sub="fallback-user")))
    canonical_path.chmod(0o644)
    profile_path.chmod(0o600)
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    profile = cloud._load_site_profile("staging")
    config = cloud._load_cached_cloud_config(profile)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._select_cached_identity("staging", config=config)

    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.reason == "identity-invalid"


def test_authenticated_transport_rejects_get_user_sub_mismatch_without_leak(
    monkeypatch, tmp_path
):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    tokens = _fresh_tokens(sub="expected-self")
    credential_path = cache / "credentials_staging"
    credential_path.write_text(json.dumps(tokens))
    credential_path.chmod(0o600)
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    cognito_session = _FakeSession(
        _FakeResponse(
            {"UserAttributes": [{"Name": "sub", "Value": "different-private-sub"}]}
        )
    )
    fake_requests = _fake_requests_module(cognito_session)
    original_import = cloud.importlib.import_module

    def fake_import(name):
        if name == "requests":
            return fake_requests
        return original_import(name)

    monkeypatch.setattr(cloud.importlib, "import_module", fake_import)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._build_authenticated_transport("staging")

    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.reason == "identity-invalid"
    assert "expected-self" not in str(exc_info.value)
    assert "different-private-sub" not in str(exc_info.value)


@pytest.mark.parametrize("expires_in", [-1, 0, 30, 60])
def test_expired_or_near_expiry_tokens_require_login_without_writes_or_network(
    monkeypatch,
    tmp_path,
    expires_in,
):
    root = tmp_path / ".frequensolve"
    cache = _prepare_profile(root)
    credential_path = cache / "credentials_staging"
    credential_path.write_text(json.dumps(_fresh_tokens(expires_in=expires_in)))
    credential_path.chmod(0o600)
    original_bytes = credential_path.read_bytes()
    original_mtime = credential_path.stat().st_mtime_ns
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    original_import = cloud.importlib.import_module

    def fake_import(name):
        if name == "requests":
            return SimpleNamespace(
                Session=lambda: pytest.fail(
                    "expired cached tokens must not make a network request"
                )
            )
        return original_import(name)

    monkeypatch.setattr(cloud.importlib, "import_module", fake_import)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._build_authenticated_transport("staging")

    assert exc_info.value.code == "AUTHENTICATION_REQUIRED"
    assert exc_info.value.reason == "login-required"
    assert credential_path.read_bytes() == original_bytes
    assert credential_path.stat().st_mtime_ns == original_mtime


@pytest.mark.parametrize(
    ("region", "expected_endpoint"),
    [
        ("us-east-1", "https://cognito-idp.us-east-1.amazonaws.com"),
        ("us-gov-west-1", "https://cognito-idp.us-gov-west-1.amazonaws.com"),
        ("cn-north-1", "https://cognito-idp.cn-north-1.amazonaws.com.cn"),
    ],
)
def test_cognito_endpoint_and_issuer_are_partition_correct(region, expected_endpoint):
    config = cloud._CloudConfig(
        region=region,
        user_pool_id=f"{region}_TestPool",
        client_id="testclient123",
        graphql_url=(
            f"https://example123.appsync-api.{region}."
            f"{'amazonaws.com.cn' if region.startswith('cn-') else 'amazonaws.com'}"
            "/graphql"
        ),
    )

    assert cloud._cognito_endpoint(region) == expected_endpoint
    assert cloud._cognito_issuer(config) == f"{expected_endpoint}/{region}_TestPool"
    assert cloud._is_safe_appsync_url(config.graphql_url, region)
    tokens = _fresh_tokens(
        issuer=f"{expected_endpoint}/{region}_TestPool",
    )
    assert (
        cloud._decode_and_validate_token(
            tokens["access_token"],
            token_use="access",
            config=config,
        )["sub"]
        == "self-user"
    )


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (
            _FakeResponse(
                b"provider token=private redirect",
                status_code=302,
                headers={"Location": "https://attacker.example"},
            ),
            "UPSTREAM_UNAVAILABLE",
        ),
        (
            _FakeResponse(
                b"provider token=private oversized",
                headers={"Content-Length": "999999"},
            ),
            "UPSTREAM_UNAVAILABLE",
        ),
        (
            _FakeResponse(
                b"x" * (cloud._COGNITO_MAX_RESPONSE_BYTES + 1),
                headers={},
            ),
            "UPSTREAM_UNAVAILABLE",
        ),
        (
            _FakeResponse(
                b'{"message":"provider token=private invalid login"}',
                status_code=400,
            ),
            "AUTHENTICATION_REQUIRED",
        ),
        (
            _FakeResponse(b"provider token=private malformed json"),
            "UPSTREAM_UNAVAILABLE",
        ),
    ],
)
def test_cognito_verification_blocks_redirects_size_and_provider_text(
    response,
    expected_code,
):
    session = _FakeSession(response)
    requests = _fake_requests_module(session)
    config = cloud._CloudConfig(
        region="us-east-1",
        user_pool_id="us-east-1_TestPool",
        client_id="testclient123",
        graphql_url=("https://example123.appsync-api.us-east-1.amazonaws.com/graphql"),
    )

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._get_verified_cognito_user(
            requests,
            config=config,
            access_token="private-access-token",
        )

    assert exc_info.value.code == expected_code
    assert "private" not in str(exc_info.value)
    assert session.calls[0]["allow_redirects"] is False
    assert session.trust_env is False
    assert session.closed is True
    assert response.closed is True
    if (
        response.status_code == 302
        or response.headers.get("Content-Length") == "999999"
    ):
        assert response.iterated is False


def test_missing_cloud_extra_returns_safe_readiness_error(monkeypatch, tmp_path):
    root = tmp_path / ".frequensolve"
    _prepare_profile(root)
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(root))
    original_import = cloud.importlib.import_module

    def fake_import(name):
        if name == "requests":
            raise ModuleNotFoundError("requests secret install path")
        return original_import(name)

    monkeypatch.setattr(cloud.importlib, "import_module", fake_import)

    with pytest.raises(cloud.CloudReadError) as exc_info:
        cloud._build_authenticated_transport("staging")

    assert exc_info.value.code == "CLOUD_SUPPORT_REQUIRED"
    assert exc_info.value.reason == "cloud-extra-missing"
    assert "requests secret install path" not in str(exc_info.value)


def test_requests_transport_sends_only_fixed_document_and_bounded_request():
    response_body = json.dumps({"data": {"membership": {}}}).encode()
    response = _FakeResponse(response_body)
    session = _FakeSession(response)
    session.trust_env = False

    transport = cloud._RequestsTransport(
        session=session,
        graphql_url=("https://example123.appsync-api.us-east-1.amazonaws.com/graphql"),
        id_token="private-test-id-token",
    )
    assert "private-test-id-token" not in repr(transport)
    assert "appsync-api" not in repr(transport)
    result = transport(
        "query Fixed { membership { hasSeat } }",
        {},
        10.0,
        16_384,
    )

    assert result == {"data": {"membership": {}}}
    captured = session.calls[0]
    assert captured["json"] == {
        "query": "query Fixed { membership { hasSeat } }",
        "variables": {},
    }
    assert captured["timeout"] == 10.0
    assert captured["stream"] is True
    assert captured["allow_redirects"] is False
    assert captured["headers"]["Authorization"] == "private-test-id-token"
    assert response.closed is True
    assert session.closed is True


def test_requests_transport_normalizes_http_network_and_size_errors():
    cases = (
        _FakeSession(
            _FakeResponse(
                b'{"provider":"token=private"}',
                headers={"Content-Length": "999999"},
            )
        ),
        _FakeSession(TimeoutError("private upstream URL and token")),
        _FakeSession(
            _FakeResponse(
                b'{"provider":"token=private redirect"}',
                status_code=302,
                headers={"Location": "https://attacker.example"},
            )
        ),
    )
    expected = (
        "RESPONSE_LIMIT_EXCEEDED",
        "UPSTREAM_UNAVAILABLE",
        "UPSTREAM_UNAVAILABLE",
    )
    for session, code in zip(cases, expected):
        transport = cloud._RequestsTransport(
            session=session,
            graphql_url=(
                "https://example123.appsync-api.us-east-1.amazonaws.com/graphql"
            ),
            id_token="private-test-id-token",
        )
        with pytest.raises(cloud.CloudReadError) as exc_info:
            transport("query Fixed { value }", {}, 10.0, 16_384)
        assert exc_info.value.code == code
        assert "private" not in str(exc_info.value)
        assert session.calls[0]["allow_redirects"] is False
        assert session.closed is True


def test_hosted_contract_path_does_not_require_cloud_or_scientific_runtime():
    script = """
import builtins
import json
import sys
from importlib.resources import files

real_import = builtins.__import__
forbidden = {
    'boto3',
    'botocore',
    'h5py',
    'numpy',
    'pandas',
    'requests',
    'scipy',
    'xarray',
    'frequensolve.orchestrator',
    'frequensolve.simulation',
}
def guarded(name, *args, **kwargs):
    if any(name == item or name.startswith(item + '.') for item in forbidden):
        raise ModuleNotFoundError(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded

import frequensolve.mcp_server.cloud as cloud
from frequensolve.mcp_server.server import build_server

async def hosted_executor(operation, arguments):
    raise AssertionError('construction must not execute an operation')

server = build_server(
    capability_profile='authenticated-cloud',
    hosted_cloud_executor=hosted_executor,
)
if server is None:
    raise SystemExit('authenticated Cloud profile construction failed')

fixtures = json.loads(
    files('frequensolve.mcp_server')
    .joinpath('contracts/customer_cloud_read_v2_fixtures.json')
    .read_text(encoding='utf-8')
)
fixture = next(
    item for item in fixtures if item['operation'] == 'getCloudReadiness'
)
validated = cloud.validate_cloud_operation_output(
    fixture['operation'],
    fixture['input'],
    fixture['output'],
)
if validated != fixture['output']:
    raise SystemExit('hosted contract validation changed the fixture')

loaded = sorted(
    name
    for name in sys.modules
    if any(name == item or name.startswith(item + '.') for item in forbidden)
)
if loaded:
    raise SystemExit('hosted path imported forbidden modules: ' + ', '.join(loaded))
print(cloud.CLOUD_READ_CONTRACT_VERSION)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "2.0.0"
