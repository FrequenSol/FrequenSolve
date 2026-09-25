from types import SimpleNamespace

import pytest

pytest.importorskip("requests")
import requests

from frequensolve.orchestrator.sites.aws import graphql_client


class FakeAuth:
    def __init__(self, token="secret-id-token", account_id=None):
        self.token = token
        self.account_id = account_id
        self.calls = 0

    def get_id_token(self):
        self.calls += 1
        return self.token

    def get_account_id(self):
        return self.account_id


class FakeResponse:
    def __init__(self, payload=None, *, json_error=None, http_error=None):
        self.payload = payload
        self.json_error = json_error
        self.http_error = http_error

    def raise_for_status(self):
        if self.http_error:
            raise self.http_error

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


def _client():
    return graphql_client.GraphQLClient(
        "https://api.example.invalid/graphql",
        FakeAuth(),
    )


def test_execute_sends_bounded_authenticated_request(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse({"data": {"value": 7}})

    monkeypatch.setattr(graphql_client.requests, "post", post)
    client = _client()

    assert client.execute("query Example($id: ID!) { value }", {"id": "item-1"}) == {
        "value": 7
    }
    assert calls == [
        (
            "https://api.example.invalid/graphql",
            {
                "headers": {
                    "Authorization": "secret-id-token",
                    "Content-Type": "application/json",
                },
                "json": {
                    "query": "query Example($id: ID!) { value }",
                    "variables": {"id": "item-1"},
                },
                "timeout": 30,
            },
        )
    ]
    assert client.auth.calls == 1


def test_execute_redacts_token_and_request_values_from_graphql_errors(monkeypatch):
    token = "secret-id-token"
    private_key = "accounts/private-user/jobs/private-job.json"
    account_id = "private-account-id"

    def post(*args, **kwargs):
        return FakeResponse(
            {
                "errors": [
                    {
                        "message": (
                            f"Rejected token {token} for object {private_key} "
                            f"in account {account_id} "
                            "(Unknown argument 'forceRun')"
                        )
                    }
                ]
            }
        )

    monkeypatch.setattr(graphql_client.requests, "post", post)
    client = graphql_client.GraphQLClient(
        "https://api.example.invalid/graphql",
        FakeAuth(token, account_id),
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.execute("mutation Submit { submitJob }", {"key": private_key})

    diagnostic = str(exc_info.value)
    assert "Unknown argument 'forceRun'" in diagnostic
    assert token not in diagnostic
    assert private_key not in diagnostic
    assert account_id not in diagnostic
    assert diagnostic.count("<redacted>") == 3


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "non-object response"),
        ({}, "object data field"),
        ({"data": None}, "object data field"),
        ({"data": []}, "object data field"),
        ({"errors": {"message": "bad"}}, "malformed error envelope"),
    ],
)
def test_execute_rejects_malformed_response_envelopes(monkeypatch, payload, message):
    monkeypatch.setattr(
        graphql_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(payload),
    )

    with pytest.raises(RuntimeError, match=message):
        _client().execute("query { value }")


def test_execute_rejects_malformed_json(monkeypatch):
    monkeypatch.setattr(
        graphql_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(json_error=ValueError("private body")),
    )

    with pytest.raises(RuntimeError, match="malformed JSON") as exc_info:
        _client().execute("query { value }")

    assert "private body" not in str(exc_info.value)


def test_execute_maps_timeout_without_echoing_request_or_token(monkeypatch):
    private_key = "accounts/private-user/jobs/job.json"

    def timeout(*args, **kwargs):
        raise requests.exceptions.Timeout(f"token=secret-id-token object={private_key}")

    monkeypatch.setattr(graphql_client.requests, "post", timeout)

    with pytest.raises(RuntimeError, match="timed out after 30 seconds") as exc_info:
        _client().execute("query Example { value }", {"key": private_key})

    diagnostic = str(exc_info.value)
    assert private_key not in diagnostic
    assert "secret-id-token" not in diagnostic
    assert exc_info.value.__cause__ is None


def test_execute_maps_http_failure_without_provider_body(monkeypatch):
    provider_error = requests.exceptions.HTTPError(
        "403 token=secret-id-token account=private-account"
    )
    monkeypatch.setattr(
        graphql_client.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(http_error=provider_error),
    )

    with pytest.raises(
        graphql_client.CloudAPIError, match="Cloud API rejected the HTTP request"
    ) as exc_info:
        _client().execute("query { value }")

    assert exc_info.value.__cause__ is None


class PollingGraphQLClient(graphql_client.GraphQLClient):
    def __init__(self, results, *, account_id="private-account-id"):
        super().__init__(
            "https://api.example.invalid/graphql",
            auth=SimpleNamespace(get_account_id=lambda: account_id),
        )
        self.results = iter(results)
        self.calls = []

    def execute(self, query, variables=None):
        self.calls.append((query, variables))
        value = next(self.results)
        if isinstance(value, BaseException):
            raise value
        return value


def test_wait_for_storage_ready_retries_transient_error_then_maps_success(
    monkeypatch,
):
    client = PollingGraphQLClient(
        [
            RuntimeError("not found"),
            {
                "listStacks": {
                    "items": [
                        {
                            "stackId": "stack-1",
                            "status": "CREATE_COMPLETE",
                            "outputs": '{"StorageBucketName":"bucket"}',
                            "createdAt": "2026-08-13T00:00:00Z",
                        }
                    ]
                }
            },
        ]
    )
    clock = iter([0, 31, 31, 31])
    monkeypatch.setattr(graphql_client.time, "time", lambda: next(clock))
    sleeps = []
    monkeypatch.setattr(graphql_client.time, "sleep", sleeps.append)

    assert client.wait_for_storage_ready(
        timeout=60,
        poll_interval=2,
        expected_stack_id="stack-1",
    ) == {
        "stackId": "stack-1",
        "bucketName": "bucket",
        "status": "CREATE_COMPLETE",
    }
    assert sleeps == [2]
    assert len(client.calls) == 2


def test_wait_for_storage_ready_propagates_timeout_at_bound(monkeypatch):
    client = PollingGraphQLClient([])
    clock = iter([100, 105])
    monkeypatch.setattr(graphql_client.time, "time", lambda: next(clock))

    with pytest.raises(RuntimeError, match="timeout: 5s"):
        client.wait_for_storage_ready(timeout=5, poll_interval=0)

    assert client.calls == []


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (requests.exceptions.Timeout("private"), True),
        (requests.exceptions.ConnectionError("private"), True),
        (requests.exceptions.SSLError("private"), False),
        (requests.exceptions.InvalidURL("private"), False),
        (requests.exceptions.RequestException("private"), False),
    ],
)
def test_only_transient_status_reads_signal_monitor_recovery(
    monkeypatch, error, retryable
):
    from frequensolve.orchestrator.utils.status_errors import TransientStatusReadError

    calls = []

    def post(*args, **kwargs):
        calls.append(kwargs["json"]["query"])
        raise error

    monkeypatch.setattr(graphql_client.requests, "post", post)
    expected = (
        TransientStatusReadError if retryable else graphql_client.CloudTransportError
    )
    with pytest.raises(expected) as raised:
        _client().get_simulation_status_details("synthetic-run-1")
    assert len(calls) == 1
    assert "private" not in str(raised.value)
    if retryable:
        assert "synthetic-run-1" in str(raised.value)
        assert "do not resubmit" in str(raised.value)

    # The shared executor never retries a mutation or turns it into a status signal.
    with pytest.raises(graphql_client.CloudTransportError):
        _client().execute("mutation Submit { submitJob }")
    assert len(calls) == 2


@pytest.mark.parametrize("retry", [False, True])
def test_retry_submission_argument_is_opt_in_for_older_backends(retry):
    client = graphql_client.GraphQLClient.__new__(graphql_client.GraphQLClient)
    captured = {}

    def execute(document, variables):
        captured.update(document=document, variables=variables)
        return {"submitJob": {"simulationId": "new-run", "status": "PENDING"}}

    client.execute = execute
    client.submit_job("project/job.json", retry=retry)
    assert ("$retry: Boolean" in captured["document"]) == retry
    assert captured["variables"].get("retry") == (True if retry else None)
