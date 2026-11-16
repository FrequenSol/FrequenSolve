from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from frequensolve.workflow.analysis import analyze_pressure
from frequensolve.workflow.auth import Connection, WorkflowAuth, WorkflowError
from frequensolve.workflow.journal import IntentJournal
from frequensolve.workflow.storage import PrivateStore


def snapshot(values):
    values = np.asarray([values], dtype=complex)
    return {
        "version": "pressure-snapshot.v1",
        "real": values.real.tolist(),
        "imaginary": values.imag.tolist(),
        "evidence": {"converged": True, "origin": "synthetic_test"},
        "metadata": {
            "source_ids": [1],
            "source_coordinates": [[0, 0, 0]],
            "receiver_ids": list(range(values.shape[1])),
            "receiver_coordinates": [[i, 0, 0] for i in range(values.shape[1])],
            "coordinate_units": "m",
            "component": "pressure",
            "units": "Pa",
            "frequency": 2,
            "laplace": 0,
            "source_scale": 1,
            "source_encoding": "scalar",
            "selection": "all",
        },
    }


def test_complex_values_zero_phase_and_difference():
    data = snapshot([3 + 4j, 0, -1 + 1e-12j, -1 - 1e-12j])
    result = analyze_pressure(data, snapshot([0, 0, 0, 0]))
    assert result["magnitude"][0][0] == 5
    assert result["principal_phase_radians"][0][1] is None
    assert result["principal_phase_radians"][0][2] > 3.14
    assert result["principal_phase_radians"][0][3] < -3.14
    assert result["comparison"]["complex_rms_difference"] == pytest.approx(
        np.sqrt(27 / 4)
    )


def test_partial_nonfinite_coverage_and_exact_metadata():
    data = snapshot([1, complex(float("nan"), 0)])
    result = analyze_pressure(data, snapshot([1, 2]))
    assert result["finite_count"] == 1
    assert result["comparison"]["coverage"] == 0.5
    for field, value in [
        ("units", "mPa"),
        ("frequency", 20),
        ("source_scale", 2),
        ("receiver_ids", [2, 1]),
    ]:
        other = snapshot([1, 2])
        other["metadata"][field] = value
        with pytest.raises(ValueError, match="match exactly"):
            analyze_pressure(data, other)
    data["evidence"]["converged"] = False
    with pytest.raises(ValueError, match="converged"):
        analyze_pressure(data)


def test_lost_response_restart_preserves_exact_intent(tmp_path):
    calls = []

    def send(operation, arguments):
        calls.append((operation, arguments))
        if len(calls) == 1:
            raise TimeoutError()
        return {"plan_id": "fake-plan"}

    root = tmp_path / "journal"
    journal = IntentJournal(
        PrivateStore(root), {"grant": "fake-grant", "project": "fake-project"}
    )
    with pytest.raises(TimeoutError):
        journal.perform("create", "plan_create", {"project_key": "fake-project"}, send)
    recovered = IntentJournal(PrivateStore(root), journal.binding)
    assert recovered.perform(
        "create", "plan_create", {"project_key": "fake-project"}, send
    ) == {"plan_id": "fake-plan"}
    assert calls[0] == calls[1]
    recovered.perform("create", "plan_create", {"project_key": "fake-project"}, send)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="different request"):
        recovered.perform("create", "plan_create", {"project_key": "changed"}, send)


def test_reconcile_accepted_preparation_retains_identity_on_uncertainty(tmp_path):
    calls = []

    def send(operation, arguments):
        calls.append((operation, arguments))
        if len(calls) == 2:
            raise TimeoutError()
        return {
            "plan_id": "fake-plan",
            "status": "paused" if len(calls) == 1 else "quoted",
        }

    journal = IntentJournal(
        PrivateStore(tmp_path / "journal"), {"project": "fake-project"}
    )
    arguments = {"project_key": "fake-project"}
    assert (
        journal.perform("create", "plan_create", arguments, send)["status"] == "paused"
    )
    with pytest.raises(TimeoutError):
        journal.perform("create", "plan_create", arguments, send, reconcile=True)
    assert journal.inspect("create")["status"] == "pending"
    restarted = IntentJournal(journal.store, journal.binding)
    assert (
        restarted.perform("create", "plan_create", arguments, send)["status"]
        == "quoted"
    )
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]


def auth(tmp_path, request):
    connection = Connection(
        "https://workflow.example.test",
        "https://workflow.example.test/cloud",
        "fake-client",
    )
    return WorkflowAuth(
        connection,
        PrivateStore(tmp_path / "tokens"),
        "sandbox",
        request=request,
        now=lambda: 1000,
    )


def tokens(connection, **changes):
    return {
        "token_type": "Bearer",
        "resource": connection.resource,
        "scope": " ".join(sorted(connection.scopes)),
        "expires_in": 3600,
        "subject": "fake-user",
        "account_id": "fake-account",
        "grant_id": "fake-grant",
        "access_token": "opaque-access",
        "refresh_token": "opaque-refresh",
        **changes,
    }


def test_concurrent_refresh_and_unknown_outcome(tmp_path):
    calls = []
    value = auth(
        tmp_path, lambda *args, **kwargs: calls.append(args) or tokens(value.connection)
    )
    value.save(tokens(value.connection, expires_in=1))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: value.credentials(), range(2)))
    assert len(calls) == 1
    assert results[0] == results[1]
    value.save(tokens(value.connection, expires_in=1))

    def uncertain(*args, **kwargs):
        calls.append(args)
        raise WorkflowError("uncertain")

    value.request = uncertain
    with pytest.raises(WorkflowError):
        value.credentials()
    with pytest.raises(WorkflowError, match="outcome is unknown"):
        value.credentials()
    assert len(calls) == 2


def test_missing_profile_and_scope_fail_without_other_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "unrelated")
    value = auth(
        tmp_path, lambda *args, **kwargs: pytest.fail("No network request expected")
    )
    with pytest.raises(WorkflowError, match="login"):
        value.credentials()
    with pytest.raises(WorkflowError, match="restricted"):
        value.save(tokens(value.connection, scope=value.connection.resource + "/read"))
    with pytest.raises(ValueError):
        Connection(
            "https://workflow.example.test",
            "https://foreign.example.test/cloud",
            "fake-client",
        )


def test_symlink_private_storage_denied(tmp_path):
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(OSError):
        with PrivateStore(link).locked("record"):
            pytest.fail("Symlink must not be opened")
