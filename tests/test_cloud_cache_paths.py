import base64
import json
import logging
from datetime import datetime, timedelta

import pytest

from frequensolve._cloud_credentials import CREDENTIAL_CACHE_BINDING_KEY
from frequensolve.orchestrator.sites.aws.cache_paths import (
    cloud_config_cache_path,
    cloud_credentials_path,
    legacy_config_cache_path,
    legacy_credentials_path,
)


def _jwt(payload):
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).rstrip(b"=")
    return f"header.{encoded.decode()}.signature"


def _cached_tokens(
    *,
    user_pool_id="pool",
    client_id="client",
    id_overrides=None,
    access_overrides=None,
):
    issuer = f"https://cognito-idp.us-east-1.amazonaws.com/{user_pool_id}"
    id_claims = {
        "iss": issuer,
        "token_use": "id",
        "aud": client_id,
    }
    id_claims.update(id_overrides or {})
    access_claims = {
        "iss": issuer,
        "token_use": "access",
        "client_id": client_id,
    }
    access_claims.update(access_overrides or {})
    return {
        "email": "user@example.com",
        "id_token": _jwt(id_claims),
        "access_token": _jwt(access_claims),
        "refresh_token": "refresh",
        "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
    }


def test_cloud_cache_paths_are_grouped_under_cloud_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert cloud_credentials_path() == (
        tmp_path / ".frequensolve" / "cloud" / "credentials"
    )
    assert cloud_config_cache_path("localhost:5173/api") == (
        tmp_path / ".frequensolve" / "cloud" / "config_localhost_5173_api.json"
    )
    assert legacy_credentials_path() == tmp_path / ".frequensolve" / "credentials"
    assert legacy_config_cache_path("localhost:5173/api") == (
        tmp_path / ".frequensolve" / "config_localhost_5173_api.json"
    )


def test_cloud_cache_paths_use_frequensolve_home_override(monkeypatch, tmp_path):
    storage_root = tmp_path / "fs-user-storage"
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(storage_root))

    assert cloud_credentials_path() == storage_root / "cloud" / "credentials"
    assert cloud_config_cache_path("localhost:5173/api") == (
        storage_root / "cloud" / "config_localhost_5173_api.json"
    )
    assert legacy_credentials_path() == storage_root / "credentials"
    assert legacy_config_cache_path("localhost:5173/api") == (
        storage_root / "config_localhost_5173_api.json"
    )


def test_cloud_profile_credential_paths_are_isolated_and_opaque(monkeypatch, tmp_path):
    storage_root = tmp_path / "fs-user-storage"
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(storage_root))

    sandbox_path = cloud_credentials_path("cloud_sandbox_test")
    production_path = cloud_credentials_path("cloud")

    assert sandbox_path.parent == storage_root / "cloud"
    assert production_path.parent == storage_root / "cloud"
    assert sandbox_path != production_path
    assert "cloud_sandbox_test" not in sandbox_path.name
    assert sandbox_path.name.endswith(".json")


def test_aws_site_config_reads_legacy_cache_and_migrates_it(monkeypatch, tmp_path):
    pytest.importorskip("boto3")
    pytest.importorskip("requests")

    from frequensolve.orchestrator.sites.aws import AWSSiteConfig

    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    config_data = {
        "auth": {
            "userPoolId": "pool",
            "clientId": "client",
            "identityPoolId": "identity",
        },
        "api": {"graphqlUrl": "https://api.example/graphql"},
        "region": "us-east-1",
    }
    legacy_path = legacy_config_cache_path("dev.example")
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps(config_data))

    assert AWSSiteConfig._fetch_config_from_domain("dev.example") == config_data
    assert cloud_config_cache_path("dev.example").exists()


def test_cognito_auth_saves_tokens_in_cloud_directory(monkeypatch, tmp_path):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())

    auth = cognito.CognitoAuth(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
    )
    tokens = {
        "email": "user@example.com",
        "id_token": "token",
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
    }

    auth.save_tokens(tokens)

    assert auth.credentials_path == cloud_credentials_path()
    assert cloud_credentials_path().exists()
    assert not legacy_credentials_path().exists()
    assert (cloud_credentials_path().stat().st_mode & 0o777) == 0o600


def test_cognito_auth_reads_legacy_tokens_and_migrates_them(monkeypatch, tmp_path):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())
    tokens = _cached_tokens()
    legacy_path = legacy_credentials_path()
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps(tokens))

    auth = cognito.CognitoAuth(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
    )

    assert auth.get_cached_tokens() == tokens
    assert cloud_credentials_path().exists()


def test_cognito_auth_cached_token_reads_are_debug_logs(monkeypatch, tmp_path, caplog):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())
    tokens = _cached_tokens()
    auth = cognito.CognitoAuth(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
    )
    auth.save_tokens(tokens)

    with caplog.at_level(logging.INFO, logger=cognito.__name__):
        assert auth.get_cached_tokens() == tokens

    assert not [
        record
        for record in caplog.records
        if record.levelno == logging.INFO
        and record.message == "Using cached credentials"
    ]


def test_profile_bound_cognito_caches_coexist_for_same_cognito_client(
    monkeypatch, tmp_path
):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    monkeypatch.setenv("FREQUENSOLVE_HOME", str(tmp_path / "frequensolve-home"))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())

    sandbox_auth = cognito.CognitoAuth(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
        profile_name="cloud_sandbox_test",
        domain="sandbox.example",
    )
    production_auth = cognito.CognitoAuth(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
        profile_name="cloud",
        domain="app.example",
    )
    sandbox_tokens = {**_cached_tokens(), "email": "sandbox@example.com"}
    production_tokens = {**_cached_tokens(), "email": "production@example.com"}

    sandbox_auth.save_tokens({**sandbox_tokens, "password": "must-not-be-written"})
    production_auth.save_tokens(production_tokens)

    assert sandbox_auth.credentials_path != production_auth.credentials_path
    assert sandbox_auth.get_cached_tokens() == sandbox_tokens
    assert production_auth.get_cached_tokens() == production_tokens
    sandbox_document = json.loads(sandbox_auth.credentials_path.read_text())
    production_document = json.loads(production_auth.credentials_path.read_text())
    assert sandbox_document[CREDENTIAL_CACHE_BINDING_KEY]["profile"] == (
        "cloud_sandbox_test"
    )
    assert production_document[CREDENTIAL_CACHE_BINDING_KEY]["profile"] == "cloud"
    assert "password" not in sandbox_document


def test_profile_bound_cognito_ignores_unbound_cache_without_migrating(
    monkeypatch, tmp_path
):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    storage_root = tmp_path / "frequensolve-home"
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(storage_root))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())
    legacy_tokens = _cached_tokens()
    unbound_path = cloud_credentials_path()
    unbound_path.write_text(json.dumps(legacy_tokens))
    original_bytes = unbound_path.read_bytes()
    auth = cognito.CognitoAuth(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
        profile_name="cloud_sandbox_test",
        domain="sandbox.example",
    )

    with pytest.raises(ValueError, match="No cached credentials"):
        auth.get_cached_tokens()

    assert not auth.credentials_path.exists()
    assert unbound_path.read_bytes() == original_bytes


def test_profile_bound_cognito_rejects_changed_binding_without_deleting_cache(
    monkeypatch, tmp_path
):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    monkeypatch.setenv("FREQUENSOLVE_HOME", str(tmp_path / "frequensolve-home"))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())
    original_auth = cognito.CognitoAuth(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
        profile_name="cloud_sandbox_test",
        domain="sandbox.example",
    )
    original_auth.save_tokens(_cached_tokens())
    original_bytes = original_auth.credentials_path.read_bytes()
    changed_auth = cognito.CognitoAuth(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="changed-identity",
        profile_name="cloud_sandbox_test",
        domain="sandbox.example",
    )

    with pytest.raises(ValueError, match="do not match the selected"):
        changed_auth.get_cached_tokens()

    assert changed_auth.credentials_path.read_bytes() == original_bytes


@pytest.mark.parametrize(
    ("id_overrides", "access_overrides"),
    (
        (
            {
                "iss": "https://cognito-idp.us-east-1.amazonaws.com/other-pool",
            },
            None,
        ),
        (
            None,
            {
                "iss": "https://cognito-idp.us-east-1.amazonaws.com/other-pool",
            },
        ),
        ({"token_use": "access"}, None),
        (None, {"token_use": "id"}),
        ({"aud": "other-client"}, None),
        (None, {"client_id": "other-client"}),
    ),
)
def test_cognito_auth_rejects_tokens_from_another_cloud_profile(
    monkeypatch, tmp_path, id_overrides, access_overrides
):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())
    auth = cognito.CognitoAuth(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
    )
    tokens = _cached_tokens(
        id_overrides=id_overrides,
        access_overrides=access_overrides,
    )
    auth.save_tokens(tokens)

    with pytest.raises(ValueError) as exc_info:
        auth.get_cached_tokens()

    message = str(exc_info.value)
    assert "do not match the selected FrequenSol Cloud profile" in message
    assert tokens["email"] not in message
    assert tokens["id_token"] not in message


def test_cognito_issuer_uses_the_aws_partition_dns_suffix():
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws.cognito import _cognito_issuer

    assert _cognito_issuer(
        region="cn-north-1",
        user_pool_id="cn-north-1_TestPool",
    ) == ("https://cognito-idp.cn-north-1.amazonaws.com.cn/cn-north-1_TestPool")
