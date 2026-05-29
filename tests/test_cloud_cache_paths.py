import base64
import json
import logging
from datetime import datetime, timedelta

import pytest

from frequensolve.orchestrator.sites.aws.cache_paths import (
    cloud_config_cache_path,
    cloud_credentials_path,
    legacy_config_cache_path,
    legacy_credentials_path,
)


def _jwt_with_issuer(user_pool_id: str) -> str:
    payload = json.dumps(
        {
            "iss": f"https://cognito-idp.us-east-1.amazonaws.com/{user_pool_id}",
            "token_use": "id",
        }
    ).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"header.{encoded}.signature"


def test_cloud_cache_paths_are_grouped_under_cloud_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert cloud_credentials_path() == (
        tmp_path / ".frequensolve" / "cloud" / "credentials"
    )
    assert cloud_credentials_path("sandbox") == (
        tmp_path / ".frequensolve" / "cloud" / "credentials_sandbox"
    )
    assert cloud_credentials_path("localhost:5173") == (
        tmp_path / ".frequensolve" / "cloud" / "credentials_localhost_5173"
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


def test_cognito_auth_saves_named_tokens_in_profile_cache(monkeypatch, tmp_path):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())

    auth = cognito.CognitoAuth(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
        credential_cache_name="sandbox",
    )
    tokens = {
        "email": "user@example.com",
        "id_token": "token",
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
    }

    auth.save_tokens(tokens)

    assert auth.credentials_path == cloud_credentials_path("sandbox")
    assert cloud_credentials_path("sandbox").exists()
    assert not cloud_credentials_path().exists()
    assert (cloud_credentials_path("sandbox").stat().st_mode & 0o777) == 0o600


def test_cognito_auth_migrates_matching_shared_cloud_tokens_to_profile_cache(
    monkeypatch, tmp_path
):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())
    tokens = {
        "email": "user@example.com",
        "id_token": _jwt_with_issuer("pool-dev"),
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
    }
    cloud_credentials_path().write_text(json.dumps(tokens))

    auth = cognito.CognitoAuth(
        user_pool_id="pool-dev",
        client_id="client",
        identity_pool_id="identity",
        credential_cache_name="cloud",
    )

    assert auth.get_cached_tokens() == tokens
    assert cloud_credentials_path("cloud").exists()
    assert json.loads(cloud_credentials_path("cloud").read_text()) == tokens


def test_cognito_auth_ignores_mismatched_shared_cloud_tokens(monkeypatch, tmp_path):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())
    tokens = {
        "email": "sandbox@example.com",
        "id_token": _jwt_with_issuer("pool-sandbox"),
    }
    cloud_credentials_path().write_text(json.dumps(tokens))

    auth = cognito.CognitoAuth(
        user_pool_id="pool-dev",
        client_id="client",
        identity_pool_id="identity",
        credential_cache_name="cloud",
    )

    with pytest.raises(ValueError) as exc_info:
        auth.get_cached_tokens()

    assert str(cloud_credentials_path("cloud")) in str(exc_info.value)
    assert 'fs.Site(profile="cloud", interactive=True)' in str(exc_info.value)
    assert not cloud_credentials_path("cloud").exists()


def test_cognito_auth_reads_legacy_tokens_and_migrates_them(monkeypatch, tmp_path):
    pytest.importorskip("boto3")

    from frequensolve.orchestrator.sites.aws import cognito

    monkeypatch.delenv("FREQUENSOLVE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cognito.boto3, "client", lambda *args, **kwargs: object())
    tokens = {"email": "user@example.com", "id_token": "token"}
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
    tokens = {"email": "user@example.com", "id_token": "token"}
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
