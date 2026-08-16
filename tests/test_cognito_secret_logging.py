import logging
from datetime import datetime, timezone

import pytest
from botocore.exceptions import ClientError

from frequensolve.orchestrator.sites.aws import cognito


def _throttled(operation_name):
    return ClientError(
        {
            "Error": {
                "Code": "TooManyRequestsException",
                "Message": "Rate exceeded",
            }
        },
        operation_name,
    )


def test_temporary_aws_credentials_and_id_token_are_not_logged(caplog):
    id_token = "sensitive-id-token"
    access_key = "sensitive-access-key"
    secret_key = "sensitive-secret-key"
    session_token = "sensitive-session-token"

    class IdentityClient:
        def get_id(self, **kwargs):
            return {"IdentityId": "us-east-1:identity"}

        def get_credentials_for_identity(self, **kwargs):
            return {
                "Credentials": {
                    "AccessKeyId": access_key,
                    "SecretKey": secret_key,
                    "SessionToken": session_token,
                    "Expiration": datetime.now(timezone.utc),
                }
            }

    instance = object.__new__(cognito.CognitoAuth)
    instance.region = "us-east-1"
    instance.user_pool_id = "pool"
    instance.identity_pool_id = "identity-pool"
    instance.identity_client = IdentityClient()
    instance.get_id_token = lambda: id_token

    with caplog.at_level(logging.DEBUG, logger=cognito.__name__):
        result = instance.get_aws_credentials()

    assert result["AccessKeyId"] == access_key
    assert result["SecretKey"] == secret_key
    assert result["SessionToken"] == session_token
    for secret in (id_token, access_key, secret_key, session_token):
        assert secret not in caplog.text
    assert "Credentials Response" not in caplog.text


def test_identity_exchange_retries_transient_throttling(monkeypatch):
    attempts = 0
    sleeps = []

    class IdentityClient:
        def get_id(self, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise _throttled("GetId")
            return {"IdentityId": "us-east-1:identity"}

        def get_credentials_for_identity(self, **kwargs):
            return {
                "Credentials": {
                    "AccessKeyId": "access",
                    "SecretKey": "secret",
                    "SessionToken": "session",
                    "Expiration": datetime.now(timezone.utc),
                }
            }

    monkeypatch.setattr(cognito.random, "uniform", lambda lower, upper: upper)
    monkeypatch.setattr(cognito.time, "sleep", sleeps.append)

    instance = object.__new__(cognito.CognitoAuth)
    instance.region = "us-east-1"
    instance.user_pool_id = "pool"
    instance.identity_pool_id = "identity-pool"
    instance.identity_client = IdentityClient()
    instance.get_id_token = lambda: "id-token"

    result = instance.get_aws_credentials()

    assert attempts == 3
    assert sleeps == [0.25, 0.5]
    assert result["IdentityId"] == "us-east-1:identity"


def test_identity_exchange_reports_clear_message_after_throttling(monkeypatch):
    attempts = 0

    class IdentityClient:
        def get_id(self, **kwargs):
            return {"IdentityId": "us-east-1:identity"}

        def get_credentials_for_identity(self, **kwargs):
            nonlocal attempts
            attempts += 1
            raise _throttled("GetCredentialsForIdentity")

    monkeypatch.setattr(cognito.random, "uniform", lambda lower, upper: lower)
    monkeypatch.setattr(cognito.time, "sleep", lambda delay: None)

    instance = object.__new__(cognito.CognitoAuth)
    instance.region = "us-east-1"
    instance.user_pool_id = "pool"
    instance.identity_pool_id = "identity-pool"
    instance.identity_client = IdentityClient()
    instance.get_id_token = lambda: "id-token"

    with pytest.raises(RuntimeError, match="temporarily busy") as exc_info:
        instance.get_aws_credentials()

    assert attempts == cognito._IDENTITY_MAX_ATTEMPTS
    assert "TooManyRequestsException" not in str(exc_info.value)
    assert "Wait a moment, then retry" in str(exc_info.value)
