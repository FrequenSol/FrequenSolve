import logging
from datetime import datetime, timezone

from frequensolve.orchestrator.sites.aws import cognito


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
