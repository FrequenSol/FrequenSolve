"""Incomplete provider sign-in must preserve the user's previous token cache."""

from unittest.mock import Mock

import pytest

pytest.importorskip("boto3")
from botocore.exceptions import ClientError

from frequensolve.orchestrator.sites.aws.cognito import CognitoAuth


def _auth(response):
    auth = object.__new__(CognitoAuth)
    auth.client_id = "fixture-client"
    auth.cognito_client = Mock()
    auth.cognito_client.initiate_auth.return_value = response
    auth.save_tokens = Mock()
    auth.get_cached_tokens = Mock(
        return_value={"email": "existing@example.test", "refresh_token": "old-refresh"}
    )
    return auth


def test_first_password_challenge_has_web_recovery_without_saving_tokens():
    auth = _auth(
        {
            "ChallengeName": "NEW_PASSWORD_REQUIRED",
            "Session": "private-provider-session",
            "ChallengeParameters": {"requiredAttributes": "private-attributes"},
        }
    )
    with pytest.raises(ValueError, match="welcome email") as error:
        auth.login("new@example.test", "temporary-password")
    assert "Then retry Python sign-in" in str(error.value)
    assert "private" not in str(error.value)
    auth.save_tokens.assert_not_called()
    assert auth.cognito_client.method_calls == [
        (
            "initiate_auth",
            (),
            {
                "ClientId": "fixture-client",
                "AuthFlow": "USER_PASSWORD_AUTH",
                "AuthParameters": {
                    "USERNAME": "new@example.test",
                    "PASSWORD": "temporary-password",
                },
            },
        )
    ]


@pytest.mark.parametrize(
    "challenge",
    [
        "MFA_SETUP",
        "SMS_MFA",
        "SOFTWARE_TOKEN_MFA",
        "SELECT_MFA_TYPE",
        "EMAIL_OTP",
        "WEB_AUTHN",
        "CUSTOM_CHALLENGE",
        "unknown-private-challenge",
    ],
)
def test_unsupported_challenge_explains_python_limit_without_exposing_payload(
    challenge,
):
    auth = _auth({"ChallengeName": challenge, "Session": "private-session"})
    with pytest.raises(ValueError, match="Python client cannot complete") as error:
        auth.login("user@example.test", "password")
    assert "browser sign-in" in str(error.value)
    assert challenge not in str(error.value)
    assert "private-session" not in str(error.value)
    auth.save_tokens.assert_not_called()


@pytest.mark.parametrize(
    "code, recovery",
    [
        ("PasswordResetRequiredException", "password recovery"),
        ("UserNotConfirmedException", "welcome email"),
    ],
)
def test_provider_account_setup_errors_have_actionable_recovery(code, recovery):
    auth = _auth({})
    auth.cognito_client.initiate_auth.side_effect = ClientError(
        {"Error": {"Code": code, "Message": "private-provider-message"}}, "InitiateAuth"
    )
    with pytest.raises(ValueError, match=recovery) as error:
        auth.login("user@example.test", "password")
    assert "private-provider-message" not in str(error.value)
    auth.save_tokens.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"AuthenticationResult": {}},
        {
            "AuthenticationResult": {"IdToken": "id", "AccessToken": "access"},
        },
    ],
)
def test_incomplete_authentication_response_does_not_replace_cache(response):
    auth = _auth(response)
    with pytest.raises(RuntimeError, match="did not return complete credentials"):
        auth.login("user@example.test", "password")
    auth.save_tokens.assert_not_called()


def test_completed_login_saves_tokens_and_refresh_keeps_original_refresh_token():
    auth = _auth(
        {
            "AuthenticationResult": {
                "IdToken": "new-id",
                "AccessToken": "new-access",
                "RefreshToken": "new-refresh",
            }
        }
    )
    tokens = auth.login("user@example.test", "password")
    assert tokens["refresh_token"] == "new-refresh"
    auth.save_tokens.assert_called_once_with(tokens)
    auth.save_tokens.reset_mock()
    auth.cognito_client.initiate_auth.return_value = {
        "AuthenticationResult": {
            "IdToken": "refreshed-id",
            "AccessToken": "refreshed-access",
        }
    }
    refreshed = auth.refresh_tokens()
    assert refreshed["refresh_token"] == "old-refresh"
    auth.save_tokens.assert_called_once_with(refreshed)


def test_refresh_challenge_preserves_existing_cache():
    auth = _auth({"ChallengeName": "SOFTWARE_TOKEN_MFA"})
    with pytest.raises(ValueError, match="additional sign-in verification"):
        auth.refresh_tokens()
    auth.save_tokens.assert_not_called()
