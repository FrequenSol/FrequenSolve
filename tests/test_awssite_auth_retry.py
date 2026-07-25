import base64
import json
from datetime import datetime, timedelta

import pytest


class FakeGraphQLClient:
    def __init__(self, api_url, auth):
        self.api_url = api_url
        self.auth = auth

    def get_storage_stack_info(self):
        return {"bucketName": "bucket"}


class FakeSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def client(self, service_name, region_name=None):
        return {"service_name": service_name, "region_name": region_name}


def _jwt(payload):
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).rstrip(b"=")
    return f"header.{encoded.decode()}.signature"


def _install_awssite_fakes(monkeypatch, auth_cls):
    pytest.importorskip("boto3")
    pytest.importorskip("requests")

    from frequensolve.orchestrator.sites.aws import aws, cognito, graphql_client

    config = aws.AWSSiteConfig(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
        api_url="https://api.example/graphql",
        region="us-east-1",
        domain="app.example",
    )
    monkeypatch.setattr(
        aws.AWSSiteConfig,
        "from_domain",
        staticmethod(lambda domain=None: config),
    )
    monkeypatch.setattr(cognito, "CognitoAuth", auth_cls)
    monkeypatch.setattr(graphql_client, "GraphQLClient", FakeGraphQLClient)
    monkeypatch.setattr(aws.boto3, "Session", FakeSession)
    return aws


def _expired_refresh_auth_class(tmp_path):
    class ExpiredRefreshAuth:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.credentials_path = tmp_path / "credentials"
            self.credentials_path.write_text("expired-cache")
            self.login_calls = []
            self.clear_calls = 0
            self.get_aws_credentials_calls = 0
            self.logged_in = False
            self.instances.append(self)

        def get_cached_tokens(self):
            return {
                "email": "cached@example.com",
                "id_token": "old-id-token",
                "access_token": "old-access-token",
                "refresh_token": "old-refresh-token",
                "expires_at": (datetime.now() - timedelta(hours=1)).isoformat(),
            }

        def login(self, email, password):
            self.login_calls.append((email, password))
            self.logged_in = True
            self.credentials_path.write_text(f"fresh-cache:{email}")

        def get_aws_credentials(self):
            self.get_aws_credentials_calls += 1
            if not self.logged_in:
                raise ValueError("Refresh token expired. Please login again.")
            return {
                "AccessKeyId": "access",
                "SecretKey": "secret",
                "SessionToken": "session",
                "Expiration": "2030-01-01T00:00:00",
                "IdentityId": "identity-id",
            }

        def clear_cached_tokens(self):
            self.clear_calls += 1
            self.credentials_path.unlink(missing_ok=True)

    return ExpiredRefreshAuth


def _valid_cached_auth_class(tmp_path):
    class ValidCachedAuth:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.credentials_path = tmp_path / "credentials"
            self.credentials_path.write_text("valid-cache")
            self.login_calls = []
            self.instances.append(self)

        def get_cached_tokens(self):
            return {
                "email": "cached@example.com",
                "id_token": "cached-id-token",
                "access_token": "cached-access-token",
                "refresh_token": "cached-refresh-token",
                "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
            }

        def login(self, email, password):
            self.login_calls.append((email, password))
            self.credentials_path.write_text(f"fresh-cache:{email}")

        def get_aws_credentials(self):
            return {
                "AccessKeyId": "access",
                "SecretKey": "secret",
                "SessionToken": "session",
                "Expiration": "2030-01-01T00:00:00",
                "IdentityId": "identity-id",
            }

        def clear_cached_tokens(self):
            self.credentials_path.unlink(missing_ok=True)

    return ValidCachedAuth


def test_aws_site_preserves_positional_verbose_and_reuses_valid_cache(
    monkeypatch, tmp_path
):
    auth_cls = _valid_cached_auth_class(tmp_path)
    aws = _install_awssite_fakes(monkeypatch, auth_cls)

    site = aws.AWSSite("app.example", None, None, False, True)

    auth = auth_cls.instances[0]
    assert site.verbose is True
    assert auth.login_calls == []
    assert auth.credentials_path.read_text() == "valid-cache"


def test_aws_site_reauthenticates_with_passed_credentials_without_deleting_cache(
    monkeypatch, tmp_path
):
    auth_cls = _expired_refresh_auth_class(tmp_path)
    aws = _install_awssite_fakes(monkeypatch, auth_cls)

    site = aws.AWSSite(
        domain="app.example",
        email="fresh@example.com",
        password="fresh-password",
    )

    auth = auth_cls.instances[0]
    assert auth.login_calls == [("fresh@example.com", "fresh-password")]
    assert auth.clear_calls == 0
    assert auth.credentials_path.read_text() == "fresh-cache:fresh@example.com"
    assert auth.get_aws_credentials_calls == 2
    assert site.config.s3_bucket == "bucket"


def test_aws_site_interactive_reauthenticates_without_deleting_cache(
    monkeypatch, tmp_path
):
    auth_cls = _expired_refresh_auth_class(tmp_path)
    aws = _install_awssite_fakes(monkeypatch, auth_cls)
    monkeypatch.setattr("builtins.input", lambda prompt: "prompted@example.com")
    monkeypatch.setattr(aws.getpass, "getpass", lambda prompt: "prompted-password")

    aws.AWSSite(domain="app.example", interactive=True)

    auth = auth_cls.instances[0]
    assert auth.login_calls == [("prompted@example.com", "prompted-password")]
    assert auth.clear_calls == 0
    assert auth.credentials_path.read_text() == "fresh-cache:prompted@example.com"


def test_aws_site_noninteractive_expired_refresh_error_names_relogin_options(
    monkeypatch, tmp_path
):
    auth_cls = _expired_refresh_auth_class(tmp_path)
    aws = _install_awssite_fakes(monkeypatch, auth_cls)

    with pytest.raises(RuntimeError) as exc_info:
        aws.AWSSite(domain="app.example")

    message = str(exc_info.value)
    auth = auth_cls.instances[0]
    assert "cached FrequenSol cloud login expired" in message
    assert str(auth.credentials_path) in message
    assert 'fs.Site(profile="cloud", interactive=True)' in message
    assert auth.clear_calls == 0
    assert auth.credentials_path.read_text() == "expired-cache"


@pytest.mark.parametrize(
    ("cached_issuer", "cached_client", "force_login"),
    (
        (
            "https://cognito-idp.us-east-1.amazonaws.com/other-pool",
            "other-client",
            False,
        ),
        (
            "https://cognito-idp.us-east-1.amazonaws.com/pool",
            "client",
            True,
        ),
    ),
)
def test_aws_site_interactive_login_replaces_an_unusable_or_forced_cache(
    monkeypatch,
    tmp_path,
    cached_issuer,
    cached_client,
    force_login,
):
    pytest.importorskip("boto3")
    pytest.importorskip("requests")

    from frequensolve.orchestrator.sites.aws import aws, cognito, graphql_client
    from frequensolve.orchestrator.sites.aws.cache_paths import cloud_credentials_path

    storage_root = tmp_path / "frequensolve-home"
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(storage_root))
    credentials_path = cloud_credentials_path()
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    credentials_path.write_text(
        json.dumps(
            {
                "email": "cached@example.com",
                "id_token": _jwt(
                    {
                        "iss": cached_issuer,
                        "token_use": "id",
                        "aud": cached_client,
                    }
                ),
                "access_token": _jwt(
                    {
                        "iss": cached_issuer,
                        "token_use": "access",
                        "client_id": cached_client,
                    }
                ),
                "refresh_token": "cached-refresh",
                "expires_at": (datetime.now() + timedelta(hours=1)).isoformat(),
            }
        )
    )

    expected_issuer = "https://cognito-idp.us-east-1.amazonaws.com/pool"
    login_calls = []

    class FakeCognitoClient:
        def initiate_auth(self, **kwargs):
            login_calls.append(kwargs)
            return {
                "AuthenticationResult": {
                    "IdToken": _jwt(
                        {
                            "iss": expected_issuer,
                            "token_use": "id",
                            "aud": "client",
                        }
                    ),
                    "AccessToken": _jwt(
                        {
                            "iss": expected_issuer,
                            "token_use": "access",
                            "client_id": "client",
                        }
                    ),
                    "RefreshToken": "fresh-refresh",
                    "ExpiresIn": 3600,
                }
            }

    class FakeIdentityClient:
        def get_id(self, **kwargs):
            return {"IdentityId": "us-east-1:test-identity"}

        def get_credentials_for_identity(self, **kwargs):
            return {
                "Credentials": {
                    "AccessKeyId": "access",
                    "SecretKey": "secret",
                    "SessionToken": "session",
                    "Expiration": datetime(2030, 1, 1),
                }
            }

    def fake_boto_client(service_name, **kwargs):
        if service_name == "cognito-idp":
            return FakeCognitoClient()
        if service_name == "cognito-identity":
            return FakeIdentityClient()
        raise AssertionError(f"Unexpected service: {service_name}")

    config = aws.AWSSiteConfig(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
        api_url="https://api.example/graphql",
        region="us-east-1",
        domain="app.example",
    )
    monkeypatch.setattr(
        aws.AWSSiteConfig,
        "from_domain",
        staticmethod(lambda domain=None: config),
    )
    monkeypatch.setattr(cognito.boto3, "client", fake_boto_client)
    monkeypatch.setattr(graphql_client, "GraphQLClient", FakeGraphQLClient)
    monkeypatch.setattr(aws.boto3, "Session", FakeSession)
    monkeypatch.setattr("builtins.input", lambda prompt: "fresh@example.com")
    monkeypatch.setattr(aws.getpass, "getpass", lambda prompt: "fresh-password")

    site = aws.AWSSite(
        domain="app.example",
        interactive=True,
        force_login=force_login,
    )

    assert len(login_calls) == 1
    assert login_calls[0]["AuthParameters"] == {
        "USERNAME": "fresh@example.com",
        "PASSWORD": "fresh-password",
    }
    assert json.loads(credentials_path.read_text())["email"] == "fresh@example.com"
    assert site.config.s3_bucket == "bucket"


def test_aws_site_failed_forced_login_preserves_existing_cache(monkeypatch, tmp_path):
    pytest.importorskip("boto3")
    pytest.importorskip("requests")
    from botocore.exceptions import ClientError

    from frequensolve.orchestrator.sites.aws import aws, cognito
    from frequensolve.orchestrator.sites.aws.cache_paths import cloud_credentials_path

    storage_root = tmp_path / "frequensolve-home"
    monkeypatch.setenv("FREQUENSOLVE_HOME", str(storage_root))
    credentials_path = cloud_credentials_path()
    credentials_path.parent.mkdir(parents=True, exist_ok=True)
    original_cache = b'{"email":"cached@example.com","refresh_token":"cached-refresh"}'
    credentials_path.write_bytes(original_cache)

    class RejectingCognitoClient:
        def initiate_auth(self, **kwargs):
            raise ClientError(
                {
                    "Error": {
                        "Code": "NotAuthorizedException",
                        "Message": "Incorrect username or password.",
                    }
                },
                "InitiateAuth",
            )

    class UnusedIdentityClient:
        pass

    def fake_boto_client(service_name, **kwargs):
        if service_name == "cognito-idp":
            return RejectingCognitoClient()
        if service_name == "cognito-identity":
            return UnusedIdentityClient()
        raise AssertionError(f"Unexpected service: {service_name}")

    config = aws.AWSSiteConfig(
        user_pool_id="pool",
        client_id="client",
        identity_pool_id="identity",
        api_url="https://api.example/graphql",
        region="us-east-1",
        domain="app.example",
    )
    monkeypatch.setattr(
        aws.AWSSiteConfig,
        "from_domain",
        staticmethod(lambda domain=None: config),
    )
    monkeypatch.setattr(cognito.boto3, "client", fake_boto_client)

    with pytest.raises(ValueError, match="Invalid email or password"):
        aws.AWSSite(
            domain="app.example",
            email="fresh@example.com",
            password="wrong-password",
            force_login=True,
        )

    assert credentials_path.read_bytes() == original_cache
