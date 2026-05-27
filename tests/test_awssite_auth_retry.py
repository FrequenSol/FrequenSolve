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
