"""Offline staging recovery, isolated from simulation admission and execution."""

import traceback
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("boto3")
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError

from frequensolve.orchestrator.sites.aws.aws import AWSSite


def provider_error(code, wrapped=False):
    error = ClientError(
        {"Error": {"Code": code, "Message": "private-key synthetic-secret"}},
        "PutObject",
    )
    if wrapped:
        try:
            raise error
        except ClientError:
            try:
                raise S3UploadFailedError("private-key synthetic-secret")
            except S3UploadFailedError as wrapper:
                return wrapper
    return error


def site_with_clients(old, new, cognito=True):
    site = AWSSite.__new__(AWSSite)
    site.config = SimpleNamespace(s3_bucket="tenant-bucket", s3_prefix="tenant")
    site.s3_client = old
    site.cognito_auth = object() if cognito else None
    site.graphql_client = Mock()
    site._refresh_s3_credentials = Mock(
        side_effect=lambda: setattr(site, "s3_client", new)
    )
    return site


@pytest.mark.parametrize("method", ["put", "sync_s3"])
@pytest.mark.parametrize("code", ["ExpiredToken", "InvalidToken"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_expired_upload_refreshes_same_identity_and_retries_one_file(
    tmp_path, method, code, wrapped
):
    source = tmp_path / "input.json"
    source.write_text("input")
    old = SimpleNamespace(upload_file=Mock(side_effect=provider_error(code, wrapped)))
    new = SimpleNamespace(upload_file=Mock())
    site = site_with_clients(old, new)

    getattr(site, method)(source, "project/input.json")

    site._refresh_s3_credentials.assert_called_once_with()
    assert old.upload_file.call_args == new.upload_file.call_args
    assert new.upload_file.call_count == 1
    assert new.upload_file.call_args.args[1] == "tenant-bucket"
    assert site.graphql_client.mock_calls == []


@pytest.mark.parametrize("method", ["put", "sync_s3"])
def test_directory_refresh_only_replays_expired_file(tmp_path, method):
    directory = tmp_path / "inputs"
    directory.mkdir()
    for name in ("a", "b", "c"):
        (directory / name).write_text(name)
    old = SimpleNamespace(
        upload_file=Mock(side_effect=[None, provider_error("ExpiredToken", True)])
    )
    new = SimpleNamespace(upload_file=Mock())
    site = site_with_clients(old, new)

    getattr(site, method)(directory, "project/inputs")

    assert old.upload_file.call_count == 2
    assert new.upload_file.call_count == 2
    assert old.upload_file.call_args_list[1] == new.upload_file.call_args_list[0]
    assert old.upload_file.call_args_list[0] not in new.upload_file.call_args_list
    assert (
        len(
            {
                c.args[0]
                for c in old.upload_file.call_args_list + new.upload_file.call_args_list
            }
        )
        == 3
    )
    site._refresh_s3_credentials.assert_called_once_with()
    assert site.graphql_client.mock_calls == []


@pytest.mark.parametrize("method", ["put", "sync_s3"])
def test_repeated_expiration_stops_after_one_refresh(tmp_path, method):
    source = tmp_path / "input"
    source.write_text("input")
    old = SimpleNamespace(upload_file=Mock(side_effect=provider_error("ExpiredToken")))
    new = SimpleNamespace(
        upload_file=Mock(side_effect=provider_error("InvalidToken", True))
    )
    site = site_with_clients(old, new)

    with pytest.raises(RuntimeError, match="credentials_expired") as caught:
        getattr(site, method)(source, "project/input")

    site._refresh_s3_credentials.assert_called_once_with()
    assert old.upload_file.call_count == new.upload_file.call_count == 1
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "synthetic-secret" not in "".join(traceback.format_exception(caught.value))
    assert site.graphql_client.mock_calls == []


@pytest.mark.parametrize("method", ["put", "sync_s3"])
@pytest.mark.parametrize(
    "code,category",
    [
        ("AccessDenied", "storage_access_denied"),
        ("400", "upload_failed"),
        ("SlowDown", "upload_failed"),
        ("synthetic-secret", "upload_failed"),
    ],
)
def test_other_provider_errors_never_refresh_or_expose_provider_code(
    tmp_path, method, code, category
):
    source = tmp_path / "input"
    source.write_text("input")
    old = SimpleNamespace(upload_file=Mock(side_effect=provider_error(code, True)))
    new = SimpleNamespace(upload_file=Mock())
    site = site_with_clients(old, new)

    with pytest.raises(RuntimeError, match=category) as caught:
        getattr(site, method)(source, "project/input")

    site._refresh_s3_credentials.assert_not_called()
    new.upload_file.assert_not_called()
    assert old.upload_file.call_count == 1
    assert caught.value.__context__ is None
    assert "synthetic-secret" not in "".join(traceback.format_exception(caught.value))
    assert site.graphql_client.mock_calls == []


@pytest.mark.parametrize("method", ["put", "sync_s3"])
def test_non_cognito_site_does_not_switch_credential_provider(tmp_path, method):
    source = tmp_path / "input"
    source.write_text("input")
    old = SimpleNamespace(upload_file=Mock(side_effect=provider_error("ExpiredToken")))
    new = SimpleNamespace(upload_file=Mock())
    site = site_with_clients(old, new, cognito=False)
    with pytest.raises(RuntimeError, match="credentials_expired"):
        getattr(site, method)(source, "project/input")
    site._refresh_s3_credentials.assert_not_called()
    new.upload_file.assert_not_called()
    assert old.upload_file.call_count == 1


@pytest.mark.parametrize("method", ["put", "sync_s3"])
def test_refresh_failure_is_sanitized_and_does_not_retry_upload(tmp_path, method):
    source = tmp_path / "input"
    source.write_text("input")
    old = SimpleNamespace(upload_file=Mock(side_effect=provider_error("ExpiredToken")))
    site = site_with_clients(old, None)
    site._refresh_s3_credentials.side_effect = RuntimeError("synthetic-secret")
    with pytest.raises(RuntimeError, match="credential_refresh_failed") as caught:
        getattr(site, method)(source, "project/input")
    assert old.upload_file.call_count == 1
    site._refresh_s3_credentials.assert_called_once_with()
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "synthetic-secret" not in "".join(traceback.format_exception(caught.value))
    assert site.graphql_client.mock_calls == []


def test_refresh_uses_only_cognito_keys_and_preserves_site_scope(monkeypatch):
    site = site_with_clients(None, None)
    site.config.region = "us-east-1"
    site.cognito_auth = SimpleNamespace(
        get_aws_credentials=Mock(
            return_value={
                "AccessKeyId": "synthetic-access",
                "SecretKey": "synthetic-secret",
                "SessionToken": "synthetic-session",
            }
        )
    )
    session = Mock()
    factory = Mock(return_value=session)
    monkeypatch.setattr(
        "frequensolve.orchestrator.sites.aws.aws.boto3.Session", factory
    )
    AWSSite._refresh_s3_credentials(site)
    factory.assert_called_once_with(
        aws_access_key_id="synthetic-access",
        aws_secret_access_key="synthetic-secret",
        aws_session_token="synthetic-session",
        region_name="us-east-1",
    )
    session.client.assert_called_once_with("s3", region_name="us-east-1")
    assert site.s3_client is session.client.return_value
    assert site.config.s3_bucket == "tenant-bucket"
