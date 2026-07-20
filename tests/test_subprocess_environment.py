import pytest

from frequensolve.orchestrator.utils.environment import (
    build_subprocess_environment,
    validate_environment,
)


@pytest.mark.parametrize(
    "name",
    [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "HPC_PASSWORD",
        "SERVICE_PASSWORD",
        "SSH_PASSPHRASE",
    ],
)
def test_subprocess_environment_does_not_inherit_credentials(monkeypatch, name):
    monkeypatch.setenv(name, "secret")

    environment = build_subprocess_environment()

    assert name not in environment


def test_subprocess_environment_applies_non_secret_defaults_and_overrides(
    monkeypatch,
):
    monkeypatch.setenv("PATH", "/test/bin")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")

    environment = build_subprocess_environment(
        defaults={"OMP_NUM_THREADS": 1, "SAFE_DEFAULT": True},
        overrides={"OMP_NUM_THREADS": 2, "SITE_VALUE": "configured"},
    )

    assert environment["PATH"] == "/test/bin"
    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["SAFE_DEFAULT"] == "True"
    assert environment["SITE_VALUE"] == "configured"


def test_subprocess_environment_discards_explicit_credentials():
    environment = build_subprocess_environment(
        overrides={"HPC_PASSWORD": "secret", "SAFE_VALUE": "present"}
    )

    assert "HPC_PASSWORD" not in environment
    assert environment["SAFE_VALUE"] == "present"


def test_profile_environment_rejects_credentials_and_invalid_names():
    with pytest.raises(ValueError, match="Credential variable"):
        validate_environment({"AWS_SECRET_ACCESS_KEY": "secret"})
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        validate_environment({"NOT-A-NAME": "value"})
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_environment(["VALUE"])
