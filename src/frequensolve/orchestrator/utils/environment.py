"""Subprocess environment policy for execution backends."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Optional

__all__ = ["build_subprocess_environment", "validate_environment"]

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_VARIABLES = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "HPC_PASSWORD",
    "SSH_PASSPHRASE",
}
_SENSITIVE_SUFFIXES = ("_PASSWORD", "_PASSPHRASE", "_SECRET", "_TOKEN")


def _is_sensitive_variable(name: str) -> bool:
    upper_name = name.upper()
    return upper_name in _SENSITIVE_VARIABLES or upper_name.endswith(
        _SENSITIVE_SUFFIXES
    )


def _without_sensitive_variables(
    values: Optional[Mapping[str, object]],
) -> dict[str, object]:
    return {
        str(name): value
        for name, value in (values or {}).items()
        if not _is_sensitive_variable(str(name))
    }


def validate_environment(values: Optional[Mapping[str, object]]) -> dict[str, str]:
    """Validate and normalize explicit non-secret environment values."""

    if values is not None and not isinstance(values, Mapping):
        raise ValueError("environment must be a mapping of names to values")
    normalized: dict[str, str] = {}
    for raw_name, raw_value in (values or {}).items():
        name = str(raw_name)
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError(f"Invalid environment variable name: {name!r}")
        if _is_sensitive_variable(name):
            raise ValueError(
                f"Credential variable {name!r} cannot be configured as a "
                "subprocess environment value"
            )
        normalized[name] = str(raw_value)
    return normalized


def build_subprocess_environment(
    *,
    overrides: Optional[Mapping[str, object]] = None,
    defaults: Optional[Mapping[str, object]] = None,
) -> dict[str, str]:
    """Build an inherited environment without forwarding known credentials."""

    environment = os.environ.copy()
    environment = {
        name: value
        for name, value in environment.items()
        if not _is_sensitive_variable(name)
    }
    for name, value in validate_environment(
        _without_sensitive_variables(defaults)
    ).items():
        environment.setdefault(name, value)
    environment.update(validate_environment(_without_sensitive_variables(overrides)))
    return environment
