"""FrequenSolver release identity and compatibility checks.

FrequenSolve distributions carry the immutable preferred FrequenSolver release
and the validation profile used for their release evidence.  This module loads
that declaration, queries a configured solver without starting MPI or a job,
and applies the requested ``warn``, ``strict``, or ``off`` policy.

Importing this module never invokes a solver and never emits a warning.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import warnings
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Callable, Literal, Mapping, Optional, Sequence, Union

LEGACY_COMPATIBILITY_SCHEMA = "frequensolve-frequensolver-compatibility/v1"
COMPATIBILITY_SCHEMA = "frequensolve-frequensolver-compatibility/v2"
IDENTITY_SCHEMA = "frequensolver-identity-1"
IDENTITY_PRODUCT = "FrequenSolver"
_SUPPORTED_IDENTITY_PAIRS = frozenset(
    {
        (IDENTITY_SCHEMA, IDENTITY_PRODUCT),
        ("fs-solver-identity-1", "FS_solver"),
    }
)
IDENTITY_QUERY_TIMEOUT_SECONDS = 15.0
POLICY_ENV_VAR = "FREQUENSOLVE_FREQUENSOLVER_POLICY"
COMPATIBILITY_RESOURCE = "frequensolver_compatibility.json"
FINAL_RELEASE_RE = re.compile(
    r"^v(?:0|[1-9][0-9]*)\." r"(?:0|[1-9][0-9]*)\." r"(?:0|[1-9][0-9]*)$"
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

CompatibilityPolicy = Literal["warn", "strict", "off"]
CompatibilityStatus = Literal["compatible", "untested", "unknown", "off"]
ValidationProfile = Literal["standard", "solver-backed"]

__all__ = [
    "COMPATIBILITY_SCHEMA",
    "IDENTITY_QUERY_TIMEOUT_SECONDS",
    "IDENTITY_SCHEMA",
    "POLICY_ENV_VAR",
    "FrequenSolverCompatibility",
    "FrequenSolverCompatibilityError",
    "FrequenSolverCompatibilityManifest",
    "FrequenSolverCompatibilityWarning",
    "FrequenSolverIdentity",
    "FrequenSolverIdentityQuery",
    "PreferredFrequenSolver",
    "check_frequensolver_compatibility",
    "load_frequensolver_compatibility",
    "query_local_frequensolver_identity",
    "query_remote_frequensolver_identity",
    "resolve_frequensolver_policy",
]


class FrequenSolverCompatibilityWarning(RuntimeWarning):
    """Warning for an untested or unidentified FrequenSolver pairing."""


class FrequenSolverCompatibilityError(RuntimeError):
    """Raised when the strict compatibility policy cannot confirm a pairing."""


@dataclass(frozen=True)
class PreferredFrequenSolver:
    """Immutable FrequenSolver release selected by FrequenSolve release evidence."""

    release: str
    git_commit: str
    release_url: str


@dataclass(frozen=True)
class FrequenSolverCompatibilityManifest:
    """Compatibility declaration embedded in a FrequenSolve distribution."""

    package_release: str
    preferred_frequensolver: Optional[PreferredFrequenSolver]
    evidence_run_id: Optional[int]
    evidence_url: Optional[str]
    schema: str = COMPATIBILITY_SCHEMA
    validation_profile: Optional[ValidationProfile] = "solver-backed"

    @property
    def solver_backed(self) -> bool:
        """Return whether the manifest records solver-backed validation."""

        return self.validation_profile == "solver-backed"


@dataclass(frozen=True)
class FrequenSolverIdentity:
    """Machine-readable identity returned by ``FrequenSolver --identity-json``."""

    version: str
    build_id: str
    git_commit: str
    schema: str = IDENTITY_SCHEMA
    product: str = IDENTITY_PRODUCT


@dataclass(frozen=True)
class FrequenSolverIdentityQuery:
    """Result of querying a local or remote configured solver executable."""

    identity: Optional[FrequenSolverIdentity]
    error: Optional[str] = None


@dataclass(frozen=True)
class FrequenSolverCompatibility:
    """Assessment of one FrequenSolve and FrequenSolver pairing."""

    status: CompatibilityStatus
    message: str
    manifest: FrequenSolverCompatibilityManifest
    identity: Optional[FrequenSolverIdentity] = None

    @property
    def confirmed(self) -> bool:
        """Return whether solver-backed evidence confirms the running pair."""

        return self.status == "compatible"


def _resource_text() -> str:
    return (
        files("frequensolve")
        .joinpath(COMPATIBILITY_RESOURCE)
        .read_text(encoding="utf-8")
    )


def _required_string(
    value: object,
    name: str,
    *,
    allow_unknown: bool = False,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    normalized = value.strip()
    if "\r" in normalized or "\n" in normalized:
        raise ValueError(f"{name} must be a single-line string")
    if not allow_unknown and normalized.lower() in {"unknown", "unavailable"}:
        raise ValueError(f"{name} must identify a released artifact")
    return normalized


def _manifest_from_mapping(
    payload: Mapping[str, object],
) -> FrequenSolverCompatibilityManifest:
    schema = payload.get("schema")
    if schema not in {COMPATIBILITY_SCHEMA, LEGACY_COMPATIBILITY_SCHEMA}:
        raise ValueError(
            "compatibility schema must be "
            f"{COMPATIBILITY_SCHEMA!r} or legacy {LEGACY_COMPATIBILITY_SCHEMA!r}, "
            f"got {schema!r}"
        )
    package_release = _required_string(
        payload.get("package_release"),
        "package_release",
        allow_unknown=True,
    )
    preferred_payload = payload.get("preferred_frequensolver")
    validation_payload = (
        payload.get("evidence")
        if schema == LEGACY_COMPATIBILITY_SCHEMA
        else payload.get("validation")
    )
    if preferred_payload is None and validation_payload is None:
        return FrequenSolverCompatibilityManifest(
            package_release=package_release,
            preferred_frequensolver=None,
            evidence_run_id=None,
            evidence_url=None,
            validation_profile=None,
            schema=str(schema),
        )
    if not isinstance(preferred_payload, Mapping):
        raise ValueError("preferred_frequensolver must be an object")
    if not isinstance(validation_payload, Mapping):
        field = "evidence" if schema == LEGACY_COMPATIBILITY_SCHEMA else "validation"
        raise ValueError(f"{field} must be an object")

    if schema == LEGACY_COMPATIBILITY_SCHEMA:
        validation_profile: ValidationProfile = "solver-backed"
        validation_field = "evidence"
    else:
        profile = validation_payload.get("profile")
        if profile not in {"standard", "solver-backed"}:
            raise ValueError("validation.profile must be 'standard' or 'solver-backed'")
        validation_profile = profile  # type: ignore[assignment]
        declared_solver_backed = validation_payload.get("solver_backed")
        if not isinstance(declared_solver_backed, bool):
            raise ValueError("validation.solver_backed must be a boolean")
        if declared_solver_backed != (validation_profile == "solver-backed"):
            raise ValueError("validation.solver_backed must match validation.profile")
        validation_field = "validation"

    run_id = validation_payload.get("run_id")
    if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
        raise ValueError(f"{validation_field}.run_id must be a positive integer")
    evidence_url = _required_string(
        validation_payload.get("url"), f"{validation_field}.url"
    )
    evidence_repository = (
        "FrequenSolveDockerImage"
        if validation_profile == "solver-backed"
        else "FrequenSolve"
    )
    expected_evidence_url = (
        f"https://github.com/FrequenSol/{evidence_repository}/actions/runs/{run_id}"
    )
    if evidence_url != expected_evidence_url:
        raise ValueError(
            f"{validation_field}.url must identify {validation_field}.run_id"
        )
    preferred_release = _required_string(
        preferred_payload.get("release"),
        "preferred_frequensolver.release",
    )
    preferred_commit = _required_string(
        preferred_payload.get("git_commit"),
        "preferred_frequensolver.git_commit",
    )
    if not FINAL_RELEASE_RE.fullmatch(preferred_release):
        raise ValueError("preferred_frequensolver.release must be a final vX.Y.Z tag")
    if not SHA_RE.fullmatch(preferred_commit):
        raise ValueError(
            "preferred_frequensolver.git_commit must be a lowercase 40-character Git SHA"
        )
    preferred_release_url = _required_string(
        preferred_payload.get("release_url"),
        "preferred_frequensolver.release_url",
    )
    expected_release_url = (
        f"https://github.com/FrequenSol/Sauce/releases/tag/{preferred_release}"
    )
    if preferred_release_url != expected_release_url:
        raise ValueError(
            "preferred_frequensolver.release_url must identify the declared "
            "immutable FrequenSolver release"
        )
    return FrequenSolverCompatibilityManifest(
        package_release=package_release,
        preferred_frequensolver=PreferredFrequenSolver(
            release=preferred_release,
            git_commit=preferred_commit,
            release_url=preferred_release_url,
        ),
        evidence_run_id=run_id,
        evidence_url=evidence_url,
        validation_profile=validation_profile,
        schema=str(schema),
    )


def load_frequensolver_compatibility(
    source: Optional[Union[str, Path]] = None,
) -> FrequenSolverCompatibilityManifest:
    """Load and validate the distribution's preferred FrequenSolver metadata.

    Args:
        source: Optional manifest path for tooling and validation. Normal
            applications should omit it and load the packaged resource.
    """

    text = Path(source).read_text(encoding="utf-8") if source else _resource_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid FrequenSolver compatibility JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("FrequenSolver compatibility metadata must be an object")
    return _manifest_from_mapping(payload)


def _identity_from_output(output: str) -> FrequenSolverIdentity:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid identity JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("identity JSON must be an object")
    expected_keys = {"schema", "product", "version", "build_id", "git_commit"}
    if set(payload) != expected_keys:
        missing = sorted(expected_keys - set(payload))
        extra = sorted(set(payload) - expected_keys)
        details = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected keys: {', '.join(extra)}")
        raise ValueError("identity JSON has " + "; ".join(details))
    schema = payload.get("schema")
    product = payload.get("product")
    if (schema, product) not in _SUPPORTED_IDENTITY_PAIRS:
        raise ValueError(
            "identity schema/product pair is unsupported: "
            f"schema={schema!r}, product={product!r}"
        )
    return FrequenSolverIdentity(
        version=_required_string(
            payload.get("version"), "identity.version", allow_unknown=True
        ),
        build_id=_required_string(
            payload.get("build_id"), "identity.build_id", allow_unknown=True
        ),
        git_commit=_required_string(
            payload.get("git_commit"),
            "identity.git_commit",
            allow_unknown=True,
        ),
        schema=str(schema),
        product=str(product),
    )


def query_local_frequensolver_identity(
    executable: Union[str, Path],
    *,
    environment: Optional[Mapping[str, str]] = None,
    timeout: float = IDENTITY_QUERY_TIMEOUT_SECONDS,
) -> FrequenSolverIdentityQuery:
    """Query a local configured executable directly with ``--identity-json``."""

    command = [str(executable), "--identity-json"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=dict(environment) if environment is not None else None,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return FrequenSolverIdentityQuery(None, f"identity command failed: {exc}")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        suffix = f": {detail}" if detail else ""
        return FrequenSolverIdentityQuery(
            None,
            f"identity command exited with {completed.returncode}{suffix}",
        )
    try:
        return FrequenSolverIdentityQuery(_identity_from_output(completed.stdout))
    except ValueError as exc:
        return FrequenSolverIdentityQuery(None, str(exc))


def query_remote_frequensolver_identity(
    executable: Union[str, Path],
    run_command: Callable[[str], str],
    *,
    setup_commands: Sequence[str] = (),
) -> FrequenSolverIdentityQuery:
    """Query an SSH-backed configured executable without an MPI launcher."""

    begin_marker = "frequensolve-frequensolver-identity-begin"
    success_marker = "frequensolve-frequensolver-identity-ok"
    command = "\n".join(
        [
            "set -e",
            *setup_commands,
            f"printf '%s\\n' {shlex.quote(begin_marker)}",
            f"{shlex.quote(str(executable))} --identity-json",
            f"printf '%s\\n' {shlex.quote(success_marker)}",
        ]
    )
    try:
        output = run_command(command)
    except Exception as exc:
        return FrequenSolverIdentityQuery(None, f"identity command failed: {exc}")
    lines = output.strip().splitlines()
    if len(lines) < 3 or lines[-1] != success_marker or begin_marker not in lines:
        return FrequenSolverIdentityQuery(
            None,
            "remote identity command did not complete successfully",
        )
    begin_index = len(lines) - 1 - lines[::-1].index(begin_marker)
    identity_output = "\n".join(lines[begin_index + 1 : -1])
    try:
        return FrequenSolverIdentityQuery(_identity_from_output(identity_output))
    except ValueError as exc:
        return FrequenSolverIdentityQuery(None, str(exc))


def resolve_frequensolver_policy(policy: Optional[str] = None) -> CompatibilityPolicy:
    """Resolve an explicit or environment-provided compatibility policy."""

    value = policy if policy is not None else os.getenv(POLICY_ENV_VAR, "warn")
    normalized = str(value).strip().lower()
    if normalized not in {"warn", "strict", "off"}:
        raise ValueError(
            "FrequenSolver compatibility policy must be 'warn', 'strict', or 'off'"
        )
    return normalized  # type: ignore[return-value]


def _known_commit(value: str) -> Optional[str]:
    normalized = value.strip().lower()
    if len(normalized) == 40 and all(char in "0123456789abcdef" for char in normalized):
        return normalized
    return None


def _versions_match(preferred: str, running: str) -> bool:
    return preferred.removeprefix("v") == running.removeprefix("v")


def _preferred_label(preferred: PreferredFrequenSolver) -> str:
    return f"{preferred.release} (commit {preferred.git_commit[:12]})"


def _warning_message(
    manifest: FrequenSolverCompatibilityManifest,
    query: FrequenSolverIdentityQuery,
    *,
    reason: str,
) -> str:
    preferred = manifest.preferred_frequensolver
    if preferred is None:
        return (
            "FrequenSolver compatibility warning\n\n"
            f"FrequenSolve {manifest.package_release} does not contain a released "
            "preferred FrequenSolver declaration. The configured pairing cannot be "
            "validated and may result in unexpected behavior."
        )
    if query.identity is None:
        running = (
            "The configured solver could not report a valid FrequenSolver identity"
            f" ({query.error or 'unknown identity error'})."
        )
    else:
        running = (
            "The configured solver reports FrequenSolver "
            f"{query.identity.version} (commit {query.identity.git_commit[:12]})."
        )
    evidence = f" Evidence: {manifest.evidence_url}." if manifest.evidence_url else ""
    if manifest.solver_backed:
        release_declaration = (
            f"was tested with FrequenSolver {_preferred_label(preferred)}."
        )
    else:
        release_declaration = (
            f"declares FrequenSolver {_preferred_label(preferred)} as its preferred "
            "immutable release, but the standard release profile did not run "
            "solver-backed validation."
        )
    return (
        "FrequenSolver compatibility warning\n\n"
        f"FrequenSolve {manifest.package_release} {release_declaration}{evidence}\n\n"
        f"{running} {reason} This pairing has not been validated and may result "
        "in unexpected behavior.\n\n"
        f"Preferred FrequenSolver: {_preferred_label(preferred)}"
    )


def _assess(
    manifest: FrequenSolverCompatibilityManifest,
    query: FrequenSolverIdentityQuery,
) -> FrequenSolverCompatibility:
    preferred = manifest.preferred_frequensolver
    if preferred is None:
        message = _warning_message(
            manifest,
            query,
            reason="No immutable release pair is declared.",
        )
        return FrequenSolverCompatibility("unknown", message, manifest, query.identity)
    identity = query.identity
    if identity is None:
        message = _warning_message(
            manifest,
            query,
            reason="The running FrequenSolver release is unknown.",
        )
        return FrequenSolverCompatibility("unknown", message, manifest)
    if not _versions_match(preferred.release, identity.version):
        message = _warning_message(
            manifest,
            query,
            reason="Its release does not match the preferred release.",
        )
        return FrequenSolverCompatibility("untested", message, manifest, identity)
    preferred_commit = _known_commit(preferred.git_commit)
    running_commit = _known_commit(identity.git_commit)
    if preferred_commit is None or running_commit is None:
        message = _warning_message(
            manifest,
            query,
            reason="An exact FrequenSolver commit could not be confirmed.",
        )
        return FrequenSolverCompatibility("unknown", message, manifest, identity)
    if preferred_commit != running_commit:
        message = _warning_message(
            manifest,
            query,
            reason="Its commit does not match the tested release commit.",
        )
        return FrequenSolverCompatibility("untested", message, manifest, identity)
    if not manifest.solver_backed:
        message = _warning_message(
            manifest,
            query,
            reason=(
                "The immutable identities match, but solver-backed release "
                "validation was not run."
            ),
        )
        return FrequenSolverCompatibility("untested", message, manifest, identity)
    return FrequenSolverCompatibility(
        "compatible",
        f"FrequenSolve {manifest.package_release} matches preferred FrequenSolver "
        f"{_preferred_label(preferred)}.",
        manifest,
        identity,
    )


def check_frequensolver_compatibility(
    executable: Union[str, Path],
    *,
    policy: Optional[str] = None,
    manifest: Optional[FrequenSolverCompatibilityManifest] = None,
    environment: Optional[Mapping[str, str]] = None,
    remote_runner: Optional[Callable[[str], str]] = None,
    setup_commands: Sequence[str] = (),
) -> FrequenSolverCompatibility:
    """Check a configured solver against this package's preferred release.

    ``warn`` (the default) emits :class:`FrequenSolverCompatibilityWarning` for
    an unknown or untested pair. ``strict`` raises
    :class:`FrequenSolverCompatibilityError`. ``off`` performs no solver query.
    """

    selected_policy = resolve_frequensolver_policy(policy)
    loaded_manifest = manifest or load_frequensolver_compatibility()
    if selected_policy == "off":
        return FrequenSolverCompatibility(
            "off",
            "FrequenSolver compatibility checking is disabled.",
            loaded_manifest,
        )
    query = (
        query_remote_frequensolver_identity(
            executable,
            remote_runner,
            setup_commands=setup_commands,
        )
        if remote_runner is not None
        else query_local_frequensolver_identity(
            executable,
            environment=environment,
        )
    )
    result = _assess(loaded_manifest, query)
    if result.confirmed:
        return result
    if selected_policy == "strict":
        raise FrequenSolverCompatibilityError(
            result.message.replace(
                "FrequenSolver compatibility warning",
                "FrequenSolver compatibility check failed",
                1,
            )
        )
    warnings.warn(result.message, FrequenSolverCompatibilityWarning, stacklevel=2)
    return result
