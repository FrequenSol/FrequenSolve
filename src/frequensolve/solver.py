"""Solver release identity, compatibility and feature checks.

FrequenSolve distributions carry the immutable preferred Solver release
and the validation profile used for their release evidence.  This module loads
that declaration, queries a configured solver without starting MPI or a job,
and applies the requested ``warn``, ``strict``, or ``off`` policy.

Job inputs that older solver builds would silently ignore or reject are gated
on the backend's ``--capabilities-json`` operation catalog
(:func:`query_local_solver_capabilities`).

Importing this module never invokes a solver and never emits a warning.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

LEGACY_COMPATIBILITY_SCHEMA = "frequensolve-solver-compatibility/v1"
COMPATIBILITY_SCHEMA = "frequensolve-solver-compatibility/v2"
IDENTITY_SCHEMA = "fs-solver-identity-1"
IDENTITY_PRODUCT = "FS_solver"
IDENTITY_QUERY_TIMEOUT_SECONDS = 15.0
POLICY_ENV_VAR = "FS_SOLVER_POLICY"
COMPATIBILITY_RESOURCE = "solver_compatibility.json"
FINAL_RELEASE_RE = re.compile(
    r"^v(?:0|[1-9][0-9]*)\." r"(?:0|[1-9][0-9]*)\." r"(?:0|[1-9][0-9]*)$"
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
CAPABILITIES_SCHEMA = "fs-solver-capabilities-2"
#: Catalog operation of a backend that honours ``extrapolate: "hold"`` on hat,
#: B-spline and tensor-hat control maps; older builds ignore the key and use
#: zero extrapolation.
CONTROL_EXTRAPOLATE_HOLD = "physics.control_map.extrapolate_hold"
#: Catalog operation of a backend that accepts several directions in one
#: ``kernel_derivative`` ``normal`` job; older builds reject such a job.
KERNEL_DERIVATIVE_NORMAL_DIRECTIONS = "imaging.normal.kernel_derivative_directions"
_CONTROL_MAP_KINDS = (None, "hat", "bspline", "tensor_hat")

CompatibilityPolicy = Literal["warn", "strict", "off"]
CompatibilityStatus = Literal["compatible", "untested", "unknown", "off"]
ValidationProfile = Literal["standard", "solver-backed"]

# Each (executable, message) warning is emitted once per process.
_EMITTED_WARNINGS: set = set()

__all__ = [
    "CAPABILITIES_SCHEMA",
    "COMPATIBILITY_SCHEMA",
    "CONTROL_EXTRAPOLATE_HOLD",
    "IDENTITY_QUERY_TIMEOUT_SECONDS",
    "IDENTITY_SCHEMA",
    "KERNEL_DERIVATIVE_NORMAL_DIRECTIONS",
    "POLICY_ENV_VAR",
    "SolverCapabilities",
    "SolverCompatibility",
    "SolverCompatibilityError",
    "SolverCompatibilityManifest",
    "SolverCompatibilityWarning",
    "SolverFeatureError",
    "SolverIdentity",
    "SolverIdentityQuery",
    "PreferredSolver",
    "check_solver_compatibility",
    "held_control_maps",
    "load_solver_compatibility",
    "query_local_solver_capabilities",
    "query_local_solver_identity",
    "query_remote_solver_capabilities",
    "query_remote_solver_identity",
    "require_held_control_support",
    "resolve_solver_policy",
    "solver_backend_name",
    "solver_backend_selection",
]


class SolverCompatibilityWarning(RuntimeWarning):
    """Warning for an untested or unidentified Solver pairing."""


class SolverCompatibilityError(RuntimeError):
    """Raised when the strict compatibility policy cannot confirm a pairing."""


class SolverFeatureError(RuntimeError):
    """Raised when a job needs a feature the configured solver does not list."""


@dataclass(frozen=True)
class PreferredSolver:
    """Immutable Solver release selected by FrequenSolve release evidence."""

    release: str
    git_commit: str
    release_url: str


@dataclass(frozen=True)
class SolverCompatibilityManifest:
    """Compatibility declaration embedded in a FrequenSolve distribution."""

    package_release: str
    preferred_solver: Optional[PreferredSolver]
    evidence_run_id: Optional[int]
    evidence_url: Optional[str]
    schema: str = COMPATIBILITY_SCHEMA
    validation_profile: Optional[ValidationProfile] = "solver-backed"

    @property
    def solver_backed(self) -> bool:
        """Return whether the manifest records solver-backed validation."""

        return self.validation_profile == "solver-backed"


@dataclass(frozen=True)
class SolverIdentity:
    """Machine-readable identity returned by ``Solver --identity-json``."""

    version: str
    build_id: str
    git_commit: str
    schema: str = IDENTITY_SCHEMA
    product: str = IDENTITY_PRODUCT


@dataclass(frozen=True)
class SolverIdentityQuery:
    """Result of querying a local or remote configured solver executable."""

    identity: Optional[SolverIdentity]
    error: Optional[str] = None


@dataclass(frozen=True)
class SolverCompatibility:
    """Assessment of one FrequenSolve and Solver pairing."""

    status: CompatibilityStatus
    message: str
    manifest: SolverCompatibilityManifest
    identity: Optional[SolverIdentity] = None

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
) -> SolverCompatibilityManifest:
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
    preferred_payload = payload.get("preferred_solver")
    validation_payload = (
        payload.get("evidence")
        if schema == LEGACY_COMPATIBILITY_SCHEMA
        else payload.get("validation")
    )
    if preferred_payload is None and validation_payload is None:
        return SolverCompatibilityManifest(
            package_release=package_release,
            preferred_solver=None,
            evidence_run_id=None,
            evidence_url=None,
            validation_profile=None,
            schema=str(schema),
        )
    if not isinstance(preferred_payload, Mapping):
        raise ValueError("preferred_solver must be an object")
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
        "preferred_solver.release",
    )
    preferred_commit = _required_string(
        preferred_payload.get("git_commit"),
        "preferred_solver.git_commit",
    )
    if not FINAL_RELEASE_RE.fullmatch(preferred_release):
        raise ValueError("preferred_solver.release must be a final vX.Y.Z tag")
    if not SHA_RE.fullmatch(preferred_commit):
        raise ValueError(
            "preferred_solver.git_commit must be a lowercase 40-character Git SHA"
        )
    preferred_release_url = _required_string(
        preferred_payload.get("release_url"),
        "preferred_solver.release_url",
    )
    expected_release_url = (
        f"https://github.com/FrequenSol/Sauce/releases/tag/{preferred_release}"
    )
    if preferred_release_url != expected_release_url:
        raise ValueError(
            "preferred_solver.release_url must identify the declared "
            "immutable Solver release"
        )
    return SolverCompatibilityManifest(
        package_release=package_release,
        preferred_solver=PreferredSolver(
            release=preferred_release,
            git_commit=preferred_commit,
            release_url=preferred_release_url,
        ),
        evidence_run_id=run_id,
        evidence_url=evidence_url,
        validation_profile=validation_profile,
        schema=str(schema),
    )


def load_solver_compatibility(
    source: Optional[Union[str, Path]] = None,
) -> SolverCompatibilityManifest:
    """Load and validate the distribution's preferred Solver metadata.

    Args:
        source: Optional manifest path for tooling and validation. Normal
            applications should omit it and load the packaged resource.
    """

    text = Path(source).read_text(encoding="utf-8") if source else _resource_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Solver compatibility JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Solver compatibility metadata must be an object")
    return _manifest_from_mapping(payload)


def _identity_from_output(output: str) -> SolverIdentity:
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
    if (
        not isinstance(schema, str)
        or not isinstance(product, str)
        or (schema, product) != (IDENTITY_SCHEMA, IDENTITY_PRODUCT)
    ):
        raise ValueError(
            "identity schema/product pair is unsupported: "
            f"schema={schema!r}, product={product!r}"
        )
    return SolverIdentity(
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


def query_local_solver_identity(
    executable: Union[str, Path],
    *,
    environment: Optional[Mapping[str, str]] = None,
    timeout: float = IDENTITY_QUERY_TIMEOUT_SECONDS,
) -> SolverIdentityQuery:
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
        return SolverIdentityQuery(None, f"identity command failed: {exc}")
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        suffix = f": {detail}" if detail else ""
        return SolverIdentityQuery(
            None,
            f"identity command exited with {completed.returncode}{suffix}",
        )
    try:
        return SolverIdentityQuery(_identity_from_output(completed.stdout))
    except ValueError as exc:
        return SolverIdentityQuery(None, str(exc))


def query_remote_solver_identity(
    executable: Union[str, Path],
    run_command: Callable[[str], str],
    *,
    setup_commands: Sequence[str] = (),
) -> SolverIdentityQuery:
    """Query an SSH-backed configured executable without an MPI launcher."""

    begin_marker = "frequensolve-solver-identity-begin"
    success_marker = "frequensolve-solver-identity-ok"
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
        return SolverIdentityQuery(None, f"identity command failed: {exc}")
    lines = output.strip().splitlines()
    if len(lines) < 3 or lines[-1] != success_marker or begin_marker not in lines:
        return SolverIdentityQuery(
            None,
            "remote identity command did not complete successfully",
        )
    begin_index = len(lines) - 1 - lines[::-1].index(begin_marker)
    identity_output = "\n".join(lines[begin_index + 1 : -1])
    try:
        return SolverIdentityQuery(_identity_from_output(identity_output))
    except ValueError as exc:
        return SolverIdentityQuery(None, str(exc))


def resolve_solver_policy(policy: Optional[str] = None) -> CompatibilityPolicy:
    """Resolve explicit policy, then the current and legacy environment names."""

    value = policy
    if value is None:
        value = os.getenv(
            POLICY_ENV_VAR,
            os.getenv(
                "FREQUENSOLVE_SOLVER_POLICY",
                os.getenv("FREQUENSOLVE_FREQUENSOLVER_POLICY", "warn"),
            ),
        )
    normalized = str(value).strip().lower()
    if normalized not in {"warn", "strict", "off"}:
        raise ValueError(
            "Solver compatibility policy must be 'warn', 'strict', or 'off'"
        )
    return normalized  # type: ignore[return-value]


def _known_commit(value: str) -> Optional[str]:
    normalized = value.strip().lower()
    if len(normalized) == 40 and all(char in "0123456789abcdef" for char in normalized):
        return normalized
    return None


def _versions_match(preferred: str, running: str) -> bool:
    return preferred.removeprefix("v") == running.removeprefix("v")


def _preferred_label(preferred: PreferredSolver) -> str:
    return f"{preferred.release} (commit {preferred.git_commit[:12]})"


def _warning_message(
    manifest: SolverCompatibilityManifest,
    query: SolverIdentityQuery,
    *,
    reason: str,
) -> str:
    preferred = manifest.preferred_solver
    if preferred is None:
        return (
            "Solver compatibility warning\n\n"
            f"FrequenSolve {manifest.package_release} does not contain a released "
            "preferred Solver declaration. The configured pairing cannot be "
            "validated and may result in unexpected behavior."
        )
    if query.identity is None:
        running = (
            "The configured solver could not report a valid Solver identity"
            f" ({query.error or 'unknown identity error'})."
        )
    else:
        running = (
            "The configured solver reports Solver "
            f"{query.identity.version} (commit {query.identity.git_commit[:12]})."
        )
    evidence = f" Evidence: {manifest.evidence_url}." if manifest.evidence_url else ""
    if manifest.solver_backed:
        release_declaration = f"was tested with Solver {_preferred_label(preferred)}."
    else:
        release_declaration = (
            f"declares Solver {_preferred_label(preferred)} as its preferred "
            "immutable release, but the standard release profile did not run "
            "solver-backed validation."
        )
    return (
        "Solver compatibility warning\n\n"
        f"FrequenSolve {manifest.package_release} {release_declaration}{evidence}\n\n"
        f"{running} {reason} This pairing has not been validated and may result "
        "in unexpected behavior.\n\n"
        f"Preferred Solver: {_preferred_label(preferred)}"
    )


def _assess(
    manifest: SolverCompatibilityManifest,
    query: SolverIdentityQuery,
) -> SolverCompatibility:
    preferred = manifest.preferred_solver
    if preferred is None:
        message = _warning_message(
            manifest,
            query,
            reason="No immutable release pair is declared.",
        )
        return SolverCompatibility("unknown", message, manifest, query.identity)
    identity = query.identity
    if identity is None:
        message = _warning_message(
            manifest,
            query,
            reason="The running Solver release is unknown.",
        )
        return SolverCompatibility("unknown", message, manifest)
    if not _versions_match(preferred.release, identity.version):
        message = _warning_message(
            manifest,
            query,
            reason="Its release does not match the preferred release.",
        )
        return SolverCompatibility("untested", message, manifest, identity)
    preferred_commit = _known_commit(preferred.git_commit)
    running_commit = _known_commit(identity.git_commit)
    if preferred_commit is None or running_commit is None:
        message = _warning_message(
            manifest,
            query,
            reason="An exact Solver commit could not be confirmed.",
        )
        return SolverCompatibility("unknown", message, manifest, identity)
    if preferred_commit != running_commit:
        message = _warning_message(
            manifest,
            query,
            reason="Its commit does not match the tested release commit.",
        )
        return SolverCompatibility("untested", message, manifest, identity)
    if not manifest.solver_backed:
        message = _warning_message(
            manifest,
            query,
            reason=(
                "The immutable identities match, but solver-backed release "
                "validation was not run."
            ),
        )
        return SolverCompatibility("untested", message, manifest, identity)
    return SolverCompatibility(
        "compatible",
        f"FrequenSolve {manifest.package_release} matches preferred Solver "
        f"{_preferred_label(preferred)}.",
        manifest,
        identity,
    )


def check_solver_compatibility(
    executable: Union[str, Path],
    *,
    policy: Optional[str] = None,
    manifest: Optional[SolverCompatibilityManifest] = None,
    environment: Optional[Mapping[str, str]] = None,
    remote_runner: Optional[Callable[[str], str]] = None,
    setup_commands: Sequence[str] = (),
) -> SolverCompatibility:
    """Check a configured solver against this package's preferred release.

    ``warn`` (the default) emits :class:`SolverCompatibilityWarning` once per
    process for each executable's unknown or untested pair. ``strict`` raises
    :class:`SolverCompatibilityError`. ``off`` performs no solver query.
    """

    selected_policy = resolve_solver_policy(policy)
    loaded_manifest = manifest or load_solver_compatibility()
    if selected_policy == "off":
        return SolverCompatibility(
            "off",
            "Solver compatibility checking is disabled.",
            loaded_manifest,
        )
    query = (
        query_remote_solver_identity(
            executable,
            remote_runner,
            setup_commands=setup_commands,
        )
        if remote_runner is not None
        else query_local_solver_identity(
            executable,
            environment=environment,
        )
    )
    result = _assess(loaded_manifest, query)
    if result.confirmed:
        return result
    if selected_policy == "strict":
        raise SolverCompatibilityError(
            result.message.replace(
                "Solver compatibility warning",
                "Solver compatibility check failed",
                1,
            )
        )
    key = (str(executable), result.message)
    if key not in _EMITTED_WARNINGS:
        _EMITTED_WARNINGS.add(key)
        warnings.warn(result.message, SolverCompatibilityWarning, stacklevel=2)
    return result


# ---------------------------------------------------------------------------
# Backend feature catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverCapabilities:
    """Catalogued operations of the solver backend that runs a job.

    ``operations`` holds the available operation IDs of the backend's
    ``--capabilities-json`` catalog, or is ``None`` when the backend could not
    be queried (``error`` says why). The catalog is partial: an operation it
    does not list is treated as unsupported, which is what a build that
    predates the operation reports.
    """

    backend: str
    operations: Optional[FrozenSet[str]] = None
    error: Optional[str] = None

    def supports(self, operation: str) -> bool:
        """Return whether the queried backend lists ``operation``."""

        return self.operations is not None and operation in self.operations

    def missing_reason(self, operation: str) -> str:
        """Describe why :meth:`supports` is false for ``operation``."""

        if self.operations is None:
            return f"its capabilities could not be queried ({self.error})"
        return f"{self.backend} --capabilities-json does not list {operation!r}"


def solver_backend_name(dimension: Any, precision: str = "single") -> str:
    """Return the ``fs{2,3}d[_s]`` backend the dispatcher runs a simulation on.

    Like ``FS_seismic``, a fractional dimension (2.5) selects its integer part
    and single precision appends ``_s``.
    """

    value = float(dimension)
    if not math.isfinite(value) or math.floor(value) not in (2, 3):
        raise ValueError(f"unsupported solver dimension {dimension!r}")
    if precision not in {"single", "double"}:
        raise ValueError("Solver precision must be 'single' or 'double'")
    return f"fs{int(math.floor(value))}d" + ("_s" if precision == "single" else "")


def solver_backend_selection(simulation: Any) -> Dict[str, Any]:
    """Return the ``dimension`` and ``precision`` that select a simulation's backend."""

    solver = getattr(simulation, "solver", None)
    return {
        "dimension": getattr(simulation, "dimension", 2),
        "precision": str(getattr(solver, "precision", None) or "single"),
    }


def _capability_catalog(output: str) -> Tuple[bool, FrozenSet[str]]:
    """Return ``(dispatcher, available operation IDs)`` of one capability report."""

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid capability JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("capability JSON must be an object")
    if payload.get("scope") == "dispatcher" or payload.get("backend_query_required"):
        return True, frozenset()
    catalog = payload.get("catalog")
    # Version 1 reports and builds before the catalog list no operations.
    entries = catalog.get("capabilities") if isinstance(catalog, Mapping) else None
    return False, frozenset(
        entry["id"]
        for entry in entries or ()
        if isinstance(entry, Mapping)
        and isinstance(entry.get("id"), str)
        and entry.get("available", True) is True
    )


def query_local_solver_capabilities(
    executable: Union[str, Path],
    *,
    dimension: Any = 2,
    precision: str = "single",
    environment: Optional[Mapping[str, str]] = None,
    timeout: float = IDENTITY_QUERY_TIMEOUT_SECONDS,
) -> SolverCapabilities:
    """Query the backend a simulation of ``dimension``/``precision`` runs on.

    A dispatcher (``FS_seismic``) reports no catalog of its own, so the backend
    it routes to is queried as the dispatcher finds it: in ``FS_SOLVER_PATH``
    of the solver environment, else beside the dispatcher.
    """

    lookup = environment if environment is not None else os.environ

    def report(path: str) -> Tuple[bool, FrozenSet[str]]:
        completed = subprocess.run(
            [path, "--capabilities-json"],
            check=False,
            capture_output=True,
            text=True,
            env=dict(environment) if environment is not None else None,
            timeout=timeout,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(
                f"{path} --capabilities-json exited with {completed.returncode}"
                + (f": {detail}" if detail else "")
            )
        return _capability_catalog(completed.stdout)

    backend = str(executable)
    try:
        dispatcher, operations = report(backend)
        if dispatcher:
            located = shutil.which(backend, path=lookup.get("PATH")) or backend
            directory = lookup.get("FS_SOLVER_PATH") or os.path.dirname(
                os.path.abspath(located)
            )
            backend = os.path.join(directory, solver_backend_name(dimension, precision))
            dispatcher, operations = report(backend)
            if dispatcher:
                raise ValueError(f"{backend} reports a dispatcher scope")
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return SolverCapabilities(backend, None, str(exc))
    return SolverCapabilities(backend, operations)


def query_remote_solver_capabilities(
    executable: Union[str, Path],
    run_command: Callable[[str], str],
    *,
    dimension: Any = 2,
    precision: str = "single",
    setup_commands: Sequence[str] = (),
) -> SolverCapabilities:
    """Query a remote backend's catalog in one login-node round trip.

    The dispatcher's routed backend (``FS_SOLVER_PATH`` after
    ``setup_commands``, else the dispatcher's directory) is reported as well
    and used when the configured executable is a dispatcher.
    """

    begin = "frequensolve-solver-capabilities-begin"
    routed = "frequensolve-solver-capabilities-backend"
    success = "frequensolve-solver-capabilities-ok"
    solver = shlex.quote(str(executable))
    backend = str(executable)
    try:
        name = solver_backend_name(dimension, precision)
        command = "\n".join(
            [
                "set -e",
                *setup_commands,
                f"printf '%s\\n' {begin}",
                f"{solver} --capabilities-json",
                f"printf '%s\\n' {routed}",
                "frequensolve_backend="
                f'"${{FS_SOLVER_PATH:-$(dirname "$(command -v {solver})")}}/{name}"',
                'if [ -x "$frequensolve_backend" ]; then '
                "printf '%s\\n' \"$frequensolve_backend\"; "
                '"$frequensolve_backend" --capabilities-json; fi',
                f"printf '%s\\n' {success}",
            ]
        )
        lines = run_command(command).strip().splitlines()
        if not lines or lines[-1] != success or begin not in lines:
            raise ValueError("remote capability query did not complete successfully")
        start = len(lines) - 1 - lines[::-1].index(begin)
        middle = start + lines[start:].index(routed)
        dispatcher, operations = _capability_catalog(
            "\n".join(lines[start + 1 : middle])
        )
        if dispatcher:
            routed_lines = lines[middle + 1 : -1]
            if not routed_lines:
                raise ValueError(f"dispatcher backend {name} was not found")
            backend = routed_lines[0]
            dispatcher, operations = _capability_catalog("\n".join(routed_lines[1:]))
            if dispatcher:
                raise ValueError(f"{backend} reports a dispatcher scope")
    except Exception as exc:
        return SolverCapabilities(backend, None, str(exc))
    return SolverCapabilities(backend, operations)


def held_control_maps(payload: Any, location: str = "") -> List[str]:
    """Return the JSON locations of control maps that use ``extrapolate: "hold"``.

    Hat, B-spline (the default kind) and tensor-hat maps are recognized
    wherever they appear: material ``parameterized`` controls, independent
    reflectivity maps and implicit-surface controls.
    """

    found: List[str] = []
    if isinstance(payload, Mapping):
        policy = payload.get("extrapolate")
        if (
            isinstance(policy, str)
            and policy.strip().lower() == "hold"
            and payload.get("kind") in _CONTROL_MAP_KINDS
        ):
            found.append(location or "/")
        for key, value in payload.items():
            found.extend(held_control_maps(value, f"{location}/{key}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(held_control_maps(value, f"{location}/{index}"))
    return found


# Held-control locations per JSON file, reused while the file is unchanged.
_HELD_CONTROL_SCANS: Dict[str, Tuple[Tuple[int, int], Tuple[str, ...]]] = {}


def _held_controls_in_file(path: Path) -> Tuple[str, ...]:
    try:
        status = path.stat()
    except OSError:
        return ()
    key = str(path.resolve())
    marker = (status.st_mtime_ns, status.st_size)
    cached = _HELD_CONTROL_SCANS.get(key)
    if cached is not None and cached[0] == marker:
        return cached[1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    found = tuple(held_control_maps(payload))
    _HELD_CONTROL_SCANS[key] = (marker, found)
    return found


def require_held_control_support(
    files: Iterable[Union[str, Path]],
    capabilities: Callable[[], Optional[SolverCapabilities]],
) -> None:
    """Reject saved job files with held control maps the solver would ignore.

    ``capabilities`` is called only when a held map is present; ``None``
    (a site that cannot report its backend, or checking turned off) accepts.
    """

    held = [
        f"{Path(path).name}:{location}"
        for path in files
        for location in _held_controls_in_file(Path(path))
    ]
    if not held:
        return
    report = capabilities()
    if report is None or report.supports(CONTROL_EXTRAPOLATE_HOLD):
        return
    shown = ", ".join(held[:3]) + (
        f" and {len(held) - 3} more" if len(held) > 3 else ""
    )
    reason = report.missing_reason(CONTROL_EXTRAPOLATE_HOLD)
    raise SolverFeatureError(
        f"Control maps use extrapolate='hold' ({shown}), but the configured "
        f"solver does not advertise it: {reason}. Solver builds without it "
        "silently apply zero extrapolation; update the solver or use "
        "extrapolate='zero'."
    )
