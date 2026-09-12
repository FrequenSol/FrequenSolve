"""Versioned Enterprise HPC profile and preflight contracts."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence

from frequensolve._optional import optional_dependency_error

try:
    from jsonschema import (  # type: ignore[import-untyped]
        Draft202012Validator,
        FormatChecker,
    )
    from jsonschema.exceptions import (  # type: ignore[import-untyped]
        SchemaError,
        ValidationError,
    )
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "Enterprise HPC profile",
        extra="hpc",
        dependencies=("jsonschema",),
        error=exc,
    ) from exc


PROFILE_SCHEMA = "frequensolve-enterprise-hpc-profile/v1"
BUNDLE_SCHEMA = "frequensolve-enterprise-hpc/v1"
COMPATIBILITY_SCHEMA = "frequensolve-enterprise-hpc-compatibility/v1"
BUNDLE_SCHEMA_ID = (
    "https://schemas.frequensol.com/frequensolve-enterprise-hpc/v1/"
    "bundle-manifest.schema.json"
)
COMPATIBILITY_SCHEMA_ID = (
    "https://schemas.frequensol.com/frequensolve-enterprise-hpc/v1/"
    "compatibility-row.schema.json"
)
BUNDLE_MANIFEST_RELATIVE = "manifests/bundle-manifest.json"
COMPATIBILITY_MANIFEST_RELATIVE = "manifests/compatibility-row.json"
BUNDLE_SCHEMA_RELATIVE = "manifests/contracts/deployment/bundle-manifest.schema.json"
COMPATIBILITY_SCHEMA_RELATIVE = (
    "manifests/contracts/deployment/compatibility-row.schema.json"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}\Z")
_SAFE_ABSOLUTE_PATH = re.compile(r"/[A-Za-z0-9._+@/=-]+\Z")


class EnterpriseHPCPreflightError(RuntimeError):
    """Raised when an Enterprise HPC site is unsafe or incompatible."""


def installed_frequensolve_artifact_sha256() -> Optional[str]:
    """Return the wheel SHA-256 after verifying pip provenance and RECORD."""

    try:
        installed = distribution("frequensolve")
        direct_url = installed.read_text("direct_url.json")
        record = installed.read_text("RECORD")
    except (PackageNotFoundError, OSError):
        return None
    if direct_url is None or record is None:
        return None
    try:
        document = json.loads(direct_url)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(document, Mapping):
        return None
    archive = document.get("archive_info")
    if not isinstance(archive, Mapping):
        return None
    candidates: list[str] = []
    legacy = archive.get("hash")
    if isinstance(legacy, str) and legacy.startswith("sha256="):
        candidates.append(legacy.removeprefix("sha256="))
    hashes = archive.get("hashes")
    if isinstance(hashes, Mapping) and isinstance(hashes.get("sha256"), str):
        candidates.append(hashes["sha256"])
    if not candidates or any(not _SHA256.fullmatch(value) for value in candidates):
        return None
    if len(set(candidates)) != 1:
        return None
    rows = list(csv.reader(record.splitlines()))
    if not rows or any(len(row) != 3 for row in rows):
        return None
    record_rows = 0
    verified_paths: set[str] = set()
    generated_cache_sources: set[str] = set()
    for relative, encoded_hash, encoded_size in rows:
        if not relative:
            return None
        if not encoded_hash and not encoded_size:
            if relative.endswith(".dist-info/RECORD"):
                record_rows += 1
                continue
            try:
                source = importlib.util.source_from_cache(relative)
            except ValueError:
                return None
            generated_cache_sources.add(source)
            continue
        if not encoded_hash.startswith("sha256=") or not encoded_size.isdigit():
            return None
        path = Path(str(installed.locate_file(relative)))
        try:
            observed = path.lstat()
            if path.is_symlink() or not path.is_file():
                return None
            digest = hashlib.sha256(path.read_bytes()).digest()
        except OSError:
            return None
        expected = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        if (
            int(encoded_size) != observed.st_size
            or encoded_hash.removeprefix("sha256=") != expected
        ):
            return None
        verified_paths.add(relative)
    if record_rows != 1 or not generated_cache_sources.issubset(verified_paths):
        return None
    return candidates[0]


def _required_text(value: Any, name: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"enterprise_hpc.{name} has an invalid value")
    return value


def _absolute_remote_path(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not _SAFE_ABSOLUTE_PATH.fullmatch(value)
    ):
        raise ValueError(f"enterprise_hpc.{name} must be an absolute remote path")
    path = PurePosixPath(value)
    if not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise ValueError(
            f"enterprise_hpc.{name} must be an absolute traversal-free path"
        )
    return str(path)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EnterpriseHPCPreflightError(
            f"Enterprise HPC {name} must be a positive integer"
        )
    return value


@dataclass(frozen=True)
class _EnterpriseHPCProfile:
    """Small generated anchor for one installed Enterprise HPC bundle."""

    profile_id: str
    bundle_root: str
    bundle_manifest_sha256: str
    schema: str = PROFILE_SCHEMA

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "_EnterpriseHPCProfile":
        """Parse a closed generated profile and reject unknown inputs."""

        allowed = {
            "schema",
            "profile_id",
            "bundle_root",
            "bundle_manifest_sha256",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                "Unsupported enterprise_hpc profile key(s): " + ", ".join(unknown)
            )
        missing = sorted(allowed - set(values))
        if missing:
            raise ValueError(
                "Missing enterprise_hpc profile key(s): " + ", ".join(missing)
            )
        if values["schema"] != PROFILE_SCHEMA:
            raise ValueError(f"enterprise_hpc.schema must be {PROFILE_SCHEMA!r}")
        return cls(
            schema=PROFILE_SCHEMA,
            profile_id=_required_text(values["profile_id"], "profile_id", _PROFILE_ID),
            bundle_root=_absolute_remote_path(values["bundle_root"], "bundle_root"),
            bundle_manifest_sha256=_required_text(
                values["bundle_manifest_sha256"],
                "bundle_manifest_sha256",
                _SHA256,
            ),
        )

    def installed_path(self, relative: str) -> str:
        """Return a fixed bundle-relative installed path."""

        return str(PurePosixPath(self.bundle_root) / relative)


@dataclass(frozen=True)
class _EnterpriseLimits:
    max_nodes: int
    max_ranks: int
    max_threads_per_rank: int

    def validate_resources(
        self,
        *,
        nodes: int,
        ranks_per_node: int,
        cores_per_node: int,
        threads_per_rank: Optional[int] = None,
    ) -> None:
        """Enforce the compatibility row's bounded scheduler request."""

        for value, label in (
            (nodes, "nodes"),
            (ranks_per_node, "ranks_per_node"),
            (cores_per_node, "cores_per_node"),
        ):
            _positive_int(value, label)
        if nodes > self.max_nodes:
            raise EnterpriseHPCPreflightError(
                "Enterprise HPC request exceeds max_nodes"
            )
        total_ranks = nodes * ranks_per_node
        if total_ranks > self.max_ranks:
            raise EnterpriseHPCPreflightError(
                "Enterprise HPC request exceeds max_ranks"
            )
        if ranks_per_node > cores_per_node:
            raise EnterpriseHPCPreflightError(
                "Enterprise HPC ranks_per_node exceeds node cores"
            )
        requested_threads = (
            cores_per_node // ranks_per_node
            if threads_per_rank is None
            else _positive_int(threads_per_rank, "threads_per_rank")
        )
        if ranks_per_node * requested_threads > cores_per_node:
            raise EnterpriseHPCPreflightError(
                "Enterprise HPC request exceeds node cores"
            )
        if requested_threads > self.max_threads_per_rank:
            raise EnterpriseHPCPreflightError(
                "Enterprise HPC request exceeds max_threads_per_rank"
            )

    def validate_allocation(self, *, nodes: int, ranks: int, cores: int) -> None:
        """Reject an attached allocation outside certified resource bounds."""

        for value, label in ((nodes, "nodes"), (ranks, "ranks"), (cores, "cores")):
            _positive_int(value, f"allocation_{label}")
        if nodes > self.max_nodes:
            raise EnterpriseHPCPreflightError(
                "Enterprise HPC allocation exceeds max_nodes"
            )
        if ranks > self.max_ranks:
            raise EnterpriseHPCPreflightError(
                "Enterprise HPC allocation exceeds max_ranks"
            )
        if cores < ranks or cores % ranks:
            raise EnterpriseHPCPreflightError(
                "Enterprise HPC allocation cores must divide evenly across ranks"
            )
        if cores // ranks > self.max_threads_per_rank:
            raise EnterpriseHPCPreflightError(
                "Enterprise HPC allocation exceeds max_threads_per_rank"
            )


@dataclass(frozen=True)
class _BundleSnapshot:
    bundle: Mapping[str, Any]
    bundle_sha256: str
    compatibility: Mapping[str, Any]
    compatibility_sha256: str
    bundle_schema: Mapping[str, Any]
    bundle_schema_sha256: str
    compatibility_schema: Mapping[str, Any]
    compatibility_schema_sha256: str


@dataclass(frozen=True)
class _RuntimeObservation:
    host: str
    scheduler_version: str
    cores_per_node: int
    mpi_launcher: str
    solver_path: str
    solver_identity: Mapping[str, str]
    frequensolve_version: str
    frequensolve_artifact_sha256: str


@dataclass(frozen=True)
class EnterpriseHPCPreflightResult:
    """Sanitized immutable identities observed before scheduler submission."""

    profile_id: str
    host: str
    scheduler: str
    scheduler_version: str
    bundle_version: str
    bundle_content_sha256: str
    bundle_manifest_sha256: str
    bundle_schema_sha256: str
    support_tier: str
    compatibility_row_id: str
    compatibility_document_sha256: str
    compatibility_schema_sha256: str
    solver_version: str
    solver_source_commit: str
    solver_build_id: str
    solver_build_identity_sha256: str
    solver_executable_sha256: str
    frequensolve_version: str
    frequensolve_artifact_sha256: str

    def to_evidence(self) -> dict[str, str]:
        """Return a customer-safe mapping suitable for a run record."""

        return dict(self.__dict__)


@dataclass(frozen=True)
class _ValidatedEnterpriseBundle:
    result: EnterpriseHPCPreflightResult
    limits: _EnterpriseLimits


def _validate_document_against_schema(
    document: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    label: str,
    expected_schema_id: str,
) -> None:
    """Apply the immutable Deployment-owned schema without copying it here."""

    if schema.get("$id") != expected_schema_id:
        raise EnterpriseHPCPreflightError(
            f"Enterprise HPC {label} schema identity is invalid"
        )
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema, format_checker=FormatChecker()).validate(document)
    except (SchemaError, ValidationError) as exc:
        raise EnterpriseHPCPreflightError(
            f"Enterprise HPC {label} does not satisfy the pinned Deployment v1 schema"
        ) from exc


def _metadata_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnterpriseHPCPreflightError(
            f"Enterprise HPC {label} metadata is incomplete"
        )
    return value


def _metadata_objects(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EnterpriseHPCPreflightError(
            f"Enterprise HPC {label} metadata is incomplete"
        )
    items = tuple(value)
    if not items or any(not isinstance(item, Mapping) for item in items):
        raise EnterpriseHPCPreflightError(
            f"Enterprise HPC {label} metadata is incomplete"
        )
    return tuple(item for item in items if isinstance(item, Mapping))


def _installed_file_digest(bundle: Mapping[str, Any], path: str) -> str:
    records = _metadata_objects(bundle.get("installedFiles"), "installed file")
    matches = [record for record in records if record.get("path") == path]
    if len(matches) != 1 or matches[0].get("kind") != "file":
        raise EnterpriseHPCPreflightError(
            f"Enterprise HPC bundle does not uniquely declare {path}"
        )
    digest = matches[0].get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise EnterpriseHPCPreflightError(
            f"Enterprise HPC bundle digest metadata for {path} is invalid"
        )
    return digest


def _identity_sha256(identity: Mapping[str, str]) -> str:
    expected = {"schema", "product", "version", "build_id", "git_commit"}
    if set(identity) != expected or any(
        not isinstance(identity.get(field), str) or not identity[field]
        for field in expected
    ):
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver public identity is incomplete"
        )
    encoded = (
        json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_bundle_snapshot(
    profile: _EnterpriseHPCProfile,
    snapshot: _BundleSnapshot,
    runtime: _RuntimeObservation,
) -> _ValidatedEnterpriseBundle:
    """Validate one digest-anchored bundle against observed runtime state."""

    if snapshot.bundle_sha256 != profile.bundle_manifest_sha256:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC bundle manifest digest does not match the profile"
        )
    trusted_files = {
        COMPATIBILITY_MANIFEST_RELATIVE: snapshot.compatibility_sha256,
        BUNDLE_SCHEMA_RELATIVE: snapshot.bundle_schema_sha256,
        COMPATIBILITY_SCHEMA_RELATIVE: snapshot.compatibility_schema_sha256,
    }
    mismatched_files = [
        path
        for path, digest in trusted_files.items()
        if _installed_file_digest(snapshot.bundle, path) != digest
    ]
    if mismatched_files:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC installed contract digest mismatch: "
            + ", ".join(sorted(mismatched_files))
        )

    _validate_document_against_schema(
        snapshot.bundle,
        snapshot.bundle_schema,
        label="bundle manifest",
        expected_schema_id=BUNDLE_SCHEMA_ID,
    )
    _validate_document_against_schema(
        snapshot.compatibility,
        snapshot.compatibility_schema,
        label="compatibility manifest",
        expected_schema_id=COMPATIBILITY_SCHEMA_ID,
    )

    bundle = snapshot.bundle
    compatibility = snapshot.compatibility
    upstream = _metadata_object(bundle.get("upstreamArtifacts"), "upstream artifact")
    sauce = _metadata_object(upstream.get("sauce"), "Sauce artifact")
    python = _metadata_object(upstream.get("frequensolve"), "FrequenSolve artifact")
    reference = _metadata_object(bundle.get("compatibility"), "compatibility reference")
    solver = _metadata_object(compatibility.get("solver"), "solver compatibility")
    consumer = _metadata_object(
        compatibility.get("frequensolve"), "FrequenSolve compatibility"
    )
    toolchain = _metadata_object(
        compatibility.get("toolchain"), "toolchain compatibility"
    )
    limits_document = _metadata_object(compatibility.get("limits"), "resource limit")
    platform = _metadata_object(compatibility.get("platform"), "platform compatibility")
    cpu = _metadata_object(platform.get("cpu"), "CPU compatibility")
    executables = _metadata_objects(bundle.get("executables"), "executable")
    if bundle.get("releaseState") not in {"candidate", "released"}:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC bundle is withdrawn or has an unsupported release state"
        )

    bundle_root = PurePosixPath(profile.bundle_root)
    try:
        solver_relative = str(
            PurePosixPath(runtime.solver_path).relative_to(bundle_root)
        )
    except ValueError as exc:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver path is outside the immutable bundle root"
        ) from exc
    executable_matches = [
        item for item in executables if item.get("path") == solver_relative
    ]
    if len(executable_matches) != 1:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver path is not uniquely declared by the bundle"
        )
    executable = executable_matches[0]
    executable_sha256 = _installed_file_digest(bundle, solver_relative)
    variants = solver.get("variants")
    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver variant metadata is invalid"
        )

    identity = runtime.solver_identity
    build_identity_sha256 = _identity_sha256(identity)
    expected = {
        "bundle schema": (bundle.get("schemaVersion"), BUNDLE_SCHEMA),
        "compatibility schema": (
            compatibility.get("schemaVersion"),
            COMPATIBILITY_SCHEMA,
        ),
        "compatibility support tier": (
            compatibility.get("supportTier"),
            bundle.get("supportTier"),
        ),
        "compatibility row": (reference.get("rowId"), compatibility.get("rowId")),
        "compatibility row document": (
            reference.get("documentSha256"),
            snapshot.compatibility_sha256,
        ),
        "solver version": (identity.get("version"), sauce.get("version")),
        "solver source commit": (
            identity.get("git_commit"),
            sauce.get("sourceCommit"),
        ),
        "solver build identity": (
            build_identity_sha256,
            sauce.get("producerIdentitySha256"),
        ),
        "compatibility solver version": (solver.get("version"), sauce.get("version")),
        "compatibility solver commit": (
            solver.get("sourceCommit"),
            sauce.get("sourceCommit"),
        ),
        "compatibility solver identity": (
            solver.get("buildIdentitySha256"),
            sauce.get("producerIdentitySha256"),
        ),
        "FrequenSolve version": (
            runtime.frequensolve_version,
            python.get("version"),
        ),
        "FrequenSolve artifact": (
            runtime.frequensolve_artifact_sha256,
            python.get("sha256"),
        ),
        "compatibility FrequenSolve version": (
            consumer.get("version"),
            python.get("version"),
        ),
        "compatibility FrequenSolve artifact": (
            consumer.get("artifactSha256"),
            python.get("sha256"),
        ),
        "compatibility scheduler": (
            str(toolchain.get("slurm", "")).removeprefix("slurm "),
            runtime.scheduler_version.lower().removeprefix("slurm "),
        ),
        "compatibility launcher": (
            toolchain.get("launcher"),
            runtime.mpi_launcher,
        ),
        "compatibility cores per node": (
            cpu.get("coresPerNode"),
            runtime.cores_per_node,
        ),
    }
    mismatches = [label for label, pair in expected.items() if pair[0] != pair[1]]
    if mismatches:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC compatibility mismatch: " + ", ".join(mismatches)
        )
    if executable.get("variant") not in variants:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver executable variant is not certified"
        )

    limits = _EnterpriseLimits(
        max_nodes=_positive_int(limits_document.get("maxNodes"), "max_nodes"),
        max_ranks=_positive_int(limits_document.get("maxRanks"), "max_ranks"),
        max_threads_per_rank=_positive_int(
            limits_document.get("maxThreadsPerRank"), "max_threads_per_rank"
        ),
    )
    return _ValidatedEnterpriseBundle(
        result=EnterpriseHPCPreflightResult(
            profile_id=profile.profile_id,
            host=runtime.host,
            scheduler="slurm",
            scheduler_version=runtime.scheduler_version,
            bundle_version=str(bundle.get("bundleVersion")),
            bundle_content_sha256=str(bundle.get("contentSha256")),
            bundle_manifest_sha256=snapshot.bundle_sha256,
            bundle_schema_sha256=snapshot.bundle_schema_sha256,
            support_tier=str(bundle.get("supportTier")),
            compatibility_row_id=str(compatibility.get("rowId")),
            compatibility_document_sha256=snapshot.compatibility_sha256,
            compatibility_schema_sha256=snapshot.compatibility_schema_sha256,
            solver_version=str(identity["version"]),
            solver_source_commit=str(identity["git_commit"]),
            solver_build_id=str(identity["build_id"]),
            solver_build_identity_sha256=build_identity_sha256,
            solver_executable_sha256=executable_sha256,
            frequensolve_version=runtime.frequensolve_version,
            frequensolve_artifact_sha256=runtime.frequensolve_artifact_sha256,
        ),
        limits=limits,
    )
