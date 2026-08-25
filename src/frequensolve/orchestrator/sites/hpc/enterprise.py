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
        "EnterpriseHPCProfile",
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

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+@/-]*\Z")
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*\Z")
_PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{2,127}\Z")
_CONTRACT_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\Z")
_BUILD_ID = re.compile(r"[ -~]+\Z")
_SAFE_ABSOLUTE_PATH = re.compile(r"/[A-Za-z0-9._+@/=-]+\Z")
_RELATIVE_PATH = re.compile(
    r"(?!/)(?!.*(?:^|/)\.\.?(?:/|$))(?!.*\\)[A-Za-z0-9._+/-]+\Z"
)
_IMMUTABLE_LOCATOR = re.compile(
    r"(?:https://github\.com/[^/]+/[^/]+/releases/download/[^/]+/[^/]+"
    r"|oci://[^@]+@sha256:[0-9a-f]{64}"
    r"|https://[^?#]+\?[^#]*(?:versionid|versionId)=[A-Za-z0-9._~-]+.*)\Z"
)
_WALL_TIME = re.compile(r"(?:(?:0|[1-9][0-9]*)-)?[0-9]{2}:[0-5][0-9]:[0-5][0-9]\Z")


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
        path = Path(installed.locate_file(relative))
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
    if record_rows != 1:
        return None
    if not generated_cache_sources.issubset(verified_paths):
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
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"enterprise_hpc.{name} contains a control character")
    return str(path)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"enterprise_hpc.{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class EnterpriseHPCProfile:
    """Strict generic Slurm profile for one immutable Enterprise HPC bundle."""

    profile_id: str
    host: str
    bundle_manifest: str
    compatibility_manifest: str
    bundle_schema_path: str
    compatibility_schema_path: str
    bundle_root: str
    solver_path: str
    work_dir: str
    scratch_dir: Optional[str]
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
    frequensolve_version: str
    frequensolve_artifact_sha256: str
    allowed_partitions: tuple[str, ...]
    max_nodes: int
    max_ranks: int
    max_threads_per_rank: int
    max_wall_time: str
    module: Optional[str] = None
    python_environment: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    mpi_launcher: str = "srun"
    schema: str = PROFILE_SCHEMA

    def __post_init__(self) -> None:
        """Enforce the contract for direct constructors as well as TOML parsing."""

        if self.schema != PROFILE_SCHEMA:
            raise ValueError(f"enterprise_hpc.schema must be {PROFILE_SCHEMA!r}")
        for name, value, pattern in (
            ("profile_id", self.profile_id, _PROFILE_ID),
            ("host", self.host, _HOST),
            ("bundle_version", self.bundle_version, _VERSION),
            ("bundle_content_sha256", self.bundle_content_sha256, _SHA256),
            ("bundle_manifest_sha256", self.bundle_manifest_sha256, _SHA256),
            ("bundle_schema_sha256", self.bundle_schema_sha256, _SHA256),
            ("compatibility_row_id", self.compatibility_row_id, _PROFILE_ID),
            (
                "compatibility_document_sha256",
                self.compatibility_document_sha256,
                _SHA256,
            ),
            (
                "compatibility_schema_sha256",
                self.compatibility_schema_sha256,
                _SHA256,
            ),
            ("solver_version", self.solver_version, _VERSION),
            ("solver_source_commit", self.solver_source_commit, _COMMIT),
            ("solver_build_id", self.solver_build_id, _BUILD_ID),
            (
                "solver_build_identity_sha256",
                self.solver_build_identity_sha256,
                _SHA256,
            ),
            ("frequensolve_version", self.frequensolve_version, _VERSION),
            (
                "frequensolve_artifact_sha256",
                self.frequensolve_artifact_sha256,
                _SHA256,
            ),
            ("max_wall_time", self.max_wall_time, _WALL_TIME),
        ):
            _required_text(value, name, pattern)
        for path_name, path_value in (
            ("bundle_manifest", self.bundle_manifest),
            ("compatibility_manifest", self.compatibility_manifest),
            ("bundle_schema_path", self.bundle_schema_path),
            ("compatibility_schema_path", self.compatibility_schema_path),
            ("bundle_root", self.bundle_root),
            ("solver_path", self.solver_path),
            ("work_dir", self.work_dir),
        ):
            _absolute_remote_path(path_value, path_name)
        root = PurePosixPath(self.bundle_root)
        for path_name, path_value in (
            ("bundle_manifest", self.bundle_manifest),
            ("compatibility_manifest", self.compatibility_manifest),
            ("bundle_schema_path", self.bundle_schema_path),
            ("compatibility_schema_path", self.compatibility_schema_path),
            ("solver_path", self.solver_path),
        ):
            try:
                relative = PurePosixPath(path_value).relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"enterprise_hpc.{path_name} must be inside bundle_root"
                ) from exc
            if str(relative) == ".":
                raise ValueError(
                    f"enterprise_hpc.{path_name} must name a file inside bundle_root"
                )
        if self.scratch_dir is not None:
            _absolute_remote_path(self.scratch_dir, "scratch_dir")
        if self.python_environment is not None:
            _absolute_remote_path(self.python_environment, "python_environment")
        if not self.allowed_partitions or len(self.allowed_partitions) != len(
            set(self.allowed_partitions)
        ):
            raise ValueError(
                "enterprise_hpc.allowed_partitions must be a non-empty unique array"
            )
        for partition in self.allowed_partitions:
            _required_text(partition, "allowed_partitions", _TOKEN)
        for integer_name, integer_value in (
            ("max_nodes", self.max_nodes),
            ("max_ranks", self.max_ranks),
            ("max_threads_per_rank", self.max_threads_per_rank),
        ):
            _positive_int(integer_value, integer_name)
        for optional_name, optional_value in (
            ("module", self.module),
            ("account", self.account),
            ("qos", self.qos),
        ):
            if optional_value is not None:
                _required_text(optional_value, optional_name, _TOKEN)
        if self.mpi_launcher not in {"srun", "mpirun", "mpiexec"}:
            raise ValueError(
                "enterprise_hpc.mpi_launcher must be srun, mpirun, or mpiexec"
            )
        if self.support_tier not in {"experimental", "compatible", "certified"}:
            raise ValueError(
                "enterprise_hpc.support_tier must be experimental, compatible, or certified"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EnterpriseHPCProfile":
        """Parse a closed profile mapping and reject unsafe values."""

        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                "Unsupported enterprise_hpc profile key(s): " + ", ".join(unknown)
            )
        schema = values.get("schema", PROFILE_SCHEMA)
        if schema != PROFILE_SCHEMA:
            raise ValueError(f"enterprise_hpc.schema must be {PROFILE_SCHEMA!r}")
        partitions = values.get("allowed_partitions")
        if (
            not isinstance(partitions, Sequence)
            or isinstance(partitions, (str, bytes))
            or not partitions
        ):
            raise ValueError(
                "enterprise_hpc.allowed_partitions must be a non-empty array"
            )
        normalized_partitions = tuple(
            _required_text(item, "allowed_partitions", _TOKEN) for item in partitions
        )
        if len(normalized_partitions) != len(set(normalized_partitions)):
            raise ValueError("enterprise_hpc.allowed_partitions contains duplicates")

        optional_tokens = {}
        for name in ("module", "account", "qos"):
            value = values.get(name)
            optional_tokens[name] = (
                None if value is None else _required_text(value, name, _TOKEN)
            )
        python_environment = values.get("python_environment")
        if python_environment is not None:
            python_environment = _absolute_remote_path(
                python_environment, "python_environment"
            )
        launcher = _required_text(
            values.get("mpi_launcher", "srun"), "mpi_launcher", _TOKEN
        )
        if launcher not in {"srun", "mpirun", "mpiexec"}:
            raise ValueError(
                "enterprise_hpc.mpi_launcher must be srun, mpirun, or mpiexec"
            )
        support_tier = _required_text(
            values.get("support_tier"), "support_tier", _TOKEN
        )
        if support_tier not in {"experimental", "compatible", "certified"}:
            raise ValueError(
                "enterprise_hpc.support_tier must be experimental, compatible, or certified"
            )

        required = {
            "profile_id",
            "host",
            "bundle_manifest",
            "compatibility_manifest",
            "bundle_schema_path",
            "compatibility_schema_path",
            "bundle_root",
            "solver_path",
            "work_dir",
            "bundle_version",
            "bundle_content_sha256",
            "bundle_manifest_sha256",
            "bundle_schema_sha256",
            "compatibility_row_id",
            "compatibility_document_sha256",
            "compatibility_schema_sha256",
            "solver_version",
            "solver_source_commit",
            "solver_build_id",
            "solver_build_identity_sha256",
            "frequensolve_version",
            "frequensolve_artifact_sha256",
            "max_nodes",
            "max_ranks",
            "max_threads_per_rank",
            "max_wall_time",
            "support_tier",
        }
        missing = sorted(name for name in required if name not in values)
        if missing:
            raise ValueError(
                "Missing enterprise_hpc profile key(s): " + ", ".join(missing)
            )
        return cls(
            schema=PROFILE_SCHEMA,
            profile_id=_required_text(values["profile_id"], "profile_id", _PROFILE_ID),
            host=_required_text(values["host"], "host", _HOST),
            bundle_manifest=_absolute_remote_path(
                values["bundle_manifest"], "bundle_manifest"
            ),
            compatibility_manifest=_absolute_remote_path(
                values["compatibility_manifest"], "compatibility_manifest"
            ),
            bundle_schema_path=_absolute_remote_path(
                values["bundle_schema_path"], "bundle_schema_path"
            ),
            compatibility_schema_path=_absolute_remote_path(
                values["compatibility_schema_path"], "compatibility_schema_path"
            ),
            bundle_root=_absolute_remote_path(values["bundle_root"], "bundle_root"),
            solver_path=_absolute_remote_path(values["solver_path"], "solver_path"),
            work_dir=_absolute_remote_path(values["work_dir"], "work_dir"),
            scratch_dir=(
                None
                if values.get("scratch_dir") is None
                else _absolute_remote_path(values["scratch_dir"], "scratch_dir")
            ),
            bundle_version=_required_text(
                values["bundle_version"], "bundle_version", _VERSION
            ),
            bundle_content_sha256=_required_text(
                values["bundle_content_sha256"], "bundle_content_sha256", _SHA256
            ),
            bundle_manifest_sha256=_required_text(
                values["bundle_manifest_sha256"], "bundle_manifest_sha256", _SHA256
            ),
            bundle_schema_sha256=_required_text(
                values["bundle_schema_sha256"], "bundle_schema_sha256", _SHA256
            ),
            compatibility_row_id=_required_text(
                values["compatibility_row_id"], "compatibility_row_id", _PROFILE_ID
            ),
            compatibility_document_sha256=_required_text(
                values["compatibility_document_sha256"],
                "compatibility_document_sha256",
                _SHA256,
            ),
            compatibility_schema_sha256=_required_text(
                values["compatibility_schema_sha256"],
                "compatibility_schema_sha256",
                _SHA256,
            ),
            solver_version=_required_text(
                values["solver_version"], "solver_version", _VERSION
            ),
            solver_source_commit=_required_text(
                values["solver_source_commit"], "solver_source_commit", _COMMIT
            ),
            solver_build_id=_required_text(
                values["solver_build_id"], "solver_build_id", _BUILD_ID
            ),
            solver_build_identity_sha256=_required_text(
                values["solver_build_identity_sha256"],
                "solver_build_identity_sha256",
                _SHA256,
            ),
            frequensolve_version=_required_text(
                values["frequensolve_version"], "frequensolve_version", _VERSION
            ),
            frequensolve_artifact_sha256=_required_text(
                values["frequensolve_artifact_sha256"],
                "frequensolve_artifact_sha256",
                _SHA256,
            ),
            allowed_partitions=normalized_partitions,
            max_nodes=_positive_int(values["max_nodes"], "max_nodes"),
            max_ranks=_positive_int(values["max_ranks"], "max_ranks"),
            max_threads_per_rank=_positive_int(
                values["max_threads_per_rank"], "max_threads_per_rank"
            ),
            max_wall_time=_required_text(
                values["max_wall_time"], "max_wall_time", _WALL_TIME
            ),
            support_tier=support_tier,
            python_environment=python_environment,
            mpi_launcher=launcher,
            **optional_tokens,
        )

    def validate_site(
        self,
        *,
        host: str,
        partition: str,
        account: Optional[str],
        qos: Optional[str],
        mpi_launcher: str,
    ) -> None:
        """Reject a contradictory host, scheduler, or launcher profile."""

        _required_text(host, "host", _HOST)
        _required_text(partition, "partition", _TOKEN)
        _required_text(mpi_launcher, "mpi_launcher", _TOKEN)
        account = account or None
        for value, name in ((account, "account"), (qos, "qos")):
            if value is not None:
                _required_text(value, name, _TOKEN)
        checks = (
            (host, self.host, "host"),
            (mpi_launcher, self.mpi_launcher, "MPI launcher"),
        )
        for actual, expected, label in checks:
            if actual != expected:
                raise ValueError(
                    f"Enterprise HPC {label} {actual!r} does not match {expected!r}"
                )
        if partition not in self.allowed_partitions:
            raise ValueError(
                f"Enterprise HPC partition {partition!r} is not allowlisted"
            )
        optional_checks: tuple[tuple[Optional[str], Optional[str], str], ...] = (
            (account, self.account, "account"),
            (qos, self.qos, "QOS"),
        )
        for optional_actual, optional_expected, optional_label in optional_checks:
            if optional_actual != optional_expected:
                raise ValueError(
                    f"Enterprise HPC {optional_label} {optional_actual!r} does not "
                    f"match {optional_expected!r}"
                )

    def validate_resources(
        self,
        *,
        nodes: int,
        ranks_per_node: int,
        cores_per_node: int,
        duration_seconds: int,
        max_wall_time_seconds: int,
    ) -> None:
        """Enforce the compatibility row's bounded scheduler request."""

        for value, label in (
            (nodes, "nodes"),
            (ranks_per_node, "ranks_per_node"),
            (cores_per_node, "cores_per_node"),
            (duration_seconds, "duration"),
            (max_wall_time_seconds, "max_wall_time"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Enterprise HPC {label} must be a positive integer")
        if nodes > self.max_nodes:
            raise ValueError("Enterprise HPC request exceeds max_nodes")
        total_ranks = nodes * ranks_per_node
        if total_ranks > self.max_ranks:
            raise ValueError("Enterprise HPC request exceeds max_ranks")
        if ranks_per_node < 1 or ranks_per_node > cores_per_node:
            raise ValueError("Enterprise HPC ranks_per_node exceeds node cores")
        threads_per_rank = cores_per_node // ranks_per_node
        if threads_per_rank > self.max_threads_per_rank:
            raise ValueError("Enterprise HPC request exceeds max_threads_per_rank")
        if duration_seconds > max_wall_time_seconds:
            raise ValueError("Enterprise HPC request exceeds max_wall_time")

    def validate_allocation(self, *, nodes: int, ranks: int, cores: int) -> None:
        """Reject an attached allocation outside certified resource bounds."""

        for value, label in ((nodes, "nodes"), (ranks, "ranks"), (cores, "cores")):
            _positive_int(value, f"allocation_{label}")
        if nodes > self.max_nodes:
            raise ValueError("Enterprise HPC allocation exceeds max_nodes")
        if ranks > self.max_ranks:
            raise ValueError("Enterprise HPC allocation exceeds max_ranks")
        if cores < ranks or cores % ranks:
            raise ValueError(
                "Enterprise HPC allocation cores must divide evenly across ranks"
            )
        if cores // ranks > self.max_threads_per_rank:
            raise ValueError("Enterprise HPC allocation exceeds max_threads_per_rank")


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
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).validate(document)
    except (SchemaError, ValidationError) as exc:
        raise EnterpriseHPCPreflightError(
            f"Enterprise HPC {label} does not satisfy the pinned Deployment v1 schema"
        ) from exc


def _metadata_object(value: Any, label: str) -> Mapping[str, Any]:
    """Return a schema-validated object with a safe defensive error."""

    if not isinstance(value, Mapping):
        raise EnterpriseHPCPreflightError(
            f"Enterprise HPC {label} metadata is incomplete"
        )
    return value


def _metadata_objects(value: Any, label: str) -> tuple[Mapping[str, Any], ...]:
    """Return a schema-validated object array with a safe defensive error."""

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


def validate_bundle_pair(
    profile: EnterpriseHPCProfile,
    bundle: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    *,
    bundle_sha256: str,
    compatibility_sha256: str,
    bundle_schema: Mapping[str, Any],
    compatibility_schema: Mapping[str, Any],
    scheduler_version: str,
    cores_per_node: int,
) -> EnterpriseHPCPreflightResult:
    """Validate immutable identities and the complete Deployment v1 shape."""

    _validate_document_against_schema(
        bundle,
        bundle_schema,
        label="bundle manifest",
        expected_schema_id=BUNDLE_SCHEMA_ID,
    )
    _validate_document_against_schema(
        compatibility,
        compatibility_schema,
        label="compatibility manifest",
        expected_schema_id=COMPATIBILITY_SCHEMA_ID,
    )
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
    limits = _metadata_object(compatibility.get("limits"), "resource limit")
    platform = _metadata_object(compatibility.get("platform"), "platform compatibility")
    cpu = _metadata_object(platform.get("cpu"), "CPU compatibility")
    executables = _metadata_objects(bundle.get("executables"), "executable")
    installed_files = _metadata_objects(bundle.get("installedFiles"), "installed file")
    if bundle.get("releaseState") not in {"candidate", "released"}:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC bundle is withdrawn or has an unsupported release state"
        )

    bundle_root = PurePosixPath(profile.bundle_root)
    try:
        solver_relative = str(
            PurePosixPath(profile.solver_path).relative_to(bundle_root)
        )
    except ValueError as exc:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver path is outside the immutable bundle root"
        ) from exc
    executable_matches = [
        item for item in executables if item.get("path") == solver_relative
    ]
    file_matches = [
        item for item in installed_files if item.get("path") == solver_relative
    ]
    if len(executable_matches) != 1 or len(file_matches) != 1:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver path is not uniquely declared by the bundle"
        )
    executable = executable_matches[0]
    installed_file = file_matches[0]
    if installed_file.get("kind") != "file" or not str(
        installed_file.get("mode", "")
    ).endswith(("1", "3", "5", "7")):
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver bundle entry is not executable"
        )
    executable_sha256 = installed_file.get("sha256")
    variants = solver.get("variants")
    if not isinstance(executable_sha256, str) or not _SHA256.fullmatch(
        executable_sha256
    ):
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver bundle digest metadata is invalid"
        )
    if isinstance(variants, (str, bytes)) or not isinstance(variants, Sequence):
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver variant metadata is invalid"
        )
    expected = {
        "bundle schema": (bundle.get("schemaVersion"), BUNDLE_SCHEMA),
        "bundle version": (bundle.get("bundleVersion"), profile.bundle_version),
        "bundle content digest": (
            bundle.get("contentSha256"),
            profile.bundle_content_sha256,
        ),
        "bundle manifest digest": (bundle_sha256, profile.bundle_manifest_sha256),
        "bundle support tier": (bundle.get("supportTier"), profile.support_tier),
        "compatibility schema": (
            compatibility.get("schemaVersion"),
            COMPATIBILITY_SCHEMA,
        ),
        "compatibility support tier": (
            compatibility.get("supportTier"),
            profile.support_tier,
        ),
        "compatibility row": (reference.get("rowId"), profile.compatibility_row_id),
        "compatibility row document": (
            reference.get("documentSha256"),
            compatibility_sha256,
        ),
        "expected compatibility document": (
            compatibility_sha256,
            profile.compatibility_document_sha256,
        ),
        "solver version": (sauce.get("version"), profile.solver_version),
        "solver source commit": (
            sauce.get("sourceCommit"),
            profile.solver_source_commit,
        ),
        "solver build identity": (
            sauce.get("producerIdentitySha256"),
            profile.solver_build_identity_sha256,
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
            python.get("version"),
            profile.frequensolve_version,
        ),
        "FrequenSolve artifact": (
            python.get("sha256"),
            profile.frequensolve_artifact_sha256,
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
            scheduler_version.lower().removeprefix("slurm "),
        ),
        "compatibility launcher": (toolchain.get("launcher"), profile.mpi_launcher),
        "compatibility max nodes": (limits.get("maxNodes"), profile.max_nodes),
        "compatibility max ranks": (limits.get("maxRanks"), profile.max_ranks),
        "compatibility max threads": (
            limits.get("maxThreadsPerRank"),
            profile.max_threads_per_rank,
        ),
        "compatibility cores per node": (cpu.get("coresPerNode"), cores_per_node),
    }
    mismatches = [label for label, pair in expected.items() if pair[0] != pair[1]]
    if mismatches:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC compatibility mismatch: " + ", ".join(mismatches)
        )
    row_id = compatibility.get("rowId")
    if row_id != reference.get("rowId"):
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC compatibility row does not match the bundle"
        )
    if executable.get("variant") not in variants:
        raise EnterpriseHPCPreflightError(
            "Enterprise HPC solver executable variant is not certified"
        )
    return EnterpriseHPCPreflightResult(
        profile_id=profile.profile_id,
        host=profile.host,
        scheduler="slurm",
        scheduler_version=scheduler_version,
        bundle_version=profile.bundle_version,
        bundle_content_sha256=profile.bundle_content_sha256,
        bundle_manifest_sha256=bundle_sha256,
        bundle_schema_sha256=profile.bundle_schema_sha256,
        support_tier=profile.support_tier,
        compatibility_row_id=profile.compatibility_row_id,
        compatibility_document_sha256=compatibility_sha256,
        compatibility_schema_sha256=profile.compatibility_schema_sha256,
        solver_version=profile.solver_version,
        solver_source_commit=profile.solver_source_commit,
        solver_build_id=profile.solver_build_id,
        solver_build_identity_sha256=profile.solver_build_identity_sha256,
        solver_executable_sha256=executable_sha256,
        frequensolve_version=profile.frequensolve_version,
        frequensolve_artifact_sha256=profile.frequensolve_artifact_sha256,
    )
