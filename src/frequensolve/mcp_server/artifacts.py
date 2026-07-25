"""Read-only inspection of narrowly supported saved simulation artifacts.

This module is deliberately independent of the MCP SDK.  Callers are expected
to run :func:`inspect_or_validate_artifact` in a cancellable worker process and
to provide a private, caller-owned temporary directory as ``sandbox_root``.
Only bounded JSON is read from configured roots.  The original tree is never
written to.
"""

from __future__ import annotations

import copy
import errno
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Optional

from frequensolve.simulation.jobs.forward import (
    FrequencyDomainJob,
    TimeDomainJob,
)
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.validation import validate_job, validate_simulation

__all__ = [
    "ArtifactSafetyError",
    "inspect_or_validate_artifact",
    "normalize_allowed_roots",
]

_RESULT_SCHEMA = "frequensolve-mcp-artifact-result/v1"
_SUPPORTED_MODES = frozenset({"inspect", "preview", "validate"})
_SUPPORTED_ROOT_TYPES = frozenset(
    {"SeismicSimulation", "FrequencyDomainJob", "TimeDomainJob"}
)
_SUPPORTED_JOB_TYPES = frozenset({"FrequencyDomainJob", "TimeDomainJob"})

_MAX_ROOTS = 32
_MAX_PATH_LENGTH = 1024
_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 50_000
_MAX_LIST_ITEMS = 10_000
_MAX_STRING_LENGTH = 64 * 1024
_MAX_KEY_LENGTH = 256
_MAX_ISSUES = 256
_MAX_SCALAR_COUNT = 100_000
_MAX_SHAPE_PRODUCT = 1_000_000

_ROOT_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_URI_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]{1,31}://")
_ENCODED_PATH_RE = re.compile(r"%(?:25|2e|2f|5c)", re.IGNORECASE)
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_SAFE_ISSUE_PATH_RE = re.compile(r"^[A-Za-z0-9_.\[\]-]{1,256}$")
_FILE_LOCATOR_RE = re.compile(
    r"(?:^|[/])[^/]+\.(?:"
    r"asdf|bin|csv|dat|h5|hdf5|json|npy|npz|rsf|segy|sgy|txt|vtk|vtu"
    r")(?:[:#?].*)?$",
    re.IGNORECASE,
)

# Concrete built-in payload tags that deserialize without selecting a job,
# project, site, result, or arbitrary user-provided Python class.
_SUPPORTED_NESTED_TYPES = frozenset(
    {
        "BaseMeshGenerator",
        "CompoundSource",
        "CoordinateSystem",
        "CoordsArray",
        "Fracture",
        "HexMeshGenerator",
        "Inline",
        "JsonDense",
        "LayeredMeshGenerator",
        "LayeredModel",
        "ModelBase",
        "Named",
        "ParaviewOutput",
        "PointSource",
        "PowerLawDispersion",
        "ReceiverFiber",
        "ReceiverNode",
        "ReceiverNodeArray",
        "RuptureSource",
        "SimpleSurface",
        "Sparse",
        "SurfaceCoordinateSystem",
        "TablulatedDispersion",
        "TetMeshGenerator",
        "TraceOutput",
        "VtkOutput",
    }
)

_REFERENCE_KEYS = frozenset(
    {
        "data_file",
        "data_path",
        "dataset",
        "dataset_path",
        "dir",
        "directory",
        "file",
        "file_grid",
        "file_path",
        "filename",
        "files",
        "href",
        "hdf5",
        "hdf5_file",
        "path",
        "paths",
        "project_path",
        "result_path",
        "simulation",
        "source_file",
        "store",
        "uri",
        "url",
    }
)
_REFERENCE_SUFFIXES = (
    "_dir",
    "_directory",
    "_file",
    "_files",
    "_path",
    "_paths",
    "_uri",
    "_url",
)

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class ArtifactSafetyError(ValueError):
    """Stable, non-sensitive failure raised by low-level safety helpers."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _DuplicateKeyError(ValueError):
    pass


class _NonFiniteNumberError(ValueError):
    pass


def normalize_allowed_roots(
    roots: Mapping[str, str | os.PathLike[str]],
) -> dict[str, str]:
    """Validate configured root IDs and return canonical absolute directories.

    Root configuration is trusted administrative input, but every returned
    directory is reopened component-by-component without following symlinks
    before artifact access.
    """

    if not isinstance(roots, Mapping) or len(roots) > _MAX_ROOTS:
        raise ArtifactSafetyError("artifact.root.invalid")

    normalized: dict[str, str] = {}
    for root_id, raw_root in roots.items():
        if not isinstance(root_id, str) or _ROOT_ID_RE.fullmatch(root_id) is None:
            raise ArtifactSafetyError("artifact.root.invalid")
        if root_id in normalized:
            raise ArtifactSafetyError("artifact.root.invalid")
        normalized[root_id] = _normalize_directory(raw_root, "artifact.root.invalid")
    return normalized


def inspect_or_validate_artifact(
    roots: Mapping[str, str | os.PathLike[str]],
    root_id: str,
    relative_path: str,
    mode: str,
    sandbox_root: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Any]:
    """Safely deserialize and validate one supported saved JSON artifact.

    ``sandbox_root`` must be a private temporary directory owned and cleaned by
    the caller.  Requiring that ownership keeps hard-cancelled worker processes
    from leaking temporary project trees.
    """

    safe_mode = mode if mode in _SUPPORTED_MODES else "unknown"
    try:
        if mode not in _SUPPORTED_MODES:
            raise ArtifactSafetyError("artifact.request.invalid")
        normalized_roots = _validate_pinned_roots(roots)
        if not isinstance(root_id, str) or root_id not in normalized_roots:
            raise ArtifactSafetyError("artifact.root.unknown")
        if sandbox_root is None:
            raise ArtifactSafetyError("artifact.sandbox.required")
        normalized_sandbox = _normalize_directory(
            sandbox_root, "artifact.sandbox.invalid"
        )
        _require_disjoint_sandbox(normalized_sandbox, normalized_roots.values())

        artifact_path = _validated_relative_path(relative_path, require_json=True)
        root = normalized_roots[root_id]
        payload = _load_json_from_root(root, artifact_path)
        root_type = _root_type(payload)

        simulation_payload: dict[str, Any]
        job_payload: Optional[dict[str, Any]]
        if root_type == "SeismicSimulation":
            _validate_payload_contract(payload, "simulation")
            simulation_payload = payload
            job_payload = None
        else:
            _validate_payload_contract(payload, "job")
            simulation_path = _job_simulation_path(payload, root)
            simulation_payload = _load_json_from_root(root, simulation_path)
            if _root_type(simulation_payload) != "SeismicSimulation":
                raise ArtifactSafetyError("artifact.simulation.unsupported")
            _validate_payload_contract(simulation_payload, "simulation")
            job_payload = payload

        work_dir = tempfile.mkdtemp(
            prefix="frequensolve-artifact-", dir=normalized_sandbox
        )
        os.chmod(work_dir, 0o700)
        artifact, report = _load_in_sandbox(
            Path(work_dir),
            simulation_payload=simulation_payload,
            job_payload=job_payload,
            root_type=root_type,
        )
        return {
            "schema": _RESULT_SCHEMA,
            "ok": bool(report.ok),
            "mode": mode,
            "artifact": _artifact_summary(artifact),
            "issues": _report_issues(report),
        }
    except ArtifactSafetyError as exc:
        return _error_result(safe_mode, exc.code)
    except (MemoryError, RecursionError):
        return _error_result(safe_mode, "artifact.json.limits")
    except Exception:
        return _error_result(safe_mode, "artifact.inspect.failed")


def _normalize_directory(raw_path: str | os.PathLike[str], failure_code: str) -> str:
    try:
        value = os.fspath(raw_path)
    except TypeError as exc:
        raise ArtifactSafetyError(failure_code) from exc
    if not isinstance(value, str) or not value or _has_control(value):
        raise ArtifactSafetyError(failure_code)
    if not os.path.isabs(value):
        raise ArtifactSafetyError(failure_code)

    # Canonicalizing trusted configuration makes common platform aliases such
    # as macOS /var -> /private/var stable.  Artifact paths themselves are
    # never canonicalized this way.
    canonical = os.path.realpath(value)
    if canonical == os.path.sep:
        raise ArtifactSafetyError(failure_code)
    descriptor: Optional[int] = None
    try:
        descriptor = _open_absolute_directory(canonical, failure_code)
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactSafetyError(failure_code)
    except OSError as exc:
        raise ArtifactSafetyError(failure_code) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return canonical


def _validate_pinned_roots(
    roots: Mapping[str, str | os.PathLike[str]],
) -> dict[str, str]:
    """Reopen startup-canonicalized roots without following a new symlink."""

    if not isinstance(roots, Mapping) or len(roots) > _MAX_ROOTS:
        raise ArtifactSafetyError("artifact.root.invalid")
    pinned: dict[str, str] = {}
    for root_id, raw_root in roots.items():
        if not isinstance(root_id, str) or _ROOT_ID_RE.fullmatch(root_id) is None:
            raise ArtifactSafetyError("artifact.root.invalid")
        try:
            root = os.fspath(raw_root)
        except TypeError as exc:
            raise ArtifactSafetyError("artifact.root.invalid") from exc
        if (
            not isinstance(root, str)
            or not root
            or _has_control(root)
            or not os.path.isabs(root)
            or os.path.normpath(root) != root
            or root == os.path.sep
        ):
            raise ArtifactSafetyError("artifact.root.invalid")
        descriptor: Optional[int] = None
        try:
            descriptor = _open_absolute_directory(root, "artifact.root.invalid")
        except OSError as exc:
            raise ArtifactSafetyError("artifact.root.invalid") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        pinned[root_id] = root
    return pinned


def _open_absolute_directory(path: str, failure_code: str) -> int:
    if not os.path.isabs(path) or getattr(os, "O_NOFOLLOW", 0) == 0:
        raise ArtifactSafetyError(failure_code)
    components = [part for part in path.split(os.path.sep) if part]
    descriptor = os.open(os.path.sep, _DIRECTORY_FLAGS)
    try:
        for component in components:
            metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactSafetyError(failure_code)
            next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            opened = os.fstat(next_descriptor)
            if (
                opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
                or not stat.S_ISDIR(opened.st_mode)
            ):
                os.close(next_descriptor)
                raise ArtifactSafetyError(failure_code)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _require_disjoint_sandbox(sandbox: str, roots: Any) -> None:
    for root in roots:
        try:
            common = os.path.commonpath((sandbox, root))
        except ValueError as exc:
            raise ArtifactSafetyError("artifact.sandbox.invalid") from exc
        if common in {sandbox, root}:
            raise ArtifactSafetyError("artifact.sandbox.invalid")


def _validated_relative_path(value: str, *, require_json: bool) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_PATH_LENGTH
        or _has_control(value)
        or value.startswith(("/", "~"))
        or "\\" in value
        or "%" in value
        or ":" in value
        or _WINDOWS_DRIVE_RE.match(value)
        or _URI_RE.search(value)
    ):
        raise ArtifactSafetyError("artifact.path.invalid")
    raw_parts = value.split("/")
    if any(
        not part or part in {".", ".."} or len(part.encode("utf-8")) > 255
        for part in raw_parts
    ):
        raise ArtifactSafetyError("artifact.path.invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or tuple(path.parts) != tuple(raw_parts):
        raise ArtifactSafetyError("artifact.path.invalid")
    if require_json and path.suffix != ".json":
        raise ArtifactSafetyError("artifact.path.unsupported")
    return path.as_posix()


def _load_json_from_root(root: str, relative_path: str) -> dict[str, Any]:
    data = _read_regular_file(root, relative_path)
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ArtifactSafetyError("artifact.json.encoding") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except _DuplicateKeyError as exc:
        raise ArtifactSafetyError("artifact.json.duplicate_key") from exc
    except _NonFiniteNumberError as exc:
        raise ArtifactSafetyError("artifact.json.nonfinite") from exc
    except RecursionError as exc:
        raise ArtifactSafetyError("artifact.json.limits") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactSafetyError("artifact.json.invalid") from exc
    _check_json_limits(payload)
    if not isinstance(payload, dict):
        raise ArtifactSafetyError("artifact.json.invalid")
    return payload


def _read_regular_file(root: str, relative_path: str) -> bytes:
    components = relative_path.split("/")
    root_descriptor: Optional[int] = None
    parent_descriptor: Optional[int] = None
    file_descriptor: Optional[int] = None
    try:
        root_descriptor = _open_absolute_directory(root, "artifact.root.invalid")
        parent_descriptor = root_descriptor
        root_descriptor = None
        for component in components[:-1]:
            metadata = _safe_lstat(component, parent_descriptor)
            if stat.S_ISLNK(metadata.st_mode):
                raise ArtifactSafetyError("artifact.path.symlink")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactSafetyError("artifact.path.invalid")
            next_descriptor = os.open(
                component, _DIRECTORY_FLAGS, dir_fd=parent_descriptor
            )
            opened = os.fstat(next_descriptor)
            if (
                opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
                or not stat.S_ISDIR(opened.st_mode)
            ):
                os.close(next_descriptor)
                raise ArtifactSafetyError("artifact.path.changed")
            os.close(parent_descriptor)
            parent_descriptor = next_descriptor

        name = components[-1]
        metadata = _safe_lstat(name, parent_descriptor)
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactSafetyError("artifact.path.symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactSafetyError("artifact.path.not_file")
        if metadata.st_size > _MAX_FILE_BYTES:
            raise ArtifactSafetyError("artifact.path.too_large")

        file_descriptor = os.open(name, _FILE_FLAGS, dir_fd=parent_descriptor)
        opened = os.fstat(file_descriptor)
        if (
            opened.st_dev != metadata.st_dev
            or opened.st_ino != metadata.st_ino
            or not stat.S_ISREG(opened.st_mode)
        ):
            raise ArtifactSafetyError("artifact.path.changed")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                file_descriptor, min(64 * 1024, _MAX_FILE_BYTES + 1 - total)
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_FILE_BYTES:
                raise ArtifactSafetyError("artifact.path.too_large")

        after = os.fstat(file_descriptor)
        if (
            after.st_dev != opened.st_dev
            or after.st_ino != opened.st_ino
            or after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise ArtifactSafetyError("artifact.path.changed")
        return b"".join(chunks)
    except ArtifactSafetyError:
        raise
    except OSError as exc:
        if exc.errno in {errno.ELOOP}:
            raise ArtifactSafetyError("artifact.path.symlink") from exc
        if exc.errno in {errno.ENOENT}:
            raise ArtifactSafetyError("artifact.path.not_found") from exc
        if exc.errno in {errno.ENOTDIR}:
            raise ArtifactSafetyError("artifact.path.invalid") from exc
        raise ArtifactSafetyError("artifact.path.unavailable") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
        if root_descriptor is not None:
            os.close(root_descriptor)


def _safe_lstat(name: str, directory_descriptor: int) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ArtifactSafetyError("artifact.path.not_found") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError
        result[key] = value
    return result


def _reject_nonfinite_constant(_value: str) -> Any:
    raise _NonFiniteNumberError


def _check_json_limits(payload: Any) -> None:
    stack: list[tuple[Any, int]] = [(payload, 1)]
    nodes = 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise ArtifactSafetyError("artifact.json.limits")
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str) or len(key) > _MAX_KEY_LENGTH:
                    raise ArtifactSafetyError("artifact.json.limits")
                nodes += 1
                if nodes > _MAX_JSON_NODES:
                    raise ArtifactSafetyError("artifact.json.limits")
                stack.append((child, depth + 1))
        elif isinstance(value, list):
            if len(value) > _MAX_LIST_ITEMS:
                raise ArtifactSafetyError("artifact.json.limits")
            stack.extend((child, depth + 1) for child in value)
        elif isinstance(value, str):
            if len(value) > _MAX_STRING_LENGTH:
                raise ArtifactSafetyError("artifact.json.limits")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ArtifactSafetyError("artifact.json.nonfinite")


def _root_type(payload: Mapping[str, Any]) -> str:
    root_type = payload.get("_type")
    if not isinstance(root_type, str) or root_type not in _SUPPORTED_ROOT_TYPES:
        raise ArtifactSafetyError("artifact.type.unsupported")
    return root_type


def _validate_payload_contract(payload: Mapping[str, Any], kind: str) -> None:
    expected_schema = "fs-simulation-1" if kind == "simulation" else "fs-job-1"
    if payload.get("schema") != expected_schema:
        raise ArtifactSafetyError("artifact.contract.unsupported")
    if kind == "job" and payload.get("workflow") != "forward":
        raise ArtifactSafetyError("artifact.contract.unsupported")
    _validate_allocation_budget(payload)
    _validate_nested_types(payload)
    _reject_external_references(payload, kind)


def _validate_allocation_budget(payload: Mapping[str, Any]) -> None:
    """Reject compact shapes that could expand beyond the worker budget."""

    if payload.get("_type") == "TimeDomainJob":
        _validate_time_domain_sweep(payload.get("f_list"))

    stack: list[Any] = [payload]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = key.strip().casefold().replace("-", "_")
                if normalized_key == "grid" and child not in (None, {}, []):
                    raise ArtifactSafetyError("artifact.allocation.unsupported")
                if normalized_key in {"n", "shape"}:
                    _validate_bounded_shape(child)
                elif normalized_key in {
                    "count",
                    "field_count",
                    "n_points",
                    "n_samples",
                    "num_points",
                    "size",
                }:
                    _validate_bounded_count(child)
                stack.append(child)
        elif isinstance(value, list):
            stack.extend(value)


def _validate_time_domain_sweep(value: Any) -> None:
    """Bound the compact uniform sweep reconstructed by ``TimeDomainJob``."""

    if not isinstance(value, list) or len(value) < 2:
        raise ArtifactSafetyError("artifact.allocation.invalid")

    real_parts: list[float] = []
    for sample in value:
        if isinstance(sample, list):
            if len(sample) != 2:
                raise ArtifactSafetyError("artifact.allocation.invalid")
            real = _finite_frequency_component(sample[0])
            _finite_frequency_component(sample[1])
        else:
            real = _finite_frequency_component(sample)
        real_parts.append(real)

    f_min = real_parts[0]
    f_max = real_parts[-1]
    df = real_parts[1] - f_min
    span = f_max - f_min
    if not math.isfinite(df) or not math.isfinite(span) or df <= 0.0 or span <= 0.0:
        raise ArtifactSafetyError("artifact.allocation.invalid")

    stop = f_max + df / 2.0
    steps = span / df
    if not math.isfinite(stop) or not math.isfinite(steps):
        raise ArtifactSafetyError("artifact.allocation.limit")
    implied_count = math.ceil(steps + 0.5)
    if implied_count > _MAX_SCALAR_COUNT:
        raise ArtifactSafetyError("artifact.allocation.limit")


def _finite_frequency_component(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactSafetyError("artifact.allocation.invalid")
    try:
        component = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ArtifactSafetyError("artifact.allocation.invalid") from exc
    if not math.isfinite(component):
        raise ArtifactSafetyError("artifact.allocation.invalid")
    return component


def _validate_bounded_shape(value: Any) -> None:
    if isinstance(value, bool):
        raise ArtifactSafetyError("artifact.allocation.invalid")
    if isinstance(value, int):
        _validate_bounded_count(value)
        return
    if not isinstance(value, list) or not 1 <= len(value) <= 4:
        raise ArtifactSafetyError("artifact.allocation.invalid")
    product = 1
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 1 <= item <= _MAX_SCALAR_COUNT
        ):
            raise ArtifactSafetyError("artifact.allocation.limit")
        product *= item
        if product > _MAX_SHAPE_PRODUCT:
            raise ArtifactSafetyError("artifact.allocation.limit")


def _validate_bounded_count(value: Any) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_SCALAR_COUNT
    ):
        raise ArtifactSafetyError("artifact.allocation.limit")


def _validate_nested_types(payload: Mapping[str, Any]) -> None:
    stack: list[tuple[Any, bool]] = [(payload, True)]
    while stack:
        value, is_root = stack.pop()
        if isinstance(value, dict):
            if "_type" in value:
                type_name = value["_type"]
                if not isinstance(type_name, str):
                    raise ArtifactSafetyError("artifact.type.unsupported")
                allowed = _SUPPORTED_ROOT_TYPES if is_root else _SUPPORTED_NESTED_TYPES
                if type_name not in allowed:
                    raise ArtifactSafetyError("artifact.type.unsupported")
            stack.extend((child, False) for child in value.values())
        elif isinstance(value, list):
            stack.extend((child, False) for child in value)


def _reject_external_references(payload: Mapping[str, Any], kind: str) -> None:
    stack: list[tuple[Any, tuple[str | int, ...]]] = [(payload, ())]
    while stack:
        value, path = stack.pop()
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = (*path, key)
                if child not in (None, "", [], {}) and _is_reference_key(key):
                    if not _allowed_reference(child_path, child, kind):
                        raise ArtifactSafetyError("artifact.reference.unsupported")
                    continue
                stack.append((child, child_path))
        elif isinstance(value, list):
            stack.extend((child, (*path, index)) for index, child in enumerate(value))
        elif isinstance(value, str):
            _reject_suspicious_string(value)


def _is_reference_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in _REFERENCE_KEYS or normalized.endswith(_REFERENCE_SUFFIXES)


def _allowed_reference(path: tuple[str | int, ...], value: Any, kind: str) -> bool:
    if path == ("project_path",):
        return isinstance(value, str)
    if kind == "job" and path == ("simulation",):
        return isinstance(value, str)
    if kind == "job" and path == ("result_path",):
        return _is_safe_output_path(value)
    if kind == "job" and len(path) >= 3 and path[0] == "Outputs" and path[-1] == "path":
        return _is_safe_output_path(value)
    if kind == "simulation" and path == ("Mesh", "generator", "path"):
        return _is_safe_output_path(value)
    return False


def _is_safe_output_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        _validated_relative_path(value, require_json=False)
    except ArtifactSafetyError:
        return False
    return True


def _reject_suspicious_string(value: str) -> None:
    if (
        _has_control(value)
        or "\\" in value
        or _ENCODED_PATH_RE.search(value)
        or _URI_RE.search(value)
        or "arn:" in value.lower()
        or value.startswith(("~", "/"))
        or _WINDOWS_DRIVE_RE.match(value)
        or _FILE_LOCATOR_RE.search(value)
    ):
        raise ArtifactSafetyError("artifact.reference.unsupported")


def _job_simulation_path(payload: Mapping[str, Any], root: str) -> str:
    reference = payload.get("simulation")
    if not isinstance(reference, str) or not reference:
        raise ArtifactSafetyError("artifact.simulation.invalid")
    if os.path.isabs(reference):
        if (
            _has_control(reference)
            or "\\" in reference
            or "%" in reference
            or ":" in reference
        ):
            raise ArtifactSafetyError("artifact.simulation.invalid")
        normalized_reference = os.path.normpath(reference)
        try:
            common = os.path.commonpath((root, normalized_reference))
        except ValueError as exc:
            raise ArtifactSafetyError("artifact.simulation.escape") from exc
        if common != root:
            raise ArtifactSafetyError("artifact.simulation.escape")
        reference = os.path.relpath(normalized_reference, root).replace(
            os.path.sep, "/"
        )
    try:
        return _validated_relative_path(reference, require_json=True)
    except ArtifactSafetyError as exc:
        code = (
            "artifact.simulation.escape"
            if ".." in reference.split("/")
            else "artifact.simulation.invalid"
        )
        raise ArtifactSafetyError(code) from exc


def _load_in_sandbox(
    work_dir: Path,
    *,
    simulation_payload: Mapping[str, Any],
    job_payload: Optional[Mapping[str, Any]],
    root_type: str,
) -> tuple[Any, Any]:
    project = work_dir / "project"
    simulation_dir = project / "simulations" / "artifact"
    job_dir = project / "jobs" / "artifact"
    simulation_dir.mkdir(parents=True, mode=0o700)
    job_dir.mkdir(parents=True, mode=0o700)

    safe_simulation = copy.deepcopy(dict(simulation_payload))
    safe_simulation["project_path"] = str(project)
    simulation_file = simulation_dir / "simulation.json"
    _write_sanitized_json(simulation_file, safe_simulation)

    try:
        simulation = SeismicSimulation.from_fs(safe_simulation)
        simulation._file = simulation_file
        simulation.relocate(project)
    except Exception as exc:
        raise ArtifactSafetyError("artifact.simulation.invalid") from exc

    if job_payload is None:
        try:
            report = validate_simulation(
                simulation, allow_unverified_remote_files=False
            )
        except Exception as exc:
            raise ArtifactSafetyError("artifact.package.invalid") from exc
        return simulation, report

    safe_job = copy.deepcopy(dict(job_payload))
    safe_job["project_path"] = str(project)
    safe_job["simulation"] = "simulations/artifact/simulation.json"
    safe_job["result_path"] = "jobs/artifact/results"
    job_file = job_dir / "job.json"
    _write_sanitized_json(job_file, safe_job)

    job_class = (
        FrequencyDomainJob if root_type == "FrequencyDomainJob" else TimeDomainJob
    )
    try:
        job = job_class.from_fs(
            safe_job,
            base_path=job_file.parent,
            project_path=project,
        )
        job._file = job_file
    except Exception as exc:
        raise ArtifactSafetyError("artifact.job.invalid") from exc
    try:
        report = validate_job(job, allow_unverified_remote_files=False)
    except Exception as exc:
        raise ArtifactSafetyError("artifact.package.invalid") from exc
    return job, report


def _write_sanitized_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )
    except Exception as exc:
        raise ArtifactSafetyError("artifact.sandbox.invalid") from exc


def _artifact_summary(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, SeismicSimulation):
        return {
            "type": "SeismicSimulation",
            "physics": str(artifact.physics),
            "dimension": artifact.dimension,
            "workflow": None,
            "frequencies": [],
            "task_count": None,
            "output_kinds": [],
        }

    output_kinds: list[str] = []
    outputs = artifact.outputs
    if outputs.traces is not None:
        output_kinds.append("traces")
    if outputs.vtk:
        output_kinds.append("vtk")
    if outputs.wavefields:
        output_kinds.append("wavefields")
    return {
        "type": artifact.__class__.__name__,
        "physics": str(artifact.simulation.physics),
        "dimension": artifact.simulation.dimension,
        "workflow": str(artifact.workflow),
        "frequencies": [_safe_frequency(value) for value in artifact.f_list],
        "task_count": int(artifact.n_tasks),
        "output_kinds": output_kinds,
    }


def _safe_frequency(value: Any) -> float | dict[str, float]:
    number = complex(value)
    if not math.isfinite(number.real) or not math.isfinite(number.imag):
        raise ArtifactSafetyError("artifact.job.invalid")
    if number.imag == 0.0:
        return float(number.real)
    return {"real": float(number.real), "imag": float(number.imag)}


def _report_issues(report: Any) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for issue in report.issues[:_MAX_ISSUES]:
        item = {
            "severity": str(issue.severity),
            "code": str(issue.code),
        }
        issue_path = str(issue.path)
        if _SAFE_ISSUE_PATH_RE.fullmatch(issue_path):
            item["path"] = issue_path
        issues.append(item)
    if len(report.issues) > _MAX_ISSUES:
        issues.append(
            {
                "severity": "warning",
                "code": "artifact.validation.truncated",
            }
        )
    return issues


def _error_result(mode: str, code: str) -> dict[str, Any]:
    return {
        "schema": _RESULT_SCHEMA,
        "ok": False,
        "mode": mode,
        "artifact": None,
        "issues": [{"severity": "error", "code": code}],
    }


def _has_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)
