"""Simulation references and recursive job overrides shared with Sauce."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Mapping


def expand_simulation_override_paths(overrides: Mapping) -> dict:
    """Expand object-member paths into Sauce's nested override representation."""
    if not isinstance(overrides, Mapping):
        raise TypeError("simulation_overrides must be a mapping")
    result = {}
    for path, value in overrides.items():
        if not isinstance(path, str):
            raise TypeError("simulation_overrides keys must be strings")
        if not path:
            raise ValueError("Simulation override paths must not be empty")
        tokens = path.removeprefix("/").split("/")
        if any(not token for token in tokens):
            raise ValueError(
                f"Simulation override path contains an empty member: {path!r}"
            )
        if "/" in path:
            decoded = []
            for token in tokens:
                if "~" in token.replace("~0", "").replace("~1", ""):
                    raise ValueError(
                        f"Invalid JSON Pointer escape in override path: {path!r}"
                    )
                decoded.append(token.replace("~1", "/").replace("~0", "~"))
            tokens = decoded
        if "/" in tokens[0]:
            raise ValueError(
                "The top-level simulation section name must not contain a slash"
            )
        if any(token.isdecimal() or token == "-" for token in tokens[1:]):
            raise ValueError(
                "Array index override paths are unsupported; replace the whole array"
            )
        patch = copy.deepcopy(value)
        for token in reversed(tokens):
            patch = {token: patch}
        result = merge_simulation_overrides(result, patch)
    return result


def merge_simulation_overrides(base: Mapping, overrides: Mapping) -> dict:
    """Merge objects recursively; replace arrays, scalars and null as Sauce does."""
    if not isinstance(overrides, Mapping):
        raise TypeError("simulation_overrides must be a mapping")
    result = dict(base)
    for key, value in overrides.items():
        if not isinstance(key, str):
            raise TypeError("simulation_overrides keys must be strings")
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = merge_simulation_overrides(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def simulation_reference_path(reference: Any, *, job_file: Path | None = None) -> Path:
    """Extract a filename, resolving File relative paths against the job directory."""
    if not isinstance(reference, Mapping):
        return Path(reference)
    if reference.get("type") != "File":
        raise ValueError("Job simulation reference must use type='File'")
    if reference.get("residency", "host") != "host":
        raise ValueError("Job simulation reference must be host-resident")
    absolute, relative = reference.get("absolute_path"), reference.get("relative_path")
    if (absolute is None) == (relative is None):
        raise ValueError("Simulation File reference requires exactly one path")
    path = Path(absolute if absolute is not None else relative)
    if relative is not None and job_file is not None:
        path = (job_file.parent / path).resolve()
    return path


def reference_overrides(reference: Any) -> dict:
    """Return a defensive copy of a File reference's override object."""
    if not isinstance(reference, Mapping):
        return {}
    return merge_simulation_overrides({}, reference.get("overrides", {}))


def effective_simulation_payload(job: Mapping, simulation: Mapping) -> dict:
    """Resolve the simulation document consumed by one saved job."""
    return merge_simulation_overrides(
        simulation, reference_overrides(job.get("simulation"))
    )


def expanded_simulation_hash(
    job: Mapping, simulation: Mapping, source_digest: str
) -> str:
    """Match Sauce's exact expanded-file hash, preserving unexpanded source digests."""
    reference = job.get("simulation")
    if not isinstance(reference, Mapping) or "overrides" not in reference:
        return source_digest
    from frequensolve.imaging._canonical_json import canonical_json_text

    text = (
        canonical_json_text(effective_simulation_payload(job, simulation), indent=2)
        + "\n"
    )
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
