from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

RUN_SCHEMA = "frequensolve-cloud-benchmark-run/v1"
CASE_SCHEMA = "frequensolve-cloud-benchmark-case/v1"
COMPARISON_SCHEMA = "frequensolve-cloud-benchmark-comparison/v1"
SECRET_KEY = re.compile(
    r"^(?:authorization|credentials?|password|secret|sessionToken|accessToken|idToken|refreshToken)$",
    re.IGNORECASE,
)
BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
AWS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def sanitize_text(value: str) -> str:
    value = BEARER.sub("Bearer <redacted>", value)
    return AWS_KEY.sub("<redacted-aws-access-key>", value)


def sanitize(value: Any, key: str | None = None) -> Any:
    if key is not None and SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(repr(value))


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_metric(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(samples),
        "min": min(samples) if samples else None,
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "max": max(samples) if samples else None,
    }


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"Unsafe benchmark identifier: {value!r}")
    return normalized
