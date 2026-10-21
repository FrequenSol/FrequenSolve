"""Tested complex-pressure operations; no alignment or physical-validity inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .storage import canonical

OPERATION = "complex-pressure.v1"
AXES = (
    "source_ids",
    "source_coordinates",
    "receiver_ids",
    "receiver_coordinates",
    "coordinate_units",
    "component",
    "units",
    "frequency",
    "laplace",
    "source_scale",
    "source_encoding",
    "selection",
)


def _values(snapshot: dict) -> np.ndarray:
    if snapshot.get("version") != "pressure-snapshot.v1" or not isinstance(
        snapshot.get("metadata"), dict
    ):
        raise ValueError("Unsupported pressure snapshot")
    metadata = snapshot["metadata"]
    if any(key not in metadata for key in AXES) or any(
        metadata[k] is None for k in AXES
    ):
        raise ValueError(
            "Pressure identity, axes, units and source scaling must be explicit"
        )
    if snapshot.get("evidence", {}).get("converged") is not True:
        raise ValueError("Analysis requires converged pressure evidence")
    values = np.asarray(snapshot["real"], dtype=float) + 1j * np.asarray(
        snapshot["imaginary"], dtype=float
    )
    expected = (len(metadata["source_ids"]), len(metadata["receiver_ids"]))
    if not all(expected) or values.size > 4096:
        raise ValueError("Pressure selection is empty or exceeds the sample limit")
    for key in ("source_ids", "receiver_ids"):
        if len(set(metadata[key])) != len(metadata[key]):
            raise ValueError("Pressure identifiers must be unique")
    for key, count in (
        ("source_coordinates", expected[0]),
        ("receiver_coordinates", expected[1]),
    ):
        coordinates = np.asarray(metadata[key], dtype=float)
        if (
            coordinates.shape not in ((count, 2), (count, 3))
            or not np.isfinite(coordinates).all()
        ):
            raise ValueError("Pressure coordinates must be complete and finite")
    if not all(np.isfinite(float(metadata[key])) for key in ("frequency", "laplace")):
        raise ValueError("Pressure frequency identity must be finite")
    if (
        values.shape != expected
        or np.asarray(snapshot["real"]).shape != expected
        or np.asarray(snapshot["imaginary"]).shape != expected
    ):
        raise ValueError("Pressure axes do not match the values")
    if (
        len(metadata["source_coordinates"]) != expected[0]
        or len(metadata["receiver_coordinates"]) != expected[1]
    ):
        raise ValueError("Pressure coordinate coverage is incomplete")
    return values


def _json_array(values: np.ndarray) -> list:
    result = values.astype(object)
    result[~np.isfinite(values)] = None
    return result.tolist()


def analyze_pressure(snapshot: dict, reference: dict | None = None) -> dict:
    values = _values(snapshot)
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    magnitude = np.abs(values)
    phase = np.where(finite & (magnitude > 0), np.angle(values), np.nan)
    result = {
        "operation": OPERATION,
        "metadata": snapshot["metadata"],
        "sample_count": int(values.size),
        "finite_count": int(finite.sum()),
        "real": _json_array(np.where(finite, values.real, np.nan)),
        "imaginary": _json_array(np.where(finite, values.imag, np.nan)),
        "magnitude": _json_array(np.where(finite, magnitude, np.nan)),
        "principal_phase_radians": _json_array(phase),
        "assurance": {
            "operation": "tested",
            "input": snapshot.get("evidence", {}),
            "scientific_claim": "Convergence alone does not establish physical or discretization accuracy",
        },
    }
    if reference is not None:
        other = _values(reference)
        comparison_keys = (
            *AXES,
            *(
                key
                for key in ("value_frame", "solver_version")
                if key in snapshot["metadata"] or key in reference["metadata"]
            ),
        )
        if any(
            canonical(snapshot["metadata"].get(key))
            != canonical(reference["metadata"].get(key))
            for key in comparison_keys
        ):
            raise ValueError(
                "Reference axes, units, frequency or source scaling do not match exactly"
            )
        paired = finite & np.isfinite(other.real) & np.isfinite(other.imag)
        difference = values - other
        count = int(paired.sum())
        result["comparison"] = {
            "paired_count": count,
            "coverage": count / values.size if values.size else 0,
            "complex_rms_difference": (
                float(np.sqrt(np.mean(np.abs(difference[paired]) ** 2)))
                if count
                else None
            ),
            "difference_real": _json_array(np.where(paired, difference.real, np.nan)),
            "difference_imaginary": _json_array(
                np.where(paired, difference.imag, np.nan)
            ),
            "absolute_complex_difference": _json_array(
                np.where(paired, np.abs(difference), np.nan)
            ),
        }
        denominator = float(np.linalg.norm(other[paired]))
        result["comparison"]["relative_l2_difference"] = (
            float(np.linalg.norm(difference[paired]) / denominator)
            if count and denominator > 0
            else None
        )
    return result


def read_snapshot(path: str | Path, expected_sha256: str) -> dict:
    """Verify the exact local analysis input on every notebook replay."""
    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 16 * 1024 * 1024
    ):
        raise ValueError("Invalid bounded pressure snapshot")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("Pressure snapshot bytes changed")
    value = json.loads(data)
    _values(value)
    return value
