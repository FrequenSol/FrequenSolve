"""Public preflight validation entry points.

Use these helpers before saving, submitting, or debugging simulations and jobs
when you want SDK-level diagnostics without invoking the solver or a JSON
Schema validation step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from frequensolve.simulation.outputs import JobOutputs
from frequensolve.units import ureg
from frequensolve.util.physics import canonical_dimension

from .geometry import _build_context, _ValidationContext
from .outputs import _validate_frequencies, _validate_outputs
from .report import ValidationReport
from .simulation import _validate_simulation

__all__ = ["validate_job", "validate_simulation"]


def validate_simulation(
    simulation: Any,
    *,
    raise_errors: bool = False,
    allow_unverified_remote_files: bool = False,
) -> ValidationReport:
    """Validate a simulation object without requiring solver JSON Schema.

    Args:
        simulation: Simulation object to validate.
        raise_errors: If ``True``, raise ``ValidationError`` when blocking
            issues are found.
        allow_unverified_remote_files: Treat absolute file references outside
            the local project as target-site files that cannot be checked
            locally. Such references produce warnings instead of errors.

    Returns:
        Validation report containing errors and warnings.
    """

    report = ValidationReport.for_package_validators()
    ctx = _build_context(
        simulation,
        report,
        allow_unverified_remote_files=allow_unverified_remote_files,
    )
    _validate_simulation(ctx)
    if raise_errors:
        report.raise_for_errors()
    return report


def validate_job(
    job: Any,
    *,
    raise_errors: bool = False,
    allow_unverified_remote_files: bool = False,
) -> ValidationReport:
    """Validate a job before saving or submitting it.

    Args:
        job: Job object with a simulation, frequency list, and outputs.
        raise_errors: If ``True``, raise ``ValidationError`` when blocking
            issues are found.
        allow_unverified_remote_files: Treat absolute file references outside
            the local project as target-site files that cannot be checked
            locally. Such references produce warnings instead of errors.

    Returns:
        Validation report containing errors and warnings.
    """

    report = ValidationReport.for_package_validators()
    simulation = getattr(job, "simulation", None)
    if simulation is None:
        report.error("job.simulation.missing", "Job requires a simulation.")
        if raise_errors:
            report.raise_for_errors()
        return report

    ctx = _build_context(
        simulation,
        report,
        allow_unverified_remote_files=allow_unverified_remote_files,
    )
    _validate_simulation(ctx)
    _validate_job(job, ctx)
    if raise_errors:
        report.raise_for_errors()
    return report


def _validate_job(job: Any, ctx: _ValidationContext) -> None:
    _validate_frequencies(getattr(job, "f_list", None), ctx.report)
    _validate_half_dimension_wavenumbers(job, ctx)
    outputs = getattr(job, "outputs", None)
    if not isinstance(outputs, JobOutputs):
        try:
            outputs = JobOutputs(outputs)
        except Exception as exc:
            ctx.report.error(
                "job.outputs.invalid",
                f"Job outputs could not be interpreted: {exc}",
                path="outputs",
            )
            return
    _validate_outputs(outputs, job, ctx)


def _validate_half_dimension_wavenumbers(job: Any, ctx: _ValidationContext) -> None:
    try:
        dimension = canonical_dimension(getattr(ctx.simulation, "dimension", None))
    except ValueError:
        return

    k_count, k_present, k_valid = _validate_wavenumber_sequence(
        getattr(job, "k_list", None),
        field_name="k_list",
        code="job.k_list.invalid",
        ctx=ctx,
    )
    if dimension == 2.5 and not k_present:
        ctx.report.error(
            "job.k_list.required",
            "2.5D jobs require signed physical k_list wavenumbers.",
            path="k_list",
        )

    k_weight_count, k_weights_present, k_weights_valid = _validate_wavenumber_sequence(
        getattr(job, "k_weights", None),
        field_name="k_weights",
        code="job.k_weights.invalid",
        ctx=ctx,
    )
    if k_weights_present and k_weights_valid and not k_present:
        ctx.report.error(
            "job.k_weights.requires_k_list",
            "k_weights requires matching k_list values.",
            path="k_weights",
        )
    elif (
        k_weights_present
        and k_weights_valid
        and k_present
        and k_valid
        and k_weight_count != k_count
    ):
        ctx.report.error(
            "job.k_weights.length_mismatch",
            "k_weights must have the same number of values as k_list.",
            path="k_weights",
        )

    k_units = getattr(job, "k_units", None)
    if k_units is not None:
        try:
            units = str(k_units).strip()
            if not units:
                raise ValueError("empty unit expression")
            ureg.Quantity(1.0, units).to("1/m")
        except Exception:
            ctx.report.error(
                "job.k_units.incompatible",
                "k_units must be valid inverse-length units.",
                path="k_units",
            )


def _validate_wavenumber_sequence(
    value: Any,
    *,
    field_name: str,
    code: str,
    ctx: _ValidationContext,
) -> tuple[int, bool, bool]:
    if value is None:
        return 0, False, True
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        array = None
    if (
        array is None
        or array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
    ):
        ctx.report.error(
            code,
            f"{field_name} must be a nonempty 1D list of finite numbers.",
            path=field_name,
        )
        return 0, True, False
    return int(array.size), True, True
