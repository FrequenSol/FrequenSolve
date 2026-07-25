"""Public preflight validation entry points.

Use these helpers before saving, submitting, or debugging simulations and jobs
when you want SDK-level diagnostics without invoking the solver or a JSON
Schema validation step.
"""

from __future__ import annotations

from typing import Any

from frequensolve.simulation.outputs import JobOutputs

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
