"""Public preflight validation entry points.

Use these helpers before saving, submitting, or debugging simulations and jobs
when you want SDK-level diagnostics without invoking the solver or a JSON
Schema validation step.
"""

from __future__ import annotations

from typing import Any

from frequensolve.simulation.outputs import JobOutputs
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
) -> ValidationReport:
    """Validate a simulation object without requiring solver JSON Schema.

    Args:
        simulation: Simulation object to validate.
        raise_errors: If ``True``, raise ``ValidationError`` when blocking
            issues are found.

    Returns:
        Validation report containing errors and warnings.
    """

    report = ValidationReport()
    ctx = _build_context(simulation, report)
    _validate_simulation(ctx)
    if raise_errors:
        report.raise_for_errors()
    return report


def validate_job(
    job: Any,
    *,
    raise_errors: bool = False,
) -> ValidationReport:
    """Validate a job before saving or submitting it.

    Args:
        job: Job object with a simulation, frequency list, and outputs.
        raise_errors: If ``True``, raise ``ValidationError`` when blocking
            issues are found.

    Returns:
        Validation report containing errors and warnings.
    """

    report = ValidationReport()
    simulation = getattr(job, "simulation", None)
    if simulation is None:
        report.error("job.simulation.missing", "Job requires a simulation.")
        if raise_errors:
            report.raise_for_errors()
        return report

    ctx = _build_context(simulation, report)
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

    k_list = getattr(job, "k_list", None)
    k_count = _optional_sequence_length(k_list)
    if dimension == 2.5 and k_count == 0:
        ctx.report.error(
            "job.k_list.required",
            "2.5D jobs require k_list wavenumbers in the job JSON.",
            path="k_list",
        )

    k_weights = getattr(job, "k_weights", None)
    k_weight_count = _optional_sequence_length(k_weights)
    if k_weight_count and k_count == 0:
        ctx.report.error(
            "job.k_weights.requires_k_list",
            "k_weights requires matching k_list values.",
            path="k_weights",
        )
    elif k_weight_count and k_weight_count != k_count:
        ctx.report.error(
            "job.k_weights.length_mismatch",
            "k_weights must have the same number of values as k_list.",
            path="k_weights",
        )


def _optional_sequence_length(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0
