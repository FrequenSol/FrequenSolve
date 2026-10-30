"""Bounded transient tolerance pilots using the normal job/site/output APIs."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from frequensolve.simulation.time import (
    CheckpointSchedule,
    TimeIntegration,
    TimeOutput,
    _count,
    _number,
)

__all__ = ["TransientToleranceStudy", "ToleranceTuningResult"]


@dataclass(frozen=True)
class ToleranceTuningResult:
    """Measured pilot differences; recommendation does not modify the parent job."""

    reference_tolerance: float
    max_relative_error: float
    recommended_tolerance: float | None
    trials: tuple[dict[str, Any], ...]


def _relative_error(reference, candidate):
    """Compare matching dense samples per source, retaining the worst source."""
    if reference.dims != candidate.dims or not reference.coords.equals(
        candidate.coords
    ):
        raise ValueError(
            "Tuning samples must have identical coordinates and dimensions"
        )
    if "source" not in reference.dims:
        raise ValueError(
            "Tolerance tuning requires dense outputs with a source dimension"
        )
    a = np.asarray(reference.transpose("source", ...), dtype=np.float64)
    b = np.asarray(candidate.transpose("source", ...), dtype=np.float64)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Tuning samples must be finite")
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    norm = np.linalg.norm(a, axis=1)
    if not len(norm) or np.any(norm == 0):
        raise ValueError(
            "Pilot reference contains a silent source; choose an informative window/output"
        )
    return float(np.max(np.linalg.norm(b - a, axis=1) / norm))


class TransientToleranceStudy:
    """Short same-dt runs at several tolerances against a tighter reference.

    Preserve the mesh, source batch, scheme and timestep. Only the pilot window,
    output cadence, checkpoint schedule and solver tolerance change. Select
    a window containing source excitation and propagation to the chosen output.
    The result measures algebraic sensitivity, not total physical accuracy.
    """

    def __init__(
        self, job, *, steps, tolerances=(1e-4, 1e-3), reference_tolerance=1e-5
    ):
        from frequensolve.simulation.jobs.transient import TransientJob

        if not isinstance(job, TransientJob):
            raise TypeError("Tolerance studies require a TransientJob")
        steps = _count(steps, "steps", 1)
        if steps > job.time.steps:
            raise ValueError("Pilot steps exceed the parent job interval")
        if job.restart is not None or job.n_modes is not None:
            raise ValueError(
                "Initial tolerance pilots require a fresh single-task transient job"
            )
        reference_tolerance = _number(reference_tolerance, "reference_tolerance")
        candidates = tuple(sorted({_number(t, "tolerance") for t in tolerances}))
        if not candidates or len(candidates) > 8:
            raise ValueError("Specify between one and eight candidate tolerances")
        if not 0 < reference_tolerance < min(candidates) or max(candidates) >= 1:
            raise ValueError("Require 0 < reference_tolerance < candidates < 1")
        self.reference_tolerance = reference_tolerance
        self.jobs = {}
        for index, tolerance in enumerate((reference_tolerance, *candidates)):
            pilot = copy.copy(job)
            pilot.name = f"{job.name}-tol-{index}"
            pilot._file = pilot._job_id = None
            pilot._external_input_fingerprint_cache = None
            pilot.time = TimeIntegration(
                job.time.start,
                job.time.start + steps * job.time.dt,
                job.time.dt,
                job.time.method,
            )
            pilot.simulation = copy.deepcopy(job.simulation)
            pilot.simulation.name = f"{job.simulation.name}-tol-{index}"
            pilot.simulation._file = None
            pilot.simulation.solver.tolerance = tolerance
            pilot.outputs = copy.deepcopy(job.outputs)
            pilot.output_schedule = TimeOutput(stride=max(1, steps // 20))
            pilot.trace_schedule = pilot.output_schedule
            pilot.checkpoint = CheckpointSchedule()
            pilot.validate_outputs()
            self.jobs[tolerance] = pilot

    def run(self, site, *, groups, max_relative_error, output="traces", **run_options):
        """Run sequentially and compare committed outputs; no production setting is changed.

        ``groups`` names one or more existing trace/wavefield groups. Select
        ``output="wavefields"`` for sampled wavefields. Each field and source
        must satisfy ``max_relative_error`` independently. Failed candidates
        are reported and excluded. A failed reference aborts the study.
        """
        from frequensolve.orchestrator.sites.base import RunFailedError

        if output not in {"traces", "wavefields"}:
            raise ValueError("output must be traces or wavefields")
        groups = (groups,) if isinstance(groups, str) else tuple(groups)
        if not groups or any(not isinstance(g, str) or not g for g in groups):
            raise ValueError("Specify output group names")
        budget = _number(max_relative_error, "max_relative_error")
        if not 0 < budget < 1:
            raise ValueError("max_relative_error must lie in (0,1)")
        if "check" in run_options:
            raise ValueError("Tolerance studies always check run success")
        reference, trials = {}, []
        for tolerance, pilot in self.jobs.items():
            row = {"tolerance": tolerance, "job": pilot.name, "passed": False}
            try:
                pilot.save()
                site.run(pilot, check=True, **run_options)
                dataset = getattr(pilot, output).open()
                samples = {group: dataset.read(group) for group in groups}
                result = pilot.operation_result()
                solves = result.solver.get("convergence", {}).get("solves", [])
                row["elapsed_seconds"] = result.timings.get("elapsed")
                row["mean_iterations"] = (
                    float(np.mean([s["iterations"] for s in solves]))
                    if solves
                    else None
                )
                if not reference:
                    # Validate even the reference's coordinates and nonzero signal.
                    for sample in samples.values():
                        _relative_error(sample, sample)
                    reference = samples
                    row.update(relative_error=0.0, passed=True, reference=True)
                else:
                    errors = {
                        g: _relative_error(reference[g], samples[g]) for g in groups
                    }
                    row.update(
                        errors=errors,
                        relative_error=max(errors.values()),
                        passed=max(errors.values()) <= budget,
                    )
            except (RunFailedError, ValueError, FileNotFoundError) as exc:
                if tolerance == self.reference_tolerance:
                    raise
                row["failure"] = str(exc)
            trials.append(row)
        accepted = [
            r["tolerance"] for r in trials if r["passed"] and not r.get("reference")
        ]
        return ToleranceTuningResult(
            self.reference_tolerance,
            budget,
            max(accepted) if accepted else None,
            tuple(trials),
        )
