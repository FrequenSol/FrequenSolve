"""Shared-frequency reflection waveform inversion and reduced curvature."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from ._artifacts import ControlStateFile
from ._backend import fingerprint, reduce_covectors
from ._rwi import _ReflectivitySystem
from .controls import ControlState, ControlVector
from .jobs import FWIOperatorJob
from .misfit import Loss
from .operators import Normal
from .problem import _vector_from_file


def _solver_options(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the native regularized least-squares solve configuration."""
    if not isinstance(value, Mapping):
        raise TypeError("reflectivity solver must be a mapping")
    defaults = dict(
        damping=1e-4,
        relative_tolerance=1e-4,
        absolute_tolerance=0.0,
        max_iterations=50,
        cache_mb=512.0,
        require_convergence=True,
        solution="reflectivity.h5",
        report="reflectivity.json",
    )
    unknown = set(value) - set(defaults) - {"frequency_weights"}
    if unknown:
        raise ValueError(f"unknown reflectivity solver options: {sorted(unknown)}")
    options = {**defaults, **value}
    for name in ("damping", "relative_tolerance", "absolute_tolerance", "cache_mb"):
        raw = options[name]
        number = float(raw)
        if (
            isinstance(raw, bool)
            or not np.isfinite(number)
            or number < 0
            or (name in {"damping", "relative_tolerance"} and number == 0)
        ):
            raise ValueError(f"invalid reflectivity solver {name}")
        options[name] = number
    count = options["max_iterations"]
    if isinstance(count, bool) or int(count) != count or count < 0:
        raise ValueError("reflectivity max_iterations must be a nonnegative integer")
    options["max_iterations"] = int(count)
    if not isinstance(options["require_convergence"], bool):
        raise ValueError("reflectivity require_convergence must be boolean")
    if "frequency_weights" in options:
        weights = np.asarray(options["frequency_weights"], dtype=float)
        if (
            weights.ndim != 1
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError(
                "reflectivity frequency_weights must be finite, nonnegative and nonzero"
            )
        options["frequency_weights"] = weights.tolist()
    for name in ("solution", "report"):
        raw = options[name]
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            raise ValueError(f"reflectivity {name} must be a nonempty path")
        options[name] = str(raw)
    return options


@dataclass
class RWILinearization:
    """A stationary reflectivity fit and its reduced background derivative."""

    _reduced_curvature = True

    value: float
    gradient: ControlVector
    state: ControlState
    point: ControlVector
    job: FWIOperatorJob
    reports: list[dict[str, Any]]
    fingerprint: str
    problem: Any = None
    _joint: Any = field(default=None, repr=False)
    _system: Any = field(default=None, repr=False)
    _products: dict = field(default_factory=dict, repr=False)
    response_reports: list = field(default_factory=list)
    _settings: dict = field(default_factory=dict, repr=False)

    @property
    def space(self):
        return self.point.space

    @property
    def report(self):
        return {
            "value": self.value,
            "reflectivity": self.reports,
            "reduced_normal": self.response_reports,
        }

    @property
    def normal(self):
        return Normal(self)

    @property
    def regularization_job(self):
        return self.job

    @property
    def support(self):
        return self.space.support

    def apply_normal(self, direction, *, regularization=None):
        """Apply the shared reduced GN operator, optionally adding outer curvature."""
        return self.apply_normal_batch([direction], regularization=regularization)[0]

    def apply_normal_batch(self, directions, *, cache=True, regularization=None):
        """Apply reduced GN to each direction with independent shared response solves.

        The reflectivity penalty participates in elimination. An explicit outer
        regularizer is added once afterwards, on the background space.
        """
        if self._joint is None:
            self._joint = self.problem._joint_linearize(self.state)
            self._system = self.problem._system(self._joint, solver=self._settings)
        elif self._joint.fingerprint not in self._joint.problem.cache:
            # The bounded native-artifact cache can evict a fitted point while
            # its reduced operator is still held by a caller. Replay its exact
            # fitted state, never the current background or a new inner fit.
            previous_space = self._joint.space
            self._joint = self._joint.problem._linearize_state(
                self.state, gradient=True
            )
            if not self._joint.space.equivalent(previous_space):
                raise RuntimeError(
                    "RWI control support changed while replaying a fitted state"
                )
            self._system = self.problem._system(self._joint, solver=self._settings)
            self.job = self._joint.job
        if regularization is not None and not regularization.space.equivalent(
            self.space
        ):
            raise ValueError("regularization belongs to a different control space")
        operator = (
            None
            if regularization is None
            else regularization.hessian_operator(self.point)
        )
        results = []
        for value in directions:
            direction = (
                value
                if isinstance(value, ControlVector)
                else ControlVector(value, self.space)
            )
            if not direction.space.equivalent(self.space):
                raise ValueError("direction belongs to a different control space")
            key = fingerprint(direction=direction.values)
            if not cache or key not in self._products:
                product, report = self._system.reduced_normal(direction.values)
                report = {**report, "direction": key, "state": self.fingerprint}
                self.response_reports.append(report)
                if cache:
                    self._products[key] = product.copy()
            else:
                product = self._products[key]
            result = ControlVector(product, self.space)
            if operator is not None:
                result = result + ControlVector(operator @ direction.values, self.space)
            results.append(result)
        return results


class RWIProblem:
    """Minimize data misfit after fitting one shared acoustic or elastic reflectivity.

    Every background trial fits one reflectivity over all frequency tasks.
    The default ``execution="sdk"`` coordinates ordinary native joint actions;
    ``execution="native"`` retains the native grouped-frequency fitting path.
    Reduced GN actions use SDK coordination in either mode. Only stationary
    fits expose a reduced gradient. The penalty
    is ``damping / 2 * ||r||²`` in physical reflectivity coefficient coordinates.
    ``reflectivity`` selects inner blocks; all other contrast fields remain
    frozen in the prediction. Outer active controls must be material blocks.
    """

    def __init__(self, problem, *, reflectivity=None, execution="sdk", **solver):
        names = [
            b.name
            for b in problem.full_space.resolved_blocks
            if b.kind == "reflectivity"
        ]
        if not names:
            raise ValueError("RWI requires reflectivity controls")
        selected = problem.full_space.restrict(
            (
                names
                if reflectivity is None
                else ([reflectivity] if isinstance(reflectivity, str) else reflectivity)
            )
        )
        if not selected.blocks:
            raise ValueError("RWI needs at least one inner reflectivity block")
        if any(b.kind != "reflectivity" for b in selected.resolved_blocks):
            raise ValueError("inner RWI controls must be reflectivity blocks")
        self._reflectivity = tuple(selected.blocks)
        background = [
            b.name for b in problem.space.resolved_blocks if b.kind != "reflectivity"
        ]
        self._problem = problem.restrict(active=background)
        if execution not in {"sdk", "native"}:
            raise ValueError("RWI execution must be 'sdk' or 'native'")
        self.execution = execution
        self.solver = _solver_options(solver)
        if "frequency_weights" in solver:
            raise ValueError("set frequency weights on the ImagingProblem")
        self._fits = {}
        self._warm = None
        self._inner_space = None
        self._joint_problem = None
        errors = self.capabilities()["errors"]
        if errors:
            raise ValueError("RWI is unsupported: " + "; ".join(errors))

    def __getattr__(self, name):
        # Do not accidentally expose ordinary-FWI actions as reduced RWI actions.
        if name in {
            "name",
            "simulation",
            "site",
            "backend",
            "workdir",
            "cache",
            "misfit",
            "smoothing",
            "min_support",
            "weights",
            "frequencies",
            "observed_data",
            "observed_groups",
            "full_space",
            "space",
            "data_space",
            "patches",
            "kernel_derivative",
            "vector",
            "is_authored",
        }:
            return getattr(self._problem, name)
        raise AttributeError(name)

    @property
    def state(self):
        return self._problem.state

    @state.setter
    def state(self, value):
        self._problem.state = value

    def identity(self):
        return {
            **self._problem.identity(),
            "rwi": self.solver,
            "execution": self.execution,
            "inner_controls": self._reflectivity,
        }

    def capabilities(self):
        base = self._problem.capabilities()
        errors = list(base["errors"])
        if self.weights is not None and not any(w > 0 for w in self.weights):
            errors.append("at least one frequency weight must be positive")
        if (
            self.execution == "native"
            and self._problem._reflectivity()["parameterization"] != "vp_ip"
        ):
            errors.append(
                "native RWI supports only acoustic vp_ip reflectivity; "
                "use execution='sdk' for elastic reflectivity"
            )
        if any(
            b.kind not in {"profile", "grid", "mesh"}
            for b in self.space.resolved_blocks
        ):
            errors.append("outer controls must be material blocks; sources stay fixed")
        terms = self.misfit.objective_terms([g.name for g in self.observed_groups])
        if set(base["comparisons"]) - {"waveform"} or any(
            Loss.from_value(t.loss).kind != "l2" for t in terms
        ):
            errors.append("the inner solve requires fixed waveform L2 objectives")
        if any(t.comparison.reliability is not None for t in terms):
            errors.append("predicted reliability must be frozen before RWI")
        if self.patches is not None or self.kernel_derivative is not None:
            errors.append("patches and kernel derivatives are not supported")
        return {**base, "errors": errors, "ok": not errors}

    def restrict(self, *args, **kwargs):
        return RWIProblem(
            self._problem.restrict(*args, **kwargs),
            reflectivity=self._reflectivity,
            execution=self.execution,
            **self.solver,
        )

    def with_controls(self, *args, **kwargs):
        raise NotImplementedError("RWI stages currently retain one control basis")

    def with_misfit(self, misfit=None, *, loss=None):
        return self.restrict(misfit=misfit, loss=loss)

    def jacobian(self, v=None):
        raise NotImplementedError(
            "RWI exposes a reduced gradient, not a reduced data Jacobian"
        )

    def state_from(self, vector):
        # FWI's accepted state must include its fitted reflectivity.
        return self.linearize(vector).state

    def linearize(self, v=None, *, gradient=True):
        problem = self._problem
        problem._ensure_registry()
        state = problem._state_at(v)
        inner = self._inner_space or self.full_space.restrict(self._reflectivity)
        # A stationary reduced objective is independent of the warm start.
        key_state = state.with_update(inner.zeros())
        key = fingerprint(**self.identity(), state=key_state.values)
        if key in self._fits:
            return self._fits[key]
        if self._warm is not None:
            state = state.with_update(self._warm)
        if self.execution == "sdk":
            result = self._fit_shared(state, key)
            # Support discovery can freeze selected inner coefficients. Their
            # values remain part of the fitted objective's cache identity.
            key_state = result.state.with_update(self._inner_space.zeros())
            key = fingerprint(**self.identity(), state=key_state.values)
            result.fingerprint = key
            self._warm = result.state.vector(
                result._joint.space.restrict(self._reflectivity)
            )
            self._fits[key] = result
            if len(self._fits) > 8:
                del self._fits[next(iter(self._fits))]
            return result
        stage_key = fingerprint(reduced_key=key, warm_start=state.values)
        stage, _ = problem._stage_state(stage_key, state)
        control_state = problem._complete_state_file(state).write(
            stage / "rwi-input.h5"
        )
        weights = (
            [1.0] * len(self.frequencies)
            if self.weights is None
            else list(self.weights)
        )
        payload = problem._reflectivity()
        payload["solver"] = {**self.solver, "frequency_weights": weights}
        job = FWIOperatorJob(
            self.backend.job_name("rwi"),
            self.simulation,
            self.frequencies,
            action="solve",
            active=list(self.space.blocks) + list(inner.blocks),
            state="state.json",
            covector="gradient.h5",
            objective="objective.json",
            control_state=control_state,
            reflectivity=payload,
            misfit=problem._misfit_payload,
            min_support=self.min_support,
        )
        problem._run_job(job)
        reports = [
            json.loads(job.reflectivity_report_file(t).read_text())
            for t in range(1, job.n_tasks + 1)
        ]
        for report in reports:
            if (
                report.get("schema") != "fs-reflectivity-solve-1"
                or report.get("converged") is not True
                or report.get("gradient_kind") != "reduced"
            ):
                raise RuntimeError(
                    "RWI inner fit is not stationary; reduced gradient unavailable"
                )
            if not all(
                np.isfinite(report[k])
                for k in ("data_objective", "regularization", "stationarity_norm")
            ):
                raise RuntimeError("nonfinite RWI solve report")
        fit = None
        for t in range(1, job.n_tasks + 1):
            part = ControlState.from_file(
                ControlStateFile.read(job.reflectivity_solution_file(t)),
                self.full_space,
            )
            if fit is not None and not np.allclose(
                part.vector(inner).values,
                fit.vector(inner).values,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "RWI frequency tasks returned different reflectivities"
                )
            fit = part
        fitted = state.with_update(fit.vector(inner))
        value = (
            sum(w * r["data_objective"] for w, r in zip(weights, reports))
            + reports[0]["regularization"]
        )
        covector = _vector_from_file(reduce_covectors(job, weights), self.space)
        result = RWILinearization(
            value,
            covector,
            fitted,
            fitted.vector(self.space),
            job,
            reports,
            key,
            self,
            _settings=dict(self.solver),
        )
        self._warm = fitted.vector(inner)
        self._fits[key] = result
        if len(self._fits) > 8:
            del self._fits[next(iter(self._fits))]
        return result

    def _joint_linearize(self, state):
        # Re-enable the selected inner blocks while retaining every frozen field
        # in the complete state and native reflectivity declaration.
        if self._joint_problem is None:
            joint = copy.copy(self._problem)
            joint._active = tuple(self.space.blocks) + self._reflectivity
            joint._masks = dict(self._problem._masks)
            # Registry discovery on the outer view has not measured inner support.
            joint._masks_adopted = False
            self._joint_problem = joint
        return self._joint_problem._linearize_state(state, gradient=True)

    def _system(self, joint, *, solver=None):
        solver = self.solver if solver is None else solver
        outer = np.concatenate(
            [
                np.arange(joint.space.slices[n].start, joint.space.slices[n].stop)
                for n in self.space.blocks
            ]
        )
        inner = np.concatenate(
            [
                np.arange(joint.space.slices[n].start, joint.space.slices[n].stop)
                for n in self._reflectivity
            ]
        )

        # Krylov products are one-use; do not retain O(ncontrol * niteration) arrays.
        def action(direction):
            return joint.apply_normal_batch([direction], cache=False)[0].values

        return _ReflectivitySystem(
            action,
            joint.space.size,
            outer,
            inner,
            damping=solver["damping"],
            solver=solver,
        )

    def _fit_shared(self, state, key):
        started = perf_counter()
        initial = self._joint_linearize(state)
        self._problem._adopt_masks(initial.support_masks)
        system = self._system(initial)
        inner = initial.space.restrict(self._reflectivity)
        self._inner_space = inner
        old = initial.state.vector(inner)
        delta, report = system.fit_step(
            initial.gradient.values[system.inner], old.values
        )
        fitted = initial.state.with_update(old + delta)
        final = initial.problem._linearize_state(fitted, gradient=True)
        if not final.space.equivalent(initial.space):
            raise RuntimeError("RWI control support changed during the inner solve")
        fitted_values = fitted.vector(inner).values
        stationarity = (
            final.gradient.values[system.inner] + self.solver["damping"] * fitted_values
        )
        norm = float(np.linalg.norm(stationarity))
        threshold = report["tolerance"]
        if not np.isfinite(norm) or norm > threshold:
            raise RuntimeError(
                f"RWI inner fit is not stationary: residual {norm:.6g} exceeds {threshold:.6g}"
            )
        penalty = 0.5 * self.solver["damping"] * float(fitted_values @ fitted_values)
        report.update(
            execution="sdk",
            stationarity_norm=norm,
            data_objective=final.value,
            regularization=penalty,
            gradient_kind="reduced",
            wall_seconds=perf_counter() - started,
            frequencies=[str(f) for f in final.frequencies],
            frequency_weights=final.frequency_weights.tolist(),
            damping=self.solver["damping"],
            state_fingerprint=final.state_fingerprint,
        )
        value = final.value + penalty
        if not np.isfinite(value):
            raise RuntimeError("nonfinite RWI objective")
        background = final.space.restrict(list(self.space.blocks))
        result = RWILinearization(
            value,
            ControlVector(final.gradient.values[system.outer], background),
            fitted,
            fitted.vector(background),
            final.job,
            [report],
            key,
            self,
            final,
            self._system(final),
            _settings=dict(self.solver),
        )
        # Reuse the final joint artifact location for coordinator diagnostics and fitted state.
        directory = final.entry.directory
        if directory is not None:
            self._problem._complete_state_file(fitted).write(
                Path(directory) / self.solver["solution"]
            )
            report_path = Path(directory) / self.solver["report"]
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
        return result

    def value(self, v=None):
        return self.linearize(v).value

    def gradient(self, v=None):
        return self.linearize(v).gradient

    def normal(self, v=None):
        return self.linearize(v).normal

    def clear_cache(self):
        self._fits.clear()
        self._warm = None
