# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Band-coupled adaptive waveform inversion using native receiver checkpoints.

The default fits a real finite-duration convolutional matching filter over the
whole band, with identity-centered regularization and an exact inner-solve
pullback. The former pointwise spectral-moment formulation is available only
by explicit selection for historical diagnostics.
"""

from __future__ import annotations

import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, NoReturn

if TYPE_CHECKING:
    from .problem import ImagingProblem
from uuid import uuid4

import numpy as np

from ._artifacts import ControlVectorFile
from ._backend import fingerprint
from ._receiver import ReceiverCollection, receiver_band
from .controls import ControlSpace, ControlState, ControlVector, _check_artifact_basis
from .jobs import FWIOperatorJob
from .misfit import Loss, Misfit, Normalization
from .problem import _observed_descriptor, _vector_from_file

__all__ = ["AWI", "AWIProblem", "AWILinearization"]


@dataclass(frozen=True)
class AWI:
    """Explicit AWI band and stabilization settings.

    ``frequencies`` are strictly increasing positive real Hz, optionally on
    one constant, nonpositive imaginary-frequency line. ``quadrature_weights``
    are positive integration weights (not ordinary FWI task weights).
    ``relative_floor`` is a positive fraction of each observed trace's
    quadrature-mean power: ``lambda = relative_floor * mean_q(|d|²)``.
    It is fixed with respect to both model and frequency within that trace.
    ``filter_lags`` gives the increasing real lag locations in seconds,
    including zero. One real filter is fitted to all frequencies of each trace.
    By default the penalty is half its normalized second lag moment.
    ``loss`` optionally applies a robust radial loss to the RMS lag in seconds;
    for Student-t, ``c2`` is the squared time scale in seconds squared. The
    filter fit remains regularized least squares and its pullback is exact.
    ``filter_kind="complex"`` fits an analytic-signal filter and can absorb a
    constant phase rotation without a time shift. ``regularization="gain_invariant"``
    leaves the zero-lag gain unpenalized and scales the remaining ridge by
    predicted power, including that model dependence in the pullback. This
    makes the objective invariant to modeled gain (also phase for complex filters).
    The default identity-centered ridge makes identical data an exact zero of the objective and gradient.
    ``formulation="spectral_moment"`` explicitly selects the old pointwise
    deconvolution diagnostic; it has known identity bias and notch sensitivity.
    ``minimum_filter_energy`` is a positive threshold on filter energy:
    a degenerate matching filter raises an error rather than returning a
    spurious zero objective. Zero-observation traces must be explicitly masked
    with zero ``trace_weights``. Remaining weights are normalized globally
    across all groups. ``source_derivative`` is explicitly 'total' or 'frozen'.
    """

    frequencies: tuple[complex, ...]
    quadrature_weights: tuple[float, ...]
    relative_floor: float
    minimum_filter_energy: float
    source_derivative: str = "total"
    filter_lags: tuple[float, ...] | None = None
    formulation: str = "finite_filter"
    loss: Loss = Loss()
    filter_kind: str = "real"
    regularization: str = "identity"

    def __post_init__(self) -> None:
        object.__setattr__(self, "loss", Loss.from_value(self.loss))
        if self.filter_kind not in {"real", "complex"}:
            raise ValueError("AWI filter_kind must be real or complex")
        if self.regularization not in {"identity", "gain_invariant"}:
            raise ValueError("AWI regularization must be identity or gain_invariant")
        if self.formulation != "finite_filter" and (
            self.filter_kind != "real" or self.regularization != "identity"
        ):
            raise ValueError("AWI filter options require finite_filter")
        if self.formulation != "finite_filter" and self.loss.kind != "l2":
            raise ValueError("robust AWI losses require finite_filter")
        f = np.asarray(self.frequencies, dtype=complex)
        q = np.asarray(self.quadrature_weights, dtype=float)
        if f.ndim != 1 or len(f) < 2 or not np.isfinite(f).all():
            raise ValueError("AWI requires at least two finite frequency samples")
        if np.any(f.real <= 0) or np.any(np.diff(f.real) <= 0):
            raise ValueError(
                "AWI frequencies must have strictly increasing positive real parts"
            )
        if np.any(f.imag > 0) or not np.all(f.imag == f.imag[0]):
            raise ValueError(
                "AWI requires one constant nonpositive imaginary frequency"
            )
        if q.shape != f.shape or not np.isfinite(q).all() or np.any(q <= 0):
            raise ValueError(
                "AWI quadrature_weights must be positive and match the band"
            )
        for name in ("relative_floor", "minimum_filter_energy"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"AWI {name} must be finite and positive")
            object.__setattr__(self, name, value)
        if self.source_derivative not in {"total", "frozen"}:
            raise ValueError("AWI source_derivative must be total or frozen")
        if self.formulation not in {"finite_filter", "spectral_moment"}:
            raise ValueError("AWI formulation must be finite_filter or spectral_moment")
        if self.formulation == "finite_filter" and self.filter_lags is None:
            raise ValueError(
                "AWI requires explicit filter_lags in seconds for its bandwise matching filter"
            )
        if self.formulation == "spectral_moment" and self.filter_lags is not None:
            raise ValueError("spectral_moment does not use filter_lags")
        if self.filter_lags is not None:
            lags = np.asarray(self.filter_lags, dtype=float)
            if (
                lags.ndim != 1
                or len(lags) < 3
                or not np.isfinite(lags).all()
                or np.any(np.diff(lags) <= 0)
                or np.count_nonzero(lags == 0) != 1
            ):
                raise ValueError(
                    "AWI filter_lags must increase and include exactly one zero lag"
                )
            if np.max(np.diff(f.real)) * np.ptp(lags) >= 1:
                raise ValueError(
                    "AWI frequency spacing is too coarse for filter_lags support"
                )
            object.__setattr__(self, "filter_lags", tuple(lags))
        object.__setattr__(self, "frequencies", tuple(f))
        object.__setattr__(self, "quadrature_weights", tuple(q))

    def to_dict(self) -> dict:
        return {
            "frequencies": [[f.real, f.imag] for f in self.frequencies],
            "quadrature_weights": list(self.quadrature_weights),
            "relative_floor": self.relative_floor,
            "minimum_filter_energy": self.minimum_filter_energy,
            "source_derivative": self.source_derivative,
            "formulation": self.formulation,
            "loss": self.loss.to_fs(),
            "filter_kind": self.filter_kind,
            "regularization": self.regularization,
            "filter_lags": None if self.filter_lags is None else list(self.filter_lags),
        }

    def evaluate(
        self,
        predicted: Any,
        predicted_df: Any,
        observed: Any,
        observed_df: Any,
        *,
        trace_weights: Any = None,
        pool: Any = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Return value and independent real-Hermitian duals for (trace, frequency) arrays.

        ``delta J = Re(vdot(g_u, delta u) + vdot(g_uf, delta u_f))``.
        Weights follow lexicographic physical (shot, receiver, component) order
        when called through :class:`AWIProblem`.
        """
        arrays = [
            np.asarray(v, dtype=np.complex128)
            for v in (predicted, predicted_df, observed, observed_df)
        ]
        shape = arrays[0].shape
        if len(shape) != 2 or shape[1] != len(self.frequencies) or shape[0] == 0:
            raise ValueError(
                "AWI arrays must have shape (nonempty traces, band frequencies)"
            )
        if any(v.shape != shape or not np.isfinite(v).all() for v in arrays):
            raise ValueError("AWI arrays must be finite and have identical shapes")
        weights = (
            np.ones(shape[0])
            if trace_weights is None
            else np.asarray(trace_weights, float)
        )
        if (
            weights.shape != (shape[0],)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or weights.sum() <= 0
        ):
            raise ValueError(
                "AWI trace_weights must be finite, nonnegative, and have positive total"
            )
        if pool is not None:
            if self.filter_lags is None:
                raise ValueError("AWI arrays require finite_filter")
            if pool.sample_count != shape[0] or pool.output_count != shape[0]:
                raise ValueError("AWI arrays require one center per receiver sample")
            value, dual = self._pooled_filter(
                arrays[0], arrays[2], weights / weights.sum(), pool
            )
            return value, dual, np.zeros(shape, complex)
        selected = weights > 0
        u, uf, d, df = (v[selected] for v in arrays)
        weights = weights[selected] / weights.sum()
        q = np.asarray(self.quadrature_weights)
        q = q / q.sum()
        power = np.sum(q * abs(d) ** 2, axis=1)
        if np.any(power <= 0) or not np.isfinite(power).all():
            raise ValueError(
                "AWI needs nonzero observed power; explicitly mask zero-energy traces"
            )
        if self.filter_lags is not None:
            if np.any(
                np.sum(q * abs(u) ** 2, axis=1) / power <= self.minimum_filter_energy
            ):
                raise ValueError(
                    "AWI predicted power is too small for a matching filter"
                )
            value, dual = self._finite_filter(u, d, weights, q, power)
            gu, guf = np.zeros(shape, complex), np.zeros(shape, complex)
            gu[selected] = dual
            return value, gu, guf
        a = abs(u) ** 2 + self.relative_floor * power[:, None]
        b = d * np.conj(u)
        c = df * np.conj(u) + d * np.conj(uf)
        s = 2 * np.real(np.conj(u) * uf)
        w = b / a
        wf = c / a - b * s / a**2
        energy = np.sum(q * abs(w) ** 2, axis=1)
        numerator = np.sum(q * abs(wf) ** 2, axis=1)
        if (
            np.any(energy <= self.minimum_filter_energy)
            or not np.isfinite(energy).all()
        ):
            raise ValueError(
                "AWI matching-filter energy is below minimum_filter_energy"
            )
        value = float(np.sum(weights * numerator / (8 * np.pi**2 * energy)))
        gw = (
            -weights[:, None]
            * q
            * numerator[:, None]
            * w
            / (4 * np.pi**2 * energy[:, None] ** 2)
        )
        gwf = weights[:, None] * q * wf / (4 * np.pi**2 * energy[:, None])
        gc = gwf / a
        gb = gw / a - gwf * s / a**2
        gs = np.real(np.conj(gwf) * (-b / a**2))
        ga = np.real(
            np.conj(gw) * (-b / a**2) + np.conj(gwf) * (-c / a**2 + 2 * b * s / a**3)
        )
        gu = np.zeros(shape, complex)
        guf = np.zeros(shape, complex)
        gu[selected] = d * np.conj(gb) + df * np.conj(gc) + 2 * gs * uf + 2 * ga * u
        guf[selected] = d * np.conj(gc) + 2 * gs * u
        if (
            not np.isfinite(value)
            or not np.isfinite(gu).all()
            or not np.isfinite(guf).all()
        ):
            raise ValueError(
                "AWI evaluation overflowed; check physical scales and stabilization"
            )
        return value, gu, guf

    def _outer_loss(self, squared_spread):
        """Loss of RMS lag in seconds and multiplier of its quadratic gradient."""
        q = np.maximum(squared_spread, 0)
        loss = self.loss
        if loss.kind == "student_t":
            scale = loss.nu * loss.c2
            return 0.5 * (loss.nu + 1) * np.log1p(q / scale), (loss.nu + 1) / (
                scale + q
            )
        if loss.kind == "huber":
            radius = np.sqrt(q)
            return (
                np.where(
                    radius <= loss.delta,
                    0.5 * q,
                    loss.delta * (radius - 0.5 * loss.delta),
                ),
                np.minimum(1, loss.delta / np.maximum(radius, np.finfo(float).tiny)),
            )
        if loss.kind == "tukey":
            t = np.maximum(1 - q / loss.delta**2, 0)
            return loss.delta**2 / 6 * (1 - t**3), t**2
        return 0.5 * q, np.ones_like(q)

    def _finite_filter(
        self,
        u: np.ndarray,
        d: np.ndarray,
        weights: np.ndarray,
        q: np.ndarray,
        power: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Real or complex lag coefficients and exact regularized inner pullback.

        min_w .5 ||diag(u) E w-d||_q² + .5 lambda ||w-e0||²
        J = .5 sum(tau² w²) / sum(w²).
        Solving for w-e0 makes identity data an exact algebraic fixed point.
        Observed-power normalization keeps the small SPD systems well scaled.
        """
        tau = np.asarray(self.filter_lags)
        basis = np.exp(-2j * np.pi * np.asarray(self.frequencies)[:, None] * tau)
        if not np.isfinite(basis).all():
            raise ValueError("AWI filter_lags overflow on this Laplace line")
        e0 = np.zeros(len(tau))
        e0[np.flatnonzero(tau == 0)[0]] = 1
        penalty_diagonal = np.ones(len(tau))
        if self.regularization == "gain_invariant":
            penalty_diagonal[tau == 0] = 0
        ridge = self.relative_floor * np.diag(penalty_diagonal)
        dual = np.empty_like(u)
        value = 0.0
        # Bound temporary storage independently of the number of receivers.
        for first in range(0, len(u), 512):
            sl = slice(first, first + 512)
            scale = np.sqrt(power[sl, None])
            a, b = u[sl] / scale, d[sl] / scale
            normal = np.einsum(
                "nf,fi,fj->nij", q * abs(a) ** 2, basis.conj(), basis, optimize=True
            )
            rhs = (q * a.conj() * (b - a)) @ basis.conj()
            if self.filter_kind == "real":
                normal, rhs = normal.real, rhs.real
            if self.regularization == "gain_invariant":
                predicted_power = np.sum(q * abs(a) ** 2, axis=1)
                normal += predicted_power[:, None, None] * ridge
            else:
                normal += ridge
            w = np.linalg.solve(normal, rhs[..., None])[..., 0] + e0
            energy = np.sum(abs(w) ** 2, axis=1)
            if np.any(energy <= self.minimum_filter_energy):
                raise ValueError(
                    "AWI matching-filter energy is below minimum_filter_energy"
                )
            moment = np.sum(tau * tau * abs(w) ** 2, axis=1) / energy
            penalty, influence = self._outer_loss(moment)
            value += float(np.sum(weights[sl] * penalty))
            gw = (
                (weights[sl] * influence)[:, None]
                * (tau * tau - moment[:, None])
                * w
                / energy[:, None]
            )
            p = np.linalg.solve(normal, gw[..., None])[..., 0]
            spectrum, adjoint = w @ basis.T, p @ basis.T
            residual = b - a * spectrum
            dual[sl] = (
                q * (adjoint.conj() * residual - a * adjoint * spectrum.conj()) / scale
            )
            if self.regularization == "gain_invariant":
                ridge_dual = np.sum((p.conj() * w).real * penalty_diagonal, axis=1)
                dual[sl] -= (
                    2 * self.relative_floor * ridge_dual[:, None] * q * a / scale
                )
        if not np.isfinite(value) or not np.isfinite(dual).all():
            raise ValueError("AWI finite-filter evaluation overflowed")
        return value, dual

    def _pooled_filter(self, u, d, weights, pool):
        """Fit shared filters to pooled powers/cross-products and scatter exact duals."""
        tau = np.asarray(self.filter_lags)
        basis = np.exp(-2j * np.pi * np.asarray(self.frequencies)[:, None] * tau)
        if not np.isfinite(basis).all():
            raise ValueError("AWI filter_lags overflow on this Laplace line")
        q = np.asarray(self.quadrature_weights)
        q = q / q.sum()
        up = pool.apply(abs(u) ** 2)
        cross = pool.apply(u.conj() * d)
        power = pool.apply(abs(d) ** 2) @ q
        selected = weights > 0
        if np.any(power[selected] <= 0):
            raise ValueError("AWI needs nonzero observed array power")
        if np.any((up[selected] @ q) / power[selected] <= self.minimum_filter_energy):
            raise ValueError("AWI predicted array power is too small")
        penalty = np.ones(len(tau))
        if self.regularization == "gain_invariant":
            penalty[tau == 0] = 0
        ridge = self.relative_floor * np.diag(penalty)
        e0 = np.asarray(tau == 0, float)
        gpower = np.zeros_like(up)
        gcross = np.zeros_like(cross)
        value = 0.0
        indices = np.flatnonzero(selected)
        for first in range(0, len(indices), 512):
            idx = indices[first : first + 512]
            pu, c = up[idx] / power[idx, None], cross[idx] / power[idx, None]
            normal = np.einsum(
                "nf,fi,fj->nij", q * pu, basis.conj(), basis, optimize=True
            )
            rhs = (q * (c - pu)) @ basis.conj()
            if self.filter_kind == "real":
                normal, rhs = normal.real, rhs.real
            normal += (
                (pu @ q)[:, None, None] * ridge
                if self.regularization == "gain_invariant"
                else ridge
            )
            w = np.linalg.solve(normal, rhs[..., None])[..., 0] + e0
            energy = np.sum(abs(w) ** 2, axis=1)
            if np.any(energy <= self.minimum_filter_energy):
                raise ValueError(
                    "AWI matching-filter energy is below minimum_filter_energy"
                )
            moment = np.sum(tau**2 * abs(w) ** 2, axis=1) / energy
            cost, influence = self._outer_loss(moment)
            value += float(weights[idx] @ cost)
            gw = (
                (weights[idx] * influence / energy)[:, None]
                * (tau**2 - moment[:, None])
                * w
            )
            p = np.linalg.solve(normal, gw[..., None])[..., 0]
            spectrum, adjoint = w @ basis.T, p @ basis.T
            gcross[idx] = q * adjoint / power[idx, None]
            gpower[idx] = -q * (adjoint.conj() * spectrum).real / power[idx, None]
            if self.regularization == "gain_invariant":
                gpower[idx] -= (
                    self.relative_floor
                    * np.sum((p.conj() * w).real * penalty, axis=1)[:, None]
                    * q
                    / power[idx, None]
                )
        dual = d * pool.scatter(gcross).conj() + 2 * u * pool.scatter(gpower)
        if not np.isfinite(value) or not np.isfinite(dual).all():
            raise ValueError("AWI pooled evaluation overflowed")
        return value, dual


class AWILinearization:
    """A receiver-band value, optional material gradient, and retained native states."""

    def __init__(
        self,
        problem: AWIProblem,
        state: ControlState,
        directory: Path,
        job: FWIOperatorJob,
        states: list[ReceiverCollection],
        value: float,
        duals: list[Path],
    ) -> None:
        self.problem = problem
        self.state = state
        self.space = problem.space
        self.point = state.vector(self.space)
        self.directory = directory
        self.job = job
        self.states = states
        self.value = value
        self.duals = duals
        self.gradient: ControlVector | None = None
        self.vjp_job: FWIOperatorJob | None = None

    def field(self, vector=None, *, input_role="dual", local_solver=None):
        """Sample a model or gradient on the retained native control basis."""
        from .problem import Linearization

        return Linearization.field(
            self, vector, input_role=input_role, local_solver=local_solver
        )

    @property
    def regularization_job(self):
        """Use the covector-producing job for native regularization and mass maps."""
        if self.vjp_job is None:
            raise ValueError("AWI regularization requires a gradient linearization")
        return self.vjp_job

    @property
    def normal(self) -> NoReturn:
        raise NotImplementedError(
            "AWI has no qualified curvature; use first-order L-BFGS"
        )

    @property
    def jacobian(self) -> NoReturn:
        raise NotImplementedError(
            "AWI is a band objective; use its material gradient with L-BFGS"
        )


class AWIProblem:
    """AWI view of an ImagingProblem with one shared authored material basis.

    Supports dense physical shots and any number of receiver groups, native
    2D/3D, profile/grid and frozen meshed material controls, and checkpoint reuse.
    Field checkpoints are owned by this view and removed on LRU eviction or ``clear_cache``. A value
    followed by a gradient at the same candidate reuses its forward states.
    Optional ``synthetic_array`` pools matching-filter powers/cross-products;
    supply unit-bearing output-channel ``coordinates`` (a mapping for multiple
    groups). Arrays remain separate by physical shot and receiver component.
    No preprocessing, source controls, or AWI curvature is enabled. Use
    ``FWI(..., optimizer=LBFGS())``.
    """

    def __init__(
        self,
        problem: ImagingProblem,
        config: AWI,
        *,
        trace_weights: Mapping[str, Any] | None = None,
        synthetic_array: Any = None,
        coordinates: Any = None,
        _storage: tuple[OrderedDict[str, AWILinearization], Path] | None = None,
    ) -> None:
        if not isinstance(config, AWI):
            raise TypeError("config must be an AWI")
        if str(problem.simulation.physics).lower() not in {
            "acoustic",
            "elastic",
            "coupled",
        }:
            raise ValueError("AWI supports acoustic, elastic, or coupled modeling")
        if any(
            b.kind not in {"profile", "grid", "mesh"} or b.complex
            for b in problem.full_space.resolved_blocks
        ):
            raise ValueError(
                "AWI requires a shared profile/grid or frozen mesh material-control basis"
            )
        if problem.simulation.acquisition.source_encoding is not None:
            raise ValueError(
                "AWI requires physical shots with implicit identity encoding"
            )
        if any(
            g.survey is not None for g in problem.simulation.acquisition.receiver_groups
        ):
            raise ValueError("initial AWI validation requires dense receiver groups")
        if not problem.observed_groups or any(
            g.observed is None or (config.filter_lags is None and g.df is None)
            for g in problem.observed_groups
        ):
            raise ValueError(
                "AWI requires fixed observed data (including df for spectral_moment)"
            )
        if tuple(map(complex, problem.frequencies)) != config.frequencies:
            raise ValueError("AWI band must exactly match the problem frequencies")
        if problem.weights is not None:
            raise ValueError(
                "put AWI band weights in quadrature_weights, not FWI task weights"
            )
        if problem.kernel_derivative is not None:
            raise ValueError(
                "AWI configures its receiver channels directly; kernel_derivative is unsupported"
            )
        misfit = problem.misfit
        if (
            misfit.preprocess
            or misfit.include_default_preprocess
            or misfit.group_preprocess
            or any(term.preprocess for term in (misfit.explicit_terms or ()))
        ):
            raise ValueError(
                "initial AWI receiver actions require no preprocessing hooks"
            )
        if any(
            misfit.projection_for(g.name).to_fs().get("kind") != "identity"
            for g in problem.observed_groups
        ):
            raise ValueError(
                "initial AWI receiver actions require identity receiver projections"
            )
        self.config = config
        self.synthetic_array = synthetic_array
        self.coordinates = coordinates
        if (synthetic_array is None) != (coordinates is None):
            raise ValueError(
                "AWI synthetic_array and output-receiver coordinates must be supplied together"
            )
        if synthetic_array is not None:
            from .synthetic_array import SyntheticArray

            if (
                not isinstance(synthetic_array, SyntheticArray)
                or config.filter_lags is None
            ):
                raise ValueError(
                    "AWI synthetic arrays require SyntheticArray and finite_filter"
                )
        self.trace_weights = {
            name: np.array(v, dtype=float, copy=True)
            for name, v in (trace_weights or {}).items()
        }
        names = {g.name for g in problem.observed_groups}
        if set(self.trace_weights) - names:
            raise ValueError("AWI trace_weights name an unknown receiver group")
        for v in self.trace_weights.values():
            if v.ndim != 1 or not np.isfinite(v).all() or np.any(v < 0):
                raise ValueError("AWI trace weights must be finite nonnegative vectors")
            v.setflags(write=False)
        # The placeholder only configures observation channels. Receiver actions
        # bypass its comparison, loss, weights and normalization entirely.
        raw = Misfit(
            normalization=Normalization(kind="explicit", value=1.0, reduction="sum")
        )
        self._problem = problem.restrict(
            misfit=raw, smoothing=None, kernel_derivative=None
        )
        self._cache, self._directory = (
            (OrderedDict(), problem.workdir / f"awi_{uuid4().hex}")
            if _storage is None
            else _storage
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._problem, name)

    @property
    def space(self) -> ControlSpace:
        return self._problem.space

    @property
    def _shared(self):
        return self._problem._shared

    @property
    def state(self) -> ControlState | None:
        return self._problem.state

    @state.setter
    def state(self, value: ControlState) -> None:
        self._problem.state = value

    @property
    def smoothing(self) -> None:
        return None

    def identity(self) -> dict[str, Any]:
        return {
            **self._problem.identity(),
            "awi": self.config.to_dict(),
            "awi_trace_weights": self.trace_weights,
            "awi_array": None
            if self.synthetic_array is None
            else self.synthetic_array.to_dict(),
            "awi_coordinates": self.coordinates,
            "observed": [_observed_descriptor(g) for g in self.observed_groups],
        }

    def restrict(
        self,
        frequencies: Any = None,
        active: Any = None,
        *,
        misfit: Any = None,
        loss: Any = None,
        weights: Any = None,
        smoothing: Any = None,
        **kwargs: Any,
    ) -> AWIProblem:
        if (
            misfit is not None
            or loss is not None
            or weights is not None
            or smoothing is not None
        ):
            raise ValueError(
                "AWI stages retain their objective and explicit quadrature; no misfit/weights/smoothing overrides"
            )
        if (
            frequencies is not None
            and tuple(map(complex, frequencies)) != self.config.frequencies
        ):
            raise ValueError(
                "create a new AWI configuration to change the integration band"
            )
        view = self._problem.restrict(active=active, **kwargs)
        return AWIProblem(
            view,
            self.config,
            trace_weights=self.trace_weights,
            synthetic_array=self.synthetic_array,
            coordinates=self.coordinates,
            _storage=(self._cache, self._directory),
        )

    def with_controls(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(
            "initial AWI stages keep one shared material-control basis"
        )

    def _bind_array(self, collection):
        """Bind physical output coordinates without mixing shots, components or groups."""
        if self.synthetic_array is None:
            return None
        from frequensolve import Q_

        names = sorted(collection.groups)
        tables = self.coordinates
        if not isinstance(tables, Mapping):
            if len(names) != 1:
                raise ValueError("Multiple AWI groups require a coordinate mapping")
            tables = {names[0]: tables}
        if set(tables) != set(names):
            raise ValueError("AWI coordinates must cover each receiver group")
        points, groups, keys = [], [], []
        group_ids = {}
        for name in names:
            if not hasattr(tables[name], "to"):
                raise TypeError("AWI output-receiver coordinates require length units")
            table = np.asarray(tables[name].to("m").magnitude)
            if table.ndim == 1:
                table = table[:, None]
            if table.ndim != 2 or not np.isfinite(table).all():
                raise ValueError("AWI output coordinates must be a finite matrix")
            for shot, receiver, component in collection.groups[name].keys:
                if not 1 <= receiver <= len(table):
                    raise ValueError(
                        "AWI coordinates do not cover physical receiver IDs"
                    )
                points.append(table[receiver - 1])
                key = (name, int(shot), int(component))
                groups.append(group_ids.setdefault(key, len(group_ids)))
                keys.append((name, int(shot), int(receiver), int(component)))
        return self.synthetic_array.bind(
            Q_(np.asarray(points), "m"), groups=groups, sample_keys=keys
        )

    def _receiver_misfit(self) -> dict[str, Any]:
        """Identity encoding makes physical-shot and RHS observation coordinates identical."""
        payload = self._problem._misfit_payload.to_fs()
        for group in payload["receiver_groups"]:
            group["observed_source_basis"] = "source_encoding"
            refs = [
                group.get("observed"),
                *group.get("observed_derivatives", {}).values(),
            ]
            for ref in refs:
                if isinstance(ref, dict):
                    ref["source_basis"] = "source_encoding"
        return payload

    def linearize(self, v: Any = None, *, gradient: bool = True) -> AWILinearization:
        self._problem.restrict(
            frequencies=self.config.frequencies[:1]
        )._ensure_registry()
        state = self._problem._state_at(v)
        key = fingerprint(**self.identity(), awi_state=state.values)
        lin = self._cache.get(key)
        if lin is not None and not self.space.equivalent(lin.space):
            self._cache.pop(key)
            shutil.rmtree(lin.directory, ignore_errors=True)
            lin = None
        if lin is None:
            self._problem._sync_simulation(state)
            _, control_state = self._problem._stage_state(key, state)
            directory = self._directory / key.split(":")[-1]
            directory.mkdir(parents=True, exist_ok=True)
            job = FWIOperatorJob(
                self.backend.job_name("awi_receiver"),
                self.simulation,
                list(self.config.frequencies),
                action="receiver_linearize",
                control_state=control_state,
                active=list(self.space.blocks),
                receiver_state=directory / "state.json",
                channel="base_df" if self.config.filter_lags is None else "base",
                source_derivative=(
                    self.config.source_derivative
                    if self.config.filter_lags is None
                    else "none"
                ),
                field_retention="checkpoint",
                misfit=self._receiver_misfit(),
            )
            try:
                self._problem._run_job(job)
                states = [
                    ReceiverCollection.read(job.receiver_state_file(i))
                    for i in range(1, job.n_tasks + 1)
                ]
                bands = receiver_band(
                    states,
                    self.config.frequencies,
                    (
                        self.config.source_derivative
                        if self.config.filter_lags is None
                        else "none"
                    ),
                )
                names = sorted(bands)
                rows = np.concatenate([bands[name] for name in names])
                weights = np.concatenate(
                    [
                        self.trace_weights.get(name, np.ones(len(bands[name])))
                        for name in names
                    ]
                )
                for name in names:
                    if name in self.trace_weights and self.trace_weights[
                        name
                    ].shape != (len(bands[name]),):
                        raise ValueError(
                            f"AWI trace weight count differs for group {name}"
                        )
                pool = self._bind_array(states[0])
                value, gu, guf = self.config.evaluate(
                    *(rows[:, :, i] for i in range(4)), trace_weights=weights, pool=pool
                )
                duals = []
                for index, collection in enumerate(states):
                    offset, values = 0, {}
                    for name in names:
                        count = len(bands[name])
                        values[name] = (
                            gu[offset : offset + count, index],
                            guf[offset : offset + count, index],
                        )
                        offset += count
                    duals.append(
                        collection.write_dual(
                            directory / f"dual_{index + 1}.json", values
                        )
                    )
                lin = AWILinearization(
                    self, state, directory, job, states, value, duals
                )
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            self._cache[key] = lin
            while len(self._cache) > self._problem.cache.capacity:
                _, evicted = self._cache.popitem(last=False)
                shutil.rmtree(evicted.directory, ignore_errors=True)
        self._cache.move_to_end(key)
        if gradient and lin.gradient is None:
            self._pullback(lin)
        return lin

    def _pullback(self, lin: AWILinearization) -> None:
        self._problem._sync_simulation(lin.state)
        job = FWIOperatorJob(
            self.backend.job_name("awi_vjp"),
            self.simulation,
            list(self.config.frequencies),
            action="receiver_vjp",
            control_state=lin.job.control_state,
            active=list(self.space.blocks),
            receiver_state=lin.directory / "state.json",
            receiver_vector=lin.directory / "dual.json",
            covector=lin.directory / "gradient.h5",
            field_reuse="checkpoint",
            misfit=self._receiver_misfit(),
        )
        self._problem._run_job(job)
        total = np.zeros(self.space.size)
        for index, collection in enumerate(lin.states, 1):
            part = ControlVectorFile.read(job.covector_file(index), native=False)
            registry = next(iter(collection.groups.values())).manifest[
                "control_registry_fingerprint"
            ]
            if (
                part.state_fingerprint != collection.fingerprint
                or part.control_registry_fingerprint != registry
            ):
                raise ValueError(
                    "AWI covector does not belong to its receiver state/registry"
                )
            _check_artifact_basis(part, lin.space)
            # Every task uses the identical authored or frozen mesh coefficient basis; runtime
            # registry fingerprints can differ with the frequency-specific mesh.
            total += _vector_from_file(part, lin.space).values
        lin.gradient = ControlVector(total, lin.space)
        lin.vjp_job = job

    def value(self, v: Any = None) -> float:
        return self.linearize(v, gradient=False).value

    def gradient(self, v: Any = None) -> ControlVector:
        result = self.linearize(v).gradient
        assert result is not None
        return result

    def clear_cache(self) -> None:
        self._cache.clear()
        if self._directory.exists():
            shutil.rmtree(self._directory)
