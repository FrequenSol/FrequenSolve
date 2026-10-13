# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Spectral attributes composed with Sauce's spectral model adjoints.

Inputs to the algebra are raw time moments M_n = U_f^(n)/(-2*pi*i)^n,
not Taylor coefficients. At positive real Laplace damping the logarithmic
curvature is an arrival variance only for nonnegative signals. Regularized
ratios of signed/complex traces are generalized attributes, not variances.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import pi
from typing import Any, Sequence

import numpy as np

from .receiver_pool import ReceiverPool

__all__ = ["SpectralAttribute", "SpectralJetLinearization", "ReceiverPool"]


def _rms_floor(values: np.ndarray, groups: np.ndarray, relative: float) -> np.ndarray:
    power = np.sum(abs(values) ** 2, axis=1)
    result = np.empty(len(values))
    for group in np.unique(groups):
        selected = groups == group
        rms = np.sqrt(np.mean(power[selected]))
        # A silent observed group is not informative for normalized attributes.
        result[selected] = relative * max(rms, np.finfo(float).tiny ** 0.25)
    return result


def _ratio(
    u: np.ndarray, du: np.ndarray, epsilon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    denominator = abs(u) ** 2 + epsilon**2
    q = u.conj() / denominator
    dq = du.conj() / denominator - q * (2 * (u.conj() * du).real / denominator)
    return q, dq


def _reliability_array(values, count):
    if values is None:
        return None
    values = np.asarray(values)
    if (
        np.iscomplexobj(values)
        or values.shape != (count,)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError(
            "reliability must be a finite real [0, 1] weight per output row"
        )
    frozen = np.array(values, dtype=float, copy=True)
    frozen.flags.writeable = False
    return frozen


def _row_weights(values, count):
    if values is None:
        return np.ones(count)
    values = np.asarray(values)
    if (
        np.iscomplexobj(values)
        or values.shape != (count,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
    ):
        raise ValueError(
            "row_weights must be finite, real, nonnegative weights per output row"
        )
    return np.array(values, dtype=float, copy=True)


def _jet_inputs(predicted, observed, groups):
    u, d = (
        np.array(predicted, complex, copy=True),
        np.array(observed, complex, copy=True),
    )
    if (
        u.ndim != 2
        or u.shape != d.shape
        or not u.size
        or not np.isfinite(u).all()
        or not np.isfinite(d).all()
    ):
        raise ValueError("jets must be finite matching (row, order) arrays")
    group = np.zeros(len(u), int) if groups is None else np.asarray(groups)
    if group.shape != (len(u),) or (
        group.dtype.kind in "fc" and not np.isfinite(group).all()
    ):
        raise ValueError("groups must identify every receiver row")
    return u, d, group


@dataclass(frozen=True)
class SpectralAttribute:
    """Fixed attribute definition; ``time_scale`` is in seconds.

    ``log_curvature`` fits the full complex second log-Laplace derivative;
    ``phase_slope`` fits Re(M1/M0), and ``amplitude_slope`` fits Im(M1/M0)
    (the physical-Hz log-amplitude slope divided by 2*pi). ``relative_time``
    fits the full complex difference of late and early windowed M1/M0.
    Window coefficients are ascending powers of seconds, either shared or
    shaped (receiver rows, coefficients). They remain fixed during derivatives.
    ``window`` applies a fixed time polynomial before phase_slope (degree k
    needs moments through k+1); observed floors use the windowed signal.
    ``normalized_jet`` removes amplitude, retaining common complex phase.
    Its order-zero case is stabilized complex phase alone.
    ``waveform_jet`` instead retains physical amplitudes and compares all raw
    time moments using fixed positive ``order_scales`` (one per input order).
    These scales do not depend on the predicted model or individual traces.
    Every loss is a mean over rows and feature components with fixed observed
    confidence weights. Floors are observed-group RMS fractions.
    ``confidence_floor`` can keep confidence fixed while changing the ratio
    floor; ``None`` retains the historical shared floor. For curvature,
    ``curvature_regularization="joint"`` regularizes the complete numerator
    M0*M2-M1**2 and preserves zero curvature for a single arrival. The default
    ``"separate"`` retains the historical separately regularized ratios.
    ``loss`` selects L2, pseudo-Huber, or a radial Student-t-style penalty.
    ``loss_scale`` is fixed in normalized attribute units, before confidence
    and row weighting; robust residual transformations are differentiated.
    """

    kind: str
    time_scale: float = 1.0
    relative_floor: float = 0.05
    early_window: Any = (1.0,)
    late_window: Any = (0.0, 0.0, 1.0)
    confidence_floor: float | None = None
    curvature_regularization: str = "separate"
    loss: str = "l2"
    loss_scale: float = 1.0
    window: Any = (1.0,)
    order_scales: Any = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "waveform",
            "phase_slope",
            "amplitude_slope",
            "log_curvature",
            "normalized_jet",
            "relative_time",
            "windowed_waveform",
            "waveform_jet",
        }:
            raise ValueError("unknown spectral attribute")
        for name in ("time_scale", "relative_floor", "loss_scale"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.loss not in {"l2", "pseudo_huber", "student_t"}:
            raise ValueError("loss must be l2, pseudo_huber or student_t")
        if self.confidence_floor is not None and (
            not np.isfinite(self.confidence_floor) or self.confidence_floor <= 0
        ):
            raise ValueError("confidence_floor must be positive and finite")
        if self.curvature_regularization not in {"separate", "joint"}:
            raise ValueError("curvature_regularization must be separate or joint")
        for name in ("early_window", "late_window", "window"):
            coefficients = np.array(getattr(self, name), dtype=float, copy=True)
            if (
                coefficients.ndim not in (1, 2)
                or coefficients.shape[-1] == 0
                or not np.isfinite(coefficients).all()
            ):
                raise ValueError("windows need finite real polynomial coefficients")
            coefficients.flags.writeable = False
            object.__setattr__(self, name, coefficients)

        if self.kind != "phase_slope" and not np.array_equal(self.window, [1.0]):
            raise ValueError(
                "window is supported by phase_slope; use early_window/late_window for other window objectives"
            )
        if not np.any(self.window):
            raise ValueError("traveltime window must not be identically zero")
        if self.order_scales is not None:
            scales = np.array(self.order_scales, dtype=float, copy=True)
            if (
                self.kind != "waveform_jet"
                or scales.ndim != 1
                or not len(scales)
                or not np.isfinite(scales).all()
                or np.any(scales <= 0)
            ):
                raise ValueError("order_scales requires positive fixed waveform_jet scales")
            scales.flags.writeable = False
            object.__setattr__(self, "order_scales", scales)

    def linearize(
        self,
        predicted: Any,
        observed: Any,
        *,
        groups: Any = None,
        row_weights: Any = None,
        reliability: Any = None,
        pool: ReceiverPool | None = None,
    ) -> "AttributeLinearization | PooledTraveltimeLinearization":
        """Return the attribute residual, exact real JVP and real-Hermitian VJP.

        ``row_weights`` is a fixed, finite, nonnegative vector, one objective
        weight per receiver row. It multiplies the squared residual (not its
        amplitude), in addition to observed confidence. The reduction still
        divides by the full output row count. ``reliability`` is a separate
        frozen [0, 1] weight prepared with :meth:`reliability_weights`; neither
        kind of weight changes floors or attributes. ``pool`` is opt-in and
        only supported by ``phase_slope``; its sparse transpose scatters all
        overlapping output loads back to the input receiver samples.
        """
        u, d, group = _jet_inputs(predicted, observed, groups)
        if self.kind == "phase_slope" and not np.array_equal(self.window, [1.0]):
            coefficients = self.window
            if u.shape[1] < coefficients.shape[-1] + 1:
                raise ValueError(
                    "windowed traveltime requires window degree + 1 spectral derivatives"
                )
            if coefficients.ndim == 2 and len(coefficients) != len(u):
                raise ValueError("receiver-dependent window row count differs")
            moment_pair = lambda x: np.column_stack(
                (self._window(x, coefficients, 0), self._window(x, coefficients, 1))
            )
            inner = replace(self, window=(1.0,)).linearize(
                moment_pair(u),
                moment_pair(d),
                groups=group,
                row_weights=row_weights,
                reliability=reliability,
                pool=pool,
            )
            return WindowedTraveltimeLinearization(inner, coefficients, u.shape)
        required = {
            "waveform": 0,
            "waveform_jet": 0,
            "phase_slope": 1,
            "amplitude_slope": 1,
            "log_curvature": 2,
            "normalized_jet": 0,
            "windowed_waveform": self.early_window.shape[-1] - 1,
            "relative_time": max(
                self.early_window.shape[-1], self.late_window.shape[-1]
            ),
        }[self.kind]
        if u.shape[1] <= required:
            raise ValueError(
                "insufficient spectral derivative orders for attribute/window"
            )
        if pool is not None:
            if self.kind != "phase_slope":
                raise ValueError(
                    "receiver pooling supports instantaneous traveltime (phase_slope) only"
                )
            pool.validate_groups(group)
            return self._linearize_pool(u, d, group, pool, row_weights, reliability)
        objective_weights = _row_weights(row_weights, len(u))
        reliability = _reliability_array(reliability, len(u))
        if reliability is not None:
            objective_weights *= reliability
        for coefficients in (self.early_window, self.late_window):
            if coefficients.ndim == 2 and coefficients.shape[0] != len(u):
                raise ValueError("receiver-dependent window row count differs")
        eps = _rms_floor(d[:, :1], group, self.relative_floor)
        confidence_floor = (
            self.relative_floor
            if self.confidence_floor is None
            else self.confidence_floor
        )
        confidence_eps = _rms_floor(d[:, :1], group, confidence_floor)
        weights = abs(d[:, 0]) ** 2 / (abs(d[:, 0]) ** 2 + confidence_eps**2)
        if not np.any(weights) and self.kind != "waveform_jet":
            raise ValueError("observed jet has no base-signal energy")
        scale = np.maximum(eps / self.relative_floor, np.finfo(float).tiny ** 0.25)
        jet_scale = self.time_scale ** np.arange(u.shape[1])
        eps_jet = _rms_floor(d / jet_scale, group, self.relative_floor)
        windows = (
            (self.early_window, self.late_window)
            if self.kind == "relative_time"
            else (self.early_window,)
        )
        window_eps = (
            [
                _rms_floor(self._window(d, c, 0)[:, None], group, self.relative_floor)
                for c in windows
            ]
            if self.kind in {"relative_time", "windowed_waveform"}
            else []
        )
        if self.kind == "relative_time":
            for c in windows:
                e = _rms_floor(self._window(d, c, 0)[:, None], group, confidence_floor)
                energy = abs(self._window(d, c, 0)) ** 2
                weights *= energy / (energy + e**2)
        if self.kind in {"waveform", "windowed_waveform", "waveform_jet"}:
            weights = np.ones(len(u))
        if self.kind == "waveform_jet":
            if self.order_scales is None or len(self.order_scales) != u.shape[1]:
                raise ValueError("waveform_jet requires one fixed scale per input moment")

        def forward(x: np.ndarray, dx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            kind, tau = self.kind, self.time_scale
            if kind == "waveform_jet":
                return x / self.order_scales, dx / self.order_scales
            if kind == "waveform":
                return x[:, :1] / scale[:, None], dx[:, :1] / scale[:, None]
            if kind == "windowed_waveform":
                norm = window_eps[0] / self.relative_floor
                return (self._window(x, self.early_window, 0) / norm)[:, None], (
                    self._window(dx, self.early_window, 0) / norm
                )[:, None]
            if kind == "normalized_jet":
                v, dv = x / jet_scale, dx / jet_scale
                norm = np.sqrt(np.sum(abs(v) ** 2, axis=1) + eps_jet**2)[:, None]
                tangent = (
                    dv / norm
                    - v * np.sum((v.conj() * dv).real, axis=1)[:, None] / norm**3
                )
                return v / norm, tangent
            if kind == "relative_time":
                values, tangents = [], []
                for c, e in zip((self.early_window, self.late_window), window_eps):
                    a, da = self._window(x, c, 0), self._window(dx, c, 0)
                    b, db = self._window(x, c, 1), self._window(dx, c, 1)
                    q, dq = _ratio(a, da, e)
                    values.append(b * q)
                    tangents.append(db * q + b * dq)
                return ((values[1] - values[0]) / tau)[:, None], (
                    (tangents[1] - tangents[0]) / tau
                )[:, None]
            q, dq = _ratio(x[:, 0], dx[:, 0], eps)
            mean, dmean = x[:, 1] * q, dx[:, 1] * q + x[:, 1] * dq
            if kind == "log_curvature":
                if self.curvature_regularization == "joint":
                    determinant = x[:, 0] * x[:, 2] - x[:, 1] ** 2
                    tangent = (
                        dx[:, 0] * x[:, 2] + x[:, 0] * dx[:, 2] - 2 * x[:, 1] * dx[:, 1]
                    )
                    return (determinant * q**2 / tau**2)[:, None], (
                        (tangent * q**2 + 2 * determinant * q * dq) / tau**2
                    )[:, None]
                return ((x[:, 2] * q - mean**2) / tau**2)[:, None], (
                    (dx[:, 2] * q + x[:, 2] * dq - 2 * mean * dmean) / tau**2
                )[:, None]
            part = np.real if kind == "phase_slope" else np.imag
            return (part(mean) / tau)[:, None], (part(dmean) / tau)[:, None]

        target, _ = forward(d, np.zeros_like(d))
        predicted_value, _ = forward(u, np.zeros_like(u))
        weight = np.sqrt(weights * objective_weights / (len(u) * target.shape[1]))[
            :, None
        ]
        error = predicted_value - target
        factor, derivative = self._loss_factors(error)
        residual = weight * factor * error
        # Exact local real-coordinate Jacobian: few orders, no numerical differencing.
        jacobian = np.empty((*target.shape, u.shape[1], 2), complex)
        for order in range(u.shape[1]):
            for component, direction in enumerate((1.0, 1.0j)):
                delta = np.zeros_like(u)
                delta[:, order] = direction
                tangent = forward(u, delta)[1]
                dt = 2 * (error.conj() * tangent).real / self.loss_scale**2
                jacobian[:, :, order, component] = weight * (
                    factor * tangent + derivative * dt * error
                )
        return AttributeLinearization(
            residual, jacobian, predicted_value, target, weights, reliability
        )

    def _loss_factors(self, error):
        t = abs(error / self.loss_scale) ** 2
        if self.loss == "student_t":
            small = t < 1e-5
            safe = np.where(small, 1.0, t)
            factor = np.where(
                small, 1 - t / 4 + 13 * t**2 / 96, np.sqrt(np.log1p(t) / safe)
            )
            derivative = np.where(
                small,
                -0.25 + 13 * t / 48,
                (t / (1 + t) - np.log1p(t)) / (2 * safe**2 * factor),
            )
        elif self.loss == "pseudo_huber":
            root = np.sqrt(1 + t)
            factor = np.sqrt(2 / (root + 1))
            derivative = -factor / (4 * root * (root + 1))
        else:
            factor, derivative = np.ones_like(t), np.zeros_like(t)
        return factor, derivative

    def reliability_weights(
        self,
        predicted,
        observed,
        *,
        groups=None,
        pool=None,
        floor_multiplier=1.0,
        power=1.0,
    ):
        """Freeze predicted-denominator reliability for an outer reweighting step.

        Reuse this returned array as ``reliability=`` for every trial model in
        the step. It multiplies the loss outside Student-t/Huber; its model
        derivative is intentionally excluded. Refreshing it changes the
        objective. Waveform and normalized-jet objectives are not M0 ratios
        and deliberately do not accept automatic denominator reliability.
        """
        if self.kind not in {
            "phase_slope",
            "amplitude_slope",
            "log_curvature",
            "relative_time",
        }:
            raise ValueError(
                "predicted-denominator reliability requires a ratio attribute"
            )
        if (
            not np.isfinite(floor_multiplier)
            or floor_multiplier <= 0
            or not np.isfinite(power)
            or power <= 0
        ):
            raise ValueError(
                "reliability floor multiplier and power must be positive and finite"
            )
        u, d, group = _jet_inputs(predicted, observed, groups)
        if pool is not None:
            if self.kind != "phase_slope":
                raise ValueError(
                    "receiver pooling supports instantaneous traveltime (phase_slope) only"
                )
            pool.validate_groups(group)
        windows = (
            (self.early_window, self.late_window)
            if self.kind == "relative_time"
            else (self.window if self.kind == "phase_slope" else np.array([1.0]),)
        )
        result = np.ones(len(u) if pool is None else pool.output_count)
        for coefficients in windows:
            if coefficients.shape[-1] > u.shape[1] or (
                coefficients.ndim == 2 and len(coefficients) != len(u)
            ):
                raise ValueError("reliability window differs from the supplied jets")
            a, b = self._window(u, coefficients, 0), self._window(d, coefficients, 0)
            energy = abs(a) ** 2
            floor2 = (
                floor_multiplier * _rms_floor(b[:, None], group, self.relative_floor)
            ) ** 2
            if pool is not None:
                energy, floor2 = pool.apply(energy), pool.apply(floor2)
            result *= (energy / (energy + floor2)) ** power
        return _reliability_array(result, len(result))

    def _linearize_pool(self, u, d, groups, pool, row_weights, reliability):
        # Pool sample products before division. Each sample is processed once;
        # the CSR map shares it among arrays, and its transpose sums duals.
        eps2 = pool.apply(_rms_floor(d[:, :1], groups, self.relative_floor) ** 2)
        confidence_floor = (
            self.relative_floor
            if self.confidence_floor is None
            else self.confidence_floor
        )
        confidence_eps2 = pool.apply(
            _rms_floor(d[:, :1], groups, confidence_floor) ** 2
        )

        def values(x):
            products = np.column_stack(
                (abs(x[:, 0]) ** 2, (x[:, 0].conj() * x[:, 1]).real)
            )
            pooled = pool.apply(products)
            denominator = pooled[:, 0] + eps2
            return pooled[:, 1] / denominator, denominator, pooled[:, 0]

        raw, denominator, _ = values(u)
        target, _, observed_power = values(d)
        confidence = observed_power / (observed_power + confidence_eps2)
        if not np.any(confidence):
            raise ValueError("observed jet has no base-signal energy")
        weights = _row_weights(row_weights, pool.output_count)
        reliability = _reliability_array(reliability, pool.output_count)
        if reliability is not None:
            weights *= reliability
        weight = np.sqrt(weights * confidence / pool.output_count)
        predicted, target = (
            (raw / self.time_scale)[:, None],
            (target / self.time_scale)[:, None],
        )
        error = predicted[:, 0] - target[:, 0]
        factor, derivative = self._loss_factors(error)
        residual = (weight * factor * error)[:, None]
        slope = weight * (factor + derivative * 2 * (error / self.loss_scale) ** 2)
        return PooledTraveltimeLinearization(
            residual,
            predicted,
            target,
            confidence,
            reliability,
            pool,
            u,
            raw,
            denominator,
            slope / self.time_scale,
        )

    @staticmethod
    def _window(jet: np.ndarray, coefficients: np.ndarray, shift: int) -> np.ndarray:
        return np.sum(
            jet[:, shift : shift + coefficients.shape[-1]] * coefficients, axis=1
        )


@dataclass
class WindowedTraveltimeLinearization:
    """Chain a fixed time polynomial through pooled or pointwise traveltime."""

    inner: Any
    coefficients: np.ndarray
    shape: tuple

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def jvp(self, tangent):
        tangent = np.asarray(tangent)
        if tangent.shape != self.shape:
            raise ValueError("jet tangent shape differs")
        pair = np.column_stack(
            [
                SpectralAttribute._window(tangent, self.coefficients, shift)
                for shift in (0, 1)
            ]
        )
        return self.inner.jvp(pair)

    def vjp(self, dual):
        pair = self.inner.vjp(dual)
        result = np.zeros(self.shape, dtype=complex)
        width = self.coefficients.shape[-1]
        for shift in (0, 1):
            result[:, shift : shift + width] += (
                pair[:, shift : shift + 1] * self.coefficients
            )
        return result


@dataclass
class AttributeLinearization:
    residual: np.ndarray
    jacobian: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    reliability: np.ndarray | None = None

    @property
    def value(self) -> float:
        return float(np.vdot(self.residual, self.residual).real / 2)

    def jvp(self, tangent: Any) -> np.ndarray:
        tangent = np.asarray(tangent)
        if tangent.shape != (len(self.residual), self.jacobian.shape[2]):
            raise ValueError("jet tangent shape differs")
        return np.einsum("nfk,nk->nf", self.jacobian[..., 0], tangent.real) + np.einsum(
            "nfk,nk->nf", self.jacobian[..., 1], tangent.imag
        )

    def vjp(self, dual: Any) -> np.ndarray:
        dual = np.asarray(dual)
        if dual.shape != self.residual.shape:
            raise ValueError("attribute dual shape differs")
        real = np.einsum("nfk,nf->nk", self.jacobian[..., 0].conj(), dual).real
        imag = np.einsum("nfk,nf->nk", self.jacobian[..., 1].conj(), dual).real
        return real + 1j * imag


@dataclass
class PooledTraveltimeLinearization:
    """Matrix-free real derivative of shared-sample traveltime arrays."""

    residual: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    confidence: np.ndarray
    reliability: np.ndarray | None
    pool: ReceiverPool
    _jets: np.ndarray
    _mean: np.ndarray
    _denominator: np.ndarray
    _slope: np.ndarray

    @property
    def value(self):
        return float(np.vdot(self.residual, self.residual).real / 2)

    def jvp(self, tangent):
        tangent = np.asarray(tangent)
        if tangent.shape != self._jets.shape:
            raise ValueError("jet tangent shape differs")
        u, v = self._jets[:, 0], self._jets[:, 1]
        du, dv = tangent[:, 0], tangent[:, 1]
        products = np.column_stack(
            (2 * (u.conj() * du).real, (du.conj() * v + u.conj() * dv).real)
        )
        delta = self.pool.apply(products)
        return (
            self._slope * (delta[:, 1] - self._mean * delta[:, 0]) / self._denominator
        )[:, None]

    def vjp(self, dual):
        dual = np.asarray(dual)
        if dual.shape != self.residual.shape:
            raise ValueError("attribute dual shape differs")
        load = dual[:, 0].real * self._slope / self._denominator
        back = self.pool.scatter(np.column_stack((load, load * self._mean)))
        result = np.zeros_like(self._jets)
        result[:, 0] = back[:, 0] * self._jets[:, 1] - 2 * back[:, 1] * self._jets[:, 0]
        result[:, 1] = back[:, 0] * self._jets[:, 0]
        return result


class SpectralJetLinearization:
    """Compose raw multi-frequency derivative-data states with nonlinear attributes.

    First implementation uses separate saved states for each derivative order.
    It reuses each state's model JVP/VJP, but does not fuse PDE solves across
    orders or objectives. Only unweighted, unprocessed waveform L2 states with
    fixed sources, matched rows/model and a common control space are accepted.
    """

    def __init__(self, linearizations: Sequence[Any]) -> None:
        self.planes = tuple(linearizations)
        if not self.planes:
            raise ValueError("at least one derivative state is required")
        first = self.planes[0]
        self.space = first.space
        self.frequencies = tuple(first.frequencies)
        keys, groups, fingerprints = None, None, None
        predicted, observed = [], []
        for order, lin in enumerate(self.planes):
            if (
                lin.frequencies != first.frequencies
                or not lin.space.equivalent(first.space)
                or not np.array_equal(lin.state.values, first.state.values)
            ):
                raise ValueError(
                    "spectral states must share frequency, model and control basis"
                )
            kernel = lin.problem.kernel_derivative or {}
            if order and (
                kernel.get("order") != order
                or kernel.get("residual") != "derivative"
                or kernel.get("axis") != "fourier"
                or kernel.get("source_derivative") != "frozen"
            ):
                raise ValueError(
                    "spectral planes must be consecutive raw Fourier derivative states"
                )
            if not order and kernel:
                raise ValueError("first plane must contain base traces")
            if not np.all(lin.frequency_weights == 1):
                raise ValueError("spectral input states need unit frequency weights")
            payload = lin.problem.misfit.to_fs(lin.problem.observed_groups)
            from ._patch_objective import validate_patch_misfit

            validate_patch_misfit(payload)
            for term in payload["objective_terms"]:
                norm = term["normalization"]
                if (
                    term["objective"]["kind"] != "l2"
                    or term["weight"] != 1
                    or norm["reduction"] != "sum"
                    or norm["scale"] != {"kind": "explicit", "value": 1.0}
                ):
                    raise ValueError(
                        "spectral input states require explicit unit-scale waveform L2"
                    )
            current = []
            for saved in lin.objective_states:
                identity = json.loads(saved.path.read_text())["fingerprints"]
                current.append(
                    tuple(
                        identity[k]
                        for k in (
                            "model",
                            "acquisition",
                            "source_signature",
                            "preprocessing",
                        )
                    )
                )
                for configs in saved.term_configs.values():
                    for config in configs:
                        if (
                            config["effective_weight_mass"] != 1
                            or config["weight"] != 1
                            or config["reduction"] != "sum"
                            or config["runtime"]["n_inactive"] != 0
                            or not np.all(np.asarray(config["scale"]["values"]) == 1)
                            or not np.all(
                                np.asarray(config["data_metric"]["weights"]) == 1
                            )
                        ):
                            raise ValueError(
                                "resolved spectral states must have active, unit-weight receiver rows"
                            )
            if fingerprints is None:
                fingerprints = current
            elif fingerprints != current:
                raise ValueError(
                    "spectral states have different physical model or acquisition fingerprints"
                )
            row_keys = [None] * lin.data_space.size
            row_groups = [None] * lin.data_space.size
            for fi in range(len(self.frequencies)):
                layouts = (
                    lin.data_space.term_layouts()
                    if len(self.frequencies) == 1
                    else lin.data_space.term_layouts(frequency=fi)
                )
                for layout in layouts:
                    for index, key in zip(layout.indices, layout.coordinate_keys):
                        row_keys[index] = (layout.id, *map(int, key)) + (
                            () if len(self.frequencies) == 1 else (fi,)
                        )
                        row_groups[index] = (layout.id, int(key[0]), int(key[2]), fi)
            if keys is None:
                keys = row_keys
                unique = {key: i for i, key in enumerate(dict.fromkeys(row_groups))}
                groups = np.array([unique[key] for key in row_groups])
            elif keys != row_keys:
                raise ValueError("spectral receiver rows differ between orders")
            factor = (-2j * pi) ** (-order)
            predicted.append(lin.simulated().values * factor)
            observed.append(lin.observed().values * factor)
        self.keys, self.groups = keys, groups
        self.predicted, self.observed = (
            np.column_stack(predicted),
            np.column_stack(observed),
        )

    @classmethod
    def from_problem(
        cls, problem: Any, *, order: int = 3, state: Any = None
    ) -> "SpectralJetLinearization":
        """Save raw Fourier jets; replace L2 term scales/reduction with unit weights.

        Receiver transforms, non-waveform comparisons and nonunit frequency
        weights are rejected. Observations must supply every requested raw order
        with the same frozen source-spectrum convention. Use individual shots
        when attributes are intended to describe physical arrival times.
        """
        if (
            isinstance(order, bool)
            or not isinstance(order, int)
            or order not in range(5)
        ):
            raise ValueError("spectral order must be between zero and four")
        if problem.weights is not None and not np.all(np.asarray(problem.weights) == 1):
            raise ValueError("raw spectral states require unit frequency weights")
        from ._patch_objective import validate_patch_misfit

        payload = problem.misfit.to_fs(problem.observed_groups)
        validate_patch_misfit(payload)
        if any(
            term["objective"]["kind"] != "l2" for term in payload["objective_terms"]
        ):
            raise ValueError("raw spectral states require L2 inputs")
        from .misfit import Misfit, Normalization, ObjectiveTerm

        base = problem.with_misfit(
            Misfit.terms(
                *[
                    ObjectiveTerm(
                        t["receiver_group"],
                        id=t["id"],
                        normalization=Normalization.explicit(1),
                    )
                    for t in payload["objective_terms"]
                ]
            )
        )
        planes = []
        for k in range(order + 1):
            selection = (
                None
                if k == 0
                else dict(
                    order=k,
                    axis="fourier",
                    residual="derivative",
                    source_derivative="frozen",
                )
            )
            planes.append(
                base.restrict(kernel_derivative=selection).linearize(
                    state, gradient=False
                )
            )
        return cls(planes)

    def through_order(self, order: int):
        """Reuse saved planes through an inclusive order, without any new solves."""
        if (
            isinstance(order, bool)
            or not isinstance(order, int)
            or not 0 <= order < len(self.planes)
        ):
            raise ValueError("order must select an available jet plane")
        return type(self)(self.planes[: order + 1])

    def weights_from_traces(self, weights: Any) -> np.ndarray:
        """Align dense trace-weight tables with this jet's receiver rows.

        Supply a ``(source RHS, component, receiver)`` table for a single term,
        or a mapping from term IDs to such tables. Canonical one-based keys
        select entries, so flattened storage order need not match the table.
        All weights must be finite, real and nonnegative. Use unencoded shots
        when constructing physical source-receiver offset weights: an encoded
        RHS does not in general have a unique physical source position.
        """
        terms = {key[0] for key in self.keys}
        if isinstance(weights, Mapping):
            if set(weights) != terms:
                raise ValueError(
                    "trace weight tables must match every objective term ID"
                )
            tables = dict(weights)
        else:
            if len(terms) != 1:
                raise ValueError(
                    "multiple terms require a mapping of trace weight tables"
                )
            tables = {next(iter(terms)): weights}
        result = np.empty(len(self.keys), dtype=float)
        for term, values in tables.items():
            table = np.asarray(values)
            if table.ndim != 3 or np.iscomplexobj(table):
                raise ValueError(
                    "trace weights need a real (source, component, receiver) table"
                )
            table = np.asarray(table, dtype=float)
            if not np.isfinite(table).all() or np.any(table < 0):
                raise ValueError("trace weights must be finite and nonnegative")
            selected = [i for i, key in enumerate(self.keys) if key[0] == term]
            indices = (
                np.array(
                    [
                        (self.keys[i][1], self.keys[i][3], self.keys[i][2])
                        for i in selected
                    ],
                    dtype=int,
                )
                - 1
            )
            if np.any(indices < 0) or np.any(indices >= np.array(table.shape)):
                raise ValueError(
                    "trace weight table does not cover the canonical row keys"
                )
            result[selected] = table[tuple(indices.T)]
        return result

    def receiver_pool(
        self, coordinates, radius, *, taper="uniform", sigma=None, sample_weights=None
    ):
        """Build reusable windows from output-receiver/channel coordinates.

        Pass a receiver-by-coordinate table for one term, or a mapping keyed by
        term ID. Length quantities are supported. These must be coordinates of
        the sampled output channels, not unexpanded device anchor coordinates.
        Shot and component partitions follow canonical keys, never row order.
        For a curved cable, 1D arc-length coordinates avoid pooling nearby but
        topologically distant sections. ``sample_weights`` follow input rows.
        """
        terms = {key[0] for key in self.keys}
        if isinstance(coordinates, Mapping):
            if set(coordinates) != terms:
                raise ValueError("coordinate tables must match every objective term ID")
            tables = coordinates
        elif len(terms) == 1:
            tables = {next(iter(terms)): coordinates}
        else:
            raise ValueError("multiple terms require a mapping of coordinate tables")
        dimensional = hasattr(radius, "to") and hasattr(radius, "magnitude")
        radius_value = radius.to("m").magnitude if dimensional else radius
        if sigma is not None and dimensional:
            sigma = sigma.to("m").magnitude
        aligned = None
        for term, values in tables.items():
            quantity = hasattr(values, "to") and hasattr(values, "magnitude")
            if quantity != dimensional:
                raise ValueError(
                    "coordinate tables and radius must have matching length units"
                )
            values = values.to("m").magnitude if quantity else values
            if np.iscomplexobj(values):
                raise ValueError("receiver coordinates must be real")
            values = np.asarray(values, dtype=float)
            if values.ndim == 1:
                values = values[:, None]
            if (
                values.ndim != 2
                or not all(values.shape)
                or not np.isfinite(values).all()
            ):
                raise ValueError("receiver coordinates must be a finite matrix")
            if aligned is None:
                aligned = np.empty((len(self.keys), values.shape[1]))
            elif values.shape[1] != aligned.shape[1]:
                raise ValueError("coordinate tables must share a dimension")
            selected = np.array([i for i, k in enumerate(self.keys) if k[0] == term])
            ids = np.array([self.keys[i][2] for i in selected], dtype=int) - 1
            if np.any(ids < 0) or np.any(ids >= len(values)):
                raise ValueError("coordinates do not cover canonical receiver IDs")
            aligned[selected] = values[ids]
        return ReceiverPool.from_coordinates(
            aligned,
            radius_value,
            groups=self.groups,
            taper=taper,
            sigma=sigma,
            sample_weights=sample_weights,
            sample_keys=self.keys,
        )

    def synthetic_array(self, array, coordinates, *, sample_weights=None):
        """Bind a SyntheticArray to output-channel coordinates in canonical order."""
        from .synthetic_array import SyntheticArray
        from frequensolve import Q_

        if not isinstance(array, SyntheticArray):
            raise TypeError("array must be a SyntheticArray")
        return self.receiver_pool(
            coordinates,
            Q_(array.radius_m, "m"),
            taper=array.kind,
            sigma=Q_(array.length_m, "m") if array.kind == "gaussian" else None,
            sample_weights=sample_weights,
        )

    def _validate_pool(self, pool):
        if (
            pool is not None
            and pool.sample_keys is not None
            and pool.sample_keys != tuple(self.keys)
        ):
            raise ValueError("pool sample keys differ from the spectral receiver rows")

    def reliability_weights(
        self, definition, *, pool=None, floor_multiplier=1.0, power=1.0
    ):
        """Prepare frozen reliability for reuse in this step's trial objectives."""
        self._validate_pool(pool)
        return definition.reliability_weights(
            self.predicted,
            self.observed,
            groups=self.groups,
            pool=pool,
            floor_multiplier=floor_multiplier,
            power=power,
        )

    def attribute(
        self,
        definition: SpectralAttribute,
        *,
        row_weights: Any = None,
        reliability: Any = None,
        pool: ReceiverPool | None = None,
    ) -> "BoundSpectralAttribute":
        """Bind an attribute with optional fixed objective weights in ``keys`` order.

        Use :meth:`weights_from_traces` to align dense trace-weight tables.
        Weights apply after nonlinear attributes and observed normalization;
        they are held fixed in the gradient, JVP, VJP and normal actions.
        Reuse ``reliability`` from the reference model during line searches.
        ``pool`` shares input samples between traveltime outputs; weights index
        those outputs (use pool.centers to select weights for rectangular maps).
        """
        self._validate_pool(pool)
        return BoundSpectralAttribute(
            self,
            definition.linearize(
                self.predicted,
                self.observed,
                groups=self.groups,
                row_weights=row_weights,
                reliability=reliability,
                pool=pool,
            ),
            definition=definition,
            pool=pool,
            reliability=_reliability_array(
                reliability, len(self.keys) if pool is None else pool.output_count
            ),
            row_weights=_row_weights(
                row_weights, len(self.keys) if pool is None else pool.output_count
            ),
        )

    def jvp(self, direction: Any) -> np.ndarray:
        return np.column_stack(
            [
                -lin.residual_sign * lin.jvp(direction).values * (-2j * pi) ** (-k)
                for k, lin in enumerate(self.planes)
            ]
        )

    def vjp(self, dual: Any, *, per_task: bool = False) -> Any:
        """Apply native band VJPs; optionally retain the individual frequency parts."""
        dual = np.asarray(dual)
        if dual.shape != self.predicted.shape:
            raise ValueError("jet dual shape differs")
        result = (
            [self.space.zeros() for _ in self.frequencies]
            if per_task
            else self.space.zeros()
        )
        for k, lin in enumerate(self.planes):
            if np.any(dual[:, k]):
                load = dual[:, k] * np.conj((-2j * pi) ** (-k))
                if per_task:
                    parts = lin.modeled_vjp(load, per_task=True)
                    result = [a + b for a, b in zip(result, parts)]
                else:
                    result = result + lin.modeled_vjp(load)
        return result


@dataclass
class BoundSpectralAttribute:
    jets: SpectralJetLinearization
    data: AttributeLinearization | PooledTraveltimeLinearization
    definition: SpectralAttribute | None = None
    pool: ReceiverPool | None = None
    reliability: Any = None
    row_weights: Any = None

    def trial(self, jets):
        """Evaluate a candidate with this reference's fixed weights and arrays."""
        if jets.keys != self.jets.keys or not np.array_equal(
            jets.observed, self.jets.observed
        ):
            raise ValueError("trial must share receiver keys and observations")
        return jets.attribute(
            self.definition,
            pool=self.pool,
            reliability=self.reliability,
            row_weights=self.row_weights,
        )

    @property
    def value(self) -> float:
        return self.data.value

    def gradient(self) -> Any:
        return self.jets.vjp(self.data.vjp(self.data.residual))

    def gradient_tasks(self):
        """Return per-frequency contributions; their sum is gradient()."""
        return self.jets.vjp(self.data.vjp(self.data.residual), per_task=True)

    def jvp(self, direction: Any) -> np.ndarray:
        return self.data.jvp(self.jets.jvp(direction))

    def vjp(self, dual: Any) -> Any:
        return self.jets.vjp(self.data.vjp(dual))

    def normal(self, direction: Any) -> Any:
        return self.vjp(self.jvp(direction))
