# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Experimental amplitude-insensitive HV composed with native spectral jets."""

from dataclasses import asdict, dataclass

import numpy as np

from ._hv_research import HVResearchSolver, endpoint_projection
from .receiver_pool import ReceiverPool
from .spectral_attributes import SpectralAttribute, SpectralJetLinearization


@dataclass(frozen=True)
class HVAttribute:
    """Research settings; endpoints use seconds and physical receiver distances.

    ``traveltime`` compares pooled Re(M1/M0); ``compressed_traveltime`` applies
    asinh about a fixed observed median; ``phase`` pools stabilized unit phasors
    and uses one shared deformation for their real and imaginary components.
    ``projective_jet`` compares the normalized pooled Hermitian matrix of
    (M0, M1/time_scale), removing common complex gain while remaining finite
    at M0=0 if M1 carries energy. It uses all three real Bloch coordinates.
    ``student_t`` and ``l2`` are pooled traveltime baselines without transport.
    ``floor_kind="local"`` uses observed array power, while "group" uses the
    observed receiver-line RMS. Projective jets always use local joint power.
    The legacy group-floor Student-t baseline retains observed RMS confidence;
    local-floor Student-t weights centers uniformly, like the HV variants.
    This uses an independent Q1 auxiliary discretization, not native HV's
    adaptive high-order solver. No qualified Hessian approximation is exposed.
    """

    kind: str = "traveltime"
    time_scale: float = 4.0
    relative_floor: float = 0.005
    floor_kind: str = "group"
    sigma_m: float = 300.0
    intervals: int = 512
    path_intervals: int = 16
    kappa: float = 0.05
    lambda_: float = 0.01
    epsilon: float = 0.001
    loss_scale: float = 0.03223448807274354

    def __post_init__(self):
        if self.kind not in {
            "traveltime",
            "compressed_traveltime",
            "phase",
            "projective_jet",
            "student_t",
            "l2",
        }:
            raise ValueError("unknown HV attribute")
        if self.floor_kind not in {"group", "local"}:
            raise ValueError("floor_kind must be group or local")
        for name in (
            "time_scale",
            "relative_floor",
            "sigma_m",
            "kappa",
            "epsilon",
            "loss_scale",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.lambda_ < 0 or not np.isfinite(self.lambda_):
            raise ValueError("lambda_ must be finite and nonnegative")
        for name in ("intervals", "path_intervals"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer >= 2")


class AttributeFeatures:
    """Fixed observed stabilization with an exact real-linear jet pullback."""

    def __init__(self, predicted, observed, pool, config):
        self.u, self.pool, self.config = predicted, pool, config
        self.epsilon = config.relative_floor * max(
            np.sqrt(np.mean(abs(observed[:, 0]) ** 2)), 1e-100
        )
        if config.floor_kind == "local":
            self.epsilon = config.relative_floor * np.sqrt(
                np.maximum(pool.apply(abs(observed[:, 0]) ** 2), 1e-200)
            )
        if config.kind == "projective_jet":
            power = (
                abs(observed[:, 0]) ** 2 + abs(observed[:, 1] / config.time_scale) ** 2
            )
            self.epsilon = config.relative_floor * np.sqrt(
                np.maximum(pool.apply(power), 1e-200)
            )
        self.predicted, self.p_state = self._forward(predicted)
        self.observed, _ = self._forward(observed)
        self.compression_slope = 1.0
        if config.kind == "compressed_traveltime":
            # Same constant on both endpoints; fixed from observations at all trials.
            center = np.median(self.observed)
            self.compression_slope = 1 / np.sqrt(1 + (self.predicted - center) ** 2)
            self.predicted = np.arcsinh(self.predicted - center)
            self.observed = np.arcsinh(self.observed - center)

    def _forward(self, jet):
        u, m = jet[:, 0], jet[:, 1]
        if self.config.kind == "projective_jet":
            v = m / self.config.time_scale
            power = self.pool.apply(np.column_stack((abs(u) ** 2, abs(v) ** 2)))
            cross = self.pool.apply(u.conj() * v)
            den = power.sum(axis=1) + self.epsilon**2
            features = (
                np.column_stack(
                    (2 * cross.real, 2 * cross.imag, power[:, 0] - power[:, 1])
                )
                / den[:, None]
            )
            return features, (den, features)
        if self.config.kind == "phase":
            den = np.sqrt(abs(u) ** 2 + self.epsilon**2)
            z = self.pool.apply(u / den)
            return np.column_stack((z.real, z.imag)), den
        den = self.pool.apply(abs(u) ** 2) + self.epsilon**2
        time = self.pool.apply((u.conj() * m).real) / den
        return (time / self.config.time_scale)[:, None], (den, time)

    def vjp(self, dual):
        dual = np.asarray(dual) * self.compression_slope
        result = np.zeros_like(self.u)
        u, m = self.u[:, 0], self.u[:, 1]
        if self.config.kind == "projective_jet":
            den, features = self.p_state
            a = dual / den[:, None]
            q = np.sum(a * features, axis=1)
            c = self.pool.scatter(2 * (a[:, 0] + 1j * a[:, 1]))
            v = m / self.config.time_scale
            result[:, 0] = c.conj() * v + 2 * u * self.pool.scatter(a[:, 2] - q)
            result[:, 1] = (
                c * u + 2 * v * self.pool.scatter(-a[:, 2] - q)
            ) / self.config.time_scale
        elif self.config.kind == "phase":
            z = self.pool.scatter(dual[:, 0] + 1j * dual[:, 1])
            den = self.p_state
            result[:, 0] = z / den - u * (z.conj() * u).real / den**3
        else:
            den, time = self.p_state
            a = dual[:, 0] / self.config.time_scale / den
            result[:, 0] = (
                self.pool.scatter(a) * m - 2 * self.pool.scatter(a * time) * u
            )
            result[:, 1] = self.pool.scatter(a) * u
        return result


class HVAttributeProblem:
    """Single-frequency research objective retaining standard imaging controls.

    All PDE work runs through FrequenSolve's saved spectral states and modeled
    VJPs. Shared raw problems reuse the forward jets across attribute variants.
    Coordinates must identify every physical output receiver, in length units.
    Restriction retains the objective; use first-order optimization only.
    """

    def __init__(self, problem, config, *, coordinates):
        if len(problem.frequencies) != 1:
            raise ValueError("HV attribute research currently requires one frequency")
        if problem.simulation.acquisition.source_encoding is not None:
            raise ValueError("HV attributes require independent physical shots")
        if not isinstance(config, HVAttribute):
            raise TypeError("config must be HVAttribute")
        x = np.asarray(coordinates.to("m").magnitude, float)
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("coordinates must be finite receiver points")
        distance = np.linalg.norm(np.diff(x, axis=0), axis=1)
        straight = np.linalg.norm(x[-1] - x[0])
        if (
            np.any(distance <= 0)
            or not np.allclose(distance, distance[0])
            or not np.isclose(distance.sum(), straight)
        ):
            raise ValueError("HV attributes require an ordered uniform straight line")
        self.raw, self.config, self.coordinates = problem, config, coordinates
        self.pool = ReceiverPool.from_coordinates(
            x, 3 * config.sigma_m, taper="gaussian", sigma=config.sigma_m
        )
        self.solver = HVResearchSolver(config.intervals, config.path_intervals)
        self.projection = endpoint_projection(len(x), config.intervals)
        self._latest = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.raw, name)

    @property
    def state(self):
        return self.raw.state

    @state.setter
    def state(self, value):
        self.raw.state = value

    def identity(self):
        return {
            **self.raw.identity(),
            "hv_attribute": asdict(self.config),
            "coordinates_m": np.asarray(self.coordinates.to("m").magnitude).tolist(),
        }

    def restrict(self, **kwargs):
        return type(self)(
            self.raw.restrict(**kwargs), self.config, coordinates=self.coordinates
        )

    def linearize(self, v=None, *, gradient=True):
        point = self.raw.vector(None if v is None else self.raw.state_from(v))
        if self._latest is not None and np.array_equal(
            point.values, self._latest.point.values
        ):
            if gradient:
                self._latest.ensure_gradient()
            return self._latest
        jets = SpectralJetLinearization.from_problem(self.raw, order=1, state=v)
        self._latest = AttributeLinearization(self, jets)
        if gradient:
            self._latest.ensure_gradient()
        return self._latest

    def value(self, v=None):
        return self.linearize(v, gradient=False).value

    def gradient(self, v=None):
        return self.linearize(v).gradient


class AttributeLinearization:
    def __init__(self, problem, jets):
        self.problem, self.jets = problem, jets
        first = jets.planes[0]
        self.space, self.state, self.point = first.space, first.state, first.point
        self.job, self.frequencies = first.job, first.frequencies
        self.gradient = None
        self.dual = np.zeros_like(jets.predicted)
        self.value, self.report = 0.0, []
        groups = np.unique(jets.groups)
        self.endpoints = []
        for group in groups:
            rows = np.flatnonzero(jets.groups == group)
            rows = rows[np.argsort([jets.keys[i][2] for i in rows])]
            if len(rows) != problem.pool.sample_count or not np.array_equal(
                [jets.keys[i][2] for i in rows], np.arange(1, len(rows) + 1)
            ):
                raise ValueError(
                    "HV attributes need every ordered physical receiver per shot/component"
                )
            p, d = jets.predicted[rows], jets.observed[rows]
            cfg = problem.config
            if cfg.kind == "student_t" and cfg.floor_kind == "group":
                bound = SpectralAttribute(
                    "phase_slope",
                    time_scale=cfg.time_scale,
                    relative_floor=cfg.relative_floor,
                    confidence_floor=0.05,
                    loss="student_t",
                    loss_scale=cfg.loss_scale,
                ).linearize(p, d, pool=problem.pool)
                self.value += bound.value / len(groups)
                self.dual[rows] = bound.vjp(bound.residual) / len(groups)
                self.endpoints.append((bound.predicted, bound.observed))
                continue
            feature = AttributeFeatures(p, d, problem.pool, cfg)
            if cfg.kind in {"student_t", "l2"}:
                error = feature.predicted - feature.observed
                if cfg.kind == "student_t":
                    ratio = (error / cfg.loss_scale) ** 2
                    self.value += (
                        0.5 * cfg.loss_scale**2 * np.mean(np.log1p(ratio)) / len(groups)
                    )
                else:
                    ratio = np.zeros_like(error)
                    self.value += 0.5 * np.mean(error**2) / len(groups)
                self.dual[rows] = feature.vjp(error / (1 + ratio) / len(rows)) / len(
                    groups
                )
                self.endpoints.append((feature.predicted, feature.observed))
                continue
            result = problem.solver.solve(
                problem.projection @ feature.predicted,
                problem.projection @ feature.observed,
                kappa=cfg.kappa,
                lam=cfg.lambda_,
                epsilon=cfg.epsilon,
            )
            self.value += result.value / len(groups)
            self.dual[rows] = feature.vjp(problem.projection.T @ result.dual) / len(
                groups
            )
            self.endpoints.append((feature.predicted, feature.observed))
            self.report.append(
                dict(
                    group=int(group),
                    iterations=result.iterations,
                    stationarity=result.stationarity,
                    max_receiver_shift_fraction=float(abs(result.velocity).max()),
                )
            )
        self.endpoints = np.asarray(self.endpoints)

    def ensure_gradient(self):
        if self.gradient is None:
            self.gradient = self.jets.vjp(self.dual)
            self.gradient._field_factory = self.field

    @property
    def regularization_job(self):
        return self.problem.raw.linearize(self.point).regularization_job

    def field(self, vector=None, **kwargs):
        return self.jets.planes[0].field(
            self.gradient if vector is None else vector, **kwargs
        )

    def trial(self, v, *, gradient=False):
        return self.problem.linearize(v, gradient=gradient)
