"""Validated physical-time integration and source functions.

Times are seconds, frequencies are hertz, and source amplitudes multiply the
physical strengths in the simulation acquisition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from frequensolve.seismic.spectra import SampledWavelet
from frequensolve.units import _scalar_in_units, is_quantity

if TYPE_CHECKING:
    from pint import Quantity

__all__ = [
    "Newmark",
    "Gauss",
    "Radau",
    "TimeIntegration",
    "TimeFunction",
    "CheckpointSchedule",
    "TimeOutput",
]


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _count(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _path(value: str | Path, name: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty path")
    raw = str(value)
    if any(c in raw for c in ("\x00", "\n", "\r", "\\")) or ".." in Path(raw).parts:
        raise ValueError(f"{name} must be a portable path without parent traversal")
    return raw


def _fields(data: Mapping[str, Any], allowed: set[str], name: str) -> dict:
    if not isinstance(data, Mapping):
        raise TypeError(f"{name} must be a mapping")
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    return dict(data)


@dataclass(frozen=True)
class Newmark:
    """Implicit Newmark method; defaults to average acceleration.

    PML requires the default coefficients. Other wave runs accept the backend's
    unconditional-stability region.
    """

    beta: float = 0.25
    gamma: float = 0.5

    def __post_init__(self) -> None:
        beta = _number(self.beta, "Newmark.beta")
        gamma = _number(self.gamma, "Newmark.gamma")
        if gamma < 0.5 or beta < 0.25 * (gamma + 0.5) ** 2:
            raise ValueError(
                "Newmark requires gamma >= 0.5 and beta >= (gamma + 0.5)^2 / 4"
            )

    def to_fs(self) -> dict:
        return {"beta": float(self.beta), "gamma": float(self.gamma)}

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> Newmark:
        return cls(**_fields(data, {"beta", "gamma"}, "newmark"))


@dataclass(frozen=True)
class Gauss:
    """Two-stage propagator with one complex solve per step.

    ``f_0=0`` selects standard fourth-order Gauss. A positive fitting frequency
    in Hz fits its undamped phase; an explicit dt resolves the stage coefficients.
    Pint frequency quantities are accepted. Changing dt preserves this frequency.
    """

    order: int = 4
    f_0: float = 0.0

    def __post_init__(self) -> None:
        if _count(self.order, "Gauss.order", 1) != 4:
            raise ValueError("Gauss currently supports order=4 only")
        frequency = _scalar_in_units(self.f_0, "Hz", "Gauss.f_0")
        if frequency < 0:
            raise ValueError("Gauss.f_0 must be nonnegative")
        object.__setattr__(self, "f_0", frequency)


@dataclass(frozen=True)
class Radau:
    """Third-order, two-stage L-stable Radau IIA with one complex solve per step.

    Requires a Sauce backend supporting radau3. CPU execution is supported.
    Choose dt or a phase-error target at f_max; phase targets do not bound damping.
    """

    order: int = 3

    def __post_init__(self) -> None:
        if _count(self.order, "Radau.order", 1) != 3:
            raise ValueError("Radau currently supports order=3 only")


def _gauss_b(theta: float) -> float:
    """Stable quadratic coefficient of the symmetric rational propagator."""
    if theta < 0.1:
        x = theta * theta
        return 1 / 12 + x * (
            1 / 720 + x * (1 / 30240 + x * (1 / 1209600 + x / 47900160))
        )
    return (1 - theta / (2 * math.tan(theta / 2))) / theta**2


def _window_radians(value: Any) -> float:
    if not isinstance(value, Real) and is_quantity(value):
        # Pint treats angles as dimensionless; conversion alone would also
        # accept percentages or steradians. Require the supported angle units.
        if str(value.units) not in {"radian", "degree"}:
            raise ValueError("window_error requires radians or degrees")
        value = value.to("radian").magnitude
    return _number(value, "window_error")


def _period_lag(x: float, scheme: str, b: float = 1 / 12) -> float:
    """Fractional harmonic phase lag, evaluated without small-step cancellation."""
    if scheme == "radau":
        # R(z)=(1+z/3)/(1-2*z/3+z*z/6); its phase is continuous below Nyquist.
        if x < 0.1:
            y = x * x
            return (
                y
                * y
                * (1 / 270 + y * (-1 / 6804 + y * (-1 / 17496 + y * 31 / 3464208)))
            )
        phase = math.atan(x / 3) + math.atan2(2 * x / 3, 1 - x * x / 6)
        return 1 - phase / x
    if x < 0.01:
        y = x * x
        if scheme == "newmark":
            return y * (1 / 12 - y * (1 / 80 - y * (1 / 448 - y / 2304)))
        if b == 1 / 12:
            return y * y * (1 / 720 - y / 12096)
        return y * (1 / 12 - b + y * (-b * b + b / 4 - 1 / 80))
    denominator = 1 if scheme == "newmark" else 1 - b * x * x
    return 1 - 2 * math.atan2(x / 2, denominator) / x


@dataclass(frozen=True)
class TimeIntegration:
    """Fixed physical-time integration, using seconds and hertz.

    Orders 3 and 4 default to third-order Radau IIA. Order 4 is a compatibility
    selector and resolves to order=3; explicit ``scheme="gauss"`` or ``Gauss()``
    selects fourth-order Gauss. Order 2 selects average-acceleration Newmark.
    Phase-based selection bounds temporal phase error, not amplitude damping.
    Either selector can be omitted; both omitted select Newmark. Legacy
    ``TimeIntegration(start, end, dt, Newmark(...))`` calls remain supported.

    Specify exactly one of ``dt``, ``period_error`` or ``window_error``.
    ``period_error`` bounds fractional harmonic phase lag per period at
    ``f_max``; ``window_error`` bounds accumulated phase lag in radians over
    ``end-start``. A Pint quantity in radians or degrees is also accepted.
    These estimates cover temporal dispersion of an undamped
    harmonic mode, not spatial, source, attenuation or linear-solve errors.
    Error-based selection requires explicit ``f_max`` and chooses a fixed dt
    dividing the interval, with at least two steps per highest-frequency period.
    Explicit dt must divide the interval and is never silently adjusted.
    """

    start: float
    end: float
    dt: float | None = None
    method: Newmark | Gauss | Radau | None = None
    order: int | None = field(default=None, kw_only=True)
    scheme: str | None = field(default=None, kw_only=True)
    f_0: float | None = field(default=None, kw_only=True)
    f_max: float | None = field(default=None, kw_only=True)
    period_error: float | None = field(default=None, kw_only=True)
    window_error: float | Quantity | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        start = _scalar_in_units(self.start, "s", "start")
        end = _scalar_in_units(self.end, "s", "end")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        duration = end - start
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("TimeIntegration requires a finite end > start")
        if self.order is not None:
            _count(self.order, "order", 1)
        method = self.method
        if method is not None:
            if any(v is not None for v in (self.scheme, self.order, self.f_0)):
                raise ValueError("Specify method or scheme/order/f_0, not both")
            if not isinstance(method, (Newmark, Gauss, Radau)):
                raise TypeError("method must be Newmark(), Gauss() or Radau()")
            scheme = (
                "newmark"
                if isinstance(method, Newmark)
                else "radau" if isinstance(method, Radau) else "gauss"
            )
            # Non-average-acceleration Newmark remains available via method.
            order = (
                (2 if method.gamma == 0.5 else 1)
                if scheme == "newmark"
                else method.order
            )
        else:
            defaults = {2: "newmark", 3: "radau", 4: "radau"}
            orders = {"newmark": 2, "radau": 3, "gauss": 4}
            scheme = (
                self.scheme
                if self.scheme is not None
                else defaults.get(self.order or 2)
            )
            if not isinstance(scheme, str) or scheme not in orders:
                raise ValueError(
                    "Supported schemes/orders are newmark/2, radau/3 and gauss/4"
                )
            order = orders[scheme] if self.order is None else self.order
            if scheme == "radau" and order == 4:
                order = 3
            if order != orders[scheme]:
                raise ValueError(f"scheme={scheme!r} requires order={orders[scheme]}")
            if scheme != "gauss" and self.f_0 is not None:
                raise ValueError("f_0 requires the gauss scheme")
            if scheme == "newmark":
                method = Newmark()
            elif scheme == "radau":
                method = Radau()
            else:
                method = Gauss(f_0=0.0 if self.f_0 is None else self.f_0)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "order", order)
        object.__setattr__(
            self, "f_0", method.f_0 if isinstance(method, Gauss) else None
        )
        if isinstance(method, Gauss) and method.f_0 != 0 and self.dt is None:
            raise ValueError("Phase-fitted Gauss requires explicit dt")

        if (
            sum(v is not None for v in (self.dt, self.period_error, self.window_error))
            != 1
        ):
            raise ValueError("Specify exactly one of dt, period_error or window_error")
        frequency = self.f_max
        if frequency is not None:
            frequency = _scalar_in_units(frequency, "Hz", "f_max")
            object.__setattr__(self, "f_max", frequency)
            if frequency <= 0 or not 0 < 2 * math.pi * frequency * duration < math.inf:
                raise ValueError("f_max must be positive with finite window phase")
        if self.dt is None:
            if frequency is None:
                raise ValueError("Error-based timestep selection requires f_max in Hz")
            if isinstance(method, Newmark) and method != Newmark():
                raise ValueError(
                    "Error-based selection requires average-acceleration Newmark, Gauss or Radau"
                )
            if self.period_error is not None:
                target = _number(self.period_error, "period_error")
                if not 0 < target < 1:
                    raise ValueError("period_error must be between 0 and 1")
            else:
                error = _window_radians(self.window_error)
                object.__setattr__(self, "window_error", error)
                if error <= 0:
                    raise ValueError("window_error must be positive (radians)")
                target = error / (2 * math.pi * frequency * duration)
            # The phase lag is monotone on this non-aliased interval. Bisection
            # is a one-time authoring cost and needs no solver or SciPy objects.
            low, high = 0.0, math.pi
            for _ in range(80):
                middle = (low + high) / 2
                if _period_lag(middle, scheme) <= target:
                    low = middle
                else:
                    high = middle
            n = 2 * math.pi * frequency * duration / low if low else math.inf
            if not math.isfinite(n) or n > 999_999_999_999:
                raise ValueError(
                    "TimeIntegration step count must fit the 12-digit output index"
                )
            dt = duration / max(1, math.ceil(n))
        else:
            dt = _scalar_in_units(self.dt, "s", "dt")
        if dt <= 0 or start + dt <= start or end - dt >= end:
            raise ValueError(
                "dt must be positive and resolvable at the interval endpoints"
            )
        n = duration / dt
        if not math.isfinite(n) or not 1 <= n <= 999_999_999_999:
            raise ValueError(
                "TimeIntegration step count must fit the 12-digit output index"
            )
        if abs(n - round(n)) > 64 * np.finfo(float).eps * max(1, abs(n)):
            raise ValueError("dt must divide the requested time interval")
        object.__setattr__(self, "dt", dt)
        if isinstance(method, Gauss) and not 0 <= self.theta_0 < math.pi:
            raise ValueError("Phase fitting requires f_0 * dt < 0.5 (below Nyquist)")

    def replace(self, **changes) -> TimeIntegration:
        """Return a validated configuration with selected inputs changed.

        Unlike ``dataclasses.replace``, this handles resolved scheme selectors
        and automatic timestep targets. Setting ``dt`` clears error targets;
        setting an error target clears ``dt`` and the other target. A fitted
        Gauss method retains its physical fitting frequency when dt changes.
        """
        values = dict(
            start=self.start,
            end=self.end,
            dt=self.dt,
            method=self.method,
            f_max=self.f_max,
            period_error=self.period_error,
            window_error=self.window_error,
        )
        if self.period_error is not None or self.window_error is not None:
            values["dt"] = None
        selectors = {k for k in ("dt", "period_error", "window_error") if k in changes}
        if selectors:
            for key in ("dt", "period_error", "window_error"):
                values[key] = None
        if any(k in changes for k in ("scheme", "order", "f_0")):
            values.pop("method")
            if "f_0" in changes and not {"scheme", "order"} & changes.keys():
                values["scheme"] = self.scheme
        values.update(changes)
        return type(self)(**values)

    def with_dt(self, dt) -> TimeIntegration:
        """Return this interval and scheme with an explicit timestep in seconds or units."""
        return self.replace(dt=dt)

    @property
    def theta_0(self) -> float:
        """Resolved fitting angle in radians per step (derived from Hz and seconds)."""
        return (
            2 * math.pi * self.method.f_0 * self.dt
            if isinstance(self.method, Gauss)
            else 0.0
        )

    @property
    def b(self) -> float | None:
        """Resolved Gauss propagator coefficient; None for other schemes."""
        return _gauss_b(self.theta_0) if isinstance(self.method, Gauss) else None

    @property
    def steps(self) -> int:
        """Number of advances, excluding the initial state."""
        return round((self.end - self.start) / self.dt)

    @property
    def times(self) -> np.ndarray:
        return self.start + np.arange(self.steps + 1) * self.dt

    @property
    def estimated_period_error(self) -> float | None:
        """Fractional phase lag at f_max; None without a supported estimate."""
        if self.f_max is None or (
            isinstance(self.method, Newmark) and self.method != Newmark()
        ):
            return None
        x = 2 * math.pi * self.f_max * self.dt
        b = self.b if isinstance(self.method, Gauss) else 1 / 12
        return _period_lag(x, self.scheme, b) if x <= math.pi else None

    @property
    def estimated_window_error(self) -> float | None:
        """Accumulated phase lag in radians at f_max."""
        error = self.estimated_period_error
        return (
            None
            if error is None
            else error * 2 * math.pi * self.f_max * (self.end - self.start)
        )

    def selection_payload(self) -> dict:
        """Frontend-only selection metadata; excluded from Sauce's Time block."""
        return {
            key: float(getattr(self, key))
            for key in ("f_max", "period_error", "window_error")
            if getattr(self, key) is not None
        }

    def to_fs(self) -> dict:
        payload = {
            "schema": "fs-time-domain-1",
            "start": float(self.start),
            "end": float(self.end),
            "time_step": float(self.dt),
            "method": {"newmark": "newmark", "gauss": "gauss4", "radau": "radau3"}[
                self.scheme
            ],
        }
        if isinstance(self.method, Newmark):
            payload["newmark"] = self.method.to_fs()
        elif isinstance(self.method, Gauss) and self.method.f_0 != 0:
            payload["gauss"] = {"f_0": float(self.method.f_0)}
        return payload

    @classmethod
    def from_fs(
        cls, data: Mapping[str, Any], *, selection: Mapping[str, Any] | None = None
    ) -> TimeIntegration:
        data = _fields(
            data,
            {"schema", "start", "end", "time_step", "method", "newmark", "gauss"},
            "TimeIntegration",
        )
        if data.get("schema", "fs-time-domain-1") != "fs-time-domain-1":
            raise ValueError("Expected fs-time-domain-1")
        name = data.get("method", "newmark")
        if name not in {"newmark", "gauss4", "radau3"}:
            raise ValueError("Expected method newmark, gauss4 or radau3")
        if name != "newmark" and "newmark" in data:
            raise ValueError("Collocation methods do not accept newmark coefficients")
        if name != "gauss4" and "gauss" in data:
            raise ValueError("Only Gauss accepts gauss coefficients")
        if name == "newmark":
            method = Newmark.from_fs(data.get("newmark", {}))
        elif name == "radau3":
            method = Radau()
        else:
            method = Gauss(**_fields(data.get("gauss", {}), {"f_0"}, "gauss"))
        selection = (
            {}
            if selection is None
            else _fields(
                selection, {"f_max", "period_error", "window_error"}, "time_selection"
            )
        )
        automatic = any(
            selection.get(key) is not None for key in ("period_error", "window_error")
        )
        time = cls(
            data["start"],
            data["end"],
            None if automatic else data["time_step"],
            method,
            **selection,
        )
        if not math.isclose(
            _number(data["time_step"], "time_step"),
            time.dt,
            rel_tol=64 * np.finfo(float).eps,
            abs_tol=0,
        ):
            raise ValueError("time_selection differs from the resolved time_step")
        return time


@dataclass(frozen=True)
class TimeFunction:
    """Shared source multiplier evaluated in physical seconds.

    Acoustic scalar/pressure monopoles and forces load its value, volume rates its first derivative, and
    isotropic moments its second derivative times local compressibility.
    Elastic forces and moments load its value. ``sampled(wavelet)`` reuses an
    existing dimensionless SampledWavelet without normalization or filtering.
    Samples are interpolated by cubic Hermite polynomials with centered interior
    slopes and zero endpoint slopes; values outside the recorded interval are
    zero. Derivative loads require zero endpoint samples to avoid impulses.
    """

    kind: str
    amplitude: float = 1.0
    delay: float = 0.0
    frequency: float | None = None
    width: float | None = None
    wavelet: SampledWavelet | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "constant",
            "ricker",
            "gaussian",
            "sine",
            "step",
            "sampled",
        }:
            raise ValueError(
                "TimeFunction kind must be constant, ricker, gaussian, sine, step or sampled"
            )
        _number(self.amplitude, "amplitude")
        object.__setattr__(self, "delay", _scalar_in_units(self.delay, "s", "delay"))
        for key, needed in (
            ("frequency", self.kind in {"ricker", "sine"}),
            ("width", self.kind == "gaussian"),
        ):
            value = getattr(self, key)
            if value is not None:
                value = _scalar_in_units(
                    value, "Hz" if key == "frequency" else "s", key
                )
                object.__setattr__(self, key, value)
            if needed and (value is None or _number(value, key) <= 0):
                raise ValueError(f"{self.kind} requires positive {key}")
            if not needed and value is not None:
                raise ValueError(f"{key} is not used by {self.kind}")
        if self.kind == "sampled":
            if not isinstance(self.wavelet, SampledWavelet):
                raise TypeError("sampled requires a SampledWavelet")
            if self.wavelet.units != "1":
                raise ValueError(
                    "Transient samples must be dimensionless; use wavelet.relative_to() "
                    "with the acquisition source strength first"
                )
            if len(self.wavelet.samples) < 2:
                raise ValueError(
                    "Transient sampled sources require at least two samples"
                )
            start = self.wavelet.t0 + self.delay
            end = start + (len(self.wavelet.samples) - 1) * self.wavelet.dt
            if (
                not np.isfinite([start, end]).all()
                or start + self.wavelet.dt <= start
                or end - self.wavelet.dt >= end
            ):
                raise ValueError("Sampled source clock must be finite and resolvable")
        elif self.wavelet is not None:
            raise ValueError(f"wavelet is not used by {self.kind}")

    def __eq__(self, other):
        if not isinstance(other, TimeFunction):
            return NotImplemented
        return self.to_fs() == other.to_fs()

    @classmethod
    def sampled(
        cls, wavelet: SampledWavelet, *, delay: float = 0.0, amplitude: float = 1.0
    ) -> TimeFunction:
        """Use recorded samples on their original clock, shifted by ``delay``.

        Fourier normalization is irrelevant to time-domain values. No taper,
        resampling, amplitude normalization or automatic causal shift is applied.
        Start the simulation before ``wavelet.t0 + delay`` for a quiet initial state.
        """
        return cls("sampled", amplitude, delay, wavelet=wavelet)

    @classmethod
    def ricker(
        cls, frequency: float, *, delay: float, amplitude: float = 1.0
    ) -> TimeFunction:
        return cls("ricker", amplitude, delay, frequency=frequency)

    @classmethod
    def gaussian(
        cls, width: float, *, delay: float, amplitude: float = 1.0
    ) -> TimeFunction:
        return cls("gaussian", amplitude, delay, width=width)

    @classmethod
    def sine(
        cls, frequency: float, *, delay: float = 0.0, amplitude: float = 1.0
    ) -> TimeFunction:
        return cls("sine", amplitude, delay, frequency=frequency)

    @classmethod
    def constant(cls, amplitude: float = 1.0) -> TimeFunction:
        return cls("constant", amplitude)

    @classmethod
    def step(cls, *, delay: float, amplitude: float = 1.0) -> TimeFunction:
        return cls("step", amplitude, delay)

    def evaluate(self, time: Any, *, derivative: int = 0) -> np.ndarray:
        """Evaluate the value or first/second derivative of the interpolant."""
        derivative = _count(derivative, "derivative")
        if derivative > 2:
            raise ValueError("Only derivatives 0, 1 and 2 are supported")
        if is_quantity(time):
            time = time.to("s").magnitude
        t = np.asarray(time, dtype=float) - self.delay
        if not np.isfinite(t).all():
            raise ValueError("time must be finite")
        if self.kind == "sampled":
            value = self._evaluate_sampled(t, derivative)
        elif self.kind == "constant":
            value = np.ones_like(t) if derivative == 0 else np.zeros_like(t)
        elif self.kind == "step":
            if derivative:
                raise ValueError(
                    "Step derivatives are impulsive; use a smooth source function"
                )
            value = (t >= 0).astype(float)
        elif self.kind == "sine":
            rate = 2 * np.pi * self.frequency
            value = rate**derivative * np.sin(rate * t + derivative * np.pi / 2)
        elif self.kind == "gaussian":
            q = t / self.width
            value = np.exp(-0.5 * q**2)
            if derivative == 1:
                value = -q / self.width * value
            elif derivative == 2:
                value = (q**2 - 1) / self.width**2 * value
        else:
            rate = (np.pi * self.frequency) ** 2
            q = rate * t**2
            value = np.exp(-q)
            if derivative == 0:
                value = (1 - 2 * q) * value
            elif derivative == 1:
                value = 2 * rate * t * (2 * q - 3) * value
            else:
                value = 2 * rate * (12 * q - 4 * q**2 - 3) * value
        return self.amplitude * value

    def _evaluate_sampled(self, time: np.ndarray, derivative: int) -> np.ndarray:
        """Evaluate the same compact Hermite interpolant as the backend."""
        w = self.wavelet
        assert w is not None
        if derivative and (w.samples[0] != 0 or w.samples[-1] != 0):
            raise ValueError(
                "Sampled source derivatives require zero endpoint samples; pad or taper explicitly"
            )
        q = (time - w.t0) / w.dt
        nearest = np.rint(q)
        q = np.where(
            np.abs(q - nearest) <= 64 * np.finfo(float).eps * np.maximum(1, np.abs(q)),
            nearest,
            q,
        )
        inside = (q >= 0) & (q <= len(w.samples) - 1)
        q = np.clip(q, 0, len(w.samples) - 1)
        i = np.minimum(np.floor(q).astype(np.int64), len(w.samples) - 2)
        u = q - i
        slopes = np.zeros_like(w.samples)
        slopes[1:-1] = (w.samples[2:] - w.samples[:-2]) / 2
        a, b, c, d = w.samples[i], w.samples[i + 1], slopes[i], slopes[i + 1]
        if derivative == 0:
            value = (
                (2 * u**3 - 3 * u**2 + 1) * a
                + (-2 * u**3 + 3 * u**2) * b
                + (u**3 - 2 * u**2 + u) * c
                + (u**3 - u**2) * d
            )
        elif derivative == 1:
            value = (
                (6 * u**2 - 6 * u) * a
                + (-6 * u**2 + 6 * u) * b
                + (3 * u**2 - 4 * u + 1) * c
                + (3 * u**2 - 2 * u) * d
            ) / w.dt
        else:
            value = (
                (12 * u - 6) * a + (-12 * u + 6) * b + (6 * u - 4) * c + (6 * u - 2) * d
            ) / w.dt**2
        return np.where(inside, value, 0.0)

    def require_quiet_start(self, time: float, *, derivative: int = 0) -> None:
        """Reject an initial load inconsistent with the backend's zero state."""
        value = self.evaluate(time, derivative=derivative)
        rate = (
            2 * np.pi * self.frequency
            if self.frequency is not None
            else (1 / self.width if self.width is not None else 1)
        )
        scale = abs(self.amplitude)
        if self.wavelet is not None:
            rate = 1 / self.wavelet.dt
            scale *= np.max(np.abs(self.wavelet.samples))
        if abs(value) > 1e-8 * scale * rate**derivative:
            raise ValueError(
                "Transient source must be initially quiet; increase delay or start earlier"
            )

    def to_fs(self) -> dict:
        result = {
            "kind": self.kind,
            "amplitude": float(self.amplitude),
            "delay": float(self.delay),
        }
        for key in ("frequency", "width"):
            if getattr(self, key) is not None:
                result[key] = float(getattr(self, key))
        if self.wavelet is not None:
            result.update(
                samples=self.wavelet.samples.tolist(),
                dt=self.wavelet.dt,
                t0=self.wavelet.t0,
            )
        return result

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> TimeFunction:
        data = _fields(
            data,
            {"kind", "amplitude", "delay", "frequency", "width", "samples", "dt", "t0"},
            "Time.source",
        )
        if data.get("kind") == "sampled":
            if not {"samples", "dt"} <= data.keys():
                raise ValueError("Sampled sources require samples and dt")
            data["wavelet"] = SampledWavelet(
                data.pop("samples"), dt=data.pop("dt"), t0=data.pop("t0", 0.0)
            )
        elif {"samples", "dt", "t0"} & data.keys():
            raise ValueError("samples, dt and t0 are only used by sampled sources")
        return cls(**{"kind": "constant", **data})


@dataclass(frozen=True)
class CheckpointSchedule:
    """Rank-local checkpoint cadence; zero disables checkpoint writes.

    Relative paths resolve below the result directory. Preserve every rank's
    checkpoint and PML sidecar together when copying a restart directory.
    """

    stride: int = 0
    path: str | Path = "checkpoints"

    def __post_init__(self) -> None:
        _count(self.stride, "checkpoint.stride")
        _path(self.path, "checkpoint.path")

    def to_fs(self) -> dict:
        return {"stride": int(self.stride), "path": str(self.path)}

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> CheckpointSchedule:
        return cls(**_fields(data, {"stride", "path"}, "Time.checkpoint"))


@dataclass(frozen=True)
class TimeOutput:
    """Output at selected physical times, a physical interval, or step stride.

    Specify at most one of ``times``, ``interval`` and ``stride``. The default
    is every step. Times/intervals must align with the integration grid. The
    final step is always written; ``write_initial`` additionally selects step
    zero. Use it for spatial output or an independent receiver trace schedule.
    """

    stride: int | None = None
    interval: float | None = None
    times: tuple[float, ...] | None = None
    write_initial: bool = True

    def __post_init__(self) -> None:
        if sum(v is not None for v in (self.stride, self.interval, self.times)) > 1:
            raise ValueError("Specify only one of output stride, interval or times")
        if self.stride is None and self.interval is None and self.times is None:
            object.__setattr__(self, "stride", 1)
        if not isinstance(self.write_initial, bool):
            raise TypeError("write_initial must be boolean")
        if self.stride is not None:
            _count(self.stride, "output.stride", 1)
        if self.interval is not None:
            interval = _scalar_in_units(self.interval, "s", "output.interval")
            if interval <= 0:
                raise ValueError("output.interval must be positive")
            object.__setattr__(self, "interval", interval)
        if self.times is not None:
            times = tuple(_scalar_in_units(v, "s", "output.times") for v in self.times)
            if not times or any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("output.times must be nonempty, unique and increasing")
            object.__setattr__(self, "times", times)

    def _cadence(self, time: TimeIntegration):
        """Validate alignment without allocating the complete output grid."""

        def aligned(value: float) -> int:
            n = value / time.dt
            if not math.isfinite(n) or abs(n) > 999_999_999_999:
                raise ValueError("Output step exceeds the supported range")
            if abs(n - round(n)) > 64 * np.finfo(float).eps * max(1, abs(n)):
                raise ValueError(
                    "Output times and interval must align with the integration grid"
                )
            return round(n)

        if self.times is not None:
            steps = np.asarray(
                [aligned(t - time.start) for t in self.times], dtype=np.int64
            )
            if np.any(steps < 0) or np.any(steps > time.steps):
                raise ValueError(
                    "Output times must lie within the integration interval"
                )
        else:
            stride = (
                aligned(self.interval)
                if self.interval is not None
                else (self.stride or 1)
            )
            if stride < 1:
                raise ValueError(
                    "Output interval must be at least one integration step"
                )
            steps = range(stride, time.steps + 1, stride)
        return steps

    def validate(self, time: TimeIntegration) -> None:
        """Check this schedule against an integration grid without allocating it."""
        self._cadence(time)

    def steps(self, time: TimeIntegration) -> np.ndarray:
        """Return selected integer steps, including required endpoint samples."""
        steps = self._cadence(time)
        return np.unique(
            np.concatenate((steps, [time.steps], [0] if self.write_initial else []))
        ).astype(np.int64)

    def to_fs(self) -> dict:
        payload = {"write_initial": self.write_initial}
        if self.times is not None:
            payload["times"] = list(self.times)
        elif self.interval is not None:
            payload["interval"] = float(self.interval)
        else:
            payload["stride"] = int(self.stride or 1)
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> TimeOutput:
        return cls(
            **_fields(
                data, {"stride", "interval", "times", "write_initial"}, "Time.output"
            )
        )
