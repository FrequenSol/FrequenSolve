"""Validated physical-time integration and analytic source functions.

Times are seconds, frequencies are hertz, and source amplitudes multiply the
physical strengths in the simulation acquisition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "Newmark",
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
class TimeIntegration:
    """One fixed physical interval; ``dt`` must divide ``end - start``.

    ``steps`` counts advances, excluding the initial state. ``times`` includes
    both endpoints. No timestep or mesh bandwidth is inferred from a wavelet.
    """

    start: float
    end: float
    dt: float
    method: Newmark = Newmark()

    def __post_init__(self) -> None:
        start, end, dt = (
            _number(getattr(self, key), key) for key in ("start", "end", "dt")
        )
        if end <= start or dt <= 0:
            raise ValueError("TimeIntegration requires end > start and dt > 0")
        n = (end - start) / dt
        if not math.isfinite(n) or not 1 <= n <= 999_999_999_999:
            raise ValueError(
                "TimeIntegration step count must fit the 12-digit output index"
            )
        if abs(n - round(n)) > 64 * np.finfo(float).eps * max(1, abs(n)):
            raise ValueError("dt must divide the requested time interval")
        if not isinstance(self.method, Newmark):
            raise TypeError("Transient wave integration requires Newmark()")

    @property
    def steps(self) -> int:
        return round((self.end - self.start) / self.dt)

    @property
    def times(self) -> np.ndarray:
        return self.start + np.arange(self.steps + 1) * self.dt

    def to_fs(self) -> dict:
        return {
            "schema": "fs-time-domain-1",
            "start": float(self.start),
            "end": float(self.end),
            "time_step": float(self.dt),
            "method": "newmark",
            "newmark": self.method.to_fs(),
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> TimeIntegration:
        data = _fields(
            data,
            {"schema", "start", "end", "time_step", "method", "newmark"},
            "TimeIntegration",
        )
        if (
            data.get("schema", "fs-time-domain-1") != "fs-time-domain-1"
            or data.get("method", "newmark") != "newmark"
        ):
            raise ValueError("Expected fs-time-domain-1 with method newmark")
        return cls(
            data["start"],
            data["end"],
            data["time_step"],
            Newmark.from_fs(data.get("newmark", {})),
        )


@dataclass(frozen=True)
class TimeFunction:
    """Shared analytic source multiplier evaluated in physical seconds.

    Acoustic forces load its value, volume rates its first derivative, and
    isotropic moments its second derivative times local compressibility.
    Elastic forces and moments load its value. Sampled or per-source signatures
    are not part of the current backend contract.
    """

    kind: str
    amplitude: float = 1.0
    delay: float = 0.0
    frequency: float | None = None
    width: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"constant", "ricker", "gaussian", "sine", "step"}:
            raise ValueError(
                "TimeFunction kind must be constant, ricker, gaussian, sine or step"
            )
        _number(self.amplitude, "amplitude")
        _number(self.delay, "delay")
        for key, needed in (
            ("frequency", self.kind in {"ricker", "sine"}),
            ("width", self.kind == "gaussian"),
        ):
            value = getattr(self, key)
            if needed and (value is None or _number(value, key) <= 0):
                raise ValueError(f"{self.kind} requires positive {key}")
            if not needed and value is not None:
                raise ValueError(f"{key} is not used by {self.kind}")

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
        """Evaluate the value or analytic first/second derivative."""
        derivative = _count(derivative, "derivative")
        if derivative > 2:
            raise ValueError("Only derivatives 0, 1 and 2 are supported")
        t = np.asarray(time, dtype=float) - self.delay
        if not np.isfinite(t).all():
            raise ValueError("time must be finite")
        if self.kind == "constant":
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

    def require_quiet_start(self, time: float, *, derivative: int = 0) -> None:
        """Reject an initial load inconsistent with the backend's zero state."""
        value = self.evaluate(time, derivative=derivative)
        rate = (
            2 * np.pi * self.frequency
            if self.frequency is not None
            else (1 / self.width if self.width is not None else 1)
        )
        if abs(value) > 1e-8 * abs(self.amplitude) * rate**derivative:
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
        return result

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> TimeFunction:
        data = _fields(
            data, {"kind", "amplitude", "delay", "frequency", "width"}, "Time.source"
        )
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
    zero. This shared schedule applies to receiver and wavefield outputs.
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
        if self.interval is not None and _number(self.interval, "output.interval") <= 0:
            raise ValueError("output.interval must be positive")
        if self.times is not None:
            times = tuple(_number(v, "output.times") for v in self.times)
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
