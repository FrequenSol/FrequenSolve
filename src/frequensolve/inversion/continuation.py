"""Explicit frequency and Laplace-domain continuation schedules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "ContinuationResult",
    "ContinuationSchedule",
    "ContinuationStage",
    "run_continuation",
]


def _frequencies(values: Sequence[complex]) -> tuple[complex, ...]:
    """Normalize one ordered, non-empty complex-frequency collection."""

    frequencies = tuple(complex(value) for value in values)
    if not frequencies:
        raise ValueError("a continuation stage requires at least one frequency")
    for value in frequencies:
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError("continuation frequencies must be finite")
        if value.real < 0.0:
            raise ValueError("continuation frequencies require nonnegative real parts")
    if len(set(frequencies)) != len(frequencies):
        raise ValueError("continuation frequencies must be unique within a stage")
    return frequencies


@dataclass(frozen=True)
class ContinuationStage:
    """One fixed objective in a frequency/Laplace continuation sequence."""

    name: str
    frequencies: tuple[complex, ...]
    max_iterations: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("continuation stage name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "frequencies", _frequencies(self.frequencies))
        if self.max_iterations is not None and int(self.max_iterations) < 1:
            raise ValueError("stage max_iterations must be positive")
        object.__setattr__(
            self,
            "max_iterations",
            None if self.max_iterations is None else int(self.max_iterations),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def maximum_real_frequency(self) -> float:
        """Return the highest ordinary frequency in the stage."""

        return max(value.real for value in self.frequencies)

    @property
    def maximum_laplace_damping(self) -> float:
        """Return the largest absolute imaginary frequency in the stage."""

        return max(abs(value.imag) for value in self.frequencies)

    def to_fs(self) -> dict[str, Any]:
        """Serialize the stage without relying on complex JSON values."""

        payload: dict[str, Any] = {
            "name": self.name,
            "frequencies_hz": [[value.real, value.imag] for value in self.frequencies],
        }
        if self.max_iterations is not None:
            payload["max_iterations"] = self.max_iterations
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ContinuationStage":
        """Deserialize one explicit continuation stage."""

        values = data.get("frequencies_hz", data.get("frequencies"))
        if values is None:
            raise ValueError("continuation stage requires frequencies_hz")
        frequencies = []
        for value in values:
            if np.isscalar(value):
                frequencies.append(complex(float(np.asarray(value)), 0.0))
            else:
                pair = list(value)
                if len(pair) != 2:
                    raise ValueError(
                        "each continuation frequency must be scalar or [real, imaginary]"
                    )
                frequencies.append(complex(float(pair[0]), float(pair[1])))
        return cls(
            name=data["name"],
            frequencies=tuple(frequencies),
            max_iterations=data.get("max_iterations"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class ContinuationSchedule:
    """Ordered fixed-objective stages for multiscale inversion."""

    stages: tuple[ContinuationStage, ...]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages or not all(
            isinstance(stage, ContinuationStage) for stage in stages
        ):
            raise ValueError("a continuation schedule requires stages")
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ValueError("continuation stage names must be unique")
        object.__setattr__(self, "stages", stages)

    @property
    def frequencies(self) -> tuple[complex, ...]:
        """Return the first-seen union needed to synthesize or load data once."""

        ordered: list[complex] = []
        seen: set[complex] = set()
        for stage in self.stages:
            for frequency in stage.frequencies:
                if frequency not in seen:
                    ordered.append(frequency)
                    seen.add(frequency)
        return tuple(ordered)

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "ContinuationSchedule":
        """Deserialize explicit stages or Cartesian frequency/Laplace bands."""

        stages = data.get("stages")
        bands = data.get("bands")
        if stages is not None and bands is not None:
            raise ValueError("continuation requires either stages or bands, not both")
        if bands is not None:
            return cls.frequency_laplace_bands(
                bands,
                damping_sign=float(data.get("damping_sign", -1.0)),
            )
        if stages is None:
            raise ValueError("continuation schedule requires stages or bands")
        return cls(tuple(ContinuationStage.from_fs(stage) for stage in stages))

    @classmethod
    def frequency_laplace_bands(
        cls,
        bands: Sequence[Mapping[str, Any]],
        *,
        damping_sign: float = -1.0,
    ) -> "ContinuationSchedule":
        """Expand frequency bands across complete Laplace-damping sequences.

        Each band supplies ordinary ``frequencies_hz`` and nonnegative,
        nonincreasing ``laplace_damping_hz``. One fixed-objective stage is
        emitted per damping value. By default it contains every frequency in
        the band, forming a Cartesian frequency/Laplace schedule. An optional
        nondecreasing ``frequency_counts`` sequence selects progressively
        longer frequency prefixes for a cheaper triangular schedule. A
        terminal damping of zero is an ordinary final continuation sample
        rather than a separate real-data phase.
        """

        authored_bands = tuple(bands)
        if not authored_bands:
            raise ValueError("frequency/Laplace continuation requires bands")
        if not np.isfinite(damping_sign) or damping_sign == 0.0:
            raise ValueError("damping_sign must be finite and nonzero")
        stages = []
        band_names = []
        for band in authored_bands:
            name = str(band["name"]).strip()
            if not name:
                raise ValueError("continuation band name cannot be empty")
            band_names.append(name)
            frequencies = tuple(float(value) for value in band["frequencies_hz"])
            if not frequencies:
                raise ValueError(f"continuation band {name!r} requires frequencies")
            if len(set(frequencies)) != len(frequencies):
                raise ValueError(
                    f"continuation band {name!r} frequencies must be unique"
                )
            if any(not np.isfinite(value) or value < 0.0 for value in frequencies):
                raise ValueError(
                    f"continuation band {name!r} frequencies must be finite and "
                    "nonnegative"
                )
            damping = tuple(float(value) for value in band["laplace_damping_hz"])
            if not damping:
                raise ValueError(
                    f"continuation band {name!r} requires Laplace damping values"
                )
            if any(not np.isfinite(value) or value < 0.0 for value in damping):
                raise ValueError(
                    f"continuation band {name!r} Laplace damping must be finite "
                    "and nonnegative"
                )
            if any(
                damping[index + 1] > damping[index] for index in range(len(damping) - 1)
            ):
                raise ValueError(
                    f"continuation band {name!r} Laplace damping must not increase"
                )
            if damping[-1] != 0.0:
                raise ValueError(
                    f"continuation band {name!r} must end at zero Laplace damping"
                )
            authored_counts = band.get("frequency_counts")
            if authored_counts is None:
                frequency_counts = (len(frequencies),) * len(damping)
            else:
                raw_counts = tuple(authored_counts)
                if any(
                    isinstance(value, (bool, np.bool_))
                    or not np.isscalar(value)
                    or not np.isfinite(float(np.asarray(value)))
                    or float(np.asarray(value)) != int(np.asarray(value))
                    for value in raw_counts
                ):
                    raise ValueError(
                        f"continuation band {name!r} frequency counts must be "
                        "finite integers"
                    )
                frequency_counts = tuple(int(value) for value in raw_counts)
                if len(frequency_counts) != len(damping):
                    raise ValueError(
                        f"continuation band {name!r} frequency counts must match "
                        "Laplace damping values"
                    )
                if any(
                    value < 1 or value > len(frequencies) for value in frequency_counts
                ):
                    raise ValueError(
                        f"continuation band {name!r} frequency counts must lie "
                        "within the authored band"
                    )
                if any(
                    frequency_counts[index + 1] < frequency_counts[index]
                    for index in range(len(frequency_counts) - 1)
                ):
                    raise ValueError(
                        f"continuation band {name!r} frequency counts must not decrease"
                    )
                if frequency_counts[-1] != len(frequencies):
                    raise ValueError(
                        f"continuation band {name!r} final frequency count must "
                        "include the complete band"
                    )
            max_iterations = band.get("max_iterations")
            metadata = dict(band.get("metadata", {}))
            authored_stage_metadata = band.get("laplace_metadata")
            if authored_stage_metadata is None:
                laplace_metadata: tuple[Mapping[str, Any], ...] = ({},) * len(damping)
            else:
                laplace_metadata = tuple(authored_stage_metadata)
                if len(laplace_metadata) != len(damping) or any(
                    not isinstance(value, Mapping) for value in laplace_metadata
                ):
                    raise ValueError(
                        f"continuation band {name!r} Laplace metadata must contain "
                        "one mapping per damping value"
                    )
            for index, (magnitude, frequency_count) in enumerate(
                zip(damping, frequency_counts)
            ):
                stage_metadata = dict(metadata)
                stage_metadata.update(laplace_metadata[index])
                stage_metadata.update(
                    {
                        "continuation_band": name,
                        "laplace_index": index,
                        "laplace_count": len(damping),
                        "laplace_damping_hz": magnitude,
                        "frequency_count": frequency_count,
                    }
                )
                stages.append(
                    ContinuationStage(
                        f"{name}__laplace_{index:02d}",
                        tuple(
                            complex(value, damping_sign * magnitude)
                            for value in frequencies[:frequency_count]
                        ),
                        max_iterations=max_iterations,
                        metadata=stage_metadata,
                    )
                )
        if len(set(band_names)) != len(band_names):
            raise ValueError("continuation band names must be unique")
        return cls(tuple(stages))

    @classmethod
    def joint_frequency_laplace(
        cls,
        frequency_bands_hz: Sequence[Sequence[float]],
        laplace_damping_hz: Sequence[float],
        *,
        max_iterations: Optional[Sequence[Optional[int]]] = None,
        names: Optional[Sequence[str]] = None,
        damping_sign: float = -1.0,
    ) -> "ContinuationSchedule":
        """Build stages that add frequencies while reducing Laplace damping.

        ``frequency_bands_hz[i]`` defines the complete real-frequency set for
        stage ``i``. The corresponding nonnegative damping magnitude is placed
        on the imaginary axis using ``damping_sign``; Sauce's current transform
        convention uses ``damping_sign=-1``.
        """

        bands = tuple(
            tuple(float(value) for value in band) for band in frequency_bands_hz
        )
        damping = tuple(float(value) for value in laplace_damping_hz)
        if not bands or len(bands) != len(damping):
            raise ValueError(
                "frequency bands and Laplace damping must have equal length"
            )
        if not np.isfinite(damping_sign) or damping_sign == 0.0:
            raise ValueError("damping_sign must be finite and nonzero")
        if any(not np.isfinite(value) or value < 0.0 for value in damping):
            raise ValueError("Laplace damping values must be finite and nonnegative")
        if any(
            damping[index + 1] > damping[index] for index in range(len(damping) - 1)
        ):
            raise ValueError("Laplace damping must not increase between stages")
        for index in range(len(bands) - 1):
            if not set(bands[index]).issubset(bands[index + 1]):
                raise ValueError(
                    "frequency continuation stages must retain earlier frequencies"
                )
        if names is None:
            stage_names = tuple(f"stage_{index + 1:02d}" for index in range(len(bands)))
        else:
            stage_names = tuple(str(value) for value in names)
            if len(stage_names) != len(bands):
                raise ValueError("continuation names must match the number of stages")
        if max_iterations is None:
            iterations: tuple[Optional[int], ...] = (None,) * len(bands)
        else:
            iterations = tuple(max_iterations)
            if len(iterations) != len(bands):
                raise ValueError("stage iteration limits must match the stages")
        stages = []
        for name, band, magnitude, limit in zip(
            stage_names, bands, damping, iterations
        ):
            stages.append(
                ContinuationStage(
                    name,
                    tuple(complex(value, damping_sign * magnitude) for value in band),
                    max_iterations=limit,
                )
            )
        return cls(tuple(stages))

    def to_fs(self) -> dict[str, Any]:
        """Serialize the explicit schedule."""

        return {"stages": [stage.to_fs() for stage in self.stages]}


@dataclass(frozen=True)
class ContinuationResult:
    """Terminal model and each stage's optimizer result."""

    model: np.ndarray
    stage_results: tuple[Any, ...]
    stage_initial_models: tuple[np.ndarray, ...] = ()


def run_continuation(
    schedule: ContinuationSchedule,
    initial_model: Sequence[float],
    solve_stage: Callable[[ContinuationStage, np.ndarray], Any],
    *,
    callback: Optional[Callable[[ContinuationStage, Any], None]] = None,
    transition: Optional[
        Callable[[ContinuationStage, ContinuationStage, np.ndarray], Sequence[float]]
    ] = None,
) -> ContinuationResult:
    """Warm-start every continuation stage from the preceding stage result.

    A stage result may expose its terminal model as either ``model`` or ``x``.
    Iteration-limit termination is intentionally left to ``solve_stage`` so a
    caller can continue after useful, deliberately inexact stage solves.

    ``transition(previous_stage, next_stage, accepted_model)`` is called only
    between stages. It may rebuild the stage-local backend/control space and
    return a different-sized initial vector. Transfer the represented physical
    model (or its latent field when references and transforms agree), not vector
    indices. The next ``solve_stage`` must rebuild its bounds, regularization,
    caches and optimizer history for that space. Dimensions may never change
    inside a stage solve. Without this hook the fixed-space behavior is unchanged.

    Results and ``stage_initial_models`` retain each stage's local vector shape;
    ``model`` belongs to the final stage only. Hooks receive copies of accepted
    vectors so an in-place transfer cannot modify earlier optimizer results.
    """

    if not isinstance(schedule, ContinuationSchedule):
        raise TypeError("schedule must be ContinuationSchedule")
    if not callable(solve_stage):
        raise TypeError("solve_stage must be callable")
    if callback is not None and not callable(callback):
        raise TypeError("continuation callback must be callable")
    if transition is not None and not callable(transition):
        raise TypeError("continuation transition must be callable")
    model = np.asarray(initial_model)
    if np.iscomplexobj(model):
        raise ValueError("continuation models must be real-valued")
    model = np.asarray(model, dtype=np.float64).reshape(-1)
    if model.size < 1 or not np.all(np.isfinite(model)):
        raise ValueError("initial continuation model must be a finite vector")
    results = []
    initial_models = []
    for index, stage in enumerate(schedule.stages):
        if index and transition is not None:
            transferred = np.asarray(
                transition(
                    schedule.stages[index - 1], stage, np.array(model, copy=True)
                )
            )
            if np.iscomplexobj(transferred):
                raise ValueError("transferred continuation models must be real-valued")
            model = np.asarray(transferred, dtype=np.float64).reshape(-1)
            if model.size < 1 or not np.all(np.isfinite(model)):
                raise ValueError(
                    "transferred continuation model must be a finite vector"
                )
        initial_models.append(np.array(model, copy=True))
        result = solve_stage(stage, np.array(model, copy=True))
        terminal = getattr(result, "model", getattr(result, "x", None))
        if terminal is None:
            raise TypeError("stage results must expose model or x")
        terminal = np.asarray(terminal)
        if np.iscomplexobj(terminal):
            raise ValueError("stage result models must be real-valued")
        terminal = np.asarray(terminal, dtype=np.float64).reshape(-1)
        if terminal.shape != model.shape or not np.all(np.isfinite(terminal)):
            raise ValueError(
                "stage result model is incompatible with the control space"
            )
        model = np.array(terminal, copy=True)
        results.append(result)
        if callback is not None:
            callback(stage, result)
    return ContinuationResult(model, tuple(results), tuple(initial_models))
