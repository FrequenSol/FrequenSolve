"""Workflow results: per-stage summaries and the complete FWI result.

:class:`StageResult` summarizes one continuation stage solved by
:class:`~frequensolve.imaging.workflows.FWI`; :class:`FWIResult` bundles the
final :class:`~frequensolve.imaging.controls.ControlState`, the optimization
history, the stage summaries and the checkpoint path, and can be saved to a
directory and loaded back onto a problem.  The artifact readers that the
design spec lists under ``results`` (:class:`ImageSet`,
:class:`ObjectiveReport`, :class:`ExtensionSolveReport`) are re-exported from
:mod:`frequensolve.imaging._artifacts`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from frequensolve.imaging._artifacts import (
    ExtensionSolveReport,
    ImageSet,
    ObjectiveReport,
)
from frequensolve.imaging.controls import ControlSpace, ControlState, ControlVector
from frequensolve.inversion.history import LossTerms, OptimizationHistory
from frequensolve.util.atomic import atomic_write_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from frequensolve.imaging.problem import ImagingProblem

__all__ = [
    "ExtensionSolveReport",
    "FWIResult",
    "ImageSet",
    "ObjectiveReport",
    "StageResult",
]

_RESULT_SCHEMA = "fs-imaging-fwi-result-1"


def _frequency_pairs(values: Sequence[Any]) -> list:
    return [[complex(v).real, complex(v).imag] for v in values]


def _frequencies_from_pairs(values: Sequence[Any]) -> Tuple[Any, ...]:
    out = []
    for pair in values:
        value = complex(pair[0], pair[1])
        out.append(value.real if value.imag == 0.0 else value)
    return tuple(out)


@dataclass(frozen=True)
class StageResult:
    """Summary of one solved (or skipped) continuation stage.

    Attributes:
        index: Position of the stage in the FWI stage list.
        name: Stage label.
        frequencies: Frequencies of the stage view.
        active: Qualified active block names of the stage view.
        iterations: Accepted optimizer iterations performed in this run.
        stage_iteration: Accepted iterations of the stage in total, including
            iterations restored from a checkpoint.
        success: Optimizer success flag (``True`` for a skipped stage).
        status: Optimizer status code.
        message: Optimizer termination message.
        initial_loss: Loss at the stage's starting point.
        final_loss: Loss at the accepted terminal point.
        evaluations: Objective evaluations of this run.
        linearizations: Sauce ``linearize`` jobs this stage submitted.
        resumed: Whether the stage started from a checkpoint.
        skipped: Whether the stage was completed by an earlier run.
        vector: Terminal vector on the stage space (``None`` after
            :meth:`FWIResult.load`).
        space: The stage's control space with adopted support masks.
        metrics: Scalar diagnostics (optimizer kind, CG iterations, ...).
    """

    index: int
    name: str
    frequencies: Tuple[Any, ...]
    active: Tuple[str, ...]
    iterations: int
    stage_iteration: int
    success: bool
    status: int
    message: str
    initial_loss: LossTerms
    final_loss: LossTerms
    evaluations: int = 0
    linearizations: int = 0
    resumed: bool = False
    skipped: bool = False
    vector: Optional[ControlVector] = None
    space: Optional[ControlSpace] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "frequencies", tuple(self.frequencies))
        object.__setattr__(self, "active", tuple(str(v) for v in self.active))
        object.__setattr__(self, "iterations", int(self.iterations))
        object.__setattr__(self, "stage_iteration", int(self.stage_iteration))
        object.__setattr__(self, "status", int(self.status))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "metrics", dict(self.metrics))
        if not isinstance(self.initial_loss, LossTerms) or not isinstance(
            self.final_loss, LossTerms
        ):
            raise TypeError("stage losses must be LossTerms")

    @property
    def objective(self) -> float:
        """Return the terminal total objective."""

        return self.final_loss.total

    @property
    def reduction(self) -> float:
        """Return ``final / initial`` total objective ratio."""

        initial = self.initial_loss.total
        if initial <= 0.0:
            return 1.0
        return self.final_loss.total / initial

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the scalar summary (vector and space are not stored)."""

        return {
            "index": self.index,
            "name": self.name,
            "frequencies_hz": _frequency_pairs(self.frequencies),
            "active": list(self.active),
            "iterations": self.iterations,
            "stage_iteration": self.stage_iteration,
            "success": bool(self.success),
            "status": self.status,
            "message": self.message,
            "initial_loss": self.initial_loss.to_fs(),
            "final_loss": self.final_loss.to_fs(),
            "evaluations": self.evaluations,
            "linearizations": self.linearizations,
            "resumed": bool(self.resumed),
            "skipped": bool(self.skipped),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_fs(cls, data: Mapping[str, Any]) -> "StageResult":
        """Deserialize a stage summary."""

        return cls(
            index=data["index"],
            name=data["name"],
            frequencies=_frequencies_from_pairs(data["frequencies_hz"]),
            active=tuple(data["active"]),
            iterations=data["iterations"],
            stage_iteration=data["stage_iteration"],
            success=bool(data["success"]),
            status=data["status"],
            message=data["message"],
            initial_loss=LossTerms.from_fs(data["initial_loss"]),
            final_loss=LossTerms.from_fs(data["final_loss"]),
            evaluations=data.get("evaluations", 0),
            linearizations=data.get("linearizations", 0),
            resumed=bool(data.get("resumed", False)),
            skipped=bool(data.get("skipped", False)),
            metrics=data.get("metrics", {}),
        )


@dataclass
class FWIResult:
    """Result of :meth:`frequensolve.imaging.workflows.FWI.run`.

    Attributes:
        state: The final complete :class:`ControlState`.
        history: The optimization history (evaluations and iterations of
            every stage, with stage metrics).
        stages: One :class:`StageResult` per stage in order.
        checkpoint: Path of the last written checkpoint, if any.
        problem: The problem the run belonged to (used by :attr:`simulation`
            and :meth:`vector`); ``None`` after :meth:`load` without one.
    """

    state: ControlState
    history: OptimizationHistory
    stages: Tuple[StageResult, ...] = ()
    checkpoint: Optional[Path] = None
    problem: Optional["ImagingProblem"] = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, ControlState):
            raise TypeError("FWIResult.state must be a ControlState")
        if not isinstance(self.history, OptimizationHistory):
            raise TypeError("FWIResult.history must be an OptimizationHistory")
        self.stages = tuple(self.stages)
        if self.checkpoint is not None:
            self.checkpoint = Path(self.checkpoint)

    # -- convenience ----------------------------------------------------------

    @property
    def final(self) -> Optional[StageResult]:
        """Return the last stage summary (``None`` for an empty run)."""

        return self.stages[-1] if self.stages else None

    @property
    def success(self) -> bool:
        """Return whether every stage terminated successfully."""

        return all(stage.success for stage in self.stages)

    @property
    def loss(self) -> Optional[LossTerms]:
        """Return the final stage's terminal loss."""

        final = self.final
        return None if final is None else final.final_loss

    @property
    def simulation(self) -> Any:
        """Return ``problem.simulation_at(state)`` (needs :attr:`problem`)."""

        if self.problem is None:
            raise ValueError("FWIResult has no problem; pass one to load()")
        return self.problem.simulation_at(self.state)

    def vector(self, space: Optional[ControlSpace] = None) -> ControlVector:
        """Return the state's active slice on ``space`` (default: the problem's)."""

        if space is None:
            if self.problem is None:
                return self.state.vector()
            space = self.problem.space
        return self.state.vector(space)

    # -- persistence ----------------------------------------------------------

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the scalar summary (state and history are separate files)."""

        return {
            "schema": _RESULT_SCHEMA,
            "problem": None if self.problem is None else self.problem.name,
            "blocks": list(self.state.space.blocks),
            "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
            "stages": [stage.to_fs() for stage in self.stages],
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Write ``state.h5``, ``history.json`` and ``result.json`` into ``path``.

        ``path`` is a directory (created when missing).  Returns it.
        """

        directory = Path(path).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        self.state.save(directory / "state.h5")
        atomic_write_json(
            directory / "history.json",
            self.history.to_fs(),
            indent=2,
            sort_keys=True,
            trailing_newline=True,
        )
        atomic_write_json(
            directory / "result.json",
            self.to_fs(),
            indent=2,
            sort_keys=True,
            trailing_newline=True,
        )
        return directory

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        problem: Optional["ImagingProblem"] = None,
        *,
        space: Optional[ControlSpace] = None,
    ) -> "FWIResult":
        """Load a saved result onto ``problem.full_space`` (or ``space``)."""

        directory = Path(path).expanduser()
        if space is None:
            if problem is None:
                raise ValueError("FWIResult.load needs a problem or a space")
            space = problem.full_space.without_support()
        data = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if data.get("schema") != _RESULT_SCHEMA:
            raise ValueError("unsupported FWI result schema")
        if list(data.get("blocks", [])) != list(space.blocks):
            raise ValueError(
                f"saved result covers blocks {data.get('blocks')}; the space has "
                f"{list(space.blocks)}"
            )
        state = ControlState.load(directory / "state.h5", space)
        history = OptimizationHistory.load(directory / "history.json")
        stages = tuple(StageResult.from_fs(item) for item in data.get("stages", []))
        checkpoint = data.get("checkpoint")
        return cls(
            state=state,
            history=history,
            stages=stages,
            checkpoint=None if checkpoint is None else Path(checkpoint),
            problem=problem,
        )

    def __repr__(self) -> str:
        final = self.final
        objective = "?" if final is None else f"{final.final_loss.total:.6g}"
        return (
            f"FWIResult(stages={len(self.stages)}, objective={objective}, "
            f"iterations={self.history.iteration_count})"
        )
