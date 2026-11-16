"""Fixed weighted sums of standard multi-frequency imaging objectives."""

from typing import Any

import numpy as np

from .controls import ControlVector

__all__ = ["WeightedObjective"]


class WeightedObjective:
    """Combine matched material-control views without custom solver launches.

    Each component uses its standard multi-frequency job, including shared
    receiver probes. Weights apply equally to values, covectors and diagonals.
    Native regularization is applied once by FWI, outside this data sum.
    """

    def __init__(self, problems: Any, weights: Any) -> None:
        self.problems = tuple(problems)
        self.weights = tuple(float(w) for w in weights)
        if (
            not self.problems
            or len(self.problems) != len(self.weights)
            or not np.isfinite(self.weights).all()
            or min(self.weights) < 0
            or sum(self.weights) <= 0
        ):
            raise ValueError("Expected matched objectives and nonnegative weights")

    def __getattr__(self, name: Any) -> Any:
        return getattr(self.problems[0], name)

    @property
    def state(self) -> Any:
        return self.problems[0].state

    @state.setter
    def state(self, value: Any) -> Any:
        for problem in self.problems:
            problem.state = value

    def identity(self) -> Any:
        return {
            "weighted_objectives": [p.identity() for p in self.problems],
            "weights": list(self.weights),
        }

    def restrict(self, **kwargs: Any) -> Any:
        if any(k in kwargs for k in ("misfit", "loss", "kernel_derivative")):
            raise ValueError("Restrict objective components before composing them")
        return type(self)([p.restrict(**kwargs) for p in self.problems], self.weights)

    def linearize(
        self, v: Any = None, *, gradient: Any = True, receiver_diagonal: Any = None
    ) -> Any:
        point = self.vector() if v is None else v
        parts = [
            p.linearize(point, gradient=gradient, receiver_diagonal=receiver_diagonal)
            for p in self.problems
        ]
        return _WeightedLinearization(self, parts)

    def value(self, v: Any = None) -> Any:
        return self.linearize(v, gradient=False).value

    def gradient(self, v: Any = None) -> Any:
        return self.linearize(v).gradient


class _WeightedLinearization:
    def __init__(self, problem: Any, parts: Any) -> None:
        self.problem, self.parts = problem, tuple(parts)
        first = parts[0]
        for part in parts[1:]:
            if (
                not first.space.equivalent(part.space)
                or not np.array_equal(first.point.values, part.point.values)
                or list(first.frequencies) != list(part.frequencies)
            ):
                raise ValueError(
                    "Weighted objectives must share model, basis and frequencies"
                )
        self.value = sum(w * p.value for w, p in zip(problem.weights, parts))
        self.report = {
            f"component_{i}/{key}": w * value
            for i, (w, p) in enumerate(zip(problem.weights, parts))
            for key, value in p.report.items()
        }
        self.gradient = (
            None
            if any(p.gradient is None for p in parts)
            else self._sum([p.gradient for p in parts])
        )
        if self.gradient is not None:
            self.gradient._field_factory = self.field

    def __getattr__(self, name: Any) -> Any:
        if name in (
            "normal",
            "jacobian",
            "apply_normal",
            "apply_normal_batch",
            "objective_residual",
        ):
            raise NotImplementedError(
                "WeightedObjective currently supports first-order FWI"
            )
        return getattr(self.parts[0], name)

    def _sum(self, vectors: Any) -> Any:
        return ControlVector(
            sum(w * v.values for w, v in zip(self.problem.weights, vectors)), self.space
        )

    @property
    def receiver_diagonal(self) -> Any:
        return self._sum([p.receiver_diagonal for p in self.parts])

    @property
    def jobs(self) -> Any:
        return tuple(j for p in self.parts for j in p.jobs)

    def field(self, vector: Any = None, **kwargs: Any) -> Any:
        return self.parts[0].field(
            self.gradient if vector is None else vector, **kwargs
        )

    def trial(self, v: Any, *, gradient: Any = False) -> Any:
        return self.problem.linearize(v, gradient=gradient)
