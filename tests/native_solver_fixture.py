"""Pair an identified solver with a source checkout during native tests."""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from frequensolve.solver import SolverCompatibilityWarning, query_local_solver_identity


@contextmanager
def native_candidate_solver(solver: Path) -> Iterator[None]:
    identity = query_local_solver_identity(solver)
    if identity.identity is None:
        pytest.fail(
            f"Native integration requires a valid solver identity: {identity.error}"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "default",
            message=(
                r"^Solver compatibility warning\n\nFrequenSolve 0\+unreleased "
                r"does not contain a released preferred Solver declaration\."
            ),
            category=SolverCompatibilityWarning,
        )
        yield
