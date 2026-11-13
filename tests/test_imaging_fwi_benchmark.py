"""Solver-backed acceptance test: the 1-D profile FWI benchmark.

Runs ``benchmarks/imaging/fwi_1d_profile.py`` in its fast configuration
(coarse mesh, two bands up to 4.5 Hz, four L-BFGS iterations per band) and
checks the adjoint at the starting model, the data-objective reduction and
the recovered ``vp`` profile against the original TCCS acceptance rules.

Thresholds: the TCCS layered case accepts a data-objective ratio of 0.2
after up to 100 iterations per stage and a model-error ratio of 0.85 (the
seam cases relax those to 0.8 and 1.1).  With eight iterations in total this
test asks for a ratio of 0.25 at the final band (the reference run reaches
0.05) and for the TCCS 0.85 on the ``vp`` profile error (reference 0.75).

The test needs a Sauce executable: ``FS_SAUCE_EXECUTABLE`` or
``LOCAL_SOLVER_EXECUTABLE`` in the environment, or the staged build at
``/tmp/FS_stage-imaging-merge/agent/install/fs2d_s``.  It skips otherwise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration

BENCHMARK = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "imaging" / "fwi_1d_profile.py"
)
MAX_DATA_OBJECTIVE_RATIO = 0.25
MAX_VP_ERROR_RATIO = 0.85
ADJOINT_TOLERANCE = 1.0e-3


def _benchmark():
    spec = importlib.util.spec_from_file_location("fwi_1d_profile", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def _executable(benchmark):
    path = benchmark.find_executable()
    if path is None:
        pytest.skip(
            "no Sauce executable: set FS_SAUCE_EXECUTABLE (or LOCAL_SOLVER_EXECUTABLE)"
        )
    return path


@pytest.mark.timeout(1800)
def test_fwi_1d_profile_benchmark(tmp_path):
    benchmark = _benchmark()
    executable = _executable(benchmark)

    case = benchmark.build_case(
        tmp_path / "project", executable=executable, coarse=True
    )
    problem = benchmark.build_problem(case, workdir=tmp_path / "fwi")
    assert problem.space.blocks == ("model.vp", "model.rho")

    # <J dv, r>_Re == <dv, J^H r> and H dv == J^H W J dv at the starting model
    report = problem.check(tolerance=ADJOINT_TOLERANCE, taylor=False)
    assert report["adjoint"]["passed"], report["adjoint"]
    assert report["normal"]["passed"], report["normal"]

    result = benchmark.run_fwi(
        case,
        workdir=tmp_path / "fwi",
        stages=benchmark.default_stages(case, iterations=4),
    )
    assert result.problem is problem
    assert len(result.stages) == 2
    assert all(stage.iterations >= 1 for stage in result.stages)

    metrics = benchmark.evaluate(result, case)
    assert np.isfinite(metrics["data_objective_ratio"])
    assert metrics["data_objective_ratio"] <= MAX_DATA_OBJECTIVE_RATIO, metrics
    assert metrics["vp"]["error_ratio"] <= MAX_VP_ERROR_RATIO, metrics["vp"]
    assert metrics["rho"]["error_ratio"] == pytest.approx(1.0)  # rho stays inactive
    assert all(
        stage["final_data_loss"] <= stage["initial_data_loss"]
        for stage in metrics["stages"]
    ), metrics["stages"]
    assert metrics["timing"]["fwi_seconds"] > 0.0
