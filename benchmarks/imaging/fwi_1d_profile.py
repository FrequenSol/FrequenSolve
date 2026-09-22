"""1-D sediment profile FWI benchmark on the ``frequensolve.imaging`` API.

Port of the TCCS single-source 1-D control benchmark
(``TCCS/1D_test/run_benchmark.py``, acoustic pressure case) reduced to its
essence: a 2-D acoustic water/sediment model whose truth is a layered
sediment ``vp``/``rho`` column with a low-velocity notch, a smooth starting
model, one scalar source and a hydrophone line, log-transformed hat profiles
on the sediment (``vp`` bounded as in TCCS), a Huber misfit with a
near-offset taper, a first-order Tikhonov penalty, projected L-BFGS with an
RMS step cap and a stochastic Gauss-Newton diagonal preconditioner, and
frequency continuation over a few bands.  Acceptance follows the original: the data objective must drop by a
set ratio and the recovered profile must be closer to the truth than the
starting model.

Run it with::

    python benchmarks/imaging/fwi_1d_profile.py --fast --workdir /tmp/fwi_1d

The Sauce executable comes from ``--executable``, ``FS_SAUCE_EXECUTABLE``,
``LOCAL_SOLVER_EXECUTABLE`` or the staged build in :data:`DEFAULT_EXECUTABLE`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from frequensolve import imaging as im
from frequensolve.mesh import BoundaryCondition
from frequensolve.model.layered import LayeredModel
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.seismic import Acquisition, ReceiverNode
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig

DEFAULT_EXECUTABLE = Path("/tmp/FS_stage-imaging-merge/agent/install/fs2d_s")

# Geometry in km: a 100 m water column over a 600 m sediment column.
WIDTH = 2.0
WATER_DEPTH = 0.1
MODEL_DEPTH = 0.7
SOURCE = (0.1, 0.02)
RECEIVER_DEPTH = 0.05
RECEIVER_X = np.linspace(0.2, 1.9, 35)

# Truth sediment column by depth below the seabed (km): (bottom, vp km/s,
# rho g/cm^3).  The third layer is the low-velocity notch.
TRUTH_LAYERS = (
    (0.10, 1.70, 1.80),
    (0.25, 2.00, 1.95),
    (0.35, 1.80, 1.85),
    (0.50, 2.30, 2.10),
    (0.60, 2.60, 2.20),
)
INITIAL_SMOOTHING = 0.15  # Gaussian sigma (km) that turns the truth into the start
PROFILE_SAMPLES = 301

# Inversion settings (the TCCS values, except where noted in ``run_fwi``).
# TCCS bounds the log ``vp`` coefficients to (-0.35, 0.25) about the starting
# model (``benchmark_acoustic_pressure_vp_*_seam*.json``).  ``limits=`` takes
# physical values, so the port uses the envelope of those boxes over the
# starting profile (:func:`vp_limits`); the per-node log bounds
# ``log(limit / vp0(node))`` then contain the TCCS box at every node and equal
# it where ``vp0`` is extreme.
VP_LOG_BOUNDS = (-0.35, 0.25)
HUBER_DELTA = 1.345
OFFSET_TAPER = (0.1, 0.25)  # km, near-offset raised cosine
# Penalties are scale free (0.5 * alpha * int |dc/dxi|^2 over the unit
# sediment column, xi = depth / thickness) and the Huber data term is
# normalized to O(1), so an O(1e-2) alpha is a real but gentle smoothness
# prior on the log coefficients.
TIKHONOV_ALPHA = 1.0e-2

# Continuation bands (Hz) and the frequencies they need.
FAST_BANDS = [[2.0, 3.0], [3.0, 4.5]]
FULL_BANDS = [[2.0, 3.0], [3.0, 4.5], [4.5, 6.0]]

__all__ = [
    "build_case",
    "build_problem",
    "default_stages",
    "evaluate",
    "find_executable",
    "initial_profile",
    "main",
    "run_fwi",
    "truth_profile",
    "vp_limits",
]


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------


def truth_profile(below: Any) -> Dict[str, np.ndarray]:
    """Return the layered truth ``vp``/``rho`` at depths ``below`` the seabed (km)."""

    below = np.asarray(below, dtype=np.float64)
    bottoms = np.array([layer[0] for layer in TRUTH_LAYERS])
    index = np.minimum(np.searchsorted(bottoms, below, side="left"), len(bottoms) - 1)
    vp = np.array([layer[1] for layer in TRUTH_LAYERS])[index]
    rho = np.array([layer[2] for layer in TRUTH_LAYERS])[index]
    return {"vp": vp, "rho": rho}


def initial_profile(below: Any) -> Dict[str, np.ndarray]:
    """Return the smooth starting ``vp``/``rho`` (Gaussian-filtered truth)."""

    dense = np.linspace(0.0, TRUTH_LAYERS[-1][0], PROFILE_SAMPLES)
    sigma = INITIAL_SMOOTHING / (dense[1] - dense[0])
    truth = truth_profile(dense)
    below = np.asarray(below, dtype=np.float64)
    return {
        key: np.interp(below, dense, gaussian_filter1d(values, sigma, mode="nearest"))
        for key, values in truth.items()
    }


def vp_limits() -> Tuple[float, float]:
    """Return the physical ``vp`` limits (km/s) behind :data:`VP_LOG_BOUNDS`.

    ``(min(vp0) exp(lower), max(vp0) exp(upper))`` over the starting sediment
    profile ``vp0``.
    """

    below = np.linspace(0.0, MODEL_DEPTH - WATER_DEPTH, PROFILE_SAMPLES)
    vp0 = initial_profile(below)["vp"]
    return (
        float(vp0.min() * np.exp(VP_LOG_BOUNDS[0])),
        float(vp0.max() * np.exp(VP_LOG_BOUNDS[1])),
    )


def _column(profile: Dict[str, np.ndarray], below: np.ndarray) -> Dict[str, Any]:
    """Author the sediment properties as vertical profiles in global depth."""

    depth = WATER_DEPTH + below
    return {
        key: xr.DataArray(values, dims=["z"], coords={"z": depth})
        for key, values in profile.items()
    }


# ---------------------------------------------------------------------------
# case
# ---------------------------------------------------------------------------


def find_executable(explicit: Optional[Any] = None) -> Optional[Path]:
    """Return the Sauce executable to use, or ``None`` when there is none."""

    candidates: List[Any] = [explicit] if explicit else []
    candidates += [
        os.environ.get("FS_SAUCE_EXECUTABLE"),
        os.environ.get("LOCAL_SOLVER_EXECUTABLE"),
        DEFAULT_EXECUTABLE,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return None


def _simulation(
    project: Project,
    name: str,
    profile: Dict[str, np.ndarray],
    frequencies: List[float],
    *,
    coarse: bool,
) -> Any:
    below = np.linspace(0.0, MODEL_DEPTH - WATER_DEPTH, PROFILE_SAMPLES)
    simulation = project.new_simulation(name=name, physics="acoustic", dimension=2)
    model = LayeredModel(dimension=2, x_limits=[0.0, WIDTH])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="water", properties={"vp": 1.5, "rho": 1.0})
    model.add_surface(name="seabed", depth=WATER_DEPTH)
    model.add_layer(name="sediment", properties=_column(profile, below))
    model.add_surface(name="bottom", depth=MODEL_DEPTH)
    simulation += model
    simulation += model.hex_mesh_generator(n=[16, 6] if coarse else [32, 12])
    simulation.mesh.set_adapt(
        elems_per_wave=2.0 if coarse else 3.0,
        order=4,
        f_low=min(frequencies),
        f_high=max(frequencies),
        adapt_order=True,
    )
    simulation += BoundaryCondition(conditions=["free"], boundaries=["z_min"])
    simulation += BoundaryCondition(
        conditions=["pml"],
        boundaries=["x_min", "x_max", "z_max"],
        pml_wavelengths=1.0,
        pml_exponent=3.0,
        pml_constant=20.0,
    )
    acquisition = Acquisition()
    acquisition.add_sources(kind="scalar", coords=[list(SOURCE)])
    hydrophone = ReceiverNode(name="hydrophone")
    hydrophone.add_component(name="p", field="pressure")
    acquisition.add_receiver_group(
        name="surface",
        device=hydrophone,
        coords=[[float(x), RECEIVER_DEPTH] for x in RECEIVER_X],
    )
    simulation += acquisition
    simulation += Discretization()
    # Single-precision FS_MG stalls near 1e-5; the Sauce e2e fixtures use 1e-4.
    simulation += SolverConfig(solve_on="final", max_iter=400, tolerance=1.0e-4)
    simulation.save()
    return simulation


def build_case(
    project_dir: Any, *, executable: Optional[Any] = None, coarse: bool = True
) -> Dict[str, Any]:
    """Build the truth and initial simulations and record the observed data.

    ``coarse`` selects the few-minute configuration (coarse mesh, two bands
    up to 4.5 Hz); ``coarse=False`` adds a 6 Hz band on a finer mesh.  The
    returned mapping carries everything :func:`run_fwi` and :func:`evaluate`
    need: the simulations, the observed job, the site, the control space,
    the misfit, the frequencies and the continuation bands.
    """

    solver = find_executable(executable)
    if solver is None:
        raise FileNotFoundError(
            "no Sauce executable: pass executable=, set FS_SAUCE_EXECUTABLE or "
            f"stage a build at {DEFAULT_EXECUTABLE}"
        )
    bands = FAST_BANDS if coarse else FULL_BANDS
    frequencies = sorted({float(f) for band in bands for f in band})
    project = Project(
        name="fwi_1d_profile", path=Path(project_dir), load_if_exists=False
    )
    below = np.linspace(0.0, MODEL_DEPTH - WATER_DEPTH, PROFILE_SAMPLES)
    truth = _simulation(
        project, "truth", truth_profile(below), frequencies, coarse=coarse
    )
    initial = _simulation(
        project, "initial", initial_profile(below), frequencies, coarse=coarse
    )

    site = LocalSite(solver=solver, n_workers=min(len(frequencies), 3))
    started = time.perf_counter()
    observed_job = FrequencyDomainJob("observed", truth, frequencies)
    outcome = site.run(observed_job, check=True)
    if not outcome.successful:
        raise RuntimeError("the observed-data forward job failed")
    timing = {"observed_seconds": time.perf_counter() - started}

    controls = im.ControlSpace(
        vp=im.DepthProfile(
            "vp", "sediment", spacing=0.05, transform="log", limits=vp_limits()
        ),
        rho=im.DepthProfile("rho", "sediment", spacing=0.1, transform="log"),
    )
    misfit = im.Misfit.huber(
        delta=HUBER_DELTA,
        preprocess=[im.Preprocess.offset_taper(d0=OFFSET_TAPER[0], d1=OFFSET_TAPER[1])],
    )
    return {
        "project": project,
        "truth": truth,
        "initial": initial,
        "observed_job": observed_job,
        "observed": im.ObservedData(observed_job),
        "site": site,
        "controls": controls,
        "misfit": misfit,
        "frequencies": frequencies,
        "bands": bands,
        "coarse": coarse,
        "timing": timing,
    }


def build_problem(case: Dict[str, Any], *, workdir: Any) -> im.ImagingProblem:
    """Return the case's :class:`~frequensolve.imaging.ImagingProblem` in ``workdir``.

    The problem is cached on ``case["problem"]`` so that a check at the
    initial point and the inversion share one linearization cache.
    """

    workdir = Path(workdir)
    problem = case.get("problem")
    if problem is not None and Path(problem.workdir) == workdir:
        return problem
    problem = im.ImagingProblem(
        case["initial"],
        controls=case["controls"],
        observed=case["observed"],
        misfit=case["misfit"],
        frequencies=case["frequencies"],
        site=case["site"],
        workdir=workdir,
        name="fwi_1d_profile",
    )
    case["problem"] = problem
    return problem


def default_stages(
    case: Dict[str, Any],
    iterations: Optional[int] = None,
    active: Any = ("vp",),
) -> list:
    """Return the continuation stages: one per band, ``iterations`` each.

    The TCCS acoustic benchmark inverts ``vp`` only (density stays at its
    smoothed start), so ``active`` defaults to the ``vp`` block; pass
    ``("vp", "rho")`` for the joint update.
    """

    if iterations is None:
        iterations = 4 if case["coarse"] else 10
    return im.Stage.bands(case["bands"], iterations=iterations, active=list(active))


def run_fwi(
    case: Dict[str, Any],
    *,
    workdir: Any,
    stages: Any = None,
    optimizer: Any = None,
) -> im.FWIResult:
    """Run the staged inversion and return its :class:`~frequensolve.imaging.FWIResult`.

    The TCCS driver capped the RMS control step at 0.015 per iteration over
    ~100 iterations; the reduced budget here uses 0.05 so that the notch is
    reachable in a handful of steps.  Its preconditioner is the same
    randomized Gauss-Newton diagonal (four probes, 1 % relative damping,
    inverse ratio 1e3), reduced to two probes in the coarse case.
    """

    problem = build_problem(case, workdir=workdir)
    if stages is None:
        stages = default_stages(case)
    if optimizer is None:
        optimizer = im.LBFGS(memory=10, step_limit=0.05, max_line_search_trials=5)
    fwi = im.FWI(
        problem,
        stages=stages,
        optimizer=optimizer,
        penalty=im.Tikhonov(alpha=TIKHONOV_ALPHA, order=1),
        preconditioner=im.Diagonal(
            probe_count=2 if case["coarse"] else 4,
            relative_damping=1.0e-2,
            maximum_inverse_ratio=1.0e3,
            seed=20260830,
        ),
        checkpoint="checkpoint.h5",
        history="history.json",
    )
    started = time.perf_counter()
    result = fwi.run(resume=True)
    case.setdefault("timing", {})["fwi_seconds"] = time.perf_counter() - started
    return result


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _profile_metrics(final: im.ControlVector, key: str) -> Dict[str, Any]:
    """RMS error of the recovered profile versus the truth on the control nodes."""

    nodes = final.to_xarray(key, frozen=0.0)
    below = np.asarray(nodes.coords[nodes.dims[0]], dtype=np.float64)
    truth = truth_profile(below)[key]
    initial = initial_profile(below)[key]
    recovered = initial * np.exp(np.asarray(nodes.values, dtype=np.float64))

    def rms(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(values))))

    initial_error = rms(initial - truth)
    final_error = rms(recovered - truth)
    return {
        "nodes_below_seabed_km": below.tolist(),
        "truth": truth.tolist(),
        "initial": initial.tolist(),
        "recovered": recovered.tolist(),
        "initial_rms_error": initial_error,
        "final_rms_error": final_error,
        "error_ratio": final_error / initial_error if initial_error > 0.0 else 1.0,
    }


def evaluate(result: im.FWIResult, case: Dict[str, Any]) -> Dict[str, Any]:
    """Return the acceptance metrics of one run.

    ``data_objective_ratio`` compares the final band's data misfit at the
    recovered model with the same misfit at the starting model (one extra
    forward solve); ``final_stage_objective_ratio`` is the TCCS definition,
    the reduction within the last stage.  Profile errors are RMS distances to
    the truth on the control nodes, for the start and the recovered model.
    """

    problem = result.problem
    assert problem is not None
    last = result.stages[-1]
    final_band = problem.restrict(frequencies=last.frequencies)
    final_vector = result.vector()
    final_data = float(final_band.value(final_vector))
    initial_data = float(final_band.value(final_band.space.zeros()))
    stages = [
        {
            "name": stage.name,
            "frequencies": [float(np.real(f)) for f in stage.frequencies],
            "iterations": stage.iterations,
            "linearizations": stage.linearizations,
            "initial_data_loss": stage.initial_loss.data,
            "final_data_loss": stage.final_loss.data,
            "objective_ratio": stage.reduction,
            "message": stage.message,
        }
        for stage in result.stages
    ]
    timing = dict(case.get("timing", {}))
    return {
        "data_objective_ratio": final_data / initial_data if initial_data else 1.0,
        "final_stage_objective_ratio": last.final_loss.data / last.initial_loss.data,
        "initial_data_loss": initial_data,
        "final_data_loss": final_data,
        "vp": _profile_metrics(final_vector, "vp"),
        "rho": _profile_metrics(final_vector, "rho"),
        "stages": stages,
        "iterations": int(sum(stage.iterations for stage in result.stages)),
        "linearizations": int(sum(stage.linearizations for stage in result.stages)),
        "success": bool(result.success),
        "timing": timing,
    }


# ---------------------------------------------------------------------------
# script
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--executable", type=Path, help="Sauce 2-D executable")
    parser.add_argument(
        "--workdir", type=Path, required=True, help="project and job directory"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="coarse mesh, two bands up to 4.5 Hz, four iterations per band",
    )
    args = parser.parse_args(argv)
    if find_executable(args.executable) is None:
        parser.error(
            "no Sauce executable: pass --executable or set FS_SAUCE_EXECUTABLE"
        )
    workdir = args.workdir.expanduser().resolve()
    case = build_case(workdir / "project", executable=args.executable, coarse=args.fast)
    result = run_fwi(case, workdir=workdir / "fwi")
    metrics = evaluate(result, case)
    metrics["workdir"] = str(workdir)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
