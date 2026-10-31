"""Native RWI background recovery with independent nonlinear elastic truth.

Run with FS_SAUCE_EXECUTABLE set to a local 2D double Sauce executable:
    python -m pytest -m integration tests/test_imaging_rwi_background_integration.py

The three comparisons fit reflectivity at the initial, updated and true
backgrounds. Model errors are measured independently of the data objective.
Generated metrics and per-action native timings live under pytest's tmp_path.
Acceptance measures improvement, not unbiased reflectivity recovery: the truth
contrast reaches material-dependent impedance boundaries, whereas Born
scattering holds those boundary coefficients fixed.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.orchestrator.sites.local import LocalSite
from tests.test_imaging_elastic_rwi_integration import _case, _executable

pytestmark = pytest.mark.integration


def _native_profile(jobs):
    """Summarize structured native task counters for the recorded outer solve."""
    actions = defaultdict(
        lambda: dict(
            tasks=0,
            elapsed=0.0,
            assembly=0.0,
            solve_forward=0.0,
            solve_adjoint=0.0,
            operator_assemblies=0,
            max_before_solve_bytes=0,
        )
    )
    builds = {}
    for job in jobs:
        for path in Path(job._result_path).glob("_fs_run/tasks/*/result.json"):
            data = json.loads(path.read_text())
            row = actions[job.action]
            row["tasks"] += 1
            for name in ("elapsed", "assembly", "solve_forward", "solve_adjoint"):
                row[name] += data.get("timings", {}).get(name, 0.0)
            row["operator_assemblies"] += data.get("misc", {}).get(
                "operator_assemblies", 0
            )
            memory = data.get("resources", {}).get("memory", {})
            row["max_before_solve_bytes"] = max(
                row["max_before_solve_bytes"],
                memory.get("before_solve", {}).get("max_per_rank_bytes", 0),
            )
            build = data.get("build", {})
            builds[build.get("build_id", "unknown")] = build
    return dict(
        scope="Outer solve only; summed task seconds, not wall time. Memory is a before-solve snapshot, not peak RSS.",
        jobs=dict(Counter(job.action for job in jobs)),
        actions=dict(actions),
        builds=list(builds.values()),
    )


def _recover_background(root, anisotropy, extra=None):
    """Exercise existing FWI/NewtonCG on a shared-frequency reduced objective."""
    root.mkdir(parents=True, exist_ok=True)
    site = LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=4,
        shutdown_on_completion=False,
    )
    started = perf_counter()
    try:
        problem, inner, expected = _case(root, site, anisotropy, "vp_vs_ip")
        active = ["vp"] + ([extra] if extra else [])
        rwi = problem.restrict(active=active).rwi(
            reflectivity=f"refl.{inner}", damping=1e-8
        )
        oracle = rwi.linearize()
        truth = oracle.state.vector(rwi.space)
        # Identity material controls are additive changes to the reference
        # model. The true background therefore has zero control coefficients.
        offset = rwi.space.pack(
            {
                name: np.full(
                    rwi.space.sizes[name], -0.1 if name == "model.vp" else -0.02
                )
                for name in rwi.space.blocks
            }
        )
        # Do not seed the recovery with the oracle's fitted contrast.
        rwi.state = oracle.state.with_update(truth + offset).with_update(
            rwi.full_space.restrict([f"refl.{inner}"]).zeros()
        )
        rwi.clear_cache()
        initial = rwi.linearize()

        def summary(fit):
            return dict(
                value=fit.value,
                model={name: fit.state[name].tolist() for name in rwi.space.blocks},
                reflectivity=fit.state[f"refl.{inner}"].tolist(),
                model_error={
                    name: float(np.linalg.norm(fit.state[name] - oracle.state[name]))
                    for name in rwi.space.blocks
                },
                reflectivity_relative_error=float(
                    np.linalg.norm(fit.state[f"refl.{inner}"] - expected)
                    / np.linalg.norm(expected)
                ),
                inner_reports=fit.reports,
            )

        metrics = dict(
            anisotropy=anisotropy,
            active=active,
            frequencies=list(rwi.frequencies),
            frequency_weights=[0.5, 1.5],
            reference_properties=dict(vp=2.5, epsilon=0.18, theta=0.37),
            depth_km=[0.2, 0.4, 0.6],
            true_reflectivity=expected.tolist(),
            oracle=summary(oracle),
            initial=summary(initial),
            iterations=[],
        )
        path = root / "background-recovery.json"

        def save():
            path.write_text(json.dumps(metrics, indent=2))

        def callback(event):
            metrics["iterations"].append(
                dict(
                    iteration=event.stage_iteration,
                    model={
                        name: event.model[name].tolist() for name in rwi.space.blocks
                    },
                    value=event.loss.total,
                    elapsed_seconds=perf_counter() - started,
                )
            )
            save()

        save()
        with rwi.backend.recording() as jobs:
            result = im.FWI(
                rwi,
                [im.Stage(rwi.frequencies, iterations=5)],
                optimizer=im.NewtonCG(
                    initial_forcing=0.1,
                    maximum_forcing=0.1,
                    minimum_forcing=1e-4,
                    max_cg_iterations=6,
                ),
                scaling="curvature",
                callback=callback,
            ).run(resume=False)
        final = rwi.linearize(result.state)
        metrics.update(
            final=summary(final),
            success=result.stages[-1].success,
            message=result.stages[-1].message,
            stage_metrics=dict(result.stages[-1].metrics),
            native_profile=_native_profile(jobs),
            total_seconds=perf_counter() - started,
        )
        save()
        return metrics
    finally:
        site.close()


@pytest.mark.parametrize(
    "anisotropy,extra", [("iso", None), ("tti", None), ("tti", "epsilon")]
)
@pytest.mark.timeout(1800)
def test_rwi_recovers_background(tmp_path, anisotropy, extra):
    metrics = _recover_background(tmp_path, anisotropy, extra)
    initial, final = metrics["initial"], metrics["final"]
    assert metrics["success"], metrics["message"]
    assert final["value"] < 0.1 * initial["value"]
    for name, error in initial["model_error"].items():
        assert final["model_error"][name] < 0.5 * error
    assert final["reflectivity_relative_error"] < initial["reflectivity_relative_error"]
