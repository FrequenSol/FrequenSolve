"""Native aperture operators, sparse encodings and encoded observation mapping."""

import numpy as np
import pytest

from frequensolve import CoordinateValue
from frequensolve import imaging as im
from frequensolve.mesh import BoundaryCondition, BoundaryConditions
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.seismic import GainDelay, SourceSignature
from frequensolve.seismic.sources import PointSource, SourceGeometry
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig
from frequensolve.units import ureg
from tests.test_imaging_integration import _executable, _simulation

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


def test_native_point_focus_reuses_l2_baseline(tmp_path, monkeypatch):
    from frequensolve.imaging.focusing import unit_misfit

    project = Project(name="reuse", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", 2.5)
    initial = _simulation(project, "initial", 2.2)
    with LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        observed = FrequencyDomainJob("observed", truth, [5.0, 6.0])
        site.run(observed, check=True)
        problem = im.ImagingProblem(
            initial,
            controls=im.DepthProfile("vp", "layer_2", count=3),
            observed=im.ObservedData(observed, source_basis="source_geometry"),
            misfit=unit_misfit(["surface"]),
            site=site,
            workdir=tmp_path / "jobs",
            name="reuse",
        )
        baseline = problem.linearize()
        actions = []
        run = problem.backend.run

        def record(job, **kwargs):
            actions.append(job.action)
            return run(job, **kwargs)

        monkeypatch.setattr(problem.backend, "run", record)
        for window in (0.0, 0.1):
            focus = problem.focus(im.Focusing(window=window))
            result = focus.linearize(baseline.point)
            assert focus.problem.linearize(baseline.point, gradient=False) is baseline
            assert np.isfinite(result.gradient.values).all()
            assert np.linalg.norm(result.gradient.values) > 0
        assert actions == ["vjp", "vjp"]


@pytest.mark.parametrize("method", ["DPG", "Galerkin"])
def test_native_aperture_gradients_with_physical_shot_observations(tmp_path, method):
    project = Project(name="focus", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", 2.5)
    initial = _simulation(project, "initial", 2.2)
    for simulation in (truth, initial):
        simulation.units.defaults["length"] = "km"
        simulation.acquisition.set_sources(
            SourceGeometry(
                kind="scalar",
                sources=[
                    PointSource("shot", CoordinateValue([500.0, 80.0], units="m"))
                ],
            )
        )
        simulation.acquisition.source_signature = SourceSignature(
            GainDelay(delay=0.013), frequencies=[5.0, 6.0]
        )
        # Material JVP/VJP exclude PML cells. Use fixed, non-PML geometry so
        # the finite difference differentiates exactly the same operator.
        simulation.BCs = BoundaryConditions(
            [
                BoundaryCondition(
                    conditions=["free"],
                    boundaries=["x_min", "x_max", "z_min", "z_max"],
                )
            ]
        )
        simulation.mesh.set_adapt(elems_per_wave=0.05, order=3, adapt_order=False)
        simulation += Discretization(method=method, form_execution="compiled")
        simulation += SolverConfig(
            grids=1,
            tolerance=1e-10,
            relaxed_assembly=False,
            schur_precision="fp64",
            solve_precision="fp64",
            workspace_precision="fp64",
            operator_precision="fp64",
            mumps_precision="double",
        )
        simulation.save()
    with LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        observed = FrequencyDomainJob("observed", truth, [5.0, 6.0])
        assert site.run(observed, check=True).successful
        problem = im.ImagingProblem(
            initial,
            controls=im.DepthProfile("vp", "layer_2", count=3),
            observed=im.ObservedData(observed, source_basis="source_geometry"),
            site=site,
            workdir=tmp_path / "jobs",
            name="focus",
        )
        direction = np.array([0.7, -0.4, 0.2])
        for strategy in ("linear", "pointwise"):
            focus = problem.focus(
                im.Focusing(
                    0.05,
                    aperture=im.SourceAperture(
                        0.04 * ureg.km, 0.02 * ureg.km, coarse=2
                    ),
                    strategy=strategy,
                )
            )
            x = focus.vector().values.copy()
            lin = focus.linearize(x)
            assert 0 <= lin.value <= 1
            assert np.isfinite(lin.gradient.values).all()
            from frequensolve.imaging.focusing import _rows, coherent_focus

            aux = (
                focus._extended(lin.state)
                if strategy == "linear"
                else focus._coarse(lin.state)
            )
            repeats = 1 if strategy == "linear" else len(focus._geometry()["coarse"])
            o = focus._observations(aux, repeats)
            weights = (
                None
                if repeats == 1
                else (focus._geometry()["w"] / focus._geometry()["w"].sum())
                @ focus._geometry()["interp"]
            )
            _, G, _ = coherent_focus(
                aux.simulated().values, o, _rows(aux), focus._kernel(), weights
            )
            jd = -aux.jvp(direction).values
            expected = np.real(np.vdot(G, jd))
            np.testing.assert_allclose(
                lin.gradient.values @ direction, expected, rtol=5e-5, atol=1e-8
            )
            # DPG freezes its Gram/test maps by contract; rebuilding the
            # forward operator in a finite difference also changes those maps.
            if method == "Galerkin":
                h = 1e-3
                fd = (
                    focus.value(x + h * direction) - focus.value(x - h * direction)
                ) / (2 * h)
                np.testing.assert_allclose(
                    lin.gradient.values @ direction, fd, rtol=2e-3, atol=1e-7
                )
