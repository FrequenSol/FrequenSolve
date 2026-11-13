"""Check PDE derivatives with independently sized meshed material controls.

These checks establish discrete derivative consistency, not quadrature convergence.
The native material_quadrature_probe separately compares the integral to an oracle.
"""

import json
import time

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging._artifacts import ControlVectorFile, ObjectiveReport
from frequensolve.imaging.jobs import FWIOperatorJob
from frequensolve.mesh import BoundaryCondition, BoundaryConditions
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig
from tests.test_imaging_integration import _executable, _simulation

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


@pytest.mark.parametrize("material_frequency", [1.0, 24.0], ids=["coarse", "fine"])
def test_independent_material_mesh_gradient(tmp_path, material_frequency):
    project = Project(
        name="mesh_derivative", path=tmp_path / "project", load_if_exists=False
    )
    truth = _simulation(project, "truth", 2.5)
    initial = _simulation(project, "initial", 2.2)
    for simulation in (truth, initial):
        simulation.BCs = BoundaryConditions(
            [
                BoundaryCondition(
                    conditions=["free"], boundaries=["x_min", "x_max", "z_min", "z_max"]
                )
            ]
        )
        simulation.mesh.set_adapt(elems_per_wave=4.0, order=3, adapt_order=False)
        simulation += Discretization(method="Galerkin", form_execution="compiled")
        simulation += SolverConfig(
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
        observed = FrequencyDomainJob("observed", truth, [6.0 - 0.3j])
        site.run(observed, check=True)
        problem = im.ImagingProblem(
            initial,
            controls=im.ControlSpace(
                vp=im.MeshParameters(
                    "vp",
                    "layer_2",
                    frequency=material_frequency,
                    epw=4.0,
                    artifact=str(tmp_path / "material.h5"),
                    transform="identity",
                )
            ),
            observed=im.ObservedData(observed),
            site=site,
            name="mesh_derivative",
        )
        started = time.perf_counter()
        lin = problem.linearize()
        discrete_seconds = time.perf_counter() - started
        comparison = FWIOperatorJob(
            "intersected",
            lin.job.simulation,
            [6.0 - 0.3j],
            action="linearize",
            active=lin.job.active,
            state="state.json",
            covector="gradient.h5",
            misfit=lin.job.misfit,
            control_state=lin.job.control_state,
            sensitivity_quadrature="material_intersections",
        )
        started = time.perf_counter()
        result = site.run(comparison, check=True)
        assert result.successful
        intersected_seconds = time.perf_counter() - started
        output = ControlVectorFile.read(comparison.covector_file(1))
        intersected = np.concatenate([output.blocks[name] for name in lin.space.blocks])
        report = ObjectiveReport.from_dict(
            json.loads(comparison.report_file(1).read_text())
        )
        np.testing.assert_allclose(report.total, lin.value, rtol=1e-12)
        relative_change = np.linalg.norm(
            intersected - lin.gradient.values
        ) / np.linalg.norm(lin.gradient.values)
        assert np.all(np.isfinite(intersected))
        if material_frequency == 1.0:
            np.testing.assert_allclose(
                intersected, lin.gradient.values, rtol=1e-8, atol=1e-14
            )
        else:
            assert (
                relative_change > 1e-5
            )  # The opt-in mode must actually reach the pullback.

        direction = np.random.default_rng(79).normal(size=lin.space.size)
        direction /= np.max(np.abs(direction))
        pairing = float(lin.gradient.values @ direction)
        point = problem.vector().values.copy()
        differences = []
        for step in (1e-2, 3e-3):
            finite_difference = (
                problem.value(point + step * direction)
                - problem.value(point - step * direction)
            ) / (2 * step)
            error = abs(pairing - finite_difference) / max(
                abs(pairing), abs(finite_difference), 1e-15
            )
            differences.append(
                dict(step=step, value=finite_difference, relative_error=error)
            )
        dot = lin.jacobian.dot_test(seed=19, tolerance=1e-6)
        metrics = dict(
            material_frequency=material_frequency,
            controls=lin.space.size,
            objective=lin.value,
            discrete_native=json.loads(
                (
                    lin.job.report_file(1).parent
                    / "_fs_run/tasks/task_000001/result.json"
                ).read_text()
            )["timings"],
            intersected_native=json.loads(
                (
                    comparison.report_file(1).parent
                    / "_fs_run/tasks/task_000001/result.json"
                ).read_text()
            )["timings"],
            intersected_relative_change=float(relative_change),
            discrete_seconds=discrete_seconds,
            intersected_seconds=intersected_seconds,
            intersected_pairing=float(intersected @ direction),
            pairing=pairing,
            finite_differences=differences,
            transpose_error=dot["relative_error"],
        )
        (tmp_path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        assert dot["passed"], metrics
        assert max(d["relative_error"] for d in differences) < 3e-3, metrics
