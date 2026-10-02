# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Meshed sediment controls with an MPI partition containing only uncontrolled water."""

import json
import shutil

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.mesh import BoundaryCondition
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.simulation import FrequencyDomainJob
from tests.test_imaging_integration import FREQUENCY, START_VP, _executable, _simulation

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


def _water_sediment_mesh(path):
    """Order five water roots ahead of one sediment root for a natural scan partition."""
    depths = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.6]
    points = [
        f"Point {2 * row + col + 1} {col} {depth}"
        for row, depth in enumerate(depths)
        for col in range(2)
    ]
    curves = [f"Seg {2 * row + 1} {2 * row + 2}" for row in range(len(depths))]
    for row in range(len(depths) - 1):
        curves.extend(
            [f"Seg {2 * row + col + 1} {2 * row + col + 3}" for col in range(2)]
        )
    quads = [
        f"Linear {1 if row < 5 else 2} {2 * row + 1} {2 * row + 2} {2 * row + 4} {2 * row + 3}"
        for row in range(6)
    ]
    path.write_text(
        '@Version = 2\n@SubVersion = 0\n@Patch = 0\n@Dimension = 2\n@Manifold = 2\n@Domains = 2\n@Units = "km"\n'
        + f"$Points\n@Count = {len(points)}\n@TypePos = 1\n@IndexPos = 2\n"
        + "\n".join(points)
        + "\n$EndPoints\n"
        + f"$Curves\n@Count = {len(curves)}\n@TypePos = 1\n"
        + "\n".join(curves)
        + "\n$EndCurves\n"
        + "$Quads\n@Count = 6\n@TypePos = 1\n@DomainPos = 2\n"
        + "\n".join(quads)
        + "\n$EndQuads\n"
    )


def test_meshed_controls_allow_water_only_rank(tmp_path):
    if shutil.which("mpiexec") is None:
        pytest.skip("MPI launcher unavailable")
    project = Project(
        name="empty_control_partition", path=tmp_path / "project", load_if_exists=False
    )
    simulation = _simulation(project, "initial", START_VP)
    mesh = tmp_path / "water_sediment.gmp"
    _water_sediment_mesh(mesh)
    simulation.mesh.mesh = None
    simulation.mesh.file = str(mesh)
    simulation.mesh.format = "gmp"
    simulation.mesh.set_parallel(distribute=True, partitioner="simple")
    simulation.mesh.set_adapt(
        elems_per_wave=0.01,
        order=2,
        f_low=FREQUENCY,
        f_high=FREQUENCY,
        adapt_order=False,
    )
    simulation.BCs.clear()
    simulation += BoundaryCondition(
        conditions=["free"], boundaries=["x_min", "x_max", "z_min", "z_max"]
    )
    simulation.save()
    with LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        observed = FrequencyDomainJob("observed", simulation, [FREQUENCY])
        assert site.run(observed, check=True).successful
        problem = im.ImagingProblem(
            simulation,
            controls=im.ControlSpace(
                vp=im.MeshParameters(
                    "vp",
                    "layer_2",
                    frequency=1,
                    epw=1,
                    artifact=str(tmp_path / "sediment.h5"),
                    transform="log",
                )
            ),
            observed=im.ObservedData(observed),
            site=site,
            name="water_sediment",
        )
        state = problem.state
        problem.state = im.ControlState(
            state.space, np.full(state.space.full_size, 0.03)
        )
        serial = problem.linearize()
        assert serial.value > 0 and np.linalg.norm(serial.gradient.values) > 0
        problem.clear_cache()
        problem.backend.submit_options["procs_per_job"] = 2
        distributed = problem.linearize()
        report = distributed.job._result_path / "_fs_run/tasks/task_000001/result.json"
        result = json.loads(report.read_text())
        assert result["execution"]["mpi_ranks"] == 2
        # The run must actually exercise the empty partition, not just launch MPI.
        assert result["misc"]["control_local_dofs_min"] == 0
        assert result["misc"]["control_local_dofs_max"] > 0
        np.testing.assert_allclose(distributed.value, serial.value, rtol=2e-4)
        relative_error = np.linalg.norm(
            distributed.gradient.values - serial.gradient.values
        ) / np.linalg.norm(serial.gradient.values)
        assert relative_error < 5e-4
        adjoint = distributed.jacobian.dot_test(seed=7, tolerance=1e-3)
        assert adjoint["passed"], adjoint
