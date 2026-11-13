"""Acoustic receiver-band AWI gradient, checkpoint cost, and L-BFGS validation."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.mesh import BoundaryCondition, BoundaryConditions
from frequensolve.orchestrator.sites.local import LocalSite
from frequensolve.project import Project
from frequensolve.seismic import ReceiverNode
from frequensolve.simulation import Discretization, FrequencyDomainJob, SolverConfig
from tests.test_imaging_integration import _executable, _simulation

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]


def _contexts(job, task):
    path = job._result_path / "_fs_run" / "tasks" / f"task_{task:06d}" / "result.json"
    report = json.loads(path.read_text())
    return [entry["context"] for entry in report["solver"]["convergence"]["solves"]]


@pytest.mark.parametrize(
    "groups,ranks,method,meshed",
    [
        (1, 1, "Galerkin", False),
        (2, 1, "Galerkin", False),
        (2, 2, "Galerkin", False),
        (2, 1, "DPG", False),
        (2, 2, "DPG", False),
        (2, 1, "Galerkin", True),
        (2, 2, "Galerkin", True),
        (2, 1, "DPG", True),
        (2, 2, "DPG", True),
    ],
)
def test_native_awi_gradient_checkpoint_reuse_and_lbfgs(
    tmp_path, groups, ranks, method, meshed
):
    project = Project(name="awi", path=tmp_path / "project", load_if_exists=False)
    truth = _simulation(project, "truth", 2.5)
    initial = _simulation(project, "initial", 2.49)
    frequencies = [f - 0.25j for f in (4, 4.5, 5, 5.5, 6)]
    for sim in (truth, initial):
        sim.BCs = BoundaryConditions(
            [
                BoundaryCondition(
                    conditions=["free"], boundaries=["x_min", "x_max", "z_min", "z_max"]
                )
            ]
        )
        sim.mesh.set_adapt(elems_per_wave=0.05, order=3, adapt_order=False)
        sim += Discretization(method=method, form_execution="compiled")
        sim += SolverConfig(
            grids=1,
            tolerance=1e-10,
            relaxed_assembly=False,
            schur_precision="fp64",
            solve_precision="fp64",
            workspace_precision="fp64",
            operator_precision="fp64",
            mumps_precision="double",
        )
        if groups == 2:
            node = ReceiverNode(name="hydrophone")
            node.add_component(name="p", field="pressure")
            sim.acquisition.add_receiver_group(
                name="deep",
                device=node,
                coords=[[x, 0.4] for x in np.linspace(0.15, 0.85, 9)],
            )
        sim.save()
    with LocalSite(
        solver=_executable(),
        n_workers=1,
        threads_per_worker=2,
        shutdown_on_completion=False,
        solver_policy="warn",
    ) as site:
        observed = FrequencyDomainJob(
            "observed", truth, frequencies, phase_derivatives=1
        )
        assert site.run(observed, check=True).successful
        base = im.ImagingProblem(
            initial,
            controls=(
                im.MeshParameters(
                    "vp",
                    "layer_2",
                    frequency=6,
                    epw=1,
                    artifact=str(project.path / "vp.h5"),
                )
                if meshed
                else im.DepthProfile("vp", "layer_2", count=3)
            ),
            observed=im.ObservedData(
                observed, derivatives={"df": observed}, source_basis="source_geometry"
            ),
            site=site,
            workdir=tmp_path / "jobs",
            name="awi",
            submit_options={"procs_per_job": ranks},
        )
        spec = im.AWI(
            frequencies,
            [0.25, 0.5, 0.5, 0.5, 0.25],
            relative_floor=1e-6,
            minimum_filter_energy=1e-12,
            source_derivative="total",
            formulation="finite_filter" if meshed else "spectral_moment",
            filter_lags=np.arange(-4, 5) * 0.125 if meshed else None,
        )
        awi = base.awi(spec)
        x = awi.vector().values.copy()
        value = awi.linearize(x, gradient=False)
        lin = awi.linearize(x)
        assert lin is value
        assert lin.value > 0 and np.isfinite(lin.gradient.values).all()
        for task in range(1, len(frequencies) + 1):
            assert _contexts(lin.job, task) == (
                ["forward"] if meshed else ["forward", "incremental_forward"]
            )
            assert _contexts(lin.vjp_job, task) == (
                ["adjoint"] if meshed else ["adjoint", "adjoint"]
            )
        assert all(len(state.groups) == groups for state in lin.states)
        direction = (
            np.random.default_rng(73).uniform(-1, 1, len(x))
            if meshed
            else np.array([0.7, -0.4, 0.2])
        )
        assert awi.restrict(frequencies=frequencies).linearize(x) is lin
        assert all(
            next(iter(s.groups.values())).manifest["partition"]["n_ranks"] == ranks
            for s in lin.states
        )
        for index, state in enumerate(lin.states, 1):
            im.ControlVector(direction, lin.space).to_file(
                state_fingerprint=state.fingerprint,
                registry_fingerprint=next(iter(state.groups.values())).manifest[
                    "control_registry_fingerprint"
                ],
            ).write(lin.directory / f"direction_{index}.h5")
        tangent = im.FWIOperatorJob(
            "receiver_tangent",
            awi.simulation,
            frequencies,
            action="receiver_jvp",
            active=list(lin.space.blocks),
            receiver_state=lin.directory / "state.json",
            receiver_vector=lin.directory / "tangent.json",
            direction=lin.directory / "direction.h5",
            misfit=awi._receiver_misfit(),
        )
        awi._problem._run_job(tangent)
        data_pairing = 0.0
        for task, dual in enumerate(lin.duals, 1):
            left = _receiver_vectors(dual)
            right = _receiver_vectors(tangent.receiver_vector_file(task))
            assert set(left) == set(right)
            data_pairing += sum(np.real(np.vdot(left[key], right[key])) for key in left)
            assert len(_contexts(tangent, task)) == (2 if meshed else 4)
        np.testing.assert_allclose(
            data_pairing, lin.gradient.values @ direction, rtol=3e-5, atol=1e-10
        )
        if method == "DPG":
            replay = im.FWIOperatorJob(
                "awi_replay",
                awi.simulation,
                frequencies,
                action="receiver_vjp",
                active=list(lin.space.blocks),
                receiver_state=lin.directory / "state.json",
                receiver_vector=lin.directory / "dual.json",
                covector=lin.directory / "replay.h5",
                field_reuse="replay",
                misfit=awi._receiver_misfit(),
            )
            awi._problem._run_job(replay)
            replay_gradient = sum(
                im.ControlVectorFile.read(
                    replay.covector_file(task), native=False
                ).pack(lin.space.blocks)
                for task in range(1, len(frequencies) + 1)
            )
            np.testing.assert_allclose(
                replay_gradient, lin.gradient.values, rtol=3e-5, atol=1e-9
            )
            for task in range(1, len(frequencies) + 1):
                assert _contexts(replay, task) == (
                    ["forward", "adjoint"]
                    if meshed
                    else ["forward", "incremental_forward", "adjoint", "adjoint"]
                )

        step = 1e-5 if method == "DPG" or meshed else 1e-3
        fd = (awi.value(x + step * direction) - awi.value(x - step * direction)) / (
            2 * step
        )
        derivative = float(lin.gradient.values @ direction)
        if method == "Galerkin":
            np.testing.assert_allclose(derivative, fd, rtol=3e-3, atol=1e-8)
        # Receiver DPG actions expose frozen-Gram derivatives, not the full
        # derivative of a reassembled DPG operator. Test their transpose and
        # checkpoint replay above; retain full-operator FD disagreement below
        # as a diagnostic rather than claiming exact discrete differentiation.
        if groups == 2:
            first = next(iter(lin.states[0].groups.values())).manifest["checkpoints"]
            assert all(
                state.manifest["checkpoints"] == first
                for state in lin.states[0].groups.values()
            )
        truth_value = awi.value(np.full(len(x), 0.01))
        print(
            "AWI initial, true-model objective, gradient",
            lin.value,
            truth_value,
            lin.gradient.values,
            flush=True,
        )
        assert truth_value < 0.01 * lin.value
        workflow = im.FWI(
            awi,
            im.Stage(frequencies, iterations=8),
            optimizer=im.LBFGS(gradient_tolerance=1e-10),
            step_limit=0.01,
            checkpoint=tmp_path / "optimization.h5",
        )
        result = workflow.run()
        assert (
            result.stages[-1].final_loss.total
            < 0.5 * result.stages[0].initial_loss.total
        )
        initial_error = np.linalg.norm(x - 0.01)
        final_error = np.linalg.norm(result.state.values - 0.01)
        assert final_error < initial_error
        metrics = dict(
            groups=groups,
            ranks=ranks,
            method=method,
            meshed=meshed,
            full_operator_fd_relative_error=abs(derivative - fd)
            / max(abs(derivative), abs(fd)),
            receiver_pairing=float(data_pairing),
            value=lin.value,
            gradient_pairing=derivative,
            centered_difference=fd,
            initial_objective=result.stages[0].initial_loss.total,
            final_objective=result.stages[-1].final_loss.total,
            initial_model_error=float(initial_error),
            final_model_error=float(final_error),
        )
        (tmp_path / "awi_validation.json").write_text(json.dumps(metrics, indent=2))
        print(metrics)


def _receiver_vectors(path):
    """Read tangent/dual rows independently, pairing group and physical identities."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest["schema"] == "fs-receiver-vector-bundle-1":
        result = {}
        for entry in manifest["groups"]:
            child = Path(entry["file"])
            result.update(
                _receiver_vectors(child if child.is_absolute() else path.parent / child)
            )
        return result
    result = {}
    for entry in manifest["shards"]:
        shard = Path(entry["file"])
        with h5py.File(shard if shard.is_absolute() else path.parent / shard) as h5:
            for key, base, df in zip(
                h5["keys"][:],
                h5["base"][:],
                (h5["df"][:] if "df" in h5 else np.zeros_like(h5["base"][:])),
            ):
                identity = (manifest["receiver_group"], *map(int, key))
                assert identity not in result
                result[identity] = np.array([complex(*base), complex(*df)])
    return result
