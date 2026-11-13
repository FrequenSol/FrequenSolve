"""Stage transitions, refreshed objectives and seeded checkpoint replay."""

import json
import os
import subprocess

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging._curvature_continuation import _CurvatureSource, _directions
from tests.statistical_mesh_fixture import write_mesh
from tests.test_imaging_curvature import solver_output, spec_digests, write_result
from tests.test_imaging_statistical_refinement import _mesh_space
from tests.test_imaging_statistics import StatisticalSite, oracle_runner
from tests.test_imaging_workflows import TIGHT, _problem

pytestmark = pytest.mark.unit


class TransferSite(StatisticalSite):
    """Independent dense numerical service behind the real SDK file boundary."""

    def run_curvature(self, path):
        request = json.loads(path.read_text())
        arrays, metadata = solver_output(request)
        method = request["method"]
        calls = self.__dict__.setdefault("transfer_calls", [])
        calls.append((request, arrays))
        if method not in {
            "bfgs_rsvd",
            "curvature_basis",
            "curvature_refresh",
            "curvature_warm_start",
        }:
            return oracle_runner(path)
        base = arrays["base_inverse_diagonal"]
        metadata["controls"] = len(base)
        if method == "curvature_basis":
            metadata["basis_tolerance"] = request["basis_tolerance"]
            _, singular, right = np.linalg.svd(
                arrays["directions"] / np.sqrt(base), full_matrices=False
            )
            keep = singular > request["basis_tolerance"] * singular.max(initial=0)
            modes = right[keep][: request.get("rank", len(singular))]
            metadata["rank"] = len(modes)
            output = dict(directions=modes * np.sqrt(base))
        else:
            inverse = np.diag(base)
            if method == "bfgs_rsvd":
                modes = arrays.get("seed_modes", np.empty((0, len(base))))
                eig = arrays.get("seed_eigenvalues", np.empty(0))
                if "seed_rank" in request:
                    metadata["seed_rank"] = len(eig)
                inverse += modes.T @ np.diag(eig) @ modes
                for step, difference in zip(
                    arrays["steps"], arrays["gradient_differences"]
                ):
                    rho = 1 / (step @ difference)
                    transform = np.eye(len(base)) - rho * np.outer(step, difference)
                    inverse = transform @ inverse @ transform.T + rho * np.outer(
                        step, step
                    )
            elif method == "curvature_refresh":
                directions = arrays["directions"]
                reduced = directions @ arrays["images"].T
                # Like Sauce: bound K's relative asymmetry, record it, symmetrize.
                tolerance = request.get("symmetry_tolerance", 1e-2)
                asymmetry = np.linalg.norm(reduced - reduced.T) / max(
                    np.linalg.norm(reduced), np.finfo(float).tiny
                )
                if asymmetry > tolerance:
                    raise RuntimeError("not symmetric within symmetry_tolerance")
                metadata.update(
                    symmetry_tolerance=tolerance, hessian_asymmetry=asymmetry
                )
                reduced = (reduced + reduced.T) / 2
                inverse += (
                    directions.T
                    @ (np.linalg.inv(reduced) - np.eye(len(directions)))
                    @ directions
                )
            else:
                modes, eig = arrays["modes"], arrays["eigenvalues"]
                correction = modes.T @ np.diag(eig) @ modes
                whitened = correction / np.sqrt(base[:, None] * base[None, :])
                lowest = np.linalg.eigvalsh(whitened).min(initial=0)
                damping = min(1, 0.99 / -lowest) if lowest < 0 else 1
                metadata["transfer_damping"] = damping
                inverse += damping * correction
            eig, modes = np.linalg.eigh(inverse - np.diag(base))
            order = np.argsort(-np.abs(eig))
            order = order[
                np.abs(eig[order]) > 1e-12 * max(1, np.abs(eig).max(initial=0))
            ]
            order = order[: request.get("rank", len(base))]
            metadata["rank"] = len(order)
            prior = arrays.get("prior_std", np.ones(len(base)))
            output = dict(
                base_inverse_diagonal=base,
                prior_std=prior,
                modes=modes[:, order].T,
                eigenvalues=eig[order],
                variance=np.diag(inverse) * prior**2,
                standard_deviation=np.sqrt(np.diag(inverse)) * prior,
            )
        write_result(request, metadata, output)


def _workflow(
    path,
    site,
    *,
    policy=None,
    checkpoint=None,
    callback=None,
    uncertainty=None,
    scaling=None,
    frequencies=(4, 6),
):
    problem = _problem(
        path, site, misfit=im.Misfit.l2(noise_std=1), frequencies=frequencies
    )
    problem.linearize()
    prior = im.GaussianPrior(problem.state, std={"vp": 2, "rho": 3})
    return im.FWI(
        problem,
        [im.Stage([frequencies[0]], 3), im.Stage([frequencies[1]], 3)],
        optimizer=im.LBFGS(**TIGHT),
        regularization=prior,
        curvature=policy,
        checkpoint=checkpoint,
        callback=callback,
        uncertainty=uncertainty,
        scaling=scaling,
    )


@pytest.mark.parametrize("method", ["refresh", "warm_start"])
@pytest.mark.parametrize("laplace", [False, True])
def test_band_transition_uses_fresh_history_and_transferred_operator(
    tmp_path, method, laplace
):
    site = TransferSite()
    workflow = _workflow(
        tmp_path,
        site,
        policy=getattr(im.CurvatureTransfer, method)(rank=3),
        frequencies=(4 - 1j, 6 - 2j) if laplace else (4, 6),
    )
    result = workflow.run()
    provenance = result.stages[1].metrics["curvature_transfer"]
    assert provenance["mode"] == method and provenance["source_stage"] == 0
    assert workflow._curvature_archive.seed_rank > 0
    assert len(workflow._curvature_archive.steps) <= result.stages[1].iterations
    assert provenance["target_state"] != provenance["source_state"]
    if method == "refresh":
        call = next(
            arrays
            for request, arrays in site.transfer_calls
            if request["method"] == "curvature_refresh"
        )
        view = workflow._views[workflow.stages[1].label(1)]
        lin = view.linearize()
        prior = workflow.regularization.bind(lin.space, problem=view)
        for direction, image in zip(call["directions"], call["images"]):
            expected = (
                lin.normal @ im.ControlVector(direction, lin.space)
            ).values + prior.curvature_diagonal() * direction
            np.testing.assert_allclose(image, expected, rtol=1e-10)
        assert provenance["hessian_actions"] == len(call["directions"])
        # Sauce's default tolerance applies; the measured asymmetry is recorded.
        assert provenance["symmetry_tolerance"] == 1e-2
        assert 0 <= provenance["hessian_asymmetry"] < 1e-8
    else:
        assert provenance["hessian_actions"] == 0 and provenance["inherited"]
        assert "hessian_asymmetry" not in provenance


def test_refresh_policy_symmetry_tolerance_reaches_sauce_and_provenance(
    tmp_path, monkeypatch
):
    from frequensolve.imaging._backend import Backend

    created = []
    curvature = Backend.curvature

    def recorded(self, **options):
        created.append(curvature(self, **options))
        return created[-1]

    monkeypatch.setattr(Backend, "curvature", recorded)
    site = TransferSite()
    policy = im.CurvatureTransfer.refresh(rank=3, symmetry_tolerance=5e-3)
    result = _workflow(
        tmp_path, site, policy=policy, uncertainty=im.BFGSUncertainty()
    ).run()
    (request,) = [
        r for r, _ in site.transfer_calls if r["method"] == "curvature_refresh"
    ]
    assert request["symmetry_tolerance"] == 5e-3
    # The policy rides on its refresh request; no backend object's default changes.
    assert created and all(native.symmetry_tolerance is None for native in created)
    for provenance in (
        result.stages[1].metrics["curvature_transfer"],
        result.uncertainty.provenance,
    ):
        assert provenance["symmetry_tolerance"] == 5e-3
        assert provenance["hessian_asymmetry"] <= 5e-3


def test_stage_reset_override_uses_fresh_baseline(tmp_path):
    workflow = _workflow(
        tmp_path, TransferSite(), policy=im.CurvatureTransfer.refresh(rank=2)
    )
    workflow.stages[1] = im.Stage([6], 2, curvature=im.CurvatureTransfer.reset())
    result = workflow.run()
    assert result.stages[1].metrics["curvature_transfer"]["mode"] == "reset"
    assert workflow._curvature_archive.seed_rank == 0


@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("scaling", [None, "curvature"])
def test_seeded_resume_matches_uninterrupted_iterates(tmp_path, boundary, scaling):
    policy = im.CurvatureTransfer.refresh(rank=3)
    expected = _workflow(
        tmp_path / "expected", TransferSite(), policy=policy, scaling=scaling
    ).run()
    location = tmp_path / "resumed"
    location.mkdir()
    checkpoint = location / "resume.h5"

    def stop(event):
        if event.stage_index == 1 and event.stage_iteration == 1:
            raise RuntimeError("interrupt after published checkpoint")

    interrupted = _workflow(
        location,
        TransferSite(),
        policy=policy,
        checkpoint=checkpoint,
        callback=stop,
        scaling=scaling,
    )
    if boundary:
        interrupted.solve_stage(interrupted.stages[0])
    else:
        with pytest.raises(RuntimeError, match="interrupt"):
            interrupted.run()
    resumed_site = TransferSite()
    resumed = _workflow(
        location, resumed_site, policy=policy, checkpoint=checkpoint, scaling=scaling
    ).run()
    np.testing.assert_allclose(
        resumed.state.values, expected.state.values, rtol=1e-11, atol=1e-11
    )
    # The final stage seeds nothing, so a mid-stage resume calls no service.
    refreshed = sum(
        request["method"] == "curvature_refresh"
        for request, _ in resumed_site.__dict__.get("transfer_calls", [])
    )
    assert refreshed == int(boundary)  # Mid-stage resume reuses the exact seed.


def test_fresh_run_drops_previous_final_curvature(tmp_path):
    workflow = _workflow(
        tmp_path, TransferSite(), policy=im.CurvatureTransfer.refresh(rank=3)
    )
    workflow.run()
    result = workflow.run(resume=False)
    assert result.stages[0].metrics["curvature_transfer"]["mode"] == "reset"
    assert "source_stage" not in result.stages[0].metrics["curvature_transfer"]


def test_refreshed_uncertainty_records_provenance_and_rejects_inherited_metric(
    tmp_path,
):
    workflow = _workflow(
        tmp_path / "refresh",
        TransferSite(),
        policy=im.CurvatureTransfer.refresh(rank=3),
        uncertainty=im.BFGSUncertainty(),
    )
    result = workflow.run()
    assert result.uncertainty.provenance["refreshed"]
    assert not result.uncertainty.provenance["inherited"]
    loaded = im.UncertaintyResult.load(
        result.uncertainty.save(tmp_path / "saved"), workflow.final_problem.full_space
    )
    assert loaded.provenance == result.uncertainty.provenance
    inherited = _workflow(
        tmp_path / "warm",
        TransferSite(),
        policy=im.CurvatureTransfer.warm_start(rank=3),
        uncertainty=im.BFGSUncertainty(),
    )
    with pytest.raises(ValueError, match="requires refreshed curvature"):
        inherited.run()


def test_profile_refinement_changes_dimension_without_interpolating_gradients(tmp_path):
    workflow = _workflow(
        tmp_path, TransferSite(), policy=im.CurvatureTransfer.refresh(rank=3)
    )
    workflow.stages[1] = im.Stage(
        [6], 3, controls={"vp": im.DepthProfile("vp", "sediment", count=9)}
    )
    result = workflow.run()
    assert result.stages[0].space.size == 8 and result.stages[1].space.size == 12
    assert workflow._curvature_archive.seed_modes.shape[1] == 12
    assert result.stages[1].metrics["curvature_transfer"]["hessian_actions"] > 0


def test_mesh_direction_lift_uses_saved_basis_and_zero_frozen_controls(tmp_path):
    coarse, _ = write_mesh(tmp_path / "coarse.h5")
    fine, _ = write_mesh(tmp_path / "fine.h5", refined=True)
    problem = _problem(tmp_path, TransferSite())
    problem.linearize()
    old = _mesh_space(problem.space, coarse).with_support(
        {"vp": [True, False, True, True]}
    )
    new = _mesh_space(problem.space, fine)
    path = tmp_path / "factors.h5"
    stored = dict(
        base_inverse_diagonal=np.ones(3),
        prior_std=np.array([2.0, 3.0, 4.0]),
        modes=[[1.0, 2.0, 3.0]],
        eigenvalues=[0.5],
        variance=[1.0, 1.0, 1.0],
    )
    metadata = {
        "rank": 1,
        "state": "coarse-band",
        "coordinates": "coarse-controls",
        "output_digests": spec_digests(stored),
    }
    with h5py.File(path, "w") as h5:
        for name, value in stored.items():
            h5[name] = value
    factors = im.CurvatureResult(path, metadata)
    source = _CurvatureSource(
        0, im.UncertaintyResult(factors, im.ControlVector([0, 0, 0], old))
    )

    class Native:
        def mesh_directions(self, before, after, directions):
            assert before["identity"] == coarse["identity"]
            assert after["identity"] == fine["identity"]
            np.testing.assert_equal(directions, [[2, 0, 6, 12]])
            return type(
                "Output", (), {"read": lambda _, name: np.array([[2, 1, 9, 12, 0, 6]])}
            )()

    directions, eigenvalues = _directions(source, new, Native(), 2)
    np.testing.assert_equal(directions, [[2, 1, 9, 12, 0, 6]])
    np.testing.assert_equal(eigenvalues, [0.5])


def test_curvature_policy_and_checkpoint_changes_fail_explicitly(tmp_path):
    with pytest.raises(TypeError, match="CurvatureTransfer"):
        im.Stage([4], 1, curvature="refresh")
    workflow = _workflow(
        tmp_path,
        TransferSite(),
        policy=im.CurvatureTransfer.refresh(rank=2),
        checkpoint=tmp_path / "checkpoint.h5",
    )
    workflow.solve_stage(workflow.stages[0])
    changed = _workflow(
        tmp_path,
        TransferSite(),
        policy=im.CurvatureTransfer.refresh(rank=3),
        checkpoint=tmp_path / "checkpoint.h5",
    )
    with pytest.raises(ValueError, match="configuration changed"):
        changed.run()


@pytest.mark.integration
@pytest.mark.parametrize("method", ["refresh", "warm_start"])
def test_native_seeded_band_transition_matches_independent_dense_service(
    tmp_path, method
):
    executable = os.environ.get("FS_CURVATURE_SOLVER")
    if not executable:
        pytest.skip("Set FS_CURVATURE_SOLVER to the new Sauce backend")

    class NativeSite(TransferSite):
        def run_curvature(self, request):
            subprocess.run(
                [executable, "--curvature", str(request), "-nthreads", "3"],
                check=True,
                capture_output=True,
            )

    policy = getattr(im.CurvatureTransfer, method)(rank=3)
    uncertainty = im.BFGSUncertainty() if method == "refresh" else None
    oracle = _workflow(
        tmp_path / "oracle", TransferSite(), policy=policy, uncertainty=uncertainty
    ).run()
    actual = _workflow(
        tmp_path / "native", NativeSite(), policy=policy, uncertainty=uncertainty
    ).run()
    np.testing.assert_allclose(
        actual.state.values, oracle.state.values, rtol=1e-9, atol=1e-9
    )
    if actual.uncertainty is not None:
        matrix = np.eye(actual.uncertainty.space.size)
        np.testing.assert_allclose(
            actual.uncertainty.covariance.matmat(matrix),
            oracle.uncertainty.covariance.matmat(matrix),
            rtol=1e-9,
            atol=1e-9,
        )


@pytest.mark.integration
def test_native_mesh_refinement_and_refresh_preserve_correlated_directions(tmp_path):
    executable = os.environ.get("FS_CURVATURE_SOLVER")
    if not executable:
        pytest.skip("Set FS_CURVATURE_SOLVER to the new Sauce backend")
    coarse, _ = write_mesh(tmp_path / "coarse.h5")
    fine, _ = write_mesh(tmp_path / "fine.h5", refined=True)
    native = im.NativeCurvature(executable, workdir=tmp_path / "native")
    old = np.array([[1, 2, 3, 4], [2, 1, -1, -2]], float)
    lifted = native.mesh_directions(coarse, fine, old).read("directions")
    interpolation = np.array(
        [
            [1, 0, 0, 0],
            [0.5, 0.5, 0, 0],
            [0, 0, 0.5, 0.5],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    np.testing.assert_allclose(lifted, old @ interpolation.T, rtol=1e-12)
    base = np.arange(1, 7, dtype=float)
    basis = native.transfer_basis(
        base, lifted, state="fine-band", coordinates=fine["identity"]
    )
    directions = basis.read("directions")
    hessian = np.diag(np.arange(2, 8, dtype=float)) + 0.1 * np.ones((6, 6))
    refreshed = native.refresh_curvature(
        base,
        directions,
        directions @ hessian,
        state="fine-band",
        coordinates=fine["identity"],
    )
    from frequensolve.imaging._curvature_transfer import _CurvatureSeed

    reduced = directions @ hessian @ directions.T
    expected = (
        np.diag(base)
        + directions.T @ (np.linalg.inv(reduced) - np.eye(len(directions))) @ directions
    )
    seed = _CurvatureSeed(refreshed)
    np.testing.assert_allclose(seed.apply(np.eye(6)), expected, rtol=1e-11, atol=1e-11)
    assert np.linalg.eigvalsh(expected).min() > 0


@pytest.mark.parametrize("kind", ["quadratic", "composite", "custom"])
def test_completed_checkpoint_rejects_changed_regularization_without_execution(
    tmp_path, kind
):
    class StablePenalty(im.Regularization):
        def __init__(self, weight):
            self.weight = weight

        def bind(self, space):
            bound = im.Quadratic(np.eye(space.size), weight=self.weight).bind(space)
            bound.identity = f"stable-penalty:{self.weight}"
            return bound

    def penalty(space, weight):
        if kind == "custom":
            return StablePenalty(weight)
        quadratic = im.Quadratic(np.eye(space.size), weight=weight)
        return (
            quadratic
            if kind == "quadratic"
            else 2 * quadratic + im.Quadratic(np.eye(space.size), weight=0.03)
        )

    checkpoint = tmp_path / "checkpoint.h5"
    original = _workflow(
        tmp_path,
        TransferSite(),
        policy=im.CurvatureTransfer.refresh(rank=2),
        checkpoint=checkpoint,
    )
    original.regularization = penalty(original.problem.space, 0.1)
    original.run(resume=False)
    site = TransferSite()
    changed = _workflow(
        tmp_path,
        site,
        policy=im.CurvatureTransfer.refresh(rank=2),
        checkpoint=checkpoint,
    )
    changed.regularization = penalty(changed.problem.space, 0.2)
    before = len(site.jobs)
    with pytest.raises(ValueError, match="regularization declaration changed"):
        changed.run(resume=True)
    assert len(site.jobs) == before
    assert not site.__dict__.get("transfer_calls")
    unchanged = _workflow(
        tmp_path,
        site,
        policy=im.CurvatureTransfer.refresh(rank=2),
        checkpoint=checkpoint,
    )
    unchanged.regularization = penalty(unchanged.problem.space, 0.1)
    before = len(site.jobs)
    result = unchanged.run(resume=True)
    assert all(stage.resumed for stage in result.stages)
    assert len(site.jobs) == before


@pytest.mark.parametrize("option", ["scaling", "preconditioner", "diagonal_options"])
def test_completed_checkpoint_rejects_changed_operator_configuration_without_execution(
    tmp_path, option
):
    checkpoint = tmp_path / "checkpoint.h5"
    original = _workflow(
        tmp_path,
        TransferSite(),
        policy=im.CurvatureTransfer.refresh(rank=2),
        checkpoint=checkpoint,
    )
    if option == "diagonal_options":
        original.preconditioner = im.Diagonal(probes="unit", relative_damping=0.01)
    original.run(resume=False)
    site = TransferSite()
    changed = _workflow(
        tmp_path,
        site,
        policy=im.CurvatureTransfer.refresh(rank=2),
        checkpoint=checkpoint,
    )
    if option == "scaling":
        changed.scaling = {"vp": 2.0, "rho": 3.0}
    elif option == "preconditioner":
        changed.preconditioner = im.Identity()
    else:
        changed.preconditioner = im.Diagonal(probes="unit", relative_damping=0.02)
    before = len(site.jobs)
    with pytest.raises(
        ValueError, match="scaling or fixed preconditioner configuration changed"
    ):
        changed.run(resume=True)
    assert len(site.jobs) == before
    assert not site.__dict__.get("transfer_calls")


def test_nontransfer_checkpoint_keeps_legacy_admission_without_new_guard_metadata(
    tmp_path,
):
    from frequensolve.inversion import OptimizationCheckpoint

    checkpoint = tmp_path / "checkpoint.h5"
    workflow = _workflow(tmp_path, TransferSite(), checkpoint=checkpoint)
    workflow.run(resume=False)
    recorded = OptimizationCheckpoint.load(checkpoint)
    assert "curvature_regularization_declaration" not in recorded.metadata
    assert "curvature_operator_configuration" not in recorded.metadata
    del recorded.metadata["curvature_config"]
    recorded.save(checkpoint)
    replay = _workflow(tmp_path, TransferSite(), checkpoint=checkpoint).run(resume=True)
    assert all(stage.resumed for stage in replay.stages)
