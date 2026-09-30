"""Statistical API behavior through the site boundary and native integration."""

import json
import os
import subprocess

import h5py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from frequensolve import imaging as im
from tests.imaging_fakes import FakeImagingSite
from tests.test_imaging_workflows import _problem


def oracle_runner(request_path):
    """Small independent dense service oracle for deterministic SDK tests."""
    request = json.loads(request_path.read_text())
    with h5py.File(request["input"]) as h5:
        arrays = {k: v[()] for k, v in h5.items() if k != "metadata"}
        meta = json.loads(h5["metadata"][()])
    if request.get("factors"):
        with h5py.File(request["factors"]) as h5:
            arrays.update(
                {name: value[()] for name, value in h5.items() if name != "metadata"}
            )
    method = request["method"]
    if method == "gaussian_prior":
        delta = arrays["point"] - arrays["reference"]
        diagonal = 1 / arrays["prior_std"] ** 2
        out = dict(
            value=[0.5 * np.sum(delta**2 * diagonal)],
            gradient=delta * diagonal,
            curvature_diagonal=diagonal,
        )
    elif method == "bfgs_rsvd":
        base = arrays["base_inverse_diagonal"]
        inverse = np.diag(base)
        for step, difference in zip(arrays["steps"], arrays["gradient_differences"]):
            rho = 1 / (step @ difference)
            transform = np.eye(len(base)) - rho * np.outer(step, difference)
            inverse = transform @ inverse @ transform.T + rho * np.outer(step, step)
        eig, modes = np.linalg.eigh(inverse - np.diag(base))
        meta["rank"] = len(eig)
        out = dict(
            base_inverse_diagonal=base,
            prior_std=arrays["prior_std"],
            modes=modes.T,
            eigenvalues=eig,
            variance=np.diag(inverse) * arrays["prior_std"] ** 2,
        )
        out["standard_deviation"] = np.sqrt(out["variance"])
    elif method.startswith("covariance_"):
        modes = arrays.get("modes", np.empty((0, len(arrays["prior_std"]))))
        inverse = (
            np.diag(arrays["base_inverse_diagonal"])
            + modes.T @ np.diag(arrays.get("eigenvalues", [])) @ modes
        )
        covariance = (
            arrays["prior_std"][:, None] * inverse * arrays["prior_std"][None, :]
        )
        if method == "covariance_action":
            out = dict(actions=arrays["vectors"] @ covariance)
        else:
            matrix = csr_matrix(
                (arrays["weights"], arrays["indices"], arrays["offsets"]),
                shape=(len(arrays["offsets"]) - 1, len(covariance)),
            ).toarray()
            variance = np.einsum("ij,jk,ik->i", matrix, covariance, matrix)
            out = dict(
                variance=variance, standard_deviation=np.sqrt(np.maximum(variance, 0))
            )
    elif method == "rickett":
        shape = tuple(arrays["grid_shape"][::-1])

        def envelope(flat):
            value = flat.reshape(shape)
            pad = request.get("padding", 0)
            if pad:
                widths = [(0, 0)] * value.ndim
                widths[-1] = (pad, pad)
                value = np.pad(
                    value, widths, mode="reflect" if shape[-1] > 1 else "edge"
                )
            n = value.shape[-1]
            multiplier = np.zeros(n)
            multiplier[0] = 1
            multiplier[1 : (n + 1) // 2] = 2
            if n % 2 == 0:
                multiplier[n // 2] = 1
            result = np.abs(
                np.fft.ifft(np.fft.fft(value, axis=-1) * multiplier, axis=-1)
            )
            return (result[..., pad : pad + shape[-1]] if pad else result).ravel()

        top, bottom = envelope(arrays["reference"]), envelope(
            arrays["normal_reference"]
        )
        damping = request.get("relative_damping", 0) * bottom.max() + request.get(
            "damping", 0
        )
        weights = np.divide(
            top, bottom + damping, out=np.zeros_like(top), where=bottom + damping > 0
        )
        out = dict(
            weights=weights,
            normalized=weights * arrays["image"],
            grid_shape=arrays["grid_shape"],
        )
    else:
        raise AssertionError(method)
    meta["schema"] = "fs-curvature-output-1"
    with h5py.File(request["output"], "w") as h5:
        h5["metadata"] = np.bytes_(json.dumps(meta))
        for key, value in out.items():
            h5[key] = value


class StatisticalSite(FakeImagingSite):
    supports_curvature = True

    def run_curvature(self, request):
        oracle_runner(request)


@pytest.mark.unit
def test_noise_declaration_serializes_fixed_sum_normalization():
    misfit = im.Misfit.l2(noise_std=2.0)
    assert misfit.normalization.value == 2
    assert misfit.normalization.reduction == "sum"
    for kw in (
        {"normalization": "observed_rms"},
        {"loss": "huber"},
        {"comparison": "phase_derivative"},
    ):
        with pytest.raises(ValueError):
            im.Misfit(noise_std=1, **kw)


@pytest.mark.unit
def test_gaussian_prior_and_automatic_fwi_uncertainty_round_trip(tmp_path):
    site = StatisticalSite(seed=11)
    problem = _problem(tmp_path, site, misfit=im.Misfit.l2(noise_std=1))
    problem.linearize()
    reference = problem.state
    prior = im.GaussianPrior(reference=reference, std={"vp": 2.0, "rho": 3.0})
    checkpoint = tmp_path / "checkpoint.h5"
    workflow = im.FWI(
        problem,
        [im.Stage(problem.frequencies, iterations=4)],
        regularization=prior,
        uncertainty=im.BFGSUncertainty(),
        checkpoint=checkpoint,
    )
    result = workflow.run()
    uq = result.uncertainty
    assert isinstance(uq, im.UncertaintyResult)
    assert uq.std("vp").shape == (5,)
    assert np.all(uq.std("vp").values > 0)
    archive = workflow._uncertainty_archive
    assert len(archive.steps) > 0
    assert workflow._optimizer_checkpoint["accepted_iterations"] > 0
    # A full-history inverse independently reconstructed from stored secants.
    inverse = np.diag(archive.base_inverse_diagonal)
    for step, difference in zip(archive.steps, archive.gradient_differences):
        rho = 1 / (step @ difference)
        t = np.eye(len(step)) - rho * np.outer(step, difference)
        inverse = t @ inverse @ t.T + rho * np.outer(step, step)
    scale = uq.factors.read("prior_std")
    expected = scale[:, None] * inverse * scale[None, :]
    direction = im.ControlVector(np.arange(1, problem.space.size + 1), problem.space)
    np.testing.assert_allclose(
        (uq.covariance @ direction).values, expected @ direction.values
    )
    result.save(tmp_path / "result")
    restored = im.FWIResult.load(tmp_path / "result", problem=problem)
    np.testing.assert_allclose(restored.uncertainty.std("vp"), uq.std("vp"))
    replay = im.FWI(
        problem,
        [im.Stage(problem.frequencies, iterations=4)],
        regularization=prior,
        uncertainty=im.BFGSUncertainty(),
        checkpoint=checkpoint,
    ).run(resume=True)
    np.testing.assert_allclose(replay.uncertainty.std("rho"), uq.std("rho"))


@pytest.mark.unit
def test_uncertainty_requires_declarations_before_propagation(tmp_path):
    site = StatisticalSite()
    problem = _problem(tmp_path, site)
    with pytest.raises(ValueError, match="GaussianPrior"):
        im.FWI(
            problem,
            [im.Stage(problem.frequencies, iterations=2)],
            uncertainty=im.BFGSUncertainty(),
        )
    assert not site.linearizations


@pytest.mark.unit
def test_illumination_reuses_frozen_joint_normal_and_preserves_rtm_default(tmp_path):
    problem = _problem(
        tmp_path,
        StatisticalSite(),
        controls=im.ControlSpace(vp=im.DepthProfile("vp", "sediment", count=5)),
    )
    lin = problem.linearize()
    reference = im.ControlVector(np.ones(lin.space.size), lin.space)
    config = im.ReferenceIllumination(reference=reference, relative_damping=0.01)
    calibration = lin.illumination(config)
    raw = lin.gradient
    first = calibration.apply(raw)
    second = calibration.apply(raw)
    np.testing.assert_allclose(first.values, second.values)
    np.testing.assert_allclose(first.values, calibration.weights.values * raw.values)
    np.testing.assert_allclose(im.rtm(problem).values, problem.gradient().values)
    np.testing.assert_allclose(
        im.rtm(problem, illumination=config).values, first.values
    )
    # A frozen calibration owns its reference even if the caller edits the vector.
    reference.values[:] *= 3
    np.testing.assert_allclose(calibration.apply(raw).values, first.values)


@pytest.mark.integration
def test_real_sauce_statistical_api(tmp_path):
    executable = os.environ.get("FS_CURVATURE_SOLVER")
    if not executable:
        pytest.skip("Set FS_CURVATURE_SOLVER to the new Sauce solver")

    class NativeSite(StatisticalSite):
        def run_curvature(self, request):
            with (request.parent / "solver.log").open("w") as log:
                subprocess.run(
                    [executable, "--curvature", str(request)],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS="1"),
                )

    problem = _problem(tmp_path, NativeSite(seed=11), misfit=im.Misfit.l2(noise_std=1))
    problem.linearize()
    prior = im.GaussianPrior(reference=problem.state, std={"vp": 2.0, "rho": 3.0})
    bound = prior.bind(problem.space, problem=problem)
    point = problem.space.zeros()
    point.values[:] = 1
    np.testing.assert_allclose(bound.gradient(point).values, 1 / bound.std**2)
    result = im.FWI(
        problem,
        [im.Stage(problem.frequencies, iterations=4)],
        regularization=prior,
        uncertainty=im.BFGSUncertainty(),
    ).run()
    assert np.all(result.uncertainty.std("vp") > 0)
    direction = im.ControlVector(
        np.ones(result.uncertainty.space.size), result.uncertainty.space
    )
    actual = result.uncertainty.covariance @ direction
    assert np.isfinite(actual.values).all()
    result.save(tmp_path / "saved")
    loaded = im.FWIResult.load(tmp_path / "saved", problem=problem)
    np.testing.assert_allclose(
        loaded.uncertainty.std("rho"), result.uncertainty.std("rho")
    )


@pytest.mark.integration
def test_native_mesh_prior_refinement_and_covariance_projection(tmp_path):
    executable = os.environ.get("FS_CURVATURE_SOLVER")
    if not executable:
        pytest.skip("Set FS_CURVATURE_SOLVER")
    from frequensolve.imaging.curvature import BFGSHistory, NativeCurvature
    from tests.statistical_mesh_fixture import write_mesh

    coarse, old_nodes = write_mesh(tmp_path / "coarse.h5")
    fine, new_nodes = write_mesh(tmp_path / "fine.h5", refined=True)
    native = NativeCurvature(executable, workdir=tmp_path / "operations")
    mean = 2 + 3 * old_nodes[:, 0] - old_nodes[:, 1]
    std = np.full(len(mean), 2.0)
    original = native.mesh_prior(coarse, coarse, mean, std)
    refined = native.mesh_prior(coarse, fine, mean, std)
    np.testing.assert_allclose(
        refined.read("reference"), 2 + 3 * new_nodes[:, 0] - new_nodes[:, 1]
    )
    np.testing.assert_allclose(original.read("measure_weights"), np.full(4, 0.25))
    np.testing.assert_allclose(
        refined.read("measure_weights"), [0.125, 0.25, 0.25, 0.125, 0.125, 0.125]
    )
    # The same constant physical perturbation has the same prior energy on either mesh.
    for result in (original, refined):
        mean = result.read("reference")
        prior = native.gaussian_prior(
            mean + 2, mean, result.read("prior_std"), state="s", coordinates="c"
        )
        np.testing.assert_allclose(prior.read("value"), [0.5])
    points = np.array([[0.25, 0.5], [0.5, 0.5], [0.75, 0.5], [2.0, 0.5]])
    sampling, valid = native.mesh_sampling(fine, points)
    np.testing.assert_allclose(sampling.sum(axis=1).A.ravel(), [1, 1, 1, 0])
    np.testing.assert_allclose(
        sampling @ refined.read("reference"), [2.25, 3.0, 3.75, 0.0]
    )
    np.testing.assert_array_equal(valid, [1, 1, 1, 0])
    factors = native.bfgs_uncertainty(
        BFGSHistory(np.ones(6), state="s", coordinates="c"),
        prior_std=refined.read("prior_std"),
    )
    projected = native.covariance(factors, projection=sampling)
    np.testing.assert_allclose(
        projected.read("variance"),
        np.asarray(sampling.power(2) @ refined.read("prior_std") ** 2).ravel(),
    )
