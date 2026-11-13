"""Receiver-dependent polynomial window validation and serialization."""

from math import comb
from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging.jobs import _kernel_derivative as _validate


def _kernel_derivative(value):
    return _validate(value, residuals=("derivative", "window"))


def test_receiver_window_round_trip():
    request = dict(
        order=4, residual="window", window={"seabed": Path("windows.h5:/coefficients")}
    )
    result = _kernel_derivative(request)
    assert result["window"] == {"seabed": "windows.h5:/coefficients"}
    assert _kernel_derivative(result) == result


@pytest.mark.parametrize(
    "spec",
    [
        dict(residual="window", window={"seabed": "x.h5:/coefficients"}),
        dict(order=4, residual="derivative", window={"seabed": "x.h5:/coefficients"}),
        dict(order=4, residual="window", window={}),
        dict(order=4, residual="window", window={"seabed": [1, 2]}),
    ],
)
def test_receiver_window_rejects_invalid_request(spec):
    with pytest.raises(ValueError):
        _kernel_derivative(spec)


def test_global_window_unchanged():
    request = dict(order=2, residual="window", window=[1, -2, 1])
    assert _kernel_derivative(request)["window"] == [1, -2, 1]


def test_polynomial_moments_and_adjoint():
    rng = np.random.default_rng(912)
    time = np.linspace(0.0, 30.0, 601)
    tau = np.array([[1.0, 4.0, 9.0], [3.0, 7.0, 11.0]])
    coefficients = np.stack([comb(4, k) * (-tau) ** (4 - k) for k in range(5)])
    weights = np.exp((-2 * np.pi * 0.05 - 2j * np.pi * 2.5) * time)
    traces = rng.normal(size=(*tau.shape, len(time)))
    jets = np.stack(
        [
            np.sum(traces * weights * (-2j * np.pi * time) ** k, axis=-1)
            for k in range(5)
        ]
    )
    factors = coefficients / (-2j * np.pi) ** np.arange(5)[:, None, None]
    actual = np.sum(factors * jets, axis=0)
    expected = np.sum(traces * weights * (time - tau[..., None]) ** 4, axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-9)
    tangent = rng.normal(size=jets.shape) + 1j * rng.normal(size=jets.shape)
    dual = rng.normal(size=tau.shape) + 1j * rng.normal(size=tau.shape)
    left = np.vdot(np.sum(factors * tangent, axis=0), dual)
    right = np.vdot(tangent, factors.conj() * dual)
    np.testing.assert_allclose(left, right, rtol=1e-13)


def test_explicit_delays_stream_and_shift(tmp_path):
    rng = np.random.default_rng(419)
    delay = rng.uniform(0, 20, (267, 69))
    polynomial = [2.0, -3.0, 0.7, 0.01, 1.0]
    with h5py.File(tmp_path / "picks.h5", "w") as source:
        picks = source.create_dataset("seconds", data=delay)
        ref = im.write_receiver_window(
            tmp_path / "window.h5", picks, coefficients=polynomial
        )
    assert ref == f"{tmp_path / 'window.h5'}:/coefficients"
    with h5py.File(tmp_path / "window.h5") as handle:
        shifted = handle["coefficients"][...]
        assert shifted.shape == (5, 267, 69)
        for time in [0.0, 10.0, 40.0]:
            actual = np.polynomial.polynomial.polyval(time, shifted)
            expected = np.polynomial.polynomial.polyval(time - delay, polynomial)
            np.testing.assert_allclose(actual, expected, atol=2e-9)
    with pytest.raises(FileExistsError):
        im.write_receiver_window(tmp_path / "window.h5", delay)


@pytest.mark.parametrize(
    "delays", [np.ones(3), np.zeros((0, 2)), [[-1]], [[np.nan]], [[1j]]]
)
def test_explicit_delays_reject_invalid(tmp_path, delays):
    with pytest.raises(ValueError):
        im.write_receiver_window(tmp_path / "invalid.h5", delays)
    assert not (tmp_path / "invalid.h5").exists()


def test_explicit_delay_quartic_matches_table(tmp_path):
    delay = np.array([[0.0, 2.0], [10.0, 30.0]])
    im.write_receiver_window(tmp_path / "window.h5", delay)
    with h5py.File(tmp_path / "window.h5") as handle:
        expected = np.stack([comb(4, k) * (-delay) ** (4 - k) for k in range(5)])
        np.testing.assert_array_equal(handle["coefficients"][...], expected)


@pytest.mark.parametrize("kind", ["operator", "gradient"])
def test_window_contents_invalidate_saved_job_fingerprints(tmp_path, kind):
    from frequensolve.imaging.jobs import ControlGradientJob, FWIOperatorJob
    from tests.test_imaging_jobs import _saved_simulation

    simulation = _saved_simulation(tmp_path)
    window = tmp_path / "window.h5"
    im.write_receiver_window(window, [[1.0]])
    selection = dict(
        order=4, residual="window", window={"surface": "window.h5:/coefficients"}
    )
    if kind == "operator":
        job = FWIOperatorJob(
            "window",
            simulation,
            [4.0],
            action="linearize",
            active=["vp"],
            state="state.json",
            covector="g.h5",
            kernel_derivative=selection,
        )
    else:
        job = ControlGradientJob(
            "window",
            simulation,
            [4.0],
            kind="rtm",
            gradient="g.h5",
            observed={"surface": window},
            kernel_derivative=selection,
        )
    job.save()
    original, task = job.fingerprint(), job.task_fingerprint(1)
    exported = job.to_fs(job.export_context(), project_relative=True)
    assert (
        exported["kernel_derivative"]["window"]["surface"] == "window.h5:/coefficients"
    )
    restored = type(job).from_fs(exported, project_path=tmp_path)
    assert restored.kernel_derivative == job.kernel_derivative
    with h5py.File(window, "r+") as handle:
        handle["coefficients"][...] *= 2.0
    assert job.fingerprint() != original
    assert job.task_fingerprint(1) != task
    assert restored.fingerprint() != original
    window.unlink()
    with pytest.raises(FileNotFoundError):
        job.fingerprint()


def test_window_contents_invalidate_problem_cache_and_stage_identity(tmp_path):
    from tests.imaging_fakes import FakeImagingSite
    from tests.test_imaging_problem import _problem

    window = tmp_path / "window.h5"
    im.write_receiver_window(window, [[1.0]])
    selection = dict(
        order=4, residual="window", window={"surface": f"{window}:/coefficients"}
    )
    _, problem = _problem(tmp_path, FakeImagingSite(seed=7))
    stage = problem.restrict(kernel_derivative=selection)
    original_identity = stage.identity()
    original = stage.linearize()
    assert stage.linearize() is original
    with h5py.File(window, "r+") as handle:
        handle["coefficients"][...] *= 2.0
    assert stage.identity() != original_identity
    assert stage.linearize() is not original
    _, shared = _problem(
        tmp_path / "shared", FakeImagingSite(seed=7), kernel_derivative=selection
    )
    identity = shared.identity()
    with h5py.File(window, "r+") as handle:
        handle["coefficients"][...] *= 2.0
    assert shared.identity() != identity
