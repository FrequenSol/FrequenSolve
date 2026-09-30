# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""History integrity and native staging, without duplicating backend algebra."""

import json
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from frequensolve.imaging import BFGSHistory, NativeCurvature
from frequensolve.inversion.optimization import LBFGSOptions, minimize_lbfgs

pytestmark = pytest.mark.unit


def checkpoint(pairs, accepted):
    return SimpleNamespace(
        optimizer_state=dict(
            schema="fs-lbfgs-restart-1",
            pairs=pairs,
            accepted_iterations=accepted,
        )
    )


def test_history_real_optimizer_and_round_trip(tmp_path):
    history = BFGSHistory([0.2, 0.1], state="quadratic-final", coordinates="linear")
    diagonal = np.array([3.0, 7.0])
    minimize_lbfgs(
        lambda x: 0.5 * np.dot(x * diagonal, x),
        lambda x: diagonal * x,
        [1.0, 2.0],
        preconditioner=lambda x, g: history.base_inverse_diagonal * g,
        options=LBFGSOptions(
            history_size=20, max_iterations=10, gradient_tolerance=1e-6
        ),
        callback=history,
    )
    assert len(history.steps) > 0
    restored = BFGSHistory.load(history.save(tmp_path / "history.h5"))
    np.testing.assert_array_equal(restored.steps, history.steps)
    np.testing.assert_array_equal(
        restored.gradient_differences, history.gradient_differences
    )
    assert restored.coordinates == history.coordinates


@pytest.mark.parametrize("pairs", [[], [dict(step=[0.0, 1.0], difference=[0.0, 2.0])]])
def test_history_rejects_reset_or_truncation(pairs):
    history = BFGSHistory([1.0, 1.0], state="s", coordinates="c")
    history(checkpoint([dict(step=[1.0, 0.0], difference=[2.0, 0.0])], 1))
    with pytest.raises(ValueError, match="truncated or reset"):
        history(checkpoint(pairs, 2))


def test_history_stage_and_optimizer_guards():
    history = BFGSHistory([1.0], state="s", coordinates="c")
    history(SimpleNamespace(stage_index=0, diagnostics=checkpoint([], 0)))
    with pytest.raises(ValueError, match="separate history per stage"):
        history(SimpleNamespace(stage_index=1, diagnostics=checkpoint([], 0)))
    with pytest.raises(ValueError, match="L-BFGS"):
        BFGSHistory([1.0], state="s", coordinates="c")(
            SimpleNamespace(optimizer_state=None)
        )


def test_stage_identity_survives_history_round_trip(tmp_path):
    history = BFGSHistory([1.0], state="s", coordinates="c")
    history(SimpleNamespace(stage_index=2, diagnostics=checkpoint([], 0)))
    restored = BFGSHistory.load(history.save(tmp_path / "history.h5"))
    with pytest.raises(ValueError, match="separate history per stage"):
        restored(SimpleNamespace(stage_index=3, diagnostics=checkpoint([], 0)))


@pytest.mark.parametrize(
    "options", [{"rank": 1.5}, {"oversampling": True}, {"seed": 2**32}]
)
def test_integer_options_do_not_silently_change_request(tmp_path, options):
    calls = []
    backend = NativeCurvature(workdir=tmp_path, runner=fake_runner(calls))
    with pytest.raises(ValueError):
        backend.bfgs_uncertainty(
            BFGSHistory([1.0], state="s", coordinates="c"), **options
        )
    assert not calls


def fake_runner(calls, mismatch=None):
    def run(path):
        request = json.loads(path.read_text())
        with h5py.File(request["input"], "r") as h5:
            arrays = {key: value[()] for key, value in h5.items() if key != "metadata"}
            metadata = json.loads(h5["metadata"][()])
        calls.append((request, arrays, metadata))
        output = dict(metadata, schema="fs-curvature-output-1")
        if mismatch:
            output.update(mismatch)
        with h5py.File(request["output"], "w") as h5:
            h5["metadata"] = np.bytes_(json.dumps(output))
            h5["marker"] = [17.0]

    return run


def test_bfgs_staging_preserves_backend_inputs(tmp_path):
    calls = []
    backend = NativeCurvature(workdir=tmp_path, runner=fake_runner(calls))
    history = BFGSHistory([0.2, 0.5], state="s", coordinates="c")
    history(checkpoint([dict(step=[1.0, 0.0], difference=[3.0, 0.0])], 1))
    result = backend.bfgs_uncertainty(history, prior_std=0.1, rank=2, seed=7)
    request, arrays, _ = calls[0]
    assert request["method"] == "bfgs_rsvd"
    assert request["rank"] == 2
    np.testing.assert_array_equal(arrays["steps"], history.steps)
    np.testing.assert_array_equal(arrays["prior_std"], [0.1, 0.1])
    assert result.path.name == "result.h5"
    np.testing.assert_array_equal(result.read("marker"), [17.0])
    backend.inverse_action(history, [1.0, 2.0])
    np.testing.assert_array_equal(calls[1][1]["vectors"], [[1.0, 2.0]])
    assert calls[0][0]["output"] != calls[1][0]["output"]


def test_rickett_normal_scheduling_and_depth_axis_packing(tmp_path):
    calls, directions = [], []

    def normal(direction):
        directions.append(direction.copy())
        return direction * 3

    backend = NativeCurvature(workdir=tmp_path, runner=fake_runner(calls))
    reference = np.arange(24.0).reshape(4, 2, 3)
    backend.rickett(
        reference,
        reference,
        normal=normal,
        state="s",
        coordinates="c",
        depth_axis=0,
        smoothing_radii=(2, 1, 0),
        damping=0.2,
    )
    np.testing.assert_array_equal(directions[0], reference.ravel())
    arrays = calls[0][1]
    np.testing.assert_array_equal(
        arrays["reference"], np.moveaxis(reference, 0, -1).ravel()
    )
    np.testing.assert_array_equal(arrays["normal_reference"], 3 * arrays["reference"])
    np.testing.assert_array_equal(arrays["grid_shape"], [4, 3, 2])
    np.testing.assert_array_equal(arrays["smoothing_radii"], [2, 0, 1])
    with pytest.raises(ValueError, match="exactly one"):
        backend.rickett(reference, reference, state="s", coordinates="c")


def test_mismatched_result_is_not_published(tmp_path):
    backend = NativeCurvature(
        workdir=tmp_path, runner=fake_runner([], dict(state="old"))
    )
    history = BFGSHistory([1.0], state="s", coordinates="c")
    with pytest.raises(ValueError, match="mismatched state"):
        backend.bfgs_uncertainty(history)
    assert not list(tmp_path.rglob("result.h5"))


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("FS_CURVATURE_SOLVER"), reason="Requires a Sauce curvature build"
)
def test_real_optimizer_history_through_native_backend(tmp_path):
    history = BFGSHistory([1.0], state="s", coordinates="prior-whitened")
    minimize_lbfgs(
        lambda x: 1.5 * np.dot(x, x),
        lambda x: 3 * x,
        [1.0],
        preconditioner=lambda x, g: history.base_inverse_diagonal * g,
        options=LBFGSOptions(history_size=10, max_iterations=5),
        callback=history,
    )
    backend = NativeCurvature(os.environ["FS_CURVATURE_SOLVER"], workdir=tmp_path)
    result = backend.bfgs_uncertainty(history, prior_std=0.1)
    np.testing.assert_allclose(result.read("variance"), [0.01 / 3], rtol=1e-12)
    np.testing.assert_allclose(
        backend.inverse_action(history, [1.0]).read("actions"), [[1 / 3]], rtol=1e-12
    )
    reference = np.cos(2 * np.pi * np.arange(16) / 16)
    result = backend.rickett(
        reference,
        4 * reference,
        normal=lambda x: 4 * x,
        state="s",
        coordinates="regular-grid",
    )
    np.testing.assert_allclose(
        result.read("normalized"), reference, rtol=5e-6, atol=1e-7
    )
