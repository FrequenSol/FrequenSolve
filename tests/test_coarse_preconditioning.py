# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Coarse correction algebra, native transfer boundaries and workflow wiring."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse.linalg import cg

from frequensolve import imaging as im
from frequensolve.imaging.workflows import _receiver_probe_request
from tests.statistical_mesh_fixture import write_mesh
from tests.test_imaging_statistical_refinement import _mesh_space
from tests.test_imaging_workflows import _problem
from tests.imaging_fakes import FakeImagingSite

pytestmark = pytest.mark.unit


class Result:
    def __init__(self, **arrays):
        self.arrays = arrays
        self.metadata = dict(
            rank=len(arrays["directions"]), condition=1.0, hessian_asymmetry=0.0
        )
        self.path = "test-factor"

    def read(self, name):
        return self.arrays[name].copy()


class NativeOracle:
    """Independent small dense reference behind the native-operation interface."""

    def __init__(self):
        self.calls = []

    def mesh_basis(self, source, target, indices):
        self.calls.append("transfer")
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
        return Result(directions=interpolation[:, indices].T)

    def transfer_basis(self, base, directions, **kwargs):
        self.calls.append("basis")
        _, singular, vt = np.linalg.svd(directions, full_matrices=False)
        return Result(
            directions=vt[singular > kwargs["basis_tolerance"] * singular.max()]
        )

    def coarse_factor(self, directions, images, **kwargs):
        self.calls.append("factor")
        reduced = directions @ images.T
        assert np.allclose(reduced, reduced.T)
        lower = np.linalg.cholesky(reduced)
        return Result(
            directions=np.linalg.solve(lower, directions),
            images=np.linalg.solve(lower, images),
        )


@pytest.fixture
def setup(tmp_path):
    base = _problem(tmp_path / "project", FakeImagingSite()).space
    before, _ = write_mesh(tmp_path / "coarse.h5")
    after, _ = write_mesh(tmp_path / "fine.h5", refined=True)
    coarse, fine = _mesh_space(base, before), _mesh_space(base, after)
    rng = np.random.default_rng(19)
    j = rng.normal(size=(12, 6))
    h = j.T @ j + np.eye(6)
    native = NativeOracle()
    calls = []

    def batch(rows, regularization=None):
        calls.append(len(rows))
        return [im.ControlVector(h @ r, fine) for r in rows]

    lin = SimpleNamespace(
        space=fine,
        point=fine.zeros(),
        job=SimpleNamespace(job_file=tmp_path / "linearization.json"),
        problem=SimpleNamespace(backend=SimpleNamespace(curvature=lambda: native)),
        apply_normal_batch=batch,
    )
    return coarse, fine, h, native, calls, lin


def test_balanced_correction_preserves_full_solution_and_uses_no_backend_at_apply(
    setup,
):
    coarse, fine, h, native, calls, lin = setup
    local = im.FromOperator(np.diag(1 / np.diag(h)))
    bound = im.CoarseCorrection(coarse, local=local).bind(fine)
    bound.update(lin)
    assert calls == [4]
    assert native.calls == ["transfer", "basis", "factor"]
    q = bound._directions.T
    e = q @ q.T
    expected = e + (np.eye(6) - e @ h) @ np.diag(1 / np.diag(h)) @ (np.eye(6) - h @ e)
    actual = np.column_stack([bound.apply(row).values for row in np.eye(6)])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual, actual.T, atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() > 0
    np.testing.assert_allclose(actual @ h @ q, q, atol=1e-12)
    b = np.arange(1, 7.0)
    result, info = cg(h, b, M=bound.operator(), rtol=1e-4)
    assert info == 0
    np.testing.assert_allclose(result, np.linalg.solve(h, b), rtol=1e-4, atol=1e-4)
    assert calls == [4] and len(native.calls) == 3


def test_regularization_is_included_and_failed_refresh_invalidates(setup):
    coarse, fine, h, native, calls, lin = setup
    penalty = im.Quadratic(np.eye(6), weight=2).bind(fine)
    bound = im.CoarseCorrection(coarse, local=im.Identity()).bind(fine)
    bound.update(lin, regularization=penalty)
    q, aq = bound._directions, bound._images
    np.testing.assert_allclose(aq, q @ (h + 2 * np.eye(6)), atol=1e-12)

    def fail(*args, **kw):
        raise RuntimeError("new Hessian failed")

    lin.apply_normal_batch = fail
    with pytest.raises(RuntimeError, match="failed"):
        bound.update(lin)
    with pytest.raises(RuntimeError, match="update"):
        bound.apply(fine.zeros())


def test_lsrtm_coarse_cg_includes_regularization_and_damping(setup, tmp_path):
    from scipy.sparse.linalg import aslinearoperator

    coarse, fine, h, _, _, lin = setup
    problem = _problem(tmp_path / "workflow", FakeImagingSite())
    rhs = np.arange(1.0, 7.0)
    lin.gradient = im.ControlVector(-rhs, fine)
    lin.normal = aslinearoperator(h)
    penalty = im.Quadratic(np.eye(6), weight=0.4).bind(fine)
    solve = im.LSRTM(
        problem,
        method="cg",
        iterations=30,
        preconditioner=im.CoarseCorrection(coarse, local=im.Identity()),
        damping=0.2,
    )
    result = solve._run_cg(lin, fine, penalty)
    total = h + 0.6 * np.eye(6)
    np.testing.assert_allclose(
        result.values, np.linalg.solve(total, rhs), rtol=1e-4, atol=1e-4
    )
    metric = solve.bound_preconditioner
    np.testing.assert_allclose(metric._images, metric._directions @ total, atol=1e-12)
    assert solve.info["converged"]
    assert solve.info["preconditioner"]["rank"] == 4
    assert solve.info["relative_residual_history"][-1] <= 1e-4


def test_budget_rejects_before_backend_and_probe_requests_are_forwarded(setup):
    coarse, fine, _, native, calls, lin = setup
    for kw in ({"max_directions": 3}, {"max_memory_bytes": 1}):
        with pytest.raises(ValueError):
            im.CoarseCorrection(coarse, **kw).bind(fine).update(lin)
    assert not calls and not native.calls
    local = im.ReceiverBlockDiagonal(
        controls=("vp", "rho"), probe_count=8, source_samples=4
    )
    assert (
        _receiver_probe_request(im.CoarseCorrection(coarse, local=local))
        == local.receiver_probe_request()
    )
    for kw in (
        {"max_directions": True},
        {"basis_tolerance": 0},
        {"symmetry_tolerance": float("nan")},
    ):
        with pytest.raises(ValueError):
            im.CoarseCorrection(coarse, **kw)


def test_support_mask_uses_transpose_of_lifted_basis(setup):
    coarse, fine, h, native, calls, lin = setup
    coarse = coarse.with_support({"vp": np.array([1, 0, 1, 1], bool)})
    fine = fine.with_support({"vp": np.array([1, 1, 0, 1, 1, 1], bool)})
    lin.space = fine
    lin.point = fine.zeros()
    hs = h[np.ix_(fine.active_indices, fine.active_indices)]
    lin.apply_normal_batch = lambda rows, regularization=None: [
        im.ControlVector(hs @ r, fine) for r in rows
    ]
    bound = im.CoarseCorrection(coarse, local=im.Identity()).bind(fine)
    bound.update(lin)
    assert bound.info["rank"] == 3
    q = bound._directions.T
    np.testing.assert_allclose(
        np.column_stack([bound.apply(col).values for col in (hs @ q).T]), q, atol=1e-12
    )


def test_memory_budget_includes_full_transfer_before_support_mask(setup):
    coarse, fine, _, native, _, lin = setup
    fine = fine.with_support({"vp": np.array([1, 0, 0, 0, 0, 0], bool)})
    lin.space, lin.point = fine, fine.zeros()
    # Active factors fit (192 bytes), but the full-block transfer also needs 192.
    metric = im.CoarseCorrection(coarse, max_memory_bytes=300).bind(fine)
    with pytest.raises(ValueError, match="approximately 384 bytes"):
        metric.update(lin)
    assert not native.calls


@pytest.mark.integration
@pytest.mark.parametrize("ranks,curved", [(1, False), (2, False), (1, True)])
def test_native_coarse_transfer_and_factor(tmp_path, ranks, curved):
    executable = os.environ.get("FS_CURVATURE_SOLVER")
    if not executable:
        pytest.skip("Set FS_CURVATURE_SOLVER")
    before, _ = write_mesh(tmp_path / "coarse.h5", curved=curved)
    after, _ = write_mesh(tmp_path / "fine.h5", refined=True, curved=curved)
    import json
    import subprocess

    def runner(path):
        method = json.loads(path.read_text())["method"]
        command = [executable, "--curvature", str(path), "-nthreads", "2"]
        if ranks > 1 and method == "coarse_factor":
            command = ["mpiexec", "-n", str(ranks), *command]
        subprocess.run(command, check=True, capture_output=True)

    native = im.NativeCurvature(workdir=tmp_path / "native", runner=runner)
    lifted = native.mesh_basis(before, after, np.arange(4)).read("directions")
    original = native.mesh_directions(before, after, np.eye(4)).read("directions")
    np.testing.assert_allclose(lifted, original, atol=1e-12)
    h = np.diag(np.arange(1, 7.0)) + np.ones((6, 6))
    result = native.coarse_factor(lifted, lifted @ h, state="test", coordinates="fine")
    q, aq = result.read("directions"), result.read("images")
    np.testing.assert_allclose(aq, q @ h, atol=1e-11)
    np.testing.assert_allclose(q @ aq.T, np.eye(4), atol=1e-11)
    for bad in (-lifted @ h, np.zeros_like(lifted)):
        with pytest.raises(Exception):
            native.coarse_factor(lifted, bad, state="test", coordinates="fine")


def test_receiver_block_inverse_is_factored_once_and_includes_regularization(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    base = _problem(tmp_path / "project", FakeImagingSite()).space
    blocks = [
        replace(b, size=3, kind="mesh", baseline=np.zeros(3), basis_identity="shared")
        for b in base.resolved_blocks
    ]
    space = base._clone(blocks, {})
    config = im.ReceiverBlockDiagonal(
        controls=("vp", "rho"), scales={"vp": 2.0, "rho": 0.5}
    )
    matrices = np.broadcast_to([[3.0, 1.0], [1.0, 2.0]], (3, 2, 2)).copy()
    lin = SimpleNamespace(
        space=space,
        point=space.zeros(),
        receiver_blocks=im.ReceiverBlocks(("vp", "rho"), matrices, {}, space),
    )
    reg = im.Quadratic(np.eye(6), weight=0.4).bind(space)
    metric = config.bind(space)
    metric.update(lin, regularization=reg)
    scales = np.diag([2.0, 0.5])
    h = scales @ (matrices[0] + 0.4 * np.eye(2)) @ scales
    ridge = config.relative_damping * np.linalg.eigvalsh(h).max()
    expected = scales @ np.linalg.inv(h + ridge * np.eye(2)) @ scales
    # Runtime application must never factor, execute a native job, or mutate input.
    monkeypatch.setattr(
        np.linalg, "eigh", lambda *a, **k: pytest.fail("runtime factorization")
    )
    vector = np.arange(1.0, 7.0)
    result = metric.apply(vector).values
    np.testing.assert_array_equal(vector, np.arange(1.0, 7.0))
    for node in range(3):
        indices = [node, node + 3]
        np.testing.assert_allclose(result[indices], expected @ vector[indices])


def test_coarse_configuration_checkpoint_tracks_mesh_and_support(setup):
    from frequensolve.imaging._curvature_continuation import (
        _checkpoint_operator_configuration,
    )

    coarse, fine, _, _, _, _ = setup
    config = im.CoarseCorrection(coarse)
    identity = _checkpoint_operator_configuration(None, config)
    masked = coarse.with_support({"vp": np.array([1, 1, 0, 1], bool)})
    assert (
        _checkpoint_operator_configuration(None, im.CoarseCorrection(masked))
        != identity
    )
    assert (
        _checkpoint_operator_configuration(None, im.CoarseCorrection(fine)) != identity
    )
