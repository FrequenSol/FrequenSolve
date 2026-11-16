"""Stage-start scaling, whitened warm starts, penalty identities and resident copies."""

import errno
import json
import os

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging._curvature_continuation import (
    _CurvatureSource,
    _directions,
    _initial_scale,
    _prepare_seed,
    _regularization_identity,
)
from frequensolve.imaging._curvature_transfer import _CurvatureSeed, _history_inverse
from frequensolve.imaging.curvature import CurvatureResult, _finite, _real_input
from tests.test_imaging_curvature import spec_digests
from tests.test_imaging_curvature_continuation import TransferSite
from tests.test_imaging_workflows import TIGHT, _problem

pytestmark = pytest.mark.unit


def _factors(path, **arrays):
    """A verified factor artifact as Sauce publishes it."""
    arrays = {name: np.asarray(value, dtype=float) for name, value in arrays.items()}
    metadata = dict(
        rank=len(arrays.get("eigenvalues", ())),
        state="source-band",
        coordinates="source-controls",
        output_digests=spec_digests(arrays),
    )
    with h5py.File(path, "w") as h5:
        h5["metadata"] = np.bytes_(json.dumps(metadata))
        for name, value in arrays.items():
            h5[name] = value
    return CurvatureResult(path, metadata)


def _fwi(path, site, *, std=2.0, policy=None, iterations=5, **options):
    problem = _problem(path, site, misfit=im.Misfit.l2(noise_std=1), frequencies=(4, 6))
    problem.linearize()
    prior = im.GaussianPrior(problem.state, std={"vp": std, "rho": 1.5 * std})
    return im.FWI(
        problem,
        [im.Stage([4], iterations), im.Stage([6], iterations)],
        optimizer=im.LBFGS(**TIGHT),
        regularization=prior,
        curvature=policy,
        **options,
    )


class _Quadratic:
    """Objective stub with an exact Hessian and a counted Hessian action."""

    def __init__(self, hessian, gradient):
        self.hessian, self._gradient, self.actions = hessian, gradient, 0

    def gradient(self, point):
        return np.array(self._gradient, dtype=float)

    def hessian_action(self, point, direction):
        self.actions += 1
        return self.hessian @ direction


def test_initial_scale_is_the_projected_preconditioned_rayleigh_step():
    hessian = np.array([[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 2.0]])
    base = np.array([1.0, 0.5, 3.0])
    point, free = np.zeros(3), (np.full(3, -np.inf), np.full(3, np.inf))
    objective = _Quadratic(hessian, [1.0, 2.0, -1.0])
    direction = base * [1.0, 2.0, -1.0]
    expected = (direction @ [1.0, 2.0, -1.0]) / (direction @ hessian @ direction)
    assert _initial_scale(objective, point, base, free) == pytest.approx(expected)
    assert objective.actions == 1
    # A component pressed against its bound does not enter the first step.
    lower = np.array([-np.inf, -np.inf, 0.0])
    projected = base * [1.0, 2.0, 0.0]
    assert _initial_scale(
        _Quadratic(hessian, [1.0, 2.0, 1.0]), point, base, (lower, free[1])
    ) == pytest.approx(
        (projected @ [1.0, 2.0, 0.0]) / (projected @ hessian @ projected)
    )
    # No descent information or no positive curvature keeps the declared scale.
    stationary = _Quadratic(hessian, [0.0, 0.0, 0.0])
    assert _initial_scale(stationary, point, base, free) == 1.0
    assert stationary.actions == 0
    assert (
        _initial_scale(_Quadratic(-hessian, [1.0, 2.0, 1.0]), point, base, free) == 1.0
    )


@pytest.mark.parametrize("std", [2.0, 0.01])
@pytest.mark.parametrize("method", ["warm_start", "refresh"])
def test_transfer_runs_scale_their_frozen_baseline_like_lbfgs(tmp_path, std, method):
    reset = _fwi(tmp_path / "reset", TransferSite(), std=std).run()
    site = TransferSite()
    workflow = _fwi(
        tmp_path / method,
        site,
        std=std,
        policy=getattr(im.CurvatureTransfer, method)(rank=3),
    )
    transfer = workflow.run()
    # A fixed unscaled baseline needed 35-112 evaluations here; L-BFGS's
    # dynamic gamma needs 20 (std 2) and 38 (std 0.01).
    assert sum(s.evaluations for s in transfer.stages) <= sum(
        s.evaluations for s in reset.stages
    )
    for stage in transfer.stages:
        scale = stage.metrics["curvature_transfer"]["initial_scale"]
        assert 0 < scale < 1  # The prior and data curvature exceed one here.
    final = workflow._curvature_archive
    np.testing.assert_allclose(
        final.base_inverse_diagonal,
        transfer.stages[1].metrics["curvature_transfer"]["initial_scale"],
    )
    if std == 0.01:  # Prior-dominated: both reach the same minimizer.
        assert transfer.stages[1].final_loss.total == pytest.approx(
            reset.stages[1].final_loss.total, rel=1e-6
        )


def test_explicit_fixed_metric_is_not_rescaled(tmp_path):
    workflow = _fwi(
        tmp_path,
        TransferSite(),
        policy=im.CurvatureTransfer.refresh(rank=3),
        preconditioner=im.Identity(),
    )
    result = workflow.run()
    assert result.stages[0].metrics["curvature_transfer"]["initial_scale"] is None
    np.testing.assert_equal(workflow._curvature_archive.base_inverse_diagonal, 1.0)


def _source(tmp_path, space, inverse, base, prior):
    """Exact source factors of ``inverse`` around the physical diagonal ``base``."""
    root = np.sqrt(base)
    whitened = (inverse - np.diag(base)) / np.outer(root, root)
    eigenvalues, vectors = np.linalg.eigh(whitened)
    modes = (root[:, None] * vectors).T  # physical modes, one per row
    factors = _factors(
        tmp_path / "source.h5",
        base_inverse_diagonal=base / prior**2,
        prior_std=prior,
        modes=modes / prior,
        eigenvalues=eigenvalues,
        variance=np.diag(inverse),
        standard_deviation=np.sqrt(np.diag(inverse)),
    )
    return _CurvatureSource(
        0, im.UncertaintyResult(factors, im.ControlVector(np.zeros(len(base)), space))
    )


@pytest.mark.parametrize(
    "backend", ["oracle", pytest.param("sauce", marks=pytest.mark.integration)]
)
def test_warm_start_recolors_the_whitened_correction_with_the_target_baseline(
    tmp_path, backend
):
    site = TransferSite()
    problem = _problem(tmp_path, site)
    problem.linearize()
    space = problem.space
    rng = np.random.default_rng(3)
    coupling = rng.normal(size=(space.size, 2))
    hessian = np.diag(rng.uniform(1, 2, space.size)) + 5 * coupling @ coupling.T
    inverse = np.linalg.inv(hessian)
    base = 1 / np.diag(hessian)  # A diagonal estimate that misses the coupling.
    source = _source(tmp_path, space, inverse, base, rng.uniform(0.5, 2, space.size))
    if backend == "oracle":
        native = im.NativeCurvature(
            workdir=tmp_path / "native", runner=site.run_curvature
        )
    else:
        executable = os.environ.get("FS_CURVATURE_SOLVER")
        if not executable:
            pytest.skip("Set FS_CURVATURE_SOLVER to the new Sauce backend")
        native = im.NativeCurvature(executable, workdir=tmp_path / "native")
    # The next band's curvature is four times larger and its re-estimated
    # diagonal (``scaling="curvature"`` or a probed Diagonal) tracks that.
    seed = _prepare_seed(
        im.CurvatureTransfer.warm_start(rank=space.size),
        source,
        space,
        native,
        base / 4,
        None,
        None,
        state="target-band",
        coordinates="target-controls",
        index=1,
    )
    np.testing.assert_allclose(seed.apply(np.eye(space.size)), inverse / 4, atol=1e-12)
    assert seed.provenance["transfer_damping"] == 1
    # Adding the physical correction to the new diagonal instead is not even
    # positive, so a single global damping would have shrunk every mode.
    physical = np.diag(base / 4) + inverse - np.diag(base)
    assert np.linalg.eigvalsh(physical).min() < 0


def test_whitened_lift_keeps_block_norms_across_a_basis_change(tmp_path):
    problem = _problem(tmp_path, TransferSite())
    problem.linearize()
    old = problem.space
    new = problem.with_controls(
        {"vp": im.DepthProfile("vp", "sediment", count=9)}
    ).space
    rng = np.random.default_rng(5)
    base, prior = rng.uniform(0.5, 2, old.size), rng.uniform(0.5, 2, old.size)
    modes = rng.normal(size=(2, old.size))
    factors = _factors(
        tmp_path / "source.h5",
        base_inverse_diagonal=base,
        prior_std=prior,
        modes=modes,
        eigenvalues=[-0.5, 0.25],
        variance=base,
        standard_deviation=np.sqrt(base),
    )
    source = _CurvatureSource(
        0, im.UncertaintyResult(factors, im.ControlVector(np.zeros(old.size), old))
    )
    physical, _ = _directions(source, new, None, 2)
    whitened, eigenvalues = _directions(source, new, None, 2, whitened=True)
    np.testing.assert_equal(eigenvalues, [-0.5, 0.25])
    source_whitened = modes / np.sqrt(base)
    for name in old.blocks:
        before = np.linalg.norm(source_whitened[:, old.slices[name]])
        after = np.linalg.norm(whitened[:, new.slices[name]])
        assert after == pytest.approx(before)
    # The unchanged rho block is copied exactly in both coordinates.
    np.testing.assert_array_equal(
        whitened[:, new.slices["model.rho"]],
        source_whitened[:, old.slices["model.rho"]],
    )
    np.testing.assert_allclose(
        physical[:, new.slices["model.rho"]],
        (modes * prior)[:, old.slices["model.rho"]],
    )


def test_native_tikhonov_is_identified_for_transfer(tmp_path):
    site = TransferSite()
    problem = _problem(tmp_path, site, smoothing={"type": "tikhonov", "lambda": 0.5})
    problem.linearize()
    workflow = im.FWI(
        problem,
        [im.Stage([4], 3), im.Stage([6], 3)],
        optimizer=im.LBFGS(**TIGHT),
        curvature=im.CurvatureTransfer.refresh(rank=2),
    )
    result = workflow.run()
    provenance = result.stages[1].metrics["curvature_transfer"]
    assert provenance["mode"] == "refresh" and provenance["hessian_actions"] > 0
    assert result.stages[1].final_loss.regularization > 0
    # The identity is the native checkpoint record, so it changes with it.
    view = workflow._views[workflow.stages[1].label(1)]
    lin = view.linearize()
    from frequensolve.imaging._native_regularization import (
        bind_workflow_regularization,
    )

    _, bound = bind_workflow_regularization(view.smoothing, lin.space, view, lin)
    identity = _regularization_identity(bound)
    assert identity == _regularization_identity(bound)
    bound.factor = 2.0
    assert _regularization_identity(bound) != identity


@pytest.mark.parametrize(
    "penalty",
    [
        im.TV(0.1),
        im.TGV(0.1, 0.2),
        2 * im.TV(0.1) + im.Quadratic(np.eye(8), weight=0.1),
        im.NativeRegularization({"type": "tv"}),
        {"type": "tgv"},
    ],
)
def test_transfer_rejects_proximal_penalties_before_any_solve(tmp_path, penalty):
    site = TransferSite()
    problem = _problem(tmp_path, site)
    with pytest.raises(ValueError, match="smooth objective"):
        im.FWI(
            problem,
            [im.Stage([4], 2), im.Stage([6], 2)],
            regularization=penalty,
            curvature=im.CurvatureTransfer.warm_start(),
        )
    smoothed = _problem(tmp_path, site, smoothing={"type": "tv"})
    with pytest.raises(ValueError, match="smooth objective"):
        im.FWI(smoothed, im.Stage([4], 2), curvature=im.CurvatureTransfer.refresh())
    # An explicit smooth penalty replaces the problem's TV smoothing.
    im.FWI(
        smoothed,
        im.Stage([4], 2),
        regularization=im.Tikhonov(0.1),
        curvature=im.CurvatureTransfer.refresh(),
    )
    assert not site.jobs


def test_staging_validates_float_blocks_without_copying():
    block = np.arange(12.0).reshape(3, 4)
    assert _real_input(block, "block") is block
    converted = _real_input(block.T, "block")  # A layout change copies once.
    assert converted.flags.c_contiguous and np.array_equal(converted, block.T)
    np.testing.assert_array_equal(_real_input([1, 2], "list"), [1.0, 2.0])
    for bad in (np.array([1.0, np.nan]), np.array([np.inf]), np.array([1j])):
        with pytest.raises(ValueError):
            _real_input(bad, "block")
    assert _finite(np.empty((0, 3))) and not _finite(np.array([-np.inf, 1.0]))


def test_seed_hands_its_block_to_the_history_and_applies_from_it(tmp_path):
    prior = np.array([2.0, 0.5])
    factors = _factors(
        tmp_path / "seed.h5",
        base_inverse_diagonal=[2.0, 3.0],
        prior_std=prior,
        modes=[[1.0, 2.0]],
        eigenvalues=[0.5],
        variance=[1.0, 1.0],
        standard_deviation=[1.0, 1.0],
    )
    seed = _CurvatureSeed(factors)
    dense = seed.apply(np.eye(2))
    block = seed.modes
    modes, eigenvalues = seed.take(prior)
    assert np.shares_memory(modes, block) and seed.rank == 0
    history = im.BFGSHistory(
        seed.physical_base / prior**2,
        state="s",
        coordinates="c",
        seed_modes=modes,
        seed_eigenvalues=eigenvalues,
    )
    apply = _history_inverse(history, prior)
    np.testing.assert_allclose(apply(None, np.eye(2)), dense)
    np.testing.assert_allclose(apply(None, np.array([1.0, 2.0])), [12, 3.5])
    with pytest.raises(ValueError):
        _history_inverse(history, np.array([1.0, 0.0]))


def _saved_uncertainty(tmp_path):
    problem = _problem(tmp_path, TransferSite())
    problem.linearize()
    space = problem.space.with_support({"vp": [True, False, True, True, True]})
    size = space.size
    factors = _factors(
        tmp_path / "factors.h5",
        base_inverse_diagonal=np.ones(size),
        prior_std=np.full(size, 2.0),
        modes=np.eye(1, size),
        eigenvalues=[-0.5],
        variance=np.full(size, 4.0),
        standard_deviation=np.full(size, 2.0),
    )
    point = im.ControlVector(np.arange(size, dtype=float), space)
    return problem, im.UncertaintyResult(factors, point, provenance={"mode": "x"})


def test_saved_uncertainty_links_factors_and_stores_coefficients_as_datasets(
    tmp_path, monkeypatch
):
    problem, result = _saved_uncertainty(tmp_path)

    def unread(*args, **kwargs):
        raise AssertionError("Construction must not read the marginals")

    monkeypatch.setattr(CurvatureResult, "read", unread)
    result = im.UncertaintyResult(result.factors, result.point)
    monkeypatch.undo()
    saved = result.save(tmp_path / "saved")
    assert os.path.samefile(saved / "covariance.h5", result.factors.path)
    record = json.loads((saved / "uncertainty.json").read_text())
    assert record["schema"] == "fs-uncertainty-2" and "point" not in record
    assert record["support"] == ["model.vp"]
    result.save(saved)  # Publishing again keeps the existing link.
    loaded = im.UncertaintyResult.load(saved, problem.full_space)
    np.testing.assert_array_equal(loaded.point.values, result.point.values)
    assert loaded.space.support_masks().keys() == result.space.support_masks().keys()
    np.testing.assert_array_equal(
        loaded.covariance.matmat(np.eye(result.space.size)),
        result.covariance.matmat(np.eye(result.space.size)),
    )
    # Version 1 results inlined the coefficient arrays as JSON lists.
    with h5py.File(saved / "uncertainty.h5") as h5:
        record["point"] = h5["point"][()].tolist()
        record["support"] = {
            name: h5["support/0"][()].astype(int).tolist() for name in record["support"]
        }
    record["schema"] = "fs-uncertainty-1"
    (saved / "uncertainty.json").write_text(json.dumps(record))
    legacy = im.UncertaintyResult.load(saved, problem.full_space)
    np.testing.assert_array_equal(legacy.point.values, result.point.values)


def test_saved_uncertainty_copies_only_where_links_are_unavailable(
    tmp_path, monkeypatch
):
    _, result = _saved_uncertainty(tmp_path)

    def cross_device(source, target):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", cross_device)
    saved = result.save(tmp_path / "copied")
    assert not os.path.samefile(saved / "covariance.h5", result.factors.path)
    assert (
        result.factors.verify()
        == CurvatureResult(saved / "covariance.h5", result.factors.metadata).verify()
    )

    def denied(source, target):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(os, "link", denied)
    with pytest.raises(PermissionError):
        result.save(tmp_path / "denied")
    assert not [p for p in (tmp_path / "denied").iterdir() if p.name.startswith(".")]
