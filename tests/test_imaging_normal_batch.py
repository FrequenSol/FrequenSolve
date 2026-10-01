"""Several Gauss-Newton normal products from one Sauce ``normal`` job."""

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging import ControlSpace, DepthProfile, ImagingProblem
from frequensolve.imaging._native_regularization import bind_workflow_regularization
from frequensolve.imaging.controls import SourceParameters
from frequensolve.imaging.jobs import RegularizationJob
from frequensolve.imaging.regularization import Quadratic, _SymmetricModelOperator
from frequensolve.imaging.workflows import _StageObjective
from tests.imaging_fakes import FakeImagingSite, layered_simulation

pytestmark = pytest.mark.unit

FREQUENCIES = [4.0, 6.0]


def _problem(tmp_path, fake, *, weights=None, source=False):
    controls = {
        "vp": DepthProfile("vp", "sediment", count=4),
        "rho": DepthProfile("rho", "sediment", count=3),
    }
    if source:
        controls = {"vp": controls["vp"], "src": SourceParameters(signature=True)}
    problem = ImagingProblem(
        layered_simulation(tmp_path / "project"),
        controls=ControlSpace(**controls),
        observed={"surface": tmp_path / "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
    )
    if weights is not None:
        problem = problem.restrict(weights=weights)
    problem.state = problem.state_from(np.linspace(-1.0, 1.0, problem.space.size))
    return problem


class _Hessian:
    """Bound smooth term with a native-style identity and an explicit Hessian."""

    def __init__(self, space, matrix):
        self.space, self.matrix = space, matrix

    def checkpoint(self):
        return {"matrix": self.matrix}

    def hessian_operator(self, v=None):
        return _SymmetricModelOperator(lambda x: self.matrix @ x, self.space)


def _normal_jobs(fake):
    return [job for job in fake.jobs if getattr(job, "action", None) == "normal"]


def _dense_normal(fake, lin):
    """Return the weighted dense ``Re(J^H W J)`` of the fake's frozen surrogate."""

    surrogate = fake.linearizations[lin.state_fingerprint]
    J, total = surrogate.J, 0.0
    for weight, frequency in zip(lin.frequency_weights, lin.frequencies):
        rows = surrogate.rows(frequency)
        total = total + weight * np.real(J[rows].conj().T @ J[rows])
    return total


@pytest.mark.parametrize("weights", [None, [0.25, 2.0]])
def test_batch_runs_one_job_and_matches_single_products(tmp_path, weights):
    fake = FakeImagingSite(seed=11)
    lin = _problem(tmp_path / "batch", fake, weights=weights).linearize()
    directions = [lin.space.random(seed) for seed in (1, 2, 3)]

    products = lin.apply_normal_batch(directions)

    (job,) = _normal_jobs(fake)
    assert len(job.directions) == 3 and job.direction is None
    assert job.f_list == FREQUENCIES  # one job carries every frequency task
    expected = _dense_normal(fake, lin)
    for direction, product in zip(directions, products):
        np.testing.assert_allclose(
            product.values, expected @ direction.values, rtol=1e-12
        )

    # Products of a separately computed single-direction job agree to rounding.
    single_fake = FakeImagingSite(seed=11)
    single = _problem(tmp_path / "single", single_fake, weights=weights).linearize()
    for direction, product in zip(directions, products):
        reference = single.apply_normal(direction.values)
        np.testing.assert_allclose(product.values, reference.values, rtol=1e-12)
    assert [len(j.directions or [None]) for j in _normal_jobs(single_fake)] == [1] * 3


def test_batch_memoizes_per_direction_and_splits_only_large_batches(tmp_path):
    fake = FakeImagingSite(seed=11)
    lin = _problem(tmp_path, fake).linearize()
    known = [lin.space.random(seed) for seed in (1, 2)]
    lin.apply_normal_batch(known)
    assert len(_normal_jobs(fake)) == 1

    # Known products are reused: one new direction runs one single-direction job.
    fresh = lin.space.random(3)
    again = lin.apply_normal_batch(np.vstack([known[1].values, fresh.values]))
    jobs = _normal_jobs(fake)
    assert len(jobs) == 2 and jobs[-1].directions is None
    assert again[0] == lin.apply_normal(known[1])
    assert len(_normal_jobs(fake)) == 2

    lin.max_normal_directions = 2
    lin.apply_normal_batch([lin.space.random(seed) for seed in (4, 5, 6)])
    assert [len(j.directions or [None]) for j in _normal_jobs(fake)[2:]] == [2, 1]


def test_normal_matmat_applies_all_columns_in_one_job(tmp_path):
    fake = FakeImagingSite(seed=11)
    lin = _problem(tmp_path, fake).linearize()
    block = np.column_stack([lin.space.random(seed).values for seed in (7, 8, 9, 10)])

    product = lin.normal @ block

    assert product.shape == block.shape
    np.testing.assert_allclose(product, _dense_normal(fake, lin) @ block, rtol=1e-12)
    (job,) = _normal_jobs(fake)
    assert len(job.directions) == 4
    assert len({path.name for path in job.covectors}) == 4


def test_source_controls_keep_one_job_per_direction(tmp_path):
    fake = FakeImagingSite(seed=11)
    lin = _problem(tmp_path, fake, source=True).linearize()
    directions = [lin.space.random(seed) for seed in (1, 2)]

    products = lin.apply_normal_batch(directions)

    jobs = _normal_jobs(fake)
    assert len(jobs) == 2 and all(job.directions is None for job in jobs)
    for direction, product in zip(directions, products):
        np.testing.assert_allclose(
            product.values, (lin.normal @ direction).values, rtol=1e-12
        )


def test_regularized_products_add_the_hessian_with_their_own_memo(tmp_path):
    fake = FakeImagingSite(seed=11)
    lin = _problem(tmp_path, fake).linearize()
    matrix = np.diag(np.linspace(1.0, 2.0, lin.space.size))
    regularization = _Hessian(lin.space, matrix)
    directions = [lin.space.random(seed) for seed in (1, 2)]

    plain = lin.apply_normal_batch(directions)
    regularized = lin.apply_normal_batch(directions, regularization=regularization)
    other = lin.apply_normal(
        directions[0], regularization=_Hessian(lin.space, 2.0 * matrix)
    )

    assert len(_normal_jobs(fake)) == 1  # the normal products are reused
    for p, r, d in zip(plain, regularized, directions):
        np.testing.assert_allclose(r.values, p.values + matrix @ d.values, rtol=1e-12)
    np.testing.assert_allclose(
        other.values, plain[0].values + 2.0 * matrix @ directions[0].values
    )


def test_stage_objective_hessian_actions_match_single_actions(tmp_path):
    fake = FakeImagingSite(seed=11)
    problem = _problem(tmp_path, fake)
    space = problem.space
    matrix = np.diag(np.linspace(0.5, 1.5, space.size))
    objective = _StageObjective(problem, space, Quadratic(matrix).bind(space), None, {})
    point = problem.vector().values
    directions = np.vstack([space.random(seed).values for seed in (3, 4, 5)])

    images = objective.hessian_actions(point, directions)
    jobs = len(_normal_jobs(fake))

    assert images.shape == directions.shape and jobs == 1
    for direction, image in zip(directions, images):
        np.testing.assert_allclose(
            image, objective.hessian_action(point, direction), rtol=1e-12
        )
    assert len(_normal_jobs(fake)) == jobs  # single actions reuse the batch
    assert objective.hessian_actions(point, np.zeros((0, space.size))).shape == (
        0,
        space.size,
    )


def _regularization_jobs(fake):
    return [job for job in fake.jobs if isinstance(job, RegularizationJob)]


def test_native_tikhonov_hessian_folds_into_the_normal_job(tmp_path):
    fake = FakeImagingSite(seed=7, support_masks={"model.vp": [1, 0, 1, 1]})
    problem = _problem(tmp_path, fake)
    lin = problem.linearize()
    _, native = bind_workflow_regularization(
        2 * im.Tikhonov(0.3), lin.space, problem, lin
    )
    directions = [lin.space.random(seed) for seed in (1, 2, 3)]
    prepared = len(_regularization_jobs(fake))

    regularized = lin.apply_normal_batch(directions, regularization=native)

    (job,) = _normal_jobs(fake)  # tasks and regularization postprocess together
    assert len(job.directions) == 3 and len(job.regularization["inputs"]) == 3
    assert len(_regularization_jobs(fake)) == prepared
    assert lin.apply_normal(directions[0], regularization=native) == regularized[0]
    plain = lin.apply_normal_batch(directions)
    assert len(_normal_jobs(fake)) == 1 and len(_regularization_jobs(fake)) == prepared
    operator = native.hessian_operator()  # the separate-job path, for reference
    for r, p, d in zip(regularized, plain, directions):
        np.testing.assert_allclose(
            r.values, p.values + np.asarray(operator @ d), rtol=1e-12
        )


def test_model_sampling_tikhonov_folds_only_at_its_own_linearization(tmp_path):
    fake = FakeImagingSite(seed=7)
    problem = _problem(tmp_path, fake)
    lin = problem.linearize()
    other = problem.linearize(lin.point.values * 0.5)
    spec = im.Tikhonov(0.3)
    _, own = bind_workflow_regularization(spec, lin.space, problem, lin)
    _, foreign = bind_workflow_regularization(spec, other.space, problem, other)
    for bound in (own, foreign):  # lengths lambda*v(x)/f sample the bound model
        for block in bound.context_identity.values():
            block["local_frequency_hz"] = 6.0
    directions = [lin.space.random(seed) for seed in (1, 2)]

    lin.apply_normal_batch(directions, regularization=own)
    assert _normal_jobs(fake)[-1].regularization is not None

    prepared = len(_regularization_jobs(fake))
    fresh = [lin.space.random(seed) for seed in (3, 4)]
    products = lin.apply_normal_batch(fresh, regularization=foreign)
    assert _normal_jobs(fake)[-1].regularization is None
    assert len(_regularization_jobs(fake)) == prepared + 2  # its own model's jobs
    for product, direction in zip(products, fresh):
        np.testing.assert_allclose(
            product.values,
            lin.apply_normal(direction).values
            + np.asarray(foreign.hessian_operator() @ direction),
            rtol=1e-12,
        )


def test_curvature_refresh_transition_costs_one_normal_job(tmp_path, monkeypatch):
    from frequensolve.imaging import _curvature_continuation as continuation
    from frequensolve.imaging import workflows
    from tests.test_imaging_curvature_continuation import TransferSite, _workflow

    def run(path):
        site = TransferSite()
        policy = im.CurvatureTransfer.refresh(rank=3)
        result = _workflow(path, site, policy=policy).run()
        return site, result

    site, folded = run(tmp_path / "folded")
    provenance = folded.stages[1].metrics["curvature_transfer"]
    count = len(
        next(
            arrays
            for request, arrays in site.transfer_calls
            if request["method"] == "curvature_refresh"
        )["directions"]
    )
    # Stage 0 measures its initial scale alone; the refresh transition evaluates
    # its retained directions and the scale probe Bg in one normal job.
    assert count > 1 and provenance["hessian_actions"] == count
    assert [len(job.directions or [None]) for job in _normal_jobs(site)] == [
        1,
        count + 1,
    ]
    assert provenance["initial_scale"] not in (None, 1.0)

    # The previous composition: a separate probe job, then transfer at scale * B.
    def separate(*args, scale_bounds=None, **options):
        policy, source, space, native, base, objective, point = args
        scale = 1.0
        if scale_bounds is not None:
            scale = continuation._initial_scale(objective, point, base, scale_bounds)
        factors, provenance = continuation._transfer_factors(
            policy, source, space, native, scale * base, objective, point, **options
        )
        if scale_bounds is not None:
            provenance["initial_scale"] = scale
        return continuation._CurvatureSeed(factors, provenance=provenance)

    monkeypatch.setattr(workflows, "_prepare_seed", separate)
    site, reference = run(tmp_path / "separate")
    assert [len(job.directions or [None]) for job in _normal_jobs(site)] == [
        1,
        1,
        count,
    ]
    assert reference.stages[1].metrics["curvature_transfer"][
        "initial_scale"
    ] == pytest.approx(provenance["initial_scale"], rel=1e-12)
    np.testing.assert_allclose(
        folded.state.values, reference.state.values, rtol=1e-9, atol=1e-12
    )


def test_stage_objective_folds_a_native_term_of_a_summed_regularization(tmp_path):
    from frequensolve.imaging.regularization import Sum, _BoundSum

    fake = FakeImagingSite(seed=7)
    problem = _problem(tmp_path, fake)
    lin = problem.linearize()
    smooth, native = bind_workflow_regularization(
        im.Quadratic(np.eye(lin.space.size), weight=0.2) + 2.0 * im.Tikhonov(0.3),
        lin.space,
        problem,
        lin,
    )
    summed = _BoundSum(
        Sum(smooth.regularization, native.regularization), lin.space, [smooth, native]
    )
    objective = _StageObjective(problem, lin.space, summed, None, {})
    point = lin.point.values
    directions = np.vstack([lin.space.random(seed).values for seed in (5, 6)])
    objective.loss(point)  # the stage start evaluates value and gradient anyway
    evaluated = len(_regularization_jobs(fake))

    images = objective.hessian_actions(point, directions)

    assert len(_normal_jobs(fake)) == 1
    assert len(_regularization_jobs(fake)) == evaluated  # folded into the normal job
    for direction, image in zip(directions, images):
        np.testing.assert_allclose(
            image, objective.hessian_action(point, direction), rtol=1e-12
        )


def test_batched_regularized_normal_job_round_trips_and_validates(tmp_path):
    from frequensolve.imaging.jobs import FWIOperatorJob
    from frequensolve.simulation.jobs.base import BaseJob
    from tests.test_imaging_jobs import _assert_valid

    fake = FakeImagingSite(seed=7)
    problem = _problem(tmp_path, fake)
    lin = problem.linearize()
    _, native = bind_workflow_regularization(im.Tikhonov(0.3), lin.space, problem, lin)
    lin.apply_normal_batch(
        [lin.space.random(seed) for seed in (1, 2)], regularization=native
    )
    (job,) = _normal_jobs(fake)

    payload = _assert_valid(job.to_fs())
    request = payload["control_sensitivities"]["Regularization"]
    assert len(payload["fwi_operator"]["directions"]) == len(request["inputs"]) == 2
    assert request["operation"] == "gradient"
    loaded = BaseJob.load(job.save())
    assert isinstance(loaded, FWIOperatorJob)
    assert loaded.directions == job.directions and loaded.covectors == job.covectors
    assert loaded.regularization == job.regularization
    staged = {pair[0] for pair in loaded.remote_input_files("/remote/project")}
    assert {*map(str, job.regularization["inputs"])} <= {*map(str, staged)}
