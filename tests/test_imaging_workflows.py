"""Workflow tests on the solver-free fake site (linear surrogate)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.geometry.grids import CartesianGrid
from frequensolve.imaging import (
    ControlSpace,
    ControlState,
    ControlVector,
    DepthProfile,
    ImageSet,
    ImagingProblem,
    Misfit,
    SourceParameters,
)
from frequensolve.imaging.results import FWIResult, StageResult
from frequensolve.imaging.workflows import (
    FWI,
    LBFGS,
    LSRTM,
    FWIIteration,
    NewtonCG,
    Stage,
    TimeReversalFocus,
    block_curvatures,
    curvature_scaling,
    rms_step_limit,
    rtm,
    sensitivity_kernel,
    sensitivity_kernel_job,
)
from frequensolve.inversion import (
    ContinuationSchedule,
    ContinuationStage,
    OptimizationCheckpoint,
    OptimizationHistory,
)
from tests.imaging_fakes import FakeImagingSite, layered_simulation
from tests.test_imaging_jobs import _assert_valid

pytestmark = pytest.mark.unit

FREQUENCIES = [4.0, 6.0]
BASE_VP = 1900.0
TIGHT = dict(gradient_tolerance=1e-10, objective_tolerance=0.0, step_tolerance=0.0)
EXACT = dict(initial_forcing=1e-8, minimum_forcing=1e-8, maximum_forcing=1e-8)


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------


class _TrackedDiagonal(im.Diagonal):
    """The real :class:`im.Diagonal` recording its bound instances and updates."""

    bound: list = []

    def bind(self, space):
        bound = super().bind(space)
        bound.updates = []
        update = bound.update

        def tracked(linearization, penalty=None):
            bound.updates.append(linearization)
            update(linearization, penalty=penalty)

        bound.update = tracked
        self.bound.append(bound)
        return bound


def _space(limits=None):
    return ControlSpace(
        vp=DepthProfile("vp", "sediment", count=5, limits=limits),
        rho=DepthProfile("rho", "sediment", count=3),
    )


def _problem(tmp_path, fake, *, name="fwi", subdir="project", **kwargs):
    sim = layered_simulation(tmp_path / subdir)
    observed = tmp_path / "observed.h5"
    if not observed.exists():
        # control-gradient jobs fingerprint their observed inputs
        observed.write_bytes(b"observed")
    options = dict(
        controls=_space(),
        observed={"surface": observed},
        frequencies=FREQUENCIES,
        site=fake,
        name=name,
        cache_capacity=4,
    )
    options.update(kwargs)
    return ImagingProblem(sim, **options)


def _surrogate(fake, problem, frequencies=None, active=None):
    """Return ``(J, d)`` of the surrogate for a view of ``problem``."""

    view = problem.restrict(frequencies=frequencies, active=active)
    lin = view.linearize(gradient=False)
    surrogate = fake.linearizations[lin.state_fingerprint]
    return surrogate.J, surrogate.d


def _least_squares(J, d, alpha=0.0, *, R=None, weights=None):
    """Regularized real least-squares solution of the surrogate.

    ``argmin 0.5 ||J m - d||_W^2 + 0.5 alpha ||m||^2 + 0.5 ||R m||^2`` via the
    dense normal equations ``(Re(J^H W J) + alpha I + R^T R) m = Re(J^H W d)``.
    ``weights`` is the per-row diagonal of ``W`` (default: unit).
    """

    W = np.ones(J.shape[0]) if weights is None else np.asarray(weights, float)
    normal = np.real(J.conj().T @ (W[:, None] * J)) + alpha * np.eye(J.shape[1])
    if R is not None:
        normal = normal + R.T @ R
    return np.linalg.solve(normal, np.real(J.conj().T @ (W * d)))


def _penalty_matrix(penalty, space):
    """Return the dense ``R`` of a quadratic penalty bound to ``space``."""

    return penalty.bind(space).operator().matrix.toarray()


def _row_weights(fake, problem, weights):
    """Expand per-frequency ``weights`` to the surrogate's data rows."""

    lin = problem.linearize(gradient=False)
    surrogate = fake.linearizations[lin.state_fingerprint]
    rows = np.ones(surrogate.J.shape[0])
    for weight, frequency in zip(weights, FREQUENCIES):
        rows[surrogate.rows(frequency)] = weight
    return rows


def _linearize_count(fake):
    return sum(1 for s in fake.submissions if s["action"] == "linearize")


@pytest.fixture
def fake():
    return FakeImagingSite(seed=11)


@pytest.fixture
def problem(tmp_path, fake):
    return _problem(tmp_path, fake)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def test_stage_constructors_and_validation():
    stages = Stage.bands([[3.0], [3.0, 5.0], [5.0, 8.0]], [10, 10, 15], active="vp")
    assert [s.frequencies for s in stages] == [(3.0,), (3.0, 5.0), (5.0, 8.0)]
    assert [s.iterations for s in stages] == [10, 10, 15]
    assert all(s.active == ("vp",) for s in stages)
    assert [s.iterations for s in Stage.bands([[3.0], [5.0]], 4)] == [4, 4]
    assert [s.label(i) for i, s in enumerate(stages)] == [
        "stage_01",
        "stage_02",
        "stage_03",
    ]

    laplace = Stage.frequency_laplace_bands(
        [
            {
                "name": "low",
                "frequencies_hz": [3.0, 4.0],
                "laplace_damping_hz": [1.0, 0.0],
            }
        ],
        iterations=[2, 3],
        active=["vp"],
    )
    assert [s.name for s in laplace] == ["low__laplace_00", "low__laplace_01"]
    assert laplace[0].frequencies == (complex(3.0, -1.0), complex(4.0, -1.0))
    assert laplace[1].frequencies == (3.0, 4.0)
    assert laplace[0].metadata["continuation_band"] == "low"
    assert [s.iterations for s in laplace] == [2, 3]
    continuation = laplace[0].to_continuation_stage(0)
    assert isinstance(continuation, ContinuationStage)
    assert continuation.max_iterations == 2 and continuation.metadata["active"] == [
        "vp"
    ]

    schedule = ContinuationSchedule.joint_frequency_laplace(
        [[3.0], [3.0, 5.0]], [0.5, 0.0], max_iterations=[4, None]
    )
    with pytest.raises(ValueError, match="no iteration budget"):
        Stage.from_schedule(schedule)
    wrapped = Stage.from_schedule(schedule, iterations=[None, 6])
    assert [s.iterations for s in wrapped] == [4, 6]
    assert wrapped[1].name == "stage_02"

    a = Stage([3.0], 2, active="vp", name="src")
    b = Stage([3.0], 2, active="rho")
    alternated = Stage.alternate([a, b], rounds=2)
    assert [s.name for s in alternated] == ["src_r1", None, "src_r2", None]
    assert [s.metadata["round"] for s in alternated] == [1, 1, 2, 2]

    huber = Stage([3.0], 1, loss="huber")
    assert huber.loss.kind == "huber"
    with pytest.raises(ValueError, match="either loss or misfit"):
        Stage([3.0], 1, loss="huber", misfit=Misfit.l2())
    with pytest.raises(ValueError, match="positive"):
        Stage([3.0], 0)
    with pytest.raises(ValueError, match="weights"):
        Stage([3.0, 4.0], 1, weights=[1.0])
    with pytest.raises(ValueError, match="unique"):
        Stage([3.0, 3.0], 1)


def test_stage_view_restricts_and_overrides_the_problem(problem, fake):
    stage = Stage([6.0], 3, active=["vp"], loss="huber", weights=[2.0])
    view = stage.view(problem)

    assert view.frequencies == [6.0] and view.space.blocks == ("model.vp",)
    assert view.misfit.loss.kind == "huber" and problem.misfit.loss.kind == "l2"
    assert view.weights == (2.0,) and problem.weights is None
    plan = view.dry_run()
    assert plan["job"]["Imaging"]["misfit"]["objective_terms"][0]["objective"] == {
        "kind": "huber",
        "delta": 1.5,
    }
    assert plan["job"]["control_sensitivities"]["weights"] == [2.0]
    assert "weights" not in view.linearize(gradient=False).job.to_fs().get(
        "control_sensitivities", {}
    )
    assert (
        view.linearize().fingerprint
        != problem.restrict(frequencies=[6.0], active=["vp"]).linearize().fingerprint
    )

    # weights scale value, gradient and normal on the objective side
    weighted = problem.restrict(weights=[1.0, 0.5])
    lin = weighted.linearize()
    plain = problem.linearize()
    J, d = _surrogate(fake, problem)
    rows4 = fake.linearizations[lin.state_fingerprint].rows(4.0)
    rows6 = fake.linearizations[lin.state_fingerprint].rows(6.0)
    residual = J @ np.zeros(8) - d
    expected = 0.5 * np.vdot(residual[rows4], residual[rows4]).real + 0.25 * (
        np.vdot(residual[rows6], residual[rows6]).real
    )
    assert lin.value == pytest.approx(expected)
    assert plain.value == pytest.approx(0.5 * np.vdot(residual, residual).real)
    np.testing.assert_allclose(
        lin.gradient.values,
        np.real(J[rows4].conj().T @ residual[rows4])
        + 0.5 * np.real(J[rows6].conj().T @ residual[rows6]),
    )
    assert weighted.check(taylor=False)["passed"]
    assert weighted.restrict(frequencies=[6.0]).weights == (0.5,)
    with pytest.raises(ValueError, match="either misfit or loss"):
        problem.restrict(misfit=Misfit.l2(), loss="huber")
    with pytest.raises(ValueError, match="support"):
        problem.restrict(support="never")


# ---------------------------------------------------------------------------
# optimizers
# ---------------------------------------------------------------------------


class _Quadratic:
    """Nonnegative quadratic ``0.5 (x - s)^T A (x - s)`` with ``A s = b``."""

    def __init__(self, A, b):
        self.A, self.b = A, b
        self.offset = 0.5 * float(b @ np.linalg.solve(A, b))

    def value(self, x):
        return 0.5 * float(x @ self.A @ x) - float(self.b @ x) + self.offset

    def gradient(self, x):
        return self.A @ x - self.b

    def hessian_action(self, x, dx):
        return self.A @ dx


def test_optimizer_configs_wrap_the_generic_minimizers_with_scaling_and_step_limit():
    rng = np.random.default_rng(3)
    root = rng.standard_normal((6, 6))
    A = root @ root.T + np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0])
    b = rng.standard_normal(6)
    solution = np.linalg.solve(A, b)
    quadratic = _Quadratic(A, b)

    lbfgs = LBFGS(memory=5, **TIGHT)
    assert lbfgs.options(7).history_size == 5 and lbfgs.options(7).max_iterations == 7
    result = lbfgs.solve(quadratic, np.zeros(6), max_iterations=200)
    np.testing.assert_allclose(result.model, solution, atol=1e-6)

    newton = NewtonCG(max_cg_iterations=6, **EXACT)
    assert newton.options().max_cg_iterations == 6
    result = newton.solve(quadratic, np.zeros(6), max_iterations=5)
    np.testing.assert_allclose(result.model, solution, atol=1e-8)
    assert result.iterations <= 2

    # diagonal change of variables leaves the minimizer invariant
    scale = 1.0 / np.sqrt(np.diag(A))
    scaled = lbfgs.solve(quadratic, np.zeros(6), scaling=scale, max_iterations=200)
    np.testing.assert_allclose(scaled.model, solution, atol=1e-6)
    np.testing.assert_allclose(scaled.gradient, A @ scaled.model - b, atol=1e-8)

    # an RMS step cap limits every accepted block update
    slices = (slice(0, 3), slice(3, 6))
    steps = []
    capped = LBFGS(step_limit=0.05).solve(
        quadratic,
        np.zeros(6),
        block_slices=slices,
        max_iterations=3,
        callback=lambda it: steps.append(it.step),
    )
    for step in steps[1:]:
        for block in slices:
            assert np.linalg.norm(step[block]) / np.sqrt(3) <= 0.05 + 1e-12
    assert capped.iterations == 3
    assert rms_step_limit(np.zeros(6), slices, 0.05) == np.inf
    assert rms_step_limit(np.ones(6), slices, 0.05) == pytest.approx(0.05)

    history = OptimizationHistory()
    lbfgs.solve(quadratic, np.zeros(6), history=history, max_iterations=2)
    assert history.iteration_count == 3  # initial point + two iterates
    first_order = SimpleNamespace(value=quadratic.value, gradient=quadratic.gradient)
    with pytest.raises(ValueError, match="hessian_action"):
        NewtonCG().solve(first_order, np.zeros(6))


# ---------------------------------------------------------------------------
# FWI
# ---------------------------------------------------------------------------


def _stages():
    return [
        Stage([4.0], 6, active=["vp"], name="low"),
        Stage([4.0, 6.0], 40, active=["vp", "rho"], name="full"),
    ]


def _stage_losses(history, index):
    return [
        r.loss.total
        for r in history.iterations
        if r.metrics["stage_index"] == index and r.metrics["stage"]
    ]


def test_fwi_lbfgs_reduces_the_objective_and_converges_on_the_surrogate(
    problem, fake, tmp_path
):
    J, d = _surrogate(fake, problem)
    solution = _least_squares(J, d)
    events = []
    fwi = FWI(
        problem,
        _stages(),
        optimizer=LBFGS(**TIGHT),
        history=tmp_path / "history.json",
        callback=events.append,
    )

    result = fwi.run()

    assert isinstance(result, FWIResult) and len(result.stages) == 2
    assert result.problem is problem and result.checkpoint is None
    assert [s.name for s in result.stages] == ["low", "full"]
    assert result.stages[0].active == ("model.vp",)
    assert result.stages[1].active == ("model.vp", "model.rho")
    assert result.stages[0].space.size == 5 and result.stages[1].space.size == 8
    np.testing.assert_allclose(result.state.values, solution, atol=1e-5)
    np.testing.assert_allclose(problem.vector().values, solution, atol=1e-5)
    np.testing.assert_allclose(result.stages[1].vector.values, solution, atol=1e-5)
    assert result.success and result.loss.regularization == 0.0
    assert result.history.status == "converged"
    assert (tmp_path / "history.json").is_file()
    for index in (0, 1):
        losses = _stage_losses(result.history, index)
        assert len(losses) >= 2
        assert np.all(np.diff(losses) <= 1e-12)
    assert result.stages[1].final_loss.total < result.stages[1].initial_loss.total
    assert result.stages[0].initial_loss.total > result.stages[0].final_loss.total
    # the second stage started from the first stage's accepted vp
    first_full = next(
        r for r in result.history.iterations if r.metrics["stage_index"] == 1
    )
    assert first_full.metrics["stage"] == "full"
    assert json.loads(first_full.metrics["frequencies"]) == [[4.0, 0.0], [6.0, 0.0]]
    assert first_full.metrics["active"] == "model.vp,model.rho"
    assert first_full.metrics["optimizer"] == "lbfgs"
    assert all(isinstance(e, FWIIteration) for e in events)
    assert events[-1].stage_index == 1 and events[-1].model.size == 8
    assert events[0].iteration == 0 and events[0].stage_iteration == 0
    # the result's simulation carries the inverted coefficients
    installed = result.simulation
    sediment = next(s for s in installed.model.subdomains if s.name == "sediment")
    np.testing.assert_allclose(
        sediment.properties["vp"].control.coefficients, solution[:5], atol=1e-5
    )
    assert result.vector().size == 8

    saved = result.save(tmp_path / "result")
    loaded = FWIResult.load(saved, problem)
    np.testing.assert_allclose(loaded.state.values, result.state.values)
    assert loaded.history.iteration_count == result.history.iteration_count
    assert [s.to_fs() for s in loaded.stages] == [s.to_fs() for s in result.stages]
    assert loaded.stages[1].vector is None and loaded.problem is problem
    assert isinstance(loaded.stages[0], StageResult)


def test_fwi_newton_cg_reaches_the_least_squares_solution_in_few_iterations(
    problem, fake
):
    J, d = _surrogate(fake, problem)
    solution = _least_squares(J, d)
    fwi = FWI(
        problem,
        Stage(FREQUENCIES, 6, name="joint"),
        optimizer=NewtonCG(max_cg_iterations=20, **EXACT),
    )

    result = fwi.run()

    stage = result.stages[0]
    assert stage.success and stage.iterations <= 3
    np.testing.assert_allclose(result.state.values, solution, atol=1e-6)
    losses = _stage_losses(result.history, 0)
    assert np.all(np.diff(losses) <= 1e-12)
    assert stage.metrics["optimizer"] == "newton_cg"
    assert (
        stage.metrics["cg_iterations"] >= 1 and stage.metrics["hessian_products"] >= 1
    )
    normal = [s for s in fake.submissions if s["action"] == "normal"]
    assert normal  # Hessian actions ran Sauce's normal action


def test_fwi_records_penalty_terms_and_solves_the_damped_problem(problem, fake):
    J, d = _surrogate(fake, problem)
    alpha = 5.0
    damped = _least_squares(J, d, alpha)
    ridge = im.Quadratic(np.eye(8), weight=alpha)
    fwi = FWI(
        problem,
        Stage(FREQUENCIES, 5),
        optimizer=NewtonCG(**EXACT),
        penalty=ridge,
    )

    result = fwi.run()

    np.testing.assert_allclose(result.state.values, damped, atol=1e-6)
    final = result.stages[0].final_loss
    assert final.regularization == pytest.approx(0.5 * alpha * float(damped @ damped))
    assert final.data > 0.0 and final.total == final.data + final.regularization
    evaluations = result.history.evaluations
    assert all(r.loss.regularization >= 0.0 for r in evaluations)
    assert evaluations[-1].metrics["penalty"] == "Quadratic"
    # a stage-level penalty overrides the workflow penalty
    override = FWI(
        problem,
        Stage(FREQUENCIES, 5, penalty=im.Tikhonov(0.0)),
        optimizer=NewtonCG(**EXACT),
        penalty=ridge,
    ).run()
    np.testing.assert_allclose(override.state.values, _least_squares(J, d), atol=1e-6)
    assert override.stages[0].final_loss.regularization == 0.0
    assert override.history.evaluations[-1].metrics["penalty"] == "Tikhonov"


@pytest.mark.parametrize(
    "optimizer, atol",
    [(NewtonCG(max_cg_iterations=30, **EXACT), 1e-6), (LBFGS(**TIGHT), 1e-3)],
    ids=["newton_cg", "lbfgs"],
)
def test_fwi_tikhonov_with_diagonal_preconditioner_solves_the_regularized_problem(
    problem, fake, optimizer, atol
):
    J, d = _surrogate(fake, problem)
    # penalties are scale free (unit-interval seminorms), so an O(1e-2)
    # alpha is a meaningful weight against the surrogate's data term
    alpha = 5.0e-2
    R = _penalty_matrix(im.Tikhonov(alpha), problem.space)
    assert R.shape == (4 + 2, 8)  # first differences of a 5- and a 3-node profile
    expected = _least_squares(J, d, R=R)
    assert not np.allclose(expected, _least_squares(J, d), atol=5e-3)
    stages = [Stage([4.0], 6, active=["vp"], name="low"), Stage(FREQUENCIES, 60)]
    fwi = FWI(
        problem,
        stages,
        optimizer=optimizer,
        penalty=im.Tikhonov(alpha),
        preconditioner=im.Diagonal(probe_count=4),
    )

    result = fwi.run()

    assert len(result.stages) == 2 and result.success
    np.testing.assert_allclose(result.state.values, expected, atol=atol)
    final = result.stages[1].final_loss
    assert final.regularization == pytest.approx(
        0.5 * float(np.linalg.norm(R @ result.state.values) ** 2)
    )
    assert final.total == pytest.approx(
        _least_squares_value(J, d, result.state.values) + final.regularization
    )
    assert result.history.evaluations[-1].metrics["penalty"] == "Tikhonov"
    for index in (0, 1):
        losses = _stage_losses(result.history, index)
        assert len(losses) >= 2 and np.all(np.diff(losses) <= 1e-9)
    # the Rademacher probes ran Sauce's normal action for both stages
    normal = [s for s in fake.submissions if s["action"] == "normal"]
    assert len(normal) >= 8


@pytest.mark.parametrize(
    "optimizer, atol",
    [(NewtonCG(max_cg_iterations=30, **EXACT), 1e-6), (LBFGS(**TIGHT), 1e-4)],
    ids=["newton_cg", "lbfgs"],
)
def test_fwi_curvature_scaling_leaves_the_minimizer_invariant(
    problem, fake, optimizer, atol
):
    J, d = _surrogate(fake, problem)
    solution = _least_squares(J, d)
    before = len(fake.submissions)
    result = FWI(problem, _stages(), optimizer=optimizer, scaling="curvature").run()

    np.testing.assert_allclose(result.state.values, solution, atol=atol)
    assert result.success and result.stages[1].active == ("model.vp", "model.rho")
    # one Rademacher JVP per block at every stage start
    jvp = [s for s in fake.submissions[before:] if s["action"] == "jvp"]
    assert len(jvp) >= 3


def test_stage_weights_scale_the_objective_and_its_minimizer(problem, fake):
    weights = [1.0, 0.25]
    J, d = _surrogate(fake, problem)
    rows = _row_weights(fake, problem, weights)
    lin = fake.linearizations[problem.linearize(gradient=False).state_fingerprint]
    residual = J @ np.zeros(8) - d
    per_frequency = [
        0.5 * float(np.vdot(residual[lin.rows(f)], residual[lin.rows(f)]).real)
        for f in FREQUENCIES
    ]
    weighted = _least_squares(J, d, weights=rows)
    assert not np.allclose(weighted, _least_squares(J, d), atol=1e-3)

    result = FWI(
        problem,
        Stage(FREQUENCIES, 6, weights=weights),
        optimizer=NewtonCG(max_cg_iterations=30, **EXACT),
    ).run()

    stage = result.stages[0]
    assert stage.initial_loss.data == pytest.approx(
        sum(w * value for w, value in zip(weights, per_frequency))
    )
    assert stage.initial_loss.data < sum(per_frequency)
    np.testing.assert_allclose(result.state.values, weighted, atol=1e-6)
    final = J @ result.state.values - d
    assert stage.final_loss.data == pytest.approx(
        0.5 * float(np.real(np.vdot(final, rows * final)))
    )


def test_fwi_respects_bounds_and_clips_iterates(tmp_path, fake):
    unbounded = _problem(tmp_path, fake, subdir="free")
    J, d = _surrogate(fake, unbounded)
    solution = _least_squares(J, d)
    limit = 0.5 * float(np.max(np.abs(solution[:5])))
    # limits are physical values; the sediment vp baseline is 1900 m/s
    bounded = _problem(
        tmp_path,
        fake,
        subdir="bounded",
        controls=_space(limits=(BASE_VP - limit, BASE_VP + limit)),
    )
    lower, upper = bounded.space.bounds
    np.testing.assert_allclose(lower[:5], -limit, atol=1e-9)
    np.testing.assert_allclose(upper[:5], limit, atol=1e-9)
    limit = float(upper[0])
    assert np.all(np.isinf(lower[5:])) and np.all(np.isinf(upper[5:]))
    iterates = []
    fwi = FWI(
        bounded,
        Stage(FREQUENCIES, 30),
        optimizer=LBFGS(**TIGHT),
        callback=lambda e: iterates.append(e.model.values),
    )

    result = fwi.run()

    for model in iterates:
        assert np.all(model[:5] <= limit + 1e-12) and np.all(
            model[:5] >= -limit - 1e-12
        )
    vp = result.state.values[:5]
    assert np.any(np.isclose(np.abs(vp), limit))
    assert result.stages[0].final_loss.total > _least_squares_value(J, d, solution)
    np.testing.assert_array_equal(
        vp, ControlVector(result.state.values, bounded.space).clip().values[:5]
    )


def _least_squares_value(J, d, m):
    residual = J @ m - d
    return 0.5 * float(np.vdot(residual, residual).real)


def test_fwi_step_limit_caps_the_block_rms_update(tmp_path, fake):
    free = _problem(tmp_path, fake, subdir="free")
    free_steps = []
    FWI(
        free,
        Stage(FREQUENCIES, 1),
        callback=lambda e: free_steps.append(e.diagnostics.step),
    ).run()
    free_rms = np.linalg.norm(free_steps[-1][:5]) / np.sqrt(5)
    cap = 0.25 * free_rms

    capped = _problem(tmp_path, fake, subdir="capped")
    steps = []
    result = FWI(
        capped,
        Stage(FREQUENCIES, 3),
        step_limit=cap,
        callback=lambda e: steps.append(e.diagnostics.step),
    ).run()

    assert result.stages[0].iterations == 3
    for step in steps[1:]:
        assert np.linalg.norm(step[:5]) / np.sqrt(5) <= cap + 1e-12
        assert np.linalg.norm(step[5:]) / np.sqrt(3) <= cap + 1e-12
    assert np.linalg.norm(steps[1]) < np.linalg.norm(free_steps[-1])


def test_fwi_checkpoints_every_iteration_and_resumes_after_an_interruption(
    tmp_path, fake
):
    class Interrupt(RuntimeError):
        pass

    def interrupt(event):
        if event.stage_index == 0 and event.stage_iteration == 2:
            raise Interrupt("simulated crash")

    problem = _problem(tmp_path, fake, subdir="resume")
    checkpoint = tmp_path / "run" / "fwi.ckpt.h5"
    options = dict(
        optimizer=LBFGS(**TIGHT),
        checkpoint=checkpoint,
        history=tmp_path / "run" / "history.json",
    )
    fwi = FWI(problem, _stages(), callback=interrupt, **options)
    with pytest.raises(Interrupt):
        fwi.run()

    assert checkpoint.is_file() and fwi.state_path.is_file()
    saved = OptimizationCheckpoint.load(checkpoint)
    assert saved.metadata["stage_index"] == 0 and saved.metadata["stage_iteration"] == 2
    assert saved.metadata["stage_completed"] is False
    assert saved.metadata["active"] == "model.vp"
    assert saved.metadata["control_ids"] == "model.vp,model.rho"
    assert saved.model.size == 5
    history = OptimizationHistory.load(tmp_path / "run" / "history.json")
    assert history.status == "stopped" and "Interrupt" in history.message
    interrupted_records = history.iteration_count
    linearizations_before = _linearize_count(fake)

    resumed = FWI(problem, _stages(), **options).run(resume=True)

    new_linearizations = _linearize_count(fake) - linearizations_before
    assert resumed.history.status == "converged"
    assert resumed.history.iteration_count > interrupted_records
    assert resumed.stages[0].resumed and not resumed.stages[0].skipped
    assert resumed.stages[0].stage_iteration == 6
    assert resumed.stages[0].iterations == 4
    assert not resumed.stages[1].resumed
    J, d = _surrogate(fake, problem)
    np.testing.assert_allclose(resumed.state.values, _least_squares(J, d), atol=1e-5)
    final = OptimizationCheckpoint.load(checkpoint)
    assert final.metadata["stage_index"] == 1 and final.metadata["stage_completed"]
    assert resumed.checkpoint == checkpoint

    # a fresh run of the same problem needs more linearizations than the resume
    reference = FakeImagingSite(seed=11)
    fresh = _problem(tmp_path, reference, subdir="fresh")
    FWI(fresh, _stages(), optimizer=LBFGS(**TIGHT)).run()
    assert new_linearizations < _linearize_count(reference)

    # a completed run resumes to a no-op result
    again = FWI(problem, _stages(), **options).run(resume=True)
    assert all(s.skipped for s in again.stages)
    assert again.stages[1].final_loss.total == pytest.approx(
        resumed.stages[1].final_loss.total
    )
    np.testing.assert_allclose(again.state.values, resumed.state.values)


def test_fwi_resume_rejects_a_mismatched_checkpoint(tmp_path, fake):
    def interrupt(event):
        if event.stage_iteration == 1:
            raise KeyboardInterrupt

    problem = _problem(tmp_path, fake, subdir="mismatch")
    checkpoint = tmp_path / "fwi.ckpt.h5"
    with pytest.raises(KeyboardInterrupt):
        FWI(problem, _stages(), checkpoint=checkpoint, callback=interrupt).run()
    assert checkpoint.is_file()

    changed_active = [Stage([4.0], 6, active=["vp", "rho"]), _stages()[1]]
    with pytest.raises(ValueError, match="activates"):
        FWI(problem, changed_active, checkpoint=checkpoint).run()
    changed_frequencies = [Stage([6.0], 6, active=["vp"]), _stages()[1]]
    with pytest.raises(ValueError, match="frequencies"):
        FWI(problem, changed_frequencies, checkpoint=checkpoint).run()

    other = _problem(tmp_path, fake, subdir="other", name="other")
    with pytest.raises(ValueError, match="belongs to problem"):
        FWI(other, _stages(), checkpoint=checkpoint).run()
    different_layout = _problem(
        tmp_path,
        fake,
        subdir="layout",
        controls=ControlSpace(
            vp=DepthProfile("vp", "sediment", count=4),
            rho=DepthProfile("rho", "sediment", count=3),
        ),
    )
    with pytest.raises(ValueError, match="control sizes"):
        FWI(different_layout, _stages(), checkpoint=checkpoint).run()
    with pytest.raises(ValueError, match="identity"):
        FWI(
            _problem(tmp_path, fake, subdir="mismatch", misfit=Misfit.huber()),
            _stages(),
            checkpoint=checkpoint,
        ).run()

    # resume=False ignores the checkpoint and starts over
    fresh = FWI(problem, _stages(), checkpoint=checkpoint).run(resume=False)
    assert not fresh.stages[0].resumed and fresh.stages[0].stage_iteration <= 6


def test_fwi_transition_refreshes_state_and_support_masks(tmp_path):
    fake = FakeImagingSite(seed=5, support_masks={"model.vp": [1, 0, 1, 1, 0]})
    problem = _problem(tmp_path, fake, min_support=0.05)
    preconditioner = _TrackedDiagonal(probes="unit")
    preconditioner.bound.clear()
    fwi = FWI(
        problem,
        [
            Stage([4.0], 3, active=["vp"]),
            Stage(FREQUENCIES, 3, active=["vp", "rho"]),
        ],
        optimizer=LBFGS(preconditioner_refresh=2),
        preconditioner=preconditioner,
    )

    result = fwi.run()

    assert result.stages[0].space.size == 3 and result.stages[1].space.size == 6
    assert result.stages[0].metrics["frozen_dofs"] == 2
    np.testing.assert_array_equal(result.state.values[[1, 4]], 0.0)
    assert np.all(result.state.values[[0, 2, 3]] != 0.0)
    assert np.all(result.state.values[5:] != 0.0)
    # stage two started from stage one's accepted vp
    stage_one = result.stages[0].vector.values
    first_full = next(
        r for r in result.history.iterations if r.metrics["stage_index"] == 1
    )
    initial_full = result.stages[1].initial_loss.total
    assert first_full.loss.total == pytest.approx(initial_full)
    J, d = _surrogate(fake, problem, frequencies=[4.0], active=["vp"])
    residual = J @ np.array([stage_one[0], 0.0, stage_one[1], stage_one[2], 0.0]) - d
    assert result.stages[0].final_loss.data == pytest.approx(
        0.5 * float(np.vdot(residual, residual).real)
    )
    # one bound preconditioner per stage, updated at the start and every 2 iterations
    assert len(preconditioner.bound) == 2
    assert len(preconditioner.bound[0].updates) == 2
    assert preconditioner.bound[0].space.size == 3
    assert preconditioner.bound[1].space.size == 6
    # the unit probes reproduced the exact Gauss-Newton diagonal of the
    # masked stage-one space at the refresh point
    first = preconditioner.bound[0]
    refreshed = first.updates[-1]
    assert refreshed.space.size == 3 and refreshed is not first.updates[0]
    J_masked = fake.linearizations[refreshed.state_fingerprint].J[:, [0, 2, 3]]
    np.testing.assert_allclose(
        first.estimate.data, np.sum(np.abs(J_masked) ** 2, axis=0), rtol=1e-8
    )
    assert first.estimate.probe_count == 3 and first.diagonal.shape == (3,)


def test_fwi_solve_stage_and_scaling_options(problem, fake):
    fwi = FWI(problem, _stages(), scaling="curvature")
    stage = fwi.stages[0]

    result = fwi.solve_stage(stage)

    assert isinstance(result, StageResult) and result.name == "low"
    assert result.iterations >= 1 and fwi.results == [result]
    assert fwi.history is not None and fwi.history.iteration_count >= 2
    np.testing.assert_allclose(problem.state.values[:5], result.vector.values)

    lin = problem.restrict(frequencies=[4.0], active=["vp"]).linearize()
    curvatures = block_curvatures(lin, seed=1)
    J = fake.linearizations[lin.state_fingerprint].J
    diagonal = np.real(np.einsum("ij,ij->j", J.conj(), J))
    assert curvatures["model.vp"] == pytest.approx(float(np.mean(diagonal)), rel=0.6)
    scale = curvature_scaling(curvatures, lin.space, max_ratio=10.0)
    assert scale.shape == (5,) and np.allclose(
        scale, 1.0 / np.sqrt(curvatures["model.vp"])
    )
    mapped = FWI(problem, [stage], scaling={"vp": 4.0}).solve_stage(stage)
    assert mapped.success
    with pytest.raises(ValueError, match="scaling"):
        FWI(problem, [stage], scaling="diag")


# ---------------------------------------------------------------------------
# LSRTM, rtm
# ---------------------------------------------------------------------------


def test_lsrtm_cg_and_lsqr_recover_the_surrogate_image(problem, fake):
    J, d = _surrogate(fake, problem)
    solution = _least_squares(J, d)

    cg = LSRTM(problem, iterations=60, method="cg", tolerance=1e-12)
    image = cg.run()
    assert isinstance(image, ControlVector) and image.space.equivalent(problem.space)
    np.testing.assert_allclose(image.values, solution, atol=1e-6)
    assert cg.info["method"] == "cg" and cg.info["converged"]

    seen = []
    lsqr = LSRTM(
        problem, iterations=100, method="lsqr", tolerance=1e-12, callback=seen.append
    )
    image = lsqr.run()
    np.testing.assert_allclose(image.values, solution, atol=1e-6)
    assert lsqr.info["converged"] and seen and seen[0] is image

    # damping and a quadratic penalty (operator rows) agree between the two solvers
    damped = _least_squares(J, d, 2.0 + 3.0)
    ridge = im.Quadratic(np.eye(8), weight=3.0)
    cg_image = LSRTM(
        problem,
        iterations=80,
        method="cg",
        damping=2.0,
        penalty=ridge,
        tolerance=1e-12,
    ).run()
    lsqr_image = LSRTM(
        problem,
        iterations=200,
        method="lsqr",
        damping=2.0,
        penalty=ridge,
        tolerance=1e-12,
    ).run()
    np.testing.assert_allclose(cg_image.values, damped, atol=1e-6)
    np.testing.assert_allclose(lsqr_image.values, damped, atol=1e-6)

    # linearizing away from the origin images the update, not the model
    v0 = problem.space.random(2)
    update = LSRTM(problem, iterations=60, method="cg", tolerance=1e-12).run(v0)
    np.testing.assert_allclose(update.values, solution - v0.values, atol=1e-6)
    with pytest.raises(ValueError, match="method"):
        LSRTM(problem, method="gmres")


def test_lsrtm_with_tikhonov_matches_the_dense_regularized_solve(problem, fake):
    J, d = _surrogate(fake, problem)
    penalty = im.Tikhonov(1.0e-2, order=2)
    R = _penalty_matrix(penalty, problem.space)
    assert R.shape == (3 + 1, 8)  # second differences of the two profiles
    expected = _least_squares(J, d, R=R)
    assert not np.allclose(expected, _least_squares(J, d), atol=1e-3)

    cg = LSRTM(problem, iterations=80, method="cg", penalty=penalty, tolerance=1e-12)
    image = cg.run()
    np.testing.assert_allclose(image.values, expected, atol=1e-6)
    assert cg.info["converged"]
    lsqr = LSRTM(
        problem, iterations=200, method="lsqr", penalty=penalty, tolerance=1e-12
    )
    np.testing.assert_allclose(lsqr.run().values, expected, atol=1e-6)

    # a TV penalty has no operator rows: cg solves the lagged-diffusivity
    # normal equations at the origin, lsqr refuses
    tv = im.TV(1.0e-2, epsilon=1.0e-2)
    hessian = tv.bind(problem.space).hessian_operator(problem.space.zeros())
    dense = np.column_stack([np.asarray(hessian @ e).reshape(-1) for e in np.eye(8)])
    assert np.linalg.norm(dense, 2) > 1.0
    expected_tv = np.linalg.solve(
        np.real(J.conj().T @ J) + dense, np.real(J.conj().T @ d)
    )
    assert not np.allclose(expected_tv, _least_squares(J, d), atol=1e-2)
    tv_image = LSRTM(problem, iterations=80, method="cg", penalty=tv, tolerance=1e-12)
    np.testing.assert_allclose(tv_image.run().values, expected_tv, atol=1e-6)
    with pytest.raises(ValueError, match="operator"):
        LSRTM(problem, method="lsqr", penalty=tv).run()


def test_rtm_returns_the_misfit_gradient(problem):
    v = problem.space.random(4)
    image = rtm(problem, v)
    assert image is problem.gradient(v)
    assert image.space.equivalent(problem.space)


# ---------------------------------------------------------------------------
# smoothing through the fake postprocess
# ---------------------------------------------------------------------------


def test_smooth_runs_the_smooth_job_as_a_fake_postprocess(problem, fake):
    v = problem.space.random(3)
    before = len(fake.submissions)

    smoothed = im.smooth(
        v, im.Smoothing(kind="tikhonov", wavelength_fraction=0.5), problem
    )

    assert isinstance(smoothed, ControlVector) and smoothed.space is problem.space
    np.testing.assert_allclose(smoothed.values, v.values)  # identity smoothing
    submission = fake.submissions[before:]
    assert len(submission) == 1 and submission[0]["postprocess_only"]
    assert submission[0]["job"].startswith("fwi_smooth_")
    job = fake.jobs[-1]
    assert isinstance(job, im.SmoothJob) and job.gradient_file(raw=True).is_file()
    payload = job.to_fs()["control_sensitivities"]
    assert payload["active"] == ["vp", "rho"] and payload["Smoothing"]["lambda"] == 0.5
    # the linearize gradient smooths the same way
    lin = problem.linearize()
    again = im.smooth(lin.gradient, im.Smoothing(kind="tv"), problem)
    np.testing.assert_allclose(again.values, lin.gradient.values)


# ---------------------------------------------------------------------------
# sensitivity kernels and time-reversal focus
# ---------------------------------------------------------------------------


def test_sensitivity_kernel_runs_on_the_fake_and_returns_an_image_set(problem, fake):
    grid = CartesianGrid(n=[4, 3], x0=[0.0, 0.0], x1=[4000.0, 1500.0])

    images = sensitivity_kernel(
        problem, grid, properties=["vp", "rho"], weights=[1.0, 0.5], keep="adjoint"
    )

    assert isinstance(images, ImageSet)
    assert images.shape == (3, 4) == grid.shape and images.parts == 2
    np.testing.assert_array_equal(images.f_list, FREQUENCIES)
    raw = images.raw
    assert list(raw.data_vars) == ["vp", "rho"]
    assert raw["vp"].dims == ("z", "x") and raw["vp"].shape == (3, 4)
    np.testing.assert_allclose(raw["vp"].coords["x"], np.linspace(0.0, 4000.0, 4))
    np.testing.assert_allclose(raw["rho"].coords["z"], np.linspace(0.0, 1500.0, 3))
    parts = [images.read_images("raw", part=task) for task in (1, 2)]
    for index, name in enumerate(("vp", "rho")):
        for part, frequency in zip(parts, FREQUENCIES):
            np.testing.assert_allclose(
                part[name].values,
                FakeImagingSite.image_values(grid, index, frequency).reshape(3, 4),
            )
        # the aggregate is the weighted stack of the parts
        np.testing.assert_allclose(
            raw[name].values, parts[0][name].values + 0.5 * parts[1][name].values
        )
    assert raw["rho"].values[0, 0] == pytest.approx(2.0 * (5.0 + 0.5 * 7.0))
    submission = fake.submissions[-1]
    assert submission["workflow"] == "rtm" and not submission["postprocess_only"]
    job = fake.jobs[-1]
    assert isinstance(job, im.ImageKernelJob)
    assert job.image_output_exists() and job.postprocess_part_outputs_exist()
    assert job.image_file() == images.image_file() == job.save_path / "image.h5"
    assert job.image_file(2) == job.save_path / "image_2.h5"

    # a single-frequency kernel at a moved point images the installed state
    single = sensitivity_kernel(
        problem, grid, frequencies=[6.0], v=problem.space.ones(), condition="up_down"
    )
    assert single.parts == 1 and list(single.raw.data_vars) == ["vp"]
    np.testing.assert_array_equal(single.f_list, [6.0])
    np.testing.assert_allclose(
        single.raw["vp"].values,
        FakeImagingSite.image_values(grid, 0, 6.0).reshape(3, 4),
    )


def test_time_reversal_focus_objective_runs_on_the_fake(problem, fake):
    softening = 12.5
    weights = [1.0, 0.5]
    focus = TimeReversalFocus(problem, softening, weights=weights)
    sizes = {"model.vp": 5, "model.rho": 3}
    J, d, space = fake.surrogate(problem.simulation, list(sizes), sizes, FREQUENCIES)

    def expected(m):
        value, gradient = 0.0, np.zeros(8)
        for weight, frequency in zip(weights, FREQUENCIES):
            rows = np.concatenate(
                [layout.indices for layout in space.term_layouts(frequency=frequency)]
            )
            residual = J[rows] @ m - d[rows]
            value += weight * softening * 0.5 * float(np.vdot(residual, residual).real)
            gradient += weight * softening * np.real(J[rows].conj().T @ residual)
        return value, gradient

    before = len(fake.submissions)
    value, gradient = focus.objective()

    assert isinstance(value, float) and isinstance(gradient, ControlVector)
    assert gradient.space is problem.space
    authored_value, authored_gradient = expected(np.zeros(8))
    assert value == pytest.approx(authored_value)
    np.testing.assert_allclose(gradient.values, authored_gradient)
    assert value > 0.0 and len(fake.submissions) == before + 1
    assert fake.submissions[-1]["workflow"] == "focus"
    job = focus.job()
    assert job.objective_value == pytest.approx(value)
    assert job.gradient_file(raw=True).is_file() and job.objective_file(2).is_file()
    # results are cached per state; a moved point runs a job with ``current``
    assert focus.objective() == (value, gradient)
    assert len(fake.submissions) == before + 1
    v = problem.space.ones()
    moved_value, moved_gradient = expected(np.ones(8))
    assert focus.value(v) == pytest.approx(moved_value)
    np.testing.assert_allclose(focus.gradient(v).values, moved_gradient)
    assert len(fake.submissions) == before + 2
    assert focus.job(v).current is not None
    # unit weights halve nothing: the 6 Hz task enters fully
    plain = TimeReversalFocus(problem, softening).objective()[0]
    assert plain > value


# ---------------------------------------------------------------------------
# sensitivity kernels and time-reversal focus (payload level)
# ---------------------------------------------------------------------------


def test_sensitivity_kernel_job_builds_a_schema_valid_zero_data_kernel(problem, fake):
    grid = CartesianGrid(n=[4, 3], x0=[0.0, 0.0], x1=[4000.0, 1500.0])

    job = sensitivity_kernel_job(
        problem, grid, properties=["vp", "rho"], frequencies=[6.0], keep="adjoint"
    )

    payload = _assert_valid(job.to_fs())
    assert job.f_list == [6.0] and job.workflow == "rtm"
    assert [image["IC"] for image in payload["Imaging"]["images"]] == [
        "fwi:acoustic",
        "fwi:acoustic",
    ]
    assert [image["property"] for image in payload["Imaging"]["images"]] == [
        "vp",
        "rho",
    ]
    assert payload["Imaging"]["misfit"]["receiver_groups"] == [
        {"name": "surface", "projection": {"kind": "identity"}, "observed": None}
    ]
    assert payload["Imaging"]["data_path"] is None
    assert job.field_retention == "adjoint" and job.observed is None
    assert job.simulation is problem.simulation
    plan = problem.backend.dry_run(job)
    assert plan["workflow"] == "rtm" and plan["n_tasks"] == 1 and fake.submissions == []

    with_data = sensitivity_kernel_job(
        problem, grid, condition="up_down", observed=True, v=problem.space.ones()
    )
    payload = _assert_valid(with_data.to_fs())
    assert payload["Imaging"]["images"][0]["IC"] == "up_down"
    assert with_data.observed == {"surface": problem.observed_groups[0].observed}
    assert with_data.simulation is not problem.simulation


def test_time_reversal_focus_builds_schema_valid_focus_jobs(problem, tmp_path, fake):
    focus = TimeReversalFocus(problem, 12.5, weights=[1.0, 0.5])
    assert focus.active == ["vp", "rho"]

    job = focus.job()
    payload = _assert_valid(job.to_fs())
    assert payload["workflow"] == "focus"
    assert payload["focus"]["softening"] == 12.5 and payload["focus"]["kind"] == "trfwi"
    assert payload["control_sensitivities"]["active"] == ["vp", "rho"]
    assert payload["control_sensitivities"]["weights"] == [1.0, 0.5]
    assert "current" not in payload["control_sensitivities"]
    assert payload["Imaging"]["misfit"]["receiver_groups"][0]["name"] == "surface"
    assert focus.job() is job

    moved = focus.job(problem.space.ones())
    payload = _assert_valid(moved.to_fs())
    assert payload["control_sensitivities"]["current"].endswith("current.h5")
    assert moved.current.is_file()
    plan = focus.dry_run(problem.space.ones())
    assert plan["workflow"] == "focus" and fake.submissions == []

    weft = TimeReversalFocus(problem.restrict(active=["vp"]), 2.0, kind="weft")
    assert weft.job().active == ["vp"]
    assert _assert_valid(weft.job().to_fs())["focus"]["kind"] == "weft"
    with pytest.raises(ValueError, match="softening"):
        TimeReversalFocus(problem, -1.0)

    sim = layered_simulation(tmp_path / "sources")
    sources = ImagingProblem(
        sim,
        controls=ControlSpace(
            vp=DepthProfile("vp", "sediment", count=2), src=SourceParameters()
        ),
        observed={"surface": "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
        name="src",
    )
    with pytest.raises(ValueError, match="model"):
        TimeReversalFocus(sources.restrict(active=["src"]), 1.0).job()


def _source_problem(tmp_path, fake):
    """Problem over ``vp`` plus per-source signatures (the full bound space)."""

    return _problem(
        tmp_path,
        fake,
        name="src",
        subdir="sources",
        controls=ControlSpace(
            vp=DepthProfile("vp", "sediment", count=5),
            src=SourceParameters(signature=True),
        ),
    )


def test_kernel_and_focus_run_on_a_problem_with_source_blocks(tmp_path, fake):
    problem = _source_problem(tmp_path, fake)
    assert problem.space.blocks == (
        "model.vp",
        "source.1.signature",
        "source.2.signature",
    )
    grid = CartesianGrid(n=[4, 3], x0=[0.0, 0.0], x1=[4000.0, 1500.0])
    update = problem.vector().values.copy()
    update[:5] = np.linspace(0.1, 0.5, 5)

    # kernels at the authored point and after a vp update (simulation_at)
    images = sensitivity_kernel(problem, grid)
    assert list(images.raw.data_vars) == ["vp"]
    job = sensitivity_kernel_job(problem, grid, v=update)
    sediment = next(s for s in job.simulation.model.subdomains if s.name == "sediment")
    np.testing.assert_allclose(
        sediment.properties["vp"].control.coefficients, update[:5]
    )
    moved = sensitivity_kernel(problem, grid, v=update)
    assert list(moved.raw.data_vars) == ["vp"]

    # focus differentiates the material blocks only
    focus = TimeReversalFocus(problem, 5.0)
    assert focus.active == ["vp"]
    plan = focus.dry_run()
    assert plan["workflow"] == "focus"
    assert _assert_valid(focus.job(update).to_fs())["control_sensitivities"][
        "active"
    ] == ["vp"]
    value, gradient = focus.objective(update)
    assert np.isfinite(value) and gradient.space is problem.space
    np.testing.assert_array_equal(gradient["src"]["source.1.signature"], 0.0)
    assert np.any(gradient["vp"] != 0.0)
    # a changed signature is authored into the kernel's simulation (q scales
    # source 1's column of the made-explicit identity encoding); focusing runs
    # on the authored simulation and still refuses a moved source block
    signature = problem.vector().values.copy()
    signature[5] = 0.5
    job = sensitivity_kernel_job(problem, grid, v=signature)
    np.testing.assert_allclose(
        job.simulation.acquisition.source_encoding.weights, np.diag([0.5, 1.0])
    )
    with pytest.raises(NotImplementedError, match=r"'source\.1\.signature'"):
        focus.job(signature)


def test_fwi_result_simulation_installs_a_state_with_source_blocks(tmp_path, fake):
    problem = _source_problem(tmp_path, fake)
    update = problem.vector().values.copy()
    update[:5] = 0.25
    result = FWIResult(
        state=problem.state_from(update),
        history=OptimizationHistory(),
        problem=problem,
    )

    sediment = next(
        s for s in result.simulation.model.subdomains if s.name == "sediment"
    )
    np.testing.assert_allclose(sediment.properties["vp"].control.coefficients, 0.25)


# ---------------------------------------------------------------------------
# resolution changes between stages
# ---------------------------------------------------------------------------


FINE_VP = {"vp": DepthProfile("vp", "sediment", count=9)}


def test_with_controls_transfers_the_state_block_wise(tmp_path, fake):
    problem = _problem(tmp_path, fake, subdir="refine")
    coarse_nodes = problem.full_space.block("vp").control.coordinates
    coarse = problem.full_space.pack(
        {"vp": np.array([0.3, -0.2, 0.5, 0.1, -0.4]), "rho": [0.2, 0.0, -0.1]}
    )
    problem.state = problem.state.with_update(problem.full_space, coarse)

    fine = problem.with_controls(FINE_VP)

    assert fine is not problem and fine.name == problem.name
    assert fine.full_space.blocks == ("model.vp", "model.rho")
    assert fine.full_space.sizes == {"model.vp": 9, "model.rho": 3}
    assert fine.backend is problem.backend and fine.cache is problem.cache
    assert fine.workdir == problem.workdir and fine.site is problem.site
    assert fine.frequencies == problem.frequencies
    assert fine.observed_groups == problem.observed_groups
    assert fine.simulation is not problem.simulation
    fine_nodes = fine.full_space.block("vp").control.coordinates
    np.testing.assert_allclose(fine_nodes[::2], coarse_nodes)
    # the coarse hat profile is exactly representable on the bisected nodes:
    # shared nodes keep their values, midpoints interpolate linearly
    vp = fine.state["vp"]
    np.testing.assert_allclose(vp[::2], coarse["vp"], atol=1e-10)
    np.testing.assert_allclose(
        vp[1::2], 0.5 * (coarse["vp"][:-1] + coarse["vp"][1:]), atol=1e-10
    )
    np.testing.assert_array_equal(fine.state["rho"], coarse["rho"])  # copied
    # the original problem is untouched
    np.testing.assert_array_equal(problem.state.values, coarse.values)

    # an affine field transfers exactly to any node set
    affine = problem.state.with_update(
        problem.full_space,
        problem.full_space.pack({"vp": 0.4 + 2.0e-4 * coarse_nodes, "rho": [0, 0, 0]}),
    )
    odd = problem.with_controls(
        {"vp": DepthProfile("vp", "sediment", count=7)}, state=affine
    )
    odd_nodes = odd.full_space.block("vp").control.coordinates
    np.testing.assert_allclose(odd.state["vp"], 0.4 + 2.0e-4 * odd_nodes, atol=1e-10)
    # a state already on the new layout is adopted as is
    adopted = problem.with_controls(FINE_VP, state=fine.state)
    np.testing.assert_array_equal(adopted.state.values, fine.state.values)
    # a complete ControlSpace works too; the fine problem linearizes on its own
    spaced = problem.with_controls(
        ControlSpace(
            vp=DepthProfile("vp", "sediment", count=9),
            rho=DepthProfile("rho", "sediment", count=3),
        )
    )
    np.testing.assert_allclose(spaced.state.values, fine.state.values)
    lin = fine.linearize()
    assert lin.space.size == 12 and lin.gradient.size == 12
    assert lin.state_fingerprint != problem.linearize().state_fingerprint

    with pytest.raises(KeyError, match="unknown block keys"):
        problem.with_controls({"vs": DepthProfile("vs", "sediment", count=3)})
    with pytest.raises(ValueError, match="cannot change the layout of interface"):
        problem.with_controls({"rho": im.InterfaceParameters("salt_top")})
    with pytest.raises(ValueError, match="neither the current nor the new layout"):
        problem.with_controls(
            FINE_VP,
            state=ControlState(
                ControlSpace(vp=DepthProfile("vp", "sediment", count=4)).bind(
                    problem.simulation
                ),
                np.zeros(4),
            ),
        )


def _refine_stages(fine=FINE_VP):
    return [
        Stage(FREQUENCIES, 6, active=["vp"], name="coarse"),
        Stage(FREQUENCIES, 40, active=["vp"], name="fine", controls=fine),
    ]


def test_fwi_stage_controls_refine_the_profile_between_stages(tmp_path, fake):
    problem = _problem(tmp_path, fake, subdir="stages")
    fwi = FWI(problem, _refine_stages(), optimizer=LBFGS(**TIGHT))

    result = fwi.run()

    coarse, fine = result.stages
    assert coarse.space.size == 5 and fine.space.size == 9
    assert result.problem is fwi.final_problem and result.problem is not problem
    assert result.problem.full_space.sizes == {"model.vp": 9, "model.rho": 3}
    assert result.state.space.blocks == ("model.vp", "model.rho")
    assert result.state.size == 12
    assert result.vector().size == 12
    # the fine stage starts from the coarse result transferred to 9 nodes
    # (the base problem keeps the coarse stage's accepted state)
    np.testing.assert_allclose(problem.state["vp"], coarse.vector.values)
    check = problem.with_controls(FINE_VP)
    np.testing.assert_allclose(check.state["vp"][::2], coarse.vector.values, atol=1e-10)
    start = check.restrict(frequencies=FREQUENCIES, active=["vp"])
    assert fine.initial_loss.data == pytest.approx(start.value())
    # and decreases the objective on the fine layout
    assert fine.final_loss.total < fine.initial_loss.total
    losses = _stage_losses(result.history, 1)
    assert len(losses) >= 2 and np.all(np.diff(losses) <= 1e-9)
    J, d = _surrogate(fake, result.problem, active=["vp"])
    np.testing.assert_allclose(
        result.state["vp"], _least_squares(J, d), atol=1e-4
    )  # the fine stage converged on its own surrogate
    np.testing.assert_array_equal(result.state["rho"], 0.0)  # never active
    assert result.simulation is not None  # simulation_at on the fine layout


def test_fwi_resumes_inside_a_refined_stage(tmp_path, fake):
    class Interrupt(RuntimeError):
        pass

    def interrupt(event):
        if event.stage_index == 1 and event.stage_iteration == 2:
            raise Interrupt("simulated crash")

    problem = _problem(tmp_path, fake, subdir="refine_resume")
    checkpoint = tmp_path / "refine" / "fwi.ckpt.h5"
    options = dict(optimizer=LBFGS(**TIGHT), checkpoint=checkpoint)
    with pytest.raises(Interrupt):
        FWI(problem, _refine_stages(), callback=interrupt, **options).run()

    saved = OptimizationCheckpoint.load(checkpoint)
    assert saved.metadata["stage_index"] == 1 and saved.metadata["stage_iteration"] == 2
    assert saved.metadata["control_ids"] == "model.vp,model.rho"
    assert saved.metadata["control_sizes"] == "9,3"
    assert saved.model.size == 9

    # a different refinement is rejected
    with pytest.raises(ValueError, match="control sizes"):
        FWI(
            problem,
            _refine_stages({"vp": DepthProfile("vp", "sediment", count=7)}),
            **options,
        ).run()

    resumed = FWI(problem, _refine_stages(), **options).run(resume=True)

    assert resumed.stages[0].skipped and resumed.stages[1].resumed
    assert resumed.stages[1].stage_iteration >= 3
    assert resumed.state.size == 12
    assert resumed.problem.full_space.sizes["model.vp"] == 9
    J, d = _surrogate(fake, resumed.problem, active=["vp"])
    np.testing.assert_allclose(resumed.state["vp"], _least_squares(J, d), atol=1e-4)
    final = OptimizationCheckpoint.load(checkpoint)
    assert (
        final.metadata["stage_completed"] and final.metadata["control_sizes"] == "9,3"
    )
    # a completed run resumes to a no-op on the fine layout
    again = FWI(problem, _refine_stages(), **options).run(resume=True)
    assert all(s.skipped for s in again.stages)
    np.testing.assert_allclose(again.state.values, resumed.state.values)


def test_stage_controls_validation():
    stage = Stage(FREQUENCIES, 2, controls=FINE_VP)
    assert isinstance(stage.controls, dict) and stage.controls["vp"].count == 9
    assert stage.to_continuation_stage().metadata["controls"] == ["vp"]
    single = Stage(FREQUENCIES, 2, controls=DepthProfile("vp", "sediment", count=4))
    assert isinstance(single.controls, ControlSpace)
    with pytest.raises(TypeError, match="block specs"):
        Stage(FREQUENCIES, 2, controls={"vp": 9})
    with pytest.raises(ValueError, match="cannot be empty"):
        Stage(FREQUENCIES, 2, controls={})
    with pytest.raises(TypeError, match="ControlSpace"):
        Stage(FREQUENCIES, 2, controls=[1, 2])
