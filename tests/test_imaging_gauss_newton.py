"""GaussNewton stage optimizer: configuration, stage glue and FWI on the fake site."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize

from frequensolve import imaging as im
from frequensolve.imaging import (
    ControlSpace,
    ControlVector,
    DepthProfile,
    ImagingProblem,
)
from frequensolve.imaging.regularization import (
    BoundRegularization,
    _SymmetricModelOperator,
)
from frequensolve.imaging.workflows import (
    FWI,
    GaussNewton,
    Stage,
    _constant_hessian,
    _StageObjective,
)
from frequensolve.inversion import (
    GaussNewtonIteration,
    OptimizationCheckpoint,
    OptimizationHistory,
)
from tests.imaging_fakes import FakeImagingSite, layered_simulation

pytestmark = pytest.mark.unit

FREQUENCIES = [4.0, 6.0]
# Absolute gradient tolerance only: the stage stops on the gradient test.
GRADIENT = dict(grad_abs_tol=1e-7, grad_rel_tole=0.0, obj_rel_tol=0.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _problem(tmp_path, fake, *, subdir="project", **kwargs):
    simulation = layered_simulation(tmp_path / subdir)
    observed = tmp_path / "observed.h5"
    if not observed.exists():
        observed.write_bytes(b"observed")
    options = dict(
        controls=ControlSpace(
            vp=DepthProfile("vp", "sediment", count=5),
            rho=DepthProfile("rho", "sediment", count=3),
        ),
        observed={"surface": observed},
        frequencies=FREQUENCIES,
        site=fake,
        name="fwi",
        cache_capacity=4,
    )
    options.update(kwargs)
    return ImagingProblem(simulation, **options)


def _surrogate(fake, problem):
    """Return ``(J, d)`` of the fake's linear surrogate for ``problem``."""

    linearization = problem.linearize(gradient=False)
    surrogate = fake.linearizations[linearization.state_fingerprint]
    return surrogate.J, surrogate.d


def _least_squares(J, d, alpha=0.0, *, R=None):
    """``argmin 0.5 ||J m - d||^2 + 0.5 alpha ||m||^2 + 0.5 ||R m||^2``."""

    normal = np.real(J.conj().T @ J) + alpha * np.eye(J.shape[1])
    if R is not None:
        normal = normal + R.T @ R
    return np.linalg.solve(normal, np.real(J.conj().T @ d))


def _resolved_space():
    """An eight-DOF space resolved without a simulation (global nodes)."""

    return ControlSpace(
        vp=DepthProfile("vp", "sediment", datum="global", nodes=[0, 1, 2, 3, 4.0]),
        rho=DepthProfile("rho", "sediment", datum="global", nodes=[0, 1, 2.0]),
    )


class _Quadratic:
    """``0.5 (x - s)^T A (x - s)`` with ``A s = b`` and a Hessian action only."""

    def __init__(self, A, b):
        self.A, self.b = A, b
        self.offset = 0.5 * float(b @ np.linalg.solve(A, b))

    def value(self, x):
        return 0.5 * float(x @ self.A @ x) - float(self.b @ x) + self.offset

    def gradient(self, x):
        return self.A @ x - self.b

    def hessian_action(self, x, dx):
        return self.A @ dx


class _MockLinearization:
    """Value, gradient and a recording ``apply_normal_batch`` of ``_MockView``."""

    def __init__(self, view, x, gradient):
        self.view, self.x = view, np.array(x)
        residual = view.residual(self.x)
        self.value = 0.5 * float(residual @ residual)
        self.gradient = (
            ControlVector(view.jacobian(self.x).T @ residual, view.space)
            if gradient
            else None
        )

    def apply_normal_batch(self, directions, *, cache=True, regularization=None):
        rows = np.array(directions, dtype=float)
        self.view.normal_calls.append((rows, cache, regularization))
        J = self.view.jacobian(self.x)
        return [ControlVector(J.T @ (J @ row), self.view.space) for row in rows]


class _MockView:
    """Nonlinear least squares ``r(m) = A m + 0.1 P m^2 - d`` behind ``linearize``."""

    def __init__(self, space, seed=0):
        rng = np.random.default_rng(seed)
        self.space = space
        self.A = rng.standard_normal((12, space.size))
        self.P = rng.standard_normal((12, space.size))
        self.d = rng.standard_normal(12)
        self.normal_calls = []

    def residual(self, m):
        return self.A @ m + 0.1 * self.P @ m**2 - self.d

    def jacobian(self, m):
        return self.A + 0.2 * self.P * m[None, :]

    def linearize(self, vector, gradient=False):
        return _MockLinearization(self, np.asarray(vector.values), gradient)


class _Quartic(im.Regularization):
    """``0.25 sum(m^4)``: smooth, but its Hessian depends on the model."""

    def bind(self, space):
        return _BoundQuartic(self, space)


class _BoundQuartic(BoundRegularization):
    def __init__(self, regularization, space):
        super().__init__(regularization, space)
        self.hessians = 0

    def value(self, v):
        return 0.25 * float(np.sum(self._values(v) ** 4))

    def gradient(self, v):
        return self._wrap(self._values(v) ** 3)

    def hessian_operator(self, v):
        self.hessians += 1
        diagonal = 3.0 * self._values(v) ** 2
        return _SymmetricModelOperator(lambda x: diagonal * x, self.space)


@pytest.fixture
def fake():
    return FakeImagingSite(seed=11)


@pytest.fixture
def problem(tmp_path, fake):
    return _problem(tmp_path, fake)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_gauss_newton_configuration_builds_validated_options():
    optimizer = GaussNewton(
        initial_radius=0.5, max_radius=4.0, scaling="marquardt", refresh=2
    )
    assert im.GaussNewton is GaussNewton and optimizer.kind == "gauss_newton"
    options = optimizer.options(7)
    assert options.max_iterations == 7 and options.initial_radius == 0.5
    assert options.max_radius == 4.0 and options.scaling == "marquardt"
    assert options.refresh == 2 and options.max_explicit == 256
    assert GaussNewton().options().max_iterations == 50
    assert "kind" not in repr(optimizer)  # checkpoints compare the repr
    with pytest.raises(ValueError, match="eta"):
        GaussNewton(eta=0.5)
    with pytest.raises(ValueError, match="scaling_max_ratio"):
        GaussNewton(scaling_max_ratio=0.5)

    quadratic = _Quadratic(np.eye(2), np.ones(2))
    unsupported = dict(
        preconditioner=lambda _m, g: g,
        step_transform=lambda _m, step: step,
        step_limit=0.1,
    )
    for name, value in unsupported.items():
        with pytest.raises(ValueError, match=f"takes no {name}"):
            optimizer.solve(quadratic, np.zeros(2), **{name: value})
    first_order = SimpleNamespace(value=quadratic.value, gradient=quadratic.gradient)
    with pytest.raises(ValueError, match="gauss_newton_matrix or hessian_action"):
        optimizer.solve(first_order, np.zeros(2))


def test_gauss_newton_builds_the_matrix_from_hessian_actions_in_scaled_coordinates():
    rng = np.random.default_rng(5)
    root = rng.standard_normal((6, 6))
    A, b = root @ root.T + np.eye(6), rng.standard_normal(6)
    solution = np.linalg.solve(A, b)
    optimizer = GaussNewton(initial_radius=1e6)

    result = optimizer.solve(_Quadratic(A, b), np.zeros(6))
    assert result.success and result.iterations == 1
    np.testing.assert_allclose(result.model, solution, atol=1e-10)

    # The change of variables x = s * y leaves the Gauss-Newton step invariant;
    # results and records are physical.
    records = []
    scaled = optimizer.solve(
        _Quadratic(A, b),
        np.zeros(6),
        scaling=np.linspace(0.5, 2.0, 6),
        callback=records.append,
    )
    assert scaled.iterations == 1
    np.testing.assert_allclose(scaled.model, solution, atol=1e-10)
    np.testing.assert_allclose(scaled.gradient, A @ scaled.model - b, atol=1e-10)
    assert isinstance(records[-1], GaussNewtonIteration)
    np.testing.assert_allclose(records[-1].model, scaled.model)
    np.testing.assert_allclose(records[-1].step, solution, atol=1e-10)


# ---------------------------------------------------------------------------
# stage objective glue (mock linearization)
# ---------------------------------------------------------------------------


def test_stage_matrix_is_one_normal_batch_plus_cached_quadratic_regularization():
    space = _resolved_space()
    view = _MockView(space)
    alpha = 0.5
    regularization = (im.Quadratic(np.eye(8), weight=alpha) + _Quartic()).bind(space)
    quadratic, quartic = regularization.terms
    assert _constant_hessian(quadratic) and not _constant_hessian(quartic)
    assert not _constant_hessian(regularization)
    quadratic_hessians = []
    operator = quadratic.hessian_operator
    quadratic.hessian_operator = lambda v=None: (
        quadratic_hessians.append(v) or operator(v)
    )
    objective = _StageObjective(view, space, regularization, None, {})
    lower, upper = np.full(8, -np.inf), np.full(8, np.inf)
    lower[6] = upper[6] = 0.0  # pinned: outside the matrix
    history = OptimizationHistory()

    result = GaussNewton(initial_radius=0.05, max_iterations=100, **GRADIENT).solve(
        objective, np.zeros(8), bounds=(lower, upper), space=space, history=history
    )

    free = [0, 1, 2, 3, 4, 5, 7]
    assert result.success and result.status == 1 and result.model[6] == 0.0
    # One normal batch of the free identity directions per matrix, bypassing
    # the linearization's memoized products.
    assert len(view.normal_calls) == result.hessian_refreshes >= 2
    for rows, cache, native in view.normal_calls:
        np.testing.assert_array_equal(rows, np.eye(8)[free])
        assert cache is False and native is None
    # The quadratic Hessian is built once per stage; the quartic at every matrix.
    assert len(quadratic_hessians) == 1
    assert quartic.hessians == result.hessian_refreshes

    def total(z):
        m = np.zeros(8)
        m[free] = z
        residual = view.residual(m)
        return 0.5 * residual @ residual + 0.5 * alpha * m @ m + 0.25 * np.sum(m**4)

    def gradient(z):
        m = np.zeros(8)
        m[free] = z
        full = view.jacobian(m).T @ view.residual(m) + alpha * m + m**3
        return full[free]

    reference = minimize(
        total, np.zeros(7), jac=gradient, method="BFGS", options=dict(gtol=1e-12)
    )
    np.testing.assert_allclose(result.model[free], reference.x, atol=1e-6)

    record = history.iterations[-1]
    assert record.metrics["optimizer"] == "gauss_newton"
    assert record.metrics["gain_ratio"] > 1e-4
    assert record.metrics["trust_region_trials"] >= 1
    assert record.metrics["hessian_refreshed"] is True
    norms = json.loads(record.metrics["block_step_norms"])
    assert set(norms) == {"model.vp", "model.rho"}
    assert np.hypot(norms["model.vp"], norms["model.rho"]) == pytest.approx(
        record.step_norm
    )


# ---------------------------------------------------------------------------
# FWI on the fake site
# ---------------------------------------------------------------------------


def test_fwi_gauss_newton_takes_the_exact_step_with_one_normal_job(problem, fake):
    J, d = _surrogate(fake, problem)
    before = len(fake.submissions)
    with pytest.warns(UserWarning, match="ignores the FWI step_limit"):
        result = FWI(
            problem,
            Stage(FREQUENCIES, 5, name="joint"),
            optimizer=GaussNewton(initial_radius=1e6),
            step_limit=1e-3,  # an RMS cap would forbid the exact step
        ).run()

    stage = result.stages[0]
    assert stage.success and stage.status == 1 and stage.iterations == 1
    np.testing.assert_allclose(result.state.values, _least_squares(J, d), atol=1e-6)
    normal = [s for s in fake.submissions[before:] if s["action"] == "normal"]
    assert len(normal) == 1  # eight directions, one job, no matrix at the end
    assert stage.metrics["optimizer"] == "gauss_newton"
    assert stage.metrics["hessian_products"] == 8
    assert stage.metrics["hessian_refreshes"] == 1
    assert stage.metrics["trust_radius"] > 0.0
    record = [r for r in result.history.iterations if r.metrics["stage_index"] == 0][-1]
    # The surrogate is linear: the Gauss-Newton model is exact.
    assert record.metrics["gain_ratio"] == pytest.approx(1.0, abs=1e-4)
    assert record.metrics["marquardt_parameter"] == 0.0
    assert record.metrics["hessian_eigenvalue_min"] > 0.0
    assert set(json.loads(record.metrics["block_step_norms"])) == {
        "model.vp",
        "model.rho",
    }


def test_fwi_gauss_newton_damps_with_a_cached_quadratic_regularization(
    problem, fake, monkeypatch
):
    from frequensolve.imaging import regularization as regularization_module

    J, d = _surrogate(fake, problem)
    alpha = 5.0
    hessians = []
    original = regularization_module._BoundQuadraticForm.hessian_operator

    def counted(self, v=None):
        hessians.append(v)
        return original(self, v)

    monkeypatch.setattr(
        regularization_module._BoundQuadraticForm, "hessian_operator", counted
    )
    events = []
    radius = 1e-2
    result = FWI(
        problem,
        Stage(FREQUENCIES, 30),
        optimizer=GaussNewton(initial_radius=radius),
        regularization=im.Quadratic(np.eye(8), weight=alpha),
        callback=events.append,
    ).run()

    stage = result.stages[0]
    assert stage.success
    np.testing.assert_allclose(
        result.state.values, _least_squares(J, d, alpha), atol=1e-6
    )
    assert stage.metrics["hessian_refreshes"] >= 2 and len(hessians) == 1
    # Accepted boundary steps of an exact model double the radius.
    radii = [event.diagnostics.trust_radius for event in events[1:3]]
    assert radii == [radius, 2 * radius]
    assert all(event.diagnostics.boundary_step for event in events[1:3])


def test_fwi_gauss_newton_includes_native_tikhonov_and_ignores_the_preconditioner(
    problem, fake
):
    J, d = _surrogate(fake, problem)
    alpha = 5.0e-2
    R = np.sqrt(alpha) * np.eye(8)
    before = len(fake.submissions)
    with pytest.warns(UserWarning, match="ignores the FWI preconditioner"):
        result = FWI(
            problem,
            Stage(FREQUENCIES, 5),
            optimizer=GaussNewton(initial_radius=1e6),
            regularization=im.Tikhonov(alpha),
            preconditioner=im.Diagonal(probe_count=4),
        ).run()

    stage = result.stages[0]
    assert stage.success and stage.iterations == 1
    np.testing.assert_allclose(
        result.state.values, _least_squares(J, d, R=R), atol=1e-6
    )
    assert stage.final_loss.regularization == pytest.approx(
        0.5 * float(np.linalg.norm(R @ result.state.values) ** 2)
    )
    # The unused diagonal preconditioner ran no probe products: the only
    # normal job is the Gauss-Newton matrix (Tikhonov runs its own jobs once).
    actions = [s["action"] for s in fake.submissions[before:]]
    assert "jvp" not in actions and actions.count("normal") == 1


def test_fwi_gauss_newton_rejects_nonsmooth_native_regularization(problem):
    with pytest.raises(ValueError, match="smooth objective"):
        FWI(
            problem,
            Stage(FREQUENCIES, 3),
            optimizer=GaussNewton(),
            regularization=im.TV(1e-3),
        ).run()


def test_fwi_gauss_newton_resumes_with_the_checkpointed_trust_radius(tmp_path, fake):
    class Interrupt(RuntimeError):
        pass

    interrupted = []

    def interrupt(event):
        interrupted.append(event)
        if event.stage_iteration == 2:
            raise Interrupt("simulated crash")

    radius = 1e-3
    problem = _problem(tmp_path, fake, subdir="resume")
    checkpoint = tmp_path / "run" / "fwi.ckpt.h5"
    options = dict(
        optimizer=GaussNewton(initial_radius=radius, scaling="marquardt"),
        checkpoint=checkpoint,
        history=tmp_path / "run" / "history.json",
    )
    stage = Stage(FREQUENCIES, 40, name="joint")
    with pytest.raises(Interrupt):
        FWI(problem, stage, callback=interrupt, **options).run()

    saved = OptimizationCheckpoint.load(checkpoint)
    assert saved.metadata["stage_iteration"] == 2
    state = json.loads(saved.metadata["restart"])["optimizer"]
    assert state["schema"] == "fs-gauss-newton-restart-1"
    assert state["scaling"] == "marquardt" and state["diagonal"] is not None
    assert state["radius"] == interrupted[-1].diagnostics.next_trust_radius
    assert state["radius"] == 4 * radius  # two accepted boundary steps

    resumed = []
    result = FWI(problem, stage, callback=resumed.append, **options).run(resume=True)

    np.testing.assert_array_equal(
        resumed[0].diagnostics.optimizer_state.diagonal,
        interrupted[-1].diagnostics.optimizer_state.diagonal,
    )
    first = next(event for event in resumed if event.iteration == 1)
    assert first.stage_iteration == 3
    assert first.diagnostics.trust_radius == 4 * radius  # not the initial radius
    assert result.stages[0].resumed and result.stages[0].success
    J, d = _surrogate(fake, problem)
    np.testing.assert_allclose(result.state.values, _least_squares(J, d), atol=1e-6)
    final = OptimizationCheckpoint.load(checkpoint)
    assert final.metadata["stage_completed"]
    assert json.loads(final.metadata["restart"]) is None
