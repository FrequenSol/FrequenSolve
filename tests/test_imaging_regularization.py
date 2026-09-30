import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import cg

from frequensolve.imaging import (
    ControlSpace,
    ControlVector,
    DepthProfile,
    GridParameters,
    ImagingProblem,
    InterfaceParameters,
    ModelOperator,
    SourceParameters,
)
from frequensolve.imaging._artifacts import ControlVectorFile
from frequensolve.imaging._backend import Backend
from frequensolve.imaging.jobs import SmoothJob
from frequensolve.imaging.regularization import (
    TGV,
    TV,
    BoundPreconditioner,
    BoundRegularization,
    Diagonal,
    FromOperator,
    Identity,
    Quadratic,
    Regularization,
    Scaled,
    Smoothing,
    Sum,
    Tikhonov,
    smooth,
)
from frequensolve.inversion.least_squares import QuadraticRegularization
from frequensolve.inversion.preconditioning import DiagonalInverseHessian
from frequensolve.orchestrator.sites.base import JobStatus, RunResult
from tests.imaging_fakes import FakeImagingSite, layered_simulation

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simulation(tmp_path):
    return layered_simulation(tmp_path / "project", save=False)


def _space():
    return ControlSpace(
        vp=DepthProfile("vp", "sediment", count=6),
        rho=DepthProfile.bspline("rho", "sediment", count=5, degree=3),
        grid=GridParameters("vp", "water", shape=[4, 3]),
        salt=InterfaceParameters("salt_top"),
        src=SourceParameters(position=True, signature=True),
    )


@pytest.fixture
def space(simulation):
    return _space().bind(simulation)


@pytest.fixture
def frozen_space(space):
    vp = np.ones(space.sizes["model.vp"], dtype=bool)
    vp[2] = False  # an interior profile node
    grid = np.ones(space.sizes["model.grid"], dtype=bool)
    grid[0] = False  # the lattice corner (i_x = 0, i_z = 0)
    return space.with_support({"vp": vp, "grid": grid})


def _random(space, seed=0):
    return space.random(seed) * 3.0


REGULARIZATIONS = ("quadratic", "sum")


def _bind(name, space):
    matrix = sp.random(9, space.size, density=0.3, random_state=3, format="csr")
    regularization = Quadratic(matrix, weight=2.0, reference=space.random(4))
    if name == "sum":
        regularization = regularization + 0.5 * Quadratic(sp.identity(space.size))
    return regularization.bind(space)


def _directional_derivative(bound, v, direction, h=1.0e-6):
    return (bound.value(v + h * direction) - bound.value(v - h * direction)) / (2 * h)


# ---------------------------------------------------------------------------
# regularization: values and gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(REGULARIZATIONS))
def test_regularization_gradients_match_finite_differences(name, space):
    bound = _bind(name, space)
    v = _random(space, 1)
    gradient = bound.gradient(v)
    assert isinstance(gradient, ControlVector) and gradient.size == space.size
    tolerance = 1.0e-6
    for seed in (2, 3):
        direction = space.random(seed)
        expected = _directional_derivative(bound, v, direction)
        actual = gradient.dot(direction)
        assert abs(actual - expected) <= tolerance * max(1.0, abs(expected))
    # ndarray inputs are accepted too
    assert bound.value(v.values) == pytest.approx(bound.value(v))
    np.testing.assert_allclose(bound.gradient(v.values).values, gradient.values)


@pytest.mark.parametrize("name", sorted(REGULARIZATIONS))
def test_hessian_operators_are_self_adjoint_and_positive_semidefinite(name, space):
    bound = _bind(name, space)
    v = _random(space, 5)
    H = bound.hessian_operator(v)
    assert isinstance(H, ModelOperator)
    assert H.shape == (space.size, space.size)
    assert H.H is H and H.T is H
    x = space.random(6)
    y = space.random(7)
    Hx = H @ x
    assert isinstance(Hx, ControlVector)
    assert Hx.dot(y) == pytest.approx((H @ y).dot(x), rel=1e-10, abs=1e-12)
    assert Hx.dot(x) >= -1e-12
    diagonal = bound.curvature_diagonal(v)
    assert diagonal.shape == (space.size,)
    assert np.all(diagonal >= 0.0)
    dense = np.column_stack(
        [(H @ np.eye(space.size)[:, i]).values for i in range(space.size)]
    )
    np.testing.assert_allclose(np.diag(dense), diagonal, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dense, dense.T, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("name", ["quadratic"])
def test_quadratic_operators_square_to_the_hessian(name, space):
    bound = _bind(name, space)
    R = bound.operator()
    assert isinstance(R, ModelOperator) and R.shape[1] == space.size
    x = space.random(8)
    residual = R @ x
    assert isinstance(residual, np.ndarray)  # untyped range
    squared = (R.H @ R) @ x
    assert isinstance(squared, ControlVector)
    np.testing.assert_allclose(squared.values, (bound.hessian_operator(x) @ x).values)
    dense = (R.matrix.T @ R.matrix).toarray()
    np.testing.assert_allclose(np.diag(dense), bound.curvature_diagonal(x))
    reference = bound.reference if bound.reference is not None else 0.0
    np.testing.assert_allclose(
        bound.gradient(x).values, R.matrix.T @ (R.matrix @ (x.values - reference))
    )
    # ``H + alpha * R.H @ R`` keeps typed vectors
    combined = bound.hessian_operator(x) + 0.5 * (R.H @ R)
    assert isinstance(combined @ x, ControlVector)


def test_quadratic_wraps_quadratic_regularization_semantics(space):
    matrix = np.random.default_rng(1).standard_normal((5, space.size))
    reference = space.random(2)
    regularization = Quadratic(matrix, weight=3.0, reference=reference)
    bound = regularization.bind(space)
    toolkit = regularization.least_squares_term(space)
    assert isinstance(toolkit, QuadraticRegularization)
    v = _random(space, 3)
    residual = toolkit.residual(v.values)
    assert bound.value(v) == pytest.approx(0.5 * residual @ residual)
    np.testing.assert_allclose(
        bound.gradient(v).values, toolkit.jacobian().T @ residual
    )
    with pytest.raises(ValueError, match="columns"):
        Quadratic(np.ones((2, space.size + 1))).bind(space)
    with pytest.raises(ValueError, match="2-D"):
        Quadratic(np.ones(3))
    with pytest.raises(ValueError, match="real"):
        Quadratic(np.ones((2, 2), dtype=complex))


# ---------------------------------------------------------------------------
# frozen DOFs, weights and unsupported blocks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def test_composition_sums_scales_and_flattens(space):
    first = Quadratic(sp.identity(space.size), weight=0.4)
    second = Quadratic(sp.diags(np.linspace(1, 2, space.size)), weight=0.3)
    quad = Quadratic(sp.identity(space.size, format="csr"), weight=0.1)
    combined = first + 0.5 * second + quad
    assert isinstance(combined, Sum)
    assert isinstance(combined.regularizations[1], Scaled)
    assert len(combined.regularizations) == 3
    assert isinstance(2.0 * first, Regularization) and isinstance(first * 2.0, Scaled)
    bound = combined.bind(space)
    v = _random(space, 13)
    parts = [first.bind(space), second.bind(space), quad.bind(space)]
    expected = parts[0].value(v) + 0.5 * parts[1].value(v) + parts[2].value(v)
    assert bound.value(v) == pytest.approx(expected)
    np.testing.assert_allclose(
        bound.gradient(v).values,
        parts[0].gradient(v).values
        + 0.5 * parts[1].gradient(v).values
        + parts[2].gradient(v).values,
    )
    np.testing.assert_allclose(
        bound.curvature_diagonal(v),
        parts[0].curvature_diagonal(v)
        + 0.5 * parts[1].curvature_diagonal(v)
        + parts[2].curvature_diagonal(v),
    )
    assert bound.operator() is not None
    quadratic_only = (first + 2.0 * quad).bind(space)
    R = quadratic_only.operator()
    assert R is not None
    x = space.random(3)
    np.testing.assert_allclose(
        ((R.H @ R) @ x).values, (quadratic_only.hessian_operator(x) @ x).values
    )
    with pytest.raises(TypeError):
        Sum(first, object())
    with pytest.raises(ValueError, match="at least one"):
        Sum()
    with pytest.raises(ValueError, match="regularization scale"):
        -1.0 * first


@pytest.mark.parametrize("spec", [Tikhonov(1), TV(1), TGV(1, 2)])
def test_model_regularizers_require_native_binding(space, spec):
    with pytest.raises(NotImplementedError, match="native callbacks"):
        spec.bind(space)


def test_regularization_validation():
    with pytest.raises(ValueError, match="alpha"):
        Tikhonov(-1.0)
    with pytest.raises(ValueError, match="order"):
        Tikhonov(1.0, order=3)
    with pytest.raises(ValueError, match="epsilon"):
        TV(1.0, epsilon=0.0)
    with pytest.raises(ValueError, match="probe_count"):
        Diagonal(probe_count=0)
    with pytest.raises(ValueError, match="probes"):
        Diagonal(probes="fancy")
    with pytest.raises(TypeError):
        FromOperator("not an operator")


# ---------------------------------------------------------------------------
# preconditioners
# ---------------------------------------------------------------------------


def _problem(tmp_path, fake, controls=None):
    sim = layered_simulation(tmp_path / "project")
    return ImagingProblem(
        sim,
        controls=controls
        or ControlSpace(
            vp=DepthProfile("vp", "sediment", count=5),
            rho=DepthProfile("rho", "sediment", count=3),
        ),
        observed={"surface": tmp_path / "observed.h5"},
        frequencies=[4.0, 6.0],
        site=fake,
        name="fwi",
    )


def test_diagonal_preconditioner_reproduces_the_gauss_newton_diagonal(tmp_path):
    fake = FakeImagingSite(seed=5)
    problem = _problem(tmp_path, fake)
    lin = problem.linearize()
    J = fake.linearizations[lin.state_fingerprint].J
    exact = np.sum(np.abs(J) ** 2, axis=0)
    regularization = Quadratic(sp.identity(lin.space.size), weight=0.2).bind(lin.space)

    unit = Diagonal(probes="unit").bind(problem.space)
    assert isinstance(unit, BoundPreconditioner)
    with pytest.raises(RuntimeError, match="update"):
        unit.apply(lin.space.zeros())
    unit.update(lin, regularization)
    np.testing.assert_allclose(unit.estimate.data, exact, rtol=1e-8)
    np.testing.assert_allclose(
        unit.estimate.regularization, regularization.curvature_diagonal(lin.point)
    )
    g = lin.space.random(3)
    applied = unit.apply(g)
    assert isinstance(applied, ControlVector) and applied.space is lin.space
    expected = DiagonalInverseHessian(
        exact + regularization.curvature_diagonal(lin.point),
        block_sizes=[5, 3],
        relative_damping=1e-2,
        maximum_inverse_ratio=1e3,
    ).apply(g.values)
    np.testing.assert_allclose(applied.values, expected)
    np.testing.assert_allclose(unit(g).values, expected)
    np.testing.assert_allclose((unit.operator() @ g).values, expected)

    random = Diagonal(probe_count=64, seed=0).bind(lin.space)
    random.update(lin)
    estimate = random.estimate.data
    assert np.all(estimate >= 0.0)
    assert np.linalg.norm(estimate - exact) <= 0.5 * np.linalg.norm(exact)
    assert random.estimate.probe_count == 64
    np.testing.assert_array_equal(random.estimate.regularization, 0.0)
    assert random.diagonal is not None and np.all(random.diagonal > 0.0)


def test_identity_and_from_operator_preconditioners(space):
    g = space.random(1)
    identity = Identity().bind(space)
    np.testing.assert_array_equal(identity.apply(g).values, g.values)
    assert identity.apply(g) is not g
    identity.update(None)  # no-op
    scale = 2.0 * np.ones(space.size)
    for op in (
        lambda x: scale * np.asarray(x),
        sp.diags(scale),
        np.diag(scale),
        FromOperator,
    ):
        if op is FromOperator:
            continue
        bound = FromOperator(op).bind(space)
        result = bound.apply(g)
        assert isinstance(result, ControlVector)
        np.testing.assert_allclose(result.values, 2.0 * g.values)
    typed = FromOperator(
        Quadratic(sp.identity(space.size)).bind(space).hessian_operator(g)
    ).bind(space)
    assert isinstance(typed.apply(g), ControlVector)
    with pytest.raises(ValueError, match="shape"):
        FromOperator(np.eye(space.size + 1)).bind(space)
    with pytest.raises(ValueError, match="entries"):
        FromOperator(lambda x: np.ones(3)).bind(space).apply(g)
    # the action drops into SciPy's cg as ``M``
    H = Quadratic(sp.identity(space.size)).bind(space).hessian_operator(
        g
    ) + sp.identity(space.size)
    solution, info = cg(
        H,
        g.values,
        M=FromOperator(np.diag(scale)).bind(space).operator(),
        rtol=1e-10,
    )
    assert info == 0
    np.testing.assert_allclose(H @ solution, g.values, atol=1e-6)


# ---------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------


def _fake_smooth_run(scale, seen):
    def run(self, job, *, postprocess_only=False, check=True):
        assert isinstance(job, SmoothJob) and postprocess_only
        seen.append(job)
        source = ControlVectorFile.read(job.input_vector)
        assert source.native
        ControlVectorFile(
            {name: scale * values for name, values in source.blocks.items()},
            native=True,
        ).write(job.gradient_file())
        return RunResult(job=job, status=JobStatus(state="completed", return_code=0))

    return run


def test_smooth_runs_a_postprocess_only_smooth_job_on_model_blocks(
    tmp_path, monkeypatch
):
    fake = FakeImagingSite(seed=2)
    controls = ControlSpace(
        vp=DepthProfile("vp", "sediment", count=5),
        src=SourceParameters(signature=True),
    )
    problem = _problem(tmp_path, fake, controls=controls)
    seen = []
    monkeypatch.setattr(Backend, "run", _fake_smooth_run(3.0, seen))
    v = problem.space.random(1)
    smoothed = smooth(v, Smoothing(kind="tv", wavelength_fraction=0.4), problem)
    assert isinstance(smoothed, ControlVector) and smoothed.space is problem.space
    np.testing.assert_allclose(smoothed["vp"], 3.0 * v["vp"])
    np.testing.assert_array_equal(
        smoothed["src.signature"]["source.1.signature"],
        v["src.signature"]["source.1.signature"],
    )
    job = seen[-1]
    payload = job.to_fs()["control_sensitivities"]
    assert (
        payload["Smoothing"]["type"] == "tv" and payload["Smoothing"]["lambda"] == 0.4
    )
    assert payload["active"] == ["vp"]
    assert "input" in payload and "gradient" in payload
    assert job.name.startswith("fwi_smooth_")
    # before any linearization the source is an unrun linearize shell
    assert job.source_job.action == "linearize" and job.source_job.covector is not None
    assert list(job.f_list) == [4.0, 6.0]
    # mappings and arrays are accepted; weights are not
    again = smooth(v.values, {"type": "tikhonov", "lambda": 0.5}, problem)
    np.testing.assert_allclose(again.values, smoothed.values)
    with pytest.raises(ValueError, match="weights"):
        smooth(v, Smoothing(), problem, weights=[1.0, 1.0])
    with pytest.raises(ValueError, match="smoothing configuration"):
        smooth(v, None, problem)


def test_smooth_reuses_the_latest_linearize_job(tmp_path, monkeypatch):
    fake = FakeImagingSite(seed=2)
    problem = _problem(tmp_path, fake)
    lin = problem.linearize()
    seen = []
    monkeypatch.setattr(Backend, "run", _fake_smooth_run(0.5, seen))
    smoothed = smooth(lin.gradient, Smoothing(kind="tikhonov"), problem)
    np.testing.assert_allclose(smoothed.values, 0.5 * lin.gradient.values)
    assert seen[-1].source_job is lin.job


def test_bound_regularization_requires_the_bound_space(space, frozen_space):
    bound = Quadratic(sp.identity(space.size)).bind(space)
    assert isinstance(bound, BoundRegularization)
    with pytest.raises(ValueError, match="entries"):
        bound.value(np.ones(3))
    with pytest.raises(ValueError, match="different control space"):
        bound.value(frozen_space.zeros())
    with pytest.raises(ValueError, match="real"):
        bound.value(np.ones(space.size, dtype=complex))


# ---------------------------------------------------------------------------
# cross-checks on the fake site (masked spaces, fake postprocess)
# ---------------------------------------------------------------------------


def test_regularization_and_diagonal_preconditioner_follow_sauce_support_masks(
    tmp_path,
):
    fake = FakeImagingSite(seed=5, support_masks={"model.vp": [1, 0, 1, 1, 0]})
    problem = _problem(tmp_path, fake)
    lin = problem.linearize()
    space = lin.space
    assert space.size == 6 and space.support.frozen_count == 2

    # A custom matrix acts on the active optimizer layout, independently of
    # native spatial regularization and its full-model fixed coefficients.
    matrix = np.arange(3 * space.size, dtype=float).reshape(3, space.size) / 10
    regularization = Quadratic(matrix, weight=3.0).bind(space)
    R = np.sqrt(3.0) * matrix
    np.testing.assert_allclose(
        regularization.curvature_diagonal(lin.point), np.sum(R * R, 0)
    )
    v = space.random(2)
    np.testing.assert_allclose(
        regularization.value(v), 0.5 * np.sum((R @ v.values) ** 2)
    )
    np.testing.assert_allclose(
        regularization.hessian_operator(v) @ v.values, R.T @ (R @ v.values)
    )
    with pytest.raises(ValueError, match="entries"):
        regularization.value(np.ones(8))
    with pytest.raises(ValueError, match="different control space"):
        regularization.value(problem.full_space.without_support().zeros())

    unit = Diagonal(probes="unit").bind(space)
    unit.update(lin, regularization)
    J = fake.linearizations[lin.state_fingerprint].J[:, [0, 2, 3, 5, 6, 7]]
    np.testing.assert_allclose(unit.estimate.data, np.sum(np.abs(J) ** 2, axis=0))
    np.testing.assert_allclose(unit.estimate.regularization, np.sum(R * R, 0))
    assert unit.estimate.probe_count == 6 and unit.diagonal.shape == (6,)
    applied = unit.apply(lin.gradient)
    assert isinstance(applied, ControlVector) and applied.space is space
    expected = DiagonalInverseHessian(
        np.sum(np.abs(J) ** 2, axis=0) + np.sum(R * R, 0),
        block_sizes=[3, 3],
        relative_damping=1e-2,
        maximum_inverse_ratio=1e3,
    ).apply(lin.gradient.values)
    np.testing.assert_allclose(applied.values, expected)
    # a regularization bound to the unmasked full space is rejected
    with pytest.raises(ValueError, match="different control space"):
        unit.update(
            lin,
            Quadratic(sp.identity(problem.full_space.without_support().size)).bind(
                problem.full_space.without_support()
            ),
        )


def test_smooth_runs_on_the_fake_site_postprocess(tmp_path):
    fake = FakeImagingSite(seed=2)
    controls = ControlSpace(
        vp=DepthProfile("vp", "sediment", count=5),
        src=SourceParameters(signature=True),
    )
    problem = _problem(tmp_path, fake, controls=controls)
    v = problem.space.random(1)

    smoothed = smooth(v, Smoothing(kind="tv", wavelength_fraction=0.4), problem)

    assert isinstance(smoothed, ControlVector) and smoothed.space is problem.space
    np.testing.assert_allclose(
        smoothed.values, v.values
    )  # the fake smooths by identity
    job = fake.jobs[-1]
    assert isinstance(job, SmoothJob) and fake.submissions[-1]["postprocess_only"]
    assert job.gradient_file().is_file() and job.gradient_file(raw=True).is_file()
    written = ControlVectorFile.read(job.gradient_file())
    assert written.native and list(written.names) == ["vp"]
    np.testing.assert_allclose(written["vp"], v["vp"])
    assert job.to_fs()["control_sensitivities"]["active"] == ["vp"]
    # a smooth job submitted as a regular run is refused by the fake
    with pytest.raises(Exception, match="postprocess-only"):
        problem.backend.run(job)
