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
    MeshParameters,
    ModelOperator,
    SourceParameters,
)
from frequensolve.imaging._artifacts import ControlRegistryManifest, ControlVectorFile
from frequensolve.imaging._backend import Backend
from frequensolve.imaging.jobs import SmoothJob
from frequensolve.imaging.regularization import (
    TGV,
    TV,
    BoundPenalty,
    BoundPreconditioner,
    Diagonal,
    FromOperator,
    Identity,
    Penalty,
    Quadratic,
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


PENALTIES = {
    "tikhonov1": Tikhonov(0.7),
    "tikhonov2": Tikhonov(0.3, order=2),
    "tikhonov_ref": Tikhonov(0.5, weights={"src": 0.2, "salt": 1.5}),
    "tv1": TV(0.5, epsilon=0.05),
    "tv2": TV(0.2, epsilon=0.1, order=2),
    "quadratic": Quadratic(
        sp.random(9, 34, density=0.3, random_state=3, format="csr"), weight=2.0
    ),
    "sum": Tikhonov(0.4) + 0.5 * TV(0.3, epsilon=0.02),
}


def _bind(name, space):
    penalty = PENALTIES[name]
    if name == "tikhonov_ref":
        penalty = Tikhonov(0.5, weights=penalty.weights, reference=_random(space, 9))
    if name == "quadratic":
        matrix = sp.random(9, space.size, density=0.3, random_state=3, format="csr")
        penalty = Quadratic(matrix, weight=2.0, reference=space.random(4))
    return penalty.bind(space)


def _directional_derivative(bound, v, direction, h=1.0e-6):
    return (bound.value(v + h * direction) - bound.value(v - h * direction)) / (2 * h)


# ---------------------------------------------------------------------------
# penalties: values and gradients
# ---------------------------------------------------------------------------


def test_tikhonov_value_and_gradient_follow_the_scaled_difference_formula(space):
    bound = Tikhonov(0.7).bind(space)
    v = _random(space)
    blocks = v.blocks()

    def first_difference(values, nodes):
        return np.diff(values) / np.diff(nodes)

    vp_nodes = space.block("vp").control.coordinates
    rho = space.block("rho").control
    knots = np.asarray(rho.knots)
    greville = np.array([knots[i + 1 : i + 4].mean() for i in range(rho.size)])
    grid = space.block("grid").control
    lattice = blocks["model.grid"].reshape(grid.shape, order="F")
    expected = (
        0.5
        * 0.7
        * (
            np.sum(first_difference(blocks["model.vp"], vp_nodes) ** 2)
            + np.sum(first_difference(blocks["model.rho"], greville) ** 2)
            + np.sum((np.diff(lattice, axis=0) / grid.spacing[0]) ** 2)
            + np.sum((np.diff(lattice, axis=1) / grid.spacing[1]) ** 2)
        )
    )
    assert bound.value(v) == pytest.approx(expected)
    assert bound(v) == pytest.approx(expected)
    # source and interface blocks carry no penalty by default
    gradient = bound.gradient(v)
    assert isinstance(gradient, ControlVector)
    np.testing.assert_array_equal(gradient["salt"], 0.0)
    np.testing.assert_array_equal(gradient["src.signature"]["source.1.signature"], 0.0)
    # constants cost nothing
    assert bound.value(space.ones()) == pytest.approx(0.0)


@pytest.mark.parametrize("name", sorted(PENALTIES))
def test_penalty_gradients_match_finite_differences(name, space):
    bound = _bind(name, space)
    v = _random(space, 1)
    gradient = bound.gradient(v)
    assert isinstance(gradient, ControlVector) and gradient.size == space.size
    tolerance = 1.0e-4 if "tv" in name or name == "sum" else 1.0e-6
    for seed in (2, 3):
        direction = space.random(seed)
        expected = _directional_derivative(bound, v, direction)
        actual = gradient.dot(direction)
        assert abs(actual - expected) <= tolerance * max(1.0, abs(expected))
    # ndarray inputs are accepted too
    assert bound.value(v.values) == pytest.approx(bound.value(v))
    np.testing.assert_allclose(bound.gradient(v.values).values, gradient.values)


@pytest.mark.parametrize("name", sorted(PENALTIES))
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


@pytest.mark.parametrize(
    "name", ["tikhonov1", "tikhonov2", "tikhonov_ref", "quadratic"]
)
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


def test_tv_hessian_is_the_lagged_diffusivity_operator(space):
    bound = TV(0.4, epsilon=0.03).bind(space)
    v = _random(space, 11)
    # gradient(v) == H(v) @ v for the IRLS operator frozen at v (reference 0)
    np.testing.assert_allclose(
        (bound.hessian_operator(v) @ v).values, bound.gradient(v).values
    )
    assert bound.value(space.zeros()) == pytest.approx(0.0)
    assert bound.value(2.0 * space.ones()) == pytest.approx(0.0)
    assert bound.operator() is None
    second = TV(0.4, epsilon=0.03, order=2).bind(space)
    nodes = space.block("vp").control.coordinates
    linear = space.pack(
        {
            "vp": nodes,
            "rho": np.zeros(5),
            "grid": np.zeros(12),
            "salt": np.zeros(3),
            "src": {
                "source.1.position": [0.0, 0.0],
                "source.2.position": [0.0, 0.0],
                "source.1.signature": [0.0],
                "source.2.signature": [0.0],
            },
        }
    )
    assert second.value(linear) == pytest.approx(0.0)  # affine profiles are free


def test_quadratic_wraps_quadratic_regularization_semantics(space):
    matrix = np.random.default_rng(1).standard_normal((5, space.size))
    reference = space.random(2)
    penalty = Quadratic(matrix, weight=3.0, reference=reference)
    bound = penalty.bind(space)
    toolkit = penalty.least_squares_term(space)
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


def test_frozen_dofs_drop_straddling_difference_rows(space, frozen_space):
    assert frozen_space.size == space.size - 2
    full = Tikhonov(1.0).bind(space)
    frozen = Tikhonov(1.0).bind(frozen_space)
    R_full = full.operator().matrix
    R_frozen = frozen.operator().matrix
    vp = space.slices["model.vp"]
    grid = space.slices["model.grid"]
    vp_rows = R_full[:, vp].getnnz(axis=1) > 0
    grid_rows = R_full[:, grid].getnnz(axis=1) > 0
    # profile: 5 edges, the two touching node 2 vanish; lattice: 9 + 8 edges,
    # the two touching the corner node vanish
    assert vp_rows.sum() == 5 and grid_rows.sum() == 17
    vp_frozen_rows = R_frozen[:, frozen_space.slices["model.vp"]].getnnz(axis=1) > 0
    grid_frozen_rows = R_frozen[:, frozen_space.slices["model.grid"]].getnnz(axis=1) > 0
    assert vp_frozen_rows.sum() == 3 and grid_frozen_rows.sum() == 15
    # the surviving rows are exactly the full rows that avoid the frozen nodes
    keep = np.ones(space.size, dtype=bool)
    keep[vp.start + 2] = False
    keep[grid.start] = False
    surviving = R_full[R_full[:, ~keep].getnnz(axis=1) == 0][:, keep]
    np.testing.assert_allclose(R_frozen.toarray(), surviving.toarray())
    # values agree on vectors that vanish at the frozen nodes
    v = _random(frozen_space, 4)
    lifted = space.from_sauce_vector(frozen_space.to_sauce_vector(v))
    edge_rows = surviving @ v.values
    assert frozen.value(v) == pytest.approx(0.5 * edge_rows @ edge_rows)
    assert frozen.value(v) <= full.value(lifted) + 1e-12
    tv = TV(1.0, epsilon=0.01)
    assert tv.bind(frozen_space).gradient(v).size == frozen_space.size
    with pytest.raises(ValueError, match="different control space"):
        frozen.value(lifted)


def test_weights_enable_ridge_terms_on_source_and_interface_blocks(space):
    v = _random(space, 12)
    weighted = Tikhonov(2.0, weights={"salt": 0.5, "src.signature": 1.0}).bind(space)
    plain = Tikhonov(2.0).bind(space)
    signature = np.concatenate(
        [
            space.to_sauce_vector(v)[space.full_slices[name]]
            for name in ("source.1.signature", "source.2.signature")
        ]
    )
    expected = 0.5 * 2.0 * (0.5 * np.sum(v["salt"] ** 2) + np.sum(signature**2))
    assert weighted.value(v) - plain.value(v) == pytest.approx(expected)
    gradient = weighted.gradient(v)
    np.testing.assert_allclose(gradient["salt"], 2.0 * 0.5 * v["salt"])
    np.testing.assert_array_equal(gradient["src.position"]["source.1.position"], 0.0)
    silenced = Tikhonov(2.0, weights={"vp": 0.0, "grid": 0.0, "rho": 0.0}).bind(space)
    assert silenced.value(v) == 0.0
    assert silenced.operator().shape == (0, space.size)
    with pytest.raises(KeyError):
        Tikhonov(1.0, weights={"nope": 1.0}).bind(space)
    with pytest.raises(ValueError, match="weight of"):
        Tikhonov(1.0, weights={"vp": -1.0}).bind(space)


def _registry_block(block_id, name, size, *, offset=1):
    return {
        "id": block_id,
        "name": name,
        "binding": [1, 1, block_id],
        "layout": [offset, size, 2, 1],
        "units": "",
        "actions": 3,
        "transform": 0,
        "scaling": [0.0, 1.0, -1.7976931348623157e308, 1.7976931348623157e308],
        "basis_identity": "",
        "distributed": False,
    }


def test_mesh_blocks_raise_unless_weighted_zero(simulation):
    bound = ControlSpace(
        mesh=MeshParameters("vp", "sediment", frequency=8.0, epw=2.0),
        rho=DepthProfile("rho", "sediment", count=4),
    ).bind(simulation)
    manifest = ControlRegistryManifest.from_dict(
        {
            "schema": "fs-control-registry-1",
            "fingerprint": "sha256:" + "0" * 64,
            "blocks": [
                _registry_block(1, "model.mesh", 7),
                _registry_block(2, "model.rho", 4, offset=8),
            ],
            "active_blocks": [1, 2],
            "active_offsets": [1, 8],
        }
    )
    sized = bound.with_manifest(manifest)
    with pytest.raises(NotImplementedError, match="mesh block 'model.mesh'"):
        Tikhonov(1.0).bind(sized)
    with pytest.raises(NotImplementedError, match="Smoothing"):
        TV(1.0).bind(sized)
    excluded = Tikhonov(1.0, weights={"mesh": 0.0}).bind(sized)
    v = sized.random(1)
    np.testing.assert_array_equal(excluded.gradient(v)["mesh"], 0.0)
    assert excluded.value(v) > 0.0


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------


def test_composition_sums_scales_and_flattens(space):
    tik = Tikhonov(0.4)
    tv = TV(0.3, epsilon=0.02)
    quad = Quadratic(sp.identity(space.size, format="csr"), weight=0.1)
    combined = tik + 0.5 * tv + quad
    assert isinstance(combined, Sum)
    assert isinstance(combined.penalties[1], Scaled)
    assert len(combined.penalties) == 3
    assert isinstance(2.0 * tik, Penalty) and isinstance(tik * 2.0, Scaled)
    bound = combined.bind(space)
    v = _random(space, 13)
    parts = [tik.bind(space), tv.bind(space), quad.bind(space)]
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
    assert bound.operator() is None  # TV has no square root
    quadratic_only = (tik + 2.0 * quad).bind(space)
    R = quadratic_only.operator()
    assert R is not None
    x = space.random(3)
    np.testing.assert_allclose(
        ((R.H @ R) @ x).values, (quadratic_only.hessian_operator(x) @ x).values
    )
    with pytest.raises(TypeError):
        Sum(tik, object())
    with pytest.raises(ValueError, match="at least one"):
        Sum()
    with pytest.raises(ValueError, match="penalty scale"):
        -1.0 * tik


def test_tgv_is_not_implemented(space):
    with pytest.raises(NotImplementedError, match="TV\\(order=2\\)"):
        TGV(1.0, 2.0).bind(space)


def test_penalty_validation():
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
    penalty = Tikhonov(0.2).bind(lin.space)

    unit = Diagonal(probes="unit").bind(problem.space)
    assert isinstance(unit, BoundPreconditioner)
    with pytest.raises(RuntimeError, match="update"):
        unit.apply(lin.space.zeros())
    unit.update(lin, penalty)
    np.testing.assert_allclose(unit.estimate.data, exact, rtol=1e-8)
    np.testing.assert_allclose(
        unit.estimate.regularization, penalty.curvature_diagonal(lin.point)
    )
    g = lin.space.random(3)
    applied = unit.apply(g)
    assert isinstance(applied, ControlVector) and applied.space is lin.space
    expected = DiagonalInverseHessian(
        exact + penalty.curvature_diagonal(lin.point),
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
    typed = FromOperator(Tikhonov(1.0).bind(space).hessian_operator(g)).bind(space)
    assert isinstance(typed.apply(g), ControlVector)
    with pytest.raises(ValueError, match="shape"):
        FromOperator(np.eye(space.size + 1)).bind(space)
    with pytest.raises(ValueError, match="entries"):
        FromOperator(lambda x: np.ones(3)).bind(space).apply(g)
    # the action drops into SciPy's cg as ``M``
    H = Tikhonov(1.0).bind(space).hessian_operator(g) + sp.identity(space.size)
    solution, info = cg(
        H, g.values, M=FromOperator(np.diag(scale)).bind(space).operator()
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


def test_bound_penalty_requires_the_bound_space(space, frozen_space):
    bound = Tikhonov(1.0).bind(space)
    assert isinstance(bound, BoundPenalty)
    with pytest.raises(ValueError, match="entries"):
        bound.value(np.ones(3))
    with pytest.raises(ValueError, match="different control space"):
        bound.value(frozen_space.zeros())
    with pytest.raises(ValueError, match="real"):
        bound.value(np.ones(space.size, dtype=complex))
