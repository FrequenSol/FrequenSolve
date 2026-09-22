import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator, cg

from frequensolve.imaging import (
    ControlSpace,
    ControlVector,
    DataVector,
    DepthProfile,
    ImagingProblem,
    Jacobian,
    ModelOperator,
    Normal,
)
from tests.imaging_fakes import FakeImagingSite, layered_simulation

pytestmark = pytest.mark.unit

FREQUENCIES = [4.0, 6.0]


@pytest.fixture
def setup(tmp_path):
    fake = FakeImagingSite(seed=11)
    sim = layered_simulation(tmp_path / "project")
    problem = ImagingProblem(
        sim,
        controls=ControlSpace(
            vp=DepthProfile("vp", "sediment", count=4),
            rho=DepthProfile("rho", "sediment", count=3),
        ),
        observed={"surface": "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
    )
    problem.state = problem.state_from(np.linspace(-1.0, 1.0, 7))
    lin = problem.linearize()
    return fake, problem, lin, fake.linearizations[lin.state_fingerprint]


N = 1  # jvp / vjp / normal submit one job carrying every frequency task


def _actions(fake):
    # the fixture moves the state before linearizing: one discovery linearize
    # at the authored baseline precedes the real one
    return [s["action"] for s in fake.submissions][1:]


# ---------------------------------------------------------------------------
# Jacobian
# ---------------------------------------------------------------------------


def test_jacobian_applies_jvp_and_vjp_with_typed_vectors(setup):
    fake, problem, lin, surrogate = setup
    J = lin.jacobian

    assert isinstance(J, Jacobian) and isinstance(J, ModelOperator)
    assert isinstance(J, LinearOperator)
    assert J.shape == (lin.data_space.size, lin.space.size) == (12, 7)
    assert J.dtype == np.complex128
    assert J.domain is lin.space and J.range is lin.data_space
    assert "ControlSpace[7]" in repr(J) and "DataSpace[12]" in repr(J)

    dv = lin.space.random(1)
    j_dv = J @ dv
    assert isinstance(j_dv, DataVector) and j_dv.space == lin.data_space
    np.testing.assert_allclose(j_dv.values, surrogate.J @ dv.values, rtol=1e-12)
    assert J.matvec(dv) == j_dv
    assert J.dot(dv) == j_dv
    np.testing.assert_allclose(np.asarray(J @ dv.values), j_dv.values)
    np.testing.assert_allclose(np.asarray(J * dv.values), j_dv.values)

    r = lin.data_space.random(2)
    jh_r = J.H @ r
    assert isinstance(jh_r, ControlVector) and jh_r.space.equivalent(lin.space)
    assert jh_r.values.dtype == np.float64
    np.testing.assert_allclose(
        jh_r.values, np.real(surrogate.J.conj().T @ r.values), rtol=1e-12
    )
    assert J.rmatvec(r) == jh_r
    assert J.H.H is J
    assert isinstance(J.H, ModelOperator) and J.H.shape == (7, 12)
    np.testing.assert_allclose(np.asarray(J.H @ r.values), jh_r.values)
    # <J dv, r>_Re == <dv, J^H r>: real covector, no factor 2
    assert j_dv.dot(r) == pytest.approx(dv.dot(jh_r), rel=1e-10)
    assert J.dot_test(seed=4)["passed"]
    transposed = J.T @ r
    np.testing.assert_allclose(
        np.asarray(transposed), np.real(surrogate.J.T @ r.values), rtol=1e-12
    )
    assert J.T.T is J


def test_jacobian_memoizes_per_input_and_rejects_foreign_vectors(setup):
    fake, problem, lin, _surrogate = setup
    J = lin.jacobian
    dv = lin.space.random(3)
    r = lin.data_space.random(4)

    J @ dv
    J @ dv
    J @ dv.values
    assert _actions(fake) == ["linearize"] + ["jvp"] * N
    J.H @ r
    J.H @ r
    assert _actions(fake) == ["linearize"] + ["jvp"] * N + ["vjp"] * N
    J @ (2.0 * dv)
    assert _actions(fake) == ["linearize"] + ["jvp"] * N + ["vjp"] * N + ["jvp"] * N
    jvps = [job for job in fake.jobs if job.action == "jvp"]
    assert len({job.name for job in jvps}) == 2  # distinct jobs per direction
    assert all(job.f_list == FREQUENCIES for job in jvps)
    # the saved-state stem and a direction stem: Sauce reads each task's sibling
    assert jvps[0].state == problem.linearize().job.state_file()
    assert jvps[0].direction.name == "direction.h5"
    assert all(jvps[0].task_input(jvps[0].direction, t).is_file() for t in (1, 2))

    other = problem.restrict(active=["vp"]).space.random(0)
    with pytest.raises(ValueError, match="different control space"):
        J @ other
    with pytest.raises(ValueError, match="different data space"):
        J.H @ problem.restrict(frequencies=[4.0]).data_space.random(0)
    with pytest.raises(ValueError):
        J @ np.ones(3)
    with pytest.raises(ValueError, match="real"):
        J.matvec(dv.values + 1j)
    with pytest.raises(ValueError, match="Scalar"):
        J @ 2.0


# ---------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------


def test_normal_is_self_adjoint_and_matches_jh_j(setup):
    fake, problem, lin, surrogate = setup
    H = lin.normal
    J = lin.jacobian

    assert isinstance(H, Normal) and H.shape == (7, 7) and H.dtype == np.float64
    assert H.H is H and H.T is H
    dv = lin.space.random(5)
    h_dv = H @ dv
    assert isinstance(h_dv, ControlVector)
    np.testing.assert_allclose(
        h_dv.values, np.real(surrogate.J.conj().T @ (surrogate.J @ dv.values))
    )
    assert h_dv == J.H @ (J @ dv)
    assert H.rmatvec(dv) == h_dv
    assert _actions(fake) == ["linearize"] + ["normal"] * N + ["jvp"] * N + ["vjp"] * N
    H @ dv
    assert len(fake.submissions) == 2 + 3 * N
    assert problem.normal() is H


# ---------------------------------------------------------------------------
# composition with SciPy
# ---------------------------------------------------------------------------


def test_composition_keeps_typed_results_and_solves_with_cg(setup):
    fake, problem, lin, surrogate = setup
    H = lin.normal
    J = lin.jacobian
    n = lin.space.size
    R = sp.diags([1.0] * n) - sp.diags([1.0] * (n - 1), 1)
    alpha = 0.5

    A = H + alpha * R.T @ R
    assert isinstance(A, ModelOperator) and A.shape == (n, n)
    assert A.domain is lin.space and A.range is lin.space
    dv = lin.space.random(6)
    a_dv = A @ dv
    assert isinstance(a_dv, ControlVector)
    JHJ = np.real(surrogate.J.conj().T @ surrogate.J)
    dense = JHJ + alpha * (R.T @ R).toarray()
    np.testing.assert_allclose(a_dv.values, dense @ dv.values, rtol=1e-10)
    assert isinstance(A.H @ dv, ControlVector)
    np.testing.assert_allclose((A.H @ dv).values, a_dv.values, rtol=1e-10)

    b = lin.gradient
    x, info = cg(A, np.asarray(b), rtol=1e-12, maxiter=500)
    assert info == 0
    np.testing.assert_allclose(dense @ x, b.values, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose((A @ x).values, b.values, rtol=1e-8, atol=1e-10)

    # products, sums, scalars, negation and adjoints of composites
    JHJ_op = J.H @ J
    assert isinstance(JHJ_op, ModelOperator) and JHJ_op.shape == (n, n)
    np.testing.assert_allclose((JHJ_op @ dv).values, JHJ @ dv.values, rtol=1e-10)
    assert isinstance((2.0 * J) @ dv, DataVector)
    np.testing.assert_allclose(((2.0 * J) @ dv).values, 2.0 * (J @ dv).values)
    np.testing.assert_allclose(((J * 2.0) @ dv).values, 2.0 * (J @ dv).values)
    np.testing.assert_allclose(((-J) @ dv).values, -(J @ dv).values)
    np.testing.assert_allclose(((J - J) @ dv).values, 0.0, atol=1e-12)
    np.testing.assert_allclose(((H + H) @ dv).values, 2.0 * (H @ dv).values)
    np.testing.assert_allclose(
        ((sp.eye(n) + H) @ dv).values, dv.values + (H @ dv).values
    )
    np.testing.assert_allclose(
        ((H - sp.eye(n)) @ dv).values, (H @ dv).values - dv.values
    )
    np.testing.assert_allclose(
        ((np.eye(n) + H) @ dv).values, dv.values + (H @ dv).values
    )
    np.testing.assert_allclose(((2.0 * J).H @ (J @ dv)).values, 2.0 * (H @ dv).values)
    scipy_side = aslinearoperator(R) @ J.H
    assert isinstance(scipy_side, LinearOperator)
    np.testing.assert_allclose(
        scipy_side @ (J @ dv).values, R @ (H @ dv).values, rtol=1e-10
    )
    with pytest.raises(ValueError):
        H + sp.eye(n + 1)
    with pytest.raises(ValueError):
        H @ np.ones((n + 1, 2))


def test_to_pylops_wraps_the_operator(setup):
    pylops = pytest.importorskip("pylops")
    _fake, _problem, lin, surrogate = setup
    J = lin.jacobian

    op = J.to_pylops()

    assert isinstance(op, pylops.LinearOperator)
    assert op.shape == J.shape and op.dtype == J.dtype
    dv = lin.space.random(7)
    np.testing.assert_allclose(op @ dv.values, surrogate.J @ dv.values, rtol=1e-12)
    r = lin.data_space.random(8)
    np.testing.assert_allclose(
        op.H @ r.values, np.real(surrogate.J.conj().T @ r.values), rtol=1e-12
    )


def test_model_operator_needs_a_shape_for_untyped_sides():
    class Scale(ModelOperator):
        def _matvec(self, x):
            return 2.0 * x

        _rmatvec = _matvec

    with pytest.raises(ValueError, match="explicit size"):
        Scale(None, None)
    op = Scale(None, None, shape=(3, 3))
    np.testing.assert_array_equal(op @ np.ones(3), 2.0)
    assert isinstance(op @ np.ones(3), np.ndarray)
    assert op.H.shape == (3, 3)
