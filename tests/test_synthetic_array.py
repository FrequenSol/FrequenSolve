import numpy as np
import pytest

import frequensolve as fs
from frequensolve import imaging as im


@pytest.mark.parametrize("kind", ["gaussian", "hann", "uniform"])
def test_windows_units_support_and_normalized_weights(kind):
    width = fs.Q_(100, "m")
    array = getattr(im.SyntheticArray, kind)(
        **{"sigma" if kind == "gaussian" else "radius": width}
    )
    x = np.array([0, 50, 100, 200, 301.0])
    pool = array.bind(fs.Q_(x / 1000, "km"))
    matrix = pool.apply(np.eye(len(x)))
    distance = abs(x[:, None] - x)
    expected = (distance <= array.radius_m).astype(float)
    if kind == "gaussian":
        expected *= np.exp(-0.5 * (distance / 100) ** 2)
    elif kind == "hann":
        expected *= 0.5 * (1 + np.cos(np.pi * np.minimum(distance / 100, 1)))
    expected /= expected.sum(axis=1)[:, None]
    np.testing.assert_allclose(matrix, expected, atol=1e-15)
    assert matrix[0, -1] == 0
    assert im.SyntheticArray.from_dict(array.to_dict()) == array


def test_array_groups_and_truncation():
    array = im.SyntheticArray.gaussian(sigma=fs.Q_(1, "m"), truncate=1)
    pool = array.bind(fs.Q_([0, 1, 2, 0], "m"), groups=[0, 0, 0, 1])
    assert pool.nnz == 8
    np.testing.assert_allclose(pool.apply([1, 1, 1, 900]), [1, 1, 1, 900])


@pytest.mark.parametrize("width", [0, -1, np.nan, np.inf, [1, 2], 1j])
def test_invalid_array_lengths(width):
    with pytest.raises((ValueError, TypeError)):
        im.SyntheticArray.gaussian(sigma=fs.Q_(width, "m"))


def test_units_and_configuration_errors():
    with pytest.raises(TypeError, match="units"):
        im.SyntheticArray.uniform(radius=10)
    with pytest.raises(Exception):
        im.SyntheticArray.uniform(radius=fs.Q_(10, "s"))
    with pytest.raises(ValueError):
        im.SyntheticArray.gaussian(sigma=fs.Q_(1, "m"), truncate=0)
    with pytest.raises(ValueError):
        im.Comparison.waveform().__class__(
            array=im.SyntheticArray.uniform(radius=fs.Q_(1, "m"))
        )
    with pytest.raises(ValueError):
        im.Comparison.phase_derivative(reliability="bad")
    with pytest.raises(TypeError):
        im.Comparison.phase_derivative(array="gaussian")
    comparison = im.Comparison.phase_derivative(reliability="predicted")
    assert im.Comparison.from_fs(comparison.to_fs()) == comparison


def fake_jets(values):
    jets = im.SpectralJetLinearization.__new__(im.SpectralJetLinearization)
    jets.keys = [("p", 1, i + 1, 1) for i in range(3)]
    jets.groups = np.zeros(3, int)
    jets.predicted = np.asarray(values, complex)
    jets.observed = np.array([[1, 2], [1, 3], [1, 4]], complex)
    return jets


def test_comparison_binding_freezes_reliability_across_trials():
    array = im.SyntheticArray.gaussian(sigma=fs.Q_(1, "m"))
    comp = im.Comparison.phase_derivative(array=array, reliability="predicted")
    jets = fake_jets([[0.001, 1], [1, 2], [1, 4]])
    bound = comp.bind(jets, coordinates=fs.Q_([0, 1, 2], "m"))
    candidate = fake_jets([[2, 1], [2, 2], [2, 4]])
    trial = bound.trial(candidate)
    np.testing.assert_array_equal(trial.reliability, bound.reliability)
    assert trial.pool is bound.pool
    assert trial.value != bound.value
    fresh = comp.bind(candidate, coordinates=fs.Q_([0, 1, 2], "m"))
    assert not np.array_equal(fresh.reliability, trial.reliability)
    assert not bound.reliability.flags.writeable


@pytest.fixture
def synthetic_problem(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tests.imaging_fakes import FakeImagingSite
    from tests.test_imaging_problem import _problem

    comparison = im.Comparison.phase_derivative(
        array=im.SyntheticArray.gaussian(sigma=fs.Q_(2000, "km")),
        reliability="predicted",
        relative_amplitude_floor=0.1,
    )
    _, problem = _problem(
        tmp_path,
        FakeImagingSite(),
        misfit=im.Misfit.student_t(
            nu=3,
            c2=0.02,
            comparison=comparison,
            normalization=im.Normalization.explicit(4, reduction="weighted_mean"),
        ),
    )
    problem = problem.restrict(weights=[0.3, 0.7])
    problem.simulation.acquisition.receiver_groups["surface"].coordinates.units = "m"
    rng = np.random.default_rng(114)
    observed = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    offset = rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2))
    matrix = (
        rng.normal(size=(6, 2, problem.space.size))
        + 1j * rng.normal(size=(6, 2, problem.space.size))
    ) * 0.01

    def from_problem(cls, raw, *, order, state=None):
        assert order == 1
        terms = raw.misfit.objective_terms(["surface"])
        assert terms[0].comparison.kind == "waveform"
        jets = cls.__new__(cls)
        jets.space = raw.space
        full_state = raw._state_at(state)
        x = full_state.vector(raw.space).values
        factor = raw.frequencies[0] / 4
        jac = matrix * factor
        jets.keys = [(terms[0].id, s, r, 1) for s in (1, 2) for r in (1, 2, 3)]
        jets.groups = np.repeat([0, 1], 3)
        jets.observed = observed.copy()
        jets.predicted = offset + np.einsum("ijk,k->ij", jac, x)
        jets.planes = [
            SimpleNamespace(
                problem=raw,
                space=raw.space,
                state=full_state,
                point=full_state.vector(raw.space),
                job=None,
            )
        ]
        jets.jvp = lambda v: np.einsum("ijk,k->ij", jac, np.asarray(v))
        jets.vjp = lambda q: im.ControlVector(
            np.einsum("ijk,ij->k", jac.conj(), q).real, raw.space
        )
        return jets

    monkeypatch.setattr(
        im.SpectralJetLinearization, "from_problem", classmethod(from_problem)
    )
    return problem


def test_public_problem_gradient_trials_and_normal(synthetic_problem):
    problem = synthetic_problem
    x = problem.vector().values
    lin = problem.linearize(x)
    assert len(lin.parts) == 2
    np.testing.assert_allclose(lin.jacobian.H @ lin.objective_residual(), lin.gradient)
    v = np.arange(len(x)) * 0.1
    h = 1e-4
    plus, minus = lin.trial(x + h * v), lin.trial(x - h * v)
    np.testing.assert_allclose(
        (plus.value - minus.value) / (2 * h), lin.gradient.values @ v, rtol=1e-4
    )
    assert plus.gradient is None
    np.testing.assert_allclose(lin.normal @ v, lin.jacobian.H @ (lin.jacobian @ v))
    for (_, a), (_, b) in zip(lin.parts, plus.parts):
        assert a.pool is b.pool
        np.testing.assert_array_equal(a.reliability, b.reliability)
    assert problem.linearize(x, gradient=False, reference=lin).value == pytest.approx(
        lin.value
    )


def test_student_t_public_parameterization(synthetic_problem):
    lin = synthetic_problem.linearize(gradient=False)
    expected = 0
    for weight, bound in lin.parts:
        error = (bound.data.predicted - bound.data.observed).ravel()
        rho = 0.5 * (3 + 1) * np.log1p(error**2 / (3 * 0.02))
        expected += weight * np.mean(bound.data.confidence * bound.reliability * rho)
    assert lin.value == pytest.approx(expected)


def test_probe_request_routes_reliability_to_native(synthetic_problem, monkeypatch):
    problem = synthetic_problem
    with pytest.raises(ValueError, match="background"):
        problem.linearize(background=True)
    sentinel = object()
    requests = []

    def native(state, **kwargs):
        requests.append(kwargs)
        return sentinel

    monkeypatch.setattr(problem, "_linearize_state", native)
    assert problem.linearize(receiver_diagonal={}) is sentinel
    assert requests[0]["receiver_diagonal"] == {}
    assert problem.misfit.comparison.reliability == "predicted"


def test_fwi_does_not_silently_refresh_trial_reliability(synthetic_problem):
    from frequensolve.imaging.workflows import _StageObjective

    with pytest.raises(ValueError, match="explicit outer loop"):
        _StageObjective(synthetic_problem, synthetic_problem.space, None, None, {})


def test_array_configuration_distinguishes_problem_identities():
    a = im.Misfit.student_t(
        comparison=im.Comparison.phase_derivative(
            array=im.SyntheticArray.gaussian(sigma=fs.Q_(100, "m"))
        )
    )
    b = im.Misfit.student_t(
        comparison=im.Comparison.phase_derivative(
            array=im.SyntheticArray.gaussian(sigma=fs.Q_(200, "m"))
        )
    )
    assert repr(a) != repr(b)


@pytest.mark.parametrize("kind", ["gaussian", "hann", "uniform"])
@pytest.mark.parametrize("source_policy", ["frozen", "total"])
def test_native_array_comparison_roundtrip_and_routing(kind, source_policy):
    from types import SimpleNamespace

    from frequensolve.imaging._synthetic_comparison import selected

    window = getattr(im.SyntheticArray, kind)(
        **{"sigma" if kind == "gaussian" else "radius": fs.Q_(75, "m")}
    )
    comparison = im.Comparison.phase_derivative(
        array=window,
        source_derivative=source_policy,
        relative_amplitude_floor=0.005,
        confidence_floor=0.05,
    )
    payload = comparison.to_fs()
    assert payload["array"]["length_m"] == 75
    assert im.Comparison.from_fs(payload) == comparison
    problem = SimpleNamespace(
        misfit=im.Misfit.student_t(comparison=comparison),
        observed_groups=[SimpleNamespace(name="surface")],
    )
    assert not selected(problem)
