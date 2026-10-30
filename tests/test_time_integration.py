"""Time scheme selection and harmonic-dispersion-controlled fixed steps."""

import cmath
import math

import numpy as np
import pint
import pytest

import frequensolve as fs


@pytest.mark.parametrize(
    "options,scheme,order",
    [
        ({}, "newmark", 2),
        ({"order": 2}, "newmark", 2),
        ({"order": 4}, "gauss", 4),
        ({"scheme": "gauss"}, "gauss", 4),
        ({"scheme": "newmark"}, "newmark", 2),
        ({"scheme": "gauss", "order": 4}, "gauss", 4),
        ({"method": fs.Gauss()}, "gauss", 4),
        ({"method": fs.Newmark(beta=0.3)}, "newmark", 2),
        ({"method": fs.Newmark(beta=0.4, gamma=0.6)}, "newmark", 1),
    ],
)
def test_scheme_defaults_and_legacy_roundtrip(options, scheme, order):
    time = fs.TimeIntegration(0, 1, 0.1, **options)
    assert (time.scheme, time.order, time.steps) == (scheme, order, 10)
    assert fs.TimeIntegration.from_fs(time.to_fs()) == time
    assert ("newmark" in time.to_fs()) == (scheme == "newmark")
    assert time.estimated_window_error is None


@pytest.mark.parametrize(
    "options",
    [
        {"order": 3},
        {"order": 4.0},
        {"order": True},
        {"scheme": "unknown"},
        {"scheme": "newmark", "order": 4},
        {"scheme": "gauss", "order": 2},
        {"method": "gauss"},
        {"method": fs.Newmark(), "order": 2},
        {"method": fs.Gauss(), "scheme": "gauss"},
    ],
)
def test_invalid_scheme(options):
    with pytest.raises((ValueError, TypeError)):
        fs.TimeIntegration(0, 1, 0.1, **options)


def harmonic_lag(x, order):
    z = 1j * x
    if order == 2:
        amplification = (1 + z / 2) / (1 - z / 2)
    else:
        amplification = (1 + z / 2 + z * z / 12) / (1 - z / 2 + z * z / 12)
    return 1 - cmath.phase(amplification) / x


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize(
    "mode,target", [("period_error", 0.001), ("window_error", 0.05)]
)
def test_selected_dt_meets_phase_bound_with_fewest_steps(order, mode, target):
    time = fs.TimeIntegration(-1, 4, order=order, f_max=4, **{mode: target})
    phase = 2 * math.pi * 4 * 5
    bound = target if mode == "period_error" else target / phase
    actual = harmonic_lag(phase / time.steps, order)
    assert actual <= bound
    assert harmonic_lag(phase / (time.steps - 1), order) > bound
    assert time.estimated_period_error == pytest.approx(actual, abs=1e-15)
    assert time.estimated_window_error == pytest.approx(actual * phase, abs=1e-13)
    assert time.times[0] == -1 and time.times[-1] == 4
    assert (
        fs.TimeIntegration.from_fs(time.to_fs(), selection=time.selection_payload())
        == time
    )


@pytest.mark.parametrize("order,denominator,power", [(2, 12, 2), (4, 720, 4)])
def test_small_step_phase_estimates_do_not_cancel(order, denominator, power):
    time = fs.TimeIntegration(0, 1, 1e-5, order=order, f_max=1)
    expected = (2 * math.pi * time.dt) ** power / denominator
    assert time.estimated_period_error == pytest.approx(expected, rel=1e-8, abs=0)


@pytest.mark.parametrize("order", [2, 4])
def test_period_target_independent_of_window_but_window_target_is_not(order):
    short = fs.TimeIntegration(0, 1, order=order, f_max=4, period_error=0.001)
    long = fs.TimeIntegration(0, 10, order=order, f_max=4, period_error=0.001)
    assert long.dt == pytest.approx(short.dt, rel=0.03)
    short = fs.TimeIntegration(0, 1, order=order, f_max=4, window_error=0.05)
    long = fs.TimeIntegration(0, 10, order=order, f_max=4, window_error=0.05)
    assert long.dt < short.dt


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"dt": 0.1, "period_error": 0.01},
        {"period_error": 0.01, "window_error": 0.1},
        {"period_error": 0},
        {"period_error": -0.1},
        {"period_error": 1},
        {"period_error": float("nan")},
        {"period_error": True},
        {"window_error": 0},
        {"window_error": float("inf")},
        {"window_error": True},
        {"window_error": 1e-100},
        {"window_error": 0.1, "f_max": None},
        {"period_error": 0.1, "f_max": 0},
        {"period_error": 0.1, "f_max": float("inf")},
        {"period_error": 0.1, "method": fs.Newmark(beta=0.3)},
        {"window_error": 0.1, "method": fs.Newmark(beta=0.4, gamma=0.6)},
    ],
)
def test_invalid_timestep_selection(options):
    with pytest.raises((ValueError, TypeError)):
        fs.TimeIntegration(0, 1, **({"f_max": 4} | options))


@pytest.mark.parametrize(
    "quantity",
    [
        fs.Q_(3, "degree"),
        fs.Q_(math.pi / 60, "radian"),
        pint.UnitRegistry().Quantity(3, "degrees"),
    ],
)
def test_window_error_accepts_pint_angles(quantity):
    time = fs.TimeIntegration(0, 5, order=4, f_max=4, window_error=quantity)
    plain = fs.TimeIntegration(0, 5, order=4, f_max=4, window_error=math.pi / 60)
    assert time.dt == plain.dt
    assert time.window_error == pytest.approx(math.pi / 60)
    assert time.estimated_window_error <= math.pi / 60
    assert isinstance(time.selection_payload()["window_error"], float)
    assert (
        fs.TimeIntegration.from_fs(time.to_fs(), selection=time.selection_payload())
        == time
    )


@pytest.mark.parametrize(
    "quantity",
    [
        fs.Q_(3, "seconds"),
        fs.Q_(1, "steradian"),
        fs.Q_(1, "dimensionless"),
        fs.Q_([1, 2], "degrees"),
        fs.Q_(-1, "radian"),
        fs.Q_(np.nan, "degree"),
    ],
)
def test_window_error_rejects_invalid_quantities(quantity):
    with pytest.raises(ValueError):
        fs.TimeIntegration(0, 1, order=4, f_max=4, window_error=quantity)


def test_selection_must_match_saved_dt():
    time = fs.TimeIntegration(0, 5, order=4, f_max=4, window_error=0.05)
    payload = time.to_fs() | {"time_step": time.dt / 2}
    with pytest.raises(ValueError, match="time_selection differs"):
        fs.TimeIntegration.from_fs(payload, selection=time.selection_payload())


def test_loose_error_target_does_not_alias():
    time = fs.TimeIntegration(0, 5, order=4, f_max=4, period_error=0.9)
    assert time.steps >= 40
    assert time.estimated_period_error is not None


@pytest.mark.parametrize("theta", [0, 1e-6, 0.01, 0.1, 0.8, 2.9])
def test_phase_fitted_propagator(theta):
    method = fs.Gauss(f_0=theta / (2 * math.pi * 0.1))
    time = fs.TimeIntegration(
        0,
        1,
        0.1,
        order=4,
        f_0=theta / (2 * math.pi * 0.1),
        f_max=theta / (2 * math.pi * 0.1) if theta else 1,
    )
    assert (
        fs.TimeIntegration.from_fs(time.to_fs(), selection=time.selection_payload())
        == time
    )
    assert time.method == method
    assert 1 / 12 <= time.b < 1 / math.pi**2
    if theta:
        z = 1j * theta
        propagator = (1 + z / 2 + time.b * z * z) / (1 - z / 2 + time.b * z * z)
        assert abs(propagator - cmath.exp(z)) < 2e-15
        assert abs(time.estimated_period_error) < 3e-15
    else:
        assert time.b == 1 / 12
        assert "gauss" not in time.to_fs()


@pytest.mark.parametrize("frequency", [-0.1, float("nan"), float("inf"), True])
def test_invalid_phase_fitting_frequency(frequency):
    with pytest.raises(ValueError):
        fs.Gauss(f_0=frequency)


def test_phase_fitting_requires_explicit_dt_and_gauss():
    with pytest.raises(ValueError, match="explicit dt"):
        fs.TimeIntegration(0, 1, order=4, f_0=0.5, f_max=2, period_error=0.001)
    with pytest.raises(ValueError, match="gauss scheme"):
        fs.TimeIntegration(0, 1, 0.1, order=2, f_0=0.5)


def test_default_solver_tolerance_is_resolved_by_backend():
    solver = fs.SolverConfig()
    assert solver.tolerance is None
    assert "tolerance" not in solver.to_fs()
    solver.tolerance = 3e-4
    assert solver.to_fs()["tolerance"] == 3e-4


def test_unit_aware_time_source_and_schedule():
    other = pint.UnitRegistry()
    time = fs.TimeIntegration(
        0 * other.ms, 1000 * other.ms, 10 * other.ms, f_max=0.02 * other.kHz
    )
    assert (time.start, time.end, time.dt, time.f_max, time.steps) == (
        0,
        1,
        0.01,
        20,
        100,
    )
    assert (
        fs.TimeIntegration.from_fs(time.to_fs(), selection=time.selection_payload())
        == time
    )
    source = fs.TimeFunction.ricker(0.01 * other.kHz, delay=300 * other.ms)
    assert source.to_fs()["frequency"] == 10
    np.testing.assert_allclose(
        source.evaluate(np.array([200, 300]) * other.ms), source.evaluate([0.2, 0.3])
    )
    schedule = fs.TimeOutput(interval=100 * other.ms)
    np.testing.assert_array_equal(schedule.steps(time), np.arange(0, 101, 10))
    assert fs.TimeOutput(times=other.Quantity([100, 200], "ms")).times == (0.1, 0.2)
    assert fs.TimeFunction.gaussian(10 * other.ms, delay=100 * other.ms).width == 0.01
    with pytest.raises(ValueError, match="compatible"):
        fs.TimeIntegration(0, 1, other.Quantity(1, "m"))
    with pytest.raises(ValueError, match="compatible"):
        fs.TimeFunction.ricker(other.Quantity(1, "s"), delay=0)


def test_time_replacement_preserves_scheme_and_resolves_selection():
    time = fs.TimeIntegration(0, 1, 0.1, order=4, f_max=4)
    refined = time.with_dt(fs.Q_(50, "ms"))
    assert refined.dt == 0.05 and refined.steps == 20 and refined.scheme == "gauss"
    assert time.dt == 0.1
    automatic = time.replace(window_error=0.01)
    assert automatic.window_error == 0.01 and automatic.estimated_window_error <= 0.01
    explicit = automatic.with_dt(0.01)
    assert explicit.window_error is None and explicit.period_error is None
    assert explicit.dt == 0.01
    assert automatic.replace(end=2).estimated_window_error <= 0.01
    assert time.replace(scheme="newmark").method == fs.Newmark()
    fitted = fs.TimeIntegration(0, 1, 0.1, method=fs.Gauss(f_0=2.0))
    assert fitted.with_dt(0.05).method.f_0 == 2.0
    assert fitted.with_dt(0.05).theta_0 == pytest.approx(fitted.theta_0 / 2)
    assert fitted.with_dt(0.05).b != fitted.b
    with pytest.raises(ValueError, match="divide"):
        time.with_dt(0.3)


def test_fitted_frequency_units_and_nyquist_limit():
    time = fs.TimeIntegration(0, 2, dt=fs.Q_(4, "ms"), order=4, f_0=fs.Q_(0.04, "kHz"))
    assert time.f_0 == 40
    assert time.theta_0 == pytest.approx(2 * math.pi * 40 * 0.004)
    assert time.to_fs()["gauss"] == {"f_0": 40.0}
    with pytest.raises(ValueError, match="below Nyquist"):
        fs.TimeIntegration(0, 1, 0.1, order=4, f_0=5)
    with pytest.raises(ValueError):
        fs.Gauss(f_0=fs.Q_(1, "s"))
