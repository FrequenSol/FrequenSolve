# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""AWI algebra, band settings, and receiver artifact identity checks."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging._receiver import (
    ReceiverCollection,
    ReceiverState,
    receiver_band,
    sha256,
)
from tests.test_imaging_jobs import _assert_valid


def config(**kwargs):
    return im.AWI(
        **(
            dict(
                frequencies=[1, 1.4, 2],
                quadrature_weights=[0.2, 0.5, 0.3],
                relative_floor=0.03,
                minimum_filter_energy=1e-15,
                source_derivative="total",
                formulation="spectral_moment",
            )
            | kwargs
        )
    )


def test_independent_complex_duals_include_ratio_and_stabilization_derivatives():
    rng = np.random.default_rng(73)
    u, uf, d, df, du, duf = (
        rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3)) for _ in range(6)
    )
    weights = [0, 0.4, 2, 1]
    spec = config()
    value, gu, guf = spec.evaluate(u, uf, d, df, trace_weights=weights)
    assert value > 0
    np.testing.assert_array_equal(gu[0], 0)
    np.testing.assert_array_equal(guf[0], 0)
    for a, b in [(du, 0 * duf), (0 * du, duf), (du, duf)]:
        step = 1e-6
        fd = (
            spec.evaluate(u + step * a, uf + step * b, d, df, trace_weights=weights)[0]
            - spec.evaluate(u - step * a, uf - step * b, d, df, trace_weights=weights)[
                0
            ]
        ) / (2 * step)
        np.testing.assert_allclose(
            fd, np.real(np.vdot(gu, a) + np.vdot(guf, b)), rtol=1e-7, atol=1e-9
        )
    scaled = config(quadrature_weights=[2, 5, 3]).evaluate(
        u, uf, d, df, trace_weights=weights
    )
    for a, b in zip((value, gu, guf), scaled):
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-14)


def test_regularized_identity_is_not_a_zero_residual():
    # Zero-centered filter damping biases identity when observed power varies
    # with frequency. Keep this limitation explicit until the objective changes.
    spec = config()
    d = np.array([[1.0, 2.0, 3.0]], complex)
    df = np.ones_like(d)
    value, gu, guf = spec.evaluate(d, df, d, df)
    assert value > 0
    assert np.linalg.norm(gu) + np.linalg.norm(guf) > 0
    small = config(relative_floor=1e-9).evaluate(d, df, d, df)
    assert small[0] < value * 1e-10


def test_pure_delay_has_half_squared_delay_objective():
    spec = config(relative_floor=1e-8)
    f = np.asarray(spec.frequencies)
    delay = 0.075
    u = np.ones((2, 3), complex)
    d = 2 * np.exp(-2j * np.pi * f * delay)[None, :] * u
    value, _, _ = spec.evaluate(u, 0 * u, d, -2j * np.pi * delay * d)
    np.testing.assert_allclose(value, 0.5 * delay**2, rtol=1e-13)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(frequencies=[1]),
        dict(frequencies=[1, 1, 2]),
        dict(frequencies=[1 - 0.1j, 2 - 0.2j, 3 - 0.1j]),
        dict(quadrature_weights=[1, -1, 1]),
        dict(relative_floor=0),
        dict(minimum_filter_energy=np.nan),
        dict(source_derivative="none"),
    ],
)
def test_invalid_band_settings(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_low_energy_is_explicit_and_mask_is_fixed():
    spec = config()
    u = np.ones((2, 3), complex)
    d = u.copy()
    d[0] = 0
    with pytest.raises(ValueError, match="observed power"):
        spec.evaluate(u, u, d, d)
    assert np.isfinite(spec.evaluate(u, u, d, d, trace_weights=[0, 1])[0])
    with pytest.raises(ValueError, match="matching-filter energy"):
        spec.evaluate(0 * u, 0 * u, u, u)


def _state(root, label, frequency=1):
    path = root / f"{label}.json"
    payload = root / f"{label}.h5"
    with h5py.File(payload, "w") as h5:
        h5["keys"] = np.array([[1, 3, 1], [1, 1, 1], [1, 2, 1]], np.int64)
        for name in ["predicted", "predicted_df", "observed", "observed_df"]:
            h5[name] = np.column_stack(([3.0, 1.0, 2.0], np.zeros(3)))
    m = dict(
        schema="fs-receiver-state-1",
        frequency_hz=frequency,
        receiver_group="line",
        channel="base_df",
        n_rows=3,
        partition={"n_ranks": 1, "compatibility": "same_mesh_partition"},
        payload_precision="complex64",
        value_convention="physical_receiver",
        physics="acoustic",
        dimension=2,
        geometry="cartesian",
        modeling_mode="full_dimension",
        control_kind="material_only",
        source_mode="physical_shot",
        units="Pa",
        source_derivative="total",
        candidate_fingerprint="candidate",
        acquisition_fingerprint="acquisition",
        observation_fingerprint="observation",
        preprocessing_fingerprint="identity",
        control_registry_fingerprint=label,
        resolved_context_fingerprint=label,
        state_fingerprint=label,
        field_retention="checkpoint",
        checkpoints=[{"base_file": "base.h5", "df_file": "df.h5"}],
        shards=[dict(file=str(payload), rank=0, sha256=sha256(payload))],
    )
    path.write_text(json.dumps(m))
    return path


def test_band_matches_keys_and_preserves_distinct_registries(tmp_path):
    paths = [_state(tmp_path, str(i), f) for i, f in enumerate([1, 1.4, 2])]
    states = [ReceiverCollection.read(p) for p in paths]
    band = receiver_band(states, config().frequencies, "total")["line"]
    np.testing.assert_array_equal(band[:, 0, 0], [1, 2, 3])
    state = states[0].groups["line"]
    dual = state.write_dual(tmp_path / "dual.json", np.ones(3), 2j * np.ones(3))
    with h5py.File(json.loads(dual.read_text())["shards"][0]["file"]) as h5:
        np.testing.assert_array_equal(h5["keys"], state.keys)
        np.testing.assert_array_equal(h5["df"][:, 1], 2)
    changed = json.loads(paths[1].read_text())
    changed["candidate_fingerprint"] = "different"
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity"):
        receiver_band(
            [states[0], ReceiverCollection.read(paths[1]), states[2]],
            config().frequencies,
            "total",
        )
    paths[0].write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        state.write_dual(tmp_path / "stale.json", np.ones(3), np.ones(3))


def test_duplicate_or_corrupt_shards_are_rejected(tmp_path):
    path = _state(tmp_path, "state")
    m = json.loads(path.read_text())
    payload = tmp_path / "state.h5"
    with h5py.File(payload, "r+") as h5:
        h5["keys"][1] = h5["keys"][0]
    with pytest.raises(ValueError, match="checksum"):
        ReceiverState.read(path)
    m["shards"][0]["sha256"] = sha256(payload)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="duplicated"):
        ReceiverState.read(path)


@pytest.mark.parametrize(
    "action", ["receiver_linearize", "receiver_jvp", "receiver_vjp"]
)
def test_receiver_jobs_serialize_and_round_trip(tmp_path, action):
    from tests.test_imaging_jobs import _round_trip, _saved_simulation

    sim = _saved_simulation(tmp_path)
    options = dict(receiver_state="receiver.json", active=["model.vp"])
    if action == "receiver_linearize":
        options.update(
            channel="base_df", source_derivative="total", field_retention="checkpoint"
        )
    elif action == "receiver_jvp":
        options.update(receiver_vector="tangent.json", direction="direction.h5")
    else:
        options.update(
            receiver_vector="dual.json",
            covector="gradient.h5",
            field_reuse="checkpoint",
        )
    job = im.FWIOperatorJob("receiver", sim, [1.0, 2.0], action=action, **options)
    payload = _assert_valid(job.to_fs())
    assert "state" not in payload["fwi_operator"]
    _round_trip(job)


@pytest.mark.parametrize(
    "options",
    [
        dict(channel="base_df", source_derivative="none"),
        dict(channel="base_df", source_derivative="total", field_reuse="checkpoint"),
        dict(channel="base_df", source_derivative="total", active=["source.1.delay"]),
    ],
)
def test_receiver_jobs_reject_invalid_channels_and_controls(tmp_path, options):
    from tests.test_imaging_jobs import _saved_simulation

    with pytest.raises(ValueError):
        im.FWIOperatorJob(
            "receiver",
            _saved_simulation(tmp_path),
            [1, 2],
            action="receiver_linearize",
            **(dict(receiver_state="state.json", active=["model.vp"]) | options),
        )


def test_multiple_groups_bind_shared_fields_and_preserve_empty_rank_duals(tmp_path):
    members = []
    for index, name in enumerate(["surface", "deep"]):
        path = _state(tmp_path, name)
        payload = json.loads(path.read_text())
        payload.update(
            receiver_group=name,
            control_registry_fingerprint="shared",
            resolved_context_fingerprint="shared",
            partition={"n_ranks": 2, "compatibility": "same_mesh_partition"},
        )
        path.write_text(json.dumps(payload))
        members.append(dict(receiver_group=name, file=path.name, sha256=sha256(path)))
    bundle = tmp_path / "groups.json"
    manifest = dict(
        schema="fs-receiver-state-bundle-1", state_fingerprint="bundle", groups=members
    )
    bundle.write_text(json.dumps(manifest))
    collection = ReceiverCollection.read(bundle)
    values = {
        name: (np.arange(3) + index, 1j * np.ones(3))
        for index, name in enumerate(collection.groups)
    }
    dual = collection.write_dual(tmp_path / "dual.json", values)
    payload = json.loads(dual.read_text())
    assert payload["source_state_sha256"] == sha256(bundle)
    assert payload["state_fingerprint"] == "bundle"
    for entry in payload["groups"]:
        member = json.loads(Path(entry["file"]).read_text())
        assert entry["sha256"] == sha256(Path(entry["file"]))
        assert member["receiver_group"] == entry["receiver_group"]
        with h5py.File(member["shards"][1]["file"]) as h5:
            assert h5["keys"].shape == (0, 3)
            assert h5["base"].shape == (0, 2)
    with pytest.raises(ValueError, match="every group"):
        collection.write_dual(tmp_path / "missing.json", {"surface": values["surface"]})
    manifest["groups"] = [members[0], members[0]]
    bundle.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        ReceiverCollection.read(bundle)
    manifest["groups"] = members
    bundle.write_text(json.dumps(manifest))
    child = tmp_path / members[1]["file"]
    payload = json.loads(child.read_text())
    payload["candidate_fingerprint"] = "changed"
    child.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        ReceiverCollection.read(bundle)
    manifest["groups"][1]["sha256"] = sha256(child)
    bundle.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="share candidate"):
        ReceiverCollection.read(bundle)


def test_finite_band_filter_identity_and_implicit_gradient():
    rng = np.random.default_rng(312)
    f = np.linspace(0.5, 3, 26) - 0.05j
    spec = im.AWI(
        f,
        np.ones(len(f)),
        relative_floor=0.01,
        minimum_filter_energy=1e-12,
        source_derivative="total",
        filter_lags=np.arange(-6, 7) * 0.125,
    )
    u, d, du = (
        rng.normal(size=(4, len(f))) + 1j * rng.normal(size=(4, len(f)))
        for _ in range(3)
    )
    zero = np.zeros_like(u)
    value, gu, guf = spec.evaluate(u, zero, d, zero, trace_weights=[1, 0, 2, 1])
    assert value > 0
    np.testing.assert_array_equal(guf, 0)
    np.testing.assert_array_equal(gu[1], 0)
    h = 1e-6
    fd = (
        spec.evaluate(u + h * du, zero, d, zero, trace_weights=[1, 0, 2, 1])[0]
        - spec.evaluate(u - h * du, zero, d, zero, trace_weights=[1, 0, 2, 1])[0]
    ) / (2 * h)
    np.testing.assert_allclose(fd, np.vdot(gu, du).real, rtol=1e-6, atol=1e-9)
    # A deep notch still gives exact zero at identity, with a positive ridge.
    d[0, 8] = 0
    identity, g, gd = spec.evaluate(d, zero, d, zero)
    assert identity == 0
    np.testing.assert_array_equal(g, 0)
    np.testing.assert_array_equal(gd, 0)
    # Perturbing one frequency changes weights elsewhere through the shared fit.
    changed = u.copy()
    changed[:, 4] += 0.1j
    other = spec.evaluate(changed, zero, d, zero)[1]
    original = spec.evaluate(u, zero, d, zero)[1]
    assert np.linalg.norm((other - original)[:, 10:]) > 1e-7


def test_finite_filter_recovers_resolved_pure_delay():
    f = np.linspace(0.5, 10, 77)
    spec = im.AWI(
        f,
        np.ones(len(f)),
        relative_floor=1e-10,
        minimum_filter_energy=1e-12,
        source_derivative="total",
        filter_lags=np.arange(-10, 11) * 0.05,
    )
    u = np.ones((1, len(f)), complex)
    zero = np.zeros_like(u)
    delay = 0.2
    d = u * np.exp(-2j * np.pi * f * delay)
    value, _, _ = spec.evaluate(u, zero, d, zero)
    np.testing.assert_allclose(value, 0.5 * delay**2, rtol=1e-6)


@pytest.mark.parametrize("delay", [-0.375, -0.25, 0.25, 0.375])
def test_sparse_narrow_band_delay_pullback_matches_scalar_objective(delay):
    # This underdetermined filter has nonmonotone delay penalties. Verify its
    # actual derivative, without treating monotonicity as a supported guarantee
    # or reversing its gradient to compensate for an unsuitable objective.
    frequencies = np.linspace(1.5, 2.5, 5) - 0.1j
    spec = im.AWI(
        frequencies,
        [0.125, 0.25, 0.25, 0.25, 0.125],
        relative_floor=0.01,
        minimum_filter_energy=1e-12,
        filter_lags=np.arange(-12, 13) * 0.125,
        loss=im.Loss.student_t(nu=2, c2=0.25),
    )
    observed = np.ones((1, len(frequencies)), complex)
    zero = np.zeros_like(observed)

    def prediction(time):
        return np.exp(-2j * np.pi * frequencies * time)[None, :]

    modeled = prediction(delay)
    _, dual, _ = spec.evaluate(modeled, zero, observed, zero)
    step = 1e-6
    finite_difference = (
        spec.evaluate(prediction(delay + step), zero, observed, zero)[0]
        - spec.evaluate(prediction(delay - step), zero, observed, zero)[0]
    ) / (2 * step)
    analytic = np.vdot(dual, -2j * np.pi * frequencies * modeled).real
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    "lags", [[0.1, 0.2, 0.3], [-1, 0, 1], [-0.1, 0, 0, 0.1], [0, np.nan, 0.2]]
)
def test_finite_filter_rejects_invalid_or_aliased_support(lags):
    with pytest.raises(ValueError, match="filter_lags|frequency spacing"):
        config(filter_lags=lags, formulation="finite_filter")


def test_band_filter_requires_explicit_support():
    with pytest.raises(ValueError, match="explicit filter_lags"):
        config(formulation="finite_filter")


def test_base_receiver_states_and_duals_do_not_require_df(tmp_path):
    path = _state(tmp_path, "base")
    payload = json.loads(path.read_text())
    payload["channel"] = "base"
    payload["source_derivative"] = "none"
    with h5py.File(payload["shards"][0]["file"], "r+") as h:
        del h["predicted_df"]
        del h["observed_df"]
    payload["shards"][0]["sha256"] = sha256(Path(payload["shards"][0]["file"]))
    path.write_text(json.dumps(payload))
    state = ReceiverState.read(path)
    np.testing.assert_array_equal(state.values[:, [1, 3]], 0)
    dual = state.write_dual(tmp_path / "dual.json", np.ones(3), np.zeros(3))
    m = json.loads(dual.read_text())
    assert m["channel"] == "base"
    with h5py.File(m["shards"][0]["file"]) as h:
        assert "base" in h and "df" not in h
    with pytest.raises(ValueError, match="cannot carry df"):
        state.write_dual(tmp_path / "invalid.json", np.ones(3), np.ones(3))


def test_finite_filter_rejects_vanishing_predictions():
    spec = config(formulation="finite_filter", filter_lags=[-0.25, 0, 0.25])
    u = np.zeros((1, 3), complex)
    with pytest.raises(ValueError, match="predicted power"):
        spec.evaluate(u, u, np.ones_like(u), u)


@pytest.mark.parametrize(
    "loss", [im.Loss.student_t(nu=2, c2=0.04), im.Loss.huber(0.2), im.Loss.tukey(0.7)]
)
def test_finite_filter_robust_loss_pullback(loss):
    rng = np.random.default_rng(119)
    f = np.linspace(1, 3, 9) - 0.1j
    spec = im.AWI(
        frequencies=f,
        quadrature_weights=np.ones(9),
        relative_floor=0.03,
        minimum_filter_energy=1e-15,
        filter_lags=np.linspace(-1, 1, 9),
        loss=loss,
    )
    u, d, du = [
        rng.normal(size=(3, 9)) + 1j * rng.normal(size=(3, 9)) for _ in range(3)
    ]
    zero = np.zeros_like(u)
    value, dual, df = spec.evaluate(u, zero, d, zero)
    step = 1e-6
    derivative = (
        spec.evaluate(u + step * du, zero, d, zero)[0]
        - spec.evaluate(u - step * du, zero, d, zero)[0]
    ) / (2 * step)
    np.testing.assert_allclose(derivative, np.vdot(dual, du).real, rtol=1e-6, atol=1e-9)
    identity, gu, _ = spec.evaluate(d, zero, d, zero)
    assert identity == 0
    np.testing.assert_array_equal(gu, 0)
    assert value >= 0
    np.testing.assert_array_equal(df, 0)


def test_student_loss_reduces_large_spread_influence():
    spec = im.AWI(
        frequencies=[1, 2],
        quadrature_weights=[1, 1],
        filter_lags=[-0.2, 0, 0.2],
        relative_floor=0.01,
        minimum_filter_energy=1e-12,
        loss=im.Loss.student_t(c2=0.01),
    )
    _, influence = spec._outer_loss(np.array([0.01, 1.0]))
    assert influence[1] < influence[0] / 20
    assert spec.to_dict()["loss"] == im.Loss.student_t(c2=0.01).to_fs()


def test_regularization_uses_the_receiver_pullback_job():
    from types import SimpleNamespace

    from frequensolve.imaging.awi import AWILinearization

    lin = object.__new__(AWILinearization)
    lin.job = SimpleNamespace(action="receiver_linearize", covector=None)
    lin.vjp_job = None
    with pytest.raises(ValueError, match="gradient linearization"):
        _ = lin.regularization_job
    lin.vjp_job = SimpleNamespace(action="receiver_vjp", covector="gradient.h5")
    assert lin.regularization_job is lin.vjp_job


@pytest.mark.parametrize("filter_kind", ["real", "complex"])
@pytest.mark.parametrize("regularization", ["identity", "gain_invariant"])
def test_filter_pullback_with_model_dependent_ridge(filter_kind, regularization):
    rng = np.random.default_rng(109)
    u, d, du = [
        rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)) for _ in range(3)
    ]
    spec = im.AWI(
        frequencies=np.linspace(1.5, 2.5, 5) - 0.1j,
        quadrature_weights=[1, 2, 2, 2, 1],
        relative_floor=0.01,
        minimum_filter_energy=1e-15,
        filter_lags=np.linspace(-1.5, 1.5, 7),
        filter_kind=filter_kind,
        regularization=regularization,
        loss=im.Loss.student_t(nu=2, c2=0.25),
    )
    zero = np.zeros_like(u)
    value, dual, _ = spec.evaluate(u, zero, d, zero)
    step = 1e-6
    fd = (
        spec.evaluate(u + step * du, zero, d, zero)[0]
        - spec.evaluate(u - step * du, zero, d, zero)[0]
    ) / (2 * step)
    np.testing.assert_allclose(fd, np.vdot(dual, du).real, rtol=2e-6, atol=1e-8)
    if regularization == "gain_invariant":
        gain = 3.7 * np.exp(0.6j) if filter_kind == "complex" else 3.7
        scaled = spec.evaluate(gain * u, zero, d, zero)
        np.testing.assert_allclose(scaled[0], value, rtol=1e-12)
        np.testing.assert_allclose(
            scaled[1], dual / np.conj(gain), rtol=1e-10, atol=1e-12
        )
        value, dual, _ = spec.evaluate(gain * d, zero, d, zero)
        assert abs(value) < 1e-20
        assert np.linalg.norm(dual) < 1e-10


def test_complex_gain_invariant_filter_has_correct_delay_direction_over_supported_basin():
    f = np.linspace(1.5, 2.5, 5) - 0.1j
    delay = np.r_[np.linspace(-1.4, -0.05, 80), np.linspace(0.05, 1.4, 80)]
    spec = im.AWI(
        frequencies=f,
        quadrature_weights=[1, 2, 2, 2, 1],
        relative_floor=0.01,
        minimum_filter_energy=1e-15,
        filter_lags=np.linspace(-1.5, 1.5, 9),
        filter_kind="complex",
        regularization="gain_invariant",
        loss=im.Loss.student_t(nu=2, c2=0.25),
    )
    u = np.exp(-2j * np.pi * delay[:, None] * f)
    _, dual, _ = spec.evaluate(u, 0 * u, np.ones_like(u), 0 * u)
    slope = np.sum((dual.conj() * (-2j * np.pi * f * u)).real, axis=1)
    assert np.all(delay * slope > 0)


@pytest.mark.parametrize("physics", ["acoustic", "elastic", "coupled"])
def test_awi_material_view_accepts_wave_physics(tmp_path, physics):
    from tests.imaging_fakes import FakeImagingSite
    from tests.test_imaging_problem import _problem

    sim, problem = _problem(
        tmp_path,
        FakeImagingSite(),
        misfit=im.Misfit.l2(include_default_preprocess=False),
    )
    problem.simulation.physics = physics
    spec = im.AWI(
        frequencies=[4, 6],
        quadrature_weights=[1, 1],
        relative_floor=0.01,
        minimum_filter_energy=1e-12,
        filter_lags=[-0.1, 0, 0.1],
    )
    view = problem.awi(spec)
    assert view.simulation.physics == physics
    assert view.config is spec


@pytest.mark.parametrize("filter_kind", ["real", "complex"])
@pytest.mark.parametrize("regularization", ["identity", "gain_invariant"])
def test_array_filter_scatter_and_identity_pool(filter_kind, regularization):
    from scipy.sparse import eye

    from frequensolve.imaging.receiver_pool import ReceiverPool

    rng = np.random.default_rng(110)
    u, d, du = [
        rng.normal(size=(6, 5)) + 1j * rng.normal(size=(6, 5)) for _ in range(3)
    ]
    spec = im.AWI(
        frequencies=np.linspace(1.5, 2.5, 5) - 0.1j,
        quadrature_weights=[1, 2, 2, 2, 1],
        relative_floor=0.01,
        minimum_filter_energy=1e-15,
        filter_lags=np.linspace(-1.5, 1.5, 7),
        filter_kind=filter_kind,
        regularization=regularization,
        loss=im.Loss.student_t(nu=2, c2=0.25),
    )
    weights = [0, 1, 2, 1, 3, 1]
    plain = spec.evaluate(u, 0 * u, d, 0 * u, trace_weights=weights)
    identical = spec.evaluate(
        u, 0 * u, d, 0 * u, trace_weights=weights, pool=ReceiverPool(eye(6))
    )
    for a, b in zip(plain, identical):
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
    pool = ReceiverPool.from_coordinates(
        np.arange(6), radius=2, taper="gaussian", sigma=1
    )
    value, dual, _ = spec.evaluate(u, 0 * u, d, 0 * u, trace_weights=weights, pool=pool)
    h = 1e-6
    fd = (
        spec.evaluate(u + h * du, 0 * u, d, 0 * u, trace_weights=weights, pool=pool)[0]
        - spec.evaluate(u - h * du, 0 * u, d, 0 * u, trace_weights=weights, pool=pool)[
            0
        ]
    ) / (2 * h)
    np.testing.assert_allclose(fd, np.vdot(dual, du).real, rtol=2e-6, atol=1e-8)
    if regularization == "gain_invariant":
        gain = 2 * np.exp(0.4j) if filter_kind == "complex" else 2
        scaled = spec.evaluate(
            gain * u, 0 * u, d, 0 * u, trace_weights=weights, pool=pool
        )
        np.testing.assert_allclose(scaled[0], value, rtol=1e-12)


def test_array_binding_keeps_shots_and_components_separate(tmp_path):
    from types import SimpleNamespace

    from frequensolve import Q_
    from tests.imaging_fakes import FakeImagingSite
    from tests.test_imaging_problem import _problem

    _, problem = _problem(
        tmp_path,
        FakeImagingSite(),
        misfit=im.Misfit.l2(include_default_preprocess=False),
    )
    spec = im.AWI(
        frequencies=[4, 6],
        quadrature_weights=[1, 1],
        relative_floor=0.01,
        minimum_filter_energy=1e-12,
        filter_lags=[-0.1, 0, 0.1],
    )
    array = im.SyntheticArray.gaussian(sigma=Q_(10, "m"))
    view = problem.awi(spec, synthetic_array=array, coordinates=Q_([[0], [1]], "m"))
    keys = np.array([[1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2], [2, 1, 1], [2, 2, 1]])
    collection = SimpleNamespace(groups={"surface": SimpleNamespace(keys=keys)})
    pool = view._bind_array(collection)
    labels = np.array([1, 10, 1, 10, 100, 100])
    np.testing.assert_allclose(pool.apply(labels), labels)
    restricted = view.restrict(active=["vp"])
    assert restricted.synthetic_array == array
    np.testing.assert_allclose(restricted._bind_array(collection).apply(labels), labels)
