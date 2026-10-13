# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import json
from types import SimpleNamespace

import numpy as np
import pytest
from frequensolve import imaging as im
from frequensolve.imaging.spectral_attributes import (
    SpectralAttribute,
    SpectralJetLinearization,
)


@pytest.mark.parametrize(
    "kind",
    [
        "waveform",
        "phase_slope",
        "amplitude_slope",
        "log_curvature",
        "normalized_jet",
        "relative_time",
        "windowed_waveform",
    ],
)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("loss", ["l2", "pseudo_huber", "student_t"])
@pytest.mark.parametrize(
    "policy", [{}, {"confidence_floor": 0.2, "curvature_regularization": "joint"}]
)
def test_real_jvp_vjp_and_loss_gradient(kind, weighted, policy, loss):
    rng = np.random.default_rng(512)
    u, d, v = [
        rng.normal(size=(11, 4)) + 1j * rng.normal(size=(11, 4)) for _ in range(3)
    ]
    u[0, 0] = 0
    d[1, 0] = 0
    groups = np.arange(11) % 3
    row_weights = np.linspace(0, 2, len(u)) if weighted else None
    definition = SpectralAttribute(
        kind,
        time_scale=2.3,
        relative_floor=0.1,
        early_window=(1.0, -0.2),
        late_window=(0.4, -0.3, 0.2),
        loss=loss,
        loss_scale=0.2,
        **policy,
    )
    linear = definition.linearize(u, d, groups=groups, row_weights=row_weights)
    h = 1e-6
    plus = definition.linearize(u + h * v, d, groups=groups, row_weights=row_weights)
    minus = definition.linearize(u - h * v, d, groups=groups, row_weights=row_weights)
    np.testing.assert_allclose(
        linear.jvp(v), (plus.residual - minus.residual) / (2 * h), rtol=2e-6, atol=1e-7
    )
    q = rng.normal(size=linear.residual.shape) + 1j * rng.normal(
        size=linear.residual.shape
    )
    assert np.vdot(q, linear.jvp(v)).real == pytest.approx(
        np.vdot(linear.vjp(q), v).real, rel=1e-12, abs=1e-10
    )
    assert (plus.value - minus.value) / (2 * h) == pytest.approx(
        np.vdot(linear.vjp(linear.residual), v).real, rel=3e-6, abs=1e-6
    )
    exact = definition.linearize(d, d, groups=groups)
    assert exact.value == 0
    assert np.count_nonzero(exact.vjp(exact.residual)) == 0


def test_objective_weights_preserve_attributes_and_floors_and_are_frozen():
    rng = np.random.default_rng(532)
    u, d = [rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3)) for _ in range(2)]
    definition = SpectralAttribute("log_curvature")
    plain = definition.linearize(u, d)
    w = np.array([0.0, 0.1, 0.5, 1.0, 2.0])
    weighted = definition.linearize(u, d, row_weights=w)
    np.testing.assert_allclose(
        weighted.predicted, plain.predicted, rtol=1e-14, atol=1e-15
    )
    np.testing.assert_allclose(
        weighted.observed, plain.observed, rtol=1e-14, atol=1e-15
    )
    np.testing.assert_allclose(
        weighted.confidence, plain.confidence, rtol=1e-14, atol=1e-15
    )
    np.testing.assert_allclose(weighted.residual, np.sqrt(w[:, None]) * plain.residual)
    np.testing.assert_allclose(
        weighted.vjp(weighted.residual), w[:, None] * plain.vjp(plain.residual)
    )
    expected = 0.5 * np.sum(w[:, None] * abs(plain.residual) ** 2)
    assert weighted.value == pytest.approx(expected)
    w[:] = 100
    assert weighted.value == pytest.approx(expected)
    zero = definition.linearize(u, d, row_weights=np.zeros(5))
    assert zero.value == 0
    assert np.count_nonzero(zero.vjp(zero.residual)) == 0


@pytest.mark.parametrize(
    "weights", [1.0, [1], [[1], [1]], [-1, 1], [np.nan, 1], [np.inf, 1], [1j, 1]]
)
def test_invalid_row_weights(weights):
    with pytest.raises(ValueError, match="row_weights"):
        SpectralAttribute("waveform").linearize(
            np.ones((2, 1)), np.ones((2, 1)), row_weights=weights
        )


def test_trace_weights_follow_canonical_keys_across_terms():
    jets = SpectralJetLinearization.__new__(SpectralJetLinearization)
    jets.keys = [("p", 2, 3, 1), ("v", 1, 1, 2), ("p", 1, 2, 2)]
    tables = {"p": np.arange(12).reshape(2, 2, 3), "v": np.ones((1, 2, 1)) * 0.4}
    np.testing.assert_array_equal(jets.weights_from_traces(tables), [8, 0.4, 4])
    with pytest.raises(ValueError, match="mapping"):
        jets.weights_from_traces(tables["p"])
    with pytest.raises(ValueError, match="term ID"):
        jets.weights_from_traces({"p": tables["p"]})
    jets.keys = [("p", 2, 3, 1), ("p", 1, 2, 2)]
    np.testing.assert_array_equal(jets.weights_from_traces(tables["p"]), [8, 4])
    with pytest.raises(ValueError, match="canonical"):
        jets.weights_from_traces(np.ones((1, 2, 3)))
    with pytest.raises(ValueError, match="real"):
        jets.weights_from_traces(np.ones((2, 3)))
    with pytest.raises(ValueError, match="nonnegative"):
        jets.weights_from_traces(-np.ones((2, 2, 3)))
    with pytest.raises(ValueError, match="finite"):
        jets.weights_from_traces(np.full((2, 2, 3), np.nan))
    with pytest.raises(ValueError, match="real"):
        jets.weights_from_traces(np.ones((2, 2, 3), complex))


def moments(times, amplitudes, s, order=3):
    return np.array(
        [[np.sum(amplitudes * times**n * np.exp(-s * times)) for n in range(order + 1)]]
    )


def test_positive_two_arrival_cumulant_and_shift_invariance():
    t, a, s = np.array([1.0, 3.0]), np.array([2.0, 4.0]), 0.3
    u = moments(t, a, s)
    attr = SpectralAttribute("log_curvature", relative_floor=1e-8)
    value = attr.linearize(u, u).predicted.item()
    p = a[0] * np.exp(-s * t[0]) / u[0, 0]
    assert value.real == pytest.approx(p * (1 - p) * (t[1] - t[0]) ** 2)
    shifted = moments(t + 4, 7 * a, s)
    assert attr.linearize(shifted, shifted).predicted.item() == pytest.approx(value)


def test_ratio_floor_introduces_single_arrival_curvature():
    """The regularized attribute is not an exact cumulant near its floor."""
    time, scale, floor = 8.0, 4.0, 0.05
    observed = moments(np.array([time]), np.array([1.0]), 0.3, order=2)
    predicted = observed * 0.1  # Same arrival time, different amplitude.
    attr = SpectralAttribute("log_curvature", time_scale=scale, relative_floor=floor)
    linear = attr.linearize(predicted, observed)
    epsilon = floor * abs(observed[0, 0])
    for jet, value in ((predicted, linear.predicted), (observed, linear.observed)):
        r = abs(jet[0, 0]) ** 2 / (abs(jet[0, 0]) ** 2 + epsilon**2)
        assert value.item() == pytest.approx(r * (1 - r) * time**2 / scale**2)
    assert linear.value > 0  # Actual single-arrival cumulants both vanish.
    joint = SpectralAttribute(
        "log_curvature",
        time_scale=scale,
        relative_floor=floor,
        curvature_regularization="joint",
    ).linearize(predicted, observed)
    assert abs(joint.predicted.item()) < 1e-14
    assert abs(joint.observed.item()) < 1e-14
    assert joint.value < 1e-28


def test_joint_curvature_preserves_moment_translation_and_separate_confidence():
    rng = np.random.default_rng(62)
    u, d = [rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3)) for _ in range(2)]
    attr = SpectralAttribute(
        "log_curvature",
        relative_floor=0.01,
        confidence_floor=0.05,
        curvature_regularization="joint",
    )
    original = attr.linearize(u, d)
    shifted = []
    for a in (u, d):
        b = a.copy()
        b[:, 1] = a[:, 1] + 3 * a[:, 0]
        b[:, 2] = a[:, 2] + 6 * a[:, 1] + 9 * a[:, 0]
        shifted.append(b)
    moved = attr.linearize(*shifted)
    np.testing.assert_allclose(
        moved.predicted, original.predicted, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        moved.observed, original.observed, rtol=1e-12, atol=1e-12
    )
    legacy = SpectralAttribute("log_curvature", relative_floor=0.05).linearize(u, d)
    np.testing.assert_allclose(original.confidence, legacy.confidence, rtol=1e-14)


@pytest.mark.parametrize(
    "options",
    [
        {"confidence_floor": 0},
        {"confidence_floor": np.nan},
        {"curvature_regularization": "unknown"},
        {"loss": "unknown"},
        {"loss_scale": 0},
        {"loss_scale": np.nan},
    ],
)
def test_invalid_regularization_policy(options):
    with pytest.raises(ValueError):
        SpectralAttribute("log_curvature", **options)


def test_relative_windows_recover_spacing():
    t = np.array([1.0, 3.0])
    u = moments(t, np.array([2.0, 4.0]), 0.3)
    # Exact two-impulse selectors, not claims about finite-width wavelets.
    attr = SpectralAttribute(
        "relative_time",
        relative_floor=1e-8,
        early_window=(3.0, -1.0),
        late_window=(-1.0, 1.0),
    )
    assert attr.linearize(u, u).predicted.item() == pytest.approx(2.0)


def test_receiver_windows_and_order_budget():
    rng = np.random.default_rng(42)
    u = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    c = rng.normal(size=(4, 3))
    attr = SpectralAttribute("relative_time", late_window=c)
    all_rows = attr.linearize(u, u, groups=np.arange(4))
    for row in range(4):
        single = SpectralAttribute("relative_time", late_window=c[row]).linearize(
            u[row : row + 1], u[row : row + 1]
        )
        np.testing.assert_allclose(all_rows.predicted[row], single.predicted[0])
    with pytest.raises(ValueError, match="insufficient"):
        attr.linearize(u[:, :3], u[:, :3])


def test_normalized_jet_gain_invariance():
    u = np.array([[1 + 2j, -2 + 3j, 4 - 1j]])
    attr = SpectralAttribute("normalized_jet", relative_floor=1e-8)
    assert attr.linearize(u * 5, u).value < 1e-25
    assert attr.linearize(1j * u, u).value > 0.1


def test_windowed_waveform_uses_only_its_window_order():
    u = np.array([[1 + 2j, 3 + 4j]])
    linear = SpectralAttribute("windowed_waveform", early_window=(1, 2)).linearize(u, u)
    assert linear.value == 0


@pytest.fixture
def model_planes(tmp_path):
    """Physical-frequency planes with the native observed-minus-simulated sign."""
    rng = np.random.default_rng(101)
    count, controls = 7, 5
    space = SimpleNamespace(zeros=lambda: np.zeros(controls))
    space.equivalent = lambda other: space is other
    layout = SimpleNamespace(
        id="p",
        indices=np.arange(count),
        coordinate_keys=np.column_stack(
            (np.ones(count), np.arange(count), np.ones(count))
        ),
    )
    data_space = SimpleNamespace(size=count, term_layouts=lambda: [layout])
    payload = {
        "receiver_groups": [{"name": "p"}],
        "objective_terms": [
            {
                "id": "p",
                "receiver_group": "p",
                "comparison": {"kind": "waveform"},
                "objective": {"kind": "l2"},
                "weight": 1,
                "normalization": im.Normalization.explicit(1).to_fs(),
            }
        ],
    }
    raw = rng.normal(size=(count, 4)) + 1j * rng.normal(size=(count, 4))
    observed = rng.normal(size=(count, 4)) + 1j * rng.normal(size=(count, 4))
    matrices = rng.normal(size=(4, count, controls)) + 1j * rng.normal(
        size=(4, count, controls)
    )
    planes = []
    for k in range(4):
        path = tmp_path / f"state{k}.json"
        path.write_text(
            json.dumps(
                {
                    "fingerprints": dict(
                        model="same",
                        acquisition="same",
                        source_signature="none",
                        preprocessing="none",
                    )
                }
            )
        )
        saved = SimpleNamespace(
            path=path,
            term_configs={
                "p": [
                    {
                        "weight": 1,
                        "effective_weight_mass": 1,
                        "reduction": "sum",
                        "runtime": {"n_inactive": 0},
                        "scale": {"values": [1]},
                        "data_metric": {"weights": [1]},
                    }
                ]
            },
        )
        planes.append(
            SimpleNamespace(
                space=space,
                state=SimpleNamespace(values=np.zeros(controls)),
                frequencies=[2 - 0.05j],
                frequency_weights=np.ones(1),
                data_space=data_space,
                objective_states=[saved],
                residual_sign=1,
                problem=SimpleNamespace(
                    kernel_derivative=(
                        None
                        if k == 0
                        else dict(
                            order=k,
                            residual="derivative",
                            axis="fourier",
                            source_derivative="frozen",
                        )
                    ),
                    misfit=SimpleNamespace(to_fs=lambda groups: payload),
                    observed_groups=[],
                ),
                simulated=lambda k=k: SimpleNamespace(values=raw[:, k]),
                observed=lambda k=k: SimpleNamespace(values=observed[:, k]),
                jvp=lambda v, k=k: SimpleNamespace(values=-matrices[k] @ v),
                modeled_vjp=lambda q, k=k: (matrices[k].conj().T @ q).real,
            )
        )
    return planes, matrices


@pytest.mark.parametrize("weighted", [False, True])
def test_model_composition_sign_hz_conversion_and_chain_rule(model_planes, weighted):
    planes, matrices = model_planes
    jets = SpectralJetLinearization(planes)
    direction = np.arange(5.0) / 10
    factors = (-2j * np.pi) ** (-np.arange(4))
    expected = np.column_stack([matrix @ direction for matrix in matrices]) * factors
    np.testing.assert_allclose(jets.jvp(direction), expected)
    definition = SpectralAttribute("log_curvature")
    row_weights = np.linspace(0, 1, len(jets.predicted)) if weighted else None
    bound = jets.attribute(definition, row_weights=row_weights)
    h = 1e-6
    plus = definition.linearize(
        jets.predicted + h * expected,
        jets.observed,
        groups=jets.groups,
        row_weights=row_weights,
    )
    minus = definition.linearize(
        jets.predicted - h * expected,
        jets.observed,
        groups=jets.groups,
        row_weights=row_weights,
    )
    assert bound.gradient() @ direction == pytest.approx(
        (plus.value - minus.value) / (2 * h), rel=2e-6
    )
    assert direction @ bound.normal(direction) == pytest.approx(
        np.vdot(bound.jvp(direction), bound.jvp(direction)).real
    )


@pytest.mark.parametrize(
    "bad", ["frequency", "scale", "model", "kernel", "floor", "rows"]
)
def test_model_composition_rejects_incompatible_planes(model_planes, bad):
    planes, _ = model_planes
    plane = planes[1]
    if bad == "frequency":
        plane.frequency_weights = np.array([2.0])
    elif bad == "scale":
        plane.objective_states[0].term_configs["p"][0]["scale"]["values"] = [2]
    elif bad == "model":
        path = plane.objective_states[0].path
        record = json.loads(path.read_text())
        record["fingerprints"]["model"] = "different"
        path.write_text(json.dumps(record))
    elif bad == "kernel":
        plane.problem.kernel_derivative["source_derivative"] = "total"
    elif bad == "floor":
        plane.objective_states[0].term_configs["p"][0]["runtime"]["n_inactive"] = 1
    else:
        layout = SimpleNamespace(
            id="p",
            indices=np.arange(7),
            coordinate_keys=np.column_stack(
                (np.ones(7), np.arange(7)[::-1], np.ones(7))
            ),
        )
        plane.data_space = SimpleNamespace(size=7, term_layouts=lambda: [layout])
    with pytest.raises(ValueError):
        SpectralJetLinearization(planes)


@pytest.mark.parametrize("order", [0, 3])
def test_factory_authors_consecutive_raw_orders(model_planes, order):
    planes, _ = model_planes
    selections = []

    class Problem:
        frequencies = [2 - 0.05j]
        weights = None
        misfit = planes[0].problem.misfit
        observed_groups = []

        def with_misfit(self, misfit):
            assert misfit.explicit_terms[0].normalization == im.Normalization.explicit(
                1
            )
            assert misfit.explicit_terms[0].id == "p"
            return self

        def restrict(self, *, kernel_derivative):
            selections.append(kernel_derivative)
            k = 0 if kernel_derivative is None else kernel_derivative["order"]
            return SimpleNamespace(linearize=lambda state, gradient=False: planes[k])

    jets = SpectralJetLinearization.from_problem(Problem(), order=order)
    assert len(jets.planes) == order + 1
    assert selections[0] is None
    assert [selection["order"] for selection in selections[1:]] == list(
        range(1, order + 1)
    )


@pytest.mark.parametrize(
    "kind", ["phase_slope", "amplitude_slope", "log_curvature", "relative_time"]
)
def test_predicted_reliability_is_frozen_outside_robust_loss(kind):
    rng = np.random.default_rng(581)
    u, d = [rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3)) for _ in range(2)]
    u[0, 0] = 0
    definition = SpectralAttribute(
        kind, relative_floor=0.1, loss="student_t", loss_scale=0.2, late_window=(0, 1)
    )
    reliability = definition.reliability_weights(u, d)
    assert reliability[0] == 0
    assert not reliability.flags.writeable
    plain = definition.linearize(u, d)
    weighted = definition.linearize(u, d, reliability=reliability)
    np.testing.assert_allclose(weighted.predicted, plain.predicted)
    np.testing.assert_allclose(weighted.confidence, plain.confidence)
    np.testing.assert_allclose(
        weighted.residual, np.sqrt(reliability[:, None]) * plain.residual
    )
    np.testing.assert_allclose(
        weighted.vjp(weighted.residual),
        reliability[:, None] * plain.vjp(plain.residual),
        rtol=1e-4,
        atol=1e-12,
    )
    v = rng.normal(size=u.shape) + 1j * rng.normal(size=u.shape)
    h = 1e-6
    plus = definition.linearize(u + h * v, d, reliability=reliability)
    minus = definition.linearize(u - h * v, d, reliability=reliability)
    np.testing.assert_allclose(
        weighted.jvp(v),
        (plus.residual - minus.residual) / (2 * h),
        rtol=1e-4,
        atol=1e-8,
    )
    # Trial values retain the reference-model reliability, never recompute it.
    np.testing.assert_array_equal(plus.reliability, weighted.reliability)


def test_reliability_threshold_power_and_window_denominators():
    d = np.ones((3, 3), complex)
    u = np.ones_like(d)
    u[:, 0] = [0.005, 0.1, 1]
    attr = SpectralAttribute("phase_slope", relative_floor=0.1)
    expected = abs(u[:, 0]) ** 2 / (abs(u[:, 0]) ** 2 + 0.1**2)
    np.testing.assert_allclose(attr.reliability_weights(u, d), expected)
    np.testing.assert_allclose(attr.reliability_weights(u, d, power=2), expected**2)
    relative = SpectralAttribute(
        "relative_time", relative_floor=0.1, late_window=(0, 1)
    )
    u[2, 1] = 0
    assert relative.reliability_weights(u, d)[2] == 0
    for kind in ("waveform", "normalized_jet"):
        with pytest.raises(ValueError, match="ratio attribute"):
            SpectralAttribute(kind).reliability_weights(u, d)
    with pytest.raises(ValueError, match="positive"):
        attr.reliability_weights(u, d, floor_multiplier=0)
    with pytest.raises(ValueError, match="reliability"):
        attr.linearize(u, d, reliability=[0, 1, 2])


@pytest.mark.parametrize("loss", ["l2", "student_t", "pseudo_huber"])
def test_pooled_identity_matches_point_attribute(loss):
    from scipy.sparse import eye

    rng = np.random.default_rng(881)
    u, d = [rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3)) for _ in range(2)]
    definition = SpectralAttribute(
        "phase_slope", loss=loss, loss_scale=0.4, time_scale=3
    )
    reliability = definition.reliability_weights(u, d)
    kwargs = dict(
        groups=np.arange(9) % 2,
        row_weights=np.linspace(0, 2, 9),
        reliability=reliability,
    )
    point = definition.linearize(u, d, **kwargs)
    pooled = definition.linearize(u, d, pool=im.ReceiverPool(eye(9)), **kwargs)
    np.testing.assert_allclose(pooled.predicted, point.predicted, rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(pooled.residual, point.residual, rtol=1e-4, atol=1e-12)
    np.testing.assert_allclose(
        pooled.vjp(pooled.residual), point.vjp(point.residual), rtol=1e-4, atol=1e-12
    )


@pytest.mark.parametrize("loss", ["l2", "student_t", "pseudo_huber"])
def test_overlapping_pool_chain_rule_and_shared_loads(loss):
    rng = np.random.default_rng(882)
    u, d, v = [rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3)) for _ in range(3)]
    u[2, 0] = 0
    pool = im.ReceiverPool(
        [[1, 2, 1, 0, 0, 0, 0], [0, 0, 1, 2, 1, 0, 0], [0, 0, 0, 0, 1, 2, 1]],
        centers=[1, 3, 5],
    )
    definition = SpectralAttribute(
        "phase_slope", loss=loss, loss_scale=0.2, time_scale=4
    )
    reliability = definition.reliability_weights(u, d, pool=pool)
    kwargs = dict(pool=pool, reliability=reliability, row_weights=[1, 0.3, 1.2])
    linear = definition.linearize(u, d, **kwargs)
    h = 1e-6
    plus = definition.linearize(u + h * v, d, **kwargs)
    minus = definition.linearize(u - h * v, d, **kwargs)
    np.testing.assert_allclose(
        linear.jvp(v), (plus.residual - minus.residual) / (2 * h), rtol=1e-4, atol=1e-8
    )
    expected = (plus.value - minus.value) / (2 * h)
    assert np.vdot(linear.vjp(linear.residual), v).real == pytest.approx(
        expected, rel=1e-4, abs=1e-8
    )
    # One logical residual distributes to its three samples, no other rows/orders.
    q = np.array([[0.0], [1.0 + 2j], [0.0]])
    dual = linear.vjp(q)
    np.testing.assert_array_equal(dual[[0, 1, 5, 6]], 0)
    np.testing.assert_array_equal(dual[:, 2], 0)
    assert np.all(np.linalg.norm(dual[2:5, :2], axis=1) > 0)


def test_pooling_uses_energy_not_coherent_amplitude_and_preserves_confidence():
    # Opposite phases cancel in a waveform average, but both carry time 3.
    d = np.array([[1, 3], [-1, -3]], complex)
    pool = im.ReceiverPool([[1, 1], [1, 1]])
    definition = SpectralAttribute(
        "phase_slope", relative_floor=0.01, confidence_floor=0.05
    )
    u = d.copy()
    u[0, 0] = 0
    pooled = definition.linearize(u, d, pool=pool)
    np.testing.assert_allclose(pooled.observed, 3 / (1 + 0.01**2))
    np.testing.assert_allclose(pooled.confidence, 1 / (1 + 0.05**2))
    assert np.isfinite(pooled.vjp(pooled.residual)).all()
    assert np.all(definition.reliability_weights(u, d, pool=pool) > 0.99)
    assert definition.reliability_weights(u, d)[0] == 0
    u[:] = 0
    np.testing.assert_array_equal(definition.reliability_weights(u, d, pool=pool), 0)
    with pytest.raises(ValueError, match="traveltime"):
        SpectralAttribute("log_curvature").linearize(
            np.ones((2, 3)), np.ones((2, 3)), pool=pool
        )
    with pytest.raises(ValueError, match="mix groups"):
        definition.linearize(d, d, pool=pool, groups=[0, 1])


def test_pooled_model_composition_preserves_native_jet_adjoint_interface(model_planes):
    planes, matrices = model_planes
    jets = SpectralJetLinearization(planes)
    pool = im.ReceiverPool.from_coordinates(
        np.arange(len(jets.predicted)), 1.1, groups=jets.groups
    )
    definition = SpectralAttribute("phase_slope", loss="student_t", loss_scale=0.2)
    reliability = jets.reliability_weights(definition, pool=pool)
    calls = []
    for order, plane in enumerate(planes):
        original = plane.modeled_vjp

        def counted(dual, order=order, original=original):
            calls.append(order)
            return original(dual)

        plane.modeled_vjp = counted
    bound = jets.attribute(definition, pool=pool, reliability=reliability)
    direction = np.arange(5.0) / 10
    tangent = jets.jvp(direction)
    h = 1e-6
    plus = definition.linearize(
        jets.predicted + h * tangent,
        jets.observed,
        pool=pool,
        groups=jets.groups,
        reliability=reliability,
    )
    minus = definition.linearize(
        jets.predicted - h * tangent,
        jets.observed,
        pool=pool,
        groups=jets.groups,
        reliability=reliability,
    )
    assert bound.gradient() @ direction == pytest.approx(
        (plus.value - minus.value) / (2 * h), rel=1e-4, abs=1e-8
    )
    assert calls == [0, 1]  # All overlapping windows share two native order loads.
    assert direction @ bound.normal(direction) == pytest.approx(
        np.vdot(bound.jvp(direction), bound.jvp(direction)).real, rel=1e-4, abs=1e-8
    )


@pytest.mark.parametrize("order", range(4))
@pytest.mark.parametrize("pooled", [False, True])
def test_time_window_traveltime_matches_shifted_moments(order, pooled):
    from frequensolve.imaging.receiver_pool import ReceiverPool

    rng = np.random.default_rng(920 + order)
    u, d, tangent = [
        rng.normal(size=(9, 5)) + 1j * rng.normal(size=(9, 5)) for _ in range(3)
    ]
    window = [0.0] * order + [1.0]
    pool = (
        ReceiverPool.from_coordinates(np.arange(9.0), 2.0, taper="gaussian", sigma=1.0)
        if pooled
        else None
    )
    definition = SpectralAttribute(
        "phase_slope",
        window=window,
        time_scale=4,
        loss="student_t",
        loss_scale=0.1,
        confidence_floor=0.05,
    )
    linear = definition.linearize(u, d, pool=pool)
    direct = SpectralAttribute(
        "phase_slope",
        time_scale=4,
        loss="student_t",
        loss_scale=0.1,
        confidence_floor=0.05,
    ).linearize(u[:, order : order + 2], d[:, order : order + 2], pool=pool)
    np.testing.assert_allclose(linear.residual, direct.residual)
    eps = 1e-6
    plus = definition.linearize(u + eps * tangent, d, pool=pool)
    minus = definition.linearize(u - eps * tangent, d, pool=pool)
    np.testing.assert_allclose(
        linear.jvp(tangent),
        (plus.residual - minus.residual) / (2 * eps),
        rtol=1e-5,
        atol=1e-8,
    )
    assert np.vdot(linear.vjp(linear.residual), tangent).real == pytest.approx(
        (plus.value - minus.value) / (2 * eps), rel=1e-5, abs=1e-9
    )


def test_traveltime_polynomial_combines_moments_before_ratio():
    definition = SpectralAttribute(
        "phase_slope", window=[1.0, -0.4, 0.1], relative_floor=1e-8
    )
    arrival = np.array([2.0, 8.0])
    amplitudes = np.array([1.0, 0.3])
    moments = np.array([sum(amplitudes * arrival**k) for k in range(4)])[None, :]
    actual = definition.linearize(moments, moments).predicted[0, 0]
    weights = amplitudes * (1 - 0.4 * arrival + 0.1 * arrival**2)
    assert actual == pytest.approx(sum(weights * arrival) / sum(weights))
    with pytest.raises(ValueError, match="requires window degree"):
        definition.linearize(moments[:, :2], moments[:, :2])


def test_normalized_jet_accepts_order_zero():
    u = np.array([[1 + 2j], [3 - 1j]])
    definition = SpectralAttribute("normalized_jet", relative_floor=1e-6)
    result = definition.linearize(u, 2 * u)
    assert result.predicted.shape == (2, 1)
    np.testing.assert_allclose(result.predicted, u / abs(u), atol=1e-11)


def test_multiple_frequencies_have_distinct_keys_and_floor_groups(model_planes):
    planes, _ = model_planes
    for plane in planes:
        plane.frequencies = [2 - 0.05j, 3 - 0.05j]
        plane.frequency_weights = np.ones(2)
        old = plane.data_space.term_layouts()[0]
        layouts = [
            SimpleNamespace(
                id=old.id,
                indices=old.indices + fi * 7,
                coordinate_keys=old.coordinate_keys,
            )
            for fi in range(2)
        ]
        plane.data_space = SimpleNamespace(
            size=14,
            term_layouts=lambda frequency, layouts=layouts: [layouts[frequency]],
        )
        u, d = plane.simulated().values, plane.observed().values
        plane.simulated = lambda u=u: SimpleNamespace(values=np.r_[u, 2 * u])
        plane.observed = lambda d=d: SimpleNamespace(values=np.r_[d, 2 * d])
    jets = SpectralJetLinearization(planes)
    assert jets.predicted.shape == (14, 4)
    assert len(set(jets.keys)) == 14
    assert set(jets.groups[:7]).isdisjoint(jets.groups[7:])
    assert jets.through_order(0).predicted.shape == (14, 1)
    np.testing.assert_array_equal(
        jets.through_order(2).predicted, jets.predicted[:, :3]
    )
