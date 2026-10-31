"""Fixed-order waveform objectives, without PDE or adjoint checks."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from frequensolve.imaging.spectral_attributes import SpectralAttribute
from frequensolve.imaging.spectral_objective import SpectralObjective


def test_waveform_jet_is_fixed_scaled_waveform_error():
    d = np.array([[1 + 2j, 3 - 1j, 2j], [0, 4, 5]], complex)
    u = 2 * d + 1j
    scales = np.array([2.0, 3.0, 5.0])
    definition = SpectralAttribute("waveform_jet", order_scales=scales)
    scales[:] = 100
    lin = definition.linearize(u, d)
    r = (u - d) / np.array([2.0, 3.0, 5.0])
    np.testing.assert_allclose(lin.residual, r / np.sqrt(6))
    np.testing.assert_allclose(lin.predicted, u / np.array([2.0, 3.0, 5.0]))
    assert lin.value == pytest.approx(0.5 * np.mean(abs(r) ** 2))
    # Closed-form L2 gradient includes physical amplitude (no trace normalization).
    np.testing.assert_allclose(
        lin.vjp(lin.residual), (u - d) / np.array([2.0, 3.0, 5.0]) ** 2 / 6
    )


def test_robust_jet_component_balancing_keeps_all_orders():
    d = np.ones((7, 3), complex)
    u = d + np.array([0.1, 2j, 7.0])
    definition = SpectralAttribute(
        "waveform_jet", order_scales=[1, 2, 3], loss="student_t", loss_scale=0.4
    )
    lin = definition.linearize(u, d)
    losses = 0.5 * np.sum(abs(lin.residual) ** 2, axis=0)
    weights = 1 / losses
    weights *= 3 / weights.sum()
    scaled = replace(
        lin,
        residual=lin.residual * np.sqrt(weights),
        jacobian=lin.jacobian * np.sqrt(weights)[None, :, None, None],
    )
    terms = 0.5 * np.sum(abs(scaled.residual) ** 2, axis=0)
    np.testing.assert_allclose(terms, np.full(3, terms[0]))
    np.testing.assert_array_equal(lin.predicted, scaled.predicted)
    assert np.count_nonzero(scaled.vjp(scaled.residual)) == 21


def test_restriction_preserves_frozen_definition_and_weights():
    class Raw:
        def restrict(self, **kwargs):
            return SimpleNamespace(selected=kwargs)

    definition = SpectralAttribute("waveform_jet", order_scales=[1, 2, 3])
    supplied = np.array([3.0, 2.0, 1.0])
    view = SpectralObjective(Raw(), definition, order=2, feature_weights=supplied)
    supplied[:] = 0
    child = view.restrict(frequencies=[2 - 0.5j])
    assert child.definition is definition
    np.testing.assert_array_equal(child.feature_weights, [3, 2, 1])
    with pytest.raises(ValueError, match="explicitly"):
        view.restrict(loss="l2")


@pytest.mark.parametrize("scales", [[0, 1], [-1, 1], [np.inf], [[1]], []])
def test_invalid_order_scales(scales):
    with pytest.raises(ValueError, match="positive fixed"):
        SpectralAttribute("waveform_jet", order_scales=scales)


def test_order_scale_length_checked():
    with pytest.raises(ValueError, match="one fixed scale"):
        SpectralAttribute("waveform_jet", order_scales=[1]).linearize(
            np.ones((2, 3)), np.ones((2, 3))
        )


def test_cached_linearization_accepts_optimizer_control_vector():
    point = SimpleNamespace(values=np.array([0.2, 0.3]))
    state = SimpleNamespace(vector=lambda space: point)

    class Raw:
        frequencies = (2 - 0.5j,)

        def state_from(self, v):
            assert v is point
            return state

        def vector(self, s=None):
            assert s is state
            return s.vector(None)

    objective = SpectralObjective(Raw(), SpectralAttribute("waveform"), order=0)
    cached = SimpleNamespace(point=point, ensure_gradient=lambda: None)
    objective._latest = cached
    assert objective.linearize(point) is cached


def test_waveform_derivatives_remain_informative_at_base_spectral_zero():
    observed = np.array([[0, 1j, 2], [0, 3, 4j]])
    predicted = observed + 1
    lin = SpectralAttribute("waveform_jet", order_scales=[1, 1, 1]).linearize(
        predicted, observed
    )
    assert lin.value == pytest.approx(0.5)
    np.testing.assert_array_equal(lin.confidence, np.ones(2))
