# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import numpy as np
import pytest
from frequensolve.imaging._hv_research import HVResearchSolver, endpoint_projection
from frequensolve.imaging.hv_attributes import HVAttribute, AttributeFeatures
from frequensolve.imaging.receiver_pool import ReceiverPool


@pytest.mark.parametrize(
    "kind", ["traveltime", "compressed_traveltime", "phase", "projective_jet"]
)
@pytest.mark.parametrize("floor_kind", ["group", "local"])
def test_feature_pullback(kind, floor_kind):
    rng = np.random.default_rng(93)
    u = rng.normal(size=(20, 2)) + 1j * rng.normal(size=(20, 2))
    d = rng.normal(size=(20, 2)) + 1j * rng.normal(size=(20, 2))
    tangent = rng.normal(size=u.shape) + 1j * rng.normal(size=u.shape)
    pool = ReceiverPool.from_coordinates(
        np.arange(20)[:, None], 3, taper="gaussian", sigma=1
    )
    cfg = HVAttribute(kind=kind, floor_kind=floor_kind)
    feature = AttributeFeatures(u, d, pool, cfg)
    load = rng.normal(size=feature.predicted.shape)
    h = 1e-5
    fd = (
        AttributeFeatures(u + h * tangent, d, pool, cfg).predicted
        - AttributeFeatures(u - h * tangent, d, pool, cfg).predicted
    ) / (2 * h)
    np.testing.assert_allclose(
        np.sum(load * fd),
        np.vdot(feature.vjp(load), tangent).real,
        rtol=1e-7,
        atol=1e-8,
    )


def test_projection_uses_all_samples():
    p = endpoint_projection(101, 8)
    np.testing.assert_allclose(p @ np.ones(101), 1, atol=1e-6)
    np.testing.assert_allclose(
        p @ np.linspace(0, 1, 101), np.linspace(0, 1, 9), atol=1e-6
    )
    assert np.all(np.linalg.norm(p, axis=0) > 0)


def test_shared_hv_rotation_and_endpoint_derivative():
    solver = HVResearchSolver(12, 4)
    x = np.linspace(0, 1, 13)
    p = np.column_stack((np.sin(2 * x), np.cos(2 * x)))
    d = np.column_stack((np.sin(2 * x + 0.2), np.cos(2 * x + 0.2)))
    opts = dict(kappa=1.0, lam=0.05, epsilon=0.01, tolerance=1e-6)
    result = solver.solve(p, d, **opts)
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]])
    rotated = solver.solve(p @ rotation, d @ rotation, **opts)
    np.testing.assert_allclose(result.value, rotated.value, rtol=1e-5)
    np.testing.assert_allclose(result.dual @ rotation, rotated.dual, atol=2e-5)
    tangent = np.random.default_rng(2).normal(size=p.shape)
    h = 1e-3
    fd = (
        solver.solve(p + h * tangent, d, **opts).value
        - solver.solve(p - h * tangent, d, **opts).value
    ) / (2 * h)
    np.testing.assert_allclose(fd, np.sum(result.dual * tangent), rtol=3e-3, atol=3e-5)
    same = solver.solve(p, p, **opts)
    assert abs(same.value) < 1e-10
    assert np.linalg.norm(same.dual) < 1e-7


def test_projective_gain_invariance_and_spectral_zero():
    rng = np.random.default_rng(19)
    u = rng.normal(size=(20, 2)) + 1j * rng.normal(size=(20, 2))
    u[10, 0] = 0
    pool = ReceiverPool(np.eye(20))
    config = HVAttribute(kind="projective_jet")
    first = AttributeFeatures(u, u, pool, config)
    gain = 2 * np.exp(0.6j)
    changed = AttributeFeatures(gain * u, gain * u, pool, config)
    np.testing.assert_allclose(first.predicted, changed.predicted, atol=1e-12)
    assert np.isfinite(first.vjp(np.ones_like(first.predicted))).all()
    assert np.max(np.linalg.norm(first.predicted, axis=1)) <= 1


def test_local_floor_preserves_weak_single_arrival_time():
    amplitude = np.exp(-np.linspace(0, 20, 40)).astype(complex)
    jets = np.column_stack((amplitude, 10 * amplitude))
    pool = ReceiverPool(np.eye(len(jets)))
    local = AttributeFeatures(jets, jets, pool, HVAttribute(floor_kind="local"))
    group = AttributeFeatures(jets, jets, pool, HVAttribute(floor_kind="group"))
    np.testing.assert_allclose(
        4 * local.predicted[:, 0], 10 / (1 + 0.005**2), rtol=1e-12
    )
    assert 4 * group.predicted[-1, 0] < 0.01
