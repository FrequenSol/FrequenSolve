# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""Energy pullbacks are quadrature metrics, not sampled covector divisions."""

import numpy as np
import pytest

from frequensolve import CartesianGrid
from frequensolve.imaging import ControlSpace, GridParameters, SourceEnergy
from frequensolve.imaging import SmoothingConfig

pytestmark = pytest.mark.unit


def case(transform="identity"):
    grid = CartesianGrid(n=[3, 3], x0=[0, 0], x1=[2, 2], units="km")
    controls = CartesianGrid(n=[2, 2], x0=[0, 0], x1=[2, 2], units="km")
    space = ControlSpace(vp=GridParameters("vp", grid=controls, transform=transform))
    return grid, space


def bind(energy=1.0, **kwargs):
    grid, space = case()
    return SourceEnergy(
        {"vp": energy}, grid, relative_damping=0.0, maximum_inverse_ratio=None, **kwargs
    ).bind(space)


def test_energy_integrates_lumped_basis_and_preserves_covector():
    metric = bind()
    # A 1D endpoint hat integrates to 1 under trapezoidal quadrature (0.75 if squared).
    np.testing.assert_allclose(metric.raw_diagonal, 1.0)
    gradient = metric.space.ones()
    np.testing.assert_allclose(metric.apply(gradient).values, 1.0)
    np.testing.assert_array_equal(gradient.values, 1)


def test_smooth_kernel_maps_to_physical_update():
    # The covector of a constant kernel k is B.T @ (w * k); the update must be k / E.
    x, z = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    basis = np.column_stack(
        [((1 - x) * (1 - z)).ravel(), (x * (1 - z)).ravel(), ((1 - x) * z).ravel(), (x * z).ravel()]
    )
    weights = np.outer([0.5, 1, 0.5], [0.5, 1, 0.5])
    metric = bind(2.0)
    gradient = metric.space.zeros()
    gradient.values[:] = basis.T @ (3.0 * weights).ravel()
    np.testing.assert_allclose(metric.apply(gradient).values, 3.0 / 2.0)


def test_frozen_columns_do_not_remove_neighbor_quadrature():
    grid, space = case()
    reduced = space.with_support({"vp": [True, False, True, True]})
    metric = SourceEnergy({"vp": 1.0}, grid).bind(reduced)
    np.testing.assert_allclose(metric.raw_diagonal, 1.0)


def test_transform_factor_is_squared():
    grid, space = case("log")
    with pytest.raises(ValueError, match="transform derivative"):
        SourceEnergy({"vp": 1.0}, grid).bind(space)
    metric = SourceEnergy({"vp": 1.0}, grid, transform_derivative={"vp": 3.0}).bind(
        space
    )
    np.testing.assert_allclose(metric.raw_diagonal, 9.0)


def test_varying_transform_matches_explicit_lumped_mass_diagonal():
    grid, space = case("inverse")
    x, z = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    basis = np.column_stack(
        [
            ((1 - x) * (1 - z)).ravel(),
            (x * (1 - z)).ravel(),
            ((1 - x) * z).ravel(),
            (x * z).ravel(),
        ]
    )
    energy = 2 + x + z
    factor = -((2 + x) ** 2)
    weights = np.outer([0.5, 1, 0.5], [0.5, 1, 0.5])
    expected = basis.T @ (weights * energy * factor**2).ravel()
    metric = SourceEnergy(
        {"vp": energy}, grid, transform_derivative={"vp": factor}
    ).bind(space)
    np.testing.assert_allclose(metric.raw_diagonal, expected)


def test_three_dimensional_tensor_measure():
    grid = CartesianGrid(n=[3, 3, 3], x0=[0, 0, 0], x1=[2, 2, 2])
    control = CartesianGrid(n=[2, 2, 2], x0=[0, 0, 0], x1=[2, 2, 2])
    space = ControlSpace(vp=GridParameters("vp", grid=control))
    metric = SourceEnergy({"vp": 1}, grid).bind(space)
    np.testing.assert_allclose(metric.raw_diagonal, 1.0)


def test_frequency_weighting_is_linear_before_inversion():
    first, second = np.arange(9).reshape(3, 3), np.ones((3, 3))
    combined = bind(0.25 * first + 0.75 * second)
    np.testing.assert_allclose(
        combined.raw_diagonal,
        0.25 * bind(first).raw_diagonal + 0.75 * bind(second).raw_diagonal,
    )


def test_explicit_quadrature_and_zero_coverage():
    weights = np.zeros((3, 3))
    weights[0, 0] = 2
    metric = bind(quadrature_weights=weights)
    np.testing.assert_array_equal(metric.raw_diagonal, [2, 0, 0, 0])
    np.testing.assert_array_equal(
        metric.apply(metric.space.ones()).values, [0.5, 0, 0, 0]
    )
    np.testing.assert_array_equal(bind(0).apply(metric.space.ones()).values, 0)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, 1j, np.ones((2, 2))])
def test_reject_invalid_energy(value):
    with pytest.raises(ValueError):
        bind(value)


def test_missing_energy_does_not_silently_use_identity():
    grid, space = case()
    with pytest.raises(ValueError, match="missing source energy"):
        SourceEnergy({}, grid).bind(space)


def test_area_scaling_is_not_lost():
    grid, space = case()
    doubled = CartesianGrid(n=[3, 3], x0=[0, 0], x1=[2000, 2000], units="m")
    # The basis coordinates convert m -> km; the supplied density must convert
    # per km^2 -> per m^2 because quadrature uses the authored grid measure.
    a = SourceEnergy({"vp": 1}, grid).bind(space)
    b = SourceEnergy({"vp": 1e-6}, doubled).bind(space)
    np.testing.assert_allclose(a.raw_diagonal, b.raw_diagonal)


def test_control_smoothing_cannot_silently_discard_illumination():
    for kind in ("source", "cross"):
        config = SmoothingConfig(illumination_normalization=kind)
        with pytest.raises(ValueError, match="SourceEnergy"):
            config.to_control_fs()
        assert config.to_image_fs()["illumination_normalization"] == kind


class _Linearization:
    def __init__(self, point):
        self.space = point.space
        self.point = point


class _Curvature:
    def __init__(self, space, value):
        self.space = space
        self.value = value

    def curvature_diagonal(self, point):
        return np.full(self.space.size, self.value)


def test_regularization_never_moves_unilluminated_coefficients():
    grid, space = case()
    energy = np.zeros(grid.shape)
    energy[0, 0] = 1.0  # only the corner hat at (x0, z0) is illuminated
    metric = SourceEnergy({"vp": energy}, grid).bind(space)
    covered = metric.raw_diagonal > 0
    assert covered.sum() == 1
    metric.update(_Linearization(space.zeros()), _Curvature(space, 1.0e-3))
    step = metric.apply(space.ones()).values
    np.testing.assert_array_equal(step[~covered], 0.0)
    assert step[covered][0] > 0


def test_anisotropic_axes_follow_grid_storage_order():
    grid = CartesianGrid(n=[5, 3], x0=[0, 0], x1=[1, 0.5], units="km")
    controls = CartesianGrid(n=[2, 2], x0=[0, 0], x1=[1, 0.5], units="km")
    space = ControlSpace(vp=GridParameters("vp", grid=controls))
    x, z = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 0.5, 3))
    energy = 1 + 3 * x + 7 * z  # (z, x) storage, asymmetric in both axes
    basis = np.column_stack(
        [
            ((1 - x) * (1 - 2 * z)).ravel(),
            (x * (1 - 2 * z)).ravel(),
            ((1 - x) * 2 * z).ravel(),
            (x * 2 * z).ravel(),
        ]
    )
    weights = np.outer([0.125, 0.25, 0.125], [0.125, 0.25, 0.25, 0.25, 0.125])
    expected = basis.T @ (weights * energy).ravel()
    metric = SourceEnergy({"vp": energy}, grid).bind(space)
    np.testing.assert_allclose(metric.raw_diagonal, expected)


def test_labelled_energy_is_aligned_by_name_and_coordinates():
    xr = pytest.importorskip("xarray")
    grid = CartesianGrid(n=[5, 3], x0=[0, 0], x1=[1, 0.5], units="km")
    controls = CartesianGrid(n=[2, 2], x0=[0, 0], x1=[1, 0.5], units="km")
    space = ControlSpace(vp=GridParameters("vp", grid=controls))
    x, z = np.linspace(0, 1, 5), np.linspace(0, 0.5, 3)
    energy = 1 + 3 * x[None, :] + 7 * z[:, None]
    reference = SourceEnergy({"vp": energy}, grid).bind(space).raw_diagonal
    labelled = xr.DataArray(energy.T, dims=("x", "z"), coords={"x": x, "z": z})
    metric = SourceEnergy({"vp": labelled}, grid).bind(space)
    np.testing.assert_allclose(metric.raw_diagonal, reference)
    shifted = labelled.assign_coords(x=x + 0.1)
    with pytest.raises(ValueError, match="coordinate 'x'"):
        SourceEnergy({"vp": shifted}, grid).bind(space)
    with pytest.raises(ValueError, match="dims"):
        SourceEnergy({"vp": labelled.rename(x="y")}, grid).bind(space)


def test_energy_keys_must_name_distinct_active_blocks():
    grid, space = case()
    with pytest.raises(ValueError, match="more than once"):
        SourceEnergy({"vp": 1.0, "model.vp": 1.0}, grid).bind(space)
    frozen = space.with_support({"vp": [False, False, False, False]})
    with pytest.raises(ValueError, match="no active controls|at least one active"):
        SourceEnergy({"vp": 1.0}, grid).bind(frozen)


def test_transform_derivative_is_not_reused_at_a_moved_model():
    grid, space = case("log")
    metric = SourceEnergy({"vp": 1.0}, grid, transform_derivative={"vp": 2.0}).bind(
        space
    )
    metric.update(_Linearization(space.zeros()))
    metric.update(_Linearization(space.zeros()))  # same model: fine
    with pytest.raises(ValueError, match="different model"):
        metric.update(_Linearization(space.ones()))


@pytest.mark.parametrize("kind", ["source", "cross"])
def test_control_contexts_reject_illumination_when_configured(kind):
    from frequensolve.imaging import NativeRegularization, Stage
    from frequensolve.imaging._artifacts import control_smoothing

    config = SmoothingConfig(illumination_normalization=kind)
    for construct in (
        lambda: control_smoothing(config),
        lambda: Stage([3.0], 1, smoothing=config),
        lambda: NativeRegularization(smoothing=config),
    ):
        with pytest.raises(ValueError, match="illumination normalization"):
            construct()
    assert control_smoothing(SmoothingConfig()).illumination_normalization == "none"
