"""Public transfer factories, physical units, and compatibility validation."""

import dataclasses

import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.units import ureg as u

pytestmark = pytest.mark.unit


def test_transfer_factories_normalize_physical_lengths():
    assert im.Transfer.l2(smooth=0.1 * u.km) == im.Transfer.l2(smooth=100 * u.m)
    assert im.Transfer.l2(smooth=100) == im.Transfer.l2(smooth=100 * u.m)
    assert im.Transfer.l2().smoothing_length == 0
    assert im.Transfer.nodal().method == "nodal"
    assert im.Transfer.nodal().smoothing_length == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        im.Transfer.l2().smoothing_length = 10


@pytest.mark.parametrize("smooth", [-1, np.nan, np.inf, -1 * u.m])
def test_transfer_rejects_invalid_lengths(smooth):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        im.Transfer.l2(smooth=smooth)


def test_transfer_rejects_wrong_units_and_vector_lengths():
    from pint import DimensionalityError

    with pytest.raises(DimensionalityError):
        im.Transfer.l2(smooth=10 * u.s)
    with pytest.raises(TypeError, match="scalar length"):
        im.Transfer.l2(smooth=[10, 20] * u.m)
    with pytest.raises(TypeError, match="scalar length"):
        im.Transfer.l2(smooth="100 m")


@pytest.mark.parametrize(
    "legacy",
    [{"mesh_transfer": "nodal"}, {"mesh_transfer": "l2"}, {"mesh_smoothing_length": 0}],
)
def test_transfer_interfaces_cannot_be_mixed(legacy):
    with pytest.raises(ValueError, match="cannot be combined"):
        im.Stage([1], iterations=1, controls={}, transfer=im.Transfer.l2(), **legacy)
    # Validation precedes registry discovery or any solver work.
    problem = object.__new__(im.ImagingProblem)
    with pytest.raises(ValueError, match="cannot be combined"):
        problem.with_controls({}, transfer=im.Transfer.l2(), **legacy)


def test_stage_transfer_policy_survives_replace():
    stage = im.Stage(
        [1],
        iterations=1,
        controls={"vp": im.MeshParameters("vp", "solid", frequency=3, epw=2)},
        transfer=im.Transfer.l2(smooth=100 * u.m),
    )
    assert dataclasses.replace(stage, iterations=2).transfer == stage.transfer
    assert (
        dataclasses.replace(stage, transfer=im.Transfer.nodal()).transfer
        == im.Transfer.nodal()
    )
    with pytest.raises(ValueError, match="require stage controls"):
        im.Stage([1], iterations=1, transfer=im.Transfer.l2())
    with pytest.raises(TypeError, match="transfer must be"):
        im.Stage([1], iterations=1, controls={}, transfer="l2")


def test_local_wavelength_policy_and_frequency_units():
    policy = im.Transfer.l2(smooth_wavelengths=0.1, frequency=0.003 * u.kHz)
    assert policy.smoothing_wavelengths == 0.1
    assert policy.frequency == 3.0
    assert policy.smoothing_length == 0.0
    assert im.Transfer.l2(smooth_wavelengths=0.1).frequency is None
    with pytest.raises(ValueError, match="choose smooth"):
        im.Transfer.l2(smooth=0, smooth_wavelengths=0.1)
    with pytest.raises(ValueError, match="frequency requires"):
        im.Transfer.l2(frequency=3)
    for fraction in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="finite nonnegative"):
            im.Transfer.l2(smooth_wavelengths=fraction)
    for frequency in [0, -1, np.inf, 1j]:
        with pytest.raises(ValueError, match="positive scalar"):
            im.Transfer.l2(smooth_wavelengths=0.1, frequency=frequency)
