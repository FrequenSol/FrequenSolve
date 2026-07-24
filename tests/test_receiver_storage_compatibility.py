import h5py
import numpy as np
import pytest
import xarray as xr

from frequensolve.seismic.receivers import (
    ReceiverFiber,
    ReceiverGroup,
    ReceiverNode,
)
from frequensolve.util.mixins import ExportContext
from frequensolve.util.store import SimulationStore


def test_receiver_fiber_angle_serializes_as_legacy_equivalent_pitch():
    fiber = ReceiverFiber(gauge_length=10.0, radius=0.5, angle=60.0)

    payload = fiber.to_fs()

    assert fiber.angle == 60.0
    assert "angle" not in payload
    assert payload["pitch"] == pytest.approx(
        2.0 * np.pi * fiber.radius / np.tan(np.deg2rad(fiber.angle))
    )


def test_receiver_fiber_angle_pitch_preserves_radius_units():
    fiber = ReceiverFiber(
        gauge_length={"value": 10.0, "units": "m"},
        radius={"value": 0.5, "units": "m"},
        angle={"value": np.pi / 3.0, "units": "rad"},
    )

    payload = fiber.to_fs()

    assert "angle" not in payload
    assert payload["pitch"]["units"] == "m"
    assert payload["pitch"]["value"] == pytest.approx(
        2.0 * np.pi * 0.5 / np.tan(np.pi / 3.0)
    )


def test_large_receiver_dataarray_preserves_custom_axis_dimension(tmp_path):
    coordinates = xr.DataArray(
        np.column_stack((np.linspace(0.0, 1.0, 201), np.zeros(201))),
        dims=("receiver", "axis"),
        coords={"axis": ["x", "z"]},
    )
    group = ReceiverGroup(
        name="surface",
        device=ReceiverNode(),
        coordinates=coordinates,
    )
    path = tmp_path / "simulation.h5"
    ctx = ExportContext(
        project_path=tmp_path,
        store=SimulationStore(path, project_path=tmp_path),
    )

    payload = group.to_fs(ctx)

    assert payload["coordinates"]["file"] == (
        "simulation.h5:inputs/acquisition/receivers/surface/coordinates"
    )
    with h5py.File(path, "r") as h5:
        dset = h5["inputs/acquisition/receivers/surface/coordinates"]
        assert list(dset.attrs["dims"]) == ["receiver", "axis"]
        assert list(dset.attrs["axis"]) == ["x", "z"]
        assert "coordinate" not in dset.attrs
