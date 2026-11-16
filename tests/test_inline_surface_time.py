# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import numpy as np
import pytest
import xarray as xr

from frequensolve.model.layered.surfaces import SimpleSurface
from frequensolve.model.property import Property
from frequensolve.util.mixins import ExportContext

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dims", [("time", "x"), ("x", "time")])
@pytest.mark.parametrize("flat", [True, False])
def test_inline_surface_frames_preserve_axis_order_and_coordinate_units(dims, flat):
    coordinates = {
        "time": {"data": [0.0, 500.0, 1000.0], "units": "ms"},
        "x": {"data": [0.0, 1.0], "units": "km"},
    }
    shape = tuple(len(coordinates[dim]["data"]) for dim in dims)
    values = np.arange(6.0).reshape(shape)
    payload = {
        "dims": list(dims),
        "coords": coordinates,
        "data": values.ravel(order="F").tolist() if flat else values.tolist(),
        "units": "m",
    }
    property_ = Property.from_value(payload)
    assert property_.darr.dims == dims
    np.testing.assert_array_equal(property_.darr.values, values)
    assert property_.darr.coords["time"].attrs["units"] == "ms"
    assert property_.darr.coords["x"].attrs["units"] == "km"
    restored = Property.from_value(property_.to_fs(preserve_inline_coordinates=True))
    xr.testing.assert_identical(restored.darr, property_.darr)


def test_flat_surface_frames_reject_inconsistent_sample_count():
    with pytest.raises(ValueError, match="coordinates require shape"):
        Property.from_value(
            {
                "dims": ["time", "x"],
                "coords": {"time": [0, 1], "x": [0, 1]},
                "data": [1, 2, 3],
            }
        )


def test_temporal_surface_export_stays_inline_with_project_store(tmp_path):
    depth = xr.DataArray(
        np.arange(6.0).reshape(3, 2),
        dims=["time", "x"],
        coords={
            "time": ("time", [0.0, 0.5, 1.0], {"units": "s"}),
            "x": ("x", [0.0, 1000.0], {"units": "m"}),
        },
        attrs={"units": "m"},
    )
    surface = SimpleSurface(name="water_top", interface=True, depth=depth)
    # A store without write methods also proves this branch creates no unsupported temporal HDF5 field.
    ctx = ExportContext(project_path=tmp_path, store=object())
    payload = surface.to_fs(ctx)
    assert "file" not in payload["depth"]
    assert payload["depth"]["data"] == depth.values.ravel(order="F").tolist()
    assert payload["depth"]["dims"] == ["time", "x"]
    restored = SimpleSurface.from_fs(payload)
    xr.testing.assert_identical(restored.data, depth)
    assert restored.to_fs(ctx) == payload


def test_temporal_depth_survives_layered_model_round_trip():
    from frequensolve.model.layered import LayeredModel
    from frequensolve.units import ureg as u

    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0] * u.km)
    depth = xr.DataArray(
        [[0.0, 0.0], [-0.001, 0.001]],
        dims=["time", "x"],
        coords={
            "time": ("time", [0.0, 1.0], {"units": "s"}),
            "x": ("x", [0.0, 1.0], {"units": "km"}),
        },
        attrs={"units": "km"},
    )
    model.add_surface(name="water_top", depth=depth)
    model.add_layer(name="water", properties={"Vp": 1.5 * u.km / u.s})
    model.add_surface(name="seabed", depth=0.012 * u.km)
    restored = LayeredModel.from_fs(model.to_fs(ExportContext(store=object())))
    top = next(surface for surface in restored.surfaces if surface.name == "water_top")
    xr.testing.assert_identical(top.data, depth)
