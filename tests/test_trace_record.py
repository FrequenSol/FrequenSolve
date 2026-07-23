import json

import numpy as np
import pytest
import xarray as xr

import frequensolve.seismic.trace_record  # noqa: F401


def test_to_segy_converts_source_and_receiver_coordinate_units(tmp_path):
    segyio = pytest.importorskip("segyio")

    simulation = {
        "Acquisition": {
            "schema": "fs-acquisition-2",
            "source_geometry": {
                "_type": "Inline",
                "kind": "scalar",
                "sources": [
                    {
                        "name": "point",
                        "coordinates": {"value": [0.5, 0.05], "units": "km"},
                    }
                ],
            },
            "receiver_groups": [
                {
                    "name": "surface",
                    "device": {
                        "_type": "ReceiverNode",
                        "components": [{"name": "p", "field": "pressure"}],
                    },
                    "coordinates": {
                        "_type": "CoordsArray",
                        "value": [[100.0, 0.0], [200.0, 10.0]],
                        "units": "m",
                    },
                }
            ],
        }
    }
    simulation_path = tmp_path / "simulation.json"
    simulation_path.write_text(json.dumps(simulation), encoding="utf-8")

    trace = xr.DataArray(
        np.zeros((3, 2), dtype=float),
        dims=("time", "receiver"),
        coords={"time": [0.0, 0.001, 0.002], "receiver": [1, 2]},
        attrs={
            "simulation": str(simulation_path),
            "project_path": str(tmp_path),
            "source_id": 1,
            "receiver_group": "surface",
        },
    )

    trace_attrs = dict(trace.attrs)
    difference = trace - xr.zeros_like(trace)
    difference.attrs.update(trace_attrs)

    out = difference.fs.to_segy(tmp_path / "trace.sgy", units_in="km", units_out="m")

    with segyio.open(str(out), mode="r", strict=False, ignore_geometry=True) as sgy:
        assert sgy.bin[segyio.BinField.MeasurementSystem] == 1

        first = sgy.header[0]
        second = sgy.header[1]
        assert first[segyio.TraceField.SourceX] == 500
        assert first[segyio.TraceField.SourceDepth] == 50
        assert first[segyio.TraceField.GroupX] == 100
        assert first[segyio.TraceField.ReceiverGroupElevation] == 0
        assert second[segyio.TraceField.GroupX] == 200
        assert second[segyio.TraceField.ReceiverGroupElevation] == 10


def test_missing_trace_metadata_error_explains_xarray_arithmetic():
    trace = xr.DataArray(
        np.zeros((3, 2), dtype=float),
        dims=("time", "receiver"),
        coords={"time": [0.0, 0.001, 0.002], "receiver": [1, 2]},
    )

    with pytest.raises(
        ValueError,
        match=(
            "missing required FrequenSolve metadata attribute 'simulation'.*"
            "Xarray arithmetic can drop DataArray attributes"
        ),
    ):
        _ = trace.fs.receiver_group
