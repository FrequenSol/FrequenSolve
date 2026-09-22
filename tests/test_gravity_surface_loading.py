import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from frequensolve import GravitySurfaceBC, SeismicSimulation
from frequensolve.seismic import Acquisition, SurfacePressureLoading
from frequensolve.util.mixins import ExportContext
from frequensolve.util.store import SimulationStore

CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-a54bdda" / "trunk" / "contracts"
)
ACQUISITION_SCHEMA = CONTRACT_ROOT / "inputs" / "fs-acquisition-2" / "schema.json"


def _acquisition_validator() -> Draft202012Validator:
    registry = Registry()
    for schema_file in CONTRACT_ROOT.rglob("*.json"):
        contents = json.loads(schema_file.read_text())
        registry = registry.with_resource(
            contents["$id"], Resource.from_contents(contents)
        )
    return Draft202012Validator(
        json.loads(ACQUISITION_SCHEMA.read_text()), registry=registry
    )


def _frequency_pressure() -> xr.DataArray:
    values = np.asarray(
        [
            [[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]],
            [[-1.0 + 0.5j, 2.0 - 3.0j], [4.0 + 1.0j, -2.0 - 1.0j]],
        ]
    )
    return xr.DataArray(
        values,
        dims=("source", "x", "frequency"),
        coords={
            "source": ["wind_sea", "swell"],
            "x": ("x", [0.0, 100.0], {"units": "m"}),
            "frequency": ("frequency", [0.1, 0.2], {"units": "Hz"}),
        },
        attrs={"units": "Pa", "system": "global"},
    )


def test_frequency_pressure_store_resolves_source_outside_xarray(tmp_path):
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)
    acquisition = Acquisition(
        boundary_loadings=SurfacePressureLoading(
            _frequency_pressure(), boundary_condition="ocean_surface"
        )
    )

    payload = acquisition.to_fs(ExportContext(tmp_path, store=store))

    _acquisition_validator().validate(payload)
    loading = payload["boundary_loadings"][0]
    assert loading["boundary_condition"] == "ocean_surface"
    assert [field["source"] for field in loading["fields"]] == [
        "wind_sea",
        "swell",
    ]
    assert loading["fields"][0]["frequencies"] == [0.1, 0.2]
    assert loading["fields"][0]["rhs_normalization"] == pytest.approx(np.sqrt(113.0))
    with h5py.File(tmp_path / "simulation.h5", "r") as h5:
        first = h5[loading["fields"][0]["data"]["dataset"]]
        assert first.shape == (2, 4)
        assert list(first.attrs["dims"].astype(str)) == ["x"]
        np.testing.assert_allclose(
            first[:],
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        )


def test_pressure_mapping_allows_independent_spatial_grids(tmp_path):
    wind = xr.DataArray(
        np.ones((3, 1), dtype=np.complex128),
        dims=("x", "frequency"),
        coords={"x": [0.0, 1.0, 2.0], "frequency": [0.25]},
        attrs={"units": "Pa"},
    )
    swell = xr.DataArray(
        2.0 * np.ones((5, 1), dtype=np.complex128),
        dims=("x", "frequency"),
        coords={"x": np.linspace(-1.0, 1.0, 5), "frequency": [0.25]},
        attrs={"units": "Pa"},
    )
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)
    loading = SurfacePressureLoading(
        {"wind": wind, "swell": swell}, boundary_condition="ocean_surface"
    )

    payload = Acquisition(boundary_loadings=loading).to_fs(
        ExportContext(tmp_path, store=store)
    )

    fields = payload["boundary_loadings"][0]["fields"]
    with h5py.File(tmp_path / "simulation.h5", "r") as h5:
        assert h5[fields[0]["data"]["dataset"]].shape == (3, 2)
        assert h5[fields[1]["data"]["dataset"]].shape == (5, 2)


def test_time_pressure_store_preserves_samples_and_transform_options(tmp_path):
    pressure = xr.DataArray(
        np.asarray([[0.0, 1.0, 0.0, -1.0], [1.0, 0.0, -1.0, 0.0]]),
        dims=("x", "time"),
        coords={
            "x": ("x", [0.0, 10.0], {"units": "m"}),
            "time": ("time", [2.0, 2.5, 3.0, 3.5], {"units": "s"}),
        },
        attrs={"units": "Pa"},
        name="storm",
    )
    store = SimulationStore(tmp_path / "simulation.h5", project_path=tmp_path)
    loading = SurfacePressureLoading(
        pressure,
        boundary_condition="ocean_surface",
        window="hann",
        detrend="mean",
    )

    payload = Acquisition(boundary_loadings=loading).to_fs(
        ExportContext(tmp_path, store=store)
    )

    field = payload["boundary_loadings"][0]["fields"][0]
    assert field["domain"] == "time"
    assert field["time_origin"] == 2.0
    assert field["time_step"] == 0.5
    assert field["window"] == "hann"
    assert field["detrend"] == "mean"
    assert field["rhs_normalization"] == 1.0
    with h5py.File(tmp_path / "simulation.h5", "r") as h5:
        np.testing.assert_allclose(h5[field["data"]["dataset"]][:], pressure.values)


def test_gravity_surface_shortcut_and_acquisition_loading_share_contract(tmp_path):
    shortcut = SeismicSimulation(
        name="shortcut",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path / "shortcut",
        BCs=[
            GravitySurfaceBC(
                "top",
                name="ocean_surface",
                pressure_spectrum=_frequency_pressure(),
            )
        ],
    )
    assert shortcut.acquisition.source_field_names() == ["wind_sea", "swell"]
    assert shortcut.acquisition.source_field_ids() == [1, 2]
    shortcut_payload = shortcut.to_fs()
    assert (
        shortcut_payload["Acquisition"]["boundary_loadings"][0]["boundary_condition"]
        == "ocean_surface"
    )

    explicit = SeismicSimulation(
        name="explicit",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path / "explicit",
        BCs=[GravitySurfaceBC("top", name="ocean_surface")],
        acquisition=Acquisition(
            boundary_loadings=SurfacePressureLoading(
                _frequency_pressure(), boundary_condition="ocean_surface"
            )
        ),
    )
    explicit_payload = explicit.to_fs()
    assert (
        explicit_payload["Acquisition"]["boundary_loadings"][0]["boundary_condition"]
        == "ocean_surface"
    )


def test_simulation_rejects_missing_or_mistargeted_gravity_loading(tmp_path):
    missing = SeismicSimulation(
        name="missing",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path / "missing",
        BCs=[GravitySurfaceBC("top", name="ocean_surface")],
    )
    with pytest.raises(ValueError, match="missing loading"):
        missing.to_fs()

    mistargeted = SeismicSimulation(
        name="mistargeted",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path / "mistargeted",
        BCs=[GravitySurfaceBC("top", name="ocean_surface", unforced=True)],
        acquisition=Acquisition(
            boundary_loadings=SurfacePressureLoading(
                _frequency_pressure(), boundary_condition="not_a_boundary"
            )
        ),
    )
    with pytest.raises(ValueError, match="unknown target"):
        mistargeted.to_fs()


def test_gravity_surface_can_be_explicitly_unforced(tmp_path):
    simulation = SeismicSimulation(
        name="homogeneous",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
        BCs=[GravitySurfaceBC("top", name="ocean_surface", unforced=True)],
    )

    payload = simulation.to_fs()

    assert "Acquisition" not in payload


@pytest.mark.parametrize(
    "conditions",
    [
        ["gravity_surface", "free"],
        ["free", "gravity_surface"],
        ["gravity_surface", "pml"],
    ],
)
@pytest.mark.parametrize("name", [None, "ocean"])
def test_combined_gravity_boundaries_preserve_complete_roundtrip(conditions, name):
    from frequensolve.mesh.boundary_conditions import BoundaryCondition

    boundary = BoundaryCondition(
        name=name,
        boundaries=["top"],
        conditions=conditions,
        pml_wavelengths=2.5,
        pml_exponent=3.0,
        pml_reflectivity=0.001,
        extra={"custom_parameter": 2.0},
    )
    payload = boundary.to_fs()
    restored = BoundaryCondition.from_fs(payload)
    assert restored.to_fs() == payload
    assert restored.conditions == conditions


def test_unnamed_gravity_boundary_roundtrip_does_not_add_identity():
    from frequensolve.mesh.boundary_conditions import BoundaryCondition

    payload = {"boundaries": ["top"], "conditions": ["gravity_surface"]}
    assert BoundaryCondition.from_fs(payload).to_fs() == payload


@pytest.mark.parametrize(
    "conditions", [["gravity_surface"], ["gravity_surface", "free"]]
)
def test_generic_gravity_boundary_loading_survives_simulation_reload(
    tmp_path, conditions
):
    from frequensolve.mesh.boundary_conditions import BoundaryCondition

    simulation = SeismicSimulation(
        name="roundtrip",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
        BCs=[
            BoundaryCondition(name="ocean", boundaries=["top"], conditions=conditions)
        ],
        acquisition=Acquisition(
            boundary_loadings=SurfacePressureLoading(
                _frequency_pressure(), boundary_condition="ocean"
            )
        ),
    )
    before = simulation.to_fs()
    loaded = SeismicSimulation.load(simulation.save())
    after = loaded.to_fs()
    assert after["BCs"] == before["BCs"]
    assert after["Acquisition"]["boundary_loadings"][0]["boundary_condition"] == "ocean"
    assert loaded.BCs["ocean"].conditions == conditions


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("count", [None, 100])
def test_external_source_count_with_loading_names_is_unknown(tmp_path, encoded, count):
    from frequensolve.seismic.sources import SourceEncoding, SourceGeometry

    geometry = SourceGeometry.hdf5(
        file=tmp_path / "external.h5", dataset="coordinates", kind="scalar", count=count
    )
    encoding = (
        SourceEncoding.hdf5(
            file=tmp_path / "encoding.h5", dataset="weights", count=count
        )
        if encoded
        else None
    )
    acquisition = Acquisition(
        source_geometry=geometry,
        source_encoding=encoding,
        boundary_loadings=SurfacePressureLoading(
            _frequency_pressure(), boundary_condition="ocean"
        ),
    )
    assert acquisition.known_source_field_count() is None


@pytest.mark.parametrize(
    "names, expected", [(["wind_sea", "other"], 3), (["wind_sea", "swell"], 2)]
)
def test_source_counts_union_loading_and_physical_identities(names, expected):
    from frequensolve.seismic.sources import SourceGeometry

    acquisition = Acquisition(
        source_geometry=SourceGeometry.points(
            kind="scalar", coords=[[0, 0], [1, 0]], names=names
        ),
        boundary_loadings=SurfacePressureLoading(
            _frequency_pressure(), boundary_condition="ocean"
        ),
    )
    assert acquisition.known_source_field_count() == expected
