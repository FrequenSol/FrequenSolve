import h5py
import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

from frequensolve.geometry.frame import Axis, SurfaceCoordinateSystem
from frequensolve.mesh.mesh_generators import LayeredMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.layered import BoreholePart, LayeredModel
from frequensolve.model.model import ModelSubdomain
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.units import ureg as u


def test_layered_model_exports_units_and_coordinate_systems():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(depth=0.0 * u.km, name="top", system="top_surface")
    model.add_layer(
        name="layer",
        properties={
            "Vp": 1.5 * u.km / u.s,
            "Rho": {
                "value": 1000.0,
                "units": "kg/m^3",
                "system": "density_grid",
            },
        },
        system="surface_depth",
    )
    model.add_surface(
        depth={"value": 0.5, "units": "km"},
        name="bottom",
        coordinate_system="depth_surface",
    )

    payload = model.to_fs()

    assert "z_ref" not in payload["surfaces"][0]
    assert "z_phys" not in payload["surfaces"][0]
    assert payload["surfaces"][0]["depth"] == {
        "value": 0.0,
        "units": "km",
        "system": "top_surface",
    }
    assert "z_ref" not in payload["surfaces"][1]
    assert "z_phys" not in payload["surfaces"][1]
    assert payload["surfaces"][1]["depth"] == {
        "value": 0.5,
        "units": "km",
        "system": "depth_surface",
    }
    assert payload["subdomains"][0]["properties"]["vp"] == {
        "value": 1.5,
        "units": "km/s",
        "system": "surface_depth",
    }
    assert payload["subdomains"][0]["properties"]["rho"] == {
        "value": 1000.0,
        "units": "kg/m^3",
        "system": "density_grid",
    }


def test_layered_model_allows_extra_mesh_block_subdomains():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="rock", mesh_block_id=1, properties={"Vp": 1.0})
    model.add_surface(name="bottom", depth=1.0)
    model += ModelSubdomain(mesh_block_id=101, properties={"Vp": 1.2})

    payload = model.to_fs()
    roundtrip = LayeredModel.from_fs(payload)

    assert len(model.layers) == 1
    assert len(model.subdomains) == 2
    assert len(payload["subdomains"]) == 2
    assert payload["subdomains"][1]["mesh_block_id"] == 101
    assert len(roundtrip.layers) == 1
    assert len(roundtrip.subdomains) == 2
    assert roundtrip.subdomains[1].mesh_block_id == 101


def test_layered_model_exports_boreholes_and_material_subdomains():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(
        name="formation_1",
        mesh_block_id=1,
        properties={"Vp": 2.2, "Rho": 2.1},
    )
    model.add_surface(name="bottom", depth=0.5)

    model.add_borehole(
        name="bh1",
        x=0.45,
        parts=[
            {
                "name": "fluid",
                "mesh_block_id": 20,
                "r": 0.035,
                "physics": "acoustic",
                "properties": {"Vp": 1.48, "Rho": 1.03},
            },
            {
                "name": "casing",
                "mesh_block_id": 21,
                "r": 0.041,
                "physics": "elastic:iso",
                "properties": {"Vp": 5.9, "Vs": 3.2, "Rho": 7.85},
            },
            {
                "name": "cement",
                "mesh_block_id": 22,
                "r": 0.065,
                "physics": "elastic:iso",
                "properties": {"Vp": 3.4, "Vs": 1.9, "Rho": 2.1},
            },
        ],
    )

    payload = model.to_fs()

    assert [subdomain["name"] for subdomain in payload["subdomains"]] == [
        "formation_1",
        "bh1_fluid",
        "bh1_casing",
        "bh1_cement",
    ]
    assert payload["subdomains"][1] == {
        "mesh_block_id": 20,
        "name": "bh1_fluid",
        "physics": "acoustic",
        "properties": {"vp": {"value": 1.48}, "rho": {"value": 1.03}},
    }
    assert payload["boreholes"] == [
        {
            "name": "bh1",
            "axis": {"x": 0.45},
            "extent": {
                "top": {"surface": "top"},
                "bottom": {"surface": "bottom"},
            },
            "parts": [
                {
                    "name": "fluid",
                    "mesh_block_id": 20,
                    "r": {"value": 0.035},
                },
                {
                    "name": "casing",
                    "mesh_block_id": 21,
                    "r": {"value": 0.041},
                },
                {
                    "name": "cement",
                    "mesh_block_id": 22,
                    "r": {"value": 0.065},
                },
            ],
        }
    ]

    roundtrip = LayeredModel.from_fs(payload).to_fs()
    assert roundtrip["boreholes"] == payload["boreholes"]
    assert roundtrip["subdomains"][2]["properties"]["vs"] == {"value": 3.2}


def test_borehole_part_rejects_legacy_outer_radius():
    with pytest.raises(TypeError, match="outer_radius"):
        BoreholePart.from_fs(
            {
                "name": "fluid",
                "mesh_block_id": 20,
                "outer_radius": 0.035,
            }
        )


def test_borehole_part_exports_variable_radius_profile():
    radius = xr.DataArray(
        [0.035, 0.040],
        dims=["z"],
        coords={"z": [0.0, 0.5]},
        attrs={"units": "m"},
    )
    radius.coords["z"].attrs["units"] = "km"

    part = BoreholePart(name="fluid", mesh_block_id=20, r=radius)

    assert part.to_fs()["r"] == {
        "value": [0.035, 0.04],
        "dims": ["z"],
        "coords": {"z": {"value": [0.0, 0.5], "units": "km"}},
        "units": "m",
    }


def test_borehole_radius_profile_materializes_to_simulation_hdf5(tmp_path):
    radius = xr.DataArray(
        [0.035, 0.040],
        dims=["z"],
        coords={"z": [0.0, 0.5]},
        attrs={"units": "m"},
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)
    model.add_borehole(
        name="bh1",
        x=0.45,
        parts=[
            {
                "name": "fluid",
                "mesh_block_id": 20,
                "r": radius,
                "physics": "acoustic",
                "properties": {"Vp": 1.48},
            }
        ],
    )

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.model = model
    sim.mesh = MeshManager(
        LayeredMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 0.5], n=[8, 4])
    )

    payload = sim.to_fs()
    r_payload = payload["Model"]["boreholes"][0]["parts"][0]["r"]

    assert r_payload["dataset"] == "inputs/model/boreholes/bh1/parts/fluid/r"
    assert r_payload["hash"].startswith("blake3:")
    with h5py.File(tmp_path / "simulations/simple/simple.h5", "r") as h5:
        dset = h5["inputs/model/boreholes/bh1/parts/fluid/r"]
        np.testing.assert_allclose(dset[:], [0.035, 0.04])
        assert list(dset.attrs["dims"]) == ["z"]


def test_layered_mesh_generator_exports_borehole_spacing_controls():
    mesh = LayeredMeshGenerator(
        l_bound=[0.0, 0.0],
        u_bound=[1.0, 0.5],
        n=[80, 4],
    )

    mesh.refine_around_borehole(
        "bh1",
        padding=0.08,
        max_size=0.005,
        max_growth=1.5,
    )

    payload = mesh.to_fs()

    assert payload["horizontal_spacing"] == {
        "include_borehole_edges": True,
        "max_growth": 1.5,
        "controls": [{"around_borehole": "bh1", "padding": 0.08, "max_size": 0.005}],
    }
    assert (
        LayeredMeshGenerator.from_fs(payload).to_fs()["horizontal_spacing"]
        == payload["horizontal_spacing"]
    )


def test_layered_model_samples_physical_grid_without_reference_remap():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    interface = xr.DataArray(
        [0.25, 0.75],
        dims=["x"],
        coords={"x": [0.0, 1.0]},
    )

    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="upper", properties={"Vp": 1.0})
    model.add_surface(name="interface", depth=interface)
    model.add_layer(name="lower", properties={"Vp": 2.0})
    model.add_surface(name="bottom", depth=1.0)

    sampled = model.sample_uniform([3, 5])

    np.testing.assert_allclose(sampled.coords["x"].values, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        sampled.coords["z"].values,
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )
    np.testing.assert_allclose(
        sampled["vp"].values,
        [
            [1.0, 2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 2.0, 2.0],
        ],
    )


def test_layered_model_samples_surface_relative_property_grid(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="elastic",
        dimension=2,
        project_path=tmp_path,
    )
    system = SurfaceCoordinateSystem(
        name="interface_relative",
        surface="interface",
        axes=[Axis("height", direction="z", positive="up")],
    )
    sim += system

    vp = xr.DataArray(
        [1.0, 2.0],
        dims=["height"],
        coords={"height": [0.0, 0.5]},
        attrs={"units": "km/s"},
    )

    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(
        name="upper",
        properties={"Vp": {"value": vp, "coordinate_system": system.name}},
    )
    model.add_surface(name="interface", depth=0.5)
    model.add_layer(name="lower", properties={"Vp": 3.0 * u.km / u.s})
    model.add_surface(name="bottom", depth=1.0)

    sim += model

    sampled = model.sample_uniform([2, 5])
    depths, log = model.get_1D_log("Vp", x=0.5, dz=0.25)

    np.testing.assert_allclose(
        sampled["vp"].values,
        [
            [2.0, 1.5, 3.0, 3.0, 3.0],
            [2.0, 1.5, 3.0, 3.0, 3.0],
        ],
    )
    np.testing.assert_allclose(depths, [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(log, [2.0, 1.5, 3.0, 3.0])
    payload = sim.model.to_fs(sim.export_context())
    assert (
        payload["subdomains"][0]["properties"]["vp"]["system"] == "interface_relative"
    )


def test_property_coordinate_system_overrides_layer_default(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="elastic",
        dimension=2,
        project_path=tmp_path,
    )
    vp = xr.DataArray(
        [1.0, 2.0],
        dims=["height"],
        coords={"height": [0.0, 0.5]},
    )

    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(
        name="upper",
        system="layer_default",
        properties={
            "Vp": {"value": vp, "coordinate_system": "property_system"},
            "Rho": 2.0,
        },
    )
    model.add_surface(name="bottom", depth=0.5)

    sim += model
    payload = sim.model.to_fs(sim.export_context())["subdomains"][0]["properties"]

    assert payload["vp"]["system"] == "property_system"
    assert payload["rho"]["system"] == "layer_default"


def test_layered_model_sampling_converts_mixed_property_units():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="upper", properties={"Vp": 1.5 * u.km / u.s})
    model.add_surface(name="interface", depth=0.5)
    model.add_layer(name="lower", properties={"Vp": 2000.0 * u.m / u.s})
    model.add_surface(name="bottom", depth=1.0)

    sampled = model.sample_uniform([2, 3])
    depths, log = model.get_1D_log("Vp", x=0.5, dz=0.5, units="m/s")

    assert sampled["vp"].attrs["units"] == "km/s"
    np.testing.assert_allclose(
        sampled["vp"].values,
        [[1.5, 2.0, 2.0], [1.5, 2.0, 2.0]],
    )
    assert model.extreme_values("Vp") == (1.5, 2.0)
    assert model.extreme_values("Vp", units="m/s") == (1500.0, 2000.0)
    np.testing.assert_allclose(depths, [0.0, 0.5])
    np.testing.assert_allclose(log, [1500.0, 2000.0])


def test_layered_model_plot_uses_property_units():
    import matplotlib.pyplot as plt

    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"Vp": 1500.0 * u.m / u.s})
    model.add_surface(name="bottom", depth=1.0)

    fig, ax = plt.subplots()
    image = model.plot(
        "Vp",
        resolution=[2, 2],
        ax=ax,
        surfaces=False,
    )

    assert image.colorbar.ax.get_ylabel() == "Vp [m/s]"
    plt.close(fig)


def test_layered_model_export_does_not_mutate_bottom_surface_interface():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"Vp": 1.0})
    model.add_surface(name="bottom", depth=1.0)
    model.surfaces[-1].interface = False

    payload = model.to_fs()

    assert payload["surfaces"][-1]["interface"] is True
    assert model.surfaces[-1].interface is False


def test_layered_model_update_from_dataset_does_not_mutate_input_dataset():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"Vp": 1.0})
    model.add_surface(name="bottom", depth=1.0)
    dataset = xr.Dataset(
        {
            "vp": xr.DataArray(
                [[np.nan, 2.0], [3.0, 4.0]],
                dims=["x", "z"],
                coords={"x": [0.0, 1.0], "z": [0.0, 1.0]},
            )
        }
    )

    model.update_from_dataset(dataset)

    assert np.isnan(dataset["vp"].values[0, 0])
    assert not np.isnan(model.layers[0].properties["vp"].get().values).any()


def test_layered_model_validates_sampling_and_completion():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])

    with pytest.raises(ValueError, match="at least two surfaces"):
        model.to_fs()

    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"Vp": 1.0})
    model.add_surface(name="bottom", depth=1.0)

    with pytest.raises(ValueError, match="sample counts"):
        model.sample_uniform([3, 3, 3])
    with pytest.raises(ValueError, match=">= 2"):
        model.sample_uniform([1, 3])


def test_layered_model_rejects_deprecated_frame_kwargs():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])

    with pytest.raises(TypeError, match="frame"):
        model.add_surface(name="top", depth=0.0, frame="reference")

    model.add_surface(name="top", depth=0.0)
    with pytest.raises(TypeError, match="frame"):
        model.add_layer(name="layer", properties={"Vp": 1.0}, frame="reference")

    with pytest.raises(ValueError, match="system"):
        model.add_layer(
            name="layer",
            properties={"Vp": 1.0},
            system="surface_depth",
            coordinate_system="other",
        )
