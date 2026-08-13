import json

import h5py
import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")

from frequensolve.geometry.frame import Axis, CoordinateSystem, SurfaceCoordinateSystem
from frequensolve.mesh.mesh_generators import LayeredMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.layered import (
    Borehole,
    BoreholeAnnularPadding,
    BoreholeLayer,
    BoreholePart,
    BoreholePlug,
    BoreholeSurface,
    Fracture,
    Layer,
    LayeredModel,
    dipping_plane_2d,
    dipping_plane_3d,
)
from frequensolve.model.layered._utils import _dataarray_with_property_metadata
from frequensolve.model.model import ModelSubdomain
from frequensolve.model.property import Property, PropertyExpression, coord, prop
from frequensolve.simulation.simulation import SeismicSimulation
from frequensolve.units import ureg as u


def test_layer_perturb_defaults_to_the_property_map_grid(monkeypatch):
    grid = xr.DataArray(
        np.ones(2),
        dims=["x"],
        coords={"x": [0.0, 1.0]},
    )
    layer = Layer(
        name="layer",
        mesh_block_id=1,
        properties={"vp": grid},
        grid=grid,
    )
    call = {}

    def record_perturbation(**kwargs):
        call.update(kwargs)

    monkeypatch.setattr(
        layer.properties["vp"],
        "stochastic_perturbation",
        record_perturbation,
    )

    layer.perturb("vp", std=0.1)

    assert call["grid"] is grid


def test_inline_property_metadata_requires_array_data():
    with pytest.raises(
        ValueError,
        match="Inline property serialization requires array data",
    ):
        _dataarray_with_property_metadata(Property.expr("vp"))


def test_layer_surface_lookup_rejects_an_unbounded_layer():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(depth=0.0, name="top")
    model.add_layer(name="incomplete")

    with pytest.raises(ValueError, match="Layer 'incomplete' has no lower surface"):
        model.lower_surface(model.layers[0])


def test_dipping_plane_2d_samples_depth_along_dip():
    plane = dipping_plane_2d(
        point={"x": 0.0, "z": 10.0},
        dip=45.0,
        x=[0.0, 1.0],
        units="m",
        name="interface",
    )

    assert plane.name == "interface"
    assert plane.dims == ("x",)
    assert plane.attrs["units"] == "m"
    assert plane.coords["x"].attrs["units"] == "m"
    np.testing.assert_allclose(plane.values, [10.0, 11.0])


def test_dipping_plane_3d_samples_depth_from_strike_and_dip():
    plane = dipping_plane_3d(
        point=(0.0, 0.0, 10.0),
        dip=45.0,
        strike=0.0,
        x=[0.0, 1.0],
        y=[0.0, 2.0],
        units="m",
    )

    assert plane.dims == ("x", "y")
    assert plane.attrs["units"] == "m"
    assert plane.coords["x"].attrs["units"] == "m"
    assert plane.coords["y"].attrs["units"] == "m"
    np.testing.assert_allclose(plane.values, [[10.0, 10.0], [11.0, 11.0]])


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


def test_layered_model_save_uses_export_context_without_child_path_binding(tmp_path):
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="rock", mesh_block_id=1, properties={"Vp": 1.0})
    model.add_surface(name="bottom", depth=1.0)

    sim = SeismicSimulation(
        name="simple",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    sim.model = model
    sim.mesh = MeshManager(
        LayeredMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[2, 2])
    )

    sim_file = sim.save()
    payload = json.loads(sim_file.read_text())

    assert sim_file == tmp_path / "simulations" / "simple" / "simple.json"
    assert payload["Model"]["_type"] == "LayeredModel"
    assert payload["Mesh"]["generator"]["path"] == "simulations/simple"


def test_layered_model_domain_limits_accept_units_and_convert():
    model = LayeredModel(
        dimension=2,
        x_limits=u.Quantity([0.0, 1000.0], "m"),
    )
    model.add_surface(depth=0.0 * u.m, name="top")
    model.add_layer(name="layer", properties={"Vp": 1500.0 * u.m / u.s})
    model.add_surface(depth=500.0 * u.m, name="bottom")

    assert model.x_limits == [0.0, 1000.0]
    assert model.x_limits_in("km") == pytest.approx((0.0, 1.0))

    payload = model.to_fs()
    assert payload["x_limits"] == {"value": [0.0, 1000.0], "units": "m"}

    sampled = model.sample_uniform([3, 3], axes_units={"x": "km", "z": "km"})
    assert sampled.coords["x"].values.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert sampled.coords["z"].values.tolist() == pytest.approx([0.0, 0.25, 0.5])
    assert sampled.coords["x"].attrs["units"] == "km"
    assert sampled.coords["z"].attrs["units"] == "km"

    mesh = model.hex_mesh_generator(n=[2, 2]).to_fs()
    assert mesh["units"] == "m"
    assert mesh["l_bound"] == pytest.approx([0.0, 0.0])
    assert mesh["u_bound"] == pytest.approx([1000.0, 500.0])

    roundtrip = LayeredModel.from_fs(payload)
    assert roundtrip.x_limits_in("km") == pytest.approx((0.0, 1.0))
    assert roundtrip.to_fs()["x_limits"] == payload["x_limits"]
    roundtrip.x_limits = [0.0, 1.0]
    assert roundtrip.to_fs()["x_limits"] == [0.0, 1.0]


def test_layered_model_3d_domain_limits_convert_each_axis():
    model = LayeredModel(
        dimension=3,
        x_limits=u.Quantity([0.0, 1000.0], "m"),
        y_limits={"value": [0.0, 2.0], "units": "km"},
    )
    model.add_surface(depth=0.0 * u.m, name="top")
    model.add_layer(name="layer", properties={"Vp": 1500.0 * u.m / u.s})
    model.add_surface(depth=500.0 * u.m, name="bottom")

    assert model.x_limits_in("km") == pytest.approx((0.0, 1.0))
    assert model.y_limits_in("m") == pytest.approx((0.0, 2000.0))

    sampled = model.sample_uniform(
        [3, 3, 3],
        axes_units={"x": "km", "y": "km", "z": "km"},
    )
    assert sampled.coords["x"].values.tolist() == pytest.approx([0.0, 0.5, 1.0])
    assert sampled.coords["y"].values.tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert sampled.coords["z"].values.tolist() == pytest.approx([0.0, 0.25, 0.5])

    mesh = model.tet_mesh_generator(n=[2, 2, 2]).to_fs()
    assert mesh["units"] == "m"
    assert mesh["l_bound"] == pytest.approx([0.0, 0.0, 0.0])
    assert mesh["u_bound"] == pytest.approx([1000.0, 2000.0, 500.0])


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


def test_layered_model_exports_fracture_geometry():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="rock", mesh_block_id=1, properties={"Vp": 1.0})

    gap = xr.DataArray(
        [0.0, 0.01, 0.0],
        dims=["x"],
        coords={"x": [0.0, 0.5, 1.0]},
    )
    model.add_fracture(name="frac", depth=0.5, gap=gap, interface=False)
    model.add_surface(name="bottom", depth=1.0)

    payload = model.to_fs()

    assert payload["surfaces"][1] == {
        "_type": "Fracture",
        "name": "frac",
        "interface": False,
        "depth": {"value": 0.5},
        "gap": {
            "dims": ["x"],
            "coords": {"x": {"data": [0.0, 0.5, 1.0]}},
            "data": [0.0, 0.01, 0.0],
        },
    }
    assert (
        LayeredModel.from_fs(payload).to_fs()["surfaces"][1] == payload["surfaces"][1]
    )


def test_layered_model_fracture_creates_material_subdomain():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="upper", mesh_block_id=1, properties={"Vp": 1.0})

    gap = xr.DataArray(
        [0.0, 0.01, 0.0],
        dims=["x"],
        coords={"x": [0.0, 0.5, 1.0]},
    )
    model.add_fracture(
        name="frac",
        depth=0.5,
        gap=gap,
        physics="acoustic",
        properties={"Vp": 1.5, "Rho": 1.0},
    )
    model.add_layer(name="lower", mesh_block_id=3, properties={"Vp": 2.0})
    model.add_surface(name="bottom", depth=1.0)

    payload = model.to_fs()

    assert [surface["name"] for surface in payload["surfaces"]] == [
        "top",
        "frac",
        "bottom",
    ]
    assert payload["surfaces"][1]["_type"] == "Fracture"
    assert payload["surfaces"][1]["mesh_block_id"] == 2
    assert [subdomain["name"] for subdomain in payload["subdomains"]] == [
        "upper",
        "frac",
        "lower",
    ]
    assert payload["subdomains"][1] == {
        "mesh_block_id": 2,
        "name": "frac",
        "physics": "acoustic",
        "properties": {"vp": {"value": 1.5}, "rho": {"value": 1.0}},
    }
    roundtrip = LayeredModel.from_fs(payload).to_fs()
    assert roundtrip["surfaces"] == payload["surfaces"]
    assert roundtrip["subdomains"][1] == payload["subdomains"][1]


def test_layered_model_noninterface_fracture_does_not_split_layers():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="rock", mesh_block_id=1, properties={"Vp": 1.0})
    model.add_fracture(
        name="frac",
        depth=0.5,
        gap=xr.DataArray([0.0, 0.01, 0.0], dims=["x"], coords={"x": [0.0, 0.5, 1.0]}),
        interface=False,
    )
    model.add_surface(name="bottom", depth=1.0)

    payload = model.to_fs()

    assert len(payload["subdomains"]) == 1
    assert payload["surfaces"][1]["interface"] is False
    assert payload["surfaces"][1]["_type"] == "Fracture"


def test_layered_model_fractures_support_units_and_iadd():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1000.0] * u.m)
    model.add_surface(name="top", depth=0.0 * u.m)
    model.add_layer(name="rock", mesh_block_id=1, properties={"Vp": 1500.0})
    gap = xr.DataArray(
        [0.0, 1.0, 0.0],
        dims=["x"],
        coords={"x": [0.0, 500.0, 1000.0]},
        attrs={"units": "cm"},
    )
    gap.coords["x"].attrs["units"] = "m"
    model += Fracture("frac", depth=500.0 * u.m, gap=gap, interface=False)
    model.add_surface(name="bottom", depth=1000.0 * u.m)

    payload = model.to_fs()["surfaces"][1]

    assert payload["depth"] == {"value": 500.0, "units": "m"}
    assert payload["gap"]["units"] == "cm"
    assert payload["gap"]["coords"]["x"]["units"] == "m"


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
            "layers": [
                {
                    "name": "fluid",
                    "mesh_block_id": 20,
                    "outer_surface": "bh1_surface_1",
                },
                {
                    "name": "casing",
                    "mesh_block_id": 21,
                    "inner_surface": "bh1_surface_1",
                    "outer_surface": "bh1_surface_2",
                },
                {
                    "name": "cement",
                    "mesh_block_id": 22,
                    "inner_surface": "bh1_surface_2",
                    "outer_surface": "bh1_surface_3",
                },
            ],
            "surfaces": [
                {"name": "bh1_surface_1", "r": {"value": 0.035}},
                {"name": "bh1_surface_2", "r": {"value": 0.041}},
                {"name": "bh1_surface_3", "r": {"value": 0.065}},
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


def test_borehole_from_fs_accepts_contract_x_shorthand_and_optional_part_name():
    borehole = Borehole.from_fs(
        {
            "name": "bh1",
            "x": {"value": 0.45, "units": "km"},
            "extent": {"top": "top", "bottom": 2},
            "parts": [
                {
                    "mesh_block_id": 20,
                    "r": {"value": 0.035},
                    "inner_surface": "bh1_axis",
                    "outer_surface": "bh1_fluid_wall",
                }
            ],
        }
    )

    assert borehole.to_fs() == {
        "name": "bh1",
        "axis": {"x": {"value": 0.45, "units": "km"}},
        "extent": {"top": {"surface": "top"}, "bottom": 2},
        "layers": [
            {
                "name": "part_20",
                "mesh_block_id": 20,
                "inner_surface": "bh1_axis",
                "outer_surface": "bh1_fluid_wall",
            }
        ],
        "surfaces": [
            {"name": "bh1_fluid_wall", "r": {"value": 0.035}},
        ],
    }


def test_borehole_builder_adds_plug_obstruction_subdomain():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=1.0)

    borehole = model.add_borehole(name="bh1", x=0.0, top="top", bottom="bottom")
    borehole.add_layer(
        "fluid",
        width=0.1,
        physics="acoustic",
        properties={"Vp": 1.48, "Rho": 1.03},
    )
    borehole.add_plug(
        "strainmeter",
        top=0.35 * u.m,
        bottom=0.65 * u.m,
        radius=0.035 * u.m,
        physics="elastic:iso",
        properties={"Vp": 3.0, "Vs": 1.4, "Rho": 2.0},
    )

    payload = model.to_fs()

    assert payload["subdomains"][-1] == {
        "mesh_block_id": 3,
        "name": "bh1_strainmeter",
        "physics": "elastic:iso",
        "properties": {
            "vp": {"value": 3.0},
            "vs": {"value": 1.4},
            "rho": {"value": 2.0},
        },
    }
    assert payload["boreholes"][0]["plugs"] == [
        {
            "name": "strainmeter",
            "mesh_block_id": 3,
            "top": {"value": 0.35, "units": "m"},
            "bottom": {"value": 0.65, "units": "m"},
            "r": {"value": 0.035, "units": "m"},
        }
    ]
    assert LayeredModel.from_fs(payload).to_fs()["boreholes"][0]["plugs"] == (
        payload["boreholes"][0]["plugs"]
    )


def test_layered_model_add_borehole_accepts_plug_specs():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=1.0)

    model.add_borehole(
        name="bh1",
        x=0.0,
        layers=[
            {
                "name": "fluid",
                "r": 0.1,
                "physics": "acoustic",
                "properties": {"Vp": 1.48, "Rho": 1.03},
            }
        ],
        plugs=[
            {
                "name": "tool",
                "top": 0.25,
                "bottom": 0.5,
                "radius": 0.04,
                "physics": "elastic:iso",
                "properties": {"Vp": 3.0, "Vs": 1.4, "Rho": 2.0},
            }
        ],
    )

    payload = model.to_fs()

    assert payload["boreholes"][0]["plugs"] == [
        {
            "name": "tool",
            "mesh_block_id": 3,
            "top": 0.25,
            "bottom": 0.5,
            "r": {"value": 0.04},
        }
    ]


def test_borehole_plug_rejects_missing_radius():
    with pytest.raises(ValueError, match="radius"):
        BoreholePlug(name="tool", mesh_block_id=1, top=0.1, bottom=0.2)


def test_borehole_add_plug_rejects_radius_alias_conflict():
    borehole = Borehole(name="bh1", axis={"x": 0.0}, extent={"top": 1, "bottom": 2})
    with pytest.raises(ValueError, match="r or radius"):
        borehole.add_plug(
            "tool",
            top=0.1,
            bottom=0.2,
            r=0.03,
            radius=0.04,
            mesh_block_id=1,
        )


def test_borehole_builder_adds_layers_between_implicit_axis_and_surfaces():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    borehole = model.add_borehole(name="bh1", x=0.45)
    borehole.add_layer(
        "fluid",
        physics="acoustic",
        properties={"Vp": 1.48, "Rho": 1.03},
    )
    borehole.add_surface("fluid_wall", r=0.035)
    borehole.add_layer(
        "casing",
        physics="elastic:iso",
        properties={"Vp": 5.9, "Vs": 3.2, "Rho": 7.85},
    )
    borehole.add_surface("casing_wall", r=0.041)

    payload = model.to_fs()

    assert borehole.surface_names == ["fluid_wall", "casing_wall"]
    assert borehole.layer_names == ["fluid", "casing"]
    assert [subdomain["name"] for subdomain in payload["subdomains"]] == [
        "formation",
        "bh1_fluid",
        "bh1_casing",
    ]
    assert payload["boreholes"][0]["layers"] == [
        {
            "name": "fluid",
            "mesh_block_id": 2,
            "outer_surface": "fluid_wall",
        },
        {
            "name": "casing",
            "mesh_block_id": 3,
            "inner_surface": "fluid_wall",
            "outer_surface": "casing_wall",
        },
    ]
    assert payload["boreholes"][0]["surfaces"] == [
        {"name": "fluid_wall", "r": {"value": 0.035}},
        {"name": "casing_wall", "r": {"value": 0.041}},
    ]


def test_borehole_builder_allows_consecutive_geometry_surfaces():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    borehole = model.add_borehole(name="bh1", x=0.45)
    borehole.add_layer(
        "fluid",
        physics="acoustic",
        properties={"Vp": 1.48, "Rho": 1.03},
    )
    borehole.add_surface("fluid_refine", r=0.025)
    borehole.add_surface("fluid_wall", r=0.035)
    borehole.add_layer(
        "casing",
        physics="elastic:iso",
        properties={"Vp": 5.9, "Vs": 3.2, "Rho": 7.85},
    )
    borehole.add_surface("casing_refine", r=0.038)
    borehole.add_surface("casing_wall", r=0.041)

    payload = model.to_fs()["boreholes"][0]

    assert borehole.surface_names == [
        "fluid_refine",
        "fluid_wall",
        "casing_refine",
        "casing_wall",
    ]
    assert payload["layers"] == [
        {
            "name": "fluid",
            "mesh_block_id": 2,
            "outer_surface": "fluid_wall",
        },
        {
            "name": "casing",
            "mesh_block_id": 3,
            "inner_surface": "fluid_wall",
            "outer_surface": "casing_wall",
        },
    ]
    assert payload["surfaces"] == [
        {"name": "fluid_refine", "r": {"value": 0.025}},
        {"name": "fluid_wall", "r": {"value": 0.035}},
        {"name": "casing_refine", "r": {"value": 0.038}},
        {"name": "casing_wall", "r": {"value": 0.041}},
    ]
    assert LayeredModel.from_fs(model.to_fs()).to_fs()["boreholes"] == [payload]


def test_borehole_canonical_layers_and_surfaces_schema_roundtrips():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)
    model += ModelSubdomain(
        name="fluid",
        mesh_block_id=20,
        physics="acoustic",
        properties={"Vp": 1.48},
    )

    model.add_borehole(
        name="bh1",
        x=0.45,
        layers=[BoreholeLayer("fluid", mesh_block_id=20)],
        surfaces=[BoreholeSurface("fluid_wall", r=0.035)],
    )

    payload = model.to_fs()["boreholes"][0]

    assert payload["layers"] == [
        {
            "name": "fluid",
            "mesh_block_id": 20,
            "outer_surface": "fluid_wall",
        }
    ]
    assert payload["surfaces"] == [{"name": "fluid_wall", "r": {"value": 0.035}}]
    assert Borehole.from_fs(payload).to_fs() == payload


def test_borehole_builder_width_auto_adds_scalar_surfaces():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    borehole = model.add_borehole(name="bh1", x=0.45)
    fluid_wall = borehole.add_layer(
        "fluid",
        width=35.0 * u.mm,
        physics="acoustic",
        properties={"Vp": 1.48, "Rho": 1.03},
    )
    casing_wall = borehole.add_layer(
        "casing",
        width=6.0 * u.mm,
        physics="elastic:iso",
        properties={"Vp": 5.9, "Vs": 3.2, "Rho": 7.85},
    )

    payload = model.to_fs()

    assert fluid_wall.name is None
    assert casing_wall.name is None
    assert borehole.surface_names == ["bh1_surface_1", "bh1_surface_2"]
    assert payload["boreholes"][0]["layers"] == [
        {
            "name": "fluid",
            "mesh_block_id": 2,
            "outer_surface": "bh1_surface_1",
        },
        {
            "name": "casing",
            "mesh_block_id": 3,
            "inner_surface": "bh1_surface_1",
            "outer_surface": "bh1_surface_2",
        },
    ]
    assert payload["boreholes"][0]["surfaces"] == [
        {"name": "bh1_surface_1", "r": {"value": 35.0, "units": "mm"}},
        {"name": "bh1_surface_2", "r": {"value": 41.0, "units": "mm"}},
    ]


def test_borehole_builder_width_rejects_variable_previous_surface():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)
    radius = xr.DataArray(
        [0.03, 0.035],
        dims=["depth"],
        coords={"depth": [0.0, 0.5]},
    )

    borehole = model.add_borehole(name="bh1", x=0.45)
    borehole.add_layer("fluid", mesh_block_id=1)
    borehole.add_surface("fluid_wall", r=radius)

    with pytest.raises(ValueError, match="variable-radius"):
        borehole.add_layer("casing", mesh_block_id=1, width=0.006)
    assert borehole.pending_layer is None


def test_borehole_builder_width_requires_scalar_number():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)
    borehole = model.add_borehole(name="bh1", x=0.45)

    with pytest.raises(TypeError, match="scalar number"):
        borehole.add_layer("fluid", mesh_block_id=1, width=[0.03, 0.04])
    assert borehole.pending_layer is None


def test_borehole_builder_supports_iadd_layer_and_surface_objects():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    borehole = model.add_borehole(name="bh1", x=0.45)
    borehole += BoreholeLayer(
        "fluid",
        physics="acoustic",
        properties={"Vp": 1.48, "Rho": 1.03},
    )
    borehole += BoreholeSurface("fluid_wall", r=0.035)
    borehole += BoreholeLayer(
        "casing",
        width=0.015,
        physics="elastic:iso",
        properties={"Vp": 5.9, "Vs": 3.2, "Rho": 7.85},
    )

    payload = model.to_fs()

    assert borehole.surface_names == ["fluid_wall", "bh1_surface_2"]
    assert payload["boreholes"][0]["layers"] == [
        {
            "name": "fluid",
            "mesh_block_id": 2,
            "outer_surface": "fluid_wall",
        },
        {
            "name": "casing",
            "mesh_block_id": 3,
            "inner_surface": "fluid_wall",
            "outer_surface": "bh1_surface_2",
        },
    ]
    assert payload["boreholes"][0]["surfaces"] == [
        {"name": "fluid_wall", "r": {"value": 0.035}},
        {"name": "bh1_surface_2", "r": {"value": 0.05}},
    ]


def test_borehole_iadd_rejects_untyped_objects():
    borehole = Borehole(
        name="bh1",
        axis={"x": 0.45},
        extent={"top": "top", "bottom": "bottom"},
    )

    with pytest.raises(ValueError, match="Cannot add"):
        borehole += {"name": "fluid"}


def test_borehole_builder_allows_optional_axis_alias():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    borehole = model.add_borehole(name="bh1", x=0.45)
    borehole.add_surface("axis", r=0.0)
    borehole.add_layer("fluid", mesh_block_id=1)
    borehole.add_surface("fluid_wall", r=0.035)

    payload = borehole.to_fs()

    assert payload["layers"][0]["inner_surface"] == "axis"
    assert payload["layers"][0]["outer_surface"] == "fluid_wall"
    assert payload["surfaces"] == [{"name": "fluid_wall", "r": {"value": 0.035}}]


def test_borehole_builder_requires_closing_pending_layer():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    borehole = model.add_borehole(name="bh1", x=0.45)
    borehole.add_layer("fluid", physics="acoustic", properties={"Vp": 1.48})

    with pytest.raises(ValueError, match="unclosed layer"):
        model.to_fs()


def test_layered_model_requires_y_for_3d_boreholes():
    model = LayeredModel(dimension=3, x_limits=[0.0, 1.0], y_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    with pytest.raises(ValueError, match="axis/y"):
        model.add_borehole(
            name="bh1",
            x=0.45,
            parts=[{"mesh_block_id": 20, "r": 0.035}],
        )


def test_layered_model_exports_3d_borehole_meshing_contract():
    model = LayeredModel(
        dimension=3,
        x_limits=[0.0, 1.0],
        y_limits=[0.0, 1.0],
    )
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    model.add_borehole(
        name="bh1",
        x=0.45,
        y=0.55,
        surfaces=[
            BoreholeSurface("fluid_refine", r=0.025),
            BoreholeSurface("fluid_wall", r=0.035),
        ],
        layers=[
            {
                "name": "fluid",
                "outer_surface": "fluid_wall",
                "physics": "acoustic",
                "properties": {"Vp": 1.48},
            }
        ],
        annular_padding={
            "n": 3,
            "outer_radius": 0.2 * u.m,
            "power": 1.5,
        },
    )

    payload = model.to_fs()

    assert payload["boreholes"] == [
        {
            "name": "bh1",
            "axis": {"x": 0.45, "y": 0.55},
            "extent": {
                "top": {"surface": "top"},
                "bottom": {"surface": "bottom"},
            },
            "layers": [
                {
                    "name": "fluid",
                    "mesh_block_id": 2,
                    "outer_surface": "fluid_wall",
                }
            ],
            "surfaces": [
                {
                    "name": "fluid_refine",
                    "r": {"value": 0.025},
                },
                {
                    "name": "fluid_wall",
                    "r": {"value": 0.035},
                },
            ],
            "annular_padding": {
                "n": 3,
                "outer_radius": {"value": 0.2, "units": "m"},
                "power": 1.5,
            },
        }
    ]
    assert LayeredModel.from_fs(payload).to_fs()["boreholes"] == payload["boreholes"]


def test_borehole_annular_padding_validates_and_serializes_units():
    padding = BoreholeAnnularPadding(
        n=2,
        outer_radius={"value": 20.0, "units": u.cm},
    )

    assert padding.to_fs() == {
        "n": 2,
        "outer_radius": {"value": 20.0, "units": "cm"},
        "power": 1.0,
    }


def test_layered_model_rejects_invalid_annular_padding():
    model = LayeredModel(
        dimension=3,
        x_limits=[0.0, 1.0],
        y_limits=[0.0, 1.0],
    )
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    with pytest.raises(
        ValueError,
        match="annular_padding/outer_radius must be positive",
    ):
        model.add_borehole(
            name="bh1",
            x=0.45,
            y=0.55,
            parts=[
                {
                    "mesh_block_id": 1,
                    "r": 0.035,
                }
            ],
            annular_padding={"n": 2},
        )


def test_layered_model_rejects_2d_annular_padding():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    with pytest.raises(ValueError, match="3D boreholes only"):
        model.add_borehole(
            name="bh1",
            x=0.45,
            parts=[
                {
                    "mesh_block_id": 1,
                    "r": 0.035,
                }
            ],
            annular_padding={"n": 1, "outer_radius": 0.08},
        )


def test_layered_model_rejects_plugs_for_3d_boreholes():
    model = LayeredModel(dimension=3, x_limits=[0.0, 1.0], y_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    with pytest.raises(ValueError, match="does not support plugs"):
        model.add_borehole(
            name="bh1",
            x=0.45,
            y=0.55,
            parts=[
                {
                    "mesh_block_id": 1,
                    "r": 0.035,
                }
            ],
            plugs=[
                {
                    "top": 0.1,
                    "bottom": 0.2,
                    "mesh_block_id": 1,
                    "r": 0.01,
                }
            ],
        )


def test_layered_model_requires_borehole_part_domains():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)

    with pytest.raises(ValueError, match="missing: 20"):
        model.add_borehole(
            name="bh1",
            x=0.45,
            parts=[{"mesh_block_id": 20, "r": 0.035}],
        )


def test_layered_model_auto_assigns_borehole_part_domain_with_properties():
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
                "r": 0.035,
                "physics": "acoustic",
                "properties": {"Vp": 1.48},
            }
        ],
    )

    payload = model.to_fs()

    assert payload["subdomains"][1]["mesh_block_id"] == 2
    assert payload["subdomains"][1]["name"] == "bh1_fluid"
    assert payload["boreholes"][0]["layers"][0]["mesh_block_id"] == 2


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
        "data": [0.035, 0.04],
        "dims": ["z"],
        "coords": {"z": {"data": [0.0, 0.5], "units": "km"}},
        "units": "m",
    }


def test_borehole_radius_profile_requires_depth_dimension():
    radius = xr.DataArray(
        [0.035, 0.040],
        dims=["x"],
        coords={"x": [0.0, 0.5]},
        attrs={"units": "m"},
    )

    with pytest.raises(ValueError, match="'z' or 'depth'"):
        BoreholePart(name="fluid", mesh_block_id=20, r=radius)


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
    r_payload = payload["Model"]["boreholes"][0]["surfaces"][0]["r"]

    assert r_payload["dataset"] == "inputs/model/boreholes/bh1/surfaces/bh1_surface_1/r"
    assert r_payload["hash"].startswith("blake3:")
    with h5py.File(tmp_path / "simulations/simple/simple.h5", "r") as h5:
        dset = h5["inputs/model/boreholes/bh1/surfaces/bh1_surface_1/r"]
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


def test_layered_mesh_generator_accepts_unit_bearing_bounds():
    mesh = LayeredMeshGenerator(
        l_bound=[0.0 * u.m, 0.0 * u.m],
        u_bound=[100.0 * u.cm, 2.0 * u.m],
        n=[4, 8],
    )

    payload = mesh.to_fs()

    assert payload["l_bound"] == [0.0, 0.0]
    assert payload["u_bound"] == [1.0, 2.0]
    assert payload["units"] == "m"


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


def test_layered_model_samples_surface_depth_coordinate_system(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="elastic",
        dimension=2,
        project_path=tmp_path,
    )
    sim += CoordinateSystem.cartesian(
        name="section",
        axes=[Axis("offset", direction="x", origin=5.0 * u.km)],
        inherit_axes=True,
    )

    interface = xr.DataArray(
        [0.25, 0.75],
        dims=["offset"],
        coords={"offset": [0.0, 1.0]},
        attrs={"units": "km"},
    )
    model = LayeredModel(
        dimension=2,
        x_limits=u.Quantity([5.0, 6.0], "km"),
    )
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(name="upper", properties={"Vp": 1.0})
    model.add_surface(
        name="interface",
        depth={"value": interface, "coordinate_system": "section"},
    )
    model.add_layer(name="lower", properties={"Vp": 2.0})
    model.add_surface(name="bottom", depth=1.0 * u.km)
    sim += model

    sampled = model.sample_uniform([3, 5], axes_units={"x": "km", "z": "km"})

    np.testing.assert_allclose(sampled.coords["x"].values, [5.0, 5.5, 6.0])
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


def test_layered_model_samples_property_in_cartesian_coordinate_system(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="elastic",
        dimension=2,
        project_path=tmp_path,
    )
    section = CoordinateSystem.cartesian(
        name="section",
        axes=[
            Axis("offset", direction="x", origin=5.0 * u.km),
            Axis("depth", direction="z"),
        ],
    )
    sim += section

    vp = xr.DataArray(
        [[1.0, 1.5, 2.0], [2.0, 2.5, 3.0], [3.0, 3.5, 4.0]],
        dims=["offset", "depth"],
        coords={"offset": [0.0, 0.5, 1.0], "depth": [0.0, 0.25, 0.5]},
    )

    model = LayeredModel(
        dimension=2,
        x_limits=u.Quantity([5.0, 6.0], "km"),
    )
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="layer",
        properties={"Vp": {"value": vp, "coordinate_system": "section"}},
    )
    model.add_surface(name="bottom", depth=0.5 * u.km)
    sim += model

    sampled = model.sample_uniform([3, 3], axes_units={"x": "km", "z": "km"})

    assert sampled.sizes["x"] == 3
    assert sampled.sizes["z"] == 3
    np.testing.assert_allclose(sampled["vp"].values, vp.values)


def test_layered_model_sample_uniform_can_expose_coordinate_system_axes(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="elastic",
        dimension=2,
        project_path=tmp_path,
    )
    sim += CoordinateSystem.cartesian(
        name="section",
        axes=[
            Axis("depth", direction="z"),
            Axis("offset", direction="x", origin=5.0 * u.km),
        ],
    )

    model = LayeredModel(
        dimension=2,
        x_limits=u.Quantity([5.0, 6.0], "km"),
    )
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(name="layer", properties={"Vp": 2.0})
    model.add_surface(name="bottom", depth=0.5 * u.km)
    sim += model

    sampled = model.sample_uniform(
        [3, 3],
        axes_units={"offset": "km", "depth": "km"},
        frame="section",
    )

    assert sampled["vp"].dims == ("depth", "offset")
    np.testing.assert_allclose(sampled.coords["offset"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(sampled.coords["depth"], [0.0, 0.25, 0.5])
    np.testing.assert_allclose(sampled.coords["x"], [5.0, 5.5, 6.0])
    np.testing.assert_allclose(sampled.coords["z"], [0.0, 0.25, 0.5])
    assert sampled.coords["offset"].attrs["units"] == "km"
    assert sampled.coords["x"].attrs["units"] == "km"
    assert sampled.attrs["frame"] == "section"


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


def test_property_grid_mismatch_error_mentions_coordinate_systems():
    vp = xr.DataArray(
        [1.0, 2.0],
        dims=["height"],
        coords={"height": [0.0, 0.5]},
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="layer", properties={"Vp": vp})
    model.add_surface(name="bottom", depth=0.5)

    with pytest.raises(ValueError) as excinfo:
        model.sample_uniform([2, 3])

    message = str(excinfo.value)
    assert "Property dimensions ['height'] are not available" in message
    assert "coordinate-system axes" in message
    assert "Note that in 2D" not in message


def test_property_unknown_coordinate_system_error_is_actionable():
    vp = xr.DataArray(
        [1.0, 2.0],
        dims=["height"],
        coords={"height": [0.0, 0.5]},
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(
        name="layer",
        properties={"Vp": {"value": vp, "coordinate_system": "missing_system"}},
    )
    model.add_surface(name="bottom", depth=0.5)

    with pytest.raises(ValueError, match="missing_system") as excinfo:
        model.sample_uniform([2, 3])

    assert "Add the CoordinateSystem to the simulation" in str(excinfo.value)


def test_property_coordinate_system_missing_axis_error_is_actionable(tmp_path):
    sim = SeismicSimulation(
        name="simple",
        physics="elastic",
        dimension=2,
        project_path=tmp_path,
    )
    sim += SurfaceCoordinateSystem(
        name="interface_relative",
        surface="top",
        axes=[Axis("up", direction="z", positive="up")],
    )
    vp = xr.DataArray(
        [1.0, 2.0],
        dims=["height"],
        coords={"height": [0.0, 0.5]},
    )

    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(
        name="layer",
        properties={"Vp": {"value": vp, "coordinate_system": "interface_relative"}},
    )
    model.add_surface(name="bottom", depth=0.5)
    sim += model

    with pytest.raises(ValueError, match="Axis named 'height'") as excinfo:
        model.sample_uniform([2, 3])

    assert "coordinate system 'interface_relative'" in str(excinfo.value)


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


def test_layered_model_samples_nested_expression_properties_per_layer():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0] * u.km)
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="upper",
        properties={
            "Vp": 2.0 * u.km / u.s,
            "Vs": 0.5 * prop("Vp"),
            "Vratio": prop("Vs") / prop("Vp"),
        },
    )
    model.add_surface(name="interface", depth=0.5 * u.km)
    model.add_layer(
        name="lower",
        properties={
            "Vp": 3000.0 * u.m / u.s,
            "Vs": 0.5 * prop("Vp"),
            "Vratio": prop("Vs") / prop("Vp"),
        },
    )
    model.add_surface(name="bottom", depth=1.0 * u.km)

    sampled_vs = model.sample_uniform(
        [2, 5],
        axes_units={"x": "km", "z": "km"},
        properties="Vs",
    )
    sampled_ratio = model.sample_uniform(
        [2, 5],
        axes_units={"x": "km", "z": "km"},
        properties="Vratio",
    )
    _, log = model.get_1D_log("Vs", x=0.5, dz=0.25, units="m/s")

    assert sampled_vs["vs"].attrs["units"] == "km/s"
    np.testing.assert_allclose(sampled_vs["vs"].isel(z=0), 1.0)
    np.testing.assert_allclose(sampled_vs["vs"].isel(z=-1), 1.5)
    np.testing.assert_allclose(sampled_ratio["vratio"], 0.5)
    np.testing.assert_allclose(log[[0, -1]], [1000.0, 1500.0])


def test_layered_model_samples_expression_coordinate_symbols(tmp_path):
    simulation = SeismicSimulation(
        name="expression-coordinates",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    depth_system = SurfaceCoordinateSystem(
        name="depth_from_top",
        surface="top",
        axes=[Axis("depth", direction="z", positive="down")],
    )
    simulation += depth_system

    qp = Property.expr(
        {
            "op": "case",
            "branches": [
                {
                    "if": {
                        "op": "<",
                        "args": [{"var": "depth"}, {"value": 0.5}],
                    },
                    "then": {"value": 60.0},
                }
            ],
            "else": {"value": 100.0},
        },
        symbols={"depth": coord(depth_system.name, "depth", units="km")},
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0] * u.km)
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(name="formation", properties={"Qp": qp})
    model.add_surface(name="bottom", depth=1.0 * u.km)
    simulation += model

    sampled = model.sample_uniform(
        [2, 5],
        axes_units={"x": "km", "z": "km"},
        properties="Qp",
    )

    np.testing.assert_allclose(sampled["qp"].isel(x=0), [60, 60, 100, 100, 100])


def test_layered_model_default_z_sampling_preserves_surface_units_for_expressions():
    depth = Property.expr(
        {"var": "z"},
        symbols={"z": coord("global", "z", units="m")},
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0] * u.km)
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(name="formation", properties={"Qp": depth})
    model.add_surface(name="bottom", depth=1.0 * u.km)

    sampled = model.sample_uniform([2, 3], properties="Qp")
    depths, log = model.get_1D_log("Qp", x=0.5, dz=0.5)

    assert sampled.coords["z"].attrs["units"] == "km"
    np.testing.assert_allclose(sampled["qp"].isel(x=0), [0.0, 500.0, 1000.0])
    np.testing.assert_allclose(depths, [0.0, 0.5])
    np.testing.assert_allclose(log, [0.0, 500.0])


def test_layered_model_plots_expression_bound_to_implicit_global_x():
    import matplotlib.pyplot as plt

    vp = Property.expr(
        {
            "op": "case",
            "branches": [
                {
                    "if": {
                        "op": "<",
                        "args": [{"var": "x"}, {"value": 500.0}],
                    },
                    "then": {"value": 1.5, "units": "km/s"},
                }
            ],
            "else": {"value": 2.0, "units": "km/s"},
        },
        symbols={"x": coord("global", "x", units="m")},
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0] * u.km)
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(name="sediment", properties={"Vp": vp})
    model.add_surface(name="bottom", depth=1.0 * u.km)

    fig, ax = plt.subplots()
    image = model.plot(
        "Vp",
        resolution=[3, 2],
        ax=ax,
        surfaces=False,
        add_colorbar=False,
    )

    np.testing.assert_allclose(image.get_array(), [[1.5, 2.0, 2.0]] * 2)
    assert image.get_clim() == pytest.approx((1.5, 2.0))
    plt.close(fig)


def test_layered_model_samples_all_implicit_global_expression_axes_in_3d():
    coordinate_sum = Property.expr(
        {
            "op": "add",
            "args": [
                {
                    "op": "add",
                    "args": [
                        {"var": "x"},
                        {"var": "y"},
                    ],
                },
                {"var": "z"},
            ],
        },
        symbols={axis: coord("global", axis, units="km") for axis in ("x", "y", "z")},
    )
    model = LayeredModel(
        dimension=3,
        x_limits=[0.0, 1.0] * u.km,
        y_limits=[0.0, 2.0] * u.km,
    )
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(name="formation", properties={"Qp": coordinate_sum})
    model.add_surface(name="bottom", depth=3.0 * u.km)

    sampled = model.sample_uniform(
        [2, 2, 2],
        axes_units={"x": "km", "y": "km", "z": "km"},
        properties="Qp",
    )

    assert sampled["qp"].isel(x=0, y=0, z=0).item() == pytest.approx(0.0)
    assert sampled["qp"].isel(x=-1, y=-1, z=-1).item() == pytest.approx(6.0)


def test_layered_model_plots_property_derived_from_independent_field_data():
    import matplotlib.pyplot as plt

    vp_base = xr.DataArray(
        [1.5, 2.0],
        dims=["z"],
        coords={"z": [0.0, 1.0]},
        attrs={"units": "km/s"},
    )
    vp_base.coords["z"].attrs["units"] = "km"
    scale = PropertyExpression(
        {
            "op": "case",
            "branches": [
                {
                    "if": {
                        "op": "<",
                        "args": [{"var": "x"}, {"value": 0.5}],
                    },
                    "then": {"value": 1.0},
                }
            ],
            "else": {"value": 1.1},
        }
    )
    vp = Property.expr(
        PropertyExpression.field("vp_base") * scale,
        units="km/s",
        symbols={"x": coord("global", "x", units="km")},
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0] * u.km)
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="sediment",
        fields={"vp_base": vp_base},
        properties={"Vp": vp},
    )
    model.add_surface(name="bottom", depth=1.0 * u.km)

    sampled = model.sample_uniform(
        [2, 3],
        axes_units={"x": "km", "z": "km"},
        properties="Vp",
    )
    fig, ax = plt.subplots()
    image = model.plot(
        "Vp",
        resolution=[2, 3],
        ax=ax,
        axes_units={"x": "km", "z": "km"},
        units="m/s",
        surfaces=False,
        add_colorbar=False,
    )

    np.testing.assert_allclose(sampled["vp"].isel(x=0), [1.5, 1.75, 2.0])
    np.testing.assert_allclose(sampled["vp"].isel(x=-1), [1.65, 1.925, 2.2])
    assert image.get_clim() == pytest.approx((1500.0, 2200.0))
    plt.close(fig)


def test_layered_model_samples_independent_field_after_bare_roundtrip():
    vp_base = xr.DataArray(
        [1.5, 2.0],
        dims=["z"],
        coords={"z": [0.0, 1.0]},
        attrs={"units": "km/s"},
    )
    vp_base.coords["z"].attrs["units"] = "km"
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0] * u.km)
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="sediment",
        fields={"vp_base": vp_base},
        properties={
            "Vp": Property.expr(
                PropertyExpression.field("vp_base") * 1.1,
                units="km/s",
            )
        },
    )
    model.add_surface(name="bottom", depth=1.0 * u.km)

    loaded = LayeredModel.from_fs(model.to_fs())
    sampled = loaded.sample_uniform([2, 3], properties="Vp")

    xr.testing.assert_identical(loaded.layers[0].fields["vp_base"].get(), vp_base)
    np.testing.assert_allclose(sampled["vp"], [[1.65, 1.925, 2.2]] * 2)


def test_layered_model_evaluates_sympy_coordinate_magnitudes_in_binding_units():
    import sympy as sp

    x, z = sp.symbols("x z", real=True)
    lateral = sp.Piecewise((1.0, x < 5.0), (1.2, True))
    depth = sp.exp(-z / 0.3)
    coordinate_symbols = {
        "x": coord("global", "x", units="km"),
        "z": coord("global", "z", units="km"),
    }
    scale = PropertyExpression.from_value(
        1.0 + (lateral - 1.0) * depth,
        symbols=coordinate_symbols,
        default_symbol="var",
    )
    vp = Property.expr(
        PropertyExpression.field("vp_base") * scale,
        units="km/s",
        symbols=coordinate_symbols,
    )
    model = LayeredModel(dimension=2, x_limits=[0.0, 10.0] * u.km)
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="sediment",
        fields={"vp_base": 2.0 * u.km / u.s},
        properties={"Vp": vp},
    )
    model.add_surface(name="bottom", depth=1.0 * u.km)

    sampled = model.sample_uniform(
        [3, 3],
        axes_units={"x": "m", "z": "m"},
        properties="Vp",
    )

    np.testing.assert_allclose(sampled["vp"].isel(x=0), 2.0)
    assert sampled["vp"].isel(x=-1, z=0).item() == pytest.approx(2.4)
    assert sampled["vp"].isel(x=-1, z=-1).item() == pytest.approx(
        2.0 * (1.0 + 0.2 * np.exp(-1.0 / 0.3))
    )


def test_layered_model_expression_reports_missing_independent_field():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(
        name="sediment",
        properties={"Vp": Property.expr({"field": "vp_base"})},
    )
    model.add_surface(name="bottom", depth=1.0)

    with pytest.raises(ValueError, match="field 'vp_base'.*not declared"):
        model.sample_uniform([2, 2], properties="Vp")


def test_layered_model_materializes_legacy_extra_field_payloads():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(
        name="sediment",
        properties={
            "Vp": Property.expr(
                PropertyExpression.field("vp_base") * 1.1,
                units="km/s",
            )
        },
    )
    model.layers["sediment"].extra["fields"] = {
        "vp_base": {"value": 1.5, "units": "km/s"}
    }
    model.add_surface(name="bottom", depth=1.0)

    sampled = model.sample_uniform([2, 2], properties="Vp")

    np.testing.assert_allclose(sampled["vp"], 1.65)
    assert sampled["vp"].attrs["units"] == "km/s"


def test_layered_model_global_expression_binding_reports_missing_physical_axis():
    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(
        name="formation",
        properties={
            "Qp": Property.expr(
                {"var": "y"},
                symbols={"y": coord("global", "y")},
            )
        },
    )
    model.add_surface(name="bottom", depth=1.0)

    with pytest.raises(ValueError, match="unavailable axis 'y'.*'global'"):
        model.sample_uniform([2, 2], properties="Qp")


def test_layered_model_expression_dependencies_report_cycles_and_missing_refs():
    cyclic = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    cyclic.add_surface(name="top", depth=0.0)
    cyclic.add_layer(
        name="cycle",
        properties={"Vp": prop("Vs"), "Vs": prop("Vp")},
    )
    cyclic.add_surface(name="bottom", depth=1.0)

    with pytest.raises(
        ValueError,
        match="Circular property expression dependency: vp -> vs -> vp",
    ):
        cyclic.sample_uniform([2, 3], properties="Vp")

    missing = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    missing.add_surface(name="top", depth=0.0)
    missing.add_layer(name="missing", properties={"Vs": 0.5 * prop("Vp")})
    missing.add_surface(name="bottom", depth=1.0)

    with pytest.raises(ValueError, match="references property 'vp'"):
        missing.sample_uniform([2, 3], properties="Vs")


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


def test_layered_model_plot_converts_surface_units_to_axis_units():
    import matplotlib.pyplot as plt

    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0 * u.m)
    model.add_layer(name="upper", properties={"Vp": 1.5})
    model.add_surface(name="interface", depth=250.0 * u.m)
    model.add_layer(name="lower", properties={"Vp": 2.0})
    model.add_surface(name="bottom", depth=500.0 * u.m)

    sampled = model.sample_uniform([2, 5], axes_units={"x": "km", "z": "km"})

    np.testing.assert_allclose(
        sampled.coords["z"].values, [0.0, 0.125, 0.25, 0.375, 0.5]
    )
    assert sampled.coords["z"].attrs["units"] == "km"
    assert sampled["vp"].isel(x=0, z=-1).item() == 2.0

    fig, ax = plt.subplots()
    model.plot(
        "Vp",
        resolution=[2, 5],
        ax=ax,
        axes_units={"x": "km", "z": "km"},
    )

    y0, y1 = ax.get_ylim()
    assert max(abs(y0), abs(y1)) <= 0.6
    plt.close(fig)


def test_layered_model_sampling_converts_property_axis_units():
    z = np.linspace(0.0, 200.0, 1001)
    vp = xr.DataArray(
        data=1.9 + 2.3 * (1.0 - np.exp(-z / 120.0)),
        dims=["z"],
        coords={"z": z},
        attrs={"units": "km/s"},
    )
    vp.coords["z"].attrs["units"] = "m"

    model = LayeredModel(dimension=2, x_limits=[0.0, 40.0] * u.m)
    model.add_surface(name="top", depth=0.0 * u.m)
    model.add_layer(name="gradient", properties={"Vp": vp})
    model.add_surface(name="bottom", depth=200.0 * u.m)

    sampled = model.sample_uniform([2, 3], axes_units={"x": "km", "z": "km"})

    assert sampled.coords["z"].values.tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert sampled["vp"].isel(x=0, z=0).item() == pytest.approx(vp.sel(z=0).item())
    assert sampled["vp"].isel(x=0, z=1).item() == pytest.approx(vp.sel(z=100).item())
    assert sampled["vp"].isel(x=0, z=2).item() == pytest.approx(vp.sel(z=200).item())


def test_layered_model_plot_draws_borehole_boundaries():
    import matplotlib.pyplot as plt

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
                "r": 0.035,
                "physics": "acoustic",
                "properties": {"Vp": 1.48},
            }
        ],
    )

    fig, ax = plt.subplots()
    model.plot(
        "Vp",
        resolution=[4, 4],
        ax=ax,
        surfaces=False,
        boreholes=True,
        add_colorbar=False,
    )

    assert len(ax.lines) >= 3
    assert any(text.get_text() == "fluid" for text in ax.texts)
    plt.close(fig)


def test_layered_model_sampling_applies_borehole_material_values():
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
                "r": 0.06,
                "physics": "acoustic",
                "properties": {"Vp": 1.48},
            }
        ],
    )

    sampled = model.sample_uniform([11, 3])

    assert sampled["vp"].sel(x=0.4, z=0.25).item() == pytest.approx(1.48)
    assert sampled["vp"].sel(x=0.5, z=0.25).item() == pytest.approx(1.48)
    assert sampled["vp"].sel(x=0.0, z=0.25).item() == pytest.approx(2.2)


def test_layered_model_plot_hides_borehole_boundaries_by_default():
    import matplotlib.pyplot as plt

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
                "r": 0.06,
                "physics": "acoustic",
                "properties": {"Vp": 1.48},
            }
        ],
    )

    fig, ax = plt.subplots()
    model.plot("Vp", resolution=[11, 3], ax=ax, surfaces=False, add_colorbar=False)

    assert len(ax.lines) == 0
    assert ax.images[0].get_array().min() == pytest.approx(1.48)
    plt.close(fig)


def test_layered_model_plot_ignores_unrequested_expression_properties():
    import matplotlib.pyplot as plt

    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(
        name="formation",
        mesh_block_id=1,
        properties={
            "Vp": 2.2,
            "Rho": {"expr": {"ref": "vp"}},
        },
    )
    model.add_surface(name="bottom", depth=0.5)

    fig, ax = plt.subplots()
    model.plot("Vp", resolution=[3, 3], ax=ax, surfaces=False, add_colorbar=False)

    np.testing.assert_allclose(ax.images[0].get_array(), 2.2)
    plt.close(fig)


def test_layered_model_plot_materializes_requested_expression_property():
    import matplotlib.pyplot as plt

    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0] * u.km)
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="upper",
        properties={"Vp": 2.0 * u.km / u.s, "Vs": 0.5 * prop("Vp")},
    )
    model.add_surface(name="interface", depth=0.5 * u.km)
    model.add_layer(
        name="lower",
        properties={"Vp": 3.0 * u.km / u.s, "Vs": 0.5 * prop("Vp")},
    )
    model.add_surface(name="bottom", depth=1.0 * u.km)

    fig, ax = plt.subplots()
    image = model.plot(
        "Vs",
        resolution=[3, 5],
        ax=ax,
        surfaces=False,
        add_colorbar=False,
    )

    assert image.get_clim() == pytest.approx((1.0, 1.5))
    np.testing.assert_allclose(image.get_array()[0], 1.0)
    np.testing.assert_allclose(image.get_array()[-1], 1.5)
    plt.close(fig)


def test_layered_model_3d_plot_uses_clean_interactive_slices(monkeypatch):
    pv = pytest.importorskip("pyvista")

    class PlotterSpy:
        instances = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.meshes = []
            self.show_kwargs = None
            self.bounds_kwargs = None
            self.background = None
            self.camera_position = None
            self.camera = type(
                "CameraSpy",
                (),
                {
                    "position": (1.0, 1.0, 1.0),
                    "focal_point": (0.0, 0.0, 0.0),
                    "up": (0.0, 0.0, 1.0),
                },
            )()
            self.__class__.instances.append(self)

        def set_background(self, color):
            self.background = color

        def add_mesh(self, mesh, **kwargs):
            self.meshes.append((mesh, kwargs))

        def set_scale(self, **kwargs):
            self.scale = kwargs

        def show_bounds(self, **kwargs):
            self.bounds_kwargs = kwargs

        def show_axes(self):
            self.axes_shown = True

        def enable_anti_aliasing(self, mode):
            self.anti_aliasing = mode

        def reset_camera(self):
            self.camera_reset = True

        def show(self, **kwargs):
            self.show_kwargs = kwargs
            return "interactive-viewer"

    monkeypatch.setattr(pv, "Plotter", PlotterSpy)

    model = LayeredModel(
        dimension=3,
        x_limits=[0.0, 1.0] * u.km,
        y_limits=[0.0, 1.0] * u.km,
    )
    model.add_surface(name="top", depth=0.0 * u.km)
    model.add_layer(
        name="formation",
        properties={
            "Vp": 2.2 * u.km / u.s,
            "Vs": 0.5 * prop("Vp"),
        },
    )
    model.add_surface(name="bottom", depth=0.5 * u.km)

    result = model.plot(
        "Vp",
        resolution=[3, 3, 3],
        surfaces=False,
        interactive=True,
    )

    plotter = PlotterSpy.instances[-1]
    assert result == "interactive-viewer"
    assert plotter.show_kwargs == {"jupyter_backend": "html"}
    assert isinstance(plotter.meshes[0][0], pv.MultiBlock)
    assert plotter.meshes[0][1]["show_edges"] is False
    assert plotter.bounds_kwargs["grid"] is None
    assert plotter.anti_aliasing == "fxaa"
    assert plotter.camera.up == (0.0, 0.0, 1.0)

    model.plot("Vs", resolution=[3, 3, 3], surfaces=False, interactive=True)
    expression_plotter = PlotterSpy.instances[-1]
    assert expression_plotter.meshes[0][1]["scalars"] == "Vs"
    assert expression_plotter.meshes[0][1]["clim"] == pytest.approx([1.1, 1.1])

    model.plot("Vp", resolution=[3, 3, 3], surfaces=False)
    static_plotter = PlotterSpy.instances[-1]
    assert static_plotter.show_kwargs == {"jupyter_backend": "static"}
    assert static_plotter.camera.position == (1.0, 1.0, -1.0)
    assert static_plotter.camera.up == (0.0, 0.0, -1.0)

    model.plot("Vp", resolution=[3, 3, 3], surfaces=False, flip_z=False)
    elevation_plotter = PlotterSpy.instances[-1]
    assert elevation_plotter.camera.position == (1.0, 1.0, 1.0)
    assert elevation_plotter.camera.up == (0.0, 0.0, 1.0)


def test_borehole_draw_plots_material_radii():
    import matplotlib.pyplot as plt

    model = LayeredModel(dimension=2, x_limits=[0.0, 1.0])
    model.add_surface(name="top", depth=0.0)
    model.add_layer(name="formation", mesh_block_id=1, properties={"Vp": 2.2})
    model.add_surface(name="bottom", depth=0.5)
    borehole = model.add_borehole(
        name="bh1",
        x=0.45,
        parts=[
            {
                "name": "fluid",
                "mesh_block_id": 20,
                "r": 0.035,
                "physics": "acoustic",
                "properties": {"Vp": 1.48},
            },
            {
                "name": "casing",
                "mesh_block_id": 21,
                "r": 0.041,
                "physics": "elastic",
                "properties": {"Vp": 5.9, "Vs": 3.2},
            },
        ],
    )

    fig, ax = plt.subplots()
    returned = borehole.draw(ax=ax, subdomains=model.subdomains, annotate=True)

    assert returned is ax
    assert len(ax.patches) == 2
    assert ax.get_aspect() == 1.0
    assert any("bh1_fluid" in text.get_text() for text in ax.texts)
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
