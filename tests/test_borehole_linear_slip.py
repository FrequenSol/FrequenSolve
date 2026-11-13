"""Borehole imperfect-bond authoring, units, and round trips."""
import numpy as np
import pytest
from pint.errors import DimensionalityError
import frequensolve as fs

u = fs.ureg


def test_linear_slip_stiffness_units_and_validation():
    bond = fs.BoreholeLinearSlip(normal_stiffness=2*u.GPa/u.m, tangential_stiffness=0)
    assert bond.to_fs() == {"model": "linear_slip", "normal_stiffness": 2e9, "tangential_stiffness": 0}
    assert fs.BoreholeLinearSlip.from_fs(bond.to_fs()) == bond
    for value in [-1, np.inf, np.nan]:
        with pytest.raises(ValueError):
            fs.BoreholeLinearSlip(normal_stiffness=value, tangential_stiffness=0)
    with pytest.raises(DimensionalityError):
        fs.BoreholeLinearSlip(normal_stiffness=1*u.Pa, tangential_stiffness=0)


def make_well():
    model = fs.LayeredModel(dimension=2, x_limits=[0, 10])
    model.add_surface(name="top", depth=0)
    model.add_layer(name="rock", physics="elastic", properties={"Vp": 3000, "Vs": 1500})
    model.add_surface(name="bottom", depth=200)
    return model, model.add_borehole(name="well", x=0)


def test_layer_interface_is_on_outer_wall_and_roundtrips():
    model, well = make_well()
    bond = fs.BoreholeLinearSlip(normal_stiffness=1e9, tangential_stiffness=0)
    well.add_layer("fluid", width=.0762, physics="acoustic", properties={"Vp": 1500})
    well.add_layer("casing", width=.0254, physics="elastic", properties={"Vp": 2600, "Vs": 1000}, interface=bond)
    well.add_layer("cement", width=.1016, physics="elastic", properties={"Vp": 3000, "Vs": 1500})
    payload = model.to_fs()
    walls = payload["boreholes"][0]["surfaces"]
    assert "interface" not in walls[0]
    assert walls[1]["interface"] == bond.to_fs()
    assert "interface" not in walls[2]
    well.set_interface("casing", None)
    assert well.surfaces[1].interface is None
    well.set_interface("casing", bond)
    with pytest.raises(ValueError, match="Unknown"):
        well.set_interface("missing", bond)
    assert fs.LayeredModel.from_fs(payload).to_fs() == payload


def test_explicit_and_object_layer_forms():
    bond = fs.BoreholeLinearSlip(normal_stiffness=0, tangential_stiffness=0)
    _, well = make_well()
    well += fs.BoreholeLayer("casing", width=.1, interface=bond, physics="elastic", properties={"Vp": 3000, "Vs": 1500})
    assert well.surfaces[-1].interface == bond
    well.add_layer("cement", interface=bond, physics="elastic", properties={"Vp": 3000, "Vs": 1500})
    well.add_surface("cement_outer", r=.2)
    assert well.surfaces[-1].interface == bond
    well.add_layer("third", mesh_block_id=well.parts[0].mesh_block_id)
    well += fs.BoreholeSurface("third_outer", r=.3, interface=bond)
    assert well.surfaces[-1].interface == bond
    with pytest.raises(ValueError, match="axis"):
        fs.BoreholeSurface("axis", r=0, interface=bond).to_fs()


def test_constructor_layer_bond_matches_interactive_authoring():
    model, _ = make_well()
    bond = fs.BoreholeLinearSlip(normal_stiffness=1e9, tangential_stiffness=0)
    well = model.add_borehole(
        name="second",
        x=1,
        layers=[fs.BoreholeLayer("solid", outer_surface="wall", properties={"Vp": 3000, "Vs": 1500}, physics="elastic", interface=bond)],
        surfaces=[fs.BoreholeSurface("wall", r=.1)],
    )
    assert well.to_fs()["surfaces"][0]["interface"] == bond.to_fs()


def test_variable_radius_bond_is_rejected_until_pml_continuation_is_supported():
    import xarray as xr
    bond = fs.BoreholeLinearSlip(normal_stiffness=1e9, tangential_stiffness=0)
    radius = xr.DataArray([.1, .2], dims=["z"], coords={"z": [0., 200.]})
    with pytest.raises(ValueError, match="constant radius"):
        fs.BoreholeSurface("wall", r=radius, interface=bond).to_fs()
