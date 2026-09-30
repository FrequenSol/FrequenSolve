"""Blend authoring survives ordinary simulation persistence and contracts."""

import numpy as np
import pytest

from frequensolve.model import BlendProperty, LayeredModel, Property, RBFSurface
from frequensolve.model.parameterization import HatControl, ParameterizedProperty
from frequensolve.simulation import SeismicSimulation
from frequensolve.units import ureg as u
from tests.test_imaging_controls import _material_validator


def test_blend_roundtrip_and_material_contract(tmp_path):
    outside = ParameterizedProperty(
        2.5,
        id="background",
        transform="identity",
        control=HatControl(axis="z", origin=0, spacing=0.5, coefficients=[0, 0.1, 0.2]),
    )
    blend = BlendProperty(
        "salt", width=20 * u.m, inside=4.2, outside=outside, units="km/s"
    )
    restored = Property.from_value(blend.to_fs())
    assert isinstance(restored, BlendProperty)
    assert isinstance(restored.outside, ParameterizedProperty)
    assert restored.to_fs() == blend.to_fs()
    model = LayeredModel(dimension=2, x_limits=[0, 1])
    model.add_surface(0, name="top")
    model.add_layer(name="host", properties={"vp": blend, "rho": 2.2})
    model.add_surface(1, name="bottom")
    model += RBFSurface("salt", 0.4, [[0.5, 0.5]], [-0.2], bias=0.05)
    assert list(_material_validator().iter_errors(model.to_fs())) == []
    sim = SeismicSimulation(
        name="blend",
        dimension=2,
        physics="acoustic",
        project_path=tmp_path,
        model=model,
    )
    sim.save()
    reloaded = SeismicSimulation.load(sim._file)
    result = reloaded.model.layers[0].properties["vp"]
    assert isinstance(result, BlendProperty)
    assert result.to_fs() == blend.to_fs()
    with pytest.raises(ValueError, match="Sauce geometry"):
        result.get()


@pytest.mark.parametrize("width", [0, -1, np.nan, np.inf])
def test_blend_rejects_invalid_width(width):
    with pytest.raises(ValueError, match="width"):
        BlendProperty("salt", width=width, inside=4.2, outside=2.5)


def test_blend_common_property_units():
    blend = BlendProperty(
        "salt", width=0.02, inside=4.2 * u.km / u.s, outside=2.5 * u.km / u.s
    )
    assert blend.to_fs()["units"] == "km/s"
    assert "units" not in blend.to_fs()["blend"]["inside"]
    with pytest.raises(ValueError, match="same property units"):
        BlendProperty(
            "salt", width=0.02, inside=4200 * u.m / u.s, outside=2.5 * u.km / u.s
        )
