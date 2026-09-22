import json

import numpy as np
import pytest

from frequensolve.model.parameterization import (
    BSplineControl,
    HatControl,
    ParameterizedProperty,
)
from frequensolve.model.property import Property


def test_hat_control_is_an_ordered_uniform_grid_without_point_ids():
    control = HatControl(
        coordinate_system="top_relative",
        axis="below",
        origin=0.1,
        spacing=0.25,
        coefficients=[0.0, 0.2, -0.1],
    )

    np.testing.assert_allclose(control.coordinates, [0.1, 0.35, 0.6])
    assert control.to_fs() == {
        "kind": "hat",
        "coordinate_system": "top_relative",
        "axis": "below",
        "origin": 0.1,
        "spacing": 0.25,
        "coefficients": [0.0, 0.2, -0.1],
    }


def test_property_control_coefficients_reject_complex_values():
    with pytest.raises(ValueError, match="real-valued"):
        HatControl(axis="z", spacing=1.0, coefficients=[0.0j, 0.0j])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"spacing": 0.0, "coefficients": [0.0, 0.0]}, "positive"),
        ({"spacing": 1.0, "coefficients": [0.0]}, "at least 2"),
        ({"spacing": 1.0, "coefficients": [0.0, np.nan]}, "finite"),
    ],
)
def test_hat_control_rejects_invalid_uniform_grids(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HatControl(axis="z", **kwargs)


def test_bspline_control_validates_coefficient_count():
    with pytest.raises(ValueError, match=r"len\(knots\) - degree - 1"):
        BSplineControl(
            axis="z",
            degree=2,
            knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            coefficients=[0.0, 0.0],
        )


def test_parameterized_property_round_trips_as_a_property_provider():
    authored = ParameterizedProperty(
        0.5,
        id="sediment_sp",
        transform="log",
        control=HatControl(
            coordinate_system="top_relative",
            axis="below",
            spacing=0.1,
            coefficients=[0.0, 0.0, 0.0],
        ),
    )

    payload = authored.to_fs()
    loaded = Property.from_value(payload)

    assert isinstance(loaded, ParameterizedProperty)
    assert loaded.id == "sediment_sp"
    assert loaded.transform == "log"
    np.testing.assert_array_equal(loaded.coefficients, np.zeros(3))
    assert loaded.to_fs() == payload


def test_parameterized_property_accepts_inline_xarray_reference_payload():
    payload = {
        "parameterized": {
            "id": "sediment_sp",
            "reference": {
                "dims": ["z"],
                "coords": {"z": {"data": [0.0, 0.1, 0.2]}},
                "data": [0.5, 0.45, 0.4],
            },
            "transform": "log",
            "control": {
                "kind": "hat",
                "axis": "below",
                "spacing": 0.1,
                "coefficients": [0.0, 0.0, 0.0],
            },
        }
    }

    loaded = Property.from_value(payload)

    assert isinstance(loaded, ParameterizedProperty)
    np.testing.assert_allclose(loaded.reference.data, [0.5, 0.45, 0.4])
    np.testing.assert_allclose(loaded.reference.data.coords["z"], [0.0, 0.1, 0.2])


def test_parameterized_property_preserves_value_and_control_units():
    authored = ParameterizedProperty(
        {"value": 0.55},
        id="sediment_sp",
        units="s/km",
        transform="log",
        control=HatControl(
            coordinate_system="top_relative",
            axis="below",
            units="m",
            spacing=50.0,
            coefficients=[0.0, 0.0, 0.0],
        ),
    )

    assert authored.to_fs() == {
        "parameterized": {
            "id": "sediment_sp",
            "reference": {"value": 0.55},
            "transform": "log",
            "control": {
                "kind": "hat",
                "coordinate_system": "top_relative",
                "axis": "below",
                "origin": 0.0,
                "spacing": 50.0,
                "coefficients": [0.0, 0.0, 0.0],
                "units": "m",
            },
        },
        "units": "s/km",
    }


def test_parameterized_property_promotes_reference_units_to_the_wrapper():
    authored = ParameterizedProperty(
        Property.file("starting_model.h5:/Vp", units="m/s"),
        id="sediment_vp",
        transform="log",
        control=HatControl(
            axis="z",
            spacing=0.5,
            coefficients=[0.0, 0.0],
        ),
    )

    payload = authored.to_fs()

    assert authored.units == "m/s"
    assert payload["units"] == "m/s"
    assert "units" not in payload["parameterized"]["reference"]
    assert Property.from_value(payload).to_fs() == payload


def test_parameterized_property_rejects_conflicting_reference_units():
    with pytest.raises(ValueError, match="units disagree with reference units"):
        ParameterizedProperty(
            Property.file("starting_model.h5:/Vp", units="m/s"),
            id="sediment_vp",
            units="km/s",
            transform="log",
            control=HatControl(
                axis="z",
                spacing=0.5,
                coefficients=[0.0, 0.0],
            ),
        )


# ----------------------------------------------------------------------
# Phase 0 control kinds: tensor_hat, mesh + property_spaces, transforms
# ----------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from jsonschema import Draft202012Validator, ValidationError  # noqa: E402
from referencing import Registry, Resource  # noqa: E402

from frequensolve.model import LayeredModel, ModelBase  # noqa: E402
from frequensolve.model.parameterization import (  # noqa: E402
    CONTROL_TRANSFORMS,
    MeshControl,
    MeshPropertySpace,
    TensorHatControl,
    control_from_fs,
)
from frequensolve.units import Q_  # noqa: E402

_CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-f533e6f" / "trunk" / "contracts"
)
_MATERIAL_SCHEMA = _CONTRACT_ROOT / "inputs" / "fs-material-model-1" / "schema.json"


def _material_validator(pointer: str = "") -> Draft202012Validator:
    registry = Registry()
    for schema_file in list(_CONTRACT_ROOT.rglob("schema.json")) + list(
        _CONTRACT_ROOT.glob("fragments/*.json")
    ):
        contents = json.loads(schema_file.read_text())
        if "$id" not in contents:
            continue  # a few pinned output schemas are unaddressable fragments
        registry = registry.with_resource(
            contents["$id"], Resource.from_contents(contents)
        )
    schema_id = json.loads(_MATERIAL_SCHEMA.read_text())["$id"]
    return Draft202012Validator({"$ref": schema_id + pointer}, registry=registry)


def _tensor_control(**overrides):
    kwargs = dict(
        axes=["x", "z"],
        shape=[2, 3],
        origin=[0.0, 10.0],
        spacing=[1.0, 2.0],
        coefficients=np.zeros(6),
    )
    kwargs.update(overrides)
    return TensorHatControl(**kwargs)


def test_tensor_hat_control_orders_the_first_axis_fastest():
    control = _tensor_control(
        coefficients=np.arange(6.0).reshape(2, 3),
        units="km",
        coordinate_system="global",
    )

    # Lattice-shaped input is flattened so axis 0 (x) varies fastest.
    np.testing.assert_array_equal(control.coefficients, [0, 3, 1, 4, 2, 5])
    np.testing.assert_array_equal(control.grid, np.arange(6.0).reshape(2, 3))
    assert control.size == 6
    assert control.ndim == 2
    np.testing.assert_allclose(
        control.coordinates,
        [[0, 10], [1, 10], [0, 12], [1, 12], [0, 14], [1, 14]],
    )
    assert control.to_fs() == {
        "kind": "tensor_hat",
        "coordinate_system": "global",
        "axes": ["x", "z"],
        "shape": [2, 3],
        "origin": [0.0, 10.0],
        "spacing": [1.0, 2.0],
        "coefficients": [0.0, 3.0, 1.0, 4.0, 2.0, 5.0],
        "units": "km",
    }
    _material_validator("#/$defs/tensorHatControlMap").validate(control.to_fs())


def test_tensor_hat_control_round_trips_flat_coefficients_in_three_dimensions():
    control = TensorHatControl(
        axes=["x", "y", "z"],
        shape=[2, 2, 2],
        origin=[0.0, 0.0, 0.0],
        spacing=[0.5, 0.5, 0.5],
        coefficients=np.arange(8.0),
    )

    restored = control_from_fs(control.to_fs())

    assert isinstance(restored, TensorHatControl)
    assert restored.to_fs() == control.to_fs()
    np.testing.assert_array_equal(restored.grid[1, 0, 0], 1.0)
    np.testing.assert_array_equal(restored.grid[0, 1, 0], 2.0)
    np.testing.assert_array_equal(restored.grid[0, 0, 1], 4.0)
    np.testing.assert_allclose(restored.coordinates[6], [0.0, 0.5, 0.5])
    replaced = restored.with_coefficients(np.zeros(8))
    assert replaced.size == 8
    np.testing.assert_array_equal(replaced.coefficients, np.zeros(8))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"axes": ["x"]}, "two or three axes"),
        ({"axes": ["x", "x"]}, "distinct"),
        ({"shape": [1, 3]}, "at least 2"),
        ({"shape": [2, 3, 2]}, "one value per lattice axis"),
        ({"spacing": [0.0, 2.0]}, "positive"),
        ({"origin": [np.nan, 0.0]}, "finite"),
        ({"coefficients": np.zeros(5)}, "product\\(shape\\)"),
        ({"coefficients": np.zeros((3, 2))}, "does not match lattice shape"),
        ({"coefficients": np.zeros(6, dtype=complex)}, "real-valued"),
    ],
)
def test_tensor_hat_control_rejects_inconsistent_lattices(overrides, message):
    with pytest.raises(ValueError, match=message):
        _tensor_control(**overrides)


def test_tensor_hat_parameterized_property_matches_sauce_contract():
    prop = ParameterizedProperty(
        {"value": 0.01, "units": "S/m"},
        id="target_sigma",
        transform="log",
        control=_tensor_control(units="km"),
    )

    payload = prop.to_fs()
    _material_validator("#/$defs/property").validate(payload)
    loaded = Property.from_value(payload)
    assert isinstance(loaded, ParameterizedProperty)
    assert isinstance(loaded.control, TensorHatControl)
    assert loaded.to_fs() == payload
    np.testing.assert_allclose(loaded.control.coordinates, prop.control.coordinates)


def test_mesh_control_has_no_inline_coefficients():
    control = MeshControl(space="materials")

    assert control.size is None
    assert control.coordinates is None
    assert control.coefficients is None
    assert control.to_fs() == {"kind": "mesh", "space": "materials"}
    assert control_from_fs(control.to_fs()) == control
    _material_validator("#/$defs/meshControlMap").validate(control.to_fs())
    with pytest.raises(ValueError, match="checkpoint"):
        control.with_coefficients([0.0])
    with pytest.raises(ValueError, match="property-space name"):
        MeshControl(space=" ")


def test_mesh_property_space_normalizes_frequency_and_epw():
    space = MeshPropertySpace(artifact="controls.h5", frequency=Q_(10, "Hz"), epw=2)
    assert space.to_fs() == {"artifact": "controls.h5", "frequency": 10.0, "epw": 2.0}

    space = MeshPropertySpace(
        artifact="controls.h5",
        frequency={"value": 0.01, "units": "kHz"},
        epw=[2, 3],
    )
    assert space.to_fs() == {
        "artifact": "controls.h5",
        "frequency": 10.0,
        "epw": [2.0, 3.0],
    }
    _material_validator("#/$defs/meshPropertySpace").validate(space.to_fs())
    assert MeshPropertySpace.from_fs(space.to_fs()) == space

    with pytest.raises(ValueError, match=r"\.h5"):
        MeshPropertySpace(artifact="controls.txt", frequency=10.0, epw=2.0)
    with pytest.raises(ValueError, match="positive"):
        MeshPropertySpace(artifact="controls.h5", frequency=0.0, epw=2.0)
    with pytest.raises(ValueError, match="hertz"):
        MeshPropertySpace(artifact="controls.h5", frequency=Q_(1, "m"), epw=2.0)
    with pytest.raises(ValueError, match="one value per dimension"):
        MeshPropertySpace(artifact="controls.h5", frequency=10.0, epw=[1, 2, 3, 4])
    with pytest.raises(ValueError, match="positive"):
        MeshPropertySpace(artifact="controls.h5", frequency=10.0, epw=[2.0, -1.0])


def test_mesh_control_with_property_spaces_matches_sauce_contract():
    vp = ParameterizedProperty(
        {"value": 3.0, "units": "km/s"},
        id="rock_vp",
        transform="log",
        control=MeshControl("materials"),
    )
    model = LayeredModel(
        name="mesh_controls",
        dimension=2,
        x_limits=[0.0, 10.0],
        property_spaces={
            "materials": {
                "artifact": "material-controls.h5",
                "frequency": 10.0,
                "epw": 2.0,
            }
        },
    )
    model.add_surface(0.0, name="top")
    model.add_layer(name="rock", properties={"vp": vp, "rho": 2.3}, physics="acoustic")
    model.add_surface(10.0, name="bottom")

    payload = model.to_fs()
    assert payload["property_spaces"] == {
        "materials": {
            "artifact": "material-controls.h5",
            "frequency": 10.0,
            "epw": 2.0,
        }
    }
    control = payload["subdomains"][0]["properties"]["vp"]["parameterized"]["control"]
    assert control == {"kind": "mesh", "space": "materials"}
    _material_validator().validate({"schema": "fs-material-model-1", **payload})

    restored = LayeredModel.from_fs(payload)
    assert isinstance(restored.property_spaces["materials"], MeshPropertySpace)
    assert restored.to_fs() == payload
    loaded_vp = restored.layers["rock"].properties["vp"]
    assert isinstance(loaded_vp, ParameterizedProperty)
    assert isinstance(loaded_vp.control, MeshControl)
    assert loaded_vp.coefficients is None


def test_base_model_serializes_property_spaces_without_surfaces():
    model = ModelBase(
        name="base",
        dimension=3,
        property_spaces={
            "materials": MeshPropertySpace("space.h5", 5.0, [2.0, 2.0, 4.0])
        },
    )

    payload = model.to_fs()
    assert payload["property_spaces"]["materials"]["epw"] == [2.0, 2.0, 4.0]
    assert "surfaces" not in payload
    restored = ModelBase.from_fs(payload)
    assert restored.property_spaces == model.property_spaces
    assert restored.to_fs() == payload
    assert "property_spaces" not in ModelBase(name="plain", dimension=2).to_fs()
    with pytest.raises(ValueError, match="1 to 64"):
        ModelBase(
            name="bad",
            dimension=2,
            property_spaces={"": {"artifact": "a.h5", "frequency": 1, "epw": 1}},
        )


@pytest.mark.parametrize("transform", CONTROL_TRANSFORMS)
def test_parameterized_property_accepts_every_contract_transform(transform):
    reference = {"value": 0.3} if transform == "logit" else {"value": 2.0, "units": "s"}
    prop = ParameterizedProperty(
        reference,
        id="block",
        transform=transform,
        control=HatControl(axis="z", spacing=1.0, coefficients=[0.0, 0.0]),
    )

    payload = prop.to_fs()
    assert payload["parameterized"]["transform"] == transform
    _material_validator("#/$defs/property").validate(payload)
    loaded = Property.from_value(payload)
    assert loaded.transform == transform
    assert loaded.to_fs() == payload


@pytest.mark.parametrize(
    "reference, transform, message",
    [
        (1.0, "sqrt", "must be one of identity, log, inverse, logit"),
        ({"value": 0.3, "units": "m/s"}, "logit", "dimensionless"),
        ({"value": 1.0}, "logit", "strictly between 0 and 1"),
        ({"value": 0.0}, "logit", "strictly between 0 and 1"),
        ({"value": 0.0, "units": "s"}, "inverse", "strictly positive"),
        ({"value": -2.0}, "inverse", "strictly positive"),
    ],
)
def test_parameterized_property_rejects_invalid_transforms(
    reference, transform, message
):
    with pytest.raises(ValueError, match=message):
        ParameterizedProperty(
            reference,
            id="block",
            transform=transform,
            control=HatControl(axis="z", spacing=1.0, coefficients=[0.0, 0.0]),
        )


def test_logit_transform_accepts_explicitly_dimensionless_references():
    prop = ParameterizedProperty(
        {"value": 0.7, "units": "dimensionless"},
        id="ip_exponent",
        transform="logit",
        control=BSplineControl(
            axis="z", degree=1, knots=[0.0, 0.0, 10.0, 10.0], coefficients=[0.0, 0.2]
        ),
    )

    payload = prop.to_fs()
    _material_validator("#/$defs/property").validate(payload)
    assert Property.from_value(payload).to_fs() == payload
    with pytest.raises(ValidationError):
        _material_validator("#/$defs/property").validate(
            {
                **payload,
                "parameterized": {**payload["parameterized"], "transform": "sqrt"},
            }
        )


def test_control_from_fs_rejects_unknown_kinds():
    with pytest.raises(ValueError, match="unsupported property-control kind"):
        control_from_fs({"kind": "wavelet"})
