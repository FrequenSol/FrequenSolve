import json

import numpy as np
import pytest

from frequensolve.model.parameterization import (
    BSplineControl,
    HatControl,
    ParameterizedProperty,
    TensorBSplineControl,
    TensorHatControl,
    control_from_fs,
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


@pytest.mark.parametrize(
    "factory",
    [
        lambda **kw: HatControl(axis="z", spacing=1.0, coefficients=[0.0, 1.0], **kw),
        lambda **kw: BSplineControl(
            axis="z",
            degree=1,
            knots=[0.0, 0.0, 1.0, 1.0],
            coefficients=[0.0, 1.0],
            **kw,
        ),
        lambda **kw: TensorHatControl(
            axes=["x", "z"],
            shape=[2, 2],
            origin=[0.0, 0.0],
            spacing=[1.0, 1.0],
            coefficients=np.zeros(4),
            **kw,
        ),
        lambda **kw: TensorBSplineControl(
            axes=["x", "z"],
            degree=1,
            knots=[[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
            coefficients=np.zeros(4),
            **kw,
        ),
    ],
    ids=["hat", "bspline", "tensor_hat", "tensor_bspline"],
)
def test_controls_round_trip_the_extrapolate_policy(factory):
    default = factory()
    assert default.extrapolate == "zero"
    assert "extrapolate" not in default.to_fs()
    assert control_from_fs(default.to_fs()).extrapolate == "zero"

    held = factory(extrapolate="HOLD")
    assert held.extrapolate == "hold"
    payload = held.to_fs()
    assert payload["extrapolate"] == "hold"
    restored = control_from_fs(payload)
    assert restored.extrapolate == "hold"
    assert restored.to_fs() == payload
    assert held.with_coefficients(np.ones(held.size)).extrapolate == "hold"
    _material_validator(f"#/$defs/{_SCHEMA_DEF[payload['kind']]}").validate(payload)

    with pytest.raises(ValueError, match="extrapolate must be one of"):
        factory(extrapolate="linear")


_SCHEMA_DEF = {
    "hat": "hatControlMap",
    "bspline": "bsplineControlMap",
    "tensor_hat": "tensorHatControlMap",
    "tensor_bspline": "tensorBSplineControlMap",
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


def test_held_controls_extend_their_edge_values_when_applied():
    from frequensolve.model.representation import EvaluationContext

    reference = Property.from_value(10.0)
    hat = ParameterizedProperty(
        id="held_hat",
        reference=reference,
        control=HatControl(
            axis="z",
            origin=1.0,
            spacing=1.0,
            coefficients=[1.0, 2.0, 0.5],
            extrapolate="hold",
        ),
    )
    zero = ParameterizedProperty(
        id="zero_hat",
        reference=reference,
        control=HatControl(
            axis="z", origin=1.0, spacing=1.0, coefficients=[1.0, 2.0, 0.5]
        ),
    )
    context = EvaluationContext({"z": [-5.0, 1.0, 1.5, 3.0, 9.0]})
    np.testing.assert_allclose(
        hat.apply_control(10.0, context), [11.0, 11.0, 11.5, 10.5, 10.5]
    )
    np.testing.assert_allclose(
        zero.apply_control(10.0, context), [10.0, 11.0, 11.5, 10.5, 10.0]
    )

    lattice = ParameterizedProperty(
        id="held_lattice",
        reference=reference,
        control=TensorHatControl(
            axes=["x", "z"],
            shape=[2, 2],
            origin=[2.0, 0.0],
            spacing=[1.0, 1.0],
            coefficients=np.array([[1.0, 3.0], [2.0, 4.0]]),
            extrapolate="hold",
        ),
    )
    context = EvaluationContext({"x": [0.0, 2.5, 7.0, 0.0], "z": [0.5, 0.5, 1.0, -3.0]})
    # Left of the lattice the x = 2 column holds (1 -> 3 along z), right of it the x = 3 column.
    np.testing.assert_allclose(
        lattice.apply_control(10.0, context), [12.0, 12.5, 14.0, 11.0]
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


_LAND_X = [0.45, 0.45, 0.9, 1.27, 1.6, 1.6]
_LAND_Z = [0, 0, 0, 0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.15, 0.15, 0.15]


def _tensor_bspline(**overrides):
    kwargs = dict(
        axes=["x", "z"],
        degree=[1, 3],
        knots=[_LAND_X, _LAND_Z],
        coefficients=np.zeros(32),
        coordinate_system="seabed_depth",
    )
    kwargs.update(overrides)
    return TensorBSplineControl(**kwargs)


def test_tensor_bspline_control_serializes_the_sauce_contract():
    coefficients = np.arange(32.0)
    control = _tensor_bspline(coefficients=coefficients, units="km", extrapolate="hold")

    payload = control.to_fs()
    assert payload == {
        "kind": "tensor_bspline",
        "coordinate_system": "seabed_depth",
        "units": "km",
        "extrapolate": "hold",
        "axes": [
            {"axis": "x", "degree": 1, "knots": _LAND_X},
            {"axis": "z", "degree": 3, "knots": [float(v) for v in _LAND_Z]},
        ],
        "coefficients": coefficients.tolist(),
    }
    assert list(payload) == [
        "kind",
        "coordinate_system",
        "units",
        "extrapolate",
        "axes",
        "coefficients",
    ]
    _material_validator("#/$defs/tensorBSplineControlMap").validate(payload)
    restored = control_from_fs(payload)
    assert isinstance(restored, TensorBSplineControl)
    assert restored.to_fs() == payload
    # Defaults are omitted, like the other control maps.
    plain = _tensor_bspline(coordinate_system="global").to_fs()
    assert "units" not in plain and "extrapolate" not in plain
    _material_validator("#/$defs/tensorBSplineControlMap").validate(plain)
    with pytest.raises(ValidationError):
        _material_validator("#/$defs/tensorBSplineControlMap").validate(
            {**plain, "axes": [{**plain["axes"][0], "degree": 8}, plain["axes"][1]]}
        )


def test_tensor_bspline_control_orders_the_first_axis_fastest():
    grid = np.arange(32.0).reshape(4, 8)
    control = _tensor_bspline(coefficients=grid)

    assert control.shape == (4, 8)
    assert control.size == 32 and control.ndim == 2
    # Tensor-shaped input is flattened so axis 0 (x) varies fastest.
    np.testing.assert_array_equal(control.coefficients[:5], [0, 8, 16, 24, 1])
    np.testing.assert_array_equal(control.grid, grid)
    greville_z = [0.0, 0.01, 0.03, 0.06, 0.09, 0.12, 0.14, 0.15]
    np.testing.assert_allclose(control.axis_coordinates[0], [0.45, 0.9, 1.27, 1.6])
    np.testing.assert_allclose(control.axis_coordinates[1], greville_z)
    np.testing.assert_allclose(control.coordinates[1], [0.9, 0.0])
    np.testing.assert_allclose(control.coordinates[4], [0.45, 0.01])
    axis_x, axis_z = control.axis_controls
    assert isinstance(axis_z, BSplineControl)
    assert (axis_z.axis, axis_z.degree, axis_z.size) == ("z", 3, 8)
    assert axis_z.coordinate_system == "seabed_depth"
    np.testing.assert_array_equal(axis_x.coefficients, np.zeros(4))
    replaced = control.with_coefficients(np.ones(32))
    np.testing.assert_array_equal(replaced.grid, np.ones((4, 8)))


def test_tensor_bspline_from_nodes_builds_open_knots_on_breakpoints():
    control = TensorBSplineControl.from_nodes(
        ["x", "z"],
        {"z": np.linspace(0.0, 0.15, 6), "x": [0.45, 0.9, 1.27, 1.6]},
        degree=[1, 3],
        coordinate_system="seabed_depth",
        extrapolate="hold",
    )
    np.testing.assert_allclose(control.knots[0], _LAND_X)
    np.testing.assert_allclose(control.knots[1], _LAND_Z)
    assert control.degree == (1, 3)
    assert control.shape == (4, 8)
    np.testing.assert_array_equal(control.coefficients, np.zeros(32))
    assert control.extrapolate == "hold"
    # With open knots the end Greville abscissae are the end breakpoints.
    assert control.axis_coordinates[1][0] == 0.0
    np.testing.assert_allclose(control.axis_coordinates[1][-1], 0.15)
    # One degree for every axis; degree one is the hat basis on the nodes.
    hats = TensorBSplineControl.from_nodes(
        ["x", "y", "z"], [[0.0, 1.0], [0.0, 2.0, 3.0], [1.0, 2.0]]
    )
    assert hats.degree == (1, 1, 1) and hats.shape == (2, 3, 2)
    np.testing.assert_allclose(hats.axis_coordinates[1], [0.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"axes": ["x"]}, "two or three axes"),
        ({"axes": ["x", "x"]}, "distinct"),
        ({"axes": "xz"}, "sequence of axis names"),
        ({"degree": [0, 3]}, "between 1 and 7"),
        ({"degree": [1, 8]}, "between 1 and 7"),
        ({"degree": [1.0, 3.0]}, "between 1 and 7"),
        ({"degree": [1, 3, 2]}, "one integer or one per axis"),
        ({"knots": [_LAND_X]}, "one knot vector per axis"),
        ({"knots": [_LAND_X, _LAND_Z[::-1]]}, "axis 'z'.*nondecreasing"),
        (
            {"knots": [[0.45, 0.45, 0.45, 0.9, 1.6, 1.6], _LAND_Z]},
            "axis 'x'.*multiplicity",
        ),
        (
            {"knots": [[0.0, 0.0, 1.0], _LAND_Z], "coefficients": np.zeros(8)},
            r"degree \+ 1 controls",
        ),
        ({"coefficients": np.zeros(31)}, "coefficient count 31"),
        ({"coefficients": np.zeros((8, 4))}, "does not match the coefficient shape"),
        ({"coefficients": np.zeros(32, dtype=complex)}, "real-valued"),
        ({"coefficients": np.full(32, np.nan)}, "finite"),
        ({"units": " "}, "units cannot be empty"),
    ],
)
def test_tensor_bspline_control_rejects_inconsistent_bases(overrides, message):
    with pytest.raises(ValueError, match=message):
        _tensor_bspline(**overrides)


@pytest.mark.parametrize(
    "nodes, message",
    [
        ([[0.0, 1.0], [0.0]], "at least two values"),
        ([[0.0, 1.0], [0.0, 1.0, 1.0]], "strictly increasing"),
        ({"x": [0.0, 1.0], "y": [0.0, 1.0]}, "exactly the axes"),
        ([[0.0, 1.0]], "breakpoints per axis"),
    ],
)
def test_tensor_bspline_from_nodes_rejects_invalid_breakpoints(nodes, message):
    with pytest.raises(ValueError, match=message):
        TensorBSplineControl.from_nodes(["x", "z"], nodes, degree=2)


def _random_points(seed, low, high, count=200):
    rng = np.random.default_rng(seed)
    return {
        axis: rng.uniform(lo, hi, count) for axis, lo, hi in zip(("x", "z"), low, high)
    }


@pytest.mark.parametrize("extrapolate", ["zero", "hold"])
def test_degree_one_tensor_bspline_samples_like_the_tensor_hat(extrapolate):
    from frequensolve.model.representation import EvaluationContext

    hat = TensorHatControl(
        axes=["x", "z"],
        shape=[4, 3],
        origin=[0.5, 1.0],
        spacing=[0.25, 2.0],
        coefficients=np.random.default_rng(3).normal(size=12),
        extrapolate=extrapolate,
    )
    spline = TensorBSplineControl.from_nodes(
        ["x", "z"],
        hat.axis_coordinates,
        degree=1,
        coefficients=hat.coefficients,
        extrapolate=extrapolate,
    )
    context = EvaluationContext(_random_points(5, (0.0, -1.0), (1.75, 6.0)))
    reference = Property.from_value(2.0)
    values = [
        ParameterizedProperty(reference, id="m", control=c).apply_control(2.0, context)
        for c in (hat, spline)
    ]
    np.testing.assert_allclose(values[1], values[0], rtol=0.0, atol=1e-14)
    assert np.any(values[0] != 2.0)


def test_tensor_bspline_reproduces_bilinear_fields_from_greville_coefficients():
    from frequensolve.model.representation import EvaluationContext

    control = _tensor_bspline(coordinate_system="global")
    nodes = control.coordinates
    control = control.with_coefficients(
        (1.0 + 2.0 * nodes[:, 0]) * (1.0 + 3.0 * nodes[:, 1])
    )
    points = _random_points(7, (0.45, 0.0), (1.6, 0.15))
    prop = ParameterizedProperty(0.0, id="m", control=control)
    values = prop.apply_control(0.0, EvaluationContext(points))
    np.testing.assert_allclose(
        values, (1.0 + 2.0 * points["x"]) * (1.0 + 3.0 * points["z"]), atol=1e-12
    )


def test_tensor_bspline_holds_or_zeroes_outside_its_knot_spans():
    from frequensolve.model.representation import EvaluationContext

    nodes = _tensor_bspline().coordinates
    coefficients = (1.0 + nodes[:, 0]) * (2.0 + nodes[:, 1])
    held = ParameterizedProperty(
        0.0,
        id="held",
        control=_tensor_bspline(
            coordinate_system="global", coefficients=coefficients, extrapolate="hold"
        ),
    )
    zero = ParameterizedProperty(
        0.0,
        id="zero",
        control=_tensor_bspline(coordinate_system="global", coefficients=coefficients),
    )
    outside = EvaluationContext({"x": [0.0, 2.0, 1.0, 0.0], "z": [0.05, 0.05, 0.4, -1]})
    clamped = EvaluationContext(
        {"x": [0.45, 1.6, 1.0, 0.45], "z": [0.05, 0.05, 0.15, 0.0]}
    )
    np.testing.assert_allclose(
        held.apply_control(0.0, outside), held.apply_control(0.0, clamped)
    )
    np.testing.assert_allclose(
        held.apply_control(0.0, clamped),
        (1.0 + np.array([0.45, 1.6, 1.0, 0.45]))
        * (2.0 + np.array([0.05, 0.05, 0.15, 0])),
    )
    np.testing.assert_array_equal(zero.apply_control(0.0, outside), 0.0)


def test_tensor_bspline_parameterized_property_matches_sauce_contract():
    prop = ParameterizedProperty(
        {"value": 0.4, "units": "km/s"},
        id="land_vs",
        transform="log",
        control=_tensor_bspline(units="km", extrapolate="hold"),
    )

    payload = prop.to_fs()
    _material_validator("#/$defs/property").validate(payload)
    loaded = Property.from_value(payload)
    assert isinstance(loaded, ParameterizedProperty)
    assert isinstance(loaded.control, TensorBSplineControl)
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


@pytest.mark.parametrize(
    "transform,reference,update,expected",
    [
        ("identity", 1900.0, 20.0, 1920.0),
        ("log", 1900.0, np.log(2.0), 3800.0),
        ("inverse", 2.0, 0.5, 1.0),
        ("logit", 0.25, np.log(3.0), 0.5),
    ],
)
def test_parameterized_sampling_composes_reference_and_control(
    transform, reference, update, expected
):
    import xarray as xr

    grid = xr.DataArray(np.zeros(5), dims=["z"], coords={"z": [-1, 0, 0.5, 1, 2]})
    prop = ParameterizedProperty(
        reference,
        id="vp",
        transform=transform,
        control=HatControl("z", 1.0, [update, update]),
    )
    values = prop.get(grid)
    np.testing.assert_allclose(
        values, [reference, expected, expected, expected, reference]
    )


def test_layered_sampling_parameterized_reference_and_length_units():
    from frequensolve.model import LayeredModel

    model = LayeredModel(dimension=2, x_limits=[0.0, 0.01])
    model.add_surface(0.0, name="top")
    prop = ParameterizedProperty(
        {"value": 1900.0, "units": "m/s"},
        id="vp",
        transform="log",
        control=HatControl("z", 10.0, [0.0, np.log(2.0)], units="m"),
    )
    model.add_layer(name="sediment", properties={"vp": prop, "rho": 2000.0})
    model.add_surface(0.01, name="bottom")
    # Geometry axes are explicitly supplied in km; the control is in metres.
    sampled = model.sample_uniform(
        [3, 3], axes_units={"x": "km", "z": "km"}, properties=["vp"]
    )
    np.testing.assert_allclose(sampled["vp"].isel(x=1), [1900, 1900 * np.sqrt(2), 3800])


def test_tensor_parameterized_sampling_preserves_first_axis_fastest_storage():
    import xarray as xr

    from frequensolve.model.parameterization import TensorHatControl

    grid = xr.DataArray(
        np.zeros((3, 3)), dims=["z", "x"], coords={"z": [0, 0.5, 1], "x": [0, 0.5, 1]}
    )
    prop = ParameterizedProperty(
        10.0,
        id="vp",
        control=TensorHatControl(
            axes=["x", "z"],
            shape=[2, 2],
            origin=[0, 0],
            spacing=[1, 1],
            coefficients=[1, 2, 3, 4],
        ),
    )
    np.testing.assert_allclose(
        prop.get(grid), [[11, 11.5, 12], [12, 12.5, 13], [13, 13.5, 14]]
    )


def test_bspline_sampling_and_mesh_materialization_fail_explicitly():
    import xarray as xr

    from frequensolve.model.parameterization import MeshControl

    grid = xr.DataArray(np.zeros(3), dims=["z"], coords={"z": [0, 0.5, 1]})
    prop = ParameterizedProperty(
        1900.0,
        id="vp",
        transform="log",
        control=BSplineControl(
            axis="z", knots=[0, 0, 1, 1], degree=1, coefficients=[0, np.log(2)]
        ),
    )
    np.testing.assert_allclose(prop.get(grid), [1900, 1900 * np.sqrt(2), 3800])
    with pytest.raises(ValueError, match="explicit grid"):
        prop.get()
    mesh = ParameterizedProperty(1900.0, id="vp", control=MeshControl("vp_space"))
    with pytest.raises(NotImplementedError, match="solver-backed"):
        mesh.get(grid)


def test_layered_sampling_applies_control_in_named_surface_frame(tmp_path):
    from frequensolve.geometry.frame import Axis, SurfaceCoordinateSystem
    from frequensolve.simulation.simulation import SeismicSimulation
    from frequensolve.units import ureg as u

    simulation = SeismicSimulation(
        name="sample", physics="acoustic", dimension=2, project_path=tmp_path
    )
    simulation += SurfaceCoordinateSystem(
        name="below_top",
        surface="top",
        axes=[Axis("below", direction="z", positive="down")],
    )
    model = LayeredModel(dimension=2, x_limits=[0, 1] * u.km)
    model.add_surface(name="top", depth=0.5 * u.km)
    model.add_layer(
        name="rock",
        properties={
            "Vp": ParameterizedProperty(
                2.0,
                id="vp",
                units="km/s",
                transform="log",
                control=HatControl(
                    coordinate_system="below_top",
                    axis="below",
                    units="m",
                    spacing=500,
                    coefficients=[0, np.log(2), 0],
                ),
            )
        },
    )
    model.add_surface(name="bottom", depth=1.5 * u.km)
    simulation += model
    sampled = model.sample_uniform(
        [2, 5], axes_units={"x": "km", "z": "km"}, properties="Vp"
    )
    np.testing.assert_allclose(
        sampled["vp"].isel(x=0), [2, 2 * np.sqrt(2), 4, 2 * np.sqrt(2), 2]
    )
    assert sampled["vp"].attrs["units"] == "km/s"


def test_layered_sampling_applies_a_tensor_bspline_in_a_named_surface_frame(
    tmp_path,
):
    from frequensolve.geometry.frame import Axis, SurfaceCoordinateSystem
    from frequensolve.simulation.simulation import SeismicSimulation
    from frequensolve.units import ureg as u

    simulation = SeismicSimulation(
        name="sample", physics="acoustic", dimension=2, project_path=tmp_path
    )
    # ``x`` is inherited from the global frame; ``below`` is depth below "top".
    simulation += SurfaceCoordinateSystem(
        name="below_top",
        surface="top",
        axes=[Axis("below", direction="z", positive="down")],
    )
    control = TensorBSplineControl.from_nodes(
        ["x", "below"],
        [[0.0, 1000.0], [0.0, 500.0, 1000.0]],
        degree=[1, 2],
        coordinate_system="below_top",
        units="m",
    )
    nodes = control.coordinates
    control = control.with_coefficients(np.log(2.0) * (nodes[:, 0] + nodes[:, 1]) / 1e3)
    model = LayeredModel(dimension=2, x_limits=[0, 1] * u.km)
    model.add_surface(name="top", depth=0.5 * u.km)
    model.add_layer(
        name="rock",
        properties={
            "Vp": ParameterizedProperty(
                2.0, id="vp", units="km/s", transform="log", control=control
            )
        },
    )
    model.add_surface(name="bottom", depth=1.5 * u.km)
    simulation += model
    sampled = model.sample_uniform(
        [3, 5], axes_units={"x": "km", "z": "km"}, properties="Vp"
    )["vp"]
    x = sampled.coords["x"].broadcast_like(sampled).values
    below = sampled.coords["z"].broadcast_like(sampled).values - 0.5
    np.testing.assert_allclose(sampled.values, 2.0 * 2.0 ** (x + below))
