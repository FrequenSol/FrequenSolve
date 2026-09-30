"""Behavior of typed physical expressions, receiver lowering, and output routing."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import frequensolve as fs
from frequensolve.imaging.misfit import Preprocess
from frequensolve.physics import Field, MaterialProperty
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import (
    ReceiverComponent,
    ReceiverGroup,
    ReceiverNode,
)
from frequensolve.util.mixins import ExportContext


@pytest.fixture
def validator():
    root = Path(__file__).parent / "contracts/sauce-f533e6f/trunk/contracts"
    registry = Registry()
    for path in root.rglob("*.json"):
        contents = json.loads(path.read_text())
        if "$id" in contents:
            registry = registry.with_resource(
                contents["$id"], Resource.from_contents(contents)
            )
    schema = json.loads((root / "inputs/fs-acquisition-1/schema.json").read_text())
    return Draft202012Validator(
        {"$ref": schema["$id"] + "#/$defs/receiverGroup"}, registry=registry
    )


@pytest.mark.parametrize(
    "factory,field,material",
    [
        (fs.physics.acoustic, "pressure", "rho"),
        (fs.physics.elastic, "strain", "compliance"),
        (fs.physics.poroelastic, "fluid_flux", "porosity"),
        (fs.physics.electromagnetic, "electric", "conductivity"),
    ],
)
def test_namespaces_are_discoverable_typed_references(factory, field, material):
    namespace = factory()
    assert isinstance(getattr(namespace.fields, field), Field)
    assert isinstance(getattr(namespace.materials, material), MaterialProperty)
    assert field in dir(namespace.fields)
    with pytest.raises(AttributeError, match="Unknown quantity"):
        namespace.materials.missing


@pytest.mark.parametrize(
    "dimension,labels", [(2, ["x", "z"]), (3, ["x", "y", "z"]), (2.5, ["x", "y", "z"])]
)
def test_momentum_units_and_late_dimension_binding(dimension, labels, validator):
    elastic = fs.physics.elastic()
    acquisition = Acquisition()
    group = acquisition.add_receiver_group(
        "momentum",
        elastic.materials.rho * elastic.fields.velocity,
        [[0, 1]],
        domain="solid",
    )
    payload = group.to_fs(ExportContext(dimension=dimension, physics="elastic"))
    assert [entry["name"] for entry in payload["device"]["components"]] == labels
    assert all(
        fs.ureg.Unit(entry["units"]).dimensionality
        == fs.ureg.Unit("kg/(m^2*s)").dimensionality
        for entry in payload["device"]["components"]
    )
    assert "material_samples" not in payload
    validator.validate(payload)
    assert ReceiverGroup.from_fs(payload).to_fs() == payload


def test_constitutive_action_and_physical_shear_selection(validator):
    elastic = fs.physics.elastic()
    strain = elastic.materials.compliance @ elastic.fields.stress
    assert strain.units == "1"
    assert strain.xz.selection == ("x", "z")
    assert strain["z", "x"]._native(2) == strain.xz._native(2)
    payload = ReceiverGroup(name="strain", device=strain, coordinates=[[0, 1]]).to_fs()
    entries = payload["device"]["components"]
    assert [entry["name"] for entry in entries] == ["xx", "zz", "xz"]
    assert entries[-1]["expression"]["factor"] == pytest.approx(2**-0.5)
    assert entries[0]["expression"]["child"]["coefficient"] == {
        "tensor": "compliance",
        "basis": "mandel",
    }
    assert "material_samples" not in payload
    validator.validate(payload)


def test_predefined_strain_and_composable_scalar_components():
    elastic = fs.physics.elastic()
    expression = elastic.fields.strain.xx + 2 * elastic.fields.strain.zz
    group = ReceiverGroup(name="combined", device=expression, coordinates=[[0, 0]])
    payload = group.to_fs()["device"]["components"][0]
    assert payload["expression"]["kind"] == "add"
    assert payload["expression"]["args"][0]["child"]["packing"] == "strain"
    assert expression.shape == ()
    assert elastic.fields.velocity.shape == (None,)
    assert fs.physics.elastic(dimension=3).fields.strain.shape == (3, 3)


def test_material_diagnostics_are_explicit_and_mandel(validator):
    elastic = fs.physics.elastic()
    group = ReceiverGroup(
        name="strain",
        device=elastic.fields.strain,
        coordinates=[[0, 0]],
        materials={"rho": elastic.materials.rho, "S": elastic.materials.compliance},
        domain=1,
    )
    payload = group.to_fs()
    assert [sample["name"] for sample in payload["material_samples"]] == ["rho", "S"]
    assert payload["material_samples"][1]["coefficient"]["basis"] == "mandel"
    validator.validate(payload)
    assert ReceiverGroup.from_fs(payload).to_fs() == payload


def test_material_sampling_and_multi_output_group(validator):
    acoustic = fs.physics.acoustic()
    group = ReceiverGroup(
        name="pv",
        device={
            "pressure": acoustic.fields.pressure,
            "momentum": acoustic.materials.rho * acoustic.fields.velocity,
        },
        coordinates=[[0, 0]],
    )
    payload = group.to_fs()
    assert [entry["name"] for entry in payload["device"]["components"]] == [
        "pressure",
        "momentum.x",
        "momentum.z",
    ]
    validator.validate(payload)
    with pytest.raises(ValueError, match="materials="):
        ReceiverGroup(
            name="density", device=acoustic.materials.rho, coordinates=[[0, 0]]
        )


def test_material_expression_constants_have_units_and_conversion_is_output_only():
    acoustic = fs.physics.acoustic()
    scale = acoustic.materials.rho / (1000 * fs.ureg.kg / fs.ureg.m**3)
    assert scale.coefficient()["expr"]["args"][1] == {"value": 1000, "units": "kg/m^3"}
    assert acoustic.fields.pressure.to("MPa")._native(
        2
    ) == acoustic.fields.pressure._native(2)
    assert (acoustic.materials.rho**-0.5).coefficient()["expr"]["op"] == "pow"
    with pytest.raises(ValueError, match="incompatible"):
        acoustic.fields.pressure.to("m/s")


def test_invalid_shapes_units_physics_and_nonlinearity_fail_at_authoring():
    elastic, acoustic = fs.physics.elastic(dimension=2), fs.physics.acoustic()
    for operation in [
        lambda: elastic.fields.velocity * elastic.fields.velocity,
        lambda: elastic.fields.velocity + elastic.fields.stress,
        lambda: elastic.fields.stress.xx + elastic.fields.velocity.x,
        lambda: elastic.fields.velocity + acoustic.fields.velocity,
        lambda: elastic.fields.strain.yy,
        lambda: elastic.fields.stress @ elastic.materials.compliance,
        lambda: elastic.fields.velocity / elastic.fields.velocity,
    ]:
        with pytest.raises(ValueError):
            operation()
    group = ReceiverGroup(
        name="pressure", device=acoustic.fields.pressure, coordinates=[[0, 0]]
    )
    with pytest.raises(ValueError, match="physics"):
        group.to_fs(ExportContext(physics="em", dimension=2))


def test_legacy_components_roundtrip_unchanged():
    group = ReceiverGroup(
        name="pressure",
        device=ReceiverNode(components=[ReceiverComponent(name="p", field="pressure")]),
        coordinates=[[0, 0]],
    )
    payload = group.to_fs()
    assert "expression" not in payload["device"]["components"][0]
    assert ReceiverGroup.from_fs(payload).to_fs() == payload


def test_material_weighting_uses_frozen_trace_pair_policy_and_physical_units():
    acoustic = fs.physics.acoustic()
    hook = Preprocess.material_weighting(
        {1: 1 / acoustic.materials.rho}, units={1: "m^2/s^2"}
    )
    payload = hook.to_fs()
    assert payload["stage"] == "trace_pair"
    assert payload["params"]["model_policy"] == "frozen"
    assert payload["params"]["blocks"][0]["coefficient"]["expr"]["op"] == "div"
    assert Preprocess.from_fs(payload).to_fs() == payload
    with pytest.raises(ValueError, match="overlap"):
        Preprocess.material_weighting(
            {1: acoustic.materials.rho, (1, 2): acoustic.materials.rho},
            units={1: "Pa", (1, 2): "Pa"},
        )


@pytest.mark.parametrize("domain", [True, 1.5, [], "", "   "])
def test_receiver_domain_rejects_invalid_selectors(domain):
    with pytest.raises((TypeError, ValueError)):
        ReceiverGroup(
            name="p",
            device=fs.physics.acoustic().fields.pressure,
            coordinates=[[0, 0]],
            domain=domain,
        )


def test_expression_only_component_roundtrip():
    payload = {
        "name": "p",
        "expression": {"kind": "field", "name": "acoustic:pressure"},
    }
    component = ReceiverComponent.from_fs(payload)
    assert component.to_fs()["expression"] == payload["expression"]


def test_weighting_tensor_block_is_defined_in_physical_components():
    elastic = fs.physics.elastic()
    hook = Preprocess.material_weighting(
        {(1, 2, 3): elastic.materials.compliance}, units={(1, 2, 3): "1"}
    )
    assert hook.to_fs()["params"]["blocks"][0]["coefficient"]["basis"] == "physical"


@pytest.mark.parametrize(
    "factory,quantity,native_name",
    [
        (fs.physics.acoustic, "velocity", "acoustic:velocity_all"),
        (fs.physics.poroelastic, "velocity", "poroelastic:solid_velocity_all"),
        (fs.physics.poroelastic, "fluid_flux", "poroelastic:fluid_flux_all"),
        (fs.physics.electromagnetic, "electric", "maxwell:electric_all"),
        (fs.physics.electromagnetic, "magnetic", "maxwell:magnetic_all"),
        (fs.physics.electromagnetic, "magnetic_induction", "magnetic_induction_all"),
    ],
)
def test_primary_fields_lower_to_registered_native_names(
    factory, quantity, native_name
):
    namespace = factory(dimension=2)
    field = getattr(namespace.fields, quantity)
    assert field._native(2)["name"] == native_name
    components = field.receiver_components(quantity, 2)
    expected = ["x", "y", "z"] if namespace.name == "em" else ["x", "z"]
    assert [component.name for component in components] == expected
