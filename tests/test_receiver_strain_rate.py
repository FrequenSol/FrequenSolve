"""Native strain-rate receiver names survive export and reload."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import frequensolve as fs


@pytest.mark.parametrize("prefix", ["", "elastic:"])
@pytest.mark.parametrize("suffix", ["", "_all", "_xx", "_zz", "_xz", "_rr"])
@pytest.mark.parametrize("spelling", ["strain_rate", "strain-rate"])
def test_strain_rate_export_and_roundtrip(prefix, suffix, spelling):
    field = prefix + "strain_rate" + suffix
    component = fs.ReceiverComponent(name="rate", field=prefix + spelling + suffix)
    payload = component.to_fs()
    assert payload == {
        "name": "rate",
        "field": field,
    }
    restored = fs.ReceiverComponent.from_fs(payload)
    assert restored.field == field
    assert restored.to_fs() == payload


def test_legacy_sdk_strain_rate_payload_migrates_without_double_derivative():
    restored = fs.ReceiverComponent.from_fs(
        {"name": "rate", "field": "strain", "time_derivative": 1, "units": "1/s"}
    )
    assert restored.to_fs() == {"name": "rate", "field": "strain_rate", "units": "1/s"}


@pytest.mark.parametrize(
    "field", ["strain", "elastic:strain_xx", "p_strain_rate", "s_strain_rate"]
)
def test_other_receiver_fields_preserve_their_semantics(field):
    component = fs.ReceiverComponent(name="measurement", field=field)
    assert component.to_fs() == {"name": "measurement", "field": field}


def test_strain_rate_preserves_component_options():
    component = fs.ReceiverComponent(
        name="rate",
        field="strain_rate",
        units="1/ms",
        direction=[1.0, 0.0],
        weight=2.0,
    )
    payload = component.to_fs()
    assert payload["units"] == "1/ms"
    assert "direction" in payload
    assert "weight" in payload
    assert fs.ReceiverComponent.from_fs(payload).to_fs() == payload


def test_fiber_strain_and_rate_share_geometry_and_validate_with_sauce():
    fiber = fs.ReceiverFiber(
        gauge_length=23.93 * fs.ureg.m,
        points_per_gauge=129,
        coordinate_mode="gauge_centers",
    )
    fiber.add_component(name="strain", field="strain")
    fiber.add_component(name="rate", field="strain_rate")
    payload = fiber.to_fs()
    assert "direction" not in payload["components"][1]
    assert payload["points_per_gauge"] == 129
    assert fs.ReceiverFiber.from_fs(payload).to_fs() == payload

    root = Path(__file__).parent / "contracts/sauce-f533e6f/trunk/contracts"
    registry = Registry()
    for file in root.rglob("*.json"):
        schema = json.loads(file.read_text())
        if "$id" in schema:
            registry = registry.with_resource(
                schema["$id"], Resource.from_contents(schema)
            )
    schema = json.loads((root / "inputs/fs-acquisition-1/schema.json").read_text())
    validator = Draft202012Validator(
        {"$ref": schema["$id"] + "#/$defs/receiverComponent"}, registry=registry
    )
    for component in payload["components"]:
        validator.validate(component)
