import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

from frequensolve import SeismicSimulation
from frequensolve.simulation.outputs import outputs

CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-a54bdda" / "trunk" / "contracts"
)
SIMULATION_SCHEMA = CONTRACT_ROOT / "inputs" / "fs-simulation-1" / "schema.json"


def _sauce_simulation_validator() -> Draft202012Validator:
    registry = Registry()
    for schema_file in CONTRACT_ROOT.rglob("*.json"):
        contents = json.loads(schema_file.read_text())
        resource = Resource.from_contents(contents)
        registry = registry.with_resource(contents["$id"], resource)
    schema = json.loads(SIMULATION_SCHEMA.read_text())
    return Draft202012Validator(schema, registry=registry)


def test_electromagnetic_export_matches_pinned_sauce_contract(tmp_path):
    simulation = SeismicSimulation(
        name="em_contract",
        physics="EM",
        dimension=3,
        project_path=tmp_path,
    )

    payload = simulation.to_fs()
    payload["Outputs"] = outputs(units={"geometry": "m"}).to_fs()

    _sauce_simulation_validator().validate(payload)
    assert payload["schema"] == "fs-simulation-1"
    assert payload["physics"] == "em"
    assert payload["Outputs"]["Units"]["geometry"] == "m"


@pytest.mark.parametrize("kind", ["hat", "bspline"])
def test_parameterized_property_matches_adopted_material_contract(kind):
    from frequensolve.model.parameterization import (
        BSplineControl,
        HatControl,
        ParameterizedProperty,
    )
    from frequensolve.model.property import Property

    control = (
        HatControl(axis="z", spacing=1.0, coefficients=[0.0, 1.0])
        if kind == "hat"
        else BSplineControl(
            axis="z", degree=1, knots=[0.0, 0.0, 1.0, 1.0], coefficients=[0.0, 1.0]
        )
    )
    prop = ParameterizedProperty(Property(1500.0), id="vp", control=control)
    schema = json.loads(
        (CONTRACT_ROOT / "inputs/fs-material-model-1/schema.json").read_text()
    )
    validator = _sauce_simulation_validator().evolve(
        schema={"$ref": schema["$id"] + "#/$defs/property"}
    )
    payload = prop.to_fs()
    validator.validate(payload)
    validator.validate(ParameterizedProperty.from_fs(payload).to_fs())
    with pytest.raises(ValidationError):
        validator.validate({**payload, "value": 1500.0})
    payload["parameterized"]["control"]["kind"] = "unsupported"
    with pytest.raises(ValidationError):
        validator.validate(payload)
