import json
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from frequensolve import SeismicSimulation

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

    _sauce_simulation_validator().validate(payload)
    assert payload["schema"] == "fs-simulation-1"
    assert payload["physics"] == "em"
