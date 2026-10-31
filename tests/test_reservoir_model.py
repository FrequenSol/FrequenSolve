"""fs-reservoir-model-1 authoring, round trip, and contract validation."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from frequensolve.simulation.reservoir import (
    Alternation,
    CoreyStoneII,
    InitialState,
    InjectionStream,
    ReservoirComponent,
    ReservoirDomain,
    ReservoirFluid,
    ReservoirModel,
    ReservoirRock,
    ReservoirWater,
    ReservoirWell,
    SchedulePhase,
)

ROOT = Path(__file__).parent / "contracts/sauce-reservoir-model-1/contracts"
EXAMPLES = ROOT / "inputs/fs-reservoir-model-1/examples"


@pytest.fixture(scope="module")
def validator():
    registry = Registry()
    for file in ROOT.rglob("*.schema.json"):
        payload = json.loads(file.read_text())
        registry = registry.with_resource(
            payload["$id"], Resource.from_contents(payload)
        )
    for file in ROOT.rglob("schema.json"):
        payload = json.loads(file.read_text())
        registry = registry.with_resource(
            payload["$id"], Resource.from_contents(payload)
        )
    schema = json.loads((ROOT / "inputs/fs-reservoir-model-1/schema.json").read_text())
    return Draft202012Validator(schema, registry=registry)


def test_showcase_round_trips_exactly(validator):
    showcase = json.loads((EXAMPLES / "wag-channel-baffle.json").read_text())
    model = ReservoirModel.from_fs(showcase)
    assert model.to_fs() == showcase
    validator.validate(model.to_fs())
    assert model.end_time == 780
    assert [c.name for c in model.fluid.components] == ["methane", "butane", "decane"]


def test_minimal_example_reserializes_to_a_valid_document(validator):
    minimal = json.loads((EXAMPLES / "minimal-gas-flood.json").read_text())
    payload = ReservoirModel.from_fs(minimal).to_fs()
    validator.validate(payload)
    assert ReservoirModel.from_fs(payload).to_fs() == payload


def test_authored_model_is_contract_valid(validator):
    water = ReservoirWater(
        {"value": 10, "units": "MPa"}, 1000, 4e-10, {"value": 1, "units": "cP"}
    )
    model = ReservoirModel(
        temperature=350,
        fluid=ReservoirFluid(
            components=[
                ReservoirComponent(
                    "methane", 190.564, 4.5992e6, 0.01142, 0.016043, 9.86e-5
                ),
                ReservoirComponent("decane", 617.7, 2.103e6, 0.4884, 0.142286, 6.17e-4),
            ],
            water=water,
        ),
        rock=ReservoirRock(10e6, 0.2, 1e-10),
        relative_permeability=CoreyStoneII(0.2, 0.15, 0.1, 0.02),
        permeability={"value": 100, "units": "mD"},
        domain=ReservoirDomain(400, 40, (8, 2)),
        wells=[
            ReservoirWell("I1", "injector", [20, 20], [10, 10]),
            ReservoirWell("P1", "producer", [380, 20], [10, 10]),
        ],
        injection_streams=[
            InjectionStream("gas", "hydrocarbon", [0.99, 0.01]),
            InjectionStream("water", "water"),
        ],
        schedule=[
            SchedulePhase(300, 10, "gas", Alternation("water", 30, 30)),
            SchedulePhase(30, 0),
        ],
        initial_state=InitialState(20, 10e6, 750, 0.2, [0.3, 0.7]),
    )
    payload = model.to_fs()
    validator.validate(payload)
    assert payload["Model"]["subdomains"][0]["physics"] == "reservoir:compositional"
    assert ReservoirModel.from_fs(payload) == model


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda d: d["initial_state"].update(composition=[0.5, 0.5, 0.5]),
            "sum to one",
        ),
        (
            lambda d: d["injection_streams"][0].update(composition=[0.5, 0.5]),
            "one mole fraction per component",
        ),
        (
            lambda d: d["schedule"]["phases"][0].update(stream="steam"),
            "Unknown injection streams",
        ),
        (lambda d: d["wells"].pop(), "injector and one producer"),
        (lambda d: d["numerics"].update(amount_scale=[1, 2]), "amount_scale"),
        (
            lambda d: d["fluid"]["equation_of_state"].update(model="srk"),
            "peng_robinson",
        ),
    ],
)
def test_inconsistent_models_are_rejected(change, message):
    payload = copy.deepcopy(
        json.loads((EXAMPLES / "wag-channel-baffle.json").read_text())
    )
    change(payload)
    with pytest.raises(ValueError, match=message):
        ReservoirModel.from_fs(payload)


def test_unit_aware_schedule_and_component_exports(validator):
    from dataclasses import replace

    import frequensolve as fs

    model = ReservoirModel.from_fs(
        json.loads((EXAMPLES / "minimal-gas-flood.json").read_text())
    )
    phase = model.schedule[0]
    for duration in (fs.Q_(24, "hour"), {"value": 24, "units": "hour"}):
        changed = replace(model, schedule=(replace(phase, duration=duration),))
        assert changed.end_time == 1
        validator.validate(changed.to_fs())
    component = replace(
        model.fluid.components[0],
        critical_pressure=fs.Q_(4.6, "MPa"),
        molecular_weight={"value": 16.043, "units": "g/mol"},
    )
    changed = replace(
        model,
        fluid=replace(model.fluid, components=(component, *model.fluid.components[1:])),
    )
    payload = json.loads(json.dumps(changed.to_fs()))
    validator.validate(payload)
    assert payload["fluid"]["components"][0]["critical_pressure"] == 4.6e6
    assert payload["fluid"]["components"][0]["molecular_weight"] == pytest.approx(
        0.016043
    )
    assert fs.ReservoirModel is ReservoirModel
    with pytest.raises(ValueError, match="compatible"):
        replace(phase, duration=fs.Q_(1, "m"))
    with pytest.raises(ValueError, match="compatible"):
        replace(component, critical_pressure=fs.Q_(1, "s"))
