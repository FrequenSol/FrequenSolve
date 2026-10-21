"""Automatic sizing uses measured layouts, never aggregate/rank division."""

from copy import deepcopy

import pytest

from frequensolve.execution_plan import (
    PlanError,
    allocation,
    choose_allocation,
    digest,
    memory_request,
    revise_rows,
    totals,
    validate_request,
)


def shape(nodes=1, ranks=1, threads=8):
    return dict(
        partition="cpu",
        node_type="fake.arm",
        nodes=nodes,
        ranks_per_node=ranks,
        cpus_per_rank=threads,
    )


def observation(nodes=1, ranks=1, memory=None):
    return dict(
        version="frequency-layout-sizing.v1",
        layout=shape(nodes, ranks),
        complete=True,
        memory_scope="rank-local-resident-estimate",
        native_safety_applied=True,
        rank_memory_bytes=memory or [80 * 1024**3] * (nodes * ranks),
    )


def hardware():
    return dict(
        partition="cpu",
        node_type="fake.arm",
        schedulable_cpus=64,
        schedulable_memory_mib=120 * 1024,
        max_nodes=4,
        network_fabric="efa",
        qualified_layouts=[shape(1), shape(2), shape(2, 2), shape(4)],
        recommended_layouts=[shape(1), shape(2), shape(2, 2), shape(4)],
    )


def validate(request, evidence):
    return validate_request(
        request,
        evidence,
        hardware(),
        worker_overhead_mib=512,
        max_nodes=4,
        max_cpus=128,
        max_wall_time_seconds=3600,
    )


def test_worst_node_uses_largest_rank_estimates_and_keeps_replicated_memory():
    evidence = observation(
        2, 2, [10 * 1024**3, 30 * 1024**3, 20 * 1024**3, 40 * 1024**3]
    )
    assert (
        memory_request(evidence, shape(2, 2), worker_overhead_mib=512)
        == 70 * 1024 + 512
    )
    assert (
        memory_request(observation(2), shape(2), worker_overhead_mib=512)
        == 80 * 1024 + 512
    )


def test_frequencies_choose_different_nodes_when_serial_memory_cannot_fit():
    small = choose_allocation(
        [observation(), observation(2)],
        [hardware()],
        max_nodes=4,
        max_cpus=128,
        wall_time_seconds=900,
        worker_overhead_mib=512,
    )
    large = choose_allocation(
        [observation(memory=[180 * 1024**3]), observation(2)],
        [hardware()],
        max_nodes=4,
        max_cpus=128,
        wall_time_seconds=900,
        worker_overhead_mib=512,
    )
    assert small["requested"]["layout"]["nodes"] == 1
    assert large["requested"]["layout"]["nodes"] == 2
    assert large["estimated_runtime_seconds"] is None
    assert totals(large["requested"])["cpus"] == 16


def test_unknown_layout_or_incomplete_sizing_cannot_be_used():
    for change in (
        {"complete": False},
        {"memory_scope": "distributed-total"},
        {"rank_memory_bytes": [1]},
        {"native_safety_applied": False},
    ):
        with pytest.raises(PlanError):
            memory_request(
                {**observation(2), **change}, shape(2), worker_overhead_mib=512
            )


def test_reservation_reduction_cannot_pass_validation():
    request = dict(layout=shape(), memory_mib_per_node=1024, wall_time_seconds=900)
    with pytest.raises(PlanError, match="below the validated"):
        validate(request, observation())
    request["memory_mib_per_node"] = 90 * 1024
    assert validate(request, observation())["status"] == "valid"


def test_increased_ranks_need_new_evidence():
    request = dict(
        layout=shape(2, 2), memory_mib_per_node=110 * 1024, wall_time_seconds=900
    )
    with pytest.raises(PlanError, match="layout-specific"):
        validate(request, observation(2))


def test_edit_revisions_invalidate_only_changed_rows_and_reset_is_explicit():
    automatic = dict(
        layout=shape(), memory_mib_per_node=90 * 1024, wall_time_seconds=900
    )
    rows = [
        dict(
            frequency_index=i,
            requested=deepcopy(automatic),
            automatic=deepcopy(automatic),
            validation={"status": "valid"},
            overrides={},
        )
        for i in range(2)
    ]
    revised = revise_rows(rows, {0: {"nodes": 2, "mpi_ranks": 2}})
    assert revised[0]["validation"]["status"] == "needs_validation"
    assert revised[1]["validation"]["status"] == "valid"
    assert rows[0]["requested"]["layout"]["nodes"] == 1
    assert revise_rows(revised, {0: {"reset": True}})[0]["requested"] == automatic
    with pytest.raises(PlanError):
        revise_rows(rows, {0: {"nodes": 2, "mpi_ranks": 3}})
    with pytest.raises(PlanError):
        revise_rows(rows, {0: {"validation": {"status": "valid"}}})


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, 2**53])
def test_invalid_resource_numbers(value):
    with pytest.raises(PlanError):
        allocation(
            dict(layout=shape(value), memory_mib_per_node=1024, wall_time_seconds=900)
        )


def test_digest_rejects_float_and_is_order_independent():
    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})
    with pytest.raises(PlanError):
        digest({"frequency": 1.0})


def test_published_contract_fixture_and_incomplete_rows_validate_without_fake_allocations():
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator

    root = Path(__file__).resolve().parents[1] / "contracts/execution-plan/v1"
    schema = json.loads((root / "schema.json").read_text())
    example = json.loads((root / "example.json").read_text())
    validator = Draft202012Validator(schema)
    validator.validate(example)
    unsigned = {key: value for key, value in example.items() if key != "digest"}
    assert digest(unsigned) == example["digest"]
    example["rows"][0].update(
        automatic=None, requested=None, validation={"status": "needs_validation"}
    )
    example["digest"] = digest(
        {key: value for key, value in example.items() if key != "digest"}
    )
    validator.validate(example)
    with pytest.raises(PlanError, match="Extend planning"):
        revise_rows(example["rows"], {0: {"memory_mib_per_node": 1024}})


def test_automatic_selection_preserves_installed_memory_policy_and_thread_recommendation():
    node = hardware()
    alternate = {**shape(1), "cpus_per_rank": 1}
    node["qualified_layouts"].append(alternate)
    observations = [
        observation(1, memory=[10 * 1024**3]),
        {**observation(1, memory=[10 * 1024**3]), "layout": alternate},
    ]
    selected = choose_allocation(
        observations,
        [node],
        max_nodes=2,
        max_cpus=64,
        wall_time_seconds=3600,
        worker_overhead_mib=603,
        memory_policy={
            "calibration_numerator": 118,
            "calibration_denominator": 100,
            "worker_overhead_mib": 603,
            "minimum_memory_mib": 0,
        },
    )
    assert selected["requested"]["layout"]["cpus_per_rank"] == 8
    assert (
        selected["requested"]["memory_mib_per_node"]
        == (10 * 1024 * 118 + 99) // 100 + 603
    )


def test_shared_private_protocol_digest_matches_cloud_and_controller():
    import json
    from pathlib import Path

    fixture = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "contracts/execution-plan/v1/planning-protocol.json"
        ).read_text()
    )
    assert (
        digest(
            {
                "command": fixture["command"],
                "requests": fixture["command"]["evaluations"],
            }
        )
        == "sha256:" + fixture["submission_digest"]
    )
