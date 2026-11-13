# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

import json
from copy import deepcopy

import h5py
import numpy as np
import pytest

from frequensolve.imaging import Misfit
from frequensolve.imaging._patch_objective import (
    patch_objective_keys,
    restrict_patch_misfit,
)
from frequensolve.imaging.data import ObservedGroup, file_sha256
from frequensolve.imaging.jobs import FWIOperatorJob
from tests.test_imaging_jobs import _assert_valid
from tests.test_patch_sets import _simulation


def _parent_state(root, *, ranks=1, reduction="weighted_mean", keys=None):
    root.mkdir()
    terms = [
        {
            "id": name,
            "receiver_group": name,
            "objective": {"kind": "l2"},
            "comparison": {"kind": "waveform"},
            "weight": 2.0,
            "reduction": reduction,
            "effective_weight_mass": 6.0 if reduction == "weighted_mean" else 1.0,
            "scale": {
                "policy": "observed_rms",
                "components": ["p", "vz"],
                "values": [2.0, 3.0],
                "units": ["Pa", "m/s"],
            },
        }
        for name in ("near", "far")
    ]
    shards = []
    for rank, rows in enumerate(np.array_split(np.arange(6), ranks)):
        cache = root / f"cache_{rank}.h5"
        with h5py.File(cache, "w") as h5:
            for term in terms:
                group = h5.create_group(term["id"])
                group["row_ids"] = rows + 1
                group["n_global_rows"] = 6
                group["coordinate_keys"] = (
                    np.array([[row // 2 + 1, 7, row % 2 + 1] for row in rows]).reshape(
                        -1, 3
                    )
                    if keys is None
                    else keys[rows]
                )
        local = deepcopy(terms)
        for term in local:
            term["cache"] = {"file": cache.name, "group": term["id"]}
            term["runtime"] = {"cache_fingerprint": file_sha256(cache)}
        shard = root / f"shard_{rank}.json"
        shard.write_text(json.dumps({"terms": local}))
        shards.append({"file": shard.name, "sha256": file_sha256(shard)})
    manifest = root / "state.json"
    manifest.write_text(json.dumps({"partition": {"n_ranks": ranks}, "shards": shards}))
    payload = Misfit(normalization="observed_rms", weights=2.0).to_fs(
        [ObservedGroup(name, observed=root / f"{name}.h5") for name in ("near", "far")],
    )
    return manifest, payload


@pytest.mark.parametrize("ranks", [1, 2, 3])
@pytest.mark.parametrize("reduction", ["weighted_mean", "sum"])
def test_restriction_preserves_parent_row_value_gradient_and_normal(
    tmp_path, ranks, reduction
):
    state, payload = _parent_state(
        tmp_path / "parent", ranks=ranks, reduction=reduction
    )
    for term in payload["objective_terms"]:
        term["normalization"]["reduction"] = reduction
    original = deepcopy(payload)
    frozen = restrict_patch_misfit(payload, state, receiver_groups=["near"])
    assert payload == original
    assert [term["id"] for term in frozen["objective_terms"]] == ["near"]
    assert frozen["receiver_groups"] == [original["receiver_groups"][0]]
    term = frozen["objective_terms"][0]
    assert term["normalization"]["reduction"] == "sum"
    assert term["normalization"]["scale"]["kind"] == "explicit"
    np.testing.assert_allclose(
        term["weight"], 2.0 / (6 if reduction == "weighted_mean" else 1)
    )

    # Noncontiguous rows from the original catalogs, with component scales.
    selected = [0, 3, 5]
    residual = np.array([0.5, -2, 5, 1, -7, 3], dtype=float)
    jacobian = np.arange(18, dtype=float).reshape(6, 3) / 10
    scales = np.tile([2.0, 3.0], 3)
    parent_metric = 2.0 / (6 if reduction == "weighted_mean" else 1) / scales**2
    child_metric = np.array(
        [
            term["weight"]
            / term["normalization"]["scale"]["components"][name]["value"] ** 2
            for name in ("p", "vz", "vz")
        ]
    )
    np.testing.assert_allclose(child_metric, parent_metric[selected])
    parent_j = jacobian[selected]
    expected_value = 0.5 * np.dot(parent_metric[selected], residual[selected] ** 2)
    assert 0.5 * np.dot(child_metric, residual[selected] ** 2) == pytest.approx(
        expected_value
    )
    np.testing.assert_allclose(
        parent_j.T @ (child_metric * residual[selected]),
        parent_j.T @ (parent_metric[selected] * residual[selected]),
    )
    np.testing.assert_allclose(
        parent_j.T @ (child_metric[:, None] * parent_j),
        parent_j.T @ (parent_metric[selected, None] * parent_j),
    )

    simulation = _simulation(tmp_path / "project")
    simulation.save()
    job = FWIOperatorJob(
        "restricted",
        simulation,
        [3],
        action="linearize",
        active=[],
        state="state.json",
        misfit=frozen,
    )
    _assert_valid(job.to_fs())
    # The public misfit parser also accepts the frozen component scales.
    Misfit.from_fs(frozen)


@pytest.mark.parametrize(
    "change", ["preprocess", "projection", "comparison", "weight", "missing"]
)
def test_restriction_rejects_unsupported_or_changed_parent_context(tmp_path, change):
    state, payload = _parent_state(tmp_path / "parent")
    if change == "preprocess":
        payload["preprocess"]["include_defaults"] = True
    elif change == "projection":
        payload["receiver_groups"][0]["projection"] = {"kind": "up_down"}
    elif change == "comparison":
        payload["objective_terms"][0]["comparison"]["kind"] = "phase_derivative"
    elif change == "weight":
        payload["objective_terms"][0]["weight"] = 3
    else:
        payload["objective_terms"][0]["id"] = "absent"
    with pytest.raises(ValueError):
        restrict_patch_misfit(payload, state, receiver_groups=["near"])


def test_restriction_verifies_saved_parent_bytes_and_rank_normalization(tmp_path):
    state, payload = _parent_state(tmp_path / "parent", ranks=2)
    shard = state.parent / "shard_1.json"
    config = json.loads(shard.read_text())
    config["terms"][0]["effective_weight_mass"] = 3
    shard.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="shard hash"):
        restrict_patch_misfit(payload, state, receiver_groups=["near"])
    manifest = json.loads(state.read_text())
    manifest["shards"][1]["sha256"] = file_sha256(shard)
    state.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disagrees between ranks"):
        restrict_patch_misfit(payload, state, receiver_groups=["near"])


def _row_fixture(tmp_path):
    keys = np.array(
        [[source, 17 + i, i % 2 + 1] for i, source in enumerate((1, 1, 3, 3, 8, 8))]
    )
    state, _ = _parent_state(tmp_path / "rows", ranks=3, keys=keys)
    traces = [
        dict(
            trace_id=int(trace),
            source_id=int(source),
            component=int(component),
            receiver_id=5 + i % 2,
            receiver_position_id=5 + i % 2,
            point_first=5 + i % 2,
            point_last=5 + i % 2,
            active=True,
        )
        for i, (source, trace, component) in enumerate(keys)
    ]
    acquisition = dict(
        receiver_groups=[
            dict(name=name, sampling={"_type": "Sparse", "survey": "selected"})
            for name in ("near", "far")
        ],
        surveys=[dict(name="selected", _type="Sparse", traces=traces)],
    )
    return state, acquisition, keys


def test_patch_objective_rows_preserve_parent_receiver_identity(tmp_path):
    state, acquisition, keys = _row_fixture(tmp_path)
    original = deepcopy(acquisition)
    result = patch_objective_keys(state, acquisition)
    expected = keys.copy()
    expected[:, 1] = [5, 6, 5, 6, 5, 6]
    for canonical in result.values():
        np.testing.assert_array_equal(canonical, expected)
    assert acquisition == original
    for group in acquisition["receiver_groups"]:
        group.pop("sampling")
    for canonical in patch_objective_keys(state, acquisition).values():
        np.testing.assert_array_equal(canonical, keys)


@pytest.mark.parametrize(
    "change", ["missing", "source", "component", "point", "duplicate_id"]
)
def test_patch_objective_rows_reject_inconsistent_catalogs(tmp_path, change):
    state, acquisition, keys = _row_fixture(tmp_path)
    traces = acquisition["surveys"][0]["traces"]
    if change == "missing":
        traces.pop()
    elif change == "source":
        traces[0]["source_id"] = 2
    elif change == "component":
        traces[0]["component"] = 2
    elif change == "point":
        traces[0]["point_last"] += 1
    elif change == "duplicate_id":
        traces[1]["trace_id"] = traces[0]["trace_id"]
    with pytest.raises(ValueError):
        patch_objective_keys(state, acquisition)


def test_patch_objective_rows_reject_aliases_of_original_observations(tmp_path):
    _, acquisition, keys = _row_fixture(tmp_path)
    keys[2, 0] = 1
    state, _ = _parent_state(tmp_path / "aliased", keys=keys)
    acquisition["surveys"][0]["traces"][2]["source_id"] = 1
    with pytest.raises(ValueError, match="repeats an original observation"):
        patch_objective_keys(state, acquisition)
