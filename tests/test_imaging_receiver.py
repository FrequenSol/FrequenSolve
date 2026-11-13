"""Receiver artifacts, band identity, and native job serialization."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.imaging._receiver import (
    ReceiverCollection,
    ReceiverState,
    receiver_band,
    sha256,
)
from tests.test_imaging_jobs import _assert_valid


def _state(root, label, frequency=1):
    path = root / f"{label}.json"
    payload = root / f"{label}.h5"
    with h5py.File(payload, "w") as h5:
        h5["keys"] = np.array([[1, 3, 1], [1, 1, 1], [1, 2, 1]], np.int64)
        for name in ["predicted", "predicted_df", "observed", "observed_df"]:
            h5[name] = np.column_stack(([3.0, 1.0, 2.0], np.zeros(3)))
    m = dict(
        schema="fs-receiver-state-1",
        frequency_hz=frequency,
        receiver_group="line",
        channel="base_df",
        n_rows=3,
        partition={"n_ranks": 1, "compatibility": "same_mesh_partition"},
        payload_precision="complex64",
        value_convention="physical_receiver",
        physics="acoustic",
        dimension=2,
        geometry="cartesian",
        modeling_mode="full_dimension",
        control_kind="material_only",
        source_mode="physical_shot",
        units="Pa",
        source_derivative="total",
        candidate_fingerprint="candidate",
        acquisition_fingerprint="acquisition",
        observation_fingerprint="observation",
        preprocessing_fingerprint="identity",
        control_registry_fingerprint=label,
        resolved_context_fingerprint=label,
        state_fingerprint=label,
        field_retention="checkpoint",
        checkpoints=[{"base_file": "base.h5", "df_file": "df.h5"}],
        shards=[dict(file=str(payload), rank=0, sha256=sha256(payload))],
    )
    path.write_text(json.dumps(m))
    return path


def test_band_matches_keys_and_preserves_distinct_registries(tmp_path):
    paths = [_state(tmp_path, str(i), f) for i, f in enumerate([1, 1.4, 2])]
    states = [ReceiverCollection.read(p) for p in paths]
    band = receiver_band(states, [1, 1.4, 2], "total")["line"]
    np.testing.assert_array_equal(band[:, 0, 0], [1, 2, 3])
    state = states[0].groups["line"]
    dual = state.write_dual(tmp_path / "dual.json", np.ones(3), 2j * np.ones(3))
    with h5py.File(json.loads(dual.read_text())["shards"][0]["file"]) as h5:
        np.testing.assert_array_equal(h5["keys"], state.keys)
        np.testing.assert_array_equal(h5["df"][:, 1], 2)
    changed = json.loads(paths[1].read_text())
    changed["candidate_fingerprint"] = "different"
    paths[1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity"):
        receiver_band(
            [states[0], ReceiverCollection.read(paths[1]), states[2]],
            [1, 1.4, 2],
            "total",
        )
    paths[0].write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        state.write_dual(tmp_path / "stale.json", np.ones(3), np.ones(3))


def test_duplicate_or_corrupt_shards_are_rejected(tmp_path):
    path = _state(tmp_path, "state")
    m = json.loads(path.read_text())
    payload = tmp_path / "state.h5"
    with h5py.File(payload, "r+") as h5:
        h5["keys"][1] = h5["keys"][0]
    with pytest.raises(ValueError, match="checksum"):
        ReceiverState.read(path)
    m["shards"][0]["sha256"] = sha256(payload)
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="duplicated"):
        ReceiverState.read(path)


@pytest.mark.parametrize(
    "action", ["receiver_linearize", "receiver_jvp", "receiver_vjp"]
)
def test_receiver_jobs_serialize_and_round_trip(tmp_path, action):
    from tests.test_imaging_jobs import _round_trip, _saved_simulation

    sim = _saved_simulation(tmp_path)
    options = dict(receiver_state="receiver.json", active=["model.vp"])
    if action == "receiver_linearize":
        options.update(
            channel="base_df", source_derivative="total", field_retention="checkpoint"
        )
    elif action == "receiver_jvp":
        options.update(receiver_vector="tangent.json", direction="direction.h5")
    else:
        options.update(
            receiver_vector="dual.json",
            covector="gradient.h5",
            field_reuse="checkpoint",
        )
    job = im.FWIOperatorJob("receiver", sim, [1.0, 2.0], action=action, **options)
    payload = _assert_valid(job.to_fs())
    assert "state" not in payload["fwi_operator"]
    _round_trip(job)


@pytest.mark.parametrize(
    "options",
    [
        dict(channel="base_df", source_derivative="none"),
        dict(channel="base_df", source_derivative="total", field_reuse="checkpoint"),
        dict(channel="base_df", source_derivative="total", active=["source.1.delay"]),
    ],
)
def test_receiver_jobs_reject_invalid_channels_and_controls(tmp_path, options):
    from tests.test_imaging_jobs import _saved_simulation

    with pytest.raises(ValueError):
        im.FWIOperatorJob(
            "receiver",
            _saved_simulation(tmp_path),
            [1, 2],
            action="receiver_linearize",
            **(dict(receiver_state="state.json", active=["model.vp"]) | options),
        )


def test_multiple_groups_bind_shared_fields_and_preserve_empty_rank_duals(tmp_path):
    members = []
    for index, name in enumerate(["surface", "deep"]):
        path = _state(tmp_path, name)
        payload = json.loads(path.read_text())
        payload.update(
            receiver_group=name,
            control_registry_fingerprint="shared",
            resolved_context_fingerprint="shared",
            partition={"n_ranks": 2, "compatibility": "same_mesh_partition"},
        )
        path.write_text(json.dumps(payload))
        members.append(dict(receiver_group=name, file=path.name, sha256=sha256(path)))
    bundle = tmp_path / "groups.json"
    manifest = dict(
        schema="fs-receiver-state-bundle-1", state_fingerprint="bundle", groups=members
    )
    bundle.write_text(json.dumps(manifest))
    collection = ReceiverCollection.read(bundle)
    values = {
        name: (np.arange(3) + index, 1j * np.ones(3))
        for index, name in enumerate(collection.groups)
    }
    dual = collection.write_dual(tmp_path / "dual.json", values)
    payload = json.loads(dual.read_text())
    assert payload["source_state_sha256"] == sha256(bundle)
    assert payload["state_fingerprint"] == "bundle"
    for entry in payload["groups"]:
        member = json.loads(Path(entry["file"]).read_text())
        assert entry["sha256"] == sha256(Path(entry["file"]))
        assert member["receiver_group"] == entry["receiver_group"]
        with h5py.File(member["shards"][1]["file"]) as h5:
            assert h5["keys"].shape == (0, 3)
            assert h5["base"].shape == (0, 2)
    with pytest.raises(ValueError, match="every group"):
        collection.write_dual(tmp_path / "missing.json", {"surface": values["surface"]})
    manifest["groups"] = [members[0], members[0]]
    bundle.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        ReceiverCollection.read(bundle)
    manifest["groups"] = members
    bundle.write_text(json.dumps(manifest))
    child = tmp_path / members[1]["file"]
    payload = json.loads(child.read_text())
    payload["candidate_fingerprint"] = "changed"
    child.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        ReceiverCollection.read(bundle)
    manifest["groups"][1]["sha256"] = sha256(child)
    bundle.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="share candidate"):
        ReceiverCollection.read(bundle)


def test_base_receiver_states_and_duals_do_not_require_df(tmp_path):
    path = _state(tmp_path, "base")
    payload = json.loads(path.read_text())
    payload["channel"] = "base"
    payload["source_derivative"] = "none"
    with h5py.File(payload["shards"][0]["file"], "r+") as h:
        del h["predicted_df"]
        del h["observed_df"]
    payload["shards"][0]["sha256"] = sha256(Path(payload["shards"][0]["file"]))
    path.write_text(json.dumps(payload))
    state = ReceiverState.read(path)
    np.testing.assert_array_equal(state.values[:, [1, 3]], 0)
    dual = state.write_dual(tmp_path / "dual.json", np.ones(3), np.zeros(3))
    m = json.loads(dual.read_text())
    assert m["channel"] == "base"
    with h5py.File(m["shards"][0]["file"]) as h:
        assert "base" in h and "df" not in h
    with pytest.raises(ValueError, match="cannot carry df"):
        state.write_dual(tmp_path / "invalid.json", np.ones(3), np.ones(3))
