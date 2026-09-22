import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr
from jsonschema import Draft202012Validator

from frequensolve.imaging.data import (
    DataSpace,
    DataVector,
    ObservedData,
    ObservedGroup,
    TermLayout,
    TraceStoreRef,
    canonical_json_sha256,
    file_sha256,
    objective_layout_fingerprint,
)
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverComponent, ReceiverNode
from frequensolve.simulation.jobs import FrequencyDomainJob
from frequensolve.simulation.simulation import SeismicSimulation

CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-f533e6f" / "trunk" / "contracts"
)
OBJECTIVE_VECTOR_SCHEMA = (
    CONTRACT_ROOT / "outputs" / "fs-objective-vector-3" / "schema.json"
)
STATE = "sha256:" + "ab" * 32


def _elastic_simulation(tmp_path, *, groups=("surface",)):
    sim = SeismicSimulation(
        name="observed", physics="elastic", dimension=2, project_path=tmp_path
    )
    sim.model.x_limits = [0.0, 1.0]
    sim.model.z_limits = [0.0, 1.0]
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[1, 1])
    )
    acq = Acquisition()
    acq.add_sources(
        kind="vector", coords=np.array([[0.5, 0.1], [0.6, 0.1]]), direction=[0.0, 1.0]
    )
    for name in groups:
        device = ReceiverNode(
            name=f"{name}_device",
            components=[
                ReceiverComponent(name="vx", field="velocity"),
                ReceiverComponent(name="vz", field="velocity"),
            ],
        )
        acq.add_receiver_group(
            name=name,
            device=device,
            coords=np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        )
    sim.acquisition = acq
    return sim


@pytest.fixture
def simulation(tmp_path):
    return _elastic_simulation(tmp_path)


@pytest.fixture
def space(simulation):
    return DataSpace.from_simulation(simulation, frequencies=[5.0, 10.0])


# ---------------------------------------------------------------------------
# TraceStoreRef / ObservedGroup / ObservedData
# ---------------------------------------------------------------------------


def test_trace_store_ref_round_trips_and_normalizes():
    ref = TraceStoreRef(
        "obs.h5", dataset="/surface/", missing="Zero", source_basis="SOURCE_GEOMETRY"
    )

    assert ref.to_fs() == {
        "_type": "HDF5TraceStore",
        "file": "obs.h5",
        "dataset": "surface",
        "missing": "zero",
        "source_basis": "source_geometry",
    }
    assert TraceStoreRef.from_fs(ref.to_fs()) == ref
    assert TraceStoreRef.from_fs(
        {"_type": "SeismicStore", "file": "obs.h5"}
    ) == TraceStoreRef("obs.h5")
    packed = TraceStoreRef.packed("observed", receiver_group="surface", suffix="_df")
    assert packed.file == Path("observed") / "traces.h5"
    assert packed.dataset == "surface_df"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TraceStoreRef(""),
        lambda: TraceStoreRef("obs.h5", missing="ignore"),
        lambda: TraceStoreRef("obs.h5", source_basis="rhs"),
        lambda: TraceStoreRef.from_fs(
            {"_type": "HDF5TraceStore", "file": "a", "extra": 1}
        ),
        lambda: TraceStoreRef.from_fs({"_type": "HDF5Dense", "file": "a"}),
    ],
)
def test_trace_store_ref_rejects_invalid_descriptors(factory):
    with pytest.raises(ValueError):
        factory()


def test_observed_group_serializes_stems_and_stores():
    group = ObservedGroup(
        "surface",
        observed="observed",
        derivatives={
            "df": {"_type": "HDF5TraceStore", "file": "df.h5", "dataset": "surface"}
        },
        source_basis="source_geometry",
    )

    payload = group.to_fs()

    assert payload == {
        "name": "surface",
        "observed": "observed",
        "observed_derivatives": {
            "df": {"_type": "HDF5TraceStore", "file": "df.h5", "dataset": "surface"}
        },
        "observed_source_basis": "source_geometry",
    }
    assert ObservedGroup.from_fs(payload) == group
    assert group.df == TraceStoreRef("df.h5", dataset="surface")
    assert ObservedGroup("zero").to_fs() == {"name": "zero"}
    resolved = group.resolved(lambda path: Path("/root") / path)
    assert resolved.observed == Path("/root/observed")
    assert resolved.df.file == Path("/root/df.h5")
    with pytest.raises(ValueError):
        ObservedGroup("surface", derivatives={"ds": "x"})


def test_observed_data_infers_frequencies_and_groups_from_forward_job(
    tmp_path, simulation
):
    job = FrequencyDomainJob(name="obs", simulation=simulation, f_list=[5.0, 10.0])
    df_job = FrequencyDomainJob(
        name="obs_df", simulation=simulation, f_list=[5.0, 10.0], phase_derivatives=1
    )
    assert "surface_df" in df_job.trace_outputs.groups

    observed = ObservedData(
        job, derivatives={"df": df_job}, source_basis="source_geometry"
    )

    assert observed.frequencies == [5.0, 10.0]
    assert observed.group_names == ("surface",)
    groups = observed.resolve(simulation)
    assert [group.name for group in groups] == ["surface"]
    assert groups[0].observed == Path(job.trace_path)
    assert groups[0].source_basis == "source_geometry"
    assert groups[0].df == TraceStoreRef(
        Path(df_job.trace_path) / "traces.h5",
        dataset="surface_df",
        source_basis="source_geometry",
    )
    assert observed.groups["surface"] == groups[0]

    with_missing = ObservedData(job, missing="zero")
    ref = with_missing.resolve(simulation)[0].observed
    assert ref == TraceStoreRef(
        Path(job.trace_path) / "traces.h5", dataset="surface", missing="zero"
    )


def test_observed_data_from_stem_applies_to_every_simulation_group(tmp_path):
    simulation = _elastic_simulation(tmp_path, groups=("surface", "obc"))
    observed = ObservedData("observed_root", derivatives={"df": "observed_df"})

    assert observed.frequencies is None
    assert observed.group_names is None
    assert observed.groups == {}
    groups = observed.resolve(simulation)
    assert [group.name for group in groups] == ["surface", "obc"]
    assert groups[1].observed == Path("observed_root")
    assert groups[1].df == TraceStoreRef(
        Path("observed_df") / "traces.h5", dataset="obc_df"
    )

    explicit = ObservedData("observed_root", frequencies=[3.0])
    assert explicit.frequencies == [3.0]


def test_observed_data_from_mapping_validates_simulation_coverage(tmp_path):
    simulation = _elastic_simulation(tmp_path, groups=("surface", "obc"))
    observed = ObservedData(
        {"surface": "obs.h5", "obc": TraceStoreRef("obc.h5", dataset="obc")},
        missing="warn",
    )

    groups = observed.resolve(simulation)

    assert groups[0].observed == Path("obs.h5")
    assert groups[1].observed == TraceStoreRef("obc.h5", dataset="obc", missing="warn")
    with pytest.raises(KeyError, match="obc"):
        ObservedData({"surface": "obs.h5"}).resolve(simulation)
    with pytest.raises(ValueError, match="absent"):
        ObservedData({"surface": "a", "obc": "b", "streamer": "c"}).resolve(simulation)
    with pytest.raises(KeyError, match="derivatives"):
        ObservedData("root", derivatives={"df": {"surface": "df.h5"}}).resolve(
            simulation
        )


def test_observed_data_shared_store_and_rejects_unknown_forms(tmp_path, simulation):
    shared = ObservedData(TraceStoreRef("all.h5"), source_basis="source_geometry")
    group = shared.resolve(simulation)[0]

    assert group.observed == TraceStoreRef("all.h5", source_basis="source_geometry")
    assert group.source_basis is None
    with pytest.raises(TypeError):
        ObservedData(42)
    with pytest.raises(ValueError):
        ObservedData("root", derivatives={"ds": "root"})
    with pytest.raises(ValueError):
        ObservedData({})


# ---------------------------------------------------------------------------
# DataSpace / DataVector
# ---------------------------------------------------------------------------


def test_data_space_layout_pack_and_unpack(space):
    assert space.groups == ("surface",)
    assert space.segments[0].shape == (2, 2, 3)
    assert space.size == 2 * 2 * 2 * 3
    assert space.shape == (24,)
    values = np.arange(space.size, dtype=complex).reshape(2, 2, 2, 3)

    vector = space.pack({"surface": values})
    dataset = space.unpack(vector)

    np.testing.assert_array_equal(vector, np.arange(24))
    assert dataset["surface"].dims == ("frequency", "source", "component", "receiver")
    assert dataset["surface"].coords["component"].values.tolist() == ["vx", "vz"]
    np.testing.assert_array_equal(dataset["surface"].values, values)
    transposed = dataset["surface"].transpose(
        "receiver", "frequency", "component", "source"
    )
    np.testing.assert_array_equal(
        space.pack(xr.Dataset({"surface": transposed})), vector
    )
    with pytest.raises(KeyError):
        space.pack({"other": values})
    with pytest.raises(ValueError):
        space.pack(np.zeros(3))
    assert space.frequency_index(10.0) == 1
    assert space.frequency_index(0) == 0
    with pytest.raises(ValueError):
        space.frequency_index()


def test_data_space_vectors_and_arithmetic(space):
    zeros = space.zeros()
    a = space.random(seed=1)
    b = space.random(seed=1)
    c = space.random(seed=2)

    assert isinstance(zeros, DataVector) and zeros.norm() == 0.0
    assert a == b and a != c
    np.testing.assert_allclose((a + c - c).values, a.values)
    np.testing.assert_allclose((2.0 * a / 2.0).values, a.values)
    np.testing.assert_allclose((a * np.ones(space.size)).values, a.values)
    np.testing.assert_allclose((-a + a).values, 0.0)
    assert a.vdot(a) == pytest.approx(a.norm() ** 2)
    assert a.dot(c) == pytest.approx(np.real(np.vdot(a.values, c.values)))
    assert np.asarray(a).shape == (space.size,)
    assert a.to_dataset()["surface"].shape == (2, 2, 2, 3)
    assert a["surface"].dims == ("frequency", "source", "component", "receiver")
    assert a.to_numpy() is not a.values
    other = DataSpace.from_simulation(_elastic_simulation(Path("/nonexistent")), [5.0])
    with pytest.raises(ValueError):
        a + other.zeros()
    with pytest.raises(ValueError):
        DataVector(np.zeros(3), space)


def test_term_layout_reproduces_sauce_dense_row_ids(space):
    layout = space.term_layout("surface", frequency=10.0)
    n_src, n_comp, n_rcv = 2, 2, 3

    assert layout.id == "surface"
    assert layout.n_global_rows == 12 and layout.complete
    for row, key, index in zip(layout.row_ids, layout.coordinate_keys, layout.indices):
        rhs, receiver, component = key
        assert row == rhs + n_src * ((component - 1) + n_comp * (receiver - 1))
        expected = 12 + ((rhs - 1) * n_comp + (component - 1)) * n_rcv + (receiver - 1)
        assert index == expected
    assert sorted(layout.row_ids.tolist()) == list(range(1, 13))
    assert layout.layout_fingerprint == objective_layout_fingerprint(
        12, layout.row_ids, layout.coordinate_keys
    )
    assert [
        term.id for term in space.term_layouts(frequency=0, ids={"surface": "p"})
    ] == ["p"]
    with pytest.raises(ValueError):
        TermLayout("t", 2, [1, 1], [[1, 1, 1], [1, 1, 2]])


def test_layout_fingerprint_matches_sauce_chunked_catalog():
    n_rows = 4096 + 10
    ids = np.arange(1, n_rows + 1)
    keys = np.stack([ids, np.ones_like(ids), np.full_like(ids, 2)], axis=1)

    fingerprint = objective_layout_fingerprint(n_rows, ids, keys)

    chunks = {}
    for first in (1, 4097):
        last = min(first + 4095, n_rows)
        flat = keys[first - 1 : last].reshape(-1).tolist()
        chunks[str(first)] = canonical_json_sha256({"keys": flat})
    expected = canonical_json_sha256({"n_rows": n_rows, **chunks})
    assert fingerprint == expected
    shuffled = np.random.default_rng(0).permutation(n_rows)
    assert (
        objective_layout_fingerprint(n_rows, ids[shuffled], keys[shuffled])
        == fingerprint
    )
    assert (
        canonical_json_sha256({"b": 1, "a": [1, 2]})
        == "sha256:" + hashlib.sha256(b'{"a":[1,2],"b":1}').hexdigest()
    )


# ---------------------------------------------------------------------------
# fs-objective-vector-3
# ---------------------------------------------------------------------------


def test_objective_vector_write_read_round_trip(tmp_path, space):
    vector = space.random(seed=3)
    layouts = space.term_layouts(frequency=10.0)
    manifest_path = tmp_path / "dual.json"

    written = vector.write_objective_vector(
        manifest_path, state_fingerprint=STATE, term_layout=layouts
    )
    manifest = json.loads(written.read_text())

    Draft202012Validator(json.loads(OBJECTIVE_VECTOR_SCHEMA.read_text())).validate(
        manifest
    )
    assert manifest["schema"] == "fs-objective-vector-3"
    assert manifest["state_fingerprint"] == STATE
    assert manifest["partition"] == {
        "n_ranks": 1,
        "compatibility": "same_mesh_partition",
    }
    shard = Path(manifest["shards"][0]["file"])
    assert shard == (tmp_path / "dual_rank_0.h5").resolve()
    assert manifest["shards"][0]["sha256"] == file_sha256(shard)
    assert manifest["terms"] == [layouts[0].manifest_entry()]
    basis = {k: v for k, v in manifest.items() if k != "manifest_fingerprint"}
    assert manifest["manifest_fingerprint"] == canonical_json_sha256(basis)
    with h5py.File(shard, "r") as h5:
        assert h5["/terms/0/coordinate_keys"].shape == (12, 3)
        assert h5["/terms/0/values"].shape == (12, 2)
        assert h5["/terms/0/row_ids"].shape == (12,)

    restored = DataVector.read_objective_vector(manifest_path, space, frequency=10.0)

    expected = space.zeros().values
    expected[layouts[0].indices] = vector.values[layouts[0].indices]
    np.testing.assert_allclose(restored.values, expected)
    assert restored.to_dataset()["surface"].sel(frequency=5.0).values.sum() == 0
    layouts_back = DataVector.read_term_layouts(manifest_path)
    assert layouts_back[0].layout_fingerprint == layouts[0].layout_fingerprint
    assert layouts_back[0].indices is None
    with pytest.raises(ValueError, match="another baseline"):
        DataVector.read_objective_vector(
            manifest_path,
            space,
            frequency=10.0,
            state_fingerprint="sha256:" + "00" * 32,
        )


def test_objective_vector_reader_reassembles_multiple_shards(tmp_path, space):
    single = DataSpace(frequencies=[5.0], segments=space.segments)
    vector = single.random(seed=4)
    layout = single.term_layout("surface", id="pressure")
    manifest_path = tmp_path / "vector.json"
    vector.write_objective_vector(
        manifest_path, state_fingerprint=STATE, term_layout=layout
    )
    manifest = json.loads(manifest_path.read_text())
    original = Path(manifest["shards"][0]["file"])

    with h5py.File(original, "r") as h5:
        ids = h5["/terms/0/row_ids"][()]
        keys = h5["/terms/0/coordinate_keys"][()]
        values = h5["/terms/0/values"][()]
    order = np.random.default_rng(1).permutation(ids.size)
    split = ids.size // 3
    shards = []
    for rank, selection in enumerate((order[:split], order[split:])):
        shard = tmp_path / f"vector_rank_{rank}.h5"
        with h5py.File(shard, "w") as h5:
            h5.create_dataset("/terms/0/row_ids", data=ids[selection])
            h5.create_dataset("/terms/0/coordinate_keys", data=keys[selection])
            h5.create_dataset("/terms/0/values", data=values[selection])
        shards.append({"file": shard.name, "sha256": file_sha256(shard)})
    manifest["shards"] = shards
    manifest["partition"]["n_ranks"] = 2
    basis = {k: v for k, v in manifest.items() if k != "manifest_fingerprint"}
    manifest["manifest_fingerprint"] = canonical_json_sha256(basis)
    manifest_path.write_text(json.dumps(manifest))

    restored = DataVector.read_objective_vector(
        manifest_path, single, term_layout=layout
    )

    np.testing.assert_allclose(restored.values, vector.values)


def test_objective_vector_reader_rejects_corrupt_or_incomplete_files(tmp_path, space):
    single = DataSpace(frequencies=[5.0], segments=space.segments)
    vector = single.random(seed=5)
    manifest_path = tmp_path / "v.json"
    vector.write_objective_vector(
        manifest_path, state_fingerprint=STATE, term_layout=single.term_layouts()
    )
    manifest = json.loads(manifest_path.read_text())
    shard = Path(manifest["shards"][0]["file"])

    tampered = dict(manifest, state_fingerprint="sha256:" + "11" * 32)
    (tmp_path / "tampered.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="manifest fingerprint"):
        DataVector.read_objective_vector(tmp_path / "tampered.json", single)
    DataVector.read_objective_vector(tmp_path / "tampered.json", single, verify=False)

    with h5py.File(shard, "a") as h5:
        h5["/terms/0/values"][0, 0] += 1.0
    with pytest.raises(ValueError, match="corrupt objective shard"):
        DataVector.read_objective_vector(manifest_path, single)

    with h5py.File(shard, "a") as h5:
        ids = h5["/terms/0/row_ids"][()]
        ids[1] = ids[0]
        del h5["/terms/0/row_ids"]
        h5.create_dataset("/terms/0/row_ids", data=ids)
    manifest["shards"][0]["sha256"] = file_sha256(shard)
    basis = {k: v for k, v in manifest.items() if k != "manifest_fingerprint"}
    manifest["manifest_fingerprint"] = canonical_json_sha256(basis)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate rows"):
        DataVector.read_objective_vector(manifest_path, single)

    obsolete = dict(manifest, schema="fs-objective-vector-2")
    (tmp_path / "old.json").write_text(json.dumps(obsolete))
    with pytest.raises(ValueError, match="version 3"):
        DataVector.read_objective_vector(tmp_path / "old.json", single, verify=False)


def test_objective_vector_reader_requires_matching_layout(tmp_path, space):
    single = DataSpace(frequencies=[5.0], segments=space.segments)
    vector = single.random(seed=6)
    manifest_path = tmp_path / "v.json"
    layout = single.term_layout("surface", id="custom")
    vector.write_objective_vector(
        manifest_path, state_fingerprint=STATE, term_layout=layout
    )

    with pytest.raises(ValueError, match="pass term_layout"):
        DataVector.read_objective_vector(manifest_path, single)
    restored = DataVector.read_objective_vector(
        manifest_path, single, term_layout=layout
    )
    np.testing.assert_allclose(restored.values, vector.values)

    other = TermLayout(
        "custom",
        layout.n_global_rows,
        layout.row_ids,
        np.roll(layout.coordinate_keys, 1, axis=0),
        indices=layout.indices,
    )
    with pytest.raises(ValueError, match="layout fingerprint"):
        DataVector.read_objective_vector(manifest_path, single, term_layout=other)
    with pytest.raises(ValueError, match="sha256"):
        vector.write_objective_vector(
            tmp_path / "bad.json", state_fingerprint="abc", term_layout=layout
        )


@pytest.mark.parametrize("n_ranks", [2, 31])
def test_objective_vector_writer_covers_every_rank_including_empty(
    tmp_path, space, n_ranks
):
    vector = space.random(4)
    path = vector.write_objective_vector(
        tmp_path / "dual.json",
        state_fingerprint=STATE,
        term_layout=space.term_layouts(frequency=space.frequencies[0]),
        n_ranks=n_ranks,
    )
    manifest = DataVector.read_objective_manifest(path)
    assert len(manifest["shards"]) == manifest["partition"]["n_ranks"] == n_ranks
    restored = DataVector.read_objective_vector(
        path, space, frequency=space.frequencies[0]
    )
    for layout in space.term_layouts(frequency=space.frequencies[0]):
        np.testing.assert_array_equal(
            restored.values[layout.indices], vector.values[layout.indices]
        )
