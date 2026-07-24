from pathlib import Path

import h5py
import numpy as np
import pytest
import xarray as xr

from frequensolve.model.property import Property
from frequensolve.project import Project
from frequensolve.util import store as store_module
from frequensolve.util.store import (
    SimulationStore,
    compact_hdf5_file,
    hash_dataarray_payload,
)


def _bloated_hdf5(path: Path) -> int:
    with h5py.File(path, "w") as h5:
        obsolete = h5.create_dataset("obsolete", shape=(512, 1024), dtype="f8")
        obsolete[...] = 1.0
        del h5["obsolete"]
        live = h5.create_dataset("inputs/live", data=np.arange(16, dtype=np.float32))
        live.attrs["units"] = "km/s"
        h5.attrs["schema"] = "test"
    return path.stat().st_size


def test_compact_hdf5_file_reclaims_deleted_dataset_space(tmp_path):
    path = tmp_path / "simulation.h5"
    original_size = _bloated_hdf5(path)

    reclaimed = compact_hdf5_file(
        path,
        min_reclaim_bytes=1,
        min_reclaim_fraction=0.01,
    )

    assert reclaimed > 3.9 * 1024 * 1024
    assert path.stat().st_size < original_size / 10
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5["inputs/live"][:], np.arange(16))
        assert h5["inputs/live"].attrs["units"] == "km/s"
        assert h5.attrs["schema"] == "test"


def test_compact_hdf5_file_leaves_dense_file_unchanged(tmp_path):
    path = tmp_path / "dense.h5"
    with h5py.File(path, "w") as h5:
        values = h5.create_dataset("values", shape=(512, 1024), dtype="f8")
        values[...] = 1.0
    original_size = path.stat().st_size

    reclaimed = compact_hdf5_file(
        path,
        min_reclaim_bytes=1,
        min_reclaim_fraction=0.25,
    )

    assert reclaimed == 0
    assert path.stat().st_size == original_size


def test_simulation_store_prunes_datasets_absent_from_current_payload(tmp_path):
    project_path = tmp_path / "project"
    path = project_path / "simulations" / "model" / "model.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("inputs/current", data=np.arange(4))
        h5.create_dataset("inputs/receiver_coordinates", data=np.arange(6))
        h5.create_dataset("inputs/obsolete", data=np.arange(8))
    store = SimulationStore(path, project_path=project_path)
    payload = {
        "property": {
            "file": "simulations/model/model.h5:inputs/current",
            "format": "hdf5",
            "dataset": "inputs/current",
        },
        "remote_property": {
            "file": "/remote/shared/model.rsf",
            "format": "RSF",
        },
        "receiver_coordinates": {
            "_type": "CoordsFromFile",
            "file": "simulations/model/model.h5:inputs/receiver_coordinates",
            "format": "HDF5",
        },
    }

    removed = store.prune_unreferenced(payload)

    assert removed == ["inputs/obsolete"]
    with h5py.File(path, "r") as h5:
        assert "inputs/current" in h5
        assert "inputs/receiver_coordinates" in h5
        assert "inputs/obsolete" not in h5


def test_put_array_chunks_reuses_unchanged_factory_dataset(tmp_path):
    path = tmp_path / "simulation.h5"
    store = SimulationStore(path)
    factory_calls = 0

    def chunks():
        nonlocal factory_calls
        factory_calls += 1
        yield np.arange(6, dtype=np.float64).reshape(2, 3)
        yield np.arange(6, 12, dtype=np.float64).reshape(2, 3)

    first = store.put_array_chunks("coordinates", (4, 3), chunks)
    first_size = path.stat().st_size
    second = store.put_array_chunks("coordinates", (4, 3), chunks)

    assert second.hash == first.hash
    assert path.stat().st_size == first_size
    assert factory_calls == 3
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(h5["coordinates"][:], np.arange(12).reshape(4, 3))


def test_put_dataarray_replaces_incomplete_matching_hash(tmp_path):
    path = tmp_path / "simulation.h5"
    store = SimulationStore(path)
    data = xr.DataArray(
        np.arange(6, dtype=np.float64).reshape(3, 2),
        dims=("x", "z"),
        coords={"x": [0.0, 0.5, 1.0], "z": [0.0, 1.0]},
    )
    attrs = {"fs_kind": "property"}
    digest = hash_dataarray_payload(data, attrs=attrs, dtype=None)
    with h5py.File(path, "w") as h5:
        dset = h5.create_dataset("values", data=data.values)
        dset.attrs["fs_hash"] = f"blake3:{digest}"
        dset.attrs["fs_hash_algorithm"] = "blake3"
        dset.attrs["dims"] = ["x", "z"]
        dset.attrs["incomplete"] = True

    ref = store.put_dataarray("values", data, attrs=attrs, dtype=None)

    assert ref.hash == digest
    with h5py.File(path, "r") as h5:
        dset = h5["values"]
        assert "incomplete" not in dset.attrs
        assert list(dset.attrs["x"]) == [0.0, 0.5, 1.0]
        assert list(dset.attrs["z"]) == [0.0, 1.0]
        assert dset.attrs["fs_kind"] == "property"


def test_put_dataarray_removes_dataset_after_metadata_failure(monkeypatch, tmp_path):
    path = tmp_path / "simulation.h5"
    store = SimulationStore(path)
    data = xr.DataArray(
        np.arange(3, dtype=np.float64),
        dims=("x",),
        coords={"x": [0.0, 0.5, 1.0]},
    )
    real_h5_attr_value = store_module._h5_attr_value

    def fail_on_marker(value):
        if isinstance(value, str) and value == "trigger failure":
            raise RuntimeError("injected metadata failure")
        return real_h5_attr_value(value)

    with monkeypatch.context() as patch:
        patch.setattr(store_module, "_h5_attr_value", fail_on_marker)
        with pytest.raises(RuntimeError, match="injected metadata failure"):
            store.put_dataarray(
                "values",
                data,
                attrs={"marker": "trigger failure"},
                dtype=None,
            )

    with h5py.File(path, "r") as h5:
        assert "values" not in h5

    ref = store.put_dataarray(
        "values",
        data,
        attrs={"marker": "trigger failure"},
        dtype=None,
    )
    with h5py.File(path, "r") as h5:
        dset = h5["values"]
        assert dset.attrs["fs_hash"] == f"blake3:{ref.hash}"
        assert dset.attrs["marker"] == "trigger failure"


def test_put_dataarray_references_large_dimension_coordinate_dataset(tmp_path):
    path = tmp_path / "simulation.h5"
    store = SimulationStore(path)
    x = np.linspace(0.0, 1.0, 10_000)
    data = xr.DataArray(
        np.sin(x),
        dims=("x",),
        coords={"x": x},
    )

    ref = store.put_dataarray("values", data, dtype=None)

    with h5py.File(path, "a") as h5:
        dset = h5["values"]
        assert dset.attrs["x"] == "/values.__coordinates__/0"
        coordinate = h5[dset.attrs["x"]]
        assert coordinate.attrs["dimension"] == "x"
        assert coordinate.attrs["fs_kind"] == "dimension_coordinate"
        np.testing.assert_array_equal(coordinate[:], x)
        h5.create_dataset("obsolete", data=np.arange(3))

    removed = store.prune_unreferenced({"property": ref.to_fs()})

    assert removed == ["obsolete"]
    loaded = Property.read(Path(f"{path}:values"))
    assert loaded.dims == ("x",)
    np.testing.assert_array_equal(loaded.coords["x"], x)
    np.testing.assert_array_equal(loaded.values, data.values)

    with h5py.File(path, "a") as h5:
        del h5["values.__coordinates__/0"]

    repeated = store.put_dataarray("values", data, dtype=None)

    assert repeated.hash == ref.hash
    with h5py.File(path, "r") as h5:
        coordinate_reference = h5["values"].attrs["x"]
        assert coordinate_reference in h5
        np.testing.assert_array_equal(h5[coordinate_reference][:], x)


def test_put_array_chunks_replaces_incomplete_matching_hash(tmp_path):
    path = tmp_path / "simulation.h5"
    store = SimulationStore(path)

    def chunks():
        yield np.arange(6, dtype=np.float64).reshape(2, 3)

    first = store.put_array_chunks(
        "coordinates",
        (2, 3),
        chunks,
        dims=("receiver", "coordinate"),
        coords={"coordinate": np.asarray(["x", "y", "z"])},
        attrs={"fs_kind": "receiver_coordinates"},
    )
    with h5py.File(path, "a") as h5:
        del h5["coordinates"].attrs["coordinate"]
        h5["coordinates"].attrs["incomplete"] = True

    second = store.put_array_chunks(
        "coordinates",
        (2, 3),
        chunks,
        dims=("receiver", "coordinate"),
        coords={"coordinate": np.asarray(["x", "y", "z"])},
        attrs={"fs_kind": "receiver_coordinates"},
    )

    assert second.hash == first.hash
    with h5py.File(path, "r") as h5:
        dset = h5["coordinates"]
        assert "incomplete" not in dset.attrs
        assert list(dset.attrs["coordinate"]) == ["x", "y", "z"]
        assert dset.attrs["fs_kind"] == "receiver_coordinates"


def test_simulation_save_prunes_and_compacts_old_store_data(monkeypatch, tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    simulation = project.new_simulation(
        name="remote_model",
        physics="acoustic",
        dimension=2,
    )
    path = project.path / "simulations" / "remote_model" / "remote_model.h5"
    path.parent.mkdir(parents=True)
    original_size = _bloated_hdf5(path)

    from frequensolve.simulation import simulation as simulation_module

    real_compact = compact_hdf5_file
    monkeypatch.setattr(
        simulation_module,
        "compact_hdf5_file",
        lambda candidate: real_compact(
            candidate,
            min_reclaim_bytes=1,
            min_reclaim_fraction=0.01,
        ),
    )

    simulation.save()

    assert path.stat().st_size < original_size / 10
    with h5py.File(path, "r") as h5:
        datasets = []
        h5.visititems(
            lambda name, obj: (
                datasets.append(name) if isinstance(obj, h5py.Dataset) else None
            )
        )
    assert datasets == []


def test_project_transfer_compacts_only_the_staging_copy(monkeypatch, tmp_path):
    project = Project(name="project", path=tmp_path / "project")
    source = project.path / "simulations" / "model" / "model.h5"
    source.parent.mkdir(parents=True)
    source_size = _bloated_hdf5(source)
    captured = {}

    from frequensolve.project import project as project_module

    real_compact = compact_hdf5_file

    def compact_small_fixture(path):
        return real_compact(
            path,
            min_reclaim_bytes=1,
            min_reclaim_fraction=0.01,
        )

    class CaptureSite:
        work_dir = Path("/remote/project")

        def put(self, local, remote):
            local = Path(local)
            if not local.is_dir():
                return
            staged = local / "simulations" / "model" / "model.h5"
            captured["size"] = staged.stat().st_size
            with h5py.File(staged, "r") as h5:
                captured["values"] = h5["inputs/live"][:]

    monkeypatch.setattr(project_module, "compact_hdf5_file", compact_small_fixture)

    project._transfer(CaptureSite())

    assert captured["size"] < source_size / 10
    np.testing.assert_array_equal(captured["values"], np.arange(16))
    assert source.stat().st_size == source_size


def test_compaction_preserves_file_mode(tmp_path):
    path = tmp_path / "mode.h5"
    _bloated_hdf5(path)
    path.chmod(0o640)

    compact_hdf5_file(path, min_reclaim_bytes=1, min_reclaim_fraction=0.01)

    assert path.stat().st_mode & 0o777 == 0o640
