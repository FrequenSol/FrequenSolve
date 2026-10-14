"""Lazy readers survive cache refresh and release handles with their graphs."""

import gc

import h5py
import numpy as np
import pytest

from frequensolve.seismic.trace_store import TraceStore


def _shards(tmp_path):
    paths = []
    for index, frequency in enumerate((10.0, 20.0)):
        path = tmp_path / f"trace-{index}.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("frequency", data=[frequency])
            data = h5.create_dataset(
                "surface", data=np.full((2, 1, 1, 2), index + 1.0, dtype=np.float32)
            )
            data.attrs["dims"] = ["receiver", "component", "shot"]
            data.attrs["receiver"] = [1, 2]
            data.attrs["component"] = np.asarray(["p"], dtype="S")
            data.attrs["shot"] = [1]
        paths.append(path)
    return TraceStore(
        metadata={"groups": ["surface"], "f_map": {1: 10.0, 2: 20.0}}, files=paths
    )


def test_retained_lazy_array_survives_repeated_cache_replacement(tmp_path):
    pytest.importorskip("dask.array")
    with _shards(tmp_path) as store:
        store.consolidate(cache_dir=tmp_path / "cache")
        fd = store.read_h5("surface")
        expected = fd.values.copy()
        for _ in range(3):
            store.consolidate(cache_dir=tmp_path / "cache")
            np.testing.assert_array_equal(fd.values, expected)
            current = store.read_h5("surface")
            np.testing.assert_array_equal(current.values, expected)
            del current
            gc.collect()
            assert len(store._open_files) == 1
        del fd
        gc.collect()
        assert len(store._open_files) == 0


def test_explicit_close_bounds_retained_lazy_reader(tmp_path):
    pytest.importorskip("dask.array")
    store = _shards(tmp_path)
    store.consolidate(cache_dir=tmp_path / "cache")
    fd = store.read_h5("surface")
    store.close()
    store.close()
    with pytest.raises(RuntimeError, match="Trace reader is closed"):
        fd.values


def test_failed_refresh_preserves_existing_cache_and_readers(tmp_path):
    pytest.importorskip("dask.array")
    with _shards(tmp_path) as store:
        cache = store.consolidate(cache_dir=tmp_path / "cache")
        previous = cache.read_bytes()
        fd = store.read_h5("surface")
        expected = fd.values.copy()
        store.metadata["groups"] = ["missing"]
        with pytest.raises(KeyError):
            store.consolidate(cache_dir=tmp_path / "cache")
        assert cache.read_bytes() == previous
        np.testing.assert_array_equal(fd.values, expected)
        assert list(cache.parent.glob("*.h5")) == [cache]
