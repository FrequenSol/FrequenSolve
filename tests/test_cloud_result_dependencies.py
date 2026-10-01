"""Exercise downloaded result reading in the isolated Cloud-extra wheel lane."""

import builtins

import h5py
import numpy as np
import pytest

from frequensolve.seismic.traces import TraceDataset


@pytest.fixture
def downloaded_pressure_traces(tmp_path):
    path = tmp_path / "traces.h5"
    values = np.zeros((1, 2, 1, 1, 2), dtype=np.float32)
    values[0, :, 0, 0, 0] = [3.0, 5.0]
    values[0, :, 0, 0, 1] = [4.0, -2.0]
    strings = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frequency", data=[10.0])
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=strings),
        )
        pressure = h5.create_dataset("surface", data=values)
        pressure.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        pressure.attrs["layout_kind"] = ["dense_trace_v1"]
        pressure.attrs["receiver"] = [101, 102]
        pressure.attrs["component"] = np.array(["p"], dtype=strings)
        pressure.attrs["shot"] = [7]
    return path


def test_cloud_extra_reads_downloaded_frequency_domain_traces(
    downloaded_pressure_traces,
):
    # The base unit lane may omit Dask. The isolated cloud-extra contract first
    # requires its import, so a wheel missing this dependency cannot skip green.
    pytest.importorskip("dask.array")
    with TraceDataset.open(downloaded_pressure_traces) as traces:
        pressure = traces.fd("surface", "p", source=7)
        assert pressure.coords["frequency"].values.tolist() == [10.0]
        assert pressure.coords["receiver"].values.tolist() == [101, 102]
        np.testing.assert_allclose(
            pressure.transpose("frequency", "receiver").values,
            [[3.0 + 4.0j, 5.0 - 2.0j]],
        )


def test_missing_lazy_reader_dependency_recommends_cloud_extra(
    downloaded_pressure_traces, monkeypatch
):
    original_import = builtins.__import__

    def without_dask(name, *args, **kwargs):
        if name == "dask.array":
            raise ModuleNotFoundError("No module named 'dask'", name="dask")
        return original_import(name, *args, **kwargs)

    with TraceDataset.open(downloaded_pressure_traces) as traces:
        monkeypatch.setattr(builtins, "__import__", without_dask)
        with pytest.raises(ImportError, match=r"frequensolve\[cloud\].*dask"):
            traces.fd("surface", "p", source=7)
