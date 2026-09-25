"""Trace reads report the physical load and data units behind each gather."""

import h5py
import numpy as np
import pytest

from frequensolve.seismic.traces import TraceDataset
from frequensolve.seismic.wavelet import RickerWavelet


def _write_traces(path, *, strength=True, encoding="identity"):
    frequencies = np.arange(1.0, 26.0)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        h5.create_dataset("frequency", data=frequencies)
        h5.create_dataset("laplace", data=np.zeros(frequencies.size))
        h5.create_dataset(
            "survey/packed_layout_kind",
            data=np.array(["packed_frequency_trace_v1"], dtype=string_dtype),
        )
        data = np.zeros((frequencies.size, 1, 1, 2, 2))
        data[:, 0, 0, :, 0] = 1.0
        dset = h5.create_dataset("surface", data=data)
        dset.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        dset.attrs["receiver"] = np.array([101], dtype=np.int32)
        dset.attrs["component"] = np.array(["p"], dtype=string_dtype)
        dset.attrs["units"] = np.array(["Pa"], dtype=string_dtype)
        dset.attrs["shot"] = np.array([1, 2], dtype=np.int32)
        h5.create_dataset(
            "survey/source_encoding/encoding_kind",
            data=np.array([encoding], dtype=string_dtype),
        )
        if not strength:
            return
        geometry = h5.create_group("survey/source_geometry")
        geometry["strength"] = np.array([1.0e9, 2.5e5])
        geometry["strength_units"] = np.array(["N*m", "N"], dtype=string_dtype)
        geometry["strength_origin"] = np.array(
            ["default", "specified"], dtype=string_dtype
        )
        geometry["source_kind_name"] = np.array(
            ["monopole", "vector"], dtype=string_dtype
        )


@pytest.mark.parametrize(
    "source, expected",
    [
        (1, (1.0e9, "N*m", "default", "monopole")),
        (2, (2.5e5, "N", "specified", "vector")),
    ],
)
def test_fd_reports_source_strength_and_data_units(tmp_path, source, expected):
    path = tmp_path / "traces.h5"
    _write_traces(path)

    fd = TraceDataset.open(path).fd("surface", "p", source=source)

    strength, units, origin, kind = expected
    assert fd.attrs["units"] == "Pa"
    assert fd.attrs["source_strength"] == strength
    assert fd.attrs["source_strength_units"] == units
    assert fd.attrs["source_strength_origin"] == origin
    assert fd.attrs["source_kind"] == kind
    assert "wavelet" not in fd.attrs


def test_td_records_the_dimensionless_wavelet(tmp_path):
    path = tmp_path / "traces.h5"
    _write_traces(path)
    wavelet = RickerWavelet(f=10.0, center=0.15, scale=2.0)

    td = TraceDataset.open(path).td("surface", "p", source=1, wavelet=wavelet)

    assert td.attrs["units"] == "Pa"
    assert td.attrs["source_strength"] == 1.0e9
    assert td.attrs["wavelet"] == "RickerWavelet"
    assert td.attrs["wavelet_f"] == 10.0
    assert td.attrs["wavelet_scale"] == 2.0
    assert td.attrs["wavelet_center"] == 0.15
    assert td.attrs["wavelet_units"] == "1"


def test_encoded_fields_report_only_shared_source_properties(tmp_path):
    path = tmp_path / "traces.h5"
    _write_traces(path, encoding="dense")

    fd = TraceDataset.open(path).fd("surface", "p", source=1)

    assert fd.attrs["source_encoding"] == "dense"
    assert "source_strength" not in fd.attrs
    # The two physical sources differ in units, origin and kind.
    assert "source_strength_units" not in fd.attrs
    assert "source_strength_origin" not in fd.attrs


def test_traces_without_recorded_strength_keep_data_units(tmp_path):
    path = tmp_path / "traces.h5"
    _write_traces(path, strength=False)

    fd = TraceDataset.open(path).fd("surface", "p", source=1)

    assert fd.attrs["units"] == "Pa"
    assert not any(key.startswith("source_strength") for key in fd.attrs)
