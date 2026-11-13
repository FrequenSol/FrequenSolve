"""Behavioral contracts for streamed observed-data conversion."""

import subprocess
import sys

import h5py
import numpy as np
import pytest
import xarray as xr

import frequensolve.seismic.trace_record  # noqa: F401
from frequensolve.seismic.trace_conversion import convert_traces


@pytest.mark.parametrize("first", ["simulation", "seismic"])
def test_converter_public_import_orders_in_clean_process(first):
    # Importing trace_record in this test module otherwise masks import cycles.
    second = "seismic" if first == "simulation" else "simulation"
    code = f"""
import frequensolve.{first}
import frequensolve.{second}
from frequensolve.seismic import convert_traces
import frequensolve.seismic.trace_record
import xarray as xr
assert callable(convert_traces)
assert callable(xr.DataArray([1.0]).fs.to_trace_store)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_npy_streaming_and_frequency_cache(tmp_path):
    data = np.arange(120, dtype=np.float32).reshape(15, 8)
    source = tmp_path / "input.npy"
    np.save(source, data)
    output = convert_traces(
        source,
        tmp_path / "store.h5",
        dt=0.1,
        t0=0.025,
        component="p",
        batch_bytes=512,
        frequencies=[0, 1.37],
    )
    with h5py.File(output) as h5:
        np.testing.assert_array_equal(h5["data/time/value"], data)
        np.testing.assert_array_equal(h5["survey/traces/trace_id"], np.arange(1, 16))
        np.testing.assert_array_equal(h5["survey/traces/receiver_id"], np.arange(1, 16))
        t = 0.025 + np.arange(8) * 0.1
        expected = data @ np.exp(-2j * np.pi * t[:, None] * np.array([0, 1.37]))
        actual = h5["data/frequency/value"][...]
        np.testing.assert_allclose(
            actual[..., 0] + 1j * actual[..., 1], expected, rtol=1e-6, atol=1e-5
        )


def test_own_time_gather_accessor_preserves_ids(tmp_path):
    gather = xr.DataArray(
        np.arange(12).reshape(4, 3),
        dims=("time", "receiver"),
        coords={"time": np.arange(4) * 0.01, "receiver": [7, 2, 9]},
        attrs={"source_id": 5, "long_name": "p"},
    )
    out = gather.fs.to_trace_store(tmp_path / "observed.h5", batch_bytes=128)
    with h5py.File(out) as h5:
        np.testing.assert_array_equal(h5["data/time/value"], gather.values.T)
        np.testing.assert_array_equal(h5["survey/traces/source_id"], [5, 5, 5])
        np.testing.assert_array_equal(h5["survey/traces/receiver_id"], [7, 2, 9])


def test_dense_cube_with_components_and_lazy_blocks(tmp_path):
    da = pytest.importorskip("dask.array")
    from dask import delayed

    reads = []

    def block(i):
        reads.append(i)
        return np.full((1, 2, 3, 8), i, dtype=np.float32)

    data = da.concatenate(
        [
            da.from_delayed(delayed(block)(i), shape=(1, 2, 3, 8), dtype="f4")
            for i in range(2)
        ]
    )
    gather = xr.DataArray(
        data,
        dims=("source", "component", "receiver", "time"),
        coords={
            "source": [3, 8],
            "component": ["p", "vx"],
            "receiver": [10, 20, 30],
            "time": np.arange(8) * 0.01,
        },
    )
    out = convert_traces(
        gather, tmp_path / "cube.h5", component_map={"p": 1, "vx": 2}, batch_bytes=256
    )
    with h5py.File(out) as h5:
        np.testing.assert_array_equal(
            h5["survey/traces/source_id"], np.repeat([3, 8], 6)
        )
        np.testing.assert_array_equal(
            h5["survey/traces/receiver_id"], np.tile([10, 20, 30], 4)
        )
        np.testing.assert_array_equal(
            h5["survey/traces/component"], np.tile(np.repeat([1, 2], 3), 2)
        )
        np.testing.assert_array_equal(h5["data/time/value"][:, 0], np.repeat([0, 1], 6))
    assert reads  # Dask stays deferred until bounded read calls.


def test_frequency_gather_and_split_complex(tmp_path):
    values = np.arange(6).reshape(2, 3) + 2j
    gather = xr.DataArray(
        values,
        dims=("frequency", "receiver"),
        coords={
            "frequency": [1.37, 2.8],
            "laplace": ("frequency", [-0.2, -0.3]),
            "receiver": [2, 4, 6],
        },
        attrs={"source_id": 2, "long_name": "p"},
    )
    for name, input_ in (
        ("complex", gather),
        (
            "split",
            xr.concat(
                [gather.real, gather.imag],
                dim=xr.IndexVariable("complex", ["real", "imag"]),
            ),
        ),
    ):
        out = convert_traces(input_, tmp_path / f"{name}.h5")
        with h5py.File(out) as h5:
            actual = h5["data/frequency/value"][...]
            np.testing.assert_array_equal(
                actual[..., 0] + 1j * actual[..., 1], values.T
            )
            np.testing.assert_array_equal(h5["data/frequency/laplace"], [-0.2, -0.3])


def test_dataset_requires_variable(tmp_path):
    data = xr.Dataset(
        {name: (("trace", "time"), np.ones((2, 3))) for name in ["p", "v"]}
    )
    with pytest.raises(ValueError, match="variable"):
        convert_traces(data, tmp_path / "fail.h5", dt=0.1, component="p")
    convert_traces(data, tmp_path / "ok.h5", variable="p", dt=0.1, component="p")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "positive dt"),
        ({"dt": -1}, "positive dt"),
        ({"dt": 0.1, "source_ids": [0, 1]}, "positive int32"),
        ({"dt": 0.1, "receiver_ids": [1.5, 2]}, "positive int32"),
        ({"dt": 0.1, "frequencies": [6]}, "Nyquist"),
        ({"dt": 0.1, "batch_bytes": 0}, "batch_bytes"),
    ],
)
def test_invalid_input_leaves_no_output(tmp_path, kwargs, match):
    out = tmp_path / "store.h5"
    with pytest.raises(ValueError, match=match):
        convert_traces(np.ones((2, 4)), out, component="p", **kwargs)
    assert not out.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_failure_preserves_existing_output(tmp_path):
    out = tmp_path / "store.h5"
    out.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        convert_traces(np.ones((2, 4)), out, dt=0.1, component="p")
    data = np.ones((2, 4))
    data[-1, -1] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        convert_traces(
            data, out, dt=0.1, component="p", overwrite=True, batch_bytes=128
        )
    assert out.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [out]


def test_irregular_time_rejected(tmp_path):
    data = xr.DataArray(
        np.ones((2, 3)), dims=("receiver", "time"), coords={"time": [0, 0.1, 0.3]}
    )
    with pytest.raises(ValueError, match="uniformly"):
        convert_traces(data, tmp_path / "bad.h5", component="p")


def make_segy(path, *, inconsistent=False):
    segyio = pytest.importorskip("segyio")
    spec = segyio.spec()
    spec.sorting = 2
    spec.format = 5
    spec.samples = np.arange(8) * 2
    spec.tracecount = 3
    with segyio.create(str(path), spec) as f:
        f.bin[segyio.BinField.Interval] = 2000
        for i in range(3):
            f.trace[i] = np.arange(8, dtype=np.float32) + i
            f.header[i] = {
                segyio.TraceField.FieldRecord: 7,
                segyio.TraceField.TraceNumber: 10 + i,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: 2000,
                segyio.TraceField.DelayRecordingTime: (
                    4 if not inconsistent or i < 2 else 6
                ),
            }


def test_segy_conversion_and_late_sampling_failure(tmp_path):
    source = tmp_path / "input.sgy"
    make_segy(source)
    out = convert_traces(
        source, tmp_path / "observed.h5", component="p", batch_bytes=256
    )
    with h5py.File(out) as h5:
        np.testing.assert_array_equal(
            h5["data/time/value"], np.arange(8)[None, :] + np.arange(3)[:, None]
        )
        np.testing.assert_array_equal(h5["survey/traces/source_id"], [7, 7, 7])
        np.testing.assert_array_equal(h5["survey/traces/receiver_id"], [10, 11, 12])
        assert h5["data/time/dt"][()] == 0.002
        assert h5["data/time/t0"][()] == 0.004
    bad = tmp_path / "bad.sgy"
    make_segy(bad, inconsistent=True)
    with pytest.raises(ValueError, match="differing dt/t0"):
        convert_traces(bad, tmp_path / "bad.h5", component="p", batch_bytes=256)
    assert not (tmp_path / "bad.h5").exists()


def test_component_numbers_do_not_depend_on_batch_size(tmp_path):
    gather = xr.DataArray(
        np.ones((2, 2, 4)),
        dims=("component", "receiver", "time"),
        coords={"component": ["vx", "p"]},
    )
    for budget in (128, 4096):
        out = convert_traces(
            gather, tmp_path / f"components-{budget}.h5", dt=0.1, batch_bytes=budget
        )
        with h5py.File(out) as h5:
            np.testing.assert_array_equal(h5["survey/traces/component"], [2, 2, 1, 1])


def test_npy_uses_mmap_and_bounded_sample_reads(tmp_path, monkeypatch):
    import frequensolve.seismic.trace_conversion as conversion

    source = tmp_path / "source.npy"
    np.save(source, np.ones((100, 8), dtype=np.float32))
    original_load, original_read = np.load, conversion._ArrayTraces.read
    calls = []
    original_fromfile = np.fromfile
    sample_reads = []

    def fromfile(*args, **kwargs):
        sample_reads.append(kwargs["count"])
        return original_fromfile(*args, **kwargs)

    def load(*args, **kwargs):
        assert kwargs["mmap_mode"] == "r"
        assert kwargs["allow_pickle"] is False
        return original_load(*args, **kwargs)

    def read(self, start, stop):
        calls.append(stop - start)
        return original_read(self, start, stop)

    monkeypatch.setattr(conversion.np, "load", load)
    monkeypatch.setattr(conversion.np, "fromfile", fromfile)
    monkeypatch.setattr(conversion._ArrayTraces, "read", read)
    convert_traces(source, tmp_path / "out.h5", dt=0.1, component="p", batch_bytes=512)
    assert sum(calls) == 100
    assert max(calls) == 2
    assert max(sample_reads) == 16
    assert sum(sample_reads) == 800


@pytest.mark.integration
@pytest.mark.parametrize(
    "kind", ["npy", "npy_fft", "segy", "xarray_time", "xarray_frequency"]
)
def test_converted_store_is_readable_by_sauce(tmp_path, kind):
    """Set FS_TRACE_STORE_PROBE to Sauce's time_store_subset_probe executable."""
    import os
    import subprocess

    probe = os.environ.get("FS_TRACE_STORE_PROBE")
    if not probe:
        pytest.skip("Set FS_TRACE_STORE_PROBE to exercise native Sauce IO")
    data = np.arange(24, dtype=np.float32).reshape(3, 8)
    dt, t0 = 0.1, 0.025
    kwargs = {}
    if kind in {"npy", "npy_fft"}:
        if kind == "npy_fft":
            dt = 1 / (8 * 1.37)
        source = tmp_path / "source.npy"
        np.save(source, data)
        kwargs.update(
            dt=dt,
            t0=t0,
            frequencies=np.arange(1, 5) * 1.37 if kind == "npy_fft" else [1.37],
        )
    elif kind == "segy":
        source = tmp_path / "source.sgy"
        make_segy(source)
        data = np.arange(8)[None, :] + np.arange(3)[:, None]
        dt, t0 = 0.002, 0.004
    else:
        source = xr.DataArray(
            data.T,
            dims=("time", "receiver"),
            coords={"time": t0 + np.arange(8) * dt, "receiver": [1, 2, 3]},
            attrs={"source_id": 1, "long_name": "p"},
        )
    expected = data @ np.exp(-2j * np.pi * 1.37 * (t0 + np.arange(8) * dt))
    if kind == "xarray_frequency":
        source = xr.DataArray(
            expected[None, :],
            dims=("frequency", "receiver"),
            coords={"frequency": [1.37], "receiver": [1, 2, 3]},
        )
    output = convert_traces(source, tmp_path / "observed.h5", component="p", **kwargs)
    result = subprocess.run(
        [probe, str(output), "3", "0"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rows = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if line.startswith("TRACE_")
    )
    actual = np.array(
        [complex(*map(float, rows[f"TRACE_{i}"].split())) for i in range(1, 7)]
    )
    np.testing.assert_allclose(
        actual,
        [expected[2], expected[1], expected[2], expected[1], 0, 0],
        rtol=3e-6,
        atol=3e-6,
    )


def test_segy_identifiers_can_be_remapped_without_rewriting_source(tmp_path):
    source = tmp_path / "source.sgy"
    make_segy(source)
    before = source.read_bytes()
    output = convert_traces(
        source,
        tmp_path / "mapped.h5",
        component="p",
        source_ids=1,
        receiver_ids=[1, 3, 2],
        batch_bytes=256,
    )
    assert source.read_bytes() == before
    with h5py.File(output) as h5:
        np.testing.assert_array_equal(h5["survey/traces/source_id"], [1, 1, 1])
        np.testing.assert_array_equal(h5["survey/traces/receiver_id"], [1, 3, 2])


@pytest.mark.parametrize(
    "shape,dims,order",
    [
        ((17, 5), ("time", "receiver"), "C"),
        ((17, 3, 5), ("time", "source", "receiver"), "C"),
        ((3, 17, 5), ("source", "time", "receiver"), "C"),
        ((17, 5), ("time", "receiver"), "F"),
        ((5, 17), ("receiver", "time"), "F"),
    ],
)
def test_npy_axis_order_and_tiled_frequency_cache(tmp_path, shape, dims, order):
    data = np.array(
        np.arange(np.prod(shape)).reshape(shape), dtype=np.float32, order=order
    )
    source = tmp_path / "input.npy"
    np.save(source, data)
    output = convert_traces(
        source,
        tmp_path / "store.h5",
        dims=dims,
        component="p",
        dt=0.1,
        t0=0.025,
        batch_bytes=1024,
        frequencies=[1.37, 2.81],
    )
    canonical = np.moveaxis(data, dims.index("time"), -1).reshape(-1, 17)
    time = 0.025 + np.arange(17) * 0.1
    with h5py.File(output) as h5:
        np.testing.assert_array_equal(h5["data/time/value"], canonical)
        cached = h5["data/frequency/value"][...]
        expected = canonical @ np.exp(
            -2j * np.pi * time[:, None] * np.array([1.37, 2.81])
        )
        np.testing.assert_allclose(
            cached[..., 0] + 1j * cached[..., 1], expected, rtol=2e-6, atol=1e-5
        )


@pytest.mark.parametrize("on_bins", [False, True])
def test_batched_transforms_preserve_phase_and_normalization(tmp_path, on_bins):
    rng = np.random.default_rng(123)
    data = rng.normal(size=(13, 1024)).astype(np.float32)
    dt, t0 = 0.002, -0.013
    frequencies = (
        np.array([0, 1, 23, 97, 512]) / (1024 * dt)
        if on_bins
        else np.array([1.37, 3.73, 19.11, 23.29, 171.31])
    )
    # Force multiple trace batches and multiple frequency blocks.
    out = convert_traces(
        data,
        tmp_path / "cache.h5",
        dt=dt,
        t0=t0,
        component="p",
        batch_bytes=65536,
        frequencies=frequencies,
    )
    time = t0 + np.arange(1024) * dt
    expected = np.einsum(
        "rt,tf->rf",
        data.astype(np.float64),
        np.exp(-2j * np.pi * time[:, None] * frequencies),
        optimize=False,
    )
    with h5py.File(out) as h5:
        actual = h5["data/frequency/value"][...]
        np.testing.assert_allclose(
            actual[..., 0] + 1j * actual[..., 1], expected, rtol=3e-6, atol=2e-5
        )


def test_frequency_cache_avoids_per_trace_tiny_chunks(tmp_path):
    data = np.zeros((2048, 128), dtype=np.float32)
    out = convert_traces(
        data,
        tmp_path / "cache.h5",
        dt=0.002,
        component="p",
        frequencies=[1, 2, 3],
        batch_bytes=1024**2,
    )
    with h5py.File(out) as h5:
        rows, frequencies, complex_parts = h5["data/frequency/value"].chunks
        assert rows >= 1024
        assert frequencies == 1
        assert complex_parts == 2
