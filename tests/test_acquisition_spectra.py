"""Acquisition-clock, transform, serialization, and evaluation contracts."""

import copy
import subprocess
import sys

import h5py
import numpy as np
import pytest
import xarray as xr

from frequensolve.seismic import (
    Acquisition,
    GainDelay,
    RickerWavelet,
    SampledWavelet,
    SourceGeometry,
    SourceSignature,
    TabulatedSpectrum,
)
from frequensolve.util.mixins import ExportContext


@pytest.mark.parametrize("count", [1, 7, 8, 201])
@pytest.mark.parametrize("normalization", ["dft", "integral"])
def test_sampled_transform_matches_explicit_sum(count, normalization):
    data = np.random.default_rng(count).normal(size=count)
    w = SampledWavelet(data, dt=0.013, t0=-0.037, normalization=normalization)
    for f in [w.frequencies, np.array([0.0, 0.72, 2.137, 20.0])]:
        for laplace in [0.0, -0.23]:
            kernel = np.exp(-2j * np.pi * w.times[:, None] * (f + 1j * laplace))
            for derivative in [False, True]:
                work = data * (-2j * np.pi * w.times) if derivative else data
                expected = np.einsum("t,tf->f", work, kernel)
                if normalization == "integral":
                    expected *= w.dt
                np.testing.assert_allclose(
                    w.at_frequencies(
                        f, laplace=laplace, derivative=derivative, batch_bytes=64
                    ),
                    expected,
                    atol=2e-12,
                )


def test_delay_phase_and_product_derivative():
    impulse = SampledWavelet([0.0, 3.0, 0.0, 0.0], dt=0.02, t0=-0.01)
    response = impulse * GainDelay(gain=2 - 1j, delay=0.037)
    f = np.array([0.3, 2.5, 8.0])
    for laplace in [0.0, -0.7]:
        expected = 3 * (2 - 1j) * np.exp(-2j * np.pi * (f + 1j * laplace) * 0.047)
        np.testing.assert_allclose(
            response.at_frequencies(f, laplace=laplace), expected
        )
        df = response.at_frequencies(f, laplace=laplace, derivative=True)
        np.testing.assert_allclose(df, expected * (-2j * np.pi * 0.047))
        eps = 1e-5
        finite_difference = (
            response.at_frequencies(f + eps, laplace=laplace)
            - response.at_frequencies(f - eps, laplace=laplace)
        ) / (2 * eps)
        np.testing.assert_allclose(df, finite_difference, rtol=1e-8)


def test_recording_is_immutable_and_retains_clock():
    data = np.array([2.0, -1.0, 0.0])
    w = SampledWavelet(data, dt=0.1, t0=-0.2)
    data[:] = 7
    np.testing.assert_array_equal(w.samples, [2.0, -1.0, 0.0])
    with pytest.raises(ValueError):
        w.samples[0] = 1
    with pytest.raises(ValueError):
        w.samples.setflags(write=True)
    with pytest.raises(AttributeError):
        w.dt = 0.2
    np.testing.assert_allclose(w.times, [-0.2, -0.1, 0.0])


def test_analytical_sampling_has_no_assignment_side_effects():
    w = RickerWavelet(10)
    a = w.sample(np.arange(100) * 0.001)
    b = w.sample(np.arange(300) * 0.002)
    assert w.times is None
    assert len(a.samples) == 100
    assert len(b.samples) == 300
    w.times = np.arange(101) * 0.002
    before = w.times.copy()
    _ = w.spectrum
    w.signal = np.zeros_like(w.signal)
    np.testing.assert_array_equal(w.spectrum, 0.0)
    np.testing.assert_array_equal(w.frequencies, np.fft.rfftfreq(len(w.signal), 0.002))
    _ = w.sample(np.arange(50) * 0.01)
    np.testing.assert_array_equal(w.times, before)
    np.testing.assert_array_equal(w.signal, 0.0)


def test_import_sampled_formats_and_explicit_strength(tmp_path):
    da = xr.DataArray(
        [1000.0, 2000.0, 0.0],
        dims="time",
        coords={"time": [-0.1, 0.0, 0.1]},
        attrs={"units": "N"},
    )
    w = SampledWavelet.from_xarray(da)
    with pytest.raises(ValueError, match="dimensionless"):
        SourceSignature(w, frequencies=[0, 1])
    relative = w.relative_to(2.0, units="kN")
    np.testing.assert_array_equal(relative.samples, [0.5, 1.0, 0.0])
    assert relative.t0 == -0.1
    np.save(tmp_path / "w.npy", relative.samples)
    loaded = SampledWavelet.from_npy(tmp_path / "w.npy", dt=0.1, t0=-0.1)
    np.testing.assert_allclose(loaded.spectrum, relative.spectrum)


def test_tabulated_policies_and_derivatives():
    exact = TabulatedSpectrum([1, 2, 4], [1j, 2j, 4j], derivatives=[1j] * 3)
    np.testing.assert_array_equal(
        exact.at_frequencies([4, 1], derivative=True), [1j, 1j]
    )
    with pytest.raises(ValueError, match="Exact"):
        exact.at_frequencies([1.5])
    with pytest.raises(ValueError, match="Laplace"):
        exact.at_frequencies([1], laplace=-0.1)
    linear = TabulatedSpectrum([1, 2, 4], [1j, 3j, 5j], interpolation="linear")
    np.testing.assert_array_equal(linear.at_frequencies([1.5, 3]), [2j, 4j])
    np.testing.assert_array_equal(
        linear.at_frequencies([1.5, 2, 4], derivative=True), [2j, 1j, 1j]
    )
    with pytest.raises(ValueError, match="band"):
        linear.at_frequencies([0])
    with pytest.raises(ValueError, match="derivatives"):
        TabulatedSpectrum([1], [1]).at_frequencies([1], derivative=True)


def test_signature_export_values_derivatives_and_ids(tmp_path):
    a = SampledWavelet([0, 2, 0], dt=0.01, t0=-0.005)
    b = a * GainDelay(gain=-0.3 + 0.4j, delay=0.023)
    signature = SourceSignature(
        {1: a, 2: b, 3: a}, frequencies=[0, 2, 7], laplace=[0, -0.4], batch_bytes=64
    )
    ctx = ExportContext(tmp_path)
    spec = signature.to_fs(ctx, source_count=3)
    with h5py.File(tmp_path / spec["file"]) as h5:
        np.testing.assert_array_equal(h5["source_ids"], [1, 2, 3])
        assert h5["q"].shape == (2, 3, 3, 2)
        for d, laplace in enumerate(signature.laplace):
            for column, signal in enumerate([a, b, a]):
                for key in ["q", "q_f"]:
                    raw = h5[key][d, :, column]
                    actual = raw[:, 0] + 1j * raw[:, 1]
                    expected = signal.at_frequencies(
                        signature.frequencies, laplace=laplace, derivative=key == "q_f"
                    )
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert copy.deepcopy(signature).to_fs(ctx, source_count=3) == spec
    with pytest.raises(ValueError, match="IDs"):
        signature.to_fs(ctx, source_count=2)


def test_signature_export_failure_is_not_published(tmp_path):
    bad = SourceSignature(TabulatedSpectrum([1], [1]), frequencies=[1])
    with pytest.raises(ValueError, match="derivatives"):
        bad.to_fs(ExportContext(tmp_path), source_count=1)
    assert list((tmp_path / "source-spectra").iterdir()) == []


def test_signature_acquisition_round_trip(tmp_path):
    sig = SourceSignature(GainDelay(delay=0.02), frequencies=[1, 2])
    geometry = SourceGeometry.points(coords=[[0.0, 0.0], [1.0, 0.0]], kind="scalar")
    acquisition = Acquisition(source_geometry=geometry, source_signature=sig)
    ctx = ExportContext(tmp_path)
    payload = acquisition.to_fs(ctx)
    assert "source_signature" in payload
    assert Acquisition.from_fs(payload).to_fs(ctx) == payload


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dt=0),
        dict(dt=np.inf),
        dict(dt=0.1, t0=np.nan),
        dict(dt=0.1, normalization="peak"),
    ],
)
def test_invalid_recording_clock(kwargs):
    with pytest.raises(ValueError):
        SampledWavelet([1, 0], **kwargs)


def test_spectra_public_import_order():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import frequensolve.simulation; from frequensolve.seismic import SampledWavelet, SourceSignature, GainDelay",
        ],
        check=True,
        timeout=30,
    )


def test_receiver_response_export_is_before_encoding_and_does_not_mutate(tmp_path):
    from frequensolve.seismic import (
        ReceiverComponent,
        ReceiverGroup,
        ReceiverNode,
        ReceiverResponse,
    )

    response = ReceiverResponse(
        {1: GainDelay(gain=2, delay=0.01), 2: GainDelay(gain=-1, delay=0.03)},
        frequencies=[1, 3],
    )
    device = ReceiverNode(
        components=[ReceiverComponent(name="p", field="p", weight=2j)],
        transfer=response,
    )
    group = ReceiverGroup(
        name="r", device=device, coordinates=np.array([[0.0, 0.0], [1.0, 0.0]])
    )
    spec = group.to_fs(ExportContext(tmp_path))
    component = spec["device"]["components"][0]
    assert component["weight"] == [0.0, 2.0]
    assert "receiver_ids_dataset" in component["transfer"]
    with h5py.File(tmp_path / component["transfer"]["file"]) as h5:
        values = h5["q"][...]
        actual = values[..., 0] + 1j * values[..., 1]
        for i in range(2):
            np.testing.assert_allclose(
                actual[:, i], response.signals[i + 1].at_frequencies([1, 3]), rtol=1e-6
            )
    assert device.transfer is response
    assert device.components[0].response is None
    assert ReceiverGroup.from_fs(spec).to_fs(ExportContext(tmp_path)) == spec


def test_signature_hash_is_independent_of_block_size(tmp_path):
    ctx = ExportContext(tmp_path)
    signal = SampledWavelet(np.arange(8), dt=0.01)
    small = SourceSignature(signal, frequencies=[0, 1, 2, 3], batch_bytes=32).to_fs(
        ctx, source_count=3
    )
    large = SourceSignature(signal, frequencies=[0, 1, 2, 3], batch_bytes=1024).to_fs(
        ctx, source_count=3
    )
    assert small["hash"] == large["hash"]


def test_time_gather_from_recorded_signature_and_double_filter_guard(tmp_path):
    from frequensolve.seismic import TraceDataset

    path = tmp_path / "traces.h5"
    samples = np.zeros(32)
    samples[5] = 2.0
    waveform = SampledWavelet(samples, dt=0.01)
    spectrum = waveform.spectrum
    with h5py.File(path, "w") as h5:
        h5["frequency"] = waveform.frequencies
        h5["laplace"] = np.zeros(len(spectrum))
        strings = h5py.string_dtype()
        h5["survey/packed_layout_kind"] = np.asarray(
            ["packed_frequency_trace_v1"], dtype=strings
        )
        h5["survey/source_signature_hash"] = np.asarray(["sha256:test"], dtype=strings)
        data = np.zeros((len(spectrum), 1, 1, 1, 2), dtype="f4")
        data[:, 0, 0, 0, 0], data[:, 0, 0, 0, 1] = spectrum.real, spectrum.imag
        ds = h5.create_dataset("r", data=data)
        ds.attrs["dims"] = ["receiver", "component", "shot", "frequency"]
        ds.attrs["component"] = ["p"]
        ds.attrs["shot"] = [1]
        ds.attrs["receiver"] = [1]
    traces = TraceDataset.open(path)
    raw = traces.fd("r", "p", source=1)
    assert raw.attrs["source_signature_applied"]
    td = traces.td("r", "p", source=1)
    np.testing.assert_allclose(td.values[:, 0], samples, atol=1e-7)
    np.testing.assert_allclose(td.time.values, waveform.times)
    with pytest.raises(ValueError, match="already include"):
        traces.fd("r", "p", source=1, wavelet=waveform)
    extra = traces.fd(
        "r", "p", source=1, wavelet=waveform, allow_additional_filter=True
    )
    assert extra.attrs["source_signature_applied"]
    np.testing.assert_allclose(extra.values[:, 0], spectrum**2, rtol=1e-6, atol=1e-6)


def test_segy_wavelet_retains_recording_delay_and_units(tmp_path):
    segyio = pytest.importorskip("segyio")
    path = tmp_path / "wavelet.sgy"
    spec = segyio.spec()
    spec.format, spec.sorting, spec.tracecount = 5, 2, 1
    spec.samples = np.arange(8)
    with segyio.create(str(path), spec) as file:
        file.trace[0] = np.arange(8, dtype="f4")
        file.header[0] = {
            segyio.TraceField.TRACE_SAMPLE_INTERVAL: 2000,
            segyio.TraceField.DelayRecordingTime: -10,
        }
    wavelet = SampledWavelet.from_segy(path, units="N")
    assert wavelet.dt == 0.002 and wavelet.t0 == -0.01
    np.testing.assert_array_equal(wavelet.samples, np.arange(8))


def test_analytical_fractional_center_has_exact_phase_on_fft_grid():
    times = np.arange(512) * 0.002
    zero = RickerWavelet(12, center=0).sample(times)
    shifted = RickerWavelet(12, center=0.0373).sample(times)
    expected = zero.spectrum * np.exp(-2j * np.pi * zero.frequencies * 0.0373)
    # Nyquist must remain real for a sampled real signal; check interior bins.
    np.testing.assert_allclose(shifted.spectrum[:-1], expected[:-1], atol=2e-13)


def test_invalid_spectral_inputs_and_unsupported_damping():
    with pytest.raises(ValueError):
        SampledWavelet([1, np.nan], dt=0.1)
    with pytest.raises(ValueError):
        SampledWavelet([1 + 1j], dt=0.1)
    with pytest.raises(ValueError, match="Nyquist"):
        SampledWavelet([1, 0], dt=0.1).at_frequencies([5.001])
    with pytest.raises(ValueError, match="nonpositive"):
        GainDelay().at_frequencies([1], laplace=0.1)
    with pytest.raises(ValueError):
        SourceSignature(GainDelay(), frequencies=[1, 1])
    with pytest.raises(ValueError):
        SourceSignature(GainDelay(), frequencies=[1], laplace=[0, 0])


def test_copy_preserves_recording_immutability():
    original = SampledWavelet([1.0, 2.0], dt=0.1)
    copied = copy.deepcopy(original)
    with pytest.raises(ValueError):
        copied.samples[0] = 7
    assert copied.describe() == original.describe()


def test_plot_does_not_evaluate_or_regrid_definition(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    definition = RickerWavelet(10)
    figures = definition.plot(T_max=0.5)
    assert definition.times is None and definition.signal is None
    for figure in figures:
        plt.close(figure)


def test_converter_retains_acquisition_provenance(tmp_path):
    from frequensolve.seismic import convert_traces

    gather = xr.DataArray(
        np.ones((2, 4)),
        dims=("receiver", "time"),
        coords={"time": np.arange(4) * 0.1},
        name="p",
        attrs={
            "source_signature_applied": True,
            "source_signature_hash": "sha256:test",
            "receiver_response_applied": True,
            "units": "Pa",
        },
    )
    output = convert_traces(gather, tmp_path / "observed.h5")
    import json

    with h5py.File(output) as h5:
        provenance = json.loads(h5.attrs["conversion"])
    assert provenance["acquisition"]["source_signature_hash"] == "sha256:test"
    assert provenance["acquisition"]["units"] == "Pa"


def test_native_coordinate_tolerance_is_validated_before_export():
    with pytest.raises(ValueError, match="ambiguous"):
        SourceSignature(GainDelay(), frequencies=[1, 1.0000001])
    with pytest.raises(ValueError, match="ambiguous"):
        SourceSignature(GainDelay(), frequencies=[1], laplace=[-0.1, -0.100000001])


def test_sdk_artifacts_against_native_spectral_probe(tmp_path):
    """Optional real SDK-to-Sauce boundary, including receiver group serialization."""
    import json
    import os
    from pathlib import Path

    from frequensolve.seismic import (
        ReceiverComponent,
        ReceiverGroup,
        ReceiverNode,
        ReceiverResponse,
    )

    probe = os.environ.get("FS_ACQUISITION_SPECTRA_PROBE")
    if not probe:
        pytest.skip("Set FS_ACQUISITION_SPECTRA_PROBE to a built native probe")
    signals = {
        1: SampledWavelet([0, 2, -0.3, 0.1], dt=0.02, t0=-0.01)
        * GainDelay(delay=0.007),
        2: GainDelay(gain=-0.7 + 0.2j, delay=0.061),
    }
    frequencies, damping = [1.3, 3.7], [0.0, -0.23]
    ctx = ExportContext(project_path=tmp_path)
    signature = SourceSignature(
        signals, frequencies=frequencies, laplace=damping
    ).to_fs(ctx, source_count=2)
    receivers = ReceiverGroup(
        name="r",
        coordinates=np.array([[0.0, 0.0], [1.0, 0.0]]),
        device=ReceiverNode(
            components=[
                ReceiverComponent(
                    name="stress", field="elastic:stress_all", weight=0.5 + 0.2j
                )
            ],
            transfer=ReceiverResponse(
                signals, frequencies=frequencies, laplace=damping
            ),
        ),
    )
    receiver = receivers.to_fs(ctx)
    # Standalone probe has no project file; resolve exported references explicitly.
    signature["file"] = str(tmp_path / signature["file"])
    receiver["device"]["components"][0]["transfer"]["file"] = str(
        tmp_path / receiver["device"]["components"][0]["transfer"]["file"]
    )
    # Coordinates may also be materialized by the SDK.
    if "file" in receiver["coordinates"]:
        receiver["coordinates"]["file"] = str(
            tmp_path / receiver["coordinates"]["file"]
        )
    payload = {
        "signature": signature,
        "receiver": receiver,
        "geometry": {
            "_type": "Inline",
            "kind": "scalar",
            "sources": [
                {"coordinates": [0.0, 0.0], "name": "a"},
                {"coordinates": [1.0, 0.0], "name": "b"},
            ],
        },
        "encoding": {
            "_type": "JsonDense",
            "fields": [{"name": "mixed", "coefficients": [[0.2, 0.3], [-0.1, 0.7]]}],
        },
    }
    file = tmp_path / "input.json"
    file.write_text(json.dumps(payload))
    for laplace in damping:
        for f in frequencies:
            result = subprocess.run(
                [str(Path(probe).resolve()), str(file), str(f), str(laplace)],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            assert "ACQUISITION_SPECTRA_OK" in result.stdout
            rows = dict(
                line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
            )
            for i, signal in signals.items():
                expected = np.array(
                    [
                        signal.at_frequencies([f], laplace=laplace)[0],
                        signal.at_frequencies([f], laplace=laplace, derivative=True)[0],
                    ]
                )
                for key, factor in [
                    ("Q", 1),
                    ("C", [0.2 + 0.3j, -0.1 + 0.7j][i - 1]),
                    ("R", 0.5 + 0.2j),
                ]:
                    raw = list(map(float, rows[f"{key}_{i}"].split()))
                    np.testing.assert_allclose(
                        [complex(*raw[:2]), complex(*raw[2:])],
                        factor * expected,
                        rtol=2e-6,
                        atol=1e-7,
                    )


def test_xarray_recording_clock_units_are_converted():
    recording = xr.DataArray(
        [1.0, 0.0, -1.0], dims="time", coords={"time": [-2.0, 0.0, 2.0]}
    )
    recording.time.attrs["units"] = "ms"
    wavelet = SampledWavelet.from_xarray(recording)
    assert wavelet.dt == 0.002
    assert wavelet.t0 == -0.002


def test_large_distinct_signature_bank_uses_scalable_metadata(tmp_path):
    # Description bytes exceed the HDF5 compact-attribute limit.
    bank = {i: GainDelay(delay=i * 0.001) for i in range(1, 1501)}
    result = SourceSignature(bank, frequencies=[1.3]).to_fs(
        ExportContext(project_path=tmp_path), source_count=len(bank)
    )
    with h5py.File(tmp_path / result["file"]) as h5:
        assert len(h5["signal_definitions"]) == len(bank)
        np.testing.assert_array_equal(
            h5["signal_definition_ids"], np.arange(1, len(bank) + 1)
        )
        packed = h5["q"][0]
        np.testing.assert_allclose(
            packed[:, 0] + 1j * packed[:, 1],
            np.exp(-2j * np.pi * 1.3 * np.arange(1, len(bank) + 1) * 0.001),
            atol=1e-7,
        )


@pytest.mark.parametrize("kind", ["component", "node", "array", "encoded", "fiber"])
def test_transfer_and_reserved_response_are_independent(kind):
    import dataclasses
    import inspect
    import warnings

    from frequensolve.seismic import (
        EncodedReceiver,
        ReceiverArray,
        ReceiverComponent,
        ReceiverFiber,
        ReceiverNode,
        ReceiverResponse,
        ReceiverTransferFunction,
        SpectralResponse,
        TransferFunction,
    )

    assert TransferFunction is SpectralResponse
    assert ReceiverTransferFunction is ReceiverResponse
    constructors = {
        "component": (ReceiverComponent, dict(field="pressure")),
        "node": (ReceiverNode, {}),
        "array": (ReceiverArray, dict(offsets=[[0, 0], [1, 0]])),
        "encoded": (
            EncodedReceiver,
            dict(
                weights=np.ones((1, 2)),
                components=[ReceiverComponent(field="pressure")],
            ),
        ),
        "fiber": (ReceiverFiber, dict(gauge_length=10)),
    }
    cls, options = constructors[kind]
    h = ReceiverTransferFunction(GainDelay(delay=0.01), frequencies=[1, 2])
    reserved = {"future_reciprocal_survey": "reference"}
    assert "transfer" in inspect.signature(cls).parameters
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        item = cls(**options, response=reserved)
        assert item.transfer is None
        item.transfer = h
        assert item.response is reserved
        item.response = None
        assert item.transfer is h
        item.response = reserved
        cloned = copy.deepcopy(item)
        assert cloned.response == reserved
        assert cloned.transfer is not cloned.response
        if kind != "fiber":
            assert dataclasses.replace(item, transfer=None).response is reserved
        assert not emitted


def test_reserved_response_has_no_calibration_effect_or_warning(tmp_path):
    import warnings

    from frequensolve.seismic import (
        ReceiverComponent,
        ReceiverGroup,
        ReceiverNode,
        ReceiverTransferFunction,
    )

    h = ReceiverTransferFunction(GainDelay(gain=2j, delay=0.01), frequencies=[1, 2])
    # Even an old-shaped value is inert: it must not export an artifact or filter.
    component = ReceiverComponent(field="pressure", name="p", response=h)
    device = ReceiverNode(components=[component], response=h)
    group = ReceiverGroup(
        name="r", coordinates=np.array([[0.0, 0.0], [1.0, 0.0]]), device=device
    )
    ctx = ExportContext(project_path=tmp_path)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        payload = group.to_fs(ctx)
        assert "transfer" not in payload["device"]["components"][0]
        assert "response" not in payload["device"]["components"][0]
        assert not (tmp_path / "receiver-spectra").exists()
        component.transfer = h
        payload = group.to_fs(ctx)
        assert "transfer" in payload["device"]["components"][0]
        assert not emitted
    reserved = copy.deepcopy(payload)
    reserved["device"]["components"][0]["response"] = {"future": "reciprocal"}
    loaded = ReceiverGroup.from_fs(reserved)
    assert loaded.device.components[0].response == {"future": "reciprocal"}
    assert loaded.to_fs(ctx) == payload


def test_add_component_keeps_response_reserved():
    from frequensolve.seismic import ReceiverNode, ReceiverTransferFunction

    h = ReceiverTransferFunction(GainDelay(), frequencies=[1])
    node = ReceiverNode()
    component = node.add_component("a", "pressure", transfer=h)
    assert component.transfer is h
    assert component.response is None
    component.response = "reserved"
    assert component.transfer is h
