"""Physical-time array contracts, convolution reference and early validation."""

import numpy as np
import pytest

import frequensolve as fs
from tests.test_transient import job_at


def test_fir_reference_and_frequency_response():
    rng = np.random.default_rng(9)
    taps = rng.normal(size=(3, 4, 23))
    samples = rng.normal(size=(2, 4, 91))
    bank = fs.ConvolutionFilters(taps, dt=0.01)
    expected = np.array(
        [
            [sum(np.convolve(x, h) for x, h in zip(batch, filt)) for filt in taps]
            for batch in samples
        ]
    )
    np.testing.assert_allclose(bank.apply(samples), expected, atol=1e-12)
    freq = np.fft.rfftfreq(expected.shape[-1], 0.01)
    response = bank.frequency_response(freq)
    predicted = np.einsum(
        "foi,bif->bof", response, np.fft.rfft(samples, n=expected.shape[-1])
    )
    np.testing.assert_allclose(np.fft.rfft(expected), predicted, atol=1e-10)


def test_sparse_delay_tail():
    bank = fs.ConvolutionFilters([[[0, 0, 2], [1, 0, 0]]], dt=0.01)
    np.testing.assert_array_equal(bank.apply([[1, 2], [3, 4]]), [[3, 4, 2, 4]])


@pytest.mark.parametrize(
    "taps,dt",
    [
        ([[[1j]]], 0.1),
        ([[[np.nan]]], 0.1),
        ([[[]]], 0.1),
        ([[[1]]], 0),
        ([[[1]]], True),
    ],
)
def test_invalid_filters(taps, dt):
    with pytest.raises(ValueError):
        fs.ConvolutionFilters(taps, dt=dt)


def test_source_filter_inline_roundtrip():
    original = fs.SourceEncoding.filtered([[[1, 0], [0, 1]]], dt=0.1, names=["beam"])
    restored = fs.SourceEncoding.from_fs(original.to_fs())
    np.testing.assert_array_equal(restored.filters.taps, original.filters.taps)
    assert restored.field_names() == ["beam"]


def test_filtered_job_roundtrip(tmp_path):
    job = job_at(tmp_path)
    acq = fs.Acquisition()
    acq.add_sources(kind="pressure_monopole", coords=[[0.2, 0.2], [0.4, 0.2]])
    acq.source_encoding = fs.SourceEncoding.filtered([[[1, 0], [0, 1]]], dt=0.1)
    receiver = fs.EncodedReceiver(
        components=[fs.ReceiverComponent(name="p", field="pressure")],
        filters=fs.ConvolutionFilters([[[[1, 0], [0, -1]]]], dt=0.1),
    )
    acq.add_receiver_group("beam", receiver, [[0.2, 0.3], [0.4, 0.3]])
    job.simulation.acquisition = acq
    job.save()
    restored = fs.BaseJob.load(job.job_file)
    restored.validate_outputs()
    np.testing.assert_array_equal(
        restored.simulation.acquisition.source_encoding.filters.taps,
        acq.source_encoding.filters.taps,
    )
    np.testing.assert_array_equal(
        restored.simulation.acquisition.receiver_groups[0].device.filters.taps,
        receiver.filters.taps,
    )


@pytest.mark.parametrize("weights", [[[1j, 1]], [[1, 1j]]])
def test_transient_rejects_complex_source_weights(tmp_path, weights):
    job = job_at(tmp_path)
    acq = fs.Acquisition()
    acq.add_sources(kind="pressure_monopole", coords=[[0.2, 0.2], [0.4, 0.2]])
    acq.encode_sources(weights)
    job.simulation.acquisition = acq
    with pytest.raises(ValueError, match="source weights.*real"):
        job.validate_outputs()


def test_transient_rejects_complex_receiver_weights(tmp_path):
    job = job_at(tmp_path)
    acq = fs.Acquisition()
    acq.add_sources(kind="pressure_monopole", coords=[[0.2, 0.2]])
    acq.add_receiver_group(
        "beam",
        fs.EncodedReceiver(
            components=[fs.ReceiverComponent(name="p", field="pressure")],
            weights=[[1, 1j]],
        ),
        [[0.2, 0.3], [0.4, 0.3]],
    )
    job.simulation.acquisition = acq
    with pytest.raises(ValueError, match="receiver weights.*real"):
        job.validate_outputs()


@pytest.mark.parametrize("dt", [0.2, 0.1001])
def test_receiver_filter_clock_must_match_solver(tmp_path, dt):
    job = job_at(tmp_path)
    acq = fs.Acquisition()
    acq.add_sources(kind="pressure_monopole", coords=[[0.2, 0.2]])
    device = fs.EncodedReceiver(
        components=[fs.ReceiverComponent(name="p", field="pressure")],
        filters=fs.ConvolutionFilters([[[[1]]]], dt=dt),
    )
    acq.add_receiver_group("beam", device, [[0.2, 0.3]])
    job.simulation.acquisition = acq
    with pytest.raises(ValueError, match="timestep"):
        job.validate_outputs()


def test_source_filter_clock_can_differ(tmp_path):
    job = job_at(tmp_path)
    acq = fs.Acquisition()
    acq.add_sources(kind="pressure_monopole", coords=[[0.2, 0.2]])
    acq.source_encoding = fs.SourceEncoding.filtered([[[0, 1]]], dt=0.032)
    job.simulation.acquisition = acq
    job.validate_outputs()
    frequency_job = fs.FrequencyDomainJob("frequency", job.simulation, [10])
    with pytest.raises(ValueError, match="TransientJob"):
        frequency_job.validate_outputs()


def test_frequency_encoding_rejected(tmp_path):
    job = job_at(tmp_path)
    acq = fs.Acquisition()
    acq.add_sources(kind="pressure_monopole", coords=[[0.2, 0.2]])
    acq.source_encoding = fs.SourceEncoding.frequency_dense([[[1]], [[1]]], [1, 2])
    job.simulation.acquisition = acq
    with pytest.raises(ValueError, match="frequency-dependent"):
        job.validate_outputs()


def test_external_filter_shape_not_silently_reshaped(tmp_path):
    import h5py

    path = tmp_path / "filters.h5"
    with h5py.File(path, "w") as h5:
        h5["filters"] = np.ones((2, 1, 3))
    filters = fs.ConvolutionFilters.from_fs(
        dict(dt=0.1, shape=[1, 2, 3], file=str(path), dataset="filters")
    )
    with pytest.raises(ValueError, match="shape"):
        _ = filters.taps


def test_filter_owns_immutable_coefficients():
    taps = np.ones((1, 2, 3))
    bank = fs.ConvolutionFilters(taps, dt=0.1)
    taps[:] = 0
    assert np.all(bank.taps == 1)
    with pytest.raises(ValueError):
        bank.taps[0, 0, 0] = 2


@pytest.mark.parametrize(
    "shape,dt",
    [([1, 2, 1.5], 0.1), ([1, True, 2], 0.1), ([1, 2, 3], True), ([1, 2, 3], "0.1")],
)
def test_external_metadata_is_validated(shape, dt):
    with pytest.raises(ValueError):
        fs.ConvolutionFilters.from_fs(
            dict(shape=shape, dt=dt, file="missing.h5", dataset="filters")
        )


def test_causal_time_reversal_requires_explicit_time_origin():
    source = fs.SourceEncoding.filtered([[[1, 2]]], dt=0.1)
    with pytest.raises(ValueError, match="time origin"):
        source.time_reversed()
    receiver = fs.EncodedReceiver(
        components=[fs.ReceiverComponent(name="p", field="pressure")],
        filters=fs.ConvolutionFilters([[[[1, 2]]]], dt=0.1),
    )
    with pytest.raises(ValueError, match="time origin"):
        receiver.time_reversed()
    with pytest.raises(ValueError, match="new filter bank"):
        receiver.add_encoding("extra", [1])
