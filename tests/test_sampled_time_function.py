"""Recorded transient signatures preserve clocks, amplitudes and serialization."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline

import frequensolve as fs


def recording(**kwargs):
    return fs.SampledWavelet([0.0, 0.4, 1.0, -0.2, 0.0], dt=0.013, t0=-0.037, **kwargs)


@pytest.mark.parametrize("derivative", [0, 1, 2])
def test_interpolation_matches_independent_spline(derivative):
    w = recording()
    f = fs.TimeFunction.sampled(w, amplitude=2.3, delay=0.117)
    slopes = np.r_[0.0, (w.samples[2:] - w.samples[:-2]) / (2 * w.dt), 0.0]
    ref = CubicHermiteSpline(w.times, w.samples, slopes)
    # Both knots and off-grid times, including the negative original clock.
    t = np.r_[w.times, np.linspace(w.times[0], w.times[-1], 201)]
    np.testing.assert_allclose(
        f.evaluate(t + f.delay, derivative=derivative),
        2.3 * ref(t, nu=derivative),
        rtol=2e-11,
        atol=2e-9,
    )
    np.testing.assert_array_equal(
        f.evaluate([-0.2, 0.4], derivative=derivative), [0.0, 0.0]
    )
    assert np.shape(f.evaluate(0.1, derivative=derivative)) == ()


def test_sample_values_scaling_and_fourier_normalization():
    for normalization in ("dft", "integral"):
        w = recording(normalization=normalization)
        f = fs.TimeFunction.sampled(w, delay=0.008, amplitude=3)
        np.testing.assert_allclose(
            f.evaluate(w.times + 0.008), 3 * w.samples, atol=2e-14
        )
        assert f.wavelet is w
        assert fs.TimeFunction.from_fs(f.to_fs()) == f
    physical = fs.SampledWavelet([0, 2000, 0], dt=0.01, units="N")
    with pytest.raises(ValueError, match="dimensionless"):
        fs.TimeFunction.sampled(physical)
    f = fs.TimeFunction.sampled(physical.relative_to(2, units="kN"))
    assert f.evaluate(0.01) == 1


def test_zero_extension_does_not_silently_taper_recording():
    w = fs.SampledWavelet([0.1, 1, -0.1], dt=0.01, t0=-0.02)
    f = fs.TimeFunction.sampled(w)
    np.testing.assert_allclose(f.evaluate(w.times), w.samples)
    assert f.evaluate(-0.03) == f.evaluate(0.01) == 0
    for derivative in (1, 2):
        with pytest.raises(ValueError, match="zero endpoint"):
            f.evaluate(-1, derivative=derivative)
    f.require_quiet_start(-0.03)
    with pytest.raises(ValueError, match="initially quiet"):
        f.require_quiet_start(-0.02)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "sampled"},
        {"kind": "sampled", "samples": [0, 1]},
        {"kind": "sampled", "samples": [], "dt": 0.1},
        {"kind": "sampled", "samples": [1], "dt": 0.1},
        {"kind": "sampled", "samples": [0, 1], "dt": 0},
        {"kind": "sampled", "samples": [0, float("nan")], "dt": 0.1},
        {"kind": "sampled", "samples": [0, 1], "dt": 0.1, "frequency": 1},
        {"kind": "sampled", "samples": [0, 1], "dt": 0.1, "t0": 1e20},
        {"kind": "ricker", "frequency": 10, "samples": [0, 1], "dt": 0.1},
    ],
)
def test_invalid_sampled_payload(payload):
    with pytest.raises((ValueError, TypeError)):
        fs.TimeFunction.from_fs(payload)


def test_sampled_roundtrip_and_job_fingerprint(tmp_path):
    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=2)
    sim.discretization = fs.Discretization(method="Galerkin")
    sim.model.attenuation_model = "none"
    sim.mesh.set_adapt(elems_per_wave=1, f_adapt=20)
    job = fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(0, 1, 0.1),
        source=fs.TimeFunction.sampled(recording(), delay=0.2),
    )
    job.save()
    saved = json.loads(Path(job.job_file).read_text())
    assert saved["Time"]["source"] == job.source.to_fs()
    restored = fs.BaseJob.load(job.job_file)
    assert restored.source == job.source
    before = job.task_fingerprint(1)
    changed = recording().samples.copy()
    changed[2] += 0.01
    job.source = fs.TimeFunction.sampled(
        fs.SampledWavelet(changed, dt=0.013, t0=-0.037), delay=0.2
    )
    assert job.task_fingerprint(1) != before


def test_sampled_schema_validates_payload_and_rejects_mixed_fields():
    from jsonschema import Draft202012Validator

    root = Path(__file__).parent / "contracts/sauce-time-domain-1/contracts"
    schema = json.loads((root / "inputs/fs-time-domain-1/schema.json").read_text())
    validator = Draft202012Validator(schema["$defs"]["timeFunction"])
    source = fs.TimeFunction.sampled(recording()).to_fs()
    validator.validate(source)
    for change in (
        {"dt": 0},
        {"samples": []},
        {"frequency": 40},
        {"width": 0.1},
        {"samples": ["bad", 0]},
    ):
        assert list(validator.iter_errors({**source, **change}))
    assert list(
        validator.iter_errors({"kind": "constant", "samples": [0, 1], "dt": 0.1})
    )
