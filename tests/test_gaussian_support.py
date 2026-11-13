"""Spatial supports preserve one source identity and serialize physical parameters."""

import math

import pytest

from frequensolve.seismic.sources import GaussianSupport, PointSource, SourceGeometry
from frequensolve.units import ureg


def test_round_trip_and_single_source():
    source = PointSource(
        "shot",
        coordinates=[0, 0] * ureg.m,
        kind="vector",
        direction=[0, 1],
        amplitude=1e6 * ureg.N,
        domain=2,
        support=GaussianSupport(2 * ureg.m),
    )
    payload = source.to_fs()
    assert PointSource.from_fs(payload).to_fs() == payload
    assert payload["support"]["sigma"] == {"value": 2, "units": "m"}
    geometry = SourceGeometry.inline(kind="vector", sources=[source], domain=2)
    data = geometry.to_fs()
    assert len(data["sources"]) == 1
    assert data["domain"] == 2
    assert data["sources"][0]["name"] == "shot"
    assert "domain" not in data["sources"][0]


def test_surface_and_width_units():
    support = GaussianSupport(200 * ureg.cm, surface="seabed")
    assert support.to_fs()["surface"] == "seabed"
    assert GaussianSupport.from_fs(support.to_fs()).to_fs() == support.to_fs()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sigma": 2},
        {"sigma": 0 * ureg.m},
        {"sigma": -1 * ureg.m},
        {"sigma": math.nan * ureg.m},
        {"sigma": [1, 2] * ureg.m},
        {"sigma": 1 * ureg.s},
        {"sigma": 1 * ureg.m, "cutoff": 4},
        {"sigma": 1 * ureg.m, "cutoff": math.inf},
        {"sigma": 1 * ureg.m, "boundary": "drop"},
        {"sigma": 1 * ureg.m, "surface": ""},
        {"sigma": 1 * ureg.m, "surface": "top", "boundary": "reject"},
        {"sigma": 1 * ureg.m, "tolerance": 1e-8},
    ],
)
def test_reject_invalid_support(kwargs):
    with pytest.raises((ValueError, TypeError)):
        GaussianSupport(**kwargs)


def test_dirac_unchanged_and_ambiguous_support_rejected():
    source = PointSource([0, 0] * ureg.m, kind="vector")
    assert "support" not in source.to_fs()
    source.support = GaussianSupport(2 * ureg.m)
    source.extra["footprint"] = {"surface": "seabed"}
    with pytest.raises(ValueError, match="support or footprint"):
        source.to_fs()


def test_bulk_convenience_preserves_source_names_and_defaults():
    geometry = SourceGeometry.points(
        kind="vector",
        coords=[[0, 0], [1, 1]],
        units="m",
        names=["a", "b"],
        direction=[0, 1],
        amplitude=1e6 * ureg.N,
        support=GaussianSupport(2 * ureg.m),
    )
    data = geometry.to_fs()
    assert [p["name"] for p in data["sources"]] == ["a", "b"]
    assert all(p["support"]["sigma"]["value"] == 2 for p in data["sources"])
    assert data["defaults"]["amplitude"]["value"] == 1e6


def test_transient_auto_bandwidth_and_dirac_optout(tmp_path):
    import frequensolve as fs

    project = fs.Project(name="auto", path=tmp_path / "auto")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=2)
    sim.mesh.set_adapt(elems_per_wave=1, f_adapt=40)
    sim.acquisition.set_sources(
        fs.SourceGeometry.points(
            kind="vector", coords=[[0, 0]], units="m", direction=[0, 1]
        )
    )
    original_sources = sim.acquisition.sources.to_fs()
    source = fs.TimeFunction.ricker(15, delay=0.2)
    job = fs.TransientJob(
        "auto", sim, time=fs.TimeIntegration(0, 2, 0.004), source=source
    )
    assert job._time_payload()["point_source"] == dict(
        type="gaussian", f_max=45, amplitude_ratio=0.98
    )
    job.time = fs.TimeIntegration(0, 2, 0.004, f_max=60)
    assert job._time_payload()["point_source"]["f_max"] == 60
    job.point_source = "dirac"
    assert job._time_payload()["point_source"] == {"type": "dirac"}
    assert sim.acquisition.sources.to_fs() == original_sources


def test_sampled_auto_bandwidth_preserves_samples(tmp_path):
    import numpy as np

    import frequensolve as fs

    project = fs.Project(name="sampled", path=tmp_path / "sampled")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=2)
    sim.mesh.set_adapt(elems_per_wave=1, f_adapt=10)
    t = np.arange(2001) * 0.001
    samples = np.sin(2 * np.pi * 30 * t) * np.exp(-(((t - 1) / 0.2) ** 2))
    wavelet = fs.SampledWavelet(samples=samples, dt=0.001)
    job = fs.TransientJob(
        "auto",
        sim,
        time=fs.TimeIntegration(0, 2, 0.004),
        source=fs.TimeFunction.sampled(wavelet),
    )
    frequency = job._time_payload()["point_source"]["f_max"]
    assert 30 < frequency < 40
    np.testing.assert_array_equal(job.source.wavelet.samples, samples)


def test_one_sigma_default_preserves_normalized_support_contract():
    support = GaussianSupport(2 * ureg.m)
    assert support.cutoff == 1
    assert support.to_fs()["cutoff"] == 1
    assert support.boundary == "truncate_renormalize"


def test_lightweight_sampling_default_and_explicit_adaptive_tolerance():
    default = GaussianSupport(2 * ureg.m)
    assert default.tolerance is None
    assert "tolerance" not in default.to_fs()
    accurate = GaussianSupport(2 * ureg.m, tolerance=1e-4)
    assert accurate.to_fs()["tolerance"] == 1e-4
    assert GaussianSupport.from_fs(accurate.to_fs()).to_fs() == accurate.to_fs()
