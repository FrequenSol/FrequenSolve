# Copyright (c) 2023 - 2026 FrequenSol, LLC. All rights reserved.
# Proprietary and confidential. Unauthorized use, copying, modification,
# or distribution is prohibited except under a written license.

"""HV authoring and first-order workflow contract; native numerics have an integration test."""

import h5py
import numpy as np
import pytest

from frequensolve import imaging as im
from frequensolve.units import ureg
from tests.imaging_fakes import FakeImagingSite
from tests.test_imaging_problem import _problem

pytestmark = pytest.mark.unit


def test_hv_contract_uses_complete_waveform_and_fixed_physical_scale():
    config = im.HV(2 * ureg.Pa, lambda_=0, spatial_intervals=3)
    payload = config.payload(
        [im.ObservedGroup("line", "/observed.h5", source_basis="source_geometry")]
    )
    term = payload["objective_terms"][0]
    assert term["objective"]["kind"] == "hv"
    assert term["objective"]["lambda"] == 0
    assert term["objective"]["discretization"]["spatial_intervals"] == 3
    assert term["normalization"] == {
        "scale": {"kind": "explicit", "value": {"value": 2, "units": "Pa"}},
        "reduction": "sum",
    }
    assert payload["preprocess"] == {"include_defaults": False, "hooks": []}
    assert payload["receiver_groups"][0]["observed_source_basis"] == "source_geometry"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(scale=0),
        dict(kappa=-1),
        dict(epsilon=0),
        dict(order=0),
        dict(order=np.inf),
        dict(spatial_intervals=1.5),
        dict(max_refinements=-1),
        dict(max_iterations=True),
        dict(relative_stationarity_tolerance=np.nan),
    ],
)
def test_hv_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        im.HV(**({"scale": 1} | kwargs))


def test_hv_rejects_unsupported_controls_and_encoding_before_submission(tmp_path):
    fake = FakeImagingSite()
    _, problem = _problem(
        tmp_path,
        fake,
        controls=im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", count=3), source=im.SourceParameters()
        ),
    )
    with pytest.raises(ValueError, match="material-only"):
        problem.hv(im.HV(1))
    _, problem = _problem(tmp_path, fake)
    problem.simulation.acquisition.encode_sources(np.eye(2))
    with pytest.raises(ValueError, match="physical sources"):
        problem.hv(im.HV(1))


def test_hv_first_order_fwi_preserves_views_and_cache(tmp_path, monkeypatch):
    _, problem = _problem(tmp_path, FakeImagingSite())
    hv = problem.hv(im.HV(1))
    captured, jobs = {}, []
    original_sync = im.ImagingProblem._sync_simulation

    def sync(view, state):
        captured["state"] = state
        original_sync(view, state)

    def run(job):
        # A quadratic oracle tests orchestration, independently of HV's numerics.
        jobs.append(job)
        state = captured["state"]
        weight = sum(job.weights) if job.weights is not None else len(job.f_list)
        blocks = {
            name: value
            for name, value in state.blocks().items()
            if name.removeprefix("model.") in job.active
        }
        im.ControlVectorFile(
            {name: weight * value for name, value in blocks.items()}, native=True
        ).write(job.gradient_file())
        with h5py.File(job.objective_file(), "w") as f:
            f["value"] = 0.5 * weight * sum(v @ v for v in blocks.values())

    monkeypatch.setattr(im.ImagingProblem, "_sync_simulation", sync)
    monkeypatch.setattr(problem.backend, "run", run)
    start = np.linspace(0.1, 0.5, hv.space.size)
    problem.state = problem.state_from(start)
    lin = hv.linearize()
    assert hv.linearize(gradient=False) is lin
    assert len(jobs) == 1
    stage = hv.restrict(active=["vp"], frequencies=[4.0], weights=[0.25])
    assert isinstance(stage, im.HVProblem)
    assert stage.identity()["hv"] == hv.identity()["hv"]
    np.testing.assert_allclose(stage.gradient().values, 0.25 * stage.vector().values)
    assert jobs[-1].active == ["vp"]
    with pytest.raises(ValueError, match="cannot be replaced"):
        hv.restrict(misfit=im.Misfit.l2())
    assert lin.normal.shape == (lin.space.size, lin.space.size)
    with pytest.raises(NotImplementedError, match="residual-space"):
        _ = lin.jacobian
    prepared = hv.linearize(start, curvature=True)
    assert prepared.curvature_job is jobs[-1]
    assert jobs[-1].misfit["objective_terms"][0]["objective"][
        "export_receiver_curvature"
    ]
    assert hv.normal(start) is prepared.normal
    with pytest.raises(NotImplementedError, match="residual-space"):
        hv.jacobian(start)
    im.FWI(hv, im.Stage(frequencies=[4.0, 6.0], iterations=2)).run()
    assert hv.value() < lin.value
    count = len(jobs)
    hv.clear_cache()
    hv.linearize()
    assert len(jobs) == count + 1


def test_hv_native_override_preserves_inactive_material_state(tmp_path, monkeypatch):
    _, base = _problem(
        tmp_path,
        FakeImagingSite(),
        controls=im.ControlSpace(
            vp=im.DepthProfile("vp", "sediment", count=3),
            rho=im.DepthProfile("rho", "sediment", count=2),
        ),
    )
    # Exercise the native override path used by frozen meshed controls.
    monkeypatch.setattr(im.ImagingProblem, "_inline_authoring", lambda self: False)
    captured = []

    def run(job):
        current = im.ControlVectorFile.read(job.current, native=True)
        captured.append(current)
        current.write(job.gradient_file())
        with h5py.File(job.objective_file(), "w") as handle:
            handle["value"] = sum(float(v @ v) / 2 for v in current.blocks.values())

    monkeypatch.setattr(base.backend, "run", run)
    base.state = base.state_from(np.arange(1.0, 6.0))
    hv = base.hv(im.HV(1)).restrict(active=["vp"])
    lin = hv.linearize(np.array([0.2, 0.3, 0.4]))
    np.testing.assert_allclose(lin.gradient.values, [0.2, 0.3, 0.4])
    np.testing.assert_allclose(captured[-1].blocks["rho"], [4, 5])
    assert lin.value == pytest.approx((0.2**2 + 0.3**2 + 0.4**2 + 4**2 + 5**2) / 2)


@pytest.mark.parametrize("physics", ["acoustic", "elastic", "coupled"])
def test_hv_material_view_accepts_wave_physics(tmp_path, physics):
    sim, problem = _problem(tmp_path, FakeImagingSite())
    problem.simulation.physics = physics
    view = problem.hv(im.HV(1))
    assert view.simulation.physics == physics
    assert (
        view.hv.payload(view.observed_groups)["objective_terms"][0]["objective"]["kind"]
        == "hv"
    )
