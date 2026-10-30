"""Transient GSLS attenuation: config round trip, validation and job gates."""

import copy

import pytest

import frequensolve as fs
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.model.attenuation import AttenuationConfig
from frequensolve.model.model import ModelBase
from frequensolve.units import ureg as u

GSLS_FS = {
    "model": "GSLS",
    "mechanisms": 3,
    "frequency_min": {"value": 2.0, "units": "Hz"},
    "frequency_max": {"value": 0.04, "units": "kHz"},
    "f_ref": 10.0,
    "fit_tolerance": 0.1,
}

GSLS_BAND = {"frequency_min": 2.0, "frequency_max": 40.0}


def test_gsls_config_roundtrips_dict_object_dict():
    source = copy.deepcopy(GSLS_FS)

    config = AttenuationConfig.from_fs(source)
    payload = config.to_fs()

    assert source == GSLS_FS
    assert config.model == "gsls"
    assert config.mechanisms == 3
    assert config.fit_tolerance == 0.1
    assert config.extra == {}
    assert payload == {
        "model": "gsls",
        "reference_frequency": 10.0,
        "mechanisms": 3,
        "frequency_min": {"value": 2.0, "units": "Hz"},
        "frequency_max": {"value": 0.04, "units": "kHz"},
        "fit_tolerance": 0.1,
    }
    assert AttenuationConfig.from_fs(payload).to_fs() == payload


def test_gsls_typed_fields_accept_pint_and_omit_solver_defaults():
    config = AttenuationConfig(
        model="gsls",
        frequency_min=2.0 * u.Hz,
        frequency_max=0.04 * u.kHz,
    )

    assert config.to_fs() == {
        "model": "gsls",
        "frequency_min": {"value": 2.0, "units": "Hz"},
        "frequency_max": {"value": 0.04, "units": "kHz"},
    }


def test_gsls_reference_frequency_may_sit_on_band_edges():
    for reference in (2.0, {"value": 0.04, "units": "kHz"}):
        AttenuationConfig(model="gsls", reference_frequency=reference, **GSLS_BAND)


def test_gsls_fields_are_not_exported_for_other_models():
    for model in ("kjartansson", "none"):
        config = AttenuationConfig(
            model=model, mechanisms=3, fit_tolerance=0.1, **GSLS_BAND
        )
        assert config.to_fs() == {"model": model}


@pytest.mark.parametrize(
    "fields,error,match",
    [
        ({}, ValueError, "frequency_min and frequency_max"),
        ({"frequency_min": 2.0}, ValueError, "missing frequency_max"),
        (
            {"frequency_min": 40.0, "frequency_max": 2.0},
            ValueError,
            "frequency_min < frequency_max",
        ),
        (
            {"frequency_min": 5.0, "frequency_max": 5.0},
            ValueError,
            "frequency_min < frequency_max",
        ),
        ({**GSLS_BAND, "reference_frequency": 50.0}, ValueError, "in the band"),
        ({**GSLS_BAND, "f0": {"value": 1.0, "units": "Hz"}}, ValueError, "in the band"),
        ({**GSLS_BAND, "mechanisms": 0}, ValueError, "1..16"),
        ({**GSLS_BAND, "mechanisms": 17}, ValueError, "1..16"),
        ({**GSLS_BAND, "mechanisms": 3.0}, TypeError, "integer"),
        ({**GSLS_BAND, "mechanisms": True}, TypeError, "integer"),
        ({**GSLS_BAND, "fit_tolerance": 1.0}, ValueError, r"\(0, 1\)"),
        ({**GSLS_BAND, "fit_tolerance": 0.0}, ValueError, r"\(0, 1\)"),
        ({**GSLS_BAND, "frequency_max": -1.0}, ValueError, "finite positive"),
        (
            {"frequency_min": 2.0, "frequency_max": {"value": 40.0, "units": "m"}},
            ValueError,
            "compatible with frequency",
        ),
    ],
)
def test_gsls_rejects_invalid_solver_payloads(fields, error, match):
    with pytest.raises(error, match=match):
        AttenuationConfig.from_fs({"model": "gsls", **fields})


def test_gsls_constructor_defers_missing_band_to_export():
    config = AttenuationConfig(model="gsls", mechanisms=3)

    with pytest.raises(ValueError, match="frequency_min and frequency_max"):
        config.to_fs()

    config.frequency_min = 2.0
    config.frequency_max = 40.0
    assert config.to_fs()["mechanisms"] == 3


def test_gsls_constructor_rejects_inverted_band_and_typed_extra_conflict():
    with pytest.raises(ValueError, match="frequency_min < frequency_max"):
        AttenuationConfig(model="gsls", frequency_min=40.0, frequency_max=2.0)

    with pytest.raises(ValueError, match="both"):
        AttenuationConfig(model="gsls", mechanisms=3, extra={"mechanisms": 4})


def test_model_exports_gsls_band_from_attenuation_extra():
    model = ModelBase(name="model", dimension=2, attenuation_model="gsls")
    model._attenuation_extra = {**GSLS_BAND, "mechanisms": 3}

    assert model.to_fs()["attenuation"] == {
        "model": "gsls",
        "mechanisms": 3,
        "frequency_min": 2.0,
        "frequency_max": 40.0,
    }


def test_model_from_fs_roundtrips_gsls_band():
    # Requires ModelBase/LayeredModel.from_fs to keep AttenuationConfig.fs_extra().
    payload = ModelBase(name="model", dimension=2).to_fs()
    payload["attenuation"] = {"model": "gsls", "mechanisms": 3, **GSLS_BAND}

    roundtrip = ModelBase.from_fs(payload)

    assert roundtrip.attenuation_model == "gsls"
    assert roundtrip.to_fs()["attenuation"] == payload["attenuation"]


def _transient_job(tmp_path, attenuation_model, band=None):
    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=2)
    sim.discretization = fs.Discretization(method="Galerkin")
    sim.model.attenuation_model = attenuation_model
    if band is not None:
        sim.model._attenuation_extra = dict(band)
    sim.mesh.set_adapt(elems_per_wave=1, f_adapt=20)
    return fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(0, 1, 0.1),
        source=fs.TimeFunction.ricker(10, delay=0.3),
    )


def test_transient_job_accepts_gsls_and_exports_band(tmp_path):
    job = _transient_job(tmp_path, "gsls", {**GSLS_BAND, "mechanisms": 3})

    attenuation = job.simulation.model.to_fs()["attenuation"]

    assert attenuation == {
        "model": "gsls",
        "mechanisms": 3,
        "frequency_min": 2.0,
        "frequency_max": 40.0,
    }


def test_transient_job_rejects_gsls_without_band(tmp_path):
    with pytest.raises(ValueError, match="frequency_min and frequency_max"):
        _transient_job(tmp_path, "gsls")


def test_transient_job_rejects_kjartansson(tmp_path):
    with pytest.raises(ValueError, match="needs model.attenuation_model='gsls'"):
        _transient_job(tmp_path, "kjartansson")


def test_transient_job_defaults_unset_attenuation_to_lossless(tmp_path):
    job = _transient_job(tmp_path, None)
    assert job.simulation.model.attenuation_model == "none"
    assert job.simulation.model.to_fs()["attenuation"]["model"] == "none"


def test_frequency_domain_job_rejects_gsls(tmp_path):
    simulation = fs.SeismicSimulation(
        name="freq", physics="acoustic", dimension=2, project_path=tmp_path
    )
    simulation.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[1, 1])
    )
    simulation.model.attenuation_model = "gsls"
    simulation.model._attenuation_extra = dict(GSLS_BAND)
    job = fs.FrequencyDomainJob(name="freq", simulation=simulation, f_list=[10.0])

    report = job.validate()
    issues = [
        issue
        for issue in report.issues
        if issue.code == "model.attenuation.gsls.unsupported"
    ]

    assert len(issues) == 1
    assert "transient-only; use kjartansson" in issues[0].message
    with pytest.raises(Exception, match="transient-only"):
        job.validate(raise_errors=True)
