import copy

import pytest

import frequensolve as fs
from frequensolve.model.attenuation import AttenuationConfig
from frequensolve.model.layered import LayeredModel
from frequensolve.model.model import ModelBase, ModelSubdomain
from frequensolve.units import ureg as u


@pytest.mark.parametrize("alias", ["reference_frequency", "f0", "f_ref"])
def test_attenuation_reference_frequency_aliases_load_canonically(alias):
    source = {
        "model": "KJARTANSSON",
        alias: {"value": 0.01, "units": "kHz"},
    }
    original = copy.deepcopy(source)

    config = AttenuationConfig.from_fs(source)

    assert source == original
    assert config.model == "kjartansson"
    assert config.to_fs() == {
        "model": "kjartansson",
        "reference_frequency": {"value": 0.01, "units": "kHz"},
    }


def test_attenuation_accepts_bare_hz_pint_and_value_only_mapping():
    assert AttenuationConfig(reference_frequency=10).to_fs() == {
        "model": "kjartansson",
        "reference_frequency": 10,
    }
    assert AttenuationConfig(f_ref=0.01 * u.kHz).to_fs() == {
        "model": "kjartansson",
        "reference_frequency": {"value": 0.01, "units": "kHz"},
    }
    assert AttenuationConfig(f0={"value": 10.0}).to_fs() == {
        "model": "kjartansson",
        "reference_frequency": {"value": 10.0},
    }


def test_attenuation_none_is_case_insensitive_and_keeps_reference_optional():
    config = AttenuationConfig(model=" NoNe ")

    assert config.model == "none"
    assert config.reference_frequency is None
    assert config.to_fs() == {"model": "none"}


def test_attenuation_rejects_multiple_reference_frequency_aliases():
    with pytest.raises(ValueError, match="only one"):
        AttenuationConfig.from_fs({"reference_frequency": 10.0, "f_ref": 0.01})

    with pytest.raises(ValueError, match="only one"):
        AttenuationConfig(reference_frequency=10.0, f0=10.0)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_attenuation_rejects_nonpositive_or_nonfinite_reference(value):
    with pytest.raises(ValueError, match="finite positive"):
        AttenuationConfig(reference_frequency=value)


@pytest.mark.parametrize("value", [True, "10 Hz", [10.0]])
def test_attenuation_rejects_non_numeric_reference_scalars(value):
    with pytest.raises(TypeError, match="positive scalar"):
        AttenuationConfig(reference_frequency=value)


def test_attenuation_rejects_null_reference_or_units_in_solver_mapping():
    with pytest.raises(TypeError, match="positive scalar"):
        AttenuationConfig.from_fs({"reference_frequency": None})

    with pytest.raises(ValueError, match="compatible with frequency"):
        AttenuationConfig(reference_frequency={"value": 10.0, "units": None})


def test_attenuation_rejects_bad_model_units_and_mapping_fields():
    with pytest.raises(ValueError, match="Unsupported attenuation model"):
        AttenuationConfig(model="constant_q")

    with pytest.raises(ValueError, match="compatible with frequency"):
        AttenuationConfig(reference_frequency=10.0 * u.m)

    with pytest.raises(ValueError, match="compatible with frequency"):
        AttenuationConfig(reference_frequency={"value": 10.0, "units": "m"})

    with pytest.raises(ValueError, match="only value and units"):
        AttenuationConfig(
            reference_frequency={"value": 10.0, "units": "Hz", "scale": 1.0}
        )


def test_model_base_serializes_and_roundtrips_typed_attenuation():
    model = ModelBase(
        name="model",
        dimension=2,
        attenuation_model="NONE",
    )
    model += ModelSubdomain(mesh_block_id=1, properties={"Vp": 1500.0})

    payload = model.to_fs()
    roundtrip = ModelBase.from_fs(payload)

    assert payload["attenuation"] == {"model": "none"}
    assert model.attenuation_model == "none"
    assert roundtrip.attenuation_model == "none"
    assert roundtrip.reference_frequency is None
    assert roundtrip.to_fs() == payload


def test_model_without_attenuation_preserves_solver_default_contract():
    payload = ModelBase(name="model", dimension=2).to_fs()

    assert "attenuation" not in payload


def test_model_reference_frequency_alone_uses_default_attenuation_model():
    model = ModelBase(
        name="model",
        dimension=2,
        reference_frequency=0.01 * u.kHz,
    )

    assert model.to_fs()["attenuation"] == {
        "model": "kjartansson",
        "reference_frequency": {"value": 0.01, "units": "kHz"},
    }


def test_layered_model_and_simulation_export_attenuation_block(tmp_path):
    model = LayeredModel(
        dimension=2,
        x_limits=[0.0, 1.0],
        attenuation_model="KJARTANSSON",
        reference_frequency={"value": 0.01, "units": "kHz"},
    )
    model.add_surface(name="top", depth=0.0)
    model.add_layer(
        name="rock",
        mesh_block_id=1,
        properties={"Vp": 1500.0, "Rho": 1000.0, "Qp": 20.0},
    )
    model.add_surface(name="bottom", depth=1.0)

    payload = model.to_fs()
    roundtrip = LayeredModel.from_fs(payload)
    simulation = fs.SeismicSimulation(
        name="attenuating",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    simulation.model = model
    simulation_payload = simulation.to_fs()

    assert payload["attenuation"] == {
        "model": "kjartansson",
        "reference_frequency": {"value": 0.01, "units": "kHz"},
    }
    assert roundtrip.attenuation_model == "kjartansson"
    assert roundtrip.reference_frequency == {"value": 0.01, "units": "kHz"}
    assert roundtrip.to_fs()["attenuation"] == payload["attenuation"]
    assert simulation_payload["Model"]["attenuation"] == payload["attenuation"]


def test_attenuation_config_is_exported_from_root_namespace():
    assert fs.AttenuationConfig is AttenuationConfig
