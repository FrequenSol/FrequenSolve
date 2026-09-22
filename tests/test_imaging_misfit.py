import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

from frequensolve.imaging.data import ObservedGroup, TraceStoreRef
from frequensolve.imaging.misfit import (
    Comparison,
    Loss,
    Misfit,
    Normalization,
    ObjectiveTerm,
    Preprocess,
    ReceiverProjection,
)
from frequensolve.units import ureg as u
from frequensolve.util.mixins import ExportContext
from frequensolve.util.store import SimulationStore

CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-5e07624" / "trunk" / "contracts"
)
IMAGING_SCHEMA = CONTRACT_ROOT / "inputs" / "fs-imaging-1" / "schema.json"
IMAGING_EXAMPLES = sorted((IMAGING_SCHEMA.parent / "examples").glob("*.json"))


@pytest.fixture(scope="module")
def imaging_validator() -> Draft202012Validator:
    registry = Registry()
    for schema_file in CONTRACT_ROOT.rglob("*.json"):
        contents = json.loads(schema_file.read_text())
        if "$id" in contents:
            registry = registry.with_resource(
                contents["$id"], Resource.from_contents(contents)
            )
    return Draft202012Validator(
        json.loads(IMAGING_SCHEMA.read_text()), registry=registry
    )


def _imaging_payload(misfit: dict) -> dict:
    """Wrap a misfit block in a minimal, schema-complete imaging payload."""

    return {
        "schema": "fs-imaging-1",
        "data_path": "observed",
        "grid": {
            "system": "global",
            "dims": ["x", "z"],
            "x0": [0.0, 0.0],
            "x1": [1.0, 1.0],
            "n": [11, 11],
            "units": "km",
        },
        "misfit": misfit,
        "images": [{"name": "vp", "IC": "fwi:acoustic", "property": "vp"}],
    }


def _validate(validator, misfit: dict) -> None:
    validator.validate(_imaging_payload(json.loads(json.dumps(misfit))))


GROUPS = {
    "hydrophone": ObservedGroup(
        "hydrophone",
        observed="observed",
        derivatives={
            "df": TraceStoreRef.packed(
                "observed", receiver_group="hydrophone", suffix="_df"
            )
        },
    ),
    "das": ObservedGroup("das", observed=TraceStoreRef("das.h5", missing="zero")),
}


# ---------------------------------------------------------------------------
# Loss / Comparison / Normalization / ReceiverProjection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loss,expected",
    [
        (Loss.l2(), {"kind": "l2"}),
        (Loss("none"), {"kind": "l2"}),
        (Loss.huber(), {"kind": "huber", "delta": 1.5}),
        (Loss("hybrid", delta=0.7), {"kind": "huber", "delta": 0.7}),
        (Loss.student_t(), {"kind": "student_t", "c2": 1.0, "nu": 2.0}),
        (Loss("StudentsT", nu=3.0), {"kind": "student_t", "c2": 1.0, "nu": 3.0}),
    ],
)
def test_loss_serializes_canonical_kinds_and_round_trips(loss, expected):
    assert loss.to_fs() == expected
    assert Loss.from_fs(expected) == loss
    assert Loss.from_value(expected["kind"]).kind == loss.kind


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Loss("cauchy"),
        lambda: Loss.huber(delta=0.0),
        lambda: Loss.student_t(nu=-1.0),
        lambda: Loss("l2", delta=1.0),
        lambda: Loss("huber", nu=2.0),
    ],
)
def test_loss_rejects_invalid_parameters(factory):
    with pytest.raises(ValueError):
        factory()


def test_comparison_waveform_and_phase_derivative_round_trip():
    waveform = Comparison.from_value("waveform")
    phase = Comparison.phase_derivative(
        source_derivative="total", relative_amplitude_floor=0.05
    )

    assert waveform.to_fs() == {"kind": "waveform"}
    assert waveform.requires_derivatives == ()
    assert phase.to_fs() == {
        "kind": "phase_derivative",
        "derivative_axis": "frequency",
        "source_derivative": "total",
        "relative_amplitude_floor": 0.05,
    }
    assert phase.requires_derivatives == ("df",)
    assert Comparison.from_fs(phase.to_fs()) == phase
    with pytest.raises(ValueError):
        Comparison(kind="phase_derivative", relative_amplitude_floor=0.0)
    with pytest.raises(ValueError):
        Comparison.from_fs({"kind": "phase_derivative", "derivative_axis": "laplace"})


def test_normalization_variants_serialize_units_and_round_trip():
    explicit = Normalization.explicit(2.5 * u.Pa)
    components = Normalization.explicit(
        components={"pressure": 1.0, "vz": 0.5 * u.m / u.s}
    )
    rms = Normalization.observed_rms(minimum=1e-3, reduction="sum")
    balance = Normalization.balance_artifact("calibrate/balance.h5")

    assert explicit.to_fs() == {
        "scale": {"kind": "explicit", "value": {"value": 2.5, "units": "Pa"}},
        "reduction": "weighted_mean",
    }
    assert components.to_fs()["scale"]["components"] == {
        "pressure": 1.0,
        "vz": {"value": 0.5, "units": "m/s"},
    }
    assert rms.to_fs() == {
        "scale": {"kind": "observed_rms", "minimum": 0.001},
        "reduction": "sum",
    }
    assert balance.to_fs() == {
        "scale": {"kind": "balance_artifact", "file": "calibrate/balance.h5"},
        "reduction": "weighted_mean",
    }
    for normalization in (explicit, components, rms, balance):
        assert Normalization.from_fs(normalization.to_fs()) == normalization
    assert Normalization.from_value("observed_rms") == Normalization.observed_rms()
    assert Normalization.from_value(3.0) == Normalization.explicit(3.0)
    assert Normalization.from_value(None) == Normalization.observed_rms()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Normalization.explicit(),
        lambda: Normalization.explicit(1.0, components={"p": 1.0}),
        lambda: Normalization.explicit(-1.0),
        lambda: Normalization(kind="balance_artifact"),
        lambda: Normalization.observed_rms(reduction="mean"),
    ],
)
def test_normalization_rejects_invalid_configuration(factory):
    with pytest.raises(ValueError):
        factory()


def test_receiver_projection_serializes_and_round_trips():
    identity = ReceiverProjection.identity()
    acoustic = ReceiverProjection.up_down(1.5e6, pressure_component="p")
    elastic = ReceiverProjection.up_down(
        [1.0e6, 1.1e6],
        physics="elastic",
        shear_impedance={"_type": "HDF5Dense", "file": "z.h5", "dataset": "zs"},
        tangential_velocity_components=["velocity_x"],
        tangential_traction_components=["stress_xz"],
    )

    assert identity.to_fs() == {"kind": "identity"}
    assert acoustic.to_fs() == {
        "kind": "up_down",
        "pressure_component": "p",
        "impedance": 1.5e6,
    }
    assert elastic.to_fs()["shear_impedance"] == {
        "_type": "HDF5Dense",
        "file": "z.h5",
        "dataset": "zs",
    }
    for projection in (identity, acoustic, elastic):
        assert ReceiverProjection.from_fs(projection.to_fs()) == projection
    legacy = ReceiverProjection.from_fs(
        {
            "kind": "acoustic_upgoing",
            "impedance": 2.0,
            "vertical_velocity_component": "vz",
        }
    )
    assert legacy.normal_velocity_component == "vz"
    assert ReceiverProjection.from_value("identity") == identity


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ReceiverProjection.up_down(None),
        lambda: ReceiverProjection.up_down(-1.0),
        lambda: ReceiverProjection.up_down(1.0, physics="elastic"),
        lambda: ReceiverProjection.acoustic_upgoing(1.0, physics="elastic"),
        lambda: ReceiverProjection(kind="identity", impedance=1.0),
    ],
)
def test_receiver_projection_rejects_invalid_configuration(factory):
    with pytest.raises(ValueError):
        factory()


# ---------------------------------------------------------------------------
# Preprocess hooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hook,kind,stage,params",
    [
        (
            Preprocess.offset_power(0.5, normalize=True),
            "offset_power",
            "residual",
            {"power": 0.5, "normalize": True},
        ),
        (Preprocess.offset_power(), "offset_power", "residual", {"normalize": False}),
        (
            Preprocess.offset_taper(d0=200 * u.m, d1=1 * u.km),
            "offset_taper",
            "residual",
            {
                "scale": "absolute",
                "mode": "near",
                "d0": {"value": 200, "units": "m"},
                "d1": {"value": 1, "units": "km"},
            },
        ),
        (
            Preprocess.offset_taper(
                0.1, 0.15, units="km", mode="far", max_domain_fraction=0.5
            ),
            "offset_taper",
            "residual",
            {
                "scale": "absolute",
                "mode": "far",
                "d0": {"value": 0.1, "units": "km"},
                "d1": {"value": 0.15, "units": "km"},
                "max_domain_fraction": 0.5,
            },
        ),
        (
            Preprocess.offset_taper(0.01, 0.02, scale="domain_fraction"),
            "offset_taper",
            "residual",
            {"scale": "domain_fraction", "mode": "near", "d0": 0.01, "d1": 0.02},
        ),
        (
            Preprocess.component_scale([1.0, 1j]),
            "component_scale",
            "residual",
            {"scale": [[1.0, 0.0], [0.0, 1.0]]},
        ),
        (
            Preprocess.frequency_weight(1.0, f0=2.0, amplitude=0.5, epsilon=1e-6),
            "frequency_weight",
            "residual",
            {"power": 1.0, "f0": 2.0, "amplitude": 0.5, "epsilon": 1e-6},
        ),
        (
            Preprocess.trace_normalize("group", epsilon=1e-9),
            "trace_normalize",
            "observed",
            {"scope": "receiver_group", "epsilon": 1e-9},
        ),
        (
            Preprocess.trace_mask(source_ids=[1, 3], components=[2]),
            "trace_mask",
            "residual",
            {"source_ids": [1, 3], "components": [2]},
        ),
        (
            Preprocess.trace_mask(invalid=True, stage="observed"),
            "trace_mask",
            "observed",
            {"invalid": True},
        ),
        (
            Preprocess.amplitude_clip(3.0, clip="soft"),
            "amplitude_clip",
            "observed",
            {"threshold": 3.0, "clip": "soft"},
        ),
        (
            Preprocess.source_scalar_fit(max_iterations=30, relative_tolerance=1e-8),
            "source_scalar_fit",
            "trace_pair",
            {"norm": "inherit", "max_iterations": 30, "relative_tolerance": 1e-8},
        ),
        (
            Preprocess.source_scalar_fit(norm="hybrid", delta=1.2),
            "source_scalar_fit",
            "trace_pair",
            {
                "norm": "huber",
                "max_iterations": 20,
                "relative_tolerance": 1e-8,
                "delta": 1.2,
            },
        ),
        (
            Preprocess.source_spectrum_correction([1.0 + 0.5j], [0.0]),
            "source_spectrum_correction",
            "simulated",
            {"scale": [[1.0, 0.5]], "frequency_derivative": [[0.0, 0.0]]},
        ),
        (
            Preprocess.receiver_ar1_whiten(0.85, name="neighbor"),
            "receiver_ar1_whiten",
            "trace_pair",
            {"correlation": 0.85},
        ),
        (
            Preprocess.scholte_notch(900.0 * u.m / u.s, relative_half_width=0.03),
            "scholte_notch",
            "trace_pair",
            {
                "relative_half_width": 0.03,
                "relative_taper_width": 0.04,
                "spacing_tolerance": 1e-3,
                "spectral_derivative": "total",
                "phase_velocity": {"value": 900.0, "units": "m/s"},
            },
        ),
        (
            Preprocess.scholte_notch(
                wavenumber=0.8 / u.m, spectral_derivative="frozen"
            ),
            "scholte_notch",
            "trace_pair",
            {
                "relative_half_width": 0.04,
                "relative_taper_width": 0.04,
                "spacing_tolerance": 1e-3,
                "spectral_derivative": "frozen",
                "wavenumber": {"value": 0.8, "units": "1/m"},
            },
        ),
        (
            Preprocess.slow_velocity_mute(
                1200.0 * u.m / u.s, 1800.0 * u.m / u.s, mode="keep_slow"
            ),
            "slow_velocity_mute",
            "trace_pair",
            {
                "stop_velocity": {"value": 1200.0, "units": "m/s"},
                "pass_velocity": {"value": 1800.0, "units": "m/s"},
                "mode": "keep_slow",
                "spacing_tolerance": 1e-3,
                "spectral_derivative": "total",
            },
        ),
    ],
)
def test_preprocess_builders_emit_sauce_hook_objects(
    imaging_validator, hook, kind, stage, params
):
    payload = hook.to_fs()

    assert payload["schema"] == "fs-preprocess-hook-1"
    assert payload["kind"] == kind
    assert payload["stage"] == stage
    assert payload["params"] == params
    assert Preprocess.from_fs(payload).to_fs() == payload
    misfit = Misfit.l2(preprocess=[hook]).to_fs({"das": GROUPS["das"]})
    _validate(imaging_validator, misfit)


@pytest.mark.parametrize(
    "factory,error",
    [
        (lambda: Preprocess.offset_taper(300.0, 200.0, units="m"), ValueError),
        (
            lambda: Preprocess.offset_taper(1 * u.m, 2 * u.m, scale="domain_fraction"),
            ValueError,
        ),
        (lambda: Preprocess.offset_taper(1.0, 2.0, mode="middle"), ValueError),
        (lambda: Preprocess.trace_mask(), ValueError),
        (lambda: Preprocess.trace_mask(invalid=True), ValueError),
        (lambda: Preprocess.trace_mask(source_ids=[0]), ValueError),
        (lambda: Preprocess.trace_mask(source_ids=[1.5]), TypeError),
        (lambda: Preprocess.amplitude_clip(-1.0), ValueError),
        (lambda: Preprocess.trace_normalize("everything"), ValueError),
        (lambda: Preprocess.source_scalar_fit(max_iterations=0), ValueError),
        (lambda: Preprocess.source_scalar_fit(norm="cauchy"), ValueError),
        (lambda: Preprocess.source_spectrum_correction([1.0, 2.0], [0.0]), ValueError),
        (lambda: Preprocess.frequency_weight(epsilon=0.0), ValueError),
        (lambda: Preprocess.receiver_ar1_whiten(1.0), ValueError),
        (lambda: Preprocess.scholte_notch(), ValueError),
        (lambda: Preprocess.scholte_notch(900.0, wavenumber=0.8), ValueError),
        (lambda: Preprocess.slow_velocity_mute(1800.0, 1200.0), ValueError),
        (
            lambda: Preprocess.slow_velocity_mute(1200.0, 1800.0, mode="invalid"),
            ValueError,
        ),
        (
            lambda: Preprocess.slow_velocity_mute(
                1.0, 2.0, spectral_derivative="partial"
            ),
            ValueError,
        ),
        (lambda: Preprocess(kind="offset_power", stage="image"), ValueError),
    ],
)
def test_preprocess_builders_reject_invalid_parameters(factory, error):
    with pytest.raises(error):
        factory()


def test_trace_weight_infers_layout_and_inlines_small_arrays():
    weights = np.arange(1, 13, dtype=float).reshape(2, 2, 3)

    hook = Preprocess.trace_weight(weights, name="quality")

    assert hook.stage == "residual"
    assert hook.to_fs() == {
        "schema": "fs-preprocess-hook-1",
        "name": "quality",
        "kind": "trace_weight",
        "stage": "residual",
        "params": {
            "layout": "source_component_receiver",
            "weights": list(np.arange(1.0, 13.0)),
        },
    }
    sparse = Preprocess.trace_weight([1.0, 0.5, 0.0], layout="sparse_trace")
    assert sparse.to_fs()["params"] == {
        "layout": "sparse_trace",
        "weights": [1.0, 0.5, 0.0],
    }


def test_trace_weights_materialize_in_hdf5_store_via_misfit_scope(tmp_path):
    weights = np.arange(1, 13, dtype=float).reshape(2, 2, 3)
    store = SimulationStore(tmp_path / "inputs.h5", project_path=tmp_path)
    misfit = Misfit.l2(preprocess=[Preprocess.trace_weight(weights)])

    payload = misfit.to_fs({"das": GROUPS["das"]}, ExportContext(tmp_path, store=store))
    reference = payload["preprocess"]["hooks"][0]["params"]["weights"]

    assert reference["_type"] == "HDF5Dense"
    assert reference["file"] == "inputs.h5"
    with h5py.File(tmp_path / "inputs.h5", "r") as h5:
        np.testing.assert_allclose(h5[reference["dataset"]][:], weights)
    assert store.prune_unreferenced(payload) == []


def test_large_trace_weights_require_hdf5_context():
    with pytest.raises(ValueError, match="HDF5 store"):
        Preprocess.trace_weight(np.ones(257)).to_fs()


@pytest.mark.parametrize("weights", [[1.0, -0.1], [1.0, np.nan], [1.0, 2.0j]])
def test_trace_weights_reject_invalid_objective_weights(weights):
    with pytest.raises(ValueError):
        Preprocess.trace_weight(weights)


# ---------------------------------------------------------------------------
# ObjectiveTerm and Misfit
# ---------------------------------------------------------------------------


def test_objective_term_defaults_and_round_trip(imaging_validator):
    term = ObjectiveTerm("das", loss="huber", weight=0.3)

    payload = term.to_fs()

    assert payload == {
        "id": "das",
        "receiver_group": "das",
        "objective": {"kind": "huber", "delta": 1.5},
        "comparison": {"kind": "waveform"},
        "weight": 0.3,
        "normalization": {
            "scale": {"kind": "observed_rms"},
            "reduction": "weighted_mean",
        },
    }
    assert ObjectiveTerm.from_fs(payload) == term
    with pytest.raises(ValueError):
        ObjectiveTerm("das", id="1bad")
    with pytest.raises(ValueError):
        ObjectiveTerm("das", weight=0.0)


def test_misfit_constructors_emit_one_term_per_receiver_group(imaging_validator):
    misfit = Misfit.student_t(
        nu=3.0,
        c2=2.0,
        normalization=Normalization.explicit(1.0),
        weights={"hydrophone": 1.0, "das": 0.3},
        projection=ReceiverProjection.up_down(1.5e6),
    )

    payload = misfit.to_fs(GROUPS)

    _validate(imaging_validator, payload)
    assert [term["id"] for term in payload["objective_terms"]] == ["hydrophone", "das"]
    assert payload["objective_terms"][1]["weight"] == 0.3
    assert payload["objective_terms"][0]["objective"] == {
        "kind": "student_t",
        "c2": 2.0,
        "nu": 3.0,
    }
    assert payload["receiver_groups"] == [
        {
            "name": "hydrophone",
            "observed": "observed",
            "observed_derivatives": {
                "df": {
                    "_type": "HDF5TraceStore",
                    "file": str(Path("observed") / "traces.h5"),
                    "dataset": "hydrophone_df",
                }
            },
            "projection": {"kind": "up_down", "impedance": 1.5e6},
        },
        {
            "name": "das",
            "observed": {
                "_type": "HDF5TraceStore",
                "file": "das.h5",
                "missing": "zero",
            },
            "projection": {"kind": "up_down", "impedance": 1.5e6},
        },
    ]
    assert payload["preprocess"] == {"include_defaults": False, "hooks": []}
    assert "objective" not in payload and "comparison" not in payload


def test_misfit_accepts_sequence_of_groups_and_per_group_projection(imaging_validator):
    misfit = Misfit.l2(projection={"hydrophone": ReceiverProjection.up_down(1.0)})

    payload = misfit.to_fs(list(GROUPS.values()))

    _validate(imaging_validator, payload)
    assert payload["receiver_groups"][0]["projection"]["kind"] == "up_down"
    assert payload["receiver_groups"][1]["projection"] == {"kind": "identity"}


def test_misfit_explicit_terms_and_scoped_preprocess(imaging_validator):
    misfit = Misfit.terms(
        ObjectiveTerm(
            "hydrophone",
            loss="huber",
            weight=1.0,
            preprocess=[Preprocess.offset_power(1.0)],
        ),
        ObjectiveTerm(
            "hydrophone",
            comparison="phase_derivative",
            id="hydrophone.phase",
            weight=0.5,
        ),
        ObjectiveTerm("das", loss="l2", weight=0.3),
        preprocess=[Preprocess.offset_taper(0.05, 0.1, scale="domain_fraction")],
        group_preprocess={"das": [Preprocess.receiver_ar1_whiten(0.5)]},
        include_default_preprocess=True,
    )

    payload = misfit.to_fs(GROUPS)

    _validate(imaging_validator, payload)
    assert [term["id"] for term in payload["objective_terms"]] == [
        "hydrophone",
        "hydrophone.phase",
        "das",
    ]
    assert payload["objective_terms"][0]["preprocess"][0]["kind"] == "offset_power"
    assert payload["objective_terms"][1]["comparison"]["kind"] == "phase_derivative"
    assert payload["preprocess"]["include_defaults"] is True
    assert payload["preprocess"]["hooks"][0]["kind"] == "offset_taper"
    assert (
        payload["receiver_groups"][1]["preprocess"][0]["kind"] == "receiver_ar1_whiten"
    )
    assert misfit.required_derivatives("hydrophone") == ("df",)
    assert misfit.required_derivatives("das") == ()


def test_misfit_rejects_phase_derivative_without_observed_df():
    misfit = Misfit.l2(comparison="phase_derivative")

    with pytest.raises(ValueError, match="observed 'df' derivatives"):
        misfit.to_fs({"das": GROUPS["das"]})


def test_misfit_rejects_inconsistent_construction():
    with pytest.raises(ValueError):
        Misfit(loss="l2", terms=[ObjectiveTerm("das")])
    with pytest.raises(ValueError):
        Misfit.terms(ObjectiveTerm("das"), ObjectiveTerm("das"))
    with pytest.raises(ValueError):
        Misfit.terms()
    with pytest.raises(ValueError, match="unknown receiver group"):
        Misfit.terms(ObjectiveTerm("streamer")).to_fs(GROUPS)
    with pytest.raises(KeyError):
        Misfit.l2(weights={"das": 1.0}).to_fs(GROUPS)
    with pytest.raises(ValueError, match="group_preprocess"):
        Misfit.l2(group_preprocess={"streamer": []}).to_fs(GROUPS)


def test_misfit_from_fs_round_trips_terms_projection_and_hooks():
    misfit = Misfit.huber(
        0.8,
        comparison="phase_derivative",
        normalization=Normalization.observed_rms(minimum=1e-4),
        preprocess=[Preprocess.frequency_weight(1.0)],
        projection=ReceiverProjection.up_down(1.5e6),
        group_preprocess={"hydrophone": [Preprocess.scholte_notch(900.0)]},
    )
    groups = {"hydrophone": GROUPS["hydrophone"]}
    payload = misfit.to_fs(groups)

    restored = Misfit.from_fs(payload)

    assert Misfit.receiver_groups_from_fs(payload) == groups
    assert restored.to_fs(groups) == payload
    assert restored.explicit_terms is not None
    assert restored.projection == ReceiverProjection.up_down(1.5e6)
    assert restored.group_preprocess == misfit.group_preprocess


def test_misfit_from_fs_reads_legacy_objective_and_comparison_form():
    payload = {
        "objective": {"kind": "huber", "delta": 2.0},
        "comparison": {"kind": "waveform"},
        "receiver_groups": [{"name": "das", "observed": "obs"}],
        "preprocess": "default",
    }

    misfit = Misfit.from_fs(payload)

    assert misfit.loss == Loss.huber(2.0)
    assert misfit.comparison == Comparison.waveform()
    assert misfit.include_default_preprocess is True
    regenerated = misfit.to_fs(Misfit.receiver_groups_from_fs(payload))
    assert regenerated["objective_terms"][0]["objective"] == {
        "kind": "huber",
        "delta": 2.0,
    }


@pytest.mark.parametrize("example", IMAGING_EXAMPLES, ids=lambda path: path.stem)
def test_pinned_imaging_examples_round_trip_through_misfit(imaging_validator, example):
    payload = json.loads(example.read_text())
    imaging_validator.validate(payload)

    misfit = Misfit.from_fs(payload["misfit"])
    groups = Misfit.receiver_groups_from_fs(payload["misfit"])
    regenerated = misfit.to_fs(groups)

    _validate(imaging_validator, regenerated)
    original_terms = payload["misfit"].get("objective_terms")
    if original_terms is not None:
        assert regenerated["objective_terms"] == original_terms
    assert [g["name"] for g in regenerated["receiver_groups"]] == [
        g["name"] for g in payload["misfit"]["receiver_groups"]
    ]


def test_schema_rejects_terms_mixed_with_legacy_objective(imaging_validator):
    payload = Misfit.l2().to_fs({"das": GROUPS["das"]})
    payload["objective"] = {"kind": "l2"}

    with pytest.raises(ValidationError):
        _validate(imaging_validator, payload)
