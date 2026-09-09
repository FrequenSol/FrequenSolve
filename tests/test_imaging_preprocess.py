from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve import MisfitGroup, ObservedTraceDerivatives, PreprocessHook
from frequensolve.units import ureg as u
from frequensolve.util.mixins import ExportContext
from frequensolve.util.store import SimulationStore


def test_trace_weight_infers_source_component_receiver_layout_and_roundtrips():
    weights = np.arange(1, 13, dtype=float).reshape(2, 2, 3)
    group = MisfitGroup(name="surface", observed=None, simulated="simulated")

    hook = group.add_trace_weights(weights, name="target_quality")
    payload = group.to_fs()

    assert hook.stage == "residual"
    assert payload["preprocess"] == [
        {
            "schema": "fs-preprocess-hook-1",
            "name": "target_quality",
            "kind": "trace_weight",
            "stage": "residual",
            "params": {
                "layout": "source_component_receiver",
                "weights": list(np.arange(1.0, 13.0)),
            },
        }
    ]
    assert MisfitGroup.from_fs(payload).to_fs() == payload


def test_packed_observed_df_reference_roundtrips_with_receiver_group():
    group = MisfitGroup(
        name="surface_pressure",
        observed="observed",
        observed_derivatives=ObservedTraceDerivatives.packed(
            "observed",
            receiver_group="surface_pressure",
        ),
        simulated="simulated",
    )

    payload = group.to_fs()

    assert payload["observed_derivatives"] == {
        "df": {
            "_type": "HDF5TraceStore",
            "file": Path("observed") / "traces.h5",
            "dataset": "surface_pressure_df",
            "source_basis": "source_encoding",
        }
    }
    assert MisfitGroup.from_fs(payload).to_fs() == payload


def test_sparse_trace_weights_require_explicit_layout():
    hook = PreprocessHook.trace_weight(
        [1.0, 0.5, 0.0],
        layout="sparse_trace",
    )

    assert hook.to_fs()["params"] == {
        "layout": "sparse_trace",
        "weights": [1.0, 0.5, 0.0],
    }


def test_trace_weights_materialize_in_hdf5_without_bulk_metadata(tmp_path):
    weights = np.arange(1, 13, dtype=float).reshape(2, 2, 3)
    group = MisfitGroup(name="surface", observed=None, simulated="simulated")
    group.add_trace_weights(weights, name="quality")
    store = SimulationStore(tmp_path / "inputs.h5", project_path=tmp_path)

    payload = group.to_fs(
        ExportContext(tmp_path, store=store),
        preprocess_scope="receiver_groups/0",
    )
    reference = payload["preprocess"][0]["params"]["weights"]

    assert reference["_type"] == "HDF5Dense"
    assert reference["file"] == "inputs.h5"
    with h5py.File(tmp_path / "inputs.h5", "r") as h5:
        stored = h5[reference["dataset"]]
        assert stored.shape == (2, 2, 3)
        assert "source" not in stored.attrs
        assert "receiver" not in stored.attrs
        np.testing.assert_allclose(stored[:], weights)
    assert store.prune_unreferenced(payload) == []


def test_large_trace_weights_require_hdf5_context():
    hook = PreprocessHook.trace_weight(np.ones(257))

    with pytest.raises(ValueError, match="require.*HDF5 store"):
        hook.to_fs()


@pytest.mark.parametrize("weights", [[1.0, -0.1], [1.0, np.nan], [1.0, 2.0j]])
def test_trace_weights_reject_invalid_objective_weights(weights):
    with pytest.raises(ValueError):
        PreprocessHook.trace_weight(weights)


def test_receiver_ar1_whitening_serializes_as_linear_trace_pair_hook():
    group = MisfitGroup(name="pressure", observed="observed", simulated="simulated")

    hook = group.add_receiver_ar1_whitening(0.85, name="neighbor_noise")

    assert hook.stage == "trace_pair"
    assert group.to_fs()["preprocess"] == [
        {
            "schema": "fs-preprocess-hook-1",
            "name": "neighbor_noise",
            "kind": "receiver_ar1_whiten",
            "stage": "trace_pair",
            "params": {"correlation": 0.85},
        }
    ]


@pytest.mark.parametrize("correlation", [-1.0, 1.0, np.nan, np.inf])
def test_receiver_ar1_whitening_rejects_invalid_correlation(correlation):
    with pytest.raises(ValueError, match="correlation"):
        PreprocessHook.receiver_ar1_whiten(correlation)


def test_receiver_wavenumber_filters_serialize_as_linear_trace_pair_hooks():
    group = MisfitGroup(name="das", observed="observed", simulated="simulated")

    notch = group.add_scholte_notch(
        900.0 * u.m / u.s,
        relative_half_width=0.03,
        relative_taper_width=0.02,
        name="scholte",
    )
    fan = group.add_slow_velocity_mute(
        1200.0 * u.m / u.s,
        1800.0 * u.m / u.s,
        mode="keep_slow",
        name="s_energy",
    )

    assert notch.stage == fan.stage == "trace_pair"
    assert group.to_fs()["preprocess"] == [
        {
            "schema": "fs-preprocess-hook-1",
            "name": "scholte",
            "kind": "scholte_notch",
            "stage": "trace_pair",
            "params": {
                "relative_half_width": 0.03,
                "relative_taper_width": 0.02,
                "spacing_tolerance": 1.0e-3,
                "phase_velocity": {"value": 900.0, "units": "m/s"},
            },
        },
        {
            "schema": "fs-preprocess-hook-1",
            "name": "s_energy",
            "kind": "slow_velocity_mute",
            "stage": "trace_pair",
            "params": {
                "stop_velocity": {"value": 1200.0, "units": "m/s"},
                "pass_velocity": {"value": 1800.0, "units": "m/s"},
                "mode": "keep_slow",
                "spacing_tolerance": 1.0e-3,
            },
        },
    ]


def test_scholte_notch_accepts_measured_angular_wavenumber():
    hook = PreprocessHook.scholte_notch(wavenumber=0.8 / u.m)

    assert hook.to_fs()["params"]["wavenumber"] == {
        "value": 0.8,
        "units": "1/m",
    }


@pytest.mark.parametrize(
    "factory,match",
    [
        (lambda: PreprocessHook.scholte_notch(), "exactly one"),
        (
            lambda: PreprocessHook.scholte_notch(
                900.0,
                wavenumber=0.8,
            ),
            "exactly one",
        ),
        (
            lambda: PreprocessHook.slow_velocity_mute(1800.0, 1200.0),
            "must exceed",
        ),
        (
            lambda: PreprocessHook.slow_velocity_mute(
                1200.0,
                1800.0,
                mode="invalid",
            ),
            "unsupported",
        ),
    ],
)
def test_receiver_wavenumber_filters_reject_invalid_parameters(factory, match):
    with pytest.raises(ValueError, match=match):
        factory()
