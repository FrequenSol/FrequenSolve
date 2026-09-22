import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from frequensolve.imaging._artifacts import (
    BalanceArtifact,
    ControlRegistryManifest,
    ControlStateFile,
    ControlVectorFile,
    ExtensionSolveReport,
    ExtensionVectorField,
    ExtensionVectorFile,
    ImageSet,
    ObjectiveReport,
    SmoothingConfig,
    pack_support_mask,
    qualified_block_name,
    unpack_support_mask,
    unqualified_block_name,
)

CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-f533e6f" / "trunk" / "contracts"
)


def _string(h5, name):
    value = h5[name][()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


# ---------------------------------------------------------------------------
# block names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("vp", "model.vp"),
        ("model.vp", "model.vp"),
        ("source.2.mechanism", "source.2.mechanism"),
        ("reflectivity.ip", "reflectivity.ip"),
    ],
)
def test_qualified_block_names(name, expected):
    assert qualified_block_name(name) == expected


@pytest.mark.parametrize("name", ["", "a/b", "source.1", "source.1.wobble", "model."])
def test_qualified_block_names_reject_malformed_names(name):
    with pytest.raises(ValueError):
        qualified_block_name(name)


def test_unqualified_block_names_only_exist_for_material_blocks():
    assert unqualified_block_name("model.vp") == "vp"
    assert unqualified_block_name("vp") == "vp"
    with pytest.raises(ValueError, match="no unqualified"):
        unqualified_block_name("source.1.position")


# ---------------------------------------------------------------------------
# control vectors
# ---------------------------------------------------------------------------


def test_control_vector_file_round_trips_qualified_blocks_and_identity(tmp_path):
    vector = ControlVectorFile(
        {
            "model.vp": [1.0, 2.0, 3.0],
            "source.1.mechanism": np.array([1 + 2j, 3 - 4j]),
        },
        state_fingerprint="sha256:state",
        control_registry_fingerprint="sha256:registry",
    )
    path = vector.write(tmp_path / "direction.h5")

    with h5py.File(path, "r") as h5:
        assert _string(h5, "schema") == "fs-control-vector-1"
        assert _string(h5, "packing") == "real_interleaved"
        assert _string(h5, "state_fingerprint") == "sha256:state"
        assert _string(h5, "control_registry_fingerprint") == "sha256:registry"
        assert h5["controls/model.vp"].dtype == np.float64
        np.testing.assert_array_equal(
            h5["controls/source.1.mechanism"][()], [1.0, 2.0, 3.0, -4.0]
        )

    loaded = ControlVectorFile.read(path)
    assert loaded.native is False
    assert loaded.names == ("model.vp", "source.1.mechanism")
    assert loaded.sizes == {"model.vp": 3, "source.1.mechanism": 4}
    assert loaded.size == 7
    assert loaded.state_fingerprint == "sha256:state"
    assert loaded.control_registry_fingerprint == "sha256:registry"
    np.testing.assert_array_equal(loaded["vp"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(loaded.pack(), [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, -4.0])
    np.testing.assert_array_equal(
        loaded.pack(["source.1.mechanism", "model.vp"]),
        [1.0, 2.0, 3.0, -4.0, 1.0, 2.0, 3.0],
    )


def test_control_vector_file_requires_identity_unless_native(tmp_path):
    with pytest.raises(ValueError, match="state_fingerprint"):
        ControlVectorFile({"model.vp": [1.0]}).write(tmp_path / "bad.h5")

    native = ControlVectorFile({"model.vp": [1.0, 2.0], "rho": [3.0]}, native=True)
    path = native.write(tmp_path / "native.h5")
    with h5py.File(path, "r") as h5:
        assert "schema" not in h5
        assert sorted(h5["controls"]) == ["rho", "vp"]

    loaded = ControlVectorFile.read(path)
    assert loaded.native is True
    assert loaded.names == ("rho", "vp")
    np.testing.assert_array_equal(loaded["model.vp"], [1.0, 2.0])
    with pytest.raises(ValueError, match="not a native vector"):
        ControlVectorFile(
            {"model.vp": [1.0]},
            state_fingerprint="a",
            control_registry_fingerprint="b",
        ).write(tmp_path / "typed.h5")
        ControlVectorFile.read(tmp_path / "typed.h5", native=True)


def test_control_vector_file_round_trips_covector_support_datasets(tmp_path):
    covector = ControlVectorFile(
        {"model.vp": np.arange(10.0), "source.1.position": [1.5, 2.5]},
        state_fingerprint="s",
        control_registry_fingerprint="r",
        support={"vp": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0]},
        support_measure={"model.vp": np.linspace(0, 255, 10).astype(np.uint8)},
        support_min_support=0.01,
    )
    path = covector.write(tmp_path / "covector.h5")

    with h5py.File(path, "r") as h5:
        assert h5["support/model.vp"].dtype == np.uint8
        np.testing.assert_array_equal(
            h5["support/model.vp"][()], [0b00001101, 0b00000001]
        )
        assert h5["support_measure/model.vp"].dtype == np.uint8
        assert h5["support_min_support"].dtype == np.float64
        assert h5["support_min_support"].shape == ()
        assert h5["support_min_support"][()] == 0.01
        assert "support/source.1.position" not in h5

    loaded = ControlVectorFile.read(path)
    assert loaded.native is False
    assert loaded.support_min_support == 0.01
    assert set(loaded.support) == {"model.vp"}
    np.testing.assert_array_equal(
        loaded.support_mask("vp"), [1, 0, 1, 1, 0, 0, 0, 0, 1, 0]
    )
    np.testing.assert_array_equal(
        loaded.support_mask("source.1.position"), [True, True]
    )
    assert loaded.support_measure["model.vp"][-1] == 255
    assert loaded.support_measure["model.vp"][0] == 0

    # Direction inputs carry no support datasets.
    direction = ControlVectorFile.read(
        ControlVectorFile(
            {"model.vp": [1.0]}, state_fingerprint="s", control_registry_fingerprint="r"
        ).write(tmp_path / "direction.h5")
    )
    assert direction.support == {}
    assert direction.support_measure == {}
    assert direction.support_min_support is None
    np.testing.assert_array_equal(direction.support_mask("model.vp"), [True])

    with pytest.raises(ValueError, match="unknown block"):
        ControlVectorFile({"model.vp": [1.0]}, support={"rho": [True]})
    with pytest.raises(ValueError, match="one flag per DOF"):
        ControlVectorFile({"model.vp": [1.0, 2.0]}, support={"vp": [True]})
    with pytest.raises(ValueError, match="non-negative"):
        ControlVectorFile({"model.vp": [1.0]}, support_min_support=-0.5)
    with h5py.File(path, "a") as h5:
        h5.create_dataset("support/model.rho", data=np.zeros(1, dtype=np.uint8))
    with pytest.raises(ValueError, match="no /controls block"):
        ControlVectorFile.read(path)


def test_control_vector_from_packed_uses_ordered_sizes():
    vector = ControlVectorFile.from_packed(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        {"model.vp": 3, "model.rho": 2},
        state_fingerprint="s",
        control_registry_fingerprint="r",
    )
    np.testing.assert_array_equal(vector["rho"], [4.0, 5.0])
    with pytest.raises(ValueError, match="entries"):
        ControlVectorFile.from_packed([1.0], {"model.vp": 3})
    with pytest.raises(ValueError, match="finite"):
        ControlVectorFile({"model.vp": [np.nan]}, native=True)


# ---------------------------------------------------------------------------
# control states and support masks
# ---------------------------------------------------------------------------


def test_support_bitmask_is_lsb_first_eight_dofs_per_byte():
    mask = np.array([1, 0, 1, 1, 0, 0, 0, 0, 1, 0], dtype=bool)
    packed = pack_support_mask(mask)
    np.testing.assert_array_equal(
        packed, np.array([0b00001101, 0b00000001], dtype=np.uint8)
    )
    np.testing.assert_array_equal(unpack_support_mask(packed, 10), mask)
    np.testing.assert_array_equal(
        unpack_support_mask(bytes([0b10000010]), 8),
        [False, True, False, False, False, False, False, True],
    )
    with pytest.raises(ValueError, match="bits"):
        unpack_support_mask(packed, 17)


def test_control_state_file_round_trips_blocks_support_and_measure(tmp_path):
    state = ControlStateFile(
        {"vp": np.arange(10.0), "source.1.position": [1.5, 2.5]},
        support={"model.vp": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0]},
        support_measure={"model.vp": np.linspace(0, 255, 10).astype(np.uint8)},
        support_min_support=0.02,
    )
    path = state.write(tmp_path / "state.h5")

    with h5py.File(path, "r") as h5:
        assert _string(h5, "schema") == "fs-control-state-1"
        assert _string(h5, "packing") == "real_interleaved"
        assert h5["support/model.vp"].dtype == np.uint8
        np.testing.assert_array_equal(
            h5["support/model.vp"][()], [0b00001101, 0b00000001]
        )
        assert h5["support_measure/model.vp"].dtype == np.uint8
        assert h5["support_min_support"][()] == 0.02
        assert "support/source.1.position" not in h5

    loaded = ControlStateFile.read(path)
    assert loaded.names == ("model.vp", "source.1.position")
    assert loaded.support_min_support == 0.02
    np.testing.assert_array_equal(
        loaded.support_mask("vp"), [1, 0, 1, 1, 0, 0, 0, 0, 1, 0]
    )
    np.testing.assert_array_equal(
        loaded.support_mask("source.1.position"), [True, True]
    )
    np.testing.assert_array_equal(loaded.support_measure["model.vp"][-1], 255)
    np.testing.assert_array_equal(loaded["source.1.position"], [1.5, 2.5])

    restricted = loaded.restrict(["source.1.position"])
    assert restricted.names == ("source.1.position",)
    assert restricted.support == {}
    assert restricted.support_min_support == 0.02
    active = loaded.restrict(["vp"])
    np.testing.assert_array_equal(active.support_mask("vp"), loaded.support_mask("vp"))
    assert active.support_measure["model.vp"][-1] == 255
    updated = loaded.with_update(
        ControlVectorFile({"model.vp": np.zeros(10)}, native=False)
    )
    assert np.all(updated["vp"] == 0.0)
    np.testing.assert_array_equal(updated.support_mask("vp"), loaded.support_mask("vp"))
    assert updated.support_min_support == 0.02
    with pytest.raises(ValueError, match="wrong size"):
        loaded.with_update(ControlVectorFile({"model.vp": [1.0]}))


def test_control_state_file_round_trips_mechanism_scaling(tmp_path):
    # the Sauce export layout (imaging-api/multitask-operators):
    # /scaling/<block> float64 and /scaling_units/<block> string
    path = tmp_path / "sauce_state.h5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("schema", data=np.bytes_(b"fs-control-state-1"))
        h5.create_dataset("packing", data=np.bytes_(b"real_interleaved"))
        h5.create_dataset("controls/model.vp", data=np.zeros(3))
        h5.create_dataset("controls/source.1.mechanism", data=[0.5, 0.0, 0.25, 0.0])
        h5.create_dataset("scaling/source.1.mechanism", data=4.0e6, dtype=np.float64)
        h5.create_dataset("scaling_units/source.1.mechanism", data=np.bytes_(b"N"))

    loaded = ControlStateFile.read(path)
    assert loaded.scaling == {"source.1.mechanism": 4.0e6}
    assert loaded.scaling_units == {"source.1.mechanism": "N"}
    written = loaded.write(tmp_path / "again.h5")
    with h5py.File(written, "r") as h5:
        assert h5["scaling/source.1.mechanism"].dtype == np.float64
        assert h5["scaling/source.1.mechanism"][()] == 4.0e6
        assert _string(h5, "scaling_units/source.1.mechanism") == "N"
    again = ControlStateFile.read(written)
    assert again.scaling == loaded.scaling
    assert again.scaling_units == loaded.scaling_units
    updated = again.with_update(ControlVectorFile({"model.vp": np.ones(3)}))
    assert updated.scaling == loaded.scaling
    # files without /scaling read as task coordinates
    plain = ControlStateFile({"vp": [1.0]}).write(tmp_path / "plain.h5")
    with h5py.File(plain, "r") as h5:
        assert "scaling" not in h5 and "scaling_units" not in h5
    assert ControlStateFile.read(plain).scaling == {}

    with pytest.raises(ValueError, match="no /controls block"):
        ControlStateFile({"vp": [1.0]}, scaling={"source.1.mechanism": 1.0})
    with pytest.raises(ValueError, match="finite and positive"):
        ControlStateFile(
            {"source.1.mechanism": [1.0, 0.0]}, scaling={"source.1.mechanism": 0.0}
        )
    with pytest.raises(ValueError, match="has no /scaling"):
        ControlStateFile(
            {"source.1.mechanism": [1.0, 0.0]},
            scaling_units={"source.1.mechanism": "N"},
        )
    with pytest.raises(ValueError, match="non-empty"):
        ControlStateFile(
            {"source.1.mechanism": [1.0, 0.0]},
            scaling={"source.1.mechanism": 1.0},
            scaling_units={"source.1.mechanism": " "},
        )


def test_control_state_file_rejects_control_vector_schema(tmp_path):
    path = ControlVectorFile(
        {"model.vp": [1.0]},
        state_fingerprint="s",
        control_registry_fingerprint="r",
    ).write(tmp_path / "vector.h5")
    with pytest.raises(ValueError, match="fs-control-state-1"):
        ControlStateFile.read(path)
    with pytest.raises(ValueError, match="unknown block"):
        ControlStateFile({"vp": [1.0]}, support={"rho": [True]})


# ---------------------------------------------------------------------------
# registry manifest
# ---------------------------------------------------------------------------


def test_control_registry_manifest_reads_pinned_example():
    manifest = ControlRegistryManifest.load(
        CONTRACT_ROOT
        / "outputs"
        / "fs-control-registry-1"
        / "examples"
        / "source-position.json"
    )
    assert manifest.fingerprint.startswith("sha256:")
    assert manifest.packing == "real_interleaved"
    assert manifest.names == ("source.1.position",)
    assert manifest.active_names == ("source.1.position",)
    block = manifest.block("source.1.position")
    assert block.binding == (2, 11, 1)
    assert block.layout == (1, 2, 2, 1)
    assert block.offset == 0 and block.size == 2 and not block.complex
    assert block.supports_jvp and block.supports_vjp
    assert block.components == ("x", "z")
    assert manifest.bindings() == {"source.1.position": (2, 11, 1)}
    assert manifest.layout() == {"source.1.position": slice(0, 2)}
    assert manifest.active_layout() == {"source.1.position": slice(0, 2)}
    assert manifest.state_size == 2 and manifest.active_size == 2
    np.testing.assert_array_equal(
        manifest.unpack_state()["source.1.position"], [513.7, 34.1]
    )
    assert manifest.rank_descriptors[0]["file"] == "controls.json.rank_0.json"


def test_control_registry_manifest_orders_active_blocks_and_offsets():
    manifest = ControlRegistryManifest.from_dict(
        {
            "schema": "fs-control-registry-1",
            "fingerprint": "sha256:" + "0" * 64,
            "packing": "real_interleaved",
            "pairing": "real_euclidean",
            "coordinates": [0.0] * 7,
            "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "active_blocks": [3, 1],
            "active_offsets": [3, 0, 1],
            "blocks": [
                {
                    "id": 1,
                    "name": "model.vp",
                    "binding": [1, 1, 1],
                    "layout": [1, 3, 1, 1],
                    "units": "m/s",
                    "actions": 3,
                    "transform": 1,
                    "scaling": [0, 1, -1, 1],
                    "basis_identity": "hat",
                    "distributed": True,
                },
                {
                    "id": 2,
                    "name": "model.rho",
                    "binding": [1, 1, 2],
                    "layout": [4, 2, 1, 1],
                    "units": "kg/m3",
                    "actions": 3,
                    "transform": 1,
                    "scaling": [0, 1, -1, 1],
                    "basis_identity": "hat",
                    "distributed": True,
                },
                {
                    "id": 3,
                    "name": "source.1.signature",
                    "binding": [4, 1, 1],
                    "layout": [6, 2, 1, 2],
                    "units": "1",
                    "actions": 3,
                    "transform": 1,
                    "scaling": [0, 1, -1, 1],
                    "basis_identity": "",
                    "distributed": False,
                },
            ],
            "descriptor_rank": 0,
            "n_ranks": 1,
            "rank_descriptors": [
                {"file": "r.json", "fingerprint": "sha256:" + "1" * 64}
            ],
        }
    )
    assert manifest.active_names == ("source.1.signature", "model.vp")
    assert manifest.active_layout() == {
        "model.vp": slice(2, 5),
        "source.1.signature": slice(0, 2),
    }
    assert manifest.block("rho").slice == slice(3, 5)
    assert manifest.block("source.1.signature").complex
    np.testing.assert_array_equal(manifest.unpack_state()["model.rho"], [4.0, 5.0])
    with pytest.raises(ValueError, match="schema"):
        ControlRegistryManifest.from_dict({"schema": "nope", "blocks": []})


# ---------------------------------------------------------------------------
# extension vectors
# ---------------------------------------------------------------------------


def test_extension_vector_file_round_trips_lag_and_offset_fields(tmp_path):
    lags = ExtensionVectorField(np.arange(6.0).reshape(3, 2), axis="lag", control="vp")
    offsets = ExtensionVectorField(np.ones((4, 3)), axis="offset")
    vector = ExtensionVectorFile(
        [lags, offsets],
        fingerprint="sha256:basis",
        baseline="sha256:state",
        role="tangent",
    )
    path = vector.write(tmp_path / "taps.h5")

    with h5py.File(path, "r") as h5:
        assert _string(h5, "schema") == "fs-extension-vector-1"
        assert _string(h5, "role") == "tangent"
        assert sorted(h5["fields/1/lag"]) == ["1", "2"]
        assert sorted(h5["fields/2/offset"]) == ["1", "2", "3"]
        np.testing.assert_array_equal(h5["fields/1/lag/2"][()], [1.0, 3.0, 5.0])
        assert h5["fields/1/lag/2"].dtype == np.float64

    loaded = ExtensionVectorFile.read(path)
    assert loaded.fingerprint == "sha256:basis"
    assert loaded.baseline == "sha256:state"
    assert [field.axis for field in loaded.fields] == ["lag", "offset"]
    assert loaded.fields[0].control == "vp"
    assert loaded.fields[0].spatial_count == 3 and loaded.fields[0].n_axis == 2
    np.testing.assert_array_equal(loaded.fields[0].values, lags.values)
    assert loaded.size == 18
    packed = loaded.pack()
    np.testing.assert_array_equal(packed[:6], [0.0, 2.0, 4.0, 1.0, 3.0, 5.0])
    rebuilt = ExtensionVectorFile.from_packed(packed, loaded, role="covector")
    assert rebuilt.role == "covector"
    np.testing.assert_array_equal(rebuilt.fields[1].values, offsets.values)


def test_extension_vector_file_validates_shape_role_and_identity(tmp_path):
    with pytest.raises(ValueError, match="lag' or 'offset"):
        ExtensionVectorField(np.ones(3), axis="time")
    with pytest.raises(ValueError, match="role"):
        ExtensionVectorFile([ExtensionVectorField(np.ones(3))], role="dual")
    with pytest.raises(ValueError, match="fingerprint and baseline"):
        ExtensionVectorFile([ExtensionVectorField(np.ones(3))]).write(tmp_path / "x.h5")
    with pytest.raises(ValueError, match="at least one field"):
        ExtensionVectorFile([])


# ---------------------------------------------------------------------------
# JSON reports
# ---------------------------------------------------------------------------


def test_objective_report_reads_pinned_example():
    report = ObjectiveReport.load(
        CONTRACT_ROOT
        / "outputs"
        / "fs-objective-report-1"
        / "examples"
        / "two-term-report.json"
    )
    assert report.total == pytest.approx(0.875)
    assert report.state_fingerprint == "sha256:state"
    assert [term.id for term in report.terms] == ["kinematic", "waveform"]
    waveform = report.term("waveform")
    assert waveform.weight == 0.25
    assert waveform.weighted_value == pytest.approx(0.375)
    assert waveform.scale == (1250.0,)
    assert waveform.active_samples == 1024
    assert report.weighted_values == {"kinematic": 0.5, "waveform": 0.375}
    assert sum(report.weighted_values.values()) == pytest.approx(report.total)
    with pytest.raises(KeyError):
        report.term("missing")


def test_objective_report_rejects_wrong_schema_and_duplicate_terms(tmp_path):
    with pytest.raises(ValueError, match="schema"):
        ObjectiveReport.from_dict(
            {"schema": "fs-objective-report-2", "total": 0, "terms": []}
        )
    term = {
        "id": "a",
        "raw_sum": 1.0,
        "effective_weight_mass": 1.0,
        "normalized_value": 1.0,
        "weight": 1.0,
        "weighted_value": 1.0,
        "active_samples": 1,
        "scale": [1.0],
    }
    with pytest.raises(ValueError, match="unique"):
        ObjectiveReport.from_dict(
            {"schema": "fs-objective-report-1", "total": 2.0, "terms": [term, term]}
        )


def test_balance_artifact_reads_pinned_example():
    balance = BalanceArtifact.load(
        CONTRACT_ROOT
        / "outputs"
        / "fs-objective-balance-1"
        / "examples"
        / "baseline-residual-rms.json"
    )
    assert balance.frequency_hz == 3.0
    assert balance.laplace_damping_hz == 1.0
    term = balance.term("waveform")
    assert term.receiver_group == "hydrophones"
    assert term.components == ("pressure",)
    assert term.units == ("Pa",)
    assert term.scale == (1250.0,)
    assert term.active_samples == 2048
    assert balance.scales == {"waveform": (1250.0,)}
    assert balance.fingerprints["source_signature"] == "none"
    assert set(balance.fingerprints) >= {
        "reference_model",
        "observations",
        "acquisition",
        "preprocessing",
        "comparisons",
    }


def test_extension_solve_report_parses_cg_and_nested_reduced_normal(tmp_path):
    payload = {
        "schema": "fs-extension-solve-1",
        "baseline": "sha256:state",
        "fingerprint": "sha256:extension",
        "damping": 0.1,
        "lag_penalty": 1.0,
        "lag_scale_seconds": 0.02,
        "offset_penalty": 0.0,
        "offset_scale_meters": 0.0,
        "field_scales": [1500.0],
        "background_batches": 2,
        "resident_background_bytes": 4096,
        "regularization": 0.25,
        "method": "cg",
        "iterations": 12,
        "normal_actions": 13,
        "converged": True,
        "rhs_norm": 2.0,
        "residual_norm": 1e-7,
        "quadratic_change": -0.5,
        "quadratic_objective": 0.75,
        "reduced_normal": {
            "method": "gauss_newton_schur",
            "iterations": 7,
            "normal_actions": 8,
            "converged": True,
            "rhs_norm": 1.0,
            "residual_norm": 1e-8,
            "quadratic_change": -0.1,
        },
    }
    path = tmp_path / "inner_solve.json"
    path.write_text(json.dumps(payload))

    report = ExtensionSolveReport.load(path)
    assert report.method == "cg"
    assert report.iterations == 12 and report.normal_actions == 13
    assert report.converged is True
    assert report.field_scales == (1500.0,)
    assert report.lag_scale_seconds == 0.02
    assert report.quadratic_objective == 0.75
    assert report.reduced_objective is None
    assert report.reduced_normal is not None
    assert report.reduced_normal.method == "gauss_newton_schur"
    assert report.reduced_normal.iterations == 7
    assert report.reduced_normal.residual_norm == 1e-8

    # Sauce may embed the nested report as serialized JSON text.
    payload["reduced_normal"] = json.dumps(payload["reduced_normal"])
    assert ExtensionSolveReport.from_dict(payload).reduced_normal.iterations == 7


def test_extension_solve_report_parses_robust_reduced_gradient_fields():
    report = ExtensionSolveReport.from_dict(
        {
            "schema": "fs-extension-solve-1",
            "baseline": "b",
            "fingerprint": "f",
            "damping": 0.3,
            "method": "irls_gn_armijo",
            "iterations": 4,
            "evaluations": 9,
            "normal_actions": 20,
            "converged": True,
            "line_search_failed": False,
            "initial_gradient_norm": 1.0,
            "gradient_norm": 1e-6,
            "objective": 0.5,
            "data_objective": 0.4,
            "reduced_objective": 0.45,
            "background_gradient_stationary": True,
        }
    )
    assert report.method == "irls_gn_armijo"
    assert report.evaluations == 9
    assert report.line_search_failed is False
    assert report.reduced_objective == pytest.approx(0.45)
    assert report.background_gradient_stationary is True
    assert report.reduced_normal is None
    with pytest.raises(ValueError, match="schema"):
        ExtensionSolveReport.from_dict({"schema": "other"})


# ---------------------------------------------------------------------------
# image sets
# ---------------------------------------------------------------------------


def _write_image_group(h5, location, values, *, axis_units, units):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group = h5.create_group(location)
    group.create_dataset("properties", data=np.array(["vp"], dtype=string_dtype))
    dataset = group.create_dataset("vp", data=values)
    dataset.attrs["x0"] = np.array([0.0, 0.0])
    dataset.attrs["x1"] = np.array([2.0, 1.0])
    dataset.attrs["n_grid"] = np.array([3, 2])
    dataset.attrs["dims"] = np.array(["x", "z"], dtype=string_dtype)
    dataset.attrs["axis_units"] = np.array(axis_units, dtype=string_dtype)
    dataset.attrs["units"] = np.array([units], dtype=string_dtype)


def test_image_set_reads_raw_smoothed_and_incremental_groups(tmp_path):
    image_path = tmp_path / "image"
    image_path.mkdir()
    values = np.arange(6.0)
    with h5py.File(image_path / "image.h5", "w") as h5:
        _write_image_group(h5, "image/raw", values, axis_units=["m", "m"], units="m/s")
        _write_image_group(
            h5, "image/smoothed", 2 * values, axis_units=["km", "km"], units="km/s"
        )
        _write_image_group(
            h5, "incremental", 3 * values, axis_units=["m", "m"], units="m/s"
        )
    with h5py.File(image_path / "image_1.h5", "w") as h5:
        h5.create_dataset("frequency", data=5.0)

    images = ImageSet(path=image_path, parts=1, shape=(2, 3))
    raw = images.raw
    assert raw["vp"].dims == ("z", "x")
    np.testing.assert_array_equal(raw["vp"].values, values.reshape(2, 3))
    assert raw["vp"].attrs["units"] == "m/s"
    assert raw["vp"].coords["x"].attrs["units"] == "m"
    smoothed = images.smoothed
    np.testing.assert_array_equal(smoothed["vp"].values, (2 * values).reshape(2, 3))
    assert smoothed["vp"].coords["z"].attrs["units"] == "km"
    np.testing.assert_array_equal(
        images.incremental["vp"].values, (3 * values).reshape(2, 3)
    )
    np.testing.assert_array_equal(images.f_list, [5.0])
    assert images.image_file(1) == image_path / "image_1.h5"


def test_image_set_requires_aggregate_image(tmp_path):
    image_path = tmp_path / "image"
    image_path.mkdir()
    (image_path / "image_1.h5").touch()
    images = ImageSet(path=image_path, parts=1, shape=(2, 3))
    with pytest.raises(FileNotFoundError, match="imaging --smooth postprocess"):
        images.require_aggregate()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ImageSet(path=tmp_path / "missing", parts=1)
    explicit = ImageSet(
        path=tmp_path / "missing",
        parts=1,
        artifact_files={None: image_path / "image.h5"},
    )
    with pytest.raises(FileNotFoundError, match="No committed image"):
        explicit.image_file(2)


# ---------------------------------------------------------------------------
# smoothing
# ---------------------------------------------------------------------------


def test_smoothing_config_control_and_image_contracts():
    tv = SmoothingConfig(kind="tv", wavelength_fraction=0.2, epsilon=1e-2, iterations=7)
    assert tv.to_control_fs() == {
        "type": "tv",
        "lambda": 0.2,
        "derivative_order": 1,
        "epsilon": 1e-2,
        "iterations": 7,
        "input_role": "dual",
    }
    assert tv.to_image_fs() == {
        "type": "tv",
        "lambda": 0.2,
        "derivative_order": 1,
        "epsilon": 1e-2,
        "iterations": 7,
        "illumination_normalization": "none",
    }

    tgv = SmoothingConfig(
        kind="tgv", wavelength_fraction=0.4, alpha1=0.3, alpha2=0.7, tgv_ratio=1.25
    )
    control = tgv.to_control_fs()
    assert (
        control["alpha1"] == 0.3
        and control["alpha2"] == 0.7
        and control["tgv_ratio"] == 1.25
    )
    assert "illumination_normalization" not in control

    scaled = SmoothingConfig(
        kind="tgv", wavelength_fraction=0.5, reference_wavelength=2 * np.pi
    )
    assert scaled.to_control_fs()["reference_wavelength"] == pytest.approx(2 * np.pi)
    image = scaled.to_image_fs()
    assert "reference_wavelength" not in image
    assert image["alpha1"] == pytest.approx(0.5)
    assert image["alpha2"] == pytest.approx(0.25)
    tik = SmoothingConfig(
        kind="l2",
        wavelength_fraction=1.0,
        reference_wavelength=2 * np.pi,
        derivative_order=2,
    )
    assert tik.kind == "tikhonov"
    assert tik.resolved_alpha() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="first-order"):
        tik.to_image_fs()
    assert (
        SmoothingConfig(kind="tgv", normalize_amplitude=False).to_image_fs()[
            "normalize_amplitude"
        ]
        is False
    )
    assert (
        SmoothingConfig(illumination_normalization="linear").illumination_normalization
        == "source"
    )


def test_smoothing_config_from_value_accepts_sauce_spellings():
    config = SmoothingConfig.from_value(
        {
            "type": "tv",
            "lambda": 0.5,
            "input_role": "primal",
            "normalize_illumination": True,
        }
    )
    assert config == SmoothingConfig(
        kind="tv",
        wavelength_fraction=0.5,
        input_role="primal",
        illumination_normalization="cross",
    )
    assert SmoothingConfig.from_value(None) is None
    assert SmoothingConfig.from_value(config) is config


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"kind": "laplace"}, "unsupported smoothing kind"),
        ({"alpha1": 1.0}, "supplied together"),
        ({"kind": "tgv", "alpha": 1.0}, "alpha1 and alpha2"),
        ({"kind": "tv", "alpha1": 1.0, "alpha2": 1.0}, "only valid for TGV"),
        ({"derivative_order": 3}, "one or two"),
        ({"iterations": 0}, "positive"),
        ({"input_role": "sideways"}, "input role"),
        ({"epsilon": 0.0}, "epsilon"),
    ],
)
def test_smoothing_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SmoothingConfig(**kwargs)
