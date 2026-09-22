import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from frequensolve.geometry.grids import CartesianGrid
from frequensolve.imaging._artifacts import ControlVectorFile, SmoothingConfig
from frequensolve.imaging.jobs import (
    ControlGradientJob,
    FWIOperatorJob,
    ImageKernelJob,
    ImageSpec,
    SmoothJob,
)
from frequensolve.mesh.mesh_generators import HexMeshGenerator
from frequensolve.mesh.mesh_manager import MeshManager
from frequensolve.seismic.acquisition import Acquisition
from frequensolve.seismic.receivers import ReceiverComponent, ReceiverNode
from frequensolve.simulation.jobs import BaseJob
from frequensolve.simulation.simulation import SeismicSimulation

CONTRACT_ROOT = (
    Path(__file__).parent / "contracts" / "sauce-5e07624" / "trunk" / "contracts"
)
JOB_SCHEMA = CONTRACT_ROOT / "inputs" / "fs-job-1" / "schema.json"
JOB_EXAMPLES = CONTRACT_ROOT / "inputs" / "fs-job-1" / "examples"


def _job_validator() -> Draft202012Validator:
    def retrieve(uri):
        # Contracts referenced by fs-job-1 but not pinned here (fs-file-ref-1,
        # fs-ray-tracing-1, ...) match nothing, which keeps every oneOf branch
        # they appear in disjoint.
        return Resource.from_contents(
            {"$schema": "https://json-schema.org/draft/2020-12/schema", "not": {}}
        )

    registry = Registry(retrieve=retrieve)
    for schema_file in CONTRACT_ROOT.rglob("*.json"):
        if "examples" in schema_file.parts:
            continue
        contents = json.loads(schema_file.read_text())
        if "$id" in contents:
            registry = registry.with_resource(
                contents["$id"], Resource.from_contents(contents)
            )
    return Draft202012Validator(json.loads(JOB_SCHEMA.read_text()), registry=registry)


VALIDATOR = _job_validator()


def _assert_valid(payload):
    """Validate a job payload; ``result_path`` is added by the site at staging."""

    candidate = json.loads(
        json.dumps({**payload, "result_path": "results/job"}, default=str)
    )
    errors = sorted(
        VALIDATOR.iter_errors(candidate), key=lambda error: list(error.path)
    )
    assert not errors, "\n".join(
        f"{list(error.absolute_path)}: {error.message}" for error in errors
    )
    return candidate


def _shape(value):
    """Reduce absolute paths to basenames so payloads compare with the examples."""

    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    if isinstance(value, str) and ("/" in value) and not value.startswith("sha256"):
        return Path(value).name
    return value


def _example(name):
    return json.loads((JOB_EXAMPLES / name).read_text())


def _saved_simulation(tmp_path, name="controlled"):
    simulation = SeismicSimulation(
        name=name, physics="acoustic", dimension=2, project_path=tmp_path
    )
    simulation.save()
    return simulation


def _elastic_simulation(tmp_path):
    sim = SeismicSimulation(
        name="smooth", physics="elastic", dimension=2, project_path=tmp_path
    )
    sim.model.x_limits = [0.0, 1.0]
    sim.model.z_limits = [0.0, 1.0]
    sim.mesh = MeshManager(
        HexMeshGenerator(l_bound=[0.0, 0.0], u_bound=[1.0, 1.0], n=[1, 1])
    )
    acq = Acquisition()
    acq.add_sources(kind="vector", coords=np.array([[0.5, 0.1]]), direction=[0.0, 1.0])
    device = ReceiverNode(
        name="geophone", components=[ReceiverComponent(name="vz", field="velocity")]
    )
    acq.add_receiver_group(
        name="surface", device=device, coords=np.array([[0.0, 0.0], [1.0, 0.0]])
    )
    sim.acquisition = acq
    sim.save()
    return sim


def _round_trip(job):
    loaded = BaseJob.load(job.save())
    assert type(loaded) is type(job)
    before = job.to_fs()
    after = loaded.to_fs()
    before.pop("artifact_contract", None)
    after.pop("artifact_contract", None)
    assert json.loads(json.dumps(after, default=str)) == json.loads(
        json.dumps(before, default=str)
    )
    return loaded


# ---------------------------------------------------------------------------
# pinned examples validate with the test validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", sorted(p.name for p in JOB_EXAMPLES.glob("fwi-operator-*.json"))
)
def test_pinned_fwi_operator_examples_validate(name):
    _assert_valid(_example(name))


# ---------------------------------------------------------------------------
# FWIOperatorJob
# ---------------------------------------------------------------------------


def test_fwi_linearize_matches_pinned_example_shape(tmp_path):
    sim = _saved_simulation(tmp_path)
    job = FWIOperatorJob(
        "fwi_linearize",
        sim,
        [4.0],
        action="linearize",
        active=["model.acoustic_vp"],
        state="state.json",
        covector="gradient.h5",
        control_active=["vp"],
    )
    payload = _assert_valid(job.to_fs())

    example = _example("fwi-operator-linearize.json")
    expected = dict(example["fwi_operator"])
    expected.pop("model_gradient")  # false is the default; not emitted
    expected.pop("cache_receiver_state")  # true is the default; not emitted
    assert _shape(payload["fwi_operator"]) == expected
    assert payload["control_sensitivities"] == example["control_sensitivities"]
    assert payload["Imaging"] == {}
    assert payload["workflow"] == "fwi_operator"
    assert "Image" not in payload

    assert (
        job.state
        == tmp_path / "jobs" / "controlled" / "fwi_linearize" / "results" / "state.json"
    )
    assert job.state_file(1).name == "state_1.json"
    assert job.covector_file(2).name == "gradient_2.h5"
    assert job.covector_file(raw=True).name == "gradient_raw.h5"
    assert not job.requires_postprocess()
    loaded = _round_trip(job)
    assert loaded.active == ["model.acoustic_vp"]
    assert loaded.control_active == ["vp"]
    assert loaded.action == "linearize"


def test_fwi_jvp_matches_pinned_joint_controls_example(tmp_path):
    sim = _saved_simulation(tmp_path)
    job = FWIOperatorJob(
        "fwi_linearize",
        sim,
        [4.0],
        action="jvp",
        active=["source.2.mechanism", "model.acoustic_vp", "source.1.position"],
        state="state_1.json",
        objective_vector="jvp.json",
        direction="direction.h5",
        manifest="controls.json",
    )
    payload = _assert_valid(job.to_fs())
    assert (
        _shape(payload["fwi_operator"])
        == _example("fwi-operator-joint-controls.json")["fwi_operator"]
    )
    assert job.direction == tmp_path / "direction.h5"
    assert job.objective_vector_file(1).name == "jvp_1.json"
    assert job.manifest_file().name == "controls.json"
    loaded = _round_trip(job)
    assert loaded.active == [
        "source.2.mechanism",
        "model.acoustic_vp",
        "source.1.position",
    ]


def test_fwi_vjp_normal_and_calibrate_payloads(tmp_path):
    sim = _saved_simulation(tmp_path)
    vjp = FWIOperatorJob(
        "vjp",
        sim,
        [4.0, 6.0],
        action="vjp",
        active=["vp"],
        state="state.json",
        objective_vector="dual.json",
        covector="cov.h5",
        objective="report.json",
        control_state="baseline.h5",
        state_output="resolved.h5",
    )
    payload = _assert_valid(vjp.to_fs())
    op = payload["fwi_operator"]
    assert op["action"] == "vjp"
    assert op["controls"]["active"] == ["model.vp"]
    assert Path(op["controls"]["state"]) == tmp_path / "baseline.h5"
    assert Path(op["controls"]["state_output"]).name == "resolved.h5"
    assert Path(op["objective_vector"]) == tmp_path / "dual.json"
    # Sauce writes fs-objective-report-1 beside the state shard, not at objective
    assert vjp.report_file(2).name == "state_2_report.json"
    assert vjp.report_file().name == "state_report.json"
    assert vjp.state_output_file().name == "resolved.h5"
    _round_trip(vjp)

    normal = FWIOperatorJob(
        "normal",
        sim,
        [4.0],
        action="normal",
        active=["vp"],
        state="state.json",
        direction="d.h5",
        covector="c.h5",
        cache_receiver_state=False,
    )
    payload = _assert_valid(normal.to_fs())
    assert payload["fwi_operator"]["cache_receiver_state"] is False
    _round_trip(normal)

    calibrate = FWIOperatorJob(
        "calibrate",
        sim,
        [4.0],
        action="calibrate",
        balance="balance.json",
        misfit={
            "objective_terms": [
                {
                    "id": "waveform",
                    "receiver_group": "surface",
                    "normalization": {"scale": {"kind": "observed_rms"}},
                }
            ]
        },
    )
    payload = _assert_valid(calibrate.to_fs())
    assert payload["fwi_operator"] == {
        "action": "calibrate",
        "balance": str(calibrate.balance),
    }
    assert payload["Imaging"]["misfit"]["objective_terms"][0]["id"] == "waveform"
    assert "controls" not in payload["fwi_operator"]
    assert calibrate.balance_file(1).name == "balance_1.json"
    loaded = _round_trip(calibrate)
    assert loaded.misfit == calibrate.misfit


def test_fwi_wri_payloads_match_pinned_examples(tmp_path):
    sim = _saved_simulation(tmp_path)
    wri = FWIOperatorJob(
        "wri_wavefield_update",
        sim,
        [4.0],
        action="wri",
        covector="wri_gradient.h5",
        objective="wri_objective.h5",
        wri={"penalty": 10.0, "data_scale": "auto", "receiver_group": "surface"},
        gram_derivative="total",
        control_active=["vp"],
    )
    payload = _assert_valid(wri.to_fs())
    example = _example("fwi-operator-wri.json")
    assert _shape(payload["fwi_operator"]) == example["fwi_operator"]
    assert payload["control_sensitivities"] == example["control_sensitivities"]
    _round_trip(wri)

    schur = FWIOperatorJob(
        "wri_schur_normal",
        sim,
        [4.0],
        action="wri",
        covector="wri_normal.h5",
        direction="direction.h5",
        objective="wri_objective.h5",
        wri={
            "penalty": 10.0,
            "data_scale": "auto",
            "receiver_group": "surface",
            "curvature": "joint_schur",
        },
        gram_derivative="frozen",
        control_active=["vp"],
    )
    payload = _assert_valid(schur.to_fs())
    example = _example("fwi-operator-wri-normal.json")
    assert _shape(payload["fwi_operator"]) == example["fwi_operator"]
    assert payload["control_sensitivities"] == example["control_sensitivities"]
    loaded = _round_trip(schur)
    assert loaded.direction == tmp_path / "direction.h5"
    assert loaded.covector.name == "wri_normal.h5"

    groups = FWIOperatorJob(
        "wri_groups",
        sim,
        [4.0],
        action="wri",
        covector="g.h5",
        wri={"penalty": 10.0, "receiver_groups": ["pressure", "motion"]},
    )
    payload = _assert_valid(groups.to_fs())
    assert payload["control_sensitivities"] == {}


def test_fwi_extension_payloads_match_pinned_examples(tmp_path):
    sim = _saved_simulation(tmp_path)
    linearize = FWIOperatorJob(
        "lag_extension",
        sim,
        [3.0 + 0j],
        action="linearize",
        active=[],
        state="extension_state.json",
        extension={
            "fields": [
                {
                    "control": "acoustic_vp",
                    "lags": {
                        "count": 3,
                        "origin": -10.0,
                        "spacing": 10.0,
                        "units": "ms",
                    },
                }
            ],
            "manifest": "extension_space.json",
        },
    )
    payload = _assert_valid(linearize.to_fs())
    example = _example("fwi-operator-extension.json")
    assert _shape(payload["fwi_operator"]) == example["fwi_operator"]
    assert payload["f_list"] == [[3.0, 0.0]]
    assert linearize.extension_manifest_file().name == "extension_space.json"
    _round_trip(linearize)

    solve = FWIOperatorJob(
        "offset_extension_inner",
        sim,
        [3.0],
        action="solve",
        active=["model.acoustic_vp"],
        state="extension_state_1.json",
        model_gradient=True,
        covector="background_gradient.h5",
        extension={
            "fields": [
                {
                    "control": "acoustic_vp",
                    "offsets": {
                        "half_offsets": [[-25.0, 0.0], [0.0, 0.0], [25.0, 0.0]],
                        "units": "m",
                    },
                }
            ],
            "solver": {
                "damping": 0.1,
                "offset_penalty": 1.0,
                "offset_scale": {"value": 25.0, "units": "m"},
                "relative_tolerance": 1e-6,
                "max_iterations": 100,
                "require_convergence": True,
                "solution": "extension_taps.h5",
                "report": "inner_solve.json",
            },
        },
    )
    payload = _assert_valid(solve.to_fs())
    example = _example("fwi-operator-extension-solve.json")
    assert _shape(payload["fwi_operator"]) == example["fwi_operator"]
    assert solve.extension_solution_file(1).name == "extension_taps_1.h5"
    assert solve.extension_report_file(1).name == "inner_solve_1.json"
    assert solve.extension_solution_file().parent == solve._result_path
    loaded = _round_trip(solve)
    assert loaded.model_gradient is True
    assert loaded.extension["solver"]["offset_scale"] == {"value": 25.0, "units": "m"}

    reduced = FWIOperatorJob(
        "time_lag_reduced_normal",
        sim,
        [3.0],
        action="solve",
        active=["model.acoustic_vp"],
        state="extension_state_1.json",
        covector="physical_reduced_normal.h5",
        direction="physical_direction.h5",
        reduced_normal={"relative_tolerance": 1e-7, "max_iterations": 100},
        extension={
            "fields": [
                {
                    "control": "acoustic_vp",
                    "lags": {
                        "count": 5,
                        "origin": -20.0,
                        "spacing": 10.0,
                        "units": "ms",
                    },
                }
            ],
            "solver": {
                "damping": 0.1,
                "relative_tolerance": 1e-6,
                "max_iterations": 100,
                "require_convergence": True,
                "solution": "extension_taps.h5",
                "report": "inner_solve.json",
                "lag_penalty": 1.0,
                "lag_scale": {"value": 20.0, "units": "ms"},
            },
        },
    )
    payload = _assert_valid(reduced.to_fs())
    example = _example("fwi-operator-reduced-normal.json")
    assert _shape(payload["fwi_operator"]) == example["fwi_operator"]
    _round_trip(reduced)

    ext_jvp = FWIOperatorJob(
        "ext_jvp",
        sim,
        [3.0],
        action="jvp",
        active=[],
        state="s.json",
        objective_vector="jvp.json",
        extension={
            "fields": [
                {
                    "control": "vp",
                    "lags": {"count": 2, "origin": 0.0, "spacing": 5.0, "units": "ms"},
                }
            ],
            "direction": "taps.h5",
        },
    )
    payload = _assert_valid(ext_jvp.to_fs())
    assert (
        Path(payload["fwi_operator"]["extension"]["direction"]) == tmp_path / "taps.h5"
    )
    _round_trip(ext_jvp)


def test_fwi_reflectivity_and_source_controls_match_pinned_examples(tmp_path):
    sim = _saved_simulation(tmp_path)
    reflectivity = FWIOperatorJob(
        "joint-impedance-reflectivity",
        sim,
        [3.0],
        action="linearize",
        active=["model.acoustic_vp", "reflectivity.ip"],
        state="extension_state.json",
        covector="joint_gradient.h5",
        reflectivity={
            "parameterization": "vp_ip",
            "fields": [{"name": "ip", "layer": 2, "axis": 2, "basis": "acoustic_vp"}],
        },
    )
    payload = _assert_valid(reflectivity.to_fs())
    assert (
        _shape(payload["fwi_operator"])
        == _example("fwi-operator-reflectivity.json")["fwi_operator"]
    )
    _round_trip(reflectivity)

    sources = FWIOperatorJob(
        "fwi_linearize",
        sim,
        [4.0],
        action="linearize",
        active=["source.1.position", "source.1.mechanism"],
        state="source_state.json",
        covector="gradient.h5",
        source_controls={"location_method": "local_fd4", "reference_step": 0.0001},
    )
    payload = _assert_valid(sources.to_fs())
    example = _example("fwi-operator-source-controls.json")["fwi_operator"]
    example.pop("cache_receiver_state")
    assert _shape(payload["fwi_operator"]) == example
    _round_trip(sources)


def test_fwi_kernel_derivative_emits_both_imaging_keys(tmp_path):
    sim = _saved_simulation(tmp_path)
    job = FWIOperatorJob(
        "kernel",
        sim,
        [4.0],
        action="linearize",
        active=["vp"],
        state="s.json",
        covector="c.h5",
        misfit={"objective": {"kind": "l2"}},
        kernel_derivative={
            "order": 2,
            "axis": "laplace",
            "residual": "window",
            "window": [0.0, 1.0],
        },
    )
    payload = _assert_valid(job.to_fs())
    assert payload["kernel_derivative"] == {
        "order": 2,
        "axis": "laplace",
        "residual": "window",
        "window": [0.0, 1.0],
    }
    assert (
        payload["Image"]
        == payload["Imaging"]
        == {"misfit": {"objective": {"kind": "l2"}}}
    )
    loaded = _round_trip(job)
    assert loaded.kernel_derivative == job.kernel_derivative
    with pytest.raises(ValueError, match="residual"):
        FWIOperatorJob(
            "k",
            sim,
            [4.0],
            action="linearize",
            active=["vp"],
            state="s.json",
            kernel_derivative={"order": 1},
        )
    with pytest.raises(ValueError, match="not allowed"):
        FWIOperatorJob(
            "k",
            sim,
            [4.0],
            action="linearize",
            active=["vp"],
            state="s.json",
            kernel_derivative={"residual": "jet"},
        )


def test_fwi_smoothing_binds_the_smooth_postprocess_to_covector_parts(tmp_path):
    sim = _saved_simulation(tmp_path)
    job = FWIOperatorJob(
        "smoothed",
        sim,
        [2.0, 4.0],
        action="linearize",
        active=["model.vp", "model.rho"],
        state="s.json",
        covector="gradient.h5",
        weights=[0.25, 0.75],
        smoothing=SmoothingConfig(kind="tv", wavelength_fraction=0.2),
    )
    payload = _assert_valid(job.to_fs())
    assert payload["control_sensitivities"] == {
        "active": ["vp", "rho"],
        "gradient": str(job.covector),
        "Smoothing": {
            "type": "tv",
            "lambda": 0.2,
            "derivative_order": 1,
            "epsilon": 1e-3,
            "iterations": 5,
            "input_role": "dual",
        },
        "weights": [0.25, 0.75],
    }
    assert job.requires_postprocess()
    assert job.postprocess_file(1) == job.covector_file(1)
    assert job.postprocess_fetch_files() == [job.covector_file(raw=True), job.covector]
    assert not job.needs_postprocess()
    job.covector.parent.mkdir(parents=True)
    job.covector_file(1).touch()
    job.covector_file(2).touch()
    assert job.postprocess_part_outputs_exist()
    assert job.needs_postprocess()
    job.covector.touch()
    assert not job.needs_postprocess()
    loaded = _round_trip(job)
    assert loaded.smoothing == job.smoothing
    assert loaded.weights == [0.25, 0.75]

    with pytest.raises(ValueError, match="model.\\* blocks"):
        FWIOperatorJob(
            "s",
            sim,
            [2.0],
            action="linearize",
            active=["source.1.position"],
            state="s.json",
            covector="c.h5",
            smoothing={"type": "tv"},
        )


def test_fwi_support_keys_are_emitted_under_controls_only_when_set(tmp_path):
    sim = _saved_simulation(tmp_path)
    plain = FWIOperatorJob(
        "lin", sim, [2.0], action="linearize", active=["model.vp"], state="s.json"
    )
    plain_payload = _assert_valid(plain.to_fs())
    assert "min_support" not in plain_payload["fwi_operator"]["controls"]
    assert "support_measure" not in plain_payload["fwi_operator"]["controls"]

    job = FWIOperatorJob(
        "lin",
        sim,
        [2.0],
        action="linearize",
        active=["model.vp"],
        state="s.json",
        covector="c.h5",
        state_output="state.h5",
        min_support=0.05,
        support_measure=True,
    )
    payload = _assert_valid(job.to_fs())
    assert payload["fwi_operator"]["controls"] == {
        "active": ["model.vp"],
        "state_output": str(job.state_output),
        "min_support": 0.05,
        "support_measure": True,
    }
    assert job._input_fingerprint_payload() == plain._input_fingerprint_payload()
    loaded = _round_trip(job)
    assert loaded.min_support == 0.05
    assert loaded.support_measure is True

    threshold_only = FWIOperatorJob(
        "vjp",
        sim,
        [2.0],
        action="vjp",
        active=["model.vp"],
        state="s.json",
        covector="c.h5",
        objective_vector="dual.h5",
        min_support=0.0,
    )
    controls = _assert_valid(threshold_only.to_fs())["fwi_operator"]["controls"]
    assert controls["min_support"] == 0.0
    assert "support_measure" not in controls

    with pytest.raises(ValueError, match="non-negative"):
        FWIOperatorJob(
            "bad",
            sim,
            [2.0],
            action="linearize",
            active=[],
            state="s.json",
            min_support=-1.0,
        )
    with pytest.raises(TypeError, match="must be a number"):
        FWIOperatorJob(
            "bad",
            sim,
            [2.0],
            action="linearize",
            active=[],
            state="s.json",
            min_support=True,
        )
    with pytest.raises(ValueError, match="fwi_operator.controls"):
        FWIOperatorJob(
            "bad",
            sim,
            [2.0],
            action="calibrate",
            balance="b.json",
            support_measure=True,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"action": "jvp", "active": ["vp"], "state": "s", "objective_vector": "o"},
            "requires a direction",
        ),
        (
            {"action": "jvp", "active": ["vp"], "state": "s", "direction": "d"},
            "objective_vector output",
        ),
        (
            {
                "action": "jvp",
                "active": [],
                "state": "s",
                "direction": "d",
                "objective_vector": "o",
            },
            "nonempty active",
        ),
        (
            {
                "action": "jvp",
                "active": ["vp"],
                "state": "s",
                "direction": "d",
                "objective_vector": "o",
                "covector": "c",
            },
            "does not write a covector",
        ),
        (
            {"action": "vjp", "active": ["vp"], "state": "s", "objective_vector": "o"},
            "covector output",
        ),
        (
            {"action": "vjp", "active": ["vp"], "state": "s", "covector": "c"},
            "objective_vector input",
        ),
        (
            {
                "action": "vjp",
                "active": ["vp"],
                "state": "s",
                "covector": "c",
                "objective_vector": "o",
                "direction": "d",
            },
            "does not take a direction",
        ),
        (
            {"action": "normal", "active": ["vp"], "state": "s", "direction": "d"},
            "direction and covector",
        ),
        (
            {"action": "normal", "active": ["vp"], "state": "s", "covector": "c"},
            "direction and covector",
        ),
        (
            {"action": "linearize", "active": ["vp"], "state": "s", "direction": "d"},
            "does not take a direction",
        ),
        ({"action": "linearize", "active": ["vp"]}, "requires a state"),
        ({"action": "linearize", "state": "s"}, "requires active"),
        ({"action": "calibrate"}, "balance"),
        (
            {"action": "calibrate", "balance": "b", "active": ["vp"]},
            "not accepted by action",
        ),
        (
            {"action": "calibrate", "balance": "b", "smoothing": {"type": "tv"}},
            "control sensitivities",
        ),
        ({"action": "wri", "covector": "c"}, "wri options"),
        ({"action": "wri", "covector": "c", "wri": {"data_scale": "auto"}}, "penalty"),
        ({"action": "wri", "covector": "c", "wri": {"penalty": 0.0}}, "positive"),
        ({"action": "wri", "wri": {"penalty": 1.0}}, "model_covector"),
        (
            {
                "action": "wri",
                "covector": "c",
                "wri": {"penalty": 1.0, "curvature": "joint_schur"},
            },
            "model_direction",
        ),
        (
            {
                "action": "wri",
                "covector": "c",
                "direction": "d",
                "wri": {"penalty": 1.0},
            },
            "curvature",
        ),
        (
            {"action": "wri", "covector": "c", "state": "s", "wri": {"penalty": 1.0}},
            "not accepted by action",
        ),
        (
            {
                "action": "wri",
                "covector": "c",
                "wri": {
                    "penalty": 1.0,
                    "receiver_group": "a",
                    "receiver_groups": ["b"],
                },
            },
            "mutually exclusive",
        ),
        ({"action": "solve", "active": [], "state": "s"}, "requires an extension"),
        (
            {
                "action": "solve",
                "active": [],
                "state": "s",
                "extension": {
                    "fields": [
                        {
                            "control": "vp",
                            "lags": {
                                "count": 1,
                                "origin": 0,
                                "spacing": 1,
                                "units": "s",
                            },
                        }
                    ]
                },
            },
            "extension.solver",
        ),
        (
            {
                "action": "solve",
                "active": [],
                "state": "s",
                "covector": "c",
                "extension": {
                    "fields": [
                        {
                            "control": "vp",
                            "lags": {
                                "count": 1,
                                "origin": 0,
                                "spacing": 1,
                                "units": "s",
                            },
                        }
                    ],
                    "solver": {"damping": 1, "solution": "t", "report": "r"},
                },
            },
            "model_gradient or reduced_normal",
        ),
        (
            {
                "action": "solve",
                "active": ["vp"],
                "state": "s",
                "covector": "c",
                "reduced_normal": {},
                "extension": {
                    "fields": [
                        {
                            "control": "vp",
                            "lags": {
                                "count": 1,
                                "origin": 0,
                                "spacing": 1,
                                "units": "s",
                            },
                        }
                    ],
                    "solver": {"damping": 1, "solution": "t", "report": "r"},
                },
            },
            "physical control direction",
        ),
        (
            {
                "action": "solve",
                "active": ["vp"],
                "state": "s",
                "covector": "c",
                "direction": "d",
                "model_gradient": True,
                "reduced_normal": {},
                "extension": {
                    "fields": [
                        {
                            "control": "vp",
                            "lags": {
                                "count": 1,
                                "origin": 0,
                                "spacing": 1,
                                "units": "s",
                            },
                        }
                    ],
                    "solver": {"damping": 1, "solution": "t", "report": "r"},
                },
            },
            "not both",
        ),
        (
            {
                "action": "solve",
                "active": ["vp"],
                "state": "s",
                "covector": "c",
                "model_gradient": True,
                "objective_vector": "o",
                "extension": {
                    "fields": [
                        {
                            "control": "vp",
                            "lags": {
                                "count": 1,
                                "origin": 0,
                                "spacing": 1,
                                "units": "s",
                            },
                        }
                    ],
                    "solver": {"damping": 1, "solution": "t", "report": "r"},
                },
            },
            "observed-data target",
        ),
        (
            {"action": "linearize", "active": [], "state": "s", "reduced_normal": {}},
            "requires a model extension",
        ),
        (
            {"action": "linearize", "active": [], "state": "s", "model_gradient": True},
            "action = solve",
        ),
        (
            {
                "action": "linearize",
                "active": [],
                "state": "s",
                "extension": {
                    "fields": [
                        {
                            "control": "vp",
                            "lags": {
                                "count": 1,
                                "origin": 0,
                                "spacing": 1,
                                "units": "s",
                            },
                        }
                    ]
                },
                "reflectivity": {
                    "parameterization": "vp_ip",
                    "fields": [{"name": "ip", "layer": 1, "axis": 2, "basis": "vp"}],
                },
            },
            "mutually exclusive",
        ),
        (
            {
                "action": "linearize",
                "active": [],
                "state": "s",
                "extension": {
                    "fields": [
                        {
                            "control": "vp",
                            "lags": {
                                "count": 1,
                                "origin": 0,
                                "spacing": 1,
                                "units": "s",
                            },
                        }
                    ]
                },
                "control_active": ["vp"],
            },
            "control_sensitivities",
        ),
        (
            {
                "action": "linearize",
                "active": [],
                "state": "s",
                "extension": {
                    "fields": [
                        {
                            "control": "vp",
                            "lags": {
                                "count": 1,
                                "origin": 0,
                                "spacing": 1,
                                "units": "s",
                            },
                        }
                    ]
                },
                "covector": "c",
            },
            "extension.covector",
        ),
        (
            {
                "action": "linearize",
                "active": [],
                "state": "s",
                "extension": {"fields": [{"control": "vp"}]},
            },
            "exactly one of lags or offsets",
        ),
        (
            {
                "action": "linearize",
                "active": [],
                "state": "s",
                "extension": {
                    "fields": [
                        {
                            "control": "vp",
                            "lags": {
                                "count": 1,
                                "origin": 0,
                                "spacing": 1,
                                "units": "s",
                            },
                        }
                    ],
                    "solver": {
                        "damping": 1,
                        "solution": "t",
                        "report": "r",
                        "lag_penalty": 1.0,
                    },
                },
            },
            "lag_scale",
        ),
        (
            {
                "action": "linearize",
                "active": ["vp"],
                "state": "s",
                "reflectivity": {
                    "parameterization": "vp_ip",
                    "fields": [{"name": "ip", "layer": 1, "axis": 3, "basis": "vp"}],
                },
            },
            "axis",
        ),
        (
            {
                "action": "linearize",
                "active": ["vp"],
                "state": "s",
                "reflectivity": {
                    "parameterization": "vp_ip",
                    "fields": [{"name": "ip", "layer": 1, "axis": 2}],
                },
            },
            "exactly one of basis or control",
        ),
        (
            {
                "action": "solve",
                "active": [],
                "state": "s",
                "source_controls": {},
                "extension": {
                    "fields": [
                        {
                            "control": "vp",
                            "lags": {
                                "count": 1,
                                "origin": 0,
                                "spacing": 1,
                                "units": "s",
                            },
                        }
                    ],
                    "solver": {"damping": 1, "solution": "t", "report": "r"},
                },
            },
            "source_controls",
        ),
        ({"action": "linearize", "active": ["vp", "model.vp"], "state": "s"}, "unique"),
        ({"action": "warp"}, "action must be one of"),
    ],
)
def test_fwi_operator_action_validation(tmp_path, kwargs, message):
    sim = _saved_simulation(tmp_path)
    with pytest.raises(ValueError, match=message):
        FWIOperatorJob("bad", sim, [4.0], **kwargs)


def test_fwi_fingerprints_hash_direction_and_baseline_contents(tmp_path):
    sim = _saved_simulation(tmp_path)
    direction = ControlVectorFile(
        {"model.vp": [1.0, 0.0]},
        state_fingerprint="s",
        control_registry_fingerprint="r",
    ).write(tmp_path / "direction.h5")
    baseline = tmp_path / "baseline.h5"
    baseline.write_bytes(b"one")
    job = FWIOperatorJob(
        "jvp",
        sim,
        [4.0],
        action="jvp",
        active=["vp"],
        state="state.json",
        direction=direction,
        objective_vector="jvp.json",
        control_state=baseline,
    )
    reference = job.fingerprint()
    task_reference = job.task_fingerprint(1)
    payload = job._input_fingerprint_payload()
    assert set(payload) == {"direction", "control_state"}

    ControlVectorFile(
        {"model.vp": [0.0, 1.0]},
        state_fingerprint="s",
        control_registry_fingerprint="r",
    ).write(direction)
    assert job.fingerprint() != reference
    assert job.task_fingerprint(1) != task_reference
    changed = job.fingerprint()
    baseline.write_bytes(b"two")
    assert job.fingerprint() != changed

    # The fwi_operator and Imaging blocks participate in the job fingerprint.
    same = FWIOperatorJob(
        "jvp",
        sim,
        [4.0],
        action="jvp",
        active=["vp"],
        state="state.json",
        direction=direction,
        objective_vector="jvp.json",
        control_state=baseline,
    )
    other = FWIOperatorJob(
        "jvp",
        sim,
        [4.0],
        action="jvp",
        active=["vp"],
        state="state.json",
        direction=direction,
        objective_vector="jvp.json",
        control_state=baseline,
        misfit={"objective": {"kind": "huber", "delta": 1.5}},
    )
    assert same.fingerprint() == job.fingerprint()
    assert other.fingerprint() != job.fingerprint()
    assert "fwi_operator" in job.effective_output_request_payload()


def test_fwi_vjp_fingerprint_hashes_per_task_objective_duals(tmp_path):
    sim = _saved_simulation(tmp_path)
    dual = tmp_path / "dual.json"
    job = FWIOperatorJob(
        "vjp",
        sim,
        [4.0, 6.0],
        action="vjp",
        active=["vp"],
        state="state.json",
        objective_vector=dual,
        covector="c.h5",
    )
    (tmp_path / "dual_1.json").write_text("{}")
    (tmp_path / "dual_2.json").write_text("{}")
    payload = job._input_fingerprint_payload()
    assert set(payload["objective_vector"]) == {"1", "2"}
    shared = tmp_path / "shared.json"
    shared.write_text("{}")
    job.objective_vector = shared
    assert set(job._input_fingerprint_payload()["objective_vector"]) == {"shared"}


# ---------------------------------------------------------------------------
# ControlGradientJob
# ---------------------------------------------------------------------------


def test_control_gradient_rtm_serializes_and_round_trips(tmp_path):
    sim = _saved_simulation(tmp_path)
    observed = tmp_path / "observed.h5"
    gradient = tmp_path / "controls" / "gradient.h5"
    raw = tmp_path / "controls" / "exact.h5"
    job = ControlGradientJob(
        "rtm",
        sim,
        [3.0 - 0.5j, 5.0],
        kind="rtm",
        observed={"seabed": observed},
        gradient=gradient,
        objective_file="objective.h5",
        current="current.h5",
        active=["salt_boundary_rbf", "sediment_sp"],
        source_taper={"d0": 20.0, "d1": 40.0, "units": "m"},
        spatial_window={
            "axis": "X",
            "minimum": 250.0,
            "maximum": 750.0,
            "taper": 50.0,
            "units": "m",
        },
        weights=[0.25, 0.75],
        raw_gradient=raw,
        smoothing=SmoothingConfig(
            kind="tv", wavelength_fraction=0.2, epsilon=1e-2, iterations=7
        ),
    )
    payload = _assert_valid(job.to_fs())
    assert payload["workflow"] == "rtm"
    assert payload["f_list"] == [[3.0, -0.5], [5.0, 0.0]]
    assert payload["control_sensitivities"] == {
        "gradient": str(gradient),
        "objective": str(job.objective_file()),
        "active": ["salt_boundary_rbf", "sediment_sp"],
        "source_taper": {"d0": 20.0, "d1": 40.0, "units": "m"},
        "spatial_window": {
            "axis": "x",
            "minimum": 250.0,
            "maximum": 750.0,
            "taper": 50.0,
            "units": "m",
        },
        "current": str(tmp_path / "current.h5"),
        "weights": [0.25, 0.75],
        "Smoothing": {
            "type": "tv",
            "lambda": 0.2,
            "derivative_order": 1,
            "epsilon": 1e-2,
            "iterations": 7,
            "input_role": "dual",
        },
        "raw_gradient": str(raw),
    }
    assert payload["Imaging"]["misfit"] == {
        "objective": {"kind": "l2"},
        "preprocess": {"include_defaults": False, "hooks": []},
        "receiver_groups": [{"name": "seabed", "observed": str(observed)}],
    }
    assert "Image" not in payload and "focus" not in payload
    assert job.gradient_file(2) == gradient.with_name("gradient_2.h5")
    assert job.gradient_file(raw=True) == raw
    assert job.objective_file(1) == job._result_path / "objective_1.h5"
    assert job.postprocess_fetch_files() == [raw, gradient, job.objective_file()]
    assert job.requires_postprocess()
    assert job.trace_outputs.groups == []

    loaded = _round_trip(job)
    assert loaded.kind == "rtm"
    assert loaded.observed == {"seabed": observed}
    assert loaded.active == ["salt_boundary_rbf", "sediment_sp"]
    assert loaded.smoothing == job.smoothing
    assert loaded.weights == [0.25, 0.75]
    job.objective_file().parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(job.objective_file(), "w") as h5:
        h5.create_dataset("value", data=7.5)
    assert loaded.objective_value == 7.5


def test_control_gradient_rtm_default_gradient_lives_in_results_and_uses_all_groups(
    tmp_path,
):
    sim = _elastic_simulation(tmp_path)
    job = ControlGradientJob(
        "rtm", sim, [3.0], kind="rtm", observed="observed", gradient="gradient.h5"
    )
    assert job.gradient == job._result_path / "gradient.h5"
    assert job.observed == {"surface": tmp_path / "observed"}
    payload = _assert_valid(job.to_fs())
    assert payload["Imaging"]["misfit"]["receiver_groups"] == [
        {"name": "surface", "observed": str(tmp_path / "observed")}
    ]
    assert job.gradient_file(raw=True).name == "gradient_raw.h5"


def test_control_gradient_rtm_uses_misfit_objects_and_kernel_derivatives(tmp_path):
    sim = _saved_simulation(tmp_path)

    class Misfit:
        def to_fs(self, ctx=None, *, project_relative=False):
            return {
                "objective_terms": [
                    {
                        "id": "waveform",
                        "receiver_group": "surface",
                        "weight": 0.5,
                        "normalization": {"scale": {"kind": "observed_rms"}},
                    }
                ],
                "receiver_groups": [
                    {"name": "surface", "observed_source_basis": "source_geometry"}
                ],
            }

    job = ControlGradientJob(
        "kernel",
        sim,
        [2.5],
        kind="rtm",
        observed="observed.h5",
        gradient="g.h5",
        misfit=Misfit(),
        kernel_derivative={"order": 4, "axis": "fourier", "residual": "jet"},
    )
    payload = _assert_valid(job.to_fs())
    misfit = payload["Imaging"]["misfit"]
    assert misfit["objective_terms"][0]["weight"] == 0.5
    assert misfit["receiver_groups"] == [
        {
            "name": "surface",
            "observed_source_basis": "source_geometry",
            "observed": str(tmp_path / "observed.h5"),
        }
    ]
    assert misfit["preprocess"] == {"include_defaults": False, "hooks": []}
    assert payload["Image"] == payload["Imaging"]
    assert payload["kernel_derivative"] == {
        "order": 4,
        "axis": "fourier",
        "residual": "jet",
    }
    loaded = _round_trip(job)
    assert loaded.kernel_derivative == job.kernel_derivative


def test_control_gradient_born_serializes_incremental_traces(tmp_path):
    sim = _elastic_simulation(tmp_path)
    direction = tmp_path / "controls" / "direction.h5"
    job = ControlGradientJob(
        "born",
        sim,
        [8.0 - 0.1j],
        kind="born",
        direction=direction,
        current="current.h5",
        active=["vp"],
        spatial_window={"axis": "z", "minimum": 0.0, "maximum": 1.0},
    )
    payload = _assert_valid(job.to_fs())
    assert payload["workflow"] == "born"
    assert payload["control_sensitivities"] == {
        "direction": str(direction),
        "active": ["vp"],
        "spatial_window": {
            "axis": "z",
            "minimum": 0.0,
            "maximum": 1.0,
            "taper": 0.0,
            "units": "m",
        },
        "current": str(tmp_path / "current.h5"),
    }
    assert "Imaging" not in payload
    assert job.trace_outputs.groups == ["surface_inc"]
    assert job.trace_outputs.components == ["surface_inc:vz"]
    assert not job.requires_postprocess()
    with pytest.raises(ValueError, match="do not write a gradient"):
        job.gradient_file()
    loaded = _round_trip(job)
    assert loaded.direction == direction
    assert loaded.observed is None


def test_control_gradient_focus_matches_pinned_examples(tmp_path):
    sim = _saved_simulation(tmp_path)
    for name, kind in (("focus-trfwi.json", "trfwi"), ("focus-weft.json", "weft")):
        example = _example(name)
        job = ControlGradientJob(
            example["name"],
            sim,
            [3.0, 4.0],
            kind="focus",
            observed="observed.h5",
            gradient=example["control_sensitivities"]["gradient"],
            objective_file=example["focus"]["objective"],
            focus={"kind": kind, "softening": 0.12},
            weights=[1.0, 1.0],
        )
        payload = _assert_valid(job.to_fs())
        assert payload["workflow"] == "focus"
        assert _shape(payload["focus"]) == example["focus"]
        assert (
            _shape(payload["control_sensitivities"])["gradient"]
            == example["control_sensitivities"]["gradient"]
        )
        assert job.objective_file(2).name == example["focus"]["objective"].replace(
            ".h5", "_2.h5"
        )
        assert job.postprocess_fetch_files()[-1] == job.objective_file()
        loaded = _round_trip(job)
        assert loaded.focus == {"softening": 0.12, "kind": kind}
        assert loaded.objective_file() == job.objective_file()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"kind": "rtm", "gradient": "g"}, "require observed data"),
        ({"kind": "rtm", "observed": "o"}, "require a gradient"),
        (
            {"kind": "rtm", "observed": "o", "gradient": "g", "direction": "d"},
            "do not take a direction",
        ),
        (
            {
                "kind": "rtm",
                "observed": "o",
                "gradient": "g",
                "focus": {"softening": 1.0},
            },
            "kind = 'focus'",
        ),
        (
            {
                "kind": "rtm",
                "observed": "o",
                "gradient": "g",
                "gram_derivative": "totla",
            },
            "gram_derivative",
        ),
        (
            {
                "kind": "rtm",
                "observed": "o",
                "gradient": "g",
                "gram_derivative": "total",
                "source_taper": {"d0": 1, "d1": 2},
            },
            "total Gram",
        ),
        (
            {
                "kind": "rtm",
                "observed": "o",
                "gradient": "g",
                "gram_derivative": "total",
                "misfit": {"comparison": {"kind": "phase_derivative"}},
            },
            "phase_derivative",
        ),
        ({"kind": "born"}, "require a direction"),
        ({"kind": "born", "direction": "d", "gradient": "g"}, "do not take gradient"),
        (
            {
                "kind": "born",
                "direction": "d",
                "kernel_derivative": {"residual": "jet"},
            },
            "do not take",
        ),
        (
            {"kind": "focus", "observed": "o", "gradient": "g", "objective_file": "f"},
            "require focus",
        ),
        (
            {
                "kind": "focus",
                "observed": "o",
                "gradient": "g",
                "focus": {"softening": 1.0},
            },
            "objective_file",
        ),
        (
            {
                "kind": "focus",
                "observed": "o",
                "gradient": "g",
                "objective_file": "f",
                "focus": {"softening": 0.0},
            },
            "positive",
        ),
        (
            {
                "kind": "focus",
                "observed": "o",
                "gradient": "g",
                "objective_file": "f",
                "focus": {"softening": 1.0, "kind": "beam"},
            },
            "focus kind",
        ),
        (
            {
                "kind": "focus",
                "observed": "o",
                "gradient": "g",
                "objective_file": "f",
                "focus": {"softening": 1.0},
                "source_taper": {"d0": 1, "d1": 2},
            },
            "source_taper",
        ),
        (
            {
                "kind": "focus",
                "observed": "o",
                "gradient": "g",
                "objective_file": "f",
                "focus": {"softening": 1.0},
                "kernel_derivative": {"residual": "jet"},
            },
            "workflow = rtm",
        ),
        (
            {"kind": "rtm", "observed": "o", "gradient": "g", "active": ["a", "a"]},
            "unique",
        ),
        (
            {"kind": "rtm", "observed": "o", "gradient": "g", "weights": [1.0, 2.0]},
            "one finite",
        ),
        (
            {
                "kind": "rtm",
                "observed": "o",
                "gradient": "g",
                "spatial_window": {"axis": "q", "minimum": 0, "maximum": 1},
            },
            "axis",
        ),
        ({"kind": "adjoint"}, "kind must be one of"),
    ],
)
def test_control_gradient_validation(tmp_path, kwargs, message):
    sim = _saved_simulation(tmp_path)
    with pytest.raises(ValueError, match=message):
        ControlGradientJob("bad", sim, [3.0], **kwargs)


def test_control_gradient_fingerprints_include_observed_and_direction(tmp_path):
    sim = _saved_simulation(tmp_path)
    direction = ControlVectorFile({"sediment_sp": [1.0, 0.0]}, native=True).write(
        tmp_path / "direction.h5"
    )
    observed = tmp_path / "observed"
    observed.mkdir()
    (observed / "trace.bin").write_bytes(b"first")
    born = ControlGradientJob("born", sim, [3.0], kind="born", direction=direction)
    rtm = ControlGradientJob(
        "rtm", sim, [3.0], kind="rtm", observed={"seabed": observed}, gradient="g.h5"
    )
    born_reference = born.fingerprint()
    rtm_reference = rtm.task_fingerprint(1)
    ControlVectorFile({"sediment_sp": [0.0, 1.0]}, native=True).write(direction)
    (observed / "trace.bin").write_bytes(b"second")
    assert born.fingerprint() != born_reference
    assert rtm.task_fingerprint(1) != rtm_reference
    for job in (born, rtm):
        loaded = BaseJob.load(job.save())
        pairs = loaded.remote_input_files("/remote/project")
        local = direction if job is born else observed
        assert any(Path(pair[0]) == local for pair in pairs)


# ---------------------------------------------------------------------------
# ImageKernelJob
# ---------------------------------------------------------------------------


def test_image_kernel_rtm_serializes_imaging_block(tmp_path):
    sim = _elastic_simulation(tmp_path)
    observed = tmp_path / "observed" / "traces"
    observed.mkdir(parents=True)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])
    job = ImageKernelJob(
        "rtm",
        sim,
        [5.0, 7.0],
        grid=grid,
        images={
            "vp": ImageSpec("fwi:elastic", "vp"),
            "pressure": "pressure",
            "shot": {"IC": "up_down", "sources": [1]},
            "vs": ("fwi:elastic", "vs"),
        },
        observed=observed,
        weights=[1.0, 0.5],
        keep_forward=True,
        field_retention="forward",
        smoothing=SmoothingConfig(kind="tv", wavelength_fraction=0.3),
        save_path=tmp_path / "images",
        post_process=True,
    )
    payload = _assert_valid(job.to_fs())
    imaging = payload["Imaging"]
    assert payload["workflow"] == "rtm"
    assert "Image" not in payload
    assert imaging["schema"] == "fs-imaging-1"
    assert imaging["data_path"] == str(observed)
    assert imaging["save_path"] == str(tmp_path / "images")
    assert (
        imaging["grid"]["n"] == [3, 2] and imaging["grid"]["_type"] == "CartesianGrid"
    )
    assert imaging["images"] == [
        {"name": "vp", "IC": "fwi:elastic", "property": "vp"},
        {"name": "pressure", "IC": "pressure"},
        {"name": "shot", "IC": "up_down", "sources": [1]},
        {"name": "vs", "IC": "fwi:elastic", "property": "vs"},
    ]
    assert imaging["misfit"] == {
        "objective": {"kind": "l2"},
        "receiver_groups": [{"name": "surface", "observed": str(observed)}],
    }
    assert imaging["weights"] == [1.0, 0.5]
    assert imaging["keep_forward"] is True and imaging["field_retention"] == "forward"
    assert imaging["Smoothing"] == {
        "type": "tv",
        "lambda": 0.3,
        "derivative_order": 1,
        "epsilon": 1e-3,
        "iterations": 5,
        "illumination_normalization": "none",
    }
    assert imaging["post_process"] is True
    assert "direction" not in imaging and "gauss_newton" not in imaging
    assert job.trace_outputs.groups == ["surface"]
    assert job.requires_postprocess()
    loaded = _round_trip(job)
    assert loaded.images == job.images
    assert loaded.grid.n == [3, 2]
    assert loaded.extra == {"post_process": True}
    # Imaging.Smoothing reloads as its Sauce mapping; the emitted contract matches.
    assert loaded._smoothing_payload() == job._smoothing_payload()


def test_image_kernel_project_relative_export_and_zero_data(tmp_path):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])
    job = ImageKernelJob(
        "kernel",
        sim,
        [5.0],
        grid=grid,
        images={"vp": ("fwi:elastic", "vp")},
        smoothing={"type": "tikhonov", "lambda": 0.5},
    )
    payload = _assert_valid(job.to_fs())
    assert payload["Imaging"]["data_path"] is None
    assert payload["Imaging"]["misfit"]["receiver_groups"] == [
        {"name": "surface", "observed": None}
    ]
    assert payload["Imaging"]["Smoothing"] == {
        "type": "tikhonov",
        "lambda": 0.5,
        "illumination_normalization": "none",
    }
    relative = job.to_fs(project_relative=True)
    assert relative["Imaging"]["save_path"] == Path(
        "jobs/smooth/kernel/results/imaging"
    )
    loaded = _round_trip(job)
    assert loaded.observed is None
    assert loaded.smoothing == {
        "type": "tikhonov",
        "lambda": 0.5,
        "illumination_normalization": "none",
    }


def test_image_kernel_born_and_lsrtm_workflows(tmp_path):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])
    normal = ImageKernelJob(
        "normal",
        sim,
        [5.0, 7.0],
        grid=grid,
        images={"dVp": ("fwi:elastic", "vp")},
        workflow="born",
        zero_direction=True,
        gauss_newton=True,
    )
    payload = _assert_valid(normal.to_fs())
    assert payload["workflow"] == "born"
    assert payload["Imaging"]["zero_direction"] is True
    assert payload["Imaging"]["gauss_newton"] is True
    assert payload["Imaging"]["born_traces_only"] is False
    assert normal.trace_outputs.groups == ["surface_inc"]
    assert normal.trace_outputs.components == ["surface_inc:vz"]
    assert normal._input_fingerprint_payload() == {"direction": {"kind": "zero"}}
    _round_trip(normal)

    direction = tmp_path / "direction.h5"
    direction.write_bytes(b"cartesian")
    observed = tmp_path / "observed"
    observed.mkdir()
    gradient = ImageKernelJob(
        "gradient",
        sim,
        [5.0, 7.0],
        grid=grid,
        images={"dVp": ("fwi:elastic", "vp")},
        workflow="lsrtm_gradient",
        direction=direction,
        observed=observed,
    )
    payload = _assert_valid(gradient.to_fs())
    assert payload["workflow"] == "lsrtm_gradient"
    assert payload["Imaging"]["direction"] == str(direction)
    assert payload["Imaging"]["gauss_newton"] is True
    assert "zero_direction" not in payload["Imaging"]
    assert gradient.trace_outputs.groups == ["surface", "surface_inc"]
    assert gradient.trace_outputs.components == ["surface:vz", "surface_inc:vz"]
    reference = gradient.fingerprint()
    direction.write_bytes(b"changed")
    assert gradient.fingerprint() != reference
    loaded = _round_trip(gradient)
    assert loaded.direction == direction
    assert loaded.gauss_newton is True

    traces_only = ImageKernelJob(
        "traces",
        sim,
        [5.0],
        grid=grid,
        images={"dVp": ("fwi:elastic", "vp")},
        workflow="born",
        direction=direction,
        born_traces_only=True,
    )
    assert _assert_valid(traces_only.to_fs())["Imaging"]["born_traces_only"] is True


def test_image_kernel_kernel_derivative_emits_image_and_imaging(tmp_path):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[161, 161], x0=[0, 0], x1=[10, 10], units="km")
    job = ImageKernelJob(
        "kernel",
        sim,
        [2.5],
        grid=grid,
        images={"vp": ("fwi:acoustic", "vp")},
        observed="observed.h5",
        field_retention="none",
        kernel_derivative={"order": 4, "axis": "fourier", "residual": "derivative"},
        preprocess={"include_defaults": False, "hooks": []},
    )
    payload = _assert_valid(job.to_fs())
    example = _example("total-kernel-derivative.json")
    assert payload["Image"] == payload["Imaging"]
    assert payload["Image"]["images"] == example["Image"]["images"]
    assert payload["Image"]["grid"]["n"] == example["Image"]["grid"]["n"]
    assert payload["Image"]["preprocess"] == example["Image"]["preprocess"]
    assert (
        payload["kernel_derivative"]["order"] == example["kernel_derivative"]["order"]
    )
    _round_trip(job)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"workflow": "rtm", "direction": "d"}, "do not take a direction"),
        ({"workflow": "rtm", "gauss_newton": True}, "Born workflows"),
        ({"workflow": "born"}, "requires direction or zero_direction"),
        (
            {"workflow": "born", "direction": "d", "zero_direction": True},
            "mutually exclusive",
        ),
        (
            {"workflow": "lsrtm_gradient", "zero_direction": True},
            "requires observed data",
        ),
        (
            {
                "workflow": "lsrtm_gradient",
                "zero_direction": True,
                "observed": "o",
                "born_traces_only": True,
            },
            "trace-only",
        ),
        (
            {
                "workflow": "born",
                "zero_direction": True,
                "kernel_derivative": {"residual": "jet"},
            },
            "workflow = rtm",
        ),
        ({"workflow": "adjoint"}, "workflow must be one of"),
        ({"images": {}}, "at least one image"),
        ({"weights": [1.0, 2.0]}, "one finite"),
        ({"field_retention": "some"}, "field_retention"),
        ({"smoothing": SmoothingConfig(derivative_order=2)}, "first-order"),
    ],
)
def test_image_kernel_validation(tmp_path, kwargs, message):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])
    options = {"grid": grid, "images": {"vp": ("fwi:elastic", "vp")}, **kwargs}
    with pytest.raises(ValueError, match=message):
        ImageKernelJob("bad", sim, [5.0], **options)


def test_image_spec_normalizes_inputs():
    assert ImageSpec.from_value("pressure") == ImageSpec("pressure")
    assert ImageSpec.from_value(("fwi", "vp")) == ImageSpec("fwi", "vp")
    assert ImageSpec.from_value(
        {"IC": "fwi", "property": "vp", "sources": [2, 1]}
    ).sources == [2, 1]
    assert ImageSpec.from_value({"condition": "energy"}).to_fs("e") == {
        "name": "e",
        "IC": "energy",
    }
    with pytest.raises(ValueError, match="unique one-based"):
        ImageSpec("fwi", sources=[0])
    with pytest.raises(TypeError):
        ImageSpec.from_value(3)


# ---------------------------------------------------------------------------
# SmoothJob
# ---------------------------------------------------------------------------


def test_smooth_job_wraps_control_gradient_parts(tmp_path):
    sim = _saved_simulation(tmp_path)
    source = ControlGradientJob(
        "rtm",
        sim,
        [2.0, 4.0],
        kind="rtm",
        observed="observed.h5",
        gradient="gradient.h5",
        active=["vp"],
    )
    smooth = SmoothJob(source, smoothing=SmoothingConfig(kind="tv"), weights=[1.0, 3.0])
    payload = _assert_valid(smooth.to_fs())

    assert smooth.name == "rtm_smooth"
    assert smooth.workflow == "rtm"
    assert smooth.postprocess_only is True
    assert not smooth.supports_trace_packing
    assert payload["_type"] == "SmoothJob"
    assert payload["name"] == "rtm_smooth"
    assert payload["f_list"] == [2.0, 4.0]
    assert payload["control_sensitivities"] == {
        "gradient": str(source.gradient),
        "active": ["vp"],
        "Smoothing": SmoothingConfig(kind="tv").to_control_fs(),
        "weights": [1.0, 3.0],
    }
    assert payload["smooth_source"]["_type"] == "ControlGradientJob"
    assert "artifact_contract" not in payload["smooth_source"]
    assert smooth.postprocess_file(1) == source.gradient_file(1)
    assert smooth.postprocess_fetch_files() == source.postprocess_fetch_files()[:2]
    assert smooth.requires_postprocess()
    assert not smooth.postprocess_part_outputs_exist()
    source.gradient.parent.mkdir(parents=True)
    source.gradient_file(1).write_bytes(b"1")
    source.gradient_file(2).write_bytes(b"2")
    assert smooth.postprocess_part_outputs_exist()
    assert smooth.needs_postprocess()
    assert not smooth.is_run_current()
    assert set(smooth._input_fingerprint_payload()["parts"]) == {"1", "2"}

    loaded = _round_trip(smooth)
    assert isinstance(loaded.source_job, ControlGradientJob)
    assert loaded.source_job.gradient == source.gradient
    assert loaded.weights == [1.0, 3.0]
    assert loaded.smoothing == SmoothingConfig(kind="tv")
    assert loaded.input_vector is None


def test_smooth_job_wraps_fwi_covector_parts(tmp_path):
    sim = _saved_simulation(tmp_path)
    source = FWIOperatorJob(
        "lin",
        sim,
        [4.0],
        action="linearize",
        active=["model.vp"],
        state="s.json",
        covector="gradient.h5",
        control_active=["vp"],
    )
    smooth = SmoothJob(source, smoothing={"type": "tikhonov", "lambda": 0.5})
    payload = _assert_valid(smooth.to_fs())
    assert payload["workflow"] == "fwi_operator"
    assert payload["fwi_operator"] == source.to_fs()["fwi_operator"]
    assert payload["control_sensitivities"]["gradient"] == str(source.covector)
    assert payload["control_sensitivities"]["active"] == ["vp"]
    assert payload["control_sensitivities"]["Smoothing"]["type"] == "tikhonov"
    assert smooth.gradient_file(1) == source.covector_file(1)
    assert smooth.gradient_file(raw=True) == source.covector_file(raw=True)
    loaded = _round_trip(smooth)
    assert isinstance(loaded.source_job, FWIOperatorJob)


def test_smooth_job_wraps_cartesian_images(tmp_path):
    sim = _elastic_simulation(tmp_path)
    grid = CartesianGrid(n=[3, 2], x0=[0.0, 0.0], x1=[1.0, 1.0])
    source = ImageKernelJob(
        "rtm",
        sim,
        [5.0, 7.0],
        grid=grid,
        images={"vp": ("fwi:elastic", "vp")},
        weights=[1.0, 1.0],
    )
    smooth = SmoothJob(
        source,
        smoothing=SmoothingConfig(kind="tgv", alpha1=1.0, alpha2=2.0),
        weights=[2.0, 3.0],
    )
    payload = _assert_valid(smooth.to_fs())
    assert (
        payload["Imaging"]["Smoothing"]
        == SmoothingConfig(kind="tgv", alpha1=1.0, alpha2=2.0).to_image_fs()
    )
    assert payload["Imaging"]["weights"] == [2.0, 3.0]
    assert payload["smooth_source"]["Imaging"]["weights"] == [1.0, 1.0]
    assert "control_sensitivities" not in payload
    assert smooth.mode == "image"
    assert smooth._input_fingerprint_payload() == {}
    loaded = _round_trip(smooth)
    assert isinstance(loaded.source_job, ImageKernelJob)
    assert loaded.weights == [2.0, 3.0]
    with pytest.raises(ValueError, match="control vectors only"):
        SmoothJob(source, smoothing={"type": "tv"}, input_vector=tmp_path / "v.h5")


def test_smooth_job_input_vector_uses_control_sensitivities_input(tmp_path):
    sim = _saved_simulation(tmp_path)
    source = FWIOperatorJob(
        "lin",
        sim,
        [2.0, 6.0 - 0.5j],
        action="linearize",
        active=["model.vp", "model.rho", "source.1.position"],
        state="s.json",
        covector="gradient.h5",
    )
    # A joint covector path is used in place; non-model blocks are left out
    # of ``active`` and ignored by Sauce.
    vector = ControlVectorFile(
        {"model.vp": [1.0, 2.0], "model.rho": [3.0], "source.1.position": [4.0, 5.0]},
        state_fingerprint="s",
        control_registry_fingerprint="r",
    ).write(tmp_path / "covector.h5")
    smooth = SmoothJob(
        source,
        smoothing=SmoothingConfig(kind="tv"),
        input_vector=vector,
        gradient="smoothed.h5",
    )
    assert smooth.f_list == source.f_list
    assert smooth.input_vector == vector
    assert smooth.gradient == smooth._result_path / "smoothed.h5"
    assert smooth.control_active == ["rho", "vp"]
    assert not (smooth._result_path / "smoothed_1.h5").exists()
    assert smooth.gradient_file() == smooth.gradient
    assert smooth.gradient_file(raw=True) == smooth._result_path / "smoothed_raw.h5"
    with pytest.raises(ValueError, match="no task parts"):
        smooth.gradient_file(1)
    assert smooth.postprocess_file() == smooth.gradient
    assert smooth.postprocess_file(1) == vector
    assert smooth.postprocess_file(2) == vector
    assert smooth.postprocess_part_outputs_exist()
    assert smooth.needs_postprocess()
    assert smooth.postprocess_fetch_files() == [
        smooth.gradient_file(raw=True),
        smooth.gradient,
    ]

    payload = _assert_valid(smooth.to_fs())
    assert payload["f_list"] == [[2.0, 0.0], [6.0, -0.5]]
    assert payload["fwi_operator"]["controls"]["active"] == [
        "model.vp",
        "model.rho",
        "source.1.position",
    ]
    assert payload["control_sensitivities"] == {
        "input": str(vector),
        "gradient": str(smooth.gradient),
        "active": ["rho", "vp"],
        "Smoothing": SmoothingConfig(kind="tv").to_control_fs(),
    }
    assert "smooth_input_vector" not in payload
    assert smooth._input_fingerprint_payload()["input_vector"]["kind"] == "file"

    loaded = _round_trip(smooth)
    assert loaded.input_vector == vector
    assert loaded.gradient == smooth.gradient
    assert loaded.f_list == source.f_list
    assert loaded.weights is None

    # An in-memory vector is written once, natively, next to the gradient.
    explicit = SmoothJob(
        source,
        smoothing={"type": "tv"},
        name="custom",
        input_vector=ControlVectorFile(
            {"model.vp": [1.0, 2.0]},
            state_fingerprint="s",
            control_registry_fingerprint="r",
        ),
    )
    assert explicit.name == "custom"
    assert explicit.gradient == explicit._result_path / "smoothed.h5"
    assert explicit.input_vector == explicit._result_path / "smoothed_input.h5"
    native = ControlVectorFile.read(explicit.input_vector)
    assert native.native is True
    assert native.names == ("vp",)
    np.testing.assert_array_equal(native["vp"], [1.0, 2.0])
    explicit_payload = _assert_valid(explicit.to_fs())
    assert explicit_payload["control_sensitivities"]["input"] == str(
        explicit.input_vector
    )
    assert explicit_payload["control_sensitivities"]["active"] == ["vp"]
    assert explicit.postprocess_part_outputs_exist()

    with pytest.raises(ValueError, match="ignored when input_vector"):
        SmoothJob(
            source, smoothing={"type": "tv"}, input_vector=vector, weights=[1.0, 1.0]
        )
    with pytest.raises(ValueError, match="only used with input_vector"):
        SmoothJob(source, smoothing={"type": "tv"}, gradient="g.h5")
    with pytest.raises(ValueError, match="no model"):
        SmoothJob(
            source,
            smoothing={"type": "tv"},
            input_vector=ControlVectorFile(
                {"source.1.position": [1.0, 2.0]},
                state_fingerprint="s",
                control_registry_fingerprint="r",
            ),
        )


def test_smooth_job_rejects_sources_without_parts(tmp_path):
    sim = _saved_simulation(tmp_path)
    born = ControlGradientJob("born", sim, [3.0], kind="born", direction="d.h5")
    with pytest.raises(ValueError, match="no gradient parts"):
        SmoothJob(born, smoothing={"type": "tv"})
    calibrate = FWIOperatorJob("cal", sim, [3.0], action="calibrate", balance="b.json")
    with pytest.raises(ValueError, match="no covector"):
        SmoothJob(calibrate, smoothing={"type": "tv"})
    with pytest.raises(TypeError, match="requires an"):
        SmoothJob(object(), smoothing={"type": "tv"})
