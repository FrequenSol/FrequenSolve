import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from frequensolve.model.parameterization import (
    BSplineControl,
    HatControl,
    ParameterizedProperty,
)
from frequensolve.model.property import Property
from frequensolve.model.representation import VariationalSmoothing
from frequensolve.simulation.jobs import (
    BaseJob,
    BornControlSensitivityJob,
    ControlBlock,
    ControlSpace,
    PreprocessHook,
    RTMControlSensitivityJob,
    TimeReversalFocusJob,
)
from frequensolve.simulation.simulation import SeismicSimulation


def _saved_simulation(tmp_path):
    simulation = SeismicSimulation(
        name="controlled",
        physics="acoustic",
        dimension=2,
        project_path=tmp_path,
    )
    simulation.save()
    return simulation


@pytest.mark.parametrize("mode", ["frozen", "total"])
def test_rtm_gram_derivative_serialization_round_trip(tmp_path, mode):
    job = RTMControlSensitivityJob(
        "gram",
        _saved_simulation(tmp_path),
        [3.0],
        observed=tmp_path / "observed.h5",
        gradient=tmp_path / "gradient.h5",
        gram_derivative=mode,
    )
    payload = job.to_fs()
    config = payload["control_sensitivities"]
    assert config.get("gram_derivative", "frozen") == mode
    assert ("gram_derivative" in config) == (mode == "total")
    restored = RTMControlSensitivityJob.from_fs(payload)
    assert restored.gram_derivative == mode


def test_rtm_rejects_unknown_gram_derivative_mode(tmp_path):
    with pytest.raises(ValueError, match="gram_derivative"):
        RTMControlSensitivityJob(
            "gram",
            _saved_simulation(tmp_path),
            [3.0],
            observed=tmp_path / "observed.h5",
            gradient=tmp_path / "gradient.h5",
            gram_derivative="totla",
        )


@pytest.mark.parametrize(
    "option",
    [
        {"source_taper": {"d0": 10.0, "d1": 20.0}},
        {"spatial_window": {"axis": "z", "minimum": 0.0, "maximum": 100.0}},
        {"comparison": {"kind": "phase_derivative"}},
    ],
)
def test_total_gram_rejects_unsupported_gradient_modifications(tmp_path, option):
    with pytest.raises(ValueError, match="total Gram derivatives"):
        RTMControlSensitivityJob(
            "gram",
            _saved_simulation(tmp_path),
            [3.0],
            observed=tmp_path / "observed.h5",
            gradient=tmp_path / "gradient.h5",
            gram_derivative="total",
            **option,
        )


def test_hat_control_is_an_ordered_uniform_grid_without_point_ids():
    control = HatControl(
        coordinate_system="top_relative",
        axis="below",
        origin=0.1,
        spacing=0.25,
        coefficients=[0.0, 0.2, -0.1],
    )

    np.testing.assert_allclose(control.coordinates, [0.1, 0.35, 0.6])
    assert control.to_fs() == {
        "kind": "hat",
        "coordinate_system": "top_relative",
        "axis": "below",
        "origin": 0.1,
        "spacing": 0.25,
        "coefficients": [0.0, 0.2, -0.1],
    }


def test_property_control_coefficients_reject_complex_values():
    with pytest.raises(ValueError, match="real-valued"):
        HatControl(axis="z", spacing=1.0, coefficients=[0.0j, 0.0j])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"spacing": 0.0, "coefficients": [0.0, 0.0]}, "positive"),
        ({"spacing": 1.0, "coefficients": [0.0]}, "at least 2"),
        ({"spacing": 1.0, "coefficients": [0.0, np.nan]}, "finite"),
    ],
)
def test_hat_control_rejects_invalid_uniform_grids(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HatControl(axis="z", **kwargs)


def test_bspline_control_validates_coefficient_count():
    with pytest.raises(ValueError, match=r"len\(knots\) - degree - 1"):
        BSplineControl(
            axis="z",
            degree=2,
            knots=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            coefficients=[0.0, 0.0],
        )


def test_parameterized_property_round_trips_as_a_property_provider():
    authored = ParameterizedProperty(
        0.5,
        id="sediment_sp",
        transform="log",
        control=HatControl(
            coordinate_system="top_relative",
            axis="below",
            spacing=0.1,
            coefficients=[0.0, 0.0, 0.0],
        ),
    )

    payload = authored.to_fs()
    loaded = Property.from_value(payload)

    assert isinstance(loaded, ParameterizedProperty)
    assert loaded.id == "sediment_sp"
    assert loaded.transform == "log"
    np.testing.assert_array_equal(loaded.coefficients, np.zeros(3))
    assert loaded.to_fs() == payload


def test_parameterized_property_accepts_inline_xarray_reference_payload():
    payload = {
        "parameterized": {
            "id": "sediment_sp",
            "reference": {
                "dims": ["z"],
                "coords": {"z": {"data": [0.0, 0.1, 0.2]}},
                "data": [0.5, 0.45, 0.4],
            },
            "transform": "log",
            "control": {
                "kind": "hat",
                "axis": "below",
                "spacing": 0.1,
                "coefficients": [0.0, 0.0, 0.0],
            },
        }
    }

    loaded = Property.from_value(payload)

    assert isinstance(loaded, ParameterizedProperty)
    np.testing.assert_allclose(loaded.reference.data, [0.5, 0.45, 0.4])
    np.testing.assert_allclose(loaded.reference.data.coords["z"], [0.0, 0.1, 0.2])


def test_parameterized_property_preserves_value_and_control_units():
    authored = ParameterizedProperty(
        {"value": 0.55},
        id="sediment_sp",
        units="s/km",
        transform="log",
        control=HatControl(
            coordinate_system="top_relative",
            axis="below",
            units="m",
            spacing=50.0,
            coefficients=[0.0, 0.0, 0.0],
        ),
    )

    assert authored.to_fs() == {
        "parameterized": {
            "id": "sediment_sp",
            "reference": {"value": 0.55},
            "transform": "log",
            "control": {
                "kind": "hat",
                "coordinate_system": "top_relative",
                "axis": "below",
                "origin": 0.0,
                "spacing": 50.0,
                "coefficients": [0.0, 0.0, 0.0],
                "units": "m",
            },
        },
        "units": "s/km",
    }


def test_parameterized_property_promotes_reference_units_to_the_wrapper():
    authored = ParameterizedProperty(
        Property.file("starting_model.h5:/Vp", units="m/s"),
        id="sediment_vp",
        transform="log",
        control=HatControl(
            axis="z",
            spacing=0.5,
            coefficients=[0.0, 0.0],
        ),
    )

    payload = authored.to_fs()

    assert authored.units == "m/s"
    assert payload["units"] == "m/s"
    assert "units" not in payload["parameterized"]["reference"]
    assert Property.from_value(payload).to_fs() == payload


def test_parameterized_property_rejects_conflicting_reference_units():
    with pytest.raises(ValueError, match="units disagree with reference units"):
        ParameterizedProperty(
            Property.file("starting_model.h5:/Vp", units="m/s"),
            id="sediment_vp",
            units="km/s",
            transform="log",
            control=HatControl(
                axis="z",
                spacing=0.5,
                coefficients=[0.0, 0.0],
            ),
        )


def test_control_space_uses_lexical_blocks_and_float64_hdf5(tmp_path):
    space = ControlSpace(
        [
            ControlBlock("sediment_sp", 3, coordinates=[0.0, 0.1, 0.2]),
            ControlBlock("water_rho", 2),
        ]
    )
    values = {
        "water_rho": [4.0, 5.0],
        "sediment_sp": [1.0, 2.0, 3.0],
    }

    vector = space.pack(values)
    path = space.write_hdf5(tmp_path / "controls.h5", vector)

    assert [block.id for block in space.blocks] == ["sediment_sp", "water_rho"]
    np.testing.assert_array_equal(vector, [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(space.read_hdf5(path), vector)


def test_control_space_can_be_derived_from_parameterized_properties():
    sediment = ParameterizedProperty(
        0.5,
        id="sediment_sp",
        transform="log",
        control=HatControl(
            axis="below",
            origin=0.0,
            spacing=0.1,
            coefficients=[0.0, 0.0, 0.0],
        ),
    )

    space = ControlSpace.from_property(sediment)

    assert [block.id for block in space.blocks] == ["sediment_sp"]
    assert space.size == 3
    np.testing.assert_allclose(space.blocks[0].coordinates, [0.0, 0.1, 0.2])


def test_native_control_sensitivity_jobs_serialize_and_round_trip(tmp_path):
    simulation = _saved_simulation(tmp_path)
    direction = tmp_path / "controls" / "direction.h5"
    current = tmp_path / "controls" / "current.h5"
    observed = tmp_path / "observed.h5"
    gradient = tmp_path / "controls" / "gradient.h5"

    born = BornControlSensitivityJob(
        "born",
        simulation,
        [3.0 - 0.5j],
        direction=direction,
        current=current,
        active=["salt_boundary_rbf", "sediment_sp"],
        source_taper={"d0": 20.0, "d1": 40.0, "units": "m"},
        spatial_window={
            "axis": "X",
            "minimum": 250.0,
            "maximum": 750.0,
            "taper": 50.0,
            "units": "m",
        },
    )
    rtm = RTMControlSensitivityJob(
        "rtm",
        simulation,
        [3.0 - 0.5j],
        observed={"seabed": observed},
        gradient=gradient,
        current=current,
        active=["salt_boundary_rbf", "sediment_sp"],
        source_taper={"d0": 20.0, "d1": 40.0, "units": "m"},
        spatial_window={
            "axis": "x",
            "minimum": 250.0,
            "maximum": 750.0,
            "taper": 50.0,
            "units": "m",
        },
    )

    born_file = born.save()
    rtm_file = rtm.save()
    born_payload = json.loads(born_file.read_text())
    rtm_payload = json.loads(rtm_file.read_text())

    assert born_payload["workflow"] == "born"
    assert born_payload["control_sensitivities"] == {
        "direction": str(direction),
        "current": str(current),
        "active": ["salt_boundary_rbf", "sediment_sp"],
        "source_taper": {"d0": 20.0, "d1": 40.0, "units": "m"},
        "spatial_window": {
            "axis": "x",
            "minimum": 250.0,
            "maximum": 750.0,
            "taper": 50.0,
            "units": "m",
        },
    }
    assert rtm_payload["workflow"] == "rtm"
    assert rtm_payload["control_sensitivities"] == {
        "gradient": str(gradient),
        "current": str(current),
        "active": ["salt_boundary_rbf", "sediment_sp"],
        "source_taper": {"d0": 20.0, "d1": 40.0, "units": "m"},
        "spatial_window": {
            "axis": "x",
            "minimum": 250.0,
            "maximum": 750.0,
            "taper": 50.0,
            "units": "m",
        },
    }
    assert rtm_payload["Image"]["misfit"]["receiver_groups"] == [
        {"name": "seabed", "observed": "observed.h5"}
    ]
    assert rtm_payload["Image"]["misfit"]["preprocess"] == {
        "include_defaults": False,
        "hooks": [],
    }
    loaded_born = BaseJob.load(born_file)
    loaded_rtm = BaseJob.load(rtm_file)
    assert isinstance(loaded_born, BornControlSensitivityJob)
    assert isinstance(loaded_rtm, RTMControlSensitivityJob)
    assert loaded_born.direction == direction
    assert loaded_born.current == current
    assert loaded_born.active == ["salt_boundary_rbf", "sediment_sp"]
    assert loaded_born.source_taper == {"d0": 20.0, "d1": 40.0, "units": "m"}
    assert loaded_born.spatial_window == {
        "axis": "x",
        "minimum": 250.0,
        "maximum": 750.0,
        "taper": 50.0,
        "units": "m",
    }
    assert loaded_rtm.observed == {"seabed": observed}
    assert loaded_rtm.gradient == gradient
    assert loaded_rtm.current == current
    assert loaded_rtm.active == ["salt_boundary_rbf", "sediment_sp"]
    assert loaded_rtm.source_taper == {"d0": 20.0, "d1": 40.0, "units": "m"}
    assert loaded_rtm.spatial_window == loaded_born.spatial_window


@pytest.mark.parametrize(
    "active",
    [[], [""], ["salt/interface"], ["salt", "salt"]],
)
def test_native_control_sensitivity_jobs_reject_invalid_active_subspaces(
    tmp_path, active
):
    simulation = _saved_simulation(tmp_path)

    with pytest.raises(ValueError, match="active controls"):
        BornControlSensitivityJob(
            "born",
            simulation,
            [3.0],
            direction=tmp_path / "direction.h5",
            active=active,
        )


@pytest.mark.parametrize(
    "source_taper",
    [
        {},
        {"d0": -1.0, "d1": 2.0},
        {"d0": 2.0, "d1": 2.0},
        {"d0": 1.0, "d1": np.inf},
        {"d0": 1.0, "d1": 2.0, "units": ""},
    ],
)
def test_native_control_sensitivity_jobs_reject_invalid_source_tapers(
    tmp_path, source_taper
):
    simulation = _saved_simulation(tmp_path)

    with pytest.raises(ValueError, match="source taper"):
        BornControlSensitivityJob(
            "born",
            simulation,
            [3.0],
            direction=tmp_path / "direction.h5",
            source_taper=source_taper,
        )


@pytest.mark.parametrize(
    "spatial_window",
    [
        {},
        {"axis": "r", "minimum": 0.0, "maximum": 1.0},
        {"axis": "x", "minimum": 1.0, "maximum": 1.0},
        {"axis": "x", "minimum": 0.0, "maximum": np.inf},
        {"axis": "x", "minimum": 0.0, "maximum": 1.0, "taper": -0.1},
        {"axis": "x", "minimum": 0.0, "maximum": 1.0, "taper": 0.6},
        {"axis": "x", "minimum": 0.0, "maximum": 1.0, "units": ""},
    ],
)
def test_native_control_sensitivity_jobs_reject_invalid_spatial_windows(
    tmp_path, spatial_window
):
    simulation = _saved_simulation(tmp_path)

    with pytest.raises(ValueError, match="spatial window"):
        BornControlSensitivityJob(
            "born",
            simulation,
            [3.0],
            direction=tmp_path / "direction.h5",
            spatial_window=spatial_window,
        )


def test_rtm_control_job_serializes_frequency_postprocess_and_shard_paths(tmp_path):
    simulation = _saved_simulation(tmp_path)
    gradient = tmp_path / "controls" / "gradient.h5"
    raw_gradient = tmp_path / "controls" / "exact_vjp.h5"
    job = RTMControlSensitivityJob(
        "rtm",
        simulation,
        [2.0, 4.0],
        observed=tmp_path / "observed.h5",
        gradient=gradient,
        raw_gradient=raw_gradient,
        weights=[0.25, 0.75],
        smoothing=VariationalSmoothing(
            kind="tv",
            wavelength_fraction=0.2,
            epsilon=1.0e-2,
            iterations=7,
            input_role="dual",
        ),
    )

    payload = job.to_fs()["control_sensitivities"]

    assert payload == {
        "gradient": str(gradient),
        "weights": [0.25, 0.75],
        "Smoothing": {
            "type": "tv",
            "lambda": 0.2,
            "derivative_order": 1,
            "epsilon": 1.0e-2,
            "iterations": 7,
            "input_role": "dual",
        },
        "raw_gradient": str(raw_gradient),
    }
    assert job.gradient_file(1) == tmp_path / "controls" / "gradient_1.h5"
    assert job.gradient_file(2) == tmp_path / "controls" / "gradient_2.h5"
    assert job.gradient_file(raw=True) == raw_gradient
    assert job.postprocess_fetch_files() == [raw_gradient, gradient]
    assert job.requires_postprocess()
    assert not job.needs_postprocess()

    gradient.parent.mkdir(parents=True)
    job.gradient_file(1).touch()
    job.gradient_file(2).touch()
    assert job.needs_postprocess()
    gradient.touch()
    assert not job.needs_postprocess()


def test_control_job_compatible_fingerprint_materializes_large_trace_weights(
    tmp_path,
):
    simulation = _saved_simulation(tmp_path)
    (tmp_path / "observed.h5").write_bytes(b"observed")
    job = RTMControlSensitivityJob(
        "rtm",
        simulation,
        [2.0],
        observed=tmp_path / "observed.h5",
        gradient=tmp_path / "gradient.h5",
        preprocess=[PreprocessHook.trace_weight(np.ones(749))],
    )

    payload = job.task_policy_fingerprint_payload(1, "compatible")

    assert payload["job"]["workflow"] == "rtm"
    with h5py.File(job._local_path / "inputs.h5", "r") as h5:
        datasets = []
        h5.visititems(
            lambda _, value: (
                datasets.append(value) if isinstance(value, h5py.Dataset) else None
            )
        )
        assert [dataset.shape for dataset in datasets] == [(749,)]


def test_tgv_smoothing_serializes_native_weights():
    smoothing = VariationalSmoothing(
        kind="tgv",
        wavelength_fraction=0.4,
        alpha1=0.3,
        alpha2=0.7,
        tgv_ratio=1.25,
        epsilon=2.0e-3,
        iterations=9,
    )

    assert smoothing.to_fs() == {
        "type": "tgv",
        "lambda": 0.4,
        "derivative_order": 1,
        "epsilon": 2.0e-3,
        "iterations": 9,
        "input_role": "dual",
        "alpha1": 0.3,
        "alpha2": 0.7,
        "tgv_ratio": 1.25,
        "illumination_normalization": "none",
    }


def test_smoothing_serializes_explicit_amplitude_normalization():
    smoothing = VariationalSmoothing(
        kind="tgv",
        wavelength_fraction=0.3,
        normalize_amplitude=False,
    )

    assert smoothing.to_fs()["normalize_amplitude"] is False


def test_rtm_control_job_round_trips_smoothing_configuration(tmp_path):
    simulation = _saved_simulation(tmp_path)
    objective = tmp_path / "objective.h5"
    job = RTMControlSensitivityJob(
        "rtm",
        simulation,
        [2.0, 4.0],
        observed=tmp_path / "observed.h5",
        gradient=tmp_path / "gradient.h5",
        objective_file=objective,
        weights=[1.0, 2.0],
        smoothing={"type": "tikhonov", "lambda": 0.5, "input_role": "dual"},
    )

    loaded = BaseJob.load(job.save())

    assert isinstance(loaded, RTMControlSensitivityJob)
    assert loaded.objective_file() == objective
    assert loaded.objective_file(2) == tmp_path / "objective_2.h5"
    assert loaded.weights == [1.0, 2.0]
    assert loaded.smoothing == VariationalSmoothing(
        kind="tikhonov", wavelength_fraction=0.5, input_role="dual"
    )
    objective.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(objective, "w") as h5:
        h5.create_dataset("value", data=7.5)
    assert loaded.objective_value == 7.5


def test_time_reversal_focus_job_serializes_and_round_trips(tmp_path):
    simulation = _saved_simulation(tmp_path)
    gradient = tmp_path / "controls" / "focus_gradient.h5"
    objective = tmp_path / "controls" / "focus_objective.h5"
    observed = tmp_path / "observed.h5"
    job = TimeReversalFocusJob(
        "focus",
        simulation,
        [2.0, 4.0],
        observed={"seabed": observed},
        gradient=gradient,
        objective_file=objective,
        softening=12.5,
        weights=[0.25, 0.75],
        preprocess=[PreprocessHook.trace_weight([1.0, 0.5])],
    )

    job_file = job.save()
    payload = json.loads(job_file.read_text())

    assert payload["workflow"] == "focus"
    assert payload["focus"] == {
        "objective": str(objective),
        "softening": 12.5,
        "distance_power": 1.0,
    }
    weight_ref = payload["Image"]["misfit"]["preprocess"]["hooks"][0]["params"]
    assert weight_ref["layout"] == "receiver"
    assert weight_ref["weights"]["_type"] == "HDF5Dense"
    assert job.focus_objective_file(2) == objective.with_name("focus_objective_2.h5")

    loaded = BaseJob.load(job_file)
    assert isinstance(loaded, TimeReversalFocusJob)
    assert loaded.observed == {"seabed": observed}
    assert loaded.gradient == gradient
    assert loaded.focus_objective_file() == objective
    assert loaded.softening == 12.5
    assert loaded.weights == [0.25, 0.75]

    objective.parent.mkdir(parents=True)
    with h5py.File(objective, "w") as h5:
        h5.create_dataset("value", data=-3.25)
    assert loaded.objective_value == -3.25


def test_born_trace_output_spec_uses_incremental_receiver_groups(tmp_path):
    simulation = _saved_simulation(tmp_path)

    class Device:
        @staticmethod
        def output_components():
            return [SimpleNamespace(name="axial_strain")]

    simulation.acquisition = SimpleNamespace(
        receiver_groups=[SimpleNamespace(name="buried_das", device=Device())],
        source_field_ids=lambda: [1],
    )
    born = BornControlSensitivityJob(
        "born",
        simulation,
        [8.0 - 0.1j],
        direction=tmp_path / "direction.h5",
    )

    assert born.trace_outputs.groups == ["buried_das_inc"]
    assert born.trace_outputs.components == ["buried_das_inc:axial_strain"]


def test_control_job_fingerprints_include_input_file_contents(tmp_path):
    simulation = _saved_simulation(tmp_path)
    space = ControlSpace([ControlBlock("sediment_sp", 2)])
    direction = space.write_hdf5(tmp_path / "direction.h5", [1.0, 0.0])
    observed = tmp_path / "observed"
    observed.mkdir()
    (observed / "trace.bin").write_bytes(b"first")

    born = BornControlSensitivityJob(
        "born",
        simulation,
        [3.0 - 0.5j],
        direction=direction,
    )
    rtm = RTMControlSensitivityJob(
        "rtm",
        simulation,
        [3.0 - 0.5j],
        observed={"seabed": observed},
        gradient=tmp_path / "gradient.h5",
    )
    born_fingerprint = born.fingerprint()
    born_task_fingerprint = born.task_fingerprint(1)
    born_compatible = born.task_policy_fingerprint(1, "compatible")
    rtm_fingerprint = rtm.fingerprint()
    rtm_task_fingerprint = rtm.task_fingerprint(1)
    rtm_compatible = rtm.task_policy_fingerprint(1, "compatible")

    space.write_hdf5(direction, [0.0, 1.0])
    (observed / "trace.bin").write_bytes(b"second")

    assert born.task_policy_fingerprint(1, "compatible") != born_compatible
    assert rtm.task_policy_fingerprint(1, "compatible") != rtm_compatible
    assert born.fingerprint() != born_fingerprint
    assert born.task_fingerprint(1) != born_task_fingerprint
    assert rtm.fingerprint() != rtm_fingerprint
    assert rtm.task_fingerprint(1) != rtm_task_fingerprint


def test_loaded_control_jobs_stage_direction_and_current_files(tmp_path):
    simulation = _saved_simulation(tmp_path)
    direction = tmp_path / "direction.h5"
    direction.write_bytes(b"direction")
    observed = tmp_path / "observed.h5"
    observed.write_bytes(b"observed")
    jobs = [
        BornControlSensitivityJob("born", simulation, [3.0], direction=direction),
        RTMControlSensitivityJob(
            "rtm",
            simulation,
            [3.0],
            observed=observed,
            gradient=tmp_path / "gradient.h5",
            current=direction,
        ),
    ]
    for job in jobs:
        loaded = BaseJob.load(job.save())
        pairs = loaded.remote_input_files("/remote/project")
        assert (direction, type(direction)("/remote/project/direction.h5")) in pairs


def test_control_paths_are_project_relative_before_and_after_reload(
    tmp_path, monkeypatch
):
    from frequensolve.simulation.jobs.imaging import ObservedTraceDerivatives

    project = tmp_path / "project"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    simulation = _saved_simulation(project)
    born = BornControlSensitivityJob(
        "born-paths", simulation, [1.0], direction="direction.h5", current="current.h5"
    )
    rtm = RTMControlSensitivityJob(
        "rtm-paths",
        simulation,
        [1.0],
        observed="observed.h5",
        gradient="gradient.h5",
        current="current.h5",
        objective_file="objective.h5",
        raw_gradient="raw.h5",
        observed_derivatives=ObservedTraceDerivatives.packed(
            "derivatives.h5", receiver_group="surface"
        ),
    )
    focus = TimeReversalFocusJob(
        "focus-paths",
        simulation,
        [1.0],
        observed="observed.h5",
        gradient="focus-gradient.h5",
        objective_file="focus-objective.h5",
        softening=1.0,
    )
    assert born.direction == project / "direction.h5"
    assert born.current == project / "current.h5"
    assert rtm.observed["surface"] == project / "observed.h5"
    assert rtm.gradient == project / "gradient.h5"
    assert rtm.current == project / "current.h5"
    assert rtm._objective_file == project / "objective.h5"
    assert rtm.raw_gradient == project / "raw.h5"
    assert rtm.observed_derivatives["surface"].df.file == project / "derivatives.h5"
    assert focus.focus_objective_file() == project / "focus-objective.h5"
    for job in (born, rtm, focus):
        restored = BaseJob.load(job.save())
        assert restored.to_fs() == job.to_fs()
