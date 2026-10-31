"""Native time configuration, committed output IO and lifecycle contracts."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import frequensolve as fs
from frequensolve.orchestrator.sites.local.site import _read_task_result


def job_at(tmp_path, **kwargs):
    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=2)
    sim.discretization = fs.Discretization(method="Galerkin")
    sim.model.attenuation_model = "none"
    sim.mesh.set_adapt(elems_per_wave=1, f_adapt=20)
    job = fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(0, 1, 0.1),
        source=fs.TimeFunction.ricker(10, delay=0.3),
        **kwargs,
    )
    job.save()
    return job


@pytest.mark.parametrize(
    "schedule,steps",
    [
        (fs.TimeOutput(), list(range(11))),
        (fs.TimeOutput(stride=3), [0, 3, 6, 9, 10]),
        (fs.TimeOutput(interval=0.3, write_initial=False), [3, 6, 9, 10]),
        (fs.TimeOutput(times=[0, 0.2, 0.7], write_initial=False), [0, 2, 7, 10]),
        (fs.TimeOutput(times=[0.2, 0.7]), [0, 2, 7, 10]),
    ],
)
def test_output_schedule(schedule, steps):
    np.testing.assert_array_equal(schedule.steps(fs.TimeIntegration(0, 1, 0.1)), steps)
    assert fs.TimeOutput.from_fs(schedule.to_fs()) == schedule


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(stride=True),
        dict(stride=0),
        dict(stride=1.5),
        dict(interval=float("nan")),
        dict(interval=0),
        dict(times=[]),
        dict(times=[1, 0]),
        dict(times=[0.1, 0.1]),
        dict(times=[float("inf")]),
        dict(write_initial=1),
        dict(times=[0.1], stride=2),
        dict(interval=0.1, stride=2),
    ],
)
def test_invalid_schedule(kwargs):
    with pytest.raises((ValueError, TypeError)):
        fs.TimeOutput(**kwargs)


@pytest.mark.parametrize(
    "schedule",
    [
        fs.TimeOutput(times=[0.15]),
        fs.TimeOutput(interval=0.15),
        fs.TimeOutput(times=[-1]),
        fs.TimeOutput(times=[2]),
    ],
)
def test_schedule_must_align_and_fit(schedule):
    with pytest.raises(ValueError):
        schedule.steps(fs.TimeIntegration(0, 1, 0.1))


@pytest.mark.parametrize(
    "start,end,dt",
    [
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 0.3),
        (True, 1, 0.1),
        (0, float("inf"), 0.1),
        (0, 1, float("nan")),
    ],
)
def test_invalid_integration(start, end, dt):
    with pytest.raises(ValueError):
        fs.TimeIntegration(start, end, dt)


@pytest.mark.parametrize(
    "function",
    [
        fs.TimeFunction.ricker(4, delay=0.5),
        fs.TimeFunction.gaussian(0.1, delay=0.5),
        fs.TimeFunction.sine(4),
        fs.TimeFunction.constant(2),
    ],
)
def test_analytic_source_derivatives(function):
    t = 0.23
    h = 1e-6
    for derivative in (1, 2):
        numerical = (
            function.evaluate(t + h, derivative=derivative - 1)
            - function.evaluate(t - h, derivative=derivative - 1)
        ) / (2 * h)
        np.testing.assert_allclose(
            function.evaluate(t, derivative=derivative), numerical, rtol=1e-7, atol=1e-7
        )
    assert fs.TimeFunction.from_fs(function.to_fs()) == function


def test_roundtrip_and_time_fingerprints(tmp_path):
    job = job_at(tmp_path, output_schedule=fs.TimeOutput(times=[0.2, 0.7]))
    original = job.to_fs()
    assert original["workflow"] == "transient"
    assert not {"f_list", "time_reconstruction", "k_list"} & original.keys()
    assert job.n_tasks == 1 and not job.supports_trace_packing
    restored = fs.BaseJob.load(job.job_file)
    assert isinstance(restored, fs.TransientJob)
    assert restored.to_fs() == original
    before = job.task_fingerprint(1)
    job.output_schedule = fs.TimeOutput(stride=2)
    assert job.task_fingerprint(1) != before
    before = job.task_fingerprint(1)
    job.time = fs.TimeIntegration(0, 1, 0.05)
    assert job.task_fingerprint(1) != before
    with pytest.raises(ValueError):
        job.plan_tasks(skip_policy={"accept_failed": True})


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda s: setattr(s, "dimension", 2.5), "Cartesian"),
        (lambda s: setattr(s.model, "attenuation_model", "kolsky"), "attenuation"),
        (lambda s: setattr(s.mesh.adapt, "f_adapt", None), "f_adapt"),
        (
            lambda s: setattr(s, "discretization", fs.Discretization(method="DPG")),
            "Galerkin",
        ),
    ],
)
def test_supported_physics_validation(tmp_path, change, match):
    job = job_at(tmp_path)
    change(job.simulation)
    with pytest.raises(ValueError, match=match):
        job.validate_outputs()


def _text(h5, name, value):
    h5.create_dataset(name, data=value, dtype=h5py.string_dtype())


def commit(job, *, sparse=False, first_step=0, task=None):
    root = job._result_path
    root.mkdir(parents=True, exist_ok=True)
    prefix = Path() if task is None else Path("modes") / f"mode_{task - 1:06d}"
    (root / prefix).mkdir(parents=True, exist_ok=True)
    meta_name = (prefix / "metadata.h5").as_posix()
    meta = root / meta_name
    with h5py.File(meta, "w") as h5:
        _text(h5, "survey/receiver_groups/_catalog/group_name", ["pressure"])
        _text(h5, "survey/receiver_groups/_catalog/dataset_path", ["/pressure"])
        d = h5.create_dataset("pressure", shape=(2, 2) if sparse else (1, 1, 2, 2))
        d.attrs["dims"] = np.array(
            ["trace"] if sparse else ["receiver", "component", "source"],
            dtype=h5py.string_dtype(),
        )
        d.attrs["component"] = np.array(["p"], dtype=h5py.string_dtype())
        d.attrs["units"] = np.array(["Pa"], dtype=h5py.string_dtype())
        if sparse:
            h5["survey/receiver_groups/pressure/traces/trace_id"] = [1, 2]
            h5["survey/receiver_groups/pressure/traces/source_id"] = [1, 1]
            h5["survey/receiver_groups/pressure/traces/receiver_id"] = [1, 2]
            h5["survey/receiver_groups/pressure/traces/component"] = [1, 1]
        else:
            d.attrs["receiver"] = [1, 2]
            d.attrs["source"] = np.array(["shot"], dtype=h5py.string_dtype())
    records = [
        dict(
            id="metadata",
            role="trace_metadata",
            representation="hdf5_metadata",
            schema="fs_trace_metadata_v1",
            path=meta_name,
            retention="durable",
            bytes=meta.stat().st_size,
        )
    ]
    schedule = job.trace_schedule or job.output_schedule
    for step in schedule.steps(job.time):
        if step < first_step:
            continue
        name = (prefix / f"step_{step:012d}.h5").as_posix()
        p = root / name
        with h5py.File(p, "w") as h5:
            _text(h5, "sample_axis", "time")
            _text(h5, "trace_metadata_file", meta_name)
            h5["task_id"] = task or 1
            h5["time_step"] = step
            h5["time"] = job.time.start + step * job.time.dt
            data = np.zeros((2, 2) if sparse else (1, 1, 2, 2))
            data[..., 0] = step * (task or 1)
            h5["pressure"] = data
        records.append(
            dict(
                id=f"traces/{step}",
                role="simulated_traces",
                representation="hdf5_shard",
                schema="fs-time-step-1",
                path=name,
                retention="durable",
                bytes=p.stat().st_size,
                dependencies=["metadata"],
            )
        )
    payload = dict(
        schema="fs-operation-result-1",
        operation=dict(name="transient", generation="test"),
        fingerprints=job._artifact_contract_fingerprints(),
        status=dict(state="success", code=0),
        artifacts=records,
    )
    path = job.result_manifest_file
    if task is not None:
        payload.pop("operation")
        payload["schema"] = "fs-task-result-2"
        payload["partition"] = dict(
            task=task, task_count=job.n_tasks, azimuthal_mode=task - 1
        )
        path = job.task_manifest_file(task)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


@pytest.mark.parametrize("sparse", [False, True])
def test_native_reader_and_lifecycle(tmp_path, sparse):
    job = job_at(
        tmp_path,
        outputs=fs.TraceOutput(),
        output_schedule=fs.TimeOutput(times=[0.2, 0.7]),
    )
    commit(job, sparse=sparse)
    data = job.traces.open()
    assert data.groups == ("pressure",)
    array = data.read("pressure", steps=[2, 7])
    assert array.dims == (
        ("time", "trace") if sparse else ("time", "source", "component", "receiver")
    )
    np.testing.assert_allclose(array.time, [0.2, 0.7])
    np.testing.assert_array_equal(array.step, [2, 7])
    assert np.all(array.values[0] == 2)
    assert "frequency" not in array.dims
    assert job.results_exist()
    result = _read_task_result(job.job_file, 0)
    assert result.name == "transient" and result.successful
    job.write_run_state()
    assert job.is_run_current()
    assert job.plan_tasks()["pending_indices"] == []
    assert job.plan_tasks(force=True)["pending_indices"] == [0]
    assert job.plan_tasks(force=True, apply=True)["removed_stale_outputs"]
    assert not job.results_exist()


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "schedule",
        "axis",
        "time",
        "metadata",
        "complex",
        "fingerprint",
        "failed",
    ],
)
def test_reader_rejects_invalid_committed_outputs(tmp_path, corruption):
    job = job_at(
        tmp_path, outputs=fs.TraceOutput(), output_schedule=fs.TimeOutput(stride=5)
    )
    payload = commit(job)
    path = job._result_path / "step_000000000005.h5"
    if corruption == "missing":
        path.unlink()
    elif corruption == "schedule":
        payload["artifacts"].pop(2)
    elif corruption == "fingerprint":
        payload["fingerprints"]["job"] = "0" * 64
    elif corruption == "failed":
        payload["status"] = {"state": "failed", "code": 1}
    else:
        with h5py.File(path, "a") as h5:
            if corruption == "axis":
                del h5["sample_axis"]
                _text(h5, "sample_axis", "freq")
            elif corruption == "time":
                h5["time"][...] = 0.51
            elif corruption == "metadata":
                del h5["trace_metadata_file"]
                _text(h5, "trace_metadata_file", "../other.h5")
            elif corruption == "complex":
                h5["pressure"][..., 1] = 1
        payload["artifacts"][2]["bytes"] = path.stat().st_size
    job.result_manifest_file.write_text(json.dumps(payload))
    if corruption == "complex":
        with pytest.raises(ValueError):
            job.traces.open().read("pressure")
    else:
        assert not job.results_exist()
        with pytest.raises((ValueError, OSError)):
            job.traces.open()


def test_restart_checks_interval_and_fingerprints_content(tmp_path):
    job = job_at(tmp_path)
    path = tmp_path / "checkpoint"
    path.mkdir()
    with h5py.File(path / "newmark_manifest_R00000.h5", "w") as h5:
        h5["layout"] = [2, 0, 2, 5]
        h5["scheme"] = [0.5, 0, 1, 0.1, 0.25, 0.5]
    resumed = job.with_restart(path, name="resume")
    resumed.save()
    assert resumed.restart_step() == 5
    first = resumed.fingerprint()
    (path / "state.bin").write_bytes(b"changed")
    assert resumed.fingerprint() != first
    resumed.time = fs.TimeIntegration(0, 2, 0.1)
    with pytest.raises(ValueError, match="Restart interval"):
        resumed.restart_step()


def test_export_matches_adopted_time_contract(tmp_path):
    from pathlib import Path

    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    root = Path(__file__).parent / "contracts/sauce-time-domain-1/contracts"
    registry = Registry()
    for file in root.rglob("*.json"):
        payload = json.loads(file.read_text())
        registry = registry.with_resource(
            payload["$id"], Resource.from_contents(payload)
        )
    schema = json.loads((root / "inputs/fs-time-domain-1/schema.json").read_text())
    validator = Draft202012Validator(schema, registry=registry)
    job = job_at(tmp_path, output_schedule=fs.TimeOutput(times=[0.2, 0.7]))
    validator.validate(job.to_fs()["Time"])
    job.output_schedule = fs.TimeOutput(interval=0.2)
    validator.validate(job.to_fs()["Time"])


def test_preflight_allows_coupled_displacement_and_native_velocity(tmp_path):
    job = job_at(tmp_path)
    job.simulation.physics = "coupled"
    job += fs.WavefieldOutput(
        name="u",
        field="displacement",
        grid=fs.CartesianGrid(x0=[0, 0], x1=[1, 1], n=[3, 3]),
    )
    report = job.validate()
    assert not any(
        e.code in {"discretization.method.incompatible", "field.unsupported"}
        for e in report.errors
    )
    job += fs.WavefieldOutput(
        name="v",
        field="velocity",
        grid=fs.CartesianGrid(x0=[0, 0], x1=[1, 1], n=[3, 3]),
    )
    job.validate_outputs()
    # Site preflight must agree with the job: Sauce samples native velocity.
    assert not any(e.code == "field.unsupported" for e in job.validate().errors)


def test_remote_restart_stages_complete_directory(tmp_path):
    job = job_at(tmp_path)
    cp = tmp_path / "checkpoint"
    cp.mkdir()
    with h5py.File(cp / "newmark_manifest_R00000.h5", "w") as h5:
        h5["layout"] = [2, 0, 2, 5]
        h5["scheme"] = [0.5, 0, 1, 0.1, 0.25, 0.5]
    (cp / "pml-sidecar.h5").write_bytes(b"sidecar")
    job = job.with_restart(cp, name="restart")
    job.save()
    pairs = job.remote_input_files("/remote/project")
    assert {p.name for p, q in pairs if p.parent == cp} == {
        "newmark_manifest_R00000.h5",
        "pml-sidecar.h5",
    }
    source = job._saved_layout()
    target = source.with_project("/remote/project")
    payload = job._payload_for_layout(job.to_fs(), source=source, target=target)
    assert payload["Time"]["restart"]["path"] == str(
        target.job_dir / "inputs/transient_restart"
    )


def test_relocated_result_reader(tmp_path):
    import shutil

    job = job_at(
        tmp_path, outputs=fs.TraceOutput(), output_schedule=fs.TimeOutput(stride=5)
    )
    commit(job)
    target = tmp_path / "relocated"
    shutil.copytree(
        job._result_path, job._saved_layout().with_project(target).result_dir
    )
    data = job.traces.open(project_path=target)
    assert data.groups == ("pressure",)
    assert data.read("pressure").sizes["time"] == 3


def test_pending_run_state_does_not_report_failure(tmp_path):
    job = job_at(tmp_path)
    job.write_run_state("submitted")
    state = job.run_state()
    assert state["status"] == "submitted"
    assert state["task_summary"]["failed"] == 0
    assert state["tasks"][0]["status"] == "not_run"


def test_slurm_transient_catalog_requires_only_operation_metadata(tmp_path):
    from types import SimpleNamespace

    from frequensolve.orchestrator.sites.hpc.site import SlurmSite

    job = job_at(tmp_path, output_schedule=fs.TimeOutput(stride=5))
    payload = commit(job)
    job.staged_artifact_fingerprints = lambda name: payload["fingerprints"]
    site = SlurmSite.__new__(SlurmSite)
    site._remote_result_dir = lambda job: "/remote/results"
    calls = []

    def get_files(remote, stage, paths, **kwargs):
        calls.extend(paths)
        for path in paths:
            if path == "_fs_run/operations/transient/result.json":
                target = stage / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(payload))

    site._transfer = SimpleNamespace(get_files=get_files)
    catalog = site.fetch_artifact_catalog(job)
    assert catalog.operations["transient"].successful
    assert calls == ["_fs_run/operations/transient/result.json"]


def test_displacement_receiver_requires_direction(tmp_path):
    job = job_at(tmp_path)
    job.simulation.physics = "elastic"
    job.simulation.acquisition.add_receiver_group(
        "u",
        fs.ReceiverNode(
            name="sensor",
            components=[fs.ReceiverComponent(name="u", field="displacement")],
        ),
        coords=[[0.5, 0.5]],
    )
    with pytest.raises(ValueError, match="measurement direction"):
        job.validate_outputs()


def test_failed_trajectory_reports_failure_reason(tmp_path):
    job = job_at(tmp_path)
    job.write_run_state(
        "failed",
        tasks=[
            {
                "task_id": 0,
                "status": "error",
                "error": "Newmark solve failed",
                "duration_seconds": 1.2,
            }
        ],
    )
    failed = job.failed_tasks()
    assert len(failed) == 1 and failed[0]["frequency"] is None
    assert failed[0]["reason"] == "Newmark solve failed"
    assert failed[0]["duration_seconds"] == 1.2


@pytest.mark.parametrize("format", ["vtu", "xdmf"])
def test_transient_paraview_roundtrip(tmp_path, format):
    job = job_at(tmp_path)
    job += fs.VtkOutput(name="pressure", fields=["pressure"], format=format)
    job.save()
    restored = fs.TransientJob.from_fs(job.to_fs())
    assert restored.outputs.vtk[0].to_fs() == job.outputs.vtk[0].to_fs()
    report = job.validate()
    assert not any(i.code == "outputs.vtk.frequency_count" for i in report.errors)


@pytest.mark.parametrize(
    "output,message",
    [
        (fs.VtkOutput(), "explicit fields"),
        (fs.VtkOutput(fields=["pressure"], parts=["im"]), "real"),
        (fs.VtkOutput(fields=["pressure"], post_process=True), "post_process"),
        (fs.VtkOutput(fields=["pressure"], execute_on="adapt"), "TimeOutput"),
        (fs.VtkOutput(fields=["pressure"], format="vtr"), "VTU and XDMF"),
    ],
)
def test_transient_paraview_preflight(tmp_path, output, message):
    job = job_at(tmp_path)
    with pytest.raises(ValueError, match=message):
        job += output
        job.validate_outputs()


@pytest.mark.parametrize("field", ["velocity", "displacement"])
def test_transient_helmholtz_outputs(tmp_path, field):
    job = job_at(tmp_path)
    job.simulation.physics = "elastic"
    job.simulation.extra["helmholtz_projection"] = {
        "enabled": True,
        "field": field,
        "parts": ["p", "s"],
    }
    job += fs.VtkOutput(fields=[field, f"p_{field}", f"s_{field}"])
    job.validate_outputs()
    assert not any(i.code == "field.unsupported" for i in job.validate().errors)
    job.outputs.vtk = [
        fs.VtkOutput(fields=["pressure", "displacement", "strain_zz", f"p_{field}"])
    ]
    job.validate_outputs()
    other = "displacement" if field == "velocity" else "velocity"
    job.outputs.vtk = [fs.VtkOutput(fields=[f"p_{other}"])]
    with pytest.raises(ValueError, match="must match"):
        job.validate_outputs()


@pytest.mark.parametrize(
    "physics",
    [
        "acoustic",
        "elastic",
        "coupled",
        "acoustic_axisym",
        "elastic_axisym",
        "coupled_axisym",
    ],
)
@pytest.mark.parametrize("output_kind", ["vtk", "wavefield", "receiver"])
def test_transient_velocity_projection_is_automatic(tmp_path, physics, output_kind):
    job = job_at(tmp_path)
    job.simulation.physics = physics
    job.simulation.axisymmetric = physics.endswith("_axisym")
    if output_kind == "vtk":
        job += fs.VtkOutput(fields=["velocity", "p_velocity", "s_velocity"])
    elif output_kind == "wavefield":
        job += fs.WavefieldOutput(
            name="p",
            field="p_velocity_x",
            grid=fs.CartesianGrid(x0=[0, 0], x1=[1, 1], n=[3, 3]),
        )
    else:
        job.simulation.acquisition.add_sources(
            kind="vector" if physics.startswith("elastic") else "scalar",
            coords=[[0.5, 0.5]],
        )
        job.simulation.acquisition.add_receiver_group(
            "s",
            fs.ReceiverNode(
                components=[
                    fs.ReceiverComponent(name="s", field="s_velocity", direction=[1, 0])
                ]
            ),
            coords=[[0.5, 0.5]],
        )
    job.validate_outputs()
    assert not any(
        i.code in {"field.unsupported", "job.config.invalid"}
        for i in job.validate().errors
    )
    job.save()
    # The default selects the physical quantity; Sauce enables only requested
    # parts and schedules solves. Native-only jobs keep projection disabled.
    assert job.simulation.to_fs()["helmholtz_projection"] == {"field": "velocity"}
    restored = fs.TransientJob.from_fs(job.to_fs())
    restored.validate_outputs()
    assert restored.simulation.extra["helmholtz_projection"] == {"field": "velocity"}


def test_transient_native_velocity_does_not_enable_projection(tmp_path):
    job = job_at(tmp_path, outputs=fs.VtkOutput(fields=["velocity"]))
    assert job.simulation.to_fs()["helmholtz_projection"] == {"field": "velocity"}


@pytest.mark.parametrize(
    "config,message",
    [
        ({"enabled": True, "field": "velocity"}, "parts must select"),
        ({"enabled": True, "part": "p"}, "preserve the native"),
        ({"enabled": True, "field": "acceleration", "parts": ["p"]}, "field must"),
        ({"parts": ["p", "p"]}, "parts must select"),
    ],
)
def test_transient_helmholtz_configuration(tmp_path, config, message):
    job = job_at(tmp_path)
    job.simulation.physics = "elastic"
    job.simulation.extra["helmholtz_projection"] = config
    with pytest.raises(ValueError, match=message):
        job.validate_outputs()


@pytest.mark.parametrize("physics", ["acoustic", "elastic", "coupled"])
def test_axisymmetric_transient_velocity_configuration(tmp_path, physics):
    job = job_at(tmp_path)
    job.simulation.physics = physics + "_axisym"
    job.simulation.axisymmetric = True
    job.simulation.extra["helmholtz_projection"] = {
        "field": "velocity",
        "parts": ["p", "s"],
        "s_projection": "complement",
    }
    job.outputs.vtk = [fs.VtkOutput(fields=["velocity", "p_velocity", "s_velocity"])]
    job.validate_outputs()
    assert not any(i.code == "field.unsupported" for i in job.validate().errors)
    job.simulation.extra["helmholtz_projection"]["s_projection"] = "solenoidal"
    with pytest.raises(ValueError, match="complement"):
        job.validate_outputs()


def test_cased_borehole_transient_example(tmp_path):
    import runpy

    example = (
        Path(__file__).parents[1]
        / "tutorials/01_modeling_basics/axisymmetric_transient_borehole.py"
    )
    job = runpy.run_path(str(example))["create_job"](tmp_path / "borehole")
    assert not job.validate().errors
    model = job.simulation.model.to_fs()
    assert job.simulation.physics == "coupled_axisym"
    assert len(model["boreholes"][0]["layers"]) == 2
    assert {domain.get("physics") for domain in model["subdomains"]} >= {
        "acoustic",
        "elastic",
    }
    job.save()
    restored = fs.TransientJob.from_fs(json.loads(job.job_file.read_text()))
    np.testing.assert_array_equal(
        restored.output_schedule.steps(restored.time),
        job.output_schedule.steps(job.time),
    )


def test_acoustic_native_velocity_passes_preflight(tmp_path):
    job = job_at(tmp_path)
    job += fs.WavefieldOutput(
        name="v",
        field="velocity",
        grid=fs.CartesianGrid(x0=[0, 0], x1=[1, 1], n=[3, 3]),
    )
    job.validate_outputs()
    assert not any(e.code == "field.unsupported" for e in job.validate().errors)


def test_unsupported_transient_field_names_the_request(tmp_path):
    job = job_at(tmp_path)
    job.simulation.physics = "elastic"
    job.simulation.extra["helmholtz_projection"] = {
        "enabled": True,
        "field": "displacement",
        "parts": ["p", "s"],
    }
    job.outputs.vtk = [fs.VtkOutput(fields=["pressure"])]
    messages = [
        e.message for e in job.validate().errors if e.code == "field.unsupported"
    ]
    assert messages and all("'pressure'" in message for message in messages)


def test_elastic_moment_sources_accept_step_functions(tmp_path):
    job = job_at(tmp_path)
    job.simulation.physics = "elastic_axisym"
    job.simulation.axisymmetric = True
    job.simulation.acquisition.add_sources(kind="monopole", coords=[[0.5, 0.5]])
    job.source = fs.TimeFunction.step(delay=0.5)
    job.validate_outputs()
    job.simulation.physics = "acoustic_axisym"
    with pytest.raises(ValueError, match="smooth time functions"):
        job.validate_outputs()


def test_receiver_names_do_not_trip_harmonic_field_checks(tmp_path):
    job = job_at(tmp_path)
    job.simulation.acquisition.add_receiver_group(
        "array",
        fs.ReceiverNode(
            name="low_frequency_array",
            components=[
                fs.ReceiverComponent(name="acceleration_proxy", field="pressure")
            ],
        ),
        coords=[[0.5, 0.5]],
    )
    job.validate_outputs()
    job.simulation.acquisition.add_receiver_group(
        "accelerometer",
        fs.ReceiverNode(
            name="sensor",
            components=[fs.ReceiverComponent(name="a", field="acceleration")],
        ),
        coords=[[0.5, 0.5]],
    )
    with pytest.raises(ValueError, match="primary or constitutive"):
        job.validate_outputs()


def test_restarted_results_stay_readable_after_checkpoint_removal(tmp_path):
    import shutil

    job = job_at(tmp_path, outputs=fs.TraceOutput())
    path = tmp_path / "checkpoint"
    path.mkdir()
    with h5py.File(path / "newmark_manifest_R00000.h5", "w") as h5:
        h5["layout"] = [2, 0, 2, 5]
        h5["scheme"] = [0.5, 0, 1, 0.1, 0.25, 0.5]
    resumed = job.with_restart(path, name="resume")
    resumed.save()
    commit(resumed, first_step=5)
    resumed.write_run_state()
    assert resumed.is_run_current()

    shutil.rmtree(path)
    for current in (resumed, fs.BaseJob.load(resumed.job_file)):
        assert current.is_run_current()
        np.testing.assert_array_equal(current.traces.open().steps, range(5, 11))
    # Saving again keeps the recorded restart identity.
    resumed.save()
    assert resumed.is_run_current()


@pytest.mark.parametrize("value", [0, False, "off", "0"])
def test_slurm_transient_accepts_every_spelling_of_zero_tolerance(
    monkeypatch, tmp_path, value
):
    from types import SimpleNamespace

    from tests.test_site_gpu_partitions import _site

    class Submitted(Exception):
        pass

    def stop(**kwargs):
        raise Submitted

    site = _site(monkeypatch, tmp_path)
    monkeypatch.setattr(site, "check_solver_compatibility", stop)
    job = SimpleNamespace(workflow="transient")
    with pytest.raises(Submitted):
        site.run(job, tolerate_failures=value)
    with pytest.raises(ValueError, match="tolerate_failures=0"):
        site.run(job, tolerate_failures=1)


@pytest.mark.parametrize(
    "output",
    [
        fs.VtkOutput(fields=["pressure"], properties=["Vp"]),
        fs.VtkOutput(properties=["Vp", "Rho"], execute_on="initial"),
        fs.VtkOutput(items=[{"kind": "info", "info": "order"}]),
    ],
)
def test_transient_material_output_roundtrip(tmp_path, output):
    job = job_at(tmp_path)
    job += output
    job.validate_outputs()
    job.save()
    restored = fs.TransientJob.from_fs(job.to_fs())
    assert restored.outputs.vtk[0].to_fs() == output.to_fs()


def test_displacement_projection_rejects_native_velocity(tmp_path):
    job = job_at(tmp_path)
    job.simulation.physics = "elastic"
    job.simulation.extra["helmholtz_projection"] = {
        "enabled": True,
        "field": "displacement",
        "parts": ["p", "s"],
    }
    job.outputs.vtk = [fs.VtkOutput(fields=["velocity"])]
    with pytest.raises(ValueError, match="require velocity"):
        job.validate_outputs()


def test_independent_trace_schedule(tmp_path):
    job = job_at(
        tmp_path,
        outputs=fs.TraceOutput(),
        output_schedule=fs.TimeOutput(interval=0.5),
        trace_schedule=fs.TimeOutput(interval=0.2, write_initial=False),
    )
    restored = fs.BaseJob.load(job.job_file)
    assert restored.to_fs() == job.to_fs()
    np.testing.assert_array_equal(restored.output_schedule.steps(job.time), [0, 5, 10])
    commit(job)
    traces = restored.traces.open()
    np.testing.assert_array_equal(traces.steps, [2, 4, 6, 8, 10])
    np.testing.assert_allclose(traces.times, [0.2, 0.4, 0.6, 0.8, 1])
    before = job.task_fingerprint(1)
    job.trace_schedule = fs.TimeOutput(stride=1)
    assert job.task_fingerprint(1) != before
    with pytest.raises(ValueError, match="saved job inputs"):
        job.traces.open()


def test_trace_schedule_requires_aligned_time(tmp_path):
    with pytest.raises(ValueError, match="align"):
        job_at(tmp_path, trace_schedule=fs.TimeOutput(interval=0.15))


@pytest.mark.parametrize(
    "selection",
    [
        {"dt": 0.1},
        {"dt": 0.1, "f_0": 0.8},
        {"f_max": 4, "period_error": 0.001},
        {"f_max": 4, "window_error": fs.Q_(3, "degree")},
    ],
)
def test_gauss_job_save_load_and_solver_contract(tmp_path, selection):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    job = job_at(tmp_path)
    job.time = fs.TimeIntegration(0, 1, scheme="gauss", order=4, **selection)
    job.save()
    restored = fs.BaseJob.load(job.job_file)
    assert restored.time == job.time
    assert restored.to_fs() == job.to_fs()
    payload = restored.to_fs()["Time"]
    assert payload["method"] == "gauss4"
    assert "newmark" not in payload and "window_error" not in payload
    path = (
        Path(__file__).parent
        / "contracts/sauce-time-domain-1/contracts/inputs/fs-time-domain-1/schema.json"
    )
    registry = Registry()
    for file in path.parents[2].rglob("*.json"):
        schema = json.loads(file.read_text())
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    Draft202012Validator(json.loads(path.read_text()), registry=registry).validate(
        payload
    )
    fingerprint = restored.task_fingerprint(1)
    restored.time = fs.TimeIntegration(0, 1, restored.time.dt, order=2)
    assert restored.task_fingerprint(1) != fingerprint


def test_gauss_restart_checks_method_and_interval(tmp_path):
    job = job_at(tmp_path)
    job.time = fs.TimeIntegration(0, 1, 0.1, scheme="gauss")
    checkpoint = tmp_path / "gauss_checkpoint"
    checkpoint.mkdir()
    with h5py.File(checkpoint / "newmark_manifest_R00000.h5", "w") as h5:
        h5["layout"] = [2, 0, 3, 5]
        h5["scheme"] = [0.5, 0, 1, 0.1, 0, 0]
    resumed = job.with_restart(checkpoint, name="gauss-resume")
    resumed.save()
    assert fs.BaseJob.load(resumed.job_file).restart_step() == 5
    resumed.time = fs.TimeIntegration(0, 1, 0.1, scheme="gauss", f_0=0.8)
    with pytest.raises(ValueError, match="theta_0"):
        resumed.restart_step()
    with h5py.File(checkpoint / "newmark_manifest_R00000.h5", "a") as h5:
        h5["theta_0"] = 2 * np.pi * 0.8 * 0.1
    assert resumed.restart_step() == 5
    resumed.time = fs.TimeIntegration(0, 2, 0.1, scheme="gauss", f_0=0.8)
    with pytest.raises(ValueError, match="Restart interval"):
        resumed.restart_step()
    resumed.time = fs.TimeIntegration(0, 1, 0.1, order=2)
    with pytest.raises(ValueError, match="restart manifest"):
        resumed.restart_step()


@pytest.mark.parametrize("f_0", [0, 2])
def test_gauss_accepts_pml_and_gsls_but_custom_newmark_remains_restricted(
    tmp_path, f_0
):
    job = job_at(tmp_path)
    job.simulation += fs.BoundaryCondition(
        boundaries=["x_min"], conditions="pml:impedance"
    )
    job.time = fs.TimeIntegration(0, 1, 0.1, scheme="gauss", f_0=f_0)
    job.validate_outputs()
    job.simulation.model.attenuation_model = "gsls"
    job.simulation.model._attenuation_extra = {"frequency_min": 1, "frequency_max": 10}
    job.validate_outputs()
    job.time = fs.TimeIntegration(0, 1, 0.1, fs.Newmark(beta=0.3))
    with pytest.raises(ValueError, match="gsls"):
        job.validate_outputs()
    job.simulation.model.attenuation_model = "none"
    job.simulation.model._attenuation_extra = {}
    with pytest.raises(ValueError, match="PML"):
        job.validate_outputs()


def test_transient_configures_unset_defaults_and_preserves_explicit_choices(tmp_path):
    project = fs.Project(name="defaults", path=tmp_path / "defaults")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=2)
    sim.discretization = fs.Discretization(geometry_tolerance=1e-5)
    job = fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(0, 1, 0.01, f_max=20),
        source=fs.TimeFunction.ricker(10, delay=0.3),
    )
    assert job.simulation is sim
    assert sim.discretization.to_fs() == {
        "geometry_tolerance": 1e-5,
        "method": "Galerkin",
    }
    assert sim.model.attenuation_model == "none"
    assert sim.mesh.adapt.f_adapt == 20
    job.save()
    assert fs.BaseJob.load(job.job_file).to_fs() == job.to_fs()
    sim.mesh.set_adapt(elems_per_wave=3, order=4, f_adapt=fs.Q_(0.03, "kHz"))
    sim.solver.tolerance = 2e-4
    fs.TransientJob("second", sim, time=job.time, source=job.source)
    assert (sim.mesh.adapt.f_adapt, sim.mesh.adapt.order, sim.solver.tolerance) == (
        30,
        4,
        2e-4,
    )
    sim.discretization = fs.Discretization(method="DPG")
    with pytest.raises(ValueError, match="Galerkin"):
        fs.TransientJob("invalid", sim, time=job.time, source=job.source)


def test_transient_defaults_do_not_guess_source_bandwidth(tmp_path):
    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(name="wave", physics="acoustic", dimension=2)
    with pytest.raises(ValueError, match="bandwidth"):
        fs.TransientJob(
            "pulse",
            sim,
            time=fs.TimeIntegration(0, 1, 0.01),
            source=fs.TimeFunction.ricker(10, delay=0.3),
        )


def test_radau_job_persistence_memories_and_restart_identity(tmp_path):
    job = job_at(tmp_path)
    job.time = fs.TimeIntegration(0, 1, 0.01, fs.Radau())
    job.simulation += fs.BoundaryCondition(
        boundaries=["x_min"], conditions="pml:impedance"
    )
    job.simulation.model.attenuation_model = "gsls"
    job.simulation.model._attenuation_extra = {"frequency_min": 1, "frequency_max": 10}
    job.save()
    restored = fs.BaseJob.load(job.job_file)
    assert restored.time == job.time
    assert restored.to_fs()["Time"]["method"] == "radau3"
    checkpoint = tmp_path / "radau-checkpoint"
    checkpoint.mkdir()
    with h5py.File(checkpoint / "newmark_manifest_R00000.h5", "w") as h5:
        h5["layout"] = [2, 0, 4, 50]
        h5["scheme"] = [0.5, 0, 1, 0.01, 0, 0]
    resumed = job.with_restart(checkpoint, name="radau-resume")
    resumed.save()
    assert fs.BaseJob.load(resumed.job_file).restart_step() == 50
    resumed.time = fs.TimeIntegration(0, 1, 0.01, fs.Gauss())
    with pytest.raises(ValueError, match="restart manifest"):
        resumed.restart_step()


@pytest.mark.parametrize("order", [3, 4])
@pytest.mark.parametrize(
    "selection", [{"dt": 0.01}, {"f_max": 20, "window_error": 0.01}]
)
def test_order_based_jobs_default_to_radau_and_persist_resolved_order(
    tmp_path, order, selection
):
    job = job_at(tmp_path)
    job.time = fs.TimeIntegration(0, 1, order=order, **selection)
    job.save()
    restored = fs.BaseJob.load(job.job_file)
    assert restored.time == job.time
    assert restored.time.order == 3 and restored.time.method == fs.Radau()
    assert restored.to_fs()["Time"]["method"] == "radau3"
    assert restored.time.replace(order=4).method == fs.Radau()
