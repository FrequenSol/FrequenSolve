"""Real angular reconstruction and public transient modal authoring."""

import numpy as np
import pytest
import xarray as xr

import frequensolve as fs


def test_angular_pressure_interference():
    modes = {0: np.array([1.0, 2.0]), 1: np.array([3.0, -4.0]), 2: np.array([5.0, 6.0])}
    result = fs.reconstruct_azimuthal_pressure(modes, [0.0, np.pi / 2, np.pi])
    np.testing.assert_allclose(result, [[17.0, 6.0], [-9.0, -10.0], [5.0, 22.0]])
    np.testing.assert_allclose(fs.reconstruct_azimuthal_pressure(modes), result[0])


def test_reconstruction_preserves_samples_and_rejects_misalignment():
    first = xr.DataArray(
        [1.0, 2.0], dims="time", coords={"time": [0.0, 0.1]}, attrs={"units": "Pa"}
    )
    result = fs.reconstruct_azimuthal_pressure({0: first, 1: 2 * first}, [0.0, np.pi])
    assert result.dims == ("azimuth", "time")
    assert result.attrs["units"] == "Pa"
    np.testing.assert_allclose(result.values, [[5.0, 10.0], [-3.0, -6.0]])
    with pytest.raises(ValueError, match="coordinates"):
        fs.reconstruct_azimuthal_pressure(
            {0: first, 1: first.assign_coords(time=[0.0, 0.2])}
        )


@pytest.mark.parametrize(
    "modes",
    [
        {1: [1.0]},
        {0: [1.0], 2: [2.0]},
        {True: [1.0]},
        {0: [1j]},
        {0: [np.nan]},
        {0: [1.0], 1: [1.0, 2.0]},
    ],
)
def test_invalid_modes(modes):
    with pytest.raises(ValueError):
        fs.reconstruct_azimuthal_pressure(modes)


@pytest.mark.parametrize("angle", [np.nan, 1j, [], [[0.0]], "pi"])
def test_invalid_angles(angle):
    with pytest.raises(ValueError):
        fs.reconstruct_azimuthal_pressure({0: [1.0]}, angle)


def modal_simulation(tmp_path, mode=2):
    project = fs.Project(name="p", path=tmp_path / "p")
    sim = project.new_simulation(
        name="modal",
        physics="acoustic",
        dimension=2,
        axisymmetric=True,
        azimuthal_mode=mode,
    )
    sim.discretization = fs.Discretization(method="Galerkin")
    sim.model.attenuation_model = "none"
    sim.mesh.set_adapt(elems_per_wave=2, f_adapt=20)
    return sim


def test_modal_round_trip_and_copy(tmp_path):
    sim = modal_simulation(tmp_path)
    assert sim.to_fs()["azimuthal_mode"] == 2
    assert type(sim).from_fs(sim.to_fs()).azimuthal_mode == 2
    assert sim.copy("copy").azimuthal_mode == 2
    job = fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(0, 1, 0.01),
        source=fs.TimeFunction.ricker(10, delay=0.3),
    )
    job.validate_outputs()
    with pytest.raises(ValueError, match="TransientJob"):
        fs.FrequencyDomainJob("frequency", sim, f_list=[10]).validate_outputs()


@pytest.mark.parametrize("mode", [-1, True, 1.5, "1"])
def test_invalid_mode_setting(tmp_path, mode):
    with pytest.raises(ValueError, match="nonnegative integer"):
        modal_simulation(tmp_path, mode)


def test_mutated_geometry_is_checked_on_export(tmp_path):
    sim = modal_simulation(tmp_path)
    sim.axisymmetric = False
    with pytest.raises(ValueError, match="axisymmetric 2D acoustic"):
        sim.to_fs()


def test_positive_modes_reject_incomplete_vector_output(tmp_path):
    sim = modal_simulation(tmp_path)
    job = fs.TransientJob(
        "pulse",
        sim,
        time=fs.TimeIntegration(0, 1, 0.01),
        source=fs.TimeFunction.ricker(10, delay=0.3),
    )
    job += fs.VtkOutput(name="velocity", fields=["velocity"])
    with pytest.raises(ValueError, match="vector velocity"):
        job.validate_outputs()


def modal_job(tmp_path, n_modes=3, **kwargs):
    sim = modal_simulation(tmp_path, mode=None)
    return fs.TransientJob(
        "sweep",
        sim,
        n_modes=n_modes,
        time=fs.TimeIntegration(0, 1, 0.1),
        source=fs.TimeFunction.ricker(10, delay=0.3),
        **kwargs,
    )


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "3", 1000001])
def test_invalid_mode_count(tmp_path, count):
    with pytest.raises(ValueError, match="n_modes"):
        modal_job(tmp_path, count)


def test_modal_tasks_roundtrip_and_partition_fingerprints(tmp_path):
    job = modal_job(tmp_path)
    job.save()
    assert job.n_tasks == 3
    assert job.to_fs()["Time"]["n_modes"] == 3
    assert "azimuthal_mode" not in job.simulation.to_fs()
    assert fs.BaseJob.load(job.job_file).to_fs() == job.to_fs()
    assert len({job.task_fingerprint(i) for i in (1, 2, 3)}) == 3
    assert job.plan_tasks()["pending_indices"] == [0, 1, 2]
    job.simulation.azimuthal_mode = 1
    with pytest.raises(ValueError, match="instead of"):
        job.validate_outputs()


def test_mode_loading_reconstruction_and_partial_retry(tmp_path):
    from frequensolve.orchestrator.sites.local.site import _read_task_result
    from tests.test_transient import commit

    job = modal_job(tmp_path, outputs=fs.TraceOutput())
    job.save()
    commit(job, task=1)
    commit(job, task=3)
    assert job.current_tasks() == [1, 3]
    assert job.plan_tasks()["pending_indices"] == [1]
    assert not job.results_exist()
    with pytest.raises(OSError):
        job.traces.open()
    assert _read_task_result(job.job_file, 2).partition.azimuthal_mode == 2
    commit(job, task=2)
    assert job.results_exist()
    data = job.traces.open()
    modes = data.read_modes("pressure", steps=[2, 7])
    assert modes.dims[0] == "mode"
    np.testing.assert_array_equal(modes.mode, [0, 1, 2])
    p = data.read("pressure", steps=[2, 7], delta_theta=[0, np.pi])
    np.testing.assert_allclose(p.values[:, :, 0, 0, 0], [[22, 77], [6, 21]])
    np.testing.assert_allclose(
        job.traces.open(mode=1).read("pressure", steps=[2]).values, 4
    )
    job.write_run_state()
    assert job.is_run_current()
    assert job.plan_tasks()["pending_indices"] == []
    path = job.task_manifest_file(2)
    import json

    payload = json.loads(path.read_text())
    payload["partition"]["azimuthal_mode"] = 0
    path.write_text(json.dumps(payload))
    assert job.current_tasks() == [1, 3]
    with pytest.raises(ValueError, match="[Aa]ngular partition"):
        _read_task_result(job.job_file, 1)
    assert job.plan_tasks(apply=True)["pending_indices"] == [1]
    assert not path.exists()
    assert job.task_manifest_file(1).exists() and job.task_manifest_file(3).exists()


def test_modal_restart_selects_and_checks_each_checkpoint(tmp_path):
    import h5py

    job = modal_job(tmp_path)
    job.restart = str(tmp_path / "restart")
    for task in range(1, 4):
        path = tmp_path / "restart" / "modes" / f"mode_{task-1:06d}"
        path.mkdir(parents=True)
        (path / "azimuthal_mode_R00000.txt").write_text(str(task - 1))
        with h5py.File(path / "newmark_manifest_R00000.h5", "w") as h:
            h["layout"] = [2, 0, 2, task]
            h["scheme"] = [task * 0.1, 0, 1, 0.1, 0.25, 0.5]
    assert [job.restart_step(i) for i in (1, 2, 3)] == [1, 2, 3]
    job._input_fingerprint_payload()
    (path / "azimuthal_mode_R00000.txt").write_text("0")
    with pytest.raises(ValueError, match="azimuthal_mode"):
        job._input_fingerprint_payload()


@pytest.mark.parametrize("backend", ["slurm", "aws"])
def test_modal_remote_catalog_fetches_tasks(tmp_path, backend):
    import json
    from types import SimpleNamespace

    from frequensolve.orchestrator.sites.hpc.site import SlurmSite
    from tests.test_transient import commit

    job = modal_job(tmp_path)
    job.save()
    payloads = {
        f"_fs_run/tasks/task_{task:06d}/result.json": commit(job, task=task)
        for task in range(1, 4)
    }
    fingerprints = next(iter(payloads.values()))["fingerprints"]
    job.staged_artifact_fingerprints = lambda name: fingerprints
    job.staged_task_fingerprints = lambda name: fingerprints
    calls = []

    def transfer(remote, stage, paths, **kwargs):
        calls.extend(paths)
        for path in paths:
            if path in payloads:
                target = stage / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(payloads[path]))

    if backend == "slurm":
        site = SlurmSite.__new__(SlurmSite)
        site._remote_result_dir = lambda job: "/remote/results"
        site._transfer = SimpleNamespace(get_files=transfer)
    else:
        pytest.importorskip("boto3")
        from frequensolve.orchestrator.sites.aws.aws import AWSSite

        site = AWSSite.__new__(AWSSite)
        site._local_result_path = lambda job, project_path: job._result_path
        site._s3_result_prefix = lambda job: "results/"
        site._download_s3_files = transfer
    catalog = site.fetch_artifact_catalog(job, operations=("transient",))
    assert set(catalog.results) == {1, 2, 3}
    assert all(r.partition.azimuthal_mode == i - 1 for i, r in catalog.results.items())
    assert not any("operations/transient" in path for path in calls)
    assert set(payloads).issubset(calls)
    payloads[next(iter(payloads))]["partition"]["azimuthal_mode"] = 2
    with pytest.raises((RuntimeError, ValueError), match="angular|mode"):
        site.fetch_artifact_catalog(job)


def test_angular_task_partition_validation():
    from frequensolve.simulation.artifact_contract import (
        ArtifactContractError,
        TaskPartition,
    )

    valid = dict(task=2, task_count=3, azimuthal_mode=1)
    assert TaskPartition.from_fs(valid).to_fs() == valid
    for bad in [
        dict(task=2, azimuthal_mode=1),
        dict(valid, azimuthal_mode=0),
        dict(valid, azimuthal_mode=True),
        dict(valid, frequency=dict(real=0, imag=0)),
    ]:
        with pytest.raises(ArtifactContractError):
            TaskPartition.from_fs(bad)
