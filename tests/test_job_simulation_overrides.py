"""Job-specific simulation settings survive persistence, relocation and execution identity."""

import copy
import json
from pathlib import Path

import pytest

import frequensolve as fs
from frequensolve.simulation.overrides import (
    expanded_simulation_hash,
    merge_simulation_overrides,
)


def simulation_at(tmp_path):
    project = fs.Project(name="p", path=tmp_path / "p")
    return project.new_simulation(name="wave", physics="acoustic", dimension=2)


def transient(sim, name="pulse", f_max=8, **kwargs):
    return fs.TransientJob(
        name,
        sim,
        time=fs.TimeIntegration(0, 1, 0.01, f_max=f_max),
        source=fs.TimeFunction.ricker(3.2, delay=0.3),
        **kwargs,
    )


def test_transient_jobs_share_definition_but_resolve_own_defaults(tmp_path):
    sim = simulation_at(tmp_path)
    original = copy.deepcopy(sim.to_fs())
    a, b = transient(sim, "a", 8), transient(sim, "b", 12)
    a.save()
    source_bytes = sim._file.read_bytes()
    b.save()
    assert sim._file.read_bytes() == source_bytes
    assert sim.to_fs() == original
    assert a.simulation is b.simulation is sim
    assert a.effective_simulation.mesh.adapt.f_adapt == 8
    assert b.effective_simulation.mesh.adapt.f_adapt == 12
    for job, expected in ((a, 8), (b, 12)):
        payload = json.loads(job.job_file.read_text())
        reference = payload["simulation"]
        assert (job.job_file.parent / reference["relative_path"]).resolve() == sim._file
        assert reference["overrides"]["Mesh"]["adapt"]["f_adapt"] == expected
        restored = fs.BaseJob.load(job.job_file)
        assert restored.simulation.mesh.adapt.f_adapt is None
        assert restored.effective_simulation.mesh.adapt.f_adapt == expected
        assert restored.to_fs() == job.to_fs()
    a.time = fs.TimeIntegration(0, 1, 0.01, f_max=16)
    assert a.effective_simulation.mesh.adapt.f_adapt == 16
    assert b.effective_simulation.mesh.adapt.f_adapt == 12


def test_transient_mesh_and_job_override_precedence(tmp_path):
    sim = simulation_at(tmp_path)
    sim.mesh.set_adapt(order=3, elems_per_wave=1.5, f_adapt=6.4)
    job = transient(sim, simulation_overrides={"Mesh": {"adapt": {"f_adapt": 5.6}}})
    assert sim.mesh.adapt.f_adapt == 6.4
    assert job.effective_simulation.mesh.adapt.f_adapt == 5.6
    assert job.effective_simulation.mesh.adapt.order == 3
    assert job.effective_simulation.mesh.adapt.elems_per_wave == 1.5
    assert transient(sim, "base").effective_simulation.mesh.adapt.f_adapt == 6.4
    job.override_simulation(Mesh={"adapt": {"order": 4}})
    assert job.effective_simulation.mesh.adapt.order == 4
    with pytest.raises(ValueError, match="Galerkin"):
        transient(
            sim, "bad", simulation_overrides={"Discretization": {"method": "DPG"}}
        )


def test_generic_overrides_roundtrip_and_invalidate_fingerprints(tmp_path):
    sim = simulation_at(tmp_path)
    job = fs.FrequencyDomainJob(
        "f", sim, [8], simulation_overrides={"Solver": {"tolerance": 0.001}}
    )
    job.save()
    source = sim._file.read_bytes()
    whole, task = job.fingerprint(), job.task_fingerprint(1)
    loaded = fs.BaseJob.load(job.job_file)
    assert loaded.simulation_overrides == job.simulation_overrides
    assert loaded.effective_simulation.solver.tolerance == 0.001
    assert loaded.to_fs() == job.to_fs()
    job.override_simulation(Solver={"tolerance": 0.002})
    job.save()
    assert job.fingerprint() != whole and job.task_fingerprint(1) != task
    assert sim._file.read_bytes() == source
    legacy = fs.FrequencyDomainJob("legacy", sim, [8])
    legacy.save()
    assert isinstance(json.loads(legacy.job_file.read_text())["simulation"], str)


def test_recursive_merge_replaces_arrays_and_null_without_mutation():
    base = {
        "Mesh": {"adapt": {"order": 3, "f_adapt": 8}},
        "BCs": [1, 2],
        "optional": {"x": 1},
    }
    original = copy.deepcopy(base)
    result = merge_simulation_overrides(
        base, {"Mesh": {"adapt": {"f_adapt": 5.6}}, "BCs": [3], "optional": None}
    )
    assert result == {
        "Mesh": {"adapt": {"order": 3, "f_adapt": 5.6}},
        "BCs": [3],
        "optional": None,
    }
    assert base == original
    with pytest.raises(TypeError, match="mapping"):
        merge_simulation_overrides({}, [])


def test_acquisition_overrides_describe_receiver_and_wavefield_outputs(tmp_path):
    sim = simulation_at(tmp_path)
    sim.acquisition.add_sources(kind="pressure_monopole", coords=[[0, 0]])
    sim.acquisition.add_receiver_group(
        "original",
        fs.ReceiverNode(components=[fs.ReceiverComponent(name="p", field="pressure")]),
        [[0, 0]],
    )
    acquisition = fs.Acquisition()
    acquisition.add_sources(kind="pressure_monopole", coords=[[0, 0], [1, 0]])
    acquisition.add_receiver_group(
        "overridden",
        fs.ReceiverNode(components=[fs.ReceiverComponent(name="vz", field="vz")]),
        [[0, 0]],
    )
    grid = fs.CartesianGrid(dims=["x", "z"], x0=[0, 0], x1=[1, 1], n=[2, 2])
    job = fs.FrequencyDomainJob(
        "f",
        sim,
        [8],
        outputs=[
            fs.TraceOutput(),
            fs.WavefieldOutput(name="field", grid=grid, fields=["pressure"]),
            fs.WavefieldOutput(
                name="selected", grid=grid, fields=["pressure"], sources=[2]
            ),
        ],
        simulation_overrides={"Acquisition": acquisition.to_fs()},
    )
    job.save()
    for current in (job, fs.BaseJob.load(job.job_file)):
        assert current.trace_outputs.groups == ["overridden"]
        assert current.trace_outputs.components == ["overridden:vz"]
        assert current.trace_outputs.sources == ["1", "2"]
        assert current.wavefield_outputs["field"]["sources"] == ["1", "2"]
        assert current.wavefield_outputs["selected"]["sources"] == ["2"]
        assert current.wavefield_trace_outputs.sources == ["1", "2"]
        assert [g.name for g in current.simulation.acquisition.receiver_groups] == [
            "original"
        ]


def test_remote_staging_preserves_overrides_and_maps_their_paths(tmp_path):
    sim = simulation_at(tmp_path)
    data = sim.project_path / "input.bin"
    data.write_bytes(b"input")
    job = transient(
        sim,
        simulation_overrides={
            "Mesh": {"adapt": {"f_adapt": 5.6}},
            "custom": {"file": str(data)},
        },
    )
    local = job.save()
    original = local.read_bytes(), sim._file.read_bytes()
    remote_root = Path("/remote/project")
    staged_job, remote_job = job.save_for_remote("test", remote_root)
    staged_sim, remote_sim = job.save_simulation_for_remote("test", remote_root)
    payload = json.loads(staged_job.read_text())
    assert payload["simulation"]["absolute_path"] == str(remote_sim)
    assert payload["simulation"]["overrides"]["custom"]["file"] == str(
        remote_root / "input.bin"
    )
    assert payload["simulation"]["overrides"]["Mesh"]["adapt"]["f_adapt"] == 5.6
    assert remote_job == remote_root / job.job_file.relative_to(sim.project_path)
    assert (data, remote_root / "input.bin") in job.remote_input_files(remote_root)
    assert original == (local.read_bytes(), sim._file.read_bytes())
    provenance = job.staged_artifact_fingerprints("test")
    assert provenance["simulation"] == expanded_simulation_hash(
        payload, json.loads(staged_sim.read_text()), "unused"
    )
    assert provenance in job.downloaded_task_fingerprints()


def test_project_copy_keeps_job_override_and_relocates_reference(tmp_path):
    sim = simulation_at(tmp_path)
    job = transient(sim, simulation_overrides={"Mesh": {"adapt": {"f_adapt": 5.6}}})
    job.save()
    sim._project.save()
    copied = fs.Project.copy(sim.project_path, tmp_path / "copied")
    path = copied.path / job.job_file.relative_to(sim.project_path)
    restored = fs.BaseJob.load(path)
    assert restored.simulation.project_path == copied.path
    assert restored.simulation.mesh.adapt.f_adapt is None
    assert restored.effective_simulation.mesh.adapt.f_adapt == 5.6


def test_default_attenuation_view_does_not_serialize_material_arrays(
    tmp_path, monkeypatch
):
    sim = simulation_at(tmp_path)

    def unexpected_export(*args, **kwargs):
        raise AssertionError("A job default must not export the entire model")

    monkeypatch.setattr(sim.model, "to_fs", unexpected_export)
    job = transient(sim)
    assert job.effective_simulation.model.attenuation_model == "none"
    assert sim.model.attenuation_model is None


def test_solver_overrides_respect_compatible_skip_policy(tmp_path):
    sim = simulation_at(tmp_path)
    job = fs.FrequencyDomainJob("f", sim, [8])
    job.override_simulation(Solver={"tolerance": 0.001})
    job.save()
    compatible = job.task_policy_fingerprint(1, "compatible")
    strict = job.task_fingerprint(1)
    job.override_simulation(Solver={"tolerance": 0.002})
    job.save()
    assert compatible == job.task_policy_fingerprint(1, "compatible")
    assert strict != job.task_fingerprint(1)
    job.override_simulation(Mesh={"adapt": {"f_adapt": 5.6}})
    assert compatible != job.task_policy_fingerprint(1, "compatible")


def test_invalid_override_identity_is_reported_without_replacing_definition(tmp_path):
    sim = simulation_at(tmp_path)
    job = fs.FrequencyDomainJob("f", sim, [8])
    job.override_simulation(name="other")
    report = job.validate()
    assert any(issue.code == "job.config.invalid" for issue in report.errors)
    with pytest.raises(ValueError, match="identity"):
        job.save()
    assert sim.name == "wave"


def test_expanded_fingerprint_cache_tracks_source_and_override_changes(
    tmp_path, monkeypatch
):
    import frequensolve.simulation.overrides as module

    sim = simulation_at(tmp_path)
    job = transient(sim)
    job.save()
    calls = []
    original = module.expanded_simulation_hash

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "expanded_simulation_hash", counted)
    first = job._artifact_contract_fingerprints()
    assert first == job._artifact_contract_fingerprints()
    assert len(calls) == 1
    job.override_simulation(Mesh={"adapt": {"f_adapt": 5.6}})
    job.save()
    assert job._artifact_contract_fingerprints()["simulation"] != first["simulation"]
    assert len(calls) == 2
    payload = json.loads(sim._file.read_text())
    payload["Mesh"]["adapt"]["order"] = 4
    sim._file.write_text(json.dumps(payload))
    job._artifact_contract_fingerprints()
    assert len(calls) == 3


def test_file_reference_layout_normalizes_job_relative_parent_segments(tmp_path):
    sim = simulation_at(tmp_path)
    job = transient(sim)
    job.save()
    layout = fs.JobLayout.from_payload(
        json.loads(job.job_file.read_text()), job_file=job.job_file
    )
    assert layout.simulation_file == sim._file
    assert ".." not in layout.simulation_relpath.parts


def test_expanded_json_uses_native_float_digits_indentation_and_utf8():
    from frequensolve.imaging._canonical_json import canonical_json_text

    # This double's nlohmann/Grisu2 digits differ from Python's repr.
    payload = {"z": [-6016.951217939863, {}, []], "a": "é"}
    expected = (
        '{\n  "a": "é",\n  "z": [\n    -6016.9512179398635,\n    {},\n    []\n  ]\n}'
    )
    assert canonical_json_text(payload, indent=2) == expected


def test_typed_view_applies_coordinate_system_and_unit_overrides(tmp_path):
    sim = simulation_at(tmp_path)
    job = fs.FrequencyDomainJob("f", sim, [8])
    system = {"type": "cartesian", "name": "shifted", "origin": [1, 2]}
    job.override_simulation(
        coordinate_systems=[system], global_coordinate_system=system, f0=12
    )
    view = job.effective_simulation
    assert view.coordinate_systems["shifted"].name == "shifted"
    assert view.global_coordinate_system.name == "shifted"
    assert view.model._coordinate_systems is view.coordinate_systems
    assert view.units.f0 == 12
    assert not sim.coordinate_systems
    assert sim.global_coordinate_system is None
    assert sim.units.f0 is None


@pytest.mark.parametrize("path", ["Mesh/adapt/f_adapt", "/Mesh/adapt/f_adapt"])
def test_json_path_mapping_updates_persists_and_preserves_siblings(tmp_path, path):
    sim = simulation_at(tmp_path)
    sim.mesh.set_adapt(order=3, elems_per_wave=1.5, f_adapt=8)
    job = transient(sim)
    assert job.override_simulation({path: 5.6, "Solver/tolerance": 1e-4}) is job
    assert job.effective_simulation.mesh.adapt.f_adapt == 5.6
    assert job.effective_simulation.mesh.adapt.order == 3
    assert sim.mesh.adapt.f_adapt == 8
    job.save()
    payload = json.loads(job.job_file.read_text())
    assert payload["simulation"]["overrides"]["Mesh"]["adapt"]["f_adapt"] == 5.6
    assert not any("/" in key for key in payload["simulation"]["overrides"])
    loaded = fs.BaseJob.load(job.job_file)
    assert loaded.to_fs() == job.to_fs()
    assert loaded.effective_simulation.solver.tolerance == 1e-4


def test_path_dictionary_constructor_and_direct_edits_match_nested_overrides(tmp_path):
    sim = simulation_at(tmp_path)
    nested = transient(sim, simulation_overrides={"Mesh": {"adapt": {"f_adapt": 5.6}}})
    paths = transient(sim, simulation_overrides={"Mesh/adapt/f_adapt": 5.6})
    assert (
        paths._simulation_overrides_payload() == nested._simulation_overrides_payload()
    )
    paths.simulation_overrides["Mesh/adapt/f_adapt"] = 4.8
    assert paths.effective_simulation.mesh.adapt.f_adapt == 4.8
    frequency = fs.FrequencyDomainJob(
        "f", sim, [8], simulation_overrides={"Solver/tolerance": 0.001}
    )
    harmonic = fs.TimeDomainJob(
        "h", sim, f_max=8, df=1, simulation_overrides={"Solver/tolerance": 0.001}
    )
    assert frequency.effective_simulation.solver.tolerance == 0.001
    assert harmonic.effective_simulation.solver.tolerance == 0.001


def test_path_update_does_not_freeze_inferred_transient_defaults(tmp_path):
    job = transient(simulation_at(tmp_path))
    job.override_simulation({"Solver/tolerance": 1e-4})
    job.time = fs.TimeIntegration(0, 1, 0.01, f_max=12)
    assert job.effective_simulation.mesh.adapt.f_adapt == 12
    assert "Mesh" not in job.simulation_overrides


def test_pointer_paths_escape_nested_member_names_and_merge_in_order(tmp_path):
    job = transient(simulation_at(tmp_path))
    job.override_simulation(
        {"/custom/a~1b/~0key": None, "Mesh/adapt": {"order": 4}, "Mesh/adapt/order": 5}
    )
    assert job.effective_simulation.extra["custom"] == {"a/b": {"~key": None}}
    assert job.effective_simulation.mesh.adapt.order == 5
    job.override_simulation({"BCs": []})
    assert not job.effective_simulation.BCs


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/",
        "Mesh//adapt",
        "Mesh/adapt/",
        "/custom/~2bad",
        "BCs/0/conditions",
        "/BCs/-",
    ],
)
def test_invalid_or_array_index_paths_do_not_partially_update_job(tmp_path, path):
    job = transient(simulation_at(tmp_path))
    original = copy.deepcopy(job.simulation_overrides)
    with pytest.raises(ValueError):
        job.override_simulation({"Solver/tolerance": 1e-4, path: 5.6})
    assert job.simulation_overrides == original
