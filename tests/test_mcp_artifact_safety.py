"""Security and behavior tests for saved-artifact MCP inspection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import frequensolve as fs
import frequensolve.mcp_server.artifacts as artifact_module
from frequensolve.mcp_server.artifacts import (
    ArtifactSafetyError,
    inspect_or_validate_artifact,
    normalize_allowed_roots,
)


def _build_catalog_job(project_path: Path):
    setup = fs.load_simulation_knowledge().get_starter_scenario().setup
    project = fs.Project(path=project_path, **dict(setup["project"]))
    simulation = project.new_simulation(**dict(setup["simulation"]))

    model_config = dict(setup["model"])
    surfaces = model_config.pop("surfaces")
    layers = model_config.pop("layers")
    assert model_config.pop("type") == "LayeredModel"
    model = fs.LayeredModel(**model_config)
    for index, surface in enumerate(surfaces):
        model.add_surface(**surface)
        if index < len(layers):
            model.add_layer(**layers[index])
    simulation += model

    mesh_config = dict(setup["mesh"])
    assert mesh_config.pop("type") == "HexMeshGenerator"
    adapt = mesh_config.pop("adapt")
    source_grading = mesh_config.pop("source_grading")
    simulation += model.hex_mesh_generator(**mesh_config)
    simulation.mesh.set_adapt(**adapt)
    simulation.mesh.set_source_grading(**source_grading)

    for boundary in setup["boundary_conditions"]:
        simulation += fs.BoundaryCondition(**boundary)

    acquisition_config = setup["acquisition"]
    acquisition = fs.Acquisition()
    acquisition.add_sources(**acquisition_config["source"])
    receiver_config = acquisition_config["receiver_group"]
    receiver = fs.ReceiverNode(name=receiver_config["device_name"])
    receiver.add_component(**receiver_config["component"])
    line = receiver_config["coordinate_line"]
    coordinates = [
        [x, line["fixed"]["z"]]
        for x in np.linspace(line["start"], line["stop"], line["count"])
    ]
    acquisition.add_receiver_group(
        name=receiver_config["name"],
        device=receiver,
        coords=coordinates,
    )
    simulation += acquisition
    simulation += fs.Discretization(**setup["discretization"])
    simulation += fs.SolverConfig(**setup["solver"])

    job_config = dict(setup["job"])
    assert job_config.pop("type") == "FrequencyDomainJob"
    outputs = job_config.pop("outputs")
    vtk_outputs = [fs.VtkOutput.domain(**item) for item in outputs["vtk"]]
    job = fs.FrequencyDomainJob(
        simulation=simulation,
        outputs=vtk_outputs,
        **job_config,
    )
    project.save()
    job_file = job.save()
    return simulation._file, job_file


@pytest.fixture
def saved_artifacts(tmp_path):
    root = (tmp_path / "project").resolve()
    simulation_file, job_file = _build_catalog_job(root)
    sandbox = (tmp_path / "sandbox").resolve()
    sandbox.mkdir()
    return {
        "root": root,
        "sandbox": sandbox,
        "simulation": simulation_file.relative_to(root).as_posix(),
        "job": job_file.relative_to(root).as_posix(),
    }


def _inspect(saved, path: str, mode: str = "validate") -> dict[str, Any]:
    return inspect_or_validate_artifact(
        {"project": str(saved["root"])},
        "project",
        path,
        mode,
        saved["sandbox"],
    )


def _write_payload(root: Path, path: str, payload: Any) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _first_code(result: MappingLike) -> str:
    return result["issues"][0]["code"]


MappingLike = dict[str, Any]


def test_catalog_saved_simulation_and_frequency_job_are_supported(saved_artifacts):
    simulation = _inspect(saved_artifacts, saved_artifacts["simulation"], "inspect")
    job = _inspect(saved_artifacts, saved_artifacts["job"])

    assert simulation == {
        "schema": "frequensolve-mcp-artifact-result/v1",
        "ok": True,
        "mode": "inspect",
        "artifact": {
            "type": "SeismicSimulation",
            "physics": "acoustic",
            "dimension": 2,
            "workflow": None,
            "frequencies": [],
            "task_count": None,
            "output_kinds": [],
        },
        "issues": [],
    }
    assert job["ok"] is True
    assert job["artifact"] == {
        "type": "FrequencyDomainJob",
        "physics": "acoustic",
        "dimension": 2,
        "workflow": "forward",
        "frequencies": [10.0],
        "task_count": 1,
        "output_kinds": ["traces", "vtk"],
    }


def test_saved_job_preview_returns_the_bounded_artifact_summary(saved_artifacts):
    result = _inspect(saved_artifacts, saved_artifacts["job"], "preview")

    assert result == {
        "schema": "frequensolve-mcp-artifact-result/v1",
        "ok": True,
        "mode": "preview",
        "artifact": {
            "type": "FrequencyDomainJob",
            "physics": "acoustic",
            "dimension": 2,
            "workflow": "forward",
            "frequencies": [10.0],
            "task_count": 1,
            "output_kinds": ["traces", "vtk"],
        },
        "issues": [],
    }


def test_time_domain_job_uses_the_exact_supported_type(saved_artifacts):
    payload = json.loads((saved_artifacts["root"] / saved_artifacts["job"]).read_text())
    payload.update(
        {
            "_type": "TimeDomainJob",
            "f_list": [5.0, 10.0],
            "Outputs": {"traces": {"_type": "TraceOutput", "path": "traces"}},
        }
    )
    path = _write_payload(saved_artifacts["root"], "time-job.json", payload)

    result = _inspect(saved_artifacts, path)

    assert result["ok"] is True
    assert result["artifact"]["type"] == "TimeDomainJob"
    assert result["artifact"]["frequencies"] == [5.0, 10.0]
    assert result["artifact"]["task_count"] == 2


def test_validation_returns_stable_export_readiness_diagnostics(saved_artifacts):
    job_path = saved_artifacts["root"] / saved_artifacts["job"]
    payload = json.loads(job_path.read_text())
    payload["Outputs"]["ParaView"][0]["fields"] = ["presure"]
    job_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _inspect(saved_artifacts, saved_artifacts["job"])

    assert result["ok"] is False
    assert result["issues"] == [
        {
            "severity": "error",
            "code": "field.unsupported",
            "path": "outputs.vtk[0].fields[0]",
        }
    ]


@pytest.mark.parametrize(
    "path",
    [
        "../secret.json",
        "%2e%2e/secret.json",
        "%252e%252e/secret.json",
        "/etc/passwd",
        r"C:\secret.json",
        r"folder\secret.json",
        "~/.secret.json",
        "folder/\x01secret.json",
        "https://example.invalid/file.json",
        "artifact.txt",
    ],
)
def test_artifact_path_rejects_non_posix_or_escaping_input(saved_artifacts, path):
    result = _inspect(saved_artifacts, path)

    assert result["ok"] is False
    assert _first_code(result).startswith("artifact.path.")
    assert path not in json.dumps(result)


def test_intermediate_and_final_symlinks_are_rejected(saved_artifacts):
    simulation = saved_artifacts["root"] / saved_artifacts["simulation"]
    link_dir = saved_artifacts["root"] / "linked"
    link_dir.symlink_to(simulation.parent, target_is_directory=True)
    final_link = saved_artifacts["root"] / "simulation-link.json"
    final_link.symlink_to(simulation)

    intermediate = _inspect(saved_artifacts, "linked/" + simulation.name)
    final = _inspect(saved_artifacts, final_link.name)

    assert _first_code(intermediate) == "artifact.path.symlink"
    assert _first_code(final) == "artifact.path.symlink"


def test_allowed_root_symlink_swap_fails_closed(saved_artifacts):
    pinned_roots = normalize_allowed_roots({"project": saved_artifacts["root"]})
    original_root = saved_artifacts["root"]
    moved_root = original_root.parent / "moved-project"
    original_root.rename(moved_root)
    original_root.symlink_to(moved_root, target_is_directory=True)

    result = inspect_or_validate_artifact(
        pinned_roots,
        "project",
        saved_artifacts["job"],
        "validate",
        sandbox_root=saved_artifacts["sandbox"],
    )

    assert _first_code(result) == "artifact.root.invalid"


def test_directory_and_fifo_are_rejected_without_opening_them(saved_artifacts):
    directory = saved_artifacts["root"] / "directory.json"
    directory.mkdir()
    fifo = saved_artifacts["root"] / "pipe.json"
    os.mkfifo(fifo)

    directory_result = _inspect(saved_artifacts, directory.name)
    fifo_result = _inspect(saved_artifacts, fifo.name)

    assert _first_code(directory_result) == "artifact.path.not_file"
    assert _first_code(fifo_result) == "artifact.path.not_file"


def test_artifact_file_size_is_bounded(saved_artifacts):
    path = saved_artifacts["root"] / "large.json"
    path.write_bytes(b'{"padding":"' + (b"x" * (2 * 1024 * 1024)) + b'"}')

    result = _inspect(saved_artifacts, path.name)

    assert _first_code(result) == "artifact.path.too_large"


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (
            '{"_type":"SeismicSimulation","_type":"SeismicSimulation"}',
            "artifact.json.duplicate_key",
        ),
        (
            '{"_type":"SeismicSimulation","value":NaN}',
            "artifact.json.nonfinite",
        ),
        (
            '{"_type":"SeismicSimulation","value":1e999}',
            "artifact.json.nonfinite",
        ),
    ],
)
def test_strict_json_rejects_duplicates_and_nonfinite_numbers(
    saved_artifacts, raw, code
):
    path = saved_artifacts["root"] / "strict.json"
    path.write_text(raw, encoding="utf-8")

    result = _inspect(saved_artifacts, path.name)

    assert _first_code(result) == code


def test_json_depth_node_list_string_and_key_limits(saved_artifacts):
    deeply_nested: Any = "leaf"
    for _ in range(65):
        deeply_nested = {"child": deeply_nested}
    cases = {
        "depth.json": deeply_nested,
        "nodes.json": {"items": [[index] for index in range(25_001)]},
        "list.json": {"items": [0] * 10_001},
        "string.json": {"value": "x" * (64 * 1024 + 1)},
        "key.json": {"k" * 257: 1},
    }

    results = [
        _inspect(
            saved_artifacts,
            _write_payload(saved_artifacts["root"], filename, payload),
        )
        for filename, payload in cases.items()
    ]

    assert {_first_code(result) for result in results} == {"artifact.json.limits"}


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (
            lambda payload: payload["Mesh"]["generator"].update(
                {"n": [1_000_000, 1_000_000]}
            ),
            "artifact.allocation.limit",
        ),
        (
            lambda payload: payload["Model"]["surfaces"][0].update(
                {
                    "depth": {
                        "value": 0.0,
                        "grid": {
                            "_type": "CartesianGrid",
                            "n": [1_000_000_000],
                            "x0": [0.0],
                            "x1": [1.0],
                        },
                    }
                }
            ),
            "artifact.allocation.unsupported",
        ),
    ],
)
def test_compact_allocation_requests_fail_before_deserialization(
    saved_artifacts,
    mutation,
    code,
):
    simulation_path = saved_artifacts["root"] / saved_artifacts["simulation"]
    payload = json.loads(simulation_path.read_text())
    mutation(payload)
    simulation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _inspect(saved_artifacts, saved_artifacts["simulation"])

    assert _first_code(result) == code


@pytest.mark.parametrize(
    "frequencies",
    [
        [1.0, 1.000001, 101.0],
        [[1.0, -0.25], [1.000001, -0.25], [101.0, -0.25]],
    ],
)
def test_time_domain_implied_sweep_is_bounded_before_deserialization(
    saved_artifacts, monkeypatch, frequencies
):
    payload = json.loads((saved_artifacts["root"] / saved_artifacts["job"]).read_text())
    payload["_type"] = "TimeDomainJob"
    payload["f_list"] = frequencies
    path = _write_payload(saved_artifacts["root"], "huge-time-sweep.json", payload)

    def fail_if_loaded(*_args, **_kwargs):
        pytest.fail("unsafe payload reached package deserialization")

    monkeypatch.setattr(artifact_module, "_load_in_sandbox", fail_if_loaded)
    result = _inspect(saved_artifacts, path)

    assert _first_code(result) == "artifact.allocation.limit"


@pytest.mark.parametrize(
    "frequencies",
    [
        [1.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
        [[1.0, 0.0], [2.0], [3.0, 0.0]],
        [1.0, True, 3.0],
    ],
)
def test_invalid_time_domain_sweep_fails_before_deserialization(
    saved_artifacts, monkeypatch, frequencies
):
    payload = json.loads((saved_artifacts["root"] / saved_artifacts["job"]).read_text())
    payload["_type"] = "TimeDomainJob"
    payload["f_list"] = frequencies
    path = _write_payload(saved_artifacts["root"], "invalid-time-sweep.json", payload)

    def fail_if_loaded(*_args, **_kwargs):
        pytest.fail("invalid payload reached package deserialization")

    monkeypatch.setattr(artifact_module, "_load_in_sandbox", fail_if_loaded)
    result = _inspect(saved_artifacts, path)

    assert _first_code(result) == "artifact.allocation.invalid"


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "project", "path": "/tmp/project"},
        {"schema": "fs-simulation-1", "_type": "Project"},
        {"schema": "fs-simulation-1", "_type": "UnknownSimulation"},
        {
            "schema": "fs-job-1",
            "_type": "ImagingJob",
            "workflow": "imaging",
        },
    ],
)
def test_project_unknown_and_imaging_roots_are_rejected(saved_artifacts, payload):
    path = _write_payload(saved_artifacts["root"], "unsupported.json", payload)

    result = _inspect(saved_artifacts, path)

    assert _first_code(result) == "artifact.type.unsupported"


def test_unknown_nested_type_is_rejected_before_deserialization(saved_artifacts):
    payload = json.loads(
        (saved_artifacts["root"] / saved_artifacts["simulation"]).read_text()
    )
    payload["Model"]["extension"] = {"_type": "UserPlugin"}
    path = _write_payload(saved_artifacts["root"], "unknown-nested.json", payload)

    result = _inspect(saved_artifacts, path)

    assert _first_code(result) == "artifact.type.unsupported"


@pytest.mark.parametrize("absolute", [False, True])
def test_job_simulation_reference_cannot_escape_root(
    saved_artifacts, tmp_path, absolute
):
    payload = json.loads((saved_artifacts["root"] / saved_artifacts["job"]).read_text())
    outside = (tmp_path / "outside.json").resolve()
    outside.write_text(
        (saved_artifacts["root"] / saved_artifacts["simulation"]).read_text(),
        encoding="utf-8",
    )
    payload["simulation"] = str(outside) if absolute else "../outside.json"
    path = _write_payload(saved_artifacts["root"], "escape-job.json", payload)

    result = _inspect(saved_artifacts, path)

    assert _first_code(result) == "artifact.simulation.escape"


def test_stale_authored_project_paths_are_ignored(saved_artifacts):
    simulation_file = saved_artifacts["root"] / saved_artifacts["simulation"]
    simulation_payload = json.loads(simulation_file.read_text())
    simulation_payload["project_path"] = "/stale/private/project"
    simulation_file.write_text(json.dumps(simulation_payload), encoding="utf-8")

    job_file = saved_artifacts["root"] / saved_artifacts["job"]
    job_payload = json.loads(job_file.read_text())
    job_payload["project_path"] = "/different/stale/project"
    job_file.write_text(json.dumps(job_payload), encoding="utf-8")

    result = _inspect(saved_artifacts, saved_artifacts["job"])

    assert result["ok"] is True
    assert result["artifact"]["type"] == "FrequencyDomainJob"


@pytest.mark.parametrize(
    ("mutation_path", "value"),
    [
        (("Model", "external"), {"file": "secret.h5"}),
        (("Acquisition", "external"), {"url": "https://example.invalid/data"}),
        (
            ("Acquisition", "external"),
            {"_type": "CoordsFromFile", "file": "coords.h5"},
        ),
    ],
)
def test_embedded_external_file_and_uri_references_are_rejected(
    saved_artifacts, mutation_path, value
):
    payload = json.loads(
        (saved_artifacts["root"] / saved_artifacts["simulation"]).read_text()
    )
    target = payload
    for key in mutation_path[:-1]:
        target = target[key]
    target[mutation_path[-1]] = value
    path = _write_payload(saved_artifacts["root"], "external-reference.json", payload)

    result = _inspect(saved_artifacts, path)

    assert _first_code(result) in {
        "artifact.reference.unsupported",
        "artifact.type.unsupported",
    }


def test_job_output_paths_must_remain_relative(saved_artifacts):
    payload = json.loads((saved_artifacts["root"] / saved_artifacts["job"]).read_text())
    payload["Outputs"]["traces"]["path"] = "../private"
    path = _write_payload(saved_artifacts["root"], "escaping-output.json", payload)

    result = _inspect(saved_artifacts, path)

    assert _first_code(result) == "artifact.reference.unsupported"


def test_result_never_echoes_names_secrets_or_absolute_paths(saved_artifacts):
    payload = json.loads((saved_artifacts["root"] / saved_artifacts["job"]).read_text())
    payload["name"] = "DO_NOT_ECHO_JOB_NAME"
    payload["api_token"] = "DO_NOT_ECHO_TOKEN"
    payload["password"] = "DO_NOT_ECHO_PASSWORD"
    path = _write_payload(saved_artifacts["root"], "sensitive.json", payload)

    result = _inspect(saved_artifacts, path)
    serialized = json.dumps(result)

    assert result["ok"] is True
    assert "DO_NOT_ECHO" not in serialized
    assert str(saved_artifacts["root"]) not in serialized
    assert "/stale/" not in serialized


def test_inspection_does_not_change_the_original_tree(saved_artifacts):
    root = saved_artifacts["root"]

    def snapshot():
        return {
            path.relative_to(root).as_posix(): (
                path.read_bytes() if path.is_file() and not path.is_symlink() else None
            )
            for path in sorted(root.rglob("*"))
        }

    before = snapshot()
    result = _inspect(saved_artifacts, saved_artifacts["job"])
    after = snapshot()

    assert result["ok"] is True
    assert after == before


def test_root_normalization_and_request_errors_are_stable(saved_artifacts):
    assert normalize_allowed_roots({}) == {}
    normalized = normalize_allowed_roots({"project": saved_artifacts["root"]})
    assert normalized == {"project": str(saved_artifacts["root"])}
    for invalid_root_id in ("Project", "a" * 33):
        with pytest.raises(ArtifactSafetyError) as invalid:
            normalize_allowed_roots({invalid_root_id: saved_artifacts["root"]})
        assert invalid.value.code == "artifact.root.invalid"
    with pytest.raises(ArtifactSafetyError) as broad:
        normalize_allowed_roots({"project": "/"})
    assert broad.value.code == "artifact.root.invalid"

    missing_sandbox = inspect_or_validate_artifact(
        {"project": saved_artifacts["root"]},
        "project",
        saved_artifacts["simulation"],
        "validate",
    )
    unknown_root = inspect_or_validate_artifact(
        {"project": saved_artifacts["root"]},
        "other",
        saved_artifacts["simulation"],
        "validate",
        sandbox_root=saved_artifacts["sandbox"],
    )
    unknown_mode = inspect_or_validate_artifact(
        {"project": saved_artifacts["root"]},
        "project",
        saved_artifacts["simulation"],
        "execute",
        sandbox_root=saved_artifacts["sandbox"],
    )
    overlapping_sandbox = inspect_or_validate_artifact(
        {"project": saved_artifacts["root"]},
        "project",
        saved_artifacts["simulation"],
        "validate",
        sandbox_root=saved_artifacts["root"],
    )

    assert _first_code(missing_sandbox) == "artifact.sandbox.required"
    assert _first_code(unknown_root) == "artifact.root.unknown"
    assert _first_code(unknown_mode) == "artifact.request.invalid"
    assert _first_code(overlapping_sandbox) == "artifact.sandbox.invalid"
    assert unknown_mode["mode"] == "unknown"
