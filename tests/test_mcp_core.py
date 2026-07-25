"""Offline behavior tests for the dependency-light MCP core."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from frequensolve._version import get_versions
from frequensolve.knowledge import load_simulation_knowledge
from frequensolve.mcp_server import (
    DRAFT_CONTRACT,
    MCP_CONTRACT,
    CoreInputError,
    build_simulation_draft,
    create_simulation_draft,
    explain_validation,
    find_vetted_example,
    identity_payload,
    preview_simulation,
    render_starter_python,
    resource_payload,
    validate_simulation_draft,
)


def test_identity_payload_exposes_exact_installed_identities():
    payload = identity_payload()
    catalog = load_simulation_knowledge()
    versioneer = get_versions()

    assert payload["mcp_contract"] == MCP_CONTRACT
    assert payload["draft_contract"] == DRAFT_CONTRACT
    assert payload["package"] == {
        "name": "frequensolve",
        "version": catalog.identities.package_version,
        "declared_release": catalog.identities.declared_package_release,
        "full_revisionid": versioneer["full-revisionid"],
        "dirty": versioneer["dirty"],
    }
    assert payload["catalog"] == {
        "schema": catalog.identities.catalog_schema,
        "version": catalog.identities.catalog_version,
        "authoring_rules_schema": catalog.identities.authoring_rules_schema,
    }
    assert payload["solver"] == {
        "compatibility_schema": catalog.identities.compatibility_schema,
        "preferred_release": catalog.identities.preferred_frequensolver_release,
        "preferred_commit": catalog.identities.preferred_frequensolver_commit,
        "validation_profile": catalog.identities.solver_validation_profile,
    }
    assert payload["contracts"] == [
        {
            "name": contract.name,
            "identity": contract.identity,
            "owner": contract.owner,
            "source_revision": contract.source_revision,
        }
        for contract in catalog.identities.contracts
    ]
    json.dumps(payload, allow_nan=False)


def test_draft_is_deterministic_and_changes_only_supported_catalog_fields():
    import frequensolve.mcp_server.core as core

    first = create_simulation_draft(
        project_name="starter-project",
        simulation_name="starter_simulation",
        frequency_hz=10.0,
        receiver_count=7,
    )
    second = create_simulation_draft(
        project_name="starter-project",
        simulation_name="starter_simulation",
        frequency_hz=10.0,
        receiver_count=7,
    )
    catalog_setup = load_simulation_knowledge().get_starter_scenario().setup

    assert first == second
    assert first == {
        "schema": DRAFT_CONTRACT,
        "scenario_id": "known-small-2d-acoustic",
        "project_name": "starter-project",
        "simulation_name": "starter_simulation",
        "job_name": "frequency_10hz",
        "physics": "acoustic",
        "dimension": 2,
        "frequency_hz": 10.0,
        "receiver_count": 7,
    }

    restored = copy.deepcopy(core._setup_from_draft(first))
    restored["project"]["name"] = catalog_setup["project"]["name"]
    restored["simulation"]["name"] = catalog_setup["simulation"]["name"]
    restored["job"]["name"] = catalog_setup["job"]["name"]
    restored["job"]["f_list"] = catalog_setup["job"]["f_list"]
    restored["acquisition"]["receiver_group"]["coordinate_line"]["count"] = (
        catalog_setup["acquisition"]["receiver_group"]["coordinate_line"]["count"]
    )
    assert restored == catalog_setup


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("project_name", "", "invalid_project_name"),
        ("project_name", "../escape", "invalid_project_name"),
        ("project_name", "Uppercase", "invalid_project_name"),
        ("project_name", "a" * 65, "invalid_project_name"),
        ("simulation_name", "has a space", "invalid_simulation_name"),
        ("simulation_name", "9starts_with_number", "invalid_simulation_name"),
        ("frequency_hz", True, "invalid_frequency"),
        ("frequency_hz", 0, "invalid_frequency"),
        ("frequency_hz", -1, "invalid_frequency"),
        ("frequency_hz", float("nan"), "invalid_frequency"),
        ("frequency_hz", float("inf"), "invalid_frequency"),
        ("frequency_hz", "10", "invalid_frequency"),
        ("frequency_hz", 5.0, "invalid_frequency"),
        ("frequency_hz", 12.5, "invalid_frequency"),
        ("frequency_hz", 1_000_001, "invalid_frequency"),
        ("receiver_count", True, "invalid_receiver_count"),
        ("receiver_count", 0, "invalid_receiver_count"),
        ("receiver_count", 1002, "invalid_receiver_count"),
        ("receiver_count", 2.5, "invalid_receiver_count"),
    ],
)
def test_draft_rejects_invalid_bounds_and_names(field, value, code):
    kwargs = {
        "project_name": "project",
        "simulation_name": "simulation",
        "frequency_hz": 10.0,
        "receiver_count": 101,
    }
    kwargs[field] = value

    with pytest.raises(CoreInputError) as raised:
        create_simulation_draft(**kwargs)

    assert raised.value.code == code
    assert raised.value.args == (code,)


def test_draft_requires_exact_supported_structure():
    draft = create_simulation_draft()
    draft["guessed_option"] = "unsafe"

    with pytest.raises(CoreInputError) as raised:
        preview_simulation(draft)

    assert raised.value.code == "unsupported_draft"
    assert "unsafe" not in str(raised.value)


def test_vetted_example_lookup_is_deterministic_and_bounded():
    acoustic = find_vetted_example("acoustic")
    saved = find_vetted_example("saved-project-job-workflow")

    assert acoustic["match"]["id"] == "quickstart-2d-acoustic"
    assert acoustic["match_basis"] == "terms"
    assert saved["match"]["id"] == "saved-project-job-workflow"
    assert saved["match_basis"] == "exact"
    assert find_vetted_example("unrelated words") == find_vetted_example(
        "unrelated words"
    )
    with pytest.raises(CoreInputError, match="invalid_example_query"):
        find_vetted_example("")
    with pytest.raises(CoreInputError, match="invalid_example_query"):
        find_vetted_example("x" * 129)


def test_build_and_validation_use_real_package_objects_without_persistence(
    tmp_path, monkeypatch
):
    import frequensolve.mcp_server.core as core

    draft = create_simulation_draft(receiver_count=5)
    project_path = tmp_path / "isolated-project"
    project_path.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    def reject_operation(*_args, **_kwargs):
        raise AssertionError("persistence or execution must not be called")

    for class_object, method_names in (
        (core.Project, ("save",)),
        (
            core.FrequencyDomainJob,
            ("save", "run", "submit", "dry_run"),
        ),
    ):
        for method_name in method_names:
            if hasattr(class_object, method_name):
                monkeypatch.setattr(class_object, method_name, reject_operation)

    built = build_simulation_draft(draft, project_path=project_path)
    validated = validate_simulation_draft(draft, project_path=project_path)

    assert built == {
        "draft_contract": DRAFT_CONTRACT,
        "scenario_id": "known-small-2d-acoustic",
        "project_name": "project",
        "simulation_name": "known_small_2d_acoustic",
        "job_name": "frequency_10hz",
        "physics": "acoustic",
        "dimension": 2,
        "frequency_count": 1,
        "receiver_count": 5,
        "output_kinds": ["receiver-traces", "vtk-domain"],
    }
    assert validated == {
        "draft_contract": DRAFT_CONTRACT,
        "valid": True,
        "error_count": 0,
        "warning_count": 0,
        "diagnostics": [],
    }
    assert list(project_path.iterdir()) == []
    assert list(unrelated.iterdir()) == []


def test_project_path_is_an_existing_or_parent_owned_absolute_directory(tmp_path):
    draft = create_simulation_draft()

    with pytest.raises(CoreInputError, match="invalid_project_path"):
        build_simulation_draft(draft, project_path="relative")
    created = tmp_path / "created"
    build_simulation_draft(draft, project_path=created)
    assert created.is_dir()
    assert list(created.iterdir()) == []

    with pytest.raises(CoreInputError, match="invalid_project_path"):
        build_simulation_draft(
            draft,
            project_path=tmp_path / "missing-parent" / "project",
        )

    target = tmp_path / "target"
    target.mkdir()
    symlink = tmp_path / "link"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(CoreInputError, match="invalid_project_path"):
        build_simulation_draft(draft, project_path=symlink)


def test_build_failure_does_not_expose_raw_exception_text(tmp_path, monkeypatch):
    import frequensolve.mcp_server.core as core

    draft = create_simulation_draft()

    def fail(*_args, **_kwargs):
        raise RuntimeError("secret at /customer/private/input.h5")

    monkeypatch.setattr(core, "_build_job", fail)
    with pytest.raises(CoreInputError) as raised:
        build_simulation_draft(draft, project_path=tmp_path)

    assert raised.value.code == "draft_build_failed"
    assert "secret" not in str(raised.value)
    assert "customer" not in str(raised.value)


def test_preview_and_renderer_are_deterministic_and_never_execute():
    draft = create_simulation_draft(frequency_hz=10.0, receiver_count=9)

    preview = preview_simulation(draft)
    rendered = render_starter_python(draft)

    assert preview["frequencies_hz"] == [10.0]
    assert preview["task_count"] == 1
    assert preview["source_field_count"] == 1
    assert preview["receiver_count"] == 9
    assert preview["output_kinds"] == ["receiver-traces", "vtk-domain"]
    assert render_starter_python(draft) == rendered
    compile(rendered, "<frequensolve-starter>", "exec")
    for forbidden in (
        ".save(",
        ".submit(",
        ".run(",
        ".dry_run(",
        "Site(",
        "subprocess",
        "__import__",
        "importlib",
    ):
        assert forbidden not in rendered


def test_validation_explanations_and_fixed_resources_are_json_compatible():
    catalog = load_simulation_knowledge()
    explanation = explain_validation("field.unsupported")
    contracts = resource_payload("contracts")

    assert explanation == {
        "code": "field.unsupported",
        "severity": catalog.explain_validation("field.unsupported").severity,
        "path": catalog.explain_validation("field.unsupported").path,
        "explanation": catalog.explain_validation("field.unsupported").explanation,
        "remediation": catalog.explain_validation("field.unsupported").remediation,
    }
    with pytest.raises(CoreInputError, match="validation_code_not_found"):
        explain_validation("unknown.code")
    assert contracts["draft_constraints"]["frequency_hz"] == {
        "constant": 10.0,
        "finite": True,
    }

    for name in (
        "identity",
        "contracts",
        "catalog",
        "public-api",
        "physics",
        "authoring-rules",
        "validation-codes",
        "examples",
        "glossary",
    ):
        payload = resource_payload(name)
        json.dumps(payload, allow_nan=False)
    with pytest.raises(CoreInputError, match="resource_not_found"):
        resource_payload("arbitrary-file")


def test_core_import_does_not_require_mcp_or_server_frameworks():
    repository_root = Path(__file__).resolve().parents[1]
    script = f"""
import builtins
import sys
sys.path.insert(0, {str(repository_root / "src")!r})
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {{'mcp', 'pydantic', 'anyio'}}:
        raise AssertionError(name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import frequensolve.mcp_server.core
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
