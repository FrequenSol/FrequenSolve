from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plan_release_tags.py"


def load_planner():
    spec = importlib.util.spec_from_file_location("plan_release_tags", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_next_release_candidate_starts_at_one_for_final_version() -> None:
    planner = load_planner()

    assert planner.next_release_candidate_tag("0.2.0", []) == "v0.2.0rc1"


def test_next_release_candidate_ignores_other_versions_and_increments_current() -> None:
    planner = load_planner()

    assert (
        planner.next_release_candidate_tag(
            "0.2.0",
            ["v0.1.0rc9", "v0.2.0rc1", "v0.2.0rc3", "v0.2.1rc1"],
        )
        == "v0.2.0rc4"
    )


def test_next_release_candidate_rejects_non_final_base_versions() -> None:
    planner = load_planner()

    errors = planner.validate_final_version("0.2.0-rc.1")

    assert "version must be a final PEP 440 base version such as 0.2.0" in errors


def test_final_tag_from_release_candidate_uses_pep_440_rc_tag() -> None:
    planner = load_planner()

    assert planner.final_tag_from_release_candidate("v0.2.0rc3") == "v0.2.0"


def test_final_tag_from_release_candidate_rejects_semver_style_rc_tag() -> None:
    planner = load_planner()

    errors = planner.validate_release_candidate_tag("v0.2.0-rc.3")

    assert "release candidate tag must look like v0.2.0rc1" in errors
