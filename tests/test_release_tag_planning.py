from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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

    assert "version must be canonical ASCII X.Y.Z, such as 0.2.0" in errors


@pytest.mark.parametrize(
    "version",
    ["00.2.0", "0.02.0", "0.2.00", "０.2.0", "0.2.0 ", "0.2.0rc1"],
)
def test_next_release_candidate_rejects_noncanonical_base_versions(
    version: str,
) -> None:
    planner = load_planner()

    with pytest.raises(ValueError, match="canonical ASCII X.Y.Z"):
        planner.next_release_candidate_tag(version, [])


def test_final_tag_from_release_candidate_uses_pep_440_rc_tag() -> None:
    planner = load_planner()

    assert planner.final_tag_from_release_candidate("v0.2.0rc3") == "v0.2.0"


def test_final_tag_from_release_candidate_rejects_semver_style_rc_tag() -> None:
    planner = load_planner()

    errors = planner.validate_release_candidate_tag("v0.2.0-rc.3")

    assert (
        "release candidate tag must be canonical ASCII vX.Y.ZrcN with N >= 1" in errors
    )


@pytest.mark.parametrize(
    "tag",
    [
        "v0.2.0rc0",
        "v00.2.0rc1",
        "v0.02.0rc1",
        "v0.2.00rc1",
        "v０.2.0rc1",
        "v0.2.0rc1 ",
    ],
)
def test_final_tag_rejects_noncanonical_release_candidates(tag: str) -> None:
    planner = load_planner()

    with pytest.raises(ValueError, match="canonical ASCII vX.Y.ZrcN"):
        planner.final_tag_from_release_candidate(tag)


def test_next_release_candidate_does_not_count_noncanonical_tags() -> None:
    planner = load_planner()

    assert (
        planner.next_release_candidate_tag(
            "0.2.0",
            ["v0.2.0rc1", "v0.2.0rc02", "v0.2.0rc0", "v0.2.0rc3 "],
        )
        == "v0.2.0rc2"
    )


@pytest.mark.parametrize(
    ("tag", "is_prerelease", "expected"),
    [
        ("v0.2.0rc1", True, "testpypi"),
        ("v0.2.0", False, "pypi"),
    ],
)
def test_publication_target_is_derived_from_tag_and_release_metadata(
    tag: str,
    is_prerelease: bool,
    expected: str,
) -> None:
    planner = load_planner()

    assert planner.publication_target(tag, is_prerelease=is_prerelease) == expected


@pytest.mark.parametrize(
    ("tag", "is_prerelease", "message"),
    [
        (
            "v0.2.0rc1",
            False,
            "release candidate tags must be published as GitHub prereleases",
        ),
        (
            "v0.2.0",
            True,
            "final release tags must not be published as GitHub prereleases",
        ),
        (
            "v0.2.0-rc.1",
            True,
            "release tag must be canonical ASCII",
        ),
    ],
)
def test_publication_target_rejects_ambiguous_or_mismatched_metadata(
    tag: str,
    is_prerelease: bool,
    message: str,
) -> None:
    planner = load_planner()

    with pytest.raises(ValueError, match=message):
        planner.publication_target(tag, is_prerelease=is_prerelease)
