from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_release_version.py"
)


def load_release_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_release_version", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_accepts_matching_v_prefixed_release_tag() -> None:
    validator = load_release_validator()

    errors = validator.validate_release_version(
        version="0.2.0",
        ref_type="tag",
        ref_name="v0.2.0",
    )

    assert errors == []


def test_accepts_matching_pep_440_release_candidate_tag() -> None:
    validator = load_release_validator()

    errors = validator.validate_release_version(
        version="0.2.0rc1",
        ref_type="tag",
        ref_name="v0.2.0rc1",
    )

    assert errors == []


def test_rejects_semver_style_release_candidate_version() -> None:
    validator = load_release_validator()

    errors = validator.validate_release_version(
        version="0.2.0-rc.1",
        ref_type="tag",
        ref_name="v0.2.0-rc.1",
    )

    assert "version must be a clean PEP 440 release such as 0.2.0 or 0.2.0rc1" in errors


def test_rejects_plain_version_tags_because_versioneer_uses_v_prefix() -> None:
    validator = load_release_validator()

    errors = validator.validate_release_version(
        version="0.2.0",
        ref_type="tag",
        ref_name="0.2.0",
    )

    assert "release tag must be v0.2.0" in errors


def test_rejects_branch_derived_local_versions() -> None:
    validator = load_release_validator()

    errors = validator.validate_release_version(
        version="0.0.1+278.gccbbd6f",
        ref_type="branch",
        ref_name="v2",
    )

    assert "release must run from a tag ref" in errors
    assert "version must not include a local version segment" in errors


def test_rejects_dirty_or_untagged_versions() -> None:
    validator = load_release_validator()

    dirty_errors = validator.validate_release_version(
        version="0.2.0+1.gabcdef.dirty",
        ref_type="tag",
        ref_name="v0.2.0",
    )
    untagged_errors = validator.validate_release_version(
        version="0+untagged.1.gabcdef",
        ref_type="tag",
        ref_name="v0.2.0",
    )

    assert "version must not include dirty or untagged markers" in dirty_errors
    assert "version must not include dirty or untagged markers" in untagged_errors


def test_rejects_tag_version_mismatch() -> None:
    validator = load_release_validator()

    errors = validator.validate_release_version(
        version="0.2.1",
        ref_type="tag",
        ref_name="v0.2.0",
    )

    assert "release tag must be v0.2.1" in errors
