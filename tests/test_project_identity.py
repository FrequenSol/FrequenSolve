"""Offline contract and filesystem tests; no cloud or solver invocation."""

import concurrent.futures
import json
import os
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from frequensolve.project_identity import (
    DIRECTORY,
    FILENAME,
    MAX_BYTES,
    ProjectIdentity,
    legacy_reference,
    reference_token,
)

VECTORS = json.loads(
    (Path(__file__).parent / "contracts/project-identity-v1.json").read_text()
)
FILES_SUPPORTED = hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY")
files_only = pytest.mark.skipif(not FILES_SUPPORTED, reason="POSIX identity-file pilot")


def test_shared_contract_vectors():
    for value in VECTORS["valid_manifests"]:
        assert ProjectIdentity.from_dict(value).to_dict() == value
    for value in VECTORS["invalid_manifests"]:
        with pytest.raises(ValueError, match="INVALID_PROJECT_IDENTITY"):
            ProjectIdentity.from_dict(value)
    for vector in VECTORS["references"]:
        assert reference_token(vector["value"]) == vector["token"]
    for value in VECTORS["invalid_references"]:
        with pytest.raises(ValueError, match="INVALID_PROJECT_IDENTITY"):
            reference_token(value)


def test_legacy_uuid_keys_are_not_upgraded_by_appearance():
    with pytest.raises(ValueError, match="INVALID_PROJECT_IDENTITY"):
        legacy_reference("\ud800")
    identity = ProjectIdentity.new()
    assert reference_token(legacy_reference(identity.project_id)) != reference_token(
        identity.reference()
    )
    assert (
        legacy_reference("folder/Original Name")["project_key"]
        == "folder/Original Name"
    )
    with pytest.raises(FrozenInstanceError):
        identity.project_id = ProjectIdentity.new().project_id


@files_only
def test_move_and_explicit_copy_as_new(tmp_path):
    source = tmp_path / "same-name"
    source.mkdir()
    identity = ProjectIdentity.new()
    identity.save_new(source)
    moved = tmp_path / "renamed-folder"
    source.rename(moved)
    assert ProjectIdentity.load(moved) == identity
    other = tmp_path / "elsewhere" / "same-name"
    other.mkdir(parents=True)
    fork = identity.fork_to(other)
    assert fork != identity
    assert ProjectIdentity.load(other) == fork
    assert ProjectIdentity.load(moved) == identity
    # Explicit export/import keeps the candidate identity but grants no remote access.
    imported = tmp_path / "explicit-import"
    imported.mkdir()
    ProjectIdentity.from_dict(identity.to_dict()).save_new(imported)
    assert ProjectIdentity.load(imported) == identity


@files_only
def test_existing_or_corrupt_manifest_is_never_replaced(tmp_path):
    first = ProjectIdentity.new()
    path = first.save_new(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        ProjectIdentity.new().save_new(tmp_path)
    assert path.read_bytes() == original
    path.write_text('{"schema_version":"project-identity.v1",')
    corrupt = path.read_bytes()
    with pytest.raises(ValueError):
        ProjectIdentity.load(tmp_path)
    with pytest.raises(FileExistsError):
        ProjectIdentity.new().save_new(tmp_path)
    assert path.read_bytes() == corrupt


@files_only
def test_missing_legacy_state_does_not_create_any_manifest(tmp_path):
    before = list(tmp_path.iterdir())
    with pytest.raises(FileNotFoundError):
        ProjectIdentity.load(tmp_path)
    assert list(tmp_path.iterdir()) == before


@files_only
def test_concurrent_creates_have_one_winner(tmp_path):
    def create(_):
        value = ProjectIdentity.new()
        try:
            value.save_new(tmp_path)
            return value
        except FileExistsError:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        winners = [value for value in pool.map(create, range(16)) if value is not None]
    assert winners == [ProjectIdentity.load(tmp_path)]


@files_only
def test_rejects_metadata_directory_and_file_symlinks(tmp_path):
    root, foreign = tmp_path / "project", tmp_path / "foreign"
    root.mkdir()
    foreign.mkdir()
    (root / DIRECTORY).symlink_to(foreign, target_is_directory=True)
    with pytest.raises(OSError):
        ProjectIdentity.new().save_new(root)
    assert list(foreign.iterdir()) == []
    (root / DIRECTORY).unlink()
    (root / DIRECTORY).mkdir()
    target = foreign / "original"
    target.write_text("keep")
    (root / DIRECTORY / FILENAME).symlink_to(target)
    with pytest.raises(OSError):
        ProjectIdentity.load(root)
    with pytest.raises(FileExistsError):
        ProjectIdentity.new().save_new(root)
    assert target.read_text() == "keep"


@files_only
@pytest.mark.parametrize(
    "payload",
    [
        b"x" * (MAX_BYTES + 1),
        b"\xff",
        b"[]",
        b'{"schema_version":"project-identity.v1","project_id":"11111111-1111-4111-8111-111111111111","project_id":"22222222-2222-4222-8222-222222222222"}',
    ],
)
def test_bounded_strict_manifest_reads(tmp_path, payload):
    metadata = tmp_path / DIRECTORY
    metadata.mkdir()
    (metadata / FILENAME).write_bytes(payload)
    with pytest.raises(ValueError):
        ProjectIdentity.load(tmp_path)


@files_only
def test_directory_and_fifo_manifests_fail_without_blocking(tmp_path):
    metadata = tmp_path / DIRECTORY
    metadata.mkdir()
    path = metadata / FILENAME
    path.mkdir()
    with pytest.raises((ValueError, OSError)):
        ProjectIdentity.load(tmp_path)
    shutil.rmtree(path)
    os.mkfifo(path)
    with pytest.raises(ValueError):
        ProjectIdentity.load(tmp_path)
