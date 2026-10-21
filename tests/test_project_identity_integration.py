from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frequensolve.orchestrator.sites.aws.project_identity import ProjectIdentityMixin
from frequensolve.project.project import Project
from frequensolve.project_identity import ProjectIdentity, local_identity


def test_project_identity_moves_copies_and_missing_manifest(tmp_path):
    source = Project(name="same name", path=tmp_path / "original")
    first = source.enable_identity()
    assert source.project_id == first.project_id
    copied = Project.copy(source.path, tmp_path / "copy")
    assert copied.project_id != source.project_id
    assert copied.project_id is not None
    moved = tmp_path / "moved"
    source.path.rename(moved)
    loaded = Project.load(moved)
    assert loaded.project_id == first.project_id
    assert loaded.enable_identity() == first
    (moved / ".frequensolve" / "project-identity.json").unlink()
    with pytest.raises(ValueError, match="manifest"):
        Project.load(moved)
    with pytest.raises(ValueError, match="manifest"):
        local_identity(moved)


def test_existing_copy_target_is_untouched(tmp_path):
    source = Project(name="same", path=tmp_path / "source")
    source.enable_identity()
    target = Project(name="other", path=tmp_path / "target")
    target_id = target.enable_identity()
    sentinel = target.path / "simulations" / "keep.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        Project.copy(source.path, target.path)
    assert sentinel.read_text() == "original"
    assert ProjectIdentity.load(target.path) == target_id


def test_upload_requires_exact_server_binding_before_transfer(tmp_path):
    project = Project(name="same", path=tmp_path / "local name")
    identity = project.enable_identity()
    site = ProjectIdentityMixin()
    site.graphql_client = SimpleNamespace(
        request_project_identity=Mock(
            return_value={
                **identity.to_dict(),
                "storage_prefix": identity.storage_prefix,
            }
        )
    )
    assert site._ensure_project_identity(project.path, "same") == identity
    assert site.project_storage_key(project.path) == identity.storage_prefix
    request = site.graphql_client.request_project_identity.call_args.args[0]
    assert request["action"] == "bind"
    assert request["manifest"] == identity.to_dict()
    site.graphql_client.request_project_identity.return_value = {
        **identity.to_dict(),
        "storage_prefix": "foreign",
    }
    with pytest.raises(RuntimeError, match="different project binding"):
        site._ensure_project_identity(project.path, "renamed")
    site.graphql_client.request_project_identity.side_effect = RuntimeError("denied")
    with pytest.raises(RuntimeError, match="denied"):
        site._ensure_project_identity(project.path, "same")


def test_legacy_projects_never_register_or_change_root(tmp_path):
    project = Project(name="legacy", path=tmp_path / "legacy folder")
    site = ProjectIdentityMixin()
    site.graphql_client = SimpleNamespace(request_project_identity=Mock())
    assert site._ensure_project_identity(project.path, "legacy") is None
    assert site.project_storage_key(project.path) == project.path.name
    site.graphql_client.request_project_identity.assert_not_called()
    assert "project_id" not in project.to_fs()


def test_sync_binds_before_writing_and_uses_stable_remote_root(tmp_path):
    from pathlib import Path

    from frequensolve.orchestrator.sites.aws.aws import AWSSite as RealAWSSite

    class AWSSite(ProjectIdentityMixin):
        pass

    AWSSite.__module__ = "frequensolve.orchestrator.sites.aws.fake"
    project = Project(name="display", path=tmp_path / "folder")
    identity = project.enable_identity()
    site = AWSSite()
    site.work_dir = Path("remote")
    site._emit = Mock()
    site._ensure_storage_bucket = Mock()
    events = []
    site.graphql_client = SimpleNamespace(
        request_project_identity=Mock(
            side_effect=lambda payload: events.append("bind")
            or {**identity.to_dict(), "storage_prefix": identity.storage_prefix}
        )
    )
    site.put = Mock(side_effect=lambda *args: events.append("upload"))
    RealAWSSite.sync(site, project)
    assert events[0] == "bind"
    assert (
        site.put.call_args.args[1]
        == Path("remote") / identity.storage_prefix / "display.json"
    )
    site.put.reset_mock()
    site.graphql_client.request_project_identity.side_effect = RuntimeError("disabled")
    with pytest.raises(RuntimeError, match="disabled"):
        RealAWSSite.sync(site, project)
    site.put.assert_not_called()


def test_concurrent_copies_reject_before_writing_inputs(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    source = Project(name="source", path=tmp_path / "source")
    source.enable_identity()
    target = tmp_path / "target"
    entered, release = Event(), Event()
    original = Project._copy_project_inputs
    calls = []

    def blocked_copy(src, dest):
        calls.append((src, dest))
        entered.set()
        assert release.wait(timeout=5)
        return original(src, dest)

    monkeypatch.setattr(Project, "_copy_project_inputs", staticmethod(blocked_copy))
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(Project.copy, source.path, target)
        try:
            assert entered.wait(timeout=5)
            site = ProjectIdentityMixin()
            site.graphql_client = SimpleNamespace(request_project_identity=Mock())
            with pytest.raises(RuntimeError, match="Finish the project copy"):
                site._ensure_project_identity(target, "incomplete")
            site.graphql_client.request_project_identity.assert_not_called()
            second = pool.submit(Project.copy, source.path, target)
            with pytest.raises(FileExistsError):
                second.result(timeout=2)
        finally:
            release.set()
        copied = first.result(timeout=5)
    assert len(calls) == 1
    assert copied.project_id != source.project_id
    assert ProjectIdentity.load(target).project_id == copied.project_id


def test_failed_copy_keeps_upload_fence(tmp_path, monkeypatch):
    source = Project(name="source", path=tmp_path / "source")
    source.enable_identity()
    target = tmp_path / "target"

    def fail_after_partial_write(src, dest):
        (dest / "partial-input.json").write_text("{}")
        raise OSError("simulated input-copy failure")

    monkeypatch.setattr(
        Project, "_copy_project_inputs", staticmethod(fail_after_partial_write)
    )
    with pytest.raises(OSError, match="simulated input-copy failure"):
        Project.copy(source.path, target)
    assert ProjectIdentity.load(target).project_id != source.project_id
    assert (target / ".fs-copy.lock").exists()
    site = ProjectIdentityMixin()
    site.graphql_client = SimpleNamespace(request_project_identity=Mock())
    with pytest.raises(RuntimeError, match="Finish the project copy"):
        site._ensure_project_identity(target, "incomplete")
    site.graphql_client.request_project_identity.assert_not_called()
    with pytest.raises(FileExistsError):
        Project.copy(source.path, target)
