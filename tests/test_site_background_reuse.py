"""Background-checkpoint reuse across site jobs: capability and result cleanup."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.sites.aws.aws import AWSSite
from frequensolve.orchestrator.sites.base import BaseSite
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.hpc.site import SlurmSite
from frequensolve.orchestrator.sites.hpc.stampede3 import Stampede3Site
from frequensolve.orchestrator.sites.local.site import LocalSite

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]

_INVALID = ["/abs/file.h5", "../escape.h5", "a/../../b.h5", "", ".", "C:x", "a\\b"]


def _local_site(monkeypatch, **options):
    monkeypatch.setattr(LocalSite, "_get_solver_path", lambda self: "/opt/fs2d_s")
    return LocalSite(**options)


def _files(root, *names):
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)


def test_background_reuse_capability_per_site():
    assert BaseSite.supports_background_reuse is False
    assert LocalSite.supports_background_reuse is True
    assert SlurmSite.supports_background_reuse is True
    assert Stampede3Site.supports_background_reuse is True
    assert AWSSite.supports_background_reuse is False


def test_local_site_removes_result_files_inside_the_result_directory(
    tmp_path, monkeypatch
):
    site = _local_site(monkeypatch)
    results = tmp_path / "job" / "results"
    _files(results, "background_1.json", "ckpt/batch_0.h5", "keep.h5")
    _files(tmp_path, "outside.h5")
    job = SimpleNamespace(name="job", _result_path=results)

    site.remove_result_files(job, ["background_1.json", "ckpt/batch_0.h5", "gone.h5"])

    assert sorted(p.name for p in results.rglob("*") if p.is_file()) == ["keep.h5"]
    assert (tmp_path / "outside.h5").exists()


@pytest.mark.parametrize("bad", _INVALID)
def test_local_site_rejects_escaping_paths_before_removing_anything(
    tmp_path, monkeypatch, bad
):
    site = _local_site(monkeypatch)
    results = tmp_path / "results"
    _files(results, "a.h5")
    job = SimpleNamespace(name="job", _result_path=results)

    with pytest.raises(ValueError, match="inside the job result directory"):
        site.remove_result_files(job, ["a.h5", bad])
    assert (results / "a.h5").exists()


def test_local_site_rejects_directories_and_symlinked_escapes(tmp_path, monkeypatch):
    site = _local_site(monkeypatch)
    results = tmp_path / "results"
    _files(results, "a.h5", "sub/b.h5")
    _files(tmp_path, "outside/c.h5")
    (results / "link").symlink_to(tmp_path / "outside", target_is_directory=True)
    job = SimpleNamespace(name="job", _result_path=results)

    with pytest.raises(IsADirectoryError):
        site.remove_result_files(job, ["a.h5", "sub"])
    with pytest.raises(ValueError, match="leaves"):
        site.remove_result_files(job, ["a.h5", "link/c.h5"])
    assert (results / "a.h5").exists() and (tmp_path / "outside" / "c.h5").exists()
    with pytest.raises(TypeError, match="iterable of paths"):
        site.remove_result_files(job, "a.h5")


def test_sites_without_reuse_only_validate_paths(tmp_path):
    results = tmp_path / "results"
    _files(results, "a.h5")
    job = SimpleNamespace(name="job", _result_path=results)
    aws = object.__new__(AWSSite)

    for site in (BaseSite(), aws):
        site.remove_result_files(job, ["a.h5"])
        with pytest.raises(ValueError):
            site.remove_result_files(job, ["../a.h5"])
    assert (results / "a.h5").exists()


# -- SLURM ---------------------------------------------------------------------


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    from tests.test_site_curvature_slurm import FakeCluster, FakeSlurmSite
    from tests.test_slurm_site_refactor import DummySSHClientClass

    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    fake = FakeCluster(tmp_path)
    site = FakeSlurmSite(
        config=hpc.SlurmSiteConfig(
            hostname="login.example.edu",
            queue="debug",
            mpi_wrapper=str(fake.bin / "srun"),
            poll_interval=0,
            max_nodes=4,
            cores_per_node=8,
        ),
        solver=str(fake.install / "FS_seismic"),
        work_dir=str(tmp_path / "remote" / "work"),
        solver_policy="off",
        run_config=hpc.SlurmRunConfig(poll_interval=0),
    )
    for name in ("run_login_cmd", "put", "get"):
        setattr(site, name, getattr(fake, name))
    return site, fake


class _RemoteJob:
    name = "imaging_normal_0002"

    def _remote_path(self, work_dir):
        return Path(work_dir) / "jobs" / "sim" / self.name


def test_slurm_site_removes_result_files_with_one_login_command(slurm):
    site, fake = slurm
    job = _RemoteJob()
    results = job._remote_path(site.work_dir) / "results"
    _files(results, "background_1.json", "dir name/batch 0.h5", "keep.h5")
    _files(results.parent, "outside.h5")

    site.remove_result_files(
        job, ["background_1.json", "dir name/batch 0.h5", "missing.h5"]
    )

    assert len(fake.login) == 1 and "rm -f --" in fake.login[0]
    assert sorted(p.name for p in results.rglob("*") if p.is_file()) == ["keep.h5"]
    assert (results.parent / "outside.h5").exists()


def test_slurm_site_validates_paths_before_any_login_command(slurm):
    site, fake = slurm
    for bad in _INVALID:
        with pytest.raises(ValueError):
            site.remove_result_files(_RemoteJob(), ["a.h5", bad])
    assert fake.login == []
    site.remove_result_files(_RemoteJob(), [])
    assert fake.login == []


def test_slurm_site_batches_long_lists_and_tolerates_missing_directories(
    slurm, monkeypatch
):
    site, fake = slurm
    job = _RemoteJob()
    results = job._remote_path(site.work_dir) / "results"
    names = [f"ckpt/batch_{index}.h5" for index in range(12)]

    site.remove_result_files(job, names)  # No result directory yet.
    assert len(fake.login) == 1

    _files(results, *names)
    monkeypatch.setattr(hpc, "_REMOVE_COMMAND_BYTES", 64)
    site.remove_result_files(job, names)

    assert len(fake.login) > 2
    assert not any(results.rglob("*.h5"))


def test_slurm_site_refuses_to_remove_directories(slurm):
    site, fake = slurm
    job = _RemoteJob()
    results = job._remote_path(site.work_dir) / "results"
    _files(results, "a.h5", "sub/b.h5")

    with pytest.raises(RuntimeError, match="is a directory"):
        site.remove_result_files(job, ["a.h5", "sub"])
    assert (results / "a.h5").exists() and (results / "sub" / "b.h5").exists()
    assert os.path.isdir(results / "sub")
