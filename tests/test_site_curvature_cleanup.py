"""Removing staged Sauce curvature files from a SLURM site."""

import os
import time
from datetime import timedelta
from pathlib import Path

import pytest

from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.orchestrator.sites.hpc.curvature import SlurmCurvatureRunner
from tests.test_site_curvature_slurm import FakeCluster, FakeSlurmSite
from tests.test_slurm_site_refactor import DummySSHClientClass

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]

_DAY = 24 * 3600.0


@pytest.fixture
def slurm(tmp_path, monkeypatch):
    """A SLURM site whose login commands run in a local shell."""

    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    fake = FakeCluster(tmp_path)
    site = FakeSlurmSite(
        config=hpc.SlurmSiteConfig(hostname="login.example.edu", queue="debug"),
        solver=str(fake.install / "FS_seismic"),
        work_dir=str(tmp_path / "remote" / "work"),
        solver_policy="off",
    )
    site.run_login_cmd = fake.run_login_cmd
    return site, fake


def _mirror(site, workdir, *, size, age):
    """Stage a fake mirror of ``workdir`` holding ``size`` bytes, ``age`` s old."""

    root = Path(str(SlurmCurvatureRunner(site).remote_root(workdir)))
    files = [root / "histories" / "h.h5", root / "op-1" / "pending.h5"]
    for index, path in enumerate(files):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * (size // 2 + index * (size % 2)))
    stamp = time.time() - age
    for path in [*files, *(file.parent for file in files), root]:
        os.utime(path, (stamp, stamp))
    return root


def test_remove_one_workdir_mirror_with_a_single_command(slurm, tmp_path):
    site, fake = slurm
    kept = _mirror(site, tmp_path / "a" / "curvature", size=10, age=0)
    removed = _mirror(site, tmp_path / "b" / "curvature", size=10, age=0)

    paths = site.remove_curvature_files(tmp_path / "b" / "curvature")

    assert paths == [str(removed)]
    assert not removed.exists() and kept.exists()
    assert len(fake.login) == 1 and "rm -rf --" in fake.login[0]
    # Removing it again, or a mirror that never existed, is not an error.
    assert site.remove_curvature_files(tmp_path / "b" / "curvature") == []
    assert site.remove_curvature_files(tmp_path / "never" / "curvature") == []


def test_age_and_size_budgets_remove_least_recently_used_mirrors(slurm, tmp_path):
    site, _ = slurm
    oldest = _mirror(site, tmp_path / "w1" / "curvature", size=300, age=3 * _DAY)
    older = _mirror(site, tmp_path / "w2" / "curvature", size=200, age=2 * _DAY)
    recent = _mirror(site, tmp_path / "w3" / "curvature", size=100, age=60)
    newest = _mirror(site, tmp_path / "w4" / "curvature", size=100, age=0)
    other = Path(str(SlurmCurvatureRunner(site).curvature_dir())) / "notes"
    other.mkdir()

    assert site.remove_curvature_files(older_than=timedelta(days=2.5)) == [str(oldest)]
    assert site.remove_curvature_files(max_bytes=250) == [str(older)]
    assert recent.exists() and newest.exists()
    assert site.remove_curvature_files(older_than=_DAY, max_bytes=10_000) == []
    assert site.remove_curvature_files() == [str(recent), str(newest)]
    assert other.exists()


def test_cleanup_validates_its_budget(slurm):
    site, fake = slurm

    for options in ({"older_than": -1}, {"max_bytes": -1}, {"max_bytes": True}):
        with pytest.raises(ValueError):
            site.remove_curvature_files(**options)
    assert fake.login == []
    assert site.remove_curvature_files() == []  # No curvature directory yet.
