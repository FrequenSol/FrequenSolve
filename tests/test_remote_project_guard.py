import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import frequensolve as fs
from frequensolve.orchestrator.sites.hpc.project_guard import MARKER, claim_project
from frequensolve.orchestrator.sites.hpc.site import SlurmSite


def test_project_claim_persists_across_clients_and_input_updates(tmp_path):
    remote = tmp_path / "remote"
    claim_project(remote, "/projects/wedge")
    marker = (remote / MARKER).read_bytes()
    (remote / "simulations").mkdir()
    (remote / "simulations/wedge.json").write_text("updated inputs")
    claim_project(remote, "/projects/wedge")
    assert (remote / MARKER).read_bytes() == marker
    with pytest.raises(ValueError, match="belongs to"):
        claim_project(remote, "/projects/another-wedge")
    assert (remote / "simulations/wedge.json").read_text() == "updated inputs"


def test_concurrent_projects_cannot_claim_same_remote_directory(tmp_path):
    def attempt(source):
        try:
            claim_project(tmp_path, source)
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(2) as pool:
        accepted = list(pool.map(attempt, ["/projects/one", "/projects/two"]))
    assert sum(accepted) == 1
    assert not list(tmp_path.glob(".fs-project-claim-*"))


def test_legacy_project_is_adopted_only_with_matching_origin(tmp_path):
    (tmp_path / "simulations").mkdir()
    (tmp_path / "simulations/sim.json").write_text("existing inputs")
    manifest = tmp_path / "project.json"
    manifest.write_text(
        json.dumps(
            {"name": "project", "simulations": [{"project_path": "/projects/one"}]}
        )
    )
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        claim_project(tmp_path, "/projects/two")
    assert not (tmp_path / MARKER).exists()
    claim_project(tmp_path, "/projects/one")
    assert (tmp_path / MARKER).exists()


def test_unknown_legacy_data_and_corrupt_ownership_fail_closed(tmp_path):
    (tmp_path / "jobs").mkdir()
    (tmp_path / "jobs/job.json").write_text("existing job")
    with pytest.raises(ValueError, match="unverified"):
        claim_project(tmp_path, "/projects/one")
    (tmp_path / MARKER).write_text("partial json")
    with pytest.raises(ValueError):
        claim_project(tmp_path, "/projects/one")
    assert (tmp_path / "jobs/job.json").read_text() == "existing job"


def test_remote_guard_command_handles_quoted_paths_and_conflict(tmp_path):
    def run_login(command, *, timeout):
        return subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        ).stdout

    site = SimpleNamespace(work_dir=tmp_path / "remote ' $literal", run_login=run_login)
    source = tmp_path / "local ' $literal"
    SlurmSite._claim_remote_project(site, source)
    SlurmSite._claim_remote_project(site, source)
    with pytest.raises(RuntimeError, match="belongs to"):
        SlurmSite._claim_remote_project(site, tmp_path / "different")


@pytest.mark.parametrize("response", ["", "login banner", "{}", "[]", '{"ok":false}'])
def test_unverified_remote_response_cannot_authorize_upload(tmp_path, response):
    site = SimpleNamespace(work_dir=tmp_path, run_login=lambda *a, **k: response)
    with pytest.raises(RuntimeError, match="ownership"):
        SlurmSite._claim_remote_project(site, tmp_path / "project")


def test_direct_project_transfer_checks_ownership_before_upload(tmp_path):
    uploaded = []

    class Site:
        work_dir = tmp_path / "remote"

        def _claim_remote_project(self, path):
            raise RuntimeError("project collision")

        def put(self, *args):
            uploaded.append(args)

    project = fs.Project(name="one", path=tmp_path / "local")
    with pytest.raises(RuntimeError, match="project collision"):
        project._transfer(Site())
    assert uploaded == []


def test_direct_simulation_transfer_checks_ownership_before_staging(tmp_path):
    def reject(path):
        raise RuntimeError("project collision")

    site = SimpleNamespace(_claim_remote_project=reject)
    job = SimpleNamespace(project_path=tmp_path)
    with pytest.raises(RuntimeError, match="project collision"):
        SlurmSite._transfer_remote_simulation_inputs(site, job)
