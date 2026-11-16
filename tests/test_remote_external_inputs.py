"""Operation inputs written outside the project are staged with remote jobs.

An ``ImagingProblem`` may keep its workdir (directions, objective vectors,
control states) outside the simulation's project root.  Remote sites upload
and rewrite only project-rooted paths, so those inputs travel with the job:
below ``<job_dir>/inputs/external`` on the site, mirroring the local path.
"""

import functools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from frequensolve.imaging import ControlSpace, DepthProfile, ImagingProblem
from frequensolve.imaging.controls import SourceParameters
from frequensolve.orchestrator.sites.hpc import site as hpc
from frequensolve.simulation.jobs.remote import JobRemoteMixin
from tests.imaging_fakes import FakeImagingSite, layered_simulation
from tests.test_allocation_session import (  # noqa: F401 - fake cluster fixture
    cluster,
    published_request,
)
from tests.test_slurm_site_refactor import DummySlurmSite, DummySSHClientClass

pytestmark = pytest.mark.unit

FREQUENCIES = [4.0, 6.0]


def _problem(
    tmp_path, *, project=None, workdir=None, source=False, control_identity=False
):
    # Per-task direction copies (older solvers) exercise the task-sibling staging; direction sets are one file.
    fake = FakeImagingSite(seed=7, control_identity=control_identity)
    controls = {"vp": DepthProfile("vp", "sediment", count=4)}
    if source:
        # Source blocks are Sauce-owned: points are staged as controls.state.
        controls["src"] = SourceParameters(signature=True)
    problem = ImagingProblem(
        layered_simulation(project or tmp_path / "project"),
        controls=ControlSpace(**controls),
        observed={"surface": tmp_path / "observed.h5"},
        frequencies=FREQUENCIES,
        site=fake,
        workdir=workdir,
    )
    problem.state = problem.state_from(np.linspace(-1.0, 1.0, problem.space.size))
    return problem, fake


def _jobs(fake, action):
    return [job for job in fake.jobs if getattr(job, "action", None) == action]


def _materialize(job, remote_project):
    """Stage ``job`` like a remote site; return its remote payload and files."""

    local, remote_job = job.save_for_remote("Remote", remote_project)
    pairs = job.remote_input_files(remote_project)
    for source, target in pairs:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    Path(remote_job).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, remote_job)
    return json.loads(Path(remote_job).read_text()), pairs


def _task_resolves(stem, tasks):
    """Whether every task finds its input as Sauce resolves it on the site."""

    stem = Path(stem)
    return all(
        stem.with_name(f"{stem.stem}_{task}{stem.suffix}").is_file() or stem.is_file()
        for task in range(1, tasks + 1)
    )


def test_normal_batch_directions_outside_project_are_staged_with_the_job(tmp_path):
    workdir = tmp_path / "imaging" / "outside"
    problem, fake = _problem(tmp_path, workdir=workdir)
    lin = problem.linearize()
    lin.apply_normal_batch([lin.space.random(seed) for seed in (1, 2)])
    (job,) = _jobs(fake, "normal")
    local_directions = [str(path) for path in job.to_fs()["fwi_operator"]["directions"]]
    assert all(path.startswith(str(workdir.resolve())) for path in local_directions)

    remote_project = tmp_path / "remote" / "project"
    payload, pairs = _materialize(job, remote_project)

    remote_job_dir = (
        job._saved_layout().with_project(remote_project).job_dir / "inputs" / "external"
    )
    directions = payload["fwi_operator"]["directions"]
    assert len(directions) == 2
    for local, remote in zip(local_directions, directions):
        assert Path(remote).is_relative_to(remote_job_dir)
        # The mirrored name keeps the task-sibling resolution of the stem.
        assert Path(remote).name == Path(local).name
        assert _task_resolves(remote, len(FREQUENCIES))
    # Every staged copy lands inside the remote project.
    assert all(Path(remote).is_relative_to(remote_project) for _, remote in pairs)
    # The local job definition still names the local files.
    saved = json.loads(job.job_file.read_text())
    assert saved["fwi_operator"]["directions"] == local_directions


def test_direction_set_outside_project_is_staged_as_one_file(tmp_path):
    workdir = tmp_path / "imaging" / "outside"
    problem, fake = _problem(tmp_path, workdir=workdir, control_identity=True)
    lin = problem.linearize()
    lin.apply_normal_batch([lin.space.random(seed) for seed in (1, 2)])
    (job,) = _jobs(fake, "normal")
    local = str(job.to_fs()["fwi_operator"]["direction_set"])
    assert local.startswith(str(workdir.resolve()))

    remote_project = tmp_path / "remote" / "project"
    payload, pairs = _materialize(job, remote_project)

    remote = Path(payload["fwi_operator"]["direction_set"])
    assert remote.is_relative_to(remote_project) and remote.is_file()
    assert remote.name == Path(local).name
    # One file serves every task: nothing task-suffixed is staged beside it.
    assert [
        Path(source).name
        for source, _ in pairs
        if Path(source).parent == Path(local).parent
    ] == [Path(local).name]


def test_vjp_objective_vector_manifest_outside_project_keeps_its_files(tmp_path):
    workdir = tmp_path / "imaging" / "outside"
    problem, fake = _problem(tmp_path, workdir=workdir)
    lin = problem.linearize()
    lin.vjp(lin.data_space.ones())
    (job,) = _jobs(fake, "vjp")
    stem = Path(job.to_fs()["fwi_operator"]["objective_vector"])
    assert stem.is_relative_to(workdir.resolve())

    remote_project = tmp_path / "remote" / "project"
    payload, pairs = _materialize(job, remote_project)

    remote = Path(payload["fwi_operator"]["objective_vector"])
    assert remote.is_relative_to(remote_project / "jobs")
    assert "inputs/external" in remote.as_posix()
    assert _task_resolves(remote, len(FREQUENCIES))
    checked = 0
    for task in range(1, len(FREQUENCIES) + 1):
        manifest = remote.with_name(f"{remote.stem}_{task}{remote.suffix}")
        data = json.loads(manifest.read_text())
        # Data files a manifest names resolve beside its remote copy.
        for reference in JobRemoteMixin._iter_file_references(data):
            target = Path(reference)
            if not target.is_absolute():
                target = manifest.parent / target
            assert target.is_file(), reference
            assert target.is_relative_to(remote_project), reference
            checked += 1
    assert checked >= len(FREQUENCIES)


def test_linearize_control_state_outside_project_is_staged(tmp_path):
    workdir = tmp_path / "imaging" / "outside"
    problem, fake = _problem(tmp_path, workdir=workdir, source=True)
    problem.linearize()
    problem.state = problem.state_from(np.linspace(0.5, -0.5, problem.space.size))
    problem.linearize()
    staged = [
        job
        for job in _jobs(fake, "linearize")
        if job.to_fs()["fwi_operator"].get("controls", {}).get("state")
    ]
    assert staged
    job = staged[-1]
    local_state = Path(job.to_fs()["fwi_operator"]["controls"]["state"])
    assert local_state.is_relative_to(workdir.resolve())

    remote_project = tmp_path / "remote" / "project"
    payload, _pairs = _materialize(job, remote_project)
    remote_state = Path(payload["fwi_operator"]["controls"]["state"])
    assert "inputs/external" in remote_state.as_posix()
    assert remote_state.read_bytes() == local_state.read_bytes()


def test_default_workdir_inside_project_keeps_project_layout(tmp_path):
    problem, fake = _problem(tmp_path)
    lin = problem.linearize()
    lin.apply_normal_batch([lin.space.random(seed) for seed in (1, 2)])
    (job,) = _jobs(fake, "normal")
    project = Path(problem.simulation.project_path).resolve()
    local = [Path(path) for path in job.to_fs()["fwi_operator"]["directions"]]
    assert all(path.is_relative_to(project) for path in local)

    remote_project = tmp_path / "remote" / "project"
    payload, pairs = _materialize(job, remote_project)
    directions = [Path(path) for path in payload["fwi_operator"]["directions"]]
    assert directions == [remote_project / path.relative_to(project) for path in local]
    assert not any("inputs/external" in str(remote) for _, remote in pairs)
    assert all(_task_resolves(path, len(FREQUENCIES)) for path in directions)


def test_workdir_sharing_the_project_prefix_is_staged_not_misrewritten(tmp_path):
    # ``<runs>/project`` is a text prefix of ``<runs>/project_imaging``: the
    # project-root rewrite must not mangle the external inputs.
    runs = tmp_path / "runs"
    problem, fake = _problem(
        tmp_path, project=runs / "project", workdir=runs / "project_imaging"
    )
    lin = problem.linearize()
    lin.apply_normal_batch([lin.space.random(seed) for seed in (1, 2)])
    (job,) = _jobs(fake, "normal")
    local = [Path(path) for path in job.to_fs()["fwi_operator"]["directions"]]

    remote_project = tmp_path / "remote" / "project"
    payload, _pairs = _materialize(job, remote_project)
    external = (
        job._saved_layout().with_project(remote_project).job_dir / "inputs" / "external"
    )
    directions = [Path(path) for path in payload["fwi_operator"]["directions"]]
    assert directions == [external.joinpath(*path.parts[1:]) for path in local]
    assert all(_task_resolves(path, len(FREQUENCIES)) for path in directions)


def test_only_existing_operation_inputs_outside_the_project_move(tmp_path):
    workdir = tmp_path / "imaging" / "outside"
    problem, fake = _problem(tmp_path, workdir=workdir)
    lin = problem.linearize()
    lin.apply_normal_batch([lin.space.random(seed) for seed in (1, 2)])
    (job,) = _jobs(fake, "normal")
    project = Path(job.project_path)
    external = str(job.directions[0])
    site_only = "/scratch/shared/directions/direction.h5"
    observed = str(tmp_path / "observed.h5")
    Path(observed).write_bytes(b"observed")
    payload = {
        "fwi_operator": {"action": "normal", "directions": [site_only, external]},
        "observed": observed,
        "label": external,
    }
    root = tmp_path / "remote" / "inputs"
    moved = dict(
        job._external_input_locations(
            payload, source_roots=[project], external_root=root
        )
    )
    # An absent absolute path may already live on the site, data references
    # keep their behaviour, and every exact occurrence of a staged input moves.
    assert moved == {
        ("fwi_operator", "directions", 1): str(root / external.lstrip("/")),
        ("label",): str(root / external.lstrip("/")),
    }


def test_slurm_transfer_uploads_external_inputs_to_the_rewritten_paths(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(hpc, "SSHClientClass", DummySSHClientClass)
    workdir = tmp_path / "imaging" / "outside"
    problem, fake = _problem(tmp_path, workdir=workdir)
    lin = problem.linearize()
    lin.apply_normal_batch([lin.space.random(seed) for seed in (1, 2)])
    (job,) = _jobs(fake, "normal")

    site = DummySlurmSite("project/run")
    puts = {}

    def fake_put(local, remote):
        puts[Path(remote)] = Path(local)

    monkeypatch.setattr(site, "put", fake_put)
    monkeypatch.setattr(
        site, "put_files", lambda pairs, **kwargs: [fake_put(*pair) for pair in pairs]
    )
    monkeypatch.setattr(site, "run_login", lambda cmd: "")
    monkeypatch.setattr(site, "_claim_remote_project", lambda path: None)

    _script, remote_job = site._transfer_SLURM_job("script", job)
    payload = json.loads(puts[Path(remote_job)].read_text())
    directions = payload["fwi_operator"]["directions"]
    assert all(Path(path).is_relative_to(site.work_dir) for path in directions)
    for stem in map(Path, directions):
        for task in range(1, len(FREQUENCIES) + 1):
            sibling = stem.with_name(f"{stem.stem}_{task}{stem.suffix}")
            assert sibling in puts, sibling
            assert puts[sibling].is_relative_to(workdir.resolve())
    assert str(workdir.resolve()) not in json.dumps(
        {key: value for key, value in payload.items() if key != "fwi_operator"}
    )


@pytest.mark.hpc_hermetic
@pytest.mark.timeout(60)
def test_session_submit_stages_external_inputs_with_the_job(
    cluster, monkeypatch, tmp_path  # noqa: F811
):
    # The fake cluster's work_dir is ``tmp_path``; keep the local project and
    # the out-of-project workdir elsewhere so every upload is a real copy.
    local = tmp_path / "local"
    workdir = local / "imaging" / "outside"
    problem, fake = _problem(local, workdir=workdir)
    lin = problem.linearize()
    lin.apply_normal_batch([lin.space.random(seed) for seed in (1, 2)])
    (job,) = _jobs(fake, "normal")
    site = cluster.site
    # The fixture skips simulation/input transfer for its fake jobs.
    monkeypatch.setattr(
        site,
        "_transfer_remote_simulation_inputs",
        functools.partial(hpc.SlurmSite._transfer_remote_simulation_inputs, site),
    )
    monkeypatch.setattr(site, "_claim_remote_project", lambda path: None)

    with site.session() as session:
        handle = session.submit(job)
        request = published_request(session, handle.id)
        handle.wait(check=False)  # the fake solver cannot run a real job
    remote_job = Path(request["job_file"])
    assert remote_job.is_relative_to(site.work_dir)
    payload = json.loads(remote_job.read_text())
    directions = payload["fwi_operator"]["directions"]
    assert len(directions) == 2
    for stem in directions:
        assert Path(stem).is_relative_to(remote_job.parent / "inputs" / "external")
        assert _task_resolves(stem, len(FREQUENCIES))
    assert Path(payload["fwi_operator"]["state"]).is_relative_to(site.work_dir)
