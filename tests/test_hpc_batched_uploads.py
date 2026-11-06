"""Batched HPC uploads: one remote mkdir, few transfers, no duplicate sends."""

import os
import shlex
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from frequensolve.imaging._backend import Backend
from frequensolve.orchestrator.sites.hpc import transfer as transfer_module
from frequensolve.orchestrator.sites.hpc.session import AllocationSession
from frequensolve.orchestrator.sites.hpc.site import SlurmSite
from frequensolve.orchestrator.sites.hpc.transfer import SlurmTransferManager

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]

HOST = "login.example.edu"
WORK = PurePosixPath("/scratch/user/gulf")


class FakeRsync:
    """``subprocess.run`` double that records rsync argv and its local sources."""

    def __init__(self):
        self.calls = []
        self.statuses = []

    def __call__(self, argv, **kwargs):
        # argv: rsync [-e <ssh>] <flags/options> <source>... <target>
        sources = []
        skip = False
        for arg in argv[1:-1]:
            if skip:
                skip = False
            elif arg == "-e":
                skip = True
            elif not arg.startswith("-"):
                sources.append(arg)
        self.calls.append(SimpleNamespace(argv=list(argv), listed=sources))
        status = self.statuses.pop(0) if self.statuses else 0
        return SimpleNamespace(returncode=status, stdout="", stderr="")

    def sent(self, start=0):
        """Return the file names the calls from ``start`` on transferred."""

        return [
            PurePosixPath(item).name
            for call in self.calls[start:]
            for item in call.listed
        ]


class RecordingSFTP:
    """SFTP double that publishes atomically and can stall on one ``put``."""

    def __init__(self, fail_on=None, error=None):
        self.fail_on = fail_on
        self.error = error or TimeoutError("stalled")
        self.puts = []
        self.sizes = {}
        self.published = []
        self.removed = []

    def put(self, source, target):
        self.puts.append(target)
        if len(self.puts) == self.fail_on:
            raise self.error
        self.sizes[target] = Path(source).stat().st_size

    def stat(self, target):
        if target not in self.sizes:
            raise FileNotFoundError(target)
        return SimpleNamespace(st_size=self.sizes[target], st_mode=0o100644)

    def chmod(self, target, mode):
        pass

    def posix_rename(self, source, target):
        self.sizes[target] = self.sizes.pop(source)
        self.published.append(target)

    def remove(self, target):
        self.sizes.pop(target, None)
        self.removed.append(target)

    def close(self):
        pass


class BatchSite:
    """SLURM upload surface over a recorded login shell and fake transports."""

    put = SlurmSite.put
    put_files = SlurmSite.put_files
    _transfer_remote_simulation_inputs = SlurmSite._transfer_remote_simulation_inputs

    def __init__(self, *, method="rsync", sftp=None, attempts=3):
        self.work_dir = Path(WORK)
        self.transfer_method = method
        self.commands = []
        self.claimed = []
        self.opened = 0
        self.config = SimpleNamespace(
            hostname=HOST, transfer_attempts=attempts, transfer_stall_timeout=120.0
        )
        self.credentials = SimpleNamespace(username="user")
        self._login_client = SimpleNamespace(
            is_proxy=lambda: method == "rsync",
            get_proxy_details=lambda: ("/tmp/control.sock", "user"),
        )

        def open_sftp():
            self.opened += 1
            return sftp

        self.login_client = SimpleNamespace(open_sftp=open_sftp)
        self._transfer = SlurmTransferManager(self)

    def run_login(self, command, **kwargs):
        self.commands.append(command)
        return ""

    def _recover_login_connection(self):
        return True

    def _claim_remote_project(self, path):
        self.claimed.append(Path(path))


class StagedJob:
    """Job whose staged simulation JSON and project inputs map onto WORK."""

    def __init__(self, project, inputs, simulation):
        self.project_path = project
        self.inputs = inputs
        self.simulation = simulation

    def _project_path(self):
        return self.project_path

    def save_simulation_for_remote(self, site, remote_project):
        return self.simulation, Path(remote_project) / "simulations/sim/sim.json"

    def remote_input_files(self, remote_project):
        return [
            (path, Path(remote_project) / path.relative_to(self.project_path))
            for path in self.inputs
        ]


@pytest.fixture
def rsync(monkeypatch, tmp_path):
    fake = FakeRsync()
    monkeypatch.setattr(transfer_module.subprocess, "run", fake)
    monkeypatch.setattr(transfer_module.logger, "isEnabledFor", lambda level: False)
    lists = tmp_path / "file-lists"
    lists.mkdir()
    monkeypatch.setattr(SlurmTransferManager, "_local_tmp_parent", lambda self: lists)
    return fake


@pytest.fixture
def sleeps(monkeypatch):
    delays = []
    monkeypatch.setattr(transfer_module, "_sleep", delays.append)
    return delays


def _write(path, data=b"direction"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _ops_files(project, *, ops=("0001",), directions=15, tasks=4):
    """Write the per-task normal directions of the measured Gulf jobs."""

    return [
        _write(
            project / "imaging/fwi/abc123/ops" / op / f"direction_{k}_{task}.h5",
            f"{op}/{k}/{task}".encode(),
        )
        for op in ops
        for k in range(directions)
        for task in range(1, tasks + 1)
    ]


def _remote(project, local):
    return WORK / local.relative_to(project).as_posix()


def _mkdir_targets(command):
    words = shlex.split(command)
    assert words[:3] == ["mkdir", "-p", "--"]
    return words[3:]


def _gulf_job(tmp_path):
    project = tmp_path / "project"
    inputs = _ops_files(project)
    simulation = _write(
        project / "jobs/sim/normal/_fs_run/remote/BatchSite/sim.json", b"{}"
    )
    return StagedJob(project, inputs, simulation)


def test_put_files_sends_one_mkdir_and_one_rsync_per_directory(rsync, tmp_path):
    site = BatchSite()
    project = tmp_path / "project"
    files = _ops_files(project, ops=("0001", "0002"))
    pairs = [(local, _remote(project, local)) for local in files]

    sent = site.put_files(pairs)

    assert sent == [remote for _, remote in pairs]
    # Every parent directory in one login command, deepest parents only.
    assert len(site.commands) == 1
    assert _mkdir_targets(site.commands[0]) == [
        f"{WORK}/imaging/fwi/abc123/ops/0001",
        f"{WORK}/imaging/fwi/abc123/ops/0002",
    ]
    # One plain multi-source rsync into each existing directory: no --relative
    # path reconstruction (it replaces symlinked remote directories, and its
    # --no-implied-dirs form fails between macOS openrsync and rsync 3).
    assert [call.argv[-1] for call in rsync.calls] == [
        f"user@{HOST}:{WORK}/imaging/fwi/abc123/ops/0001/",
        f"user@{HOST}:{WORK}/imaging/fwi/abc123/ops/0002/",
    ]
    for op, call in zip(("0001", "0002"), rsync.calls):
        assert call.listed == [str(local) for local in files if f"/{op}/" in str(local)]
        for option in ("--relative", "--no-implied-dirs", "--files-from", "--from0"):
            assert not any(arg.startswith(option) for arg in call.argv)
        assert "--partial" in call.argv and "--timeout=120" in call.argv
        assert call.argv[1] == "-e" and "ssh" in call.argv[2]


def test_put_files_groups_by_remote_directory(rsync, tmp_path):
    site = BatchSite()
    project = tmp_path / "project"
    ops = _ops_files(project, directions=1, tasks=2)
    staged = _write(project / "jobs/sim/normal/_fs_run/remote/BatchSite/sim.json")
    external = _write(tmp_path / "outside/ops/direction_1.h5")
    external_remote = (
        WORK / "jobs/sim/normal/inputs/external" / external.as_posix().lstrip("/")
    )
    renamed = _write(tmp_path / "sweep.tmp")
    pairs = [
        *[(local, _remote(project, local)) for local in ops],
        (staged, WORK / "simulations/sim/sim.json"),
        (external, external_remote),
        (renamed, WORK / "jobs/sim/normal/logs/batch/sweep-1.slurm"),
    ]

    assert site.put_files(pairs) == [remote for _, remote in pairs]

    assert len(site.commands) == 1
    assert set(_mkdir_targets(site.commands[0])) == {
        f"{WORK}/imaging/fwi/abc123/ops/0001",
        f"{WORK}/simulations/sim",
        str(external_remote.parent),
        f"{WORK}/jobs/sim/normal/logs/batch",
    }
    targets = {call.argv[-1]: call.listed for call in rsync.calls}
    assert targets == {
        f"user@{HOST}:{WORK}/imaging/fwi/abc123/ops/0001/": [str(local) for local in ops],
        f"user@{HOST}:{WORK}/simulations/sim/": [str(staged)],
        f"user@{HOST}:{external_remote.parent}/": [str(external)],
        # Stored under another name: a single-file rsync to the exact path.
        f"user@{HOST}:{WORK}/jobs/sim/normal/logs/batch/sweep-1.slurm": [str(renamed)],
    }


def test_put_files_sftp_publishes_every_file_over_one_session(monkeypatch, tmp_path):
    monkeypatch.setattr(
        transfer_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("SFTP uploads must not start rsync"),
    )
    sftp = RecordingSFTP()
    site = BatchSite(method="sftp", sftp=sftp)
    project = tmp_path / "project"
    files = _ops_files(project, directions=3, tasks=2)
    pairs = [(local, _remote(project, local)) for local in files]

    site.put_files(pairs)

    assert site.opened == 1
    assert sftp.published == [str(remote) for _, remote in pairs]
    assert len(site.commands) == 1
    assert _mkdir_targets(site.commands[0]) == [f"{WORK}/imaging/fwi/abc123/ops/0001"]


def test_rsync_batch_retries_a_stall_with_the_same_file_list(rsync, sleeps, tmp_path):
    site = BatchSite()
    project = tmp_path / "project"
    files = _ops_files(project, directions=2, tasks=2)
    rsync.statuses = [30, 0]  # rsync I/O timeout, then success

    site.put_files([(local, _remote(project, local)) for local in files])

    first, second = rsync.calls
    assert first.argv == second.argv and first.listed == second.listed
    assert len(first.listed) == 4
    assert sleeps == [2.0]
    assert len(site.commands) == 1  # the directories are not created again


def test_sftp_batch_retry_resumes_with_unpublished_files(sleeps, tmp_path):
    sftp = RecordingSFTP(fail_on=2)
    site = BatchSite(method="sftp", sftp=sftp)
    project = tmp_path / "project"
    files = _ops_files(project, directions=2, tasks=2)
    pairs = [(local, _remote(project, local)) for local in files]

    site.put_files(pairs)

    assert sftp.published == [str(remote) for _, remote in pairs]
    assert len(sftp.puts) == len(files) + 1  # only the stalled file is resent
    assert len(sftp.removed) == 1  # its partial upload
    assert site.opened == 2 and sleeps == [2.0]


def test_put_files_validates_every_pair_before_any_transfer(
    monkeypatch, rsync, tmp_path
):
    monkeypatch.setattr(transfer_module, "_wait_for_path", lambda p: Path(p).exists())
    site = BatchSite()
    present = _write(tmp_path / "project/a.h5")

    with pytest.raises(ValueError, match="traversal-free"):
        site.put_files([(present, WORK / "a.h5"), (present, f"{WORK}/../b.h5")])
    with pytest.raises(FileNotFoundError):
        site.put_files([(present, WORK / "a.h5"), (tmp_path / "missing.h5", WORK)])

    assert site.commands == [] and rsync.calls == []


def test_fully_staged_batch_runs_no_remote_command(rsync, tmp_path):
    site = BatchSite()
    project = tmp_path / "project"
    pairs = [(local, _remote(project, local)) for local in _ops_files(project)]

    site.put_files(pairs)
    commands, calls = len(site.commands), len(rsync.calls)

    assert site.put_files(pairs) == []
    assert (len(site.commands), len(rsync.calls)) == (commands, calls)


def test_second_pass_skips_staged_files_and_resends_changed_content(rsync, tmp_path):
    site = BatchSite()
    job = _gulf_job(tmp_path)

    # Pass 1: a caller stages the operation inputs before submitting.
    Backend(site, job.project_path / "imaging")._stage_remote_inputs(job)
    assert len(site.commands) == 1 and len(rsync.calls) == 1
    assert len(rsync.calls[0].listed) == 60

    # Same size and timestamps, different bytes: only content says it changed.
    changed = job.inputs[5]
    before = changed.stat()
    changed.write_bytes(bytes(reversed(changed.read_bytes())))
    os.utime(changed, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert changed.stat().st_size == before.st_size

    calls = len(rsync.calls)
    site._transfer_remote_simulation_inputs(job)

    assert sorted(rsync.sent(calls)) == sorted(["sim.json", changed.name])
    assert len(site.commands) == 2  # one mkdir for the second pass

    # The job may rewrite site files once it runs: the round is over.
    calls = len(rsync.calls)
    site._transfer_remote_simulation_inputs(job)
    assert len(rsync.sent(calls)) == 61


def test_upload_to_a_directory_drops_staged_records_below_it(rsync, tmp_path):
    site = BatchSite()
    project = tmp_path / "project"
    pairs = [(local, _remote(project, local)) for local in _ops_files(project)]
    site.put_files(pairs)

    site.put(project / "imaging", WORK / "imaging")

    calls = len(rsync.calls)
    assert len(site.put_files(pairs)) == len(pairs)
    assert len(rsync.sent(calls)) == len(pairs)


def test_old_staged_records_are_sent_again(monkeypatch, rsync, tmp_path):
    now = [1000.0]
    monkeypatch.setattr(transfer_module, "_monotonic", lambda: now[0])
    site = BatchSite()
    project = tmp_path / "project"
    pairs = [(local, _remote(project, local)) for local in _ops_files(project)]
    site.put_files(pairs)

    now[0] += transfer_module._STAGED_RECORD_LIFETIME / 2
    assert site.put_files(pairs) == []
    now[0] += transfer_module._STAGED_RECORD_LIFETIME
    assert len(site.put_files(pairs)) == len(pairs)


def test_job_inputs_travel_in_one_batch_that_ends_the_round(tmp_path):
    calls = []
    simulation = tmp_path / "staged/sim.json"
    inputs = [
        (tmp_path / f"d_{task}.h5", Path(WORK) / f"d_{task}.h5") for task in (1, 2)
    ]

    class Site:
        work_dir = Path(WORK)
        _transfer_remote_simulation_inputs = (
            SlurmSite._transfer_remote_simulation_inputs
        )

        def _claim_remote_project(self, path):
            calls.append(("claim", path))

        def put(self, *args, **kwargs):
            pytest.fail("job inputs must not be uploaded one at a time")

        def put_files(self, pairs, **kwargs):
            calls.append(("put_files", list(pairs), kwargs))

    job = SimpleNamespace(
        project_path=tmp_path,
        save_simulation_for_remote=lambda site, root: (
            simulation,
            Path(root) / "simulations/sim/sim.json",
        ),
        remote_input_files=lambda root: inputs,
    )

    Site()._transfer_remote_simulation_inputs(job)

    assert calls == [
        ("claim", tmp_path),
        (
            "put_files",
            [(simulation, Path(WORK) / "simulations/sim/sim.json"), *inputs],
            {"before_launch": True},
        ),
    ]


def test_allocation_session_inputs_travel_once_per_submit(rsync, tmp_path):
    site = BatchSite()
    job = _gulf_job(tmp_path)
    session = AllocationSession(site)

    # The imaging backend leaves job inputs to the session's submit ...
    Backend(session, job.project_path / "imaging")._stage_remote_inputs(job)
    assert site.commands == [] and rsync.calls == []

    # ... which sends them with the simulation JSON in one batch.
    site._transfer_remote_simulation_inputs(job)
    assert len(site.commands) == 1
    assert len(rsync.calls) == 2  # project tree, staged simulation JSON
    assert sorted(rsync.sent()) == sorted(
        ["sim.json", *(path.name for path in job.inputs)]
    )
    assert site.claimed == [job.project_path]
