"""Control-socket SFTP transfers fail when they stall, not when they are large."""

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from frequensolve.orchestrator.utils import ssh as ssh_module
from frequensolve.orchestrator.utils.ssh import SSHProxy

pytestmark = [pytest.mark.unit, pytest.mark.hpc_hermetic]


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _Transfer:
    """A transfer process that exits after ``polls`` waits of simulated time."""

    def __init__(self, clock, polls):
        self.clock = clock
        self.polls = polls
        self.returncode = None
        self.killed = False
        self.stdin = SimpleNamespace(write=lambda data: None, close=lambda: None)

    def wait(self, timeout=None):
        if self.killed or self.polls == 0:
            self.returncode = -9 if self.killed else 0
            return self.returncode
        self.polls -= 1
        self.clock.now += timeout
        raise subprocess.TimeoutExpired("sftp", timeout)

    def kill(self):
        self.killed = True


def _monitor(monkeypatch, process, progress):
    monkeypatch.setattr(ssh_module.subprocess, "Popen", lambda argv, **_: process)
    return ssh_module._run_with_progress(
        ["sftp"],
        "put a b\n",
        progress,
        idle_timeout=120.0,
        poll_interval=5.0,
        clock=process.clock,
    )


def test_progressing_transfer_runs_far_beyond_the_idle_limit(monkeypatch):
    clock = _Clock()
    process = _Transfer(clock, polls=10_000)  # About 14 hours of transfer.
    sizes = iter(range(1, 10**9))

    assert _monitor(monkeypatch, process, lambda: next(sizes)) == (0, "")
    assert clock.now == pytest.approx(50_000.0)
    assert not process.killed


@pytest.mark.parametrize("stalled", [42, None])
def test_stalled_transfer_is_killed_after_the_idle_limit(monkeypatch, stalled):
    clock = _Clock()
    process = _Transfer(clock, polls=10_000)
    sizes = iter([0, 10, 20, *([stalled] * 10_000)])

    with pytest.raises(TimeoutError, match="no progress for 120 seconds"):
        _monitor(monkeypatch, process, lambda: next(sizes))

    assert process.killed
    # The size last changed at the fourth poll (t = 20 s).
    assert clock.now == pytest.approx(140.0)


# -- real processes ------------------------------------------------------------

_FAKE_SFTP = """\
#!{python}
import os, shlex, sys, time
verb, source, target = shlex.split(sys.stdin.read())
if float(os.environ.get("FAKE_SFTP_STALL", "0")):
    time.sleep(float(os.environ["FAKE_SFTP_STALL"]))
data = open(source, "rb").read()
with open(target, "wb") as out:
    for start in range(0, len(data), 64):
        out.write(data[start:start + 64])
        out.flush()
        time.sleep(float(os.environ.get("FAKE_SFTP_DELAY", "0")))
"""

_FAKE_SSH = """\
#!{python}
import os, shlex, sys
words = shlex.split(sys.argv[-1])
assert words[:3] == ["stat", "-Lc", "%s"], words
try:
    print(os.stat(words[-1]).st_size)
except OSError:
    sys.exit(1)
"""


@pytest.fixture
def openssh(tmp_path, monkeypatch):
    """Put fake ``sftp``/``ssh`` executables on PATH; the remote is local."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("sftp", _FAKE_SFTP), ("ssh", _FAKE_SSH)):
        path = bin_dir / name
        path.write_text(body.format(python=sys.executable))
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    source = tmp_path / "source.bin"
    source.write_bytes(os.urandom(64 * 30))
    return SSHProxy(str(tmp_path / "control"), "user", "login", command_timeout=2.0)


@pytest.mark.parametrize("direction", ["put", "get"])
def test_slow_transfers_that_keep_moving_complete(
    openssh, tmp_path, monkeypatch, direction
):
    monkeypatch.setenv("FAKE_SFTP_DELAY", "0.1")  # About 3 s against a 2 s limit.
    target = tmp_path / "target.bin"

    started = time.monotonic()
    getattr(openssh.open_sftp(), direction)(str(tmp_path / "source.bin"), str(target))

    assert time.monotonic() - started > 2.0
    assert target.read_bytes() == (tmp_path / "source.bin").read_bytes()


@pytest.mark.parametrize("direction", ["put", "get"])
def test_stuck_transfers_fail_instead_of_hanging(
    openssh, tmp_path, monkeypatch, direction
):
    monkeypatch.setenv("FAKE_SFTP_STALL", "60")

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="SFTP transfer made no progress"):
        getattr(openssh.open_sftp(), direction)(
            str(tmp_path / "source.bin"), str(tmp_path / "target.bin")
        )

    assert time.monotonic() - started < 20.0
