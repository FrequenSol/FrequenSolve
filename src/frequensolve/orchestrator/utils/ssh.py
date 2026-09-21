"""
SSH manager for SLURM/HPC sites that uses a master socket to
avoid re-authenticating each time a connection is made.
"""

import shlex
import subprocess
import time
from typing import Any, Protocol, cast

from frequensolve._optional import optional_dependency_error
from frequensolve.util.setup_logger import init_logger

try:
    from paramiko import SSHClient
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "SSH-backed HPC support",
        extra="hpc",
        dependencies=("paramiko",),
        error=exc,
    ) from exc

__all__ = [
    "SSHProxy",
    "SSHClientClass",
    "SSH_COMMAND_TIMEOUT_SECONDS",
    "SSH_CONNECT_TIMEOUT_SECONDS",
    "control_socket_ssh_options",
]

SSH_CONNECT_TIMEOUT_SECONDS = 15
SSH_COMMAND_TIMEOUT_SECONDS = 120
SSH_SERVER_ALIVE_INTERVAL_SECONDS = 15
SSH_SERVER_ALIVE_COUNT_MAX = 2

logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/hpc.log")


class _CommandResult(Protocol):
    @property
    def stdout(self) -> bytes | str: ...

    @property
    def stderr(self) -> bytes | str: ...

    @property
    def returncode(self) -> int: ...


class ReadableByteStream(Protocol):
    """Minimal common stdout/stderr contract for Paramiko and SSHProxy."""

    def read(self) -> bytes: ...


class SSHTransport(Protocol):
    """Paramiko transport behavior used to open a compute-node tunnel."""

    def open_channel(
        self,
        kind: str,
        dest_addr: tuple[str, int],
        src_addr: tuple[str, int],
    ) -> object: ...


SSHCommandResult = tuple[object, ReadableByteStream, ReadableByteStream]


def control_socket_ssh_options(control_path: str) -> list[str]:
    """Return non-interactive OpenSSH options for a verified control socket."""

    return [
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ControlPath={control_path}",
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
        "-o",
        f"ServerAliveInterval={SSH_SERVER_ALIVE_INTERVAL_SECONDS}",
        "-o",
        f"ServerAliveCountMax={SSH_SERVER_ALIVE_COUNT_MAX}",
    ]


class BytesIO:
    """Small byte-stream adapter matching Paramiko stdout/stderr objects.

    Args:
        initial_bytes: Initial byte payload.
    """

    def __init__(self, initial_bytes: bytes, *, channel: Any = None):
        self._bytes = initial_bytes
        self._pos = 0
        self.channel = channel

    def read(self) -> bytes:
        """Return all remaining bytes and advance to end-of-stream."""

        remaining = self._bytes[self._pos :]
        self._pos = len(self._bytes)
        return remaining

    def readline(self) -> bytes:
        """Return the next line as bytes, or ``b""`` at end-of-stream."""

        if self._pos >= len(self._bytes):
            return b""
        newline_pos = self._bytes.find(b"\n", self._pos)
        if newline_pos == -1:
            line = self._bytes[self._pos :]
            self._pos = len(self._bytes)
        else:
            line = self._bytes[self._pos : newline_pos + 1]
            self._pos = newline_pos + 1
        return line

    def decode(self) -> str:
        """Decode the underlying bytes payload as text."""

        return self._bytes.decode()


class _ExitStatusChannel:
    """Expose a subprocess return code through Paramiko's channel contract."""

    def __init__(self, returncode: int) -> None:
        self._returncode = returncode

    def recv_exit_status(self) -> int:
        """Return the completed SSH subprocess status."""

        return self._returncode


class SSHProxy:
    """SSH control-socket proxy with a Paramiko-like command interface.

    Args:
        control_path: Path to an existing SSH control socket.
        username: SSH username.
        host: Login host.
        compute_host: Optional compute host reached through the login host.
    """

    def __init__(
        self,
        control_path: str,
        username: str,
        host: str,
        compute_host: str | None = None,
        command_timeout: float = SSH_COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        self.control_path = control_path
        self.username = username
        self.host = host
        self.compute_host = compute_host
        self.command_timeout = command_timeout

    def _login_ssh_command(self) -> list[str]:
        return [
            "ssh",
            *control_socket_ssh_options(self.control_path),
            f"{self.username}@{self.host}",
        ]

    def invoke_shell(self) -> subprocess.Popen[bytes]:
        """Create an interactive shell using subprocess.Popen."""
        import os
        import pty

        master, slave = pty.openpty()

        cmd = [*self._login_ssh_command()[:-1], "-q", self._login_ssh_command()[-1]]

        process = subprocess.Popen(
            cmd,
            stdin=slave,
            stdout=slave,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

        os.close(slave)
        master_file = os.fdopen(master, "rb+", buffering=0)

        if self.compute_host is not None:
            master_file.write(
                (
                    "ssh -o StrictHostKeyChecking=yes -o BatchMode=yes "
                    f"{shlex.quote(str(self.compute_host))}\n"
                ).encode()
            )
            master_file.flush()
            time.sleep(1)

        master_file.flush()
        process.stdin = process.stdout = master_file

        return process

    def _filter_output(self, result: _CommandResult) -> tuple[bytes, bytes]:
        """Filter unwanted messages from command output."""
        stdout = result.stdout
        stderr = result.stderr

        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        if isinstance(stderr, bytes):
            stderr = stderr.decode()

        # Filter out mount messages and empty lines
        stdout_lines = []
        stderr_lines = []
        for line in stdout.splitlines():
            line = line.strip()
            if len(line) > 0 and not line.endswith('is mounted not "FULL" nor "IDLE"!'):
                stdout_lines.append(line)
        for line in stderr.splitlines():
            line = line.strip()
            if len(line) > 0 and not line.endswith('is mounted not "FULL" nor "IDLE"!'):
                stderr_lines.append(line)

        return ("\n".join(stdout_lines).encode(), "\n".join(stderr_lines).encode())

    def exec_command(
        self, command: str, *, timeout: float | None = None
    ) -> SSHCommandResult:
        """Execute a command through the proxy.

        Args:
            command: Remote shell command.
            timeout: Optional per-command timeout in seconds. Defaults to this
                proxy's configured command timeout.

        Returns:
            ``(stdin, stdout, stderr)`` byte streams compatible with Paramiko's
            ``exec_command`` return shape.
        """

        if self.compute_host is not None:
            result = self._exec_on_compute(command, timeout=timeout)
        else:
            result = self._exec_on_login(command, timeout=timeout)

        stdout_bytes, stderr_bytes = self._filter_output(result)
        channel = _ExitStatusChannel(result.returncode)

        return (
            BytesIO(b"", channel=channel),
            BytesIO(stdout_bytes, channel=channel),
            BytesIO(stderr_bytes, channel=channel),
        )

    def exec_command_term(self, command: str) -> SSHCommandResult:
        """Execute a login-node command with a pseudo-terminal.

        Args:
            command: Remote shell command.

        Returns:
            ``(stdin, stdout, stderr)`` byte streams.
        """

        result = self._exec_on_login(command, term=True)
        stdout_bytes, stderr_bytes = self._filter_output(result)
        channel = _ExitStatusChannel(result.returncode)

        return (
            BytesIO(b"", channel=channel),
            BytesIO(stdout_bytes, channel=channel),
            BytesIO(stderr_bytes, channel=channel),
        )

    def _exec_on_login(
        self,
        command: str,
        term: bool = False,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        cmd = self._login_ssh_command()
        if term:
            cmd.insert(-1, "-t")
        cmd.append(command)
        return self._run_ssh(cmd, target=self.host, timeout=timeout)

    def _exec_on_compute(
        self, command: str, timeout: float | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        """Execute a command on a compute node via the login node."""
        if self.compute_host is None:
            raise RuntimeError("Compute host is not configured")
        proxy_command = shlex.join(
            [
                *self._login_ssh_command()[:-1],
                "-W",
                "%h:%p",
                self._login_ssh_command()[-1],
            ]
        )
        cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
            "-o",
            f"ServerAliveInterval={SSH_SERVER_ALIVE_INTERVAL_SECONDS}",
            "-o",
            f"ServerAliveCountMax={SSH_SERVER_ALIVE_COUNT_MAX}",
            "-o",
            f"ProxyCommand={proxy_command}",
            f"{self.username}@{self.compute_host}",
            command,
        ]
        return self._run_ssh(cmd, target=self.compute_host, timeout=timeout)

    def _run_ssh(
        self,
        cmd: list[str],
        *,
        target: str,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        effective_timeout = self.command_timeout if timeout is None else timeout
        logger.debug(
            "Running non-interactive SSH command via control socket: target=%s "
            "timeout=%ss command=%s",
            target,
            effective_timeout,
            cmd[-1],
        )
        started = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "SSH command timed out after %.1fs for target %s",
                time.monotonic() - started,
                target,
            )
            raise TimeoutError(
                f"SSH command to {target} timed out after {effective_timeout} seconds"
            ) from None
        except OSError as exc:
            logger.error("SSH command could not start (%s)", type(exc).__name__)
            raise RuntimeError(
                f"SSH command could not start ({type(exc).__name__})"
            ) from None
        logger.debug(
            "SSH command finished in %.3fs for target %s with return code %s",
            time.monotonic() - started,
            target,
            result.returncode,
        )
        if result.returncode == 255:
            raise RuntimeError(f"SSH connection to {target} failed with status 255")
        return result

    def close(self) -> None:
        """Close the proxy.

        Existing SSH control sockets are owned by the user's SSH process, so
        this method intentionally performs no action.
        """

        pass  # Nothing to close - using existing socket

    def get_transport(self) -> None:
        """Return ``None`` because subprocess SSH proxies expose no transport."""

        return None  # No transport for proxy

    def open_sftp(self) -> "_OpenSSHControlSFTP":
        """Open an SFTP adapter that reuses this verified control socket."""

        return _OpenSSHControlSFTP(
            control_path=self.control_path,
            username=self.username,
            host=self.host,
            command_timeout=self.command_timeout,
        )


class _OpenSSHControlSFTP:
    """Small Paramiko-compatible SFTP surface backed by OpenSSH batch mode."""

    def __init__(
        self,
        *,
        control_path: str,
        username: str,
        host: str,
        command_timeout: float,
    ) -> None:
        self.control_path = control_path
        self.username = username
        self.host = host
        self.command_timeout = command_timeout

    @staticmethod
    def _quote_path(value: str) -> str:
        raw = str(value)
        if not raw or any(character in raw for character in ("\x00", "\n", "\r")):
            raise ValueError("SFTP path contains an invalid character")
        return shlex.quote(raw)

    def _run_batch(self, command: str) -> None:
        result = subprocess.run(
            [
                "sftp",
                "-q",
                "-b",
                "-",
                *control_socket_ssh_options(self.control_path),
                f"{self.username}@{self.host}",
            ],
            input=f"{command}\n",
            capture_output=True,
            text=True,
            timeout=self.command_timeout,
        )
        if result.returncode != 0:
            raise OSError("SFTP operation failed")

    def _run_ssh(self, command: str) -> str:
        result = subprocess.run(
            [
                "ssh",
                *control_socket_ssh_options(self.control_path),
                f"{self.username}@{self.host}",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=self.command_timeout,
        )
        if result.returncode != 0:
            raise OSError("SFTP metadata operation failed")
        return result.stdout

    def stat(self, path: str) -> Any:
        """Return the remote mode and size needed by the transfer manager."""

        output = self._run_ssh("stat -Lc '%f %s' -- " + self._quote_path(path)).strip()
        try:
            encoded_mode, encoded_size = output.split()
            mode = int(encoded_mode, 16)
            size = int(encoded_size)
        except (TypeError, ValueError):
            raise OSError("SFTP metadata response was invalid") from None
        return type("SFTPAttributes", (), {"st_mode": mode, "st_size": size})()

    def put(self, local_path: str, remote_path: str) -> None:
        """Upload one file through OpenSSH SFTP."""

        self._run_batch(
            f"put {self._quote_path(local_path)} {self._quote_path(remote_path)}"
        )

    def get(self, remote_path: str, local_path: str) -> None:
        """Download one file through OpenSSH SFTP."""

        self._run_batch(
            f"get {self._quote_path(remote_path)} {self._quote_path(local_path)}"
        )

    def chmod(self, path: str, mode: int) -> None:
        """Set one remote file mode through OpenSSH SFTP."""

        self._run_batch(f"chmod {mode:04o} {self._quote_path(path)}")

    def posix_rename(self, old_path: str, new_path: str) -> None:
        """Atomically rename one remote file when the server supports it."""

        self._run_batch(
            f"rename {self._quote_path(old_path)} {self._quote_path(new_path)}"
        )

    def rename(self, old_path: str, new_path: str) -> None:
        """Rename one remote file."""

        self.posix_rename(old_path, new_path)

    def remove(self, path: str) -> None:
        """Remove one remote file."""

        self._run_batch(f"rm {self._quote_path(path)}")

    def close(self) -> None:
        """Close the stateless adapter."""

        return None


class SSHClientClass:
    """Wrapper normalizing Paramiko and ``SSHProxy`` clients.

    Args:
        client: Paramiko ``SSHClient`` or ``SSHProxy`` instance.

    Raises:
        ValueError: If ``client`` is not a supported SSH client type.
    """

    def __init__(self, client: SSHClient | SSHProxy) -> None:
        # Paramiko is an optional, incompletely typed boundary. Public wrapper
        # methods below expose the shared contract instead of leaking it.
        self.client: Any = client
        self._proxy: SSHProxy | None = None
        if isinstance(client, SSHClient):
            _, stdout, _ = self.client.exec_command("echo $HOSTNAME")
            out = stdout.read().decode().strip()
            self._hostname = out.split("@")[0] if "@" in out else out.split(".")[0]
            self._is_proxy = False
        elif isinstance(client, SSHProxy):
            _, stdout, _ = self.client.exec_command("echo $HOSTNAME")
            self._hostname = stdout.read().decode().strip().split(".")[0]
            self._is_proxy = True
            self._proxy = client
        else:
            raise ValueError(f"Unsupported client type: {type(client)}")

    @property
    def hostname(self) -> str:
        """Get the hostname."""
        return self._hostname

    def exec_command(
        self, command: str, *, timeout: float | None = None
    ) -> SSHCommandResult:
        """Execute a remote command.

        Args:
            command: Remote shell command.
            timeout: Optional per-command timeout in seconds.

        Returns:
            ``(stdin, stdout, stderr)`` as returned by the underlying client.
        """
        if timeout is not None:
            result = self.client.exec_command(command, timeout=timeout)
        else:
            result = self.client.exec_command(command)
        return cast(SSHCommandResult, result)

    def close(self) -> None:
        """Close the underlying SSH connection or proxy."""
        self.client.close()

    def get_transport(self) -> SSHTransport | None:
        """Get the Paramiko transport when available."""
        if isinstance(self.client, SSHClient):
            return cast(SSHTransport | None, self.client.get_transport())
        return None

    def is_proxy(self) -> bool:
        """Return whether the underlying client is an ``SSHProxy``."""
        return self._is_proxy

    def get_proxy_details(self) -> tuple[str | None, str | None]:
        """Return proxy control path and username when using ``SSHProxy``."""
        if self._proxy is not None:
            return self._proxy.control_path, self._proxy.username
        return None, None
