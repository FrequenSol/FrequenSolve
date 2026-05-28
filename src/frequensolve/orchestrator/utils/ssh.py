"""
SSH manager for SLURM/HPC sites that uses a master socket to
avoid re-authenticating each time a connection is made.
"""

import subprocess
import time

from frequensolve._optional import optional_dependency_error

try:
    from paramiko import SSHClient
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "SSH-backed HPC support",
        extra="hpc",
        dependencies=("paramiko",),
        error=exc,
    ) from exc

__all__ = ["SSHProxy", "SSHClientClass"]


class BytesIO:
    """Small byte-stream adapter matching Paramiko stdout/stderr objects.

    Args:
        initial_bytes: Initial byte or string payload.
    """

    def __init__(self, initial_bytes):
        self._bytes = initial_bytes
        self._pos = 0

    def read(self):
        """Return all remaining bytes and advance to end-of-stream."""

        remaining = self._bytes[self._pos :]
        self._pos = len(self._bytes)
        if isinstance(remaining, str):
            return remaining.encode()
        return remaining

    def readline(self):
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

    def decode(self):
        """Decode the underlying bytes payload as text."""

        return self._bytes.decode()


class SSHProxy:
    """SSH control-socket proxy with a Paramiko-like command interface.

    Args:
        control_path: Path to an existing SSH control socket.
        username: SSH username.
        host: Login host.
        compute_host: Optional compute host reached through the login host.
    """

    def __init__(self, control_path, username, host, compute_host=None):
        self.control_path = control_path
        self.username = username
        self.host = host
        self.compute_host = compute_host

    def invoke_shell(self):
        """Create an interactive shell using subprocess.Popen."""
        import os
        import pty

        master, slave = pty.openpty()

        cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-q",  # Add quiet flag
            f"{self.username}@{self.host}",
        ]

        process = subprocess.Popen(
            cmd,
            stdin=slave,
            stdout=slave,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            universal_newlines=True,
        )

        os.close(slave)
        master_file = os.fdopen(master, "rb+", buffering=0)

        if self.compute_host is not None:
            master_file.write(f"ssh {self.compute_host}\n".encode())
            master_file.flush()
            time.sleep(1)

        master_file.flush()
        process.stdin = process.stdout = master_file

        return process

    def _filter_output(self, result):
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

    def exec_command(self, command):
        """Execute a command through the proxy.

        Args:
            command: Remote shell command.

        Returns:
            ``(stdin, stdout, stderr)`` byte streams compatible with Paramiko's
            ``exec_command`` return shape.
        """

        if self.compute_host is not None:
            result = self._exec_on_compute(command)
        else:
            result = self._exec_on_login(command)

        stdout, stderr = self._filter_output(result)

        stdin = BytesIO(b"")
        stdout = BytesIO(stdout)
        stderr = BytesIO(stderr)

        return (stdin, stdout, stderr)

    def exec_command_term(self, command):
        """Execute a login-node command with a pseudo-terminal.

        Args:
            command: Remote shell command.

        Returns:
            ``(stdin, stdout, stderr)`` byte streams.
        """

        result = self._exec_on_login(command, term=True)
        stdout, stderr = self._filter_output(result)

        stdin = BytesIO(b"")
        stdout = BytesIO(stdout)
        stderr = BytesIO(stderr)

        return (stdin, stdout, stderr)

    def _exec_on_login(self, command, term=False):
        cmd = ["ssh", "-o", "StrictHostKeyChecking=no"]
        if term:
            cmd.append("-t")
        cmd.extend(
            [
                f"{self.username}@{self.host}",
                command,
            ]
        )
        return subprocess.run(cmd, capture_output=True)

    def _exec_on_compute(self, command):
        """Execute a command on a compute node via the login node."""
        cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-J",
            f"{self.username}@{self.host}",
            f"{self.username}@{self.compute_host}",
            command,
        ]
        return subprocess.run(cmd, capture_output=True)

    def close(self):
        """Close the proxy.

        Existing SSH control sockets are owned by the user's SSH process, so
        this method intentionally performs no action.
        """

        pass  # Nothing to close - using existing socket

    def get_transport(self):
        """Return ``None`` because subprocess SSH proxies expose no transport."""

        return None  # No transport for proxy


class SSHClientClass:
    """Wrapper normalizing Paramiko and ``SSHProxy`` clients.

    Args:
        client: Paramiko ``SSHClient`` or ``SSHProxy`` instance.

    Raises:
        ValueError: If ``client`` is not a supported SSH client type.
    """

    def __init__(self, client):
        self.client = client
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

    def exec_command(self, command):
        """Execute a remote command.

        Args:
            command: Remote shell command.

        Returns:
            ``(stdin, stdout, stderr)`` as returned by the underlying client.
        """
        return self.client.exec_command(command)

    def close(self):
        """Close the underlying SSH connection or proxy."""
        self.client.close()

    def get_transport(self):
        """Get the Paramiko transport when available."""
        if isinstance(self.client, SSHClient):
            return self.client.get_transport()
        return None

    def is_proxy(self):
        """Return whether the underlying client is an ``SSHProxy``."""
        return self._is_proxy

    def get_proxy_details(self):
        """Return proxy control path and username when using ``SSHProxy``."""
        if self.is_proxy():
            return self._proxy.control_path, self._proxy.username
        return None, None
