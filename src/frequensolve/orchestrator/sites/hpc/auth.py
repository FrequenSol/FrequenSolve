"""Authentication helpers for SLURM-backed HPC sites."""

from __future__ import annotations

import glob
import os
import re
import socket
import subprocess
import threading
from pathlib import Path
from typing import Optional

from frequensolve._optional import optional_dependency_error

try:
    from paramiko import (
        AuthenticationException,
        HostKeys,
        SSHClient,
        SSHException,
        Transport,
    )
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "SlurmSite",
        extra="hpc",
        dependencies=("paramiko",),
        error=exc,
    ) from exc

from frequensolve.orchestrator.sites.hpc.slurm_helpers import validate_slurm_job_id
from frequensolve.orchestrator.utils.ssh import (
    SSH_CONNECT_TIMEOUT_SECONDS,
    SSHProxy,
    control_socket_ssh_options,
)
from frequensolve.util.setup_logger import init_logger

logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/hpc.log")

_COMPUTE_HOST = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*|\[[0-9A-Fa-f:]+\])\Z")


def _verify_server_host_key(
    transport: Transport,
    host: str,
    *,
    known_hosts_file: Optional[Path] = None,
    known_hosts_name: Optional[str] = None,
) -> None:
    """Verify *host* against an explicit or standard known-hosts source."""

    known_hosts = HostKeys()
    paths = (
        (known_hosts_file,)
        if known_hosts_file is not None
        else (
            Path("~/.ssh/known_hosts").expanduser(),
            Path("/etc/ssh/ssh_known_hosts"),
        )
    )
    for path in paths:
        try:
            known_hosts.load(str(path))
        except OSError:
            continue

    server_key = transport.get_remote_server_key()
    lookup_name = known_hosts_name or host
    host_keys = known_hosts.lookup(lookup_name)
    expected_key = host_keys.get(server_key.get_name()) if host_keys else None
    if expected_key != server_key:
        raise SSHException(
            f"SSH host key for {lookup_name!r} is unknown or does not match "
            "the configured known-hosts source."
        )


class SlurmAuthenticator:
    """Owns SSH login-node and compute-node authentication flows.

    Args:
        site: ``SlurmSite`` instance that provides credentials and remote
            command helpers.
    """

    def __init__(self, site):
        self.site = site

    def authenticate(self, host: Optional[str] = None):
        """Connect to the login node using Paramiko or an SSH control socket.

        Args:
            host: Optional login host override.

        Returns:
            Paramiko ``SSHClient`` or ``SSHProxy`` using an existing control
            socket.

        Raises:
            RuntimeError: If called outside the main thread.
            ValueError: If no login host is configured.
        """

        site = self.site
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Authentication must be called from the main thread")
        host = host or getattr(site.config, "hostname", None) or site.default_host
        if not host:
            raise ValueError("No login host configured for SLURM site")

        logger.info("Starting authentication with host: %s", host)

        config = getattr(site, "config", None)
        ssh_port = getattr(config, "ssh_port", 22)
        known_hosts_file = getattr(config, "known_hosts_file", None)
        known_hosts_name = getattr(config, "known_hosts_name", None)
        allow_ssh_agent = getattr(config, "allow_ssh_agent", True)
        allow_keyboard_interactive = getattr(config, "allow_keyboard_interactive", True)
        control_dir = os.path.expanduser("~/.ssh/control")
        explicit_trust = known_hosts_file is not None or known_hosts_name is not None
        explicit_auth_policy = not allow_ssh_agent or not allow_keyboard_interactive
        if (
            ssh_port == 22
            and not explicit_trust
            and not explicit_auth_policy
            and os.path.exists(control_dir)
        ):
            for control_path in glob.glob(f"{control_dir}/*"):
                try:
                    result = subprocess.run(
                        [
                            "ssh",
                            "-q",
                            *control_socket_ssh_options(control_path),
                            f"{site.credentials.username}@{host}",
                            "echo 'Connection test'",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=SSH_CONNECT_TIMEOUT_SECONDS + 5,
                    )

                    if result.returncode == 0:
                        logger.debug("Found working control socket at %s", control_path)
                        proxy_client = SSHProxy(
                            control_path=control_path,
                            username=site.credentials.username,
                            host=host,
                        )
                        logger.info("Secure connection established with host: %s", host)
                        return proxy_client

                except subprocess.TimeoutExpired:
                    logger.debug(
                        "Timed out probing SSH control socket %s", control_path
                    )
                    continue
                except Exception as exc:
                    logger.debug(
                        "Failed to use control socket %s (%s)",
                        control_path,
                        type(exc).__name__,
                    )
                    continue
        return self._interactive_authentication(host)

    def get_job_host(self, job_id: int) -> str:
        """Return the host assigned to a running SLURM job.

        Args:
            job_id: SLURM job id.

        Returns:
            Compute-node host name.

        Raises:
            RuntimeError: If the job is not running or the host cannot be
                determined.
        """

        site = self.site
        validated_job_id = validate_slurm_job_id(job_id)
        status = site.run_login(f"squeue -j {validated_job_id} -h -o %t").strip()
        if status != "R":
            raise RuntimeError(f"Job {validated_job_id} is not running")

        hostname = site.run_login(f"squeue -j {validated_job_id} -h -o %B").strip()
        if not hostname:
            raise RuntimeError(f"Could not get hostname for job {validated_job_id}")
        if not _COMPUTE_HOST.fullmatch(hostname):
            raise RuntimeError("SLURM returned an invalid compute-node hostname")

        return hostname

    def connect_to_job_host(self, job_id: int):
        """Connect to the compute node assigned to a running SLURM job.

        Args:
            job_id: SLURM job id for a running allocation.

        Returns:
            Paramiko ``SSHClient`` or ``SSHProxy`` connected to the compute
            node.
        """

        site = self.site
        job_host = self.get_job_host(job_id)
        logger.debug("Got compute node hostname: %s", job_host)

        if site._login_client.is_proxy():
            logger.debug("Using proxy connection to connect to compute node")
            control_path, username = site._login_client.get_proxy_details()
            if not control_path or not username:
                raise RuntimeError("Missing proxy details")
            return SSHProxy(control_path, username, site.login_client.host, job_host)

        logger.debug("Using SSH tunneling to connect to compute node")
        transport = site._login_client.get_transport()
        if not transport:
            raise RuntimeError("No transport available for SSH tunneling")

        try:
            channel = transport.open_channel("direct-tcpip", (job_host, 22), ("", 0))
        except Exception as exc:
            raise RuntimeError(
                f"SSH compute-node tunnel failed ({type(exc).__name__})"
            ) from None
        job_client = SSHClient()

        try:
            job_client.load_system_host_keys()
            job_client.connect(
                job_host,
                username=site.credentials.username,
                sock=channel,
                allow_agent=True,
                look_for_keys=False,
                timeout=SSH_CONNECT_TIMEOUT_SECONDS,
                banner_timeout=SSH_CONNECT_TIMEOUT_SECONDS,
                auth_timeout=SSH_CONNECT_TIMEOUT_SECONDS,
            )
            logger.info("Connected to job host: %s", job_host)
            return job_client
        except Exception as exc:
            logger.error("Failed to connect to job host (%s)", type(exc).__name__)
            job_client.close()
            channel.close()
            raise RuntimeError(
                f"SSH compute-node connection failed ({type(exc).__name__})"
            ) from None

    def _interactive_authentication(self, host: str):
        """Normal authentication flow when a control socket is not available."""

        site = self.site
        login_client = SSHClient()
        config = getattr(site, "config", None)
        ssh_port = getattr(config, "ssh_port", 22)
        known_hosts_file = getattr(config, "known_hosts_file", None)
        known_hosts_name = getattr(config, "known_hosts_name", None)
        allow_ssh_agent = getattr(config, "allow_ssh_agent", True)
        allow_keyboard_interactive = getattr(config, "allow_keyboard_interactive", True)

        try:
            sock = socket.create_connection(
                (host, ssh_port), timeout=SSH_CONNECT_TIMEOUT_SECONDS
            )
        except (socket.timeout, TimeoutError):
            raise TimeoutError(
                "SSH login connection timed out after "
                f"{SSH_CONNECT_TIMEOUT_SECONDS} seconds"
            ) from None
        except OSError as exc:
            raise RuntimeError(
                f"SSH login connection failed ({type(exc).__name__})"
            ) from None
        try:
            transport = Transport(sock)
        except Exception as exc:
            sock.close()
            raise RuntimeError(
                f"SSH transport setup failed ({type(exc).__name__})"
            ) from None
        try:
            transport.start_client(timeout=SSH_CONNECT_TIMEOUT_SECONDS)
            _verify_server_host_key(
                transport,
                host,
                known_hosts_file=known_hosts_file,
                known_hosts_name=known_hosts_name,
            )
        except Exception:
            transport.close()
            raise

        authenticated = False
        if allow_ssh_agent:
            try:
                from paramiko.agent import Agent

                logger.debug("Attempting agent-based authentication.")
                agent = Agent()
                for key in agent.get_keys():
                    try:
                        transport.auth_publickey(site.credentials.username, key)
                        if transport.is_authenticated():
                            authenticated = True
                            break
                    except Exception as exc:
                        logger.debug(
                            "Agent key authentication failed (%s)",
                            type(exc).__name__,
                        )
                        continue
            except Exception as exc:
                logger.debug(
                    "Agent-based authentication exception (%s)", type(exc).__name__
                )

        if not authenticated:
            logger.debug("Attempting configured private-key authentication.")
            try:
                key = site.credentials.ssh_key
                transport.auth_publickey(site.credentials.username, key)
                authenticated = transport.is_authenticated()
            except (FileNotFoundError, OSError, SSHException) as exc:
                logger.debug(
                    "Private-key authentication unavailable (%s)",
                    type(exc).__name__,
                )

        if not authenticated and allow_keyboard_interactive:
            logger.debug("Attempting keyboard-interactive authentication.")

            def handler(title, instructions, prompt_list):
                responses = []
                for prompt, echo in prompt_list:
                    normalized_prompt = prompt.lower()
                    if "password" in normalized_prompt:
                        responses.append(site.credentials.password)
                    elif any(
                        marker in normalized_prompt
                        for marker in ("token", "2fa", "code", "passcode")
                    ):
                        responses.append(site.credentials.duo_code)
                    else:
                        responses.append("")
                return responses

            try:
                transport.auth_interactive(site.credentials.username, handler)
                authenticated = transport.is_authenticated()
                if authenticated:
                    logger.debug("Keyboard-interactive authentication successful.")
                else:
                    logger.debug("Keyboard-interactive authentication failed.")
            except Exception as exc:
                logger.debug(
                    "Keyboard-interactive authentication exception (%s)",
                    type(exc).__name__,
                )

        if not transport.is_authenticated():
            logger.error("Authentication failed for the configured SSH user")
            transport.close()
            raise AuthenticationException("Authentication failed.")

        transport.set_keepalive(120)
        login_client._transport = transport
        logger.info("Secure connection established with host: %s", host)
        return login_client
