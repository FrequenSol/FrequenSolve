"""Authentication helpers for SLURM-backed HPC sites."""

from __future__ import annotations

import glob
import os
import socket
import subprocess
import threading
from typing import Optional

from frequensolve._optional import optional_dependency_error

try:
    from paramiko import AuthenticationException, AutoAddPolicy, SSHClient, Transport
except ModuleNotFoundError as exc:
    raise optional_dependency_error(
        "SlurmSite",
        extra="hpc",
        dependencies=("paramiko", "python-dotenv"),
        error=exc,
    ) from exc

from frequensolve.orchestrator.ssh import SSHProxy
from frequensolve.util.setup_logger import init_logger

logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/hpc.log")


class SlurmAuthenticator:
    """Owns SSH login-node and compute-node authentication flows."""

    def __init__(self, site):
        self.site = site

    def authenticate(self, host: Optional[str] = None):
        """Connect to the login node using Paramiko or an SSH control socket."""

        site = self.site
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Authentication must be called from the main thread")
        host = host or getattr(site.config, "hostname", None) or site.default_host
        if not host:
            raise ValueError("No login host configured for SLURM site")

        logger.info("Starting authentication with host: %s", host)

        control_dir = os.path.expanduser("~/.ssh/control")
        if os.path.exists(control_dir):
            for control_path in glob.glob(f"{control_dir}/*"):
                try:
                    result = subprocess.run(
                        [
                            "ssh",
                            "-q",
                            "-o",
                            "StrictHostKeyChecking=no",
                            "-o",
                            f"ControlPath={control_path}",
                            f"{site.credentials.username}@{host}",
                            "echo 'Connection test'",
                        ],
                        capture_output=True,
                        text=True,
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

                except Exception as exc:
                    logger.debug(
                        "Failed to use control socket %s: %s", control_path, exc
                    )
                    continue
        return self._interactive_authentication(host)

    def get_job_host(self, job_id: int) -> str:
        """Return the host assigned to a running SLURM job."""

        site = self.site
        status = site.run_login(f"squeue -j {job_id} -h -o %t").strip()
        if status != "R":
            raise RuntimeError(f"Job {job_id} is not running")

        hostname = site.run_login(f"squeue -j {job_id} -h -o %B").strip()
        if not hostname:
            raise RuntimeError(f"Could not get hostname for job {job_id}")

        return hostname

    def connect_to_job_host(self, job_id: int):
        """Connect to the compute node assigned to a running SLURM job."""

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

        channel = transport.open_channel("direct-tcpip", (job_host, 22), ("", 0))
        job_client = SSHClient()
        job_client.set_missing_host_key_policy(AutoAddPolicy())

        try:
            job_client.connect(
                job_host,
                username=site.credentials.username,
                sock=channel,
                allow_agent=True,
                look_for_keys=False,
            )
            logger.info("Connected to job host: %s", job_host)
            return job_client
        except Exception:
            logger.exception("Failed to connect to job host: %s", job_host)
            channel.close()
            raise

    def _interactive_authentication(self, host: str):
        """Normal authentication flow when a control socket is not available."""

        site = self.site
        login_client = SSHClient()
        login_client.set_missing_host_key_policy(AutoAddPolicy())

        sock = socket.create_connection((host, 22))
        transport = Transport(sock)
        transport.start_client()

        authenticated = False
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
                    logger.debug("Agent key authentication failed: %s", exc)
                    continue
        except Exception as exc:
            logger.debug("Agent-based authentication exception: %s", exc)

        if not authenticated:
            logger.debug("Attempting keyboard-interactive authentication.")

            def handler(title, instructions, prompt_list):
                responses = []
                for prompt, echo in prompt_list:
                    if "Password" in prompt:
                        responses.append(site.credentials.password)
                    elif "Token" in prompt or "2FA" in prompt or "Code" in prompt:
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
                logger.debug("Keyboard-interactive authentication exception: %s", exc)

        if not transport.is_authenticated():
            logger.error(
                "Authentication failed for user: %s", site.credentials.username
            )
            raise AuthenticationException("Authentication failed.")

        transport.set_keepalive(120)
        login_client._transport = transport
        logger.info("Secure connection established with host: %s", host)
        return login_client
