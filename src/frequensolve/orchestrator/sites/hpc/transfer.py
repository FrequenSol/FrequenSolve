"""File transfer helpers for SLURM-backed HPC sites."""

from __future__ import annotations

import logging
import os
import shlex
import stat
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Union

from frequensolve.orchestrator.sites.base import _wait_for_path
from frequensolve.orchestrator.utils.ssh import control_socket_ssh_options
from frequensolve.util.setup_logger import init_logger

logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/hpc.log")


class SlurmTransferManager:
    """Owns remote file transfer mechanics for a SLURM site.

    Args:
        site: ``SlurmSite`` instance that provides SSH clients and transfer
            settings.
    """

    def __init__(self, site):
        self.site = site

    def put(self, local_path: Union[str, Path], remote_path: Union[str, Path]):
        """Transfer a local file or directory to a remote path.

        Args:
            local_path: Local source file or directory.
            remote_path: Remote destination path.

        Raises:
            FileNotFoundError: If ``local_path`` does not exist.
            RuntimeError: If rsync fails.
        """

        site = self.site
        logger.debug("Transferring %s to %s", local_path, remote_path)
        if not _wait_for_path(local_path):
            logger.error("Local path %s does not exist", local_path)
            raise FileNotFoundError(f"Local path {local_path} does not exist")

        local_path = Path(local_path)
        remote_path = Path(remote_path)

        try:
            site.run_login(f"mkdir -p {remote_path.parent}")

            if self._uses_sftp():
                sftp = site.login_client.open_sftp()
                try:
                    if local_path.is_dir():
                        self._put_dir(sftp, local_path, remote_path)
                    else:
                        sftp.put(str(local_path), str(remote_path))
                finally:
                    sftp.close()
            else:
                source = f"{local_path}/" if local_path.is_dir() else str(local_path)
                self._run_rsync(source, self._remote_spec(remote_path))

            logger.debug("Transfer completed successfully")

        except Exception:
            logger.exception(
                "Error during file transfer: %s -> %s", local_path, remote_path
            )
            raise

    def get(
        self,
        remote_path: Union[str, Path],
        local_path: Union[str, Path],
        overwrite: bool = False,
    ):
        """Transfer a remote file or directory to a local path.

        Args:
            remote_path: Remote source file or directory.
            local_path: Local destination path.
            overwrite: Accepted for API compatibility; transfer backends decide
                replacement behavior.

        Raises:
            RuntimeError: If rsync fails.
        """

        site = self.site
        logger.debug("Attempting to transfer from %s to %s", remote_path, local_path)

        local_path = Path(local_path)
        remote_path = Path(remote_path)

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if self._uses_sftp():
                logger.debug("Transferring %s to %s (SFTP)", remote_path, local_path)
                sftp = site.login_client.open_sftp()
                try:
                    if self._sftp_is_dir(sftp, remote_path):
                        self._get_dir(sftp, remote_path, local_path)
                    else:
                        sftp.get(str(remote_path), str(local_path))
                finally:
                    sftp.close()
            else:
                remote_str = (
                    f"{remote_path}/" if remote_path.suffix == "" else str(remote_path)
                )
                local_str = f"{local_path}/" if local_path.is_dir() else str(local_path)
                self._run_rsync(self._remote_spec(remote_str), local_str)

            logger.debug("Transfer completed successfully")

        except Exception:
            logger.exception(
                "Error during file transfer: %s -> %s", remote_path, local_path
            )
            raise

    def _remote_spec(self, remote_path: Union[str, Path]) -> str:
        return (
            f"{self.site.credentials.username}@"
            f"{self.site.config.hostname}:{remote_path}"
        )

    def _run_rsync(self, source: str, target: str) -> None:
        debug_output = logger.isEnabledFor(logging.DEBUG)
        rsync_flags = "-avzP" if debug_output else "-az"
        rsync_cmd = ["rsync", rsync_flags]
        if not debug_output:
            rsync_cmd.append("--partial")
        rsync_cmd.extend([source, target])
        if self.site._login_client.is_proxy():
            control_path, _ = self.site._login_client.get_proxy_details()
            if not control_path:
                raise RuntimeError(
                    "Authenticated SSH proxy did not provide its control socket path"
                )
            ssh_command = shlex.join(["ssh", *control_socket_ssh_options(control_path)])
            rsync_cmd[1:1] = ["-e", ssh_command]
        logger.debug("rsync: %s", rsync_cmd)
        if debug_output:
            logger.debug("Streaming rsync file names and progress to the console")
            result = subprocess.run(rsync_cmd, text=True)
        else:
            result = subprocess.run(rsync_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr or "see the streamed rsync output above"
            raise RuntimeError(f"rsync failed: {detail}")

    def _uses_sftp(self) -> bool:
        """Return whether transfers should use the authenticated SFTP channel."""

        if self.site.transfer_method == "sftp":
            return True
        if self.site.transfer_method != "rsync":
            raise ValueError(
                "transfer_method must be either 'rsync' or 'sftp', got "
                f"{self.site.transfer_method!r}"
            )
        if self.site._login_client.is_proxy():
            return False
        logger.debug(
            "Using SFTP because rsync cannot reuse the authenticated Paramiko "
            "connection"
        )
        return True

    def _local_tmp_parent(self) -> Optional[Path]:
        tmp_dir = getattr(self.site, "local_host_tmp_dir", None)
        if callable(tmp_dir):
            tmp_dir = tmp_dir()
        if tmp_dir is None:
            host_config = getattr(self.site, "local_host_config", None)
            tmp_dir = getattr(host_config, "tmp_path", None)
            if tmp_dir is None:
                tmp_dir = getattr(host_config, "tmp_dir", None)
        if tmp_dir is None or str(tmp_dir).strip() == "":
            return None
        path = Path(tmp_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _remote_tmp_file(self, suffix: str) -> Path:
        tmp_dir = Path(getattr(self.site, "remote_tmp_dir", None) or "/tmp")
        return tmp_dir / f"frequensolve-{uuid.uuid4().hex}{suffix}"

    def _put_dir(self, sftp, local_dir: Path, remote_dir: Path):
        with tempfile.NamedTemporaryFile(
            suffix=".tar.gz",
            dir=self._local_tmp_parent(),
        ) as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                tar.add(local_dir, arcname=local_dir.name)

            remote_tar = self._remote_tmp_file(".tar.gz")
            self.site.run_login(f"mkdir -p {shlex.quote(str(remote_tar.parent))}")
            sftp.put(tmp.name, str(remote_tar))

            _, _, stderr = self.site.run_login_cmd(
                f"cd {shlex.quote(str(remote_dir.parent))} && "
                f"tar xzf {shlex.quote(str(remote_tar))} && "
                f"rm -f {shlex.quote(str(remote_tar))}"
            )

            err = stderr.read().decode().strip()
            if err:
                logger.error("Error extracting directory on remote: %s", err)
                raise RuntimeError(f"Failed to extract directory on remote: {err}")

    def _get_dir(self, sftp, remote_dir: Path, local_dir: Path):
        remote_tar = self._remote_tmp_file(".tar.gz")
        _, _, stderr = self.site.run_login_cmd(
            f"mkdir -p {shlex.quote(str(remote_tar.parent))} && "
            f"cd {shlex.quote(str(remote_dir.parent))} && "
            f"tar czf {shlex.quote(str(remote_tar))} "
            f"{shlex.quote(remote_dir.name)}"
        )

        err = stderr.read().decode().strip()
        if err:
            raise RuntimeError(f"Failed to create tar payload on remote: {err}")

        with tempfile.NamedTemporaryFile(
            suffix=".tar.gz",
            dir=self._local_tmp_parent(),
            delete=False,
        ) as tmp:
            local_tar = tmp.name

        try:
            sftp.get(str(remote_tar), local_tar)

            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(path=local_dir.parent, filter="data")
        finally:
            try:
                os.remove(local_tar)
            except FileNotFoundError:
                pass
            self.site.run_login(f"rm -f {shlex.quote(str(remote_tar))}")

    @staticmethod
    def _sftp_is_dir(sftp, remote_path: Union[str, Path]) -> bool:
        try:
            return stat.S_ISDIR(sftp.stat(str(remote_path)).st_mode)
        except OSError:
            return False
