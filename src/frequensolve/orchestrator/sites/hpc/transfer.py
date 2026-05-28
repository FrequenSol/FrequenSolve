"""File transfer helpers for SLURM-backed HPC sites."""

from __future__ import annotations

import os
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Union

from frequensolve.orchestrator.sites.base import _wait_for_path
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

            if site.transfer_method == "sftp":
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
            if site.transfer_method == "sftp":
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
        rsync_cmd = ["rsync", "-azP", source, target]
        logger.debug("rsync: %s", rsync_cmd)
        result = subprocess.run(rsync_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"rsync failed: {result.stderr}")

    def _put_dir(self, sftp, local_dir: Path, remote_dir: Path):
        with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                tar.add(local_dir, arcname=local_dir.name)

            remote_tar = str(remote_dir.parent / f"{remote_dir.name}.tar.gz")
            sftp.put(tmp.name, remote_tar)

            _, _, stderr = self.site.run_login_cmd(
                f"cd {remote_dir.parent} && "
                f"tar xzf {remote_dir.name}.tar.gz && "
                f"rm {remote_dir.name}.tar.gz"
            )

            err = stderr.read().decode().strip()
            if err:
                logger.error("Error extracting directory on remote: %s", err)
                raise RuntimeError(f"Failed to extract directory on remote: {err}")

    def _get_dir(self, sftp, remote_dir: Path, local_dir: Path):
        remote_tar = str(remote_dir.parent / f"{remote_dir.name}.tar.gz")
        _, _, stderr = self.site.run_login_cmd(
            f"cd {remote_dir.parent} && "
            f"tar czf {remote_dir.name}.tar.gz {remote_dir.name}"
        )

        err = stderr.read().decode().strip()
        if err:
            raise RuntimeError(f"Failed to create tar payload on remote: {err}")

        local_tar = str(local_dir.parent / f"{local_dir.name}.tar.gz")
        sftp.get(remote_tar, local_tar)

        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(path=local_dir.parent)

        os.remove(local_tar)
        self.site.run_login(f"rm {remote_tar}")

    @staticmethod
    def _sftp_is_dir(sftp, remote_path: Union[str, Path]) -> bool:
        try:
            return stat.S_ISDIR(sftp.stat(str(remote_path)).st_mode)
        except OSError:
            return False
