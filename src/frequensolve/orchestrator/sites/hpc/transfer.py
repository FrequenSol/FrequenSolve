"""File transfer helpers for SLURM-backed HPC sites."""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Union

from frequensolve.orchestrator.sites.base import _wait_for_path
from frequensolve.orchestrator.sites.config_file import _host_tmp_path_for_config
from frequensolve.orchestrator.sites.hpc.slurm_helpers import ssh_exit_status
from frequensolve.orchestrator.utils.ssh import control_socket_ssh_options
from frequensolve.util.setup_logger import init_logger

logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/hpc.log")

_REMOTE_USERNAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_REMOTE_HOST = re.compile(r"(?:[A-Za-z0-9_][A-Za-z0-9_.-]*|\[[0-9A-Fa-f:]+\])\Z")


def _validated_remote_path(value: Union[str, Path]) -> PurePosixPath:
    """Return a remote path that is safe for SSH shell commands.

    Absolute paths and traversal-free relative paths are both supported.  A
    relative path keeps the historical public API behavior of resolving from
    the remote account's working directory.
    """

    raw = str(value)
    if (
        not raw
        or "\x00" in raw
        or "\n" in raw
        or "\r" in raw
        or "\\" in raw
        or any(part == ".." for part in raw.split("/"))
    ):
        raise ValueError("remote path must be a non-empty traversal-free POSIX path")
    return PurePosixPath(raw)


class SlurmTransferManager:
    """Owns remote file transfer mechanics for a SLURM site.

    Args:
        site: ``SlurmSite`` instance that provides SSH clients and transfer
            settings.
    """

    def __init__(self, site: Any) -> None:
        self.site = site

    def put(self, local_path: Union[str, Path], remote_path: Union[str, Path]) -> None:
        """Transfer a local file or directory to a remote path.

        Args:
            local_path: Local source file or directory.
            remote_path: Remote destination path.

        Raises:
            FileNotFoundError: If ``local_path`` does not exist.
            RuntimeError: If rsync fails.
        """

        site = self.site
        logger.debug("Starting HPC upload")
        if not _wait_for_path(local_path):
            logger.error("Local path %s does not exist", local_path)
            raise FileNotFoundError(f"Local path {local_path} does not exist")

        local_path = Path(local_path)
        validated_remote_path = _validated_remote_path(remote_path)

        try:
            site.run_login(
                f"mkdir -p -- {shlex.quote(str(validated_remote_path.parent))}"
            )

            if self._uses_sftp():
                sftp = site.login_client.open_sftp()
                try:
                    if local_path.is_dir():
                        self._put_dir(sftp, local_path, validated_remote_path)
                    else:
                        self._put_file(sftp, local_path, validated_remote_path)
                finally:
                    sftp.close()
            else:
                source = f"{local_path}/" if local_path.is_dir() else str(local_path)
                self._run_rsync(source, self._remote_spec(validated_remote_path))

            logger.debug("Transfer completed successfully")

        except (FileNotFoundError, ValueError, RuntimeError):
            raise
        except Exception as exc:
            logger.error("HPC upload failed (%s)", type(exc).__name__)
            raise RuntimeError(f"HPC upload failed ({type(exc).__name__})") from None

    def get(
        self,
        remote_path: Union[str, Path],
        local_path: Union[str, Path],
        overwrite: bool = False,
    ) -> None:
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
        logger.debug("Starting HPC download")

        local_path = Path(local_path)
        validated_remote_path = _validated_remote_path(remote_path)

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if self._uses_sftp():
                logger.debug(
                    "Transferring %s to %s (SFTP)",
                    validated_remote_path,
                    local_path,
                )
                sftp = site.login_client.open_sftp()
                try:
                    if self._sftp_is_dir(sftp, validated_remote_path):
                        self._get_dir(sftp, validated_remote_path, local_path)
                    else:
                        self._get_file(sftp, validated_remote_path, local_path)
                finally:
                    sftp.close()
            else:
                remote_str = (
                    f"{validated_remote_path}/"
                    if validated_remote_path.suffix == ""
                    else str(validated_remote_path)
                )
                local_str = f"{local_path}/" if local_path.is_dir() else str(local_path)
                self._run_rsync(self._remote_spec(remote_str), local_str)

            logger.debug("Transfer completed successfully")

        except (FileNotFoundError, ValueError, RuntimeError):
            raise
        except Exception as exc:
            logger.error("HPC download failed (%s)", type(exc).__name__)
            raise RuntimeError(f"HPC download failed ({type(exc).__name__})") from None

    def _remote_spec(self, remote_path: Union[str, Path, PurePosixPath]) -> str:
        username = str(self.site.credentials.username)
        hostname = str(self.site.config.hostname)
        raw_path = str(remote_path)
        trailing_slash = raw_path.endswith("/")
        path = _validated_remote_path(raw_path)
        if not _REMOTE_USERNAME.fullmatch(username) or not _REMOTE_HOST.fullmatch(
            hostname
        ):
            raise ValueError("SSH username or hostname contains unsafe characters")
        rendered_path = str(path) + ("/" if trailing_slash else "")
        return f"{username}@{hostname}:{shlex.quote(rendered_path)}"

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
            raise RuntimeError(f"rsync transfer failed with status {result.returncode}")

    @staticmethod
    def _verify_sftp_size(
        sftp: Any, remote_path: PurePosixPath, local_path: Path
    ) -> None:
        """Verify one SFTP object against its local counterpart."""

        remote_size = sftp.stat(str(remote_path)).st_size
        if remote_size != local_path.stat().st_size:
            raise RuntimeError("SFTP transfer size verification failed")

    def _put_file(
        self, sftp: Any, local_path: Path, remote_path: PurePosixPath
    ) -> None:
        """Upload and verify one file before atomically publishing it."""

        temporary = remote_path.with_name(
            f".{remote_path.name}.frequensolve-{uuid.uuid4().hex}.partial"
        )
        try:
            try:
                destination_mode = stat.S_IMODE(sftp.stat(str(remote_path)).st_mode)
                destination_exists = True
            except OSError:
                destination_mode = stat.S_IMODE(local_path.stat().st_mode)
                destination_exists = False

            sftp.put(str(local_path), str(temporary))
            self._verify_sftp_size(sftp, temporary, local_path)
            sftp.chmod(str(temporary), destination_mode)
            try:
                sftp.posix_rename(str(temporary), str(remote_path))
            except (AttributeError, OSError):
                if destination_exists:
                    raise RuntimeError(
                        "SFTP server cannot atomically replace an existing file; "
                        "use rsync or enable the POSIX rename extension"
                    ) from None
                sftp.rename(str(temporary), str(remote_path))
        except Exception:
            try:
                sftp.remove(str(temporary))
            except OSError:
                pass
            raise

    def _get_file(
        self, sftp: Any, remote_path: PurePosixPath, local_path: Path
    ) -> None:
        """Download and verify one file before atomically publishing it."""

        local_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_path.with_name(
            f".{local_path.name}.frequensolve-{uuid.uuid4().hex}.partial"
        )
        fd = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o666,
        )
        os.close(fd)
        try:
            if local_path.exists():
                destination_mode = stat.S_IMODE(local_path.stat().st_mode)
            else:
                destination_mode = stat.S_IMODE(temporary.stat().st_mode)
            sftp.get(str(remote_path), str(temporary))
            self._verify_sftp_size(sftp, remote_path, temporary)
            temporary.chmod(destination_mode)
            os.replace(temporary, local_path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

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

    def _local_tmp_parent(self) -> Path:
        path = _host_tmp_path_for_config(getattr(self.site, "_site_config_path", None))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _remote_tmp_file(self, suffix: str) -> PurePosixPath:
        tmp_dir = _validated_remote_path(
            getattr(self.site, "remote_tmp_dir", None) or "/tmp"
        )
        return tmp_dir / f"frequensolve-{uuid.uuid4().hex}{suffix}"

    def _remove_remote_tmp_file(self, path: Union[str, Path, PurePosixPath]) -> None:
        """Best-effort cleanup for a uniquely owned remote staging file."""

        try:
            self.site.run_login(f"rm -f {shlex.quote(str(path))}")
        except Exception as exc:
            logger.warning(
                "Remote transfer staging cleanup failed (%s)", type(exc).__name__
            )

    def _put_dir(self, sftp: Any, local_dir: Path, remote_dir: PurePosixPath) -> None:
        with tempfile.NamedTemporaryFile(
            suffix=".tar.gz",
            dir=self._local_tmp_parent(),
        ) as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                tar.add(local_dir, arcname=remote_dir.name)

            remote_tar = self._remote_tmp_file(".tar.gz")
            self.site.run_login(f"mkdir -p {shlex.quote(str(remote_tar.parent))}")
            self._put_file(sftp, Path(tmp.name), remote_tar)

            try:
                token = uuid.uuid4().hex
                staging_dir = remote_dir.with_name(
                    f".{remote_dir.name}.frequensolve-{token}.partial"
                )
                backup_dir = remote_dir.with_name(
                    f".{remote_dir.name}.frequensolve-{token}.backup"
                )
                _, stdout, stderr = self.site.run_login_cmd(
                    "set -eu; "
                    f"staging={shlex.quote(str(staging_dir))}; "
                    f"backup={shlex.quote(str(backup_dir))}; "
                    f"destination={shlex.quote(str(remote_dir))}; "
                    f"archive={shlex.quote(str(remote_tar))}; "
                    f"entry={shlex.quote(remote_dir.name)}; "
                    'cleanup() { rm -rf -- "$staging"; }; '
                    "trap cleanup EXIT HUP INT TERM; "
                    f"mkdir -p -- {shlex.quote(str(remote_dir.parent))}; "
                    'mkdir -- "$staging"; '
                    'tar xzf "$archive" -C "$staging"; '
                    'test -d "$staging/$entry"; '
                    "had_destination=0; "
                    'if [ -e "$destination" ] || [ -L "$destination" ]; then '
                    'mv -- "$destination" "$backup"; had_destination=1; fi; '
                    'if mv -- "$staging/$entry" "$destination"; then '
                    '[ "$had_destination" -eq 0 ] || rm -rf -- "$backup"; '
                    "else status=$?; "
                    'if [ "$had_destination" -eq 1 ] && '
                    '[ ! -e "$destination" ] && [ ! -L "$destination" ]; then '
                    'mv -- "$backup" "$destination" || true; fi; '
                    'exit "$status"; fi'
                )

                err = stderr.read().decode().strip()
                exit_status = ssh_exit_status(stdout, stderr)
                if err or exit_status not in {None, 0}:
                    raise RuntimeError("Remote directory extraction failed")
            finally:
                self._remove_remote_tmp_file(remote_tar)

    def _get_dir(self, sftp: Any, remote_dir: PurePosixPath, local_dir: Path) -> None:
        remote_tar = self._remote_tmp_file(".tar.gz")
        try:
            _, stdout, stderr = self.site.run_login_cmd(
                f"mkdir -p {shlex.quote(str(remote_tar.parent))} && "
                f"resolved_dir=$(readlink -f -- {shlex.quote(str(remote_dir))}) && "
                'test -d "$resolved_dir" && '
                f"tar czf {shlex.quote(str(remote_tar))} "
                '-C "$resolved_dir" .'
            )

            err = stderr.read().decode().strip()
            exit_status = ssh_exit_status(stdout, stderr)
            if err or exit_status not in {None, 0}:
                raise RuntimeError("Remote directory archive failed")

            with tempfile.NamedTemporaryFile(
                suffix=".tar.gz",
                dir=self._local_tmp_parent(),
                delete=False,
            ) as tmp:
                local_tar = tmp.name

            try:
                sftp.get(str(remote_tar), local_tar)
                self._verify_sftp_size(
                    sftp,
                    remote_tar,
                    Path(local_tar),
                )

                with tarfile.open(local_tar, "r:gz") as tar:
                    with tempfile.TemporaryDirectory(
                        prefix=f".{local_dir.name}.frequensolve-",
                        dir=local_dir.parent,
                    ) as staging_name:
                        staging = Path(staging_name)
                        extracted = staging / remote_dir.name
                        extracted.mkdir()
                        tar.extractall(path=extracted, filter="data")

                        backup = None
                        if local_dir.exists():
                            backup = local_dir.with_name(
                                f".{local_dir.name}.frequensolve-"
                                f"{uuid.uuid4().hex}.backup"
                            )
                            os.replace(local_dir, backup)
                        try:
                            os.replace(extracted, local_dir)
                        except Exception:
                            if backup is not None and not local_dir.exists():
                                os.replace(backup, local_dir)
                            raise
                        else:
                            if backup is not None:
                                try:
                                    if backup.is_dir():
                                        shutil.rmtree(backup)
                                    else:
                                        backup.unlink()
                                except OSError as exc:
                                    logger.warning(
                                        "Could not remove replaced local result (%s)",
                                        type(exc).__name__,
                                    )
            finally:
                try:
                    os.remove(local_tar)
                except FileNotFoundError:
                    pass
        finally:
            self._remove_remote_tmp_file(remote_tar)

    @staticmethod
    def _sftp_is_dir(sftp: Any, remote_path: Union[str, Path, PurePosixPath]) -> bool:
        try:
            return stat.S_ISDIR(sftp.stat(str(remote_path)).st_mode)
        except OSError:
            return False
