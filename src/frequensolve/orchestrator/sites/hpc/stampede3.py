"""Stampede3-specific SLURM site configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from frequensolve.orchestrator.sites.hpc.config import Stampede3Config
from frequensolve.orchestrator.sites.hpc.site import SlurmRunConfig, SlurmSite
from frequensolve.orchestrator.utils.credentials import Credentials
from frequensolve.orchestrator.utils.pool import PoolInfo
from frequensolve.orchestrator.utils.ssh import SSHClientClass

__all__ = ["TACCLoginCredentials", "Stampede3Site"]


class TACCLoginCredentials(Credentials):
    """TACC credentials used by Stampede3.

    Environment variables named by ``user_env``, ``pw_env``, and
    ``ssh_key_env`` are read by the shared credential helper when explicit
    values are not supplied.
    """

    user_env: str = "TACC_USERNAME"
    pw_env: str = "TACC_PASSWORD"
    ssh_key_env: str = "SSH_PASSPHRASE"


@dataclass(kw_only=True, init=False)
class Stampede3Site(SlurmSite):
    """Stampede3 remote execution site.

    This class only supplies Stampede3-specific defaults. SSH, transfer,
    provisioning, SLURM submission, status polling, and result fetching are
    implemented by :class:`frequensolve.orchestrator.sites.hpc.SlurmSite`.

    Args:
        rel_path: Site configuration path or name used to load Stampede3
            credentials and defaults.
        transfer_method: File-transfer backend, either ``"rsync"`` or
            ``"sftp"``.
        default_queue: SLURM queue/partition used when no run config supplies
            one.
        run_config: Optional per-run SLURM settings.
        verbose: Whether to enable verbose site logging.
    """

    credentials: TACCLoginCredentials
    config: Stampede3Config
    pool: PoolInfo
    transfer_method: Literal["rsync", "sftp"] = "rsync"
    _login_client: SSHClientClass
    _compute_client: SSHClientClass
    _work_dir: Path
    _executable: str
    _FS_dir: Path

    site_name = "Stampede3"
    credentials_cls = TACCLoginCredentials
    config_cls = Stampede3Config
    default_queue = "skx-dev"
    default_host = "stampede3.tacc.utexas.edu"
    work_dir_env = "STAMPEDE3_WORK_DIR"
    solver_executable_env = "STAMPEDE3_SOLVER_EXECUTABLE"
    default_solver_executable = (
        "/work2/06472/jbadger/shared/stampede3/FS_stable/FS_seismic"
    )

    def __init__(
        self,
        rel_path: Union[str, Path],
        transfer_method: Literal["rsync", "sftp"] = "rsync",
        default_queue: str = "skx-dev",
        run_config: Optional[SlurmRunConfig] = None,
        verbose: bool = False,
    ):
        super().__init__(
            rel_path=rel_path,
            transfer_method=transfer_method,
            default_queue=default_queue,
            run_config=run_config,
            verbose=verbose,
        )
