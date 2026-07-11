"""Stampede3-specific SLURM site configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from frequensolve.orchestrator.sites.hpc.config import Stampede3Config
from frequensolve.orchestrator.sites.hpc.site import SlurmRunConfig, SlurmSite
from frequensolve.orchestrator.utils.credential_store import CredentialStore
from frequensolve.orchestrator.utils.credentials import Credentials
from frequensolve.orchestrator.utils.pool import PoolInfo
from frequensolve.orchestrator.utils.ssh import SSHClientClass

__all__ = ["TACCLoginCredentials", "Stampede3Site"]


class TACCLoginCredentials(Credentials):
    """TACC credentials used by Stampede3."""


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
        username: TACC username.
        credential: Keyring lookup name.
        ssh_key: Optional SSH private-key path.
        solver: Remote solver executable path.
        work_dir: Remote work-directory root.
        modules: Environment modules loaded before solver execution.
        environment: Non-secret environment values exported before solver
            execution.
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

    site_name = "Stampede3"
    credentials_cls = TACCLoginCredentials
    config_cls = Stampede3Config
    default_queue = "skx-dev"
    default_host = "stampede3.tacc.utexas.edu"
    default_solver_executable = None

    def __init__(
        self,
        rel_path: Union[str, Path],
        transfer_method: Literal["rsync", "sftp"] = "rsync",
        default_queue: str = "skx-dev",
        credentials: Optional[TACCLoginCredentials] = None,
        username: Optional[str] = None,
        credential: Optional[str] = None,
        ssh_key: Optional[Union[str, Path]] = None,
        credential_store: Optional[CredentialStore] = None,
        solver: Optional[Union[str, Path]] = None,
        work_dir: Optional[Union[str, Path]] = None,
        modules: Optional[list[str]] = None,
        environment: Optional[dict[str, object]] = None,
        run_config: Optional[SlurmRunConfig] = None,
        verbose: bool = False,
    ):
        super().__init__(
            rel_path=rel_path,
            transfer_method=transfer_method,
            default_queue=default_queue,
            credentials=credentials,
            username=username,
            credential=credential,
            ssh_key=ssh_key,
            credential_store=credential_store,
            solver=solver,
            work_dir=work_dir,
            modules=modules,
            environment=environment,
            run_config=run_config,
            verbose=verbose,
        )
