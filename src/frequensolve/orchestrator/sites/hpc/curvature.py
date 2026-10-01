"""Remote staging and compute-node execution of Sauce curvature requests.

Each local curvature work directory (the parent of every operation directory)
maps to one remote root below the site's scratch directory, or its work
directory when no scratch directory is configured. Files below the local work
directory keep their relative layout there, so relative HDF5 external links
such as ``../histories/<digest>.h5`` resolve remotely. Other inputs (saved
factors, property-space meshes) are stored under ``external/`` with a name
derived from their local path, size and modification time.

A file is uploaded only when no remote copy of the same size exists, and
factors that Sauce wrote remotely as an earlier operation's ``pending.h5`` are
read in place. Sauce runs on compute nodes only, as a batch job or inside the
site's attached allocation; the login node only prepares directories,
submits, polls and transfers. On success the output and ``solver.log`` are
fetched beside the local request, whose caller verifies them. Mirrors persist
for reuse until :meth:`SlurmSite.remove_curvature_files` deletes them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from frequensolve.orchestrator.sites.config_file import _host_tmp_path_for_config
from frequensolve.orchestrator.sites.curvature import (
    FILE_KEYS,
    dispatcher_hint,
    external_link_targets,
    read_request,
    single_rank,
)
from frequensolve.orchestrator.sites.hpc.slurm_helpers import (
    read_stream,
    shell_word_batches,
    ssh_exit_status,
    temporary_text_file,
)
from frequensolve.orchestrator.sites.hpc.transfer import _remote_command_failed
from frequensolve.util.setup_logger import init_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from frequensolve.orchestrator.sites.hpc.site import SlurmRunConfig, SlurmSite

__all__ = ["SlurmCurvatureRunner"]

logger = init_logger(name=__name__, log_file="/tmp/log/frequensolve/hpc.log")

_MODES = ("auto", "batch", "attached")
_TERMINAL = frozenset({"complete", "failed", "cancelled", "timeout"})
_LOG_LINES = 40
_STATUS = "exit_status"
_PID = "launch.pid"
# OpenMP placement for ranks that the launcher binds to disjoint cores.
_PLACEMENT = (("OMP_PLACES", "cores"), ("OMP_PROC_BIND", "close"))
_SIZE_LOOP = (
    'for f in {paths}; do if [ -f "$f" ]; then '
    'printf "FS_SIZE %s\\n" "$(wc -c < "$f")"; '
    'else printf "FS_SIZE -1\\n"; fi; done'
)
# Names remote_root gives workdir mirrors; cleanup never touches other entries.
_MIRROR = re.compile(r"[A-Za-z0-9._-]+-[0-9a-f]{16}\Z")
# Quoted paths per removal command; well below Linux's 128 KiB argument cap.
_REMOVE_BYTES = 64 * 1024
# Bytes and newest modification time of each workdir mirror, on the login node.
_USAGE_SCRIPT = """\
import json, os, sys, time
base, wanted = sys.argv[1], sys.argv[2:]
try:
    names = wanted or sorted(os.listdir(base))
except OSError:
    names = []
roots = {}
for name in names:
    root = os.path.join(base, name)
    if os.path.islink(root) or not os.path.isdir(root):
        continue
    size, newest = 0, os.lstat(root).st_mtime
    for directory, folders, files in os.walk(root):
        for entry in folders:
            newest = max(newest, os.lstat(os.path.join(directory, entry)).st_mtime)
        for entry in files:
            info = os.lstat(os.path.join(directory, entry))
            size += info.st_size
            newest = max(newest, info.st_mtime)
    roots[name] = [size, newest]
print("FS_CURVATURE_USAGE " + json.dumps({"now": time.time(), "roots": roots}))
"""


def _quote(path: Any) -> str:
    return shlex.quote(str(path))


def _interval(config: Any, site: Any) -> float:
    """Return the resolved poll interval; zero is a valid explicit choice."""

    value = config.poll_interval
    return float(site.config.poll_interval if value is None else value)


@dataclass(frozen=True)
class _Staged:
    """A local file, its remote location and remote copies that may exist."""

    local: Path
    remote: PurePosixPath
    aliases: Tuple[PurePosixPath, ...] = ()
    fresh: bool = False


class SlurmCurvatureRunner:
    """Run one ``fs-curvature-request-1`` file through a :class:`SlurmSite`."""

    def __init__(self, site: "SlurmSite") -> None:
        self.site = site
        self.workdir = Path("/")
        self.root = PurePosixPath("/")

    # -- layout ---------------------------------------------------------------

    def curvature_dir(self) -> PurePosixPath:
        """Return the remote directory holding every workdir mirror."""

        return PurePosixPath(str(self.site.scratch_dir or self.site.work_dir)) / (
            "curvature"
        )

    def remote_root(self, workdir: Path) -> PurePosixPath:
        """Return the one remote directory mirroring a local curvature workdir."""

        label = re.sub(r"[^A-Za-z0-9._-]+", "_", workdir.parent.name)[:48]
        digest = hashlib.sha256(str(workdir).encode()).hexdigest()[:16]
        return self.curvature_dir() / f"{label or 'workdir'}-{digest}"

    def _inside(self, path: Path) -> Optional[PurePosixPath]:
        try:
            relative = path.relative_to(self.workdir)
        except ValueError:
            return None
        return self.root.joinpath(*relative.parts)

    def _remote(self, path: Path) -> PurePosixPath:
        mirrored = self._inside(path)
        if mirrored is not None:
            return mirrored
        stat = path.stat()
        key = hashlib.sha256(
            f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
        ).hexdigest()[:24]
        return self.root / "external" / key / path.name

    # -- staging --------------------------------------------------------------

    def _collect(
        self, path: Path, staged: Dict[Path, _Staged], *, fresh: bool = False
    ) -> None:
        """Add a file and every file its HDF5 external links reach."""

        import h5py

        if path in staged:
            return
        if not path.is_file():
            raise FileNotFoundError(f"Curvature input {path} does not exist")
        inside = self._inside(path) is not None
        remote = self._remote(path)
        aliases: Tuple[PurePosixPath, ...] = ()
        if inside and not fresh and path.name == "result.h5":
            # Sauce wrote this result remotely under its pending name.
            aliases = (remote.with_name("pending.h5"),)
        staged[path] = _Staged(path, remote, aliases, fresh)
        if not h5py.is_hdf5(path):
            return
        for name in external_link_targets(path):
            target = Path(os.path.normpath(path.parent / name))
            if (
                PurePosixPath(name).is_absolute()
                or not inside
                or self._inside(target) is None
            ):
                raise ValueError(
                    f"{path} links {name}; remote curvature staging supports "
                    f"relative links between files below {self.workdir}"
                )
            self._collect(target, staged)

    def _login(self, command: str, *, purpose: str) -> str:
        """Run a login-node command, judged by its exit status."""

        _, stdout, stderr = self.site.run_login_cmd(command)
        output = read_stream(stdout)
        error = read_stream(stderr)
        if _remote_command_failed(error, ssh_exit_status(stdout, stderr)):
            raise RuntimeError(
                f"Could not {purpose} on {self.site.site_name}: "
                f"{error or 'remote command failed'}"
            )
        return output

    def _upload(
        self, directory: PurePosixPath, staged: List[_Staged]
    ) -> Dict[Path, PurePosixPath]:
        """Upload files the remote root lacks; return each file's remote copy."""

        site = self.site
        checked = [entry for entry in staged if not entry.fresh]
        candidates = [
            path for entry in checked for path in (entry.remote, *entry.aliases)
        ]
        command = (
            f"mkdir -p -- {_quote(directory)} && "
            f"rm -f -- {_quote(directory / _STATUS)} {_quote(directory / _PID)}"
        )
        if candidates:
            command += " && " + _SIZE_LOOP.format(
                paths=" ".join(_quote(path) for path in candidates)
            )
        listing = self._login(command, purpose="prepare remote curvature files")
        sizes = [
            int(line.split()[1])
            for line in listing.splitlines()
            if line.startswith("FS_SIZE ")
        ]
        if len(sizes) != len(candidates):
            raise RuntimeError("Remote curvature file listing was incomplete")
        remote_sizes = dict(zip(candidates, sizes))
        resolved: Dict[Path, PurePosixPath] = {}
        for entry in staged:
            size = entry.local.stat().st_size
            match = (
                None
                if entry.fresh
                else next(
                    (
                        path
                        for path in (entry.remote, *entry.aliases)
                        if remote_sizes[path] == size
                    ),
                    None,
                )
            )
            if match is None:
                site._emit(f"Uploading {entry.local} -> {entry.remote}")
                site.put(entry.local, str(entry.remote), compress=False)
                match = entry.remote
            resolved[entry.local] = match
        return resolved

    def _put_text(self, text: str, remote: PurePosixPath, suffix: str) -> None:
        with temporary_text_file(
            text,
            suffix=suffix,
            prefix="curvature",
            directory=_host_tmp_path_for_config(
                getattr(self.site, "_site_config_path", None)
            ),
        ) as path:
            self.site.put(path, str(remote))

    # -- execution ------------------------------------------------------------

    def _script(
        self,
        directory: PurePosixPath,
        *,
        ranks: int,
        threads: int,
        **batch: Any,
    ) -> str:
        site = self.site
        return site._render_template(
            "curvature/curvature.sh",
            keep_trailing_newline=True,
            batch_job=bool(batch),
            status_shell=_quote(directory / _STATUS),
            pid_shell=_quote(directory / _PID),
            directory_shell=_quote(directory),
            runtime_setup=site._runtime_setup_lines(),
            n_procs=ranks,
            n_threads=threads,
            mpi_shell=_quote(site.mpi_cmd),
            mpi_args_shell=" ".join(_quote(value) for value in site.mpi_args),
            executable_shell=_quote(site.executable),
            thread_placement=[
                f"export {name}={value}"
                for name, value in _PLACEMENT
                if name not in site.environment
            ],
            request_shell=_quote(directory / "request.json"),
            **batch,
        )

    def _batch_resources(
        self, single: bool, overrides: Dict[str, Any]
    ) -> Tuple["SlurmRunConfig", int]:
        """Resolve nodes, ranks and threads for a batch job from the run config."""

        site = self.site
        base = site.curvature_run_config or site.run_config
        config, extra = base.resolved(site.config, **overrides)
        if extra:
            raise TypeError(
                "Unexpected curvature option(s): " + ", ".join(sorted(extra))
            )
        partition = site.config_for_partition(str(config.queue))
        cores = max(1, int(partition.cores_per_node))
        if single:
            nodes, per_node = 1, 1
        else:
            nodes = int(config.nodes)
            sockets = int(getattr(partition, "sockets_per_node", 1) or 1)
            per_node = int(config.ranks_per_node or max(1, sockets))
        duration = partition.validate_request(nodes, nodes * per_node, config.duration)
        effective = config.merged(
            nodes=nodes, ranks_per_node=per_node, duration=duration
        )
        return effective, max(1, cores // per_node)

    def _submit_batch(
        self,
        request: Dict[str, Any],
        directory: PurePosixPath,
        config: "SlurmRunConfig",
        threads: int,
    ) -> str:
        from frequensolve.orchestrator.sites.hpc.site import _safe_slurm_directive

        site = self.site
        account = (
            config.account
            if config.account is not None
            else getattr(site.config, "account", None)
        )
        ranks = int(config.nodes) * int(config.ranks_per_node or 1)
        notify_on = _safe_slurm_directive(config.notify_on, "notify_on")
        script = directory / "curvature.slurm"
        self._put_text(
            self._script(
                directory,
                ranks=ranks,
                threads=threads,
                name="fs-curvature",
                launch_log=_safe_slurm_directive(
                    directory / "launch-%j.log", "curvature directory"
                ),
                n_nodes=int(config.nodes),
                # Every rank needs its threads' CPUs in the allocation; TACC's
                # ibrun allocates whole nodes and places ranks itself.
                ranks_per_node=int(config.ranks_per_node or 1),
                cpus_per_task=(
                    None if Path(str(site.mpi_cmd)).name == "ibrun" else threads
                ),
                queue=_safe_slurm_directive(config.queue, "queue"),
                account=_safe_slurm_directive(account or None, "account"),
                duration=_safe_slurm_directive(config.duration, "duration"),
                notify_on=notify_on.upper() if notify_on else None,
                notify_email=_safe_slurm_directive(config.notify_email, "notify_email"),
            ),
            script,
            ".slurm",
        )
        if site.enterprise_hpc is not None:
            site._enterprise_hpc_preflight(run_config=config)
        command = " ".join(
            ["sbatch", *(_quote(arg) for arg in config.slurm_args), _quote(script)]
        )
        job_id = site._submit_sbatch(command)
        site._emit(
            f"Submitted curvature {request.get('method')} to "
            f"{site.site_name}:{config.queue} as job {job_id} "
            f"({ranks} rank(s) x {threads} thread(s))"
        )
        return job_id

    def _launch_attached(
        self, request: Dict[str, Any], directory: PurePosixPath, single: bool
    ) -> str:
        """Start the request on the allocation's batch host and return its id."""

        site = self.site
        if site._compute_client is None:
            site._attach_compute_client()
        pool = site.pool
        nodes, processes, cores = (
            max(1, int(pool.nhost or 1)),
            max(1, int(pool.nproc or 1)),
            max(1, int(pool.ncore or 1)),
        )
        ranks, threads = (
            (1, max(1, cores // nodes))
            if single
            else (processes, max(1, cores // processes))
        )
        script = directory / "curvature.sh"
        self._put_text(
            self._script(directory, ranks=ranks, threads=threads),
            script,
            ".sh",
        )
        launch = (
            f"cd {_quote(directory)} && "
            "if command -v setsid >/dev/null 2>&1; then detach=setsid; "
            "else detach=; fi; "
            f"$detach nohup bash {_quote(script)} > launch.log 2>&1 < /dev/null &"
        )
        _, stdout, stderr = site.run_compute_cmd(launch)
        error = read_stream(stderr)
        read_stream(stdout)
        if _remote_command_failed(error, ssh_exit_status(stdout, stderr)):
            raise RuntimeError(
                f"Could not start curvature in allocation {pool.id}: {error}"
            )
        site._emit(
            f"Started curvature {request.get('method')} in {site.site_name} "
            f"allocation {pool.id} ({ranks} rank(s) x {threads} thread(s))"
        )
        return str(pool.id)

    def _exit_status(self, directory: PurePosixPath) -> Optional[int]:
        status = _quote(directory / _STATUS)
        text = self.site.run_login(
            f'if [ -f {status} ]; then printf "FS_EXIT %s\\n" "$(cat -- {status})"; fi'
        )
        for line in text.splitlines():
            if line.startswith("FS_EXIT "):
                try:
                    return int(line.split()[1])
                except (IndexError, ValueError):
                    return None
        return None

    def _wait(
        self, directory: PurePosixPath, job_id: str, *, interval: float, label: str
    ) -> Tuple[Optional[int], str]:
        """Poll the completion record and the scheduler until either finishes."""

        site = self.site
        last = None
        while True:
            code = self._exit_status(directory)
            if code is not None:
                return code, "complete" if code == 0 else "failed"
            state = site.update_status(job_id)
            if state != last:
                site._emit(f"{site.site_name} {label} {job_id}: {state}")
                last = state
            if state in _TERMINAL:
                return self._exit_status(directory), state
            time.sleep(interval)

    def _stop(self, directory: PurePosixPath, job_id: str, *, attached: bool) -> None:
        """Best-effort cancellation when waiting is interrupted."""

        try:
            if attached:
                pid = _quote(directory / _PID)
                self.site.run_compute_cmd(
                    f"if [ -f {pid} ]; then p=$(cat -- {pid}); "
                    'kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null; '
                    "fi; true"
                )
            else:
                self.site.cancel_job(job_id)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Could not stop curvature run %s (%s)", job_id, exc)

    def _failure(
        self, directory: PurePosixPath, log: Path, method: str, detail: str
    ) -> RuntimeError:
        site = self.site
        try:
            site.get(str(directory / "solver.log"), log)
        except Exception as exc:
            logger.debug("Could not fetch curvature solver log: %s", exc)
        try:
            tails = site.run_login(
                f"cd {_quote(directory)} && for f in solver.log launch*.log; do "
                'if [ -f "$f" ]; then printf "==> %s <==\\n" "$f"; '
                f'tail -n {_LOG_LINES} -- "$f"; fi; done'
            )
        except Exception as exc:
            tails = f"(remote logs unavailable: {type(exc).__name__})"
        hint = dispatcher_hint(site.executable, tails)
        return RuntimeError(
            f"Sauce curvature {method} failed on {site.site_name} ({detail}); "
            f"remote directory {directory}\n{tails}"
            + (f"\nHint: {hint}." if hint else "")
        )

    # -- entry point ----------------------------------------------------------

    def run(self, request: Path, *, mode: str = "auto", **overrides: Any) -> None:
        """Stage, execute and fetch one request; see the module documentation."""

        if mode not in _MODES:
            raise ValueError("mode must be 'auto', 'batch', or 'attached'")
        site = self.site
        request = Path(os.path.abspath(request))
        local = read_request(request)
        if not local.get("input") or not local.get("output"):
            raise ValueError(f"Curvature request {request} needs input and output")
        method = str(local.get("method"))
        single = single_rank(local)
        enterprise = site.enterprise_hpc is not None
        if mode == "attached" and enterprise:
            raise ValueError(
                "Enterprise HPC curvature runs are validated batch jobs; use mode='batch'"
            )
        active = bool(site.pool.id) and site.provisioned
        if mode == "attached" and not active:
            raise RuntimeError(
                "No active compute allocation is attached; use mode='batch' or "
                "mode='auto'"
            )
        attached = mode == "attached" or (mode == "auto" and not enterprise and active)
        if attached and set(overrides) - {"poll_interval"}:
            raise ValueError(
                "Resource overrides apply to batch curvature jobs; an attached "
                "allocation keeps its own resources (pass mode='batch')"
            )
        # Resolve and validate resources before transferring any data.
        resources = None if attached else self._batch_resources(single, overrides)
        if resources is not None and enterprise:
            site._enterprise_hpc_preflight(run_config=resources[0])
        elif not enterprise:
            site.check_solver_compatibility()

        self.workdir = request.parent.parent
        self.root = self.remote_root(self.workdir)
        directory = self._remote(request.parent)
        staged: Dict[Path, _Staged] = {}
        for key in ("input", "factors", "source_mesh", "target_mesh"):
            if key in local:
                path = Path(os.path.abspath(local[key]))
                self._collect(path, staged, fresh=key == "input")
        resolved = self._upload(directory, list(staged.values()))

        output = Path(os.path.abspath(local["output"]))
        remote_output = directory / output.name
        remote = dict(local)
        for key in FILE_KEYS:
            if key in local:
                remote[key] = str(
                    remote_output
                    if key == "output"
                    else resolved[Path(os.path.abspath(local[key]))]
                )
        self._put_text(
            json.dumps(remote, indent=2) + "\n", directory / "request.json", ".json"
        )

        if attached:
            run_config, _ = (site.curvature_run_config or site.run_config).resolved(
                site.config, **overrides
            )
            interval = _interval(run_config, site)
            job_id = self._launch_attached(local, directory, single)
            label = "curvature in allocation"
        else:
            assert resources is not None
            run_config, threads = resources
            interval = _interval(run_config, site)
            job_id = self._submit_batch(local, directory, run_config, threads)
            label = "curvature job"
        try:
            code, state = self._wait(directory, job_id, interval=interval, label=label)
        except BaseException:
            self._stop(directory, job_id, attached=attached)
            raise
        log = request.parent / "solver.log"
        if code != 0:
            detail = (
                f"exit status {code}"
                if code is not None
                else f"Slurm state {state} before Sauce finished"
            )
            raise self._failure(directory, log, method, f"{label} {job_id}: {detail}")
        site.get(str(remote_output), output, compress=False)
        try:
            site.get(str(directory / "solver.log"), log)
        except Exception as exc:
            logger.warning("Could not fetch curvature solver log: %s", exc)

    # -- cleanup --------------------------------------------------------------

    def _usage(self, names: Optional[List[str]]) -> Tuple[float, Dict[str, Any]]:
        """Return the remote clock and ``{mirror: [bytes, newest mtime]}``."""

        output = self._login(
            " ".join(
                [
                    "python3",
                    "-c",
                    _quote(_USAGE_SCRIPT),
                    _quote(self.curvature_dir()),
                    *(_quote(name) for name in names or ()),
                ]
            ),
            purpose="measure staged curvature files",
        )
        for line in output.splitlines():
            if line.startswith("FS_CURVATURE_USAGE "):
                usage = json.loads(line.split(" ", 1)[1])
                return float(usage["now"]), dict(usage["roots"])
        raise RuntimeError("Remote curvature usage listing was incomplete")

    def remove(
        self,
        workdir: Optional[Path] = None,
        *,
        older_than: Optional[Union[float, timedelta]] = None,
        max_bytes: Optional[int] = None,
    ) -> List[PurePosixPath]:
        """Delete workdir mirrors; see :meth:`SlurmSite.remove_curvature_files`."""

        if isinstance(older_than, timedelta):
            older_than = older_than.total_seconds()
        if older_than is not None and not float(older_than) >= 0:
            raise ValueError("older_than must be a nonnegative duration")
        if max_bytes is not None and (isinstance(max_bytes, bool) or max_bytes < 0):
            raise ValueError("max_bytes must be a nonnegative integer")
        names = None
        if workdir is not None:
            # NativeCurvature resolves its workdir; direct runs keep abspath.
            local = Path(workdir).expanduser()
            names = list(
                dict.fromkeys(
                    self.remote_root(path).name
                    for path in (local.resolve(), Path(os.path.abspath(local)))
                )
            )
        if names is not None and older_than is None and max_bytes is None:
            selected = names
        else:
            now, roots = self._usage(names)
            # Least recently modified first.
            ordered = sorted(roots.items(), key=lambda item: float(item[1][1]))
            if older_than is None and max_bytes is None:
                selected = [name for name, _ in ordered]
            else:
                selected = [
                    name
                    for name, (_, newest) in ordered
                    if older_than is not None and now - float(newest) >= older_than
                ]
                if max_bytes is not None:
                    kept = [item for item in ordered if item[0] not in selected]
                    total = sum(int(size) for _, (size, _) in kept)
                    for name, (size, _) in kept:
                        if total <= max_bytes:
                            break
                        selected.append(name)
                        total -= int(size)
        targets = [
            _quote(self.curvature_dir() / name)
            for name in selected
            if _MIRROR.match(name)
        ]
        removed: List[PurePosixPath] = []
        for batch in shell_word_batches(targets, _REMOVE_BYTES):
            output = self._login(
                f"status=0; for d in {batch}; do "
                'if [ -d "$d" ] && [ ! -L "$d" ]; then '
                'if rm -rf -- "$d"; then printf "FS_REMOVED %s\\n" "$d"; '
                "else status=1; fi; fi; done; exit $status",
                purpose="remove staged curvature files",
            )
            removed.extend(
                PurePosixPath(line.split(" ", 1)[1])
                for line in output.splitlines()
                if line.startswith("FS_REMOVED ")
            )
        for path in removed:
            self.site._emit(f"Removed staged curvature files {path}")
        return removed
