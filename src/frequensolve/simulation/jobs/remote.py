"""Remote-staging helpers mixed into FrequenSolve jobs.

The mixin writes local job and simulation JSON, rewrites project-rooted paths
for a remote filesystem layout, and enumerates input files that must accompany
the staged payloads on local, cloud, or HPC execution sites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

from frequensolve.model.property import rsf_binary_path
from frequensolve.simulation.simulation import BaseSimulation, CustomJSONEncoder
from frequensolve.util.mixins import ExportContext
from frequensolve.util.store import compact_hdf5_file

if TYPE_CHECKING:
    from frequensolve.simulation.jobs.base import JobLayout

_PROJECT_FILE_REFERENCE_KEYS = frozenset(
    {
        "file",
        "data_path",
        "layout_file",
        "observed",
        "receiver_file",
        "relation_file",
        "source_file",
        "srf_file",
    }
)

_STAGED_PROVENANCE_SCHEMA = "fs-staged-provenance-1"


@runtime_checkable
class _SaveableSimulation(Protocol):
    """Concrete simulations must support saving before a job can be staged."""

    def save(self) -> Path: ...


class JobRemoteMixin:
    """Save and stage job inputs for local and remote execution.

    The mixin rewrites project-rooted payload paths so locally saved jobs and
    simulations can be staged into a remote project layout without mutating the
    local job definition.
    """

    if TYPE_CHECKING:
        name: str
        simulation: BaseSimulation
        _file: Optional[Path]

        @property
        def _local_path(self) -> Path: ...

        @property
        def _result_path(self) -> Path: ...

        @property
        def project_path(self) -> Path: ...

        preserve_task_outputs: bool

        def to_fs(
            self, ctx: Optional[ExportContext] = None, *, project_relative: bool = False
        ) -> Dict[str, Any]: ...
        def _refresh_external_input_fingerprint_for_save(self) -> None: ...
        def _set_output_request_fingerprint(self, data: Dict[str, Any]) -> None: ...
        def _serialized_artifact_metadata(self) -> Optional[Dict[str, Any]]: ...
        @staticmethod
        def _sha256_file(path: Path) -> str: ...
        @staticmethod
        def _compatibility_hash_payloads(
            job_data: Dict[str, Any], simulation_data: Dict[str, Any]
        ) -> str: ...

        @staticmethod
        def _write_json_file(path: Path, payload: Dict[str, Any]) -> None: ...

        def export_context(self) -> ExportContext: ...

    def save(self) -> Path:
        """Save the simulation and project-relative job JSON to disk.

        Returns:
            Path to the saved local job JSON.

        Raises:
            ValueError: If the simulation is not attached to a project or job
                output validation fails.
        """

        from frequensolve.simulation.jobs.base import JobLayout

        JobLayout._layout_name(getattr(self, "name"), field="name")
        if not isinstance(self.simulation, _SaveableSimulation):
            raise TypeError("Job simulation must support saving before staging")
        self.simulation.save()
        self._refresh_external_input_fingerprint_for_save()
        file = self._local_path / f"{self.name}.json"
        self._file = file
        ctx = self.export_context()
        data = self.to_fs(ctx, project_relative=True)
        data["result_path"] = str(self._result_path.relative_to(self.project_path))
        self._set_output_request_fingerprint(data)
        data["artifact_contract"] = self._serialized_artifact_metadata()
        self._write_json_file(file, data)
        assert ctx.store is not None
        ctx.store.prune_unreferenced(data)
        if ctx.store.path.exists():
            compact_hdf5_file(ctx.store.path)
        return file

    def save_for_remote(
        self,
        site: str,
        remote_project: Union[Path, str],
        *,
        include_job_id: bool = True,
    ) -> tuple[Path, Path]:
        """Stage a remote job JSON without replacing the local definition.

        Args:
            site: Site name used to choose the local staging directory.
            remote_project: Project root visible on the remote site.
            include_job_id: Include the previous execution id as descriptor metadata.
                Disable when the transport owns immutable run identity separately.

        Returns:
            Tuple ``(staged_file, remote_job_file)`` where ``staged_file`` is
            local and ``remote_job_file`` is the path expected on the site.

        Raises:
            ValueError: If staged JSON still contains local project roots that
                would not be valid remotely.
        """

        from frequensolve.simulation.jobs.base import JobLayout

        local_file = self.save()
        with open(local_file, "r") as f:
            payload = json.load(f)
        local_layout = JobLayout.from_payload(payload, job_file=local_file)
        remote_layout = local_layout.with_project(remote_project)
        source_projects = self._remote_source_projects(local_layout, payload)
        data = self._payload_for_layout(
            payload,
            source=local_layout,
            target=remote_layout,
            source_projects=source_projects,
        )
        self._assert_remote_payload_has_no_local_roots(
            data,
            source_projects=source_projects,
            target_project=remote_layout.project,
            payload_name="job JSON",
        )

        if not include_job_id:
            data.pop("job_id", None)
        stage_dir = self._result_path / "_fs_run" / "remote" / site
        staged_file = stage_dir / Path(local_file).name
        if include_job_id:
            self._write_json_file(staged_file, data)
        else:
            staged_file.parent.mkdir(parents=True, exist_ok=True)
            staged_file.write_text(json.dumps(data, sort_keys=True, indent=3))
        self._write_staged_provenance(
            site,
            {
                "job": {
                    "digest": self._sha256_file(staged_file),
                }
            },
        )
        return staged_file, remote_layout.job_file

    def save_simulation_for_remote(
        self, site: str, remote_project: Union[Path, str]
    ) -> tuple[Path, Path]:
        """Stage this job's simulation JSON for a remote project layout.

        Args:
            site: Site name used to choose the local staging directory.
            remote_project: Project root visible on the remote site.

        Returns:
            Tuple ``(staged_file, remote_simulation_file)`` where
            ``staged_file`` is local and ``remote_simulation_file`` is the path
            expected on the site.

        Raises:
            ValueError: If staged JSON still contains local project roots that
                would not be valid remotely.
        """

        self.save()
        local_layout = self._saved_layout()
        remote_layout = local_layout.with_project(remote_project)
        with open(local_layout.simulation_file, "r") as f:
            data = json.load(f)
        source_projects = self._remote_source_projects(local_layout, data)
        data = self._map_payload_project_roots(
            data,
            source_projects=source_projects,
            target_project=remote_layout.project,
        )
        data["project_path"] = str(remote_layout.project)
        self._assert_remote_payload_has_no_local_roots(
            data,
            source_projects=source_projects,
            target_project=remote_layout.project,
            payload_name="simulation JSON",
        )
        try:
            staged_relpath = remote_layout.simulation_file.relative_to(
                remote_layout.project
            )
        except ValueError:
            staged_relpath = Path(remote_layout.simulation_file.name)

        staged_file = self._result_path / "_fs_run" / "remote" / site / staged_relpath
        self._write_json_file(staged_file, data)
        provenance = self._read_staged_provenance(site)
        provenance["simulation"] = {"digest": self._sha256_file(staged_file)}
        staged_job = (
            self._result_path / "_fs_run" / "remote" / site / local_layout.job_file.name
        )
        if self.preserve_task_outputs and staged_job.is_file():
            provenance["compatibility"] = self._compatibility_hash_payloads(
                json.loads(staged_job.read_text()), data
            )
        self._write_staged_provenance(site, provenance)
        return staged_file, remote_layout.simulation_file

    def staged_artifact_fingerprints(self, site: str) -> Optional[Dict[str, str]]:
        """Return exact staged JSON fingerprints for remote task currentness."""

        provenance = self._read_staged_provenance(site)
        job = provenance.get("job")
        simulation = provenance.get("simulation")
        if not isinstance(job, Mapping) or not isinstance(simulation, Mapping):
            return None
        job_digest = job.get("digest")
        simulation_digest = simulation.get("digest")
        if not self._valid_sha256_digest(job_digest) or not self._valid_sha256_digest(
            simulation_digest
        ):
            return None
        output_request = getattr(self, "_output_request_fingerprint_cache", None)
        output_digest = (
            output_request.get("digest")
            if isinstance(output_request, Mapping)
            else None
        )
        if not self._valid_sha256_digest(output_digest):
            return None
        return {
            "job": str(job_digest),
            "simulation": str(simulation_digest),
            "outputs": str(output_digest),
        }

    def staged_task_fingerprints(self, site: str) -> Optional[Dict[str, str]]:
        """Validate retained tasks against the actual staged retry identity."""
        if self.preserve_task_outputs:
            digest = self._read_staged_provenance(site).get("compatibility")
            if not self._valid_sha256_digest(digest):
                return None
            return {"compatibility": str(digest)}
        return self.staged_artifact_fingerprints(site)

    @staticmethod
    def _valid_sha256_digest(value: object) -> bool:
        if not isinstance(value, str) or len(value) != len("sha256:") + 64:
            return False
        if not value.startswith("sha256:"):
            return False
        try:
            int(value.removeprefix("sha256:"), 16)
        except ValueError:
            return False
        return True

    def _staged_provenance_path(self, site: str) -> Path:
        """Return the job-scoped compact provenance sidecar path."""

        if (
            not isinstance(site, str)
            or not site
            or site in {".", ".."}
            or Path(site).name != site
            or "\\" in site
        ):
            raise ValueError("Remote site staging name must be one path component")
        return self._result_path / "_fs_run" / "remote" / site / "provenance.json"

    def _read_staged_provenance(self, site: str) -> Dict[str, Any]:
        frozen = getattr(self, "_frozen_staged_provenance", None)
        if isinstance(frozen, Mapping) and site in frozen:
            return dict(frozen[site])
        path = self._staged_provenance_path(site)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema": _STAGED_PROVENANCE_SCHEMA}
        if not isinstance(payload, dict):
            return {"schema": _STAGED_PROVENANCE_SCHEMA}
        if payload.get("schema") != _STAGED_PROVENANCE_SCHEMA:
            return {"schema": _STAGED_PROVENANCE_SCHEMA}
        return {
            key: value
            for key, value in payload.items()
            if key in {"schema", "job", "simulation", "compatibility"}
        }

    def _write_staged_provenance(
        self,
        site: str,
        payload: Mapping[str, Any],
    ) -> None:
        """Atomically replace compact provenance for the current staging."""

        data = {
            "schema": _STAGED_PROVENANCE_SCHEMA,
            **{
                key: dict(value)
                for key, value in payload.items()
                if key in {"job", "simulation"} and isinstance(value, Mapping)
            },
        }
        if self._valid_sha256_digest(payload.get("compatibility")):
            data["compatibility"] = payload["compatibility"]
        self._write_json_file(self._staged_provenance_path(site), data)

    def remote_input_files(
        self, remote_project: Union[Path, str]
    ) -> List[tuple[Path, Path]]:
        """Return local input files that must accompany remote job inputs.

        Args:
            remote_project: Project root visible on the remote site.

        Returns:
            List of ``(local_path, remote_path)`` pairs for mesh, property, and
            artifact files referenced by the staged payloads.
        """

        remote_project = Path(remote_project)
        files = []
        seen = set()
        local_layout = self._saved_layout()
        payloads = []

        def add_pair(pair: Optional[tuple[Path, Path]]) -> None:
            if pair is None:
                return
            local, remote = pair
            key = (Path(local), Path(remote))
            if key in seen:
                return
            seen.add(key)
            files.append(pair)

        mesh = getattr(self.simulation, "mesh", None)
        mesh_file = getattr(mesh, "file", None)
        if mesh_file is not None:
            payloads.append(mesh_file)

        for payload_file in [local_layout.job_file, local_layout.simulation_file]:
            if not payload_file.exists():
                continue
            with open(payload_file, "r") as f:
                payloads.append(json.load(f))

        source_projects = self._remote_source_projects(local_layout, *payloads)

        if mesh_file is not None:
            add_pair(
                self._remote_project_file_pair(
                    mesh_file,
                    source_project=local_layout.project,
                    remote_project=remote_project,
                    source_projects=source_projects,
                )
            )

        for payload in payloads:
            if not isinstance(payload, Mapping):
                continue
            for file_ref in self._iter_file_references(payload):
                pair = self._remote_project_file_pair(
                    file_ref,
                    source_project=local_layout.project,
                    remote_project=remote_project,
                    source_projects=source_projects,
                )
                add_pair(pair)
                for sidecar_ref in self._rsf_sidecar_references(pair):
                    add_pair(
                        self._remote_project_file_pair(
                            sidecar_ref,
                            source_project=local_layout.project,
                            remote_project=remote_project,
                            source_projects=source_projects,
                        )
                    )
        return files

    @staticmethod
    def _map_payload_paths(
        value: Any,
        *,
        source_project: Path,
        target_project: Path,
    ) -> Any:
        """Map absolute project paths in a JSON-like payload to another project."""

        if isinstance(value, Mapping):
            return {
                key: JobRemoteMixin._map_payload_paths(
                    item,
                    source_project=source_project,
                    target_project=target_project,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                JobRemoteMixin._map_payload_paths(
                    item,
                    source_project=source_project,
                    target_project=target_project,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            return [
                JobRemoteMixin._map_payload_paths(
                    item,
                    source_project=source_project,
                    target_project=target_project,
                )
                for item in value
            ]
        if isinstance(value, Path):
            return JobRemoteMixin._map_payload_paths(
                str(value),
                source_project=source_project,
                target_project=target_project,
            )
        if isinstance(value, str):
            source = str(source_project)
            if source and source in value:
                return value.replace(source, str(target_project))
        return value

    @staticmethod
    def _unique_paths(paths: Iterable[Union[str, Path]]) -> List[Path]:
        unique: List[Path] = []
        seen: set[str] = set()
        for value in paths:
            if value is None:
                continue
            try:
                path = Path(value).expanduser()
            except TypeError:
                continue
            candidates = [path]
            if path.is_absolute():
                try:
                    resolved = path.resolve()
                except OSError:
                    resolved = path
                candidates.append(resolved)
            for candidate in candidates:
                text = str(candidate)
                if text in {"", "."} or candidate.anchor == text:
                    continue
                if text in seen:
                    continue
                seen.add(text)
                unique.append(candidate)
        return unique

    @staticmethod
    def _project_root_from_artifact_path(value: Union[str, Path]) -> Optional[Path]:
        text = JobRemoteMixin._strip_file_locator(value)
        try:
            path = Path(text).expanduser()
        except TypeError:
            return None
        if not path.is_absolute():
            return None
        parts = path.parts
        for marker in ("simulations", "jobs"):
            if marker not in parts:
                continue
            index = parts.index(marker)
            if index <= 0:
                continue
            return Path(*parts[:index])
        return None

    @staticmethod
    def _project_root_from_job_path(value: Union[str, Path]) -> Optional[Path]:
        path = Path(value).expanduser()
        if path.suffix:
            path = path.parent
        parts = path.parts
        if "jobs" not in parts:
            return None
        index = parts.index("jobs")
        if index <= 0:
            return None
        return Path(*parts[:index])

    @staticmethod
    def _payload_project_roots(value: Any) -> List[Path]:
        roots: List[Path] = []
        if isinstance(value, Mapping):
            project_path = value.get("project_path")
            if isinstance(project_path, (str, Path)):
                roots.append(Path(project_path).expanduser())
            for item in value.values():
                roots.extend(JobRemoteMixin._payload_project_roots(item))
        elif isinstance(value, (list, tuple)):
            for item in value:
                roots.extend(JobRemoteMixin._payload_project_roots(item))
        elif isinstance(value, (str, Path)):
            root = JobRemoteMixin._project_root_from_artifact_path(value)
            if root is not None:
                roots.append(root)
        return roots

    def _remote_source_projects(
        self,
        local_layout: JobLayout,
        *payloads: Any,
    ) -> List[Path]:
        roots: List[Union[str, Path]] = [local_layout.project]
        for payload in payloads:
            roots.extend(self._payload_project_roots(payload))
        roots.append(self.project_path)
        simulation = getattr(self, "simulation", None)
        if simulation is not None:
            sim_project = getattr(simulation, "project_path", None)
            if sim_project is not None:
                roots.append(sim_project)
        return self._unique_paths(roots)

    @staticmethod
    def _map_payload_project_roots(
        payload: Mapping[str, Any],
        *,
        source_projects: Iterable[Path],
        target_project: Path,
    ) -> Dict[str, Any]:
        data: Any = dict(payload)
        source_projects = sorted(
            JobRemoteMixin._unique_paths(source_projects),
            key=lambda path: len(str(path)),
            reverse=True,
        )
        for source_project in source_projects:
            data = JobRemoteMixin._map_payload_paths(
                data,
                source_project=source_project,
                target_project=target_project,
            )
        return data

    @staticmethod
    def _assert_remote_payload_has_no_local_roots(
        payload: Mapping[str, Any],
        *,
        source_projects: Iterable[Path],
        target_project: Path,
        payload_name: str,
    ) -> None:
        text = json.dumps(payload, cls=CustomJSONEncoder)
        target_text = str(target_project)
        remaining = [
            str(path)
            for path in JobRemoteMixin._unique_paths(source_projects)
            if str(path) != target_text and str(path) in text
        ]
        if not remaining:
            return
        compact = ", ".join(remaining[:3])
        if len(remaining) > 3:
            compact = f"{compact}, ..."
        raise ValueError(
            f"Remote-staged {payload_name} still contains local project path(s): "
            f"{compact}. The job was not submitted because the solver would try "
            "to open local files on the remote site."
        )

    @staticmethod
    def _payload_for_layout(
        payload: Mapping[str, Any],
        *,
        source: JobLayout,
        target: JobLayout,
        source_projects: Iterable[Path] = (),
    ) -> Dict[str, Any]:
        data = JobRemoteMixin._map_payload_project_roots(
            payload,
            source_projects=[source.project, *source_projects],
            target_project=target.project,
        )
        data["project_path"] = str(target.project)
        data["simulation"] = str(target.simulation_file)
        data["result_path"] = str(target.result_dir)
        return data

    def _saved_layout(self) -> JobLayout:
        from frequensolve.simulation.jobs.base import JobLayout

        job_file = self._file if self._file is not None else self.save()
        with open(job_file, "r") as f:
            payload = json.load(f)
        return JobLayout.from_payload(payload, job_file=job_file)

    @staticmethod
    def _iter_file_references(value: Any) -> Iterable[str]:
        if isinstance(value, Mapping):
            for section, keys in (
                ("control_sensitivities", ("direction", "current")),
                ("Image", ("direction",)),
            ):
                config = value.get(section)
                if isinstance(config, Mapping):
                    for key in keys:
                        path = config.get(key)
                        if isinstance(path, (str, Path)):
                            yield JobRemoteMixin._strip_file_locator(path)
            derivatives = value.get("observed_derivatives")
            if isinstance(derivatives, Mapping):
                df = derivatives.get("df")
                if isinstance(df, (str, Path)):
                    yield JobRemoteMixin._strip_file_locator(df)
            for key, item in value.items():
                if key in _PROJECT_FILE_REFERENCE_KEYS and isinstance(
                    item, (str, Path)
                ):
                    yield JobRemoteMixin._strip_file_locator(item)
                yield from JobRemoteMixin._iter_file_references(item)
        elif isinstance(value, list):
            for item in value:
                yield from JobRemoteMixin._iter_file_references(item)

    @staticmethod
    def _strip_file_locator(value: Union[str, Path]) -> str:
        text = str(value)
        if ":" not in text:
            return text
        file_part, _ = text.split(":", 1)
        if Path(file_part).suffix:
            return file_part
        return text

    @staticmethod
    def _rsf_sidecar_references(pair: Optional[tuple[Path, Path]]) -> Iterable[Path]:
        if pair is None:
            return []
        local, _remote = pair
        local = Path(local)
        if local.suffix.lower() != ".rsf" or not local.exists():
            return []
        sidecar = rsf_binary_path(local)
        if sidecar is None or not sidecar.exists():
            return []
        return [sidecar]

    def _remote_project_file_pair(
        self,
        value: Union[str, Path],
        *,
        source_project: Path,
        remote_project: Path,
        source_projects: Iterable[Path] = (),
    ) -> Optional[tuple[Path, Path]]:
        path = Path(value)
        source_roots = self._unique_paths([source_project, *source_projects])
        relative: Optional[Path] = None
        if path.is_absolute():
            for root in source_roots:
                try:
                    relative = path.relative_to(root)
                    break
                except ValueError:
                    continue
            if relative is None:
                return None
            local_candidates = [root / relative for root in source_roots]
            if path not in local_candidates:
                local_candidates.append(path)
        else:
            relative = path
            local_candidates = [root / relative for root in source_roots]
        for local in local_candidates:
            if local.exists():
                return local, remote_project / relative
        return None
