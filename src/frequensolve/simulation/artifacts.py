from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np


def _real_frequency(value: Union[float, complex]) -> float:
    if isinstance(value, complex):
        return float(value.real)
    if isinstance(value, np.generic):
        return _real_frequency(value.item())
    return float(value)


def _as_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _relative_to(path: Path, base: Optional[Path]) -> str:
    if base is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


@dataclass(frozen=True)
class OutputArtifact:
    """One file produced by a solver run."""

    path: Path
    relative_path: Optional[str] = None
    kind: Optional[str] = None
    schema: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(
        cls, data: Mapping[str, Any], result_path: Optional[Union[str, Path]] = None
    ) -> "OutputArtifact":
        root = _as_path(result_path) if result_path is not None else None
        raw_path = data.get("path") or data.get("relative_path")
        path = _as_path(raw_path)
        if not path.is_absolute() and root is not None:
            path = root / path
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"path", "relative_path", "kind", "schema"}
        }
        return cls(
            path=path,
            relative_path=data.get("relative_path"),
            kind=data.get("kind"),
            schema=data.get("schema"),
            metadata=metadata,
        )

    def to_fs(self, project_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        base = _as_path(project_path) if project_path is not None else None
        payload = {
            "path": str(self.path),
            "relative_path": self.relative_path or _relative_to(self.path, base),
        }
        if self.kind is not None:
            payload["kind"] = self.kind
        if self.schema is not None:
            payload["schema"] = self.schema
        payload.update(self.metadata)
        return payload


@dataclass(frozen=True)
class RunMetadata:
    """Sauce/Python run metadata collected beside a result directory."""

    manifest: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, Any] = field(default_factory=dict)
    error: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def read(cls, result_path: Union[str, Path]) -> "RunMetadata":
        result_path = _as_path(result_path)
        run_dir = result_path / "_fs_run"

        def read_json(path: Path) -> Dict[str, Any]:
            if not path.exists():
                return {}
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                return {}

        return cls(
            manifest=read_json(run_dir / "run_manifest.json"),
            outputs=read_json(run_dir / "outputs.json"),
            timings=read_json(run_dir / "timings.json"),
            error=read_json(run_dir / "error.json"),
            state=read_json(result_path / "_fs_python_run.json"),
        )

    @property
    def successful(self) -> bool:
        if self.manifest:
            return self.manifest.get("exit_status") == "success"
        if self.state:
            return self.state.get("status") in {"completed", "skipped"}
        return False

    @property
    def job_file_hash(self) -> Optional[str]:
        return self.manifest.get("job_file_sha256")

    @property
    def simulation_file_hash(self) -> Optional[str]:
        return self.manifest.get("simulation_file_sha256")


@dataclass(frozen=True)
class TraceManifest:
    """Typed description of a job's per-frequency trace files."""

    files: List[Path]
    frequencies: Dict[int, float]
    groups: List[str]
    simulation: Path
    result_path: Path
    output_path: Path
    project_path: Optional[Path] = None
    components: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    artifacts: List[OutputArtifact] = field(default_factory=list)
    run: RunMetadata = field(default_factory=RunMetadata)

    @classmethod
    def from_job(
        cls,
        job,
        *,
        project_path: Optional[Union[str, Path]] = None,
        resolve_legacy: bool = False,
    ) -> "TraceManifest":
        sim = job.simulation
        source_project = _as_path(job.project_path).resolve()
        local_project = (
            _as_path(project_path).resolve()
            if project_path is not None
            else source_project
        )
        output = job.trace_outputs
        result_path = cls._map_project_path(
            job._result_path, source_project, local_project
        )
        output_path = cls._map_project_path(output.path, source_project, local_project)
        frequencies = {
            index: _real_frequency(freq)
            for index, freq in enumerate(output.frequencies, start=1)
        }
        files = [
            output_path / f"traces_{index}.h5"
            for index in range(1, len(frequencies) + 1)
        ]
        if resolve_legacy:
            files = [cls.resolve_trace_file(path) for path in files]

        artifacts = cls._read_artifacts(result_path)
        simulation_path = cls._map_project_path(
            sim._file, source_project, local_project
        )
        return cls(
            files=files,
            frequencies=frequencies,
            groups=list(output.groups),
            simulation=simulation_path,
            result_path=result_path,
            output_path=output_path,
            project_path=local_project,
            components=list(output.components),
            sources=list(output.sources),
            artifacts=artifacts,
            run=RunMetadata.read(result_path),
        )

    @classmethod
    def combine(
        cls,
        manifests: Iterable["TraceManifest"],
        duplicate: str = "first",
    ) -> "TraceManifest":
        manifests = list(manifests)
        if not manifests:
            raise ValueError("At least one TraceManifest is required")
        if duplicate not in {"first", "last", "error"}:
            raise ValueError("duplicate must be 'first', 'last', or 'error'")

        groups = manifests[0].groups
        for manifest in manifests[1:]:
            if manifest.groups != groups:
                raise ValueError("Cannot combine trace manifests with different groups")

        entries: Dict[float, Path] = {}
        duplicates = []
        for manifest in manifests:
            ordered = sorted(
                manifest.frequencies.items(), key=lambda item: int(item[0])
            )
            for (_, freq), file in zip(ordered, manifest.files):
                if freq in entries:
                    duplicates.append(freq)
                    if duplicate == "error":
                        raise ValueError(f"Duplicate trace frequency: {freq}")
                    if duplicate == "first":
                        continue
                entries[freq] = file

        frequencies = {
            index: float(freq) for index, freq in enumerate(sorted(entries), start=1)
        }
        files = [entries[freq] for freq in sorted(entries)]
        first = manifests[0]
        artifacts = [
            artifact for manifest in manifests for artifact in manifest.artifacts
        ]
        run = first.run
        if duplicates:
            state = dict(run.state)
            state["duplicate_frequencies"] = sorted(set(duplicates))
            run = RunMetadata(
                manifest=run.manifest,
                outputs=run.outputs,
                timings=run.timings,
                error=run.error,
                state=state,
            )
        return cls(
            files=files,
            frequencies=frequencies,
            groups=groups,
            simulation=first.simulation,
            result_path=first.result_path,
            output_path=first.output_path,
            project_path=first.project_path,
            components=first.components,
            sources=first.sources,
            artifacts=artifacts,
            run=run,
        )

    @staticmethod
    def _map_project_path(
        path: Union[str, Path], source_project: Path, project_path: Path
    ) -> Path:
        path = _as_path(path)
        if path.is_absolute():
            try:
                return project_path / path.resolve().relative_to(source_project)
            except Exception:
                return path
        return project_path / path

    @staticmethod
    def resolve_trace_file(path: Union[str, Path]) -> Path:
        path = _as_path(path)
        if path.exists():
            return path
        legacy = path.with_name(path.name.replace("traces_", "receivers_", 1))
        if legacy.exists():
            return legacy
        return path

    @staticmethod
    def _read_artifacts(result_path: Path) -> List[OutputArtifact]:
        outputs = RunMetadata.read(result_path).outputs
        files = outputs.get("files", []) if outputs else []
        return [OutputArtifact.from_fs(file, result_path=result_path) for file in files]

    @property
    def existing_files(self) -> List[Path]:
        return [
            self.resolve_trace_file(file)
            for file in self.files
            if self.resolve_trace_file(file).exists()
        ]

    @property
    def complete(self) -> bool:
        return bool(self.files) and all(
            self.resolve_trace_file(file).exists() for file in self.files
        )

    def to_fs(self) -> Dict[str, Any]:
        return {
            "schema": "frequensolve-trace-manifest-1",
            "files": [str(file) for file in self.files],
            "frequencies": self.frequencies,
            "groups": self.groups,
            "simulation": str(self.simulation),
            "result_path": str(self.result_path),
            "output_path": str(self.output_path),
            "components": self.components,
            "sources": self.sources,
            "artifacts": [
                artifact.to_fs(self.project_path) for artifact in self.artifacts
            ],
        }


@dataclass(frozen=True)
class TraceOutputSpec:
    """Resolved trace output request for a job."""

    path: Path
    frequencies: Sequence[Union[float, complex]]
    groups: List[str]
    components: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TraceOutputHandle:
    """Convenience handle exposed as ``job.traces``."""

    job: Any

    @property
    def manifest(self) -> TraceManifest:
        return self.job.trace_manifest

    def open(self, upscale: int = 1, project_path: Optional[Union[str, Path]] = None):
        from frequensolve.seismic.traces import TraceDataset

        return TraceDataset.from_manifest(
            TraceManifest.from_job(
                self.job,
                project_path=project_path,
                resolve_legacy=True,
            ),
            upscale=upscale,
        )

    def to_fs(self) -> Dict[str, Any]:
        return self.manifest.to_fs()

    def __getitem__(self, key: str) -> Any:
        return self.to_fs()[key]
