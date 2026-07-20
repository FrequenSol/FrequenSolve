"""Structured access to solver output artifacts and result metadata.

The objects in this module read and describe files produced by a FrequenSolve
run, including trace manifests, run metadata, and lightweight handles for trace
and wavefield outputs that are materialized on demand.
"""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

__all__ = [
    "OutputArtifact",
    "RunMetadata",
    "TraceManifest",
    "TraceOutputHandle",
    "TraceOutputSpec",
    "WavefieldOutputHandle",
]


@dataclass(frozen=True)
class OutputArtifact:
    """Structured record for one file produced by a solver run.

    Args:
        path: Absolute or result-relative output path.
        relative_path: Optional path relative to the project or result root.
        kind: Optional artifact kind such as ``"h5"``, ``"vtk"``, or
            ``"json"``.
        schema: Optional artifact schema identifier.
        metadata: Additional solver-reported artifact fields.
    """

    path: Path
    relative_path: Optional[str] = None
    kind: Optional[str] = None
    schema: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_fs(
        cls, data: Mapping[str, Any], result_path: Optional[Union[str, Path]] = None
    ) -> "OutputArtifact":
        """Deserialize an artifact record relative to an optional result root.

        Args:
            data: Serialized artifact mapping.
            result_path: Optional result directory used to resolve relative
                artifact paths.

        Returns:
            ``OutputArtifact`` with path and metadata restored.
        """

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
        """Serialize the artifact record.

        Args:
            project_path: Optional base path used to emit ``relative_path``.

        Returns:
            JSON-compatible artifact payload.
        """

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
    """Fast solver and Python run metadata collected beside a result directory.

    Args:
        manifest: Parsed ``_fs_run/run_manifest.json`` payload.
        outputs: Parsed ``_fs_run/outputs.json`` payload.
        timings: Parsed ``_fs_run/timings.json`` payload.
        error: Parsed ``_fs_run/error.json`` payload.
        state: Parsed Python-side ``_fs_python_run.json`` payload.
        result_path: Result directory these metadata files came from.
    """

    manifest: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, Any] = field(default_factory=dict)
    error: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    result_path: Optional[Path] = None

    @classmethod
    def read(cls, result_path: Union[str, Path]) -> "RunMetadata":
        """Read solver metadata JSON files beside a result directory.

        Args:
            result_path: Job result directory.

        Returns:
            ``RunMetadata`` with missing or invalid JSON files represented by
            empty dictionaries.
        """

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
            result_path=result_path,
        )

    @property
    def successful(self) -> bool:
        """Return whether recorded solver or Python metadata indicate success.

        Returns:
            ``True`` when solver manifest status is ``"success"`` or the
            Python-side state was completed/skipped.
        """

        if self.manifest:
            status = self.manifest.get("exit_status")
            if isinstance(status, Mapping):
                status = status.get("status")
            return status == "success"
        if self.state:
            return self.state.get("status") in {"completed", "skipped"}
        return False

    @property
    def job_file_hash(self) -> Optional[str]:
        """Return the SHA-256 hash of the job JSON, if recorded.

        Returns:
            Hash string from solver metadata, or ``None`` if unavailable.
        """

        if "job_file_sha256" in self.manifest:
            return self.manifest.get("job_file_sha256")
        inputs = self.manifest.get("inputs", {})
        if isinstance(inputs, Mapping):
            job_file = inputs.get("job_file", {})
            if isinstance(job_file, Mapping):
                return job_file.get("hash")
        return None

    @property
    def simulation_file_hash(self) -> Optional[str]:
        """Return the SHA-256 hash of the simulation JSON, if recorded.

        Returns:
            Hash string from solver metadata, or ``None`` if unavailable.
        """

        if "simulation_file_sha256" in self.manifest:
            return self.manifest.get("simulation_file_sha256")
        inputs = self.manifest.get("inputs", {})
        if isinstance(inputs, Mapping):
            simulation_file = inputs.get("simulation_file", {})
            if isinstance(simulation_file, Mapping):
                return simulation_file.get("hash")
        return None

    @property
    def artifacts(self) -> List[OutputArtifact]:
        """Return output files reported by the fast solver for this run.

        Returns:
            Artifact records resolved relative to ``result_path``.
        """

        files = self.outputs.get("files", []) if self.outputs else []
        return [
            OutputArtifact.from_fs(file, result_path=self.result_path) for file in files
        ]

    def output_files(
        self,
        *,
        kind: Optional[str] = None,
        suffix: Optional[Union[str, Sequence[str]]] = None,
        base: Optional[Union[str, Path]] = None,
        existing: bool = False,
    ) -> List[Path]:
        """Return output file paths filtered by kind, suffix, and base name.

        Args:
            kind: Optional artifact kind or file family to keep.
            suffix: Optional filename suffix or suffixes to keep.
            base: Optional output base name, such as a ParaView request name.
            existing: When true, return only paths that currently exist.

        Returns:
            Deduplicated list of matching output paths.
        """

        suffixes: Optional[tuple[str, ...]]
        if suffix is None:
            suffixes = None
        elif isinstance(suffix, str):
            suffixes = (suffix,)
        else:
            suffixes = tuple(suffix)
        normalized_base = _normalize_base(base)

        files = []
        for artifact in self.artifacts:
            path = artifact.path
            if not _artifact_matches_kind(artifact, kind):
                continue
            if suffixes is not None and not path.name.endswith(suffixes):
                continue
            if not _path_matches_base(path, normalized_base):
                continue
            if existing and not path.exists():
                continue
            files.append(path)
        files.extend(
            self._discover_output_files(
                kind=kind,
                suffixes=suffixes,
                base=normalized_base,
            )
        )
        deduped = []
        seen = set()
        for path in files:
            key = _path_identity_key(path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return deduped

    def _discover_output_files(
        self,
        *,
        kind: Optional[str],
        suffixes: Optional[tuple[str, ...]],
        base: Optional[str],
    ) -> List[Path]:
        if self.result_path is None or not self.result_path.exists():
            return []

        scan_suffixes = suffixes or _kind_suffixes(kind)
        if not scan_suffixes:
            return []

        files = []
        for suffix in scan_suffixes:
            files.extend(self.result_path.rglob(f"*{suffix}"))

        return [
            path
            for path in sorted(set(files))
            if path.is_file()
            and "_fs_run" not in path.relative_to(self.result_path).parts
            and _artifact_matches_kind(
                OutputArtifact(path=path, kind=path.suffix.lstrip(".")),
                kind,
            )
            and (suffixes is None or path.name.endswith(suffixes))
            and _path_matches_base(path, base)
        ]


@dataclass(frozen=True)
class TraceManifest:
    """Typed description of a job's per-frequency trace files.

    Args:
        files: Expected per-task trace files.
        frequencies: Mapping from one-based task number to frequency.
        groups: Receiver or wavefield groups contained in the trace product.
        simulation: Simulation JSON path associated with the traces.
        result_path: Job result directory.
        output_path: Directory containing trace outputs.
        project_path: Optional project root used for relative artifact paths.
        laplace: Mapping from one-based task number to Laplace damping value.
        components: Component labels available in the trace product.
        sources: Source ids available in the trace product.
        wavefields: Wavefield output metadata keyed by group.
        artifacts: Solver-reported artifacts associated with the run.
        run: Parsed run metadata associated with the result directory.
    """

    files: List[Path]
    frequencies: Dict[int, float]
    groups: List[str]
    simulation: Path
    result_path: Path
    output_path: Path
    project_path: Optional[Path] = None
    laplace: Dict[int, float] = field(default_factory=dict)
    components: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    wavefields: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[OutputArtifact] = field(default_factory=list)
    run: RunMetadata = field(default_factory=RunMetadata)

    @classmethod
    def from_job(
        cls,
        job,
        *,
        output: Optional["TraceOutputSpec"] = None,
        project_path: Optional[Union[str, Path]] = None,
        resolve_legacy: bool = False,
    ) -> "TraceManifest":
        """Build the expected trace manifest for a job and output spec.

        Args:
            job: Job whose outputs are being described.
            output: Optional resolved trace output spec. Defaults to receiver
                traces for ``job``.
            project_path: Optional local project root used to remap paths from
                a copied or fetched job.
            resolve_legacy: Prefer existing legacy ``receivers_*`` files when
                modern ``traces_*`` files are missing.

        Returns:
            Trace manifest with expected files, frequencies, groups, artifacts,
            and run metadata.
        """

        sim = job.simulation
        source_project = _as_path(job.project_path).resolve()
        local_project = (
            _as_path(project_path).resolve()
            if project_path is not None
            else source_project
        )
        output = job.trace_outputs if output is None else output
        result_path = cls._map_project_path(
            job._result_path, source_project, local_project
        )
        output_path = cls._map_project_path(output.path, source_project, local_project)
        frequencies = {
            index: _real_frequency(freq)
            for index, freq in enumerate(output.frequencies, start=1)
        }
        laplace = {
            index: _laplace_frequency(freq)
            for index, freq in enumerate(output.frequencies, start=1)
        }
        files = [
            output_path / f"traces_{index}.h5"
            for index in range(1, len(frequencies) + 1)
        ]
        if resolve_legacy:
            files = [cls.resolve_trace_file(path) for path in files]

        artifacts = cls._read_artifacts(result_path)
        simulation_path = cls._simulation_path(job, sim, source_project, local_project)
        return cls(
            files=files,
            frequencies=frequencies,
            groups=list(output.groups),
            simulation=simulation_path,
            result_path=result_path,
            output_path=output_path,
            project_path=local_project,
            laplace=laplace,
            components=list(output.components),
            sources=list(output.sources),
            wavefields=copy.deepcopy(output.wavefields),
            artifacts=artifacts,
            run=RunMetadata.read(result_path),
        )

    @classmethod
    def combine(
        cls,
        manifests: Iterable["TraceManifest"],
        duplicate: str = "first",
    ) -> "TraceManifest":
        """Merge manifests from multiple jobs into one frequency-ordered view.

        Args:
            manifests: Trace manifests to combine.
            duplicate: Policy for duplicate frequencies: ``"first"``,
                ``"last"``, or ``"error"``.

        Returns:
            Combined manifest sorted by frequency.

        Raises:
            ValueError: If no manifests are supplied, duplicate policy is
                invalid, group layouts differ, or duplicate frequencies are
                rejected.
        """

        manifests = list(manifests)
        if not manifests:
            raise ValueError("At least one TraceManifest is required")
        if duplicate not in {"first", "last", "error"}:
            raise ValueError("duplicate must be 'first', 'last', or 'error'")

        groups = manifests[0].groups
        for manifest in manifests[1:]:
            if manifest.groups != groups:
                raise ValueError("Cannot combine trace manifests with different groups")

        entries: Dict[float, tuple[Path, float]] = {}
        duplicates = []
        for manifest in manifests:
            ordered = sorted(
                manifest.frequencies.items(), key=lambda item: int(item[0])
            )
            for (task_index, freq), file in zip(ordered, manifest.files):
                if freq in entries:
                    duplicates.append(freq)
                    if duplicate == "error":
                        raise ValueError(f"Duplicate trace frequency: {freq}")
                    if duplicate == "first":
                        continue
                laplace = float(
                    manifest.laplace.get(
                        task_index, manifest.laplace.get(str(task_index), 0.0)
                    )
                )
                entries[freq] = (file, laplace)

        frequencies = {
            index: float(freq) for index, freq in enumerate(sorted(entries), start=1)
        }
        files = [entries[freq][0] for freq in sorted(entries)]
        laplace = {
            index: float(entries[freq][1])
            for index, freq in enumerate(sorted(entries), start=1)
        }
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
            laplace=laplace,
            components=first.components,
            sources=first.sources,
            wavefields=copy.deepcopy(first.wavefields),
            artifacts=artifacts,
            run=run,
        )

    @staticmethod
    def resolve_trace_file(path: Union[str, Path]) -> Path:
        """Resolve modern ``traces_*`` or legacy ``receivers_*`` files.

        Args:
            path: Preferred modern trace path.

        Returns:
            Existing modern path, existing legacy path, or the original modern
            path when neither exists.
        """

        path = _as_path(path)
        if path.exists():
            return path
        legacy = path.with_name(path.name.replace("traces_", "receivers_", 1))
        if legacy.exists():
            return legacy
        return path

    @property
    def packed_file(self) -> Optional[Path]:
        """Return the packed trace file named by the trace manifest.

        Returns:
            Existing packed trace path, or ``None`` when no packed trace product
            is recorded.
        """

        products = self._packed_products()
        if len(products) != 1 or not products[0][0].exists():
            return None
        return products[0][0]

    @property
    def packed_files(self) -> List[Path]:
        """Return existing packed trace files named by available manifests.

        Returns:
            Existing packed trace products. Wavefield outputs may produce one
            packed file per output name.
        """

        return [
            path for path, _freq, _laplace in self._packed_products() if path.exists()
        ]

    @property
    def packed_frequencies(self) -> Dict[int, float]:
        """Return packed trace frequencies keyed by one-based task number.

        Returns:
            Frequency mapping from the packed trace manifest, or from the packed
            HDF5 file when the manifest omits frequency rows. Returns an empty
            mapping when no usable frequency index is recorded.
        """

        products = self._packed_products()
        if not products:
            return {}
        path, frequencies, _laplace = products[0]
        return frequencies or self._packed_file_frequency_map(path)

    @property
    def packed_laplace(self) -> Dict[int, float]:
        """Return packed trace Laplace values keyed by one-based task number.

        Returns:
            Laplace damping mapping from the packed trace manifest, or an empty
            mapping when unavailable.
        """

        products = self._packed_products()
        return {} if not products else products[0][2]

    @property
    def missing_packed_frequencies(self) -> Dict[int, float]:
        """Return expected job frequencies missing from the packed product.

        Returns:
            Mapping from expected one-based task number to missing frequency.
        """

        products = self._packed_products()
        if not products or any(not path.exists() for path, _freq, _laplace in products):
            return dict(self.frequencies)
        product_frequencies = [
            frequencies or self._packed_file_frequency_map(path)
            for path, frequencies, _laplace in products
        ]
        missing = {}
        for key, frequency in self.frequencies.items():
            if any(
                not frequencies or not _frequency_values_contain(frequencies, frequency)
                for frequencies in product_frequencies
            ):
                missing[int(key)] = _real_frequency(frequency)
        return missing

    @property
    def packed_complete(self) -> bool:
        """Return whether the packed trace file covers every expected frequency.

        Returns:
            ``True`` only when a packed product exists and no expected
            frequencies are missing.
        """

        products = self._packed_products()
        if not products or any(not path.exists() for path, _freq, _laplace in products):
            return False
        return not self.missing_packed_frequencies

    def packed_incomplete_message(self) -> str:
        """Return a clear message describing missing packed frequencies.

        Returns:
            Human-readable diagnostic naming the packed product and missing
            task/frequency ranges.
        """

        missing = self.missing_packed_frequencies
        detail = _compact_frequency_ranges(missing)
        packed_files = self.packed_files
        if packed_files:
            packed_file = ", ".join(str(path) for path in packed_files)
        else:
            packed_file = str(self.output_path / "traces.h5")
        return (
            f"Packed trace product {packed_file} is missing {len(missing)} of "
            f"{len(self.frequencies)} expected frequencies: {detail}"
        )

    @property
    def existing_files(self) -> List[Path]:
        """Return existing trace files, preferring complete packed output.

        Returns:
            Existing packed product when complete; otherwise existing
            per-frequency trace shards.
        """

        packed_files = self.packed_files
        if packed_files and self.packed_complete:
            return packed_files
        return [
            self.resolve_trace_file(file)
            for file in self.files
            if self.resolve_trace_file(file).exists()
        ]

    @property
    def complete(self) -> bool:
        """Return whether every expected trace output is available on disk.

        Returns:
            ``True`` when either a complete packed product exists or every
            expected per-frequency trace file exists.
        """

        if self._packed_products():
            return self.packed_complete
        return bool(self.files) and all(
            self.resolve_trace_file(file).exists() for file in self.files
        )

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this manifest for diagnostics or artifact handoff.

        Returns:
            JSON-compatible trace manifest payload.
        """

        return {
            "schema": "frequensolve-trace-manifest-1",
            "files": [str(file) for file in self.files],
            "frequencies": self.frequencies,
            "groups": self.groups,
            "simulation": str(self.simulation),
            "result_path": str(self.result_path),
            "output_path": str(self.output_path),
            "laplace": self.laplace,
            "components": self.components,
            "sources": self.sources,
            "wavefields": copy.deepcopy(self.wavefields),
            "artifacts": [
                artifact.to_fs(self.project_path) for artifact in self.artifacts
            ],
        }

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

    @classmethod
    def _simulation_path(
        cls,
        job,
        sim,
        source_project: Path,
        project_path: Path,
    ) -> Path:
        sim_file = getattr(sim, "_file", None)
        if sim_file is not None:
            return cls._map_project_path(sim_file, source_project, project_path)

        job_file = getattr(job, "_file", None)
        if job_file is None:
            try:
                job_file = job.job_file
            except Exception:
                job_file = None
        if job_file is not None:
            job_file = _as_path(job_file)
            if job_file.exists():
                try:
                    payload = json.loads(job_file.read_text())
                except (OSError, json.JSONDecodeError):
                    payload = {}
                simulation_ref = payload.get("simulation")
                if simulation_ref:
                    simulation_path = _as_path(simulation_ref)
                    if simulation_path.is_absolute():
                        return cls._map_project_path(
                            simulation_path,
                            source_project,
                            project_path,
                        )
                    mapped = project_path / simulation_path
                    if mapped.exists():
                        return mapped
                    job_relative = job_file.parent / simulation_path
                    if job_relative.exists():
                        return job_relative
                    return mapped

        name = getattr(sim, "name", "simulation")
        return project_path / "simulations" / str(name) / f"{name}.json"

    @staticmethod
    def _read_artifacts(result_path: Path) -> List[OutputArtifact]:
        outputs = RunMetadata.read(result_path).outputs
        files = outputs.get("files", []) if outputs else []
        return [OutputArtifact.from_fs(file, result_path=result_path) for file in files]

    def _packed_products(self) -> List[tuple[Path, Dict[int, float], Dict[int, float]]]:
        manifest_files = [self.output_path / "manifest.json"]
        names = [*self.groups, *self.wavefields]
        if names:
            manifest_files.extend(
                self.output_path / str(name) / "manifest.json" for name in names
            )
        else:
            manifest_files.extend(sorted(self.output_path.glob("*/manifest.json")))

        products: List[tuple[Path, Dict[int, float], Dict[int, float]]] = []
        seen = set()
        for manifest_file in manifest_files:
            packed = self._packed_manifest_file(self.result_path, manifest_file)
            if packed is None:
                continue
            key = str(packed[0].resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            products.append(packed)
        return products

    @staticmethod
    def _packed_manifest(
        result_path: Path,
        output_path: Path,
    ) -> Optional[tuple[Path, Dict[int, float], Dict[int, float]]]:
        return TraceManifest._packed_manifest_file(
            result_path,
            output_path / "manifest.json",
        )

    @staticmethod
    def _packed_manifest_file(
        result_path: Path,
        manifest_file: Path,
    ) -> Optional[tuple[Path, Dict[int, float], Dict[int, float]]]:
        if not manifest_file.exists():
            return None
        try:
            data = json.loads(manifest_file.read_text())
        except json.JSONDecodeError:
            return None

        packed = data.get("packed", {}) if isinstance(data, Mapping) else {}
        if not isinstance(packed, Mapping):
            return None
        raw_path = packed.get("path") or packed.get("relative_path")
        if not raw_path:
            return None

        path = _as_path(raw_path)
        if not path.is_absolute():
            path = result_path / path

        frequencies: Dict[int, float] = {}
        laplace: Dict[int, float] = {}
        rows = data.get("frequencies", [])
        if isinstance(rows, list):
            sortable = [
                (index, row)
                for index, row in enumerate(rows, start=1)
                if isinstance(row, Mapping) and "frequency" in row
            ]
            sortable.sort(key=lambda item: int(item[1].get("task_id", item[0])))
            frequencies = {}
            laplace = {}
            for original_index, row in sortable:
                task_id = int(row.get("task_id", original_index))
                frequencies[task_id] = _real_frequency(row["frequency"])
                laplace[task_id] = _laplace_value(
                    row.get("laplace", _laplace_frequency(row["frequency"]))
                )
        return path, frequencies, laplace

    @staticmethod
    def _packed_file_frequency_map(path: Path) -> Dict[int, float]:
        try:
            from frequensolve.seismic.trace_store import TraceStore

            frequencies = TraceStore._read_trace_frequencies(path)
        except (OSError, KeyError, ValueError):
            return {}
        return {
            index: _real_frequency(frequency)
            for index, frequency in enumerate(frequencies, start=1)
        }


@dataclass(frozen=True)
class TraceOutputSpec:
    """Resolved trace output request for a job.

    Args:
        path: Output directory containing trace files.
        frequencies: Frequencies represented by this trace output.
        groups: Receiver or wavefield group names.
        components: Component labels available in each group.
        sources: Source ids represented in the output.
        wavefields: Optional wavefield metadata keyed by group.
    """

    path: Path
    frequencies: Sequence[Union[float, complex]]
    groups: List[str]
    components: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    wavefields: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceOutputHandle:
    """Convenience handle exposed as ``job.traces``.

    Args:
        job: Job whose receiver trace output is configured or opened.
    """

    job: Any

    def __call__(self, path: Union[str, Path] = "traces", **kwargs) -> Any:
        """Configure receiver trace output for the owning job.

        Args:
            path: Trace output directory relative to the job result directory.
            **kwargs: Additional ``TraceOutput`` options.

        Returns:
            The owning job, allowing fluent configuration.
        """

        from frequensolve.simulation.outputs import TraceOutput

        self.job.outputs.traces = TraceOutput(path=path, **kwargs)
        return self.job

    @property
    def manifest(self) -> TraceManifest:
        """Return the resolved manifest describing receiver traces.

        Returns:
            ``TraceManifest`` for the owning job's receiver trace output.
        """

        return self.job.trace_manifest

    def open(self, upscale: int = 1, project_path: Optional[Union[str, Path]] = None):
        """Open receiver traces as a ``TraceDataset``.

        Args:
            upscale: Optional trace upscaling factor.
            project_path: Optional local project root used to remap paths after
                fetching results from another project location.

        Returns:
            ``TraceDataset`` backed by current receiver trace outputs.

        Raises:
            ValueError: If trace files or frequency metadata are incomplete.
        """

        from frequensolve.seismic.traces import TraceDataset

        manifest = TraceManifest.from_job(
            self.job,
            project_path=project_path,
            resolve_legacy=True,
        )
        return TraceDataset.from_manifest(manifest, upscale=upscale)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the current receiver trace manifest.

        Returns:
            JSON-compatible trace manifest payload.
        """

        return self.manifest.to_fs()

    def __getitem__(self, key: str) -> Any:
        """Return one field from the serialized trace manifest.

        Args:
            key: Manifest field name.

        Returns:
            Value from :meth:`to_fs`.
        """

        return self.to_fs()[key]


@dataclass(frozen=True)
class WavefieldOutputHandle:
    """Convenience handle exposed as ``job.wavefields``.

    Args:
        job: Job whose wavefield trace output is configured or opened.
    """

    job: Any

    def __call__(self, *args, **kwargs) -> Any:
        """Add a wavefield output request to the owning job.

        Args:
            *args: Positional arguments forwarded to ``outputs.wavefield``.
            **kwargs: Keyword arguments forwarded to ``outputs.wavefield``.

        Returns:
            The owning job, allowing fluent configuration.
        """

        from frequensolve.simulation.outputs import wavefield

        self.job += wavefield(*args, **kwargs)
        return self.job

    @property
    def manifest(self) -> TraceManifest:
        """Return the resolved manifest describing wavefield trace files.

        Returns:
            ``TraceManifest`` for the owning job's wavefield trace output.
        """

        return self.job.wavefield_manifest

    def open(self, upscale: int = 1, project_path: Optional[Union[str, Path]] = None):
        """Open wavefield outputs as a ``TraceDataset``.

        Args:
            upscale: Optional trace upscaling factor.
            project_path: Optional local project root used to remap paths after
                fetching results from another project location.

        Returns:
            ``TraceDataset`` backed by current wavefield outputs.

        Raises:
            ValueError: If no wavefield groups are configured or trace metadata
                are incomplete.
        """

        from frequensolve.seismic.traces import TraceDataset

        manifest = TraceManifest.from_job(
            self.job,
            output=self.job.wavefield_trace_outputs,
            project_path=project_path,
            resolve_legacy=True,
        )
        if not manifest.groups:
            raise ValueError("Job has no wavefield outputs")
        return TraceDataset.from_manifest(manifest, upscale=upscale)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize the current wavefield trace manifest.

        Returns:
            JSON-compatible wavefield trace manifest payload.
        """

        return self.manifest.to_fs()

    def __getitem__(self, key: str) -> Any:
        """Return one field from the serialized wavefield manifest.

        Args:
            key: Manifest field name.

        Returns:
            Value from :meth:`to_fs`.
        """

        return self.to_fs()[key]


class JobArtifactMixin:
    """Artifact discovery and convenience handles for saved job outputs.

    The mixin exposes receiver traces, wavefield traces, VTK outputs, and
    packed-trace cache management on concrete job classes.
    """

    def expected_trace_files(self) -> List[Path]:
        """Return the task-numbered trace files expected for this job.

        Returns:
            One path per frequency task in solver task order.
        """

        return list(self.trace_manifest.files)

    def trace_outputs_exist(self) -> bool:
        """Return whether receiver traces are complete and reusable.

        Returns:
            ``True`` when all expected receiver traces exist, either as shards
            or as a complete packed trace product.
        """

        manifest = self.trace_manifest
        if manifest.packed_files and not manifest.packed_complete:
            warnings.warn(
                manifest.packed_incomplete_message(),
                RuntimeWarning,
                stacklevel=2,
            )
        if manifest.complete:
            return True
        return all(
            self._trace_output_path_for_task(task, manifest=manifest)[1]
            for task in range(1, self.n_tasks + 1)
        )

    def invalidate_trace_cache(self) -> None:
        """Remove derived trace VDS files so reads reflect current HDF5 shards.

        Returns:
            ``None``.
        """

        candidates = [
            self._result_path / "_fs_run" / "cache",
            self.trace_outputs.path,
        ]
        for directory in candidates:
            if not directory.exists():
                continue
            for path in directory.glob("*_vds.h5"):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def remove_packed_trace_products(self) -> bool:
        """Remove packed trace products so packing rebuilds from shards.

        Returns:
            ``True`` if any packed trace file or manifest was removed.
        """

        removed = False
        manifests = [self.trace_manifest]
        try:
            wavefield_manifest = self.wavefield_manifest
        except Exception:
            wavefield_manifest = None
        if wavefield_manifest is not None and wavefield_manifest.groups:
            manifests.append(wavefield_manifest)

        for manifest in manifests:
            candidates = [
                *manifest.packed_files,
                manifest.output_path / "traces.h5",
                manifest.output_path / "manifest.json",
                *[
                    manifest.output_path / str(name) / "manifest.json"
                    for name in [*manifest.groups, *manifest.wavefields]
                ],
            ]
            for path in candidates:
                if path is None:
                    continue
                try:
                    Path(path).unlink()
                    removed = True
                except FileNotFoundError:
                    pass

        if removed:
            self.invalidate_trace_cache()
        return removed

    @property
    def trace_manifest(self) -> TraceManifest:
        """Return the receiver trace manifest expected for this job.

        Returns:
            ``TraceManifest`` for receiver traces.
        """

        return TraceManifest.from_job(self)

    @property
    def wavefield_manifest(self) -> TraceManifest:
        """Return the wavefield trace manifest expected for this job.

        Returns:
            ``TraceManifest`` for wavefield outputs.
        """

        return TraceManifest.from_job(self, output=self.wavefield_trace_outputs)

    @property
    def vtk_outputs(self) -> dict:
        """Return configured VTK/visualization outputs.

        Returns:
            Mapping from visualization output name to output path.
        """
        self.outputs.ensure_unique_names()
        return {out.name: self._result_path / out.path for out in self.outputs.vtk}

    @property
    def paraview_outputs(self) -> dict:
        """Return visualization outputs using the historical property name."""

        return self.vtk_outputs

    @property
    def trace_path(self) -> Path:
        """Return the resolved receiver trace output directory.

        Returns:
            Path where receiver traces are expected.
        """
        return self.trace_outputs.path

    @property
    def trace_outputs(self) -> TraceOutputSpec:
        """Return the resolved receiver trace output specification.

        Returns:
            ``TraceOutputSpec`` containing output path, job frequencies,
            receiver groups, components, and source ids.
        """
        sim = self.simulation
        groups = []
        components = []
        sources = []

        for group in sim.acquisition.receiver_groups:
            groups.append(group.name)
            for component in group.device.components:
                components.append(f"{group.name}:{component.name}")

        for source_id in sim.acquisition.source_field_ids():
            sources.append(str(source_id))

        return TraceOutputSpec(
            path=self._result_path / self.outputs.traces.path,
            frequencies=self.f_list,
            groups=groups,
            components=components,
            sources=sources,
        )

    @property
    def traces(self) -> TraceOutputHandle:
        """Return the receiver trace handle for this job.

        Returns:
            ``TraceOutputHandle`` used to configure, inspect, or open traces.
        """
        return TraceOutputHandle(self)

    @property
    def wavefield_trace_outputs(self) -> TraceOutputSpec:
        """Return the resolved wavefield trace output specification.

        Returns:
            ``TraceOutputSpec`` containing shared output path, frequencies,
            wavefield groups, components, sources, and grid metadata.

        Raises:
            ValueError: If any wavefield output is missing a grid or configured
                wavefield outputs use more than one path.
        """

        groups = []
        components = []
        sources = set()
        wavefields = {}
        output_paths = set()

        self.outputs.ensure_unique_names()
        for out in self.outputs.wavefields:
            if out.grid is None:
                raise ValueError("WavefieldOutput requires a grid")
            output_paths.add(str(out.path))
            fields = out.fields if out.fields is not None else ["primary"]
            component_names = out.component_names
            component_specs = out.component_payloads()
            components_for_group = [
                f"{out.name}:{component_name}" for component_name in component_names
            ]
            source_ids = (
                [str(source) for source in out.sources]
                if out.sources is not None
                else [
                    str(source_id)
                    for source_id in self.simulation.acquisition.source_field_ids()
                ]
            )
            groups.append(out.name)
            components.extend(components_for_group)
            sources.update(source_ids)
            wavefield = {
                "name": out.name,
                "path": str(self._result_path / out.path),
                "fields": list(fields),
                "components": components_for_group,
                "component_names": component_names,
                "component_specs": component_specs,
                "sources": source_ids,
                "grid": copy.deepcopy(out.grid),
            }
            if out.device is not None:
                wavefield["device"] = out.device.to_fs()
            wavefields[out.name] = wavefield

        if len(output_paths) > 1:
            raise ValueError(
                "job.wavefields.open() requires all wavefield outputs to share "
                "one path"
            )
        output_path = (
            Path(next(iter(output_paths)))
            if output_paths
            else (
                Path(self.outputs.wavefields[0].path)
                if self.outputs.wavefields
                else Path("wavefields")
            )
        )

        return TraceOutputSpec(
            path=self._result_path / output_path,
            frequencies=self.f_list,
            groups=groups,
            components=components,
            sources=sorted(sources, key=lambda value: int(value)),
            wavefields=wavefields,
        )

    @property
    def wavefields(self) -> WavefieldOutputHandle:
        """Return the wavefield output handle for this job.

        Returns:
            ``WavefieldOutputHandle`` used to configure, inspect, or open
            wavefield traces.
        """

        return WavefieldOutputHandle(self)

    @property
    def wavefield_outputs(self) -> dict:
        """Return legacy wavefield output metadata.

        Returns:
            Mapping from wavefield output name to solver output metadata.

        Raises:
            ValueError: If any wavefield output is missing a grid.
        """
        wave_out = {}
        self.outputs.ensure_unique_names()
        for out in self.outputs.wavefields:
            if out.grid is None:
                raise ValueError("WavefieldOutput requires a grid")
            fields = out.fields if out.fields is not None else ["primary"]
            component_names = out.component_names
            component_specs = out.component_payloads()
            components = [
                f"{out.name}:{component_name}" for component_name in component_names
            ]
            sources = (
                [str(source) for source in out.sources]
                if out.sources is not None
                else [
                    str(source_id)
                    for source_id in self.simulation.acquisition.source_field_ids()
                ]
            )
            wave_out[out.name] = {
                "domain": (self.__class__.__name__,),
                "path": str(self._result_path / out.path),
                "frequencies": self.f_list,
                "grid": copy.deepcopy(out.grid),
                "fields": list(fields),
                "components": components,
                "component_names": component_names,
                "component_specs": component_specs,
                "sources": sources,
            }
            if out.device is not None:
                wave_out[out.name]["device"] = out.device.to_fs()
        return wave_out

    @staticmethod
    def _legacy_trace_file(path: Path) -> Path:
        return path.with_name(path.name.replace("traces_", "receivers_", 1))

    @classmethod
    def _trace_file_exists(cls, path: Path) -> bool:
        return path.exists() or cls._legacy_trace_file(path).exists()

    def _packed_trace_has_current_task(
        self,
        task: int,
        *,
        manifest: Optional[TraceManifest] = None,
    ) -> bool:
        manifest = self.trace_manifest if manifest is None else manifest
        if not manifest.packed_files:
            return False
        if task < 1 or task > self.n_tasks:
            return False
        packed_frequencies = manifest.packed_frequencies
        if not packed_frequencies:
            return False
        expected_frequency = self._real_frequency_value(self.f_list[task - 1])
        return np.isclose(
            list(packed_frequencies.values()),
            expected_frequency,
            rtol=0.0,
            atol=1.0e-9,
        ).any()

    def _candidate_frequency_trace_files(
        self,
        task: int,
        *,
        manifest: Optional[TraceManifest] = None,
    ) -> List[Path]:
        manifest = self.trace_manifest if manifest is None else manifest
        files = list(manifest.files)
        candidates: List[Path] = []
        if 1 <= task <= len(files):
            path = Path(files[task - 1])
            candidates.extend([path, self._legacy_trace_file(path)])

        def add_shard_dir(shard_dir: Path) -> None:
            if not shard_dir.exists():
                return
            modern = sorted(shard_dir.glob("f_*.h5"))
            files = modern
            if not files:
                files = [
                    path
                    for pattern in (
                        "traces_*.h5",
                        "receivers_*.h5",
                        "trace_frequency_*.h5",
                    )
                    for path in sorted(shard_dir.glob(pattern))
                ]
            candidates.extend(files)

        def add_root(root: Path) -> None:
            for shard_dir in (
                root / "shards",
                root / "traces" / "shards",
                root / "wavefields" / "shards",
            ):
                add_shard_dir(shard_dir)
            if root.exists():
                for pattern in (
                    "f_*.h5",
                    "traces_*.h5",
                    "receivers_*.h5",
                    "trace_frequency_*.h5",
                ):
                    candidates.extend(sorted(root.glob(pattern)))

        roots = [manifest.output_path, manifest.result_path]
        for root in roots:
            add_root(root)
        for group in [*manifest.groups, *manifest.wavefields]:
            group_dir = manifest.output_path / str(group)
            add_shard_dir(group_dir)

        out: List[Path] = []
        seen = set()
        for path in candidates:
            key = str(Path(path).resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            out.append(Path(path))
        return out

    def _matching_frequency_trace_file(
        self,
        task: int,
        *,
        manifest: Optional[TraceManifest] = None,
    ) -> Optional[Path]:
        if task < 1 or task > self.n_tasks:
            return None
        expected_frequency = self._real_frequency_value(self.f_list[task - 1])
        from frequensolve.seismic.trace_store import TraceStore

        for path in self._candidate_frequency_trace_files(task, manifest=manifest):
            if not path.exists():
                continue
            try:
                if TraceStore._is_packed_trace_file(path):
                    continue
                frequencies = TraceStore._read_trace_frequencies(path)
            except (OSError, KeyError, ValueError):
                continue
            if any(
                np.isclose(
                    float(frequency),
                    expected_frequency,
                    rtol=0.0,
                    atol=1.0e-9,
                )
                for frequency in frequencies
            ):
                return path
        return None

    def _trace_output_path_for_task(
        self,
        task: int,
        *,
        manifest: Optional[TraceManifest] = None,
    ) -> tuple[Path, bool]:
        manifest = self.trace_manifest if manifest is None else manifest
        files = list(manifest.files)
        path = Path(files[task - 1])
        existing = path if path.exists() else self._legacy_trace_file(path)
        if existing.exists():
            return existing, True
        shard = self._matching_frequency_trace_file(task, manifest=manifest)
        if shard is not None:
            return shard, True
        if manifest.packed_complete and self._packed_trace_has_current_task(
            task,
            manifest=manifest,
        ):
            packed_files = manifest.packed_files
            if packed_files:
                return packed_files[0], True
        return path, False

    def _stored_trace_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.project_path))
        except Exception:
            return str(path)

    def _resolve_stored_trace_path(self, value: Any) -> Optional[Path]:
        if value is None:
            return None
        path = Path(str(value))
        if not path.is_absolute():
            path = self.project_path / path
        path = self._legacy_trace_file(path) if not path.exists() else path
        return path


def _real_frequency(value: Union[float, complex]) -> float:
    if isinstance(value, complex):
        return float(value.real)
    if isinstance(value, np.generic):
        return _real_frequency(value.item())
    return float(value)


def _laplace_frequency(value: Union[float, complex]) -> float:
    if isinstance(value, complex):
        return float(value.imag)
    if isinstance(value, np.generic):
        return _laplace_frequency(value.item())
    return 0.0


def _laplace_value(value: Any) -> float:
    if isinstance(value, np.generic):
        return _laplace_value(value.item())
    if isinstance(value, complex):
        return float(value.imag)
    return float(value)


def _frequency_values_contain(
    values: Mapping[Any, float],
    frequency: Union[float, complex],
) -> bool:
    expected = _real_frequency(frequency)
    return any(
        np.isclose(_real_frequency(value), expected, rtol=0.0, atol=1.0e-9)
        for value in values.values()
    )


def _format_frequency(value: float) -> str:
    return f"{float(value):.6g} Hz"


def _compact_frequency_ranges(
    values: Mapping[int, float],
    *,
    max_ranges: int = 6,
) -> str:
    items = sorted((int(task), float(freq)) for task, freq in values.items())
    if not items:
        return "none"

    ranges: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for task, frequency in items:
        if current and task != current[-1][0] + 1:
            ranges.append(current)
            current = []
        current.append((task, frequency))
    if current:
        ranges.append(current)

    parts = []
    for group in ranges[:max_ranges]:
        first_task, first_frequency = group[0]
        last_task, last_frequency = group[-1]
        if first_task == last_task:
            parts.append(f"task {first_task}: {_format_frequency(first_frequency)}")
        else:
            first = _format_frequency(first_frequency)
            last = _format_frequency(last_frequency)
            parts.append(f"tasks {first_task}-{last_task}: " f"{first}-{last}")
    if len(ranges) > max_ranges:
        parts.append(f"+{len(ranges) - max_ranges} more ranges")
    return "; ".join(parts)


def _as_path(path: Union[str, Path]) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _relative_to(path: Path, base: Optional[Path]) -> str:
    if base is None:
        return str(path)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path)


def _normalize_base(base: Optional[Union[str, Path]]) -> Optional[str]:
    if base is None:
        return None
    normalized = Path(base).stem if isinstance(base, Path) else Path(str(base)).stem
    normalized = normalized.strip()
    return normalized or None


def _path_matches_base(path: Path, base: Optional[str]) -> bool:
    if base is None:
        return True
    stem = path.stem
    if stem == base:
        return True
    suffix = stem.removeprefix(f"{base}_")
    return suffix != stem and len(suffix) == 5 and suffix.isdigit()


def _normalize_kind(kind: Optional[str]) -> Optional[str]:
    if kind is None:
        return None
    normalized = str(kind).strip().lower()
    return normalized or None


def _kind_suffixes(kind: Optional[str]) -> tuple[str, ...]:
    normalized = _normalize_kind(kind)
    if normalized is None:
        return ()
    if normalized == "vtk":
        return (".vtk", ".vtu", ".vtr", ".vtp", ".vts")
    if normalized in {
        "vtu",
        "vtr",
        "vtp",
        "vts",
        "vtk",
        "h5",
        "hdf5",
        "json",
        "xmf",
        "xmdf",
    }:
        suffix = ".h5" if normalized == "hdf5" else f".{normalized}"
        return (suffix,)
    return ()


def _path_matches_kind(path: Path, kind: Optional[str]) -> bool:
    normalized = _normalize_kind(kind)
    if normalized is None:
        return True
    suffix = path.suffix.lower()
    return suffix in _kind_suffixes(normalized)


def _path_identity_key(path: Path) -> tuple[Any, ...]:
    """Return a stable key for deduplicating output paths.

    Existing files are keyed by filesystem identity so aliases such as
    ``ParaView`` and ``paraview`` collapse when they point at the same file.
    Non-existing files fall back to their normalized path string.
    """

    try:
        stat = path.stat()
    except OSError:
        return ("path", str(path.resolve(strict=False)))
    return ("stat", stat.st_dev, stat.st_ino)


def _artifact_matches_kind(artifact: "OutputArtifact", kind: Optional[str]) -> bool:
    normalized = _normalize_kind(kind)
    if normalized is None:
        return True
    if artifact.kind is not None and str(artifact.kind).strip().lower() == normalized:
        return True
    return _path_matches_kind(artifact.path, normalized)
