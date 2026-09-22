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
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)

import numpy as np

from frequensolve.simulation.artifact_catalog import load_artifact_catalog
from frequensolve.simulation.artifact_contract import (
    ARTIFACT_CONTRACT_VERSION,
    ArtifactCatalog,
    ArtifactContractError,
    ArtifactRecord,
    ArtifactRequest,
)
from frequensolve.simulation.task_index import (
    TaskIndex,
    TaskIndexEntry,
    load_task_catalog,
)

if TYPE_CHECKING:
    from frequensolve.seismic.trace_pack import TracePackManifest

__all__ = [
    "RunMetadata",
    "TraceManifest",
    "TraceOutputHandle",
    "TraceOutputSpec",
    "WavefieldOutputHandle",
]


@dataclass(frozen=True)
class RunMetadata:
    """Logical task artifacts and Python orchestration state for one run."""

    state: Dict[str, Any] = field(default_factory=dict)
    result_path: Optional[Path] = None
    artifacts: tuple[ArtifactRecord, ...] = ()
    task_status: Dict[int, str] = field(default_factory=dict)
    tasks: Mapping[int, TaskIndexEntry] = field(default_factory=dict)

    @classmethod
    def read(
        cls, result_path: Union[str, Path], *, tasks: Iterable[int] = ()
    ) -> "RunMetadata":
        """Read the fixed task index and Python state without discovery."""

        result_path = _as_path(result_path)
        state_path = result_path / "_fs_python_run.json"
        try:
            state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        except json.JSONDecodeError:
            state = {}
        if not isinstance(state, Mapping):
            state = {}
        index_path = result_path / "_fs_run" / "tasks.h5"
        index = TaskIndex.read(result_path) if index_path.is_file() else None
        if index is None:
            catalog = ArtifactCatalog.read_task_results(result_path, tasks=tasks)
            artifacts = tuple(
                artifact
                for result in catalog.results.values()
                for artifact in result.artifacts
            )
            task_status = {
                task: result.state for task, result in catalog.results.items()
            }
            indexed_tasks: Mapping[int, TaskIndexEntry] = {}
        else:
            artifacts = index.artifacts
            task_status = {task: entry.status for task, entry in index.tasks.items()}
            indexed_tasks = index.tasks
        return cls(
            state=dict(state),
            result_path=result_path,
            artifacts=artifacts,
            task_status=task_status,
            tasks=indexed_tasks,
        )

    @property
    def successful(self) -> bool:
        """Return whether Python state or every indexed task is successful."""

        if self.state:
            return self.state.get("status") in {"completed", "skipped"}
        return bool(self.task_status) and all(
            status in {"success", "skipped"} for status in self.task_status.values()
        )

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
        deduped = []
        seen = set()
        for path in files:
            key = _path_identity_key(path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return deduped


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
        time_reconstruction: Optional frequency-to-time reconstruction settings.
        artifacts: Solver-reported artifacts associated with the run.
        run: Parsed run metadata associated with the result directory.
        artifact_contract: Producer contract used to resolve trace payloads.
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
    time_reconstruction: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    run: RunMetadata = field(default_factory=RunMetadata)
    artifact_contract: Optional[str] = None
    pack: Optional[TracePackManifest] = None
    wavefield_packs: tuple[TracePackManifest, ...] = ()

    @classmethod
    def from_job(
        cls,
        job,
        *,
        output: Optional["TraceOutputSpec"] = None,
        project_path: Optional[Union[str, Path]] = None,
    ) -> "TraceManifest":
        """Build the expected trace manifest for a job and output spec.

        Args:
            job: Job whose outputs are being described.
            output: Optional resolved trace output spec. Defaults to receiver
                traces for ``job``.
            project_path: Optional local project root used to remap paths from
                a copied or fetched job.
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
        receiver_output = output is None
        output = job.trace_outputs if output is None else output
        assert output is not None
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
        run = RunMetadata.read(result_path)
        fingerprint_reader = getattr(job, "_task_reuse_fingerprints", None)
        fingerprints = fingerprint_reader() if callable(fingerprint_reader) else None
        pack = (
            cls._receiver_pack(
                result_path,
                output.frequencies,
                fingerprints=fingerprints,
            )
            if receiver_output
            else None
        )
        wavefield_packs: tuple[TracePackManifest, ...] = ()
        if not receiver_output:
            wavefield_packs = cls._sampled_wavefield_packs(
                result_path, output.frequencies, fingerprints=fingerprints
            )
        if wavefield_packs:
            files = list(
                dict.fromkeys(
                    segment.path
                    for product in wavefield_packs
                    for segment in product.segments
                )
            )
            artifacts = [
                segment.artifact
                for product in wavefield_packs
                for segment in product.segments
            ]
        elif pack is not None:
            entries = pack.entries_for(output.frequencies)
            segments = pack.selected_segments(entries)
            files = [segment.path for segment in segments]
            artifacts = [pack.artifact, *(segment.artifact for segment in segments)]
        elif receiver_output:
            files, artifacts = cls._receiver_trace_artifacts(
                result_path,
                output.frequencies,
                fingerprints=fingerprints,
            )
        else:
            files, artifacts = cls._wavefield_trace_artifacts(
                result_path,
                output.frequencies,
                fingerprints=fingerprints,
            )
        artifact_contract = ARTIFACT_CONTRACT_VERSION
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
            time_reconstruction=copy.deepcopy(getattr(job, "time_reconstruction", {})),
            artifacts=artifacts,
            run=run,
            artifact_contract=artifact_contract,
            pack=pack,
            wavefield_packs=wavefield_packs,
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
                state=state,
                result_path=run.result_path,
                artifacts=run.artifacts,
                task_status=run.task_status,
                tasks=run.tasks,
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
            time_reconstruction=copy.deepcopy(first.time_reconstruction),
            artifacts=artifacts,
            run=run,
            artifact_contract=(
                first.artifact_contract
                if all(
                    manifest.artifact_contract == first.artifact_contract
                    for manifest in manifests
                )
                else None
            ),
            pack=None,
        )

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
        """Return packed trace frequencies keyed by packed-entry number.

        Returns:
            Frequency mapping from the packed trace manifest. Unique solver task
            ids are retained as keys; accumulated manifests with repeated task
            ids use unique packed dataset/entry numbers instead. Falls back to
            the packed HDF5 file when the manifest omits frequency rows.
        """

        products = self._packed_products()
        if not products:
            return {}
        return {
            key: value
            for _path, frequencies, _laplace in products
            for key, value in frequencies.items()
        }

    @property
    def packed_laplace(self) -> Dict[int, float]:
        """Return packed trace Laplace values keyed by packed-entry number.

        Returns:
            Laplace damping mapping from the packed trace manifest, or an empty
            mapping when unavailable.
        """

        products = self._packed_products()
        return {
            key: value
            for _path, _frequencies, laplace in products
            for key, value in laplace.items()
        }

    @property
    def missing_packed_frequencies(self) -> Dict[int, float]:
        """Return expected job frequencies missing from the packed product.

        Returns:
            Mapping from expected one-based task number to missing frequency.
        """

        products = self._packed_products()
        if not products or any(not path.exists() for path, _freq, _laplace in products):
            return dict(self.frequencies)
        product_frequencies = self.packed_frequencies
        product_laplace = self.packed_laplace
        missing = {}
        for key, frequency in self.frequencies.items():
            expected_laplace = self.laplace.get(
                key,
                self.laplace.get(str(key)),
            )
            if not _frequency_laplace_values_contain(
                product_frequencies,
                product_laplace,
                frequency,
                expected_laplace,
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
        return [Path(file) for file in self.files if Path(file).exists()]

    @property
    def complete(self) -> bool:
        """Return whether every expected trace output is available on disk.

        Returns:
            ``True`` when either a complete packed product exists or every
            expected per-frequency trace file exists.
        """

        if self._packed_products():
            return self.packed_complete
        return bool(self.files) and all(Path(file).exists() for file in self.files)

    def to_fs(self) -> Dict[str, Any]:
        """Serialize this manifest for diagnostics or artifact handoff.

        Returns:
            JSON-compatible trace manifest payload.
        """

        payload = {
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
            "time_reconstruction": copy.deepcopy(self.time_reconstruction),
            "artifacts": [artifact.to_fs() for artifact in self.artifacts],
        }
        if self.artifact_contract is not None:
            payload["artifact_contract"] = self.artifact_contract
        return payload

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
    def _receiver_pack(
        result_path: Path,
        expected_frequencies: Sequence[Union[float, complex]],
        *,
        fingerprints: Optional[Mapping[str, str]] = None,
    ) -> Optional[TracePackManifest]:
        """Resolve one complete receiver pack through the fixed operation result."""

        from frequensolve.seismic.trace_pack import TracePackManifest

        tasks = range(1, len(expected_frequencies) + 1)
        catalog = load_artifact_catalog(
            result_path,
            tasks=tasks,
            operations=("pack",),
        )
        task_pack = TracePackManifest.from_tasks(
            catalog, expected_frequencies, fingerprints=fingerprints
        )
        if task_pack is not None:
            return task_pack
        if "pack" not in catalog.operations:
            return None
        operation_result = catalog.operations["pack"]
        if fingerprints is not None and any(
            operation_result.fingerprints.get(name) != digest
            for name, digest in fingerprints.items()
            if name != "compatibility"
        ):
            return None
        request = ArtifactRequest(
            id="traces",
            role="simulated_traces",
            representations=("packed_manifest",),
            retention="durable",
        )
        records = catalog.select(request, operation="pack")
        if not records:
            return None
        if len(records) != 1:
            raise ArtifactContractError(
                f"pack operation contains {len(records)} receiver trace manifests"
            )
        pack = TracePackManifest.read(records[0], catalog=catalog)
        entries = pack.entries_for(expected_frequencies)
        if not entries:
            return None
        for entry in entries:
            if not TraceManifest._task_fingerprints_match(
                catalog.task_catalog, entry.task, fingerprints
            ):
                return None
            if (
                fingerprints is not None
                and "compatibility" in fingerprints
                and not any(
                    record.id == "traces"
                    and record.generation == entry.source_generation
                    for record in catalog.artifacts_for_task(entry.task)
                )
            ):
                return None
        return pack

    @classmethod
    def _receiver_trace_artifacts(
        cls,
        result_path: Path,
        expected_frequencies: Sequence[Union[float, complex]],
        *,
        fingerprints: Optional[Mapping[str, str]] = None,
    ) -> tuple[List[Path], List[ArtifactRecord]]:
        """Resolve receiver shards exclusively from committed v2 task results."""

        tasks = range(1, len(expected_frequencies) + 1)
        catalog = load_task_catalog(result_path, tasks=tasks)
        committed_tasks = (
            catalog.tasks if isinstance(catalog, TaskIndex) else catalog.results
        )
        if not committed_tasks:
            # An unstarted job has no committed products yet. Keep the
            # manifest useful for configuration/currentness inspection without
            # inventing physical filenames. Once any task commits, gaps are
            # contract errors rather than inferred paths.
            return [], []
        request = ArtifactRequest(
            id="traces",
            role="simulated_traces",
            representations=("hdf5_shard",),
            retention="durable",
        )
        records: List[ArtifactRecord] = []
        for task, expected in enumerate(expected_frequencies, start=1):
            cls._require_task_frequency(catalog, task, expected)
            if not cls._task_fingerprints_match(catalog, task, fingerprints):
                return [], []
            records.append(catalog.require_one(request, task=task))
        return [record.path for record in records], records

    @staticmethod
    def _sampled_wavefield_packs(
        result_path: Path,
        frequencies: Sequence[Union[float, complex]],
        *,
        fingerprints: Optional[Mapping[str, str]] = None,
    ) -> tuple[TracePackManifest, ...]:
        """Resolve every sampled wavefield family from explicit task/pack records."""
        from frequensolve.seismic.trace_pack import TracePackManifest

        catalog = load_artifact_catalog(
            result_path, tasks=range(1, len(frequencies) + 1), operations=("pack",)
        )
        families = dict.fromkeys(
            record.id
            for record in catalog.artifacts_for_task(1)
            if record.role == "sampled_wavefield"
        )
        products = []
        for family in families:
            product = TracePackManifest.from_tasks(
                catalog, frequencies, family_id=family, fingerprints=fingerprints
            )
            if product is None:
                matches = catalog.select(
                    ArtifactRequest(
                        id=family,
                        role="sampled_wavefield",
                        representations=("packed_manifest",),
                        retention="durable",
                    ),
                    operation="pack",
                )
                if len(matches) != 1:
                    return ()
                product = TracePackManifest.read(matches[0], catalog=catalog)
                entries = product.entries_for(frequencies)
                if not entries:
                    return ()
                for entry in entries:
                    if not TraceManifest._task_fingerprints_match(
                        catalog.task_catalog, entry.task, fingerprints
                    ):
                        return ()
                    if not any(
                        record.id == family
                        and record.generation == entry.source_generation
                        for record in catalog.artifacts_for_task(entry.task)
                    ):
                        return ()
            products.append(product)
        return tuple(products)

    @classmethod
    def _wavefield_trace_artifacts(
        cls,
        result_path: Path,
        expected_frequencies: Sequence[Union[float, complex]],
        *,
        fingerprints: Optional[Mapping[str, str]] = None,
    ) -> tuple[List[Path], List[ArtifactRecord]]:
        """Resolve retained wavefields from task-local logical records."""

        tasks = range(1, len(expected_frequencies) + 1)
        catalog = load_task_catalog(result_path, tasks=tasks)
        committed_tasks = (
            catalog.tasks if isinstance(catalog, TaskIndex) else catalog.results
        )
        if not committed_tasks:
            return [], []
        request = ArtifactRequest(
            role="wavefield",
            representations=("collection_manifest", "hdf5_container"),
            retention="durable",
        )
        records: List[ArtifactRecord] = []
        for task, expected in enumerate(expected_frequencies, start=1):
            cls._require_task_frequency(catalog, task, expected)
            if not cls._task_fingerprints_match(catalog, task, fingerprints):
                return [], []
            matches = catalog.select(
                ArtifactRequest(
                    role="sampled_wavefield",
                    representations=("hdf5_shard",),
                    retention="durable",
                ),
                task=task,
            )
            if not matches:
                matches = catalog.select(request, task=task)
            if not matches:
                raise ArtifactContractError(
                    f"task {task} has no retained wavefield artifact"
                )
            records.extend(matches)
        return [record.path for record in records], records

    @staticmethod
    def _require_task_frequency(
        catalog: Union[TaskIndex, ArtifactCatalog],
        task: int,
        expected: Union[float, complex],
    ) -> None:
        if isinstance(catalog, TaskIndex):
            result_frequency = (
                None if task not in catalog.tasks else catalog.tasks[task].frequency
            )
        else:
            result = catalog.results.get(task)
            result_frequency = None if result is None else result.partition.frequency
        if result_frequency is None:
            raise ArtifactContractError(f"task {task} has no committed result")
        if result_frequency != complex(expected):
            raise ArtifactContractError(
                f"task {task} result frequency {result_frequency!r} does not match "
                f"expected frequency {complex(expected)!r}"
            )

    @staticmethod
    def _task_fingerprints_match(
        catalog: Union[TaskIndex, ArtifactCatalog],
        task: int,
        expected: Optional[Mapping[str, str]],
    ) -> bool:
        if expected is None:
            return True
        if isinstance(catalog, TaskIndex):
            entry = catalog.tasks.get(task)
            actual = None if entry is None else entry.fingerprints
        else:
            result = catalog.results.get(task)
            actual = None if result is None else result.fingerprints
        return actual is not None and all(
            actual.get(name) == digest for name, digest in expected.items()
        )

    def _packed_products(self) -> List[tuple[Path, Dict[int, float], Dict[int, float]]]:
        if self.pack is None:
            return []
        by_segment: Dict[str, List[Any]] = {}
        for entry in self.pack.entries:
            if entry.task in self.frequencies:
                by_segment.setdefault(entry.segment_id, []).append(entry)
        products = []
        for segment in self.pack.segments:
            entries = by_segment.get(segment.id, [])
            if not entries:
                continue
            products.append(
                (
                    segment.path,
                    {entry.task: float(entry.frequency.real) for entry in entries},
                    {entry.task: float(entry.frequency.imag) for entry in entries},
                )
            )
        return products


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
            for variant_name, _ in _spectral_derivative_variants(self, group.name):
                groups.append(variant_name)
                for component in group.device.output_components():
                    components.append(f"{variant_name}:{component.name}")

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

        wavefields = self._wavefield_metadata()
        output_paths = {str(out.path) for out in self.outputs.wavefields}
        if len(output_paths) > 1:
            raise ValueError(
                "job.wavefields.open() requires all wavefield outputs to share one path"
            )
        output_path = Path(next(iter(output_paths), "wavefields"))
        sources = {source for item in wavefields.values() for source in item["sources"]}
        return TraceOutputSpec(
            path=self._result_path / output_path,
            frequencies=self.f_list,
            groups=list(wavefields),
            components=[
                component
                for item in wavefields.values()
                for component in item["components"]
            ],
            sources=sorted(sources, key=lambda value: int(value)),
            wavefields={
                name: {"name": name, **item} for name, item in wavefields.items()
            },
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
        return {
            name: {
                "domain": (self.__class__.__name__,),
                "frequencies": self.f_list,
                **item,
            }
            for name, item in self._wavefield_metadata().items()
        }

    def _wavefield_metadata(self) -> dict:
        """Build shared metadata for legacy and trace wavefield views."""
        wave_out = {}
        self.outputs.ensure_unique_names()
        for out in self.outputs.wavefields:
            if out.grid is None:
                raise ValueError("WavefieldOutput requires a grid")
            fields = out.fields if out.fields is not None else ["primary"]
            component_names = out.component_names
            component_specs = out.component_payloads()
            sources = (
                [str(source) for source in out.sources]
                if out.sources is not None
                else [
                    str(source_id)
                    for source_id in self.simulation.acquisition.source_field_ids()
                ]
            )
            for variant_name, derivative_order in _spectral_derivative_variants(
                self, out.name
            ):
                components = [
                    f"{variant_name}:{component_name}"
                    for component_name in component_names
                ]
                wave_out[variant_name] = {
                    "path": str(self._result_path / out.path),
                    "grid": copy.deepcopy(out.grid),
                    "fields": list(fields),
                    "components": components,
                    "component_names": component_names,
                    "component_specs": component_specs,
                    "sources": sources,
                }
                if derivative_order:
                    wave_out[variant_name]["base_wavefield"] = out.name
                    wave_out[variant_name]["phase_derivative_order"] = derivative_order
                if out.properties:
                    wave_out[variant_name]["requested_properties"] = list(
                        out.properties
                    )
                    wave_out[variant_name]["property_output"] = "packed_static"
                    wave_out[variant_name]["properties"] = {
                        name: {
                            "dataset": f"/properties/{name}",
                            "static": True,
                        }
                        for name in out.properties
                    }
                if out.device is not None:
                    wave_out[variant_name]["device"] = out.device.to_fs()
        return wave_out

    def _trace_output_path_for_task(
        self,
        task: int,
        *,
        manifest: Optional[TraceManifest] = None,
    ) -> tuple[Path, bool]:
        del manifest
        result = self._committed_task_result(task)
        if result is None:
            return self.trace_outputs.path, False
        catalog = load_task_catalog(self._result_path, tasks=(task,))
        try:
            record = catalog.require_one(
                ArtifactRequest(
                    id="traces",
                    role="simulated_traces",
                    representations=("hdf5_shard",),
                    retention="durable",
                ),
                task=task,
            )
        except ArtifactContractError:
            return self.trace_outputs.path, False
        return record.path, record.path.is_file()

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
        return path


def _real_frequency(value: Union[float, complex]) -> float:
    if isinstance(value, complex):
        return float(value.real)
    if isinstance(value, np.generic):
        return _real_frequency(value.item())
    return float(value)


def _spectral_derivative_variants(job: Any, base_name: str) -> List[tuple[str, int]]:
    """Return base and solver-generated spectral derivative dataset names."""

    workflow = str(getattr(job, "workflow", ""))
    if workflow not in {"forward_df", "forward_ds"}:
        return [(str(base_name), 0)]

    order = getattr(job, "phase_derivatives", None)
    if order is None or int(order) == 0:
        order = getattr(job, "derivative_order", 1)
    order = max(1, int(order if order is not None else 1))
    axis = "f" if workflow == "forward_df" else "s"
    variants = [(str(base_name), 0)]
    for derivative_order in range(1, order + 1):
        suffix = f"_d{derivative_order}{axis}" if derivative_order > 1 else f"_d{axis}"
        variants.append((f"{base_name}{suffix}", derivative_order))
    return variants


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


def _frequency_laplace_values_contain(
    values: Mapping[Any, float],
    laplace_values: Mapping[Any, float],
    frequency: Union[float, complex],
    laplace: Optional[float] = None,
) -> bool:
    expected = _real_frequency(frequency)
    for key, value in values.items():
        if not np.isclose(
            _real_frequency(value),
            expected,
            rtol=0.0,
            atol=1.0e-9,
        ):
            continue
        if laplace is None or not laplace_values:
            return True
        actual_laplace = laplace_values.get(
            key,
            laplace_values.get(str(key)),
        )
        if actual_laplace is None:
            return True
        if np.isclose(
            _laplace_value(actual_laplace),
            _laplace_value(laplace),
            rtol=0.0,
            atol=1.0e-12,
        ):
            return True
    return False


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
            parts.append(f"tasks {first_task}-{last_task}: {first}-{last}")
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
    if normalized == "xdmf":
        return (".xmf",)
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


def _artifact_matches_kind(artifact: ArtifactRecord, kind: Optional[str]) -> bool:
    normalized = _normalize_kind(kind)
    if normalized is None:
        return True
    if artifact.representation.strip().lower() == normalized:
        return True
    if artifact.role.strip().lower() == normalized:
        return True
    return _path_matches_kind(artifact.path, normalized)
