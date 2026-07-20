"""High-level accessors for solver trace output datasets."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from frequensolve.seismic.trace_store import TraceStore, TraceSummary
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.jobs.artifacts import TraceManifest
from frequensolve.simulation.sampling import UniformSweepSampling

__all__ = ["TraceDataset"]


@dataclass
class TraceDataset:
    """Small facade for reading solver trace outputs.

    This is the preferred replacement for direct `TraceStore` use. It keeps
    the common API centered on traces while preserving compatibility with the
    current solver's legacy `receivers_*.h5` files.
    """

    manifest: TraceManifest
    upscale: int = 1
    _store: Optional[TraceStore] = None

    def __post_init__(self) -> None:
        """Validate manifest coverage before any files are opened."""

        if self.upscale < 1:
            raise ValueError("TraceDataset upscale must be >= 1")
        if not self.manifest.files:
            raise ValueError("TraceDataset requires at least one trace file")
        if not self.manifest.frequencies:
            raise ValueError("TraceDataset requires frequency metadata")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return normalized trace metadata used by ``TraceStore``."""

        f_list = np.sort(np.asarray(list(self.manifest.frequencies.values())))
        df = float(np.diff(f_list).min()) if len(f_list) > 1 else 1.0
        laplace_map = {
            int(index): float(
                self.manifest.laplace.get(
                    index, self.manifest.laplace.get(str(index), 0.0)
                )
            )
            for index in self.manifest.frequencies
        }
        return {
            "project": self.manifest.project_path,
            "simulation": self.manifest.simulation,
            "result_path": self.manifest.result_path,
            "output_path": self.manifest.output_path,
            "groups": self.manifest.groups,
            "df": df,
            "f_max": float(f_list[-1]),
            "f_map": dict(self.manifest.frequencies),
            "laplace_map": laplace_map,
            "laplace_map_keys": [int(index) for index in self.manifest.laplace],
            "wavefields": dict(self.manifest.wavefields),
            **(
                {
                    "duplicate_frequencies": self.manifest.run.state[
                        "duplicate_frequencies"
                    ]
                }
                if "duplicate_frequencies" in self.manifest.run.state
                else {}
            ),
        }

    @property
    def files(self) -> List[str]:
        """Return trace file paths as strings."""

        return [str(file) for file in self.manifest.files]

    @property
    def paths(self) -> List[Path]:
        """Return trace file paths as ``Path`` objects."""

        return [Path(file) for file in self.manifest.files]

    @classmethod
    def from_job(
        cls,
        job,
        upscale: int = 1,
        project_path: Optional[Path] = None,
    ) -> "TraceDataset":
        """Build a trace dataset from a job's trace manifest.

        Args:
            job: Job object with trace artifact metadata.
            upscale: Default upscaling factor for time-domain reads.
            project_path: Optional project path used to resolve relative
                artifacts.

        Returns:
            ``TraceDataset`` for the job's requested frequencies.
        """

        return cls.from_manifest(
            TraceManifest.from_job(
                job,
                project_path=project_path,
                resolve_legacy=True,
            ),
            upscale=upscale,
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: TraceManifest,
        upscale: int = 1,
    ) -> "TraceDataset":
        """Build a trace dataset from a resolved trace manifest.

        The manifest may point at a packed trace product or per-frequency shard
        files. When packed output contains stale frequencies, this method
        narrows the dataset to the frequencies requested by the manifest and
        falls back to matching shards when possible.

        Args:
            manifest: Trace manifest generated from a job or artifact handle.
            upscale: Default upscaling factor for time-domain reads.

        Returns:
            ``TraceDataset`` whose manifest contains only usable trace files and
            frequencies.

        Raises:
            ValueError: If no requested frequencies are present in the packed
                product.
            FileNotFoundError: If expected wavefield trace artifacts are
                missing.
        """

        packed_files = manifest.packed_files
        if packed_files:
            packed_incomplete = not manifest.packed_complete
            files = packed_files
            frequencies = dict(manifest.frequencies)
            laplace = dict(manifest.laplace)
            packed_frequencies = manifest.packed_frequencies
            if packed_frequencies:
                shard_files, shard_frequencies, shard_laplace = (
                    cls._matching_frequency_trace_files(
                        manifest,
                        frequencies,
                        laplace,
                    )
                )
                if shard_frequencies and set(map(int, shard_frequencies)) == set(
                    map(int, frequencies)
                ):
                    files = shard_files
                    frequencies = shard_frequencies
                    laplace = shard_laplace
                    if packed_incomplete:
                        warnings.warn(
                            f"{manifest.packed_incomplete_message()}; using "
                            "matching per-frequency trace files instead.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                else:
                    if packed_incomplete:
                        warnings.warn(
                            manifest.packed_incomplete_message(),
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    missing = {int(key) for key in manifest.missing_packed_frequencies}
                    frequencies = {
                        key: frequency
                        for key, frequency in frequencies.items()
                        if int(key) not in missing
                    }
                    laplace = {
                        key: laplace.get(key, laplace.get(str(key), 0.0))
                        for key in frequencies
                    }
                    if manifest.frequencies and not frequencies:
                        product_names = ", ".join(str(path) for path in packed_files)
                        raise ValueError(
                            f"Packed trace product {product_names} contains no "
                            "frequencies requested by this job. "
                            f"{manifest.packed_incomplete_message()}"
                        )
        else:
            files = [TraceManifest.resolve_trace_file(file) for file in manifest.files]
            frequencies = dict(manifest.frequencies)
            laplace = dict(manifest.laplace)
            shard_files, shard_frequencies, shard_laplace = (
                cls._matching_frequency_trace_files(
                    manifest,
                    frequencies,
                    laplace,
                )
            )
            if shard_frequencies and set(map(int, shard_frequencies)) == set(
                map(int, frequencies)
            ):
                files = shard_files
                frequencies = shard_frequencies
                laplace = shard_laplace
        cls._raise_if_wavefield_artifacts_missing(manifest, files, frequencies)
        return cls(
            manifest=TraceManifest(
                files=files,
                frequencies=frequencies,
                groups=list(manifest.groups),
                simulation=manifest.simulation,
                result_path=manifest.result_path,
                output_path=manifest.output_path,
                project_path=manifest.project_path,
                laplace=laplace,
                components=list(manifest.components),
                sources=list(manifest.sources),
                wavefields=dict(manifest.wavefields),
                artifacts=list(manifest.artifacts),
                run=manifest.run,
            ),
            upscale=upscale,
        )

    @staticmethod
    def _frequency_key_for_value(
        frequencies: Dict[int, float],
        value: float,
    ) -> Optional[int]:
        for key, frequency in frequencies.items():
            if np.isclose(float(value), float(frequency), rtol=0.0, atol=1.0e-9):
                return int(key)
        return None

    @staticmethod
    def _candidate_frequency_trace_files(manifest: TraceManifest) -> List[Path]:
        candidates = [TraceManifest.resolve_trace_file(file) for file in manifest.files]

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

        out = []
        seen = set()
        for path in candidates:
            key = str(Path(path).resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            out.append(Path(path))
        return out

    @classmethod
    def _matching_frequency_trace_files(
        cls,
        manifest: TraceManifest,
        frequencies: Dict[int, float],
        laplace: Dict[int, float],
    ) -> tuple[List[Path], Dict[int, float], Dict[int, float]]:
        files_by_key: Dict[int, Path] = {}
        laplace_by_key: Dict[int, float] = {}
        for path in cls._candidate_frequency_trace_files(manifest):
            if not path.exists():
                continue
            try:
                if TraceStore._is_packed_trace_file(path):
                    continue
                values = TraceStore._read_trace_frequencies(path)
            except (OSError, KeyError, ValueError):
                continue
            if len(values) != 1:
                continue
            key = cls._frequency_key_for_value(frequencies, values[0])
            if key is None or key in files_by_key:
                continue
            files_by_key[key] = path
            try:
                laplace_values = TraceStore._read_trace_laplace_values(path)
            except (OSError, KeyError, ValueError):
                laplace_values = []
            if laplace_values:
                laplace_by_key[key] = float(laplace_values[0])

        ordered_keys = [int(key) for key in frequencies if int(key) in files_by_key]
        files = [files_by_key[key] for key in ordered_keys]
        matched_frequencies = {key: frequencies[key] for key in ordered_keys}
        matched_laplace = {
            key: laplace_by_key.get(key, laplace.get(key, laplace.get(str(key), 0.0)))
            for key in ordered_keys
        }
        return files, matched_frequencies, matched_laplace

    @staticmethod
    def _has_existing_trace_artifact(
        files: Iterable[Path],
        frequencies: Dict[int, float],
    ) -> bool:
        records = [Path(file) for file in files]
        if any(record.exists() for record in records):
            return True

        expected = []
        for value in frequencies.values():
            try:
                expected.append(float(np.real(value)))
            except (TypeError, ValueError):
                continue
        return any(
            TraceStore._packed_trace_covers_frequencies(candidate, expected)
            for candidate in TraceStore._candidate_packed_trace_files(records)
        )

    @staticmethod
    def _raise_if_wavefield_artifacts_missing(
        manifest: TraceManifest,
        files: Iterable[Path],
        frequencies: Dict[int, float],
    ) -> None:
        if not manifest.wavefields:
            return
        files = list(files)
        if TraceDataset._has_existing_trace_artifact(files, frequencies):
            return

        groups = ", ".join(repr(group) for group in manifest.groups)
        groups = groups or "requested wavefields"
        packed_hint = manifest.output_path / "traces.h5"
        named_packed_hint = manifest.output_path / "<wavefield>.h5"
        shard_hint = manifest.output_path / "shards" / "f_*_hz.h5"
        named_shard_hint = manifest.output_path / "<wavefield>" / "f_*_hz.h5"
        raise FileNotFoundError(
            f"No wavefield trace files were found for {groups} in "
            f"{manifest.output_path}. Expected a packed wavefield file such as "
            f"{packed_hint} or {named_packed_hint}, or per-frequency files "
            f"matching {shard_hint} or {named_shard_hint}. "
            "Rerun the job after enabling solver wavefield output, or fetch the "
            "wavefield output directory if it was produced remotely."
        )

    @classmethod
    def open(
        cls,
        source,
        upscale: int = 1,
        project_path: Optional[Path] = None,
    ) -> "TraceDataset":
        """Open traces from a manifest, job, or trace HDF5 file.

        Args:
            source: ``TraceManifest``, job-like object, or HDF5 trace file path.
            upscale: Default upscaling factor for time-domain reads.
            project_path: Optional project path used for job artifact resolution.

        Returns:
            ``TraceDataset``.
        """

        if isinstance(source, TraceManifest):
            return cls.from_manifest(source, upscale=upscale)
        if hasattr(source, "trace_manifest"):
            return cls.from_job(source, upscale=upscale, project_path=project_path)
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")
        frequencies = TraceStore._read_trace_frequencies(path)
        laplace = TraceStore._read_trace_laplace_values(path)
        groups = TraceStore.discover_trace_groups(path)
        manifest = TraceManifest(
            files=[TraceManifest.resolve_trace_file(path)],
            frequencies={
                index: frequency for index, frequency in enumerate(frequencies, start=1)
            },
            laplace={index: value for index, value in enumerate(laplace, start=1)},
            groups=groups,
            simulation=path,
            result_path=path.parent,
            output_path=path.parent,
            project_path=path.parent,
        )
        return cls.from_manifest(manifest, upscale=upscale)

    @classmethod
    def from_jobs(
        cls,
        jobs: Iterable,
        upscale: int = 1,
        project_path: Optional[Path] = None,
        duplicate: str = "first",
    ) -> "TraceDataset":
        """Create one dataset from multiple jobs.

        Args:
            jobs: Jobs whose trace manifests should be combined.
            upscale: Default upscaling factor for time-domain reads.
            project_path: Optional project path used for artifact resolution.
            duplicate: Duplicate-frequency policy: ``"first"``, ``"last"``, or
                ``"error"``.

        Returns:
            Combined ``TraceDataset``.
        """

        datasets = [
            cls.from_job(job, upscale=upscale, project_path=project_path)
            for job in jobs
        ]
        return cls.combine(datasets, duplicate=duplicate)

    @classmethod
    def combine(
        cls,
        datasets: Iterable["TraceDataset"],
        duplicate: str = "first",
    ) -> "TraceDataset":
        """Combine existing trace datasets.

        Args:
            datasets: Trace datasets to combine.
            duplicate: Duplicate-frequency policy: ``"first"``, ``"last"``, or
                ``"error"``.

        Returns:
            Combined ``TraceDataset``.
        """

        datasets = list(datasets)
        if not datasets:
            raise ValueError("At least one TraceDataset is required")
        if duplicate not in {"first", "last", "error"}:
            raise ValueError("duplicate must be 'first', 'last', or 'error'")

        manifest = TraceManifest.combine(
            [dataset.manifest for dataset in datasets],
            duplicate=duplicate,
        )
        if "duplicate_frequencies" in manifest.run.state:
            warnings.warn(
                "Duplicate trace frequencies were encountered while combining jobs; "
                f"using duplicate='{duplicate}'.",
                RuntimeWarning,
            )
        return cls.from_manifest(manifest, upscale=datasets[0].upscale)

    @property
    def store(self) -> TraceStore:
        """Return the lazily opened ``TraceStore`` backing this dataset."""

        if self._store is None:
            store = TraceStore(
                metadata=self.metadata,
                files=self.files,
                upscale=self.upscale,
                cache_dir=self._default_cache_dir(),
            )
            self._store = store
        return self._store

    def close(self) -> None:
        """Close any open HDF5 handles owned by this dataset."""

        if self._store is not None:
            self._store.close()

    def __enter__(self) -> "TraceDataset":
        """Enter a context manager for deterministic trace-store cleanup."""

        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Close the backing trace store when leaving a context manager."""

        self.close()
        return False

    def __len__(self) -> int:
        """Return the number of trace files in the dataset manifest."""

        return len(self.manifest.files)

    def __repr__(self) -> str:
        return (
            "TraceDataset("
            f"files={len(self.manifest.files)}, "
            f"frequencies={len(self.manifest.frequencies)}, "
            f"groups={len(self.manifest.groups)}"
            ")"
        )

    def _default_cache_dir(self) -> Optional[Path]:
        if self.manifest.result_path is None:
            return None
        return Path(self.manifest.result_path) / "_fs_run" / "cache"

    def consolidate(
        self,
        cache_dir: Optional[Path] = None,
    ) -> Path:
        """Build or refresh the trace VDS cache and return its path.

        Args:
            cache_dir: Optional cache directory. Defaults to the job run cache
                directory when available.

        Returns:
            Path to the consolidated HDF5 file.
        """

        return self.store.consolidate(
            cache_dir=cache_dir or self._default_cache_dir(),
        )

    def times(self, upscale: Optional[int] = None) -> np.ndarray:
        """Return reconstructed time samples for this dataset.

        Args:
            upscale: Optional upscaling factor. Defaults to ``self.upscale``.

        Returns:
            One-dimensional time sample array.
        """

        upscale = self.upscale if upscale is None else upscale
        sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=self.metadata["f_max"],
            df=self.metadata["df"],
            upscale=upscale,
        )
        return sampling.T_list

    @property
    def groups(self) -> list[str]:
        """Return receiver/wavefield group names available in the dataset."""

        return self.store.groups

    def components(self, group: str):
        """Return component names available for a group."""

        return self.store.components(group)

    def sources(self, group: str):
        """Return source ids available for a group."""

        return self.store.sources(group)

    def receivers(self, group: str):
        """Return receiver coordinate metadata for a group."""

        return self.store.receivers(group)

    def frequencies(self, group: Optional[str] = None):
        """Return available modeled frequencies.

        Args:
            group: Optional group name. When omitted, manifest frequencies are
                returned.
        """

        if group is not None:
            return self.store.frequencies(group)
        return np.sort(np.asarray(list(self.metadata["f_map"].values())))

    def laplace(self, group: Optional[str] = None):
        """Return Laplace damping values for all frequencies or one group."""

        return self.store.laplace(group)

    def format_summary(self, colorize: bool = False) -> TraceSummary:
        """Return a human-readable trace summary."""

        return self.store.format_summary(colorize=colorize)

    @property
    def summary(self) -> TraceSummary:
        """Return a notebook-friendly trace summary."""

        return self.store.summary

    def print_summary(
        self,
        *,
        colorize: Optional[bool] = None,
        file: Optional[Any] = None,
    ) -> TraceSummary:
        """Print and return a trace summary."""

        return self.store.print_summary(colorize=colorize, file=file)

    def open_frequency_domain(self, group: str):
        """Open a raw frequency-domain xarray object for ``group``."""

        return self.store.read_h5(group)

    def survey_tables(self) -> Dict[str, Any]:
        """Return solver survey tables embedded in trace output files."""

        return self.store.survey_tables()

    def frequency_domain(
        self,
        group: str,
        component: str,
        source: int = 1,
        wavelet: Optional[Wavelet] = None,
        **kwargs,
    ):
        """Read one frequency-domain gather.

        Args:
            group: Receiver group name.
            component: Component name.
            source: One-based source id.
            wavelet: Optional wavelet used to scale the frequency-domain data.
            **kwargs: Additional ``TraceStore.read_FD`` options.
        """

        return self.store.read_FD(group, component, source, wavelet=wavelet, **kwargs)

    def time_domain(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        laplace_compensation: str = "auto",
        **kwargs,
    ):
        """Read one reconstructed time-domain gather.

        Args:
            group: Receiver group name.
            component: Component name.
            source: One-based source id.
            wavelet: Wavelet used for inverse transform.
            upscale: Reconstruction upscaling factor.
            T_max: Optional maximum time to return.
            laplace_compensation: Laplace compensation mode.
            **kwargs: Additional ``TraceStore.read_TD`` options.
        """

        return self.store.read_TD(
            group,
            component,
            source,
            wavelet,
            upscale=upscale,
            T_max=T_max,
            laplace_compensation=laplace_compensation,
            **kwargs,
        )

    def laplace_domain(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        **kwargs,
    ):
        """Read one Laplace-domain gather."""

        return self.store.read_LD(
            group,
            component,
            source,
            wavelet,
            upscale=upscale,
            T_max=T_max,
            **kwargs,
        )

    def fd(
        self,
        group: str,
        component: str,
        source: int = 1,
        wavelet: Optional[Wavelet] = None,
        **kwargs,
    ):
        """Alias for ``frequency_domain``."""

        return self.frequency_domain(
            group,
            component,
            source=source,
            wavelet=wavelet,
            **kwargs,
        )

    def td(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        laplace_compensation: str = "auto",
        **kwargs,
    ):
        """Alias for ``time_domain``."""

        return self.time_domain(
            group,
            component,
            source=source,
            wavelet=wavelet,
            upscale=upscale,
            T_max=T_max,
            laplace_compensation=laplace_compensation,
            **kwargs,
        )

    def ld(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        **kwargs,
    ):
        """Alias for ``laplace_domain``."""

        return self.laplace_domain(
            group,
            component,
            source=source,
            wavelet=wavelet,
            upscale=upscale,
            T_max=T_max,
            **kwargs,
        )
