from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from frequensolve.seismic.trace_store import TraceStore, TraceSummary
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.artifacts import TraceManifest
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
        if self.upscale < 1:
            raise ValueError("TraceDataset upscale must be >= 1")
        if not self.manifest.files:
            raise ValueError("TraceDataset requires at least one trace file")
        if not self.manifest.frequencies:
            raise ValueError("TraceDataset requires frequency metadata")

    @property
    def metadata(self) -> Dict[str, Any]:
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
        return [str(file) for file in self.manifest.files]

    @property
    def paths(self) -> List[Path]:
        return [Path(file) for file in self.manifest.files]

    @classmethod
    def from_job(
        cls,
        job,
        upscale: int = 1,
        project_path: Optional[Path] = None,
    ) -> "TraceDataset":
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
        packed_file = manifest.packed_file
        if packed_file is not None:
            files = [packed_file]
            frequencies = manifest.packed_frequencies or dict(manifest.frequencies)
            laplace = manifest.packed_laplace or dict(manifest.laplace)
        else:
            files = [TraceManifest.resolve_trace_file(file) for file in manifest.files]
            frequencies = dict(manifest.frequencies)
            laplace = dict(manifest.laplace)
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
                artifacts=list(manifest.artifacts),
                run=manifest.run,
            ),
            upscale=upscale,
        )

    @classmethod
    def open(
        cls,
        source,
        upscale: int = 1,
        project_path: Optional[Path] = None,
    ) -> "TraceDataset":
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
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __len__(self) -> int:
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
        """Build or refresh the trace VDS cache and return its path."""

        return self.store.consolidate(
            cache_dir=cache_dir or self._default_cache_dir(),
        )

    def times(self, upscale: Optional[int] = None) -> np.ndarray:
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
        return self.store.groups

    def components(self, group: str):
        return self.store.components(group)

    def sources(self, group: str):
        return self.store.sources(group)

    def receivers(self, group: str):
        return self.store.receivers(group)

    def frequencies(self, group: Optional[str] = None):
        if group is not None:
            return self.store.frequencies(group)
        return np.sort(np.asarray(list(self.metadata["f_map"].values())))

    def laplace(self, group: Optional[str] = None):
        return self.store.laplace(group)

    def format_summary(self, colorize: bool = False) -> TraceSummary:
        return self.store.format_summary(colorize=colorize)

    @property
    def summary(self) -> TraceSummary:
        return self.store.summary

    def print_summary(
        self,
        *,
        colorize: Optional[bool] = None,
        file: Optional[Any] = None,
    ) -> TraceSummary:
        return self.store.print_summary(colorize=colorize, file=file)

    def open_frequency_domain(self, group: str):
        return self.store.read_h5(group)

    def survey_tables(self) -> Dict[str, Any]:
        return self.store.survey_tables()

    def frequency_domain(
        self,
        group: str,
        component: str,
        source: int = 1,
        wavelet: Optional[Wavelet] = None,
        **kwargs,
    ):
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
        return self.laplace_domain(
            group,
            component,
            source=source,
            wavelet=wavelet,
            upscale=upscale,
            T_max=T_max,
            **kwargs,
        )
