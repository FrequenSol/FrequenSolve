from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from frequensolve.seismic.record_database import TraceStore
from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.sampling import UniformSweepSampling


@dataclass
class TraceDataset:
    """Small facade for reading solver trace outputs.

    This is the preferred replacement for direct `TraceStore` use. It keeps
    the common API centered on traces while preserving compatibility with the
    current solver's legacy `receivers_*.h5` files.
    """

    metadata: Dict[str, Any]
    files: List[str]
    upscale: int = 1
    _store: Optional[TraceStore] = None

    @classmethod
    def from_job(
        cls,
        job,
        upscale: int = 1,
        project_path: Optional[Path] = None,
    ) -> "TraceDataset":
        traces = job.traces
        source_project = Path(job.project_path).resolve()
        project_path = (
            Path(project_path).resolve() if project_path is not None else source_project
        )

        f_map = dict(traces["frequencies"])
        for key, value in list(f_map.items()):
            if isinstance(value, complex):
                f_map[key] = value.real
        f_list = np.sort(list(f_map.values()))
        f_max = f_list[-1]
        df = np.diff(f_list).min() if len(f_list) > 1 else 1.0

        files = [
            str(
                cls._resolve_trace_file(
                    cls._map_project_path(path, source_project, project_path)
                )
            )
            for path in traces["files"]
        ]
        simulation = cls._map_project_path(
            traces["simulation"], source_project, project_path
        )
        metadata = {
            "project": project_path,
            "simulation": simulation,
            "groups": traces["groups"],
            "df": df,
            "f_max": f_max,
            "f_map": f_map,
        }
        return cls(metadata=metadata, files=files, upscale=upscale)

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

        groups = datasets[0].metadata["groups"]
        for dataset in datasets[1:]:
            if dataset.metadata["groups"] != groups:
                raise ValueError("Cannot combine trace datasets with different groups")

        entries: Dict[float, Dict[str, Any]] = {}
        duplicates = []
        for dataset in datasets:
            f_items = sorted(
                dataset.metadata["f_map"].items(), key=lambda item: int(item[0])
            )
            for (_, freq), file in zip(f_items, dataset.files):
                key = float(freq)
                if key in entries:
                    duplicates.append(key)
                    if duplicate == "error":
                        raise ValueError(f"Duplicate trace frequency: {key}")
                    if duplicate == "first":
                        continue
                entries[key] = {"frequency": key, "file": file}

        ordered = [entries[freq] for freq in sorted(entries)]
        f_list = np.array([entry["frequency"] for entry in ordered], dtype=float)
        df = np.diff(f_list).min() if len(f_list) > 1 else 1.0
        metadata = {
            "project": datasets[0].metadata["project"],
            "simulation": datasets[0].metadata["simulation"],
            "simulations": [dataset.metadata["simulation"] for dataset in datasets],
            "groups": groups,
            "df": df,
            "f_max": float(f_list[-1]),
            "f_map": {i + 1: float(freq) for i, freq in enumerate(f_list)},
            "combined": True,
            "source_count": len(datasets),
        }
        if duplicates:
            metadata["duplicate_frequencies"] = sorted(set(duplicates))
            warnings.warn(
                "Duplicate trace frequencies were encountered while combining jobs; "
                f"using duplicate='{duplicate}'.",
                RuntimeWarning,
            )
        return cls(
            metadata=metadata,
            files=[entry["file"] for entry in ordered],
            upscale=datasets[0].upscale,
        )

    @staticmethod
    def _map_project_path(path: str, source_project: Path, project_path: Path) -> Path:
        path = Path(path)
        if path.is_absolute():
            try:
                return project_path / path.resolve().relative_to(source_project)
            except Exception:
                return path
        return project_path / path

    @staticmethod
    def _resolve_trace_file(path: str) -> Path:
        path = Path(path)
        if path.exists():
            return path
        legacy = path.with_name(path.name.replace("traces_", "receivers_", 1))
        if legacy.exists():
            return legacy
        return path

    @property
    def store(self) -> TraceStore:
        if self._store is None:
            store = TraceStore(
                metadata=self.metadata,
                files=self.files,
                upscale=self.upscale,
            )
            store.consolidate_h5()
            self._store = store
        return self._store

    @property
    def record_db(self) -> TraceStore:
        """Compatibility alias for the internal trace store."""
        return self.store

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

    @property
    def summary(self) -> str:
        return self.store.summary

    def open_frequency_domain(self, group: str):
        return self.store.read_h5(group)

    def survey_tables(self) -> Dict[str, Any]:
        return self.store.survey_tables()

    def fd(
        self,
        group: str,
        component: str,
        source: int = 1,
        wavelet: Optional[Wavelet] = None,
        **kwargs,
    ):
        return self.store.read_FD(group, component, source, wavelet=wavelet, **kwargs)

    def td(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        **kwargs,
    ):
        return self.store.read_TD(
            group,
            component,
            source,
            wavelet,
            upscale=upscale,
            T_max=T_max,
            **kwargs,
        )

    def read_FD(self, group: str, component: str, shot: int, wavelet=None, **kwargs):
        return self.fd(group, component, source=shot, wavelet=wavelet, **kwargs)

    def read_TD(
        self,
        group: str,
        component: str,
        shot: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        **kwargs,
    ):
        return self.td(
            group,
            component,
            source=shot,
            wavelet=wavelet,
            upscale=upscale,
            T_max=T_max,
            **kwargs,
        )
