"""Internal HDF5 trace-store reader.

``TraceStore`` backs the public ``TraceDataset`` facade.
"""

import html
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import h5py
import numpy as np
from xarray import DataArray

from frequensolve.seismic.wavelet import Wavelet
from frequensolve.simulation.sampling import UniformSweepSampling
from frequensolve.util.fft import get_fft_backend

__all__: list[str] = []

_ROOT_TRACE_METADATA_DATASETS = {
    "frequency",
    "laplace",
    "metadata",
    "task_id",
    "trace_metadata_file",
    "trace_data",
    "trace_index",
}
_TRACE_METADATA_FILE = "trace_metadata.h5"
_MODERN_TRACE_SHARD_GLOB = "f_*.h5"
_LEGACY_TRACE_SHARD_GLOBS = (
    "traces_*.h5",
    "receivers_*.h5",
    "trace_frequency_*.h5",
)


class TraceSummary(str):
    """Notebook-friendly string for trace summaries."""

    def __new__(cls, text: str) -> "TraceSummary":
        return super().__new__(cls, text)

    def __repr__(self) -> str:
        return str(self)

    def _repr_pretty_(self, printer, cycle: bool) -> None:
        printer.text(str(self))

    def _repr_html_(self) -> str:
        text = html.escape(str(self))
        return f'<pre style="white-space: pre-wrap; margin: 0;">{text}</pre>'


def _decode_h5_strings(values):
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.asarray(
            [
                (
                    item.decode("utf-8", "ignore").rstrip("\x00").rstrip()
                    if isinstance(item, bytes)
                    else str(item).rstrip("\x00").rstrip()
                )
                for item in values.ravel()
            ],
            dtype=object,
        ).reshape(values.shape)
    return values


def _decode_dim_list(values) -> list[str]:
    return [
        (
            item.decode("utf-8", "ignore").rstrip("\x00").rstrip()
            if isinstance(item, bytes)
            else str(item).rstrip("\x00").rstrip()
        )
        for item in values
    ]


def _unique_preserve_order(values):
    out = []
    seen = set()
    for value in values:
        key = value.item() if hasattr(value, "item") else value
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return np.asarray(out)


def _attr_strings(value) -> list[str]:
    return [str(item) for item in _decode_h5_strings(np.asarray(value)).ravel()]


def _dataset_strings(h5, path: str) -> list[str]:
    if path not in h5:
        return []
    return [str(item) for item in _decode_h5_strings(h5[path][()]).ravel()]


def _dataset_ints(h5, path: str) -> np.ndarray:
    if path not in h5:
        return np.asarray([], dtype=np.int64)
    return np.asarray(h5[path][()]).ravel().astype(np.int64)


def _clean_h5_path(path: str) -> str:
    return "/" + str(path).strip("/")


def _trace_data_tail(path: str) -> str:
    return str(path).strip("/")


def _trace_axis_dims(dset) -> list[str]:
    dims = _decode_dim_list(dset.attrs["dims"])
    layout_kind = _attr_strings(dset.attrs.get("layout_kind", []))
    if "dense_trace_v1" in layout_kind:
        return list(reversed(dims))
    return dims


def _trace_data_dims(dset) -> list[str]:
    dims = _decode_dim_list(dset.attrs["dims"])
    has_frequency = bool(dims) and dims[-1] == "frequency"
    if has_frequency:
        dims = dims[:-1]
    layout_kind = _attr_strings(dset.attrs.get("layout_kind", []))
    if "dense_trace_v1" in layout_kind:
        dims = list(reversed(dims))
    return ["frequency", *dims]


def _copy_h5_attrs(source, target) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _as_file_list(files: Iterable[Union[str, Path]]) -> List[Union[str, Path]]:
    if isinstance(files, (str, Path)):
        return [files]
    return list(files)


@dataclass(init=False)
class TraceStore:
    """Low-level reader for FrequenSolve HDF5 trace products.

    ``TraceDataset`` is the preferred public facade. Use ``TraceStore`` when
    code needs direct access to consolidated HDF5 groups or legacy read
    methods.

    Args:
        metadata: Trace metadata including frequency map, groups, and project
            paths.
        files: Trace HDF5 files or per-frequency shard files.
        upscale: Default upscaling factor for reconstructed time-domain reads.
        cache_dir: Optional directory for virtual HDF5 consolidation files.
    """

    metadata: Dict[str, Any]
    files: List[str]
    _upscale: int
    _consolidated: Optional[Path] = None
    _cache_dir: Optional[Path] = None
    _open_files: List[h5py.File]
    _packed_group_files: Dict[str, Path]

    def __init__(
        self,
        metadata: Dict[str, Any],
        files: Optional[Iterable[Union[str, Path]]] = None,
        upscale: int = 1,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        if files is None:
            raise TypeError("TraceStore requires trace files")
        files = _as_file_list(files)
        self.metadata = metadata
        self.files = [str(file) for file in files]
        self.upscale = upscale
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._consolidated = None
        self._open_files = []
        self._packed_group_files = {}

    @classmethod
    def from_job(cls, job, upscale: int = 1):
        """Create a TraceStore from a simulation job.

        Args:
            job: A BaseJob-like object.
            upscale: Time-domain upscale factor.

        Returns:
            ``TraceStore`` initialized from the job's trace artifact metadata.
        """
        traces = job.traces
        proj_path = Path(job.project_path).resolve()

        f_map = dict(traces["frequencies"])
        for key, value in f_map.items():
            f_map[key] = value
            if isinstance(value, complex):
                f_map[key] = value.real
        f_list = np.sort(list(f_map.values()))
        f_max = f_list[-1]
        if len(f_list) > 1:
            df = np.diff(f_list).min()
        else:
            df = 1.0

        meta = {
            "project": proj_path,
            "simulation": proj_path / traces["simulation"],
            "groups": traces["groups"],
            "df": df,
            "f_max": f_max,
            "f_map": f_map,
        }

        db = cls(metadata=meta, files=traces["files"], upscale=upscale)
        db.consolidate()
        return db

    def __enter__(self) -> "TraceStore":
        """Enter a context manager for deterministic HDF5 cleanup."""

        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Close HDF5 handles when leaving a context manager."""

        self.close()
        return False

    def close(self) -> None:
        """Close HDF5 file handles owned by this reader."""

        for handle in self._open_files:
            try:
                handle.close()
            except Exception:
                pass
        self._open_files.clear()

    @property
    def upscale(self) -> int:
        """Return the default upscaling factor for time-domain reads."""

        return self._upscale

    @upscale.setter
    def upscale(self, upscale: int) -> None:
        """Set the default upscaling factor for time-domain reads."""

        self._upscale = upscale

    def times(self, upscale: Optional[int] = None) -> np.ndarray:
        """Return reconstructed trace sample times.

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

    def __len__(self) -> int:
        """Returns the number of traces in the store."""
        size = 0
        for group in self.groups:
            recv = self.receivers(group)
            shot = self.shots(group)
            comp = self.components(group)
            size += len(recv) * len(shot) * len(comp)
        return size

    @property
    def groups(self) -> list[str]:
        """Return receiver or wavefield groups available in the trace store."""

        self._ensure_consolidated()
        if self._packed_group_files:
            configured = [str(group) for group in self.metadata.get("groups", [])]
            if configured:
                return [
                    group for group in configured if group in self._packed_group_files
                ]
            return list(self._packed_group_files)
        with h5py.File(self._consolidated, "r") as f:
            return self._h5_trace_groups(f, self.metadata.get("groups"))

    def dims(self, group) -> list[str]:
        """Return xarray dimension names for a trace group."""

        with h5py.File(self._trace_file_for_group(group), "r") as f:
            if self._is_indexed_packed_h5(f) and group not in f:
                dset = f[self._indexed_trace_paths(f, group)[0]]
            else:
                dset = f[group]
            return [
                "source" if dim == "shot" else dim for dim in _trace_data_dims(dset)
            ]

    def components(self, group) -> list[str]:
        """Return component labels available in a trace group."""

        with h5py.File(self._trace_file_for_group(group), "r") as f:
            for path in (
                f"survey/receiver_groups/{group}/traces/component_name",
                f"survey/receiver_groups/{group}/components/component_name",
                "survey/components/component_name",
            ):
                if path in f:
                    return _unique_preserve_order(_decode_h5_strings(f[path][()]))
            dset = (
                f[self._indexed_trace_paths(f, group)[0]]
                if self._is_indexed_packed_h5(f) and group not in f
                else f[group]
            )
            if "component" in dset.attrs:
                return _decode_h5_strings(dset.attrs["component"])
            return np.arange(1, dset.shape[-2] + 1)

    def sources(self, group) -> list[str]:
        """Return source ids available in a trace group."""

        with h5py.File(self._trace_file_for_group(group), "r") as f:
            for path in (
                f"survey/receiver_groups/{group}/traces/source_id",
                "survey/sources/source_id",
            ):
                if path in f:
                    return _unique_preserve_order(f[path][()])
            dset = (
                f[self._indexed_trace_paths(f, group)[0]]
                if self._is_indexed_packed_h5(f) and group not in f
                else f[group]
            )
            if "shot" in dset.attrs:
                return dset.attrs["shot"]
            return np.arange(1, dset.shape[-1] + 1)

    def shots(self, group) -> list[str]:
        """Compatibility alias for ``sources``."""

        return self.sources(group)

    def frequencies(self, group) -> list[str]:
        """Return frequencies available in a trace group."""

        with h5py.File(self._trace_file_for_group(group), "r") as f:
            if self._is_indexed_packed_h5(f) and group not in f:
                values = [
                    row["frequency"]
                    for row in self._filter_expected_indexed_rows(
                        self._indexed_trace_rows(f, group)
                    )
                    if row["frequency"] is not None
                ]
                if values:
                    return np.asarray(values, dtype=float)
            if "frequency" in f:
                frequencies = np.asarray(f["frequency"][()]).ravel()
                laplace = (
                    np.asarray(f["laplace"][()]).ravel() if "laplace" in f else None
                )
                indices = self._expected_frequency_laplace_indices(frequencies, laplace)
                if indices is not None:
                    return frequencies[indices].astype(float)
                return self._filter_expected_frequencies(frequencies)
            dset = (
                f[self._indexed_trace_paths(f, group)[0]]
                if self._is_indexed_packed_h5(f) and group not in f
                else f[group]
            )
            if "frequency" in dset.attrs:
                return self._filter_expected_frequencies(dset.attrs["frequency"])
            return np.array(list(self.metadata["f_map"].values()))

    def laplace(self, group: Optional[str] = None) -> np.ndarray:
        """Return Laplace offsets for all traces or a specific group."""

        if group is None and self._packed_group_files:
            return self._metadata_laplace_values(self._expected_frequencies())
        path = (
            self._trace_file_for_group(group)
            if group is not None
            else self._trace_file_for_group(self.groups[0]) if self.groups else None
        )
        if path is None:
            return self._metadata_laplace_values(self._expected_frequencies())
        with h5py.File(path, "r") as f:
            if group is not None and self._is_indexed_packed_h5(f) and group not in f:
                rows = self._filter_expected_indexed_rows(
                    self._indexed_trace_rows(f, group)
                )
                return np.asarray(
                    [row.get("laplace", 0.0) for row in rows], dtype=float
                )
            if "laplace" in f:
                values = np.asarray(f["laplace"][()]).ravel()
                if values.size:
                    if "frequency" in f:
                        indices = self._expected_frequency_laplace_indices(
                            f["frequency"][()],
                            values,
                        )
                        if (
                            indices is not None
                            and max(indices, default=-1) < values.size
                        ):
                            values = values[indices]
                    return values.astype(float)
            if group is not None:
                frequencies = np.asarray(self.frequencies(group), dtype=float)
            else:
                frequencies = np.asarray(self._expected_frequencies(), dtype=float)
            return self._metadata_laplace_values(frequencies)

    def receivers(self, group) -> list[str]:
        """Return receiver ids available in a trace group."""

        with h5py.File(self._trace_file_for_group(group), "r") as f:
            for path in (
                f"survey/receiver_groups/{group}/traces/receiver_id",
                f"survey/receiver_groups/{group}/receivers/receiver_id",
                "survey/receivers/receiver_id",
            ):
                if path in f:
                    return _unique_preserve_order(f[path][()])
            dset = (
                f[self._indexed_trace_paths(f, group)[0]]
                if self._is_indexed_packed_h5(f) and group not in f
                else f[group]
            )
            if "receiver" in dset.attrs:
                return dset.attrs["receiver"]
            dims = np.asarray(_decode_dim_list(dset.attrs["dims"])[::-1])
            ind = np.where(dims == "receiver")[0][0]
            return np.arange(1, dset.shape[ind] + 1)

    def survey_tables(self) -> Dict[str, Any]:
        """Return embedded survey metadata tables from the trace store."""

        self._ensure_consolidated()

        def read_group(group):
            out = {}
            for key, item in group.items():
                if isinstance(item, h5py.Dataset):
                    value = item[()]
                    out[key] = _decode_h5_strings(value).tolist()
                elif isinstance(item, h5py.Group):
                    out[key] = read_group(item)
            return out

        if self._packed_group_files:
            tables: Dict[str, Any] = {}
            for path in self._packed_group_files.values():
                with h5py.File(path, "r") as f:
                    if "survey" not in f:
                        continue
                    tables.update(read_group(f["survey"]))
            return tables
        with h5py.File(self._consolidated, "r") as f:
            if "survey" not in f:
                return {}
            return read_group(f["survey"])

    def format_summary(self, colorize: bool = False) -> TraceSummary:
        """Return a human-readable summary of groups, sources, and frequencies."""

        def _gray(text: str, light: bool = True) -> str:
            if colorize:
                if light:
                    return f"\033[38;5;248m{text}\033[0m"
                else:
                    return f"\033[90m{text}\033[0m"
            return text

        out = ""
        for group in self.groups:
            freq = self.frequencies(group)

            if len(freq) == 0:
                expected = self._expected_frequencies()
                expected_text = (
                    f" Expected frequencies from the job metadata: {expected} Hz."
                    if expected
                    else ""
                )
                raise ValueError(
                    f"Cannot summarize trace group {group!r}: no frequencies are "
                    f"available.{expected_text} The trace files may be empty, "
                    "incomplete, or inconsistent with the job metadata; inspect "
                    "the job logs and fetched trace artifacts."
                )

            recv = self.receivers(group)
            shot = self.shots(group)
            comp = self.components(group)

            out += f"{group}\n"
            out += f"  {_gray('Receivers')}\t: {recv[0]} - {recv[-1]}\n"
            if len(shot) > 1:
                out += f"  {_gray('Shots')}\t\t: {shot[0]} - {shot[-1]}\n"
            else:
                out += f"  {_gray('Shot')}\t\t: {shot[0]}\n"
            out += f"  {_gray('Components')}\t: {comp}\n"
            if len(freq) > 1:
                df = freq[1] - freq[0]
                out += f"  {_gray('Frequencies')}\t: {freq[0]:.2f} - {freq[-1]:.2f} Hz (Δf={df:.2f})\n"
                out += f"  {_gray('Window')}\t: {0:.2f} - {1.0/df:.2f} s\n"
            else:
                out += f"  {_gray('Frequency')}\t: {freq[0]:.2f} Hz\n"
            out += "\n"
        return TraceSummary(out)

    @property
    def summary(self) -> TraceSummary:
        """Return a notebook-friendly trace summary."""

        return self.format_summary()

    def print_summary(
        self,
        *,
        colorize: Optional[bool] = None,
        file: Optional[Any] = None,
    ) -> TraceSummary:
        """Print and return a trace summary.

        Args:
            colorize: Whether to include ANSI color codes. Defaults to terminal
                detection.
            file: Optional text stream.

        Returns:
            Printed ``TraceSummary``.
        """

        file = sys.stdout if file is None else file
        if colorize is None:
            isatty = getattr(file, "isatty", None)
            colorize = bool(isatty()) if callable(isatty) else False
        summary = self.format_summary(colorize=colorize)
        print(str(summary), end="" if str(summary).endswith("\n") else "\n", file=file)
        return summary

    def __str__(self) -> str:
        return str(self.summary)

    def _ensure_consolidated(self) -> None:
        if self._packed_group_files and all(
            path.exists() for path in self._packed_group_files.values()
        ):
            return
        if self._consolidated is None or not Path(self._consolidated).exists():
            self.consolidate()

    def _trace_file_for_group(self, group: str) -> Path:
        self._ensure_consolidated()
        if group in self._packed_group_files:
            return self._packed_group_files[group]
        if self._consolidated is None:
            raise FileNotFoundError("No consolidated trace file is available")
        return Path(self._consolidated)

    @staticmethod
    def _consolidated_path(
        first_record: Union[str, Path],
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        first_record = Path(first_record)
        stem = first_record.stem
        prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
        directory = Path(cache_dir) if cache_dir is not None else first_record.parent
        return directory / f"{prefix}_vds.h5"

    @staticmethod
    def _read_trace_frequency(file: Path) -> float:
        return TraceStore._read_trace_frequencies(file)[0]

    @staticmethod
    def _read_trace_frequencies(file: Path) -> list[float]:
        with h5py.File(file, "r") as h5:
            if "frequency" not in h5:
                raise KeyError(f"'frequency' dataset not found in {file}")
            values = np.asarray(h5["frequency"][()]).ravel()
            if values.size == 0:
                raise ValueError(f"'frequency' dataset is empty in {file}")
            return [float(value) for value in values]

    @staticmethod
    def _read_trace_laplace_values(file: Path) -> list[float]:
        with h5py.File(file, "r") as h5:
            frequency_count = (
                len(np.asarray(h5["frequency"][()]).ravel()) if "frequency" in h5 else 1
            )
            if "trace_index/laplace" in h5:
                values = np.asarray(h5["trace_index/laplace"][()]).ravel()
            elif "laplace" in h5:
                values = np.asarray(h5["laplace"][()]).ravel()
            else:
                return [0.0] * frequency_count
            if values.size == 0:
                return [0.0] * frequency_count
            if values.size == 1 and frequency_count > 1:
                return [float(np.real(values[0]))] * frequency_count
            return [float(np.real(value)) for value in values]

    @staticmethod
    def _read_trace_laplace(file: Path) -> float:
        return TraceStore._read_trace_laplace_values(file)[0]

    @staticmethod
    def _read_trace_metadata_reference(file: Path) -> Optional[str]:
        try:
            with h5py.File(file, "r") as h5:
                values = _dataset_strings(h5, "trace_metadata_file")
        except (OSError, KeyError, ValueError):
            return None
        return values[0] if values else None

    @staticmethod
    def _is_trace_metadata_file(path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with h5py.File(path, "r") as h5:
                schema = _dataset_strings(h5, "metadata/schema_version")
                if schema:
                    return "fs_trace_metadata_v1" in schema
                return "survey" in h5 and "frequency" not in h5
        except (OSError, KeyError, ValueError):
            return False

    def _candidate_trace_metadata_files(
        self,
        records: Iterable[Path],
    ) -> list[Path]:
        records = [Path(record) for record in records]
        candidates: list[Path] = []
        seen: set[str] = set()

        def add(path: Path) -> None:
            key = str(path.resolve(strict=False))
            if key in seen:
                return
            seen.add(key)
            candidates.append(path)

        roots: list[Path] = []
        for root in (
            self.metadata.get("output_path"),
            self.metadata.get("result_path"),
        ):
            if root is not None:
                roots.append(Path(root))
        for record in records:
            roots.extend([record.parent, record.parent.parent])

        for record in records:
            if not record.exists():
                continue
            reference = self._read_trace_metadata_reference(record)
            if not reference:
                continue
            reference_path = Path(reference)
            if reference_path.is_absolute():
                add(reference_path)
                continue
            add(record.parent / reference_path)
            add(record.parent.parent / reference_path)
            for root in roots:
                add(root / reference_path)

        for root in roots:
            add(root / _TRACE_METADATA_FILE)
            add(root / "traces" / _TRACE_METADATA_FILE)
            if root.name == "shards":
                add(root.parent / _TRACE_METADATA_FILE)
        return candidates

    def _trace_metadata_file_for_records(
        self,
        records: Iterable[Path],
    ) -> Optional[Path]:
        for candidate in self._candidate_trace_metadata_files(records):
            if self._is_trace_metadata_file(candidate):
                return candidate
        return None

    @staticmethod
    def _template_dataset_path_for_group(h5, group: str) -> str:
        group = str(group).strip("/")
        if group in h5 and isinstance(h5[group], h5py.Dataset):
            return group
        catalog = "survey/receiver_groups/_catalog"
        if catalog in h5:
            group_names = _dataset_strings(h5, f"{catalog}/group_name")
            dataset_paths = _dataset_strings(h5, f"{catalog}/dataset_path")
            for group_name, dataset_path in zip(group_names, dataset_paths):
                path = _trace_data_tail(dataset_path)
                if (group_name == group or path == group) and path in h5:
                    return path
        raise KeyError(f"Group '{group}' not found in trace metadata")

    @staticmethod
    def _is_indexed_packed_h5(h5) -> bool:
        if "trace_index" not in h5 or "trace_data" not in h5:
            return False
        layout = _dataset_strings(h5, "trace_index/layout_kind")
        if layout:
            return "indexed_frequency_trace_v1" in layout
        return "trace_index/datasets/packed_path" in h5

    @staticmethod
    def _indexed_dataset_rows(h5, source_path: Optional[str] = None) -> list[dict]:
        if "trace_index/datasets/packed_path" not in h5:
            return []
        dataset_numbers = _dataset_ints(h5, "trace_index/datasets/dataset_number")
        source_paths = _dataset_strings(h5, "trace_index/datasets/source_path")
        packed_paths = _dataset_strings(h5, "trace_index/datasets/packed_path")
        n_rows = min(len(dataset_numbers), len(source_paths), len(packed_paths))
        if n_rows == 0:
            return []

        frequency_numbers = _dataset_ints(h5, "trace_index/dataset_number")
        frequencies = (
            np.asarray(h5["trace_index/frequency"][()]).ravel()
            if "trace_index/frequency" in h5
            else (
                np.asarray(h5["frequency"][()]).ravel()
                if "frequency" in h5
                else np.asarray([])
            )
        )
        frequency_by_number = {
            int(number): float(np.real(freq))
            for number, freq in zip(frequency_numbers, frequencies)
        }
        if not frequency_by_number and len(frequencies):
            frequency_by_number = {
                index: float(np.real(freq))
                for index, freq in enumerate(frequencies, start=1)
            }
        laplace_values = (
            np.asarray(h5["trace_index/laplace"][()]).ravel()
            if "trace_index/laplace" in h5
            else (
                np.asarray(h5["laplace"][()]).ravel()
                if "laplace" in h5
                else np.zeros_like(frequencies, dtype=float)
            )
        )
        if laplace_values.size == 1 and frequency_numbers.size > 1:
            laplace_values = np.full(frequency_numbers.size, laplace_values[0])
        laplace_by_number = {
            int(number): float(np.real(value))
            for number, value in zip(frequency_numbers, laplace_values)
        }
        if not laplace_by_number and len(laplace_values):
            laplace_by_number = {
                index: float(np.real(value))
                for index, value in enumerate(laplace_values, start=1)
            }

        wanted = _clean_h5_path(source_path) if source_path is not None else None
        rows = []
        for index in range(n_rows):
            source = _clean_h5_path(source_paths[index])
            packed = _clean_h5_path(packed_paths[index])
            number = int(dataset_numbers[index])
            if wanted is not None and source != wanted:
                continue
            if packed not in h5:
                continue
            rows.append(
                {
                    "dataset_number": number,
                    "source_path": source,
                    "packed_path": packed,
                    "frequency": frequency_by_number.get(number),
                    "laplace": laplace_by_number.get(number, 0.0),
                }
            )
        rows.sort(
            key=lambda row: (
                float("inf") if row["frequency"] is None else row["frequency"],
                row["dataset_number"],
            )
        )
        return rows

    @staticmethod
    def _indexed_source_path_for_group(h5, group: str) -> str:
        group = str(group).strip("/")
        catalog = "survey/receiver_groups/_catalog"
        if catalog in h5:
            group_names = _dataset_strings(h5, f"{catalog}/group_name")
            dataset_paths = _dataset_strings(h5, f"{catalog}/dataset_path")
            for group_name, dataset_path in zip(group_names, dataset_paths):
                if group_name == group or _trace_data_tail(dataset_path) == group:
                    return _clean_h5_path(dataset_path)
        return _clean_h5_path(group)

    @staticmethod
    def _indexed_trace_rows(h5, group: str) -> list[dict]:
        source_path = TraceStore._indexed_source_path_for_group(h5, group)
        rows = TraceStore._indexed_dataset_rows(h5, source_path)
        if not rows:
            raise KeyError(f"Group '{group}' not found in indexed packed trace file")
        return rows

    @staticmethod
    def _indexed_trace_paths(h5, group: str) -> list[str]:
        return [row["packed_path"] for row in TraceStore._indexed_trace_rows(h5, group)]

    @staticmethod
    def _h5_trace_groups(h5, configured: Optional[Iterable[str]] = None) -> list[str]:
        configured_groups = [str(group) for group in configured or []]
        if configured_groups:
            return [
                name
                for name in configured_groups
                if (
                    name in h5
                    and isinstance(h5[name], h5py.Dataset)
                    or (
                        TraceStore._is_indexed_packed_h5(h5)
                        and TraceStore._indexed_dataset_rows(
                            h5, TraceStore._indexed_source_path_for_group(h5, name)
                        )
                    )
                )
            ]
        if TraceStore._is_indexed_packed_h5(h5):
            catalog = "survey/receiver_groups/_catalog"
            if catalog in h5:
                group_names = _dataset_strings(h5, f"{catalog}/group_name")
                dataset_paths = _dataset_strings(h5, f"{catalog}/dataset_path")
                groups = []
                for index, dataset_path in enumerate(dataset_paths):
                    group = (
                        group_names[index]
                        if index < len(group_names) and group_names[index]
                        else _trace_data_tail(dataset_path)
                    )
                    if TraceStore._indexed_dataset_rows(h5, dataset_path):
                        groups.append(group)
                return groups
            source_paths = [
                row["source_path"] for row in TraceStore._indexed_dataset_rows(h5)
            ]
            return [
                _trace_data_tail(path)
                for path in _unique_preserve_order(source_paths)
                if not _trace_data_tail(path).startswith("k_domain/")
            ]
        return [
            name
            for name, item in h5.items()
            if isinstance(item, h5py.Dataset)
            and name not in _ROOT_TRACE_METADATA_DATASETS
            and "dims" in item.attrs
        ]

    @staticmethod
    def discover_trace_groups(
        file: Union[str, Path],
        configured: Optional[Iterable[str]] = None,
    ) -> list[str]:
        """Discover trace group names in an HDF5 trace file.

        Args:
            file: HDF5 trace file path.
            configured: Optional configured group order/filter.

        Returns:
            List of trace group names.
        """

        with h5py.File(file, "r") as h5:
            return TraceStore._h5_trace_groups(h5, configured)

    @staticmethod
    def _is_packed_trace_file(path: Path) -> bool:
        try:
            with h5py.File(path, "r") as h5:
                if "frequency" not in h5:
                    return False
                frequency = np.asarray(h5["frequency"][()])
                if frequency.ndim == 0:
                    return False
                if TraceStore._is_indexed_packed_h5(h5):
                    return bool(TraceStore._indexed_dataset_rows(h5))
                groups = TraceStore._h5_trace_groups(h5)
                return any(
                    _decode_dim_list(h5[group].attrs["dims"])[-1] == "frequency"
                    for group in groups
                )
        except OSError:
            return False

    @staticmethod
    def _candidate_packed_trace_files(records: Iterable[Path]) -> list[Path]:
        candidates: list[Path] = []
        seen = set()
        for record in records:
            stem = record.stem
            prefix = stem.rsplit("_", 1)[0] if "_" in stem else stem
            for candidate in (
                record.parent / f"{prefix}.h5",
                record.parent / "traces.h5",
                record.parent.parent / f"{prefix}.h5",
                record.parent.parent / "traces.h5",
            ):
                key = str(candidate.resolve(strict=False))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
        return candidates

    def _expected_frequencies(self) -> list[float]:
        f_map = self.metadata.get("f_map", {})
        if not isinstance(f_map, dict):
            return []
        values = []
        for value in f_map.values():
            try:
                values.append(float(np.real(value)))
            except (TypeError, ValueError):
                continue
        return sorted(values)

    def _expected_frequency_indices(
        self,
        available: Iterable[float],
    ) -> Optional[list[int]]:
        expected = self._expected_frequencies()
        if not expected:
            return None
        available_values = np.asarray(available, dtype=float).ravel()
        indices: list[int] = []
        used: set[int] = set()
        for frequency in expected:
            matches = np.flatnonzero(
                np.isclose(
                    available_values,
                    float(frequency),
                    rtol=0.0,
                    atol=1.0e-9,
                )
            )
            for match in matches:
                index = int(match)
                if index not in used:
                    indices.append(index)
                    used.add(index)
                    break
        return indices

    def _expected_frequency_laplace_indices(
        self,
        available: Iterable[float],
        laplace: Optional[Iterable[float]],
    ) -> Optional[list[int]]:
        expected = self._expected_frequencies()
        if not expected:
            return None
        available_values = np.asarray(available, dtype=float).ravel()
        laplace_values = (
            np.asarray(laplace, dtype=float).ravel() if laplace is not None else None
        )
        if laplace_values is None or laplace_values.size != available_values.size:
            return self._expected_frequency_indices(available_values)
        if not self._metadata_has_laplace_values(expected):
            return self._expected_frequency_indices(available_values)

        expected_laplace = self._metadata_laplace_values(expected)
        indices: list[int] = []
        used: set[int] = set()
        for frequency, laplace_value in zip(expected, expected_laplace):
            real_matches = np.isclose(
                available_values,
                float(frequency),
                rtol=0.0,
                atol=1.0e-9,
            )
            laplace_matches = np.isclose(
                laplace_values,
                float(laplace_value),
                rtol=0.0,
                atol=1.0e-12,
            )
            for match in np.flatnonzero(real_matches & laplace_matches):
                index = int(match)
                if index not in used:
                    indices.append(index)
                    used.add(index)
                    break
        return indices

    def _metadata_has_laplace_values(self, frequencies: Iterable[float]) -> bool:
        f_map = self.metadata.get("f_map", {})
        laplace_map = self.metadata.get("laplace_map", {})
        explicit_keys = self.metadata.get("laplace_map_keys")
        if explicit_keys is None:
            if not isinstance(laplace_map, dict) or not laplace_map:
                return False
            explicit = {str(key) for key in laplace_map}
        else:
            explicit = {str(key) for key in explicit_keys}
        if not explicit:
            return False
        if not isinstance(f_map, dict):
            return True

        pairs = []
        for index, frequency in f_map.items():
            try:
                pairs.append((str(index), float(np.real(frequency))))
            except (TypeError, ValueError):
                continue
        used: set[int] = set()
        for frequency in frequencies:
            match = None
            for index, (key, known_frequency) in enumerate(pairs):
                if index in used:
                    continue
                if np.isclose(
                    float(frequency),
                    known_frequency,
                    rtol=0.0,
                    atol=1.0e-9,
                ):
                    match = (index, key)
                    break
            if match is None:
                return False
            used.add(match[0])
            if match[1] not in explicit:
                return False
        return True

    def _filter_expected_frequencies(self, values: Iterable[float]) -> np.ndarray:
        frequencies = np.asarray(values, dtype=float).ravel()
        indices = self._expected_frequency_indices(frequencies)
        if indices is None:
            return frequencies
        return frequencies[indices]

    def _filter_expected_frequency_data(self, data: DataArray) -> DataArray:
        if "frequency" not in data.dims or "frequency" not in data.coords:
            return data
        laplace = data.coords["laplace"].values if "laplace" in data.coords else None
        indices = self._expected_frequency_laplace_indices(
            data.coords["frequency"].values,
            laplace,
        )
        if indices is None:
            return data
        return data.isel(frequency=indices)

    def _filter_expected_indexed_rows(self, rows: list[dict]) -> list[dict]:
        comparable_rows = [row for row in rows if row["frequency"] is not None]
        indices = self._expected_frequency_laplace_indices(
            [row["frequency"] for row in comparable_rows],
            [row.get("laplace", 0.0) for row in comparable_rows],
        )
        if indices is None:
            return rows
        filtered = []
        frequency_index = 0
        wanted = set(indices)
        for row in rows:
            if row["frequency"] is None:
                continue
            if frequency_index in wanted:
                filtered.append(row)
            frequency_index += 1
        return filtered

    def _metadata_laplace_values(self, frequencies: Iterable[float]) -> np.ndarray:
        f_map = self.metadata.get("f_map", {})
        laplace_map = self.metadata.get("laplace_map", {})
        pairs = []
        if isinstance(f_map, dict) and isinstance(laplace_map, dict):
            for index, frequency in f_map.items():
                try:
                    pairs.append(
                        (
                            float(np.real(frequency)),
                            float(
                                laplace_map.get(index, laplace_map.get(str(index), 0.0))
                            ),
                        )
                    )
                except (TypeError, ValueError):
                    continue
        out = []
        for frequency in frequencies:
            value = 0.0
            for known_frequency, known_laplace in pairs:
                if np.isclose(float(frequency), known_frequency, rtol=0.0, atol=1.0e-9):
                    value = known_laplace
                    break
            out.append(value)
        return np.asarray(out, dtype=float)

    @classmethod
    def _packed_trace_covers_frequencies(
        cls, path: Path, expected: Iterable[float]
    ) -> bool:
        if not path.exists() or not cls._is_packed_trace_file(path):
            return False
        expected_values = list(expected)
        if not expected_values:
            return True
        try:
            available = np.asarray(cls._read_trace_frequencies(path), dtype=float)
        except (OSError, KeyError, ValueError):
            return False
        return all(
            np.any(np.isclose(available, float(freq), rtol=0.0, atol=1.0e-9))
            for freq in expected_values
        )

    @classmethod
    def _packed_trace_contains_groups(
        cls,
        path: Path,
        groups: Iterable[str],
    ) -> bool:
        requested = [str(group) for group in groups]
        if not requested:
            return True
        if not path.exists():
            return False
        try:
            available = set(cls.discover_trace_groups(path, requested))
        except (OSError, KeyError, ValueError):
            return False
        return all(group in available for group in requested)

    def _packed_trace_file_for_records(self, records: Iterable[Path]) -> Optional[Path]:
        expected = self._expected_frequencies()
        groups = [str(group) for group in self.metadata.get("groups", [])]
        for candidate in self._candidate_packed_trace_files(records):
            if self._packed_trace_covers_frequencies(
                candidate, expected
            ) and self._packed_trace_contains_groups(candidate, groups):
                return candidate
        return None

    def _packed_group_file_map(self, records: Iterable[Path]) -> Dict[str, Path]:
        expected = self._expected_frequencies()
        configured = [str(group) for group in self.metadata.get("groups", [])]
        group_files: Dict[str, Path] = {}
        for record in records:
            if not record.exists() or not self._is_packed_trace_file(record):
                continue
            if not self._packed_trace_covers_frequencies(record, expected):
                continue
            try:
                groups = self.discover_trace_groups(record, configured)
            except (OSError, KeyError, ValueError):
                continue
            for group in groups:
                group_files.setdefault(str(group), record)
        if configured and not all(group in group_files for group in configured):
            return {}
        return group_files

    def _candidate_frequency_shard_files(self, records: Iterable[Path]) -> list[Path]:
        records = [Path(record) for record in records]
        candidates: list[Path] = []
        seen: set[str] = set()
        record_keys = {str(path.resolve(strict=False)) for path in records}

        def add_file(path: Path) -> None:
            key = str(path.resolve(strict=False))
            if key in seen:
                return
            seen.add(key)
            candidates.append(path)

        def add_shard_dir(shard_dir: Path) -> None:
            if not shard_dir.exists():
                return
            modern = sorted(shard_dir.glob(_MODERN_TRACE_SHARD_GLOB))
            files = modern
            if not files:
                files = [
                    path
                    for pattern in _LEGACY_TRACE_SHARD_GLOBS
                    for path in sorted(shard_dir.glob(pattern))
                ]
            for path in files:
                add_file(path)

        def add_root(root: Optional[Union[str, Path]]) -> None:
            if root is None:
                return
            root = Path(root)
            for shard_dir in (
                root / "shards",
                root / "traces" / "shards",
                root / "wavefields" / "shards",
            ):
                add_shard_dir(shard_dir)
            if root.exists():
                for pattern in (_MODERN_TRACE_SHARD_GLOB, *_LEGACY_TRACE_SHARD_GLOBS):
                    for path in sorted(root.glob(pattern)):
                        add_file(path)

        add_root(self.metadata.get("output_path"))
        add_root(self.metadata.get("result_path"))
        for record in records:
            add_root(record.parent)
            add_root(record.parent.parent)

        configured_groups = [str(group) for group in self.metadata.get("groups", [])]
        expected = self._expected_frequencies()
        if expected:
            files_by_frequency: dict[int, Path] = {}
        else:
            files = []

        for candidate in candidates:
            key = str(candidate.resolve(strict=False))
            if (
                key in record_keys
                or not candidate.exists()
                or self._is_packed_trace_file(candidate)
            ):
                continue
            try:
                frequencies = self._read_trace_frequencies(candidate)
            except (OSError, KeyError, ValueError):
                continue
            if len(frequencies) != 1:
                continue
            if configured_groups:
                try:
                    available_groups = set(
                        self.discover_trace_groups(candidate, configured_groups)
                    )
                except (OSError, KeyError, ValueError):
                    available_groups = set()
                if not all(group in available_groups for group in configured_groups):
                    metadata_file = self._trace_metadata_file_for_records(
                        [candidate, *records]
                    )
                    if metadata_file is not None:
                        try:
                            available_groups = set(
                                self.discover_trace_groups(
                                    metadata_file,
                                    configured_groups,
                                )
                            )
                        except (OSError, KeyError, ValueError):
                            available_groups = set()
                if not all(group in available_groups for group in configured_groups):
                    continue
            if not expected:
                files.append(candidate)
                continue
            for index, frequency in enumerate(expected):
                if index in files_by_frequency:
                    continue
                if np.isclose(
                    float(frequencies[0]),
                    float(frequency),
                    rtol=0.0,
                    atol=1.0e-9,
                ):
                    files_by_frequency[index] = candidate
                    break

        if not expected:
            return files
        files = [files_by_frequency[index] for index in sorted(files_by_frequency)]
        if files and len(files) < len(expected):
            warnings.warn(
                "Trace shards are missing "
                f"{len(expected) - len(files)} of {len(expected)} expected "
                "frequencies; building a VDS from the available shards.",
                RuntimeWarning,
                stacklevel=2,
            )
        return files

    def _order_frequency_files(self, files: Iterable[Path]) -> list[Path]:
        expected = self._expected_frequencies()
        files = list(files)
        if not expected:
            return files

        files_by_frequency: dict[int, Path] = {}
        for path in files:
            try:
                frequencies = self._read_trace_frequencies(path)
            except (OSError, KeyError, ValueError):
                continue
            if len(frequencies) != 1:
                continue
            for index, frequency in enumerate(expected):
                if index in files_by_frequency:
                    continue
                if np.isclose(
                    float(frequencies[0]),
                    float(frequency),
                    rtol=0.0,
                    atol=1.0e-9,
                ):
                    files_by_frequency[index] = path
                    break
        ordered = [files_by_frequency[index] for index in sorted(files_by_frequency)]
        return ordered or files

    def consolidate(self, cache_dir: Optional[Union[str, Path]] = None) -> Path:
        """Create or select a packed trace file with frequency as leading axis.

        Args:
            cache_dir: Optional directory for the generated virtual dataset.

        Returns:
            Path to a packed or virtual HDF5 trace file.
        """

        records = [Path(file) for file in self.files]
        if not records:
            raise FileNotFoundError("No trace files were provided")
        self._packed_group_files = {}

        packed = self._packed_trace_file_for_records(records)
        if packed is not None:
            self.close()
            self._consolidated = packed
            return self._consolidated

        group_files = self._packed_group_file_map(records)
        if group_files:
            self.close()
            self._packed_group_files = group_files
            self._consolidated = next(iter(group_files.values()))
            return self._consolidated

        if (
            len(records) == 1
            and records[0].exists()
            and self._is_packed_trace_file(records[0])
        ):
            self.close()
            self._consolidated = records[0]
            return self._consolidated

        available = []
        missing = []
        shard_files: list[Path] = []
        for record in records:
            if record.exists():
                available.append(record)
            else:
                missing.append(record)

        if missing:
            shard_files = self._candidate_frequency_shard_files(records)
            if shard_files:
                available = self._order_frequency_files([*available, *shard_files])
                message = (
                    "Trace files are missing; creating a VDS from "
                    f"{len(available)} matching frequency shard(s)."
                    if not any(record.exists() for record in records)
                    else "Trace files are missing; filling the VDS from matching "
                    "frequency shard(s)."
                )
                warnings.warn(
                    message,
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif not available:
                for record in missing:
                    warnings.warn(
                        "Trace file is missing and will be omitted from the VDS: "
                        f"{record}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

        if not available:
            raise FileNotFoundError("No trace files exist")

        if missing and any(record.exists() for record in records) and not shard_files:
            for record in missing:
                warnings.warn(
                    f"Trace file is missing and will be omitted from the VDS: {record}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        metadata_record = self._trace_metadata_file_for_records(
            [*records, *available]
        ) or (records[0] if records[0].exists() else available[0])
        output_record = records[0] if records else metadata_record
        freqs = [self._read_trace_frequency(record) for record in available]
        laplace = []
        metadata_laplace = self._metadata_laplace_values(freqs)
        for index, record in enumerate(available):
            try:
                laplace.append(self._read_trace_laplace(record))
            except (OSError, KeyError, ValueError):
                laplace.append(float(metadata_laplace[index]))
        cache_dir = Path(cache_dir) if cache_dir is not None else self._cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        new_file = self._consolidated_path(output_record, cache_dir=cache_dir)
        new_file.parent.mkdir(parents=True, exist_ok=True)

        self.close()
        if new_file.exists():
            new_file.unlink()

        with h5py.File(metadata_record, "r") as first, h5py.File(new_file, "w") as nf:
            nf.create_dataset("frequency", data=np.asarray(freqs, dtype=float))
            nf.create_dataset("laplace", data=np.asarray(laplace, dtype=float))
            if "survey" in first:
                first.copy("survey", nf)

            for group in self.metadata["groups"]:
                template_path = self._template_dataset_path_for_group(first, group)
                source = first[template_path]
                source_shape = source.shape
                payload_path = _trace_data_tail(template_path)
                layout = h5py.VirtualLayout(
                    shape=(len(available),) + source_shape,
                    dtype=source.dtype,
                )
                for index, record in enumerate(available):
                    with h5py.File(record, "r") as shard:
                        if payload_path not in shard:
                            raise KeyError(
                                f"Dataset '{payload_path}' not found in HDF5 {record}"
                            )
                        if shard[payload_path].shape != source_shape:
                            raise ValueError(
                                "Trace payload shape does not match metadata "
                                f"template for group {group!r}: {record}"
                            )
                    layout[index] = h5py.VirtualSource(
                        str(record), payload_path, shape=source_shape
                    )
                nf.create_virtual_dataset(group, layout)

                dset = nf[group]
                _copy_h5_attrs(source, dset)
                # The VDS stores axes in the template's physical HDF5 order.
                # Layout markers describe the template payload order and would
                # cause readers to reinterpret the VDS axes a second time.
                if "layout_kind" in dset.attrs:
                    del dset.attrs["layout_kind"]
                dims = [*_trace_axis_dims(source), "frequency"]
                dset.attrs["dims"] = dims
                for axis, dim in enumerate(dims[:-1]):
                    coord = (
                        source.attrs[dim]
                        if dim in source.attrs
                        else np.arange(1, source_shape[axis] + 1)
                    )
                    if len(coord) < 10000:
                        dset.attrs[dim] = coord
                dset.attrs["frequency"] = freqs

        self._consolidated = Path(new_file)
        return self._consolidated

    def read_h5(self, group: str) -> DataArray:
        """Open a lazy frequency-domain xarray view for one trace group.

        Args:
            group: Receiver or wavefield group name.

        Returns:
            Lazy ``xarray.DataArray`` with physical coordinates and complex-axis
            metadata.
        """

        try:
            import dask.array as da
        except ModuleNotFoundError as exc:
            from frequensolve._optional import optional_dependency_error

            raise optional_dependency_error(
                "TraceDataset lazy HDF5 reading",
                extra="hpc",
                dependencies=("dask",),
                error=exc,
            ) from exc

        h5 = h5py.File(self._trace_file_for_group(group), "r")
        self._open_files.append(h5)
        indexed_paths: list[str] = []
        indexed_frequencies: list[float] = []
        indexed_laplace: list[float] = []
        if self._is_indexed_packed_h5(h5) and group not in h5:
            indexed_rows = self._filter_expected_indexed_rows(
                self._indexed_trace_rows(h5, group)
            )
            if not indexed_rows:
                raise ValueError(
                    "Packed trace file does not contain any requested "
                    f"frequencies for group {group!r}"
                )
            indexed_paths = [row["packed_path"] for row in indexed_rows]
            indexed_frequencies = [
                row["frequency"] for row in indexed_rows if row["frequency"] is not None
            ]
            indexed_laplace = [row.get("laplace", 0.0) for row in indexed_rows]
            dset = h5[indexed_paths[0]]
        else:
            dset = h5[group]
        dims = _trace_data_dims(dset)
        data_shape = (len(indexed_paths), *dset.shape) if indexed_paths else dset.shape
        data_ndim = len(data_shape)
        if data_ndim == len(dims) + 1:
            dims.append("complex")
        dims = ["source" if dim == "shot" else dim for dim in dims]
        coords = {}
        for axis, dim in enumerate(dims):
            if dim == "complex":
                coords[dim] = (
                    ["real", "imag"]
                    if data_shape[axis] == 2
                    else np.arange(1, data_shape[axis] + 1)
                )
                continue
            attr_dim = "shot" if dim == "source" and "shot" in dset.attrs else dim
            survey_paths = []
            if dim == "receiver":
                survey_paths = [
                    f"survey/receiver_groups/{group}/traces/receiver_id",
                    f"survey/receiver_groups/{group}/receivers/receiver_id",
                    "survey/receivers/receiver_id",
                ]
            elif dim == "source":
                survey_paths = [
                    f"survey/receiver_groups/{group}/traces/source_id",
                    "survey/sources/source_id",
                ]
            elif dim == "component":
                survey_paths = [
                    f"survey/receiver_groups/{group}/traces/component_name",
                    f"survey/receiver_groups/{group}/components/component_name",
                    "survey/components/component_name",
                ]

            survey_path = next((path for path in survey_paths if path in h5), None)
            if survey_path is not None:
                values = _unique_preserve_order(_decode_h5_strings(h5[survey_path][()]))
                coords[dim] = (
                    values
                    if len(values) == data_shape[axis]
                    else np.arange(1, data_shape[axis] + 1)
                )
            elif dim == "frequency" and indexed_frequencies:
                coords[dim] = np.asarray(indexed_frequencies, dtype=float)
            elif dim == "frequency" and "frequency" in h5:
                coords[dim] = h5["frequency"][()]
            elif attr_dim in dset.attrs:
                coords[dim] = dset.attrs[attr_dim]
            else:
                coords[dim] = np.arange(1, data_shape[axis] + 1)
        if "complex" in dims and "complex" not in coords:
            coords["complex"] = ["real", "imag"]

        if indexed_paths:
            arrays = []
            for path in indexed_paths:
                item = h5[path]
                if item.shape != dset.shape:
                    raise ValueError(
                        "Indexed packed trace datasets for group "
                        f"{group!r} do not have a common shape"
                    )
                arrays.append(da.from_array(item, chunks=item.shape))
            data = da.stack(arrays, axis=0)
        else:
            chunks = (dset.shape[0], 1, 1, *dset.shape[3:])
            data = da.from_array(dset, chunks=chunks)
        fd = DataArray(data, dims=dims, coords=coords)
        if "frequency" in fd.dims:
            if indexed_laplace:
                laplace = np.asarray(indexed_laplace, dtype=float)
            elif "laplace" in h5:
                laplace = np.asarray(h5["laplace"][()]).ravel().astype(float)
            else:
                laplace = self._metadata_laplace_values(fd.coords["frequency"].values)
            if laplace.size == 1 and fd.sizes["frequency"] > 1:
                laplace = np.full(fd.sizes["frequency"], float(laplace[0]))
            if laplace.size == fd.sizes["frequency"]:
                fd = fd.assign_coords(laplace=("frequency", laplace))
            fd = self._filter_expected_frequency_data(fd)
        return fd

    def read_FD(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Optional[Wavelet] = None,
        **kwargs,
    ):
        """Read one complex frequency-domain gather.

        Args:
            group: Receiver group name.
            component: Component name.
            source: One-based source id.
            wavelet: Optional wavelet used to scale frequency samples.
            **kwargs: Optional wavelet/taper controls.

        Returns:
            Complex ``xarray.DataArray`` indexed by frequency and receiver axes.
        """

        if wavelet is None:
            dset = self.read_h5(group)
            gather = dset.sel(component=component, source=source)
            fd = gather.sel(complex="real") + 1j * gather.sel(complex="imag")
            fd = fd.fillna(0)
            fd.attrs.update(
                self._trace_array_attrs(
                    group,
                    component=component,
                    source=source,
                    domain="frequency",
                )
            )
            return fd
        else:
            sampling = UniformSweepSampling(
                f_min=0.0,
                f_max=self.metadata["f_max"],
                df=self.metadata["df"],
            )
            wavelet.times = sampling.T_list
            dset = self.read_h5(group)
            gather = dset.sel(component=component, source=source)
            fd = gather.sel(complex="real") + 1j * gather.sel(complex="imag")
            fd = fd.fillna(0)
            fd = self._apply_wavelet_to_fd(fd, wavelet, **kwargs)
            fd.attrs.update(
                self._trace_array_attrs(
                    group,
                    component=component,
                    source=source,
                    domain="frequency",
                )
            )
            return fd

    def _trace_array_attrs(
        self,
        group: str,
        *,
        component: str,
        source: int,
        domain: str,
    ) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {
            "source_id": source,
            "receiver_group": group,
            "project_path": str(self.metadata["project"]),
            "simulation": str(self.metadata["simulation"]),
            "long_name": f"{component}",
            "domain": domain,
        }
        wavefield = self.metadata.get("wavefields", {}).get(group)
        if isinstance(wavefield, dict):
            attrs["wavefield_output"] = group
            grid = wavefield.get("grid")
            if grid is not None:
                attrs["wavefield_grid"] = grid
            if "path" in wavefield:
                attrs["wavefield_path"] = wavefield["path"]
        return attrs

    def _apply_wavelet_to_fd(
        self,
        fd: DataArray,
        wavelet: Wavelet,
        **kwargs,
    ) -> DataArray:
        freqs = wavelet.frequencies
        spectrum = DataArray(
            wavelet.spectrum, dims=["frequency"], coords={"frequency": freqs}
        )
        w = spectrum.interp(
            frequency=fd.coords["frequency"].values, kwargs={"fill_value": 0}
        )
        if "f_taper" in kwargs:
            alpha = kwargs["f_taper"]
            if alpha > 0:
                from scipy.signal.windows import tukey

                dim = "frequency"
                data = tukey(2 * fd.sizes[dim], alpha=alpha)
                data = data[fd.sizes[dim] :]
                window = DataArray(
                    data,
                    dims=[dim],
                    coords={dim: fd[dim]},
                )
                w *= window
        return fd * w

    @staticmethod
    def _normalize_laplace_compensation(value: Union[str, bool]) -> str:
        if isinstance(value, bool):
            return "on" if value else "off"
        value = str(value).lower()
        if value not in {"auto", "on", "off"}:
            raise ValueError("laplace_compensation must be 'auto', 'on', or 'off'")
        return value

    @staticmethod
    def _active_frequency_mask(gather: DataArray) -> Optional[np.ndarray]:
        if "frequency" not in gather.dims:
            return None
        frequency_size = gather.sizes["frequency"]
        axes = tuple(axis for axis, dim in enumerate(gather.dims) if dim != "frequency")
        try:
            amplitude = np.abs(gather.data)
            if axes:
                amplitude = amplitude.max(axis=axes)
            compute = getattr(amplitude, "compute", None)
            if callable(compute):
                amplitude = compute()
        except Exception:
            return None
        amplitude = np.asarray(amplitude).ravel()
        if amplitude.size != frequency_size:
            return None
        return np.isfinite(amplitude) & (amplitude > 0.0)

    @staticmethod
    def _uniform_laplace(gather: DataArray) -> float:
        if "laplace" not in gather.coords:
            return 0.0
        values = np.asarray(gather.coords["laplace"].values, dtype=float).ravel()
        if values.size == 0:
            return 0.0
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0
        first = float(finite[0])
        if np.allclose(finite, first, rtol=0.0, atol=1.0e-12):
            return first

        active = TraceStore._active_frequency_mask(gather)
        if active is not None and active.size == values.size:
            finite = values[np.isfinite(values) & active]
            if finite.size == 0:
                return 0.0
            first = float(finite[0])
            if np.allclose(finite, first, rtol=0.0, atol=1.0e-12):
                return first

        raise ValueError("Time-domain reconstruction requires a uniform Laplace offset")

    @staticmethod
    def _damping_factor(laplace: float, period: float) -> float:
        return float(np.exp(-2.0 * np.pi * laplace * period))

    def read_TD(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        laplace_compensation: Union[str, bool] = "auto",
        **kwargs,
    ) -> DataArray:
        """Read one reconstructed time-domain gather.

        Args:
            group: Receiver group name.
            component: Component name.
            source: One-based source id.
            wavelet: Wavelet used for inverse transformation.
            upscale: Reconstruction upscaling factor.
            T_max: Optional maximum time to return.
            laplace_compensation: ``"auto"``, ``"on"``, ``"off"``, or boolean
                compatibility value controlling Laplace damping compensation.
            **kwargs: Optional frequency-domain read controls.

        Returns:
            Time-domain ``xarray.DataArray``.
        """

        compensation_mode = self._normalize_laplace_compensation(laplace_compensation)
        sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=self.metadata["f_max"],
            df=self.metadata["df"],
            upscale=upscale,
        )
        fd = self.read_FD(group, component, source, wavelet, **kwargs)
        laplace = self._uniform_laplace(fd)
        wavelet.times = sampling.T_list

        fd = fd.interp(frequency=sampling.F_list, kwargs={"fill_value": 0})
        fft = get_fft_backend()
        td = fft.irfft(fd.data, axis=0)
        dims = ["time" if d == "frequency" else d for d in fd.dims]
        coords = {}
        for d in dims:
            if d in fd.coords:
                coords[d] = fd.coords[d]
            else:
                coords[d] = sampling.T_list[:-1] - wavelet.center

        td = DataArray(data=td, dims=dims, coords=coords)
        compensated = compensation_mode == "on" or (
            compensation_mode == "auto" and not np.isclose(laplace, 0.0)
        )
        if compensated:
            exp = DataArray(
                np.exp(-2 * np.pi * laplace * td.coords["time"]),
                dims=["time"],
                coords={"time": td.coords["time"]},
            )
            td = td * exp
        if T_max is not None:
            td = td.sel(time=slice(None, T_max))

        td.attrs.update(
            self._trace_array_attrs(
                group,
                component=component,
                source=source,
                domain="time",
            )
        )
        td.attrs["laplace"] = laplace
        td.attrs["laplace_compensated"] = compensated
        td.attrs["damping_factor"] = self._damping_factor(laplace, sampling.T)
        for d in td.dims:
            td.coords[d].attrs["long_name"] = d.title()
            if d == "time":
                td.coords[d].attrs["units"] = "s"
                td.coords[d].attrs["description"] = "Time"
            elif d == "frequency":
                td.coords[d].attrs["units"] = "Hz"
                td.coords[d].attrs["description"] = "Frequency"
        return td

    def read_LD(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        **kwargs,
    ) -> DataArray:
        """Read one Laplace-domain time gather without compensation."""

        td = self.read_TD(
            group,
            component,
            source,
            wavelet,
            upscale=upscale,
            T_max=T_max,
            laplace_compensation="off",
            **kwargs,
        )
        td.attrs["domain"] = "laplace_time"
        td.attrs["laplace_compensated"] = False
        return td

    def CosineWindow(self, t0: float, tf: float, taper: float) -> DataArray:
        """Create a cosine-tapered time window.

        Args:
            t0: Start time of the flat window.
            tf: End time of the flat window.
            taper: Taper duration on each side.

        Returns:
            Window ``DataArray`` indexed by time.
        """

        sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=self.metadata["f_max"],
            df=self.metadata["df"],
            upscale=1,
        )
        dt = sampling.T_list[1] - sampling.T_list[0]
        N = len(sampling.T_list)
        start = np.argmin(np.abs(sampling.T_list - t0))
        end = np.argmin(np.abs(sampling.T_list - tf))
        w = np.zeros(N)
        w[start:end] = 1.0

        L = int(taper / dt)
        if L > 0:
            L_left = min(L, start)
            if L_left > 0:
                idx = np.arange(start - L_left, start)
                n = np.arange(L_left)
                w[idx] = 0.5 - 0.5 * np.cos(np.pi * (n + 1) / (L_left + 1))
            L_right = min(L, N - end)
            if L_right > 0:
                idx = np.arange(end, end + L_right)
                n = np.arange(L_right)
                w[idx] = 0.5 + 0.5 * np.cos(np.pi * (n + 1) / (L_right + 1))
        window = DataArray(w, dims=["time"], coords={"time": sampling.T_list})
        return window

    def read_windowed_TD(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        window: DataArray,
        N_window: int,
        upscale: int = 1,
        T_max: Optional[float] = None,
        **kwargs,
    ) -> DataArray:
        """Read a time-domain gather after applying a frequency-domain window.

        Args:
            group: Receiver group name.
            component: Component name.
            source: One-based source id.
            wavelet: Wavelet used for inverse transformation.
            window: Time-domain window.
            N_window: Number of frequency bins on each side of the window
                kernel.
            upscale: Reconstruction upscaling factor.
            T_max: Optional maximum time to return.
            **kwargs: Optional frequency-domain read controls.

        Returns:
            Windowed time-domain ``xarray.DataArray``.
        """

        sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=self.metadata["f_max"],
            df=self.metadata["df"],
            upscale=upscale,
        )
        fd = self.read_FD(group, component, source, wavelet, **kwargs)
        wavelet.times = sampling.T_list

        fd = fd.interp(frequency=sampling.F_list, kwargs={"fill_value": 0})

        N = len(window.data)

        fft = get_fft_backend()
        W = fft.fft(window.data) / N
        offs = np.arange(-N_window, N_window + 1)
        offs2 = np.arange(-2 * N_window, 2 * N_window + 1)
        idxs = offs % N
        kernel = W[idxs]

        # import matplotlib.pyplot as plt
        # plt.plot(offs2,abs(W[offs2]),'r--')
        # plt.plot(offs,abs(kernel),'b-')
        # plt.show()

        X = fd.data.compute()
        Y = 0.0
        for off, c in zip(offs, kernel):
            Y += c * np.roll(X, shift=off, axis=0)

        td = fft.irfft(Y, axis=0)
        dims = ["time" if d == "frequency" else d for d in fd.dims]
        coords = {}
        for d in dims:
            if d in fd.coords:
                coords[d] = fd.coords[d]
            else:
                coords[d] = sampling.T_list[:-1] - wavelet.center

        td = DataArray(data=td, dims=dims, coords=coords)
        if T_max is not None:
            td = td.sel(time=slice(None, T_max))

        td.attrs.update(
            self._trace_array_attrs(
                group,
                component=component,
                source=source,
                domain="time",
            )
        )
        for d in td.dims:
            td.coords[d].attrs["long_name"] = d.title()
            if d == "time":
                td.coords[d].attrs["units"] = "s"
                td.coords[d].attrs["description"] = "Time"
            elif d == "frequency":
                td.coords[d].attrs["units"] = "Hz"
                td.coords[d].attrs["description"] = "Frequency"
        return td
