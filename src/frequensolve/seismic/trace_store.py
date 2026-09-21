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
_TD_EAGER_MAX_BYTES = 64 * 1024**2
_TD_LAZY_CHUNK_BYTES = 16 * 1024**2


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

    @staticmethod
    def _read_h5_group(group: h5py.Group) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                out[key] = _decode_h5_strings(item[()]).tolist()
            elif isinstance(item, h5py.Group):
                out[key] = TraceStore._read_h5_group(item)
        return out

    def properties(self, group: str) -> list[str]:
        """Return static material properties stored with a wavefield group."""

        with h5py.File(self._trace_file_for_group(group), "r") as h5:
            if "properties" not in h5 or not isinstance(h5["properties"], h5py.Group):
                return []
            return list(h5["properties"].keys())

    def material_property(self, group: str, name: str) -> DataArray:
        """Read one realized material property on a wavefield receiver grid."""

        path = f"properties/{name}"
        with h5py.File(self._trace_file_for_group(group), "r") as h5:
            if path not in h5:
                properties = h5.get("properties")
                available = (
                    list(properties.keys())
                    if isinstance(properties, h5py.Group)
                    else []
                )
                raise KeyError(
                    f"Property {name!r} not found for {group!r}; available: {available}"
                )
            dset = h5[path]
            values = np.asarray(dset[()]).reshape(-1)
            units = _decode_h5_strings(dset.attrs.get("units", [])).reshape(-1)

        receiver = np.asarray(self.receivers(group))
        if receiver.size != values.size:
            receiver = np.arange(1, values.size + 1)
        return DataArray(
            values,
            dims=("receiver",),
            coords={"receiver": receiver},
            name=name,
            attrs={
                "units": str(units[0]) if units.size else "",
                "receiver_group": group,
                "static": True,
            },
        )

    def survey_tables(self) -> Dict[str, Any]:
        """Return embedded survey metadata tables from the trace store."""

        self._ensure_consolidated()

        if self._packed_group_files:
            tables: Dict[str, Any] = {}
            for path in self._packed_group_files.values():
                with h5py.File(path, "r") as f:
                    if "survey" not in f:
                        continue
                    tables.update(self._read_h5_group(f["survey"]))
            return tables
        with h5py.File(self._consolidated, "r") as f:
            if "survey" not in f:
                return {}
            return self._read_h5_group(f["survey"])

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
                out += f"  {_gray('Window')}\t: {0:.2f} - {1.0 / df:.2f} s\n"
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
        expected_values = np.asarray(expected, dtype=float)
        if expected_values.size == available_values.size and np.allclose(
            expected_values, available_values, rtol=0.0, atol=1.0e-9
        ):
            return None
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

    def _select_gather(
        self, dset: DataArray, group: str, component: str, source: int
    ) -> DataArray:
        if "trace" not in dset.dims:
            return dset.sel(component=component, source=source)

        with h5py.File(self._trace_file_for_group(group), "r") as h5:
            survey = self._read_h5_group(h5["survey"]) if "survey" in h5 else {}
        receiver_group = survey.get("receiver_groups", {}).get(group, {})
        trace_table = receiver_group.get("traces", {})
        trace_count = dset.sizes["trace"]

        def column(name: str, default: Optional[Any] = None) -> np.ndarray:
            values = trace_table.get(name, default)
            if values is None:
                raise ValueError(f"Sparse trace catalog is missing {name!r}")
            values = np.asarray(values)
            if len(values) != trace_count:
                raise ValueError(
                    f"Sparse trace catalog field {name!r} has {len(values)} rows; "
                    f"expected {trace_count}"
                )
            return values

        source_ids = column("source_id")
        receiver_ids = column("receiver_id")
        component_ids = column("component", trace_table.get("component_id"))
        component_names = trace_table.get("component_name")
        if component_names is None:
            component_table = receiver_group.get("components", {})
            if "component_name" not in component_table:
                component_table = survey.get("components", {})
            identifiers = component_table.get(
                "component", component_table.get("component_id", [])
            )
            names = component_table.get("component_name", [])
            component_map = dict(zip(map(int, identifiers), map(str, names)))
            component_names = [
                component_map.get(int(value), value) for value in component_ids
            ]
        component_names = np.asarray(component_names)
        if len(component_names) != trace_count:
            raise ValueError(
                "Sparse trace component catalog does not match the trace axis"
            )

        selected = np.flatnonzero(
            (component_names == component) & (source_ids == source)
        )
        if not len(selected):
            raise KeyError(
                f"No sparse traces found for component {component!r}, source {source!r}"
            )
        gather = dset.isel(trace=selected).rename({"trace": "receiver"})

        receiver_catalog = receiver_group.get("receivers", {})
        if "coordinates" not in receiver_catalog:
            receiver_catalog = survey.get("receivers", {})
        catalog_ids = np.asarray(receiver_catalog.get("receiver_id", []))
        coordinates = np.asarray(receiver_catalog.get("coordinates", []), dtype=float)
        if coordinates.ndim == 2 and coordinates.shape[1] == len(catalog_ids):
            coordinates = coordinates.T
        coordinate_values: Dict[str, Any] = {}
        if coordinates.size:
            if coordinates.ndim != 2 or coordinates.shape[0] != len(catalog_ids):
                raise ValueError(
                    "Sparse receiver coordinates do not match the receiver catalog"
                )
            rows = {
                int(identifier): index for index, identifier in enumerate(catalog_ids)
            }
            if not all(int(receiver_ids[index]) in rows for index in selected):
                raise ValueError(
                    "Sparse trace receiver ids are missing from the receiver catalog"
                )
            names = ("x", "z") if coordinates.shape[1] == 2 else ("x", "y", "z")
            coordinate_values = {
                f"receiver_{name}": (
                    "receiver",
                    [
                        coordinates[rows[int(receiver_ids[index])], axis]
                        for index in selected
                    ],
                )
                for axis, name in enumerate(names[: coordinates.shape[1]])
            }

        return gather.assign_coords(
            {
                "receiver": ("receiver", receiver_ids[selected]),
                "trace_id": (
                    "receiver",
                    column("trace_id", np.arange(1, trace_count + 1))[selected],
                ),
                "receiver_id": ("receiver", receiver_ids[selected]),
                "source_id": ("receiver", source_ids[selected]),
                "component": ("receiver", component_names[selected]),
                "weight": (
                    "receiver",
                    column("weight", np.ones(trace_count, dtype=float))[selected],
                ),
                **coordinate_values,
            }
        )

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
        coords = self._trace_coordinates(
            h5,
            dset,
            group,
            dims,
            data_shape,
            indexed_frequencies=indexed_frequencies,
        )

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

    @staticmethod
    def _trace_coordinates(
        h5: Any,
        dset: Any,
        group: str,
        dims: list[str],
        data_shape: tuple[int, ...],
        *,
        indexed_frequencies: Iterable[float] = (),
    ) -> Dict[str, np.ndarray]:
        """Build physical trace coordinates without constructing data tasks."""

        coords = {}
        indexed_frequencies = list(indexed_frequencies)
        for axis, dim in enumerate(dims):
            if dim == "complex":
                coords[dim] = (
                    np.array(["real", "imag"])
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
            coords["complex"] = np.array(["real", "imag"])
        return coords

    @staticmethod
    def _coordinate_index(values: Any, value: Any, dim: str) -> int:
        """Return the exact coordinate position requested by a trace selection."""

        values = np.asarray(values)
        matches = np.flatnonzero(values == value)
        if matches.size == 0:
            raise KeyError(f"{dim!r} coordinate {value!r} was not found")
        return int(matches[0])

    def _read_fd_eager(
        self,
        group: str,
        component: str,
        source: int,
        *,
        max_bytes: int,
    ) -> Optional[DataArray]:
        """Read a small selected frequency-domain gather directly with h5py."""

        if max_bytes <= 0:
            return None

        with h5py.File(self._trace_file_for_group(group), "r") as h5:
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
                    row["frequency"]
                    for row in indexed_rows
                    if row["frequency"] is not None
                ]
                indexed_laplace = [row.get("laplace", 0.0) for row in indexed_rows]
                dset = h5[indexed_paths[0]]
            else:
                dset = h5[group]

            dims = _trace_data_dims(dset)
            data_shape = (
                (len(indexed_paths), *dset.shape) if indexed_paths else dset.shape
            )
            if len(data_shape) == len(dims) + 1:
                dims.append("complex")
            dims = ["source" if dim == "shot" else dim for dim in dims]
            if tuple(dims) != (
                "frequency",
                "source",
                "component",
                "receiver",
                "complex",
            ):
                return None

            coords = self._trace_coordinates(
                h5,
                dset,
                group,
                dims,
                data_shape,
                indexed_frequencies=indexed_frequencies,
            )
            if len(coords["complex"]) != 2:
                return None

            source_index = self._coordinate_index(coords["source"], source, "source")
            component_index = self._coordinate_index(
                coords["component"], component, "component"
            )
            n_frequency = len(coords["frequency"])
            n_receiver = len(coords["receiver"])
            nbytes = (
                n_frequency * n_receiver * len(coords["complex"]) * dset.dtype.itemsize
            )
            if nbytes > max_bytes:
                return None

            if indexed_paths:
                raw = np.stack(
                    [
                        np.asarray(h5[path][source_index, component_index, :, :])
                        for path in indexed_paths
                    ],
                    axis=0,
                )
            else:
                raw = np.asarray(dset[:, source_index, component_index, :, :])

            values = raw[..., 0] + 1j * raw[..., 1]
            values = np.where(np.isnan(values), 0.0, values)
            frequency = np.asarray(coords["frequency"], dtype=float)
            receiver = np.asarray(coords["receiver"])
            if indexed_laplace:
                laplace = np.asarray(indexed_laplace, dtype=float)
            elif "laplace" in h5:
                laplace = np.asarray(h5["laplace"][()]).ravel().astype(float)
            else:
                laplace = self._metadata_laplace_values(frequency)
            if laplace.size == 1 and frequency.size > 1:
                laplace = np.full(frequency.size, float(laplace[0]))

            indices = self._expected_frequency_laplace_indices(frequency, laplace)
            if indices is not None:
                frequency = frequency[indices]
                values = values[indices]
                if laplace.size == len(coords["frequency"]):
                    laplace = laplace[indices]

            fd = DataArray(
                values,
                dims=("frequency", "receiver"),
                coords={"frequency": frequency, "receiver": receiver},
            )
            if laplace.size == frequency.size:
                fd = fd.assign_coords(laplace=("frequency", laplace))
            fd.attrs.update(
                self._trace_array_attrs(
                    group,
                    component=component,
                    source=source,
                    domain="frequency",
                )
            )
            return fd

    @staticmethod
    def _coalesce_td_chunks(fd: DataArray) -> DataArray:
        """Bound the Dask graph used by a large lazy TD reconstruction."""

        if not callable(getattr(fd.data, "rechunk", None)):
            return fd
        chunks: Dict[str, int] = {"frequency": -1}
        if "receiver" in fd.dims:
            itemsize = np.dtype(fd.dtype).itemsize
            per_receiver = max(1, fd.sizes["frequency"] * itemsize)
            chunks["receiver"] = max(1, _TD_LAZY_CHUNK_BYTES // per_receiver)
        return fd.chunk(chunks)

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

        if wavelet is not None:
            sampling = UniformSweepSampling(
                f_min=0.0,
                f_max=self.metadata["f_max"],
                df=self.metadata["df"],
            )
            wavelet.times = sampling.T_list

        dset = self.read_h5(group)
        gather = self._select_gather(dset, group, component, source)
        fd = gather.sel(complex="real") + 1j * gather.sel(complex="imag")
        fd = fd.fillna(0)
        if wavelet is not None:
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
    def _sample_wavelet_spectrum(
        wavelet: Wavelet,
        times: np.ndarray,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        """Sample a wavelet spectrum at the requested frequencies."""

        from scipy.interpolate import CubicSpline

        wavelet.times = times
        wavelet_frequencies = np.asarray(wavelet.frequencies, dtype=float)
        wavelet_spectrum = np.asarray(wavelet.spectrum, dtype=np.complex128)
        spline = CubicSpline(wavelet_frequencies, wavelet_spectrum, extrapolate=False)
        return np.nan_to_num(spline(frequencies))

    def _read_raw_selected_fd(
        self,
        group: str,
        component: str,
        source: int,
        *,
        max_bytes: int,
    ) -> DataArray:
        """Read an unweighted base or derivative frequency gather."""

        fd = self._read_fd_eager(
            group,
            component,
            source,
            max_bytes=max_bytes,
        )
        if fd is None:
            fd = self.read_FD(group, component, source)
            fd = self._coalesce_td_chunks(fd)
        return fd

    def _derivative_assisted_spectrum(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        *,
        target_df: float,
        upscale: int,
        high_frequency_taper: Union[bool, float],
        interpolation_time_shift: float,
        max_bytes: int,
        reconstruction: str = "hermite",
    ) -> tuple[DataArray, UniformSweepSampling]:
        """Reconstruct or extend a spectrum using base and ``df`` channels."""

        from scipy.interpolate import CubicHermiteSpline

        if target_df <= 0.0:
            raise ValueError("target_df must be positive")
        interpolation_time_shift = float(interpolation_time_shift)
        if not np.isfinite(interpolation_time_shift):
            raise ValueError("interpolation_time_shift must be finite")

        base = self._read_raw_selected_fd(
            group,
            component,
            source,
            max_bytes=max_bytes,
        )
        order = self._frequency_derivative_order(group)
        derivative_group = (
            f"{group.rsplit('_', 1)[0]}_d{order + 1}f" if order else f"{group}_df"
        )
        try:
            derivative = self._read_raw_selected_fd(
                derivative_group,
                component,
                source,
                max_bytes=max_bytes,
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Derivative-assisted time reconstruction requires the packed "
                f"frequency-derivative trace group {derivative_group!r}"
            ) from exc

        base_frequency = np.asarray(base.coords["frequency"].values, dtype=float)
        derivative_frequency = np.asarray(
            derivative.coords["frequency"].values, dtype=float
        )
        if base_frequency.size < 2:
            raise ValueError(
                "Derivative-assisted time reconstruction requires at least two "
                "frequencies"
            )
        if base.dims != derivative.dims:
            raise ValueError("Base and frequency-derivative trace dimensions differ")
        if base_frequency.shape != derivative_frequency.shape or not np.allclose(
            base_frequency,
            derivative_frequency,
            rtol=0.0,
            atol=max(1.0e-9, target_df * 1.0e-8),
        ):
            raise ValueError("Base and frequency-derivative trace frequencies differ")

        frequency_order = np.argsort(base_frequency)
        base_frequency = base_frequency[frequency_order]
        base_values = base.data
        derivative_values = derivative.data
        if callable(getattr(base_values, "compute", None)):
            base_values = base_values.compute()
        if callable(getattr(derivative_values, "compute", None)):
            derivative_values = derivative_values.compute()
        base_values = np.asarray(base_values)[frequency_order]
        derivative_values = np.asarray(derivative_values)[frequency_order]

        solved_f_max = float(base_frequency[-1])
        coarse_steps = np.diff(base_frequency)
        coarse_df = float(np.median(coarse_steps))
        taper_width = 0.0
        if isinstance(high_frequency_taper, (bool, np.bool_)):
            if high_frequency_taper:
                taper_width = coarse_df
        else:
            taper_width = float(high_frequency_taper)
            if taper_width < 0.0:
                raise ValueError("high_frequency_taper width must be non-negative")

        untapered_sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=solved_f_max,
            df=target_df,
            upscale=upscale,
        )
        reconstruction_f_max = solved_f_max
        if taper_width > 0.0:
            taper_steps = max(1, int(np.ceil(taper_width / target_df)))
            reconstruction_f_max += taper_steps * target_df
        reconstruction_steps = int(np.ceil(reconstruction_f_max / target_df))
        reconstruction_f_max = reconstruction_steps * target_df
        sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=reconstruction_f_max,
            df=target_df,
            upscale=upscale,
        )

        # Interpolate the solver response before applying the wavelet. Extending
        # an already weighted response to zero lets its endpoint derivative
        # create a large cubic lobe even where the source spectrum should be
        # rapidly decaying. A fixed, sufficiently oversampled wavelet grid is
        # independent of the requested rolloff width, so adding the continuation
        # cannot change the wavelet at any existing frequency.
        base_wavelet_sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=solved_f_max,
            df=target_df,
        )
        declared_wavelet_f_max = float(
            getattr(wavelet, "f_max", solved_f_max) or solved_f_max
        )
        wavelet_upscale = max(
            4,
            int(np.ceil(declared_wavelet_f_max / solved_f_max)),
        )
        wavelet_sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=solved_f_max,
            df=target_df,
            upscale=wavelet_upscale,
        )
        base_wavelet_value = self._sample_wavelet_spectrum(
            wavelet,
            base_wavelet_sampling.T_list,
            sampling.F_list,
        )
        oversampled_wavelet_value = self._sample_wavelet_spectrum(
            wavelet,
            wavelet_sampling.T_list,
            sampling.F_list,
        )
        # Match the DFT normalization used by the standard path, which samples
        # the wavelet on the base (non-upscaled) time grid.
        oversampled_wavelet_value *= (
            base_wavelet_sampling.T_list.size / wavelet_sampling.T_list.size
        )
        solved_endpoint_index = int(np.rint(solved_f_max / target_df))
        endpoint_denominator = oversampled_wavelet_value[solved_endpoint_index]
        if not np.isclose(endpoint_denominator, 0.0):
            oversampled_wavelet_value *= (
                base_wavelet_value[solved_endpoint_index] / endpoint_denominator
            )
        wavelet_value = np.where(
            sampling.F_list <= solved_f_max + target_df * 1.0e-8,
            base_wavelet_value,
            oversampled_wavelet_value,
        )

        expand = (slice(None),) + (None,) * (base_values.ndim - 1)

        # Remove a reference delay before interpolation. For the Fourier
        # convention used here a delayed response is exp(-2j*pi*f*tau), so the
        # demodulating factor is exp(+2j*pi*f*tau). Its derivative contributes
        # the second product-rule term below. This is applied to the unweighted
        # solver response; the wavelet is multiplied in only after the Hermite
        # spline has been evaluated on the dense frequency grid.
        response = base_values
        response_derivative = derivative_values
        if interpolation_time_shift != 0.0:
            phase = np.exp(2.0j * np.pi * base_frequency * interpolation_time_shift)[
                expand
            ]
            response = phase * response
            response_derivative = phase * (
                response_derivative
                + 2.0j * np.pi * interpolation_time_shift * base_values
            )

        knot_frequency = base_frequency
        knot_value = response
        knot_derivative = response_derivative
        if taper_width > 0.0:
            zero_shape = (1, *knot_value.shape[1:])
            knot_frequency = np.append(knot_frequency, reconstruction_f_max)
            knot_value = np.concatenate(
                (knot_value, np.zeros(zero_shape, dtype=knot_value.dtype)), axis=0
            )
            knot_derivative = np.concatenate(
                (
                    knot_derivative,
                    np.zeros(zero_shape, dtype=knot_derivative.dtype),
                ),
                axis=0,
            )

        interpolator = CubicHermiteSpline(
            knot_frequency,
            knot_value,
            knot_derivative,
            axis=0,
            extrapolate=False,
        )
        evaluation_frequency = np.array(sampling.F_list, copy=True)
        # Frequencies loaded from a job can differ by a few ulps from the dense
        # linspace grid (for example 24.999999999999996 versus 25.0). Hermite
        # extrapolation is disabled, so evaluating the nominal endpoint just
        # beyond its knot would otherwise produce NaN and silently zero the
        # highest solved bin. Snap dense samples back to coincident knots.
        for knot in knot_frequency:
            grid_index = int(np.rint(knot / target_df))
            if 0 <= grid_index < evaluation_frequency.size and np.isclose(
                evaluation_frequency[grid_index],
                knot,
                rtol=0.0,
                atol=target_df * 1.0e-8,
            ):
                evaluation_frequency[grid_index] = knot
        values = np.nan_to_num(interpolator(evaluation_frequency))
        if interpolation_time_shift != 0.0:
            restore_phase = np.exp(
                -2.0j * np.pi * sampling.F_list * interpolation_time_shift
            )
            values = (
                values * restore_phase[(slice(None),) + (None,) * (values.ndim - 1)]
            )
        values = values * wavelet_value[expand]
        # NumPy-style irfft applies a 1/N normalization. Extending the spectrum
        # for a rolloff increases N, so unchanged Fourier coefficients would
        # otherwise reduce every time-domain component, including frequencies
        # below solved_f_max. Scale by the FFT-length ratio so X/N (the physical
        # spectral amplitude) is invariant on all pre-existing bins. This is
        # one only when no rolloff changes the reconstruction grid.
        fft_length_scale = sampling.nTime / untapered_sampling.nTime
        if fft_length_scale != 1.0:
            values = values * fft_length_scale
        dims = list(base.dims)
        coords = {
            dim: (sampling.F_list if dim == "frequency" else base.coords[dim])
            for dim in dims
        }
        spectrum = DataArray(values, dims=dims, coords=coords)
        spectrum.attrs.update(base.attrs)
        spectrum.attrs.update(
            {
                "reconstruction": reconstruction,
                "target_df": target_df,
                "solved_frequency_count": int(base_frequency.size),
                "reconstructed_frequency_count": int(sampling.nFreq),
                "solved_f_max": solved_f_max,
                "high_frequency_taper_width": taper_width,
                "fft_length_scale": fft_length_scale,
                "wavelet_application": "post_interpolation",
                "interpolation_time_shift": interpolation_time_shift,
            }
        )
        if "laplace" in base.coords:
            laplace = self._uniform_laplace(base)
            spectrum = spectrum.assign_coords(
                laplace=("frequency", np.full(sampling.nFreq, laplace))
            )
        return spectrum, sampling

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

    @staticmethod
    def _frequency_derivative_order(group: str) -> int:
        """Return the raw physical-frequency derivative order for a group."""

        for order, suffix in ((1, "_df"), (2, "_d2f"), (3, "_d3f"), (4, "_d4f")):
            if str(group).endswith(suffix):
                return order
        return 0

    def read_TD(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        laplace_compensation: Union[str, bool] = "auto",
        reconstruction: Optional[str] = None,
        target_df: Optional[float] = None,
        high_frequency_taper: Optional[Union[bool, float]] = None,
        interpolation_time_shift: Optional[float] = None,
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
            reconstruction: ``"standard"`` or derivative-assisted ``"hermite"``.
                Defaults to the job's saved time-reconstruction setting.
            target_df: Dense output frequency spacing for Hermite reconstruction.
            high_frequency_taper: ``True`` adds a one-solved-interval
                continuation to a zero-value, zero-slope endpoint; a positive
                number specifies its width in hertz. Standard reconstruction
                requires a matching first-derivative trace group when enabled.
            interpolation_time_shift: Time shift in seconds used to remove a
                linear phase trend before Hermite interpolation and restore it
                afterward. Defaults to the value saved with the job.
            **kwargs: Optional frequency-domain read controls.

        Returns:
            Time-domain ``xarray.DataArray``.
        """

        eager_max_bytes = kwargs.pop("eager_max_bytes", _TD_EAGER_MAX_BYTES)
        if eager_max_bytes is None:
            eager_max_bytes = 0
        if int(eager_max_bytes) < 0:
            raise ValueError("eager_max_bytes must be non-negative or None")
        compensation_mode = self._normalize_laplace_compensation(laplace_compensation)
        reconstruction_config = self.metadata.get("time_reconstruction", {})
        if reconstruction is None:
            reconstruction = reconstruction_config.get("method", "standard")
        reconstruction = str(reconstruction).lower()
        if reconstruction not in {"standard", "hermite"}:
            raise ValueError("reconstruction must be 'standard' or 'hermite'")
        if high_frequency_taper is None:
            high_frequency_taper = reconstruction_config.get(
                "high_frequency_taper", False
            )
        derivative_assisted = False

        if reconstruction == "hermite":
            if "f_taper" in kwargs:
                raise ValueError(
                    "f_taper is not supported with Hermite reconstruction; "
                    "use high_frequency_taper"
                )
            if target_df is None:
                target_df = reconstruction_config.get("target_df")
            if target_df is None:
                sample_every = int(
                    reconstruction_config.get(
                        "sample_every",
                        reconstruction_config.get("frequency_reduction", 4),
                    )
                )
                target_df = self.metadata["df"] / sample_every
            if interpolation_time_shift is None:
                interpolation_time_shift = reconstruction_config.get(
                    "interpolation_time_shift", 0.0
                )
            fd, sampling = self._derivative_assisted_spectrum(
                group,
                component,
                source,
                wavelet,
                target_df=float(target_df),
                upscale=upscale,
                high_frequency_taper=high_frequency_taper,
                interpolation_time_shift=float(interpolation_time_shift),
                max_bytes=int(eager_max_bytes),
            )
            derivative_assisted = True
        else:
            if high_frequency_taper:
                if "f_taper" in kwargs:
                    raise ValueError(
                        "f_taper and high_frequency_taper cannot be used together"
                    )
                fd, sampling = self._derivative_assisted_spectrum(
                    group,
                    component,
                    source,
                    wavelet,
                    target_df=float(self.metadata["df"]),
                    upscale=upscale,
                    high_frequency_taper=high_frequency_taper,
                    interpolation_time_shift=0.0,
                    max_bytes=int(eager_max_bytes),
                    reconstruction="standard",
                )
                derivative_assisted = True
            else:
                sampling = UniformSweepSampling(
                    f_min=0.0,
                    f_max=self.metadata["f_max"],
                    df=self.metadata["df"],
                    upscale=upscale,
                )
                fd = self._read_fd_eager(
                    group,
                    component,
                    source,
                    max_bytes=int(eager_max_bytes),
                )
                if fd is not None:
                    source_sampling = UniformSweepSampling(
                        f_min=0.0,
                        f_max=self.metadata["f_max"],
                        df=self.metadata["df"],
                    )
                    wavelet.times = source_sampling.T_list
                    fd = self._apply_wavelet_to_fd(fd, wavelet, **kwargs)
                else:
                    fd = self.read_FD(group, component, source, wavelet, **kwargs)
                    fd = self._coalesce_td_chunks(fd)
        laplace = self._uniform_laplace(fd)
        wavelet.times = sampling.T_list

        if reconstruction == "standard":
            fd = fd.interp(frequency=sampling.F_list, kwargs={"fill_value": 0})

        # For a real time signal, successive derivatives with respect to real
        # Fourier frequency alternate between anti-Hermitian and Hermitian
        # symmetry. Multiplication by i**order restores Hermitian symmetry so
        # irfft returns the real time-moment representation. Derivatives with
        # respect to imaginary (Laplace) frequency already retain Hermitian
        # symmetry and need no phase rotation.
        derivative_order = self._frequency_derivative_order(group)
        if derivative_order:
            fd = fd.copy(data=fd.data * (1j**derivative_order))
        fft = get_fft_backend()
        td = fft.irfft(fd.data, axis=0)
        dims = ["time" if d == "frequency" else d for d in fd.dims]
        coords = {
            d: (
                fd.coords[d]
                if d in fd.coords
                else sampling.T_list[:-1] - wavelet.center
            )
            for d in dims
        }
        coords.update(
            {
                name: coordinate
                for name, coordinate in fd.coords.items()
                if name not in coords and coordinate.dims == ("receiver",)
            }
        )

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
        td.attrs["reconstruction"] = reconstruction
        td.attrs["phase_derivative_order"] = derivative_order
        if derivative_assisted:
            for name in (
                "target_df",
                "solved_frequency_count",
                "reconstructed_frequency_count",
                "solved_f_max",
                "high_frequency_taper_width",
                "fft_length_scale",
                "wavelet_application",
                "interpolation_time_shift",
            ):
                td.attrs[name] = fd.attrs[name]
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
