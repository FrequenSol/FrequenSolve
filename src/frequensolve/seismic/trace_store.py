"""Internal HDF5 trace-store reader.

``TraceStore`` backs the public ``TraceDataset`` facade.
"""

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

_ROOT_TRACE_METADATA_DATASETS = {"frequency", "laplace", "task_id"}


def _decode_h5_strings(values):
    values = np.asarray(values)
    if values.dtype.kind in {"S", "O"}:
        return np.array(
            [
                item.decode("utf-8", "ignore") if isinstance(item, bytes) else str(item)
                for item in values
            ]
        )
    return values


def _decode_dim_list(values) -> list[str]:
    return [
        item.decode("utf-8", "ignore") if isinstance(item, bytes) else str(item)
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


def _as_file_list(files: Iterable[Union[str, Path]]) -> List[Union[str, Path]]:
    if isinstance(files, (str, Path)):
        return [files]
    return list(files)


@dataclass(init=False)
class TraceStore:
    metadata: Dict[str, Any]
    files: List[str]
    _upscale: int
    _consolidated: Optional[Path] = None
    _cache_dir: Optional[Path] = None
    _open_files: List[h5py.File]

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

    @classmethod
    def from_job(cls, job, upscale: int = 1):
        """Create a TraceStore from a simulation job.

        Args:
            job: A SimulationJob-like object.
            upscale: Time-domain upscale factor.
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
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
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
        return self._upscale

    @upscale.setter
    def upscale(self, upscale: int) -> None:
        self._upscale = upscale

    def times(self, upscale: Optional[int] = None) -> np.ndarray:
        """Returns the trace sample times."""

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
        self._ensure_consolidated()
        with h5py.File(self._consolidated, "r") as f:
            return self._h5_trace_groups(f, self.metadata.get("groups"))

    def dims(self, group) -> list[str]:
        self._ensure_consolidated()
        with h5py.File(self._consolidated, "r") as f:
            return [
                "source" if dim == "shot" else dim for dim in _trace_data_dims(f[group])
            ]

    def components(self, group) -> list[str]:
        self._ensure_consolidated()
        with h5py.File(self._consolidated, "r") as f:
            for path in (
                f"survey/receiver_groups/{group}/traces/component_name",
                f"survey/receiver_groups/{group}/components/component_name",
            ):
                if path in f:
                    return _unique_preserve_order(_decode_h5_strings(f[path][()]))
            if "component" in f[group].attrs:
                return _decode_h5_strings(f[group].attrs["component"])
            return np.arange(1, f[group].shape[-2] + 1)

    def sources(self, group) -> list[str]:
        self._ensure_consolidated()
        with h5py.File(self._consolidated, "r") as f:
            path = f"survey/receiver_groups/{group}/traces/source_id"
            if path in f:
                return _unique_preserve_order(f[path][()])
            if "shot" in f[group].attrs:
                return f[group].attrs["shot"]
            return np.arange(1, f[group].shape[-1] + 1)

    def shots(self, group) -> list[str]:
        """Compatibility alias for ``sources``."""

        return self.sources(group)

    def frequencies(self, group) -> list[str]:
        self._ensure_consolidated()
        with h5py.File(self._consolidated, "r") as f:
            if "frequency" in f:
                return f["frequency"][()]
            if "frequency" in f[group].attrs:
                return f[group].attrs["frequency"]
            return np.array(list(self.metadata["f_map"].values()))

    def receivers(self, group) -> list[str]:
        self._ensure_consolidated()
        with h5py.File(self._consolidated, "r") as f:
            for path in (
                f"survey/receiver_groups/{group}/traces/receiver_id",
                f"survey/receiver_groups/{group}/receivers/receiver_id",
            ):
                if path in f:
                    return _unique_preserve_order(f[path][()])
            if "receiver" in f[group].attrs:
                return f[group].attrs["receiver"]
            dims = np.asarray(_decode_dim_list(f[group].attrs["dims"])[::-1])
            ind = np.where(dims == "receiver")[0][0]
            return np.arange(1, f[group].shape[ind] + 1)

    def survey_tables(self) -> Dict[str, Any]:
        if self._consolidated is None:
            self.consolidate()

        def read_group(group):
            out = {}
            for key, item in group.items():
                if isinstance(item, h5py.Dataset):
                    value = item[()]
                    out[key] = _decode_h5_strings(value).tolist()
                elif isinstance(item, h5py.Group):
                    out[key] = read_group(item)
            return out

        with h5py.File(self._consolidated, "r") as f:
            if "survey" not in f:
                return {}
            return read_group(f["survey"])

    @property
    def summary(self, colorize: bool = True) -> str:
        def _gray(text: str, light: bool = True) -> str:
            if colorize:
                if light:
                    return f"\033[38;5;248m{text}\033[0m"
                else:
                    return f"\033[90m{text}\033[0m"
            return text

        out = ""
        for group in self.groups:
            recv = self.receivers(group)
            shot = self.shots(group)
            comp = self.components(group)
            freq = self.frequencies(group)

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
        return out

    def __str__(self) -> str:
        return self.summary

    def _ensure_consolidated(self) -> None:
        if self._consolidated is None or not Path(self._consolidated).exists():
            self.consolidate()

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
    def _h5_trace_groups(h5, configured: Optional[Iterable[str]] = None) -> list[str]:
        configured_groups = [str(group) for group in configured or []]
        if configured_groups:
            return [
                name
                for name in configured_groups
                if name in h5 and isinstance(h5[name], h5py.Dataset)
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
                groups = TraceStore._h5_trace_groups(h5)
                return any(
                    _decode_dim_list(h5[group].attrs["dims"])[-1] == "frequency"
                    for group in groups
                )
        except OSError:
            return False

    def consolidate(self, cache_dir: Optional[Union[str, Path]] = None) -> Path:
        """Create a virtual HDF5 file with frequency as the leading axis."""

        records = [Path(file) for file in self.files]
        if not records:
            raise FileNotFoundError("No trace files were provided")

        if (
            len(records) == 1
            and records[0].exists()
            and self._is_packed_trace_file(records[0])
        ):
            self.close()
            self._consolidated = records[0]
            return self._consolidated

        available = []
        for record in records:
            if record.exists():
                available.append(record)
            else:
                warnings.warn(
                    f"Trace file is missing and will be omitted from the VDS: {record}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if not available:
            raise FileNotFoundError("No trace files exist")

        metadata_record = records[0] if records[0].exists() else available[0]
        freqs = [self._read_trace_frequency(record) for record in available]
        cache_dir = Path(cache_dir) if cache_dir is not None else self._cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        new_file = self._consolidated_path(metadata_record, cache_dir=cache_dir)

        self.close()
        if new_file.exists():
            new_file.unlink()

        with h5py.File(metadata_record, "r") as first, h5py.File(new_file, "w") as nf:
            nf.create_dataset("frequency", data=np.asarray(freqs, dtype=float))
            if "survey" in first:
                first.copy("survey", nf)

            for group in self.metadata["groups"]:
                if group not in first:
                    raise KeyError(
                        f"Group '{group}' not found in HDF5 {metadata_record}"
                    )

                source = first[group]
                source_shape = source.shape
                layout = h5py.VirtualLayout(
                    shape=(len(available),) + source_shape,
                    dtype=source.dtype,
                )
                for index, record in enumerate(available):
                    layout[index] = h5py.VirtualSource(
                        str(record), group, shape=source_shape
                    )
                nf.create_virtual_dataset(group, layout)

                dset = nf[group]
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
        try:
            import dask.array as da
        except ModuleNotFoundError as exc:
            from frequensolve._optional import optional_dependency_error

            raise optional_dependency_error(
                "TraceDataset lazy HDF5 reading",
                extra="parallel",
                dependencies=("dask",),
                error=exc,
            ) from exc

        self._ensure_consolidated()
        h5 = h5py.File(self._consolidated, "r")
        self._open_files.append(h5)
        dset = h5[group]
        dims = _trace_data_dims(dset)
        if len(dset.shape) == len(dims) + 1:
            dims.append("complex")
        dims = ["source" if dim == "shot" else dim for dim in dims]
        coords = {}
        for axis, dim in enumerate(dims):
            if dim == "complex":
                coords[dim] = (
                    ["real", "imag"]
                    if dset.shape[axis] == 2
                    else np.arange(1, dset.shape[axis] + 1)
                )
                continue
            attr_dim = "shot" if dim == "source" and "shot" in dset.attrs else dim
            survey_paths = []
            if dim == "receiver":
                survey_paths = [
                    f"survey/receiver_groups/{group}/traces/receiver_id",
                    f"survey/receiver_groups/{group}/receivers/receiver_id",
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
                ]

            survey_path = next((path for path in survey_paths if path in h5), None)
            if survey_path is not None:
                values = _unique_preserve_order(_decode_h5_strings(h5[survey_path][()]))
                coords[dim] = (
                    values
                    if len(values) == dset.shape[axis]
                    else np.arange(1, dset.shape[axis] + 1)
                )
            elif dim == "frequency" and "frequency" in h5:
                coords[dim] = h5["frequency"][()]
            elif attr_dim in dset.attrs:
                coords[dim] = dset.attrs[attr_dim]
            else:
                coords[dim] = np.arange(1, dset.shape[axis] + 1)
        if "complex" in dims and "complex" not in coords:
            coords["complex"] = ["real", "imag"]

        chunks = (dset.shape[0], 1, 1, *dset.shape[3:])
        data = da.from_array(dset, chunks=chunks)
        fd = DataArray(data, dims=dims, coords=coords)
        return fd

    def read_FD(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Optional[Wavelet] = None,
        **kwargs,
    ):
        if wavelet is None:
            dset = self.read_h5(group)
            gather = dset.sel(component=component, source=source)
            fd = gather.sel(complex="real") + 1j * gather.sel(complex="imag")
            fd = fd.fillna(0)
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

            freqs = wavelet.frequencies
            spectrum = DataArray(
                wavelet.spectrum, dims=["frequency"], coords={"frequency": freqs}
            )
            w = spectrum.interp(
                frequency=gather.coords["frequency"].values, kwargs={"fill_value": 0}
            )
            fd = gather.sel(complex="real") + 1j * gather.sel(complex="imag")
            fd = fd.fillna(0)

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
            fd = fd * w
            return fd

    def read_TD(
        self,
        group: str,
        component: str,
        source: int,
        wavelet: Wavelet,
        upscale: int = 1,
        T_max: Optional[float] = None,
        **kwargs,
    ) -> DataArray:
        sampling = UniformSweepSampling(
            f_min=0.0,
            f_max=self.metadata["f_max"],
            df=self.metadata["df"],
            upscale=upscale,
        )
        fd = self.read_FD(group, component, source, wavelet, **kwargs)
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
        if T_max is not None:
            td = td.sel(time=slice(None, T_max))

        td.attrs["source_group"] = source
        td.attrs["receiver_group"] = group
        # NOTE: this is a temporary hack since receivers read from project path
        td.attrs["project_path"] = str(self.metadata["project"])
        td.attrs["simulation"] = str(self.metadata["simulation"])
        td.attrs["long_name"] = f"{component}"
        for d in td.dims:
            td.coords[d].attrs["long_name"] = d.title()
            if d == "time":
                td.coords[d].attrs["units"] = "s"
                td.coords[d].attrs["description"] = "Time"
            elif d == "frequency":
                td.coords[d].attrs["units"] = "Hz"
                td.coords[d].attrs["description"] = "Frequency"
        return td

    def CosineWindow(self, t0: float, tf: float, taper: float) -> DataArray:
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

        td.attrs["source_group"] = source
        td.attrs["receiver_group"] = group
        # NOTE: this is a temporary hack since receivers read from project path
        td.attrs["project_path"] = str(self.metadata["project"])
        td.attrs["simulation"] = str(self.metadata["simulation"])
        td.attrs["long_name"] = f"{component}"
        for d in td.dims:
            td.coords[d].attrs["long_name"] = d.title()
            if d == "time":
                td.coords[d].attrs["units"] = "s"
                td.coords[d].attrs["description"] = "Time"
            elif d == "frequency":
                td.coords[d].attrs["units"] = "Hz"
                td.coords[d].attrs["description"] = "Frequency"
        return td
